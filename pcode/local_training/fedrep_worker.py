# -*- coding: utf-8 -*-

import copy
import numpy as np
from typing import List

from pcode.local_training.base_worker import BaseWorker


"""
Implementation of
    "Exploiting Shared Representations for Personalized Federated Learning",
    https://arxiv.org/abs/2102.07078.
"""


class FedRepWorker(BaseWorker):
    def __init__(self, conf):
        super(FedRepWorker, self).__init__(conf)
        self.fedrep_personal_layers = conf.fedrep_personal_layers
        self.n_personalized_epochs = conf.n_personalized_epochs
        self.eval_dataset = conf.eval_dataset  # can be test_loader or val_loader
        self.is_in_childworker = True

    def run(self):
        # according to their work, they first train the head layers (personal layers) with base layers frozen
        # then train the base layers with personal layers frozen.
        # here we denote the training of head layers as personalized train,
        # and denote the training of base layers as normal training.
        while True:
            self._listen_to_master()

            # check if we need to terminate the training or not.
            if self._terminate_by_early_stopping():
                return

            # receive global model (note as self.model) and train the personal layers.
            self._recv_model_from_master()
            # a personal model to hold head layers' parameters
            if not hasattr(self, "personal_model"):
                self.personal_model = copy.deepcopy(self.model)

            # local training in an alternative fashion.
            state = self._partial_train(model=self.model, training_part="head")
            state = self._partial_train(model=state["model"], training_part="base")
            params_to_send = state["model"].state_dict()

            # personalized training.
            self.is_in_personalized_training = True
            self.n_local_epochs = self.n_personalized_epochs
            self._fill_local_head_weights(target_model=self.model, source_model=self.personal_model)
            p_state = self._partial_train(model=self.model, training_part="head")
            p_state = self._partial_train(model=p_state["model"], training_part="base")
            self.is_in_personalized_training = False
            self.personal_model.load_state_dict(copy.deepcopy(p_state["model"].state_dict()))

            # evaluate the model.
            perf = self._evaluate_all_test_sets(p_state)

            # display the info and sync the model & perf.
            self._display_info(p_state, perf)
            self._send_model_to_master(params_to_send, perf)
            del state, p_state

            if self._terminate_by_complete_training():
                return

    def _partial_train(self, model, training_part):
        # train the personal layers and freeze the base layers.
        model = copy.deepcopy(model)
        if training_part == "base":
            self._turn_off_partial_grads(model, which_part="head")
        else:
            self._turn_off_partial_grads(model, which_part="base")
        state = super(FedRepWorker, self).train(model)

        return state

    def _fill_local_head_weights(self, target_model, source_model):
        # change local personal layers of target model by source model
        model_tmp = copy.deepcopy(target_model.state_dict())
        local_params = copy.deepcopy(source_model.state_dict())
        from_layer = len(model_tmp) - self.fedrep_personal_layers
        for layer in list(model_tmp.keys())[from_layer::]:
            model_tmp[layer] = copy.deepcopy(local_params[layer])
        target_model.load_state_dict(model_tmp)

    def _turn_off_partial_grads(self, model, which_part="head"):
        # turn off the grads of personal layers or base layers.
        trainable_len = sum(1 for _ in model.parameters())
        flag = True if which_part == "head" else False
        for i, param in enumerate(model.parameters()):
            if i < trainable_len - self.fedrep_personal_layers:
                param.requires_grad = flag
            else:
                param.requires_grad = not flag
