# -*- coding: utf-8 -*-

import copy
import numpy as np
from typing import List

from pcode.local_training.base_worker import BaseWorker
import pcode.create_dataset as create_dataset


"""
Implementation of
    "Ditto: Fair and Robust Federated Learning Through Personalization",
    https://arxiv.org/abs/2012.04221.
"""


class DittoWorker(BaseWorker):
    def __init__(self, conf):
        super(DittoWorker, self).__init__(conf)
        self.regularized_factor = conf.regularized_factor
        self.n_personalized_epochs = conf.n_personalized_epochs
        self.eval_dataset = conf.eval_dataset  # can be test_loader or val_loader
        self.is_in_childworker = True

    def run(self):
        while True:
            self._listen_to_master()

            # check if we need to terminate the training or not.
            if self._terminate_by_early_stopping():
                return

            # receive global model (note as self.model) and perform standard local training.
            self._recv_model_from_master()
            # a personal model to hold local parameters.
            if not hasattr(self, "personal_model"):
                self.personal_model = copy.deepcopy(self.model)

            state = self.train(model=self.model)
            params_to_send = state["model"].state_dict()

            # personalization.
            p_state = self._personalized_train(model=self.personal_model)
            self.personal_model.load_state_dict(copy.deepcopy(p_state["model"].state_dict()))

            # evaluate the model.
            perf = self._evaluate_all_test_sets(p_state)

            # display the info and sync the model & perf.
            self._display_info(p_state, perf)
            self._send_model_to_master(params_to_send, perf)
            del state, p_state

            if self._terminate_by_complete_training():
                return

    def _personalized_train(self, model):
        # regularized local training based on global model
        self.is_in_personalized_training = True
        state = self._init_train_process(model=model)
        self.model.to(self.graph.device)
        self.n_local_epochs = self.n_personalized_epochs

        while not self._is_finished_one_comm_round(state):
            for _input, _target in state["train_loader"]:
                # load data
                with self.timer("load_data", epoch=state["scheduler"].epoch_):
                    data_batch = create_dataset.load_data_batch(
                        self.conf, _input, _target, is_training=True
                    )

                # inference and get current performance.
                with self.timer("forward_pass", epoch=state["scheduler"].epoch_):
                    state["optimizer"].zero_grad()
                    loss, _ = self._inference(
                        data_batch, state["model"], state["tracker"]
                    )

                with self.timer("backward_pass", epoch=state["scheduler"].epoch_):
                    loss.backward()
                    state["optimizer"].step()
                    # add regularization.
                    self._add_regularization(state)
                    state["scheduler"].step()

                if self.conf.display_log:
                    self._display_logging(state)
                if self._is_diverge(state):
                    self._terminate_comm_round(state)
                    return

            # refresh the logging cache at the end of each epoch.
            state["tracker"].reset()
            if self.conf.logger.meet_cache_limit():
                self.conf.logger.save_json()

        self._terminate_comm_round(state)
        self.is_in_personalized_training = False
        return state

    def _add_regularization(self, state):
        # add the regularization term's gradient to weights
        for global_w, local_w in zip(
            self.model.parameters(), state["model"].parameters()
        ):
            if local_w.grad is not None:
                local_w.grad.data.add_(
                    -state["optimizer"].param_groups[0]["lr"]
                    * self.regularized_factor
                    * (local_w.data - global_w.data)
                )

