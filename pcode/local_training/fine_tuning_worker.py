# -*- coding: utf-8 -*-

import copy
import json
import os
import torch
import numpy as np
from typing import List
from pcode.local_training.base_worker import BaseWorker
import pcode.local_training.utils as utils


"""
Implementation of FedAvg with fine-tuning,
    which personalizes the model through local finetuning.
"""


class FineTuningWorker(BaseWorker):
    def __init__(self, conf):
        super(FineTuningWorker, self).__init__(conf)
        self.n_personalized_epochs = conf.n_personalized_epochs
        self.eval_dataset = conf.eval_dataset  # can be test_loader or val_loader
        self.is_in_childworker = True

    def run(self):
        while True:
            self._listen_to_master()

            # check if we need to terminate the training or not.
            if self._terminate_by_early_stopping():
                return

            # receive the perform standard local training.
            self._recv_model_from_master()
            # a personal model to hold local parameters.
            if not hasattr(self, "personal_model"):
                self.personal_model = copy.deepcopy(self.model)

            state = self.train(model=self.model)
            params_to_send = copy.deepcopy(state["model"].state_dict())

            # personalization.
            p_state = self._personalized_train(model=self.model)

            # evaluate the personalized model.
            perf = self._evaluate_all_test_sets(p_state)

            # display the personalized perf and sync the model & perf.
            self._display_info(p_state, perf)
            self._send_model_to_master(params_to_send, perf)
            del state, p_state

            if self._terminate_by_complete_training():
                return

    def _personalized_train(self, model):
        self.is_in_personalized_training = True
        self.n_local_epochs = self.n_personalized_epochs
        state = super(FineTuningWorker, self).train(model)
        self.is_in_personalized_training = False
        return state

