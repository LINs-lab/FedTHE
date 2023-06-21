# -*- coding: utf-8 -*-

import copy
import math
import numpy as np
from typing import List
from pcode.local_training.base_worker import BaseWorker
import pcode.create_dataset as create_dataset
import pcode.utils.loss as loss
from pcode.utils.stat_tracker import RuntimeTracker
import pcode.local_training.utils as utils
import torch
import torch.nn as nn


"""
Implementation of
    "On Bridging Generic and Personalized Federated Learning".
    https://arxiv.org/abs/2107.00778
"""


class FedRodWorker(BaseWorker):
    def __init__(self, conf):
        super(FedRodWorker, self).__init__(conf)
        self.n_personalized_epochs = conf.n_personalized_epochs
        self.eval_dataset = conf.eval_dataset  # can be test_loader or val_loader
        self.is_in_childworker = True
        self.num_head = 2
        self.conf = conf

    def run(self):
        while True:
            self._listen_to_master()

            # check if we need to terminate the training or not.
            if self._terminate_by_early_stopping():
                return

            # receive global model (note as self.model) and perform standard local training.
            self._recv_model_from_master()
            # create personal model and register hook.
            if not hasattr(self, "personal_head"):
                self.personal_head = nn.Linear(self.conf.rep_len, utils.get_num_classes(self.conf.data))

            # local training on the received model with balanced risk minimization.
            state = self._brm_train(model=self.model)
            params_to_send = copy.deepcopy(state["model"].state_dict())

            # personalization.
            p_state = self._personalized_train(model=self.model)

            # evaluate the model.
            perf = self._evaluate_all_test_sets(p_state)

            # display the info and sync the model & perf.
            self._display_info(p_state, perf)
            self._send_model_to_master(params_to_send, perf)
            del state, p_state

            if self._terminate_by_complete_training():
                return

    def _brm_train(self, model):
        # change the criterion to balanced loss and do local training.
        self.criterion = loss.BalancedSoftmaxLoss(self._get_target_histogram())
        state = super(FedRodWorker, self).train(model)

        return state

    def _personalized_train(self, model):
        self.is_in_personalized_training = True
        self.n_local_epochs = self.n_personalized_epochs
        self.erm_criterion = nn.CrossEntropyLoss(reduction="mean")
        # define dataloader, optimizer, scheduler and tracker
        state = self._init_train_process(model=model)
        self.personal_head.to(self.graph.device)
        state["optimizer"].add_param_group({"params": self.personal_head.parameters()})
        # freeze the model, except the personal head
        state["model"].requires_grad_(True)
        self.personal_head.requires_grad_(True)
        self.rep_layer = utils.determine_hook(state["model"], name=self.conf.arch)
        self.rep_layer.register_forward_hook(utils.hook)
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
                    loss, _ = self._multi_head_inference(data_batch, state["model"], state["tracker"])

                with self.timer("backward_pass", epoch=state["scheduler"].epoch_):
                    loss.backward()
                    state["optimizer"].step()
                    state["scheduler"].step()

                if self.conf.display_log:
                    self._display_logging(state)
                if self._is_diverge(state):
                    self._terminate_comm_round(state)
                    return state

            # refresh the logging cache at the end of each epoch.
            state["tracker"].reset()
            if self.logger.meet_cache_limit():
                self.logger.save_json()

        # terminate
        self._terminate_comm_round(state)
        self.is_in_personalized_training = False
        return state

    def _validate(self, state, dataset_name):
        # switch to evaluation mode.
        state["model"].eval()
        num = 1
        num_batches = self.get_num_batches(state, dataset_name)

        # evaluate on test_loader.
        tracker_te = RuntimeTracker(metrics_to_track=self.metrics.metric_names)
        for _input, _target in state[dataset_name]:
            # load data and check performance.
            data_batch = create_dataset.load_data_batch(
                self.conf, _input, _target, is_training=False
            )
            with torch.no_grad():
                self._multi_head_inference(data_batch, state["model"], tracker=tracker_te)

            if num >= num_batches: break
            else: num += 1

        return tracker_te()

    def _multi_head_inference(self, data_batch, model, tracker=None):
        g_out = model(data_batch["input"])
        brm_loss = self.criterion(g_out, data_batch["target"])
        # we dont want to bp gradients to feature extractor, so detach here
        p_out = self.personal_head(self.rep_layer.rep.detach())
        local_head_logits = g_out.detach() + p_out
        erm_loss = self.erm_criterion(local_head_logits, data_batch["target"])
        loss = brm_loss + erm_loss
        performance = self.metrics.evaluate(
            loss, local_head_logits, data_batch["target"]  # check here
        )
        tracker.update_metrics(
            [loss.item()] + performance, n_samples=data_batch["input"].size(0)
        )

        return loss, local_head_logits

    def _get_target_histogram(self, display=True):
        local_data_loaders = self.fl_data_cls.intra_client_data_partition_and_create_dataloaders(
            localdata_id=self.client_id - 1,  # localdata_id starts from 0 while client_id starts from 1.
            other_ids=self._get_other_ids(),
            is_in_childworker=self.is_in_childworker,
            local_train_ratio=self.conf.local_train_ratio,
            batch_size=1,
            display_log=False,
        )
        hist = torch.zeros(utils.get_num_classes(self.conf.data))
        train_loader = local_data_loaders["train"]
        for _, _target in train_loader:
            hist[_target.item()] += 1
        if display:
            self.logger.log(
                f"\tWorker-{self.graph.worker_id} (client-{self.client_id}) training histogram is like {hist}"
            )

        return hist
