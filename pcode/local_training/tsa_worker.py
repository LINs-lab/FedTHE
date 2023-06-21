# -*- coding: utf-8 -*-
import torch
import copy
import functools
import pcode.utils.loss as loss
import numpy as np
from typing import List
from pcode.local_training.base_worker import BaseWorker
import pcode.create_dataset as create_dataset
from pcode.utils.stat_tracker import RuntimeTracker
import pcode.local_training.utils as utils
from pcode.utils.mathdict import MathDict
import torch.nn.functional as F
import torch.nn as nn
import pcode.datasets.moco_data as moco_data
import pcode.create_optimizer as create_optimizer

"""
Implementation of test-time self-supervised aggregation in FL, refer to "Test-Agnostic Long-Tailed Recognition by Test-Time Aggregating Diverse Experts with Self-Supervision".
    https://arxiv.org/abs/2107.09249
"""


class TsaWorker(BaseWorker):
    def __init__(self, conf):
        super(TsaWorker, self).__init__(conf)
        self.conf = conf
        self.n_personalized_epochs = conf.n_personalized_epochs
        self.eval_dataset = conf.eval_dataset  # can be test_loader or val_loader
        self.is_in_childworker = True
        # test-time self-supervised aggregation
        self.num_head = 2
        self.tsa_steps = 10
        self.agg_weight = torch.nn.Parameter(torch.rand((self.conf.batch_size, self.num_head)).cuda(), requires_grad=True)
        self.agg_weight.data.fill_(1 / self.num_head)
        normalization, unnormalization = utils.get_normalization(self.conf.data)
        self.augmentation = moco_data.TwoCropsTransform(normalization, unnormalization)
        self.agg_weight.data.fill_(1 / self.num_head)

    def run(self):
        while True:
            self._listen_to_master()

            # check if we need to terminate the training or not.
            if self._terminate_by_early_stopping():
                return

            # receive the perform standard local training.
            self._recv_model_from_master()
            # a personal model to hold local parameters.
            if not hasattr(self, "personal_head"):
                # create three-head model here, instead of copy self.model
                self.personal_head = nn.Linear(self.conf.rep_len, utils.get_num_classes(self.conf.data), bias=False)

            state = self._brm_train(model=self.model)
            params_to_send = copy.deepcopy(state["model"].state_dict())  # use deepcopy

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

    def _brm_train(self, model):
        # change the criterion to balanced loss and do local training.
        # self.criterion = self._get_brm_loss()
        state = super(TsaWorker, self).train(model)
        return state

    def _personalized_train(self, model):
        self.is_in_personalized_training = True
        self.n_local_epochs = self.n_personalized_epochs
        self.erm_criterion = nn.CrossEntropyLoss(reduction="mean")
        # define dataloader, optimizer, scheduler and tracker
        state = self._init_train_process(model=model)
        self.personal_head.to(self.graph.device)
        # we want to optimize personal head
        state["optimizer"] = create_optimizer.define_optimizer(
            self.conf, model=self.personal_head, optimizer_name=self.conf.optimizer, lr=self._get_round_lr()
        )
        # freeze the model, except the personal head
        state["model"].requires_grad_(False)
        self.personal_head.requires_grad_(True)
        self.rep_layer = utils.determine_hook(state["model"], name=self.conf.arch)
        self.rep_layer.register_forward_hook(utils.hook)
        while not self._is_finished_one_comm_round(state):
            for _input, _target in state["train_loader"]:
                # load data
                with self.timer("load_data", epoch=state["scheduler"].epoch_):
                    data_batch = create_dataset.load_data_batch(
                        self.conf, _input, _target, is_training=True,
                    )

                # inference and get current performance.
                with self.timer("forward_pass", epoch=state["scheduler"].epoch_):
                    state["optimizer"].zero_grad()
                    g_out = state["model"](data_batch["input"])
                    p_out = self.personal_head(self.rep_layer.rep)
                    loss = self.erm_criterion(p_out, data_batch["target"])
                    agg_out = torch.stack([g_out, p_out], dim=1).mean(dim=1)
                    performance = self.metrics.evaluate(loss, agg_out, data_batch["target"])
                    state["tracker"].update_metrics(
                        [loss.item()] + performance, n_samples=data_batch["input"].size(0)
                    )

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

    def _validate_training(self, state, dataset, num_epochs, dataset_name, display=True):
        self.is_in_personalized_training = True
        state["model"].eval()
        tracker_te = RuntimeTracker(metrics_to_track=self.metrics.metric_names)
        lr = self._get_round_lr()
        # agg weight to optimize
        cos = torch.nn.CosineSimilarity(dim=1, eps=1e-6)
        for _input, _target in dataset:
            # test-time self-supervised aggregation
            data_batch = create_dataset.load_data_batch(
                self.conf, _input, _target, is_training=False
            )
            temperature = torch.hstack((torch.ones((data_batch["input"].shape[0], 1)).cuda(), torch.ones((data_batch["input"].shape[0], 1)).cuda()))

            self.agg_weight = torch.nn.Parameter(torch.tensor(temperature).cuda(), requires_grad=True)

            state["optimizer"] = torch.optim.Adam([self.agg_weight], lr=10*lr)
            self._calculate_samplewise_weight(state, data_batch, num_epochs, dataset_name, cos, display)
            # do inference for current batch
            with torch.no_grad():
                self._multi_head_inference(data_batch, state["model"], tracker_te)

        self.is_in_personalized_training = False
        self.agg_weight.data.fill_(1 / self.num_head)
        return tracker_te

    def _calculate_samplewise_weight(self, state, data_batch, num_epochs, dataset_name, opt, display=True):
        # augment the inputs
        aug_inputs = self.augmentation(data_batch["input"])
        # keep the gradients recorded
        outputs_1_g = state["model"](aug_inputs[0])
        outputs_1_p = self.personal_head(self.rep_layer.rep)
        outputs_2_g = state["model"](aug_inputs[1])
        outputs_2_p = self.personal_head(self.rep_layer.rep)
        for _ in range(num_epochs):
            # normalize the aggregation weight by softmax
            agg_softmax = torch.nn.functional.softmax(self.agg_weight)
            agg_output1 = agg_softmax[:, 0].unsqueeze(1) * outputs_1_g.detach() \
                            + agg_softmax[:, 1].unsqueeze(1) * outputs_1_p.detach()
            agg_output2 = agg_softmax[:, 0].unsqueeze(1) * outputs_2_g.detach() \
                            + agg_softmax[:, 1].unsqueeze(1) * outputs_2_p.detach()
            agg_output1_softmax = F.softmax(agg_output1, dim=1)
            agg_output2_softmax = F.softmax(agg_output2, dim=1)
            cos_similarity = opt(agg_output1_softmax, agg_output2_softmax).mean()
            loss = -cos_similarity
            state["optimizer"].zero_grad()
            loss.backward()

            state["optimizer"].step()

    def _multi_head_inference(self, data_batch, model, tracker=None):
        # inference procedure for multi-head nets.
        agg_softmax = torch.nn.functional.softmax(self.agg_weight)
        # do the forward pass and get the output.
        g_out = model(data_batch["input"])
        p_out = self.personal_head(self.rep_layer.rep)
        agg_output = agg_softmax[:, 0].unsqueeze(1) * g_out \
                        + agg_softmax[:, 1].unsqueeze(1) * p_out
        # agg_output_softmax = F.softmax(agg_output, dim=1)
        agg_output_softmax = agg_output
        # evaluate the output and get the loss, performance.
        loss = self.criterion(agg_output_softmax, data_batch["target"])
        performance = self.metrics.evaluate(loss, agg_output_softmax, data_batch["target"])

        # update tracker.
        if tracker is not None:
            tracker.update_metrics(
                [loss.item()] + performance, n_samples=data_batch["input"].size(0)
            )

        return loss

    def _validate(self, state, dataset_name):
        # switch to evaluation mode.
        state["model"].eval()
        self.rep_layer = utils.determine_hook(state["model"], name=self.conf.arch)
        self.rep_layer.register_forward_hook(utils.hook)
        # test-time self-supervised aggregation
        tracker_te = self._validate_training(state, state[dataset_name], self.tsa_steps, dataset_name)
        return tracker_te()

    def _get_target_histogram(self):
        local_data_loaders = self.fl_data_cls.intra_client_data_partition_and_create_dataloaders(
            localdata_id=self.client_id - 1,  # localdata_id starts from 0 while client_id starts from 1.
            other_ids=self._get_other_ids(),
            is_in_childworker=self.is_in_childworker,
            local_train_ratio=self.conf.local_train_ratio,
            batch_size=1,
        )
        # Warning: only for CIFAR and mnist now
        hist = torch.zeros(10)
        train_loader = local_data_loaders["train"]
        for _, _target in train_loader:
            hist[_target.item()] += 1
        self.logger.log(
            f"\tWorker-{self.graph.worker_id} (client-{self.client_id}) training histogram is like {hist}"
        )

        return hist

