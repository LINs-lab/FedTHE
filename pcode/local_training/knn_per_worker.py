# -*- coding: utf-8 -*-

import copy
import json
import os
import torch
import numpy as np
from typing import List
import pcode.create_dataset as create_dataset
from pcode.local_training.base_worker import BaseWorker
from pcode.utils.stat_tracker import RuntimeTracker
import pcode.local_training.utils as utils

"""
Implementation of
    "Personalized Federated Learning through Local Memorization".
    https://arxiv.org/abs/2111.09360
Some scripts refers to https://github.com/omarfoq/knn-per/blob/main/client.py.
"""


class KNNPerWorker(BaseWorker):
    def __init__(self, conf):
        super(KNNPerWorker, self).__init__(conf)
        self.n_personalized_epochs = conf.n_personalized_epochs
        self.eval_dataset = conf.eval_dataset  # can be test_loader or val_loader
        self.is_in_childworker = True
        self.datastore = {}  # placeholder for data store.
        self.capacity = None  # the maximum number of examples that a data store can maintain.
        self.strategy = "FIFO"  # eliminate strategy. Only useful in their setting.
        self.scale = 1.  # the scale of the gaussian kernel.
        self.dimension = self.conf.rep_len
        self.client_wise_lambda = 0.5  # only a placeholder here.
        self.num_classes = utils.get_num_classes(self.conf.data)
        self.k = 10  # the knn parameter.
        self.lambda_list = np.linspace(0, 1, 101)
        self.eval_round = [100]

    def run(self):
        while True:
            self._listen_to_master()

            # check if we need to terminate the training or not.
            if self._terminate_by_early_stopping():
                return

            # receive the perform standard local training.
            self._recv_model_from_master()

            state = self.train(model=self.model)
            params_to_send = copy.deepcopy(state["model"].state_dict())  # use deepcopy

            # personalization.
            p_state = self._personalized_train(model=self.model)

            # evaluate the personalized model.
            if self.comm_round in self.eval_round:
                max_in_distribution_acc = 0.0
                perf = None
                for curr_lambda in self.lambda_list:
                    self.client_wise_lambda = curr_lambda
                    curr_perf = self._evaluate_all_test_sets(p_state)
                    if curr_perf[0] > max_in_distribution_acc:
                        max_in_distribution_acc = curr_perf[0]
                        perf = curr_perf
            else:
                perf = [0.0] * 6

            # display the personalized perf and sync the model & perf.
            self._display_info(p_state, perf)
            self._send_model_to_master(params_to_send, perf)
            del state, p_state

            if self._terminate_by_complete_training():
                return

    def _personalized_train(self, model):
        # No actual personalized training, only do forward pass and build data store.
        self.is_in_personalized_training = True
        self.n_local_epochs = self.n_personalized_epochs
        # define dataloader, optimizer, scheduler and tracker
        state = self._init_train_process(model=model)
        # freeze the model.
        state["model"].requires_grad_(False)
        state["model"].eval()
        self.rep_layer = utils.determine_hook(state["model"], name=self.conf.arch)
        self.rep_layer.register_forward_hook(utils.hook)
        while not self._is_finished_one_comm_round(state):
            datastore = []
            for _input, _target in state["train_loader"]:
                # load data
                with self.timer("load_data", epoch=state["scheduler"].epoch_):
                    data_batch = create_dataset.load_data_batch(
                        self.conf, _input, _target, is_training=True,
                    )

                # inference and get current performance.
                with self.timer("forward_pass", epoch=state["scheduler"].epoch_):
                    out = state["model"](data_batch["input"])
                    loss = self.criterion(out, data_batch["target"])
                    performance = self.metrics.evaluate(loss, out, data_batch["target"])
                    state["tracker"].update_metrics(
                        [loss.item()] + performance, n_samples=data_batch["input"].size(0)
                    )
                    for i, label in enumerate(data_batch["target"]):
                        datastore.append((self.rep_layer.rep[i, :].unsqueeze(0), label))

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
        self._build_datastore(datastore)
        self._terminate_comm_round(state)
        self.is_in_personalized_training = False
        return state

    def _validate(self, state, dataset_name):
        # switch to evaluation mode.
        state["model"].eval()
        num = 1
        num_batches = self.get_num_batches(state, dataset_name)
        self.rep_layer = utils.determine_hook(state["model"], name=self.conf.arch)
        self.rep_layer.register_forward_hook(utils.hook)

        # evaluate on test_loader.
        tracker_te = RuntimeTracker(metrics_to_track=self.metrics.metric_names)
        for _input, _target in state[dataset_name]:
            # load data and check performance.
            data_batch = create_dataset.load_data_batch(
                self.conf, _input, _target, is_training=False
            )
            with torch.no_grad():
                self._knn_inference(data_batch, state["model"], tracker=tracker_te)

            if num >= num_batches: break
            else: num += 1

        return tracker_te()

    def _knn_inference(self, data_batch, model, tracker=None):
        # do the forward pass and get the output from the model.
        model_output = model(data_batch["input"])
        # get the output from KNN.
        distances, neighbors_labels = self._search_datastore(self.rep_layer.rep)
        similarities = torch.exp(-distances / (self.dimension * self.scale))
        masks = torch.zeros(((self.num_classes,) + similarities.shape)).cuda()
        for class_id in range(self.num_classes):
            masks[class_id] = neighbors_labels == class_id
        knn_output = (similarities * masks).sum(axis=2) / similarities.sum(axis=1)

        # mix the two outputs.
        output = self.client_wise_lambda * model_output + (1 - self.client_wise_lambda) * knn_output.T
        # evaluate the output and get the loss, performance.
        loss = self.criterion(output, data_batch["target"])
        performance = self.metrics.evaluate(loss, output, data_batch["target"])

        # update tracker.
        if tracker is not None:
            tracker.update_metrics(
                [loss.item()] + performance, n_samples=data_batch["input"].size(0)
            )
        return loss, output

    def _build_datastore(self, datastore):
        # if self.capacity is not None, then filter the datastore to size of self.capacity
        self.datastore["features"] = torch.vstack([ent[0] for ent in datastore])
        self.datastore["labels"] = torch.tensor([ent[1] for ent in datastore])

    def _search_datastore(self, query):
        # search for self.k nearest neighbors for a given query.
        dist_matrix = torch.cdist(query, self.datastore["features"], p=2)
        dist, ind = torch.topk(dist_matrix, self.k, dim=1, largest=False, sorted=False)

        return dist, self.datastore["labels"][ind]

    def _clear_datastore(self):
        self.datastore = None

