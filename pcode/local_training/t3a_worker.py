# -*- coding: utf-8 -*-
from re import L
import torch
import copy
import functools
import pcode.utils.loss as loss
import torch.distributed as dist
import numpy as np
from typing import List
from pcode.local_training.base_worker import BaseWorker
import pcode.create_dataset as create_dataset
from pcode.utils.stat_tracker import RuntimeTracker
import pcode.local_training.utils as utils
from pcode.utils.mathdict import MathDict
import torch.nn.functional as F

"""
Implementation of
    T3A algorithm from "Test-Time Classifier Adjustment Module for Model-Agnostic Domain Generalization",
    https://proceedings.neurips.cc/paper/2021/file/1415fe9fea0fa1e45dddcff5682239a0-Paper.pdf
    https://github.com/matsuolab/T3A/blob/master/domainbed/adapt_algorithms.py
"""


class T3aWorker(BaseWorker):
    def __init__(self, conf):
        super(T3aWorker, self).__init__(conf)
        self.conf = conf
        self.n_personalized_epochs = conf.n_personalized_epochs
        self.eval_dataset = conf.eval_dataset  # can be test_loader or val_loader
        self.is_in_childworker = True

        # T3A specific
        self.ent = None
        self.supports = None
        self.labels = None
        self.filter_K = conf.t3a_filter_k
        self.num_classes = utils.get_num_classes(conf.data)

    def run(self):
        while True:
            self._listen_to_master()

            # check if we need to terminate the training or not.
            if self._terminate_by_early_stopping():
                return

            # receive the perform standard local training.
            self._recv_model_from_master()

            state = self.train(model=self.model)
            params_to_send = copy.deepcopy(state["model"].state_dict())

            # personalization. use local trained model and further do fine-tuning.
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
        state = super(T3aWorker, self).train(model)
        self.is_in_personalized_training = False
        return state

    def _validate(self, state, dataset_name):
        state["model"].eval()
        self.rep_layer = utils.determine_hook(state["model"], name=self.conf.arch)
        self.rep_layer.register_forward_hook(utils.hook)
        # warmup initialization
        model_param = copy.deepcopy(state["model"].state_dict())
        # get the weight matrix of classifier
        warmup_supports = list(model_param.values())[-1] if self.conf.arch == "simple_cnn" else list(model_param.values())[-2]
        warmup_prob = warmup_supports @ warmup_supports.T
        warmup_ent = -(warmup_prob.softmax(1) * warmup_prob.log_softmax(1)).sum(1)
        warmup_labels = torch.nn.functional.one_hot(warmup_prob.argmax(1), num_classes=self.num_classes).float()

        self.supports = warmup_supports.detach()
        self.labels = warmup_labels.detach()
        self.ent = warmup_ent.detach()
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
                self._t3a_inference(data_batch, state, tracker_te)

            if num >= num_batches:
                break
            else:
                num += 1

        return tracker_te()

    def _t3a_inference(self, data_batch, state, tracker=None):
        batch_outputs = []
        for i in range(data_batch["input"].shape[0]):
            single_output = self.adapt_and_test_single(state, data_batch["input"][i].unsqueeze(0))
            batch_outputs.append(single_output)
        batch_outputs = torch.cat(batch_outputs)
        # evaluate the output and get the loss, performance.
        loss = self.criterion(batch_outputs, data_batch["target"])
        performance = self.metrics.evaluate(loss, batch_outputs, data_batch["target"])
        # update tracker.
        if tracker is not None:
            tracker.update_metrics(
                [loss.item()] + performance, n_samples=data_batch["input"].size(0)
            )

        return loss, batch_outputs

    def adapt_and_test_single(self, state, input):
        p = state["model"](input).detach()
        z = self.rep_layer.rep.detach()
        yhat = torch.nn.functional.one_hot(p.argmax(1), num_classes=self.num_classes).float()
        ent = -(p.softmax(1) * p.log_softmax(1)).sum(1)

        self.supports = torch.cat([self.supports, z])
        self.labels = torch.cat([self.labels, yhat])
        self.ent = torch.cat([self.ent, ent])

        supports, labels = self.select_supports()
        supports = torch.nn.functional.normalize(supports, dim=1)
        weights = (supports.T @ (labels))
        return z @ torch.nn.functional.normalize(weights, dim=0)

    def select_supports(self):
        ent_s = self.ent
        y_hat = self.labels.argmax(dim=1).long()
        filter_K = self.filter_K
        if filter_K == -1:
            indices = torch.LongTensor(list(range(len(ent_s))))

        indices = []
        indices1 = torch.LongTensor(list(range(len(ent_s))))
        for i in range(self.num_classes):
            _, indices2 = torch.sort(ent_s[y_hat == i])
            indices.append(indices1[y_hat==i][indices2][:filter_K])
        indices = torch.cat(indices)

        self.supports = self.supports[indices]
        self.labels = self.labels[indices]
        self.ent = self.ent[indices]

        return self.supports, self.labels




