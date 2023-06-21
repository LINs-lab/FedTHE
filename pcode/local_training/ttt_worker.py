# -*- coding: utf-8 -*-

import copy
import json
import os
import torch
import numpy as np
from typing import List
from pcode.local_training.base_worker import BaseWorker
import pcode.local_training.utils as utils
import torch.nn as nn
import pcode.create_dataset as create_dataset
import pcode.create_optimizer as create_optimizer
from pcode.utils.stat_tracker import RuntimeTracker
import torchvision.transforms as trn

"""
Implementation of
    Test-Time Training with Self-Supervision for Generalization under Distribution Shifts.
    https://arxiv.org/abs/1909.13231
"""


class TTTWorker(BaseWorker):
    def __init__(self, conf):
        super(TTTWorker, self).__init__(conf)
        self.n_personalized_epochs = conf.n_personalized_epochs
        self.eval_dataset = conf.eval_dataset  # can be test_loader or val_loader
        self.is_in_childworker = True
        self.rotation_type = 'rand'  # or 'expand'
        self.adapt_iters = 1
        self.ttt_lr = 0.001  # follow their default setting
        self.is_online_setting = True

    def run(self):
        while True:
            self._listen_to_master()

            # check if we need to terminate the training or not.
            if self._terminate_by_early_stopping():
                return

            # receive the perform standard local training.
            self._recv_model_from_master()
            # a personal model to hold local parameters.
            if not hasattr(self, "ss_head"):
                self.ss_head = nn.Linear(self.conf.rep_len, 4)

            state = self.train(model=self.model)
            params_to_send = copy.deepcopy(state["model"].state_dict())  # use deepcopy

            # personalization. use local trained model and further do fine-tuning.
            p_state = self._personalized_train(model=self.model)

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
        # training in multitask fashion (main task + ss task)
        state = self._init_train_process(model=model)
        # prepare for ss head, and add its parameters to optimizer
        self.ss_head.to(self.graph.device)
        state["optimizer"].add_param_group({"params": self.ss_head.parameters()})
        # get the feature layer
        self.rep_layer = utils.determine_hook(state["model"], name=self.conf.arch)
        self.rep_layer.register_forward_hook(utils.hook)

        while not self._is_finished_one_comm_round(state):
            for _input, _target in state["train_loader"]:
                # load data
                with self.timer("load_data", epoch=state["scheduler"].epoch_):
                    data_batch = create_dataset.load_data_batch(
                        self.conf, _input, _target, is_training=True
                    )
                    rotated_batch = self.rotate_batch(data_batch["input"], self.rotation_type)

                # inference and get current performance.
                with self.timer("forward_pass", epoch=state["scheduler"].epoch_):
                    state["optimizer"].zero_grad()
                    loss, _ = self._inference(
                        data_batch, state["model"], state["tracker"]
                    )
                    _ = state["model"](rotated_batch["input"])  # get the rep
                    output_ssh = self.ss_head(self.rep_layer.rep)
                    loss_ssh = self.criterion(output_ssh, rotated_batch["target"])
                    loss += loss_ssh

                with self.timer("backward_pass", epoch=state["scheduler"].epoch_):
                    loss.backward()
                    state["optimizer"].step()
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

    def _validate(self, state, dataset_name):
        # switch to evaluation mode.
        state["model"].eval()
        num = 1
        num_batches = self.get_num_batches(state, dataset_name)

        # prepare the optmizer for TTT
        # get the feature layer
        self.rep_layer = utils.determine_hook(state["model"], name=self.conf.arch)
        self.rep_layer.register_forward_hook(utils.hook)
        state["optimizer"] = torch.optim.SGD(list(state["model"].parameters())+list(self.ss_head.parameters()), lr=0.001)

        model_param = copy.deepcopy(state["model"].state_dict())
        head_param = copy.deepcopy(self.ss_head.state_dict())

        # evaluate on test_loader.
        tracker_te = RuntimeTracker(metrics_to_track=self.metrics.metric_names)
        for _input, _target in state[dataset_name]:
            # load data and check performance.
            data_batch = create_dataset.load_data_batch(
                self.conf, _input, _target, is_training=False
            )

            self._ttt_inference(data_batch, state, tracker_te)

            if num >= num_batches: break
            else: num += 1

        state["model"].load_state_dict(model_param)
        self.ss_head.load_state_dict(head_param)

        return tracker_te()

    def _ttt_inference(self, data_batch, state, tracker=None):
        batch_outputs = self.adapt_and_test_batch(state, data_batch)
        # evaluate the output and get the loss, performance.
        loss = self.criterion(batch_outputs, data_batch["target"])
        performance = self.metrics.evaluate(loss, batch_outputs, data_batch["target"])
        # update tracker.
        if tracker is not None:
            tracker.update_metrics(
                [loss.item()] + performance, n_samples=data_batch["input"].size(0)
            )

        return loss, batch_outputs

    def adapt_and_test_single(self, state, ori_input):
        state["model"].train()  # follow their default setting
        self.ss_head.train()  # follow their default setting
        # if confident enough, dont adapt
        for iteration in range(self.adapt_iters):
            # duplicate input to a batch
            ori_inputs = ori_input.repeat(self.conf.batch_size, 1, 1, 1)
            rotated_inputs = self.rotate_batch(ori_inputs, self.rotation_type)
            state["optimizer"].zero_grad()
            _ = state["model"](rotated_inputs["input"])
            outputs = self.ss_head(self.rep_layer.rep)
            loss = self.criterion(outputs, rotated_inputs["target"])
            loss.backward()
            state["optimizer"].step()

        input = ori_input.unsqueeze(0)
        with torch.no_grad():
            outputs = state["model"](input.cuda())
        return outputs

    def adapt_and_test_batch(self, state, data_batch):
        model_param = copy.deepcopy(state["model"].state_dict())
        head_param = copy.deepcopy(self.ss_head.state_dict())
        batch_outputs = []
        for i in range(data_batch["input"].shape[0]):
            batch_outputs.append(self.adapt_and_test_single(state, data_batch["input"][i]))
            if not self.is_online_setting:
                state["model"].load_state_dict(model_param)
                self.ss_head.load_state_dict(head_param)

        batch_outputs = torch.cat(batch_outputs)

        return batch_outputs

    def rotate_batch(self, batch, label):
        if label == 'rand':
            labels = torch.randint(4, (len(batch),), dtype=torch.long)
        elif label == 'expand':
            labels = torch.cat([torch.zeros(len(batch), dtype=torch.long),
                                torch.zeros(len(batch), dtype=torch.long) + 1,
                                torch.zeros(len(batch), dtype=torch.long) + 2,
                                torch.zeros(len(batch), dtype=torch.long) + 3])
            batch = batch.repeat((4,1,1,1))
        else:
            assert isinstance(label, int)
            labels = torch.zeros((len(batch),), dtype=torch.long) + label
        return {"input": rotate_batch_with_labels(batch, labels), "target": labels.cuda()}


def tensor_rot_90(x):
    return x.flip(2).transpose(1, 2)

def tensor_rot_180(x):
    return x.flip(2).flip(1)

def tensor_rot_270(x):
    return x.transpose(1, 2).flip(2)

def rotate_batch_with_labels(batch, labels):
    images = []
    for img, label in zip(batch, labels):
        if label == 1:
            img = tensor_rot_90(img)
        elif label == 2:
            img = tensor_rot_180(img)
        elif label == 3:
            img = tensor_rot_270(img)
        images.append(img.unsqueeze(0))
    return torch.cat(images)






