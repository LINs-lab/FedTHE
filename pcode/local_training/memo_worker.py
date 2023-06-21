# -*- coding: utf-8 -*-

import numpy as np
import math
from pcode.local_training.base_worker import BaseWorker
import pcode.create_dataset as create_dataset
import pcode.create_optimizer as create_optimizer
from pcode.utils.stat_tracker import RuntimeTracker
import torch.nn as nn
import copy
import torch
from pcode.models.memo_resnet import ResNetCifar
from pcode.datasets.aug_data import aug
from PIL import Image
import pcode.local_training.utils as utils
import torchvision.transforms as trn
"""
Implementation of
    MEMO: Test Time Robustness via Adaptation and Augmentation
    https://openreview.net/pdf?id=XrGEkCOREX2
"""


class MemoWorker(BaseWorker):
    def __init__(self, conf, is_personal=True):
        super(MemoWorker, self).__init__(conf)
        self.n_personalized_epochs = conf.n_personalized_epochs
        self.eval_dataset = conf.eval_dataset
        self.is_in_childworker = True
        self.is_personal = is_personal

    def run(self):
        while True:
            self._listen_to_master()

            # check if we need to terminate the training or not.
            if self._terminate_by_early_stopping():
                return

            # receive global model (note as self.model) and perform standard local training.
            self._recv_model_from_master()

            state = self.train(model=self.model)
            params_to_send = copy.deepcopy(state["model"].state_dict())

            # a personal model to hold local parameters.
            if not hasattr(self, "personal_model"):
                self.personal_model = copy.deepcopy(self.model)

            # evaluate the model.
            self.is_in_personalized_training = True
            if not self.is_personal:
                state = self._init_train_process(self.model)
            else:
                state = self._personalized_train(model=self.model)
            state["optimizer"].lr = 0.0005
            perf = self._evaluate_all_test_sets(state)
            self.is_in_personalized_training = False

            # display the info and sync the model & perf.
            self._display_info(state, perf)
            self._send_model_to_master(params_to_send, perf)
            del state

            if self._terminate_by_complete_training():
                return

    def _personalized_train(self, model):
        self.n_local_epochs = self.n_personalized_epochs
        state = super(MemoWorker, self).train(model)
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
            self._memo_inference(data_batch, state, tracker_te)

            if num >= num_batches: break
            else: num += 1

        return tracker_te()

    def _memo_inference(self, data_batch, state, tracker=None):
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

    def adapt_and_test_single(self, state, image, ori_input, normalize):
        state["model"].eval()
        for _ in range(2):
            inputs = [aug(image, normalize) for _ in range(32)]
            inputs = torch.stack(inputs).cuda()
            state["optimizer"].zero_grad()
            outputs = state["model"](inputs)
            loss, _ = utils.marginal_entropy(outputs)
            loss.backward()
            state["optimizer"].step()

        input = ori_input.unsqueeze(0)
        with torch.no_grad():
            outputs = state["model"](input.cuda())
        return outputs

    def adapt_and_test_batch(self, state, data_batch):
        model_param = copy.deepcopy(state["model"].state_dict())
        batch_outputs = []
        for i in range(data_batch["input"].shape[0]):
            normalize, unnormalize = utils.get_normalization(self.conf.data)
            convert_img = trn.Compose([unnormalize, trn.ToPILImage()])
            image = convert_img(data_batch["input"][i])
            batch_outputs.append(self.adapt_and_test_single(state, image, data_batch["input"][i], normalize))
            state["model"].load_state_dict(model_param)

        batch_outputs = torch.cat(batch_outputs)

        return batch_outputs


