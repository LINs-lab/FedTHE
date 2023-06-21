# -*- coding: utf-8 -*-

import copy
import json
import os
import torch
import numpy as np
from typing import List
from pcode.local_training.base_worker import BaseWorker
import pcode.local_training.utils as utils
import pcode.create_dataset as create_dataset
import torch.distributed as dist

"""
Implementation of distributionally robust federated averaging (DRFA),
    https://arxiv.org/abs/2102.12660
    The worker is similar to FedAvg + FT, whereas additional receiving t' and sending loss tensor steps are needed.
"""


class DRFAWorker(BaseWorker):
    def __init__(self, conf, is_fine_tune=False):
        super(DRFAWorker, self).__init__(conf)
        self.n_personalized_epochs = conf.n_personalized_epochs
        self.eval_dataset = conf.eval_dataset  # can be test_loader or val_loader
        self.is_in_childworker = True
        self.is_fine_tune = is_fine_tune

    def run(self):
        while True:
            self._listen_to_master()

            # listen t' from server
            msg = torch.zeros((self.conf.n_participated,), dtype=int)
            dist.broadcast(tensor=msg, src=0)
            dist.barrier()
            self.t_prime = msg[self.graph.rank - 1].cuda()

            # check if we need to terminate the training or not.
            if self._terminate_by_early_stopping():
                return

            # receive the perform standard local training.
            self._recv_model_from_master()

            # a placeholder for t' iteration
            if not hasattr(self, "t_prime_model"):
                self.t_prime_model = copy.deepcopy(self.model)

            state = self.train(model=self.model)
            params_to_send = copy.deepcopy(state["model"].state_dict())  # use deepcopy

            # personalization.
            if self.is_fine_tune:
                p_state = self._personalized_train(model=self.model)
                # evaluate the personalized model.
                perf = self._evaluate_all_test_sets(p_state)
            else:
                p_state = self._init_train_process(self.model)
                perf = self._evaluate_all_test_sets(p_state)

            # display the personalized perf and sync the model & perf.
            self._display_info(p_state, perf)
            self._send_model_to_master(params_to_send, perf)

            # send t_prime_model
            self._send_model_to_master(copy.deepcopy(self.t_prime_model.state_dict()), message=None)
            # receive averaged params
            self._recv_t_prime_model_from_master()
            # do evaluation and get the loss
            loss_tensor = self._eval_model_on_random_batch(self.t_prime_model, state)
            # send the loss tensor
            dist.send(tensor=loss_tensor, dst=0)
            dist.barrier()

            del state, p_state

            if self._terminate_by_complete_training():
                return

    def _eval_model_on_random_batch(self, model, state):
        model.cuda()
        model.eval()
        # state here only offers the dataset
        input, target = next(iter(state["train_loader"]))
        data_batch = create_dataset.load_data_batch(
            self.conf, input, target, is_training=True,
        )
        with torch.no_grad():
            # update test history by exponential moving average.
            output = state["model"](data_batch["input"])
            loss = self.criterion(output, data_batch["target"])
        return loss

    def _recv_t_prime_model_from_master(self):
        # related to the function `_send_model_to_selected_clients` in `master.py`
        old_buffer = copy.deepcopy(self.model_tb.buffer)
        dist.recv(self.model_tb.buffer, src=0)
        new_buffer = copy.deepcopy(self.model_tb.buffer)
        self.model_tb.unpack(self.model_state_dict.values())
        self.t_prime_model.load_state_dict(self.model_state_dict)
        self.logger.log(
            f"Worker-{self.graph.worker_id} (client-{self.client_id}) received the t_prime model ({self.arch}) from Master. The model status {'is updated' if old_buffer.norm() != new_buffer.norm() else 'is not updated'}."
        )
        dist.barrier()

    def train(self, model):
        # define dataloader, optimizer, scheduler and tracker
        state = self._init_train_process(model=model)
        nstep = 0
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

            nstep += 1
            if nstep == self.t_prime:
                self.t_prime_model.load_state_dict(copy.deepcopy(model.state_dict()))
                self.logger.log(
                        f"Client saved t' model at nstep: {nstep}."
                )
        # terminate
        self._terminate_comm_round(state)

        return state

    def _personalized_train(self, model):
        self.is_in_personalized_training = True
        self.n_local_epochs = self.n_personalized_epochs
        state = super(DRFAWorker, self).train(model)
        self.is_in_personalized_training = False
        return state

