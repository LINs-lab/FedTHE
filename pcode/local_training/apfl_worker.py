# -*- coding: utf-8 -*-

import numpy as np
import torch
import pcode.create_dataset as create_dataset
from pcode.utils.stat_tracker import RuntimeTracker
import copy
import numpy as np
from typing import List
from pcode.local_training.base_worker import BaseWorker
import pcode.local_training.utils as utils


"""
Implementation of
    Adaptive Personalized Federated Learning
    reference: https://github.com/MLOPTPSU/FedTorch/blob/main/fedtorch/comms/trainings/federated/apfl.py
"""


class APFLWorker(BaseWorker):
    def __init__(self, conf):
        super(APFLWorker, self).__init__(conf)
        self.n_personalized_epochs = conf.n_personalized_epochs
        self.eval_dataset = conf.eval_dataset  # can be test_loader or val_loader
        self.is_in_childworker = True
        self.alpha = 0.5

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

            state, _, _ = self.train(model=self.model, p_model=self.personal_model)
            params_to_send = copy.deepcopy(state["model"].state_dict())

            # personalization. use local trained model and further do fine-tuning.
            p_state = self._personalized_train(model=self.model, p_model=self.personal_model)
            self.personal_model.load_state_dict(copy.deepcopy(p_state["model"].state_dict()))

            # evaluate the personalized model.
            perf = self._evaluate_all_test_sets(p_state)

            # display the personalized perf and sync the model & perf.
            self._display_info(p_state, perf)
            self._send_model_to_master(params_to_send, perf)
            del state, p_state

            if self._terminate_by_complete_training():
                return

    def train(self, model, p_model):
        # regularized local training based on global model
        state = self._init_train_process(model=model)
        p_state = self._init_train_process(model=p_model)
        curr_alpha = self.alpha

        while not self._is_finished_one_comm_round(state):
            for i, (_input, _target) in enumerate(state["train_loader"]):
                # load data
                with self.timer("load_data", epoch=state["scheduler"].epoch_):
                    data_batch = create_dataset.load_data_batch(
                        self.conf, _input, _target, is_training=True
                    )

                # inference and get current performance.
                with self.timer("forward_pass", epoch=state["scheduler"].epoch_):
                    state["optimizer"].zero_grad()
                    output = state["model"](data_batch["input"])
                    # evaluate the output and get the loss, performance.
                    loss = self.criterion(output, data_batch["target"])
                    performance = self.metrics.evaluate(loss, output, data_batch["target"])

                    # update tracker.
                    if state["tracker"] is not None:
                        state["tracker"].update_metrics(
                            [loss.item()] + performance, n_samples=data_batch["input"].size(0)
                        )

                with self.timer("backward_pass", epoch=state["scheduler"].epoch_):
                    loss.backward()
                    state["optimizer"].step()
                    state["scheduler"].step()

                # personalization
                p_state["optimizer"].zero_grad()
                loss, _ = self._inference(data_batch, state["model"], p_state["model"], curr_alpha, p_state["tracker"])

                loss.backward()
                p_state["optimizer"].step()
                p_state["scheduler"].step()

                # update alpha
                if i == 0:  # in their impl., they update alpha per comm round, which is sync gap.
                    curr_alpha = update_alpha(state["model"], p_state["model"], curr_alpha, self._get_round_lr())

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
        return state, p_state, curr_alpha

    def _inference(self, data_batch, model, p_model, alpha, tracker=None):
        """Inference on the given model and get loss and accuracy."""
        output = model(data_batch["input"])
        output_p = p_model(data_batch["input"])
        output_mix = alpha * output + (1 - alpha) * output_p
        loss = self.criterion(output_mix, data_batch["target"])
        performance = self.metrics.evaluate(loss, output_mix, data_batch["target"])

        # update tracker.
        if tracker is not None:
            tracker.update_metrics(
                [loss.item()] + performance, n_samples=data_batch["input"].size(0)
            )
        return loss, output

    def _personalized_train(self, model, p_model):
        self.is_in_personalized_training = True
        self.n_local_epochs = self.n_personalized_epochs
        state, p_state, self.alpha = self.train(model, p_model)
        p_state["g_model"] = state["model"]
        self.is_in_personalized_training = False

        return p_state

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
                self._inference(data_batch, state["g_model"], state["model"], self.alpha, tracker_te)

            if num >= num_batches: break
            else: num+=1

        return tracker_te()


def update_alpha(model_local, model_personal, alpha, eta):
    grad_alpha = 0
    for l_params, p_params in zip(model_local.parameters(), model_personal.parameters()):
        dif = p_params.data - l_params.data
        grad = alpha * p_params.grad.data + (1-alpha)*l_params.grad.data
        grad_alpha += dif.view(-1).T.dot(grad.view(-1))

    grad_alpha += 0.02 * alpha
    alpha_n = alpha - eta*grad_alpha
    alpha_n = np.clip(alpha_n.item(),0.0,1.0)
    return alpha_n