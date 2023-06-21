# -*- coding: utf-8 -*-
import copy
import math
import functools
from typing import List, Dict
import numpy as np
import torch
import torch.nn as nn
import torch.distributed as dist

import pcode.create_model as create_model
import pcode.create_dataset as create_dataset
import pcode.create_scheduler as create_scheduler

import pcode.create_metrics as create_metrics
from pcode.utils.tensor_buffer import TensorBuffer
from pcode.utils.timer import Timer

from pcode.utils.logging import display_training_stat
import pcode.create_optimizer as create_optimizer
from pcode.utils.stat_tracker import RuntimeTracker

import pcode.local_training.utils as utils
from pcode.utils.mathdict import MathDict
from pcode.utils.logging import display_perf
from pcode.datasets.partition_data import DataPartitioner
import pcode.datasets.corr_data as corr_data

"""Base worker of FL algorithms: 
    it has no specific personalized FL techniques, and it will not be evaluated on personalized data.
"""


class BaseWorker(object):
    def __init__(self, conf):
        self.conf = conf
        self.graph = conf.graph
        self.logger = conf.logger
        self.random_state = conf.random_state
        self.is_in_childworker = False

        # some initializations.
        self.rank = self.graph.rank
        self.graph.worker_id = self.graph.rank
        self.device = torch.device("cuda" if self.graph.on_cuda else "cpu")
        self.comm_round = 0

        # define the timer for different operations.
        # if we choose the `train_fast` mode, then we will not track the time.
        self.timer = Timer(
            verbosity_level=1 if conf.track_time and not conf.train_fast else 0,
            log_fn=self.logger.log_metric,
        )

        # create dataset (as well as the potential data_partitioner) for training.
        dist.barrier()
        self.fl_data_cls = create_dataset.FLData(
            conf=self.conf,
            logger=self.logger,
            graph=self.graph,
            random_state=self.random_state,
            batch_size=self.conf.batch_size,
            img_resolution=self.conf.img_resolution,
            corr_seed=self.conf.corr_seed,
            corr_severity=self.conf.corr_severity,
            local_n_epochs=self.conf.local_n_epochs,
            num_workers=self.conf.num_workers,
            pin_memory=self.conf.pin_memory,
        )
        self.fl_data_cls.define_dataset(
            data_name=self.conf.data,
            data_dir=self.conf.data_dir,
            is_personalized=self.conf.is_personalized,
            test_partition_ratio=self.conf.test_partition_ratio,
            extra_arg="cifar10.1" if self.conf.data == "cifar10" else self.conf.natural_shifted_imagenet_type
        )
        self.fl_data_cls.inter_clients_data_partition(
            dataset=self.fl_data_cls.dataset["train"],
            n_clients=self.conf.n_clients,
            partition_data_conf=self.conf.partition_data_conf,
        )
        self.logger.log(f"Worker-{self.graph.worker_id} initialized the data.")

        # define the criterion.
        self.criterion = nn.CrossEntropyLoss(reduction="mean")
        self.logger.log(
            f"Worker-{self.graph.worker_id} initialized dataset/criterion.\n"
        )

        # to determine which learning rate should be used.
        self.is_in_personalized_training = False

        # determine representation length
        self.determine_rep_len()

    def run(self):
        while True:
            self._listen_to_master()

            # check if we need to terminate the training or not.
            if self._terminate_by_early_stopping():
                return

            # receive the perform standard local training.
            self._recv_model_from_master()
            state = self.train(model=self.model)
            params_to_send = state["model"].state_dict()

            # display the info and sync the model & perf.
            self._display_info(state, performance=[])
            self._send_model_to_master(params_to_send, message=[])

            if self._terminate_by_complete_training():
                return

    def _listen_to_master(self):
        # listen to master, related to the function `_activate_selected_clients` in `master.py`.
        msg = torch.zeros((5, self.conf.n_participated))
        dist.broadcast(tensor=msg, src=0)
        (
            self.client_id,
            self.client_prob,
            self.comm_round,
            self.n_local_epochs,
            self.local_batch_size,
        ) = (msg[:, self.graph.rank - 1].cpu().numpy().tolist())
        self.client_id = int(self.client_id)
        self.local_batch_size = int(self.local_batch_size)

        # once we receive the signal, we init for the local training.
        self.arch, self.model = create_model.define_model(
            self.conf, to_consistent_model=False, client_id=self.client_id
        )
        self.model_state_dict = self.model.state_dict()
        self.model_tb = TensorBuffer(list(self.model_state_dict.values()))
        self.metrics = create_metrics.Metrics(self.model, task="classification")
        dist.barrier()

    def _recv_model_from_master(self):
        # related to the function `_send_model_to_selected_clients` in `master.py`
        old_buffer = copy.deepcopy(self.model_tb.buffer)
        dist.recv(self.model_tb.buffer, src=0)
        new_buffer = copy.deepcopy(self.model_tb.buffer)
        self.model_tb.unpack(self.model_state_dict.values())
        self.model.load_state_dict(self.model_state_dict)
        self.model_state_dict = copy.deepcopy(self.model.state_dict())
        self.init_model = self._turn_off_grad(copy.deepcopy(self.model).to(self.device))
        self.logger.log(
            f"Worker-{self.graph.worker_id} (client-{self.client_id}) received the model ({self.arch}) from Master. The model status {'is updated' if old_buffer.norm() != new_buffer.norm() else 'is not updated'}."
        )
        dist.barrier()

    def _init_train_process(self, model):
        model = copy.deepcopy(model)
        model.to(self.graph.device)
        model.train()

        # init the model and dataloader.
        local_data_loaders = self.fl_data_cls.intra_client_data_partition_and_create_dataloaders(
            localdata_id=self.client_id - 1,  # localdata_id starts from 0 while client_id starts from 1.
            other_ids=self._get_other_ids(),
            is_in_childworker=self.is_in_childworker,
            local_train_ratio=self.conf.local_train_ratio,
            batch_size=self.local_batch_size,
        )

        # initialize train and test set
        train_loader = local_data_loaders["train"]
        val_loader = local_data_loaders["validation"]
        test_loader = local_data_loaders["test"]
        corr_test_loader = local_data_loaders["corr_test"]
        mixed_test_loader = local_data_loaders["mixed_test"]
        if "imagenet" in self.conf.data:
            natural_shift_v2_test_loader = local_data_loaders["natural_shift_test"]
            natural_shift_a_test_loader = local_data_loaders["natural_shift_test_a"]
            natural_shift_r_test_loader = local_data_loaders["natural_shift_test_r"]
            ooc_corr_test_loader = None
        elif "cifar10" in self.conf.data:
            natural_shift_test_loader = local_data_loaders["natural_shift_test"]
            ooc_test_loader = local_data_loaders["ooc_test"]
            ooc_corr_test_loader = local_data_loaders["ooc_corr_test"]

        # define optimizer, scheduler and runtime tracker.
        optimizer = create_optimizer.define_optimizer(
            self.conf, model=model, optimizer_name=self.conf.optimizer, lr=self._get_round_lr()
        )
        scheduler = create_scheduler.Scheduler(
            self.conf,
            optimizer=optimizer,
            lr=self._get_round_lr(),
            num_batches_per_device_per_epoch=local_data_loaders[
                "num_batches_per_device_per_epoch"
            ],
            display_status=False,
        )
        tracker = RuntimeTracker(metrics_to_track=self.metrics.metric_names)

        # return the state
        return {
            "model": model,
            "model_state_dict": copy.deepcopy(model.state_dict()),
            "train_loader": train_loader,
            "val_loader": val_loader,
            "test_loader": test_loader,
            "corr_test_loader": corr_test_loader if "cifar10" in self.conf.data else natural_shift_a_test_loader,
            "ooc_test_loader": ooc_test_loader if "cifar10" in self.conf.data else natural_shift_r_test_loader,
            "natural_shift_test_loader": natural_shift_test_loader if "cifar10" in self.conf.data else natural_shift_v2_test_loader,
            "ooc_corr_test_loader": ooc_corr_test_loader,
            "mixed_test_loader": mixed_test_loader,
            "optimizer": optimizer,
            "scheduler": scheduler,
            "tracker": tracker,
        }

    def train(self, model):
        # define dataloader, optimizer, scheduler and tracker
        state = self._init_train_process(model=model)

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

        # terminate
        self._terminate_comm_round(state)

        return state

    def _send_model_to_master(self, params_to_send: Dict, message: List):
        dist.barrier()
        self.logger.log(
            f"Worker-{self.graph.worker_id} (client-{self.client_id}) sending the model ({self.arch}) back to Master."
        )

        # we init a tensor.
        if message is not None:
            message = torch.FloatTensor(message + [0.0] * (self.conf.comm_buffer_size - len(message)))
        else:
            message = torch.FloatTensor([0.0] * self.conf.comm_buffer_size)

        # init the model.
        flatten_model = TensorBuffer(list(params_to_send.values()))
        to_be_sent = torch.cat([flatten_model.buffer.cpu(), message])
        dist.send(tensor=to_be_sent, dst=0)
        dist.barrier()

    def _inference(self, data_batch, model, tracker=None):
        """Inference on the given model and get loss and accuracy."""
        # do the forward pass and get the output.
        output = model(data_batch["input"])

        # evaluate the output and get the loss, performance.
        loss = self.criterion(output, data_batch["target"])
        performance = self.metrics.evaluate(loss, output, data_batch["target"])

        # update tracker.
        if tracker is not None:
            tracker.update_metrics(
                [loss.item()] + performance, n_samples=data_batch["input"].size(0)
            )
        return loss, output

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
                self._inference(data_batch, state["model"], tracker_te)

            if num >= num_batches: break
            else: num+=1

        return tracker_te()

    def _evaluate_model(self, state, dataset_name):
        tracker_te = RuntimeTracker(metrics_to_track=self.metrics.metric_names)

        if dataset_name == "test_loader" and state[dataset_name] is None:
            return [list(tracker_te().values())[1]]
        if dataset_name == "val_loader" and state[dataset_name] is None:
            return [list(tracker_te().values())[1]]
        if dataset_name == "corr_test_loader" and state[dataset_name] is None:
            return [list(tracker_te().values())[1]]
        if dataset_name == "ooc_test_loader" and state[dataset_name] is None:
            return [list(tracker_te().values())[1]]
        if dataset_name == "natural_shift_test_loader" and state[dataset_name] is None:
            return [list(tracker_te().values())[1]]
        if dataset_name == "ooc_corr_test_loader" and state[dataset_name] is None:
            return [list(tracker_te().values())[1]]
        if dataset_name == "mixed_test_loader" and state[dataset_name] is None:
            return [list(tracker_te().values())[1]]

        if dataset_name == "corr_test_loader":
            # multiple times evaluation for corr test data.
            num = 2
            corr_perf = 0
            for i in range(num):
                corr_perf += np.array(list(self._validate(state, dataset_name).values()))
            return [(corr_perf / num).tolist()[1]]  # only care about top1.
        else:
            return [list(self._validate(state, dataset_name).values())[1]]

    def _evaluate_all_test_sets(self, p_state):
        perf = self._evaluate_model(p_state, "test_loader")
        corr_perf = self._evaluate_model(p_state, "corr_test_loader")
        perf.extend(corr_perf)
        label_shift_perf = self._evaluate_model(p_state, "ooc_test_loader")
        perf.extend(label_shift_perf)
        natural_shift_perf = self._evaluate_model(p_state, "natural_shift_test_loader")
        perf.extend(natural_shift_perf)
        label_shift_corr_perf = self._evaluate_model(p_state, "ooc_corr_test_loader")
        perf.extend(label_shift_corr_perf)
        mixed_test_perf = self._evaluate_model(p_state, "mixed_test_loader")
        perf.extend(mixed_test_perf)

        return perf

    def _get_round_lr(self):
        """get a round-wise lr."""
        if self.is_in_personalized_training:
            lr = self.conf.personal_lr
        else:
            lr = self.conf.lr

        if self.conf.round_milestones_ratios is None:
            return lr
        else:
            # extract milestone_ratios
            if not hasattr(self, "_round_milestones"):
                _round_milestones = (
                    [0]
                    + [
                        int(float(x) * self.conf.n_comm_rounds)
                        for x in self.conf.round_milestones_ratios.split(",")
                    ]
                    + [self.conf.n_comm_rounds]
                )
                self._round_milestones = list(
                    zip(_round_milestones[:-1], _round_milestones[1:])
                )

            # get number of decay
            for idx, (l_round_milestone, r_round_milestone) in enumerate(
                self._round_milestones
            ):
                if r_round_milestone > self.comm_round >= l_round_milestone:
                    return lr * (self.conf.lr_decay ** idx)
            return lr * (self.conf.lr_decay ** idx)

    def _turn_off_grad(self, model):
        for param in model.parameters():
            param.requires_grad = False
        return model

    def _terminate_comm_round(self, state):
        state["scheduler"].clean()
        self.logger.save_json()

    def _terminate_by_early_stopping(self):
        if self.comm_round == -1:
            dist.barrier()
            self.logger.log(
                f"Worker-{self.graph.worker_id} finished the FL by early-stopping."
            )
            return True
        else:
            return False

    def _terminate_by_complete_training(self):
        if self.comm_round == self.conf.n_comm_rounds:
            dist.barrier()
            self.logger.log(
                f"Worker-{self.graph.worker_id} finished the FL: (total comm_rounds={self.comm_round})."
            )
            return True
        else:
            return False

    def _is_finished_one_comm_round(self, state):
        return True if state["scheduler"].epoch_ >= self.n_local_epochs else False

    def _get_other_ids(self):
        # function that returns ids of test datasets that we want to test on (useful in OOD case)
        all_clients = list(range(1, self.conf.n_clients + 1))
        all_clients.remove(self.client_id)
        return all_clients

    def _display_info(self, state, performance, extra=None):
        """move logging here."""
        self.logger.log(
            f"Worker-{self.graph.worker_id} (client-{self.client_id}) in local training (current rounds={self.comm_round}"
        )
        self.logger.log(
            f"\tIt has {len(state['train_loader'])} train batches and {len(state['test_loader']) if state['test_loader'] is not None else -1} test batches."
        )
        self.logger.log(f"\tIts local evaluation results: {performance}).")
        self.logger.log(
            f"\tWorker-{self.graph.worker_id} (client-{self.client_id}) finished one round of FL: (comm_round={self.comm_round})."
        )
        # some extra message to display
        if extra is not None:
            self.logger.log(extra)

    def _display_logging(self, state):
        # display the logging info.
        display_training_stat(
            self.conf,
            state["tracker"],
            client_id=self.client_id,
            comm_round=self.comm_round,
            epoch=state["scheduler"].epoch_,
            local_index=state["scheduler"].local_index,
        )

        # display tracking time.
        if (
            self.conf.display_tracked_time
            and state["scheduler"].local_index % self.conf.summary_freq == 0
        ):
            self.logger.log(self.timer.summary())

    def _is_diverge(self, state):
        # check divergence.
        if state["tracker"].stat["loss"].avg > 1e3 or np.isnan(
            state["tracker"].stat["loss"].avg
        ):
            self.logger.log(
                f"Worker-{self.graph.worker_id} (client-{self.client_id}) diverges!!!!!Early stop it."
            )
            return True

    def load_checkpoint(self, path):
        return torch.load(path)

    def check_trainable_grads(self, model):
        for i, param in enumerate(model.named_parameters()):
            print(param[0])
            print(param[1].requires_grad)

    def get_num_batches(self, state, dataset_name):
        if "cifar10" in self.conf.data:
            if dataset_name in ["ooc_test_loader", "ooc_corr_test_loader"]:
                num_batches = math.ceil(len(state[dataset_name]) / (self.conf.n_participated - 1))
            else:
                num_batches = len(state["test_loader"])
        elif "imagenet" in self.conf.data:
            num_batches = len(state[dataset_name])

        return num_batches

    def determine_rep_len(self):
        if self.conf.arch == "simple_cnn":
            self.conf.rep_len = 64
        elif "resnet" in self.conf.arch:
            resnet_size = int(self.conf.arch.replace("resnet", ""))
            if "cifar" in self.conf.data:
                self.conf.rep_len = 64*4 if resnet_size >= 44 else 64
            elif "imagenet" in self.conf.data:
                self.conf.rep_len = 2048 if resnet_size >= 44 else 256
        elif "vision_transformer" in self.conf.arch:
            if "cifar10" in self.conf.data:
                self.conf.rep_len = 64
        elif "vgg" in self.conf.arch:
            if "cifar10" in self.conf.data:
                self.conf.rep_len = 256
        elif "compact_conv_transformer" in self.conf.arch:
            if "cifar10" in self.conf.data:
                self.conf.rep_len = 128
        else:
            raise NotImplementedError


