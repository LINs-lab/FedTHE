# -*- coding: utf-8 -*-
import os
import copy

import numpy as np
from pcode.utils.stat_tracker import RuntimeTracker
import torch
import torch.distributed as dist
import functools
import pcode.master_utils as master_utils
import pcode.create_coordinator as create_coordinator
import pcode.create_aggregator as create_aggregator
import pcode.create_client_sampler as create_client_sampler
import pcode.create_dataset as create_dataset
import pcode.create_metrics as create_metrics
import pcode.create_model as create_model
import pcode.utils.checkpoint as checkpoint
from pcode.utils.tensor_buffer import TensorBuffer
import pcode.utils.loss as loss
from pcode.utils.early_stopping import EarlyStoppingTracker
from pcode.utils.logging import display_perf


class Master(object):
    def __init__(self, conf):
        self.conf = conf
        self.graph = conf.graph
        self.logger = conf.logger
        self.random_state = conf.random_state

        # some initializations.
        self.client_ids = list(range(1, 1 + conf.n_clients))
        self.world_ids = list(range(1, 1 + conf.n_participated))

        # define arch for master and clients.
        self._create_arch()

        # define the criterion and metrics.
        self.criterion = loss.CrossEntropyLoss(reduction="mean")
        self.metrics = create_metrics.Metrics(self.master_model, task="classification")
        self.logger.log("Master initialized model/dataset/criterion/metrics.")

        # define client sampler.
        self.client_sampler = create_client_sampler.ClientSampler(
            random_state=conf.random_state,
            logger=conf.logger,
            n_clients=conf.n_clients,
            n_participated=conf.n_participated,
            local_n_epochs=conf.local_n_epochs,
            min_local_epochs=conf.min_local_epochs,
            batch_size=conf.batch_size,
            min_batch_size=conf.min_batch_size,
        )
        self.logger.log(f"Master initialized the client_sampler.")

        # define data for training/val/test.
        self._create_data()

        # define the aggregators and coordinator.
        self.aggregator = create_aggregator.Aggregator(
            fl_aggregate=self.conf.fl_aggregate,
            model=self.master_model,
            criterion=self.criterion,
            metrics=self.metrics,
            dataset=self.fl_data_cls.dataset,
            test_loaders=self.eval_loaders,
            clientid2arch=self.clientid2arch,
            logger=self.logger,
            global_lr=self.conf.global_lr,
        )
        self.coordinator = create_coordinator.Coordinator(self.metrics)
        self.logger.log("Master initialized the aggregator/coordinator.")

        # to record the perf.
        self.perf = {"round": 0, "global_top1": 0.0, "top1": 0.0}

        # define early_stopping_tracker.
        self.early_stopping_tracker = EarlyStoppingTracker(
            patience=conf.early_stopping_rounds
        )

        # save arguments to disk.
        self.is_finished = False
        checkpoint.save_arguments(conf)

    def _create_arch(self):
        # create master model.
        _, self.master_model = create_model.define_model(
            self.conf, to_consistent_model=False
        )

        # create client model (may have different archs, but not supported yet).
        self.used_client_archs = set(
            [
                create_model.determine_arch(
                    client_id=client_id,
                    n_clients=self.conf.n_clients,
                    arch=self.conf.arch,
                    use_complex_arch=True,
                    arch_info=self.conf.arch_info,
                )
                for client_id in range(1, 1 + self.conf.n_clients)
            ]
        )

        self.logger.log(f"The client will use archs={self.used_client_archs}.")
        self.logger.log("Master created model templates for client models.")
        self.client_models = dict(
            create_model.define_model(self.conf, to_consistent_model=False, arch=arch)
            for arch in self.used_client_archs
        )
        self.clientid2arch = dict(
            (
                client_id,
                create_model.determine_arch(
                    client_id=client_id,
                    n_clients=self.conf.n_clients,
                    arch=self.conf.arch,
                    use_complex_arch=True,
                    arch_info=self.conf.arch_info,
                ),
            )
            for client_id in range(1, 1 + self.conf.n_clients)
        )
        self.logger.log(
            f"Master initialize the clientid2arch mapping relations: {self.clientid2arch}."
        )

    def _create_data(self):
        # create dataset (as well as the potential data_partitioner) for training.
        dist.barrier()
        self.fl_data_cls = create_dataset.FLData(
            logger=self.logger,
            graph=self.graph,
            random_state=self.random_state,
            batch_size=self.conf.batch_size,
            img_resolution=self.conf.img_resolution,
            local_n_epochs=self.conf.local_n_epochs,
            num_workers=self.conf.num_workers,
            pin_memory=self.conf.pin_memory,
        )
        self.fl_data_cls.define_dataset(
            data_name=self.conf.data, data_dir=self.conf.data_dir
        )
        self.fl_data_cls.inter_clients_data_partition(
            dataset=self.fl_data_cls.dataset["train"],
            n_clients=self.conf.n_clients,
            partition_data_conf=self.conf.partition_data_conf,
        )
        self.logger.log("Master initialized the data.")

        # create test loaders.
        # client_id starts from 1 to the # of clients.
        test_loader = self.fl_data_cls.create_dataloader(
            self.fl_data_cls.dataset["test"], shuffle=False
        )
        self.eval_loaders = {0: test_loader}
        self.logger.log("Master initialized the local test data with workers.")

    def run(self):
        # we init the sampling probability from an uniform distribution.
        self.comm_round = 1

        # start training.
        for comm_round in range(1, 1 + self.conf.n_comm_rounds):
            self.comm_round = comm_round
            self.logger.log(
                f"Master starting one round of federated learning: (comm_round={comm_round})."
            )

            # detect early stopping.
            self._check_early_stopping()

            # init the activation tensor and broadcast to all clients (either start or stop).
            self.client_sampler.select_clients(
                model=self.master_model,
                flatten_local_models=None,
                criterion=self.criterion,
                metrics=self.metrics,
            )
            self._activate_selected_clients(
                self.client_sampler.selected_client_ids,
                self.client_sampler.selected_client_probs,
                self.comm_round,
                self.client_sampler.get_n_local_epoch(),
                self.client_sampler.get_n_local_mini_batchsize(),
            )

            # will decide to send the model or stop the training.
            if not self.is_finished:
                # broadcast the model to activated clients.
                self._send_model_to_selected_clients(
                    self.client_sampler.selected_client_ids
                )
            else:
                dist.barrier()
                self.logger.log(
                    f"Master finished the federated learning by early-stopping: (current comm_rounds={self.comm_round}, total_comm_rounds={self.conf.n_comm_rounds})"
                )
                return

            # wait to receive the local models.
            (
                flatten_local_models,
                extra_messages,
            ) = self._receive_models_from_selected_clients(
                self.client_sampler.selected_client_ids
            )

            # aggregate the local models and evaluate on the validation dataset.
            global_top1_perfs = self._aggregate_model_and_evaluate(flatten_local_models)

            # keep tracking the local performance
            self._track_perf(
                extra_messages=extra_messages, global_top1_perfs=global_top1_perfs
            )

            # in case we want to save a checkpoint of model
            self._save_checkpoint()
            self.logger.save_json()

            # evaluate the aggregated model.
            self.logger.log("Master finished one round of federated learning.\n")

        # formally stop the training (the master has finished all communication rounds).
        dist.barrier()
        self._finishing()

    def _save_checkpoint(self):
        if (
            self.conf.save_every_n_round is not None
            and self.comm_round % self.conf.save_every_n_round == 0
        ):
            torch.save(
                self.master_model.state_dict(),
                os.path.join(
                    self.conf.checkpoint_root, f"{self.conf.arch}_{self.comm_round}.pt"
                ),
            )

    def _activate_selected_clients(
        self,
        selected_client_ids,
        selected_client_probs,
        comm_round,
        list_of_local_n_epochs,
        list_of_local_mini_batch_size,
    ):
        # Activate the selected clients:
        # the first row indicates the client id,
        # the second row indicates the current_comm_round,
        # the third row indicates the expected local_n_epochs
        selected_client_ids = np.array(selected_client_ids)
        activation_msg = torch.zeros((5, len(selected_client_ids)))
        activation_msg[0, :] = torch.Tensor(selected_client_ids)
        activation_msg[1, :] = torch.Tensor(list(selected_client_probs.values()))
        activation_msg[2, :] = comm_round
        activation_msg[3, :] = torch.Tensor(list_of_local_n_epochs)
        activation_msg[4, :] = torch.Tensor(list_of_local_mini_batch_size)

        dist.broadcast(tensor=activation_msg, src=0)
        self.logger.log(f"Master activated the selected clients.")
        dist.barrier()

    def _send_model_to_selected_clients(self, selected_client_ids):
        # the master_model can be large; the client_models can be small and different.
        self.logger.log("Master send the models to workers.")
        for worker_rank, selected_client_id in enumerate(selected_client_ids, start=1):
            arch = self.clientid2arch[selected_client_id]
            client_model_state_dict = self.client_models[arch].state_dict()
            flatten_model = TensorBuffer(list(client_model_state_dict.values()))
            dist.send(tensor=flatten_model.buffer, dst=worker_rank)
            self.logger.log(
                f"\tMaster send the current model={arch} to process_id={worker_rank}."
            )
        dist.barrier()

    def _receive_models_from_selected_clients(self, selected_client_ids):
        self.logger.log("Master waits to receive the local models.")
        dist.barrier()

        # init the placeholders to recv the local models from workers.
        flatten_local_models = dict()
        extra_messages = dict()

        for selected_client_id in selected_client_ids:
            arch = self.clientid2arch[selected_client_id]
            client_tb = TensorBuffer(
                list(self.client_models[arch].state_dict().values())
            )
            message = torch.FloatTensor([0.0] * 100)
            client_tb.buffer = torch.cat([torch.zeros_like(client_tb.buffer), message])
            flatten_local_models[selected_client_id] = client_tb

        # async to receive model from clients.
        reqs = {}
        for client_id, world_id in zip(selected_client_ids, self.world_ids):
            req = dist.irecv(
                tensor=flatten_local_models[client_id].buffer, src=world_id
            )
            reqs[client_id] = req

        for client_id, req in reqs.items():
            req.wait()

        for client_id in selected_client_ids:
            extra_messages[client_id] = flatten_local_models[client_id].buffer[-100:]
            flatten_local_models[client_id].buffer = flatten_local_models[
                client_id
            ].buffer[:-100]

        dist.barrier()
        self.logger.log("Master received all local models.")
        return flatten_local_models, extra_messages

    def _avg_over_archs(self, flatten_local_models):
        # get unique arch from this comm. round.
        archs = set(
            [
                self.clientid2arch[client_idx]
                for client_idx in flatten_local_models.keys()
            ]
        )
        assert len(archs) == 1, "we only support the case of same arch."

        # average for each arch.
        archs_fedavg_models = {}
        for arch in archs:
            # extract local_models from flatten_local_models.
            _flatten_local_models = {}
            for client_idx, flatten_local_model in flatten_local_models.items():
                if self.clientid2arch[client_idx] == arch:
                    _flatten_local_models[client_idx] = flatten_local_model

            # average corresponding local models.
            self.logger.log(
                f"Master uniformly average over {len(_flatten_local_models)} received models ({arch})."
            )
            fedavg_model = self.aggregator.aggregate(
                master_model=self.master_model,
                client_models=self.client_models,
                flatten_local_models=_flatten_local_models,
                aggregate_fn_name="_s1_federated_average",
            )
            archs_fedavg_models[arch] = fedavg_model
        return archs_fedavg_models

    def _aggregate_model_and_evaluate(self, flatten_local_models):
        # uniformly average local models with the same architecture.
        fedavg_models = self._avg_over_archs(flatten_local_models)
        fedavg_model = list(fedavg_models.values())[0]

        # update self.master_model in place.
        self.master_model.load_state_dict(fedavg_model.state_dict())
        # update self.client_models in place.
        for arch, _fedavg_model in fedavg_models.items():
            self.client_models[arch].load_state_dict(self.master_model.state_dict())

        # evaluate the aggregated model on the test data.
        perf = master_utils.do_validation(
            self.conf,
            self.coordinator,
            self.master_model,
            self.criterion,
            self.metrics,
            self.eval_loaders,
            split=self.conf.eval_dataset,
            label="global_model",
            comm_round=self.comm_round,
        ).dictionary["top1"]

        return perf

    def _track_perf(self, extra_messages, global_top1_perfs):
        # using the extra_message received from clients to get the ave perf for clients' local evaluations.
        # also track the perf of global model
        self.perf["round"] = self.comm_round
        self.perf["global_top1"] = global_top1_perfs

        # logging.
        display_perf(self.conf, self.perf)

    def _check_early_stopping(self):
        # to use early_stopping checker, we need to ensure patience > 0.
        meet_flag = False

        # consider both of target_perf and early_stopping
        if self.conf.target_perf is not None:
            assert 100 >= self.conf.target_perf > 0

            # meet the target perf.
            if (
                self.coordinator.key_metric.cur_perf is not None
                and self.coordinator.key_metric.cur_perf > self.conf.target_perf
            ):
                self.logger.log("Master early stopping: meet target perf.")
                self.meet_target = True
                meet_flag = True
            # or has no progress and early stop it.
            elif self.early_stopping_tracker(self.coordinator.key_metric.cur_perf):
                self.logger.log(
                    "Master early stopping: not meet target perf but has no patience."
                )
                meet_flag = True

        # only consider the early stopping.
        else:
            if self.early_stopping_tracker(self.coordinator.key_metric.cur_perf):
                meet_flag = True

        if meet_flag:
            # we perform the early-stopping check:
            # (1) before the local training and (2) after the update of the comm_round.
            _comm_round = self.comm_round - 1
            self.comm_round = -1
            self._finishing(_comm_round)

    def _finishing(self, _comm_round=None):
        self.logger.save_json()
        self.logger.log(f"Master finished the federated learning.")
        self.is_finished = True
        self.finished_comm = _comm_round
        checkpoint.save_arguments(self.conf)
        os.system(f"echo {self.conf.checkpoint_root} >> {self.conf.job_id}")
