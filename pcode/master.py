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
from pcode.datasets.partition_data import DataPartitioner


class Master(object):
    def __init__(self, conf):
        self.conf = conf
        self.graph = conf.graph
        self.logger = conf.logger
        self.random_state = conf.random_state

        # some initializations.
        self.client_ids = list(range(1, 1 + conf.n_clients))
        self.world_ids = list(range(1, 1 + conf.n_participated))
        self.is_in_childworker = False

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
        self.perf = {
            "method": self.conf.personalization_scheme["method"],
            "round": 0,
            "global_top1": 0.0,
            "top1": 0.0,
            "corr_top1": 0.0,
            "ooc_top1": 0.0,
            "natural_shift_top1": 0.0,
            "ooc_corr_top1": 0.0,
            "mixed_top1": 0.0,
        }

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
        # self.conf.comm_buffer_size = self.conf.rep_len + 10

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
        # add an old_client_models here for the purpose of client sampling
        self.old_client_models = dict(
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
        self.logger.log("Master initialized the data.")

        # create test loaders.
        # client_id starts from 1 to the # of clients.
        if self.conf.is_personalized:
            # if personalization is enabled, self.dataset["test"] becomes backup test set.
            # Then we should obtain the clients' validation or test set from merged train set.
            self.eval_loaders = {}
            list_of_local_mini_batch_size = self.client_sampler.get_n_local_mini_batchsize(
                self.client_ids
            )
            eval_datasets = []
            _create_dataloader_fn = functools.partial(
                self.fl_data_cls.create_dataloader, batch_size=list_of_local_mini_batch_size[0], shuffle=True
            )
            local_train_ratio = self.conf.local_train_ratio
            local_test_ratio = (1 - local_train_ratio) / 2
            for client_id in self.client_ids:
                data_to_load = self.fl_data_cls.data_partitioner.use(client_id - 1)
                local_data_partitioner = DataPartitioner(
                    data_to_load,
                    partition_sizes=[
                        local_train_ratio,
                        1 - (local_train_ratio + local_test_ratio),
                        local_test_ratio,
                        ],
                    partition_type="random",
                    partition_alphas=None,
                    consistent_indices=False,
                    random_state=self.random_state,
                    graph=self.graph,
                    logger=self.logger,
                )
                if self.conf.eval_dataset == "val_loader":
                    eval_datasets.append(local_data_partitioner.use(1))
                elif self.conf.eval_dataset == "test_loader":
                    eval_datasets.append(local_data_partitioner.use(2))
            self.eval_loaders = {0: _create_dataloader_fn(torch.utils.data.ConcatDataset(eval_datasets))}

        else:
            test_loader = self.fl_data_cls.create_dataloader(
                self.fl_data_cls.dataset["test"], shuffle=False
            )
            self.eval_loaders = {0: test_loader}
        self.logger.log(f"Master initialized the local test data with workers.")

    def run(self):
        # we init the sampling probability from an uniform distribution.
        self.comm_round = 1

        # initialize lambda for drfa, need to be moved elsewhere in the future.
        if "drfa" in self.conf.personalization_scheme["method"]:
            self.drfa_lambda = np.array([1/self.conf.n_clients] * self.conf.n_clients)

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

            # manually enforce client sampling on master.
            if "drfa" in self.conf.personalization_scheme["method"]:
                # Define online clients for the current round of communication for Federated Learning setting
                master_selected_lambda_idx = self.random_state.choice(
                    self.client_ids, self.conf.n_master_sampled_clients, replace=False
                ).tolist()  # for sampling the lambda
                master_selected_lambda_idx.sort()
                self.logger.log(
                        f"Sanity Check (random): Master sampled lambda idxs are: {master_selected_lambda_idx}."
                )
                self.master_selected_clients_idx = self.random_state.choice(self.client_ids, self.conf.n_master_sampled_clients, replace=False, p=self.drfa_lambda).tolist()
                self.master_selected_clients_idx.sort()
                self.logger.log(
                        f"DRFA: Master sampled client idxs are: {self.master_selected_clients_idx}."
                )
            else:
                # TODO: an more elegant way of handling this.
                self.master_selected_clients_idx = self.random_state.choice(
                    self.client_ids, self.conf.n_master_sampled_clients, replace=False
                ).tolist()
                self.master_selected_clients_idx.sort()
                self.logger.log(
                        f"Master sampled client idxs are: {self.master_selected_clients_idx}."
                )

            self._activate_selected_clients(
                self.client_sampler.selected_client_ids,
                self.client_sampler.selected_client_probs,
                self.comm_round,
                self.client_sampler.get_n_local_epoch(),
                self.client_sampler.get_n_local_mini_batchsize(),
            )

            # method-specific communications, maybe put these in different masters in the future.
            if "THE" in self.conf.personalization_scheme["method"]:
                self._send_global_rep()
            elif "drfa" in self.conf.personalization_scheme["method"]:
                self._send_random_iter_index()

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
            flatten_local_models, extra_messages = self._receive_models_from_selected_clients(
                self.client_sampler.selected_client_ids
            )
            # mask out unselected models as way of naive client sampling.
            # need a more elegant implementation for sampling.
            flatten_local_models = {sel: flatten_local_models[sel] for sel in self.master_selected_clients_idx}

            if "drfa" in self.conf.personalization_scheme["method"]:
                # receive t' models
                t_prime_local_models = self._receive_models_from_selected_clients(
                    self.client_sampler.selected_client_ids
                )
                t_prime_local_models, _ = {sel: t_prime_local_models[sel] for sel in self.master_selected_clients_idx}
                # uniformly average local t_prime_models
                avg_t_prime_models = self._avg_over_archs(t_prime_local_models)
                avg_t_prime_model = list(avg_t_prime_models.values())[0]
                # send
                self.logger.log("Master send the averaged t_prime_models to workers.")
                for worker_rank, selected_client_id in enumerate(self.client_sampler.client_ids, start=1):
                    t_prime_model_state_dict = avg_t_prime_model.state_dict()
                    flatten_model = TensorBuffer(list(t_prime_model_state_dict.values()))
                    dist.send(tensor=flatten_model.buffer, dst=worker_rank)
                dist.barrier()
                # async to receive loss tensor from clients.
                reqs = {}
                drfa_loss_tensor = {id: torch.tensor(0.0) for id in self.client_sampler.selected_client_ids}
                for client_id, world_id in zip(self.client_sampler.selected_client_ids, self.world_ids):
                    req = dist.irecv(
                        tensor=drfa_loss_tensor[client_id], src=world_id
                    )
                    reqs[client_id] = req

                for client_id, req in reqs.items():
                    req.wait()
                dist.barrier()
                filtered_drfa_loss_tensor = torch.zeros(len(self.client_sampler.client_ids))
                for sel in master_selected_lambda_idx:
                    filtered_drfa_loss_tensor[sel - 1] = drfa_loss_tensor[sel]
                self.drfa_lambda += self.conf.drfa_lambda_lr * self.conf.drfa_sync_gap * filtered_drfa_loss_tensor.numpy()
                self.drfa_lambda = master_utils.euclidean_proj_simplex(torch.tensor(self.drfa_lambda)).numpy()
                self.conf.drfa_lambda_lr *= 0.9
                # avoid zero probability
                lambda_zeros = self.drfa_lambda <= 1e-3
                if lambda_zeros.sum() > 0:
                    self.drfa_lambda[lambda_zeros] = 1e-3
                self.drfa_lambda /= np.sum(self.drfa_lambda)
                self.drfa_lambda[-1] = max(0, 1 - np.sum(self.drfa_lambda[0:-1]))  # to avoid round error
                self.logger.log(f"Current lambdas are {self.drfa_lambda}.\n")

            # aggregate the local models and evaluate on the validation dataset.
            global_top1_perfs = self._aggregate_model_and_evaluate(flatten_local_models)

            # keep tracking the local performance
            self._track_perf(extra_messages=extra_messages, global_top1_perfs=global_top1_perfs)

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

    def _send_global_rep(self):
        if not hasattr(self, "global_rep"):
            self.global_rep = torch.ones((self.conf.rep_len,))
        dist.broadcast(tensor=self.global_rep, src=0)
        self.logger.log(f"Master sent global representation to the selected clients.")
        dist.barrier()

    def _send_random_iter_index(self):
        t_prime = torch.tensor(torch.randint(low=1, high=45*int(self.conf.local_n_epochs), size=(1,)).tolist() * len(self.client_sampler.client_ids)) # hard code a 'high'
        dist.broadcast(tensor=t_prime, src=0)
        self.logger.log(f"DRFA: Master sampled iteration t' {t_prime} and send to the selected clients.")
        dist.barrier()

    def _send_model_to_selected_clients(self, selected_client_ids):
        # the master_model can be large; the client_models can be small and different.
        self.logger.log("Master send the models to workers.")
        for worker_rank, selected_client_id in enumerate(selected_client_ids, start=1):
            arch = self.clientid2arch[selected_client_id]
            if selected_client_id in self.master_selected_clients_idx:
                client_model_state_dict = self.client_models[arch].state_dict()
                flatten_model = TensorBuffer(list(client_model_state_dict.values()))
                dist.send(tensor=flatten_model.buffer, dst=worker_rank)
                self.logger.log(
                    f"\tMaster send the current model={arch} to process_id={worker_rank}."
                )
            else:
                client_model_state_dict = self.old_client_models[arch].state_dict()
                flatten_model = TensorBuffer(list(client_model_state_dict.values()))
                dist.send(tensor=flatten_model.buffer, dst=worker_rank)
                self.logger.log(
                    f"\tMaster send the previous model={arch} to process_id={worker_rank}."
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
            message = torch.FloatTensor([0.0] * self.conf.comm_buffer_size)
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
            extra_messages[client_id] = flatten_local_models[client_id].buffer[-self.conf.comm_buffer_size:]
            flatten_local_models[client_id].buffer = flatten_local_models[
                                                         client_id
                                                     ].buffer[:-self.conf.comm_buffer_size]

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
                aggregate_fn_name="_s1_federated_average" if self.conf.personalization_scheme["method"] != "GMA" else "_gma_fedavg",
                weights=self.drfa_lambda if self.conf.personalization_scheme["method"] == "drfa" else None,
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
            # add an old_client_models here for the purpose of client sampling.
            self.old_client_models[arch].load_state_dict(self.client_models[arch].state_dict())
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

        if self.conf.is_personalized:
            # extract local performance from activated clients and average them.
            top1, corr_top1, ooc_top1, natural_shift_top1, ooc_corr_top1, mixed_top1 = 0.0, 0.0, 0.0, 0.0, 0.0, 0.0
            self.global_rep = torch.zeros((self.conf.rep_len,))
            for message in extra_messages.values():
                top1 = top1 + message[0]/len(extra_messages)
                corr_top1 = corr_top1 + message[1]/len(extra_messages)
                ooc_top1 = ooc_top1 + message[2]/len(extra_messages)
                natural_shift_top1 = natural_shift_top1 + message[3]/len(extra_messages)
                ooc_corr_top1 = ooc_corr_top1 + message[4]/len(extra_messages)
                mixed_top1 = mixed_top1 + message[5]/len(extra_messages)
                self.global_rep = self.global_rep + torch.tensor(message[6:6+self.conf.rep_len])/len(extra_messages)

            self.perf["top1"] = top1.item()
            self.perf["corr_top1"] = corr_top1.item()
            self.perf["ooc_top1"] = ooc_top1.item()
            self.perf["natural_shift_top1"] = natural_shift_top1.item()
            self.perf["ooc_corr_top1"] = ooc_corr_top1.item()
            self.perf["mixed_top1"] = mixed_top1.item()
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
