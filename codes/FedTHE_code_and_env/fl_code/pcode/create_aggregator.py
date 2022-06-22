# -*- coding: utf-8 -*-
import json
import copy
import torch

import pcode.datasets.prepare_data as prepare_data
import pcode.datasets.partition_data as partition_data

import pcode.aggregation.fedavg as fedavg
import pcode.aggregation.gma_fedavg as gma_fedavg


class Aggregator(object):
    def __init__(
        self,
        fl_aggregate,
        model,
        criterion,
        metrics,
        dataset,
        test_loaders,
        clientid2arch,
        logger,
        global_lr=1,
    ):
        self.fl_aggregate = fl_aggregate
        self.logger = logger
        self.model = copy.deepcopy(model)
        self.criterion = criterion
        self.metrics = metrics
        self.dataset = dataset
        self.test_loaders = test_loaders
        self.clientid2arch = clientid2arch
        self.global_lr = global_lr

        # define the aggregation function.
        self._define_aggregate_fn()

    def _define_aggregate_fn(self):
        if (
            self.fl_aggregate is None
            or self.fl_aggregate["scheme"] == "federated_average"
        ):
            self.aggregate_fn = None
        else:
            raise NotImplementedError

    def _s1_federated_average(self):
        # global-wise averaging scheme.
        def f(**kwargs):
            return fedavg.fedavg(
                global_lr=self.global_lr,
                clientid2arch=self.clientid2arch,
                n_selected_clients=kwargs["n_selected_clients"],
                master_model=kwargs["master_model"],
                flatten_local_models=kwargs["flatten_local_models"],
                client_models=kwargs["client_models"],
            )

        return f

    def _gma_fedavg(self):
        def f(**kwargs):
            return gma_fedavg.gma_fedavg(
                global_lr=self.global_lr,
                clientid2arch=self.clientid2arch,
                n_selected_clients=kwargs["n_selected_clients"],
                master_model=kwargs["master_model"],
                flatten_local_models=kwargs["flatten_local_models"],
                client_models=kwargs["client_models"],
            )

        return f

    def aggregate(
        self,
        master_model,
        client_models,
        flatten_local_models,
        aggregate_fn_name=None,
        **kwargs,
    ):
        n_selected_clients = len(flatten_local_models)

        # apply advanced aggregate_fn.
        self.logger.log(
            f"Aggregator via {aggregate_fn_name if aggregate_fn_name is not None else self.fl_aggregate['scheme']}: {f'scheme={json.dumps(self.fl_aggregate)}' if self.fl_aggregate is not None else ''}"
        )
        _aggregate_fn = (
            self.aggregate_fn
            if aggregate_fn_name is None
            else getattr(self, aggregate_fn_name)()
        )
        return _aggregate_fn(
            master_model=master_model,
            client_models=client_models,
            flatten_local_models=flatten_local_models,
            n_selected_clients=n_selected_clients,
            **kwargs,
        )
