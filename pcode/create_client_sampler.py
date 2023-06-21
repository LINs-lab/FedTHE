# -*- coding: utf-8 -*-
import copy
import numpy as np
import torch

import pcode.master_utils as master_utils
import pcode.create_dataset as create_dataset
import pcode.utils.tensor_buffer as tensor_buffer
from pcode.utils.stat_tracker import SimpleTracker


class ClientSampler(object):
    def __init__(
        self,
        random_state,
        logger,
        n_clients,
        n_participated,
        local_n_epochs,
        min_local_epochs,
        batch_size,
        min_batch_size,
        client_sampling_method="uniform",
    ):
        # init
        self.logger = logger
        self.random_state = random_state
        self.n_clients = n_clients
        self.client_ids = list(range(1, 1 + n_clients))
        self.n_participated = n_participated
        self.local_n_epochs = local_n_epochs
        self.min_local_epochs = min_local_epochs
        self.batch_size = batch_size
        self.min_batch_size = min_batch_size
        self.sampling_method = client_sampling_method

        self.tracker = SimpleTracker(things_to_track=["probs"])
        self._selected_client_ids, self._selected_client_probs = None, None

    def select_clients(self, model, flatten_local_models, criterion, metrics):
        # assign values.
        self._model = model
        self._flatten_local_models = flatten_local_models
        self._criterion = criterion
        self._metrics = metrics

        # use different sampling methods.
        self.probs = np.array([1.0 / self.n_clients for _ in range(self.n_clients)])
        client_ids, client_probs = self._select_clients_via_probability(
            probs=self.probs
        )

        assert type(client_ids) is list
        self._selected_client_ids = client_ids
        self._selected_client_probs = client_probs
        del self._model, self._flatten_local_models

        # store client sampling results.
        self.tracker.update_metrics(metric_stat=[self.probs.tolist()])

    def _select_clients_via_probability(self, probs=None):
        self.logger.log(
            f"Master selected {self.n_participated} via {self.sampling_method} from {self.n_clients} clients with probs={probs}, summed to {sum(probs) if probs is not None else None}."
        )
        selected_client_ids = self.random_state.choice(
            self.client_ids, self.n_participated, replace=False, p=probs
        ).tolist()
        selected_client_ids.sort()
        self.logger.log(
            f"Master selected {self.n_participated} from {self.n_clients} clients with probs={probs}: {selected_client_ids}"
        )
        return (
            selected_client_ids,
            dict(
                (client_id, probs[client_id - 1]) for client_id in selected_client_ids
            ),
        )

    def restart(self):
        self._selected_client_ids = None
        client_ids, client_probs = self._select_clients_via_probability(
            probs=[1.0 / self.n_clients] * self.n_clients
        )
        self._selected_client_ids = client_ids
        self._selected_client_probs = client_probs

        # store client sampling results.
        self.tracker.update_metrics(metric_stat=[self.probs.tolist()])

    def get_n_local_epoch(self, selected_ids=None):
        selected_ids = (
            self._selected_client_ids if selected_ids is None else selected_ids
        )

        if self.min_local_epochs is None:
            return [self.local_n_epochs] * len(selected_ids)
        else:
            # here we only consider to (uniformly) randomly sample the local epochs.
            assert self.min_local_epochs >= 1.0
            self.random_local_n_epochs = self.random_state.uniform(
                low=self.min_local_epochs, high=self.local_n_epochs, size=self.n_clients
            )
            return [
                self.random_local_n_epochs[client_id - 1] for client_id in selected_ids
            ]

    def get_n_local_mini_batchsize(self, selected_ids=None):
        selected_ids = (
            self._selected_client_ids if selected_ids is None else selected_ids
        )

        if self.min_batch_size is None:
            return [self.batch_size] * len(selected_ids)
        else:
            # here we only consider to (uniformly) randomly sample the local epochs.
            assert self.min_batch_size >= 1
            self.random_local_mini_batch_size = self.random_state.uniform(
                low=self.min_batch_size, high=self.batch_size, size=self.n_clients
            )
            return [
                self.random_local_mini_batch_size[client_id - 1]
                for client_id in selected_ids
            ]

    @property
    def selected_client_ids(self):
        return self._selected_client_ids

    @property
    def selected_client_probs(self):
        return self._selected_client_probs
