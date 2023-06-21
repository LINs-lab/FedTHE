# -*- coding: utf-8 -*-
import copy

import torch

from pcode.utils.module_state import ModuleState
from pcode.utils.tensor_buffer import TensorBuffer
import pcode.master_utils as master_utils
import pcode.aggregation.utils as agg_utils


def _fedavg(
    global_lr,
    clientid2arch,
    n_selected_clients,
    master_model,
    flatten_local_models,
    client_models,
    weights = None,
):
    if weights is None:
        weights = [
            torch.FloatTensor([1.0 / n_selected_clients]) for _ in range(n_selected_clients)
        ]
    else:
        print(f"aggregation weights are {weights}")

    # NOTE: the arch for different local models needs to be the same as the master model.
    # retrieve the local models.
    local_models = {}
    for client_idx, flatten_local_model in flatten_local_models.items():
        _arch = clientid2arch[client_idx]
        _model = copy.deepcopy(client_models[_arch])
        _model_state_dict = _model.state_dict()
        flatten_local_model.unpack(_model_state_dict.values())
        _model.load_state_dict(_model_state_dict)
        local_models[client_idx] = _model

    # uniformly average the local models.
    # assume we use the runtime stat from the last model.
    _model = copy.deepcopy(_model)
    local_states = [
        ModuleState(copy.deepcopy(local_model.state_dict()))
        for _, local_model in local_models.items()
    ]
    model_state = local_states[0] * weights[0]
    for idx in range(1, len(local_states)):
        model_state += local_states[idx] * weights[idx]
    model_state.load_state_to_module(_model)

    # apply global learning rate.
    master_model_tb = TensorBuffer(list(master_model.parameters()))
    _model_params = list(_model.parameters())
    _model_tb = TensorBuffer(_model_params)
    _model_tb.buffer = master_model_tb.buffer - global_lr * (
        master_model_tb.buffer - _model_tb.buffer
    )
    _model_tb.unpack(_model_params)
    return _model


def fedavg(
    global_lr,
    clientid2arch,
    n_selected_clients,
    master_model,
    flatten_local_models,
    client_models,
):
    return _fedavg(
        global_lr,
        clientid2arch,
        n_selected_clients,
        master_model,
        flatten_local_models,
        client_models,
    )
