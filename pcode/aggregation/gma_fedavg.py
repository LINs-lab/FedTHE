# -*- coding: utf-8 -*-
import copy

import torch
from functools import reduce
from pcode.utils.module_state import ModuleState
from pcode.utils.tensor_buffer import TensorBuffer
import pcode.master_utils as master_utils
import pcode.aggregation.utils as agg_utils


def _gma_fedavg(
        global_lr,
        clientid2arch,
        n_selected_clients,
        master_model,
        flatten_local_models,
        client_models,
        agreement_threshold: float,
):
    weights = [
        torch.FloatTensor([1.0 / n_selected_clients]) for _ in range(n_selected_clients)
    ]

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
    master_states = copy.deepcopy(master_model.state_dict())
    local_states = [
        ModuleState(copy.deepcopy(local_model.state_dict()))
        for _, local_model in local_models.items()
    ]
    local_states[0].gen_grad_sign(other_states=master_states)
    raw_mask = torch.Tensor(local_states[0].state_list_sign) * weights[0]
    model_state = local_states[0] * weights[0]
    for idx in range(1, len(local_states)):
        model_state += local_states[idx] * weights[idx]
        # generate sign dict
        local_states[idx].gen_grad_sign(other_states=master_states)
        raw_mask += torch.Tensor(local_states[idx].state_list_sign) * weights[idx]
    model_state.load_state_to_module(_model)
    raw_mask = torch.abs(raw_mask)
    soft_mask = torch.where(raw_mask.double() >= agreement_threshold, 1.0, raw_mask.double())

    # apply global learning rate. use the mask here.
    master_model_tb = TensorBuffer(list(master_model.parameters()))
    _model_params = list(_model.parameters())
    _model_tb = TensorBuffer(_model_params)
    _model_tb.buffer = master_model_tb.buffer - global_lr * soft_mask * (
            master_model_tb.buffer - _model_tb.buffer
    )
    _model_tb.unpack(_model_params)
    return _model


def gma_fedavg(
        global_lr,
        clientid2arch,
        n_selected_clients,
        master_model,
        flatten_local_models,
        client_models,
        agreement_threshold=0.1,
):
    return _gma_fedavg(
        global_lr,
        clientid2arch,
        n_selected_clients,
        master_model,
        flatten_local_models,
        client_models,
        agreement_threshold,
    )