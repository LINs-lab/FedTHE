# -*- coding: utf-8 -*-
import torch.distributed as dist

import pcode.models as models


def define_model(
    conf,
    show_stat=True,
    to_consistent_model=True,
    use_complex_arch=True,
    client_id=None,
    arch=None,
):
    arch, model = define_cv_classification_model(
        conf, client_id, use_complex_arch, arch
    )

    # consistent the model.
    if to_consistent_model:
        consistent_model(model, conf.graph, conf.logger)

    # get the model stat info.
    if show_stat:
        get_model_stat(client_id, model, arch, conf.graph, conf.logger)
    return arch, model


"""define loaders for different models."""


def determine_arch(client_id, n_clients, arch, use_complex_arch, arch_info):
    # the client_id starts from 1.
    _id = client_id if client_id is not None else 0
    if use_complex_arch:
        if _id == 0:
            arch = arch_info["master"]
        else:
            archs = arch_info["worker"]
            if len(arch_info["worker"]) == 1:
                arch = archs[0]
            else:
                assert "num_clients_per_model" in arch_info
                assert arch_info["num_clients_per_model"] * len(archs) == n_clients
                arch = archs[int((_id - 1) / arch_info["num_clients_per_model"])]
    return arch


def define_cv_classification_model(conf, client_id, use_complex_arch, arch):
    # determine the arch.
    arch = (
        determine_arch(
            client_id,
            n_clients=conf.n_clients,
            arch=conf.arch,
            use_complex_arch=use_complex_arch,
            arch_info=conf.arch_info,
        )
        if arch is None
        else arch
    )

    # use the determined arch to init the model.
    if "wideresnet" in arch:
        model = models.__dict__["wideresnet"](conf)
    elif "resnet" in arch and "resnet_evonorm" not in arch:
        model = models.__dict__["resnet"](conf, arch=arch)
    elif "resnet_evonorm" in arch:
        model = models.__dict__["resnet_evonorm"](conf, arch=arch)
    elif "regnet" in arch.lower():
        model = models.__dict__["regnet"](conf, arch=arch)
    elif "densenet" in arch:
        model = models.__dict__["densenet"](conf)
    elif "vgg" in arch:
        model = models.__dict__["vgg"](conf)
    elif "mobilenetv2" in arch:
        model = models.__dict__["mobilenetv2"](conf)
    elif "shufflenetv2" in arch:
        model = models.__dict__["shufflenetv2"](conf, arch=arch)
    elif "efficientnet" in arch:
        model = models.__dict__["efficientnet"](conf)
    elif "federated_averaging_cnn" in arch:
        model = models.__dict__["simple_cnn"](conf)
    elif "moderate_cnn" in arch:
        model = models.__dict__["moderate_cnn"](conf)
    elif "vision_transformer" == arch:
        model = models.__dict__["vision_transformer"](conf)
    elif "vision_transformer_small" in arch:
        model = models.__dict__["vision_transformer_small"](conf)
    elif "compact_conv_transformer" in arch:
        model = models.__dict__["compact_conv_transformer"](conf)
    else:
        model = models.__dict__[arch](conf)
    return arch, model


"""some utilities functions."""


def get_model_stat(client_id, model, arch, graph, logger):
    logger.log(
        "\t=> {} created model '{}. Total params: {}M".format(
            "Master"
            if graph.rank == 0
            else f"Worker-{graph.worker_id} (client-{client_id})",
            arch,
            sum(p.numel() for p in model.parameters() if p.requires_grad) / 1e6,
        )
    )


def consistent_model(model, graph, logger):
    """it might because of MPI, the model for each process is not the same.

    This function is proposed to fix this issue,
    i.e., use the  model (rank=0) as the global model.
    """
    logger.log("\tconsistent model for process (rank {})".format(graph.rank))
    for param in model.parameters():
        param.data = param.data if graph.rank == 0 else param.data - param.data
        dist.all_reduce(param.data, op=dist.ReduceOp.SUM)
