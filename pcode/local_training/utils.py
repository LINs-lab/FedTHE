# -*- coding: utf-8 -*-
import math
import numpy as np
import torch
import torch.nn as nn
import random
import torchvision.transforms as transforms


def determine_hook(model, name):
    if name == "simple_cnn":
        rep_layer = model.classifier
    elif "resnet" in name:
        rep_layer = model.classifier
    elif "vgg" in name:
        rep_layer = model.classifier
    elif "compact_conv_transformer" in name:
        rep_layer = model.fc
    elif "vision_transformer" in name:
        rep_layer = model.mlp_head
    else:
        raise NotImplementedError
    return rep_layer


def hook(module, input, output):
    module.rep = input[0]  # the input of fc layer


def create_mixed_batch(list_of_batch):
    mixed_batches = []
    batch_size = list_of_batch[0]["input"].shape[0]
    list_of_input = [batch["input"] for batch in list_of_batch]
    list_of_target = [batch["target"] for batch in list_of_batch]
    mixed_batch = torch.cat(list_of_input, dim=0)
    mixed_target = torch.cat(list_of_target, dim=0)
    indices = torch.randperm(mixed_batch.shape[0])
    for i in range(mixed_batch.shape[0]//batch_size):
        curr_ind = indices[i * batch_size: (i+1) * batch_size]
        mixed_batches.append((mixed_batch[curr_ind], mixed_target[curr_ind]))
    return mixed_batches


def marginal_entropy(outputs):
    logits = outputs - outputs.logsumexp(dim=-1, keepdim=True)
    avg_logits = logits.logsumexp(dim=0) - np.log(logits.shape[0])
    min_real = torch.finfo(avg_logits.dtype).min
    avg_logits = torch.clamp(avg_logits, min=min_real)
    return -(avg_logits * torch.exp(avg_logits)).sum(dim=-1), avg_logits


def get_num_classes(data_name):
    if data_name in ["cifar10", "svhn", "cinic"]:
        return 10
    elif data_name == "cifar100":
        return 100
    elif "imagenet" in data_name:
        # return 1000
        return 86
    elif data_name == "femnist":
        return 62

def get_normalization(data_name):
    if data_name == "cifar10":
        normalize = transforms.Normalize((0.4914, 0.4822, 0.4465), (0.247, 0.243, 0.261))
        unnormalize = transforms.Compose([transforms.Normalize(mean=[0., 0., 0.], std=[1/0.247, 1/0.243, 1/0.261]),
                                          transforms.Normalize(mean=[-0.4914, -0.4822, -0.4465], std=[1., 1., 1.])])
        return normalize, unnormalize
    elif data_name == "cifar100":
        normalize = transforms.Normalize((0.5071, 0.4867, 0.4408), (0.2675, 0.2565, 0.2761))
        unnormalize = transforms.Compose([transforms.Normalize(mean=[0., 0., 0.], std=[1/0.2675, 1/0.2565, 1/0.2761]),
                                          transforms.Normalize(mean=[-0.5071, -0.4867, -0.4408], std=[1., 1., 1.])])
        return normalize, unnormalize
    elif "imagenet" in data_name:
        normalize = transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
        unnormalize = transforms.Compose([transforms.Normalize(mean=[0., 0., 0.], std=[1/0.5, 1/0.5, 1/0.5]),
                                          transforms.Normalize(mean=[-0.5, -0.5, -0.5], std=[1., 1., 1.])])
        # normalize = transforms.Normalize((0.4810, 0.4574, 0.4078), (0.2146, 0.2104, 0.2138))
        # unnormalize = transforms.Compose([transforms.Normalize(mean=[0., 0., 0.], std=[1/0.2146, 1/0.2104, 1/0.2138]),
        #                                   transforms.Normalize(mean=[-0.4810, -0.4574, -0.4078], std=[1., 1., 1.])])
        return normalize, unnormalize


def random_reinit_model(conf, model):
    if conf.random_reinit_local_model is None:
        return
    else:
        assert "resnet" in conf.arch or "federated_averaging_cnn" in conf.arch
        names = [
            (_name, _module)
            for _name, _module in model.named_modules()
            if (
                len(list(_module.children())) == 0
                and "bn" not in _name
                and ("conv" in _name or "classifier" in _name)
            )
        ]

        if conf.random_reinit_local_model == "last":
            name_module = names[-1]
            weight_initialization(name_module[1])
        elif "random" in conf.random_reinit_local_model:
            name_module = names[conf.random_state.choice(len(names))]
            weight_initialization(name_module[1])
        else:
            raise NotImplementedError

        conf.logger.log(
            f"Worker-{conf.graph.worker_id} (client-{conf.graph.client_id}) received the model from Master and reinitialize layer={name_module[0]}."
        )


def weight_initialization(m):
    if isinstance(m, nn.Conv2d):
        n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
        m.weight.data.normal_(0, math.sqrt(2.0 / n))
    elif isinstance(m, nn.BatchNorm2d):
        m.weight.data.fill_(1)
        m.bias.data.zero_()
    elif isinstance(m, nn.Linear):
        m.weight.data.normal_(mean=0, std=0.01)
        # torch.nn.init.xavier_uniform(m.weight.data)
