# -*- coding: utf-8 -*-
import torch
import torch.nn as nn
import torch.nn.functional as F
import math

__all__ = ["simple_cnn"]


def _decide_num_classes(dataset):
    if dataset == "cifar10" or dataset == "svhn":
        return 10
    elif dataset == "cifar100":
        return 100
    elif "imagenet" in dataset:
        return 1000
    elif "mnist" == dataset:
        return 10
    elif "femnist" == dataset:
        return 62
    else:
        raise NotImplementedError(f"this dataset ({dataset}) is not supported yet.")


class CNNMnist(nn.Module):
    def __init__(self, dataset, w_conv_bias=False, w_fc_bias=True):
        super(CNNMnist, self).__init__()

        # decide the num of classes.
        self.num_classes = _decide_num_classes(dataset)

        self.conv1 = nn.Conv2d(1, 10, kernel_size=5, bias=w_conv_bias)
        self.conv2 = nn.Conv2d(10, 20, kernel_size=5, bias=w_conv_bias)
        self.conv2_drop = nn.Dropout2d()
        self.fc1 = nn.Linear(320, 50, bias=w_fc_bias)
        self.classifier = nn.Linear(50, self.num_classes, bias=w_fc_bias)

    def forward(self, x):
        x = F.relu(F.max_pool2d(self.conv1(x), 2))
        x = F.relu(F.max_pool2d(self.conv2_drop(self.conv2(x)), 2))
        x = x.view(-1, x.shape[1] * x.shape[2] * x.shape[3])
        x = F.relu(self.fc1(x))
        x = F.dropout(x, training=self.training)
        x = self.classifier(x)
        return x


class CNNfemnist(nn.Module):
    def __init__(
        self, dataset, w_conv_bias=True, w_fc_bias=True, save_activations=True
    ):
        super(CNNfemnist, self).__init__()

        # decide the num of classes.
        self.num_classes = _decide_num_classes(dataset)

        # define layers.
        self.conv1 = nn.Conv2d(1, 32, 5, bias=w_conv_bias)
        self.pool = nn.MaxPool2d(2, 2)
        self.conv2 = nn.Conv2d(32, 64, 5, bias=w_conv_bias)
        self.fc1 = nn.Linear(1024, 2048, bias=w_fc_bias)
        self.classifier = nn.Linear(2048, self.num_classes, bias=w_fc_bias)

        # a placeholder for activations in the intermediate layers.
        self.save_activations = save_activations
        self.activations = None

    def forward(self, x):
        activation1 = self.conv1(x)
        x = self.pool(F.relu(activation1))

        activation2 = self.conv2(x)

        x = self.pool(F.relu(activation2))
        x = x.view(-1, 1024)
        x = F.relu(self.fc1(x))
        x = self.classifier(x)

        if self.save_activations:
            self.activations = [activation1, activation2]
        return x


class CNNCifar(nn.Module):
    """Modify the original CNNCifar to match the ConvNet in FedRoD."""

    def __init__(
        self,
        dataset,
        w_conv_bias=True,
        w_fc_bias=True,
        save_activations=True,
        with_bn=False,
    ):
        super(CNNCifar, self).__init__()

        # decide the num of classes.
        self.num_classes = _decide_num_classes(dataset)
        # use BN or not
        self.with_bn = with_bn

        # define layers.
        self.conv1 = nn.Conv2d(3, 32, 5, bias=w_conv_bias)
        self.pool = nn.MaxPool2d(2, 2)
        self.conv2 = nn.Conv2d(32, 64, 5, bias=w_conv_bias)
        self.fc1 = nn.Linear(64 * 5 * 5, 64, bias=w_fc_bias)
        if self.with_bn:
            self.bn1 = nn.BatchNorm2d(32)
            self.bn2 = nn.BatchNorm2d(64)
        self.classifier = nn.Linear(64, self.num_classes, bias=w_fc_bias)

        # a placeholder for activations in the intermediate layers.
        self.save_activations = save_activations
        self.activations = None
        self.rep = None

    def forward(self, x):
        activation1 = self.conv1(x)
        if self.with_bn:
            x = self.pool(F.relu(self.bn1(activation1)))
            activation2 = self.conv2(x)
            x = self.pool(F.relu(self.bn2(activation2)))
        else:
            x = self.pool(F.relu(activation1))
            activation2 = self.conv2(x)
            x = self.pool(F.relu(activation2))

        x = x.view(-1, 64 * 5 * 5)
        # x = x.reshape(-1, 64 * 5 * 5)
        x = F.relu(self.fc1(x))
        self.rep = x
        x = self.classifier(x)

        if self.save_activations:
            self.activations = [activation1, activation2]
        return x


class MultiheadCNNCifar(nn.Module):
    """Multihead CNN used in FedRoD and TSA"""

    def __init__(
        self,
        dataset,
        num_head=2,
        w_conv_bias=True,
        w_fc_bias=True,
        save_activations=True,
        with_bn=False,
        output_agg="mean",
        rep_dim=64,
    ):
        super(MultiheadCNNCifar, self).__init__()
        assert num_head > 1
        self.num_classes = _decide_num_classes(dataset)
        self.with_bn = with_bn
        self.num_head = num_head
        self.output_agg = output_agg

        # define layers.
        self.conv1 = nn.Conv2d(3, 32, 5, bias=w_conv_bias)
        self.pool = nn.MaxPool2d(2, 2)
        self.conv2 = nn.Conv2d(32, 64, 5, bias=w_conv_bias)
        self.fc1 = nn.Linear(64 * 5 * 5, rep_dim, bias=w_fc_bias)
        if self.with_bn:
            self.bn1 = nn.BatchNorm2d(32)
            self.bn2 = nn.BatchNorm2d(64)

        # self.classifier = nn.ModuleList(
        #     [nn.Linear(64, self.num_classes, bias=False) for _ in range(self.num_head)])
        if num_head == 3:
            self.classifier = nn.ModuleList(
                [
                    nn.Linear(rep_dim, self.num_classes, bias=False),
                    nn.Linear(rep_dim, self.num_classes, bias=False),
                    CosNorm_Classifier(rep_dim, self.num_classes),
                ]
            )
        elif num_head == 2:
            self.classifier = nn.ModuleList(
                [
                    nn.Linear(rep_dim, self.num_classes, bias=False),
                    CosNorm_Classifier(rep_dim, self.num_classes),
                ]
            )
        # self.classifier = nn.ModuleList(
        #     [nn.Sequential(nn.Linear(64 * 5 * 5, 64, bias=w_fc_bias), nn.ReLU(), nn.Linear(64, self.num_classes, bias=False)) for _ in range(self.num_head)])
        # a placeholder for activations in the intermediate layers.
        self.save_activations = save_activations
        self.activations = None

    def forward(self, x):
        activation1 = self.conv1(x)
        if self.with_bn:
            x = self.pool(F.relu(self.bn1(activation1)))
            activation2 = self.conv2(x)
            x = self.pool(F.relu(self.bn2(activation2)))
        else:
            x = self.pool(F.relu(activation1))
            activation2 = self.conv2(x)
            x = self.pool(F.relu(activation2))

        x = x.view(-1, 64 * 5 * 5)
        # x = x.reshape(-1, 64 * 5 * 5)
        x = F.relu(self.fc1(x))
        outs = []
        for ind in range(self.num_head):
            outs.append(self.classifier[ind](x))

        if self.save_activations:
            self.activations = [activation1, activation2]

        if self.output_agg == "mean":
            final_output = torch.stack(outs, dim=1).mean(dim=1)
        elif self.output_agg == "sum":
            final_output = torch.stack(outs, dim=1).sum(dim=1)

        return {
            "logits": torch.stack(outs, dim=1),
            "output": final_output,
            "g_out": self.classifier[0](x),
            "p_out": self.classifier[1](x),
            "representation": x,
        }


class NormedLinear(nn.Module):
    def __init__(self, in_features, out_features):
        super(NormedLinear, self).__init__()
        self.weight = nn.Parameter(torch.Tensor(in_features, out_features))
        self.weight.data.uniform_(-1, 1).renorm_(2, 1, 1e-5).mul_(1e5)

    def forward(self, x):
        out = F.normalize(x, dim=1).mm(F.normalize(self.weight, dim=0))
        return out


class CosNorm_Classifier(nn.Module):
    def __init__(self, in_dims, out_dims, scale=16, margin=0.5, init_std=0.001):
        super(CosNorm_Classifier, self).__init__()
        self.in_dims = in_dims
        self.out_dims = out_dims
        self.scale = scale
        self.margin = margin
        self.weight = nn.Parameter(torch.Tensor(out_dims, in_dims).cuda())
        self.reset_parameters()

    def reset_parameters(self):
        stdv = 1.0 / math.sqrt(self.weight.size(1))
        self.weight.data.uniform_(-stdv, stdv)

    def forward(self, input, *args):
        norm_x = torch.norm(input.clone(), 2, 1, keepdim=True)
        # ex = (norm_x / (1 + norm_x)) * (input / norm_x)
        ex = input / norm_x
        ew = self.weight / torch.norm(self.weight, 2, 1, keepdim=True)
        return torch.mm(self.scale * ex, ew.t())


def simple_cnn(conf):
    dataset = conf.data

    if "cifar" in dataset or dataset == "svhn":
        return CNNCifar(
            dataset,
            w_conv_bias=conf.w_conv_bias,
            w_fc_bias=conf.w_fc_bias,
            with_bn=conf.w_bn,
        )
    elif "mnist" == dataset:
        return CNNMnist(dataset, w_conv_bias=conf.w_conv_bias, w_fc_bias=conf.w_fc_bias)
    elif "femnist" == dataset:
        return CNNfemnist(
            dataset, w_conv_bias=conf.w_conv_bias, w_fc_bias=conf.w_fc_bias
        )
    else:
        raise NotImplementedError(f"not supported yet.")


if __name__ == "__main__":
    personal_model = CNNCifar(
        "cifar10", w_conv_bias=True, w_fc_bias=False, with_bn=True
    )

    print(list(personal_model.state_dict().values())[-1].shape)
