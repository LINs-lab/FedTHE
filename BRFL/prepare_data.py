# -*- coding: utf-8 -*-
import os

import torchvision
import torchvision.datasets as datasets
import torchvision.transforms as transforms
import numpy as np
import torch

import loaders.imagenet_folder as imagenet_folder
from loaders.cifar10_1_folder import define_cifar10_1_folder
from loaders.imagenet_variant_folder import define_imagenet_variant_folder

"""The entry for different supported dataset."""

def _get_cifar(data_name, root, split, download):
    is_train = split == "train"

    if data_name == "cifar10":
        dataset_loader = datasets.CIFAR10
    elif data_name == "cifar10_1":
        dataset_loader = define_cifar10_1_folder
        assert split == "test"
    else:
        raise NotImplementedError(f"invalid data_name={data_name}.")

    # we don't add any transform now because we will merge train and test set.
    transform = transforms.Compose([transforms.ToTensor()])
    return dataset_loader(
        root=root,
        train=is_train,
        transform=transform,
        download=download,
    )


def _get_imagenet(data_name, root, split):
    is_train = split == "train"

    if data_name == "imagenet32":
        dataset_loader = imagenet_folder.ImageNetDS
        transform = transforms.Compose([transforms.ToTensor()])
    elif data_name in ["imagenet_r", "imagenet_a", "imagenet_v2_matched-frequency", "imagenet_v2_threshold0.7", "imagenet_v2_topimages"]:
        dataset_loader = define_imagenet_variant_folder
        transform = transforms.Compose([transforms.Resize((32, 32)), transforms.ToTensor()])
        assert split == "test"
    else:
        raise NotImplementedError(f"invalid data_name={data_name}.")

    return dataset_loader(
        root=root,
        transform=transform,
        train=is_train,
        data_name=data_name,
        img_size=32
    )


def get_dataset(
    data_name,
    datasets_path,
    split="train",
    download=True,
):
    # create data folder if it does not exist.
    root = os.path.join(datasets_path, data_name)

    if data_name in ["cifar10", "cifar10_1"]:
        return _get_cifar(data_name, root, split, download)
    elif data_name in ["imagenet32", "imagenet_r", "imagenet_a", "imagenet_v2_matched-frequency", "imagenet_v2_threshold0.7", "imagenet_v2_topimages"]:
        return _get_imagenet(data_name, root, split)
    else:
        raise NotImplementedError


def transform_data_batch(_input, _target, is_training=True):
    # Do the transform right before feeding in the model, such that only local training data is augmented.
    transform = _get_transform(is_training=is_training)
    _input = transform(_input)
    if torch.cuda.is_available():
        _input, _target = _input.cuda(), _target.cuda()
    return _input, _target


def _get_transform(is_training):
    normalize = transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
    if is_training:
        transform = transforms.Compose(
            [
                transforms.RandomCrop(32, padding=4),
                transforms.RandomHorizontalFlip(),
                normalize,
            ]
        )
    else:
        transform = transforms.Compose([normalize])

    return transform
