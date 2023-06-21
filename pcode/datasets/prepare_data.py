# -*- coding: utf-8 -*-
import os

import torchvision
import torchvision.datasets as datasets
import torchvision.transforms as transforms
import numpy as np
import torch

import pcode.datasets.loader.imagenet_folder as imagenet_folder
import pcode.datasets.loader.pseudo_imagenet_folder as pseudo_imagenet_folder
from pcode.datasets.loader.svhn_folder import define_svhn_folder
from pcode.datasets.loader.femnist import define_femnist_folder
from pcode.datasets.loader.cifar10_1_folder import define_cifar10_1_folder
from pcode.datasets.loader.imagenet_natural_shift_folder import define_natural_shifted_imagenet_folder
import pcode.utils.op_paths as op_paths

"""the entry for classification tasks."""


def _get_cifar(data_name, root, split, transform, target_transform, download):
    is_train = split == "train"

    # decide normalize parameter.
    if data_name == "cifar10":
        dataset_loader = datasets.CIFAR10
    elif data_name == "cifar100":
        dataset_loader = datasets.CIFAR100
    else:
        raise NotImplementedError(f"invalid data_name={data_name}.")

    # we don't add any transform now because we will merge train and test set.
    transform = transforms.Compose([transforms.ToTensor()])
    return dataset_loader(
        root=root,
        train=is_train,
        transform=transform,
        target_transform=target_transform,
        download=download,
    )


def _get_cifar10_1(data_name, datasets_path, split="test"):
    assert data_name == "cifar10.1"
    assert split == "test"
    transform = transforms.Compose([transforms.ToTensor()])
    root = os.path.join(datasets_path, "cifar10_1")
    return define_cifar10_1_folder(root, transform)


def _get_cinic(root, split, transform, target_transform, download):
    is_train = split == "train"
    transform = transforms.Compose([transforms.ToTensor()])
    # download dataset.
    if download:
        # create the dir.
        op_paths.build_dir(root, force=False)

        # check_integrity.
        is_valid_download = True
        for _type in ["train", "valid", "test"]:
            _path = os.path.join(root, _type)
            if len(os.listdir(_path)) == 10:
                num_files_per_folder = [
                    len(os.listdir(os.path.join(_path, _x))) for _x in os.listdir(_path)
                ]
                num_files_per_folder = [x == 9000 for x in num_files_per_folder]
                is_valid_download = is_valid_download and all(num_files_per_folder)
            else:
                is_valid_download = False

        if not is_valid_download:
            # download.
            torchvision.datasets.utils.download_and_extract_archive(
                url="https://datashare.is.ed.ac.uk/bitstream/handle/10283/3192/CINIC-10.tar.gz",
                download_root=root,
                filename="cinic-10.tar.gz",
                md5=None,
            )
        else:
            print("Files already downloaded and verified.")

    return torchvision.datasets.ImageFolder(root=root, transform=transform)


def _get_stl10(root, split, transform, target_transform, download, img_resolution=None):
    # right now this function is only used for unlabeled dataset.
    is_train = split == "train"
    # TODO
    # define the normalization operation.
    normalize = transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))

    if is_train:
        split = "train+unlabeled"
        transform = transforms.Compose(
            [transforms.RandomHorizontalFlip(), transforms.RandomCrop((96, 96), 4)]
            + (
                [torchvision.transforms.Resize((img_resolution, img_resolution))]
                if img_resolution is not None
                else []
            )
            + [transforms.ToTensor(), normalize]
        )
    else:
        transform = transforms.Compose(
            (
                [torchvision.transforms.Resize((img_resolution, img_resolution))]
                if img_resolution is not None
                else []
            )
            + [transforms.ToTensor(), normalize]
        )
    return datasets.STL10(
        root=root,
        split=split,
        transform=transform,
        target_transform=target_transform,
        download=download,
    )


def _get_svhn(root, split, transform, target_transform, download):
    is_train = split == "train"
    transform = transforms.Compose([transforms.ToTensor()])
    return define_svhn_folder(
        root=root,
        is_train=is_train,
        transform=transform,
        target_transform=target_transform,
        download=download,
    )


def _get_femnist(root, split, transform, target_transform, download):
    is_train = split == "train"
    transform = transforms.Compose([transforms.ToTensor()])
    return define_femnist_folder(
        root=root,
        is_train=is_train,
        transform=transform,
        target_transform=target_transform,
        download=download,
    )


def _get_imagenet(data_name, datasets_path, split):
    is_train = split == "train"
    root = os.path.join(datasets_path, "downsampled_" + data_name)
    transform = transforms.Compose([transforms.ToTensor()])

    return imagenet_folder.ImageNetDS(
        root=root, img_size=32, train=is_train, transform=transform
    )


def _get_natural_shifted_imagenet(data_name, datasets_path, split="test"):
    assert split == "test"
    # root = os.path.join(datasets_path, data_name)
    root = datasets_path
    transform = transforms.Compose([transforms.Resize((32, 32)), transforms.ToTensor()])
    return define_natural_shifted_imagenet_folder(root, data_name, transform)


def _get_pseudo_imagenet(root, split="train", img_resolution=None):
    is_train = split == "train"
    assert is_train
    # TODO
    # define normalize.
    normalize = transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
    # define the transform.
    transform = transforms.Compose(
        [transforms.RandomHorizontalFlip(), transforms.RandomCrop((112, 112), 4)]
        + (
            [transforms.Resize((img_resolution, img_resolution))]
            if img_resolution is not None
            else []
        )
        + [transforms.ToTensor(), normalize]
    )
    # return the dataset.
    return pseudo_imagenet_folder.ImageNetDS(
        root=root, train=is_train, transform=transform
    )


"""the entry for different supported dataset."""


def get_dataset(
    data_name,
    datasets_path,
    split="train",
    transform=None,
    target_transform=None,
    download=True,
    img_resolution=None,
):
    # create data folder if it does not exist.
    root = os.path.join(datasets_path, data_name)

    if data_name == "cifar10" or data_name == "cifar100":
        return _get_cifar(data_name, root, split, transform, target_transform, download)
    elif data_name == "cifar10.1":
        return _get_cifar10_1(data_name, datasets_path, split)
    elif data_name in ["imagenet_r", "imagenet_a", "imagenet_v2_matched-frequency", "imagenet_v2_threshold0.7", "imagenet_v2_topimages"]:
        return _get_natural_shifted_imagenet(data_name, datasets_path, split)
    elif data_name == "cinic":
        return _get_cinic(root, split, transform, target_transform, download)
    elif "stl10" in data_name:
        return _get_stl10(
            root, split, transform, target_transform, download, img_resolution
        )
    elif data_name == "svhn":
        return _get_svhn(root, split, transform, target_transform, download)
    elif data_name == "femnist":
        return _get_femnist(root, split, transform, target_transform, download)
    elif "pseudo_imagenet" in data_name:
        return _get_pseudo_imagenet(root, split, img_resolution)
    elif "imagenet" in data_name:
        return _get_imagenet(data_name, datasets_path, split)
    else:
        raise NotImplementedError
