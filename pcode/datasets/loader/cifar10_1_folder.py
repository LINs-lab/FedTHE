# -*- coding: utf-8 -*-
import os
import numpy as np
from PIL import Image

import torch.utils.data as data
import torchvision
from torchvision.datasets.utils import download_url


def define_cifar10_1_folder(root, transform):
    return CIFAR10Val1(
        root=root,
        transform=transform,
    )


class CIFAR10Val1(object):
    """Borrowed from https://github.com/modestyachts/CIFAR-10.1"""

    stats = {
        "v4": {
            "data": "https://github.com/modestyachts/CIFAR-10.1/raw/master/datasets/cifar10.1_v4_data.npy",
            "labels": "https://github.com/modestyachts/CIFAR-10.1/raw/master/datasets/cifar10.1_v4_labels.npy",
        },
        "v6": {
            "data": "https://github.com/modestyachts/CIFAR-10.1/raw/master/datasets/cifar10.1_v6_data.npy",
            "labels": "https://github.com/modestyachts/CIFAR-10.1/raw/master/datasets/cifar10.1_v6_labels.npy",
        },
    }

    def __init__(self, root, data_name=None, version=None, transform=None):
        version = "v6" if version is None else version
        assert version in ["v4", "v6"]

        self.data_name = data_name
        self.path_data = os.path.join(root, f"cifar10.1_{version}_data.npy")
        self.path_labels = os.path.join(root, f"cifar10.1_{version}_labels.npy")
        self._download(root, version)

        self.data = np.load(self.path_data)
        self.targets = np.load(self.path_labels).tolist()
        self.data_size = len(self.data)

        self.transform = transform

    def _download(self, root, version):
        if self._check_integrity():
            print("Files already downloaded and verified")
            return

        download_url(url=self.stats[version]["data"], root=root)
        download_url(url=self.stats[version]["labels"], root=root)

    def _check_integrity(self) -> bool:
        if os.path.exists(self.path_data) and os.path.exists(self.path_labels):
            return True
        else:
            return False

    def __getitem__(self, index):
        img_array = self.data[index]
        target = self.targets[index]

        if self.transform is not None:
            img = self.transform(img_array)

        return img, target

    def __len__(self):
        return self.data_size
