# -*- coding: utf-8 -*-
import os
import tarfile
import numpy as np
from PIL import Image

import torch.utils.data as data
import torchvision
from torchvision.datasets.utils import download_url
from torchvision.datasets import ImageFolder


def define_natural_shifted_imagenet_folder(root, data_name, transform):
    return ImageNetValNaturalShift(
        root=root,
        data_name=data_name,
        transform=transform,
    )


class ImageNetValNaturalShift(object):
    """Borrowed from
    (1) https://github.com/hendrycks/imagenet-r/,
    (2) https://github.com/hendrycks/natural-adv-examples,
    (3) https://github.com/modestyachts/ImageNetV2.

    For imagenet_v2, run the following script once before loading the data, to avoid folder name problem.
    for path in glob.glob(your_path_to_imagenetv2):
        if os.path.isdir(path):
            for subpath in glob.glob(f'{path}/*'):
                dirname = subpath.split('/')[-1]
                os.rename(subpath, '/'.join(subpath.split('/')[:-1]) + '/' + dirname.zfill(4))
    """

    stats = {
        "imagenet_r": {
            "data_and_labels": "https://people.eecs.berkeley.edu/~hendrycks/imagenet-r.tar",
            "folder_name": "imagenet-r",
        },
        "imagenet_a": {
            "data_and_labels": "https://people.eecs.berkeley.edu/~hendrycks/imagenet-a.tar",
            "folder_name": "imagenet-a",
        },
        "imagenet_v2_matched-frequency": {
            "data_and_labels": "https://s3-us-west-2.amazonaws.com/imagenetv2public/imagenetv2-matched-frequency.tar.gz",
            "folder_name": "imagenetv2-matched-frequency-format-val",
        },
        "imagenet_v2_threshold0.7": {
            "data_and_labels": "https://s3-us-west-2.amazonaws.com/imagenetv2public/imagenetv2-threshold0.7.tar.gz",
            "folder_name": "imagenetv2-threshold0.7-format-val",
        },
        "imagenet_v2_topimages": {
            "data_and_labels": "https://s3-us-west-2.amazonaws.com/imagenetv2public/imagenetv2-top-images.tar.gz",
            "folder_name": "imagenetv2-topimages-format-val",
        },
    }

    a_cls = [0, 1, 2, 5, 6, 10, 12, 16, 17, 18, 20, 21, 22, 23, 24, 25, 26, 29, 31, 32, 33, 35, 36, 37, 38, 39, 40, 41, 42, 45, 48, 51, 52, 53, 54, 56, 57, 59, 60, 63, 64, 65, 66, 68, 69, 71, 74, 76, 79, 84, 85, 86, 87, 91, 93, 94, 95, 96, 117, 121, 129, 135, 143, 145, 147, 148, 149, 150, 158, 159, 162, 166, 177, 180, 181, 182, 183, 184, 185, 186, 187, 188, 190, 194, 195, 199]
    r_cls = [3, 6, 7, 8, 9, 13, 14, 16, 17, 18, 20, 21, 22, 23, 24, 26, 27, 28, 30, 31, 32, 33, 37, 38, 51, 56, 57, 63, 71, 75, 80, 81, 82, 83, 84, 85, 86, 87, 88, 89, 90, 91, 92, 98, 101, 107, 111, 112, 113, 116, 117, 118, 120, 124, 125, 128, 130, 132, 139, 145, 152, 153, 156, 157, 158, 159, 160, 161, 165, 166, 168, 169, 176, 179, 180, 181, 183, 184, 185, 186, 189, 191, 192, 196, 197, 199]
    v2_cls = [6, 11, 13, 22, 23, 39, 47, 71, 76, 79, 90, 94, 96, 97, 99, 105, 107, 113, 125, 130, 132, 144, 150, 151, 207, 234, 235, 254, 277, 291, 301, 308, 309, 310, 311, 314, 315, 319, 323, 327, 330, 334, 335, 347, 361, 372, 397, 401, 407, 425, 428, 430, 437, 457, 462, 470, 472, 483, 579, 609, 658, 701, 763, 768, 774, 776, 779, 780, 815, 820, 833, 847, 907, 932, 933, 934, 937, 943, 945, 947, 951, 954, 957, 980, 981, 988]

    def __init__(self, root, data_name, transform=None,  version=None):

        self.cls_list = {"imagenet_r": self.r_cls,
                         "imagenet_a": self.a_cls,
                         "imagenet_v2_matched-frequency": self.v2_cls,
                         "imagenet_v2_threshold0.7": self.v2_cls,
                         "imagenet_v2_topimages": self.v2_cls}
        self.data_name = data_name
        self.path_data_and_labels_tar = os.path.join(
            root, self.stats[data_name]["data_and_labels"].split("/")[-1]
        )
        self.path_data_and_labels = os.path.join(
            root, self.stats[data_name]["folder_name"]
        )

        self._download(root)

        self.image_folder = ImageFolder(self.path_data_and_labels)
        indices = [i for i in range(len(self.image_folder.targets)) if self.image_folder.targets[i] in self.cls_list[data_name]]
        self.data = [self.image_folder.samples[ind] for ind in indices]
        self.targets = [self.image_folder.targets[ind] for ind in indices]

        # make the targets consistent between original downsampled set and natual shifted set
        cls_dict = {}
        for i, v in enumerate(self.cls_list[data_name]):
            cls_dict[v] = i  # cls_dict = {6: 0, 11: 1, 13: 2 ...}
        self.targets = [cls_dict[target] for target in self.targets]

        self.transform = transform

    def _download(self, root):
        download_url(url=self.stats[self.data_name]["data_and_labels"], root=root)

        if self._check_integrity():
            print("Files already downloaded, verified, and uncompressed.")
            return
        self._uncompress(root)

    def _uncompress(self, root):
        with tarfile.open(self.path_data_and_labels_tar) as file:
            file.extractall(root)

    def _check_integrity(self) -> bool:
        if os.path.exists(self.path_data_and_labels):
            return True
        else:
            return False

    def __getitem__(self, index):
        path, _ = self.data[index]
        target = self.targets[index]
        img = self.image_folder.loader(path)

        if self.transform is not None:
            img = self.transform(img)

        return img, target

    def __len__(self):
        return len(self.data)
