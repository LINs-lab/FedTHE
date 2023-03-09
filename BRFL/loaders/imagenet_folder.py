# -*- coding: utf-8 -*-
import os
import pickle

import numpy as np
from PIL import Image
import torch.utils.data as data
import torchvision.datasets as datasets
from torchvision.datasets.utils import check_integrity

class ImageNetDS(data.Dataset):
    """`Downsampled ImageNet <https://patrykchrabaszcz.github.io/Imagenet32/>`_ Datasets.
    Args:
        root (string): Root directory of dataset where directory
            ``ImagenetXX_train`` exists.
        img_size (int): Dimensions of the images: 64,32,16,8
        train (bool, optional): If True, creates dataset from training set, otherwise
            creates from test set.
        transform (callable, optional): A function/transform that  takes in an PIL image
            and returns a transformed version. E.g, ``transforms.RandomCrop``
        target_transform (callable, optional): A function/transform that takes in the
            target and transforms it.
    """

    base_folder = "imagenet{}"
    train_list = [
        ["train_data_batch_1", ""],
        ["train_data_batch_2", ""],
        ["train_data_batch_3", ""],
        ["train_data_batch_4", ""],
        ["train_data_batch_5", ""],
        ["train_data_batch_6", ""],
        ["train_data_batch_7", ""],
        ["train_data_batch_8", ""],
        ["train_data_batch_9", ""],
        ["train_data_batch_10", ""],
    ]

    test_list = [["val_data", ""]]

    overlapping_cls = [446, 387, 389, 398, 399, 464, 472, 603, 608, 611, 411, 415, 417, 418, 420, 213, 647, 653, 620, 425, 426, 439, 14, 173, 125, 64, 211, 143, 62, 190, 622, 629, 630, 224, 631, 634, 635, 639, 643, 225, 188, 13, 53, 165, 44, 137, 455, 223, 265, 683, 258, 908, 733, 817, 851, 591, 239, 701, 227, 269, 871, 942, 219, 876, 751, 355, 962, 221, 600, 263, 247, 250, 831, 975, 993, 885, 737, 743, 735, 746, 320, 323, 326, 362, 954, 327]

    def __init__(
            self, root, img_size, train=True, transform=None, target_transform=None, data_name=None,
    ):
        self.root = root
        self.transform = transform
        self.target_transform = target_transform
        self.train = train  # training set or test set
        self.img_size = img_size

        self.base_folder = self.base_folder.format(img_size)

        # now load the picked numpy arrays
        # only use overlap classes
        if self.train:
            self.data = []
            self.targets = []
            for fentry in self.train_list:
                f = fentry[0]
                file = os.path.join(self.root, self.base_folder, f)
                with open(file, "rb") as fo:
                    entry = pickle.load(fo)
                    indices = [i for i in range(len(entry["labels"])) if entry["labels"][i] in self.overlapping_cls]
                    self.data.append([entry["data"][ind] for ind in indices])
                    self.targets += [entry["labels"][ind] for ind in indices]
                    self.mean = entry["mean"]

            self.data = np.concatenate(self.data)
        else:
            f = self.test_list[0][0]
            file = os.path.join(self.root, self.base_folder, f)
            with open(file, "rb") as fo:
                entry = pickle.load(fo)
                indices = [i for i in range(len(entry["labels"])) if entry["labels"][i] in self.overlapping_cls]
                self.data = np.array([entry["data"][ind] for ind in indices])
                self.targets = [entry["labels"][ind] for ind in indices]

        self.data = self.data.reshape((self.data.shape[0], 3, img_size, img_size))
        self.data = self.data.transpose((0, 2, 3, 1))  # convert to HWC

        # make the targets consistent between original downsampled set and natual shifted set
        cls_dict = {}
        for i, v in enumerate(self.overlapping_cls):
            cls_dict[v] = i  # cls_dict = {6: 0, 11: 1, 13: 2 ...}
        self.targets = [cls_dict[target] for target in self.targets]

    def __getitem__(self, index):
        """
        Args:
            index (int): Index
        Returns:
            tuple: (image, target) where target is index of the target class.
        """
        if self.train:
            img, target = self.data[index], self.targets[index]
        else:
            img, target = self.data[index], self.targets[index]

        # doing this so that it is consistent with all other datasets to return a PIL Image
        img = Image.fromarray(img)

        if self.transform is not None:
            img = self.transform(img)

        if self.target_transform is not None:
            target = self.target_transform(target)

        return img, target

    def __len__(self):
        return len(self.data)

    def _check_integrity(self):
        root = self.root
        for fentry in self.train_list + self.test_list:
            filename, md5 = fentry[0], fentry[1]
            fpath = os.path.join(root, self.base_folder, filename)
            if not check_integrity(fpath, md5):
                return False
        return True