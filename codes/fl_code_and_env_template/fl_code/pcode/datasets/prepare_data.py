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
import pcode.utils.op_paths as op_paths

"""the entry for classification tasks."""


def _get_cifar(data_name, root, split, transform, target_transform, download):
    is_train = split == "train"

    # decide normalize parameter.
    if data_name == "cifar10":
        dataset_loader = datasets.CIFAR10
        # normalize = transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010))
        normalize = transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
    elif data_name == "cifar100":
        dataset_loader = datasets.CIFAR100
        # normalize = transforms.Normalize((0.5071, 0.4867, 0.4408), (0.2675, 0.2565, 0.2761))
        normalize = transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
    else:
        raise NotImplementedError(f"invalid data_name={data_name}.")

    # decide data type.
    if is_train:
        transform = transforms.Compose(
            [
                transforms.RandomHorizontalFlip(),
                transforms.RandomCrop((32, 32), 4),
                transforms.ToTensor(),
                normalize,
            ]
        )
    else:
        transform = transforms.Compose([transforms.ToTensor(), normalize])
    return dataset_loader(
        root=root,
        train=is_train,
        transform=transform,
        target_transform=target_transform,
        download=download,
    )


def _get_cinic(root, split, transform, target_transform, download):
    is_train = split == "train"

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

    # decide normalize parameter.
    normalize = transforms.Normalize(
        mean=(0.47889522, 0.47227842, 0.43047404),
        std=(0.24205776, 0.23828046, 0.25874835),
    )

    # decide data type.
    if is_train:
        transform = transforms.Compose(
            [
                transforms.RandomHorizontalFlip(),
                transforms.RandomCrop((32, 32), 4),
                transforms.ToTensor(),
                normalize,
            ]
        )
    else:
        transform = transforms.Compose([transforms.ToTensor(), normalize])
    return torchvision.datasets.ImageFolder(root=root, transform=transform)


def _get_mnist(root, split, transform, target_transform, download):
    is_train = split == "train"
    normalize = transforms.Normalize((0.1307,), (0.3081,))

    transform = transforms.Compose([transforms.ToTensor(), normalize])
    return datasets.MNIST(
        root=root,
        train=is_train,
        transform=transform,
        target_transform=target_transform,
        download=download,
    )


def _get_stl10(root, split, transform, target_transform, download, img_resolution=None):
    # right now this function is only used for unlabeled dataset.
    is_train = split == "train"

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
    normalize = transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))

    transform = transforms.Compose([transforms.ToTensor(), normalize])
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


def _get_imagenet(data_name, datasets_path, split, use_lmdb_data=True):
    is_train = split == "train"
    is_downsampled = any([s in data_name for s in ["8", "16", "32", "64"]])
    root = os.path.join(
        datasets_path, "lmdb" if not is_downsampled else "downsampled_lmdb"
    )

    # get transform for is_downsampled=True.
    if is_downsampled:
        normalize = transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))

        if is_train:
            transform = transforms.Compose(
                [
                    transforms.RandomCrop(32, padding=4),
                    transforms.RandomHorizontalFlip(),
                    transforms.ToTensor(),
                    normalize,
                ]
            )
        else:
            transform = transforms.Compose([transforms.ToTensor(), normalize])
    else:
        transform = None

    # safety check.
    assert use_lmdb_data and not is_downsampled
    assert not use_lmdb_data and is_downsampled

    # load.
    if use_lmdb_data:
        if is_train:
            root = os.path.join(
                root, "{}train.lmdb".format(data_name + "_" if is_downsampled else "")
            )
        else:
            root = os.path.join(
                root, "{}val.lmdb".format(data_name + "_" if is_downsampled else "")
            )
        return imagenet_folder.define_imagenet_folder(
            name=data_name,
            root=root,
            flag=True,
            transform=transform,
            is_image=True and not is_downsampled,
        )
    else:
        return imagenet_folder.ImageNetDS(
            root=root, img_size=int(data_name[8:]), train=is_train, transform=transform
        )


def _get_pseudo_imagenet(root, split="train", img_resolution=None):
    is_train = split == "train"
    assert is_train

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


def get_cifar10c(root, split="test", shuffle=False, severity=5):
    """only for test purpose"""
    import requests
    from tqdm import tqdm
    import shutil

    def download_file(url: str, save_dir: str, total_bytes: int) -> str:
        """Downloads large files from the given URL.
        From: https://stackoverflow.com/a/16696317
        :param url: The URL of the file.
        :param save_dir: The directory where the file should be saved.
        :param total_bytes: The total bytes of the file.
        :return: The path to the downloaded file.
        """
        CHUNK_SIZE = 65536
        local_filename = save_dir + url.split("/")[-1]
        print(f"Starting download from {url}")
        with requests.get(url, stream=True) as r:
            r.raise_for_status()
            with open(local_filename, "wb") as f:
                iters = total_bytes // CHUNK_SIZE
                for chunk in tqdm(r.iter_content(chunk_size=CHUNK_SIZE), total=iters):
                    f.write(chunk)

        return local_filename

    corruptions = [
        "gaussian_noise",
        "shot_noise",
        "impulse_noise",
        "defocus_blur",
        "glass_blur",
        "motion_blur",
        "zoom_blur",
        "snow",
        "frost",
        "fog",
        "brightness",
        "contrast",
        "elastic_transform",
        "pixelate",
        "jpeg_compression",
    ]
    data_dir = root
    n_total_cifar = 10000
    n_examples = 10000
    assert 1 <= severity <= 5
    assert n_examples <= n_total_cifar

    if not os.path.exists(data_dir):
        os.makedirs(data_dir)

        if not os.path.exists(data_dir):
            os.makedirs(data_dir)

    data_root_dir = data_dir + "/CIFAR-10-C/"

    corruption_links = ("2535967", {"CIFAR-10-C.tar"})
    ZENODO_ENTRY_POINT = "https://zenodo.org/api"
    RECORDS_ENTRY_POINT = f"{ZENODO_ENTRY_POINT}/records/"
    url = f"{RECORDS_ENTRY_POINT}/{corruption_links[0]}"
    res = requests.get(url)
    files = res.json()["files"]
    files_to_download = list(
        filter(lambda file: file["key"] in corruption_links[1], files)
    )

    for file in files_to_download:
        file_url = file["links"]["self"]
        file_checksum = file["checksum"].split(":")[-1]
        filename = download_file(file_url, data_dir, file["size"])
        print("Download finished, extracting...")
        shutil.unpack_archive(filename, extract_dir=data_dir, format=file["type"])
        print("Downloaded and extracted.")

    # Download labels
    labels_path = data_root_dir + "labels.npy"
    labels = np.load(labels_path)

    x_test_list, y_test_list = [], []
    n_pert = len(corruptions)
    for corruption in corruptions:
        corruption_file_path = data_root_dir + (corruption + ".npy")

        images_all = np.load(corruption_file_path)
        images = images_all[(severity - 1) * n_total_cifar : severity * n_total_cifar]
        n_img = int(np.ceil(n_examples / n_pert))
        x_test_list.append(images[:n_img])
        # Duplicate the same labels potentially multiple times
        y_test_list.append(labels[:n_img])

    x_test, y_test = np.concatenate(x_test_list), np.concatenate(y_test_list)
    if shuffle:
        rand_idx = np.random.permutation(np.arange(len(x_test)))
        x_test, y_test = x_test[rand_idx], y_test[rand_idx]

    # Make it in the PyTorch format
    x_test = np.transpose(x_test, (0, 3, 1, 2))
    # Make it compatible with our models
    x_test = x_test.astype(np.float32) / 255
    # Make sure that we get exactly n_examples but not a few samples more
    x_test = torch.tensor(x_test)[:n_examples]
    y_test = torch.tensor(y_test)[:n_examples]

    return x_test, y_test


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
    elif data_name == "cinic":
        return _get_cinic(root, split, transform, target_transform, download)
    elif "stl10" in data_name:
        return _get_stl10(
            root, split, transform, target_transform, download, img_resolution
        )
    elif data_name == "svhn":
        return _get_svhn(root, split, transform, target_transform, download)
    elif data_name == "mnist":
        return _get_mnist(root, split, transform, target_transform, download)
    elif data_name == "femnist":
        return _get_femnist(root, split, transform, target_transform, download)
    elif "pseudo_imagenet" in data_name:
        return _get_pseudo_imagenet(root, split, img_resolution)
    elif "imagenet" in data_name:
        return _get_imagenet(data_name, datasets_path, split)
    else:
        raise NotImplementedError
