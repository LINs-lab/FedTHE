# -*- coding: utf-8 -*-
import argparse
import os
import torch
import numpy as np
from torch.utils.data import Subset

import prepare_data as prepare_data
import partition_data as partition_data
import create_ood_test as create_ood_test

if __name__ == "__main__":
    # Hyperparameters.
    parser = argparse.ArgumentParser()
    parser.add_argument('--seed', type=int, default=6)
    parser.add_argument('--data_path', type=str, default=os.getcwd())
    parser.add_argument('--num_clients', type=int, default=20)
    parser.add_argument('--data_type', type=str, default="cifar10")
    parser.add_argument('--local_tr_ratio', type=float, default=0.6)
    parser.add_argument('--local_te_ratio', type=float, default=0.2)
    parser.add_argument('--non_iid_alpha', type=float, default=0.1)
    parser.add_argument(
        '--corr_severity',
        help="Severity of corruption (only affect corrupted tests), between 0 and 5",
        type=int,
        default=5
    )
    parser.add_argument(
        '--weighted_sampling_mixed_test',
        help="Whether sampling each test distribution approx. equally (for cifar10)",
        type=bool,
        default=True
    )

    args = parser.parse_args()

    random_state = np.random.RandomState(args.seed)
    # Sanity check.
    assert args.local_tr_ratio > 0 and args.local_te_ratio >= 0
    assert args.data_type == "cifar10" or args.data_type == "imagenet"

    dataset = {}
    if args.data_type == "cifar10":

        # Prepare dataset.
        dataset["cifar10_tr"] = prepare_data.get_dataset(
            data_name="cifar10", datasets_path=args.data_path, split="train",
        )
        dataset["cifar10_te"] = prepare_data.get_dataset(
            data_name="cifar10", datasets_path=args.data_path, split="test",
        )

        # Merge, then split.
        fl_data = torch.utils.data.ConcatDataset([dataset["cifar10_tr"], dataset["cifar10_te"]])
        fl_data.indices = list(range(len(dataset["cifar10_tr"]) + len(dataset["cifar10_te"])))
        fl_data.targets = dataset["cifar10_tr"].targets + dataset["cifar10_te"].targets

        non_iid_indices = partition_data.inter_client_non_iid_partition(
            fl_data, args.non_iid_alpha, random_state, args.num_clients
        )

        # Intra-split into local train/val/test.
        indices_per_client = partition_data.intra_client_uniform_partition(
            non_iid_indices, random_state, args.local_tr_ratio, args.local_te_ratio
        )

        # Create ID/OOD train & test datasets.
        fl_data_per_client = {}
        fl_data_per_client["train"], fl_data_per_client["val"], fl_data_per_client["test"] = {}, {}, {}
        for i, indices in indices_per_client.items():
            fl_data_per_client["train"][i] = Subset(fl_data, indices["train"])
            fl_data_per_client["val"][i] = Subset(fl_data, indices["val"])
            fl_data_per_client["test"][i] = Subset(fl_data, indices["test"])
        fl_data_per_client["corr_test"] = create_ood_test.get_corr_data(
            fl_data,
            indices_per_client,
            random_state,
            severity=args.corr_severity
        )  # common corruptions
        fl_data_per_client["ooc_test"] = create_ood_test.get_ooc_data(
            fl_data,
            indices_per_client,
            random_state,
        )  # out-of-client (label shift) test
        fl_data_per_client["natural_shift_test"] = create_ood_test.get_natural_shift_data(
            fl_data,
            indices_per_client,
            random_state,
            data_path=args.data_path,
            data_name="cifar10_1",
        )  # cifar10.1 (natural shift) test
        fl_data_per_client["mixed_test"] = create_ood_test.get_mixed_data(
            fl_data_per_client,
            random_state,
            weighted_sampling=args.weighted_sampling_mixed_test,
        )  # mixed test

        # `fl_data_per_client` thus becomes a dict where fl_data_per_client[test_type][client_id] gives the corresponding local dataset.
        # Final operations, e.g., create dataloaders.
        test_loader = torch.utils.data.DataLoader(
            fl_data_per_client["test"][0],
            batch_size=32,
            shuffle=True,
            drop_last=False,
        )
        for imgs, labels in test_loader:
            imgs, labels = prepare_data.transform_data_batch(imgs, labels, is_training=False)

    elif args.data_type == "imagenet":

        # Prepare dataset.
        dataset["imagenet_tr"] = prepare_data.get_dataset(
            data_name="imagenet32", datasets_path=args.data_path, split="train",
        )
        dataset["imagenet_te"] = prepare_data.get_dataset(
            data_name="imagenet32", datasets_path=args.data_path, split="test",
        )

        # Merge, then split.
        fl_data = torch.utils.data.ConcatDataset([dataset["imagenet_tr"], dataset["imagenet_te"]])
        fl_data.indices = list(range(len(dataset["imagenet_tr"]) + len(dataset["imagenet_te"])))
        fl_data.targets = dataset["imagenet_tr"].targets + dataset["imagenet_te"].targets

        non_iid_indices = partition_data.inter_client_non_iid_partition(
            fl_data, args.non_iid_alpha, random_state, args.num_clients
        )

        # Intra-split into local train/val/test.
        indices_per_client = partition_data.intra_client_uniform_partition(
            non_iid_indices, random_state, args.local_tr_ratio, args.local_te_ratio
        )

        # Create ID/OOD train & test datasets.
        fl_data_per_client = {}
        fl_data_per_client["train"], fl_data_per_client["val"], fl_data_per_client["test"] = {}, {}, {}
        for i, indices in indices_per_client.items():
            fl_data_per_client["train"][i] = Subset(fl_data, indices["train"])
            fl_data_per_client["val"][i] = Subset(fl_data, indices["val"])
            fl_data_per_client["test"][i] = Subset(fl_data, indices["test"])
        fl_data_per_client["a_test"] = create_ood_test.get_natural_shift_data(
            fl_data,
            indices_per_client,
            random_state,
            data_path=args.data_path,
            data_name="imagenet_a",
        )  # ImageNet-A
        fl_data_per_client["v2_test"] = create_ood_test.get_natural_shift_data(
            fl_data,
            indices_per_client,
            random_state,
            data_path=args.data_path,
            data_name="imagenet_v2_matched-frequency",
        )  # ImageNet-V2
        fl_data_per_client["r_test"] = create_ood_test.get_natural_shift_data(
            fl_data,
            indices_per_client,
            random_state,
            data_path=args.data_path,
            data_name="imagenet_r",
        )  # ImageNet-R
        fl_data_per_client["mixed_test"] = create_ood_test.get_mixed_data(
            fl_data_per_client,
            random_state,
        )  # mixed test

        # `fl_data_per_client` thus becomes a dict where fl_data_per_client[test_type][client_id] gives the corresponding local dataset.
        # Final operations, e.g., create dataloaders.
        test_loader = torch.utils.data.DataLoader(
            fl_data_per_client["test"][0],
            batch_size=128,
            shuffle=True,
            drop_last=False,
        )
        for imgs, labels in test_loader:
            imgs, labels = prepare_data.transform_data_batch(imgs, labels, is_training=False)

