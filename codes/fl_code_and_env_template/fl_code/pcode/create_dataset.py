# -*- coding: utf-8 -*-
import functools
import torch

from pcode.datasets.partition_data import DataPartitioner
import pcode.datasets.prepare_data as prepare_data
import torchvision.transforms as transforms

"""create dataset and load the data_batch."""


def load_data_batch(conf, _input, _target, is_training=True):
    """Load a mini-batch and record the loading time."""
    if conf.graph.on_cuda:
        _input, _target = _input.cuda(), _target.cuda()
    _data_batch = {"input": _input, "target": _target}
    return _data_batch


class FLData:
    def __init__(
        self,
        logger,
        graph,
        random_state,
        batch_size,
        img_resolution,
        local_n_epochs,
        num_workers,
        pin_memory,
    ):
        self.logger = logger
        self.graph = graph
        self.random_state = random_state

        self.batch_size = batch_size
        self.img_resolution = img_resolution
        self.local_n_epochs = local_n_epochs
        self.num_workers = num_workers
        self.pin_memory = pin_memory

    def define_dataset(self, data_name, data_dir, display_log=True):
        """This function manipulates the original dataset (before partitioning by clients):
        1. the train_dataset will be splitted into train_dataset and val_dataset.
        2. part of test_dataset will be combined with the train_dataset.
        """
        # prepare general train/test, and create the validation from train if necessary.
        train_dataset, val_dataset, test_dataset = self._define_tr_val_te_dataset(
            train_dataset=prepare_data.get_dataset(
                data_name, data_dir, split="train", img_resolution=self.img_resolution
            ),
            test_dataset=prepare_data.get_dataset(
                data_name, data_dir, split="test", img_resolution=self.img_resolution
            ),
        )

        if display_log:
            self.logger.log(
                "Data stat for original dataset: we have {} samples for train, {} samples for val, {} samples for test.".format(
                    len(train_dataset),
                    len(val_dataset) if val_dataset is not None else 0,
                    len(test_dataset),
                )
            )
        self.dataset = {
            "train": train_dataset,
            "val": val_dataset,
            "test": test_dataset,
        }

    def _define_tr_val_te_dataset(
        self, train_dataset, test_dataset, train_data_ratio=1.0, val_data_ratio=0.0
    ):
        assert val_data_ratio >= 0
        partition_sizes = [
            (1 - val_data_ratio) * train_data_ratio,
            (1 - val_data_ratio) * (1 - train_data_ratio),
            val_data_ratio,
        ]

        data_partitioner = DataPartitioner(
            train_dataset,
            partition_sizes,
            partition_type="origin",
            partition_alphas=None,
            consistent_indices=False,
            random_state=self.random_state,
            graph=self.graph,
            logger=self.logger,
        )
        train_dataset = data_partitioner.use(0)

        # split for val data.
        if val_data_ratio > 0:
            val_dataset = data_partitioner.use(2)
            return train_dataset, val_dataset, test_dataset
        else:
            return train_dataset, None, test_dataset

    def inter_clients_data_partition(self, dataset, n_clients, partition_data_conf):
        if "size_conf" not in partition_data_conf:
            partition_sizes = [1.0 / n_clients for _ in range(n_clients)]
        else:
            # a few simple cases.
            # size_conf=1:2:3 indicates three data fraction cases.
            cases = [float(x) for x in partition_data_conf["size_conf"].split(":")]
            cases = (
                (cases * int(n_clients / len(cases) + 1))[:n_clients]
                if n_clients > 1
                else [1]
            )
            sum_cases = sum(cases)
            partition_sizes = [1.0 / sum_cases * case for case in cases]
            self.random_state.shuffle(partition_sizes)

        # create data partitioner.
        self.data_partitioner = DataPartitioner(
            dataset,
            partition_sizes=partition_sizes,
            partition_type=partition_data_conf["distribution"],
            partition_alphas=partition_data_conf["non_iid_alpha"],
            consistent_indices=True,
            random_state=self.random_state,
            graph=self.graph,
            logger=self.logger,
        )

    def intra_client_data_partition_and_create_dataloaders(
        self,
        localdata_id,
        batch_size=None,
        shuffle=True,
        display_log=True,
    ):
        """partition clients' data to train, val, test."""
        assert hasattr(self, "data_partitioner")
        batch_size = self.batch_size if batch_size is None else int(batch_size)

        # get the partitioned dataset.
        log_message = f"Data partition for train (client_id={localdata_id + 1})."
        data_to_load = self.data_partitioner.use(localdata_id)
        _create_dataloader_fn = functools.partial(
            self.create_dataloader, batch_size=batch_size, shuffle=shuffle
        )

        # create dataloaders.
        data_loader_local_tr = _create_dataloader_fn(data_to_load)
        data_loaders = {
            "train": data_loader_local_tr,
            "num_batches_per_device_per_epoch": len(data_loader_local_tr),
        }

        if display_log:
            self.logger.log(
                f"{log_message}: # of tr batches={len(data_loaders['train'])}, # batch_size={batch_size}."
            )
        return data_loaders

    def create_dataloader(self, dataset, batch_size=None, shuffle=True):
        batch_size = self.batch_size if batch_size is None else int(batch_size)

        return torch.utils.data.DataLoader(
            dataset,
            batch_size=batch_size,
            shuffle=shuffle,
            num_workers=self.num_workers,
            pin_memory=self.pin_memory,
            drop_last=False,
        )
