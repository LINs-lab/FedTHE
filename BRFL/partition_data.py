# -*- coding: utf-8 -*-
import math
import numpy as np
import functools

def inter_client_non_iid_partition(dataset, non_iid_alpha, random_state, num_clients):
    num_classes = len(np.unique(dataset.targets))
    indices = dataset.indices
    num_indices = len(indices)
    indices2targets = np.array(
        [
            (idx, target)
            for idx, target in enumerate(dataset.targets)
            if idx in indices
        ]
    )
    # Partition.
    non_iid_indices = build_non_iid_by_dirichlet(
        random_state=random_state,
        indices2targets=indices2targets,
        non_iid_alpha=non_iid_alpha,
        num_classes=num_classes,
        num_indices=num_indices,
        n_workers= num_clients,
    )
    non_iid_indices = functools.reduce(lambda a, b: a + b, non_iid_indices)

    # Convert list indices to dict.
    from_index = 0
    partitioned_dict = {}
    partition_size = int(num_indices/num_clients)
    for i in range(num_clients):
        to_index = from_index + partition_size
        partitioned_dict[i] = non_iid_indices[from_index:to_index]
        from_index = to_index

    return partitioned_dict


def intra_client_uniform_partition(indices, random_state, local_tr_ratio, local_te_ratio):
    for i, ind in indices.items():
        random_state.shuffle(ind)
        indices[i] = {
            "train": ind[:int(len(ind) * local_tr_ratio)],
            "test": ind[int(len(ind) * local_tr_ratio): int(len(ind) * local_tr_ratio) + int(len(ind) * local_te_ratio)],
            "val": ind[int(len(ind) * local_tr_ratio) + int(len(ind) * local_te_ratio):]
        }
    return indices


def build_non_iid_by_dirichlet(
    random_state, indices2targets, non_iid_alpha, num_classes, num_indices, n_workers
):
    n_auxi_workers = 10

    # random shuffle targets indices.
    random_state.shuffle(indices2targets)

    # partition indices.
    from_index = 0
    splitted_targets = []
    num_splits = math.ceil(n_workers / n_auxi_workers)
    split_n_workers = [
        n_auxi_workers
        if idx < num_splits - 1
        else n_workers - n_auxi_workers * (num_splits - 1)
        for idx in range(num_splits)
    ]
    split_ratios = [_n_workers / n_workers for _n_workers in split_n_workers]
    for idx, ratio in enumerate(split_ratios):
        to_index = from_index + int(n_auxi_workers / n_workers * num_indices)
        splitted_targets.append(
            indices2targets[
                from_index : (num_indices if idx == num_splits - 1 else to_index)
            ]
        )
        from_index = to_index

    idx_batch = []
    for _targets in splitted_targets:
        # rebuild _targets.
        _targets = np.array(_targets)
        _targets_size = len(_targets)

        # use auxi_workers for this subset targets.
        _n_workers = min(n_auxi_workers, n_workers)
        n_workers = n_workers - n_auxi_workers

        # get the corresponding idx_batch.
        min_size = 0
        while min_size < int(0.50 * _targets_size / _n_workers):
            _idx_batch = [[] for _ in range(_n_workers)]
            for _class in range(num_classes):
                # get the corresponding indices in the original 'targets' list.
                idx_class = np.where(_targets[:, 1] == _class)[0]
                idx_class = _targets[idx_class, 0]

                # sampling.
                try:
                    proportions = random_state.dirichlet(
                        np.repeat(non_iid_alpha, _n_workers)
                    )
                    # balance
                    proportions = np.array(
                        [
                            p * (len(idx_j) < _targets_size / _n_workers)
                            for p, idx_j in zip(proportions, _idx_batch)
                        ]
                    )
                    proportions = proportions / proportions.sum()
                    proportions = (np.cumsum(proportions) * len(idx_class)).astype(int)[
                        :-1
                    ]
                    _idx_batch = [
                        idx_j + idx.tolist()
                        for idx_j, idx in zip(
                            _idx_batch, np.split(idx_class, proportions)
                        )
                    ]
                    sizes = [len(idx_j) for idx_j in _idx_batch]
                    min_size = min([_size for _size in sizes])
                except ZeroDivisionError:
                    pass
        idx_batch += _idx_batch
    return idx_batch