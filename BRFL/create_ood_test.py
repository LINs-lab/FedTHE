# -*- coding: utf-8 -*-
import numpy as np
import torch
from torch.utils.data import ConcatDataset, Subset

import prepare_data as prepare_data
import corr_data as corr_data

import warnings

warnings.simplefilter("ignore", UserWarning)


def get_corr_data(data, indices_per_client, random_state, severity=5):
    corr_test = {}
    for i, ind in indices_per_client.items():
        curr_data = torch.utils.data.Subset(data, ind["test"])
        corr_test[i] = corr_data.CorruptData(data=curr_data, random_state=random_state, severity=severity)
    return corr_test


def get_ooc_data(data, indices_per_client, random_state):
    ooc_test = {}
    # print(indices_per_client)
    for i, ind in indices_per_client.items():
        ind_other_tests = np.concatenate([indices_per_client[j]["test"] for j in range(len(indices_per_client)) if j!=i])
        # print(ind_other_tests)
        other_ind = random_state.choice(ind_other_tests, len(ind["test"]))
        ooc_test[i] = torch.utils.data.Subset(data, other_ind)
    return ooc_test


def get_natural_shift_data(data, indices_per_client, random_state, data_path, data_name):
    natural_shift_data = prepare_data.get_dataset(
        data_name=data_name, datasets_path=data_path, split="test",
    )
    natural_shift_test = {}
    test_indices = [indices_per_client[i]["test"] for i in range(len(indices_per_client))]
    hist = record_class_distribution(test_indices, data.targets)
    partitioned_indices = partition_by_histogram(natural_shift_data, hist, random_state)
    for i in range(len(partitioned_indices)):
        natural_shift_test[i] = Subset(
            natural_shift_data, partitioned_indices[i]
        )

    return natural_shift_test

def get_mixed_data(fl_data_per_client, random_state, weighted_sampling=False):
    mixed_test = {}
    if not weighted_sampling:
        for i in range(len(fl_data_per_client["test"])):
            id_ood_tests = [fl_data_per_client[t][i] for t in fl_data_per_client.keys() if "test" in t]
            id_ood_tests = ConcatDataset(id_ood_tests)
            id_ood_tests.indices = list(range(len(id_ood_tests)))
            ind = random_state.choice(id_ood_tests.indices, len(fl_data_per_client["test"][0]))
            mixed_test[i] = Subset(id_ood_tests, ind)
    else:
        for i in range(len(fl_data_per_client["test"])):
            id_ood_tests = [fl_data_per_client[t][i] for t in fl_data_per_client.keys() if "test" in t]
            shortest_test_ind = np.argmin([len(test) for test in id_ood_tests])  # naturally shifted test, normally
            weighted_id_ood_tests = []
            for j in range(len(id_ood_tests)):
                if j!=shortest_test_ind:
                    weighted_id_ood_tests.append(
                        Subset(
                            id_ood_tests[j],
                            indices=random_state.choice(
                                len(id_ood_tests[j]),
                                len(id_ood_tests[shortest_test_ind]),
                                replace=False
                            )
                        )
                    )
                else:
                    weighted_id_ood_tests.append(id_ood_tests[j])
            mixed_test[i] = ConcatDataset(weighted_id_ood_tests)

    return mixed_test


def record_class_distribution(partitions, targets):
    cls_hist = []

    targets_np = np.array(targets)
    # compute unique values here
    num_class = len(np.unique(targets_np))
    for partition in partitions:
        unique_elements, counts_elements = np.unique(
            targets_np[partition], return_counts=True
        )
        temp = np.zeros(num_class, dtype=int)
        temp[unique_elements] = counts_elements
        cls_hist.append(list(temp))

    return cls_hist


def partition_by_histogram(data, hist, random_state):
    num_class = len(np.unique(data.targets))
    split_point = []
    targets_np = data.targets
    indices = np.argsort(targets_np)
    targets_np = np.array(targets_np)[indices]
    last = targets_np[0]
    for i, target in enumerate(targets_np):
        if target != last:
            split_point.append(i)
            last = target
    indices = np.array_split(indices, split_point)
    # convert the dict to array, for every class indices, partition it by count_array
    count_array = np.floor(np.array(hist) * (len(data)/num_class) / np.sum(np.array(hist), axis=0)).T
    count_array = np.cumsum(count_array, axis=1).astype(int)
    # get partitioned indices
    per_class_client_indices = np.array_split(indices[0], count_array[0])
    partitioned_indices = per_class_client_indices
    for i in range(num_class - 1):
        per_class_client_indices = np.array_split(indices[i+1], count_array[i+1])
        partitioned_indices = [np.concatenate([sub_arr, per_class_client_indices[j]])
            for j, sub_arr in enumerate(partitioned_indices)]
    return partitioned_indices

