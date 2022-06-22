# -*- coding: utf-8 -*-
import argparse
import os
from os.path import join

from sklearn.datasets import load_svmlight_file

import numpy as np
from tensorpack.dataflow import dataset, PrefetchDataZMQ, LMDBSerializer


def get_args():
    parser = argparse.ArgumentParser(description="aug data.")

    # define arguments.
    parser.add_argument("--data", default="imagenet", help="a specific dataset name")
    parser.add_argument("--data_dir", default=None)
    parser.add_argument("--data_type", default="train", type=str)

    # parse args.
    args = parser.parse_args()

    # check args.
    assert args.data_dir is not None
    assert (
        "train" in args.data_type or "val" in args.data_type or "test" in args.data_type
    )
    return args


def build_dirs(path):
    try:
        os.makedirs(path)
    except OSError as e:
        print(e)


"""sequential imagenet"""


def sequential_imagenet(root_path, data_type):
    # define path.
    lmdb_path = join(root_path, "lmdb")
    build_dirs(lmdb_path)
    lmdb_file_path = join(lmdb_path, data_type + ".lmdb")

    class BinaryILSVRC12(dataset.ILSVRC12Files):
        def __iter__(self):
            for fname, label in super(BinaryILSVRC12, self).__iter__():
                with open(fname, "rb") as f:
                    jpeg = f.read()
                jpeg = np.asarray(bytearray(jpeg), dtype="uint8")
                yield [jpeg, label]

    ds0 = BinaryILSVRC12(root_path, data_type, shuffle=False)
    ds1 = PrefetchDataZMQ(ds0, nr_proc=1)
    LMDBSerializer.save(ds1, lmdb_file_path)


"""sequential epsilon or rcv1"""


_DATASET_MAP = {
    "epsilon_train": "https://www.csie.ntu.edu.tw/\~cjlin/libsvmtools/datasets/binary/epsilon_normalized.bz2",
    "epsilon_test": "https://www.csie.ntu.edu.tw/~cjlin/libsvmtools/datasets/binary/epsilon_normalized.t.bz2",
    "rcv1_train": "https://www.csie.ntu.edu.tw/~cjlin/libsvmtools/datasets/binary/rcv1_train.binary.bz2",
    "rcv1_test": "https://www.csie.ntu.edu.tw/~cjlin/libsvmtools/datasets/binary/rcv1_test.binary.bz2",
    "url": "https://www.csie.ntu.edu.tw/~cjlin/libsvmtools/datasets/binary/url_combined.bz2",
    "higgs_train": "https://www.csie.ntu.edu.tw/~cjlin/libsvmtools/datasets/binary/HIGGS.bz2",
    "higgs_test": "https://www.csie.ntu.edu.tw/~cjlin/libsvmtools/datasets/binary/HIGGS.bz2",
}


def maybe_download_and_extract(root, data_url, file_path):
    if not os.path.exists(root):
        os.makedirs(root)

    file_name = data_url.split("/")[-1]

    if len([x for x in os.listdir(root) if x == file_name]) == 0:
        os.system("wget -t inf {} -O {}".format(data_url, file_path))


def _get_dense_tensor(tensor):
    if "sparse" in str(type(tensor)):
        return tensor.toarray()
    elif "numpy" in str(type(tensor)):
        return tensor


def _correct_binary_labels(labels, is_01_classes=True):
    classes = set(labels)

    if -1 in classes and is_01_classes:
        labels[labels == -1] = 0
    return labels


class Epsilon_or_RCV1(object):
    def __init__(self, root, name, split):
        # get file url and file path.
        data_url = _DATASET_MAP["{}_{}".format(name, split)]
        file_path = os.path.join(root, data_url.split("/")[-1])

        # download dataset or not.
        maybe_download_and_extract(root, data_url, file_path)

        # load dataset.
        dataset = load_svmlight_file(file_path)
        self.features, self.labels = self._get_images_and_labels(dataset)

    def _get_images_and_labels(self, data):
        features, labels = data

        features = _get_dense_tensor(features)
        labels = _get_dense_tensor(labels)
        labels = _correct_binary_labels(labels)
        return features, labels

    def __len__(self):
        return self.features.shape[0]

    def __iter__(self):
        idxs = list(range(self.__len__()))
        for k in idxs:
            yield [self.features[k], self.labels[k]]

    def get_data(self):
        return self.__iter__()

    def size(self):
        return self.__len__()

    def reset_state(self):
        """
        Reset state of the dataflow.
        It **has to** be called once and only once before producing datapoints.
        Note:
            1. If the dataflow is forked, each process will call this method
               before producing datapoints.
            2. The caller thread of this method must remain alive to keep this dataflow alive.
        For example, RNG **has to** be reset if used in the DataFlow,
        otherwise it won't work well with prefetching, because different
        processes will have the same RNG state.
        """
        pass


def sequential_epsilon_or_rcv1(root_path, name, data_type):
    data = Epsilon_or_RCV1(root_path, name, data_type)
    lmdb_file_path = join(root_path, "{}_{}.lmdb".format(name, data_type))

    print("dump_dataflow_to_lmdb for {}".format(lmdb_file_path))
    ds1 = PrefetchDataZMQ(data, nr_proc=1)
    LMDBSerializer.save(ds1, lmdb_file_path)


def main(args):
    if args.data == "imagenet":
        sequential_imagenet(args.data_dir, args.data_type)
    else:
        sequential_epsilon_or_rcv1(args.data_dir, args.data, args.data_type)


if __name__ == "__main__":
    args = get_args()
    main(args)

# python build_sequential_data.py --data_dir /mlo-container-scratch/tlin/dataset/ILSVRC/raw_data/imagenet --data_type train
# python build_sequential_data.py --data_dir /mlo-container-scratch/tlin/dataset/ILSVRC/raw_data/imagenet --data_type val
