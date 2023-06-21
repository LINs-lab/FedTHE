# -*- coding: utf-8 -*-
import time
import shutil
import json
from os.path import join

import torch

from pcode.utils.op_paths import build_dirs
from pcode.utils.op_files import is_jsonable


def get_checkpoint_folder_name(conf):
    # get optimizer info.
    optim_info = "{}".format(conf.optimizer)

    # get n_participated
    conf.n_participated = int(conf.n_clients * conf.participation_ratio + 0.5)

    # concat them together.
    # return "_{}_lr-{}_plr-{}_rounds-{}_lo_epochs-{}_per_epochs-{}_n_clients_{}_n_par_ratio-{}_non_iid-{}".format(
    #     conf.personalization_scheme["method"],
    #     conf.lr,
    #     conf.personal_lr,
    #     conf.n_comm_rounds,
    #     conf.local_n_epochs,
    #     conf.n_personalized_epochs,
    #     conf.n_clients,
    #     conf.participation_ratio,
    #     conf.partition_data_conf["non_iid_alpha"]
    # )
    return "_{}_lr-{}_plr-{}_lo_epochs-{}_per_epochs-{}_n_clients_{}_non_iid-{}".format(
        conf.personalization_scheme["method"],
        conf.lr,
        conf.personal_lr,
        conf.local_n_epochs,
        conf.n_personalized_epochs,
        conf.n_clients,
        conf.partition_data_conf["non_iid_alpha"]
    )


def init_checkpoint(conf, rank=None):
    # init checkpoint_root for the main process.
    conf.checkpoint_root = join(
        conf.checkpoint,
        conf.data,
        conf.arch,
        conf.experiment,
        conf.personalization_scheme["method"],
        conf.timestamp + get_checkpoint_folder_name(conf),
    )
    if conf.save_some_models is not None:
        conf.save_some_models = conf.save_some_models.split(",")

    if rank is None:
        # if the directory does not exists, create them.
        build_dirs(conf.checkpoint_root)
    else:
        conf.checkpoint_dir = join(conf.checkpoint_root, rank)
        build_dirs(conf.checkpoint_dir)


def _save_to_checkpoint(state, dirname, filename):
    checkpoint_path = join(dirname, filename)
    torch.save(state, checkpoint_path)
    return checkpoint_path


def save_arguments(conf):
    # save the configure file to the checkpoint.
    with open(join(conf.checkpoint_root, "arguments.json"), "w") as fp:
        json.dump(
            dict(
                [
                    (k, v)
                    for k, v in conf.__dict__.items()
                    if is_jsonable(v) and type(v) is not torch.Tensor
                ]
            ),
            fp,
            indent=" ",
        )


def save_to_checkpoint(conf, state, is_best, dirname, filename, save_all=False):
    # save full state.
    checkpoint_path = _save_to_checkpoint(state, dirname, filename)
    best_model_path = join(dirname, "model_best.pth.tar")
    if is_best:
        shutil.copyfile(checkpoint_path, best_model_path)
    if save_all:
        shutil.copyfile(
            checkpoint_path,
            join(
                dirname, "checkpoint_c_round_%s.pth.tar" % state["current_comm_round"]
            ),
        )
    elif conf.save_some_models is not None:
        if str(state["current_comm_round"]) in conf.save_some_models:
            shutil.copyfile(
                checkpoint_path,
                join(
                    dirname,
                    "checkpoint_c_round_%s.pth.tar" % state["current_comm_round"],
                ),
            )
