# -*- coding: utf-8 -*-
"""define all global parameters here."""
from os.path import join
import argparse

import pcode.models as models
from pcode.utils.param_parser import str2bool


def get_args():
    ROOT_DIRECTORY = "./"
    RAW_DATA_DIRECTORY = join(ROOT_DIRECTORY, "data/")
    TRAINING_DIRECTORY = join(RAW_DATA_DIRECTORY, "checkpoint")

    model_names = sorted(
        name for name in models.__dict__ if name.islower() and not name.startswith("__")
    )

    # feed them to the parser.
    parser = argparse.ArgumentParser(description="None")

    # add arguments.
    parser.add_argument("--work_dir", default=None, type=str)
    parser.add_argument("--remote_exec", default=False, type=str2bool)

    # dataset.
    parser.add_argument("--data", default="cifar10", help="a specific dataset name")
    parser.add_argument("--data_dir", default=None)
    parser.add_argument("--val_data_ratio", type=float, default=0)
    parser.add_argument(
        "--train_data_ratio", type=float, default=1, help="after the train/val split."
    )
    parser.add_argument("--img_resolution", type=int, default=None)
    parser.add_argument(
        "--use_lmdb_data",
        default=False,
        type=str2bool,
        help="use sequential lmdb dataset for better loading.",
    )
    parser.add_argument(
        "--partition_data_conf",
        default=None,
        type=str,
        help="decide if each worker will access to all data.",
    )
    parser.add_argument("--pin_memory", default=True, type=str2bool)
    parser.add_argument(
        "-j",
        "--num_workers",
        default=4,
        type=int,
        help="number of data loading workers (default: 4)",
    )

    # dataset settings in personalized and OOD case
    # dataset partition and reunion process mainly based on partition_data.py and ConcatDataset
    parser.add_argument(
        "--test_partition_ratio",
        default=0,
        type=float,
        help="this indicates the ratio of testset that merges with trainset",
    )
    parser.add_argument(
        "--local_train_ratio",
        default=0.6,
        type=float,
        help="the ratio of local training set, i.e. 0.6 means 3:2 for local train:(validation & test) on clients",
    )
    parser.add_argument(
        "--eval_dataset",
        default="test_loader",
        type=str,
        help="use test_loader or val_loader",
    )
    # if we want to use other client's test set as ooc set
    parser.add_argument("--test_ooc_perf_on_others", default=True, type=str2bool)
    parser.add_argument("--natural_shifted_imagenet_type", default="imagenet_v2_matched-frequency", type=str)
    # flag to determine how the mixed test set is sampled. only support cifar10
    parser.add_argument("--weighted_sampling_mixed_test", default=True, type=str)

    # model
    parser.add_argument(
        "--arch",
        default="resnet20",
        help="model architecture: " + " | ".join(model_names) + " (default: resnet20)",
    )
    parser.add_argument("--group_norm_num_groups", default=None, type=int)
    parser.add_argument(
        "--complex_arch", type=str, default="master=resnet20,worker=resnet8:resnet14"
    )
    parser.add_argument("--w_conv_bias", default=False, type=str2bool)
    parser.add_argument("--w_fc_bias", default=False, type=str2bool)
    parser.add_argument("--freeze_bn", default=False, type=str2bool)
    parser.add_argument("--freeze_bn_affine", default=False, type=str2bool)
    parser.add_argument("--resnet_scaling", default=1, type=float)
    parser.add_argument("--vgg_scaling", default=None, type=int)
    parser.add_argument("--evonorm_version", default=None, type=str)

    # personalization scheme.
    parser.add_argument("--is_personalized", default=True, type=str2bool)
    parser.add_argument("--is_corrupted", default=True, type=str2bool)
    parser.add_argument("--corr_severity", default=5, type=int, help="from 1 to 5")
    parser.add_argument("--corr_seed", default=5, type=int)
    parser.add_argument("--personalization_scheme", default=None, type=str)
    parser.add_argument("--is_in_childworker", default=None, type=str2bool)
    parser.add_argument("--n_personalized_epochs", default=1, type=int)
    parser.add_argument("--personal_lr", default=0.01, type=float)
    parser.add_argument("--with_BN", default=False, type=str2bool)
    # method specific parameters
    # FedRep
    parser.add_argument("--fedrep_personal_layers", default=1, type=int)
    # Ditto, note: please set weight decay to 0 when using Ditto
    parser.add_argument(
        "--regularized_factor",
        default=1.0,
        type=float,
        help="0 means training local models; a large value favors global model optimization",
    )
    parser.add_argument(
        "--is_dynamic_lambda",
        default=False,
        type=str2bool,
        help="whether or not using different lambda between clients, not implement now",
    )
    # FedRod
    parser.add_argument("--fedrod_personal_layers", default=1, type=int)
    # T3A
    parser.add_argument("--t3a_filter_k", default=50, type=int)
    # FedTHE and FedTHE+.
    parser.add_argument("--rep_len", default=64, type=int)
    parser.add_argument("--is_rep_history_reused", default=False, type=str2bool)
    parser.add_argument("--THE_steps", default=20, type=int)
    parser.add_argument("--THE_alpha", default=0.1, type=float)
    parser.add_argument("--THE_beta", default=0.1, type=float)
    # DFRA
    parser.add_argument("--drfa_sync_gap", default=1, type=int)
    parser.add_argument("--drfa_lambda_lr", default=0.01, type=float)

    # data, training and learning scheme.
    parser.add_argument("--comm_buffer_size", type=int, default=100)
    parser.add_argument("--n_comm_rounds", type=int, default=100)
    parser.add_argument(
        "--target_perf", type=float, default=None, help="it is between [0, 100]."
    )
    parser.add_argument("--early_stopping_rounds", type=int, default=None)
    parser.add_argument("--local_n_epochs", type=float, default=1)
    parser.add_argument("--min_local_epochs", type=float, default=None)
    parser.add_argument("--random_reinit_local_model", default=None, type=str)
    parser.add_argument("--reshuffle_per_epoch", default=False, type=str2bool)
    parser.add_argument("--batch_size", "-b", default=32, type=int)
    parser.add_argument("--min_batch_size", default=None, type=int)
    parser.add_argument("--base_batch_size", default=None, type=int)
    parser.add_argument("--n_clients", default=20, type=int, help="# of the clients for FL.")
    parser.add_argument("--participation_ratio",
        default=1.0,
        type=float,
        help="number of participated ratio per communication rounds",
    )
    parser.add_argument(
        "--n_participated",
        default=None,
        type=int,
        help="it will be determined by n_clients * participation_ratio",
    )
    parser.add_argument(
        "--n_master_sampled_clients",
        default=20,
        type=int,
        help="number of clients sampled by master",
    )  # Note: this determines the accepted number of client updates.

    parser.add_argument("--fl_aggregate", default="scheme=federated_average", type=str)
    parser.add_argument("--non_iid_alpha", default=None, type=str)
    parser.add_argument("--train_fast", type=str2bool, default=False)

    # learning rate scheme
    parser.add_argument("--lr", type=float, default=0.01)
    parser.add_argument("--global_lr", type=float, default=1)
    parser.add_argument(
        "--lr_scheduler",
        type=str,
        default="MultiStepLR",
        choices=["MultiStepLR", "ExponentialLR", "ReduceLROnPlateau"],
    )
    parser.add_argument("--lr_milestones", type=str, default=None)
    parser.add_argument("--lr_milestone_ratios", type=str, default=None)
    parser.add_argument("--round_milestones_ratios", type=str, default=None)
    parser.add_argument("--lr_decay", type=float, default=0.1)
    parser.add_argument("--lr_patience", type=int, default=10)
    parser.add_argument("--lr_scaleup", type=str2bool, default=False)
    parser.add_argument("--lr_scaleup_init_lr", type=float, default=None)
    parser.add_argument("--lr_scaleup_factor", type=int, default=None)
    parser.add_argument("--lr_warmup", type=str2bool, default=False)
    parser.add_argument("--lr_warmup_epochs", type=int, default=5)
    parser.add_argument("--lr_warmup_epochs_upper_bound", type=int, default=150)

    parser.add_argument("--adam_beta_1", default=0.9, type=float)
    parser.add_argument("--adam_beta_2", default=0.999, type=float)
    parser.add_argument("--adam_eps", default=1e-8, type=float)

    # optimizer
    parser.add_argument("--optimizer", type=str, default="sgd")

    # momentum scheme
    parser.add_argument("--momentum_factor", default=0.9, type=float)
    parser.add_argument("--use_nesterov", default=False, type=str2bool)

    # regularization
    parser.add_argument(
        "--weight_decay", default=5e-4, type=float, help="weight decay (default: 1e-4)"
    )
    parser.add_argument("--drop_rate", default=0.5, type=float)

    # some SOTA training schemes, e.g., larc, label smoothing.
    parser.add_argument("--weighted_loss", default=None, type=str)
    parser.add_argument("--weighted_beta", default=0, type=float)
    parser.add_argument("--weighted_gamma", default=0, type=float)

    # configuration for different models.
    parser.add_argument("--densenet_growth_rate", default=12, type=int)
    parser.add_argument("--densenet_bc_mode", default=False, type=str2bool)
    parser.add_argument("--densenet_compression", default=0.5, type=float)

    parser.add_argument("--wideresnet_widen_factor", default=4, type=int)

    parser.add_argument("--mlp_num_layers", default=2, type=int)
    parser.add_argument("--mlp_hidden_size", default=128, type=int)

    parser.add_argument("--rnn_n_hidden", default=200, type=int)
    parser.add_argument("--rnn_n_layers", default=2, type=int)
    parser.add_argument("--rnn_bptt_len", default=35, type=int)
    parser.add_argument("--rnn_clip", type=float, default=0.25)
    parser.add_argument("--rnn_use_pretrained_emb", type=str2bool, default=True)
    parser.add_argument("--rnn_tie_weights", type=str2bool, default=True)
    parser.add_argument("--rnn_weight_norm", type=str2bool, default=False)

    parser.add_argument("--transformer_n_layers", default=6, type=int)
    parser.add_argument("--transformer_n_head", default=8, type=int)
    parser.add_argument("--transformer_dim_model", default=512, type=int)
    parser.add_argument("--transformer_dim_inner_hidden", default=2048, type=int)
    parser.add_argument("--transformer_n_warmup_steps", default=4000, type=int)

    # miscs
    parser.add_argument("--same_seed_process", type=str2bool, default=True)
    parser.add_argument("--manual_seed", type=int, default=6, help="manual seed")
    parser.add_argument(
        "--evaluate",
        "-e",
        dest="evaluate",
        type=str2bool,
        default=False,
        help="evaluate model on validation set",
    )
    parser.add_argument("--display_log", default=False, type=str2bool)
    parser.add_argument("--summary_freq", default=256, type=int)
    parser.add_argument("--timestamp", default=None, type=str)
    parser.add_argument("--track_time", default=False, type=str2bool)
    parser.add_argument("--track_detailed_time", default=False, type=str2bool)
    parser.add_argument("--display_tracked_time", default=False, type=str2bool)

    # checkpoint
    parser.add_argument("--resume", default=None, type=str)
    parser.add_argument("--checkpoint", default=TRAINING_DIRECTORY, type=str)
    parser.add_argument("--save_every_n_round", type=int, default=None)
    parser.add_argument("--checkpoint_index", type=str, default=None)
    parser.add_argument("--save_all_models", type=str2bool, default=False)
    parser.add_argument("--save_some_models", type=str, default=None)

    # device
    parser.add_argument(
        "--python_path", type=str, default="/opt/conda/bin/python"
    )
    parser.add_argument("--world", default=None, type=str)
    parser.add_argument("--world_conf", default=None, type=str)
    parser.add_argument("--on_cuda", type=str2bool, default=True)
    parser.add_argument("--hostfile", type=str, default=None)
    parser.add_argument("--mpi_path", type=str, default="/.openmpi")
    parser.add_argument("--mpi_env", type=str, default=None)

    """meta info."""
    parser.add_argument("--experiment", type=str, default="debug")
    parser.add_argument("--job_name", type=str, default="default")
    parser.add_argument("--job_id", type=str, default="/tmp/jobrun_logs")
    parser.add_argument("--script_path", default="exps/exp_cifar10_cnn.py", type=str)
    parser.add_argument("--script_class_name", default=None, type=str)
    parser.add_argument("--num_jobs_per_node", default=1, type=int)
    parser.add_argument("--wait_in_seconds_per_job", default=30, type=int)

    # parse conf.
    conf = parser.parse_args()
    return conf


if __name__ == "__main__":
    args = get_args()
