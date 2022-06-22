# -*- coding: utf-8 -*-
import os
import argparse
import pickle
import matplotlib.pyplot as plt

import pcode.utils.op_files as op_files
from pcode.tools.show_results import load_raw_info_from_experiments
from pcode.tools.plot import plot_overall_perf, plot_round_perf, display_best_args

"""parse and define arguments for different tasks."""


def get_args():
    # feed them to the parser.
    parser = argparse.ArgumentParser(description="Extract results.")

    # add arguments.
    parser.add_argument(
        "--in_dir",
        type=str,
        default="./checkpoint_report_finetune1/cifar10/simple_cnn/debug/",
    )
    parser.add_argument("--out_name", type=str, default="summary.pickle")
    parser.add_argument("--img_path", type=str, default="./checkpoint_normal/plots/")

    # parse aˇˇrgs.
    args = parser.parse_args()

    # an argument safety check.
    check_args(args)
    return args


def check_args(args):
    assert args.in_dir is not None

    # define out path.
    args.out_path = os.path.join(args.in_dir, args.out_name)


"""write the results to path."""


def main(args):
    # save the parsed results to path.
    run = True
    if not run:
        info = load_raw_info_from_experiments(args.in_dir)
        op_files.write_pickle(info, args.out_path)

    if run:
        prefix = "./checkpoint_"
        suffix = "/cifar10/simple_cnn/debug/summary.pickle"
        markers = ["-", "--", ":", "-."]
        fig, ax = plt.subplots()
        # ax.set_prop_cycle(color=['#8ECFC9', '#FFBE7A', '#FA7F6F', "82B0D2"])
        # ax.set_prop_cycle(color=['green', 'red', 'orange', "blue"])
        ax.set_prop_cycle(color=["green", "blue"])
        lr = 0.01
        personal_lr = 0.01
        local_epochs = 5
        personal_epochs = 1

        # method = ["finetune_brm", "fedrod", "fedrod_g", "fedrodfix"]
        # method = ["finetune_brm", "2head_init_ent_euclidean_19", "fedrod_g", "fedrodfix"]
        method = ["report_finetune", "report_finetune1"]
        # map = ["ft_brm", "2head", "noreuse", "reuse"]
        map = ["tune head", "tune all"]
        info = pickle.load(open(prefix + method[0] + suffix, "rb"))
        info1 = pickle.load(open(prefix + method[1] + suffix, "rb"))
        # info2 = pickle.load(open(prefix + method[2] + suffix, "rb"))
        # info3 = pickle.load(open(prefix + method[3] + suffix, "rb"))

        show_one_cluster(
            args,
            ax,
            map[0],
            info,
            markers[0],
            lr,
            personal_lr,
            local_epochs,
            personal_epochs,
        )
        show_one_cluster(
            args,
            ax,
            map[1],
            info1,
            markers[1],
            lr,
            personal_lr,
            local_epochs,
            personal_epochs,
        )
        # show_one_cluster(args, ax, map[2], info2, markers[2], lr, personal_lr, local_epochs, personal_epochs)
        # show_one_cluster(args, ax, map[3], info3, markers[3], lr, personal_lr, local_epochs, personal_epochs)

        ax.grid(True)
        plt.savefig("pic", dpi=300)
        plt.show()


def show_one_cluster(
    args,
    ax,
    method,
    info,
    marker,
    lr,
    personal_lr,
    local_epochs,
    personal_epochs,
    is_smooth=True,
):

    # plot_overall_perf(
    #     method,
    #     ax,
    #     info=info,
    #     y_metric="test-top1",
    #     condition={"lr": lr, "personal_lr": personal_lr, "local_n_epochs": local_epochs, "n_personalized_epochs": personal_epochs},
    #     legend_variable="n_personalized_epochs",
    #     is_save=False,
    #     is_smooth=is_smooth,
    #     smooth_space=3,
    #     marker=marker,
    #     path=args.img_path)

    plot_overall_perf(
        method,
        ax,
        info=info,
        y_metric="test-corr_top1",
        condition={
            "lr": lr,
            "personal_lr": personal_lr,
            "local_n_epochs": local_epochs,
            "n_personalized_epochs": personal_epochs,
        },
        legend_variable="n_personalized_epochs",
        is_save=False,
        is_smooth=is_smooth,
        smooth_space=3,
        marker=marker,
        path=args.img_path,
    )

    # plot_overall_perf(
    #     method,
    #     ax,
    #     info=info,
    #     y_metric="test-global_top1",
    #     condition={"lr": lr, "personal_lr": personal_lr, "local_n_epochs": local_epochs, "n_personalized_epochs": personal_epochs},
    #     legend_variable="n_personalized_epochs",
    #     is_save=False,
    #     is_smooth=is_smooth,
    #     smooth_space=3,
    #     marker=marker,
    #     path=args.img_path)

    plot_overall_perf(
        method,
        ax,
        info=info,
        y_metric="test-ood_top1",
        condition={
            "lr": lr,
            "personal_lr": personal_lr,
            "local_n_epochs": local_epochs,
            "n_personalized_epochs": personal_epochs,
        },
        legend_variable="n_personalized_epochs",
        is_save=False,
        is_smooth=is_smooth,
        smooth_space=3,
        marker=marker,
        path=args.img_path,
    )


if __name__ == "__main__":
    args = get_args()

    main(args)
