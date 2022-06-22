# -*- coding: utf-8 -*-
from operator import itemgetter
import numpy as np
import matplotlib.pyplot as plt

from pcode.tools.show_results import reorder_records
from pcode.tools.plot_utils import (
    determine_color_and_lines,
    plot_one_case,
    smoothing_func,
    configure_figure,
    build_legend,
    groupby_indices,
)


"""plot the curve in terms of time."""


def plot_curve_wrt_time(
    ax,
    records,
    x_wrt_sth,
    y_wrt_sth,
    xlabel,
    ylabel,
    title=None,
    markevery_list=None,
    is_smooth=True,
    smooth_space=100,
    l_subset=0.0,
    r_subset=1.0,
    reorder_record_item=None,
    remove_duplicate=True,
    legend=None,
    legend_loc="lower right",
    legend_ncol=2,
    bbox_to_anchor=[0, 0],
    ylimit_bottom=None,
    ylimit_top=None,
    use_log=False,
    num_cols=3,
):
    """Each info consists of
        ['tr_loss', 'tr_top1', 'tr_time', 'te_top1', 'te_step', 'te_time'].
    """
    # parse a list of records.
    num_records = len(records)
    distinct_conf_set = set()

    # re-order the records.
    if reorder_record_item is not None:
        records = reorder_records(records, based_on=reorder_record_item)

    count = 0
    for ind, (args, info) in enumerate(records):
        # build legend.
        _legend = build_legend(args, legend)
        if _legend in distinct_conf_set and remove_duplicate:
            continue
        else:
            distinct_conf_set.add(_legend)

        # split the y_wrt_sth if it can be splitted.
        if ";" in y_wrt_sth:
            has_multiple_y = True
            list_of_y_wrt_sth = y_wrt_sth.split(";")
        else:
            has_multiple_y = False
            list_of_y_wrt_sth = [y_wrt_sth]

        for _y_wrt_sth in list_of_y_wrt_sth:
            # determine the style of line, color and marker.
            line_style, color_style, mark_style = determine_color_and_lines(
                num_rows=num_records // num_cols, num_cols=num_cols, ind=count
            )
            if markevery_list is not None:
                mark_every = markevery_list[count]
            else:
                mark_style = None
                mark_every = None

            # update the counter.
            count += 1

            # determine if we want to smooth the curve.
            if "tr_step" in x_wrt_sth or "tr_epoch" in x_wrt_sth:
                info["tr_step"] = list(range(1, 1 + len(info["tr_loss"])))
            if "tr_epoch" == x_wrt_sth:
                x = info["tr_step"]
                x = [
                    1.0 * _x / args["num_batches_train_per_device_per_epoch"]
                    for _x in x
                ]
            else:
                x = info[x_wrt_sth]
                if "time" in x_wrt_sth:
                    x = [(time - x[0]).seconds + 1 for time in x]
            y = info[_y_wrt_sth]

            if is_smooth:
                x, y = smoothing_func(x, y, smooth_space)

            # only plot subtset.
            _l_subset, _r_subset = int(len(x) * l_subset), int(len(x) * r_subset)
            _x = x[_l_subset:_r_subset]
            _y = y[_l_subset:_r_subset]

            # use log scale for y
            if use_log:
                _y = np.log(_y)

            # plot
            ax = plot_one_case(
                ax,
                x=_x,
                y=_y,
                label=_legend if not has_multiple_y else _legend + f", {_y_wrt_sth}",
                line_style=line_style,
                color_style=color_style,
                mark_style=mark_style,
                mark_every=mark_every,
                remove_duplicate=remove_duplicate,
            )

    ax.set_ylim(bottom=ylimit_bottom, top=ylimit_top)
    ax = configure_figure(
        ax,
        xlabel=xlabel,
        ylabel=ylabel,
        title=title,
        has_legend=legend is not None,
        legend_loc=legend_loc,
        legend_ncol=legend_ncol,
        bbox_to_anchor=bbox_to_anchor,
    )
    return ax


def plot_overall_perf(
    method,
    ax,
    info,
    y_metric,
    condition,
    is_save,
    path,
    legend_variable,
    marker,
    is_smooth=True,
    smooth_space=100,
):
    """Function that plot accuracy for the whole training process."""
    # extract useful information.
    exp_args, records, perfs = extract_info(info, y_metric)
    # make the plot.
    x = records[0]["test-step"]
    satisfying_perfs = [
        (exp_args[ind][legend_variable], perf[y_metric])
        for ind, perf in enumerate(perfs)
        if y_metric in perf.keys() and is_meet_condition(exp_args[ind], condition)
    ]
    print("number of matched records:", len(satisfying_perfs))
    for key, perf in satisfying_perfs:
        y = perf
        if is_smooth:
            x, y = smoothing_func(x, y, smooth_space)
        ax.plot(
            x,
            y,
            linewidth=2.0,
            label=method
            + y_metric.replace("test", "")
            .replace("_top1", "")
            .replace("corr", "corrupted")
            .replace("ood", "OoC")
            .replace("top1", "original"),
            linestyle=marker,
        )

    ax.set_xlabel("Communication rounds")
    ax.set_ylabel("Test accuracy")
    # ax.set_title(condition)

    # plt.legend(loc="lower left")
    plt.legend()
    if is_save:
        plt.savefig(path + y_metric + ".png")
    # plt.show()

    return ax


def plot_round_perf(info, comm_round, y_metrics, x_metric, condition, is_save, path):
    """Function that plot accuracy on a given comm_round."""
    # extract useful information.
    exp_args, records, perfs = extract_info(info, y_metrics)
    # make the plot.
    x = sorted(set([exp_arg[x_metric] for exp_arg in exp_args]))

    fig, axs = plt.subplots(1, len(y_metrics))
    for i, metric in enumerate(y_metrics):
        satisfying_perfs = sorted(
            [
                (exp_args[ind][x_metric], perf)
                for ind, perf in enumerate(perfs)
                if metric in perf.keys() and is_meet_condition(exp_args[ind], condition)
            ],
            key=lambda v: v[0],
        )

        y = [perf[1][metric][comm_round] for perf in satisfying_perfs]
        axs[i - 1].plot(x, y, linewidth=2.0, label=metric)
        axs[i - 1].legend()
        axs[i - 1].set_xlabel(x_metric)
        axs[i - 1].set_ylabel("Test accuracy")
        axs[i - 1].set_title("comm_round = " + str(comm_round))

    if is_save:
        plt.savefig(path + "comm_round" + str(comm_round) + ".png")
    plt.show()
    return axs


def extract_info(info, metrics):
    # extract useful information: experiment arguments and records of perf.
    exp_args = [ele_info[1]["arguments"] for ele_info in info]
    # each ele of records is a dict and it uses list of perf as values.
    records = [ele_info[1]["records0"] for ele_info in info]
    perfs = []
    for record in records:
        curr_perf = {}
        for key in record.keys():
            if key in metrics:
                curr_perf[key] = record[key]
        perfs.append(curr_perf)

    print("num of records:", len(perfs))
    print("num of exp_args:", len(exp_args))
    print("example performace:", perfs)

    return exp_args, records, perfs


def is_meet_condition(args, condition: dict):
    for k, v in condition.items():
        if args[k] == v:
            continue
        else:
            return False
    return True


def display_best_args(info, metric, measurement, arg_condition):
    """"""
    # extract useful infomation
    exp_args, _, perfs = extract_info(info, metric)
    # find the ind
    if measurement == "max":
        best = [max(perf[metric]) for perf in perfs]
        ind = np.argmax(best)
    else:
        best = [max(perf[metric]) for perf in perfs]
        ind = np.argmax(best)
    best_args = [
        (key, arg) for key, arg in exp_args[ind].items() if key in arg_condition
    ]
    # output its args
    print("The best value for " + metric + ": ", best[ind])
    print("Its arguemnts: ", best_args)

    return best_args
