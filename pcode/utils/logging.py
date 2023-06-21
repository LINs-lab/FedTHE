# -*- coding: utf-8 -*-
import os
import json
import time
import platform

from pcode.utils.op_files import write_txt


class Logger:
    """
    Very simple prototype logger that will store the values to a JSON file
    """

    def __init__(self, file_folder):
        """
        :param filename: ending with .json
        :param auto_save: save the JSON file after every addition
        """
        self.file_folder = file_folder
        self.file_json = os.path.join(file_folder, "log-1.json")
        self.file_txt = os.path.join(file_folder, "log.txt")
        self.values = []

    def log_metric(self, name, values, tags, display=False):
        """
        Store a scalar metric

        :param name: measurement, like 'accuracy'
        :param values: dictionary, like { epoch: 3, value: 0.23 }
        :param tags: dictionary, like { split: train }
        """
        self.values.append({"measurement": name, **values, **tags})
        if display:
            print(
                "{name}: {values} ({tags})".format(name=name, values=values, tags=tags)
            )

    def log(self, value, display=True):
        content = time.strftime("%Y-%m-%d %H:%M:%S") + "\t" + value
        print(content)
        self.save_txt(content)

    def save_json(self):
        """
        Save the internal memory to a file
        """
        with open(self.file_json, "w") as fp:
            json.dump(self.values, fp, indent=" ")

        # reset 'values' and redirect the json file to other name.
        if self.meet_cache_limit():
            self.values = []
            self.redirect_new_json()

    def save_txt(self, value):
        write_txt(value + "\n", self.file_txt, type="a")

    def redirect_new_json(self):
        """get the number of existing json files under the current folder."""
        existing_json_files = [
            file for file in os.listdir(self.file_folder) if "json" in file
        ]
        self.file_json = os.path.join(
            self.file_folder, "log-{}.json".format(len(existing_json_files) + 1)
        )

    def meet_cache_limit(self):
        return True if len(self.values) > 1e4 else False


def display_args(conf):
    print("\n\nparameters: ")
    for arg in vars(conf):
        print("\t" + str(arg) + "\t" + str(getattr(conf, arg)))

    print(f"\n\nexperiment platform: on {platform.node()} {conf.graph.device}")
    for name in ["n_participated", "world", "rank", "devices", "on_cuda"]:
        print("\t{}: {}".format(name, getattr(conf.graph, name)))
    print("\n\n")


def display_training_stat(conf, tracker, **kwargs):
    current_time = time.strftime("%Y-%m-%d %H:%M:%S")

    # display the runtime training information.
    conf.logger.log_metric(
        name="runtime",
        values={
            "time": current_time,
            "worker_id": conf.graph.worker_id,
            **kwargs,
            **tracker(),
        },
        tags={"split": "train"},
        display=True,
    )


def display_general_stat(conf, tracker, split, label):
    current_time = time.strftime("%Y-%m-%d %H:%M:%S")

    # display the runtime training information.
    conf.logger.log_metric(
        name="runtime",
        values={
            "time": current_time,
            "rank": conf.graph.rank,
            "comm_round": conf.graph.comm_round,
            **tracker(),
        },
        tags={"split": split, "type": label},
        display=True,
    )
    conf.logger.save_json()


def display_perf(conf, perf_summary):
    current_time = time.strftime("%Y-%m-%d %H:%M:%S")

    # add to the json log.
    conf.logger.log_metric(
        name="runtime",
        values={"time": current_time, **perf_summary},
        tags={"split": "test"},
    )
    # add to the txt log.
    conf.logger.log(
        f"Master aggregates the ave perf of {conf.n_participated} local models in round {perf_summary['round']}."
    )
    conf.logger.log(
        f"Current local ave top1 perf: comm_round {perf_summary['round']}, "
        f"top1: {perf_summary['top1']}"
    )
    conf.logger.log(
        f"Current local ave corr top1 perf: comm_round {perf_summary['round']}, "
        f"top1: {perf_summary['corr_top1']}"
    )
    conf.logger.log(
        f"Current local ave ooc top1 perf: comm_round {perf_summary['round']}, "
        f"top1: {perf_summary['ooc_top1']}"
    )
    conf.logger.log(
        f"Current local ave natural shift top1 perf: comm_round {perf_summary['round']}, "
        f"top1: {perf_summary['natural_shift_top1']}"
    )
    conf.logger.log(
        f"Current local ave corr ooc top1 perf: comm_round {perf_summary['round']}, "
        f"top1: {perf_summary['ooc_corr_top1']}"
    )
    conf.logger.log(
        f"Current local ave mixed top1 perf: comm_round {perf_summary['round']}, "
        f"top1: {perf_summary['mixed_top1']}"
    )


def display_best_test_stat(conf, coordinator, comm_round):
    for name, best_tracker in coordinator.best_trackers.items():
        conf.logger.log(
            "Best performance of {} \
            (best comm_round {:.3f}, current comm_round {:.3f}): {}.".format(
                name, best_tracker.get_best_perf_loc, comm_round, best_tracker.best_perf
            )
        )
