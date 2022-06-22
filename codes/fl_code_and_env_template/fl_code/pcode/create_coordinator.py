# -*- coding: utf-8 -*-
import pcode.utils.stat_tracker as stat_tracker


class Coordinator(object):
    def __init__(self, metrics, best_perf=None):
        # init
        self.metrics_names = metrics.metric_names
        self.build_best_trackers(best_perf)

    def build_best_trackers(self, best_perf):
        self.best_trackers = {}
        for name in ["loss"] + self.metrics_names:
            self.best_trackers[name] = stat_tracker.BestPerf(
                best_perf=best_perf,
                larger_is_better=True if "loss" not in name else False,
            )

    def update_perf(self, performance, comm_round):
        for name, perf in performance.items():
            self.best_trackers[name].update(perf, comm_round)

    def __call__(self):
        return dict(
            (name, (best_tracker.best_perf, best_tracker.get_best_perf_loc))
            for name, best_tracker in self.best_trackers.items()
        )

    @property
    def key_metric(self):
        return self.best_trackers[self.metrics_names[0]]
