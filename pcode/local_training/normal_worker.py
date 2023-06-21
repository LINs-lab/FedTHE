# -*- coding: utf-8 -*-
import copy
from pcode.local_training.base_worker import BaseWorker


"""Standard FedAvg, and no specific personalized FL techniques."""


class NormalWorker(BaseWorker):
    def __init__(self, conf):
        super(NormalWorker, self).__init__(conf)
        self.eval_dataset = conf.eval_dataset  # can be test_loader or val_loader
        self.is_in_childworker = True

    def run(self):
        while True:
            self._listen_to_master()

            # check if we need to terminate the training or not.
            if self._terminate_by_early_stopping():
                return

            # receive the perform standard local training.
            self._recv_model_from_master()

            state = self.train(model=self.model)
            params_to_send = copy.deepcopy(state["model"].state_dict())

            # evaluate the model.
            state = self._init_train_process(self.model)
            perf = self._evaluate_all_test_sets(state)

            # display the info and sync the model & perf.
            self._display_info(state, perf)
            self._send_model_to_master(params_to_send, perf)

            del state

            if self._terminate_by_complete_training():
                return


