# -*- coding: utf-8 -*-
import copy
from pcode.local_training.base_worker import BaseWorker


"""Standard FedAvg, and no specific personalized FL techniques."""


class NormalWorker(BaseWorker):
    def __init__(self, conf):
        super(NormalWorker, self).__init__(conf)
        self.eval_dataset = conf.eval_dataset  # can be test_loader or val_loader
        self.is_in_childworker = True

        # BN related
        self.reuse_BN_params = conf.reuse_BN_params

    def run(self):
        while True:
            self._listen_to_master()

            # check if we need to terminate the training or not.
            if self._terminate_by_early_stopping():
                return

            # receive the perform standard local training.
            self._recv_model_from_master()
            # a personal model to hold local parameters. In Normal worker, it is used to hold BN params if needed.
            if not hasattr(self, "personal_model"):
                self.personal_model = copy.deepcopy(self.model)

            # reload BN params
            if self.reuse_BN_params:
                self._fill_BN_params(self.model, self.personal_model)

            state = self.train(model=self.model)
            params_to_send = copy.deepcopy(state["model"].state_dict())

            self.personal_model.load_state_dict(params_to_send)

            # evaluate the model.
            state = self._init_train_process(self.model)
            perf = self._evaluate_all_test_sets(state, is_order_sensitive=False)

            # display the info and sync the model & perf.
            self._display_info(state, perf)
            self._send_model_to_master(params_to_send, perf)

            del state

            if self._terminate_by_complete_training():
                return

    def _fill_BN_params(self, target_model, source_model):
        # source_model typically is self.personal_model
        model_tmp = copy.deepcopy(target_model.state_dict())
        params = source_model.state_dict()
        for layer in model_tmp:
            if "bn" in layer:
                model_tmp[layer] = copy.deepcopy(params[layer])
        target_model.load_state_dict(model_tmp)
