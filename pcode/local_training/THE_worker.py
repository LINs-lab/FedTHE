# -*- coding: utf-8 -*-
import torch
import copy
import itertools
import pcode.utils.loss as loss
import torch.distributed as dist
from pcode.local_training.base_worker import BaseWorker
import pcode.create_dataset as create_dataset
from pcode.utils.stat_tracker import RuntimeTracker
import pcode.local_training.utils as utils
import pcode.create_optimizer as create_optimizer
import torch.nn.functional as F
import torch.nn as nn
from pcode.datasets.aug_data import aug
import torchvision.transforms as transforms


"""
Implementation of FedTHE (Ours)
    Test-Time Robust Personalization for Federated Learning
    https://arxiv.org/abs/2205.10920.
    When `is_finetune` is True, it becomes FedTHE+ (Ours).
"""


class THEWorker(BaseWorker):
    def __init__(self, conf, is_fine_tune=False):
        super(THEWorker, self).__init__(conf)
        self.conf = conf
        self.n_personalized_epochs = self.conf.n_personalized_epochs
        self.eval_dataset = self.conf.eval_dataset  # can be test_loader or val_loader
        self.is_in_childworker = True
        # test-time self-supervised aggregation
        self.num_head = 2
        self.THE_steps = self.conf.THE_steps
        self.agg_weight = torch.nn.Parameter(torch.rand((self.conf.batch_size, self.num_head)).cuda(), requires_grad=True)
        self.agg_weight.data.fill_(1 / self.num_head)
        self.alpha = self.conf.THE_alpha
        self.beta = self.conf.THE_beta

        self.is_tune_net = is_fine_tune
        self.is_rep_history_reused = self.conf.is_rep_history_reused

    def run(self):
        while True:
            self._listen_to_master()

            # receive global representation from server.
            self.global_rep = torch.zeros((self.conf.rep_len,))
            dist.broadcast(tensor=self.global_rep, src=0)
            dist.barrier()
            self.global_rep = self.global_rep.cuda()

            # check if we need to terminate the training or not.
            if self._terminate_by_early_stopping():
                return

            # receive the perform standard local training.
            self._recv_model_from_master()

            # create personal model and register hook.
            if not hasattr(self, "personal_head"):
                self.personal_head = nn.Linear(self.conf.rep_len, utils.get_num_classes(self.conf.data), bias=False)

            state = self._brm_train(model=self.model)
            params_to_send = copy.deepcopy(state["model"].state_dict())  # use deepcopy

            # personalization.
            p_state = self._personalized_train(model=self.model)

            # evaluate the personalized model.
            perf = self._evaluate_all_test_sets(p_state)

            # display the personalized perf and sync the model & perf.
            self._display_info(p_state, perf)

            # also send local rep
            perf.extend(self.local_rep.cpu().squeeze().tolist())
            self._send_model_to_master(params_to_send, perf)
            del state, p_state

            if self._terminate_by_complete_training():
                return

    def _brm_train(self, model):
        # change the criterion to balanced loss and do local training.
        self.criterion = loss.BalancedSoftmaxLoss(self._get_target_histogram())
        state = super(THEWorker, self).train(model)
        return state

    def _personalized_train(self, model):
        self.is_in_personalized_training = True
        self.n_local_epochs = self.n_personalized_epochs
        self.erm_criterion = nn.CrossEntropyLoss(reduction="mean")
        # define dataloader, optimizer, scheduler and tracker
        state = self._init_train_process(model=model)
        self.personal_head.to(self.graph.device)
        # we want to optimize personal head
        state["optimizer"] = create_optimizer.define_optimizer(
            self.conf, model=self.personal_head, optimizer_name=self.conf.optimizer, lr=self._get_round_lr()
        )
        # freeze the model, except the personal head
        state["model"].requires_grad_(False)
        self.personal_head.requires_grad_(True)
        self.rep_layer = utils.determine_hook(state["model"], name=self.conf.arch)
        self.rep_layer.register_forward_hook(utils.hook)
        while not self._is_finished_one_comm_round(state):
            self.local_rep = []
            self.per_class_rep = {i: [] for i in range(utils.get_num_classes(self.conf.data))}
            for _input, _target in state["train_loader"]:
                # load data
                with self.timer("load_data", epoch=state["scheduler"].epoch_):
                    data_batch = create_dataset.load_data_batch(
                        self.conf, _input, _target, is_training=True,
                    )

                # inference and get current performance.
                with self.timer("forward_pass", epoch=state["scheduler"].epoch_):
                    state["optimizer"].zero_grad()
                    g_out = state["model"](data_batch["input"])
                    p_out = self.personal_head(self.rep_layer.rep)
                    loss = self.erm_criterion(p_out, data_batch["target"])
                    agg_out = torch.stack([g_out, p_out], dim=1).mean(dim=1)
                    performance = self.metrics.evaluate(loss, agg_out, data_batch["target"])
                    state["tracker"].update_metrics(
                        [loss.item()] + performance, n_samples=data_batch["input"].size(0)
                    )
                    for i, label in enumerate(data_batch["target"]):
                        self.per_class_rep[label.item()].append(self.rep_layer.rep[i, :].unsqueeze(0))

                with self.timer("backward_pass", epoch=state["scheduler"].epoch_):
                    loss.backward()
                    state["optimizer"].step()
                    state["scheduler"].step()

                if self.conf.display_log:
                    self._display_logging(state)
                if self._is_diverge(state):
                    self._terminate_comm_round(state)
                    return state

            # refresh the logging cache at the end of each epoch.
            state["tracker"].reset()
            if self.logger.meet_cache_limit():
                self.logger.save_json()

        # terminate
        self._compute_prototype()
        self._terminate_comm_round(state)
        self.is_in_personalized_training = False
        return state

    def _compute_prototype(self):
        # compute the average representation of local training set.
        for (k, v) in self.per_class_rep.items():
            if len(v) != 0:
                self.local_rep.append(torch.cat(v).cuda())
        self.local_rep = torch.cat(self.local_rep).mean(dim=0).cuda()

    def _test_time_tune(self, state, data_batch, num_steps=3):
        # turn on model grads.
        state["model"].requires_grad_(True)
        self.personal_head.requires_grad_(True)
        # set optimizer.
        fe_optim = torch.optim.SGD(state["model"].parameters(), lr=0.0005)
        fe_optim.add_param_group({"params": self.personal_head.parameters()})
        g_pred, p_pred = [], []
        # do the unnormalize to ensure consistency.
        normalize, unnormalize = utils.get_normalization(self.conf.data)
        convert_img = transforms.Compose([unnormalize, transforms.ToPILImage()])
        agg_softmax = torch.nn.functional.softmax(self.agg_weight).detach()
        model_param = copy.deepcopy(state["model"].state_dict())
        p_head_param = copy.deepcopy(self.personal_head.state_dict())
        for i in range(data_batch["input"].shape[0]):
            image = convert_img(data_batch["input"][i])
            for _ in range(num_steps):
                # generate a batch of augmentations and minimize prediction entropy.
                inputs = [aug(image, normalize) for _ in range(16)]
                inputs = torch.stack(inputs).cuda()
                fe_optim.zero_grad()
                g_out = state["model"](inputs)
                p_out = self.personal_head(self.rep_layer.rep)
                agg_output = agg_softmax[i, 0] * g_out + agg_softmax[i, 1] * p_out
                loss, _ = utils.marginal_entropy(agg_output)
                loss.backward()
                fe_optim.step()
            with torch.no_grad():
                g_out = state["model"](data_batch["input"][i].unsqueeze(0).cuda())
                p_out = self.personal_head(self.rep_layer.rep)
                g_pred.append(g_out)
                p_pred.append(p_out)
            state["model"].load_state_dict(model_param)
            self.personal_head.load_state_dict(p_head_param)
        # turn off grads.
        state["model"].requires_grad_(False)
        self.personal_head.requires_grad_(False)
        return torch.cat(g_pred), torch.cat(p_pred)

    def _validate_training(self, state, dataset, num_epochs):
        self.is_in_personalized_training = True
        # dont requires gradients.
        state["model"].requires_grad_(False)
        self.personal_head.requires_grad_(False)
        state["model"].eval()
        tracker_te = RuntimeTracker(metrics_to_track=self.metrics.metric_names)
        # if enabled, then the test history is reused between test sets.
        if self.is_rep_history_reused:
            if not hasattr(self, "test_history"):
                self.test_history = None
        else:
            self.test_history = None

        for _input, _target in dataset:
            data_batch = create_dataset.load_data_batch(
                self.conf, _input, _target, is_training=False
            )
            with torch.no_grad():
                # update test history by exponential moving average.
                _ = state["model"](data_batch["input"])
                test_rep = self.rep_layer.rep.detach()
                test_history = None
                for i in range(test_rep.shape[0]):
                    if test_history is None and self.test_history is None:
                        test_history = [test_rep[0, :]]
                    elif test_history is None and self.test_history is not None:
                        test_history = [self.test_history[-1, :]]
                    else:
                        test_history.append(self.alpha * test_rep[i, :] + (1 - self.alpha) * test_history[-1])
                self.test_history = torch.stack(test_history)
                temperature = torch.hstack((torch.ones((test_rep.shape[0], 1)).cuda(), torch.ones((test_rep.shape[0], 1)).cuda()))

            self.agg_weight = torch.nn.Parameter(torch.tensor(temperature).cuda(), requires_grad=True)
            state["optimizer"] = torch.optim.Adam([self.agg_weight], lr=10*self._get_round_lr())

            self._calculate_samplewise_weight(state, data_batch, num_epochs)
            if self.is_tune_net:
                # test-timely tune the whole net.
                g_pred, p_pred = self._test_time_tune(state, data_batch, num_steps=3)

            # do inference for current batch
            with torch.no_grad():
                if self.is_tune_net:
                    self._multi_head_inference(data_batch, state["model"], tracker_te, g_pred, p_pred)
                else:
                    self._multi_head_inference(data_batch, state["model"], tracker_te)

        self.is_in_personalized_training = False
        self.agg_weight.data.fill_(1 / self.num_head)
        return tracker_te

    def _calculate_samplewise_weight(self, state, data_batch, num_epochs):
        # function that optimize the ensemble weights.
        g_out = state["model"](data_batch["input"])
        test_rep = self.rep_layer.rep
        p_out = self.personal_head(test_rep)
        grad_norm, loss_batch = [], []
        for _ in range(num_epochs):
            # normalize the aggregation weight by softmax
            agg_softmax = torch.nn.functional.softmax(self.agg_weight)
            agg_output = agg_softmax[:, 0].unsqueeze(1) * g_out.detach() \
                         + agg_softmax[:, 1].unsqueeze(1) * p_out.detach()
            # formulate test representation.
            test_rep = self.beta * self.rep_layer.rep + (1 - self.beta) * self.test_history
            p_feat_al = torch.norm((test_rep - self.local_rep), dim=1)
            g_feat_al = torch.norm((test_rep - self.global_rep), dim=1)
            cos = torch.nn.CosineSimilarity(dim=1, eps=1e-6)
            sim = cos(F.softmax(g_out).detach(), F.softmax(p_out).detach())
            # loss function based on prediction similarity, entropy minimization and feature alignment.
            loss = (-sim * (agg_output.softmax(1) * agg_output.log_softmax(1)).sum(1) + \
                    (1 - sim) * (agg_softmax[:, 0] * g_feat_al.detach() + agg_softmax[:, 1] * p_feat_al.detach())).mean(0)
            state["optimizer"].zero_grad()
            loss.backward()

            if torch.norm(self.agg_weight.grad) < 1e-5:
                break
            grad_norm.append(torch.norm(self.agg_weight.grad).item())
            loss_batch.append(loss.item())
            state["optimizer"].step()

    def _multi_head_inference(self, data_batch, model, tracker=None, g_pred=None, p_pred=None):
        # inference procedure for multi-head nets.
        agg_softmax = torch.nn.functional.softmax(self.agg_weight)
        if g_pred is not None and p_pred is not None:
            agg_output = agg_softmax[:, 0].unsqueeze(1) * g_pred \
                         + agg_softmax[:, 1].unsqueeze(1) * p_pred
        else:
            # do the forward pass and get the output.
            g_out = model(data_batch["input"])
            p_out = self.personal_head(self.rep_layer.rep)
            agg_output = agg_softmax[:, 0].unsqueeze(1) * g_out \
                         + agg_softmax[:, 1].unsqueeze(1) * p_out
        # evaluate the output and get the loss, performance.
        loss = self.criterion(agg_output, data_batch["target"])
        performance = self.metrics.evaluate(loss, agg_output, data_batch["target"])

        # update tracker.
        if tracker is not None:
            tracker.update_metrics(
                [loss.item()] + performance, n_samples=data_batch["input"].size(0)
            )

        return loss

    def _validate(self, state, dataset_name):
        # switch to evaluation mode.
        state["model"].eval()
        self.rep_layer = utils.determine_hook(state["model"], name=self.conf.arch)
        self.rep_layer.register_forward_hook(utils.hook)
        # test-time self-supervised aggregation
        tracker_te = self._validate_training(state, state[dataset_name], self.THE_steps)
        return tracker_te()

    def _get_target_histogram(self, display=True):
        local_data_loaders = self.fl_data_cls.intra_client_data_partition_and_create_dataloaders(
            localdata_id=self.client_id - 1,  # localdata_id starts from 0 while client_id starts from 1.
            other_ids=self._get_other_ids(),
            is_in_childworker=self.is_in_childworker,
            local_train_ratio=self.conf.local_train_ratio,
            batch_size=1,
            display_log=False,
        )
        hist = torch.zeros(utils.get_num_classes(self.conf.data))
        train_loader = local_data_loaders["train"]
        for _, _target in train_loader:
            hist[_target.item()] += 1
        if display:
            self.logger.log(
                f"\tWorker-{self.graph.worker_id} (client-{self.client_id}) training histogram is like {hist}"
            )

        return hist

