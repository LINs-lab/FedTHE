# -*- coding: utf-8 -*-
import torch
import copy
import os
from tqdm import tqdm
import functools
import torch.distributed as dist
import numpy as np
import torch.utils.data as data
from typing import List
from pcode.local_training.base_worker import BaseWorker
import pcode.create_dataset as create_dataset
from pcode.utils.stat_tracker import RuntimeTracker
import pcode.local_training.utils as utils
from pcode.utils.mathdict import MathDict
import torch.nn.functional as F
import torch.nn as nn
from pcode.models.simple_cnns import CosNorm_Classifier
from pcode.datasets.aug_data import aug
import pcode.create_optimizer as create_optimizer
import torchvision.transforms as transforms
import pcode.datasets.corr_data as corr_data
import torchvision
from pcode.datasets.partition_data import DataPartitioner
from pcode.datasets.partition_data import (
    record_class_distribution,
    partition_by_other_histogram,
)

"""
Our test-time worker in single machine setting. It will load a pretrained model and only finetune the personalized head,
and evaluate on distribution shift case.
"""


class THESingleWorker(BaseWorker):
    def __init__(self, conf, is_fine_tune=False):
        super(THESingleWorker, self).__init__(conf)
        self.conf = conf
        self.is_in_childworker = True
        self.brm_loss = conf.brm_loss
        # test-time self-supervised aggregation
        self.num_head = 2
        self.THE_steps = conf.THE_steps
        self.agg_weight = torch.nn.Parameter(
            torch.rand((self.conf.batch_size, self.num_head)).cuda(), requires_grad=True
        )
        self.agg_weight.data.fill_(1 / self.num_head)

        # round list
        self.THE_round_list = [1]
        self.check_round_list = [1, 2]
        self.is_tune_net = is_fine_tune
        self.is_rep_history_reused = conf.is_rep_history_reused
        self.pretrained_through = conf.pretrained_through

        if self.pretrained_through == "mocov2":
            self.ckpt_epochs = 800
            self.ckpt_path = (
                "/data1/user/pretrained_model/moco_v2_"
                + str(self.ckpt_epochs)
                + "ep_pretrain.pth.tar"
            )
        elif self.pretrained_through == "mocov1":
            self.ckpt_epochs = 200
            self.ckpt_path = (
                "/data1/user/pretrained_model/moco_v1_"
                + str(self.ckpt_epochs)
                + "ep_pretrain.pth.tar"
            )
        elif self.pretrained_through == "imagenet":
            pass

    def run(self):
        while True:
            self._listen_to_master()

            # receive global representation from server. all zeros in this case.
            self.global_rep = torch.zeros((self.conf.rep_len,))
            dist.broadcast(tensor=self.global_rep, src=0)
            dist.barrier()
            self.global_rep = self.global_rep.cuda()

            # check if we need to terminate the training or not.
            if self._terminate_by_early_stopping():
                return

            # receive the perform standard local training.
            self._recv_model_from_master()
            params_to_send = copy.deepcopy(self.model.state_dict())  # use deepcopy

            if not hasattr(self, "pretrained_model"):
                model = torchvision.models.resnet34
                self.pretrained_model = model(pretrained=False, num_classes=10)
                if self.pretrained_through == "imagenet":
                    pretrained_resnet = model(pretrained=True, progress=True)
                    state_dict = pretrained_resnet.state_dict()
                    del state_dict["fc.weight"]
                    del state_dict["fc.bias"]
                elif (
                    self.pretrained_through == "mocov2"
                    or self.pretrained_through == "mocov1"
                ):
                    ckpt = torch.load(self.ckpt_path)
                    state_dict = ckpt["state_dict"]
                    for k in list(state_dict.keys()):
                        # retain only encoder_q up to before the embedding layer
                        if k.startswith("module.encoder_q") and not k.startswith(
                            "module.encoder_q.fc"
                        ):
                            # remove prefix
                            state_dict[k[len("module.encoder_q.") :]] = state_dict[k]
                        # delete renamed or unused k
                        del state_dict[k]
                msg = self.pretrained_model.load_state_dict(state_dict, strict=False)
                assert set(msg.missing_keys) == {"fc.weight", "fc.bias"}
                # init the fc layer
                self.pretrained_model.fc.weight.data.normal_(mean=0.0, std=0.01)
                self.pretrained_model.fc.bias.data.zero_()

            if not hasattr(self, "personal_head"):
                self.conf.rep_len = 512
                # self.personal_head = CosNorm_Classifier(self.conf.rep_len, utils.get_num_classes(self.conf.data))
                self.personal_head = nn.Linear(
                    self.conf.rep_len, utils.get_num_classes(self.conf.data)
                )
                # init the fc layer
                self.personal_head.weight.data.normal_(mean=0.0, std=0.01)
                self.personal_head.bias.data.zero_()

            # freeze pretrained model params
            for name, param in self.pretrained_model.named_parameters():
                if name not in ["fc.weight", "fc.bias"]:
                    param.requires_grad = False

            g_tr_loader, p_tr_loader, test_loaders = self._get_dataset()
            self.global_rep = self._train_global_head(
                model=self.pretrained_model, train_loader=g_tr_loader
            )

            perf = []
            perf.append(
                self._evaluate_model(
                    model=self.pretrained_model,
                    model_name="global",
                    val_loader=test_loaders["original"],
                )
            )
            perf.append(
                self._evaluate_model(
                    model=self.pretrained_model,
                    model_name="global",
                    val_loader=test_loaders["local"],
                )
            )
            perf.append(
                self._evaluate_model(
                    model=self.pretrained_model,
                    model_name="global",
                    val_loader=test_loaders["synthetic_shift"],
                )
            )
            perf.append(
                self._evaluate_model(
                    model=self.pretrained_model,
                    model_name="global",
                    val_loader=test_loaders["natural_shift"],
                )
            )
            self.display_results(perf)
            # personalization.
            p_state = self._personalized_train(
                model=self.pretrained_model, train_loader=p_tr_loader
            )
            perf = []

            perf.append(
                self._evaluate_model(
                    model=self.pretrained_model,
                    model_name="personal",
                    val_loader=test_loaders["original"],
                )
            )
            perf.append(
                self._evaluate_model(
                    model=self.pretrained_model,
                    model_name="personal",
                    val_loader=test_loaders["local"],
                )
            )
            perf.append(
                self._evaluate_model(
                    model=self.pretrained_model,
                    model_name="personal",
                    val_loader=test_loaders["synthetic_shift"],
                )
            )
            perf.append(
                self._evaluate_model(
                    model=self.pretrained_model,
                    model_name="personal",
                    val_loader=test_loaders["natural_shift"],
                )
            )
            self.display_results(perf)
            # evaluation without THE or THE-FT
            self.comm_round = 2
            perf = []
            perf.append(
                self._evaluate_model(
                    model=self.pretrained_model,
                    model_name="personal",
                    val_loader=test_loaders["original"],
                )
            )
            perf.append(
                self._evaluate_model(
                    model=self.pretrained_model,
                    model_name="personal",
                    val_loader=test_loaders["local"],
                )
            )
            perf.append(
                self._evaluate_model(
                    model=self.pretrained_model,
                    model_name="personal",
                    val_loader=test_loaders["synthetic_shift"],
                )
            )
            perf.append(
                self._evaluate_model(
                    model=self.pretrained_model,
                    model_name="personal",
                    val_loader=test_loaders["natural_shift"],
                )
            )

            self.display_results(perf)
            # default setting
            self._send_model_to_master(
                params_to_send, []
            )  # no need to send perf, because this is in single machine.
            del p_state

            if self._terminate_by_complete_training():
                return

    def _train_global_head(self, model, train_loader):
        # define dataloader, optimizer, scheduler and tracker
        def hook(module, input, output):
            module.rep = input[0].detach()  # the input of fc layer

        model.to(self.graph.device)
        model.eval()  # for keeping the statistics
        model.fc.register_forward_hook(hook)
        rep_layer = model.fc
        parameters = list(filter(lambda p: p.requires_grad, model.parameters()))
        optimizer = torch.optim.SGD(
            parameters,
            lr=self.conf.lr,
            momentum=self.conf.momentum_factor,
            weight_decay=self.conf.weight_decay,
        )
        state = {"model": model, "optimizer": optimizer, "train_loader": train_loader}
        self.erm_criterion = nn.CrossEntropyLoss()
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
            optimizer, int(self.conf.local_n_epochs)
        )
        for _ in tqdm(range(int(self.conf.local_n_epochs))):
            global_rep = []
            train_loss, total, correct = 0.0, 0, 0
            self.global_per_class_rep = {
                i: [] for i in range(utils.get_num_classes(self.conf.data))
            }
            for _input, _target in tqdm(state["train_loader"]):
                if self.conf.graph.on_cuda:
                    _input, _target = _input.cuda(), _target.cuda()
                data_batch = {"input": _input, "target": _target}
                # inference and get current performance.
                state["optimizer"].zero_grad()
                outputs = state["model"](data_batch["input"])
                loss = self.erm_criterion(outputs, data_batch["target"])
                for i, label in enumerate(_target):
                    self.global_per_class_rep[label.item()].append(
                        rep_layer.rep[i, :].unsqueeze(0)
                    )

                loss.backward()
                state["optimizer"].step()

                train_loss += loss.item()
                total += _target.size(0)
                _, predicted = outputs.max(1)
                correct += predicted.eq(_target).sum().item()

            scheduler.step()
            print("current training loss:", train_loss)
            print("current training acc:", 100.0 * correct / total)

        # compute the average representation of local training set.
        for (k, v) in self.global_per_class_rep.items():
            if len(v) != 0:
                global_rep.append(torch.cat(v).cuda())
        global_rep = torch.cat(global_rep).mean(dim=0).cuda()
        return global_rep

    def _evaluate_model(self, model, model_name, val_loader):
        self.is_in_personalized_training = True
        # dont requires gradients.
        model.requires_grad_(False)
        self.personal_head.requires_grad_(False)
        # define dataloader, optimizer, scheduler and tracker
        model.eval()
        tracker_te = RuntimeTracker(metrics_to_track=self.metrics.metric_names)
        state = {"model": model}
        if model_name == "global":
            for _input, _target in tqdm(val_loader):
                # inference and get current performance.
                if self.conf.graph.on_cuda:
                    _input, _target = _input.cuda(), _target.to(dtype=torch.long).cuda()
                data_batch = {"input": _input, "target": _target}
                with torch.no_grad():
                    self._inference(data_batch, state["model"], tracker_te)
            return list(tracker_te().values())[1]
        if model_name == "personal":
            if self.comm_round in self.THE_round_list:
                # test-time self-supervised aggregation
                global_rep = self.global_rep
                # if enabled, then the test history is reused between test sets.
                if self.is_rep_history_reused:
                    if not hasattr(self, "test_history"):
                        self.test_history = None
                else:
                    self.test_history = None
                # determine the number of batches to sample.
                for _input, _target in tqdm(val_loader):
                    if self.conf.graph.on_cuda:
                        _input, _target = (
                            _input.cuda(),
                            _target.to(dtype=torch.long).cuda(),
                        )
                    data_batch = {"input": _input, "target": _target}
                    with torch.no_grad():
                        # update test history by exponential moving average.
                        _ = state["model"](data_batch["input"])
                        test_rep = self.rep_layer.rep.detach()
                        a = 0.1
                        b = test_rep.shape[0]
                        test_history = None
                        for i in range(b):
                            if test_history is None and self.test_history is None:
                                test_history = [test_rep[0, :]]
                            elif test_history is None and self.test_history is not None:
                                test_history = [self.test_history[-1, :]]
                            else:
                                test_history.append(
                                    a * test_rep[i, :] + (1 - a) * test_history[-1]
                                )
                        self.test_history = torch.stack(test_history)
                        temperature = torch.hstack(
                            (torch.ones((b, 1)).cuda(), torch.ones((b, 1)).cuda())
                        )

                    self.agg_weight = torch.nn.Parameter(
                        torch.tensor(temperature).cuda(), requires_grad=True
                    )
                    state["optimizer"] = torch.optim.Adam([self.agg_weight], lr=0.1)

                    self._calculate_samplewise_weight(
                        state, data_batch, self.THE_steps, global_rep
                    )
                    if self.is_tune_net:
                        # test-timely tune the whole net.
                        g_pred, p_pred = self._test_time_tune(
                            state, data_batch, num_steps=3
                        )

                    # do inference for current batch
                    with torch.no_grad():
                        if self.is_tune_net:
                            self._multi_head_inference(
                                data_batch, state["model"], tracker_te, g_pred, p_pred
                            )
                        else:
                            self._multi_head_inference(
                                data_batch, state["model"], tracker_te
                            )

                    self.agg_weight.data.fill_(1 / self.num_head)
            else:
                for _input, _target in tqdm(val_loader):
                    if self.conf.graph.on_cuda:
                        _input, _target = (
                            _input.cuda(),
                            _target.to(dtype=torch.long).cuda(),
                        )
                    data_batch = {"input": _input, "target": _target}
                    self.agg_weight = torch.tile(
                        torch.tensor([0.5, 0.5]), (_target.shape[0], 1)
                    ).cuda()
                    with torch.no_grad():
                        self._multi_head_inference(
                            data_batch, state["model"], tracker_te
                        )
            return list(tracker_te().values())[1]

    def _personalized_train(self, model, train_loader):
        self.is_in_personalized_training = True
        self.n_local_epochs = int(self.conf.n_personalized_epochs)
        self.erm_criterion = nn.CrossEntropyLoss()
        # define dataloader, optimizer, scheduler
        optimizer = create_optimizer.define_optimizer(
            self.conf,
            model=self.personal_head,
            optimizer_name=self.conf.optimizer,
            lr=self.conf.personal_lr,
        )
        state = {"model": model, "optimizer": optimizer, "train_loader": train_loader}
        # freeze the model, except the personal head
        state["model"].requires_grad_(False)
        self.personal_head.requires_grad_(True)

        def hook(module, input, output):
            module.rep = input[0].detach()  # the input of fc layer

        model.to(self.graph.device)
        self.personal_head.to(self.graph.device)
        model.eval()
        model.fc.register_forward_hook(hook)
        self.rep_layer = model.fc
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
            optimizer, self.n_local_epochs
        )
        for i in tqdm(range(self.n_local_epochs)):
            if i == self.n_local_epochs - 1:
                self.rep = []
                self.per_class_rep = {
                    i: [] for i in range(utils.get_num_classes(self.conf.data))
                }
            train_loss, total, correct = 0.0, 0, 0
            for _input, _target in tqdm(state["train_loader"]):
                # load data
                if self.conf.graph.on_cuda:
                    _input, _target = _input.cuda(), _target.cuda()
                data_batch = {"input": _input, "target": _target}

                state["optimizer"].zero_grad()
                g_out = state["model"](data_batch["input"])
                p_out = self.personal_head(self.rep_layer.rep)
                loss = self.erm_criterion(p_out, data_batch["target"])
                agg_out = torch.stack([g_out, p_out], dim=1).mean(dim=1)

                if hasattr(self, "per_class_rep"):
                    for j, label in enumerate(data_batch["target"]):
                        self.per_class_rep[label.item()].append(
                            self.rep_layer.rep[i, :].unsqueeze(0)
                        )

                loss.backward()
                state["optimizer"].step()

                train_loss += loss.item()
                total += _target.size(0)
                _, predicted = agg_out.max(1)
                correct += predicted.eq(_target).sum().item()

            scheduler.step()

            print("current training loss:", train_loss)
            print("current training acc:", 100.0 * correct / total)

        # terminate
        self._compute_prototype()
        self.is_in_personalized_training = False
        return state

    def _compute_prototype(self):
        # compute the average representation of local training set.
        for (k, v) in self.per_class_rep.items():
            if len(v) != 0:
                self.rep.append(torch.cat(v).cuda())
        self.rep = torch.cat(self.rep).mean(dim=0).cuda()

    def _test_time_tune(self, state, data_batch, num_steps=3):
        # turn on model grads.
        state["model"].requires_grad_(True)
        self.personal_head.requires_grad_(True)
        # set optimizer.
        fe_optim = torch.optim.SGD(state["model"].parameters(), lr=0.0005)
        fe_optim.add_param_group({"params": self.personal_head.parameters()})
        g_pred, p_pred = [], []
        loss_batch = []
        # do the unnormalize to ensure consistency.
        unnormalize = transforms.Compose(
            [
                transforms.Normalize(
                    mean=[0.0, 0.0, 0.0], std=[1 / 0.229, 1 / 0.224, 1 / 0.225]
                ),
                transforms.Normalize(
                    mean=[-0.485, -0.456, -0.406], std=[1.0, 1.0, 1.0]
                ),
            ]
        )
        normalize = transforms.Normalize((0.485, 0.456, 0.406), (0.228, 0.224, 0.225))
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
                loss, logits = utils.marginal_entropy(agg_output)
                loss.backward()
                fe_optim.step()
                loss_batch.append(loss.item())
            with torch.no_grad():
                g_out = state["model"](data_batch["input"][i].unsqueeze(0).cuda())
                p_out = self.personal_head(self.rep_layer.rep)
                g_pred.append(g_out)
                p_pred.append(p_out)
            # if self.client_id == 1:
            #     print("marginal entropy loss:", loss_batch)
            #     loss_batch = []
            state["model"].load_state_dict(model_param)
            self.personal_head.load_state_dict(p_head_param)
        # turn off grads.
        state["model"].requires_grad_(False)
        self.personal_head.requires_grad_(False)
        return torch.cat(g_pred), torch.cat(p_pred)

    def _calculate_samplewise_weight(
        self, state, data_batch, num_epochs, global_rep, display=True
    ):
        # function that optimize the ensemble weights.
        g_out = state["model"](data_batch["input"])
        test_rep = self.rep_layer.rep
        p_out = self.personal_head(test_rep)
        grad_norm, loss_batch = [], []
        for _ in range(num_epochs):
            # normalize the aggregation weight by softmax
            agg_softmax = torch.nn.functional.softmax(self.agg_weight)
            agg_output = (
                agg_softmax[:, 0].unsqueeze(1) * g_out.detach()
                + agg_softmax[:, 1].unsqueeze(1) * p_out.detach()
            )
            p_rep = self.rep
            g_rep = global_rep
            # formulate test representation.
            beta = 0.3
            test_rep = beta * test_rep + (1 - beta) * self.test_history
            p_feat_al = torch.norm((test_rep - p_rep), dim=1)
            g_feat_al = torch.norm((test_rep - g_rep), dim=1)
            cos = torch.nn.CosineSimilarity(dim=1, eps=1e-6)
            sim = cos(F.softmax(g_out).detach(), F.softmax(p_out).detach())
            # loss function based on prediction similarity, entropy minimization and feature alignment.
            loss = (
                -sim * (agg_output.softmax(1) * agg_output.log_softmax(1)).sum(1)
                + (1 - sim)
                * (
                    agg_softmax[:, 0] * g_feat_al.detach()
                    + agg_softmax[:, 1] * p_feat_al.detach()
                )
            ).mean(0)
            state["optimizer"].zero_grad()
            loss.backward()

            if torch.norm(self.agg_weight.grad) < 1e-5:
                break
            grad_norm.append(torch.norm(self.agg_weight.grad).item())
            loss_batch.append(loss.item())
            state["optimizer"].step()

        # if display:
        #     if self.client_id == 1:
        #         print(torch.nn.functional.softmax(self.agg_weight))
        #         print("gradient norm:", grad_norm)
        #         print("loss batch:", loss_batch)

    def _multi_head_inference(
        self, data_batch, model, tracker=None, g_pred=None, p_pred=None
    ):
        # inference procedure for multi-head nets.
        agg_softmax = torch.nn.functional.softmax(self.agg_weight)
        if g_pred is not None and p_pred is not None:
            agg_output = (
                agg_softmax[:, 0].unsqueeze(1) * g_pred
                + agg_softmax[:, 1].unsqueeze(1) * p_pred
            )
        else:
            # do the forward pass and get the output.
            g_out = model(data_batch["input"])
            p_out = self.personal_head(self.rep_layer.rep)
            agg_output = (
                agg_softmax[:, 0].unsqueeze(1) * g_out
                + agg_softmax[:, 1].unsqueeze(1) * p_out
            )
        # agg_output_softmax = F.softmax(agg_output, dim=1)
        agg_output_softmax = agg_output
        # evaluate the output and get the loss, performance.
        loss = self.criterion(agg_output_softmax, data_batch["target"])
        performance = self.metrics.evaluate(
            loss, agg_output_softmax, data_batch["target"]
        )

        # update tracker.
        if tracker is not None:
            tracker.update_metrics(
                [loss.item()] + performance, n_samples=data_batch["input"].size(0)
            )

        # if self.client_id == 1 and self.comm_round in self.check_round_list:
        #     print(data_batch["target"])
        #     print("---------------------------------------------------------")
        return loss

    def _get_dataset(self):
        assert self.conf.data == "cifar10"
        test_loaders = {}
        tr_transform = transforms.Compose(
            [
                # transforms.RandomCrop(32, padding=4),
                # transforms.RandomHorizontalFlip(),
                transforms.Resize(224),
                transforms.ToTensor(),
                transforms.Normalize((0.485, 0.456, 0.406), (0.228, 0.224, 0.225)),
            ]
        )
        te_transform = transforms.Compose(
            [
                transforms.Resize(224),
                transforms.ToTensor(),
                transforms.Normalize((0.485, 0.456, 0.406), (0.228, 0.224, 0.225)),
            ]
        )
        if self.conf.data == "cifar10":
            cifar_data = torchvision.datasets.CIFAR10
        else:
            raise NotImplementedError
        whole_trainset = cifar_data(
            root=os.path.join(self.conf.data_dir, self.conf.data),
            train=True,
            download=True,
            transform=tr_transform,
        )
        # some part of the whole dataset as global train loader
        sample_len = int(0.5 * len(whole_trainset))
        g_tr_set, p_tr_set = torch.utils.data.random_split(
            whole_trainset, [sample_len, len(whole_trainset) - sample_len]
        )
        p_tr_set.targets = [p_tr_set[i][1] for i in range(len(p_tr_set))]
        sample_len = int(0.8 * len(g_tr_set))
        g_tr_set, g_te_set = torch.utils.data.random_split(
            g_tr_set, [sample_len, len(g_tr_set) - sample_len]
        )
        # partition the dataset into 2 non-iid dataset
        partition_sizes = [0.5, 0.5]
        data_partitioner = DataPartitioner(
            p_tr_set,
            partition_sizes,
            partition_type=self.conf.partition_data_conf["distribution"],
            partition_alphas=self.conf.partition_data_conf["non_iid_alpha"],
            consistent_indices=False,
            random_state=self.random_state,
            graph=self.graph,
            logger=self.logger,
        )
        p_tr_set = data_partitioner.use(0)
        sample_len = int(0.8 * len(p_tr_set))
        p_tr_set, p_te_set = torch.utils.data.random_split(
            p_tr_set, [sample_len, len(p_tr_set) - sample_len]
        )
        original_indices = [
            data_partitioner.original_indices[i] for i in p_te_set.indices
        ]

        # introduce various of ood_te_loaders, some are from lp_ft_worker.py
        if self.conf.data == "cifar10":
            # generate natural shift test
            cifar10_1_images = np.load(
                self.conf.data_dir + "/lpft/cifar10.1_" + "v6" + "_data.npy"
            )
            cifar10_1_labels = np.load(
                self.conf.data_dir + "/lpft/cifar10.1_" + "v6" + "_labels.npy"
            )
            ns_te_set = cifar_data(
                root=os.path.join(self.conf.data_dir, self.conf.data),
                train=False,
                download=True,
                transform=te_transform,
            )
            ns_te_set.data = cifar10_1_images
            ns_te_set.targets = cifar10_1_labels
            # partition for natural shift test
            _, hist = record_class_distribution(
                data_partitioner.partitions,
                data_partitioner.data.targets,
                print_fn=print,
            )
            ns_te_set = partition_by_other_histogram(hist, ns_te_set)[0]

            # generate synthetic shift test
            corr_te_set = cifar_data(
                root=os.path.join(self.conf.data_dir, self.conf.data),
                train=True,
                download=True,
                transform=transforms.ToTensor(),
            )
            corr_te_set.data = corr_te_set.data[original_indices]
            corr_te_set.targets = [corr_te_set.targets[i] for i in original_indices]
            corr_te_set = corr_data.define_corr_data(
                corr_te_set, self.conf.corr_seed, severity=5
            )

            corr_te_set1 = cifar_data(
                root=os.path.join(self.conf.data_dir, self.conf.data),
                train=True,
                download=True,
                transform=te_transform,
            )
            corr_te_set1.targets = np.array(
                [corr_te_set[i][1] for i in range(len(corr_te_set))]
            )
            corr_te_set1.data = np.array(
                255
                * torch.stack([corr_te_set[i][0] for i in corr_te_set.indices]).permute(
                    0, 2, 3, 1
                )
            ).astype(np.uint8)

        else:
            ns_te_set, corr_te_set = None, None
            raise NotImplementedError

        # create data loaders
        g_tr_loader = data.DataLoader(
            g_tr_set, batch_size=self.conf.batch_size, shuffle=True, num_workers=1
        )
        p_tr_loader = data.DataLoader(
            p_tr_set, batch_size=self.conf.batch_size, shuffle=True, num_workers=1
        )
        test_loaders["original"] = torch.utils.data.DataLoader(
            g_te_set, batch_size=self.conf.batch_size, shuffle=True, num_workers=1
        )
        test_loaders["local"] = torch.utils.data.DataLoader(
            p_te_set, batch_size=self.conf.batch_size, shuffle=True, num_workers=1
        )
        test_loaders["synthetic_shift"] = torch.utils.data.DataLoader(
            corr_te_set1, batch_size=self.conf.batch_size, shuffle=True, num_workers=1
        )
        test_loaders["natural_shift"] = torch.utils.data.DataLoader(
            ns_te_set, batch_size=self.conf.batch_size, shuffle=True, num_workers=1
        )

        return g_tr_loader, p_tr_loader, test_loaders

    def display_results(self, perf):
        print(perf)
