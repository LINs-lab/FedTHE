# -*- coding: utf-8 -*-
import torch
import copy
from tqdm import tqdm
import functools
import pcode.utils.loss as loss
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
import pcode.create_optimizer as create_optimizer
import torchvision.transforms as transforms
import torchvision
import pickle

"""
Test for linear probing and fine-tuning.
"""


class LPFTWorker(BaseWorker):
    def __init__(self, conf):
        super(LPFTWorker, self).__init__(conf)
        self.conf = conf
        self.n_personalized_epochs = conf.n_personalized_epochs
        self.is_in_childworker = True

        self.pretrained_through = conf.pretrained_through  # "mocov2" or "imagenet"
        self.mode = conf.mode  # "fine_tuning" or "lp_ft" or "linear_probing"
        self.tr_setting = conf.tr_setting  # "class" or "original" or "sample"

        # TODO: move these parameters to example.py
        # TODO: check parameter setting

        if self.tr_setting == "sample":
            self.sample_ratio = conf.sample_ratio
        elif self.tr_setting == "class":
            self.exclude_list = [8, 9]
        elif self.tr_setting == "original":
            pass

        self.corruption_type = "gaussian_noise"
        self.corruption_severity = 3
        self.lp_lr = [10, 20, 30]
        self.ft_lr = [0.001, 0.005, 0.01]
        self.lp_ft_lr = [0.0001, 0.0005, 0.002]

        if self.mode == "linear_probing":
            self.conf.personal_lr = self.lp_lr[self.conf.personal_lr_setting - 1]
        elif self.mode == "fine_tuning":
            self.conf.personal_lr = self.ft_lr[self.conf.personal_lr_setting - 1]
        elif self.mode == "lp_ft":
            # initialize personal lr as lp does.
            self.conf.personal_lr = self.lp_lr[self.conf.personal_lr_setting - 1]
            self.n_personalized_epochs = int(conf.n_personalized_epochs / 2)

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

            # check if we need to terminate the training or not.
            if self._terminate_by_early_stopping():
                return

            # receive the perform standard local training.
            self._recv_model_from_master()
            params_to_send = copy.deepcopy(self.model.state_dict())  # use deepcopy

            if not hasattr(self, "pretrained_model"):
                self.pretrained_model = torchvision.models.resnet50(
                    pretrained=False, num_classes=10
                )
                if self.pretrained_through == "imagenet":
                    pretrained_resnet50 = torchvision.models.resnet50(
                        pretrained=True, progress=True
                    )
                    state_dict = pretrained_resnet50.state_dict()
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
                # in case of reuse, create a deepcopy
                self.pretrained_copy = copy.deepcopy(self.pretrained_model)

            # prepare datasets
            (
                trainset,
                tr_loader,
                id_te_loader,
                cifar10_1_te_loader,
                cifarc_te_loader,
                stl10_te_loader,
            ) = self._prepare_all_dataset(tr_setting=self.tr_setting)
            # operations depending on the current mode
            if self.mode == "linear_probing" or self.mode == "lp_ft":
                for name, param in self.pretrained_model.named_parameters():
                    if name not in ["fc.weight", "fc.bias"]:
                        param.requires_grad = False
                # init the fc layer
                self.pretrained_model.fc.weight.data.normal_(mean=0.0, std=0.01)
                self.pretrained_model.fc.bias.data.zero_()
            elif self.mode == "fine_tuning":
                self.pretrained_model.requires_grad_(True)
            else:
                raise NotImplementedError

            # sanity check
            for i, param in enumerate(self.pretrained_model.named_parameters()):
                print(param[0], param[1].requires_grad)
            # training
            state = self._personalized_train(
                model=self.pretrained_model, tr_loader=tr_loader
            )
            # in case of lp_ft, do a fine-tuning session
            if self.mode == "lp_ft":
                # change the tr_transform and reload the tr_loader
                tr_transform = transforms.Compose(
                    [
                        # transforms.RandomCrop(32, padding=4),
                        transforms.Resize(224),
                        transforms.RandomHorizontalFlip(),
                        transforms.ToTensor(),
                        transforms.Normalize(
                            (0.485, 0.456, 0.406), (0.228, 0.224, 0.225)
                        ),
                    ]
                )
                trainset.transform = tr_transform
                tr_loader = data.DataLoader(
                    trainset, batch_size=128, shuffle=True, num_workers=1
                )
                self.conf.personal_lr = self.lp_ft_lr[self.conf.personal_lr_setting - 1]
                self.pretrained_model.requires_grad_(True)
                self.mode = (
                    "fine_tuning"  # to set model.train() in _personalized_train()
                )
                state = self._personalized_train(
                    model=state["model"], tr_loader=tr_loader
                )
                self.mode = "lp_ft"
            # evaluate on id_te_loader and ood_te_loader
            id_perf = self._evaluate_model(state["model"], dataset=id_te_loader)
            cifar10_1_perf = self._evaluate_model(
                state["model"], dataset=cifar10_1_te_loader
            )
            cifarc_perf = self._evaluate_model(state["model"], dataset=cifarc_te_loader)
            stl10_perf = self._evaluate_model(state["model"], dataset=stl10_te_loader)

            self._display_args()
            self._record_results(id_perf, cifar10_1_perf, cifarc_perf, stl10_perf)
            # default setting
            self._send_model_to_master(params_to_send, [])

            if self._terminate_by_complete_training():
                return

    def _evaluate_model(self, model, dataset):
        # define dataloader, optimizer, scheduler and tracker
        model.to(self.graph.device)
        model.eval()
        state = {"model": model}
        test_loss, total, correct = 0.0, 0, 0
        for _input, _target in tqdm(dataset):
            # inference and get current performance.
            if self.conf.graph.on_cuda:
                _input, _target = _input.cuda(), _target.to(dtype=torch.long).cuda()
            data_batch = {"input": _input, "target": _target}
            with torch.no_grad():
                outputs = state["model"](data_batch["input"])
                loss = self.erm_criterion(outputs, data_batch["target"])
                test_loss += loss.item()
                total += _target.size(0)
                _, predicted = outputs.max(1)
                correct += predicted.eq(_target).sum().item()

        print("current test loss:", test_loss)
        print("current test acc:", 100.0 * correct / total)

        return 100.0 * correct / total

    def _prepare_all_dataset(self, tr_setting):
        if self.mode == "linear_probing" or self.mode == "lp_ft":
            tr_transform = transforms.Compose(
                [
                    # transforms.RandomCrop(32, padding=4),
                    # transforms.RandomHorizontalFlip(),
                    transforms.Resize(224),
                    transforms.ToTensor(),
                    transforms.Normalize((0.485, 0.456, 0.406), (0.228, 0.224, 0.225)),
                ]
            )
        else:
            tr_transform = transforms.Compose(
                [
                    # transforms.RandomCrop(32, padding=4),
                    transforms.Resize(224),
                    transforms.RandomHorizontalFlip(),
                    transforms.ToTensor(),
                    transforms.Normalize((0.485, 0.456, 0.406), (0.228, 0.224, 0.225)),
                ]
            )
        transform = transforms.Compose(
            [
                transforms.Resize(224),
                transforms.ToTensor(),
                transforms.Normalize((0.485, 0.456, 0.406), (0.228, 0.224, 0.225)),
            ]
        )
        # prepare original or modified cifar
        trainset = torchvision.datasets.CIFAR10(
            root=self.conf.data_dir + "/lpft/cifar10/",
            train=True,
            download=True,
            transform=tr_transform,
        )
        if tr_setting == "sample":
            sample_len = int(self.sample_ratio * len(trainset))
            trainset, _ = torch.utils.data.random_split(
                trainset, [sample_len, len(trainset) - sample_len]
            )
        elif tr_setting == "class":
            mask = ~(
                np.array(trainset.targets).reshape(-1, 1) == self.exclude_list
            ).any(axis=1)
            trainset.data = trainset.data[mask]
            trainset.targets = np.array(trainset.targets)[mask].tolist()

        tr_loader = data.DataLoader(
            trainset, batch_size=128, shuffle=True, num_workers=1
        )
        # prepare original cifar test set (ID)
        testset = torchvision.datasets.CIFAR10(
            root=self.conf.data_dir + "/lpft/cifar10/",
            train=False,
            download=True,
            transform=transform,
        )
        id_te_loader = torch.utils.data.DataLoader(
            testset, batch_size=100, shuffle=False, num_workers=2
        )
        # prepare distribution shifted cifar test set(OOD)
        ood_images = np.load(
            self.conf.data_dir + "/lpft/cifar10.1_" + "v6" + "_data.npy"
        )
        ood_labels = np.load(
            self.conf.data_dir + "/lpft/cifar10.1_" + "v6" + "_labels.npy"
        )
        ood_te_set = torchvision.datasets.CIFAR10(
            root=self.conf.data_dir + "/lpft/cifar10/",
            train=False,
            download=True,
            transform=transform,
        )
        ood_te_set.data = ood_images
        ood_te_set.targets = ood_labels
        cifar10_1_te_loader = torch.utils.data.DataLoader(
            ood_te_set, batch_size=100, shuffle=False, num_workers=1
        )

        cifar10c_path = self.conf.data_dir + "/lpft/cifar10c/CIFAR-10-C/"
        ood_images = np.load(cifar10c_path + self.corruption_type + ".npy")
        ood_images = ood_images[
            (self.corruption_severity - 1) * 10000 : self.corruption_severity * 10000
        ]
        ood_te_set = torchvision.datasets.CIFAR10(
            root=self.conf.data_dir + "/lpft/cifar10/",
            train=False,
            download=True,
            transform=transform,
        )
        ood_te_set.data = ood_images
        cifar10c_te_loader = torch.utils.data.DataLoader(
            ood_te_set, batch_size=100, shuffle=False, num_workers=1
        )

        ood_te_set = torchvision.datasets.STL10(
            root=self.conf.data_dir + "/lpft/stl10/",
            split="test",
            download=True,
            transform=transform,
        )
        # remove the class monkey because cifar10 don't have monkey class
        monkey_cls = 7
        target_map = [0, 2, 1, 3, 4, 5, 7, 6, 8, 9]
        stl10_mask = ~(np.array(ood_te_set.labels).reshape(-1, 1) == monkey_cls).any(
            axis=1
        )
        ood_te_set.data = ood_te_set.data[stl10_mask]
        ood_te_set.labels = np.array(ood_te_set.labels)[stl10_mask].tolist()
        ood_te_set.labels = [target_map[label] for label in ood_te_set.labels]
        stl10_te_loader = torch.utils.data.DataLoader(
            ood_te_set, batch_size=100, shuffle=False, num_workers=1
        )

        # return all three
        return (
            trainset,
            tr_loader,
            id_te_loader,
            cifar10_1_te_loader,
            cifar10c_te_loader,
            stl10_te_loader,
        )

    def _personalized_train(self, model, tr_loader):
        self.is_in_personalized_training = True
        self.n_local_epochs = self.n_personalized_epochs
        # define dataloader, optimizer, scheduler and tracker
        model.to(self.graph.device)
        if self.mode == "linear_probing" or self.mode == "lp_ft":
            model.eval()
        else:
            model.train()
        self.lr = self.conf.personal_lr
        self.erm_criterion = nn.CrossEntropyLoss()
        parameters = list(filter(lambda p: p.requires_grad, model.parameters()))
        optimizer = torch.optim.SGD(
            parameters, lr=self.lr, momentum=0.9, weight_decay=0.0
        )
        state = {"model": model, "optimizer": optimizer, "train_loader": tr_loader}
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
            optimizer, self.n_local_epochs
        )
        for _ in tqdm(range(self.n_local_epochs)):
            train_loss, total, correct = 0.0, 0, 0
            # scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, len(state["train_loader"]))
            for _input, _target in state["train_loader"]:
                # load data
                if self.conf.graph.on_cuda:
                    _input, _target = _input.cuda(), _target.to(dtype=torch.long).cuda()
                data_batch = {"input": _input, "target": _target}

                state["optimizer"].zero_grad()
                outputs = state["model"](data_batch["input"])
                loss = self.erm_criterion(outputs, data_batch["target"])

                loss.backward()
                state["optimizer"].step()
                # scheduler.step()
                # print(optimizer.param_groups[0]["lr"])
                train_loss += loss.item()
                total += _target.size(0)
                _, predicted = outputs.max(1)
                correct += predicted.eq(_target).sum().item()
            scheduler.step()
            # print(optimizer.param_groups[0]["lr"])
            print("current training loss:", train_loss)
            print("current training acc:", 100.0 * correct / total)

        # terminate
        self.is_in_personalized_training = False
        return state

    def _display_args(self):
        print(self.mode)
        print(self.pretrained_through)
        print(self.tr_setting)
        print(self.n_personalized_epochs)
        print(self.lr)

    def _record_results(self, id_perf, cifar10_1_perf, cifarc_perf, stl10_perf):
        path = "/data1/user/results/section3/results.json"
        arg_and_perf = {
            "mode": self.mode,
            "pretrained_through": self.pretrained_through,
            "training_setting": self.tr_setting,
            "n_personalized_epoch": self.n_personalized_epochs,
            "lr": self.lr,
            "id_perf": id_perf,
            "cifar10_1_perf": cifar10_1_perf,
            "cifarc_perf": cifarc_perf,
            "stl10_perf": stl10_perf,
        }
        # save results
        with open(path, "ab+") as f:
            pickle.dump(arg_and_perf, f)
            f.close()
