# -*- coding: utf-8 -*-
import functools
import numpy as np

import torch
import torch.nn as nn
import torch.nn.functional as F
from pcode.utils.misc import onehot


"""standard loss with label smoothing.
borrowed from https://github.com/eladhoffer/utils.pytorch/blob/master/cross_entropy.py
"""


def _is_long(x):
    if hasattr(x, "data"):
        x = x.data
    return isinstance(x, torch.LongTensor) or isinstance(x, torch.cuda.LongTensor)


def cross_entropy(
    inputs,
    target,
    weight=None,
    ignore_index=-100,
    reduction="mean",
    smooth_eps=None,
    smooth_dist=None,
    from_logits=True,
):
    """cross entropy loss, with support for target distributions and label smoothing https://arxiv.org/abs/1512.00567"""
    smooth_eps = smooth_eps or 0

    # ordinary log-liklihood - use cross_entropy from nn
    if _is_long(target) and smooth_eps == 0:
        if from_logits:
            return F.cross_entropy(
                inputs, target, weight, ignore_index=ignore_index, reduction=reduction
            )
        else:
            return F.nll_loss(
                inputs, target, weight, ignore_index=ignore_index, reduction=reduction
            )

    if from_logits:
        # log-softmax of inputs
        lsm = F.log_softmax(inputs, dim=-1)
    else:
        lsm = inputs

    masked_indices = None
    num_classes = inputs.size(-1)

    if _is_long(target) and ignore_index >= 0:
        masked_indices = target.eq(ignore_index)

    if smooth_eps > 0 and smooth_dist is not None:
        if _is_long(target):
            target = onehot(target, num_classes).type_as(inputs)
        if smooth_dist.dim() < target.dim():
            smooth_dist = smooth_dist.unsqueeze(0)
        target.lerp_(smooth_dist, smooth_eps)

    if weight is not None:
        lsm = lsm * weight.unsqueeze(0)

    if _is_long(target):
        eps_sum = smooth_eps / num_classes
        eps_nll = 1.0 - eps_sum - smooth_eps
        likelihood = lsm.gather(dim=-1, index=target.unsqueeze(-1)).squeeze(-1)
        loss = -(eps_nll * likelihood + eps_sum * lsm.sum(-1))
    else:
        loss = -(target * lsm).sum(-1)

    if masked_indices is not None:
        loss.masked_fill_(masked_indices, 0)

    if reduction == "sum":
        loss = loss.sum()
    elif reduction == "mean":
        if masked_indices is None:
            loss = loss.mean()
        else:
            loss = loss.sum() / float(loss.size(0) - masked_indices.sum())

    return loss


class CrossEntropyLoss(nn.CrossEntropyLoss):
    """CrossEntropyLoss - with ability to recieve distrbution as targets, and optional label smoothing"""

    def __init__(
        self,
        weight=None,
        ignore_index=-100,
        reduction="mean",
        smooth_eps=None,
        smooth_dist=None,
        from_logits=True,
    ):
        super(CrossEntropyLoss, self).__init__(
            weight=weight, ignore_index=ignore_index, reduction=reduction
        )
        self.smooth_eps = smooth_eps
        self.smooth_dist = smooth_dist
        self.from_logits = from_logits

    def forward(self, input, target, smooth_eps=None, smooth_dist=None):
        if smooth_dist is None:
            smooth_dist = self.smooth_dist
        if smooth_eps is None:
            smooth_eps = self.smooth_eps
        return cross_entropy(
            input,
            target,
            weight=self.weight,
            ignore_index=self.ignore_index,
            reduction=self.reduction,
            smooth_eps=smooth_eps,
            smooth_dist=smooth_dist,
            from_logits=self.from_logits,
        )


def binary_cross_entropy(
    inputs, target, weight=None, reduction="mean", smooth_eps=None, from_logits=False
):
    """cross entropy loss, with support for label smoothing https://arxiv.org/abs/1512.00567"""
    smooth_eps = smooth_eps or 0
    if smooth_eps > 0:
        target = target.float()
        target.add_(smooth_eps).div_(2.0)
    if from_logits:
        return F.binary_cross_entropy_with_logits(
            inputs, target, weight=weight, reduction=reduction
        )
    else:
        return F.binary_cross_entropy(
            inputs, target, weight=weight, reduction=reduction
        )


def binary_cross_entropy_with_logits(
    inputs, target, weight=None, reduction="mean", smooth_eps=None, from_logits=True
):
    return binary_cross_entropy(
        inputs, target, weight, reduction, smooth_eps, from_logits
    )


class BCELoss(nn.BCELoss):
    def __init__(
        self,
        weight=None,
        size_average=None,
        reduce=None,
        reduction="mean",
        smooth_eps=None,
        from_logits=False,
    ):
        super(BCELoss, self).__init__(weight, size_average, reduce, reduction)
        self.smooth_eps = smooth_eps
        self.from_logits = from_logits

    def forward(self, input, target):
        return binary_cross_entropy(
            input,
            target,
            weight=self.weight,
            reduction=self.reduction,
            smooth_eps=self.smooth_eps,
            from_logits=self.from_logits,
        )


class BCEWithLogitsLoss(BCELoss):
    def __init__(
        self,
        weight=None,
        size_average=None,
        reduce=None,
        reduction="mean",
        smooth_eps=None,
        from_logits=True,
    ):
        super(BCEWithLogitsLoss, self).__init__(
            weight,
            size_average,
            reduce,
            reduction,
            smooth_eps=smooth_eps,
            from_logits=from_logits,
        )


"""reweighted loss for inbalanced classes."""


def focal_loss(labels, logits, alpha, gamma):
    """Compute the focal loss between `logits` and the ground truth `labels`.
    Focal loss = -alpha_t * (1-pt)^gamma * log(pt)
    where pt is the probability of being classified to the true class.
    pt = p (if true class), otherwise pt = 1 - p. p = sigmoid(logit).
    Args:
      labels: A float tensor of size [batch, num_classes].
      logits: A float tensor of size [batch, num_classes].
      alpha: A float tensor of size [batch_size]
        specifying per-example weight for balanced cross entropy.
      gamma: A float scalar modulating loss from hard and easy examples.
    Returns:
      focal_loss: A float32 scalar representing normalized total loss.
    """
    BCLoss = F.binary_cross_entropy_with_logits(
        input=logits, target=labels, reduction="none"
    )

    if gamma == 0.0:
        modulator = 1.0
    else:
        modulator = torch.exp(
            -gamma * labels * logits - gamma * torch.log(1 + torch.exp(-1.0 * logits))
        )

    loss = modulator * BCLoss

    weighted_loss = alpha * loss
    focal_loss = torch.sum(weighted_loss)

    focal_loss /= torch.sum(labels)
    return focal_loss


def CB_loss(
    logits, labels, samples_per_cls, no_of_classes, loss_type, beta, gamma, use_cuda
):
    """Compute the Class Balanced Loss between `logits` and the ground truth `labels`.
    Class Balanced Loss: ((1-beta)/(1-beta^n))*Loss(labels, logits)
    where Loss is one of the standard losses used for Neural Networks.
    Args:
      labels: A int tensor of size [batch].
      logits: A float tensor of size [batch, no_of_classes].
      samples_per_cls: A python list of size [no_of_classes].
      no_of_classes: total number of classes. int
      loss_type: string. One of "sigmoid", "focal", "softmax".
      beta: float. Hyperparameter for Class balanced loss.
      gamma: float. Hyperparameter for Focal loss.
    Returns:
      cb_loss: A float tensor representing class balanced loss
    """
    effective_num = 1.0 - np.power(beta, samples_per_cls)
    weights = (1.0 - beta) / np.array(effective_num)
    weights = weights / np.sum(weights) * no_of_classes

    labels_one_hot = F.one_hot(labels, no_of_classes).float()

    weights = torch.tensor(weights).float()
    weights = weights.cuda() if use_cuda else weights
    weights = weights.unsqueeze(0)
    weights = weights.repeat(labels_one_hot.shape[0], 1) * labels_one_hot
    weights = weights.sum(1)
    weights = weights.unsqueeze(1)
    weights = weights.repeat(1, no_of_classes)

    if loss_type == "focal":
        cb_loss = focal_loss(labels_one_hot, logits, weights, gamma)
    elif loss_type == "sigmoid":
        cb_loss = F.binary_cross_entropy_with_logits(
            input=logits, target=labels_one_hot, weights=weights
        )
    elif loss_type == "softmax":
        pred = logits.softmax(dim=1)
        cb_loss = F.binary_cross_entropy(
            input=pred, target=labels_one_hot, weight=weights
        )
    else:
        raise NotImplementedError
    return cb_loss


def get_weighted_loss_criterion(dataset, target_info, loss_type, beta, gamma, use_cuda):
    no_of_classes = len(target_info)
    samples_per_cls = [x[1] for x in target_info]
    fn = functools.partial(
        CB_loss,
        samples_per_cls=samples_per_cls,
        no_of_classes=no_of_classes,
        loss_type=loss_type,
        beta=beta,
        gamma=gamma,
        use_cuda=use_cuda,
    )
    return fn


class BalancedSoftmaxLoss(nn.Module):
    def __init__(self, histogram):
        super().__init__()
        self.histogram = (
            histogram
        )  # need to input number of instances for each class here.
        # self.histogram = torch.Tensor([h if h >= 1 else 1e-4 for h in self.histogram])
        self.histogram = histogram + 1  # smooth in case we have zero in histogram

    def forward(self, x, y, reduction="mean"):
        # x is minibatch * classes
        # hist = self.histogram.type_as(y)
        hist = self.histogram.cuda()
        x = x + hist.unsqueeze(0).log().expand(x.shape[0], -1)
        return F.cross_entropy(input=x, target=y, reduction=reduction)


class WeightedCrossEntropy(nn.Module):
    def __init__(self, histogram):
        super().__init__()
        self.histogram = 1 / torch.sqrt(histogram + 1)

    def forward(self, x, y):
        # x is minibatch * classes
        # hist = self.histogram.type_as(y)
        hist = torch.Tensor([self.histogram[target] for target in y])
        hist = hist / torch.sum(hist)
        hist = hist.cuda()
        losses = F.cross_entropy(input=x, target=y, reduction="none")
        loss = torch.sum(hist * losses)
        return loss


"""
Implementation of 
    "Focal Loss for Dense Object Detection": https://arxiv.org/abs/1708.02002
    "Learning Imbalanced Datasets with Label-Distribution-Aware Margin Loss": https://arxiv.org/abs/1906.07413
The scripts are referred to https://github.com/kaidic/LDAM-DRW/blob/master/losses.py
"""


def _focal_loss(input_values, gamma):
    """Computes the focal loss"""
    p = torch.exp(-input_values)
    loss = (1 - p) ** gamma * input_values
    return loss.mean()


class FocalLoss(nn.Module):
    def __init__(self, weight=None, gamma=0.0):
        super(FocalLoss, self).__init__()
        assert gamma >= 0
        self.gamma = gamma
        self.weight = weight

    def forward(self, input, target):
        return _focal_loss(
            F.cross_entropy(input, target, reduction="none", weight=self.weight),
            self.gamma,
        )


class LDAMLoss(nn.Module):
    def __init__(self, cls_num_list, max_m=0.5, weight=None, s=30):
        # s might be multiplied to adjust effect normalization causes.
        # https://github.com/frank-xwang/RIDE-LongTailRecognition/issues/4
        # https://github.com/kaidic/LDAM-DRW/issues/13
        super(LDAMLoss, self).__init__()
        # use a smooth term to avoid inf in m_list.
        smooth_term = 1
        m_list = 1.0 / np.sqrt(np.sqrt(cls_num_list + smooth_term))
        m_list = (m_list * (max_m / torch.max(m_list))).cuda()
        m_list = torch.cuda.FloatTensor(m_list)
        self.m_list = m_list
        assert s > 0
        self.s = s
        self.weight = weight

    def forward(self, x, target):
        index = torch.zeros_like(x, dtype=torch.uint8)
        index.scatter_(1, target.data.view(-1, 1), 1)

        index_float = index.type(torch.cuda.FloatTensor)
        batch_m = torch.matmul(self.m_list[None, :], index_float.transpose(0, 1))
        batch_m = batch_m.view((-1, 1))
        x_m = x - batch_m

        output = torch.where(index, x_m, x)
        return F.cross_entropy(self.s * output, target, weight=self.weight)


class SupConLoss(nn.Module):
    """Supervised Contrastive Learning: https://arxiv.org/pdf/2004.11362.pdf.
    It also supports the unsupervised contrastive loss in SimCLR"""
    def __init__(self, temperature=0.07, contrast_mode='all',
                 base_temperature=0.07):
        super(SupConLoss, self).__init__()
        self.temperature = temperature
        self.contrast_mode = contrast_mode
        self.base_temperature = base_temperature

    def forward(self, features, labels=None, mask=None):
        """Compute loss for model. If both `labels` and `mask` are None,
        it degenerates to SimCLR unsupervised loss:
        https://arxiv.org/pdf/2002.05709.pdf
        Args:
            features: hidden vector of shape [bsz, n_views, ...].
            labels: ground truth of shape [bsz].
            mask: contrastive mask of shape [bsz, bsz], mask_{i,j}=1 if sample j
                has the same class as sample i. Can be asymmetric.
        Returns:
            A loss scalar.
        """
        device = (torch.device('cuda')
                  if features.is_cuda
                  else torch.device('cpu'))

        if len(features.shape) < 3:
            raise ValueError('`features` needs to be [bsz, n_views, ...],'
                             'at least 3 dimensions are required')
        if len(features.shape) > 3:
            features = features.view(features.shape[0], features.shape[1], -1)

        batch_size = features.shape[0]
        if labels is not None and mask is not None:
            raise ValueError('Cannot define both `labels` and `mask`')
        elif labels is None and mask is None:
            mask = torch.eye(batch_size, dtype=torch.float32).to(device)
        elif labels is not None:
            labels = labels.contiguous().view(-1, 1)
            if labels.shape[0] != batch_size:
                raise ValueError('Num of labels does not match num of features')
            mask = torch.eq(labels, labels.T).float().to(device)
        else:
            mask = mask.float().to(device)

        contrast_count = features.shape[1]
        contrast_feature = torch.cat(torch.unbind(features, dim=1), dim=0)
        if self.contrast_mode == 'one':
            anchor_feature = features[:, 0]
            anchor_count = 1
        elif self.contrast_mode == 'all':
            anchor_feature = contrast_feature
            anchor_count = contrast_count
        else:
            raise ValueError('Unknown mode: {}'.format(self.contrast_mode))

        # compute logits
        anchor_dot_contrast = torch.div(
            torch.matmul(anchor_feature, contrast_feature.T),
            self.temperature)
        # for numerical stability
        logits_max, _ = torch.max(anchor_dot_contrast, dim=1, keepdim=True)
        logits = anchor_dot_contrast - logits_max.detach()

        # tile mask
        mask = mask.repeat(anchor_count, contrast_count)
        # mask-out self-contrast cases
        logits_mask = torch.scatter(
            torch.ones_like(mask),
            1,
            torch.arange(batch_size * anchor_count).view(-1, 1).to(device),
            0
        )
        mask = mask * logits_mask

        # compute log_prob
        exp_logits = torch.exp(logits) * logits_mask
        log_prob = logits - torch.log(exp_logits.sum(1, keepdim=True))

        # compute mean of log-likelihood over positive
        mean_log_prob_pos = (mask * log_prob).sum(1) / mask.sum(1)

        # loss
        loss = - (self.temperature / self.base_temperature) * mean_log_prob_pos
        loss = loss.view(anchor_count, batch_size).mean()

        return loss
