# -*- coding: utf-8 -*-
import functools

import torch

import pcode.create_dataset as create_dataset
import pcode.utils.checkpoint as checkpoint
from pcode.utils.stat_tracker import RuntimeTracker
from pcode.utils.logging import display_general_stat, display_best_test_stat
from pcode.utils.mathdict import MathDict
import math


def inference(model, criterion, metrics, data_batch, tracker=None, is_training=True):
    """Inference on the given model and get loss and accuracy."""
    # do the forward pass and get the output.
    output = model(data_batch["input"])

    # evaluate the output and get the loss, performance.
    loss = criterion(output, data_batch["target"])
    performance = metrics.evaluate(loss, output, data_batch["target"])

    # update tracker.
    if tracker is not None:
        tracker.update_metrics(
            [loss.item()] + performance, n_samples=data_batch["input"].size(0)
        )
    return loss, output


def do_validation(
    conf,
    coordinator,
    model,
    criterion,
    metrics,
    data_loaders,
    split,
    label,
    comm_round,
    performance=None,
):
    """Evaluate the model on the test dataset and save to the checkpoint."""
    # wait until the whole group enters this function, and then evaluate.
    conf.logger.log("Master enters the validation phase.")
    if performance is None:
        performance = get_avg_perf_on_dataloaders(
            conf,
            model=model,
            criterion=criterion,
            metrics=metrics,
            data_loaders=data_loaders,
            split=split,
            label=label,
            comm_round=comm_round,
        )

    # remember best performance and display the val info.
    coordinator.update_perf(performance, comm_round)
    display_best_test_stat(conf, coordinator, comm_round)

    # save to the checkpoint.
    if label == "global_model":
        # add this because personalized model logging hasn't been implemented
        conf.logger.log("Master finished the validation.")
        if not conf.train_fast:
            checkpoint.save_to_checkpoint(
                conf,
                {
                    "arch": conf.arch,
                    "current_comm_round": comm_round,
                    "best_perf": coordinator.best_trackers["top1"].best_perf,
                    "state_dict": model.state_dict(),  # models here would be a single model
                },
                coordinator.best_trackers["top1"].is_best,
                dirname=conf.checkpoint_root,
                filename="checkpoint.pth.tar",
                save_all=conf.save_all_models,
            )
            conf.logger.log("Master saved to checkpoint.")

    return performance


def get_avg_perf_on_dataloaders(
    conf, model, criterion, metrics, data_loaders, split, label, comm_round
):
    print(f"\tGet averaged performance from {len(data_loaders)} data_loaders.")
    performance = []
    for client_id, data_loader in data_loaders.items():
        _performance = validate(
            conf,
            model,
            criterion,
            metrics,
            data_loader,
            split,
            label=f"{label}-{client_id}",
            comm_round=comm_round,
            display=True,
        )
        performance.append(MathDict(_performance))
    performance = functools.reduce(lambda a, b: a + b, performance) / len(performance)
    return performance


def validate(
    conf, model, criterion, metrics, data_loader, split, label, comm_round, display=True
):
    """A function for model evaluation."""
    if data_loader is None:
        return None

    # switch to evaluation mode.
    model.eval()

    # place the model to the device.
    if conf.graph.on_cuda:
        model = model.cuda()

    # determine the number of batches
    num_batches = math.ceil(len(data_loader) / conf.n_participated)
    num = 1

    # evaluate on test_loader.
    tracker_te = RuntimeTracker(metrics_to_track=metrics.metric_names)

    for _input, _target in data_loader:
        # load data and check performance.
        data_batch = create_dataset.load_data_batch(
            conf, _input, _target, is_training=False
        )

        with torch.no_grad():
            inference(
                model, criterion, metrics, data_batch, tracker_te, is_training=False
            )

        if num >= num_batches:
            break
        else:
            num += 1

    # place back model to the cpu.
    if conf.graph.on_cuda:
        model = model.cpu()

    if display:
        # display_general_stat(conf, tracker_te, split, label)
        conf.logger.log(
            f"The {split} performance @ round {comm_round} = {tracker_te()}."
        )
    return tracker_te()


def euclidean_proj_simplex(v, s=1):
    """ Compute the Euclidean projection on a positive simplex
    Solves the optimisation problem (using the algorithm from [1]):
        min_w 0.5 * || w - v ||_2^2 , s.t. \sum_i w_i = s, w_i >= 0 
    Parameters
    ----------
    v: (n,) numpy array,
       n-dimensional vector to project
    s: int, optional, default: 1,
       radius of the simplex
    Returns
    -------
    w: (n,) numpy array,
       Euclidean projection of v on the simplex
    Notes
    -----
    The complexity of this algorithm is in O(n log(n)) as it involves sorting v.
    Better alternatives exist for high-dimensional sparse vectors (cf. [1])
    However, this implementation still easily scales to millions of dimensions.
    References
    ----------
    [1] Efficient Projections onto the .1-Ball for Learning in High Dimensions
        John Duchi, Shai Shalev-Shwartz, Yoram Singer, and Tushar Chandra.
        International Conference on Machine Learning (ICML 2008)
        http://www.cs.berkeley.edu/~jduchi/projects/DuchiSiShCh08.pdf
    """
    assert s > 0, "Radius s must be strictly positive (%d <= 0)" % s
    n, = v.shape  # will raise ValueError if v is not 1-D
    # check if we are already on the simplex
    if v.sum() == s and (v >= 0).all():
        # best projection: itself!
        return v
    # get the array of cumulative sums of a sorted (decreasing) copy of v
    u = torch.flip(torch.sort(v)[0],dims=(0,))
    cssv = torch.cumsum(u,dim=0)
    # get the number of > 0 components of the optimal solution
    non_zero_vector = torch.nonzero(u * torch.arange(1, n+1) > (cssv - s), as_tuple=False)
    if len(non_zero_vector) == 0:
        rho=0.0
    else:
        rho = non_zero_vector[-1].squeeze()
    # compute the Lagrange multiplier associated to the simplex constraint
    theta = (cssv[rho] - s) / (rho + 1.0)
    # compute the projection by thresholding v using theta
    w = (v - theta).clamp(min=0)
    return w
