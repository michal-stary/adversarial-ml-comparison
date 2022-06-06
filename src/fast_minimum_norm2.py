# Adapted from https://github.com/maurapintor/Fast-Minimum-Norm-FMN-Attack

import math
from functools import partial
from typing import Optional

import torch
from torch import nn, Tensor
from torch.autograd import grad

from adv_lib.utils.losses import difference_of_logits
from adv_lib.utils.projections import l1_ball_euclidean_projection


def l0_projection_(δ: Tensor, ε: Tensor) -> Tensor:
    """In-place l0 projection"""
    δ = δ.flatten(1)
    δ_abs = δ.abs()
    sorted_indices = δ_abs.argsort(dim=1, descending=True).gather(1, (ε.long().unsqueeze(1) - 1).clamp_(min=0))
    thresholds = δ_abs.gather(1, sorted_indices)
    δ.mul_(δ_abs >= thresholds)


def l1_projection_(δ: Tensor, ε: Tensor) -> Tensor:
    """In-place l1 projection"""
    l1_ball_euclidean_projection(x=δ.flatten(1), ε=ε, inplace=True)


def l2_projection_(δ: Tensor, ε: Tensor) -> Tensor:
    """In-place l2 projection"""
    δ = δ.flatten(1)
    l2_norms = δ.norm(p=2, dim=1, keepdim=True).clamp_(min=1e-12)
    δ.mul_(ε.unsqueeze(1) / l2_norms).clamp_(max=1)


def linf_projection_(δ: Tensor, ε: Tensor) -> Tensor:
    """In-place linf projection"""
    δ = δ.flatten(1)
    ε = ε.unsqueeze(1)
    torch.maximum(torch.minimum(δ, ε, out=δ), -ε, out=δ)


def l0_mid_points(x0: Tensor, x1: Tensor, ε: Tensor) -> Tensor:
    n_features = x0[0].numel()
    δ = x1 - x0
    l0_projection_(δ=δ, ε=n_features * ε)
    return δ


def l1_mid_points(x0: Tensor, x1: Tensor, ε: Tensor) -> Tensor:
    threshold = (1 - ε).unsqueeze(1)
    δ = (x1 - x0).flatten(1)
    δ_abs = δ.abs()
    mask = δ_abs > threshold
    mid_points = δ_abs.sub_(threshold).copysign_(δ)
    mid_points.mul_(mask)
    return x0 + mid_points.reshape(x0.shape)


def l2_mid_points(x0: Tensor, x1: Tensor, ε: Tensor) -> Tensor:
    ε = ε.unsqueeze(1)
    return x0.flatten(1).mul(1 - ε).add_(ε * x1.flatten(1)).view_as(x0)


def linf_mid_points(x0: Tensor, x1: Tensor, ε: Tensor) -> Tensor:
    ε = ε.unsqueeze(1)
    δ = (x1 - x0).flatten(1)
    return x0 + torch.maximum(torch.minimum(δ, ε, out=δ), -ε, out=δ).view_as(x0)


def fmn2(model: nn.Module,
        inputs: Tensor,
        labels: Tensor,
        norm: float,
        targeted: bool = False,
        steps: int = 10,
        α_init: float = 1.0,
        α_final: Optional[float] = None,
        γ_init: float = 0.05,
        γ_final: float = 0.001,
        starting_points: Optional[Tensor] = None,
        binary_search_steps: int = 10,
        line_search_steps: int = 10,
        top_explore: int = 3) -> Tensor:
    """
    Fast Minimum-Norm attack from https://arxiv.org/abs/2102.12827.

    Parameters
    ----------
    model : nn.Module
        Model to attack.
    inputs : Tensor
        Inputs to attack. Should be in [0, 1].
    labels : Tensor
        Labels corresponding to the inputs if untargeted, else target labels.
    norm : float
        Norm to minimize in {0, 1, 2 ,float('inf')}.
    targeted : bool
        Whether to perform a targeted attack or not.
    steps : int
        Number of optimization steps.
    α_init : float
        Initial step size.
    α_final : float
        Final step size after cosine annealing.
    γ_init : float
        Initial factor by which ε is modified: ε = ε * (1 + or - γ).
    γ_final : float
        Final factor, after cosine annealing, by which ε is modified.
    starting_points : Tensor
        Optional warm-start for the attack.
    binary_search_steps : int
        Number of binary search steps to find the decision boundary between inputs and starting_points.
    line_search_steps : int
        Number of line/binary search steps to find the decision boundary in the direction of the gradient.
    top_explore : int
        Number of classes explored by initial line search.

    Returns
    -------
    adv_inputs : Tensor
        Modified inputs to be adversarial to the model.

    """
    _dual_projection_mid_points = {
        0: (None, l0_projection_, l0_mid_points),
        1: (float('inf'), l1_projection_, l1_mid_points),
        2: (2, l2_projection_, l2_mid_points),
        float('inf'): (1, linf_projection_, linf_mid_points),
    }
    if inputs.min() < 0 or inputs.max() > 1: raise ValueError('Input values should be in the [0, 1] range.')
    device = inputs.device
    batch_size = len(inputs)
    batch_view = lambda tensor: tensor.view(batch_size, *[1] * (inputs.ndim - 1))
    dual, projection, mid_point = _dual_projection_mid_points[norm]
    α_final = α_final or α_init / 100
    multiplier = 1 if targeted else -1


    ## use linesearch to get the starting points

    def boundary_search(model: nn.Module,
                        inputs: Tensor,
                        c_inds: Tensor,
                        direction: Tensor,
                        distance_to_boundary: Tensor,
                        steps: int) -> Tensor:

        # projections not implemented - so suitable only for L2 line search
        assert norm == 2

        smallest_adv = torch.ones_like(inputs, device=device) * torch.inf
        largest_nadv = torch.zeros_like(inputs, device=device)
        ε = torch.ones(batch_size, device=device) * .5

        smallest_adv_distance = torch.ones_like(distance_to_boundary) * torch.inf
        largest_nadv_distance = torch.zeros_like(distance_to_boundary)

        adv_found = torch.ones_like(distance_to_boundary) == 0

        for step in range(steps):
            δ = direction.clone()
            δ *= batch_view(distance_to_boundary)

            # get the points
            points = inputs + δ

            # TODO think through if model goes into other adversarial class
            pred = model(points).argmax(dim=1)
            is_adv = pred == c_inds

            # update points
            is_smaller = distance_to_boundary < smallest_adv_distance
            is_larger  = distance_to_boundary > largest_nadv_distance
            smallest_adv = smallest_adv.where(batch_view(~is_adv | ~is_smaller), δ)
            largest_nadv = largest_nadv.where(batch_view(is_adv | ~is_larger), δ)

            # update distances
            smallest_adv_distance = smallest_adv_distance.where(~is_adv | ~is_smaller,  distance_to_boundary)
            largest_nadv_distance = largest_nadv_distance.where(is_adv | ~is_larger,  distance_to_boundary)
            adv_found |= is_adv

            # line search
            distance_to_boundary[~adv_found] *= 2

            # binary search
            # TODO - correct the epsilons
            # TODO - remove useless computation of midpoints for not adv points
            # distance_to_boundary = distance_to_boundary.where(~adv_found, mid_point(x0=largest_nadv, x1=smallest_adv, ε=ε).flatten(1).norm(p=norm, dim=1))
            distance_to_boundary = distance_to_boundary.where(~adv_found, (smallest_adv_distance + largest_nadv_distance)/2)
        return smallest_adv_distance, smallest_adv

    δ = torch.zeros_like(inputs)
    δ.requires_grad_(True)

    # get initial predictions
    adv_inputs = inputs + δ
    logits = model(adv_inputs)

    # skip correct class
    top_explore = min(top_explore, logits.shape[1]-1)

    # get top k classes indices (except the correct one)
    logits_ = logits.clone()
    logits_[torch.arange(batch_size), labels] = -torch.inf
    class_indcs = torch.argsort(logits_, dim=1, descending=True)[:, :top_explore]

    distance_to_boundaries = []
    smallest_advs = []
    print(class_indcs)
    for c_inds in class_indcs.T:
        print("ah", c_inds)
        # logit diff to the c_inds
        labels_infhot = torch.zeros_like(logits).scatter_(1, c_inds.unsqueeze(1), float('inf'))
        logit_diff_func = partial(difference_of_logits, labels=c_inds, labels_infhot=labels_infhot)
        logit_diffs = logit_diff_func(logits=logits)

        # check if this make sense (maybe should be -1)
        loss = logit_diffs
        print(loss, "l")
        δ_grad = grad(loss.sum(), δ, only_inputs=True, retain_graph=True)[0]

        # initial linear approximation of distance to the boundary
        distance_to_boundary = loss.detach().abs() / δ_grad.flatten(1).norm(p=dual, dim=1).clamp_(min=1e-12)

        # normalize tu unit length
        δ_grad /= batch_view(δ_grad.flatten(1).norm(p=norm, dim=1))

        print(distance_to_boundary, "init")
        # do a boundary search in δ_grad direction for each points
        distance_to_boundary, smallest_adv = boundary_search(model, inputs, c_inds, δ_grad, distance_to_boundary, line_search_steps)

        distance_to_boundaries.append(distance_to_boundary)
        smallest_advs.append(smallest_adv)
        print(distance_to_boundary, "refined")

    distance_to_boundaries = torch.stack(distance_to_boundaries).T
    smallest_advs = torch.swapaxes(torch.stack(smallest_advs), 0, 1)

    best_indc = torch.argmin(distance_to_boundaries, dim=1, keepdim=True)
    best_classes = torch.gather(class_indcs, 1, best_indc)
    best_distances = torch.gather(distance_to_boundaries, 1, best_indc)

    # HACK TO REUSE THE CODE FOR ADV INIT in origFMN
    # add either the points found by search or nothing is search was not succesful in any class
    starting_points = inputs + smallest_advs[torch.arange(batch_size), best_indc.squeeze(), :].where(batch_view(best_distances) != torch.inf, torch.zeros_like(inputs))
    ε = torch.ones(batch_size, device=device)
    binary_search_steps = 0
    # /HACK END

    # see what class has the real closest boundary
    print(best_indc, best_distances, best_classes)


    # If starting_points is provided, search for the boundary
    if starting_points is not None:
        is_adv = model(starting_points).argmax(dim=1)
        #if not is_adv.all():
        #    raise ValueError('Starting points are not all adversarial.')
        lower_bound = torch.zeros(batch_size, device=device)
        upper_bound = torch.ones(batch_size, device=device)
        for _ in range(binary_search_steps):
            ε = (lower_bound + upper_bound) / 2
            mid_points = mid_point(x0=inputs, x1=starting_points, ε=ε)
            pred_labels = model(mid_points).argmax(dim=1)
            is_adv = (pred_labels == labels) if targeted else (pred_labels != labels)
            lower_bound = torch.where(is_adv, lower_bound, ε)
            upper_bound = torch.where(is_adv, ε, upper_bound)

        δ = mid_point(x0=inputs, x1=starting_points, ε=ε) - inputs
    else:
        δ = torch.zeros_like(inputs)
    δ.requires_grad_(True)

    if norm == 0:
        ε = torch.ones(batch_size, device=device) if starting_points is None else δ.flatten(1).norm(p=0, dim=0)
    else:
        ε = torch.full((batch_size,), float('inf'), device=device)

    # Init trackers
    worst_norm = torch.maximum(inputs, 1 - inputs).flatten(1).norm(p=norm, dim=1)
    best_norm = worst_norm.clone()
    best_adv = inputs.clone()
    adv_found = torch.zeros(batch_size, dtype=torch.bool, device=device)


    ## Exploit
    for i in range(steps):
        cosine = (1 + math.cos(math.pi * i / steps)) / 2
        α = α_final + (α_init - α_final) * cosine
        γ = γ_final + (γ_init - γ_final) * cosine

        δ_norm = δ.data.flatten(1).norm(p=norm, dim=1)
        adv_inputs = inputs + δ
        logits = model(adv_inputs)
        pred_labels = logits.argmax(dim=1)

        if i == 0:
            labels_infhot = torch.zeros_like(logits).scatter_(1, labels.unsqueeze(1), float('inf'))
            logit_diff_func = partial(difference_of_logits, labels=labels, labels_infhot=labels_infhot)

        logit_diffs = logit_diff_func(logits=logits)
        loss = multiplier * logit_diffs
        δ_grad = grad(loss.sum(), δ, only_inputs=True)[0]

        is_adv = (pred_labels == labels) if targeted else (pred_labels != labels)
        is_smaller = δ_norm < best_norm
        is_both = is_adv & is_smaller
        adv_found.logical_or_(is_adv)
        best_norm = torch.where(is_both, δ_norm, best_norm)
        best_adv = torch.where(batch_view(is_both), adv_inputs.detach(), best_adv)

        if norm == 0:
            ε = torch.where(is_adv,
                            torch.minimum(torch.minimum(ε - 1, (ε * (1 - γ)).floor_()), best_norm),
                            torch.maximum(ε + 1, (ε * (1 + γ)).floor_()))
            ε.clamp_(min=0)
        else:
            distance_to_boundary = loss.detach().abs() / δ_grad.flatten(1).norm(p=dual, dim=1).clamp_(min=1e-12)
            ε = torch.where(is_adv,
                            torch.minimum(ε * (1 - γ), best_norm),
                            torch.where(adv_found, ε * (1 + γ), δ_norm + distance_to_boundary))

        # clip ε
        ε = torch.minimum(ε, worst_norm)

        # normalize gradient
        grad_l2_norms = δ_grad.flatten(1).norm(p=2, dim=1).clamp_(min=1e-12)
        δ_grad.div_(batch_view(grad_l2_norms))

        # gradient ascent step
        δ.data.add_(δ_grad, alpha=α)

        # project in place
        projection(δ=δ.data, ε=ε)

        # clamp
        δ.data.add_(inputs).clamp_(min=0, max=1).sub_(inputs)

    return best_adv
