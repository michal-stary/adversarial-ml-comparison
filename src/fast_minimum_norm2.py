# Adapted from https://github.com/maurapintor/Fast-Minimum-Norm-FMN-Attack

import math
from functools import partial
from typing import Optional

import torch
from torch import nn, Tensor
from torch.autograd import grad

from adv_lib.utils.losses import difference_of_logits
from adv_lib.utils.projections import l1_ball_euclidean_projection
from adv_lib.attacks.auto_pgd import l1_projection as croce_l1_projection
from adv_lib.attacks import fmn

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

def l1_mid_points_(x0: Tensor, x1: Tensor, ε: Tensor) -> Tensor:
    δ = (x1 - x0).flatten(1)
    
    # move from 0-1 eps range to 0-||delta||_1 range 
    eps =ε* δ.norm(p=1, dim=1)
#     print("initial eps", ε)
    
    δ_abs = δ.abs()
    argsorted = torch.argsort(δ_abs, dim=1, descending=True)
    delta = torch.zeros_like(δ)
    
    
#     print(δ)
    # try all coordinates in the worst case, but no more
    for i in range(len(δ[0])):
        free = (eps - delta.abs().sum(dim=1)).clamp(min=0)
        
        if torch.isclose(free, torch.zeros_like(free)).all():
            break
            
#         print(delta, free)
        vector_of_x0vals_for_ith_largest_grads = x0.flatten(1)[torch.arange(len(ε)), argsorted[:, i]]
        limit_up = 1 - vector_of_x0vals_for_ith_largest_grads
        limit_down = vector_of_x0vals_for_ith_largest_grads
        
        
#         print(δ.shape)
        is_positive = δ[torch.arange(len(ε)), argsorted[:, i]].sign() == 1
        
        adder = torch.where(is_positive,torch.minimum(free, limit_up), torch.maximum(-free, -limit_down))
#         print("aa", adder)
        delta[torch.arange(len(ε)), argsorted[:, i]] += adder
#         print(delta)
#     i = 0 
    
#     while i < len() (delta.sum(dim=1) < ε).any():
#     print(delta.abs().sum(dim=1))  
    return x0 + delta.view_as(x0)
    
    
    
    
    
    
    
#     threshold = (1 - ε).unsqueeze(1)
#     δ = (x1 - x0).flatten(1)
#     δ_abs = δ.abs()
#     mask = δ_abs > threshold
#     mid_points = δ_abs.sub_(threshold).copysign_(δ)
#     mid_points.mul_(mask)
#     return x0 + mid_points.reshape(x0.shape)



def l2_mid_points(x0: Tensor, x1: Tensor, ε: Tensor) -> Tensor:
    ε = ε.unsqueeze(1)
    return x0.flatten(1).mul(1 - ε).add_(ε * x1.flatten(1)).view_as(x0)


def linf_mid_points(x0: Tensor, x1: Tensor, ε: Tensor) -> Tensor:
    ε = ε.unsqueeze(1)
    δ = (x1 - x0).flatten(1)
    return x0 + torch.maximum(torch.minimum(δ, ε, out=δ), -ε, out=δ).view_as(x0)

def linf_mid_points_(x0: Tensor, x1: Tensor, ε: Tensor) -> Tensor:
    ε = ε.unsqueeze(1)
    δ = (x1 - x0).flatten(1)
    return x0 + (δ.sign()*eps).view_as(x0)



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
        top_explore: int = 10,
        balanced_init: bool = False, 
        targeted_line: bool = False, 
        fmn_init: bool = True,
        ifmn_α_init: Optional[float] = None,
        track_grad_size: bool = False,
        steepest_line: bool = False,
        croce_l1: bool = False) -> Tensor:
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
    
    if steepest_line:
        _dual_projection_mid_points = {
        0: (None, l0_projection_, l0_mid_points),
        1: (float('inf'), l1_projection_, l1_mid_points_),
        2: (2, l2_projection_, l2_mid_points),
        float('inf'): (1, linf_projection_, linf_mid_points_),
    }
    
    
    if inputs.min() < 0 or inputs.max() > 1: raise ValueError('Input values should be in the [0, 1] range.')
    device = inputs.device
    batch_size = len(inputs)
    batch_view = lambda tensor: tensor.view(batch_size, *[1] * (inputs.ndim - 1))
    dual, projection, mid_point = _dual_projection_mid_points[norm]
    α_final = α_final or α_init / 100
    multiplier = 1 if targeted else -1


    
    if track_grad_size:
        x0_g_tracker = []
        xinit_g_tracker = []
    
    def binary_search(points):
#         if starting_points is not None:
#         is_adv = model(starting_points).argmax(dim=1)
#         #if not is_adv.all():
        #    raise ValueError('Starting points are not all adversarial.')
        lower_bound = torch.zeros(batch_size, device=device)
        upper_bound = torch.ones(batch_size, device=device)
        for _ in range(binary_search_steps):
            ε = (lower_bound + upper_bound) / 2
            mid_points = mid_point(x0=inputs, x1=points, ε=ε)
            pred_labels = model(mid_points).argmax(dim=1)
            is_adv = (pred_labels == labels) if targeted else (pred_labels != labels)
            lower_bound = torch.where(is_adv, lower_bound, ε)
            upper_bound = torch.where(is_adv, ε, upper_bound)

        δ = mid_point(x0=inputs, x1=points, ε=ε) - inputs
        return δ
    
    
    ## use linesearch to get the starting points

    def boundary_search(model: nn.Module,
                        inputs: Tensor,
                        c_inds: Tensor,
                        direction: Tensor,
                        distance_to_boundary: Tensor,
                        steps: int, 
                        targeted_line: bool) -> Tensor:

        # projections not implemented - so suitable only for L2 line search
#         assert norm == 2

#         smallest_adv = torch.ones_like(inputs, device=device) * torch.inf
#         largest_nadv = torch.zeros_like(inputs, device=device)
#         ε = torch.ones(batch_size, device=device) * 1

        
#         lower_bound = torch.zeros(batch_size, device=device)
#         upper_bound = torch.ones(batch_size, device=device)
        
        smallest_adv_distance = torch.ones_like(distance_to_boundary) * torch.inf
        largest_nadv_distance = torch.zeros_like(distance_to_boundary)

        adv_found = torch.ones_like(distance_to_boundary) == 0

        for step in range(steps):
#             # direction * distance for not adv found yet, bs search midpoints delta for the rest
#             δ = torch.where(adv_found, mid_points - inputs, direction.clone()*batch_view(distance_to_boundary))
            
            δ = direction.clone()*batch_view(distance_to_boundary)
            
            # cut to box
            δ.data.add_(inputs).clamp_(min=0, max=1).sub_(inputs)

            # get the points
            points = inputs + δ

            pred = model(points).argmax(dim=1)
            is_adv = (pred == c_inds) if targeted_line else (pred != c_inds)

            # binary search - stays lower/upper bound for not adv found points, move for the rest
#             lower_bound = torch.where(~adv_found | is_adv, lower_bound, ε)
#             upper_bound = torch.where(adv_found & is_adv, ε, upper_bound)

            
            # update points
            is_smaller = distance_to_boundary < smallest_adv_distance
            is_larger  = distance_to_boundary > largest_nadv_distance
            
            # todo - deal with correct casting
#             smallest_adv = smallest_adv.where(batch_view(~is_adv | ~is_smaller), δ)
#             largest_nadv = largest_nadv.where(batch_view(is_adv | ~is_larger), δ)

            # update distances
            smallest_adv_distance = smallest_adv_distance.where(~is_adv | ~is_smaller,  distance_to_boundary)
            largest_nadv_distance = largest_nadv_distance.where(is_adv | ~is_larger,  distance_to_boundary)
            adv_found |= is_adv

            # TODO - correct the epsilons
            # TODO - remove useless computation of midpoints for not adv points
            # distance_to_boundary = distance_to_boundary.where(~adv_found, mid_point(x0=largest_nadv, x1=smallest_adv, ε=ε).flatten(1).norm(p=norm, dim=1))
            if balanced_init:
                assert norm == 2
                distance_to_boundary = distance_to_boundary.where(~adv_found, (smallest_adv_distance + largest_nadv_distance)/2)


            # binary search
#             ε = (lower_bound + upper_bound) / 2
#             mid_points = mid_point(x0=inputs, x1=starting_points, ε=ε)
         
            
            # line search - increase for those not found yet
            distance_to_boundary[~adv_found] *= 2

            
#         δ = mid_point(x0=inputs, x1=starting_points, ε=ε) - inputs


        return smallest_adv_distance, adv_found

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
#     print(class_indcs)
    for c_inds in class_indcs.T:
#         print("ah", c_inds)
        # logit diff to the c_inds
        labels_infhot = torch.zeros_like(logits).scatter_(1, c_inds.unsqueeze(1), float('inf'))
        logit_diff_func = partial(difference_of_logits, labels=c_inds, labels_infhot=labels_infhot)
        logit_diffs = logit_diff_func(logits=logits)

        # check if this make sense (maybe should be -1)
        loss = logit_diffs

        δ_grad = grad(loss.sum(), δ, only_inputs=True, retain_graph=True)[0]
        
        if track_grad_size:
            x0_g_tracker.append(δ_grad.flatten(1).norm(p=norm, dim=1))
        
        # initial linear approximation of distance to the boundary
        distance_to_boundary = loss.detach().abs() / δ_grad.flatten(1).norm(p=2, dim=1).clamp_(min=1e-12)

#         i_distance_to_boundary = distance_to_boundary.clone()
        # normalize tu unit length
        δ_grad /= batch_view(δ_grad.flatten(1).norm(p=2, dim=1))

        if not fmn_init:
            # do a boundary search in L2 δ_grad direction for each points
            distance_to_boundary, adv_found = boundary_search(model, inputs, c_inds if targeted_line else labels, δ_grad, distance_to_boundary, 
                                                              line_search_steps, targeted_line=targeted_line)

            # get samples 
            line_delta = batch_view(distance_to_boundary) * δ_grad
            smallest_adv_samples = line_delta.data.add_(inputs).clamp_(min=0, max=1)

        else:
            smallest_adv_samples = fmn(model, inputs, c_inds, norm=norm, targeted=True, steps=line_search_steps, 
                                       α_init=α_init if ifmn_α_init is None else ifmn_α_init, α_final=0.01)
            adv_found = ~torch.isclose((smallest_adv_samples-inputs).flatten(1).norm(p=norm, dim=1),torch.zeros_like(distance_to_boundary))
                
        if balanced_init:
            # bs already done
            binary_delta = smallest_adv_samples - inputs
        else:
            # do a bs with projections
            binary_delta = binary_search(smallest_adv_samples)

        # if no adversarial was found, use the original inputs 
        smallest_adv = binary_delta.where(batch_view(adv_found), torch.zeros_like(inputs))

        # calculate the new distance to boundary, torch inf for not succesful
        distance_to_boundary = torch.where(adv_found, smallest_adv.flatten(1).norm(p=norm, dim=1), torch.ones_like(adv_found)*torch.inf)
        
        
        distance_to_boundaries.append(distance_to_boundary)
        smallest_advs.append(smallest_adv)

        summary = False
        if summary:
            print("summary")
            new_loss = logit_diff_func(logits=model(inputs + smallest_adv)).detach()
            for i in range(len(c_inds)):
                print(c_inds[i].item(), loss[i].detach().item(), new_loss[i].item(), i_distance_to_boundary[i].item(), distance_to_boundary[i].item())
    
    distance_to_boundaries = torch.stack(distance_to_boundaries).T
    smallest_advs = torch.swapaxes(torch.stack(smallest_advs), 0, 1)
    
#     print(distance_to_boundaries)
#     print(smallest_advs[1,0].sum())
    best_indc = torch.argmin(distance_to_boundaries, dim=1, keepdim=True)
#     best_classes = torch.gather(class_indcs, 1, best_indc)
    best_distances = torch.gather(distance_to_boundaries, 1, best_indc)
    
#     print(best_indc, best_distances)
    # see what class has the real closest boundary
    for t in [*zip(best_indc.tolist(), best_distances.tolist())]:
        print(t)

    
#     starting_points = 
        
        
    # avoid undefined behaviour
    assert (starting_points is None) ^ (top_explore is None)

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
    elif top_explore is not None:
        # start with found delta or initial sample if no adversarial found or the original is already adversary
        orig_adv = logits.argmax(dim=1) != labels
        no_adv = best_distances == torch.inf
#         print(orig_adv.shape)
#         print(no_adv.shape)
#         print(batch_view((no_adv | orig_adv)))
#         print(smallest_advs)
#         print(smallest_advs.sum())
        δ = torch.where(batch_view(no_adv | orig_adv.unsqueeze(1)), torch.zeros_like(inputs), smallest_advs[torch.arange(batch_size), best_indc.squeeze(), :])
#         print(δ)
#         print(δ.sum())
    else:
        δ = torch.zeros_like(inputs)
    δ.requires_grad_(True)

    if norm == 0:
        ε = torch.ones(batch_size, device=device) if starting_points is None else δ.flatten(1).norm(p=0, dim=0)
    else:
        ε = torch.full((batch_size,), float('inf'), device=device)

    
    
    
    ## try multiple step sizes
    # later
    
    
    
    
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
        
        if track_grad_size:
            xinit_g_tracker.append(δ_grad.flatten(1).norm(p=norm, dim=1))
            
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
        
        if norm == 1 and croce_l1:
             δ.data += croce_l1_projection(inputs, δ, ε)
        else:
            # project in place
            projection(δ=δ.data, ε=ε)

        # clamp
        δ.data.add_(inputs).clamp_(min=0, max=1).sub_(inputs)
    if track_grad_size:
        return best_adv, x0_g_tracker, xinit_g_tracker
    return best_adv
