# Standard library
from typing import Optional, Union

# Local application
from .utils import Pred, handles_probabilistic

# Third party
import torch
import torch.nn.functional as F
from pytorch_msssim import ssim as ssim_func
import numpy as np

@handles_probabilistic
def mse(
    pred: Pred,
    target: Union[torch.FloatTensor, torch.DoubleTensor],
    aggregate_only: bool = False,
    lat_weights: Optional[Union[torch.FloatTensor, torch.DoubleTensor]] = None,
) -> Union[torch.FloatTensor, torch.DoubleTensor]:
    error = (pred - target).square()
    if lat_weights is not None:
        error = error * lat_weights
    per_channel_losses = error.mean([0, 2, 3])
    loss = error.mean()
    if aggregate_only:
        return loss
    return torch.cat((per_channel_losses, loss.unsqueeze(0)))


@handles_probabilistic
def msess(
    pred: Pred,
    target: Union[torch.FloatTensor, torch.DoubleTensor],
    climatology: Union[torch.FloatTensor, torch.DoubleTensor],
    aggregate_only: bool = False,
    lat_weights: Optional[Union[torch.FloatTensor, torch.DoubleTensor]] = None,
) -> Union[torch.FloatTensor, torch.DoubleTensor]:
    pred_mse = mse(pred, target, aggregate_only, lat_weights)
    clim_mse = mse(climatology, target, aggregate_only, lat_weights)
    return 1 - pred_mse / clim_mse


@handles_probabilistic
def mae(
    pred: Pred,
    target: Union[torch.FloatTensor, torch.DoubleTensor],
    aggregate_only: bool = False,
    lat_weights: Optional[Union[torch.FloatTensor, torch.DoubleTensor]] = None,
) -> Union[torch.FloatTensor, torch.DoubleTensor]:
    error = (pred - target).abs()
    if lat_weights is not None:
        error = error * lat_weights
    per_channel_losses = error.mean([0, 2, 3])
    loss = error.mean()
    if aggregate_only:
        return loss
    return torch.cat((per_channel_losses, loss.unsqueeze(0)))


@handles_probabilistic
def rmse(
    pred: Pred,
    target: Union[torch.FloatTensor, torch.DoubleTensor],
    aggregate_only: bool = False,
    lat_weights: Optional[Union[torch.FloatTensor, torch.DoubleTensor]] = None,
    mask=None,
) -> Union[torch.FloatTensor, torch.DoubleTensor]:
    error = (pred - target).square()
    if lat_weights is not None:
        error = error * lat_weights
    if mask is not None:
        error = error * mask
        eps = 1e-9
        masked_lat_weights = torch.mean(mask, dim=(1, 2, 3), keepdim=True) + eps
        error = error / masked_lat_weights
    per_channel_losses = error.mean([2, 3]).sqrt().mean(0)
    loss = per_channel_losses.mean()
    if aggregate_only:
        return loss
    return torch.cat((per_channel_losses, loss.unsqueeze(0)))


@handles_probabilistic
def acc(
    pred: Pred,
    target: Union[torch.FloatTensor, torch.DoubleTensor],
    climatology: Optional[Union[torch.FloatTensor, torch.DoubleTensor]],
    aggregate_only: bool = False,
    lat_weights: Optional[Union[torch.FloatTensor, torch.DoubleTensor]] = None,
    mask=None,
) -> Union[torch.FloatTensor, torch.DoubleTensor]:
    pred = pred - climatology
    target = target - climatology
    per_channel_acc = []
    for i in range(pred.shape[1]):
        pred_prime = pred[:, i] - pred[:, i].mean()
        target_prime = target[:, i] - target[:, i].mean()
        if mask is not None:
            eps = 1e-9
            numer = (mask * lat_weights * pred_prime * target_prime).sum()
            denom1 = ((mask + eps) * lat_weights * pred_prime.square()).sum()
            denom2 = ((mask + eps) * lat_weights * target_prime.square()).sum()
        else:
            numer = (lat_weights * pred_prime * target_prime).sum()
            denom1 = (lat_weights * pred_prime.square()).sum()
            denom2 = (lat_weights * target_prime.square()).sum()
        numer = (lat_weights * pred_prime * target_prime).sum()
        denom1 = (lat_weights * pred_prime.square()).sum()
        denom2 = (lat_weights * target_prime.square()).sum()
        per_channel_acc.append(numer / (denom1 * denom2).sqrt())
    per_channel_acc = torch.stack(per_channel_acc)
    result = per_channel_acc.mean()
    if aggregate_only:
        return result
    return torch.cat((per_channel_acc, result.unsqueeze(0)))


@handles_probabilistic
def pearson(
    pred: Pred,
    target: Union[torch.FloatTensor, torch.DoubleTensor],
    aggregate_only: bool = False,
) -> Union[torch.FloatTensor, torch.DoubleTensor]:
    pred = _flatten_channel_wise(pred)
    target = _flatten_channel_wise(target)
    pred = pred - pred.mean(1, keepdims=True)
    target = target - target.mean(1, keepdims=True)
    per_channel_coeffs = F.cosine_similarity(pred, target)
    coeff = torch.mean(per_channel_coeffs)
    if not aggregate_only:
        coeff = coeff.unsqueeze(0)
        coeff = torch.cat((per_channel_coeffs, coeff))
    return coeff


@handles_probabilistic
def bce(
    pred: Pred,
    target: Union[torch.FloatTensor, torch.DoubleTensor],
    aggregate_only: bool = True,
) -> Union[torch.FloatTensor, torch.DoubleTensor]:
    bce_loss = torch.nn.BCEWithLogitsLoss(reduction='none')(pred, target)
    # Calculate the mean BCE loss per class
    per_class_bce = bce_loss.mean(dim=0)  # Mean across the batch dimension
    # Calculate the overall mean BCE loss
    bce = per_class_bce.mean()
    if aggregate_only:
        return bce
    # Return individual class BCE losses along with the aggregate
    return torch.cat((per_class_bce, bce.unsqueeze(0)))

@handles_probabilistic
def mean_bias(
    pred: Pred,
    target: Union[torch.FloatTensor, torch.DoubleTensor],
    aggregate_only: bool = False,
) -> Union[torch.FloatTensor, torch.DoubleTensor]:
    per_channel_mb = []
    for i in range(pred.shape[1]):
        per_channel_mb.append(target[:, i].mean() - pred[:, i].mean())
    per_channel_mb = torch.stack(per_channel_mb)
    result = per_channel_mb.mean()
    if aggregate_only:
        return result
    return torch.cat((per_channel_mb, result.unsqueeze(0)))


def _flatten_channel_wise(x: torch.Tensor) -> torch.Tensor:
    """
    :param x: A tensor of shape [B,C,H,W].
    :type x: torch.Tensor

    :return: A tensor of shape [C,B*H*W].
    :rtype: torch.Tensor
    """
    subtensors = torch.tensor_split(x, x.shape[1], 1)
    result = torch.stack([t.flatten() for t in subtensors])
    return result


def gaussian_crps(
    pred: torch.distributions.Normal,
    target: Union[torch.FloatTensor, torch.DoubleTensor],
    aggregate_only: bool = False,
    lat_weights: Optional[Union[torch.FloatTensor, torch.DoubleTensor]] = None,
) -> Union[torch.FloatTensor, torch.DoubleTensor]:
    mean, std = pred.loc, pred.scale
    z = (target - mean) / std
    standard_normal = torch.distributions.Normal(
        torch.zeros_like(pred), torch.ones_like(pred)
    )
    pdf = torch.exp(standard_normal.log_prob(z))
    cdf = standard_normal.cdf(z)
    crps = std * (z * (2 * cdf - 1) + 2 * pdf - 1 / torch.pi)
    if lat_weights is not None:
        crps = crps * lat_weights
    per_channel_losses = crps.mean([0, 2, 3])
    loss = crps.mean()
    if aggregate_only:
        return loss
    return torch.cat((per_channel_losses, loss.unsqueeze(0)))


def gaussian_spread(
    pred: torch.distributions.Normal,
    aggregate_only: bool = False,
    lat_weights: Optional[Union[torch.FloatTensor, torch.DoubleTensor]] = None,
) -> Union[torch.FloatTensor, torch.DoubleTensor]:
    variance = torch.square(pred.scale)
    if lat_weights is not None:
        variance = variance * lat_weights
    per_channel_losses = variance.mean([2, 3]).sqrt().mean(0)
    loss = variance.mean()
    if aggregate_only:
        return loss
    return torch.cat((per_channel_losses, loss.unsqueeze(0)))


def gaussian_spread_skill_ratio(
    pred: torch.distributions.Normal,
    target: Union[torch.FloatTensor, torch.DoubleTensor],
    aggregate_only: bool = False,
    lat_weights: Optional[Union[torch.FloatTensor, torch.DoubleTensor]] = None,
) -> Union[torch.FloatTensor, torch.DoubleTensor]:
    spread = gaussian_spread(pred, aggregate_only, lat_weights)
    error = rmse(pred, target, aggregate_only, lat_weights)
    return spread / error


def nrmses(
    pred: Union[torch.FloatTensor, torch.DoubleTensor],
    target: Union[torch.FloatTensor, torch.DoubleTensor],
    clim: Union[torch.FloatTensor, torch.DoubleTensor],
    aggregate_only: bool = False,
    lat_weights: Optional[Union[torch.FloatTensor, torch.DoubleTensor]] = None,
) -> Union[torch.FloatTensor, torch.DoubleTensor]:
    y_normalization = clim.squeeze()
    error = (pred.mean(dim=0) - target.mean(dim=0)) ** 2  # (C, H, W)
    if lat_weights is not None:
        error = error * lat_weights.squeeze(0)
    per_channel_losses = error.mean(dim=(-2, -1)).sqrt() / y_normalization  # C
    loss = per_channel_losses.mean()
    if aggregate_only:
        return loss
    return torch.cat((per_channel_losses, loss.unsqueeze(0)))


def nrmseg(
    pred: Union[torch.FloatTensor, torch.DoubleTensor],
    target: Union[torch.FloatTensor, torch.DoubleTensor],
    clim: Union[torch.FloatTensor, torch.DoubleTensor],
    aggregate_only: bool = False,
    lat_weights: Optional[Union[torch.FloatTensor, torch.DoubleTensor]] = None,
) -> Union[torch.FloatTensor, torch.DoubleTensor]:
    y_normalization = clim.squeeze()
    if lat_weights is not None:
        pred = pred * lat_weights
        target = target * lat_weights
    pred = pred.mean(dim=(-2, -1))  # N, C
    target = target.mean(dim=(-2, -1))  # N, C
    error = (pred - target) ** 2
    per_channel_losses = error.mean(0).sqrt() / y_normalization  # C
    loss = per_channel_losses.mean()
    if aggregate_only:
        return loss
    return torch.cat((per_channel_losses, loss.unsqueeze(0)))


def pnsr(
    pred: Union[torch.FloatTensor, torch.DoubleTensor],
    target: Union[torch.FloatTensor, torch.DoubleTensor],
    aggregate_only: bool = False,
    lat_weights: Optional[Union[torch.FloatTensor, torch.DoubleTensor]] = None,
) -> Union[torch.FloatTensor, torch.DoubleTensor]:

    mse_value = mse(pred, target, False, lat_weights)[:pred.shape[1]] # C
    # per_channel_max_value = 1.0
    per_channel_losses = []
    for i in range(pred.shape[1]):
        pnsr_value = 20 * torch.log10(1.0 / torch.sqrt(torch.Tensor([mse_value[i]])))
        per_channel_losses.append(pnsr_value)
    per_channel_losses = torch.tensor(per_channel_losses).squeeze()
    loss = per_channel_losses.mean()
    if aggregate_only:
        return loss
    return torch.cat((per_channel_losses, loss.unsqueeze(0)))


def ssim(
    pred: Union[torch.FloatTensor, torch.DoubleTensor],
    target: Union[torch.FloatTensor, torch.DoubleTensor],
    aggregate_only: bool = False,
    lat_weights: Optional[Union[torch.FloatTensor, torch.DoubleTensor]] = None,
) -> Union[torch.FloatTensor, torch.DoubleTensor]:
    if lat_weights is not None:
        pred = pred * lat_weights
        target = target * lat_weights
        
    per_channel_losses = ssim_func(torch.swapaxes(pred, 0, 1),
                                   torch.swapaxes(target, 0, 1),
                                   data_range=1.0,
                                   size_average=False)
    loss = per_channel_losses.mean()
    if aggregate_only:
        return loss
    return torch.cat((per_channel_losses, loss.unsqueeze(0)))


def kge(
    pred: Union[torch.FloatTensor, torch.DoubleTensor],
    target: Union[torch.FloatTensor, torch.DoubleTensor],
    aggregate_only: bool = False,
    lat_weights: Optional[Union[torch.FloatTensor, torch.DoubleTensor]] = None,
) -> Union[torch.FloatTensor, torch.DoubleTensor]:
      
    if lat_weights is not None:
        pred = pred * lat_weights
        target = target * lat_weights
    
    num_channels = pred.shape[1]
    per_channel_losses = torch.empty(num_channels)

    for i in range(num_channels):
        pred_channel = torch.ravel(pred[:, i])
        target_channel = torch.ravel(target[:, i])
        kge, r, alpha, beta = kge_torch(pred_channel, target_channel)
        per_channel_losses[i] = kge
        
    per_channel_losses = torch.tensor(per_channel_losses).squeeze()
    loss = per_channel_losses.mean()
    if aggregate_only:
        return loss
    return torch.cat((per_channel_losses, loss.unsqueeze(0)))



def kge_torch(simulations, evaluation):
    """Original Kling-Gupta Efficiency (KGE) and its three components
    (r, α, β) as per `Gupta et al., 2009
    <https://doi.org/10.1016/j.jhydrol.2009.08.003>`_.

    Note, all four values KGE, r, α, β are returned, in this order.

    :Calculation Details:
        .. math::
           E_{\\text{KGE}} = 1 - \\sqrt{[r - 1]^2 + [\\alpha - 1]^2
           + [\\beta - 1]^2}
        .. math::
           r = \\frac{\\text{cov}(e, s)}{\\sigma({e}) \\cdot \\sigma(s)}
        .. math::
           \\alpha = \\frac{\\sigma(s)}{\\sigma(e)}
        .. math::
           \\beta = \\frac{\\mu(s)}{\\mu(e)}

        where *e* is the *evaluation* series, *s* is (one of) the
        *simulations* series, *cov* is the covariance, *σ* is the
        standard deviation, and *μ* is the arithmetic mean.

    """
    # calculate error in timing and dynamics r
    # (Pearson's correlation coefficient)
    sim_mean = torch.mean(simulations, dim=0)
    obs_mean = torch.mean(evaluation)

    r_num = torch.sum((simulations - sim_mean) * (evaluation - obs_mean), dim=0)
    r_den = torch.sqrt(torch.sum((simulations - sim_mean) ** 2, dim=0) *
                       torch.sum((evaluation - obs_mean) ** 2))
    r = r_num / r_den

    # calculate error in spread of flow alpha
    alpha = torch.std(simulations, dim=0) / torch.std(evaluation)
    
    # calculate error in volume beta (bias of mean discharge)
    beta = torch.sum(simulations, dim=0) / torch.sum(evaluation)
    
    # calculate the Kling-Gupta Efficiency KGE
    kge_ = 1 - torch.sqrt((r - 1) ** 2 + (alpha - 1) ** 2 + (beta - 1) ** 2)

    return torch.stack((kge_, r, alpha, beta))

