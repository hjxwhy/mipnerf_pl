import torch
from torch import nn
from einops import repeat
import collections
from collections import namedtuple
import math
import torch.nn.functional as F
import functorch
from functorch.compile import memory_efficient_fusion
from copy import deepcopy
from typing import List
import time
import functools
import random
from collections import namedtuple
from einops import rearrange
import numpy as np
import math



Rays = collections.namedtuple(
    'Rays',
    ('origins', 'directions', 'viewdirs', 'radii', 'lossmult', 'near', 'far', 'depth', 'normal', 'mask'))
Rays_keys = Rays._fields

def l2_normalize(x, eps=1e-6):        
    """Normalize x to unit length along last axis."""
    return x / torch.sqrt(torch.maximum(torch.sum(x**2, axis=-1, keepdims=True), torch.tensor(eps, device = x.device)))

def distloss(weight, samples):
    '''
    mip-nerf 360 sec.4
    weight: [B, N]
    samples:[N, N+1]
    '''
    interval = samples[:, 1:] - samples[:, :-1]
    mid_points = (samples[:, 1:] + samples[:, :-1]) * 0.5
    loss_uni = (1/3) * (interval * weight.pow(2)).sum(-1).mean()
    ww = weight.unsqueeze(-1) * weight.unsqueeze(-2)          # [B,N,N]
    mm = (mid_points.unsqueeze(-1) - mid_points.unsqueeze(-2)).abs()  # [B,N,N]
    loss_bi = (ww * mm).sum((-1,-2)).mean()
    return loss_uni + loss_bi

@torch.jit.script
def lift_gaussian(directions, t_mean, t_var, r_var, diagonal: bool):
    """Lift a Gaussian defined along a ray to 3D coordinates."""
    mean = torch.unsqueeze(directions, dim=-2) * torch.unsqueeze(t_mean, dim=-1)  # [B, 1, 3]*[B, N, 1] = [B, N, 3]
    d_norm_denominator = torch.sum(directions ** 2, dim=-1, keepdim=True) + 1e-10
    # min_denominator = torch.full_like(d_norm_denominator, 1e-10)
    # d_norm_denominator = torch.maximum(min_denominator, d_norm_denominator)

    #if diagonal:
    d_outer_diag = directions ** 2  # eq (16)
    null_outer_diag = 1 - d_outer_diag / d_norm_denominator
    t_cov_diag = torch.unsqueeze(t_var, dim=-1) * torch.unsqueeze(d_outer_diag,
                                                                    dim=-2)  # [B, N, 1] * [B, 1, 3] = [B, N, 3]
    xy_cov_diag = torch.unsqueeze(r_var, dim=-1) * torch.unsqueeze(null_outer_diag, dim=-2)
    cov_diag = t_cov_diag + xy_cov_diag
    return mean, cov_diag
    """else:
        d_outer = torch.unsqueeze(directions, dim=-1) * torch.unsqueeze(directions,
                                                                        dim=-2)  # [B, 3, 1] * [B, 1, 3] = [B, 3, 3]
        eye = torch.eye(directions.shape[-1], device=directions.device)  # [B, 3, 3]
        # [B, 3, 1] * ([B, 3] / [B, 1])[..., None, :] = [B, 3, 3]
        null_outer = eye - torch.unsqueeze(directions, dim=-1) * (directions / d_norm_denominator).unsqueeze(-2)
        t_cov = t_var.unsqueeze(-1).unsqueeze(-1) * d_outer.unsqueeze(-3)  # [B, N, 1, 1] * [B, 1, 3, 3] = [B, N, 3, 3]
        xy_cov = t_var.unsqueeze(-1).unsqueeze(-1) * null_outer.unsqueeze(
            -3)  # [B, N, 1, 1] * [B, 1, 3, 3] = [B, N, 3, 3]
        cov = t_cov + xy_cov
        return mean, cov"""

@torch.jit.script
def compute_parameters(t0, t1, base_radius):
    mu = (t0 + t1) / 2
    hw = (t1 - t0) / 2
    t_mean = mu + (2 * mu * hw ** 2) / (3 * mu ** 2 + hw ** 2)
    t_var = (hw ** 2) / 3 - (4 / 15) * ((hw ** 4 * (12 * mu ** 2 - hw ** 2)) /
                                        (3 * mu ** 2 + hw ** 2) ** 2)
    r_var = base_radius ** 2 * ((mu ** 2) / 4 + (5 / 12) * hw ** 2 - 4 / 15 *
                                (hw ** 4) / (3 * mu ** 2 + hw ** 2))
    return t_mean, t_var, r_var


def conical_frustum_to_gaussian(directions, t0, t1, base_radius, diagonal: bool, stable: bool=True):
    """Approximate a conical frustum as a Gaussian distribution (mean+cov).
    Assumes the ray is originating from the origin, and base_radius is the
    radius at dist=1. Doesn't assume `directions` is normalized.
    Args:
        directions: torch.tensor float32 3-vector, the axis of the cone
        t0: float, the starting distance of the frustum.
        t1: float, the ending distance of the frustum.
        base_radius: float, the scale of the radius as a function of distance.
        diagonal: boolean, whether or the Gaussian will be diagonal or full-covariance.
        stable: boolean, whether or not to use the stable computation described in
        the paper (setting this to False will cause catastrophic failure).
    Returns:
        a Gaussian (mean and covariance).
    """
    #if stable:
    """mu = (t0 + t1) / 2
    hw = (t1 - t0) / 2
    t_mean = mu + (2 * mu * hw ** 2) / (3 * mu ** 2 + hw ** 2)
    t_var = (hw ** 2) / 3 - (4 / 15) * ((hw ** 4 * (12 * mu ** 2 - hw ** 2)) /
                                        (3 * mu ** 2 + hw ** 2) ** 2)
    r_var = base_radius ** 2 * ((mu ** 2) / 4 + (5 / 12) * hw ** 2 - 4 / 15 *
                                (hw ** 4) / (3 * mu ** 2 + hw ** 2))"""
    t_mean, t_var, r_var = compute_parameters(t0, t1, base_radius)
    """else:
        t_mean = (3 * (t1 ** 4 - t0 ** 4)) / (4 * (t1 ** 3 - t0 ** 3))
        r_var = base_radius ** 2 * (3 / 20 * (t1 ** 5 - t0 ** 5) / (t1 ** 3 - t0 ** 3))
        t_mosq = 3 / 5 * (t1 ** 5 - t0 ** 5) / (t1 ** 3 - t0 ** 3)
        t_var = t_mosq - t_mean ** 2"""
    return lift_gaussian(directions, t_mean, t_var, r_var, diagonal)

@torch.jit.script
def cast_rays(t_samples: torch.Tensor, origins: torch.Tensor, directions: torch.Tensor, radii: torch.Tensor, ray_shape: str, diagonal: bool=True):
    """Cast rays (cone- or cylinder-shaped) and featurize sections of it.
    Args:
        t_samples: float array [B, n_sample+1], the "fencepost" distances along the ray.
        origins: float array [B, 3], the ray origin coordinates.
        directions [B, 3]: float array, the ray direction vectors.
        radii[B, 1]: float array, the radii (base radii for cones) of the rays.
        ray_shape: string, the shape of the ray, must be 'cone' or 'cylinder'.
        diagonal: boolean, whether or not the covariance matrices should be diagonal.
    Returns:
        a tuple of arrays of means and covariances.
    """
    t0 = t_samples[..., :-1]  # [B, n_samples]
    t1 = t_samples[..., 1:]
    #if ray_shape == 'cone':
    gaussian_fn = conical_frustum_to_gaussian
    #elif ray_shape == 'cylinder':
    #    raise NotImplementedError
    #else:
    #    assert False
    means, covs = gaussian_fn(directions, t0, t1, radii, diagonal)
    means = means + torch.unsqueeze(origins, dim=-2)
    return means, covs


def sample_along_rays_360(origins, directions, radii, num_samples, near, far, randomized, disparity, ray_shape):
    batch_size = origins.shape[0]
    t_samples = torch.linspace(0., 1., num_samples + 1, device=origins.device)
    far_inv = 1 / far
    near_inv = 1 / near
    t_inv = far_inv * t_samples + (1 - t_samples) * near_inv

    if randomized:
        mids = 0.5 * (t_inv[..., 1:] + t_inv[..., :-1])
        upper = torch.cat([mids, t_inv[..., -1:]], -1)
        lower = torch.cat([t_inv[..., :1], mids], -1)
        t_rand = torch.rand(batch_size, num_samples + 1, device=origins.device)
        t_inv = lower + (upper - lower) * t_rand
    else:
        # Broadcast t_inv to make the returned shape consistent.
        t_inv = torch.broadcast_to(t_inv, [batch_size, num_samples + 1])
    t = 1 / t_inv
    means, covs = cast_rays(t, origins, directions, radii, ray_shape, False)
    return t_inv, (means, covs)

#@torch.jit.script
def sample_along_rays(origins: torch.Tensor, directions: torch.Tensor, radii: torch.Tensor, num_samples, near: torch.Tensor, 
    far: torch.Tensor, randomized: bool, disparity: bool, ray_shape: str, rays: namedtuple, sigma_depth):
    """
    Stratified sampling along the rays.
    Args:
        origins: torch.Tensor, [batch_size, 3], ray origins.
        directions: torch.Tensor, [batch_size, 3], ray directions.
        radii: torch.Tensor, [batch_size, 3], ray radii.
        num_samples: int.
        near: torch.Tensor, [batch_size, 1], near clip.
        far: torch.Tensor, [batch_size, 1], far clip.
        randomized: bool, use randomized stratified sampling.
        disparity: bool, sampling linearly in disparity rather than depth.
        ray_shape: string, which shape ray to assume.
    Returns:
    t_samples: torch.Tensor, [batch_size, num_samples], sampled z values.
    means: torch.Tensor, [batch_size, num_samples, 3], sampled means.
    covs: torch.Tensor, [batch_size, num_samples, 3, 3], sampled covariances.
    """
    if sigma_depth is not None: 
        num_t_samples = num_samples // 4 * 3 + 1
        num_ss_samples = num_samples // 4
    else: 
        num_t_samples = num_samples + 1
    batch_size = origins.shape[0]

    t_samples = torch.linspace(0., 1., num_t_samples, device=origins.device)

    if disparity:
        t_samples = 1. / (1. / near * (1. - t_samples) + 1. / far * t_samples)
    else:
        # t_samples = near * (1. - t_samples) + far * t_samples
        t_samples = near + (far - near) * t_samples

    if randomized:
        mids = 0.5 * (t_samples[..., 1:] + t_samples[..., :-1])
        upper = torch.cat([mids, t_samples[..., -1:]], -1)
        lower = torch.cat([t_samples[..., :1], mids], -1)
        t_rand = torch.rand(batch_size, num_t_samples, device=origins.device)
        t_samples = lower + (upper - lower) * t_rand
    #else:
         # Broadcast t_samples to make the returned shape consistent.
    #    t_samples = torch.broadcast_to(t_samples, [batch_size, num_samples + 1])
    if sigma_depth is not None: 
        ss_tsamples = depth_guided_supersampling(rays=rays, n_samples=num_ss_samples, start_sigma=sigma_depth)
        t_samples, _ = torch.sort(torch.cat([t_samples, ss_tsamples], -1), -1)

    means, covs = cast_rays(t_samples, origins, directions, radii, ray_shape)
    return t_samples, (means, covs)

#@torch.jit.script
def sorted_piecewise_constant_pdf(bins, weights, num_samples, randomized):
    """
    Piecewise-Constant PDF sampling from sorted bins.
    Args:
        bins: torch.Tensor, [batch_size, num_bins + 1].
        weights: torch.Tensor, [batch_size, num_bins].
        num_samples: int, the number of samples.
        randomized: bool, use randomized samples.
    Returns:
        t_samples: torch.Tensor, [batch_size, num_samples].
    """
    # Pad each weight vector (only if necessary) to bring its sum to `eps`. This
    # avoids NaNs when the input is zeros or small, but has no effect otherwise.
    eps = 1e-5
    weight_sum = torch.sum(weights, dim=-1, keepdim=True)  # [B, 1]
    padding = torch.maximum(torch.zeros_like(weight_sum), eps - weight_sum)
    weights += padding / weights.shape[-1]
    weight_sum += padding

    # Compute the PDF and CDF for each weight vector, while ensuring that the CDF
    # starts with exactly 0 and ends with exactly 1.
    pdf = weights / weight_sum
    cdf = torch.cumsum(pdf[..., :-1], dim=-1)
    cdf = torch.minimum(torch.ones_like(cdf), cdf)
    cdf = torch.cat([torch.zeros(list(cdf.shape[:-1]) + [1], device=cdf.device),
                     cdf,
                     torch.ones(list(cdf.shape[:-1]) + [1], device=cdf.device)],
                    dim=-1)  # [B, N]

    # Draw uniform samples.
    if randomized:
        s = 1 / num_samples
        u = (torch.arange(num_samples, device=cdf.device) * s)[None, ...]
        u = u + torch.empty(list(cdf.shape[:-1]) + [num_samples], device=cdf.device).uniform_(
            to=(s - torch.finfo(torch.float32).eps))
        # `u` is in [0, 1) --- it can be zero, but it can never be 1.
        u = torch.minimum(u, torch.full_like(u, 1. - torch.finfo(torch.float32).eps, device=u.device))
    else:
        # Match the behavior of jax.random.uniform() by spanning [0, 1-eps].
        u = torch.linspace(0., 1. - torch.finfo(torch.float32).eps, num_samples, device=cdf.device)
        u = torch.broadcast_to(u, list(cdf.shape[:-1]) + [num_samples])
    u = u.contiguous()
    #try:
    inds = torch.searchsorted(cdf, u, right=True)
    """except:
        # for lower version torch that does not have torch.searchsorted,
        # you need to manually install from
        # https://github.com/aliutkus/torchsearchsorted
        from torchsearchsorted import searchsorted
        inds = searchsorted(cdf, u, side='right')"""
    below = torch.max(torch.zeros_like(inds - 1), inds - 1)
    above = torch.min((cdf.shape[-1] - 1) * torch.ones_like(inds), inds)
    inds_g = torch.stack([below, above], -1)  # (batch, N_samples, 2)

    matched_shape = [inds_g.shape[0], inds_g.shape[1], cdf.shape[-1]]
    cdf_g = torch.gather(cdf.unsqueeze(1).expand(matched_shape), 2, inds_g)
    bins_g = torch.gather(bins.unsqueeze(1).expand(matched_shape), 2, inds_g)
    denom = (cdf_g[..., 1] - cdf_g[..., 0])
    denom = torch.where(denom < 1e-5, torch.ones_like(denom), denom)
    t = (u - cdf_g[..., 0]) / denom
    samples = bins_g[..., 0] + t * (bins_g[..., 1] - bins_g[..., 0])
    return samples

def sorted_piecewise_constant_pdf_(
    bins, weights, num_samples, randomized, float_min_eps=2**-14
):

    eps = 1e-5
    weight_sum = weights.sum(dim=-1, keepdims=True)
    padding = torch.fmax(torch.zeros_like(weight_sum), eps - weight_sum)
    weights += padding / weights.shape[-1]
    weight_sum += padding

    pdf = weights / weight_sum
    cdf = torch.fmin(
        torch.ones_like(pdf[..., :-1]), torch.cumsum(pdf[..., :-1], axis=-1)
    )
    cdf = torch.cat(
        [
            torch.zeros(list(cdf.shape[:-1]) + [1], device=weights.device),
            cdf,
            torch.ones(list(cdf.shape[:-1]) + [1], device=weights.device),
        ],
        axis=-1,
    )

    if randomized:
        s = 1 / num_samples
        u = torch.arange(num_samples, device=weights.device) * s
        u += torch.rand_like(u) * (s - float_min_eps)
        u = torch.fmin(u, torch.ones_like(u) * (1.0 - float_min_eps))
    else:
        u = torch.linspace(0.0, 1.0 - float_min_eps, num_samples, device=cdf.device)
        u = torch.broadcast_to(u, list(cdf.shape[:-1]) + [num_samples])

    mask = u[..., None, :] >= cdf[..., :, None]

    bin0 = (mask * bins[..., None] + ~mask * bins[..., :1, None]).max(dim=-2)[0]
    bin1 = (~mask * bins[..., None] + mask * bins[..., -1:, None]).min(dim=-2)[0]
    # Debug Here
    cdf0 = (mask * cdf[..., None] + ~mask * cdf[..., :1, None]).max(dim=-2)[0]
    cdf1 = (~mask * cdf[..., None] + mask * cdf[..., -1:, None]).min(dim=-2)[0]

    t = torch.clip(torch.nan_to_num((u - cdf0) / (cdf1 - cdf0), 0), 0, 1)
    samples = bin0 + t * (bin1 - bin0)

    return samples


#@torch.jit.script
def resample_along_rays(origins, directions, radii, t_samples, weights, randomized, ray_shape, stop_grad,
                        resample_padding):
    """Resampling.
    Args:
        origins: torch.Tensor, [batch_size, 3], ray origins.
        directions: torch.Tensor, [batch_size, 3], ray directions.
        radii: torch.Tensor, [batch_size, 3], ray radii.
        t_samples: torch.Tensor, [batch_size, num_samples+1].
        weights: torch.Tensor [batch_size, num_samples], weights for t_samples
        randomized: bool, use randomized samples.
        ray_shape: string, which kind of shape to assume for the ray.
        stop_grad: bool, whether or not to backprop through sampling.
        resample_padding: float, added to the weights before normalizing.
    Returns:
        t_samples: torch.Tensor, [batch_size, num_samples+1].
        points: torch.Tensor, [batch_size, num_samples, 3].
    """
    # Do a blurpool.
    if stop_grad:
        with torch.no_grad():
            weights_pad = torch.cat([weights[..., :1], weights, weights[..., -1:]], dim=-1)
            weights_max = torch.maximum(weights_pad[..., :-1], weights_pad[..., 1:])
            weights_blur = 0.5 * (weights_max[..., :-1] + weights_max[..., 1:])

            # Add in a constant (the sampling function will renormalize the PDF).
            weights = weights_blur + resample_padding

            new_t_vals = sorted_piecewise_constant_pdf(
                t_samples,
                weights,
                t_samples.shape[-1],
                randomized,
            )
    else:
        weights_pad = torch.cat([weights[..., :1], weights, weights[..., -1:]], dim=-1)
        weights_max = torch.maximum(weights_pad[..., :-1], weights_pad[..., 1:])
        weights_blur = 0.5 * (weights_max[..., :-1] + weights_max[..., 1:])

        # Add in a constant (the sampling function will renormalize the PDF).
        weights = weights_blur + resample_padding

        new_t_vals = sorted_piecewise_constant_pdf(
            t_samples,
            weights,
            t_samples.shape[-1],
            randomized,
        )
    means, covs = cast_rays(new_t_vals, origins, directions, radii, ray_shape)
    return new_t_vals, (means, covs)

@torch.jit.script
def expected_sin(x, x_var):
    """Estimates mean of sin(z), z ~ N(x, var)."""
    # When the variance is wide, shrink sin towards zero.
    #y = torch.exp(-0.5 * x_var) * torch.sin(x)  # [B, N, 2*3*L]
    #y_var = torch.maximum(torch.zeros_like(x), 0.5 * (1 - torch.exp(-2 * x_var) * torch.cos(2 * x)) - y**2)
    return torch.exp(-0.5 * x_var) * torch.sin(x)#, y_var

"""
def integrated_pos_enc_360(means_covs):
    P = torch.tensor([[0.8506508, 0, 0.5257311],
                      [0.809017, 0.5, 0.309017],
                      [0.5257311, 0.8506508, 0],
                      [1, 0, 0],
                      [0.809017, 0.5, -0.309017],
                      [0.8506508, 0, -0.5257311],
                      [0.309017, 0.809017, -0.5],
                      [0, 0.5257311, -0.8506508],
                      [0.5, 0.309017, -0.809017],
                      [0, 1, 0],
                      [-0.5257311, 0.8506508, 0],
                      [-0.309017, 0.809017, -0.5],
                      [0, 0.5257311, 0.8506508],
                      [-0.309017, 0.809017, 0.5],
                      [0.309017, 0.809017, 0.5],
                      [0.5, 0.309017, 0.809017],
                      [0.5, -0.309017, 0.809017],
                      [0, 0, 1],
                      [-0.5, 0.309017, 0.809017],
                      [-0.809017, 0.5, 0.309017],
                      [-0.809017, 0.5, -0.309017]]).T
    means, covs = means_covs
    P = P.to(means.device)
    means, x_cov = parameterization(means, covs)
    y = torch.matmul(means, P)
    y_var = torch.sum((torch.matmul(x_cov, P)) * P, -2)
    return expected_sin(torch.cat([y, y + 0.5 * torch.tensor(np.pi)], dim=-1), torch.cat([y_var] * 2, dim=-1))"""

@torch.jit.script
def integrated_pos_enc(means_covs: tuple[torch.Tensor, torch.Tensor], min_deg: int, max_deg: int, diagonal: bool=True):
    """Encode `means` with sinusoids scaled by 2^[min_deg:max_deg-1].
    Args:
        means_covs:[B, N, 3] a tuple containing: means, torch.Tensor, variables to be encoded.
        covs, [B, N, 3] torch.Tensor, covariance matrices.
        min_deg: int, the min degree of the encoding.
        max_deg: int, the max degree of the encoding.
        diagonal: bool, if true, expects input covariances to be diagonal (full otherwise).
    Returns:
        encoded: torch.Tensor, encoded variables.
    """
    
    mean, var = means_covs
    pi = torch.as_tensor(math.pi, device=mean.device)
    scales = 2**torch.arange(min_deg, max_deg, device=mean.device)[..., None]
    shape = mean.shape[:-1] + (-1,)
    scaled_mean = (mean[..., None, :] * scales).view(shape)
    scaled_var = (var[..., None, :] * scales**2).view(shape)
    
    return expected_sin(
      torch.cat([scaled_mean, scaled_mean + 0.5 * pi], dim=-1),
      torch.cat([scaled_var] * 2, dim=-1))


def integrated_pos_enc_(means_covs, min_deg: int, max_deg: int, diagonal: bool=True):
    """Encode `means` with sinusoids scaled by 2^[min_deg:max_deg-1].
    Args:
        means_covs:[B, N, 3] a tuple containing: means, torch.Tensor, variables to be encoded.
        covs, [B, N, 3] torch.Tensor, covariance matrices.
        min_deg: int, the min degree of the encoding.
        max_deg: int, the max degree of the encoding.
        diagonal: bool, if true, expects input covariances to be diagonal (full otherwise).
    Returns:
        encoded: torch.Tensor, encoded variables.
    """
    if diagonal:
        means, covs_diag = means_covs
        scales = diagonal_scale(min_deg, max_deg) #torch.tensor([2 ** i for i in range(min_deg, max_deg)], device=means.device)  # [L]
        # [B, N, 1, 3] * [L, 1] = [B, N, L, 3]->[B, N, 3L]
        y = rearrange(torch.unsqueeze(means, dim=-2) * scales, #torch.unsqueeze(scales, dim=-1),
                      pattern='batch sample scale_dim mean_dim -> batch sample (scale_dim mean_dim)')
        # [B, N, 1, 3] * [L, 1] = [B, N, L, 3]->[B, N, 3L]
        y_var = rearrange(torch.unsqueeze(covs_diag, dim=-2) * scales, #torch.unsqueeze(scales, dim=-1) ** 2,
                          pattern='batch sample scale_dim cov_dim -> batch sample (scale_dim cov_dim)')
    else:
        means, x_cov = means_covs
        num_dims = means.shape[-1]
        # [3, L]
        basis = torch.cat([2 ** i * torch.eye(num_dims, device=means.device) for i in range(min_deg, max_deg)], 1)
        y = torch.matmul(means, basis)  # [B, N, 3] * [3, 3L] = [B, N, 3L]
        y_var = torch.sum((torch.matmul(x_cov, basis)) * basis, -2)
    # sin(y + 0.5 * torch.tensor(np.pi)) = cos(y) 中国的学生脑子一定出现那句 “奇变偶不变 符号看象限”
    return expected_sin(torch.cat([y, y + 0.5 * torch.as_tensor(math.pi, device=y.device)], dim=-1), torch.cat([y_var] * 2, dim=-1))

@torch.jit.script
def diagonal_scale(min_deg: int, max_deg: int):
    return 2**torch.arange(min_deg, max_deg, device='cuda:0')[..., None]

@torch.jit.script
def fourier_feat(xb):
    return torch.sin(torch.cat([xb, xb + 0.5 * torch.as_tensor(math.pi, device=xb.device)], dim=-1))

#@torch.jit.script
def pos_enc(x, min_deg, max_deg, append_identity=True):
    """The positional encoding used by the original NeRF paper."""
    #scales = torch.tensor([2 ** i for i in range(min_deg, max_deg)], device=x.device)
    scales = diagonal_scale(min_deg, max_deg)
    # [B, 1, 3] * [L, 1] = [B, L, 3] -> [B, 3L]
    xb = rearrange(torch.unsqueeze(x, dim=-2) * scales, #torch.unsqueeze(scales, dim=-1),
                   pattern='batch scale_dim x_dim -> batch (scale_dim x_dim)')
    four_feat = fourier_feat(xb) #torch.sin(torch.cat([xb, xb + 0.5 *hardcoded_pi], dim=-1))  # [B, 2*3*L]
    if append_identity:
        return torch.cat([x] + [four_feat], dim=-1)  # [B, 2*3*L+3]
    else:
        return four_feat

@torch.jit.script
def compute_weights(density, t_samples, dirs):
    t_mids = 0.5 * (t_samples[..., :-1] + t_samples[..., 1:])
    t_interval = t_samples[..., 1:] - t_samples[..., :-1]  # [B, N]
    # models/mip.py:8 here sample point by multiply the interval with the direction without normalized, so
    # the delta is norm(t1*d-t2*d) = (t1-t2)*norm(d)
    delta = t_interval * torch.linalg.norm(torch.unsqueeze(dirs, dim=-2), dim=-1)
    # Note that we're quietly turning density from [..., 0] to [...].
    density_delta = density[..., 0] * delta

    alpha = 1 - torch.exp(-density_delta)
    trans = torch.exp(-torch.cat([
        torch.zeros_like(density_delta[..., :1]),
        torch.cumsum(density_delta[..., :-1], dim=-1)
    ], dim=-1))
    weights = alpha * trans  # [B, N]
    return weights, t_mids

#@torch.jit.script
def volumetric_rendering(rgb, density, t_samples, dirs, white_bkgd: bool):
    """Volumetric Rendering Function.
    Args:
        rgb: torch.Tensor, color, [batch_size, num_samples, 3]
        density: torch.Tensor, density, [batch_size, num_samples, 1].
        t_samples: torch.Tensor, [batch_size, num_samples+1].
        dirs: torch.Tensor, [batch_size, 3].
        white_bkgd: bool.
    Returns:
        comp_rgb: torch.Tensor, [batch_size, 3].
        disp: torch.Tensor, [batch_size].
        acc: torch.Tensor, [batch_size].
        weights: torch.Tensor, [batch_size, num_samples]
    """
    
    weights, t_mids = compute_weights(density, t_samples, dirs)
    comp_rgb = (torch.unsqueeze(weights, dim=-1) * rgb).sum(axis=-2)  # [B, N, 1] * [B, N, 3] -> [B, 3]
    acc = weights.sum(axis=-1)
    distance = (weights * t_mids).sum(axis=-1) 
    ## alternative way to compute distance (mean)
    #eps = torch.tensor(1e-7, device=dirs.device)   
    #expectation = lambda x: (weights * x).sum(axis=-1) / torch.maximum(eps, acc)
    #distance = expectation(t_mids)
    #distance = torch.exp(expectation(torch.log(t_mids)))
    #distance = torch.clamp(torch.nan_to_num(distance), t_samples[:, 0], t_samples[:, -1])
    #disp = jnp.where((disp > 0) & (disp < inv_eps) & (acc > eps), disp, inv_eps)
    #eps = torch.tensor(1e-5, device=comp_rgb.device)
    
    #distance = (torch.clip(torch.nan_to_num(torch.exp(expectation(torch.log(t_mids))), 1e5), t_samples[..., 0], t_samples[..., -1]))
    if white_bkgd:
        comp_rgb = comp_rgb + (1. - torch.unsqueeze(acc, dim=-1))
    return comp_rgb, distance, acc, weights


def rearrange_render_image(rays, chunk_size=4096):
    # change Rays to list: [origins, directions, viewdirs, radii, lossmult, near, far]
    #ignore_keys = ['depth', 'normal']
    single_image_rays = [getattr(rays, key) for key in Rays_keys]
    val_mask = single_image_rays[-5]

    # flatten each Rays attribute and put on device
    single_image_rays = [rays_attr.reshape(-1, rays_attr.shape[-1]) for rays_attr in single_image_rays]
    # get the amount of full rays of an image
    length = single_image_rays[0].shape[0]
    # divide each Rays attr into N groups according to chunk_size,
    # the length of the last group <= chunk_size
    single_image_rays = [[rays_attr[i:i + chunk_size] for i in range(0, length, chunk_size)] for
                         rays_attr in single_image_rays]
    # get N, the N for each Rays attr is the same
    length = len(single_image_rays[0])
    # generate N Rays instances
    single_image_rays = [Rays(*[rays_attr[i] for rays_attr in single_image_rays]) for i in range(length)]
    return single_image_rays, val_mask


def contract(x):
    # x: [N, 3]
    return (2 - 1 / (torch.norm(x, dim=-1, keepdim=True))) * x / torch.norm(x, dim=-1, keepdim=True)

#reparametrization trick to make it differentiable
def depth_guided_supersampling(rays, n_samples, start_sigma=0.1):
    #with torch.no_grad():
    near, far = rays.near, rays.far # both (N_rays, 1)
    depth = rays.depth
    #mean is depth unless depth is outside of frustrum, pick random point there
    mean = depth.clone().detach()
    #mean[depth == 0] = (torch.rand(1, device=depth.device) * (far-near))+near
    #mean[depth == 0] = (torch.rand(depth[depth == 0].shape, device=depth.device) * (far[depth == 0]-near[depth == 0])) + near[depth == 0]
    mean[depth == 0] = torch.rand(depth[depth == 0].shape, device=depth.device)
    mean[depth == 0] = mean[depth == 0] * (far[depth == 0] - near[depth == 0]) + near[depth == 0]
    #mean[depth == 0] = far[depth==0] - torch.ones_like(far[depth==0]).exponential_(lambd=1)
    # this 1.5 factor helps nerf gather more samples at the back of the ray, since the density is biased to the front
    #sigma
    sigma = torch.ones((1, n_samples//4), device=depth.device)*start_sigma
    normal = torch.normal(mean=torch.zeros_like(depth).expand(depth.shape[0], n_samples//4), std=1)
    zvals_0_level = normal * (1/2**0)*sigma + mean
    zvals_1_level = normal * (1/2**1)*sigma + mean
    zvals_2_level = normal * (1/2**2)*sigma + mean
    zvals_3_level = normal * (1/2**4)*sigma + mean
    zvals_ = torch.cat([zvals_0_level,zvals_1_level, zvals_2_level,zvals_3_level], -1)
    #zvals_ = torch.cat([zvals_0_level,zvals_1_level], -1)
    #zvals_ = zvals_3_level
    #zvals_ = torch.clip(zvals_, near, far)

    return zvals_

"""
def parameterization(means, covs):
    '''
    means: [B, N, 3]
    covs: [B, N, 3, 3]
    '''

    B, N, _ = means.shape
    means = means.reshape([-1, 3])
    if len(covs.shape) == 4:
        covs = covs.reshape(-1, 3, 3)
    else:
        covs = covs.reshape(-1, 3)
    contr_mask = (torch.norm(means, dim=-1, keepdim=True) > 1).detach()
    with torch.no_grad():
        jac = vmap(jacrev(contract))(means)
        print('11', jac.shape, covs.shape)
    means = torch.where(contr_mask, contract(means), means)
    covs = torch.where(contr_mask.unsqueeze(-1).expand(jac.shape), jac, covs)
    return means.reshape([B, N, 3]), covs.reshape([B, N, 3, 3])"""


if __name__ == '__main__':
    torch.manual_seed(0)
    batch_size = 4096
    origins = torch.rand([batch_size, 3]).to('cuda')
    directions = torch.rand(batch_size, 3).to('cuda')
    radii = torch.rand([batch_size, 1]).to('cuda')
    num_samples = 64
    near = torch.rand([batch_size, 1]).to('cuda')
    far = torch.rand([batch_size, 1]).to('cuda')
    viewdir = torch.rand([batch_size, 3]).to('cuda')
    randomized = True
    disparity = False
    ray_shape = 'cone'

    means = torch.rand(4096, 32, 3, requires_grad=True).to('cuda')
    covs = torch.rand(4096, 32, 3, requires_grad=True).to('cuda')
    min_degree = torch.as_tensor([0.], device='cuda')

    from torch.profiler import profile, record_function, ProfilerActivity
    with profile(activities=[
        ProfilerActivity.CPU, ProfilerActivity.CUDA], record_shapes=True) as prof:
        with record_function("model_inference"):
            #integrated_pos_enc((means, covs), 0, 16, True)
            pos_enc(directions, 0, 4, True)

            pass
    print(prof.key_averages().table(sort_by="cuda_time_total", row_limit=10))



def _xavier_init(linear):
    """
    Performs the Xavier weight initialization of the linear layer `linear`.
    """
    torch.nn.init.xavier_uniform_(linear.weight.data)


class MLP(torch.nn.Module):
    """
    A simple MLP.
    """

    def __init__(self, net_depth: int, net_width: int, net_depth_condition: int, net_width_condition: int,
                 skip_index: int, num_rgb_channels: int, num_density_channels: int, activation: str,
                 xyz_dim: int, view_dim: int):
        """
          net_depth: The depth of the first part of MLP.
          net_width: The width of the first part of MLP.
          net_depth_condition: The depth of the second part of MLP.
          net_width_condition: The width of the second part of MLP.
          activation: The activation function.
          skip_index: Add a skip connection to the output of every N layers.
          num_rgb_channels: The number of RGB channels.
          num_density_channels: The number of density channels.
        """
        super(MLP, self).__init__()
        self.skip_index: int = skip_index  # Add a skip connection to the output of every N layers.
        layers = []
        for i in range(net_depth):
            if i == 0:
                dim_in = xyz_dim
                dim_out = net_width
            elif (i - 1) % skip_index == 0 and i > 1:
                dim_in = net_width + xyz_dim
                dim_out = net_width
            else:
                dim_in = net_width
                dim_out = net_width
            linear = torch.nn.Linear(dim_in, dim_out)
            _xavier_init(linear)
            if activation == 'relu':
                layers.append(torch.nn.Sequential(linear, torch.nn.ReLU(True)))
            else:
                raise NotImplementedError
        self.layers = torch.nn.ModuleList(layers)
        del layers
        self.density_layer = torch.nn.Linear(net_width, num_density_channels)
        _xavier_init(self.density_layer)
        self.extra_layer = torch.nn.Linear(net_width, net_width)  # extra_layer is not the same as NeRF
        _xavier_init(self.extra_layer)
        layers = []
        for i in range(net_depth_condition):
            if i == 0:
                dim_in = net_width + view_dim
                dim_out = net_width_condition
            else:
                dim_in = net_width_condition
                dim_out = net_width_condition
            linear = torch.nn.Linear(dim_in, dim_out)
            _xavier_init(linear)
            if activation == 'relu':
                layers.append(torch.nn.Sequential(linear, torch.nn.ReLU(True)))
            else:
                raise NotImplementedError
        self.view_layers = torch.nn.Sequential(*layers)
        del layers
        self.color_layer = torch.nn.Linear(net_width_condition, num_rgb_channels)

    def forward(self, x, view_direction=None):
        """Evaluate the MLP.

        Args:
            x: torch.Tensor(float32), [batch, num_samples, feature], points.
            view_direction: torch.Tensor(float32), [batch, feature], if not None, this
            variable will be part of the input to the second part of the MLP
            concatenated with the output vector of the first part of the MLP. If
            None, only the first part of the MLP will be used with input x. In the
            original paper, this variable is the view direction.

        Returns:
            raw_rgb: torch.Tensor(float32), with a shape of
                [batch, num_samples, num_rgb_channels].
            raw_density: torch.Tensor(float32), with a shape of
                [batch, num_samples, num_density_channels].
        """
        num_samples = x.shape[1]
        inputs = x  # [B, N, 2*3*L]
        for i, layer in enumerate(self.layers):
            x = layer(x)
            if i % self.skip_index == 0 and i > 0:
                x = torch.cat([x, inputs], dim=-1)
        raw_density = self.density_layer(x)
        if view_direction is not None:
            # Output of the first part of MLP.
            bottleneck = self.extra_layer(x)
            # Broadcast condition from [batch, feature] to
            # [batch, num_samples, feature] since all the samples along the same ray
            # have the same viewdir.
            # view_direction: [B, 2*3*L] -> [B, N, 2*3*L]
            view_direction = repeat(view_direction, 'batch feature -> batch sample feature', sample=num_samples)
            x = torch.cat([bottleneck, view_direction], dim=-1)
            # Here use 1 extra layer to align with the original nerf model.
            x = self.view_layers(x)
        raw_rgb = self.color_layer(x)
        return raw_rgb, raw_density


class MipNerf(torch.nn.Module):
    """Nerf NN Model with both coarse and fine MLPs."""

    def __init__(self, num_samples: int = 128,
                 num_levels: int = 2,
                 resample_padding: float = 0.01,
                 stop_resample_grad: bool = True,
                 use_viewdirs: bool = True,
                 disparity: bool = False,
                 depth_sampling: bool = False,
                 ray_shape: str = 'cone',
                 min_deg_point: int = 0,
                 max_deg_point: int = 16,
                 deg_view: int = 4,
                 density_activation: str = 'softplus',
                 density_noise: float = 0.,
                 density_bias: float = -1.,
                 rgb_activation: str = 'sigmoid',
                 rgb_padding: float = 0.001,
                 disable_integration: bool = False,
                 append_identity: bool = True,
                 mlp_net_depth: int = 8,
                 mlp_net_width: int = 256,
                 mlp_net_depth_condition: int = 1,
                 mlp_net_width_condition: int = 128,
                 mlp_skip_index: int = 4,
                 mlp_num_rgb_channels: int = 3,
                 mlp_num_density_channels: int = 1,
                 mlp_net_activation: str = 'relu'):
        super(MipNerf, self).__init__()
        self.num_levels = num_levels  # The number of sampling levels.
        self.num_samples = num_samples  # The number of samples per level.
        self.disparity = disparity  # If True, sample linearly in disparity, not in depth.
        self.depth_sampling = depth_sampling #If True, sample ray more densily around depth
        self.ray_shape = ray_shape  # The shape of cast rays ('cone' or 'cylinder').
        self.disable_integration = disable_integration  # If True, use PE instead of IPE.
        self.min_deg_point = min_deg_point  # Min degree of positional encoding for 3D points.
        self.max_deg_point = max_deg_point  # Max degree of positional encoding for 3D points.
        self.use_viewdirs = use_viewdirs  # If True, use view directions as a condition.
        self.deg_view = deg_view  # Degree of positional encoding for viewdirs.
        self.density_noise = density_noise  # Standard deviation of noise added to raw density.
        self.density_bias = density_bias  # The shift added to raw densities pre-activation.
        self.resample_padding = resample_padding  # Dirichlet/alpha "padding" on the histogram.
        self.stop_resample_grad = stop_resample_grad  # If True, don't backprop across levels')
        mlp_xyz_dim = (max_deg_point - min_deg_point) * 3 * 2
        mlp_view_dim = deg_view * 3 * 2
        mlp_view_dim = mlp_view_dim + 3 if append_identity else mlp_view_dim
        self.mlp = MLP(mlp_net_depth, mlp_net_width, mlp_net_depth_condition, mlp_net_width_condition,
                       mlp_skip_index, mlp_num_rgb_channels, mlp_num_density_channels, mlp_net_activation,
                       mlp_xyz_dim, mlp_view_dim)
        if rgb_activation == 'sigmoid':  # The RGB activation.
            self.rgb_activation = torch.nn.Sigmoid()
        else:
            raise NotImplementedError
        self.rgb_padding = rgb_padding
        if density_activation == 'softplus':  # Density activation.
            self.density_activation = torch.nn.Softplus()
        else:
            raise NotImplementedError
        self.chunksize = 1024
        self.depth_network = DepthVarianceNetwork(init_val = 2.30258509) #0.1 variance

    def forward(self, rays: namedtuple, randomized: bool, white_bkgd: bool, compute_normals: bool = False, eps = 1.0):
        """The mip-NeRF Model.
        Args:
            rays: util.Rays, a namedtuple of ray origins, directions, and viewdirs.
            randomized: bool, use randomized stratified sampling.
            white_bkgd: bool, if True, use white as the background (black o.w.).
        Returns:
            ret: list, [*(rgb, distance, acc)]
        """
        batch_size = rays[0].shape[0]
        ret = torch.empty((2,batch_size, 3), device=rays[0].device)
        t_samples, weights = None, None
        for i_level in range(self.num_levels):
            # key, rng = random.split(rng)
            if i_level == 0:
                # Stratified sampling along rays
                if self.depth_sampling: s = self.depth_network(torch.zeros([1, 3], device=rays[0].device))[:, :1].clip(1e-6, 1e6)
                else: s = None
                t_samples, means_covs = sample_along_rays(
                    rays.origins,
                    rays.directions,
                    rays.radii,
                    self.num_samples,
                    rays.near,
                    rays.far,
                    randomized,
                    self.disparity,
                    self.ray_shape,
                    rays,
                    s
                )
            else:
                t_samples, means_covs = resample_along_rays(
                    rays.origins,
                    rays.directions,
                    rays.radii,
                    t_samples,
                    weights,
                    randomized,
                    self.ray_shape,
                    self.stop_resample_grad,
                    resample_padding=self.resample_padding,
                )
                # t_samples: (BS, n_samples+1); mean_covs: (tuple: ((1024, 128, 3), (1024, 128, 3)))
            if self.disable_integration:
                means_covs = (means_covs[0], torch.zeros_like(means_covs[1]))
        
            if (not compute_normals) or (i_level !=1):
                samples_enc = integrated_pos_enc(
                    means_covs,
                    self.min_deg_point,
                    self.max_deg_point,
                )  # samples_enc: [B, N, 2*3*L]  L:(max_deg_point - min_deg_point)

                # Point attribute predictions
                if self.use_viewdirs:
                    viewdirs_enc = pos_enc(
                        rays.viewdirs,
                        min_deg=0,
                        max_deg=self.deg_view,
                        append_identity=True,
                    )

                    raw_rgb, raw_density = self.mlp(samples_enc, viewdirs_enc)
                else:
                    raw_rgb, raw_density = self.mlp(samples_enc)
            
            elif compute_normals and i_level == 1:
                raw_rgb, raw_density, normals = self.gradient(means_covs, rays.viewdirs)

            # Add noise to regularize the density predictions if needed.
            if randomized and (self.density_noise > 0):
                raw_density += self.density_noise * torch.randn(raw_density.shape, dtype=raw_density.dtype)

            # Volumetric rendering.
            rgb = self.rgb_activation(raw_rgb)  # [B, N, 3]
            rgb = rgb * (1 + 2 * self.rgb_padding) - self.rgb_padding
            density = self.density_activation(raw_density + self.density_bias)  # [B, N, 1]
            comp_rgb, depth, acc, weights = volumetric_rendering(
                rgb,
                density,
                t_samples,
                rays.directions,
                white_bkgd=white_bkgd,
            )
            #ret.append((comp_rgb, distance, acc, weights, t_samples))
            #ret = {'rgb': comp_rgb, 'depth': distance, 'acc':acc, 'weights': weights, 'samples': t_samples} #coarse & fine
            ret[i_level] = comp_rgb
            

        if not compute_normals: normals = torch.zeros((batch_size, self.num_samples, 3), device=rays[0].device)
        if not self.depth_sampling: s = torch.as_tensor([0.] , device=rays[0].device)
        weighted_depthloss = lossfun_depth_weight(rays.depth, t_samples, weights, eps)
        out = {'rgb_coarse': ret[0], 'rgb_fine': ret[1], 'depth': depth, 'normal': normals, 'weights': weights, 's': s, 
                            'distortion': lossfun_distortion(t_samples, weights), 'weighted_depthloss': weighted_depthloss}
        return out

    def gradient(self, means_covs, viewdirs):
        means_covs[0].requires_grad_()
        means_covs[1].requires_grad_()
        GraphBools = self.training           
        with torch.enable_grad():
            samples_enc = integrated_pos_enc(
                means_covs,
                self.min_deg_point,
                self.max_deg_point,
            )
            viewdirs_enc = pos_enc(
                    viewdirs,
                    min_deg=0,
                    max_deg=self.deg_view,
                    append_identity=True,
                ) 
            raw_rgb, raw_density = self.mlp(samples_enc, viewdirs_enc)
            d_output = torch.ones_like(raw_density, requires_grad=False, device=raw_density.device)
            normals = torch.autograd.grad(
                outputs=raw_density,
                inputs=means_covs,
                grad_outputs=d_output,
                create_graph=GraphBools,
                retain_graph=GraphBools,
                only_inputs=True)[0]
            
            normals = -l2_normalize(torch.nan_to_num(normals)) 

        return raw_rgb, raw_density, normals


    def chunked_inference(self, i_level, rays, means_covs, chunksize=2048, compute_normals: bool = False):
        B = means_covs[0].shape[0]
        out_chunks = []
        for i in range(0, B, chunksize):
            means_covs_ = means_covs[0][i:i+chunksize], means_covs[1][i:i+chunksize]
            if (not compute_normals) or (i_level !=1):
                samples_enc = integrated_pos_enc(
                    means_covs,
                    self.min_deg_point,
                    self.max_deg_point,
                )  # samples_enc: [B, N, 2*3*L]  L:(max_deg_point - min_deg_point)

                # Point attribute predictions
                if self.use_viewdirs:
                    viewdirs_enc = pos_enc(
                        rays.viewdirs,
                        min_deg=0,
                        max_deg=self.deg_view,
                        append_identity=True,
                    )

                    raw_rgb, raw_density = self.mlp(samples_enc, viewdirs_enc)
                    #out_chunks += [self.mlp(samples_enc, viewdirs_enc)]
                else:
                    raw_rgb, raw_density = self.mlp(samples_enc)
            
            elif compute_normals and i_level == 1:
                raw_rgb, raw_density, normals = self.gradient(means_covs, rays.viewdirs)
                
            raw_rgb, raw_density = zip(*out_chunks)
            raw_rgb, raw_density = torch.cat(raw_rgb, 0), torch.cat(raw_density, 0)
        return raw_rgb, raw_density


class DepthVarianceNetwork(nn.Module):
    def __init__(self, init_val):
        super(DepthVarianceNetwork, self).__init__()
        self.register_parameter('variance', nn.Parameter(torch.tensor(init_val)))

    def forward(self, x):
        return torch.ones([len(x), 1], device=x.device) * torch.exp(-self.variance.to(x.device))

def lossfun_distortion(t, w):   #t=z_vals, w=weights. Loss from mip-nerf 360
    """Compute iint w[i] w[j] |t[i] - t[j]| di dj."""
    # The loss incurred between all pairs of intervals.
    ut = (t[..., 1:] + t[..., :-1]) / 2
    dut = torch.abs(ut[..., :, None] - ut[..., None, :])
    loss_inter = torch.sum(w * torch.sum(w[..., None, :] * dut, axis=-1), axis=-1)

    # The loss incurred within each individual interval with itself.
    loss_intra = torch.sum(w**2 * (t[..., 1:] - t[..., :-1]), axis=-1) / 3

    return loss_inter + loss_intra

def lossfun_depth_weight(depth: torch.Tensor, tvals_, w, eps):
    """Penalize sum of weights for empty interval
       Penalize squared distance from depth for near inteval"""
    dummy_1 = torch.as_tensor([1.0], device = depth.device)
    tvals = tvals_[..., :-1]
    depth_t = depth.broadcast_to(tvals.shape)
    sigma = (eps / 3.) ** 2
    mask_near = ((tvals > (depth - eps)) & (tvals < (depth + eps))).to(depth.dtype).reshape(tvals.shape[0], -1)
    mask_empty = (tvals < (depth - eps)).to(depth.dtype).reshape(tvals.shape[0], -1)
    dist = mask_near * (tvals - depth_t)
    dist = 1.0 / (sigma * math.sqrt(2 * math.pi)) * torch.exp(-(dist ** 2 / (2 * sigma ** 2)))
    dist = (dist/ dist.max()) * mask_near
    near_losses = 0.1 * (((mask_near * w - dist) ** 2).sum() / torch.maximum(mask_near.sum(), dummy_1))
    empty_losses = 1.0 * (((mask_empty * w) ** 2).sum() / torch.maximum(mask_empty.sum(), dummy_1))
    return near_losses + empty_losses
    

    """
    depth_t = jnp.broadcast_to(batch['depth'], tvals.shape)
    sigma = (eps / 3.) ** 2
    t_from_ndc = 1.0 / (1.0 - tvals)
    mask_near = ((tvals > (depth_t - eps)) & (tvals < (depth_t + eps))).astype(jnp.float32)
    mask_near *= depth_mask.reshape(tvals.shape[0], -1)
    mask_empty = (tvals > (depth_t + eps)).astype(jnp.float32)
    mask_empty *= depth_mask.reshape(tvals.shape[0], -1)
    dist = mask_near * (tvals - depth_t)
    distr = 1.0 / (sigma * jnp.sqrt(2 * jnp.pi)) * jnp.exp(-(dist ** 2 / (2 * sigma ** 2)))
    distr /= distr.max()
    distr *= mask_near
    n_losses.append(((mask_near * weights - distr) ** 2).sum() / jnp.maximum(depth_mask.sum(), 1.0))
    e_losses.append(((mask_empty * weights) ** 2).sum() / jnp.maximum(depth_mask.sum(), 1.0))
    # seg_losses.append(((sky_mask.reshape(weights.shape[0], -1) * weights)**2).sum() / jnp.maximum(sky_mask.sum(), 1.0))
    # seg_losses.append((sky_mask * depth).mean())
    # z_from_ndc = depth_mask * (1.0 / (1.0 - depth_mask * depth))
    # print(jnp.max(z_from_ndc))
    # print(jnp.max(depth))
    # inv_depth = depth_mask * (1.0 / jnp.maximum(batch['depth'].squeeze(), 1.0))
    d_losses.append(
    (depth_mask * (depth - batch['depth'].squeeze()) ** 2).sum() / jnp.maximum(depth_mask.sum(), 1.0))
    """

# Utility to profile the workload
def profile_workload(forward_func, grad_output, iteration_count=100, label="", parameters = None):
    # Perform warm-up iterations
    for _ in range(3):
        # Run model, forward and backward
        output = forward_func()
        output.backward(grad_output)
        # delete gradiens to avoid profiling the gradient accumulation
        #for p in parameters:
        #    p.grad = None

    # Synchronize the GPU before starting the timer
    torch.cuda.synchronize()
    start = time.perf_counter()
    for _ in range(iteration_count):
        # Run model, forward and backward
        output = forward_func()
        output.backward(grad_output)
        # delete gradiens to avoid profiling the gradient accumulation
        #for p in parameters:
        #    p.grad = None

    # Synchronize the GPU before stopping the timer
    torch.cuda.synchronize()
    stop = time.perf_counter()
    iters_per_second = iteration_count / (stop - start)
    if label:
        print(label)
    print("Average iterations per second: {:.2f}".format(iters_per_second))

if __name__ == '__main__':
    import collections
    Rays = collections.namedtuple(
    'Rays',
    ('origins', 'directions', 'viewdirs', 'radii', 'lossmult', 'near', 'far', 'depth', 'normal', 'mask'))
    # 
    """ 'Rays', ('origins', 'directions', 'viewdirs', 'radii', 'lossmult', 'near', 'far', 'depth', 'normal', 'mask'))"""
    torch.manual_seed(0)
    batch_size = 4096
    origins = torch.rand([batch_size, 3]).to('cuda')
    directions = torch.rand(batch_size, 3).to('cuda')
    radii = torch.rand([batch_size, 1]).to('cuda')
    num_samples = 64
    near = torch.rand([batch_size, 1]).to('cuda')
    far = torch.rand([batch_size, 1]).to('cuda')
    viewdir = torch.rand([batch_size, 3]).to('cuda')
    normal = torch.rand([batch_size, 3]).to('cuda')
    lossmult = torch.tensor([0.]).to('cuda')
    depth = torch.rand([batch_size, 1]).to('cuda')
    mask = (depth > 0.5).to(torch.float).to('cuda')
    randomized = True
    disparity = False
    ray_shape = 'cone'
    rays = Rays(origins, directions, origins, radii, lossmult, near, far, depth, normal, mask)

    means = torch.rand(4096, 32, 3, requires_grad=True).to('cuda')
    covs = torch.rand(4096, 32, 3, requires_grad=True).to('cuda')
    min_degree = torch.as_tensor([0.], device='cuda')

    """from torch.profiler import profile, record_function, ProfilerActivity
    with profile(activities=[
        ProfilerActivity.CPU, ProfilerActivity.CUDA], record_shapes=True) as prof:
        with record_function("model_inference"):
            #integrated_pos_enc((means, covs), 0, 16, True)
            sample_along_rays(origins, directions, radii, num_samples, near, far, randomized, disparity, ray_shape, rays, None)
            
            pass
    print(prof.key_averages().table(sort_by="cuda_time_total", row_limit=10))"""
    profile_workload