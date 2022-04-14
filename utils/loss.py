import torch
from typing import Tuple

def calc_mse(x: torch.Tensor, y: torch.Tensor):
    """
    Calculates the mean square error between tensors `x` and `y`.
    """
    return torch.mean((x - y) ** 2)


def calc_psnr(x: torch.Tensor, y: torch.Tensor):
    """
    Calculates the Peak-signal-to-noise ratio between tensors `x` and `y`.
    """
    mse = calc_mse(x, y)
    psnr = -10.0 * torch.log10(mse)
    return psnr

def accum_loss(ret:Tuple, gt_rgb:torch.Tensor, loss_weight:torch.Tensor):
    losses = []
    psnrs = []
    for (rgb, _, _) in ret:
        losses.append(
            (loss_weight * (rgb - gt_rgb[..., :3]) ** 2).sum() / loss_weight.sum())
        psnrs.append(calc_psnr(rgb, gt_rgb[..., :3]))
    # The loss is a sum of coarse and fine MSEs
    # mse_corse, mse_fine = losses
    # psnr_corse, psnr_fine = psnrs
    return losses, psnrs