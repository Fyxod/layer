"""Small image metrics used by gradient sanity probes."""
from __future__ import annotations

import math

import torch
import torch.nn.functional as F


def mse(x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
    return F.mse_loss(x.float(), y.float())


def psnr(x: torch.Tensor, y: torch.Tensor) -> float:
    value = float(mse(x, y).detach().cpu())
    if value <= 1e-12:
        return float("inf")
    return float(10.0 * math.log10(1.0 / value))


def ssim_global(x: torch.Tensor, y: torch.Tensor) -> float:
    """A cheap global SSIM-style diagnostic, not a full sliding-window SSIM."""

    x = x.float()
    y = y.float()
    c1 = 0.01**2
    c2 = 0.03**2
    mux = x.mean(dim=(-2, -1), keepdim=True)
    muy = y.mean(dim=(-2, -1), keepdim=True)
    vx = ((x - mux) ** 2).mean(dim=(-2, -1), keepdim=True)
    vy = ((y - muy) ** 2).mean(dim=(-2, -1), keepdim=True)
    cov = ((x - mux) * (y - muy)).mean(dim=(-2, -1), keepdim=True)
    score = ((2 * mux * muy + c1) * (2 * cov + c2)) / ((mux**2 + muy**2 + c1) * (vx + vy + c2))
    return float(score.mean().detach().cpu())
