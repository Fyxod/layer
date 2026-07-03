"""Activation selection, pooling, and statistics."""
from __future__ import annotations

from typing import Any

import torch
import torch.nn.functional as F


def first_tensor(value: Any) -> torch.Tensor | None:
    if torch.is_tensor(value):
        return value
    if isinstance(value, dict):
        for item in value.values():
            found = first_tensor(item)
            if found is not None:
                return found
    if isinstance(value, (tuple, list)):
        for item in value:
            found = first_tensor(item)
            if found is not None:
                return found
    return None


def pool_activation(tensor: torch.Tensor) -> torch.Tensor:
    """Pool an activation into a [B, C] style embedding and L2-normalize."""

    x = tensor.float()
    if x.ndim == 4:
        pooled = x.mean(dim=(-2, -1))
    elif x.ndim == 3:
        pooled = x.mean(dim=1)
    elif x.ndim == 2:
        pooled = x
    elif x.ndim > 4:
        pooled = x.flatten(start_dim=2).mean(dim=-1)
    elif x.ndim == 1:
        pooled = x[None]
    else:
        pooled = x.reshape(x.shape[0], -1)
    return F.normalize(pooled, p=2, dim=1, eps=1e-8)


def activation_stats(tensor: torch.Tensor) -> dict[str, float]:
    x = tensor.detach().float()
    if x.numel() == 0:
        return {
            "activation_norm": 0.0,
            "activation_mean": 0.0,
            "activation_std": 0.0,
            "activation_sparsity": 0.0,
            "activation_spatial_variance": 0.0,
        }
    spatial_variance = 0.0
    if x.ndim == 4:
        spatial_variance = float(x.var(dim=(-2, -1), unbiased=False).mean().cpu())
    elif x.ndim == 3:
        spatial_variance = float(x.var(dim=1, unbiased=False).mean().cpu())
    return {
        "activation_norm": float(x.norm().cpu()),
        "activation_mean": float(x.mean().cpu()),
        "activation_std": float(x.std(unbiased=False).cpu()),
        "activation_sparsity": float((x.abs() < 1e-6).float().mean().cpu()),
        "activation_spatial_variance": spatial_variance,
    }
