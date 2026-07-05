"""Lightweight WOOD-style geometry blocks for Stage-A gradient sanity only."""
from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Any

import torch
import torch.nn.functional as F


@dataclass
class GradientGeometryConfig:
    seed: int = 1234
    init: str = "small_random"
    init_fraction: float = 0.05
    tps_size: int = 5
    delaunay_size: int = 5
    dct_size: int = 4
    fft_phase_size: int = 8
    edge_falloff_px: float = 16.0
    tps_px_limit: float = 3.6
    delaunay_px_limit: float = 5.1
    rolling_px_limit: float = 4.6
    dct_px_limit: float = 4.1
    fft_phase_limit_rad: float = math.pi
    max_combined_disp_px: float = 8.0
    tps_enabled: bool = True
    delaunay_enabled: bool = True
    rolling_enabled: bool = True
    dct_enabled: bool = True
    fft_phase_enabled: bool = True


def dct_basis(size: int, height: int, width: int, device: torch.device) -> torch.Tensor:
    yy = torch.arange(height, device=device).float()
    xx = torch.arange(width, device=device).float()
    basis = []
    for uy in range(size):
        for ux in range(size):
            by = torch.cos(math.pi * (yy + 0.5) * uy / height)[:, None]
            bx = torch.cos(math.pi * (xx + 0.5) * ux / width)[None, :]
            b = by * bx
            b = b / (b.abs().max().clamp_min(1e-8))
            basis.append(b)
    return torch.stack(basis, dim=0)


def flow_stats(field: torch.Tensor, prefix: str) -> dict[str, float]:
    mag = torch.sqrt(field.detach().float().square().sum(dim=1) + 1e-12)
    suffix = "_disp_px" if prefix == "combined" else "_disp"
    return {
        f"{prefix}_mean{suffix}": float(mag.mean().cpu()),
        f"{prefix}_max{suffix}": float(mag.max().cpu()),
        f"{prefix}_p95{suffix}": float(torch.quantile(mag.flatten(), 0.95).cpu()),
    }


def smoothness_tv(field: torch.Tensor) -> torch.Tensor:
    return (field[:, :, :, 1:] - field[:, :, :, :-1]).abs().mean() + (
        field[:, :, 1:] - field[:, :, :-1]
    ).abs().mean()


def jacobian_diagnostics(field: torch.Tensor) -> dict[str, float]:
    dx, dy = field[:, 0], field[:, 1]
    ddx = F.pad((dx[:, :, 2:] - dx[:, :, :-2]) / 2.0, (1, 1))
    dxy = F.pad((dx[:, 2:] - dx[:, :-2]) / 2.0, (0, 0, 1, 1))
    dyx = F.pad((dy[:, :, 2:] - dy[:, :, :-2]) / 2.0, (1, 1))
    ddy = F.pad((dy[:, 2:] - dy[:, :-2]) / 2.0, (0, 0, 1, 1))
    det = (1.0 + ddx) * (1.0 + ddy) - dxy * dyx
    return {
        "jacobian_det_min": float(det.detach().float().min().cpu()),
        "foldover_fraction": float((det.detach().float() < 0).float().mean().cpu()),
        "smoothness_tv": float(smoothness_tv(field.detach().float()).cpu()),
    }


class GradientProbeGeometry(torch.nn.Module):
    """Small differentiable geometry module for A7 gradient sanity.

    This is intentionally a short-probe module, not the full Stage-B attack.
    It exposes the same component names used in WOOD so we can measure whether
    candidate layers send useful gradients into each geometry family.
    """

    def __init__(self, height: int, width: int, channels: int, device: torch.device, config: GradientGeometryConfig) -> None:
        super().__init__()
        self.height = int(height)
        self.width = int(width)
        self.channels = int(channels)
        self.config = config
        generator = torch.Generator(device=device).manual_seed(config.seed)
        yy, xx = torch.meshgrid(
            torch.linspace(-1, 1, height, device=device),
            torch.linspace(-1, 1, width, device=device),
            indexing="ij",
        )
        self.register_buffer("base_grid", torch.stack([xx, yy], dim=-1)[None])
        self.register_buffer("yy_norm", yy[None, None])
        self.register_buffer("dct_basis", dct_basis(config.dct_size, height, width, device))
        distances = torch.minimum(
            torch.minimum(torch.arange(width, device=device)[None], torch.arange(width - 1, -1, -1, device=device)[None]),
            torch.minimum(torch.arange(height, device=device)[:, None], torch.arange(height - 1, -1, -1, device=device)[:, None]),
        ).float()
        t = (distances / max(float(config.edge_falloff_px), 1.0)).clamp(0, 1)
        self.register_buffer("edge", (t * t * (3 - 2 * t))[None, None])

        def init(shape: tuple[int, ...], limit: float) -> torch.Tensor:
            if config.init == "neutral":
                return torch.zeros(*shape, device=device)
            return torch.randn(*shape, generator=generator, device=device) * (limit * config.init_fraction)

        self.tps_raw = torch.nn.Parameter(init((1, 2, config.tps_size, config.tps_size), config.tps_px_limit))
        self.delaunay_raw = torch.nn.Parameter(init((1, 2, config.delaunay_size, config.delaunay_size), config.delaunay_px_limit))
        self.rolling_params = torch.nn.Parameter(init((2,), config.rolling_px_limit))
        self.dct_coeffs = torch.nn.Parameter(init((2, self.dct_basis.shape[0]), config.dct_px_limit))
        self.fft_phase = torch.nn.Parameter(init((channels, config.fft_phase_size, config.fft_phase_size), config.fft_phase_limit_rad))
        self.project_()

    def _zero(self) -> torch.Tensor:
        return self.base_grid.new_zeros((1, 2, self.height, self.width))

    def _upsample_ctrl(self, ctrl: torch.Tensor, limit: float, enabled: bool, mode: str) -> torch.Tensor:
        if not enabled:
            return self._zero()
        field = F.interpolate(ctrl.clamp(-limit, limit), size=(self.height, self.width), mode=mode, align_corners=True)
        return field * self.edge

    def tps_field(self) -> torch.Tensor:
        return self._upsample_ctrl(self.tps_raw, self.config.tps_px_limit, self.config.tps_enabled, "bicubic")

    def delaunay_field(self) -> torch.Tensor:
        return self._upsample_ctrl(self.delaunay_raw, self.config.delaunay_px_limit, self.config.delaunay_enabled, "bilinear")

    def rolling_field(self) -> torch.Tensor:
        if not self.config.rolling_enabled:
            return self._zero()
        params = self.rolling_params.clamp(-self.config.rolling_px_limit, self.config.rolling_px_limit)
        x = params[0] * self.yy_norm + params[1] * torch.sin(math.pi * self.yy_norm)
        y = torch.zeros_like(x)
        return torch.cat([x, y], dim=1).expand(1, 2, self.height, self.width) * self.edge

    def dct_field(self) -> torch.Tensor:
        if not self.config.dct_enabled:
            return self._zero()
        coeffs = self.dct_coeffs.clamp(-self.config.dct_px_limit, self.config.dct_px_limit)
        return torch.einsum("ck,khw->chw", coeffs, self.dct_basis)[None] * self.edge

    def fft_stage(self, image: torch.Tensor) -> tuple[torch.Tensor, dict[str, float]]:
        if not self.config.fft_phase_enabled:
            return image, {
                "fft_phase_norm": 0.0,
                "fft_phase_max": 0.0,
                "fft_phase_mean_abs": 0.0,
                "legacy_fft_strength_equivalent": 0.0,
                "fft_spatial_delta_mse": 0.0,
            }
        phase = self.fft_phase.clamp(-self.config.fft_phase_limit_rad, self.config.fft_phase_limit_rad)
        phase = F.interpolate(
            phase[None],
            size=(self.height, self.width),
            mode="bilinear",
            align_corners=True,
        )[0]
        freq = torch.fft.fft2(image.float(), dim=(-2, -1))
        shifted = freq * torch.exp(1j * phase[None])
        out = torch.fft.ifft2(shifted, dim=(-2, -1)).real.clamp(0, 1).to(dtype=image.dtype)
        delta = out.float() - image.float()
        return out, {
            "fft_phase_norm": float(phase.detach().float().norm().cpu()),
            "fft_phase_max": float(phase.detach().float().abs().max().cpu()),
            "fft_phase_mean_abs": float(phase.detach().float().abs().mean().cpu()),
            "legacy_fft_strength_equivalent": float(phase.detach().float().abs().mean().cpu() / math.pi * 1_000_000.0),
            "fft_spatial_delta_mse": float((delta.square().mean()).detach().cpu()),
        }

    def spatial_warp(self, image: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor, dict[str, torch.Tensor]]:
        fields = {
            "tps": self.tps_field(),
            "delaunay": self.delaunay_field(),
            "rolling": self.rolling_field(),
            "dct": self.dct_field(),
        }
        displacement = sum(fields.values())
        if self.config.max_combined_disp_px and self.config.max_combined_disp_px > 0:
            mag = torch.sqrt(displacement.square().sum(dim=1, keepdim=True) + 1e-12)
            displacement = displacement * torch.clamp(float(self.config.max_combined_disp_px) / mag.clamp_min(1e-6), max=1.0)
        grid = self.base_grid.clone()
        grid[..., 0] += 2.0 * displacement[:, 0] / max(self.width - 1, 1)
        grid[..., 1] += 2.0 * displacement[:, 1] / max(self.height - 1, 1)
        warped = F.grid_sample(image, grid, mode="bilinear", padding_mode="reflection", align_corners=True).clamp(0, 1)
        return warped, displacement, fields

    def forward(self, image: torch.Tensor) -> tuple[torch.Tensor, dict[str, Any]]:
        spatial, displacement, fields = self.spatial_warp(image)
        perturbed, fft_stats = self.fft_stage(spatial)
        diag = self.diagnostics(displacement, fields)
        diag.update(fft_stats)
        return perturbed, {"displacement": displacement, "fields": fields, "diagnostics": diag}

    def diagnostics(self, displacement: torch.Tensor, fields: dict[str, torch.Tensor]) -> dict[str, float]:
        out: dict[str, float] = {}
        out.update(flow_stats(displacement, "combined"))
        out.update(jacobian_diagnostics(displacement))
        for name, field in fields.items():
            out.update(flow_stats(field, name))
        return out

    def grad_norms(self) -> dict[str, float]:
        def norm(param: torch.Tensor) -> float:
            if param.grad is None:
                return 0.0
            return float(param.grad.detach().float().norm().cpu())

        values = {
            "tps_grad_norm": norm(self.tps_raw),
            "delaunay_grad_norm": norm(self.delaunay_raw),
            "rolling_grad_norm": norm(self.rolling_params),
            "dct_grad_norm": norm(self.dct_coeffs),
            "fft_grad_norm": norm(self.fft_phase),
        }
        values["total_grad_norm"] = float(sum(v * v for v in values.values()) ** 0.5)
        return values

    def project_(self) -> dict[str, Any]:
        with torch.no_grad():
            specs = [
                ("tps", self.tps_raw, self.config.tps_px_limit, self.config.tps_enabled),
                ("delaunay", self.delaunay_raw, self.config.delaunay_px_limit, self.config.delaunay_enabled),
                ("rolling", self.rolling_params, self.config.rolling_px_limit, self.config.rolling_enabled),
                ("dct", self.dct_coeffs, self.config.dct_px_limit, self.config.dct_enabled),
                ("fft", self.fft_phase, self.config.fft_phase_limit_rad, self.config.fft_phase_enabled),
            ]
            total = clamped = at_min = at_max = 0
            boundary = []
            for name, param, limit, enabled in specs:
                param.nan_to_num_(0.0)
                if not enabled:
                    param.zero_()
                    continue
                total += param.numel()
                clamped += int(((param < -limit) | (param > limit)).sum().cpu())
                param.clamp_(-limit, limit)
                nmin = int((param <= -limit + 1e-8).sum().cpu())
                nmax = int((param >= limit - 1e-8).sum().cpu())
                at_min += nmin
                at_max += nmax
                if nmin or nmax:
                    boundary.append(name)
            return {
                "num_total_params": int(total),
                "num_clamped_total": int(clamped),
                "fraction_clamped_total": float(clamped / max(total, 1)),
                "num_at_min_total": int(at_min),
                "num_at_max_total": int(at_max),
                "components_at_boundary": ",".join(sorted(set(boundary))),
            }
