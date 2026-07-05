"""Phase A7 gradient sanity scan for selected identity-sensitive layers."""
from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import torch
import torch.nn.functional as F

from .activation_hooks import resolve_module
from .cases import slugify
from .gradient_geometry import GradientGeometryConfig, GradientProbeGeometry
from .instruct_backend import InstructLayerBackend
from .io import read_csv, write_csv, write_json
from .metrics import mse, psnr, ssim_global
from .pooling import first_tensor, pool_activation


def parse_geometry_config(payload: dict[str, Any] | None) -> GradientGeometryConfig:
    if not payload:
        return GradientGeometryConfig()
    allowed = set(GradientGeometryConfig.__dataclass_fields__)
    values = {key: value for key, value in payload.items() if key in allowed}
    for block_name in ("limits", "sizes", "components"):
        block = payload.get(block_name, {})
        if isinstance(block, dict):
            for key, value in block.items():
                if key in allowed:
                    values[key] = value
    components = payload.get("components", {})
    for name in ["tps", "delaunay", "rolling", "dct", "fft_phase"]:
        block = components.get(name, {}) if isinstance(components, dict) else {}
        enabled_key = f"{name}_enabled"
        if "enabled" in block and enabled_key in allowed:
            values[enabled_key] = bool(block["enabled"])
    return GradientGeometryConfig(**values)


def capture_activation_tensor(
    backend: InstructLayerBackend,
    image_tensor: torch.Tensor,
    prompt: str,
    timestep_index: int,
    layer_name: str,
) -> torch.Tensor:
    if layer_name == "vae_image_latent":
        return backend.encode_image_latent(image_tensor)
    module = resolve_module(backend.pipe, layer_name)
    captured: dict[str, torch.Tensor] = {}

    def hook(_module: torch.nn.Module, _inputs: Any, output: Any) -> None:
        tensor = first_tensor(output)
        if tensor is None:
            raise RuntimeError(f"No tensor output captured for {layer_name}")
        captured["tensor"] = tensor

    handle = module.register_forward_hook(hook)
    try:
        backend.unet_forward(image_tensor, prompt=prompt, timestep_index=timestep_index)
    finally:
        handle.remove()
    if "tensor" not in captured:
        raise RuntimeError(f"Hook did not fire for {layer_name}")
    return captured["tensor"]


def _flatten_normalized(tensor: torch.Tensor) -> torch.Tensor:
    x = tensor.float().flatten(start_dim=1)
    return F.normalize(x, p=2, dim=1, eps=1e-8)


def _mean_std_embedding(tensor: torch.Tensor) -> torch.Tensor:
    x = tensor.float()
    if x.ndim == 4:
        mean = x.mean(dim=(-2, -1))
        std = x.std(dim=(-2, -1), unbiased=False)
    elif x.ndim == 3:
        mean = x.mean(dim=1)
        std = x.std(dim=1, unbiased=False)
    elif x.ndim == 2:
        mean = x
        std = torch.zeros_like(x)
    else:
        flat = x.flatten(start_dim=1)
        mean = flat
        std = torch.zeros_like(flat)
    return F.normalize(torch.cat([mean, std], dim=1), p=2, dim=1, eps=1e-8)


def activation_reference(tensor: torch.Tensor, objective_variant: str) -> torch.Tensor:
    """Build a detached reference for a gradient-sanity objective variant."""

    if objective_variant == "pooled_cosine":
        return pool_activation(tensor).detach()
    if objective_variant == "mean_std_cosine":
        return _mean_std_embedding(tensor).detach()
    if objective_variant == "normalized_activation_mse":
        return _flatten_normalized(tensor).detach()
    raise ValueError(f"Unsupported objective_variant: {objective_variant}")


def activation_objective(tensor: torch.Tensor, reference: torch.Tensor, objective_variant: str) -> torch.Tensor:
    """Return Z for the selected activation objective variant."""

    if objective_variant == "pooled_cosine":
        current = pool_activation(tensor)
        cosine = F.cosine_similarity(reference.float(), current.float(), dim=1).mean()
        return 1.0 - cosine
    if objective_variant == "mean_std_cosine":
        current = _mean_std_embedding(tensor)
        cosine = F.cosine_similarity(reference.float(), current.float(), dim=1).mean()
        return 1.0 - cosine
    if objective_variant == "normalized_activation_mse":
        current = _flatten_normalized(tensor)
        return F.mse_loss(current.float(), reference.float())
    raise ValueError(f"Unsupported objective_variant: {objective_variant}")


def select_layers(scores_dir: Path, explicit_layers: list[str] | None, max_layers: int) -> list[str]:
    if explicit_layers:
        return explicit_layers[:max_layers]
    rows = read_csv(scores_dir / "layer_identity_scores.csv")
    layers: list[str] = []
    preferred = ["vae_image_latent", "unet.conv_in", "unet.up_blocks.0", "unet.mid_block"]
    for layer in preferred:
        if any(row["layer_name"] == layer for row in rows):
            layers.append(layer)
    attention = next((row["layer_name"] for row in rows if ".attn" in row["layer_name"]), None)
    if attention:
        layers.append(attention)
    for row in rows:
        if row["layer_name"] not in layers:
            layers.append(row["layer_name"])
        if len(layers) >= max_layers:
            break
    return layers[:max_layers]


def plot_gradient_outputs(rows: list[dict[str, Any]], output_dir: Path) -> list[str]:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    paths: list[str] = []
    if not rows:
        return paths
    final_rows = [
        row
        for row in rows
        if int(row["step"])
        == max(
            int(r["step"])
            for r in rows
            if r["layer_name"] == row["layer_name"]
            and r.get("objective_variant", "pooled_cosine") == row.get("objective_variant", "pooled_cosine")
            and r.get("prompt") == row.get("prompt")
        )
    ]
    labels = [f"{row['layer_name']} / {row.get('objective_variant', 'pooled_cosine')}"[:48] for row in final_rows]
    totals = [float(row["total_grad_norm"]) for row in final_rows]
    plt.figure(figsize=(10, 5))
    plt.bar(range(len(labels)), totals)
    plt.xticks(range(len(labels)), labels, rotation=70, ha="right", fontsize=7)
    plt.ylabel("total gradient norm")
    plt.title("Gradient norm by selected layer")
    plt.tight_layout()
    path = output_dir / "gradient_norm_by_layer.png"
    plt.savefig(path, dpi=180)
    plt.close()
    paths.append(path.as_posix())

    plt.figure(figsize=(9, 5))
    for key in sorted({(row["layer_name"], row.get("objective_variant", "pooled_cosine"), row.get("prompt", "")) for row in rows}):
        layer, variant, prompt = key
        layer_rows = sorted(
            [
                row
                for row in rows
                if row["layer_name"] == layer
                and row.get("objective_variant", "pooled_cosine") == variant
                and row.get("prompt", "") == prompt
            ],
            key=lambda r: int(r["step"]),
        )
        plt.plot([int(r["step"]) for r in layer_rows], [float(r["Z"]) for r in layer_rows], marker="o", label=layer[:28])
    plt.xlabel("Adam step")
    plt.ylabel("Z = 1 - cosine(original, perturbed)")
    plt.title("Short-step Z curves")
    plt.legend(fontsize=7)
    plt.tight_layout()
    path = output_dir / "short_step_Z_curves.png"
    plt.savefig(path, dpi=180)
    plt.close()
    paths.append(path.as_posix())
    return paths


def run_gradient_scan(
    root: Path,
    scores_dir: Path,
    extraction_dir: Path,
    output_dir: Path,
    explicit_layers: list[str] | None = None,
    explicit_image_ids: list[str] | None = None,
    prompts: list[str] | None = None,
    objective_variants: list[str] | None = None,
    max_layers: int = 5,
    max_cases: int = 3,
    prompt: str = "add black sunglasses",
    timestep_index: int = 6,
    steps: int = 5,
    learning_rate: float = 0.03,
    geometry_config: dict[str, Any] | None = None,
) -> dict[str, Any]:
    output_dir.mkdir(parents=True, exist_ok=True)
    manifest = read_csv(extraction_dir / "manifest" / "identity_manifest.csv")
    if explicit_image_ids:
        wanted = set(explicit_image_ids)
        manifest = [row for row in manifest if row.get("image_id") in wanted]
    manifest = manifest[:max_cases]
    layers = select_layers(scores_dir, explicit_layers, max_layers=max_layers)
    prompt_list = prompts or [prompt]
    variant_list = objective_variants or ["pooled_cosine"]
    backend = InstructLayerBackend(torch.device("cuda"))
    geom_config = parse_geometry_config(geometry_config)
    all_rows: list[dict[str, Any]] = []
    failures: list[dict[str, Any]] = []

    for layer_name in layers:
        for objective_variant in variant_list:
            for current_prompt in prompt_list:
                for image_row in manifest:
                    image_tensor = backend.load_image_tensor(image_row["image_path"])
                    _, channels, height, width = image_tensor.shape
                    geometry = GradientProbeGeometry(height, width, channels, backend.device, geom_config)
                    optimizer = torch.optim.Adam([p for p in geometry.parameters() if p.requires_grad], lr=learning_rate)
                    try:
                        with torch.no_grad():
                            original_activation = capture_activation_tensor(backend, image_tensor, current_prompt, timestep_index, layer_name)
                            original_ref = activation_reference(original_activation, objective_variant)
                        initial_Z = None
                        last_Z = None
                        for step in range(0, steps + 1):
                            optimizer.zero_grad(set_to_none=True)
                            perturbed, geom_payload = geometry(image_tensor)
                            activation = capture_activation_tensor(backend, perturbed, current_prompt, timestep_index, layer_name)
                            Z = activation_objective(activation, original_ref, objective_variant)
                            loss = -Z
                            if step > 0:
                                loss.backward()
                                grad_norms = geometry.grad_norms()
                                finite_grad = all(torch.isfinite(p.grad).all().item() for p in geometry.parameters() if p.grad is not None)
                                optimizer.step()
                                clamp_stats = geometry.project_()
                            else:
                                grad_norms = {
                                    "tps_grad_norm": 0.0,
                                    "delaunay_grad_norm": 0.0,
                                    "rolling_grad_norm": 0.0,
                                    "dct_grad_norm": 0.0,
                                    "fft_grad_norm": 0.0,
                                    "total_grad_norm": 0.0,
                                }
                                finite_grad = True
                                clamp_stats = geometry.project_()
                            if initial_Z is None:
                                initial_Z = float(Z.detach().cpu())
                            last_Z = float(Z.detach().cpu())
                            row = {
                                "layer_name": layer_name,
                                "objective_variant": objective_variant,
                                "image_id": image_row["image_id"],
                                "identity_id": image_row["identity_id"],
                                "prompt": current_prompt,
                                "timestep_index": int(timestep_index),
                                "step": int(step),
                                "initial_Z": initial_Z,
                                "Z": float(Z.detach().cpu()),
                                "loss": float(loss.detach().cpu()),
                                "finite_gradient_flag": bool(finite_grad),
                                "nan_or_inf_flag": not torch.isfinite(Z).item(),
                                "psnr": psnr(perturbed, image_tensor),
                                "ssim": ssim_global(perturbed, image_tensor),
                                "mse": float(mse(perturbed, image_tensor).detach().cpu()),
                                **geom_payload["diagnostics"],
                                **grad_norms,
                                **clamp_stats,
                            }
                            all_rows.append(row)
                        case_dir = output_dir / "debug_cases" / slugify(layer_name) / objective_variant / slugify(current_prompt) / image_row["image_id"]
                        case_dir.mkdir(parents=True, exist_ok=True)
                        write_json(
                            case_dir / "summary.json",
                            {
                                "layer_name": layer_name,
                                "objective_variant": objective_variant,
                                "image_id": image_row["image_id"],
                                "prompt": current_prompt,
                                "initial_Z": initial_Z,
                                "final_Z": last_Z,
                                "Z_increase": None if initial_Z is None or last_Z is None else last_Z - initial_Z,
                            },
                        )
                    except Exception as error:
                        failures.append(
                            {
                                "layer_name": layer_name,
                                "objective_variant": objective_variant,
                                "prompt": current_prompt,
                                "image_id": image_row["image_id"],
                                "error": repr(error),
                            }
                        )

    write_csv(output_dir / "gradient_sanity.csv", all_rows)
    write_csv(output_dir / "gradient_failures.csv", failures)
    ranking_rows: list[dict[str, Any]] = []
    for layer in layers:
        for objective_variant in variant_list:
            for current_prompt in prompt_list:
                layer_rows = [
                    row
                    for row in all_rows
                    if row["layer_name"] == layer
                    and row.get("objective_variant", "pooled_cosine") == objective_variant
                    and row.get("prompt") == current_prompt
                ]
                if not layer_rows:
                    ranking_rows.append({"layer_name": layer, "objective_variant": objective_variant, "prompt": current_prompt, "status": "failed", "reason": "no rows"})
                    continue
                initial_values = [float(row["Z"]) for row in layer_rows if int(row["step"]) == 0]
                final_values = [float(row["Z"]) for row in layer_rows if int(row["step"]) == steps]
                final_grad = [float(row["total_grad_norm"]) for row in layer_rows if int(row["step"]) == steps]
                finite = all(bool(row["finite_gradient_flag"]) and not bool(row["nan_or_inf_flag"]) for row in layer_rows)
                ranking_rows.append(
                    {
                        "layer_name": layer,
                        "objective_variant": objective_variant,
                        "prompt": current_prompt,
                        "status": "ok" if finite else "numerical_issue",
                        "mean_initial_Z": float(sum(initial_values) / max(len(initial_values), 1)),
                        "mean_final_Z": float(sum(final_values) / max(len(final_values), 1)),
                        "mean_Z_increase": float(
                            (sum(final_values) / max(len(final_values), 1)) - (sum(initial_values) / max(len(initial_values), 1))
                        ),
                        "mean_final_total_grad_norm": float(sum(final_grad) / max(len(final_grad), 1)),
                        "finite_all_rows": bool(finite),
                        "num_cases": len({row["image_id"] for row in layer_rows}),
                    }
                )
    ranking_rows.sort(key=lambda row: (float(row.get("mean_Z_increase", -1e9)), float(row.get("mean_final_total_grad_norm", -1e9))), reverse=True)
    selected = [
        row
        for row in ranking_rows
        if row.get("status") == "ok"
        and float(row.get("mean_final_total_grad_norm", 0.0)) > 0
        and float(row.get("mean_Z_increase", -1.0)) >= 0
    ][:3]
    rejected = [row for row in ranking_rows if row not in selected]
    write_csv(output_dir / "layer_gradient_rankings.csv", ranking_rows)
    write_json(output_dir / "selected_layers.json", {"layers": selected})
    write_json(output_dir / "rejected_layers.json", {"layers": rejected})
    graph_paths = plot_gradient_outputs(all_rows, output_dir)
    summary = {
        "layers_tested": layers,
        "num_rows": len(all_rows),
        "num_failures": len(failures),
        "steps": steps,
        "prompts": prompt_list,
        "timestep_index": timestep_index,
        "objective_variants": variant_list,
        "selected_layers": selected,
        "graph_paths": graph_paths,
        "note": "This is a short gradient sanity scan, not a full geometry attack.",
    }
    write_json(output_dir / "gradient_scan_summary.json", summary)
    return summary
