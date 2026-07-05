"""Stage-B layer-targeted geometry smoke for InstructPix2Pix identity layers."""
from __future__ import annotations

import json
import math
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image, ImageDraw, ImageFont

from .cases import slugify
from .gradient_analysis import capture_activation_tensor, parse_geometry_config
from .gradient_geometry import GradientProbeGeometry
from .instruct_backend import InstructLayerBackend
from .io import append_jsonl, package_versions, read_csv, read_json, write_csv, write_json
from .metrics import mse, psnr, ssim_global
from .pooling import pool_activation


DEFAULT_STAGE_B_LAYERS = ["vae_image_latent", "unet.conv_in"]
DEFAULT_STAGE_B_PROMPTS = ["add black sunglasses", "add headphones"]


@dataclass
class StageBSmokeConfig:
    layers: list[str] = field(default_factory=lambda: list(DEFAULT_STAGE_B_LAYERS))
    prompts: list[str] = field(default_factory=lambda: list(DEFAULT_STAGE_B_PROMPTS))
    timestep_index: int = 6
    iterations: int = 50
    learning_rate: float = 0.03
    max_cases: int = 3
    seed: int = 1234
    image_ids: list[str] = field(default_factory=list)
    geometry: dict[str, Any] = field(default_factory=dict)
    generate_final_edits: bool = True
    final_edit_seed: int = 1234
    final_edit_num_inference_steps: int = 20
    final_edit_guidance_scale: float = 7.5
    final_edit_image_guidance_scale: float = 1.5


def tensor_to_pil(tensor: torch.Tensor) -> Image.Image:
    x = tensor.detach().float().clamp(0, 1)[0].permute(1, 2, 0).cpu().numpy()
    return Image.fromarray((x * 255.0 + 0.5).astype(np.uint8))


def save_tensor_image(path: Path, tensor: torch.Tensor) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tensor_to_pil(tensor).save(path)


def tensor_l2(x: torch.Tensor, y: torch.Tensor) -> float:
    return float(torch.sqrt((x.float() - y.float()).square().mean()).detach().cpu())


def pil_to_tensor_local(image: Image.Image, device: torch.device) -> torch.Tensor:
    arr = np.asarray(image.convert("RGB"), dtype=np.float32) / 255.0
    return torch.from_numpy(arr).permute(2, 0, 1).unsqueeze(0).to(device)


def flow_to_image(field: torch.Tensor) -> Image.Image:
    flow = field.detach().float()[0].cpu()
    dx = flow[0]
    dy = flow[1]
    mag = torch.sqrt(dx.square() + dy.square())
    vmax = float(torch.quantile(mag.flatten(), 0.995).clamp_min(1e-6))
    r = ((dx / vmax).clamp(-1, 1) * 0.5 + 0.5)
    g = ((dy / vmax).clamp(-1, 1) * 0.5 + 0.5)
    b = (mag / vmax).clamp(0, 1)
    rgb = torch.stack([r, g, b], dim=-1).numpy()
    return Image.fromarray((rgb * 255.0 + 0.5).astype(np.uint8))


def save_flow(path: Path, field: torch.Tensor) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    flow_to_image(field).save(path)


def captioned_panel(image: Image.Image | None, title: str, size: tuple[int, int] = (220, 220)) -> Image.Image:
    width, height = size
    panel = Image.new("RGB", (width, height + 34), "white")
    draw = ImageDraw.Draw(panel)
    if image is None:
        draw.rectangle([0, 0, width - 1, height - 1], fill=(245, 245, 245), outline=(180, 180, 180))
        draw.line([0, 0, width - 1, height - 1], fill=(180, 80, 80), width=3)
        draw.text((width // 2 - 28, height // 2 - 7), "missing", fill=(120, 40, 40))
    else:
        thumbnail = image.convert("RGB").copy()
        thumbnail.thumbnail((width, height), Image.Resampling.LANCZOS)
        x = (width - thumbnail.width) // 2
        y = (height - thumbnail.height) // 2
        panel.paste(thumbnail, (x, y))
        draw.rectangle([0, 0, width - 1, height - 1], outline=(210, 210, 210))
    draw.text((8, height + 9), title[:32], fill=(20, 20, 20))
    return panel


def make_comparison_sheet(
    path: Path,
    title: str,
    original: Image.Image | None,
    perturbed: Image.Image | None,
    clean_edit: Image.Image | None,
    perturbed_edit: Image.Image | None,
    flow: Image.Image | None,
) -> None:
    panels = [
        captioned_panel(original, "original"),
        captioned_panel(perturbed, "perturbed"),
        captioned_panel(clean_edit, "clean edit"),
        captioned_panel(perturbed_edit, "perturbed edit"),
        captioned_panel(flow, "combined flow"),
    ]
    margin = 14
    title_h = 42
    width = sum(p.width for p in panels) + margin * (len(panels) + 1)
    height = max(p.height for p in panels) + title_h + margin
    sheet = Image.new("RGB", (width, height), "white")
    draw = ImageDraw.Draw(sheet)
    draw.text((margin, 12), title[:160], fill=(0, 0, 0))
    x = margin
    for panel in panels:
        sheet.paste(panel, (x, title_h))
        x += panel.width + margin
    path.parent.mkdir(parents=True, exist_ok=True)
    sheet.save(path)


def safe_float(value: Any, default: float = float("nan")) -> float:
    try:
        return float(value)
    except Exception:
        return default


def load_stage_b_config(path: Path | None) -> StageBSmokeConfig:
    if path is None or not path.exists():
        return StageBSmokeConfig()
    payload = read_json(path)
    allowed = set(StageBSmokeConfig.__dataclass_fields__)
    values = {key: value for key, value in payload.items() if key in allowed}
    return StageBSmokeConfig(**values)


def select_manifest_rows(extraction_dir: Path, max_cases: int, image_ids: list[str] | None = None) -> list[dict[str, str]]:
    rows = read_csv(extraction_dir / "manifest" / "identity_manifest.csv")
    if image_ids:
        wanted = set(image_ids)
        rows = [row for row in rows if row.get("image_id") in wanted]
    # Keep diverse identities first when possible.
    selected: list[dict[str, str]] = []
    seen_identities: set[str] = set()
    for row in rows:
        identity_id = row.get("identity_id", "")
        if identity_id in seen_identities:
            continue
        selected.append(row)
        seen_identities.add(identity_id)
        if len(selected) >= max_cases:
            return selected
    for row in rows:
        if row not in selected:
            selected.append(row)
        if len(selected) >= max_cases:
            break
    return selected


def final_edit_image(
    backend: InstructLayerBackend,
    image: Image.Image,
    prompt: str,
    seed: int,
    num_inference_steps: int,
    guidance_scale: float,
    image_guidance_scale: float,
) -> Image.Image:
    generator = torch.Generator(device=backend.device).manual_seed(int(seed))
    with torch.no_grad():
        output = backend.pipe(
            prompt=prompt,
            image=image,
            num_inference_steps=int(num_inference_steps),
            guidance_scale=float(guidance_scale),
            image_guidance_scale=float(image_guidance_scale),
            generator=generator,
        )
    return output.images[0].convert("RGB")


def image_metrics(prefix: str, left: torch.Tensor, right: torch.Tensor) -> dict[str, float]:
    return {
        f"{prefix}_psnr": psnr(left, right),
        f"{prefix}_ssim": ssim_global(left, right),
        f"{prefix}_mse": float(mse(left, right).detach().cpu()),
        f"{prefix}_l2": tensor_l2(left, right),
    }


def capture_pooled(
    backend: InstructLayerBackend,
    image_tensor: torch.Tensor,
    prompt: str,
    timestep_index: int,
    layer_name: str,
) -> torch.Tensor:
    activation = capture_activation_tensor(backend, image_tensor, prompt, timestep_index, layer_name)
    return pool_activation(activation)


def run_single_stage_b_case(
    backend: InstructLayerBackend,
    image_row: dict[str, str],
    layer_name: str,
    prompt: str,
    config: StageBSmokeConfig,
    output_dir: Path,
) -> dict[str, Any]:
    output_dir.mkdir(parents=True, exist_ok=True)
    history_path = output_dir / "history.csv"
    history_jsonl_path = output_dir / "history.jsonl"
    if history_jsonl_path.exists():
        history_jsonl_path.unlink()
    resolved = {
        "layer_name": layer_name,
        "prompt": prompt,
        "image_row": image_row,
        "config": config.__dict__,
        "package_versions": package_versions(),
        "objective": "Z = 1 - cosine_similarity(pool(layer(original)), pool(layer(perturbed))); loss = -Z",
    }
    write_json(output_dir / "config_resolved.json", resolved)

    start_time = time.perf_counter()
    image_tensor = backend.load_image_tensor(image_row["image_path"])
    _, channels, height, width = image_tensor.shape
    geom_config = parse_geometry_config(config.geometry)
    geom_config.seed = int(config.seed)
    geometry = GradientProbeGeometry(height, width, channels, backend.device, geom_config)
    optimizer = torch.optim.Adam([p for p in geometry.parameters() if p.requires_grad], lr=float(config.learning_rate))

    with torch.no_grad():
        original_ref = capture_pooled(backend, image_tensor, prompt, int(config.timestep_index), layer_name).detach()

    rows: list[dict[str, Any]] = []
    failures: list[dict[str, Any]] = []
    best_Z = -float("inf")
    best_iter = 0
    best_payload: dict[str, Any] | None = None

    try:
        for iteration in range(0, int(config.iterations) + 1):
            iter_start = time.perf_counter()
            optimizer.zero_grad(set_to_none=True)
            perturbed, geom_payload = geometry(image_tensor)
            pooled = capture_pooled(backend, perturbed, prompt, int(config.timestep_index), layer_name)
            cosine = F.cosine_similarity(original_ref.float(), pooled.float(), dim=1).mean()
            Z = 1.0 - cosine
            loss = -Z
            if iteration > 0:
                loss.backward()
                grad_norms = geometry.grad_norms()
                finite_gradient = all(
                    torch.isfinite(param.grad).all().item()
                    for param in geometry.parameters()
                    if param.grad is not None
                )
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
                finite_gradient = True
                clamp_stats = geometry.project_()
            seconds_elapsed = time.perf_counter() - start_time
            row = {
                "iteration": int(iteration),
                "Z": float(Z.detach().cpu()),
                "Z_total": float(Z.detach().cpu()),
                "loss": float(loss.detach().cpu()),
                "layer_name": layer_name,
                "prompt": prompt,
                "timestep_index": int(config.timestep_index),
                "image_id": image_row["image_id"],
                "identity_id": image_row["identity_id"],
                "seed": int(config.seed),
                "learning_rate": float(config.learning_rate),
                "finite_gradient_flag": bool(finite_gradient),
                "nan_or_inf_flag": not torch.isfinite(Z).item(),
                **image_metrics("input", perturbed, image_tensor),
                **geom_payload["diagnostics"],
                **grad_norms,
                **clamp_stats,
                "seconds_iter": float(time.perf_counter() - iter_start),
                "seconds_elapsed": float(seconds_elapsed),
                "peak_vram_gb": float(torch.cuda.max_memory_allocated() / (1024**3)) if torch.cuda.is_available() else float("nan"),
            }
            rows.append(row)
            append_jsonl(history_jsonl_path, row)
            if row["Z"] > best_Z:
                best_Z = float(row["Z"])
                best_iter = int(iteration)
                best_payload = {
                    "perturbed": perturbed.detach().clone(),
                    "displacement": geom_payload["displacement"].detach().clone(),
                    "fields": {name: value.detach().clone() for name, value in geom_payload["fields"].items()},
                }
    except Exception as error:
        failures.append({"error": repr(error), "layer_name": layer_name, "prompt": prompt, "image_id": image_row["image_id"]})
        write_json(output_dir / "FAILED.json", {"status": "failed", "failures": failures})
        raise

    write_csv(history_path, rows)

    with torch.no_grad():
        final_perturbed, final_geom_payload = geometry(image_tensor)
        final_pooled = capture_pooled(backend, final_perturbed, prompt, int(config.timestep_index), layer_name)
        final_cosine = F.cosine_similarity(original_ref.float(), final_pooled.float(), dim=1).mean()
        final_Z = float((1.0 - final_cosine).detach().cpu())

    final_displacement = final_geom_payload["displacement"].detach()
    original_pil = tensor_to_pil(image_tensor)
    perturbed_pil = tensor_to_pil(final_perturbed)
    save_tensor_image(output_dir / "original.png", image_tensor)
    save_tensor_image(output_dir / "perturbed.png", final_perturbed)
    save_flow(output_dir / "combined_flow.png", final_displacement)
    for component_name, field in final_geom_payload["fields"].items():
        save_flow(output_dir / f"{component_name}_flow.png", field)

    clean_edit_pil: Image.Image | None = None
    perturbed_edit_pil: Image.Image | None = None
    output_metrics: dict[str, float] = {}
    final_edit_error: str | None = None
    if config.generate_final_edits:
        try:
            clean_edit_pil = final_edit_image(
                backend,
                original_pil,
                prompt,
                seed=int(config.final_edit_seed),
                num_inference_steps=int(config.final_edit_num_inference_steps),
                guidance_scale=float(config.final_edit_guidance_scale),
                image_guidance_scale=float(config.final_edit_image_guidance_scale),
            )
            perturbed_edit_pil = final_edit_image(
                backend,
                perturbed_pil,
                prompt,
                seed=int(config.final_edit_seed),
                num_inference_steps=int(config.final_edit_num_inference_steps),
                guidance_scale=float(config.final_edit_guidance_scale),
                image_guidance_scale=float(config.final_edit_image_guidance_scale),
            )
            clean_edit_pil.save(output_dir / "clean_edited.png")
            perturbed_edit_pil.save(output_dir / "perturbed_edited.png")
            clean_tensor = pil_to_tensor_local(clean_edit_pil, backend.device)
            perturbed_edit_tensor = pil_to_tensor_local(perturbed_edit_pil, backend.device)
            output_metrics = image_metrics("output", perturbed_edit_tensor, clean_tensor)
        except Exception as error:
            final_edit_error = repr(error)

    flow_pil = flow_to_image(final_displacement)
    make_comparison_sheet(
        output_dir / "comparison_sheet.png",
        f"{layer_name} | {prompt} | {image_row['image_id']}",
        original_pil,
        perturbed_pil,
        clean_edit_pil,
        perturbed_edit_pil,
        flow_pil,
    )

    initial_Z = float(rows[0]["Z"]) if rows else float("nan")
    final_row = rows[-1] if rows else {}
    summary = {
        "status": "done",
        "layer_name": layer_name,
        "prompt": prompt,
        "image_id": image_row["image_id"],
        "identity_id": image_row["identity_id"],
        "image_path": image_row["image_path"],
        "iterations": int(config.iterations),
        "timestep_index": int(config.timestep_index),
        "seed": int(config.seed),
        "initial_Z": initial_Z,
        "final_Z": final_Z,
        "logged_final_Z": safe_float(final_row.get("Z")),
        "Z_increase": final_Z - initial_Z,
        "best_Z": best_Z,
        "best_iter": best_iter,
        "final_loss": -final_Z,
        "final_input_psnr": psnr(final_perturbed, image_tensor),
        "final_input_ssim": ssim_global(final_perturbed, image_tensor),
        "final_input_mse": float(mse(final_perturbed, image_tensor).detach().cpu()),
        "final_input_l2": tensor_l2(final_perturbed, image_tensor),
        "final_combined_mean_disp_px": safe_float(final_geom_payload["diagnostics"].get("combined_mean_disp_px")),
        "final_combined_p95_disp_px": safe_float(final_geom_payload["diagnostics"].get("combined_p95_disp_px")),
        "final_combined_max_disp_px": safe_float(final_geom_payload["diagnostics"].get("combined_max_disp_px")),
        "final_jacobian_det_min": safe_float(final_geom_payload["diagnostics"].get("jacobian_det_min")),
        "final_foldover_fraction": safe_float(final_geom_payload["diagnostics"].get("foldover_fraction")),
        "final_smoothness_tv": safe_float(final_geom_payload["diagnostics"].get("smoothness_tv")),
        "final_fraction_clamped_total": safe_float(final_row.get("fraction_clamped_total")),
        "final_total_grad_norm": safe_float(final_row.get("total_grad_norm")),
        "seconds_elapsed": float(time.perf_counter() - start_time),
        "history_path": history_path.as_posix(),
        "run_dir": output_dir.as_posix(),
        "comparison_sheet": (output_dir / "comparison_sheet.png").as_posix(),
        "final_edit_error": final_edit_error,
        **output_metrics,
    }
    write_json(output_dir / "summary.json", summary)
    write_json(output_dir / "DONE.json", {"status": "done", "summary_path": "summary.json"})
    if best_payload is not None:
        # Store lightweight metadata only; tensors are intentionally not saved.
        write_json(
            output_dir / "best_metadata.json",
            {
                "best_iter": best_iter,
                "best_Z": best_Z,
                "note": "Best tensors are not saved to avoid committing large checkpoint artifacts.",
            },
        )
    return summary


def make_top_sheet(summaries: list[dict[str, Any]], output_path: Path, limit: int = 12) -> None:
    selected = sorted(summaries, key=lambda row: safe_float(row.get("Z_increase")), reverse=True)[:limit]
    if not selected:
        output_path.parent.mkdir(parents=True, exist_ok=True)
        Image.new("RGB", (600, 140), "white").save(output_path)
        return
    thumbs = []
    for row in selected:
        sheet_path = Path(row["comparison_sheet"])
        image = Image.open(sheet_path).convert("RGB") if sheet_path.exists() else None
        panel = captioned_panel(
            image,
            f"{row['layer_name'][:20]} | dZ={safe_float(row.get('Z_increase')):.4g}",
            size=(420, 170),
        )
        thumbs.append(panel)
    cols = 2
    margin = 16
    title_h = 42
    cell_w = max(panel.width for panel in thumbs)
    cell_h = max(panel.height for panel in thumbs)
    rows = math.ceil(len(thumbs) / cols)
    canvas = Image.new("RGB", (cols * cell_w + (cols + 1) * margin, rows * cell_h + title_h + (rows + 1) * margin), "white")
    draw = ImageDraw.Draw(canvas)
    draw.text((margin, 12), "Stage-B smoke top runs by Z increase", fill=(0, 0, 0))
    for idx, panel in enumerate(thumbs):
        col = idx % cols
        row = idx // cols
        x = margin + col * (cell_w + margin)
        y = title_h + margin + row * (cell_h + margin)
        canvas.paste(panel, (x, y))
    output_path.parent.mkdir(parents=True, exist_ok=True)
    canvas.save(output_path)


def write_stage_b_report(summaries: list[dict[str, Any]], output_dir: Path) -> Path:
    output_dir.mkdir(parents=True, exist_ok=True)
    sorted_rows = sorted(summaries, key=lambda row: safe_float(row.get("Z_increase")), reverse=True)
    selected_layers = sorted({row["layer_name"] for row in sorted_rows})
    lines = [
        "# Stage B targeted layer smoke report",
        "",
        "This is a short layer-targeted geometry smoke, not a full attack sweep.",
        "",
        "## Layers tested",
        "",
        *[f"- `{layer}`" for layer in selected_layers],
        "",
        "## Summary",
        "",
        f"- Runs completed: {len(sorted_rows)}",
        "- Objective: `Z = 1 - cosine_similarity(pool(layer(original)), pool(layer(perturbed)))`",
        "- Loss: `loss = -Z`",
        "- Trainable values: geometry parameters only",
        "",
        "## Top runs by Z increase",
        "",
        "| layer | image | prompt | initial Z | final Z | Z increase | input SSIM | max disp px | output SSIM |",
        "|---|---:|---|---:|---:|---:|---:|---:|---:|",
    ]
    for row in sorted_rows[:20]:
        lines.append(
            "| {layer} | {image} | {prompt} | {initial:.6g} | {final:.6g} | {inc:.6g} | {ssim:.4g} | {disp:.4g} | {out_ssim:.4g} |".format(
                layer=row.get("layer_name", ""),
                image=row.get("image_id", ""),
                prompt=row.get("prompt", ""),
                initial=safe_float(row.get("initial_Z")),
                final=safe_float(row.get("final_Z")),
                inc=safe_float(row.get("Z_increase")),
                ssim=safe_float(row.get("final_input_ssim")),
                disp=safe_float(row.get("final_combined_max_disp_px")),
                out_ssim=safe_float(row.get("output_ssim")),
            )
        )
    lines.extend(
        [
            "",
            "## Decision note",
            "",
            "Use this smoke only as Gate 4 evidence. Proceed to a 150-iteration run only if Z increases without numerical failure and the geometry diagnostics remain acceptable.",
            "",
        ]
    )
    path = output_dir / "stage_b_decision_report.md"
    path.write_text("\n".join(lines), encoding="utf-8")
    return path


def aggregate_stage_b_outputs(output_dir: Path) -> dict[str, Any]:
    output_dir.mkdir(parents=True, exist_ok=True)
    summaries = []
    for path in sorted(output_dir.glob("runs/**/summary.json")):
        try:
            summaries.append(json.loads(path.read_text(encoding="utf-8")))
        except Exception:
            continue
    summaries.sort(key=lambda row: safe_float(row.get("Z_increase")), reverse=True)
    write_csv(output_dir / "stage_b_all_runs.csv", summaries)
    write_csv(output_dir / "stage_b_top_runs.csv", summaries[:20])
    top_sheet = output_dir / "stage_b_top_sheet.jpg"
    make_top_sheet(summaries, top_sheet)
    report_path = write_stage_b_report(summaries, output_dir)
    summary = {
        "num_runs": len(summaries),
        "layers": sorted({row.get("layer_name") for row in summaries}),
        "prompts": sorted({row.get("prompt") for row in summaries}),
        "top_runs": summaries[:10],
        "top_sheet": top_sheet.as_posix(),
        "decision_report": report_path.as_posix(),
        "note": "Stage-B smoke only; not a full attack sweep.",
    }
    write_json(output_dir / "stage_b_summary.json", summary)
    return summary


def run_stage_b_smoke(
    root: Path,
    extraction_dir: Path,
    output_dir: Path,
    config: StageBSmokeConfig,
) -> dict[str, Any]:
    output_dir.mkdir(parents=True, exist_ok=True)
    write_json(output_dir / "stage_b_smoke_config_resolved.json", config.__dict__)
    manifest_rows = select_manifest_rows(extraction_dir, max_cases=int(config.max_cases), image_ids=config.image_ids)
    backend = InstructLayerBackend(torch.device("cuda"))
    all_summaries: list[dict[str, Any]] = []
    failures: list[dict[str, Any]] = []
    for layer_name in config.layers:
        for prompt in config.prompts:
            for image_row in manifest_rows:
                run_dir = (
                    output_dir
                    / "runs"
                    / slugify(layer_name)
                    / slugify(prompt)
                    / slugify(image_row["image_id"])
                )
                try:
                    summary = run_single_stage_b_case(
                        backend=backend,
                        image_row=image_row,
                        layer_name=layer_name,
                        prompt=prompt,
                        config=config,
                        output_dir=run_dir,
                    )
                    all_summaries.append(summary)
                except Exception as error:
                    failure = {
                        "layer_name": layer_name,
                        "prompt": prompt,
                        "image_id": image_row.get("image_id"),
                        "error": repr(error),
                        "run_dir": run_dir.as_posix(),
                    }
                    failures.append(failure)
                    write_json(run_dir / "FAILED.json", failure)
    write_csv(output_dir / "stage_b_failures.csv", failures)
    aggregate = aggregate_stage_b_outputs(output_dir)
    aggregate["num_failures"] = len(failures)
    aggregate["failures"] = failures
    write_json(output_dir / "stage_b_summary.json", aggregate)
    return aggregate
