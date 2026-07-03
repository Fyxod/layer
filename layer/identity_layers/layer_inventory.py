"""Layer inventory for InstructPix2Pix identity-layer discovery."""
from __future__ import annotations

from pathlib import Path
from typing import Any

import torch

from .activation_hooks import ShapeHookCapture
from .cases import build_identity_manifest
from .instruct_backend import InstructLayerBackend
from .io import package_versions, write_csv, write_json


DEFAULT_PROMPT = "add black sunglasses"


def candidate_group(module_path: str, module_type: str) -> str:
    if module_path == "vae_image_latent":
        return "vae_conditioning"
    if module_path == "unet.conv_in":
        return "unet_input"
    if ".attn1" in module_path:
        return "self_attention"
    if ".attn2" in module_path:
        return "cross_attention"
    if ".ff" in module_path:
        return "feed_forward"
    if "transformer_blocks" in module_path:
        return "transformer_block"
    if module_path.startswith("unet.down_blocks"):
        return "down_block"
    if module_path.startswith("unet.mid_block"):
        return "mid_block"
    if module_path.startswith("unet.up_blocks"):
        return "up_block"
    if "ResnetBlock" in module_type or "Resnet" in module_type:
        return "resnet"
    return "other"


def parent_block(module_path: str) -> str:
    parts = module_path.split(".")
    if len(parts) >= 3 and parts[1] in {"down_blocks", "up_blocks"}:
        return ".".join(parts[:3])
    if len(parts) >= 2 and parts[1] == "mid_block":
        return "unet.mid_block"
    return ".".join(parts[:2]) if len(parts) >= 2 else module_path


def enumerate_candidates(pipe: Any) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = [
        {
            "layer_name": "vae_image_latent",
            "module_path": "vae_image_latent",
            "module_type": "VAEImageLatentPseudoLayer",
            "parent_block": "vae",
            "candidate_group": "vae_conditioning",
            "supports_forward_hook": False,
        }
    ]
    wanted_roots = {
        "conv_in",
        "down_blocks",
        "mid_block",
        "up_blocks",
    }
    wanted_fragments = ("transformer_blocks", ".attn1", ".attn2", ".ff")
    for name, module in pipe.unet.named_modules():
        if not name:
            continue
        include = False
        if name in wanted_roots or name.startswith(("down_blocks.", "mid_block", "up_blocks.")):
            # Keep block-level modules and semantically useful attention/FF layers.
            depth = name.count(".")
            include = depth <= 1 or any(fragment in f".{name}" for fragment in wanted_fragments)
        if not include and any(fragment in f".{name}" for fragment in wanted_fragments):
            include = True
        if not include:
            continue
        module_path = f"unet.{name}"
        module_type = module.__class__.__name__
        rows.append(
            {
                "layer_name": module_path,
                "module_path": module_path,
                "module_type": module_type,
                "parent_block": parent_block(module_path),
                "candidate_group": candidate_group(module_path, module_type),
                "num_parameters": int(sum(p.numel() for p in module.parameters(recurse=False))),
                "supports_forward_hook": True,
            }
        )
    return rows


def recommend_initial_layers(candidates: list[dict[str, Any]], max_attention_layers: int = 24) -> list[dict[str, Any]]:
    selected: list[dict[str, Any]] = []
    always = {
        "vae_image_latent",
        "unet.conv_in",
        "unet.down_blocks.0",
        "unet.down_blocks.1",
        "unet.down_blocks.2",
        "unet.down_blocks.3",
        "unet.mid_block",
        "unet.up_blocks.0",
        "unet.up_blocks.1",
        "unet.up_blocks.2",
        "unet.up_blocks.3",
    }
    by_path = {row["module_path"]: row for row in candidates}
    for path in always:
        if path in by_path:
            selected.append(by_path[path])

    attention_like = [
        row
        for row in candidates
        if row["candidate_group"] in {"self_attention", "cross_attention", "feed_forward", "transformer_block"}
    ]
    # Evenly sample transformer-ish candidates so the first scan is tractable.
    if attention_like:
        stride = max(1, len(attention_like) // max_attention_layers)
        for row in attention_like[::stride][:max_attention_layers]:
            if row not in selected:
                selected.append(row)
    return selected


def run_inventory(
    root: Path,
    mat_root: Path,
    output_dir: Path,
    prompt: str = DEFAULT_PROMPT,
    timestep_index: int = 6,
    face_ids: list[str] | None = None,
) -> dict[str, Any]:
    output_dir.mkdir(parents=True, exist_ok=True)
    manifest_payload = build_identity_manifest(mat_root, output_dir / "manifest", face_ids=face_ids)
    if not manifest_payload["manifest"]:
        raise FileNotFoundError(f"No MAT face images found under {mat_root / 'data'}")
    image_path = manifest_payload["manifest"][0]["image_path"]

    backend = InstructLayerBackend(torch.device("cuda"))
    candidates = enumerate_candidates(backend.pipe)
    hook_paths = [row["module_path"] for row in candidates if row.get("supports_forward_hook")]

    image_tensor = backend.load_image_tensor(image_path)
    with torch.no_grad():
        vae_latent = backend.encode_image_latent(image_tensor)
        with ShapeHookCapture(backend.pipe, hook_paths) as capture:
            backend.unet_forward(image_tensor, prompt=prompt, timestep_index=timestep_index)

    shape_map = {name: record.__dict__ for name, record in capture.records.items()}
    rows: list[dict[str, Any]] = []
    shape_rows: list[dict[str, Any]] = []
    for row in candidates:
        merged = dict(row)
        if row["module_path"] == "vae_image_latent":
            merged.update(
                {
                    "output_type": "torch.Tensor",
                    "output_shape": list(vae_latent.shape),
                    "dtype": str(vae_latent.dtype),
                    "device": str(vae_latent.device),
                    "output_dtype": str(vae_latent.dtype),
                    "output_device": str(vae_latent.device),
                    "number_of_elements": int(vae_latent.numel()),
                    "estimated_memory": float(vae_latent.numel() * vae_latent.element_size() / (1024**2)),
                    "estimated_memory_mb": float(vae_latent.numel() * vae_latent.element_size() / (1024**2)),
                    "hook_error": None,
                }
            )
        else:
            rec = shape_map.get(row["module_path"], {})
            merged.update(
                {
                    "output_type": "torch.Tensor" if rec.get("output_shape") else None,
                    "output_shape": rec.get("output_shape"),
                    "dtype": rec.get("output_dtype"),
                    "device": rec.get("output_device"),
                    "output_dtype": rec.get("output_dtype"),
                    "output_device": rec.get("output_device"),
                    "number_of_elements": rec.get("number_of_elements"),
                    "estimated_memory": rec.get("estimated_memory_mb"),
                    "estimated_memory_mb": rec.get("estimated_memory_mb"),
                    "hook_error": rec.get("error"),
                }
            )
        rows.append(merged)
        shape_rows.append(
            {
                "layer_name": merged["layer_name"],
                "module_path": merged["module_path"],
                "module_type": merged["module_type"],
                "candidate_group": merged["candidate_group"],
                "output_type": merged.get("output_type"),
                "output_shape": str(merged.get("output_shape")),
                "dtype": merged.get("dtype"),
                "device": merged.get("device"),
                "number_of_elements": merged.get("number_of_elements"),
                "estimated_memory": merged.get("estimated_memory"),
                "hook_error": merged.get("hook_error"),
            }
        )

    recommended = recommend_initial_layers(rows)
    tree_lines = []
    for name, module in backend.pipe.unet.named_modules():
        if name:
            tree_lines.append(f"unet.{name}\t{module.__class__.__name__}")

    payload = {
        "model_id": backend.settings.model_id,
        "prompt": prompt,
        "timestep_index": timestep_index,
        "timesteps": backend.resolved_timesteps(),
        "inventory_image_path": image_path,
        "frozen_flags": backend.frozen_flags(),
        "package_versions": package_versions(),
        "num_candidates": len(rows),
        "num_recommended_initial_layers": len(recommended),
        "candidates": rows,
    }
    environment = {
        "package_versions": payload["package_versions"],
        "device": "cuda",
        "mat_root_required": True,
    }
    model_config = {
        "model_id": backend.settings.model_id,
        "torch_dtype": backend.settings.torch_dtype,
        "num_inference_steps": backend.settings.num_inference_steps,
        "guidance_scale": backend.settings.guidance_scale,
        "image_guidance_scale": backend.settings.image_guidance_scale,
        "seed": backend.settings.seed,
        "frozen_flags": backend.frozen_flags(),
        "resolved_timesteps": backend.resolved_timesteps(),
    }
    experiment_plan = {
        "milestone": "Milestone 1 — Identity Layer Scan",
        "scope": [
            "layer inventory",
            "activation hooks",
            "pooled activation extraction",
            "identity pair construction",
            "layer identity scoring",
            "layer/timestep ranking",
            "report generation",
        ],
        "explicitly_not_implemented": [
            "gradient sanity scan",
            "geometry attack",
            "pixel noise",
            "adversarial patches",
            "finetuning",
            "LoRA",
            "SPSA",
            "CEM",
            "landmarks",
            "face alignment",
            "face detection",
        ],
    }
    write_json(output_dir / "layer_inventory.json", payload)
    write_json(output_dir / "environment.json", environment)
    write_json(output_dir / "model_config.json", model_config)
    write_json(output_dir / "experiment_plan.json", experiment_plan)
    write_csv(output_dir / "layer_inventory.csv", rows)
    write_csv(output_dir / "activation_shapes.csv", shape_rows)
    write_json(output_dir / "recommended_initial_layers.json", {"layers": recommended})
    (output_dir / "model_tree.txt").write_text("\n".join(tree_lines) + "\n", encoding="utf-8")
    return payload
