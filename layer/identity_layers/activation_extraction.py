"""Pooled identity activation extraction for the Milestone 1 scan."""
from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np
import torch

from .activation_hooks import PooledHookCapture
from .cases import build_identity_manifest, load_identity_manifest, slugify
from .instruct_backend import InstructLayerBackend
from .io import append_jsonl, read_json, write_csv, write_json
from .pooling import activation_stats, pool_activation


DEFAULT_PROMPTS = ["add black sunglasses", "add headphones"]
DEFAULT_TIMESTEP_INDICES = [3, 6, 10, 14]


def load_scan_layers(inventory_dir: Path, explicit_layers: list[str] | None = None, max_layers: int | None = None) -> list[str]:
    if explicit_layers:
        layers = explicit_layers
    else:
        rec_path = inventory_dir / "recommended_initial_layers.json"
        if not rec_path.exists():
            raise FileNotFoundError(f"Missing recommended layer list: {rec_path}")
        payload = read_json(rec_path)
        layers = [row["module_path"] for row in payload.get("layers", [])]
    if max_layers is not None:
        layers = layers[: int(max_layers)]
    if "vae_image_latent" not in layers:
        layers = ["vae_image_latent", *layers]
    return list(dict.fromkeys(layers))


def run_extraction(
    root: Path,
    mat_root: Path,
    inventory_dir: Path,
    output_dir: Path,
    prompts: list[str] | None = None,
    timestep_indices: list[int] | None = None,
    face_ids: list[str] | None = None,
    layers: list[str] | None = None,
    max_layers: int | None = None,
    canonical_only: bool = False,
    dataset_manifest: Path | None = None,
) -> dict[str, Any]:
    output_dir.mkdir(parents=True, exist_ok=True)
    if dataset_manifest is not None:
        manifest_payload = load_identity_manifest(root, dataset_manifest, output_dir / "manifest")
    else:
        manifest_payload = build_identity_manifest(
            mat_root,
            output_dir / "manifest",
            face_ids=face_ids,
            canonical_only=canonical_only,
        )
    manifest = manifest_payload["manifest"]
    if not manifest:
        raise FileNotFoundError(f"No images found under {mat_root / 'data'}")

    scan_layers = load_scan_layers(inventory_dir, explicit_layers=layers, max_layers=max_layers)
    hook_layers = [layer for layer in scan_layers if layer != "vae_image_latent"]
    scan_prompts = prompts or DEFAULT_PROMPTS
    scan_timestep_indices = timestep_indices or DEFAULT_TIMESTEP_INDICES

    backend = InstructLayerBackend(torch.device("cuda"))
    rows: list[dict[str, Any]] = []
    failures_path = output_dir / "failures.jsonl"
    embeddings_dir = output_dir / "pooled_embeddings"
    embeddings_dir.mkdir(parents=True, exist_ok=True)

    write_json(
        output_dir / "resolved_timesteps.json",
        {
            "num_inference_steps": backend.settings.num_inference_steps,
            "resolved_scheduler_timesteps": backend.resolved_timesteps(),
            "selected_timestep_indices": scan_timestep_indices,
            "note": "Scheduler timesteps are resolved by diffusers at runtime; timestep_index selects from this list.",
        },
    )

    first_tensor = backend.load_image_tensor(manifest[0]["image_path"])
    with torch.no_grad():
        first_latent = backend.encode_image_latent(first_tensor)
        fixed_noise = backend.fixed_noise(first_latent)
        prompt_metadata = []
        for prompt in scan_prompts:
            embedding = backend.encode_prompt(prompt)
            prompt_metadata.append(
                {
                    "prompt": prompt,
                    "embedding_shape": list(embedding.shape),
                    "embedding_dtype": str(embedding.dtype),
                    "embedding_norm": float(embedding.float().norm().detach().cpu()),
                }
            )
    write_json(
        output_dir / "fixed_noise_metadata.json",
        {
            "seed": backend.settings.seed,
            "noise_shape": list(fixed_noise.shape),
            "noise_dtype": str(fixed_noise.dtype),
            "noise_device": str(fixed_noise.device),
            "scheduler_init_noise_sigma": float(backend.pipe.scheduler.init_noise_sigma),
            "reference_image_path_for_shape": manifest[0]["image_path"],
        },
    )
    write_json(output_dir / "prompt_embeddings_metadata.json", {"prompts": prompt_metadata})

    for image_row in manifest:
        image_tensor = backend.load_image_tensor(image_row["image_path"])
        with torch.no_grad():
            vae_latent = backend.encode_image_latent(image_tensor)
            vae_pooled = pool_activation(vae_latent).detach().cpu().numpy()[0]
            vae_stats = activation_stats(vae_latent)
        for prompt in scan_prompts:
            for timestep_index in scan_timestep_indices:
                try:
                    collected: dict[str, dict[str, Any]] = {}
                    if "vae_image_latent" in scan_layers:
                        collected["vae_image_latent"] = {
                            "vector": vae_pooled,
                            "stats": vae_stats,
                            "output_shape": list(vae_latent.shape),
                            "output_dtype": str(vae_latent.dtype),
                        }
                    with torch.no_grad():
                        with PooledHookCapture(backend.pipe, hook_layers) as capture:
                            backend.unet_forward(image_tensor, prompt=prompt, timestep_index=timestep_index)
                        for layer_name, activation in capture.activations.items():
                            collected[layer_name] = {
                                "vector": activation.pooled.numpy()[0],
                                "stats": activation.stats,
                                "output_shape": activation.output_shape,
                                "output_dtype": activation.output_dtype,
                            }
                        for layer_name, error in capture.failures.items():
                            append_jsonl(
                                failures_path,
                                {
                                    "image_id": image_row["image_id"],
                                    "prompt": prompt,
                                    "timestep_index": timestep_index,
                                    "layer_name": layer_name,
                                    "error": error,
                                },
                            )
                    for layer_name, item in collected.items():
                        stem = "__".join(
                            [
                                slugify(layer_name),
                                image_row["image_id"],
                                slugify(prompt),
                                f"t{timestep_index:02d}",
                            ]
                        )
                        rel_path = Path("pooled_embeddings") / f"{stem}.npy"
                        np.save(output_dir / rel_path, item["vector"].astype(np.float32))
                        rows.append(
                            {
                                "embedding_path": rel_path.as_posix(),
                                "layer_name": layer_name,
                                "module_path": layer_name,
                                "prompt": prompt,
                                "timestep_index": int(timestep_index),
                                "image_id": image_row["image_id"],
                                "identity_id": image_row["identity_id"],
                                "image_path": image_row["image_path"],
                                "vector_dim": int(item["vector"].shape[0]),
                                "selected_tensor_rule": "first_tensor_recursive",
                                "embedding_finite": bool(np.isfinite(item["vector"]).all()),
                                "output_shape": str(item.get("output_shape")),
                                "output_dtype": item.get("output_dtype"),
                                **item["stats"],
                            }
                        )
                except Exception as error:
                    append_jsonl(
                        failures_path,
                        {
                            "image_id": image_row["image_id"],
                            "image_path": image_row["image_path"],
                            "prompt": prompt,
                            "timestep_index": timestep_index,
                            "error": repr(error),
                        },
                    )

    write_csv(output_dir / "embeddings_index.csv", rows)
    write_csv(output_dir / "pooled_embeddings_index.csv", rows)
    write_csv(output_dir / "activation_statistics.csv", rows)
    scan_condition_rows = [
        {
            "prompt": prompt,
            "timestep_index": timestep_index,
            "layer_name": layer_name,
            "seed": backend.settings.seed,
            "num_inference_steps": backend.settings.num_inference_steps,
            "guidance_scale": backend.settings.guidance_scale,
            "image_guidance_scale": backend.settings.image_guidance_scale,
            "text_embedding_fixed": True,
            "noise_tensor_fixed": True,
            "scheduler_fixed": True,
            "model_precision": backend.settings.torch_dtype,
        }
        for prompt in scan_prompts
        for timestep_index in scan_timestep_indices
        for layer_name in scan_layers
    ]
    write_csv(output_dir / "scan_conditions.csv", scan_condition_rows)
    summary = {
        "num_images": len(manifest),
        "num_embeddings": len(rows),
        "num_layers_requested": len(scan_layers),
        "layers": scan_layers,
        "prompts": scan_prompts,
        "timestep_indices": scan_timestep_indices,
        "manifest_summary": manifest_payload["summary"],
    }
    write_json(output_dir / "extraction_manifest.json", summary)
    return summary
