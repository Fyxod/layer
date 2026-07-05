"""Phase A6 baseline comparison for identity-layer scan outputs."""
from __future__ import annotations

import csv
import json
from pathlib import Path
from typing import Any

import numpy as np
import torch
from PIL import Image

from .identity_metrics import auc_rank, cosine_similarity, mean_or_nan
from .instruct_backend import InstructLayerBackend
from .io import read_csv, write_csv, write_json
from .pooling import pool_activation


def l2_normalize(vector: np.ndarray) -> np.ndarray:
    vector = vector.astype(np.float32).reshape(-1)
    norm = float(np.linalg.norm(vector))
    if norm <= 1e-12:
        return vector
    return vector / norm


def raw_image_embedding(path: Path, size: int = 32) -> np.ndarray:
    image = Image.open(path).convert("RGB").resize((size, size), Image.Resampling.BICUBIC)
    arr = np.asarray(image, dtype=np.float32) / 255.0
    return l2_normalize(arr.reshape(-1))


def score_vectors(
    name: str,
    vectors: dict[str, np.ndarray],
    same_pairs: list[dict[str, str]],
    diff_pairs: list[dict[str, str]],
    extra: dict[str, Any] | None = None,
) -> dict[str, Any]:
    pair_rows = []
    for pair_type, pairs, label in [("same_identity", same_pairs, 1), ("different_identity", diff_pairs, 0)]:
        for pair in pairs:
            left = vectors.get(pair["left_image_id"])
            right = vectors.get(pair["right_image_id"])
            if left is None or right is None:
                continue
            cos = cosine_similarity(left, right)
            pair_rows.append(
                {
                    "baseline_name": name,
                    "pair_type": pair_type,
                    "label_same_identity": label,
                    "cosine_similarity": cos,
                    "cosine_distance": float(1.0 - cos),
                    "euclidean_distance": float(np.linalg.norm(left - right)),
                }
            )
    same_cos = [float(row["cosine_distance"]) for row in pair_rows if row["pair_type"] == "same_identity"]
    diff_cos = [float(row["cosine_distance"]) for row in pair_rows if row["pair_type"] == "different_identity"]
    same_euc = [float(row["euclidean_distance"]) for row in pair_rows if row["pair_type"] == "same_identity"]
    diff_euc = [float(row["euclidean_distance"]) for row in pair_rows if row["pair_type"] == "different_identity"]
    labels = [int(row["label_same_identity"]) for row in pair_rows]
    auc = auc_rank(labels, [-float(row["cosine_distance"]) for row in pair_rows])
    return {
        "baseline_name": name,
        "num_vectors": len(vectors),
        "num_pair_scores": len(pair_rows),
        "mean_same_identity_cosine_distance": mean_or_nan(same_cos),
        "mean_different_identity_cosine_distance": mean_or_nan(diff_cos),
        "mean_same_identity_euclidean_distance": mean_or_nan(same_euc),
        "mean_different_identity_euclidean_distance": mean_or_nan(diff_euc),
        "identity_separation": mean_or_nan(diff_cos) - mean_or_nan(same_cos),
        "identity_auc": auc if auc is not None else float("nan"),
        **(extra or {}),
    }


def load_layer_vectors(
    extraction_dir: Path,
    layer_name: str,
    prompt: str | None = None,
    timestep_index: int | None = None,
) -> dict[str, np.ndarray]:
    index_path = extraction_dir / "pooled_embeddings_index.csv"
    if not index_path.exists():
        index_path = extraction_dir / "embeddings_index.csv"
    rows = read_csv(index_path)
    selected = []
    for row in rows:
        if row["layer_name"] != layer_name:
            continue
        if prompt is not None and row["prompt"] != prompt:
            continue
        if timestep_index is not None and int(row["timestep_index"]) != int(timestep_index):
            continue
        selected.append(row)
    grouped: dict[str, list[np.ndarray]] = {}
    for row in selected:
        grouped.setdefault(row["image_id"], []).append(np.load(extraction_dir / row["embedding_path"]).astype(np.float32))
    return {image_id: l2_normalize(np.mean(vectors, axis=0)) for image_id, vectors in grouped.items()}


def compute_unet_prediction_vectors(
    manifest_rows: list[dict[str, str]],
    prompts: list[str],
    timestep_indices: list[int],
    output_dir: Path,
) -> dict[str, dict[str, np.ndarray]]:
    """Compute final UNet-prediction baseline vectors grouped by condition."""

    backend = InstructLayerBackend(torch.device("cuda"))
    results: dict[str, dict[str, np.ndarray]] = {}
    for prompt in prompts:
        for timestep_index in timestep_indices:
            key = f"final_unet_prediction__{prompt}__t{timestep_index}"
            vectors: dict[str, np.ndarray] = {}
            for row in manifest_rows:
                image_tensor = backend.load_image_tensor(row["image_path"])
                with torch.no_grad():
                    payload = backend.unet_forward(image_tensor, prompt=prompt, timestep_index=timestep_index)
                    pooled = pool_activation(payload["prediction"]).detach().cpu().numpy()[0]
                vectors[row["image_id"]] = pooled.astype(np.float32)
            results[key] = vectors
    return results


def plot_baseline_comparison(rows: list[dict[str, Any]], output_dir: Path) -> Path:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    sorted_rows = sorted(rows, key=lambda row: float(row["identity_separation"]), reverse=True)
    labels = [row["baseline_name"][:38] for row in sorted_rows]
    values = [float(row["identity_separation"]) for row in sorted_rows]
    aucs = [float(row["identity_auc"]) for row in sorted_rows]
    fig, axes = plt.subplots(2, 1, figsize=(12, max(6, len(rows) * 0.25)), sharex=True)
    axes[0].bar(range(len(rows)), values)
    axes[0].set_ylabel("identity separation")
    axes[0].set_title("Baseline and selected-layer identity separation")
    axes[1].bar(range(len(rows)), aucs)
    axes[1].set_ylabel("AUC")
    axes[1].set_ylim(0, 1.05)
    axes[1].set_xticks(range(len(rows)))
    axes[1].set_xticklabels(labels, rotation=75, ha="right", fontsize=7)
    fig.tight_layout()
    path = output_dir / "baseline_comparison.png"
    fig.savefig(path, dpi=180)
    plt.close(fig)
    return path


def run_baseline_comparison(
    root: Path,
    extraction_dir: Path,
    scores_dir: Path,
    output_dir: Path,
    compute_unet_baseline: bool = True,
    max_top_layers: int = 8,
) -> dict[str, Any]:
    output_dir.mkdir(parents=True, exist_ok=True)
    manifest = read_csv(extraction_dir / "manifest" / "identity_manifest.csv")
    same_pairs = read_csv(extraction_dir / "manifest" / "same_identity_pairs.csv")
    diff_pairs = read_csv(extraction_dir / "manifest" / "different_identity_pairs.csv")
    extraction_manifest = json.loads((extraction_dir / "extraction_manifest.json").read_text(encoding="utf-8"))
    prompts = list(extraction_manifest.get("prompts", ["add black sunglasses"]))
    timestep_indices = list(extraction_manifest.get("timestep_indices", [6]))
    rows: list[dict[str, Any]] = []

    raw_vectors = {row["image_id"]: raw_image_embedding(Path(row["image_path"])) for row in manifest}
    rows.append(score_vectors("raw_resized_image_32x32", raw_vectors, same_pairs, diff_pairs, {"baseline_group": "raw_image"}))

    rng = np.random.default_rng(1234)
    raw_matrix = np.stack([raw_vectors[row["image_id"]] for row in manifest], axis=0)
    projection = rng.normal(size=(raw_matrix.shape[1], 512)).astype(np.float32) / np.sqrt(raw_matrix.shape[1])
    random_vectors = {
        row["image_id"]: l2_normalize(raw_vectors[row["image_id"]] @ projection)
        for row in manifest
    }
    rows.append(score_vectors("random_projection_raw_512", random_vectors, same_pairs, diff_pairs, {"baseline_group": "random"}))

    vae_vectors = load_layer_vectors(extraction_dir, "vae_image_latent")
    if vae_vectors:
        rows.append(score_vectors("vae_conditioning_latent_extracted", vae_vectors, same_pairs, diff_pairs, {"baseline_group": "vae"}))

    layer_rows = read_csv(scores_dir / "layer_identity_scores.csv")
    for layer_row in layer_rows[:max_top_layers]:
        layer_name = layer_row["layer_name"]
        vectors = load_layer_vectors(extraction_dir, layer_name)
        if not vectors:
            continue
        rows.append(
            score_vectors(
                f"instruct_layer__{layer_name}",
                vectors,
                same_pairs,
                diff_pairs,
                {
                    "baseline_group": "selected_instruct_layer",
                    "source_layer_rank_score": layer_row.get("layer_rank_score"),
                    "source_mean_identity_separation": layer_row.get("mean_identity_separation"),
                    "source_mean_identity_auc": layer_row.get("mean_identity_auc"),
                },
            )
        )

    if compute_unet_baseline:
        unet_vectors = compute_unet_prediction_vectors(manifest, prompts[:2], timestep_indices[:4], output_dir)
        for name, vectors in unet_vectors.items():
            rows.append(score_vectors(name, vectors, same_pairs, diff_pairs, {"baseline_group": "final_unet_prediction"}))

    rows.sort(key=lambda row: float(row["identity_separation"]), reverse=True)
    write_csv(output_dir / "baseline_comparison.csv", rows)
    write_csv(
        output_dir / "instruct_vs_arcface_correlation.csv",
        [
            {
                "status": "not_computed",
                "reason": "ArcFace embeddings were not provided/available for this baseline run.",
                "note": "ArcFace is reserved as an external evaluation reference, not an Instruct objective.",
            }
        ],
    )
    graph_path = plot_baseline_comparison(rows, output_dir)
    summary = {
        "num_baselines": len(rows),
        "top_baselines": rows[:20],
        "graph_path": graph_path.as_posix(),
        "arcface_status": "not_computed",
    }
    write_json(output_dir / "baseline_comparison_summary.json", summary)
    return summary
