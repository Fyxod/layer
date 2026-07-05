"""Phase A6 baseline comparison for identity-layer scan outputs."""
from __future__ import annotations

import csv
import json
import sys
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


def _as_float(value: Any, default: float = float("nan")) -> float:
    try:
        return float(value)
    except Exception:
        return default


def raw_image_embedding(path: Path, size: int = 32) -> np.ndarray:
    image = Image.open(path).convert("RGB").resize((size, size), Image.Resampling.BICUBIC)
    arr = np.asarray(image, dtype=np.float32) / 255.0
    return l2_normalize(arr.reshape(-1))


def image_tensor(path: Path, device: torch.device) -> torch.Tensor:
    image = Image.open(path).convert("RGB")
    array = np.asarray(image, dtype=np.float32) / 255.0
    tensor = torch.from_numpy(array).permute(2, 0, 1).unsqueeze(0).to(device)
    return tensor


def pair_key(pair: dict[str, str]) -> str:
    left = pair["left_image_id"]
    right = pair["right_image_id"]
    if left <= right:
        return f"{left}::{right}"
    return f"{right}::{left}"


def pair_distance_map(
    vectors: dict[str, np.ndarray],
    same_pairs: list[dict[str, str]],
    diff_pairs: list[dict[str, str]],
) -> dict[str, dict[str, Any]]:
    distances: dict[str, dict[str, Any]] = {}
    for pair_type, pairs, label in [("same_identity", same_pairs, 1), ("different_identity", diff_pairs, 0)]:
        for pair in pairs:
            left = vectors.get(pair["left_image_id"])
            right = vectors.get(pair["right_image_id"])
            if left is None or right is None:
                continue
            cos = cosine_similarity(left, right)
            distances[pair_key(pair)] = {
                "pair_type": pair_type,
                "label_same_identity": label,
                "cosine_similarity": cos,
                "cosine_distance": float(1.0 - cos),
                "euclidean_distance": float(np.linalg.norm(left - right)),
            }
    return distances


def score_vectors(
    name: str,
    vectors: dict[str, np.ndarray],
    same_pairs: list[dict[str, str]],
    diff_pairs: list[dict[str, str]],
    extra: dict[str, Any] | None = None,
) -> dict[str, Any]:
    pair_rows = list(pair_distance_map(vectors, same_pairs, diff_pairs).values())
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


def _rankdata(values: np.ndarray) -> np.ndarray:
    """Average-rank implementation for Spearman correlation without scipy."""

    order = np.argsort(values, kind="mergesort")
    ranks = np.empty(len(values), dtype=np.float64)
    sorted_values = values[order]
    start = 0
    while start < len(values):
        end = start + 1
        while end < len(values) and sorted_values[end] == sorted_values[start]:
            end += 1
        average_rank = 0.5 * (start + end - 1)
        ranks[order[start:end]] = average_rank
        start = end
    return ranks


def _corrcoef(left: np.ndarray, right: np.ndarray) -> float:
    if len(left) < 2 or len(right) < 2:
        return float("nan")
    if float(np.std(left)) <= 1e-12 or float(np.std(right)) <= 1e-12:
        return float("nan")
    return float(np.corrcoef(left, right)[0, 1])


def pearson_spearman(left: list[float], right: list[float]) -> tuple[float, float]:
    left_arr = np.asarray(left, dtype=np.float64)
    right_arr = np.asarray(right, dtype=np.float64)
    pearson = _corrcoef(left_arr, right_arr)
    spearman = _corrcoef(_rankdata(left_arr), _rankdata(right_arr))
    return pearson, spearman


def default_face_repo_candidates(root: Path) -> list[Path]:
    candidates = [
        root.parent / "face",
        Path("/home/interns/Desktop/face"),
        Path("C:/Users/parth/Desktop/experiment/face"),
    ]
    return candidates


def default_arcface_checkpoint_candidates(root: Path) -> list[Path]:
    return [
        root.parent / "face" / "models" / "arcface" / "iresnet100.pth",
        Path("/home/interns/Desktop/face/models/arcface/iresnet100.pth"),
        Path("C:/Users/parth/Desktop/experiment/face/models/arcface/iresnet100.pth"),
    ]


def resolve_existing_path(explicit: Path | None, candidates: list[Path]) -> Path | None:
    if explicit is not None:
        return explicit.expanduser().resolve()
    for candidate in candidates:
        try:
            if candidate.exists():
                return candidate.resolve()
        except OSError:
            continue
    return None


def load_face_arcface_class(face_repo: Path | None) -> type[Any]:
    """Load the frozen ArcFace wrapper from the sibling FACE repo.

    LAYER intentionally treats ArcFace as an external evaluation baseline. The
    FACE repo already contains the checked iResNet-100 implementation and
    checkpoint setup, so this function imports that implementation when
    available rather than copying the model code into LAYER.
    """

    if face_repo is not None:
        face_repo = face_repo.resolve()
        if str(face_repo) not in sys.path:
            sys.path.insert(0, str(face_repo))
    from face.models.arcface import ArcFaceIResNet100  # type: ignore

    return ArcFaceIResNet100


def compute_arcface_vectors(
    manifest_rows: list[dict[str, str]],
    root: Path,
    output_dir: Path,
    checkpoint_path: Path | None = None,
    face_repo: Path | None = None,
    device_name: str = "cuda",
    fp16: bool = False,
) -> tuple[dict[str, np.ndarray], dict[str, Any]]:
    checkpoint = resolve_existing_path(checkpoint_path, default_arcface_checkpoint_candidates(root))
    resolved_face_repo = resolve_existing_path(face_repo, default_face_repo_candidates(root))
    if checkpoint is None or not checkpoint.exists():
        raise FileNotFoundError(
            "ArcFace checkpoint not found. Expected one of: "
            + ", ".join(str(path) for path in default_arcface_checkpoint_candidates(root))
            + " or pass --arcface-checkpoint."
        )
    if resolved_face_repo is None or not resolved_face_repo.exists():
        raise FileNotFoundError(
            "FACE repo not found for ArcFace model import. Expected sibling /home/interns/Desktop/face "
            "or pass --face-repo."
        )

    device = torch.device(device_name if device_name == "cpu" or torch.cuda.is_available() else "cpu")
    ArcFaceIResNet100 = load_face_arcface_class(resolved_face_repo)
    model = ArcFaceIResNet100(checkpoint_path=checkpoint, device=device, fp16=fp16)
    vectors: dict[str, np.ndarray] = {}
    with torch.no_grad():
        for row in manifest_rows:
            tensor = image_tensor(Path(row["image_path"]), device=device)
            embedding = model.embedding(tensor).detach().cpu().numpy()[0].astype(np.float32)
            vectors[row["image_id"]] = l2_normalize(embedding)
    metadata = {
        "arcface_status": "computed",
        "arcface_checkpoint_path": str(checkpoint),
        "arcface_face_repo": str(resolved_face_repo),
        "arcface_device": str(device),
        "arcface_fp16": bool(fp16),
        "arcface_num_vectors": len(vectors),
        **model.metadata(),
    }
    write_json(output_dir / "arcface_metadata.json", metadata)
    return vectors, metadata


def correlation_rows_against_arcface(
    baseline_vectors: dict[str, dict[str, np.ndarray]],
    arcface_vectors: dict[str, np.ndarray],
    same_pairs: list[dict[str, str]],
    diff_pairs: list[dict[str, str]],
) -> list[dict[str, Any]]:
    arcface_distances = pair_distance_map(arcface_vectors, same_pairs, diff_pairs)
    rows: list[dict[str, Any]] = []
    for name, vectors in baseline_vectors.items():
        distances = pair_distance_map(vectors, same_pairs, diff_pairs)
        common_keys = sorted(set(arcface_distances) & set(distances))
        arc_cos = [float(arcface_distances[key]["cosine_distance"]) for key in common_keys]
        base_cos = [float(distances[key]["cosine_distance"]) for key in common_keys]
        arc_euc = [float(arcface_distances[key]["euclidean_distance"]) for key in common_keys]
        base_euc = [float(distances[key]["euclidean_distance"]) for key in common_keys]
        pearson_cos, spearman_cos = pearson_spearman(arc_cos, base_cos)
        pearson_euc, spearman_euc = pearson_spearman(arc_euc, base_euc)
        rows.append(
            {
                "status": "computed",
                "baseline_name": name,
                "num_common_pairs": len(common_keys),
                "pearson_cosine_distance_vs_arcface": pearson_cos,
                "spearman_cosine_distance_vs_arcface": spearman_cos,
                "pearson_euclidean_distance_vs_arcface": pearson_euc,
                "spearman_euclidean_distance_vs_arcface": spearman_euc,
                "note": "Correlation is between pairwise distances, not an optimization objective.",
            }
        )
    rows.sort(key=lambda row: _as_float(row["spearman_cosine_distance_vs_arcface"]), reverse=True)
    return rows


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
    compute_arcface_baseline: bool = False,
    arcface_checkpoint: Path | None = None,
    face_repo: Path | None = None,
    arcface_device: str = "cuda",
    arcface_fp16: bool = False,
) -> dict[str, Any]:
    output_dir.mkdir(parents=True, exist_ok=True)
    manifest = read_csv(extraction_dir / "manifest" / "identity_manifest.csv")
    same_pairs = read_csv(extraction_dir / "manifest" / "same_identity_pairs.csv")
    diff_pairs = read_csv(extraction_dir / "manifest" / "different_identity_pairs.csv")
    extraction_manifest = json.loads((extraction_dir / "extraction_manifest.json").read_text(encoding="utf-8"))
    prompts = list(extraction_manifest.get("prompts", ["add black sunglasses"]))
    timestep_indices = list(extraction_manifest.get("timestep_indices", [6]))
    rows: list[dict[str, Any]] = []
    baseline_vectors_for_correlation: dict[str, dict[str, np.ndarray]] = {}

    raw_vectors = {row["image_id"]: raw_image_embedding(Path(row["image_path"])) for row in manifest}
    rows.append(score_vectors("raw_resized_image_32x32", raw_vectors, same_pairs, diff_pairs, {"baseline_group": "raw_image"}))
    baseline_vectors_for_correlation["raw_resized_image_32x32"] = raw_vectors

    rng = np.random.default_rng(1234)
    raw_matrix = np.stack([raw_vectors[row["image_id"]] for row in manifest], axis=0)
    projection = rng.normal(size=(raw_matrix.shape[1], 512)).astype(np.float32) / np.sqrt(raw_matrix.shape[1])
    random_vectors = {
        row["image_id"]: l2_normalize(raw_vectors[row["image_id"]] @ projection)
        for row in manifest
    }
    rows.append(score_vectors("random_projection_raw_512", random_vectors, same_pairs, diff_pairs, {"baseline_group": "random"}))
    baseline_vectors_for_correlation["random_projection_raw_512"] = random_vectors

    vae_vectors = load_layer_vectors(extraction_dir, "vae_image_latent")
    if vae_vectors:
        rows.append(score_vectors("vae_conditioning_latent_extracted", vae_vectors, same_pairs, diff_pairs, {"baseline_group": "vae"}))
        baseline_vectors_for_correlation["vae_conditioning_latent_extracted"] = vae_vectors

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
        baseline_vectors_for_correlation[f"instruct_layer__{layer_name}"] = vectors

    if compute_unet_baseline:
        unet_vectors = compute_unet_prediction_vectors(manifest, prompts[:2], timestep_indices[:4], output_dir)
        for name, vectors in unet_vectors.items():
            rows.append(score_vectors(name, vectors, same_pairs, diff_pairs, {"baseline_group": "final_unet_prediction"}))
            baseline_vectors_for_correlation[name] = vectors

    arcface_metadata: dict[str, Any] = {
        "arcface_status": "not_computed",
        "reason": "ArcFace baseline was not requested for this run.",
        "note": "ArcFace is reserved as an external evaluation reference, not an Instruct objective.",
    }
    if compute_arcface_baseline:
        try:
            arcface_vectors, arcface_metadata = compute_arcface_vectors(
                manifest_rows=manifest,
                root=root,
                output_dir=output_dir,
                checkpoint_path=arcface_checkpoint,
                face_repo=face_repo,
                device_name=arcface_device,
                fp16=arcface_fp16,
            )
            rows.append(
                score_vectors(
                    "arcface_iresnet100",
                    arcface_vectors,
                    same_pairs,
                    diff_pairs,
                    {"baseline_group": "arcface_external_reference"},
                )
            )
            correlation_rows = correlation_rows_against_arcface(
                baseline_vectors_for_correlation,
                arcface_vectors,
                same_pairs,
                diff_pairs,
            )
            write_csv(output_dir / "instruct_vs_arcface_correlation.csv", correlation_rows)
        except Exception as error:
            arcface_metadata = {
                "arcface_status": "failed",
                "error": repr(error),
                "note": "ArcFace is reserved as an external evaluation reference, not an Instruct objective.",
            }
            write_json(output_dir / "arcface_metadata.json", arcface_metadata)
            write_csv(
                output_dir / "instruct_vs_arcface_correlation.csv",
                [
                    {
                        "status": "failed",
                        "reason": repr(error),
                        "note": "ArcFace baseline failed; Instruct baseline rows were still written.",
                    }
                ],
            )
    else:
        write_csv(
            output_dir / "instruct_vs_arcface_correlation.csv",
            [
                {
                    "status": "not_computed",
                    "reason": "ArcFace baseline was not requested for this run.",
                    "note": "ArcFace is reserved as an external evaluation reference, not an Instruct objective.",
                }
            ],
        )

    rows.sort(key=lambda row: float(row["identity_separation"]), reverse=True)
    write_csv(output_dir / "baseline_comparison.csv", rows)
    graph_path = plot_baseline_comparison(rows, output_dir)
    summary = {
        "num_baselines": len(rows),
        "top_baselines": rows[:20],
        "graph_path": graph_path.as_posix(),
        **arcface_metadata,
    }
    write_json(output_dir / "baseline_comparison_summary.json", summary)
    return summary
