"""Identity-pair scoring for pooled layer activations."""
from __future__ import annotations

from collections import defaultdict
from pathlib import Path
from typing import Any

import numpy as np

from .io import read_csv, write_csv, write_json


def cosine_similarity(left: np.ndarray, right: np.ndarray) -> float:
    denom = float(np.linalg.norm(left) * np.linalg.norm(right))
    if denom <= 1e-12:
        return 0.0
    return float(np.dot(left, right) / denom)


def auc_rank(labels: list[int], scores: list[float]) -> float | None:
    positives = [score for label, score in zip(labels, scores) if label == 1]
    negatives = [score for label, score in zip(labels, scores) if label == 0]
    if not positives or not negatives:
        return None
    wins = 0.0
    total = len(positives) * len(negatives)
    for pos in positives:
        for neg in negatives:
            if pos > neg:
                wins += 1.0
            elif pos == neg:
                wins += 0.5
    return float(wins / total)


def run_scoring(extraction_dir: Path, output_dir: Path) -> dict[str, Any]:
    output_dir.mkdir(parents=True, exist_ok=True)
    index_rows = read_csv(extraction_dir / "embeddings_index.csv")
    same_pairs = read_csv(extraction_dir / "manifest" / "same_identity_pairs.csv")
    diff_pairs = read_csv(extraction_dir / "manifest" / "different_identity_pairs.csv")
    if not index_rows:
        raise FileNotFoundError(f"No embeddings found in {extraction_dir}")

    embeddings: dict[tuple[str, str, str, int], np.ndarray] = {}
    meta: dict[tuple[str, str, str, int], dict[str, str]] = {}
    for row in index_rows:
        key = (row["layer_name"], row["prompt"], row["image_id"], int(row["timestep_index"]))
        embeddings[key] = np.load(extraction_dir / row["embedding_path"]).astype(np.float32)
        meta[key] = row

    prompts = sorted({row["prompt"] for row in index_rows})
    timesteps = sorted({int(row["timestep_index"]) for row in index_rows})
    layers = sorted({row["layer_name"] for row in index_rows})
    pair_scores: list[dict[str, Any]] = []

    for layer in layers:
        for prompt in prompts:
            for timestep in timesteps:
                for pair_type, pairs, label in [
                    ("same_identity", same_pairs, 1),
                    ("different_identity", diff_pairs, 0),
                ]:
                    for pair in pairs:
                        left_key = (layer, prompt, pair["left_image_id"], timestep)
                        right_key = (layer, prompt, pair["right_image_id"], timestep)
                        if left_key not in embeddings or right_key not in embeddings:
                            continue
                        cos = cosine_similarity(embeddings[left_key], embeddings[right_key])
                        euclidean = float(np.linalg.norm(embeddings[left_key] - embeddings[right_key]))
                        pair_scores.append(
                            {
                                "layer_name": layer,
                                "prompt": prompt,
                                "timestep_index": timestep,
                                "pair_type": pair_type,
                                "label_same_identity": label,
                                "left_image_id": pair["left_image_id"],
                                "right_image_id": pair["right_image_id"],
                                "left_identity_id": pair["left_identity_id"],
                                "right_identity_id": pair["right_identity_id"],
                                "cosine_similarity": cos,
                                "euclidean_distance": euclidean,
                            }
                        )

    grouped: dict[tuple[str, str, int], list[dict[str, Any]]] = defaultdict(list)
    for row in pair_scores:
        grouped[(row["layer_name"], row["prompt"], int(row["timestep_index"]))].append(row)

    layer_scores: list[dict[str, Any]] = []
    for (layer, prompt, timestep), rows in grouped.items():
        same = [float(row["cosine_similarity"]) for row in rows if row["pair_type"] == "same_identity"]
        diff = [float(row["cosine_similarity"]) for row in rows if row["pair_type"] == "different_identity"]
        labels = [int(row["label_same_identity"]) for row in rows]
        scores = [float(row["cosine_similarity"]) for row in rows]
        mean_same = float(np.mean(same)) if same else float("nan")
        mean_diff = float(np.mean(diff)) if diff else float("nan")
        auc = auc_rank(labels, scores)
        layer_scores.append(
            {
                "layer_name": layer,
                "prompt": prompt,
                "timestep_index": timestep,
                "num_same_pairs": len(same),
                "num_different_pairs": len(diff),
                "mean_same_cosine": mean_same,
                "mean_different_cosine": mean_diff,
                "identity_separation": mean_same - mean_diff if same and diff else float("nan"),
                "identity_auc": auc if auc is not None else float("nan"),
                "mean_same_euclidean": float(
                    np.mean([float(row["euclidean_distance"]) for row in rows if row["pair_type"] == "same_identity"])
                )
                if same
                else float("nan"),
                "mean_different_euclidean": float(
                    np.mean([float(row["euclidean_distance"]) for row in rows if row["pair_type"] == "different_identity"])
                )
                if diff
                else float("nan"),
            }
        )

    layer_scores.sort(key=lambda row: (float(row["identity_auc"]), float(row["identity_separation"])), reverse=True)
    write_csv(output_dir / "pair_scores.csv", pair_scores)
    write_csv(output_dir / "layer_scores.csv", layer_scores)
    write_csv(output_dir / "top_identity_layers.csv", layer_scores[:50])
    summary = {
        "num_pair_scores": len(pair_scores),
        "num_layer_prompt_timestep_scores": len(layer_scores),
        "top_layers": layer_scores[:20],
        "warning": "Scores from the default MAT auto-manifest are a development diagnostic, not a final identity benchmark.",
    }
    write_json(output_dir / "identity_score_summary.json", summary)
    return summary
