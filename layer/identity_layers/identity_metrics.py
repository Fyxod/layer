"""Identity-pair scoring for pooled InstructPix2Pix activations."""
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
    """AUC where larger score means more likely same identity."""

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


def verification_metrics(labels: list[int], cosine_distances: list[float]) -> dict[str, float]:
    """Threshold cosine distance; below threshold predicts same identity."""

    if not labels or len(set(labels)) < 2:
        return {
            "identity_auc": float("nan"),
            "best_verification_threshold": float("nan"),
            "best_verification_accuracy": float("nan"),
            "equal_error_rate": float("nan"),
        }
    thresholds = sorted(set(cosine_distances))
    if thresholds:
        thresholds = [min(thresholds) - 1e-6, *thresholds, max(thresholds) + 1e-6]
    best_acc = -1.0
    best_threshold = float("nan")
    best_eer = float("nan")
    best_gap = float("inf")
    labels_arr = np.asarray(labels)
    distances = np.asarray(cosine_distances)
    for threshold in thresholds:
        pred_same = distances <= threshold
        truth_same = labels_arr == 1
        acc = float((pred_same == truth_same).mean())
        false_accept = float(((pred_same == 1) & (truth_same == 0)).sum() / max(1, (truth_same == 0).sum()))
        false_reject = float(((pred_same == 0) & (truth_same == 1)).sum() / max(1, (truth_same == 1).sum()))
        gap = abs(false_accept - false_reject)
        if acc > best_acc:
            best_acc = acc
            best_threshold = float(threshold)
        if gap < best_gap:
            best_gap = gap
            best_eer = float((false_accept + false_reject) / 2.0)
    auc = auc_rank(labels, [-distance for distance in cosine_distances])
    return {
        "identity_auc": auc if auc is not None else float("nan"),
        "best_verification_threshold": best_threshold,
        "best_verification_accuracy": best_acc,
        "equal_error_rate": best_eer,
    }


def mean_or_nan(values: list[float]) -> float:
    return float(np.mean(values)) if values else float("nan")


def std_or_zero(values: list[float]) -> float:
    return float(np.std(values)) if values else 0.0


def _score_group(rows: list[dict[str, Any]]) -> dict[str, float]:
    same_cos_dist = [float(row["cosine_distance"]) for row in rows if row["pair_type"] == "same_identity"]
    diff_cos_dist = [float(row["cosine_distance"]) for row in rows if row["pair_type"] == "different_identity"]
    same_euc = [float(row["euclidean_distance"]) for row in rows if row["pair_type"] == "same_identity"]
    diff_euc = [float(row["euclidean_distance"]) for row in rows if row["pair_type"] == "different_identity"]
    labels = [int(row["label_same_identity"]) for row in rows]
    cosine_distances = [float(row["cosine_distance"]) for row in rows]
    verify = verification_metrics(labels, cosine_distances)
    identity_separation = mean_or_nan(diff_cos_dist) - mean_or_nan(same_cos_dist)
    return {
        "num_same_pairs": len(same_cos_dist),
        "num_different_pairs": len(diff_cos_dist),
        "mean_same_identity_cosine_distance": mean_or_nan(same_cos_dist),
        "mean_different_identity_cosine_distance": mean_or_nan(diff_cos_dist),
        "mean_same_identity_euclidean_distance": mean_or_nan(same_euc),
        "mean_different_identity_euclidean_distance": mean_or_nan(diff_euc),
        "identity_separation": identity_separation,
        **verify,
    }


def _write_graphs(output_dir: Path, condition_scores: list[dict[str, Any]], pair_scores: list[dict[str, Any]]) -> list[str]:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    paths: list[str] = []
    if condition_scores:
        top_sep = sorted(condition_scores, key=lambda row: float(row["identity_separation"]), reverse=True)[:25]
        labels = [f"{Path(row['layer_name']).name or row['layer_name']}\nt{row['timestep_index']}" for row in top_sep]
        plt.figure(figsize=(12, 6))
        plt.bar(range(len(top_sep)), [float(row["identity_separation"]) for row in top_sep])
        plt.xticks(range(len(top_sep)), labels, rotation=75, ha="right", fontsize=7)
        plt.ylabel("mean different distance - mean same distance")
        plt.title("Identity separation by layer/timestep")
        plt.tight_layout()
        path = output_dir / "identity_separation_by_layer.png"
        plt.savefig(path, dpi=180)
        plt.close()
        paths.append(path.as_posix())

        top_auc = sorted(condition_scores, key=lambda row: float(row["identity_auc"]), reverse=True)[:25]
        labels = [f"{Path(row['layer_name']).name or row['layer_name']}\nt{row['timestep_index']}" for row in top_auc]
        plt.figure(figsize=(12, 6))
        plt.bar(range(len(top_auc)), [float(row["identity_auc"]) for row in top_auc])
        plt.xticks(range(len(top_auc)), labels, rotation=75, ha="right", fontsize=7)
        plt.ylabel("AUC")
        plt.ylim(0.0, 1.05)
        plt.title("Same/different identity verification AUC")
        plt.tight_layout()
        path = output_dir / "identity_auc_by_layer.png"
        plt.savefig(path, dpi=180)
        plt.close()
        paths.append(path.as_posix())

        layers = list(dict.fromkeys(row["layer_name"] for row in condition_scores[:40]))
        timesteps = sorted({int(row["timestep_index"]) for row in condition_scores})
        if layers and timesteps:
            matrix = np.full((len(layers), len(timesteps)), np.nan)
            for i, layer in enumerate(layers):
                for j, timestep in enumerate(timesteps):
                    values = [
                        float(row["identity_separation"])
                        for row in condition_scores
                        if row["layer_name"] == layer and int(row["timestep_index"]) == timestep
                    ]
                    if values:
                        matrix[i, j] = float(np.mean(values))
            plt.figure(figsize=(8, max(6, len(layers) * 0.25)))
            plt.imshow(matrix, aspect="auto", interpolation="nearest")
            plt.colorbar(label="identity separation")
            plt.yticks(range(len(layers)), [Path(layer).name or layer for layer in layers], fontsize=7)
            plt.xticks(range(len(timesteps)), [str(t) for t in timesteps])
            plt.xlabel("timestep index")
            plt.title("Layer × timestep identity-separation heatmap")
            plt.tight_layout()
            path = output_dir / "layer_timestep_heatmap.png"
            plt.savefig(path, dpi=180)
            plt.close()
            paths.append(path.as_posix())

    same = [float(row["cosine_distance"]) for row in pair_scores if row["pair_type"] == "same_identity"]
    diff = [float(row["cosine_distance"]) for row in pair_scores if row["pair_type"] == "different_identity"]
    if same and diff:
        plt.figure(figsize=(8, 5))
        plt.hist(diff, bins=30, alpha=0.65, label="different identity")
        plt.hist(same, bins=30, alpha=0.65, label="same identity")
        plt.xlabel("cosine distance")
        plt.ylabel("count")
        plt.title("Same vs different identity distance distributions")
        plt.legend()
        plt.tight_layout()
        path = output_dir / "same_vs_different_distributions.png"
        plt.savefig(path, dpi=180)
        plt.close()
        paths.append(path.as_posix())
    return paths


def run_scoring(extraction_dir: Path, output_dir: Path) -> dict[str, Any]:
    output_dir.mkdir(parents=True, exist_ok=True)
    index_path = extraction_dir / "pooled_embeddings_index.csv"
    if not index_path.exists():
        index_path = extraction_dir / "embeddings_index.csv"
    index_rows = read_csv(index_path)
    same_pairs = read_csv(extraction_dir / "manifest" / "same_identity_pairs.csv")
    diff_pairs = read_csv(extraction_dir / "manifest" / "different_identity_pairs.csv")
    if not index_rows:
        raise FileNotFoundError(f"No pooled embeddings found in {extraction_dir}")
    dataset_summary_path = extraction_dir / "manifest" / "dataset_summary.json"
    dataset_summary = {}
    if dataset_summary_path.exists():
        import json

        dataset_summary = json.loads(dataset_summary_path.read_text(encoding="utf-8"))

    embeddings: dict[tuple[str, str, str, int], np.ndarray] = {}
    for row in index_rows:
        key = (row["layer_name"], row["prompt"], row["image_id"], int(row["timestep_index"]))
        embeddings[key] = np.load(extraction_dir / row["embedding_path"]).astype(np.float32)

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
                                "cosine_distance": float(1.0 - cos),
                                "euclidean_distance": float(np.linalg.norm(embeddings[left_key] - embeddings[right_key])),
                            }
                        )

    condition_groups: dict[tuple[str, str, int], list[dict[str, Any]]] = defaultdict(list)
    for row in pair_scores:
        condition_groups[(row["layer_name"], row["prompt"], int(row["timestep_index"]))].append(row)

    verification_rows: list[dict[str, Any]] = []
    for (layer, prompt, timestep), rows in condition_groups.items():
        verification_rows.append({"layer_name": layer, "prompt": prompt, "timestep_index": timestep, **_score_group(rows)})
    verification_rows.sort(key=lambda row: (float(row["identity_separation"]), float(row["identity_auc"])), reverse=True)

    prompt_groups: dict[tuple[str, str], list[dict[str, Any]]] = defaultdict(list)
    timestep_groups: dict[tuple[str, int], list[dict[str, Any]]] = defaultdict(list)
    layer_groups: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in verification_rows:
        prompt_groups[(row["layer_name"], row["prompt"])].append(row)
        timestep_groups[(row["layer_name"], int(row["timestep_index"]))].append(row)
        layer_groups[row["layer_name"]].append(row)

    layer_prompt_scores = [
        {
            "layer_name": layer,
            "prompt": prompt,
            "mean_identity_separation": mean_or_nan([float(row["identity_separation"]) for row in rows]),
            "mean_identity_auc": mean_or_nan([float(row["identity_auc"]) for row in rows]),
            "num_conditions": len(rows),
        }
        for (layer, prompt), rows in prompt_groups.items()
    ]
    layer_timestep_scores = [
        {
            "layer_name": layer,
            "timestep_index": timestep,
            "mean_identity_separation": mean_or_nan([float(row["identity_separation"]) for row in rows]),
            "mean_identity_auc": mean_or_nan([float(row["identity_auc"]) for row in rows]),
            "num_conditions": len(rows),
        }
        for (layer, timestep), rows in timestep_groups.items()
    ]
    layer_identity_scores = []
    for layer, rows in layer_groups.items():
        separations = [float(row["identity_separation"]) for row in rows]
        aucs = [float(row["identity_auc"]) for row in rows]
        prompt_means = [
            float(row["mean_identity_separation"])
            for row in layer_prompt_scores
            if row["layer_name"] == layer and np.isfinite(float(row["mean_identity_separation"]))
        ]
        timestep_means = [
            float(row["mean_identity_separation"])
            for row in layer_timestep_scores
            if row["layer_name"] == layer and np.isfinite(float(row["mean_identity_separation"]))
        ]
        prompt_instability = std_or_zero(prompt_means)
        timestep_instability = std_or_zero(timestep_means)
        mean_sep = mean_or_nan(separations)
        mean_auc = mean_or_nan(aucs)
        layer_identity_scores.append(
            {
                "layer_name": layer,
                "mean_identity_separation": mean_sep,
                "mean_identity_auc": mean_auc,
                "best_identity_separation": float(np.max(separations)) if separations else float("nan"),
                "best_identity_auc": float(np.max(aucs)) if aucs else float("nan"),
                "prompt_instability": prompt_instability,
                "timestep_instability": timestep_instability,
                "layer_rank_score": float(mean_sep + mean_auc - prompt_instability - timestep_instability),
                "num_conditions": len(rows),
            }
        )
    layer_identity_scores.sort(key=lambda row: float(row["layer_rank_score"]), reverse=True)
    layer_prompt_scores.sort(key=lambda row: float(row["mean_identity_separation"]), reverse=True)
    layer_timestep_scores.sort(key=lambda row: float(row["mean_identity_separation"]), reverse=True)

    graph_paths = _write_graphs(output_dir, verification_rows, pair_scores)
    write_csv(output_dir / "pair_scores.csv", pair_scores)
    write_csv(output_dir / "verification_metrics.csv", verification_rows)
    write_csv(output_dir / "layer_identity_scores.csv", layer_identity_scores)
    write_csv(output_dir / "layer_timestep_scores.csv", layer_timestep_scores)
    write_csv(output_dir / "layer_prompt_scores.csv", layer_prompt_scores)
    # Compatibility aliases for the initial implementation/report code.
    write_csv(output_dir / "layer_scores.csv", verification_rows)
    write_csv(output_dir / "top_identity_layers.csv", layer_identity_scores[:50])
    ranked = {
        "ranking_basis": "layer_rank_score = mean_identity_separation + mean_identity_auc - prompt_instability - timestep_instability",
        "top_layers": layer_identity_scores[:50],
        "top_layer_prompt_timestep_conditions": verification_rows[:50],
        "dataset_summary": dataset_summary,
        "warning": dataset_summary.get("warning"),
    }
    write_json(output_dir / "ranked_layers.json", ranked)
    summary = {
        "num_pair_scores": len(pair_scores),
        "num_layer_prompt_timestep_scores": len(verification_rows),
        "num_layer_identity_scores": len(layer_identity_scores),
        "top_layers": layer_identity_scores[:20],
        "top_conditions": verification_rows[:20],
        "graph_paths": graph_paths,
        "dataset_summary": dataset_summary,
        "warning": ranked["warning"],
    }
    write_json(output_dir / "identity_score_summary.json", summary)
    return summary
