"""Report generation for the identity-layer scan."""
from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np

from .io import read_csv, read_json, write_json


def _safe_float(value: Any) -> float:
    try:
        return float(value)
    except Exception:
        return float("nan")


def _plot_reports(scores: list[dict[str, str]], pair_scores: list[dict[str, str]], output_dir: Path) -> list[str]:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    graph_dir = output_dir / "graphs"
    graph_dir.mkdir(parents=True, exist_ok=True)
    graph_paths: list[str] = []
    if not scores:
        return graph_paths

    top = sorted(scores, key=lambda r: _safe_float(r.get("identity_separation")), reverse=True)[:25]
    labels = [f"{Path(r['layer_name']).name or r['layer_name']}\nt{r['timestep_index']}" for r in top]

    plt.figure(figsize=(12, 6))
    plt.bar(range(len(top)), [_safe_float(r.get("identity_separation")) for r in top])
    plt.xticks(range(len(top)), labels, rotation=75, ha="right", fontsize=7)
    plt.ylabel("mean same cosine - mean different cosine")
    plt.title("Identity separation by layer/timestep")
    plt.tight_layout()
    path = graph_dir / "identity_separation_by_layer.png"
    plt.savefig(path, dpi=180)
    plt.close()
    graph_paths.append(path.relative_to(output_dir).as_posix())

    top_auc = sorted(scores, key=lambda r: _safe_float(r.get("identity_auc")), reverse=True)[:25]
    labels = [f"{Path(r['layer_name']).name or r['layer_name']}\nt{r['timestep_index']}" for r in top_auc]
    plt.figure(figsize=(12, 6))
    plt.bar(range(len(top_auc)), [_safe_float(r.get("identity_auc")) for r in top_auc])
    plt.xticks(range(len(top_auc)), labels, rotation=75, ha="right", fontsize=7)
    plt.ylabel("AUC")
    plt.ylim(0.0, 1.05)
    plt.title("Same-vs-different identity AUC by layer/timestep")
    plt.tight_layout()
    path = graph_dir / "identity_auc_by_layer.png"
    plt.savefig(path, dpi=180)
    plt.close()
    graph_paths.append(path.relative_to(output_dir).as_posix())

    same = [_safe_float(row["cosine_similarity"]) for row in pair_scores if row.get("pair_type") == "same_identity"]
    diff = [_safe_float(row["cosine_similarity"]) for row in pair_scores if row.get("pair_type") == "different_identity"]
    if same and diff:
        plt.figure(figsize=(8, 5))
        plt.hist(diff, bins=30, alpha=0.65, label="different identity")
        plt.hist(same, bins=30, alpha=0.65, label="same identity")
        plt.xlabel("cosine similarity")
        plt.ylabel("count")
        plt.title("Same vs different pooled-activation cosine distributions")
        plt.legend()
        plt.tight_layout()
        path = graph_dir / "same_vs_different_distributions.png"
        plt.savefig(path, dpi=180)
        plt.close()
        graph_paths.append(path.relative_to(output_dir).as_posix())

    # Heatmap: best separation by layer/timestep across prompts.
    layers = list(dict.fromkeys(row["layer_name"] for row in scores[:40]))
    timesteps = sorted({int(row["timestep_index"]) for row in scores})
    if layers and timesteps:
        matrix = np.full((len(layers), len(timesteps)), np.nan)
        for i, layer in enumerate(layers):
            for j, timestep in enumerate(timesteps):
                vals = [
                    _safe_float(row["identity_separation"])
                    for row in scores
                    if row["layer_name"] == layer and int(row["timestep_index"]) == timestep
                ]
                if vals:
                    matrix[i, j] = np.nanmean(vals)
        plt.figure(figsize=(8, max(6, len(layers) * 0.25)))
        plt.imshow(matrix, aspect="auto", interpolation="nearest")
        plt.colorbar(label="identity separation")
        plt.yticks(range(len(layers)), [Path(layer).name or layer for layer in layers], fontsize=7)
        plt.xticks(range(len(timesteps)), [str(t) for t in timesteps])
        plt.xlabel("timestep index")
        plt.title("Layer/timestep identity separation heatmap")
        plt.tight_layout()
        path = graph_dir / "layer_timestep_heatmap.png"
        plt.savefig(path, dpi=180)
        plt.close()
        graph_paths.append(path.relative_to(output_dir).as_posix())

    return graph_paths


def build_report(root: Path, output_dir: Path) -> dict[str, Any]:
    output_dir.mkdir(parents=True, exist_ok=True)
    inventory_dir = root / "identity_layers" / "outputs" / "layer_inventory"
    extraction_dir = root / "identity_layers" / "outputs" / "activation_scan"
    scores_dir = root / "identity_layers" / "outputs" / "identity_scores"
    scores = read_csv(scores_dir / "layer_scores.csv")
    pair_scores = read_csv(scores_dir / "pair_scores.csv")
    inventory_payload = read_json(inventory_dir / "layer_inventory.json") if (inventory_dir / "layer_inventory.json").exists() else {}
    extraction_payload = read_json(extraction_dir / "extraction_manifest.json") if (extraction_dir / "extraction_manifest.json").exists() else {}
    score_payload = read_json(scores_dir / "identity_score_summary.json") if (scores_dir / "identity_score_summary.json").exists() else {}
    graph_paths = _plot_reports(scores, pair_scores, output_dir)

    top_rows = scores[:25]
    table_md = "\n".join(
        [
            "| rank | layer | prompt | timestep | same cos | diff cos | separation | AUC |",
            "|---:|---|---|---:|---:|---:|---:|---:|",
            *[
                f"| {i+1} | `{row['layer_name']}` | {row['prompt']} | {row['timestep_index']} | "
                f"{_safe_float(row['mean_same_cosine']):.4f} | {_safe_float(row['mean_different_cosine']):.4f} | "
                f"{_safe_float(row['identity_separation']):.4f} | {_safe_float(row['identity_auc']):.4f} |"
                for i, row in enumerate(top_rows)
            ],
        ]
    )
    graph_md = "\n\n".join([f"![{Path(path).stem}]({path})" for path in graph_paths])
    md = f"""# InstructPix2Pix Identity Layer Scan

Milestone 1 scans frozen InstructPix2Pix internal activations for identity-separable layers. This is not a geometry attack and does not modify model weights.

## Run summary

- Model: `{inventory_payload.get('model_id', 'timbrooks/instruct-pix2pix')}`
- Inventory candidates: {inventory_payload.get('num_candidates', 'unknown')}
- Recommended initial layers: {inventory_payload.get('num_recommended_initial_layers', 'unknown')}
- Extracted embeddings: {extraction_payload.get('num_embeddings', 'unknown')}
- Layer/prompt/timestep scores: {score_payload.get('num_layer_prompt_timestep_scores', 'unknown')}
- Note: default MAT auto-manifest pairs are development diagnostics unless replaced with a richer identity dataset.

## Top identity-separable layer/timestep rows

{table_md}

## Graphs

{graph_md}
"""
    html_graphs = "\n".join([f'<figure><img src="{path}" alt="{Path(path).stem}"><figcaption>{Path(path).stem}</figcaption></figure>' for path in graph_paths])
    html_rows = "\n".join(
        [
            "<tr>"
            f"<td>{i+1}</td><td><code>{row['layer_name']}</code></td><td>{row['prompt']}</td>"
            f"<td>{row['timestep_index']}</td><td>{_safe_float(row['identity_separation']):.4f}</td>"
            f"<td>{_safe_float(row['identity_auc']):.4f}</td>"
            "</tr>"
            for i, row in enumerate(top_rows)
        ]
    )
    html = f"""<!doctype html>
<html><head><meta charset="utf-8"><title>InstructPix2Pix Identity Layer Scan</title>
<style>
body {{ font-family: Inter, Arial, sans-serif; max-width: 1100px; margin: 32px auto; line-height: 1.45; color: #1f2937; }}
h1, h2 {{ color: #111827; }}
table {{ border-collapse: collapse; width: 100%; font-size: 13px; }}
th, td {{ border: 1px solid #d1d5db; padding: 6px 8px; vertical-align: top; }}
th {{ background: #f3f4f6; }}
figure {{ border: 1px solid #e5e7eb; padding: 12px; border-radius: 10px; margin: 18px 0; }}
img {{ max-width: 100%; display: block; margin: 0 auto; }}
code {{ background: #f3f4f6; padding: 1px 4px; border-radius: 4px; }}
</style></head><body>
<h1>InstructPix2Pix Identity Layer Scan</h1>
<p>Milestone 1 scans frozen InstructPix2Pix internal activations for identity-separable layers. This is not a geometry attack and does not modify model weights.</p>
<h2>Run summary</h2>
<ul>
<li>Model: <code>{inventory_payload.get('model_id', 'timbrooks/instruct-pix2pix')}</code></li>
<li>Inventory candidates: {inventory_payload.get('num_candidates', 'unknown')}</li>
<li>Recommended initial layers: {inventory_payload.get('num_recommended_initial_layers', 'unknown')}</li>
<li>Extracted embeddings: {extraction_payload.get('num_embeddings', 'unknown')}</li>
<li>Layer/prompt/timestep scores: {score_payload.get('num_layer_prompt_timestep_scores', 'unknown')}</li>
</ul>
<h2>Top identity-separable layer/timestep rows</h2>
<table><thead><tr><th>rank</th><th>layer</th><th>prompt</th><th>timestep</th><th>separation</th><th>AUC</th></tr></thead><tbody>
{html_rows}
</tbody></table>
<h2>Graphs</h2>
{html_graphs}
</body></html>
"""
    (output_dir / "identity_layer_scan_report.md").write_text(md, encoding="utf-8")
    (output_dir / "identity_layer_scan_report.html").write_text(html, encoding="utf-8")
    summary = {
        "report_markdown": (output_dir / "identity_layer_scan_report.md").as_posix(),
        "report_html": (output_dir / "identity_layer_scan_report.html").as_posix(),
        "graph_paths": graph_paths,
        "top_rows": top_rows,
    }
    write_json(output_dir / "report_summary.json", summary)
    return summary
