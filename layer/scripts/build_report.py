"""Build a FACE/WOOD-style LAYER report from existing Stage-B outputs."""
from __future__ import annotations

import argparse
import csv
import html
import json
import math
import shutil
from collections import defaultdict
from pathlib import Path
from typing import Any

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from PIL import Image, ImageChops, ImageDraw, ImageEnhance, ImageOps


TITLE = "LAYER: InstructPix2Pix Identity-Layer Geometry Results"
SUBTITLE = "Stage-B targeted layer optimization with Stage-A identity-layer context"
AUTHOR = "Parth Katiyar"
OUTPUT_BASENAME = "layer_stage_b_report"

# Default run folder under identity_layers/outputs. Override from CLI with:
# --results-dir identity_layers/outputs/stage_b_constrained_spatial_200
RUN_FOLDER_NAME = "stage_b_broad_400"

COMPRESS_REPORT = False

COMPRESSED_QUALITY = {
    "image_size": (360, 360),
    "strip_format": "jpg",
    "strip_quality": 82,
    "graph_dpi": 120,
    "pdf_image_quality": 70,
}

HIGH_QUALITY = {
    "image_size": (512, 512),
    "strip_format": "png",
    "strip_quality": None,
    "graph_dpi": 150,
    "pdf_image_quality": 92,
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, default=Path.cwd())
    parser.add_argument("--results-dir", type=Path, default=None, help="Specific Stage-B output folder to report.")
    parser.add_argument("--run-folder", default=RUN_FOLDER_NAME, help="Folder under identity_layers/outputs; falls back to latest if missing.")
    parser.add_argument("--output-root", type=Path, default=None)
    parser.add_argument("--no-pdf", action="store_true")
    quality = parser.add_mutually_exclusive_group()
    quality.add_argument("--compress-report", dest="compress_report", action="store_true")
    quality.add_argument("--no-compress-report", dest="compress_report", action="store_false")
    parser.set_defaults(compress_report=COMPRESS_REPORT)
    return parser.parse_args()


def quality_settings(compress_report: bool) -> dict[str, Any]:
    return dict(COMPRESSED_QUALITY if compress_report else HIGH_QUALITY)


def slug(value: str) -> str:
    out = []
    for ch in value.lower():
        if ch.isalnum():
            out.append(ch)
        elif ch in {" ", "-", "_", "/", ".", ":"}:
            out.append("_")
    text = "".join(out)
    while "__" in text:
        text = text.replace("__", "_")
    return text.strip("_")


def to_float(value: Any) -> float | None:
    if value is None or value == "":
        return None
    try:
        number = float(value)
    except Exception:
        return None
    return number if math.isfinite(number) else None


def fmt(value: Any, digits: int = 4) -> str:
    number = to_float(value)
    if number is None:
        return "" if value is None else str(value)
    if abs(number) >= 100:
        return f"{number:.2f}"
    if abs(number) >= 10:
        return f"{number:.3f}"
    if abs(number) >= 1:
        return f"{number:.{digits}f}"
    return f"{number:.{digits}g}"


def read_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def read_csv_rows(path: Path) -> list[dict[str, str]]:
    if not path.exists():
        return []
    with path.open(newline="", encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


def write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        path.write_text("", encoding="utf-8")
        return
    fields = list(dict.fromkeys(key for row in rows for key in row.keys()))
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def relative(path: Path, base: Path) -> str:
    return path.resolve().relative_to(base.resolve()).as_posix()


def resolve_results_dir(root: Path, results_dir: Path | None, run_folder: str) -> Path:
    if results_dir is not None:
        path = results_dir if results_dir.is_absolute() else root / results_dir
        if path.exists():
            return path
        print(f"[layer-report] requested results dir missing, falling back: {path}")
    outputs = root / "identity_layers" / "outputs"
    if run_folder:
        path = outputs / run_folder
        if path.exists():
            return path
        print(f"[layer-report] requested run folder missing, falling back to latest: {path}")
    candidates = []
    for child in sorted(outputs.iterdir()) if outputs.exists() else []:
        if child.is_dir() and child.name.startswith("stage_b_") and (child / "stage_b_all_runs.csv").exists():
            candidates.append(child)
    if not candidates:
        raise FileNotFoundError(f"No Stage-B result folders found under {outputs}")
    return sorted(candidates, key=lambda p: p.stat().st_mtime)[-1]


def valid_flag(summary: dict[str, Any]) -> str:
    ssim = to_float(summary.get("final_input_ssim"))
    psnr = to_float(summary.get("final_input_psnr"))
    max_disp = to_float(summary.get("final_combined_max_disp_px"))
    fft_max = None
    hist = summary.get("_history_final") or {}
    if isinstance(hist, dict):
        fft_max = to_float(hist.get("fft_phase_max")) or to_float(hist.get("fft_phase_max_abs"))
    if ssim is not None and ssim < 0.75:
        return "invalid_input_destroyed"
    if psnr is not None and psnr < 16:
        return "invalid_input_destroyed"
    if fft_max is not None and fft_max >= 3.0:
        return "invalid_fft_saturated"
    if ssim is not None and ssim < 0.90:
        return "weak_input_preservation"
    if max_disp is not None and max_disp >= 7.9:
        return "weak_at_displacement_cap"
    return "valid_or_visually_check"


def collect_runs(results_dir: Path) -> tuple[list[dict[str, Any]], list[dict[str, str]]]:
    runs: list[dict[str, Any]] = []
    missing: list[dict[str, str]] = []
    for summary_path in sorted(results_dir.glob("runs/**/summary.json")):
        run_dir = summary_path.parent
        summary = read_json(summary_path)
        history_path = run_dir / "history.csv"
        history_rows = read_csv_rows(history_path)
        history_final = history_rows[-1] if history_rows else {}
        summary["_history_final"] = history_final
        layer = str(summary.get("layer_name") or run_dir.parts[-3])
        prompt = str(summary.get("prompt") or run_dir.parts[-2])
        image_id = str(summary.get("image_id") or run_dir.name)
        images = {
            "original": run_dir / "original.png",
            "perturbed": run_dir / "perturbed.png",
            "clean_edit": run_dir / "clean_edited.png",
            "perturbed_edit": run_dir / "perturbed_edited.png",
            "combined_flow": run_dir / "combined_flow.png",
            "comparison_sheet": run_dir / "comparison_sheet.png",
        }
        for label, path in images.items():
            if not path.exists():
                missing.append({"layer": layer, "prompt": prompt, "image_id": image_id, "artifact": label, "path": str(path)})
        runs.append(
            {
                "layer_name": layer,
                "prompt": prompt,
                "image_id": image_id,
                "identity_id": str(summary.get("identity_id", "")),
                "case": f"{image_id} / {prompt}",
                "run_dir": run_dir,
                "summary": summary,
                "history_rows": history_rows,
                "images": images,
                "validity": valid_flag(summary),
            }
        )
    runs.sort(key=lambda r: (r["layer_name"], r["prompt"], r["image_id"]))
    return runs, missing


def load_stage_a_context(root: Path) -> dict[str, Any]:
    out: dict[str, Any] = {}
    baseline = root / "identity_layers" / "outputs" / "baseline_comparison" / "baseline_comparison.csv"
    grad = root / "identity_layers" / "outputs" / "gradient_scan" / "layer_gradient_rankings.csv"
    selected = root / "identity_layers" / "outputs" / "gradient_scan" / "selected_layers.json"
    corr = root / "identity_layers" / "outputs" / "baseline_comparison" / "instruct_vs_arcface_correlation.csv"
    out["baseline_rows"] = read_csv_rows(baseline)[:20]
    out["gradient_rows"] = read_csv_rows(grad)
    out["arcface_corr_rows"] = read_csv_rows(corr)[:20]
    out["selected_layers"] = read_json(selected).get("layers", []) if selected.exists() else []
    return out


def image_or_placeholder(path: Path, size: tuple[int, int]) -> Image.Image:
    if path.exists():
        img = Image.open(path).convert("RGB")
        img = ImageOps.contain(img, size, Image.Resampling.LANCZOS)
        canvas = Image.new("RGB", size, "white")
        canvas.paste(img, ((size[0] - img.width) // 2, (size[1] - img.height) // 2))
        return canvas
    canvas = Image.new("RGB", size, "#f3f4f6")
    draw = ImageDraw.Draw(canvas)
    draw.rectangle([0, 0, size[0] - 1, size[1] - 1], outline="#cbd5e1")
    draw.line([0, 0, size[0] - 1, size[1] - 1], fill="#b91c1c", width=3)
    draw.text((size[0] // 2 - 35, size[1] // 2 - 8), "missing", fill="#7f1d1d")
    return canvas


def diff_image(left: Path, right: Path, out: Path, size: tuple[int, int], boost: float = 7.0) -> Path:
    out.parent.mkdir(parents=True, exist_ok=True)
    if not left.exists() or not right.exists():
        image_or_placeholder(Path("__missing__"), size).save(out)
        return out
    a = Image.open(left).convert("RGB").resize(size, Image.Resampling.LANCZOS)
    b = Image.open(right).convert("RGB").resize(size, Image.Resampling.LANCZOS)
    d = ImageChops.difference(a, b)
    d = ImageEnhance.Brightness(d).enhance(boost)
    d = ImageOps.autocontrast(d, cutoff=0.5)
    d.save(out)
    return out


def make_strip(run: dict[str, Any], output_root: Path, quality: dict[str, Any]) -> str:
    strip_dir = output_root / "assets" / "strips"
    diff_dir = output_root / "assets" / "diffs"
    strip_dir.mkdir(parents=True, exist_ok=True)
    size = tuple(quality["image_size"])
    run_slug = slug(f"{run['layer_name']}_{run['prompt']}_{run['image_id']}")
    input_diff = diff_image(run["images"]["perturbed"], run["images"]["original"], diff_dir / f"input_diff_{run_slug}.jpg", size)
    edit_diff = diff_image(run["images"]["perturbed_edit"], run["images"]["clean_edit"], diff_dir / f"edit_diff_{run_slug}.jpg", size)
    cells = [
        ("Original", run["images"]["original"]),
        ("Perturbed", run["images"]["perturbed"]),
        ("Input diff x7", input_diff),
        ("Clean edit", run["images"]["clean_edit"]),
        ("Perturbed edit", run["images"]["perturbed_edit"]),
        ("Edit diff x7", edit_diff),
        ("Combined flow", run["images"]["combined_flow"]),
    ]
    label_h = 34
    canvas = Image.new("RGB", (size[0] * len(cells), size[1] + label_h), "white")
    draw = ImageDraw.Draw(canvas)
    for idx, (label, path) in enumerate(cells):
        x = idx * size[0]
        canvas.paste(image_or_placeholder(path, size), (x, 0))
        draw.text((x + 8, size[1] + 9), label, fill="black")
    ext = str(quality["strip_format"])
    out = strip_dir / f"{run_slug}.{ext}"
    if ext.lower() in {"jpg", "jpeg"}:
        canvas.save(out, quality=int(quality["strip_quality"]), optimize=True)
    else:
        canvas.save(out, optimize=True)
    return relative(out, output_root)


def metric(summary: dict[str, Any], key: str) -> float | None:
    value = to_float(summary.get(key))
    if value is not None:
        return value
    hist = summary.get("_history_final")
    if isinstance(hist, dict):
        return to_float(hist.get(key))
    return None


def per_run_rows(runs: list[dict[str, Any]]) -> list[dict[str, Any]]:
    rows = []
    for run in runs:
        s = run["summary"]
        rows.append(
            {
                "validity": run["validity"],
                "layer_name": run["layer_name"],
                "prompt": run["prompt"],
                "image_id": run["image_id"],
                "initial_Z": s.get("initial_Z"),
                "final_Z": s.get("final_Z"),
                "Z_increase": s.get("Z_increase"),
                "input_ssim": s.get("final_input_ssim"),
                "input_psnr": s.get("final_input_psnr"),
                "input_l2": s.get("final_input_l2"),
                "output_ssim": s.get("output_ssim"),
                "output_l2": s.get("output_l2"),
                "max_disp_px": s.get("final_combined_max_disp_px"),
                "p95_disp_px": s.get("final_combined_p95_disp_px"),
                "foldover": s.get("final_foldover_fraction"),
                "fraction_clamped": s.get("final_fraction_clamped_total"),
                "run_dir": run["run_dir"].as_posix(),
            }
        )
    return sorted(rows, key=lambda r: to_float(r.get("Z_increase")) or -999, reverse=True)


def aggregate_rows(runs: list[dict[str, Any]]) -> list[dict[str, Any]]:
    groups: dict[tuple[str, str], list[dict[str, Any]]] = defaultdict(list)
    for run in runs:
        groups[(run["layer_name"], run["prompt"])].append(run)
    rows = []
    for (layer, prompt), group in sorted(groups.items()):
        def avg(key: str) -> float | None:
            vals = [to_float(r["summary"].get(key)) for r in group]
            vals = [v for v in vals if v is not None]
            return sum(vals) / len(vals) if vals else None

        rows.append(
            {
                "layer_name": layer,
                "prompt": prompt,
                "num_runs": len(group),
                "mean_Z_increase": avg("Z_increase"),
                "max_Z_increase": max((to_float(r["summary"].get("Z_increase")) or 0.0) for r in group),
                "mean_input_ssim": avg("final_input_ssim"),
                "min_input_ssim": min((to_float(r["summary"].get("final_input_ssim")) or 0.0) for r in group),
                "mean_output_ssim": avg("output_ssim"),
                "mean_output_l2": avg("output_l2"),
                "invalid_count": sum(1 for r in group if r["validity"].startswith("invalid")),
            }
        )
    return rows


def matrix_rows(runs: list[dict[str, Any]]) -> list[dict[str, Any]]:
    layers = sorted({r["layer_name"] for r in runs})
    prompts = sorted({r["prompt"] for r in runs})
    images = sorted({r["image_id"] for r in runs})
    iterations = sorted({int(to_float(r["summary"].get("iterations")) or 0) for r in runs})
    return [
        {
            "layers": len(layers),
            "prompts": len(prompts),
            "images": len(images),
            "runs": len(runs),
            "iterations": ", ".join(str(i) for i in iterations),
            "layer_names": ", ".join(layers),
            "prompt_names": ", ".join(prompts),
        }
    ]


def _series(run: dict[str, Any], key: str) -> tuple[list[float], list[float]]:
    xs, ys = [], []
    for row in run["history_rows"]:
        x = to_float(row.get("iteration") or row.get("iter") or row.get("step"))
        y = to_float(row.get(key))
        if x is not None and y is not None:
            xs.append(x)
            ys.append(y)
    return xs, ys


def plot_lines(path: Path, title: str, ylabel: str, runs: list[dict[str, Any]], key: str, quality: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    plt.figure(figsize=(10, 5.2), dpi=int(quality["graph_dpi"]))
    for run in runs:
        xs, ys = _series(run, key)
        if xs:
            label = f"{run['layer_name']} / {run['prompt'].replace('add ', '')} / {run['image_id'].replace('_01','')}"
            plt.plot(xs, ys, linewidth=1.5, label=label)
    plt.title(title)
    plt.xlabel("iteration")
    plt.ylabel(ylabel)
    plt.grid(True, alpha=0.25)
    plt.legend(fontsize=6, ncol=2)
    plt.tight_layout()
    plt.savefig(path)
    plt.close()


def plot_component_lines(path: Path, runs: list[dict[str, Any]], quality: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    keys = ["tps_max_disp", "delaunay_max_disp", "rolling_max_disp", "dct_max_disp", "fft_phase_max", "combined_max_disp_px"]
    by_key: dict[str, list[tuple[float, float]]] = {k: [] for k in keys}
    for run in runs:
        for row in run["history_rows"]:
            x = to_float(row.get("iteration"))
            if x is None:
                continue
            for key in keys:
                y = to_float(row.get(key))
                if y is not None:
                    by_key[key].append((x, y))
    plt.figure(figsize=(9, 5.2), dpi=int(quality["graph_dpi"]))
    for key, pairs in by_key.items():
        if not pairs:
            continue
        grouped: dict[int, list[float]] = defaultdict(list)
        for x, y in pairs:
            grouped[int(x)].append(y)
        xs = sorted(grouped)
        ys = [sum(grouped[x]) / len(grouped[x]) for x in xs]
        plt.plot(xs, ys, linewidth=1.8, label=key)
    plt.title("Mean component magnitude vs iteration")
    plt.xlabel("iteration")
    plt.ylabel("pixels / radians")
    plt.grid(True, alpha=0.25)
    plt.legend(fontsize=8)
    plt.tight_layout()
    plt.savefig(path)
    plt.close()


def plot_scatter(path: Path, runs: list[dict[str, Any]], quality: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    plt.figure(figsize=(7.5, 5.5), dpi=int(quality["graph_dpi"]))
    for run in runs:
        x = to_float(run["summary"].get("final_input_ssim"))
        y = to_float(run["summary"].get("Z_increase"))
        if x is not None and y is not None:
            color = "#dc2626" if run["validity"].startswith("invalid") else "#2563eb"
            plt.scatter([x], [y], s=58, color=color)
            if y > 0.05 or x < 0.8:
                plt.text(x, y, run["image_id"].split("_")[0], fontsize=7)
    plt.title("Z increase vs input SSIM")
    plt.xlabel("final input SSIM")
    plt.ylabel("Z increase")
    plt.grid(True, alpha=0.25)
    plt.tight_layout()
    plt.savefig(path)
    plt.close()


def make_graphs(runs: list[dict[str, Any]], output_root: Path, quality: dict[str, Any]) -> list[dict[str, Any]]:
    graph_dir = output_root / "assets" / "graphs"
    specs = [
        ("Z vs iteration", "Z", "Z", "z_vs_iteration.png"),
        ("Loss vs iteration", "loss", "loss", "loss_vs_iteration.png"),
        ("Input SSIM vs iteration", "input_ssim", "SSIM", "input_ssim_vs_iteration.png"),
        ("Input PSNR vs iteration", "input_psnr", "PSNR", "input_psnr_vs_iteration.png"),
        ("Combined max displacement vs iteration", "combined_max_disp_px", "pixels", "combined_max_disp_vs_iteration.png"),
        ("FFT spatial delta MSE vs iteration", "fft_spatial_delta_mse", "MSE", "fft_delta_mse_vs_iteration.png"),
    ]
    graphs = []
    for title, key, ylabel, name in specs:
        path = graph_dir / name
        plot_lines(path, title, ylabel, runs, key, quality)
        graphs.append({"title": title, "path": relative(path, output_root)})
    component_path = graph_dir / "component_magnitude_vs_iteration.png"
    plot_component_lines(component_path, runs, quality)
    graphs.append({"title": "Component magnitude vs iteration", "path": relative(component_path, output_root)})
    scatter_path = graph_dir / "z_increase_vs_input_ssim.png"
    plot_scatter(scatter_path, runs, quality)
    graphs.append({"title": "Z increase vs input SSIM", "path": relative(scatter_path, output_root)})
    return [{"title": "Cross-run graphs", "graphs": graphs}]


def table_html(rows: list[dict[str, Any]], cols: list[tuple[str, str]]) -> str:
    if not rows:
        return "<p>No rows available.</p>"
    parts = ["<div class='table-wrap'><table><thead><tr>"]
    for _, label in cols:
        parts.append(f"<th>{html.escape(label)}</th>")
    parts.append("</tr></thead><tbody>")
    for row in rows:
        parts.append("<tr>")
        for key, _ in cols:
            value = row.get(key, "")
            parts.append(f"<td>{html.escape(fmt(value) if isinstance(value, (int, float)) else str(value))}</td>")
        parts.append("</tr>")
    parts.append("</tbody></table></div>")
    return "\n".join(parts)


def md_table(rows: list[dict[str, Any]], cols: list[tuple[str, str]]) -> str:
    if not rows:
        return "No rows available.\n"
    lines = ["| " + " | ".join(label for _, label in cols) + " |", "| " + " | ".join("---" for _ in cols) + " |"]
    for row in rows:
        lines.append("| " + " | ".join(str(fmt(row.get(key)) if isinstance(row.get(key), (int, float)) else row.get(key, "")) for key, _ in cols) + " |")
    return "\n".join(lines) + "\n"


def html_escape(value: Any) -> str:
    return html.escape(str(value))


def graph_section_html(section: dict[str, Any]) -> str:
    parts = [f"<div class='graph-section'><h3>{html_escape(section['title'])}</h3><div class='graph-grid'>"]
    for graph in section["graphs"]:
        parts.append(
            f"<figure class='graph'><figcaption>{html_escape(graph['title'])}</figcaption>"
            f"<img src='{html_escape(graph['path'])}' alt='{html_escape(graph['title'])}'></figure>"
        )
    parts.append("</div></div>")
    return "\n".join(parts)


def build_html(data: dict[str, Any]) -> str:
    matrix_cols = [("layers", "layers"), ("prompts", "prompts"), ("images", "images"), ("runs", "runs"), ("iterations", "iterations"), ("layer_names", "layer names")]
    agg_cols = [("layer_name", "layer"), ("prompt", "prompt"), ("num_runs", "runs"), ("mean_Z_increase", "mean dZ"), ("max_Z_increase", "max dZ"), ("mean_input_ssim", "mean input SSIM"), ("min_input_ssim", "min input SSIM"), ("mean_output_ssim", "mean output SSIM"), ("invalid_count", "invalid")]
    per_cols = [("validity", "validity"), ("layer_name", "layer"), ("prompt", "prompt"), ("image_id", "image"), ("Z_increase", "dZ"), ("final_Z", "final Z"), ("input_ssim", "input SSIM"), ("input_psnr", "input PSNR"), ("output_ssim", "output SSIM"), ("output_l2", "output L2"), ("max_disp_px", "max disp")]
    grad_cols = [("layer_name", "layer"), ("mean_initial_Z", "initial Z"), ("mean_final_Z", "final Z"), ("mean_Z_increase", "dZ"), ("mean_final_total_grad_norm", "grad norm"), ("finite_all_rows", "finite")]
    arc_cols = [("baseline_name", "baseline"), ("spearman_cosine_distance_vs_arcface", "Spearman vs ArcFace"), ("pearson_cosine_distance_vs_arcface", "Pearson vs ArcFace")]
    css = """
    :root { --ink:#111827; --muted:#64748b; --line:#d8dee9; --panel:#ffffff; --soft:#f8fafc; --red:#b91c1c; --blue:#2563eb; }
    body { margin:0; font-family: Inter, Segoe UI, Arial, sans-serif; color:var(--ink); background:#f3f6fb; }
    main { max-width:1180px; margin:0 auto; padding:40px 28px 70px; background:white; }
    h1 { font-size:34px; margin:0 0 6px; } h2 { margin-top:34px; border-bottom:2px solid var(--line); padding-bottom:8px; }
    h3 { margin-top:24px; } .subtitle { color:var(--muted); font-size:18px; margin:0 0 4px; } .author { color:var(--muted); margin-top:0; }
    .card { border:1px solid var(--line); border-radius:14px; background:var(--panel); padding:18px; margin:18px 0 26px; box-shadow:0 1px 2px rgba(15,23,42,.04); }
    .callout { border-left:5px solid var(--red); background:#fff7ed; padding:14px 16px; border-radius:10px; }
    code, pre { background:#f1f5f9; border-radius:6px; padding:2px 5px; } pre { padding:12px; overflow:auto; }
    .table-wrap { overflow-x:auto; max-width:100%; margin:12px 0 22px; } table { border-collapse:collapse; width:100%; font-size:12.5px; }
    th,td { border:1px solid var(--line); padding:7px 8px; vertical-align:top; } th { background:var(--soft); text-align:left; }
    .strip { width:100%; border:1px solid var(--line); border-radius:10px; display:block; background:white; }
    .graph-section { border:1px solid var(--line); border-radius:14px; background:var(--panel); padding:18px; margin:18px 0 28px; }
    .graph-grid { display:grid; grid-template-columns:repeat(auto-fit,minmax(430px,1fr)); gap:18px; }
    .graph { border:1px solid var(--line); border-radius:10px; padding:12px; background:white; margin:0; }
    .graph figcaption { font-weight:650; margin:0 0 10px; } .graph img { width:100%; display:block; }
    .path { font-family:Consolas,monospace; font-size:12px; word-break:break-all; color:#334155; }
    @media print { main { max-width:none; } .card { break-inside:avoid; } }
    """
    parts = [
        "<!doctype html><html lang='en'><head><meta charset='utf-8'><meta name='viewport' content='width=device-width, initial-scale=1'>",
        f"<title>{html_escape(TITLE)}</title><style>{css}</style></head><body><main>",
        f"<h1>{html_escape(TITLE)}</h1><p class='subtitle'>{html_escape(SUBTITLE)}</p><p class='author'>{html_escape(AUTHOR)}</p>",
        "<div class='card'><h2>Overview</h2><p>This report summarizes the current Stage-B identity-layer geometry runs. InstructPix2Pix weights are frozen. Only differentiable geometry parameters are optimized.</p></div>",
        "<h2>1. Method</h2>",
        "<p>Stage A ranked internal representations and tested whether identity-sensitive layers produced usable gradients to geometry parameters. Stage B targets selected frozen InstructPix2Pix layers directly.</p>",
        "<pre>Z = 1 - cosine_similarity(pool(layer(original)), pool(layer(perturbed)))\nloss = -Z</pre>",
        "<p>No visual counter-loss was used in these runs. This is important: destructive inputs can score highly if the internal layer distance increases.</p>",
        "<h2>2. Stage-A context</h2>",
        "<h3>Gradient-selected layers</h3>",
        table_html(data["stage_a"].get("gradient_rows", []), grad_cols),
        "<h3>ArcFace correlation context</h3>",
        table_html(data["stage_a"].get("arcface_corr_rows", [])[:10], arc_cols),
        "<h2>3. Run matrix</h2>",
        table_html(data["matrix_rows"], matrix_cols),
        "<h2>4. Aggregate results</h2>",
        table_html(data["aggregate_rows"], agg_cols),
        "<h2>5. Per-run final values</h2>",
        table_html(data["per_run_rows"], per_cols),
        "<div class='callout'><b>Current broad-run conclusion:</b> the highest-Z <code>unet.conv_in</code> runs are invalid as attacks because input preservation collapsed. The green images are caused by FFT phase saturation and DCT/spatial movement under an unconstrained <code>loss = -Z</code> objective.</div>",
        "<h2>6. Image strips</h2>",
    ]
    for run in data["runs_sorted_for_display"]:
        s = run["summary"]
        parts.append(
            "<div class='card'>"
            f"<h3>{html_escape(run['layer_name'])} / {html_escape(run['prompt'])} / {html_escape(run['image_id'])}</h3>"
            f"<p><b>Validity:</b> {html_escape(run['validity'])} &nbsp; "
            f"<b>dZ:</b> {fmt(s.get('Z_increase'))} &nbsp; <b>input SSIM:</b> {fmt(s.get('final_input_ssim'))} &nbsp; "
            f"<b>output SSIM:</b> {fmt(s.get('output_ssim'))}</p>"
            f"<img class='strip' src='{html_escape(run['strip_path'])}' alt='strip'>"
            f"<p class='path'>{html_escape(str(run['run_dir']))}</p></div>"
        )
    parts.append("<h2>7. Graphs</h2>")
    for section in data["graph_sections"]:
        parts.append(graph_section_html(section))
    notes = [
        f"Results folder: {data['results_dir']}",
        f"Runs collected: {len(data['runs'])}",
        f"Missing artifacts: {len(data['missing'])}",
    ]
    parts.append("<h2>8. Notes</h2><ul>" + "".join(f"<li>{html_escape(note)}</li>" for note in notes) + "</ul>")
    parts.append("</main></body></html>")
    return "\n".join(parts)


def build_markdown(data: dict[str, Any]) -> str:
    matrix_cols = [("layers", "layers"), ("prompts", "prompts"), ("images", "images"), ("runs", "runs"), ("iterations", "iterations"), ("layer_names", "layer names")]
    agg_cols = [("layer_name", "layer"), ("prompt", "prompt"), ("num_runs", "runs"), ("mean_Z_increase", "mean dZ"), ("max_Z_increase", "max dZ"), ("mean_input_ssim", "mean input SSIM"), ("invalid_count", "invalid")]
    per_cols = [("validity", "validity"), ("layer_name", "layer"), ("prompt", "prompt"), ("image_id", "image"), ("Z_increase", "dZ"), ("input_ssim", "input SSIM"), ("output_ssim", "output SSIM"), ("max_disp_px", "max disp")]
    lines = [
        f"# {TITLE}",
        "",
        SUBTITLE,
        "",
        f"Author: {AUTHOR}",
        "",
        "## Method",
        "",
        "`Z = 1 - cosine_similarity(pool(layer(original)), pool(layer(perturbed)))`",
        "",
        "`loss = -Z`",
        "",
        "No visual counter-loss was used.",
        "",
        "## Run matrix",
        "",
        md_table(data["matrix_rows"], matrix_cols),
        "## Aggregate results",
        "",
        md_table(data["aggregate_rows"], agg_cols),
        "## Per-run final values",
        "",
        md_table(data["per_run_rows"], per_cols),
        "## Image strips",
        "",
    ]
    for run in data["runs_sorted_for_display"]:
        lines.extend([f"### {run['layer_name']} / {run['prompt']} / {run['image_id']}", "", f"![strip]({run['strip_path']})", ""])
    lines.extend(["## Graphs", ""])
    for section in data["graph_sections"]:
        lines.extend([f"### {section['title']}", ""])
        for graph in section["graphs"]:
            lines.extend([f"#### {graph['title']}", "", f"![{graph['title']}]({graph['path']})", ""])
    return "\n".join(lines)


def make_pdf(data: dict[str, Any], output_root: Path, pdf_path: Path, quality: dict[str, Any]) -> None:
    from reportlab.lib import colors
    from reportlab.lib.pagesizes import letter
    from reportlab.lib.units import inch
    from reportlab.platypus import Image as RLImage
    from reportlab.platypus import KeepTogether, PageBreak, Paragraph, SimpleDocTemplate, Spacer, Table, TableStyle
    from reportlab.lib.styles import getSampleStyleSheet

    styles = getSampleStyleSheet()
    doc = SimpleDocTemplate(str(pdf_path), pagesize=letter, rightMargin=0.42 * inch, leftMargin=0.42 * inch, topMargin=0.45 * inch, bottomMargin=0.45 * inch)
    story: list[Any] = []

    def p(text: str, style: str = "BodyText") -> None:
        story.append(Paragraph(text, styles[style]))
        story.append(Spacer(1, 0.08 * inch))

    def add_table(rows: list[dict[str, Any]], cols: list[tuple[str, str]], font_size: int = 6) -> None:
        if not rows:
            p("No rows available.")
            return
        table_data = [[label for _, label in cols]]
        for row in rows:
            table_data.append([fmt(row.get(key)) if isinstance(row.get(key), (int, float)) else str(row.get(key, "")) for key, _ in cols])
        table = Table(table_data, repeatRows=1)
        table.setStyle(TableStyle([
            ("BACKGROUND", (0, 0), (-1, 0), colors.HexColor("#f1f5f9")),
            ("GRID", (0, 0), (-1, -1), 0.25, colors.HexColor("#cbd5e1")),
            ("FONT", (0, 0), (-1, -1), "Helvetica", font_size),
            ("VALIGN", (0, 0), (-1, -1), "TOP"),
        ]))
        story.append(table)
        story.append(Spacer(1, 0.13 * inch))

    def make_pdf_image(rel_path: str, max_w: float, max_h: float) -> Any | None:
        path = output_root / rel_path
        if not path.exists():
            return None
        pdf_dir = output_root / "assets" / "pdf_images"
        pdf_dir.mkdir(parents=True, exist_ok=True)
        out = pdf_dir / f"{slug(rel_path)}.jpg"
        with Image.open(path) as raw:
            img = raw.convert("RGB")
            img.thumbnail((int(max_w / inch * 155), int(max_h / inch * 155)), Image.Resampling.LANCZOS)
            img.save(out, quality=int(quality["pdf_image_quality"]), optimize=True)
            w, h = img.size
        scale = min(max_w / w, max_h / h)
        return RLImage(str(out), width=w * scale, height=h * scale)

    def pdf_image(rel_path: str, max_w: float, max_h: float) -> None:
        image = make_pdf_image(rel_path, max_w, max_h)
        if image is not None:
            story.append(image)
            story.append(Spacer(1, 0.12 * inch))

    story.append(Paragraph(TITLE, styles["Title"]))
    p(SUBTITLE, "Heading2")
    p(f"Author: {AUTHOR}")
    p("Objective: Z = 1 - cosine_similarity(pool(layer(original)), pool(layer(perturbed))). Loss: loss = -Z. No visual counter-loss was used.")
    p("Run matrix", "Heading2")
    add_table(data["matrix_rows"], [("layers", "layers"), ("prompts", "prompts"), ("images", "images"), ("runs", "runs"), ("iterations", "iterations")], 7)
    p("Aggregate results", "Heading2")
    add_table(data["aggregate_rows"], [("layer_name", "layer"), ("prompt", "prompt"), ("num_runs", "runs"), ("mean_Z_increase", "mean dZ"), ("max_Z_increase", "max dZ"), ("mean_input_ssim", "mean SSIM"), ("min_input_ssim", "min SSIM"), ("invalid_count", "invalid")], 5.5)
    p("Per-run final values", "Heading2")
    add_table(data["per_run_rows"], [("validity", "validity"), ("layer_name", "layer"), ("prompt", "prompt"), ("image_id", "image"), ("Z_increase", "dZ"), ("input_ssim", "SSIM"), ("output_ssim", "out SSIM"), ("max_disp_px", "max disp")], 4.8)
    story.append(PageBreak())
    p("Image strips", "Heading2")
    for run in data["runs_sorted_for_display"]:
        image = make_pdf_image(run["strip_path"], 7.4 * inch, 1.65 * inch)
        block: list[Any] = [
            Paragraph(f"{run['layer_name']} / {run['prompt']} / {run['image_id']} - {run['validity']}", styles["Heading3"]),
            Spacer(1, 0.06 * inch),
        ]
        if image is not None:
            block.extend([image, Spacer(1, 0.12 * inch)])
        story.append(KeepTogether(block))
    story.append(PageBreak())
    p("Graphs", "Heading2")
    for section in data["graph_sections"]:
        p(section["title"], "Heading3")
        for graph in section["graphs"]:
            image = make_pdf_image(graph["path"], 7.2 * inch, 4.2 * inch)
            block = [
                Paragraph(graph["title"], styles["Heading4"]),
                Spacer(1, 0.06 * inch),
            ]
            if image is not None:
                block.extend([image, Spacer(1, 0.12 * inch)])
            story.append(KeepTogether(block))
    doc.build(story)


def build_report(root: Path, results_dir: Path, output_root: Path, compress_report: bool, no_pdf: bool) -> dict[str, Any]:
    quality = quality_settings(compress_report)
    if output_root.exists():
        shutil.rmtree(output_root)
    output_root.mkdir(parents=True, exist_ok=True)
    (output_root / "assets" / "tables").mkdir(parents=True, exist_ok=True)
    runs, missing = collect_runs(results_dir)
    for run in runs:
        run["strip_path"] = make_strip(run, output_root, quality)
    runs_sorted = sorted(runs, key=lambda r: (0 if r["validity"].startswith("invalid") else 1, -(to_float(r["summary"].get("Z_increase")) or 0.0)))
    graphs = make_graphs(runs, output_root, quality)
    data = {
        "results_dir": str(results_dir),
        "runs": runs,
        "runs_sorted_for_display": runs_sorted,
        "missing": missing,
        "stage_a": load_stage_a_context(root),
        "matrix_rows": matrix_rows(runs),
        "aggregate_rows": aggregate_rows(runs),
        "per_run_rows": per_run_rows(runs),
        "graph_sections": graphs,
        "compress_report": compress_report,
    }
    write_csv(output_root / "assets" / "tables" / "run_matrix_summary.csv", data["matrix_rows"])
    write_csv(output_root / "assets" / "tables" / "aggregate_summary.csv", data["aggregate_rows"])
    write_csv(output_root / "assets" / "tables" / "per_run_final_values.csv", data["per_run_rows"])
    (output_root / "missing_artifacts.md").write_text(
        "# Missing artifacts\n\n" + ("\n".join(f"- {m['layer']} / {m['prompt']} / {m['image_id']}: {m['artifact']} ({m['path']})" for m in missing) if missing else "None.\n"),
        encoding="utf-8",
    )
    (output_root / f"{OUTPUT_BASENAME}.html").write_text(build_html(data), encoding="utf-8")
    (output_root / f"{OUTPUT_BASENAME}.md").write_text(build_markdown(data), encoding="utf-8")
    if not no_pdf:
        make_pdf(data, output_root, output_root / f"{OUTPUT_BASENAME}.pdf", quality)
    summary = {
        "title": TITLE,
        "results_dir": str(results_dir),
        "output_root": str(output_root),
        "num_runs": len(runs),
        "num_missing": len(missing),
        "num_invalid": sum(1 for r in runs if r["validity"].startswith("invalid")),
        "html": f"{OUTPUT_BASENAME}.html",
        "markdown": f"{OUTPUT_BASENAME}.md",
        "pdf": None if no_pdf else f"{OUTPUT_BASENAME}.pdf",
        "compress_report": compress_report,
    }
    (output_root / "report_data_summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    return summary


def main() -> None:
    args = parse_args()
    root = args.root.resolve()
    results_dir = resolve_results_dir(root, args.results_dir, args.run_folder)
    output_root = args.output_root or root / "identity_layers" / "outputs" / "reports" / "stage_b_current"
    summary = build_report(root, results_dir, output_root, bool(args.compress_report), bool(args.no_pdf))
    print(f"[layer-report] results: {results_dir}")
    print(f"[layer-report] output: {output_root}")
    print(f"[layer-report] runs: {summary['num_runs']}, invalid: {summary['num_invalid']}, missing: {summary['num_missing']}")
    print(f"[layer-report] html: {output_root / summary['html']}")
    print(f"[layer-report] md: {output_root / summary['markdown']}")
    if summary.get("pdf"):
        print(f"[layer-report] pdf: {output_root / summary['pdf']}")


if __name__ == "__main__":
    main()
