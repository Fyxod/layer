"""CLI: run Stage-A Phase A6 baseline comparison."""
from __future__ import annotations

import argparse
from pathlib import Path

from layer.identity_layers.baseline_comparison import run_baseline_comparison


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, default=Path.cwd())
    parser.add_argument("--extraction-dir", type=Path, default=None)
    parser.add_argument("--scores-dir", type=Path, default=None)
    parser.add_argument("--output-dir", type=Path, default=None)
    parser.add_argument("--max-top-layers", type=int, default=8)
    parser.add_argument("--skip-unet-baseline", action="store_true")
    parser.add_argument("--compute-arcface", action="store_true")
    parser.add_argument("--arcface-checkpoint", type=Path, default=None)
    parser.add_argument("--face-repo", type=Path, default=None)
    parser.add_argument("--arcface-device", type=str, default="cuda")
    parser.add_argument("--arcface-fp16", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    extraction_dir = args.extraction_dir or args.root / "identity_layers" / "outputs" / "activation_scan"
    scores_dir = args.scores_dir or args.root / "identity_layers" / "outputs" / "identity_scores"
    output_dir = args.output_dir or args.root / "identity_layers" / "outputs" / "baseline_comparison"
    summary = run_baseline_comparison(
        root=args.root,
        extraction_dir=extraction_dir,
        scores_dir=scores_dir,
        output_dir=output_dir,
        compute_unet_baseline=not args.skip_unet_baseline,
        max_top_layers=args.max_top_layers,
        compute_arcface_baseline=args.compute_arcface,
        arcface_checkpoint=args.arcface_checkpoint,
        face_repo=args.face_repo,
        arcface_device=args.arcface_device,
        arcface_fp16=args.arcface_fp16,
    )
    print(f"Wrote baseline comparison to: {output_dir}")
    print(f"Baselines/layers compared: {summary['num_baselines']}")
    print(f"ArcFace status: {summary['arcface_status']}")


if __name__ == "__main__":
    main()
