"""CLI: run Stage-B targeted layer geometry smoke."""
from __future__ import annotations

import argparse
from pathlib import Path

from layer.identity_layers.stage_b_attack import (
    StageBSmokeConfig,
    load_stage_b_config,
    run_stage_b_smoke,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, default=Path.cwd())
    parser.add_argument("--extraction-dir", type=Path, default=None)
    parser.add_argument("--output-dir", type=Path, default=None)
    parser.add_argument("--config", type=Path, default=None)
    parser.add_argument("--layers", nargs="*", default=None)
    parser.add_argument("--prompts", nargs="*", default=None)
    parser.add_argument("--image-ids", nargs="*", default=None)
    parser.add_argument("--iters", type=int, default=None)
    parser.add_argument("--max-cases", type=int, default=None)
    parser.add_argument("--learning-rate", type=float, default=None)
    parser.add_argument("--timestep-index", type=int, default=None)
    parser.add_argument("--skip-final-edits", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    config_path = args.config or args.root / "identity_layers" / "configs" / "stage_b_smoke.json"
    config = load_stage_b_config(config_path)
    if args.layers:
        config.layers = list(args.layers)
    if args.prompts:
        config.prompts = list(args.prompts)
    if args.image_ids:
        config.image_ids = list(args.image_ids)
    if args.iters is not None:
        config.iterations = int(args.iters)
    if args.max_cases is not None:
        config.max_cases = int(args.max_cases)
    if args.learning_rate is not None:
        config.learning_rate = float(args.learning_rate)
    if args.timestep_index is not None:
        config.timestep_index = int(args.timestep_index)
    if args.skip_final_edits:
        config.generate_final_edits = False

    extraction_dir = args.extraction_dir or args.root / "identity_layers" / "outputs" / "activation_scan"
    output_dir = args.output_dir or args.root / "identity_layers" / "outputs" / "stage_b_smoke"
    summary = run_stage_b_smoke(
        root=args.root,
        extraction_dir=extraction_dir,
        output_dir=output_dir,
        config=config,
    )
    print(f"Wrote Stage-B smoke outputs to: {output_dir}")
    print(f"Runs completed: {summary.get('num_runs')}")
    print(f"Failures: {summary.get('num_failures', 0)}")
    print(f"Top sheet: {summary.get('top_sheet')}")
    print(f"Decision report: {summary.get('decision_report')}")


if __name__ == "__main__":
    main()
