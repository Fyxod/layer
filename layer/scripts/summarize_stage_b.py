"""CLI: summarize existing Stage-B smoke outputs."""
from __future__ import annotations

import argparse
from pathlib import Path

from layer.identity_layers.stage_b_attack import aggregate_stage_b_outputs


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, default=Path.cwd())
    parser.add_argument("--results-dir", type=Path, default=None)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    results_dir = args.results_dir or args.root / "identity_layers" / "outputs" / "stage_b_smoke"
    summary = aggregate_stage_b_outputs(results_dir)
    print(f"Summarized Stage-B outputs in: {results_dir}")
    print(f"Runs found: {summary.get('num_runs')}")
    print(f"Top sheet: {summary.get('top_sheet')}")
    print(f"Decision report: {summary.get('decision_report')}")


if __name__ == "__main__":
    main()
