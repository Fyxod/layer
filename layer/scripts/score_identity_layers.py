"""CLI: score same/different identity separation for extracted activations."""
from __future__ import annotations

import argparse
from pathlib import Path

from layer.identity_layers.identity_metrics import run_scoring


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, default=Path.cwd())
    parser.add_argument("--extraction-dir", type=Path, default=None)
    parser.add_argument("--output-dir", type=Path, default=None)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    extraction_dir = args.extraction_dir or args.root / "identity_layers" / "outputs" / "activation_scan"
    output_dir = args.output_dir or args.root / "identity_layers" / "outputs" / "identity_scores"
    summary = run_scoring(extraction_dir=extraction_dir, output_dir=output_dir)
    print(f"Wrote identity scores to: {output_dir}")
    print(f"Pair scores: {summary['num_pair_scores']}")
    print(f"Layer/prompt/timestep rows: {summary['num_layer_prompt_timestep_scores']}")


if __name__ == "__main__":
    main()
