"""CLI: run Stage-A Phase A7 gradient sanity scan."""
from __future__ import annotations

import argparse
from pathlib import Path

from layer.identity_layers.gradient_analysis import run_gradient_scan
from layer.identity_layers.io import read_json


def _split_csv(value: str | None) -> list[str] | None:
    if not value:
        return None
    return [part.strip() for part in value.split(",") if part.strip()]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, default=Path.cwd())
    parser.add_argument("--config", type=Path, default=None)
    parser.add_argument("--scores-dir", type=Path, default=None)
    parser.add_argument("--extraction-dir", type=Path, default=None)
    parser.add_argument("--output-dir", type=Path, default=None)
    parser.add_argument("--layers", default=None, help="Comma-separated layer names. Defaults to ranked top candidates.")
    parser.add_argument("--image-ids", default=None, help="Comma-separated image IDs.")
    parser.add_argument("--prompts", default=None, help="Comma-separated prompts. Overrides --prompt and config prompts.")
    parser.add_argument("--objective-variants", default=None, help="Comma-separated objective variants.")
    parser.add_argument("--max-layers", type=int, default=None)
    parser.add_argument("--max-cases", type=int, default=None)
    parser.add_argument("--prompt", default=None)
    parser.add_argument("--timestep-index", type=int, default=None)
    parser.add_argument("--steps", type=int, default=None)
    parser.add_argument("--learning-rate", type=float, default=None)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    config_path = args.config or args.root / "identity_layers" / "configs" / "gradient_scan.json"
    config = read_json(config_path) if config_path.exists() else {}
    scores_dir = args.scores_dir or args.root / "identity_layers" / "outputs" / "identity_scores"
    extraction_dir = args.extraction_dir or args.root / "identity_layers" / "outputs" / "activation_scan"
    output_dir = args.output_dir or args.root / "identity_layers" / "outputs" / "gradient_scan"
    summary = run_gradient_scan(
        root=args.root,
        scores_dir=scores_dir,
        extraction_dir=extraction_dir,
        output_dir=output_dir,
        explicit_layers=_split_csv(args.layers) or config.get("layers"),
        explicit_image_ids=_split_csv(args.image_ids) or config.get("image_ids"),
        prompts=_split_csv(args.prompts) or config.get("prompts"),
        objective_variants=_split_csv(args.objective_variants) or config.get("objective_variants"),
        max_layers=args.max_layers if args.max_layers is not None else int(config.get("max_layers", 5)),
        max_cases=args.max_cases if args.max_cases is not None else int(config.get("max_cases", 3)),
        prompt=args.prompt or config.get("prompt", "add black sunglasses"),
        timestep_index=args.timestep_index if args.timestep_index is not None else int(config.get("timestep_index", 6)),
        steps=args.steps if args.steps is not None else int(config.get("steps", 5)),
        learning_rate=args.learning_rate if args.learning_rate is not None else float(config.get("learning_rate", 0.03)),
        geometry_config=config.get("geometry", {}),
    )
    print(f"Wrote gradient sanity scan to: {output_dir}")
    print(f"Rows: {summary['num_rows']}")
    print(f"Failures: {summary['num_failures']}")
    print(f"Selected layers: {[row['layer_name'] for row in summary['selected_layers']]}")


if __name__ == "__main__":
    main()
