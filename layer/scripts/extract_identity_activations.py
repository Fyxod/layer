"""CLI: extract pooled activations for identity layer scoring."""
from __future__ import annotations

import argparse
from pathlib import Path

from layer.identity_layers.activation_extraction import run_extraction
from layer.identity_layers.io import read_json


def _split_csv(value: str | None) -> list[str] | None:
    if not value:
        return None
    return [part.strip() for part in value.split(",") if part.strip()]


def _split_int_csv(value: str | None) -> list[int] | None:
    if not value:
        return None
    return [int(part.strip()) for part in value.split(",") if part.strip()]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, default=Path.cwd())
    parser.add_argument("--mat-root", type=Path, default=Path("/home/interns/Desktop/mat"))
    parser.add_argument("--config", type=Path, default=None)
    parser.add_argument("--inventory-dir", type=Path, default=None)
    parser.add_argument("--output-dir", type=Path, default=None)
    parser.add_argument("--prompts", default=None, help="Comma-separated prompt list.")
    parser.add_argument("--timestep-indices", default=None, help="Comma-separated timestep indices.")
    parser.add_argument("--faces", nargs="*", default=None)
    parser.add_argument("--dataset-manifest", type=Path, default=None)
    parser.add_argument("--layers", default=None, help="Comma-separated module paths.")
    parser.add_argument("--max-layers", type=int, default=None)
    parser.add_argument("--canonical-only", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    config_path = args.config or args.root / "identity_layers" / "configs" / "identity_scan.json"
    config = read_json(config_path) if config_path.exists() else {}
    inventory_dir = args.inventory_dir or args.root / "identity_layers" / "outputs" / "layer_inventory"
    output_dir = args.output_dir or args.root / "identity_layers" / "outputs" / "activation_scan"
    summary = run_extraction(
        root=args.root,
        mat_root=args.mat_root,
        inventory_dir=inventory_dir,
        output_dir=output_dir,
        prompts=_split_csv(args.prompts) or config.get("prompts"),
        timestep_indices=_split_int_csv(args.timestep_indices) or config.get("timestep_indices"),
        face_ids=args.faces or config.get("face_ids"),
        layers=_split_csv(args.layers),
        max_layers=args.max_layers if args.max_layers is not None else config.get("max_layers"),
        canonical_only=bool(args.canonical_only or config.get("canonical_only", False)),
        dataset_manifest=args.dataset_manifest or (Path(config["dataset_manifest"]) if config.get("dataset_manifest") else None),
    )
    print(f"Wrote pooled activations to: {output_dir}")
    print(f"Embeddings: {summary['num_embeddings']}")
    print(f"Layers: {summary['num_layers_requested']}")


if __name__ == "__main__":
    main()
