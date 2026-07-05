"""CLI: inventory InstructPix2Pix layers for the identity scan."""
from __future__ import annotations

import argparse
from pathlib import Path

from layer.identity_layers.io import read_json
from layer.identity_layers.layer_inventory import run_inventory


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, default=Path.cwd())
    parser.add_argument("--mat-root", type=Path, default=Path("/home/interns/Desktop/mat"))
    parser.add_argument("--config", type=Path, default=None)
    parser.add_argument("--prompt", default=None)
    parser.add_argument("--timestep-index", type=int, default=None)
    parser.add_argument("--faces", nargs="*", default=None)
    parser.add_argument("--dataset-manifest", type=Path, default=None)
    parser.add_argument("--output-dir", type=Path, default=None)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    config_path = args.config or args.root / "identity_layers" / "configs" / "layer_inventory.json"
    config = read_json(config_path) if config_path.exists() else {}
    output_dir = args.output_dir or args.root / "identity_layers" / "outputs" / "layer_inventory"
    payload = run_inventory(
        root=args.root,
        mat_root=args.mat_root,
        output_dir=output_dir,
        prompt=args.prompt or config.get("prompt", "add black sunglasses"),
        timestep_index=args.timestep_index if args.timestep_index is not None else int(config.get("timestep_index", 6)),
        face_ids=args.faces or config.get("face_ids"),
        dataset_manifest=args.dataset_manifest or (Path(config["dataset_manifest"]) if config.get("dataset_manifest") else None),
    )
    print(f"Wrote layer inventory to: {output_dir}")
    print(f"Candidates: {payload['num_candidates']}")
    print(f"Recommended initial layers: {payload['num_recommended_initial_layers']}")


if __name__ == "__main__":
    main()
