"""CLI: build the Milestone 1 identity layer scan report."""
from __future__ import annotations

import argparse
from pathlib import Path

from layer.identity_layers.reporting import build_report


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, default=Path.cwd())
    parser.add_argument("--output-dir", type=Path, default=None)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    output_dir = args.output_dir or args.root / "identity_layers" / "outputs" / "reports"
    summary = build_report(root=args.root, output_dir=output_dir)
    print(f"Report HTML: {summary['report_html']}")
    print(f"Report MD: {summary['report_markdown']}")


if __name__ == "__main__":
    main()
