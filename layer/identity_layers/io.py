"""Small JSON/CSV/logging helpers for LAYER."""
from __future__ import annotations

import csv
import json
import subprocess
from pathlib import Path
from typing import Any


def write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, sort_keys=True, allow_nan=True) + "\n", encoding="utf-8")


def read_json(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


def write_csv(path: Path, rows: list[dict[str, Any]], fieldnames: list[str] | None = None) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        path.write_text("", encoding="utf-8")
        return
    fields = fieldnames or list(dict.fromkeys(key for row in rows for key in row.keys()))
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)


def read_csv(path: Path) -> list[dict[str, str]]:
    if not path.exists():
        return []
    with path.open(newline="", encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


def append_jsonl(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(payload, sort_keys=True, allow_nan=True) + "\n")


def package_versions() -> dict[str, Any]:
    versions: dict[str, Any] = {}
    for name in ["torch", "diffusers", "transformers", "accelerate", "numpy", "PIL", "matplotlib"]:
        try:
            if name == "PIL":
                import PIL

                versions["pillow"] = PIL.__version__
            else:
                module = __import__(name)
                versions[name] = getattr(module, "__version__", "available")
        except Exception as error:
            versions[name] = f"unavailable: {error!r}"
    try:
        import torch

        versions["cuda_available"] = bool(torch.cuda.is_available())
        versions["cuda_version"] = torch.version.cuda
        versions["gpu_name"] = torch.cuda.get_device_name(0) if torch.cuda.is_available() else None
    except Exception as error:
        versions["torch_runtime_error"] = repr(error)
    try:
        result = subprocess.run(["git", "rev-parse", "HEAD"], capture_output=True, text=True, timeout=5)
        versions["git_commit"] = result.stdout.strip() if result.returncode == 0 else None
    except Exception:
        versions["git_commit"] = None
    return versions
