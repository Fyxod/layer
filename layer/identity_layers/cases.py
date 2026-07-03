"""MAT image discovery and identity-pair construction."""
from __future__ import annotations

import itertools
from pathlib import Path
from typing import Any

from .io import write_csv, write_json


DEFAULT_FACES = ("face_002", "face_005")
DEFAULT_IMAGE_NAMES = ("instruct_512.png", "master_1024.png", "flux_768.png")


def slugify(value: str) -> str:
    out = []
    for char in value.lower():
        if char.isalnum():
            out.append(char)
        elif char in {" ", "_", "-", "/", "."}:
            out.append("_")
    text = "".join(out)
    while "__" in text:
        text = text.replace("__", "_")
    return text.strip("_")


def build_identity_manifest(
    mat_root: Path,
    output_dir: Path,
    face_ids: list[str] | None = None,
    image_names: list[str] | None = None,
    canonical_only: bool = False,
) -> dict[str, Any]:
    """Build a lightweight identity-probe manifest from MAT data.

    By default this uses all available canonical MAT images per identity
    (`instruct_512`, `master_1024`, `flux_768`) so the development dataset has
    positive same-identity pairs. This is for hook/scoring validation; the
    report explicitly records that these are limited development pairs when no
    richer user-provided dataset is supplied.
    """

    faces = face_ids or list(DEFAULT_FACES)
    names = ["instruct_512.png"] if canonical_only else list(image_names or DEFAULT_IMAGE_NAMES)
    rows: list[dict[str, Any]] = []
    for face_id in faces:
        folder = mat_root / "data" / face_id
        if not folder.exists():
            continue
        for name in names:
            path = folder / name
            if path.exists():
                rows.append(
                    {
                        "image_path": path.as_posix(),
                        "identity_id": face_id,
                        "image_id": f"{face_id}_{slugify(path.stem)}",
                        "source": "mat",
                        "split": "dev",
                        "notes": "auto_manifest_from_mat_face_folder",
                    }
                )
        if not any(row["identity_id"] == face_id for row in rows):
            images = sorted(p for p in folder.iterdir() if p.suffix.lower() in {".png", ".jpg", ".jpeg", ".webp"})
            for path in images[: (1 if canonical_only else 3)]:
                rows.append(
                    {
                        "image_path": path.as_posix(),
                        "identity_id": face_id,
                        "image_id": f"{face_id}_{slugify(path.stem)}",
                        "source": "mat",
                        "split": "dev",
                        "notes": "auto_manifest_fallback_image",
                    }
                )

    # Remove accidental path duplicates while preserving order.
    seen = set()
    deduped = []
    for row in rows:
        key = Path(row["image_path"]).resolve().as_posix().lower()
        if key in seen:
            continue
        seen.add(key)
        deduped.append(row)
    rows = deduped

    same_rows = []
    diff_rows = []
    for left, right in itertools.combinations(rows, 2):
        pair = {
            "left_image_id": left["image_id"],
            "right_image_id": right["image_id"],
            "left_identity_id": left["identity_id"],
            "right_identity_id": right["identity_id"],
            "left_image_path": left["image_path"],
            "right_image_path": right["image_path"],
        }
        if left["identity_id"] == right["identity_id"]:
            same_rows.append({**pair, "pair_type": "same_identity"})
        else:
            diff_rows.append({**pair, "pair_type": "different_identity"})

    identities = sorted({row["identity_id"] for row in rows})
    summary = {
        "num_images": len(rows),
        "num_identities": len(identities),
        "identity_ids": identities,
        "num_same_identity_pairs": len(same_rows),
        "num_different_identity_pairs": len(diff_rows),
        "canonical_only": canonical_only,
        "warning": None,
    }
    if len(rows) < 3 or len(same_rows) == 0 or len(diff_rows) == 0:
        summary["warning"] = "Development manifest is too small for a strong identity conclusion; add more identities/images."

    write_csv(output_dir / "identity_manifest.csv", rows)
    write_csv(output_dir / "same_identity_pairs.csv", same_rows)
    write_csv(output_dir / "different_identity_pairs.csv", diff_rows)
    write_json(output_dir / "dataset_summary.json", summary)
    return {"manifest": rows, "same_pairs": same_rows, "different_pairs": diff_rows, "summary": summary}
