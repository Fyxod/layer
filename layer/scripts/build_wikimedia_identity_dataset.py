"""Build a small free-media identity probe dataset from Wikimedia Commons.

The script downloads freely licensed/public-domain Commons images, resizes them
to 512x512 with generic letterbox padding (no face detection/alignment), and
writes the manifest/pair files required by the LAYER identity scan.
"""
from __future__ import annotations

import argparse
import csv
import html
import json
import re
import time
import urllib.parse
import urllib.request
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from PIL import Image, ImageDraw, ImageFont

from layer.identity_layers.cases import slugify


COMMONS_API = "https://commons.wikimedia.org/w/api.php"
USER_AGENT = "LAYER-IdentityProbe/0.1 (https://github.com/Fyxod/layer; research dataset builder)"
GLOBAL_REJECT_TITLE_TERMS = (
    "cabinet",
    "family portrait",
    "impersonator",
    "impersonators",
    "wax",
    "statue",
    "views portrait",
    "visitors",
    "gallery visitors",
    "with artistic gymnastic",
    "mural",
    "caricature",
    "cartoon",
    "drawing",
    "signature",
    "logo",
)


@dataclass(frozen=True)
class IdentitySpec:
    identity_id: str
    display_name: str
    required_terms: tuple[str, ...]
    queries: tuple[str, ...]


IDENTITY_SPECS: tuple[IdentitySpec, ...] = (
    IdentitySpec("barack_obama", "Barack Obama", ("obama",), ("Barack Obama portrait", "Barack Obama official portrait", "Barack Obama speaking")),
    IdentitySpec("michelle_obama", "Michelle Obama", ("michelle", "obama"), ("Michelle Obama portrait", "Michelle Obama speaking", "Michelle Obama official")),
    IdentitySpec("joe_biden", "Joe Biden", ("biden",), ("Joe Biden portrait", "Joe Biden official portrait", "Joe Biden speaking")),
    IdentitySpec("kamala_harris", "Kamala Harris", ("kamala", "harris"), ("Kamala Harris portrait", "Kamala Harris official portrait", "Kamala Harris speaking")),
    IdentitySpec("hillary_clinton", "Hillary Clinton", ("hillary", "clinton"), ("Hillary Clinton portrait", "Hillary Clinton speaking", "Hillary Clinton official portrait")),
    IdentitySpec("donald_trump", "Donald Trump", ("trump",), ("Donald Trump portrait", "Donald Trump official portrait", "Donald Trump speaking")),
    IdentitySpec("angela_merkel", "Angela Merkel", ("merkel",), ("Angela Merkel portrait", "Angela Merkel speaking", "Angela Merkel official")),
    IdentitySpec("emmanuel_macron", "Emmanuel Macron", ("macron",), ("Emmanuel Macron portrait", "Emmanuel Macron official", "Emmanuel Macron speaking")),
    IdentitySpec("justin_trudeau", "Justin Trudeau", ("trudeau",), ("Justin Trudeau portrait", "Justin Trudeau official portrait", "Justin Trudeau speaking")),
    IdentitySpec("jacinda_ardern", "Jacinda Ardern", ("ardern",), ("Jacinda Ardern portrait", "Jacinda Ardern official", "Jacinda Ardern speaking")),
    IdentitySpec("narendra_modi", "Narendra Modi", ("modi",), ("Narendra Modi portrait", "Narendra Modi official portrait", "Narendra Modi speaking")),
    IdentitySpec("rishi_sunak", "Rishi Sunak", ("sunak",), ("Rishi Sunak portrait", "Rishi Sunak official", "Rishi Sunak speaking")),
    IdentitySpec("greta_thunberg", "Greta Thunberg", ("greta", "thunberg"), ("Greta Thunberg portrait", "Greta Thunberg speaking", "Greta Thunberg photo")),
    IdentitySpec("malala_yousafzai", "Malala Yousafzai", ("malala", "yousafzai"), ("Malala Yousafzai portrait", "Malala Yousafzai speaking", "Malala Yousafzai photo")),
    IdentitySpec("elon_musk", "Elon Musk", ("musk",), ("Elon Musk portrait", "Elon Musk speaking", "Elon Musk photo")),
    IdentitySpec("tim_cook", "Tim Cook", ("tim cook",), ("Tim Cook portrait", "Tim Cook speaking", "Tim Cook Apple")),
    IdentitySpec("serena_williams", "Serena Williams", ("serena", "williams"), ("Serena Williams portrait", "Serena Williams photo", "Serena Williams speaking")),
    IdentitySpec("roger_federer", "Roger Federer", ("federer",), ("Roger Federer portrait", "Roger Federer photo", "Roger Federer press")),
    IdentitySpec("meryl_streep", "Meryl Streep", ("streep",), ("Meryl Streep portrait", "Meryl Streep photo", "Meryl Streep press")),
    IdentitySpec("tom_hanks", "Tom Hanks", ("hanks",), ("Tom Hanks portrait", "Tom Hanks photo", "Tom Hanks press")),
)


def strip_html(value: str | None) -> str:
    if not value:
        return ""
    value = re.sub(r"<[^>]+>", "", value)
    return html.unescape(value).strip()


def commons_get(params: dict[str, Any]) -> dict[str, Any]:
    query = urllib.parse.urlencode(params)
    request = urllib.request.Request(f"{COMMONS_API}?{query}", headers={"User-Agent": USER_AGENT})
    with urllib.request.urlopen(request, timeout=30) as response:
        return json.loads(response.read().decode("utf-8"))


def search_commons(query: str, limit: int = 30) -> list[dict[str, Any]]:
    payload = commons_get(
        {
            "action": "query",
            "format": "json",
            "generator": "search",
            "gsrnamespace": 6,
            "gsrsearch": query,
            "gsrlimit": limit,
            "prop": "imageinfo",
            "iiprop": "url|mime|size|extmetadata",
            "iiurlwidth": 1024,
        }
    )
    pages = payload.get("query", {}).get("pages", {})
    return sorted(pages.values(), key=lambda page: int(page.get("index", 9999)))


def metadata_value(extmetadata: dict[str, Any], key: str) -> str:
    value = extmetadata.get(key, {})
    if isinstance(value, dict):
        return strip_html(str(value.get("value", "")))
    return strip_html(str(value))


def title_matches(title: str, terms: tuple[str, ...]) -> bool:
    lowered = title.lower()
    if any(term in lowered for term in GLOBAL_REJECT_TITLE_TERMS):
        return False
    return all(term.lower() in lowered for term in terms)


def candidate_from_page(page: dict[str, Any], spec: IdentitySpec) -> dict[str, Any] | None:
    infos = page.get("imageinfo") or []
    if not infos:
        return None
    info = infos[0]
    mime = str(info.get("mime", ""))
    if not mime.startswith("image/") or mime in {"image/svg+xml", "image/gif"}:
        return None
    width = int(info.get("width") or 0)
    height = int(info.get("height") or 0)
    if width < 350 or height < 350:
        return None
    title = str(page.get("title", ""))
    if not title_matches(title, spec.required_terms):
        return None
    ext = info.get("extmetadata", {}) or {}
    return {
        "title": title,
        "source_url": info.get("url"),
        "download_url": info.get("thumburl") or info.get("url"),
        "description_url": info.get("descriptionurl"),
        "mime": mime,
        "width": width,
        "height": height,
        "license_short_name": metadata_value(ext, "LicenseShortName"),
        "license_url": metadata_value(ext, "LicenseUrl"),
        "artist": metadata_value(ext, "Artist"),
        "credit": metadata_value(ext, "Credit"),
        "attribution_required": metadata_value(ext, "AttributionRequired"),
        "usage_terms": metadata_value(ext, "UsageTerms"),
        "image_description": metadata_value(ext, "ImageDescription"),
    }


def download_bytes(url: str) -> bytes:
    request = urllib.request.Request(url, headers={"User-Agent": USER_AGENT})
    with urllib.request.urlopen(request, timeout=45) as response:
        return response.read()


def preprocess_image(raw_path: Path, out_path: Path, size: int) -> None:
    image = Image.open(raw_path).convert("RGB")
    image.thumbnail((size, size), Image.Resampling.LANCZOS)
    canvas = Image.new("RGB", (size, size), (18, 18, 18))
    left = (size - image.width) // 2
    top = (size - image.height) // 2
    canvas.paste(image, (left, top))
    out_path.parent.mkdir(parents=True, exist_ok=True)
    canvas.save(out_path, "JPEG", quality=92, optimize=True)


def write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        path.write_text("", encoding="utf-8")
        return
    fields = list(dict.fromkeys(key for row in rows for key in row.keys()))
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)


def make_pairs(manifest_rows: list[dict[str, Any]]) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    same: list[dict[str, Any]] = []
    diff: list[dict[str, Any]] = []
    for i, left in enumerate(manifest_rows):
        for right in manifest_rows[i + 1 :]:
            pair = {
                "left_image_id": left["image_id"],
                "right_image_id": right["image_id"],
                "left_identity_id": left["identity_id"],
                "right_identity_id": right["identity_id"],
                "left_image_path": left["image_path"],
                "right_image_path": right["image_path"],
            }
            if left["identity_id"] == right["identity_id"]:
                same.append({**pair, "pair_type": "same_identity"})
            else:
                diff.append({**pair, "pair_type": "different_identity"})
    return same, diff


def create_contact_sheet(root: Path, manifest_rows: list[dict[str, Any]], output_path: Path, thumb: int = 120) -> None:
    cols = 6
    rows = (len(manifest_rows) + cols - 1) // cols
    label_h = 34
    sheet = Image.new("RGB", (cols * thumb, rows * (thumb + label_h)), "white")
    draw = ImageDraw.Draw(sheet)
    font = ImageFont.load_default()
    for idx, row in enumerate(manifest_rows):
        image_path = root / row["image_path"]
        if not image_path.exists():
            image_path = Path(row["image_path"])
        try:
            image = Image.open(image_path).convert("RGB")
        except Exception:
            continue
        image.thumbnail((thumb, thumb), Image.Resampling.LANCZOS)
        x = (idx % cols) * thumb
        y = (idx // cols) * (thumb + label_h)
        sheet.paste(image, (x + (thumb - image.width) // 2, y + (thumb - image.height) // 2))
        draw.text((x + 3, y + thumb + 2), row["identity_id"][:18], fill="black", font=font)
        draw.text((x + 3, y + thumb + 15), row["image_id"].split("_")[-1], fill="gray", font=font)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    sheet.save(output_path, "JPEG", quality=88, optimize=True)


def build_dataset(root: Path, output_dir: Path, identities: int, images_per_identity: int, image_size: int) -> dict[str, Any]:
    output_dir.mkdir(parents=True, exist_ok=True)
    raw_dir = output_dir / "raw"
    images_dir = output_dir / "images"
    manifest_rows: list[dict[str, Any]] = []
    attribution_rows: list[dict[str, Any]] = []
    identity_summaries: list[dict[str, Any]] = []

    selected_identity_count = 0
    for spec in IDENTITY_SPECS:
        if selected_identity_count >= identities:
            break
        seen_titles: set[str] = set()
        candidates: list[dict[str, Any]] = []
        for query in spec.queries:
            try:
                for page in search_commons(query):
                    candidate = candidate_from_page(page, spec)
                    if candidate is None or not candidate.get("download_url"):
                        continue
                    if candidate["title"] in seen_titles:
                        continue
                    seen_titles.add(candidate["title"])
                    candidates.append(candidate)
                    if len(candidates) >= images_per_identity:
                        break
            except Exception as exc:
                identity_summaries.append({"identity_id": spec.identity_id, "query": query, "error": repr(exc)})
            if len(candidates) >= images_per_identity:
                break
            time.sleep(0.2)

        downloaded = 0
        for candidate in candidates[:images_per_identity]:
            image_idx = downloaded + 1
            image_id = f"{spec.identity_id}_{image_idx:02d}"
            raw_ext = ".jpg" if "jpeg" in candidate["mime"] or "jpg" in candidate["mime"] else ".png"
            raw_path = raw_dir / spec.identity_id / f"{image_id}{raw_ext}"
            processed_rel = Path("identity_layers") / "datasets" / "wikimedia_identity_probe" / "images" / spec.identity_id / f"{image_id}.jpg"
            processed_path = root / processed_rel
            try:
                raw_path.parent.mkdir(parents=True, exist_ok=True)
                raw_path.write_bytes(download_bytes(candidate["download_url"]))
                preprocess_image(raw_path, processed_path, image_size)
            except Exception as exc:
                identity_summaries.append({"identity_id": spec.identity_id, "title": candidate["title"], "download_error": repr(exc)})
                continue
            manifest_rows.append(
                {
                    "image_path": processed_rel.as_posix(),
                    "identity_id": spec.identity_id,
                    "image_id": image_id,
                    "source": "wikimedia_commons",
                    "split": "probe",
                    "notes": f"free_media_resized_{image_size}_no_face_alignment",
                }
            )
            attribution_rows.append(
                {
                    "identity_id": spec.identity_id,
                    "display_name": spec.display_name,
                    "image_id": image_id,
                    **candidate,
                    "processed_image_path": processed_rel.as_posix(),
                }
            )
            downloaded += 1

        identity_summaries.append(
            {
                "identity_id": spec.identity_id,
                "display_name": spec.display_name,
                "images_selected": downloaded,
                "candidate_titles": [candidate["title"] for candidate in candidates[:images_per_identity]],
            }
        )
        if downloaded >= 2:
            selected_identity_count += 1

    valid_identities = sorted({row["identity_id"] for row in manifest_rows})
    same_pairs, diff_pairs = make_pairs(manifest_rows)
    summary = {
        "dataset_name": "wikimedia_identity_probe",
        "source": "Wikimedia Commons MediaWiki API",
        "license_policy": "Wikimedia Commons accepts freely licensed or public-domain media; per-file license metadata is stored in source_attribution.csv.",
        "image_size": image_size,
        "num_identities": len(valid_identities),
        "num_images": len(manifest_rows),
        "images_per_identity_requested": images_per_identity,
        "num_same_identity_pairs": len(same_pairs),
        "num_different_identity_pairs": len(diff_pairs),
        "identity_ids": valid_identities,
        "identity_summaries": identity_summaries,
        "warning": None if len(valid_identities) >= 10 else "Dataset below preferred 10-30 identities; add more identities/images before final conclusions.",
    }
    write_csv(output_dir / "identity_manifest.csv", manifest_rows)
    write_csv(output_dir / "same_identity_pairs.csv", same_pairs)
    write_csv(output_dir / "different_identity_pairs.csv", diff_pairs)
    write_csv(output_dir / "source_attribution.csv", attribution_rows)
    (output_dir / "dataset_summary.json").write_text(json.dumps(summary, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    create_contact_sheet(root, manifest_rows, output_dir / "dataset_contact_sheet.jpg")
    return summary


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, default=Path.cwd())
    parser.add_argument("--output-dir", type=Path, default=None)
    parser.add_argument("--identities", type=int, default=15)
    parser.add_argument("--images-per-identity", type=int, default=3)
    parser.add_argument("--image-size", type=int, default=512)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    output_dir = args.output_dir or args.root / "identity_layers" / "datasets" / "wikimedia_identity_probe"
    summary = build_dataset(args.root, output_dir, args.identities, args.images_per_identity, args.image_size)
    print(f"Wrote dataset to: {output_dir}")
    print(f"Identities: {summary['num_identities']}")
    print(f"Images: {summary['num_images']}")
    print(f"Same pairs: {summary['num_same_identity_pairs']}")
    print(f"Different pairs: {summary['num_different_identity_pairs']}")


if __name__ == "__main__":
    main()
