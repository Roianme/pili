"""Create output folder structure and move converted files from temp work dir."""

from __future__ import annotations

import logging
import shutil
from datetime import datetime
from pathlib import Path

from classifier import Bucket

logger = logging.getLogger(__name__)

BUCKETS: tuple[Bucket, ...] = ("clean", "review", "rejected", "burst")
OUTPUT_BUCKETS: tuple[str, ...] = ("usable", "review", "usable/burst", "defects")
BUCKET_TO_OUTPUT: dict[Bucket, str] = {
    "clean": "usable",
    "review": "review",
    "burst": "usable/burst",
    "rejected": "defects",
}

TYPE_SUBFOLDERS: dict[str, str] = {
    "video": "videos",
    "photo": "photos",
    "audio": "audio",
}


def _type_subfolder(detected_type: str) -> str:
    try:
        return TYPE_SUBFOLDERS[detected_type]
    except KeyError as exc:
        raise ValueError(f"Unsupported detected_type for move: {detected_type}") from exc


def _create_bucket_tree(output_folder: Path) -> None:
    for bucket in OUTPUT_BUCKETS:
        if bucket == "usable/burst":
            (output_folder / bucket).mkdir(parents=True, exist_ok=True)
            continue
        for subfolder in TYPE_SUBFOLDERS.values():
            (output_folder / bucket / subfolder).mkdir(parents=True, exist_ok=True)


def setup_output_folder(target_folder: Path | str) -> Path:
    """
    Create sibling output folder TargetFolder_sorted/ with bucket/type subfolders.

    If that path already exists, append a timestamp suffix per Section 13.
    """
    target = Path(target_folder).resolve()
    base_name = f"{target.name}_sorted"
    parent = target.parent
    output_folder = parent / base_name

    if output_folder.exists():
        stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        output_folder = parent / f"{base_name}_{stamp}"
        logger.warning("Output folder exists; using %s", output_folder)

    output_folder.mkdir(parents=True, exist_ok=True)
    _create_bucket_tree(output_folder)
    return output_folder.resolve()


def _allocate_destination(dest_dir: Path, filename: str) -> Path:
    candidate = dest_dir / filename
    if not candidate.exists():
        return candidate

    path = Path(filename)
    counter = 1
    while True:
        candidate = dest_dir / f"{path.stem}_{counter}{path.suffix}"
        if not candidate.exists():
            return candidate
        counter += 1


def move_file(
    converted_path: Path | str,
    bucket: Bucket,
    detected_type: str,
    output_folder: Path | str,
) -> Path:
    """
    Move a converted file from temp work dir into bucket/type subfolder.

    Never overwrites an existing file; appends _1, _2, ... on collision.
    Does not modify the original TargetFolder.
    """
    if bucket not in BUCKETS:
        raise ValueError(f"Invalid bucket: {bucket}")
    if bucket == "burst" and detected_type != "photo":
        raise ValueError("Burst bucket is only supported for photos")

    source = Path(converted_path)
    if not source.is_file():
        raise FileNotFoundError(f"Converted file not found: {source}")

    root = Path(output_folder).resolve()
    output_bucket = BUCKET_TO_OUTPUT[bucket]
    dest_dir = root / output_bucket
    if output_bucket != "usable/burst":
        dest_dir = dest_dir / _type_subfolder(detected_type)
    dest_dir.mkdir(parents=True, exist_ok=True)

    destination = _allocate_destination(dest_dir, source.name)
    shutil.move(str(source), str(destination))
    logger.info("Moved %s -> %s", source, destination)
    return destination.resolve()
