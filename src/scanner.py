"""Recursive folder scan with content-based file type detection."""

from __future__ import annotations

import logging
import os
from pathlib import Path
from typing import Literal, TypedDict

import magic

logger = logging.getLogger(__name__)

DetectedType = Literal["video", "photo", "audio", "unknown"]


class FileRecord(TypedDict):
    original_path: str
    detected_type: Literal["video", "photo", "audio"]
    extension: str
    filename: str


VIDEO_EXTENSIONS: frozenset[str] = frozenset(
    {
        ".mp4",
        ".mov",
        ".mxf",
        ".avi",
        ".mkv",
        ".wmv",
        ".mts",
        ".m2ts",
        ".3gp",
        ".flv",
        ".webm",
        ".ts",
        ".vob",
    }
)

PHOTO_EXTENSIONS: frozenset[str] = frozenset(
    {
        ".jpg",
        ".jpeg",
        ".png",
        ".bmp",
        ".tiff",
        ".tif",
        ".webp",
        ".heic",
        ".heif",
        ".arw",
        ".cr2",
        ".cr3",
        ".nef",
        ".orf",
        ".raf",
        ".dng",
        ".rw2",
    }
)

AUDIO_EXTENSIONS: frozenset[str] = frozenset(
    {
        ".mp3",
        ".wav",
        ".aac",
        ".m4a",
        ".flac",
        ".ogg",
        ".wma",
        ".aiff",
        ".opus",
    }
)

IGNORED_EXTENSIONS: frozenset[str] = frozenset(
    {
        ".pdf",
        ".docx",
        ".xlsx",
        ".txt",
        ".xml",
        ".zip",
        ".exe",
        ".lnk",
    }
)

SUPPORTED_EXTENSIONS: frozenset[str] = (
    VIDEO_EXTENSIONS | PHOTO_EXTENSIONS | AUDIO_EXTENSIONS
)

# MIME types that do not follow image/*, video/*, or audio/* prefixes.
_EXTRA_MIME_TYPES: dict[str, DetectedType] = {
    "application/mp4": "video",
    "application/mxf": "video",
    "application/x-matroska": "video",
    "application/ogg": "audio",
    "application/x-flac": "audio",
    "application/quicktime": "video",
    "video/quicktime": "video",
}

_INCONCLUSIVE_MIMES: frozenset[str] = frozenset(
    {
        "application/octet-stream",
        "binary/octet-stream",
    }
)


def _normalize_extension(path: Path) -> str:
    return path.suffix.lower()


def _extension_type(extension: str) -> DetectedType:
    if extension in VIDEO_EXTENSIONS:
        return "video"
    if extension in PHOTO_EXTENSIONS:
        return "photo"
    if extension in AUDIO_EXTENSIONS:
        return "audio"
    return "unknown"


def _mime_to_type(mime: str) -> DetectedType:
    normalized = mime.split(";")[0].strip().lower()
    if not normalized:
        return "unknown"

    if normalized in _EXTRA_MIME_TYPES:
        return _EXTRA_MIME_TYPES[normalized]

    if normalized.startswith("video/"):
        return "video"
    if normalized.startswith("audio/"):
        return "audio"
    if normalized.startswith("image/"):
        return "photo"

    return "unknown"


def _detect_mime(path: Path) -> str | None:
    try:
        return magic.from_file(str(path), mime=True)
    except Exception as exc:
        logger.warning("Could not read file type for %s: %s", path, exc)
        return None


def classify_file(path: Path) -> DetectedType:
    """
    Classify a file using libmagic, with extension fallback only when
    magic is inconclusive (e.g. RAW photos as application/octet-stream) or
    when MIME is application/mp4 (which could be .m4a audio).
    """
    extension = _normalize_extension(path)

    if extension in IGNORED_EXTENSIONS:
        return "unknown"

    mime = _detect_mime(path)
    if mime:
        detected = _mime_to_type(mime)
        if detected != "unknown":
            return detected

        normalized = mime.split(";")[0].strip().lower()
        # For ambiguous MIME types, fall back to extension
        if normalized in _INCONCLUSIVE_MIMES or normalized == "application/mp4":
            return _extension_type(extension)

        return "unknown"

    if extension in SUPPORTED_EXTENSIONS:
        return _extension_type(extension)

    return "unknown"


def _build_record(path: Path, detected_type: Literal["video", "photo", "audio"]) -> FileRecord:
    return FileRecord(
        original_path=str(path.resolve()),
        detected_type=detected_type,
        extension=_normalize_extension(path),
        filename=path.name,
    )


def scan_folder(target_folder: Path | str) -> list[FileRecord]:
    """
    Recursively scan target_folder and return supported media FileRecords.

    Unknown or unsupported files are skipped and logged.
    """
    root = Path(target_folder)
    if not root.is_dir():
        raise NotADirectoryError(f"Target folder does not exist or is not a directory: {root}")

    records: list[FileRecord] = []

    for dirpath, _, filenames in os.walk(root):
        for name in filenames:
            path = Path(dirpath) / name
            if not path.is_file():
                continue

            extension = _normalize_extension(path)
            detected = classify_file(path)

            if detected == "unknown":
                reason = (
                    f"ignored extension {extension}"
                    if extension in IGNORED_EXTENSIONS
                    else "unsupported or unrecognized type"
                )
                logger.info("Skipping %s (%s)", path, reason)
                continue

            records.append(_build_record(path, detected))

    records.sort(key=lambda record: record["original_path"])
    return records
