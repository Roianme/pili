"""Combine QC results and duplicate flags into a final bucket."""

from __future__ import annotations

from pathlib import Path
from typing import Any, Literal, TypedDict

from duplicate import DuplicatePair, format_duplicate_flag
from qc_video import QCLevel, QCResult

Bucket = Literal["clean", "review", "rejected", "burst"]


class ClassifierResult(TypedDict):
    bucket: Bucket
    reasons: list[str]


def _bucket_from_qc(qc_result: QCResult) -> Bucket:
    """Section 5e priority: rejected > review > clean."""
    checks: list[QCLevel] = [
        qc_result["duration_check"],
        qc_result["blur_check"],
        qc_result["content_check"],
        qc_result["saturation_check"],
        qc_result["entropy_check"],
        qc_result["exposure_check"],
        qc_result["shake_check"],
    ]
    if "rejected" in checks:
        return "rejected"
    if "review" in checks:
        return "review"
    return "clean"


def _duplicate_flags(file_path: str | Path, duplicate_pairs: list[DuplicatePair]) -> list[str]:
    resolved = str(Path(file_path).resolve())
    flags: list[str] = []
    for pair in duplicate_pairs:
        file_a = str(Path(pair["file_a"]).resolve())
        file_b = str(Path(pair["file_b"]).resolve())
        if resolved in (file_a, file_b):
            flags.append(format_duplicate_flag(pair, resolved))
    return flags


def _is_in_burst(file_path: str | Path, burst_groups: list[dict[str, Any]] | None) -> bool:
    if not burst_groups:
        return False
    resolved = str(Path(file_path).resolve())
    for group in burst_groups:
        if resolved in group.get("files", []):
            return True
    return False


def classify_file(
    qc_result: QCResult,
    duplicate_pairs: list[DuplicatePair],
    file_path: str | Path,
    config: dict[str, Any] | None = None,
    burst_groups: list[dict[str, Any]] | None = None,
) -> ClassifierResult:
    """
    Classify one file into clean, review, rejected, or burst.

    Duplicate pairs force review unless QC already produced rejected.
    Burst groups take priority over clean/review and duplicate review, but not rejected.
    config is accepted for pipeline consistency; not used in classification logic.
    """
    _ = config

    qc_bucket = _bucket_from_qc(qc_result)
    reasons = list(qc_result["reasons"])
    duplicate_reasons = _duplicate_flags(file_path, duplicate_pairs)
    in_burst = _is_in_burst(file_path, burst_groups)

    if qc_bucket == "rejected":
        bucket: Bucket = "rejected"
        if duplicate_reasons:
            reasons.extend(duplicate_reasons)
    elif in_burst:
        if duplicate_reasons:
            reasons.extend(duplicate_reasons)
        bucket = "burst"
    elif duplicate_reasons:
        reasons.extend(duplicate_reasons)
        bucket = "review"
    else:
        bucket = qc_bucket

    return ClassifierResult(bucket=bucket, reasons=reasons)
