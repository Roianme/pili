from __future__ import annotations

from pathlib import Path
from typing import Any, Literal, TypedDict


class ReportEntry(TypedDict, total=False):
    bucket: Literal["clean", "review", "rejected", "burst", "skipped"]
    final_path: str
    original_path: str
    converted_from: str
    metadata: dict[str, str]
    flags: list[str]
    reason: str


class ReportData(TypedDict):
    run_date: str
    source_folder: str
    output_folder: str
    total_files_found: int
    files_processed: int
    files_skipped: int
    skipped_note: str
    converted_counts: dict[str, int]
    results: dict[str, int]
    entries: list[ReportEntry]


SEPARATOR = "=" * 40
DETAIL_INDENT = "".rjust(11)


def write_report(output_folder: Path | str, report_data: ReportData) -> Path:
    output_root = Path(output_folder)
    output_root.mkdir(parents=True, exist_ok=True)
    report_path = output_root / "_report.txt"

    lines: list[str] = []
    lines.append(SEPARATOR)
    lines.append("CLIPSORTER REPORT")
    lines.append(f"Run date: {report_data['run_date']}")
    lines.append(f"Source folder: {report_data['source_folder']}")
    lines.append(f"Output folder: {report_data['output_folder']}")
    lines.append(SEPARATOR)
    lines.append("")
    lines.append("SUMMARY")
    lines.append("-------")
    lines.append(f"Total files found:        {report_data['total_files_found']}")
    lines.append(f"Files processed:          {report_data['files_processed']}")

    skipped_line = f"Files skipped:            {report_data['files_skipped']}"
    skipped_note = report_data.get("skipped_note", "")
    if skipped_note:
        skipped_line += f"  ({skipped_note})"
    lines.append(skipped_line)

    converted = report_data.get("converted_counts", {})
    lines.append(f"Converted to mp4:         {converted.get('mp4', 0)}")
    lines.append(f"Converted to jpg:         {converted.get('jpg', 0)}")
    lines.append(f"Converted to mp3:         {converted.get('mp3', 0)}")
    lines.append("")
    lines.append("Results:")
    results = report_data.get("results", {})
    lines.append(f"  Usable:{results.get('usable', 0):>21}")
    lines.append(f"  Review:{results.get('review', 0):>21}")
    lines.append(f"  Defects:{results.get('defects', 0):>20}")
    lines.append("")
    lines.append(SEPARATOR)
    lines.append("DETAIL LOG")
    lines.append(SEPARATOR)
    lines.append("")

    for entry in report_data.get("entries", []):
        bucket_label = entry.get("bucket", "skipped").upper()
        token = f"[{bucket_label}]"
        padding = " " * max(1, 11 - len(token))
        final_path = entry.get("final_path", "")
        lines.append(f"{token}{padding}{final_path}")

        original = entry.get("original_path")
        if original:
            lines.append(f"{DETAIL_INDENT}Original: {original}")

        converted_from = entry.get("converted_from")
        if converted_from:
            lines.append(f"{DETAIL_INDENT}Converted from: {converted_from}")

        metadata = entry.get("metadata") or {}
        if metadata:
            metadata_items = _format_metadata(metadata)
            if metadata_items:
                lines.append(f"{DETAIL_INDENT}{metadata_items}")

        flags = entry.get("flags") or []
        if flags and bucket_label != "SKIPPED":
            lines.append(f"{DETAIL_INDENT}Flags: {' | '.join(flags)}")

        reason = entry.get("reason")
        if reason:
            if bucket_label == "SKIPPED":
                lines.append(f"{DETAIL_INDENT}Reason: {reason}")
            elif not metadata and not flags:
                lines.append(f"{DETAIL_INDENT}{reason}")

        lines.append("")

    lines.append(SEPARATOR)
    lines.append("END OF REPORT")
    lines.append(SEPARATOR)

    report_path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    return report_path


def _format_metadata(metadata: dict[str, str]) -> str:
    pieces: list[str] = []
    ordered_keys = ["Duration", "Blur", "Exposure", "Shake", "Silence"]
    for key in ordered_keys:
        if key in metadata:
            pieces.append(f"{key}: {metadata[key]}")
    for key, value in metadata.items():
        if key not in ordered_keys:
            pieces.append(f"{key}: {value}")
    return " | ".join(pieces)
