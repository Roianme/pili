from __future__ import annotations

import argparse
import logging
import sys
from datetime import datetime
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parent
SRC_DIR = ROOT / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

try:
    from tqdm import tqdm
except ImportError:  # pragma: no cover
    tqdm = None

import classifier
import config_loader
import converter
import duplicate
import mover
import reporter
import scanner
from qc_audio import analyze_audio
from qc_photo import analyze_photo
from qc_video import analyze_video


logger = logging.getLogger(__name__)


def _build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="ClipSorter media sorting pipeline.")
    parser.add_argument("target_folder", help="Path to the source media folder")
    parser.add_argument(
        "--config",
        dest="config_path",
        help="Path to a custom config.json file",
    )
    parser.add_argument(
        "--verbose",
        action="store_true",
        help="Enable verbose output",
    )
    return parser


def _configure_logging(verbose: bool) -> None:
    level = logging.DEBUG if verbose else logging.INFO
    logging.basicConfig(
        level=level,
        format="%(asctime)s %(levelname)s %(name)s: %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
        handlers=[logging.StreamHandler(sys.stdout)],
    )


def _progress(iterable, **kwargs):
    if tqdm is None:
        return iterable
    return tqdm(iterable, **kwargs)


def _relative_path(root: Path, path: Path) -> str:
    try:
        rel = path.relative_to(root)
    except ValueError:
        rel = path
    return f"/{rel.as_posix()}"


def _scan_source_folder(source_folder: Path) -> tuple[list[scanner.FileRecord], list[dict[str, Any]]]:
    supported = scanner.scan_folder(source_folder)
    supported_set = {Path(record["original_path"]).resolve() for record in supported}
    skipped_entries: list[dict[str, Any]] = []

    for path in sorted(source_folder.rglob("*")):
        if not path.is_file():
            continue
        resolved = path.resolve()
        if resolved in supported_set:
            continue
        skipped_entries.append(
            {
                "bucket": "skipped",
                "final_path": _relative_path(source_folder, path),
                "original_path": _relative_path(source_folder, path),
                "reason": "Unsupported file type",
            }
        )

    return supported, skipped_entries


def _summary_text(total: int, processed: int, skipped: int) -> str:
    return f"Scanning files...        {total} files found ({processed} supported, {skipped} skipped)"


def _format_metadata(qc_result: dict[str, Any], detected_type: str) -> dict[str, str]:
    metadata: dict[str, str] = {}
    duration_status = qc_result["duration_check"].upper()
    if duration_status != "PASS":
        duration_reason = next((msg for msg in qc_result["reasons"] if msg.lower().startswith("duration")), None)
        metadata["Duration"] = f"{duration_status} ({duration_reason})" if duration_reason else duration_status
    else:
        metadata["Duration"] = "PASS"

    if detected_type == "video":
        blur_status = qc_result["blur_check"].upper()
        exposure_status = qc_result["exposure_check"].upper()
        shake_status = qc_result["shake_check"].upper()
        blur_reason = next((msg for msg in qc_result["reasons"] if msg.startswith("Blur:")), None)
        exposure_reason = next((msg for msg in qc_result["reasons"] if msg.startswith("Exposure:")), None)
        shake_reason = next((msg for msg in qc_result["reasons"] if msg.startswith("Shake:")), None)
        metadata["Blur"] = blur_status if blur_reason is None else f"{blur_status} ({blur_reason[len('Blur: '):]})"
        metadata["Exposure"] = exposure_status if exposure_reason is None else f"{exposure_status} ({exposure_reason[len('Exposure: '):]})"
        metadata["Shake"] = shake_status if shake_reason is None else f"{shake_status} ({shake_reason[len('Shake: '):]})"
    elif detected_type == "photo":
        blur_status = qc_result["blur_check"].upper()
        exposure_status = qc_result["exposure_check"].upper()
        blur_reason = next((msg for msg in qc_result["reasons"] if msg.startswith("Blur:")), None)
        exposure_reason = next((msg for msg in qc_result["reasons"] if msg.startswith("Exposure:")), None)
        metadata["Blur"] = blur_status if blur_reason is None else f"{blur_status} ({blur_reason[len('Blur: '):]})"
        metadata["Exposure"] = exposure_status if exposure_reason is None else f"{exposure_status} ({exposure_reason[len('Exposure: '):]})"
    elif detected_type == "audio":
        silence_status = qc_result["exposure_check"].upper()
        silence_reason = next((msg for msg in qc_result["reasons"] if msg.startswith("Silence:")), None)
        metadata["Duration"] = metadata["Duration"]
        metadata["Silence"] = silence_status if silence_reason is None else f"{silence_status} ({silence_reason[len('Silence: '):]})"
    return metadata


def _score_photo_for_burst(qc_result: dict[str, Any]) -> tuple[int, int, int]:
    blur_score = 2 if qc_result.get("blur_check") == "pass" else 1
    exposure_score = 2 if qc_result.get("exposure_check") == "pass" else 1
    reason_penalty = len(qc_result.get("reasons", []))
    return (blur_score, exposure_score, -reason_penalty)


def _choose_best_burst_representatives(
    burst_groups: list[dict[str, Any]],
    qc_results: dict[str, dict[str, Any]],
) -> set[str]:
    selected: set[str] = set()
    for group in burst_groups:
        best_file = None
        best_score: tuple[int, int, int] | None = None
        for raw_path in group.get("files", []):
            path = str(Path(raw_path).resolve())
            qc_result = qc_results.get(path, {
                "blur_check": "pass",
                "exposure_check": "pass",
                "reasons": [],
            })
            score = _score_photo_for_burst(qc_result)
            if best_score is None or score > best_score or (score == best_score and Path(path).name < Path(best_file).name):
                best_score = score
                best_file = path
        if best_file:
            selected.add(best_file)
    return selected


def _converted_from_text(record: scanner.FileRecord) -> str:
    extension = record["extension"].lower()
    canonical = {
        "video": ".mp4",
        "audio": ".mp3",
        "photo": ".jpg",
    }
    if extension == canonical[record["detected_type"]]:
        return f"{extension} (no conversion needed)"
    return extension


def _build_report_entries(
    source_folder: Path,
    output_folder: Path,
    converted_records: list[converter.ConvertedFileRecord],
    qc_results: dict[str, dict[str, Any]],
    classifications: dict[str, classifier.ClassifierResult],
    skipped_entries: list[dict[str, Any]],
    moved_paths: dict[str, str],
) -> list[dict[str, Any]]:
    entries: list[dict[str, Any]] = []

    for record in converted_records:
        original = Path(record["original_path"])
        converted_path = record.get("converted_path")
        final_path = moved_paths.get(str(Path(converted_path or "")))
        if final_path is None:
            entries.append(
                {
                    "bucket": "skipped",
                    "final_path": _relative_path(source_folder, original),
                    "original_path": _relative_path(source_folder, original),
                    "reason": "Conversion failed or skipped",
                }
            )
            continue

        qc_result = qc_results.get(str(Path(record["converted_path"])))
        classification = classifications.get(str(Path(record["converted_path"])))
        metadata = _format_metadata(qc_result if qc_result is not None else {
            "duration_check": "pass",
            "blur_check": "pass",
            "exposure_check": "pass",
            "shake_check": "pass",
            "reasons": [],
        }, record["detected_type"])

        entries.append(
            {
                "bucket": classification["bucket"] if classification is not None else "clean",
                "final_path": final_path,
                "original_path": _relative_path(source_folder, original),
                "converted_from": _converted_from_text(record),
                "metadata": metadata,
                "flags": classification["reasons"] if classification is not None else [],
            }
        )

    entries.extend(skipped_entries)
    return entries


def _run_pipeline(target_folder: Path, config_path: Path | None, verbose: bool) -> int:
    _configure_logging(verbose)

    source_folder = target_folder.resolve()
    config = config_loader.load_config(config_path)
    work_dir = converter.get_work_dir()

    print("ClipSorter v1.0")
    print(f"Source: {source_folder}")

    supported_records, unsupported_entries = _scan_source_folder(source_folder)
    total_files_found = sum(1 for path in source_folder.rglob("*") if path.is_file())
    files_processed = len(supported_records)
    files_skipped = len(unsupported_entries)

    output_root = mover.setup_output_folder(source_folder)
    print(f"Output: {output_root}")

    print(_summary_text(total_files_found, files_processed, files_skipped))

    if tqdm is not None:
        for _ in _progress([None], desc="Scanning files", total=1, unit="stage"):
            pass

    converted_records: list[converter.ConvertedFileRecord] = []
    print("Converting formats...", end=" ")
    if tqdm is not None:
        conv_iter = tqdm(supported_records, desc="Converting formats", unit="file", dynamic_ncols=True, ncols=80, ascii=True)
        for record in conv_iter:
            try:
                conv_iter.set_description(f"Converting {Path(record['original_path']).name}")
                converted_records.append(converter.convert_file(record, config, work_dir=work_dir))
            except Exception as exc:
                logger.exception("Conversion failed for %s", record["original_path"])
                unsupported_entries.append(
                    {
                        "bucket": "skipped",
                        "final_path": _relative_path(source_folder, Path(record["original_path"])),
                        "original_path": _relative_path(source_folder, Path(record["original_path"])),
                        "reason": str(exc),
                    }
                )
    else:
        for record in supported_records:
            try:
                converted_records.append(converter.convert_file(record, config, work_dir=work_dir))
            except Exception as exc:
                logger.exception("Conversion failed for %s", record["original_path"])
                unsupported_entries.append(
                    {
                        "bucket": "skipped",
                        "final_path": _relative_path(source_folder, Path(record["original_path"])),
                        "original_path": _relative_path(source_folder, Path(record["original_path"])),
                        "reason": str(exc),
                    }
                )
    print("Done")

    qc_results: dict[str, dict[str, Any]] = {}
    print("Running QC checks...", end=" ")
    if tqdm is not None:
        qc_iter = tqdm(converted_records, desc="Running QC checks", unit="file", dynamic_ncols=True, ncols=80, ascii=True)
        for record in qc_iter:
            converted_path = record.get("converted_path")
            if not converted_path or record.get("skipped"):
                qc_iter.set_description("QC skipped")
                continue
            try:
                qc_iter.set_description(f"QC {Path(converted_path).name}")
                if record["detected_type"] == "video":
                    qc_results[converted_path] = analyze_video(converted_path, config)
                elif record["detected_type"] == "photo":
                    qc_results[converted_path] = analyze_photo(converted_path, config)
                elif record["detected_type"] == "audio":
                    qc_results[converted_path] = analyze_audio(converted_path, config)
                else:
                    qc_results[converted_path] = {
                        "duration_check": "pass",
                        "blur_check": "pass",
                        "exposure_check": "pass",
                        "shake_check": "pass",
                        "reasons": [],
                    }
            except Exception:
                logger.exception("QC failed for %s", converted_path)
                qc_results[converted_path] = {
                    "duration_check": "review",
                    "blur_check": "review",
                    "exposure_check": "review",
                    "shake_check": "review",
                    "reasons": ["QC analysis failed"],
                }
    else:
        for record in converted_records:
            converted_path = record.get("converted_path")
            if not converted_path or record.get("skipped"):
                continue
            try:
                if record["detected_type"] == "video":
                    qc_results[converted_path] = analyze_video(converted_path, config)
                elif record["detected_type"] == "photo":
                    qc_results[converted_path] = analyze_photo(converted_path, config)
                elif record["detected_type"] == "audio":
                    qc_results[converted_path] = analyze_audio(converted_path, config)
                else:
                    qc_results[converted_path] = {
                        "duration_check": "pass",
                        "blur_check": "pass",
                        "exposure_check": "pass",
                        "shake_check": "pass",
                        "reasons": [],
                    }
            except Exception:
                logger.exception("QC failed for %s", converted_path)
                qc_results[converted_path] = {
                    "duration_check": "review",
                    "blur_check": "review",
                    "exposure_check": "review",
                    "shake_check": "review",
                    "reasons": ["QC analysis failed"],
                }
    print("Done")

    photo_paths = [r["converted_path"] for r in converted_records if r.get("converted_path") and r["detected_type"] == "photo"]
    video_paths = [r["converted_path"] for r in converted_records if r.get("converted_path") and r["detected_type"] == "video"]
    audio_paths = [r["converted_path"] for r in converted_records if r.get("converted_path") and r["detected_type"] == "audio"]

    duplicate_pairs = []
    print("Detecting duplicates...", end=" ")
    if tqdm is not None:
        with tqdm(total=1, desc="Detecting duplicates", dynamic_ncols=True, ncols=80, ascii=True) as dup_bar:
            try:
                duplicate_pairs = duplicate.find_duplicates(
                    photo_paths=photo_paths,
                    video_paths=video_paths,
                    audio_paths=audio_paths,
                    config=config,
                )
            except Exception:
                logger.exception("Duplicate detection failed")
            dup_bar.update(1)
    else:
        try:
            duplicate_pairs = duplicate.find_duplicates(
                photo_paths=photo_paths,
                video_paths=video_paths,
                audio_paths=audio_paths,
                config=config,
            )
        except Exception:
            logger.exception("Duplicate detection failed")
    print(f"Done — {len(duplicate_pairs)} duplicate pairs found")

    burst_groups: list[dict[str, Any]] = []
    print("Detecting burst groups...", end=" ")
    if tqdm is not None:
        with tqdm(total=1, desc="Detecting burst groups", dynamic_ncols=True, ncols=80, ascii=True) as burst_bar:
            try:
                burst_groups = duplicate.find_burst_groups(photo_paths, config)
            except Exception:
                logger.exception("Burst detection failed")
            burst_bar.update(1)
    else:
        try:
            burst_groups = duplicate.find_burst_groups(photo_paths, config)
        except Exception:
            logger.exception("Burst detection failed")
    print(f"Done — {len(burst_groups)} burst groups found")

    selected_burst_representatives = _choose_best_burst_representatives(burst_groups, qc_results)

    classifications: dict[str, classifier.ClassifierResult] = {}
    print("Classifying files...", end=" ")
    if tqdm is not None:
        cls_iter = tqdm(converted_records, desc="Classifying files", unit="file", dynamic_ncols=True, ncols=80, ascii=True)
        for record in cls_iter:
            converted_path = record.get("converted_path")
            if not converted_path or record.get("skipped"):
                cls_iter.set_description("Classifying skipped")
                continue
            qc_result = qc_results.get(converted_path)
            if qc_result is None:
                qc_result = {
                    "duration_check": "pass",
                    "blur_check": "pass",
                    "exposure_check": "pass",
                    "shake_check": "pass",
                    "reasons": [],
                }
            try:
                cls_iter.set_description(f"Classifying {Path(converted_path).name}")
                classifications[converted_path] = classifier.classify_file(
                    qc_result,
                    duplicate_pairs,
                    converted_path,
                    config=config,
                    burst_groups=burst_groups,
                )
            except Exception:
                logger.exception("Classification failed for %s", converted_path)
                classifications[converted_path] = {"bucket": "review", "reasons": ["Classification failed"]}
    else:
        for record in converted_records:
            converted_path = record.get("converted_path")
            if not converted_path or record.get("skipped"):
                continue
            qc_result = qc_results.get(converted_path)
            if qc_result is None:
                qc_result = {
                    "duration_check": "pass",
                    "blur_check": "pass",
                    "exposure_check": "pass",
                    "shake_check": "pass",
                    "reasons": [],
                }
            try:
                classifications[converted_path] = classifier.classify_file(
                    qc_result,
                    duplicate_pairs,
                    converted_path,
                    config=config,
                    burst_groups=burst_groups,
                )
            except Exception:
                logger.exception("Classification failed for %s", converted_path)
                classifications[converted_path] = {"bucket": "review", "reasons": ["Classification failed"]}
    print("Done")

    print("Moving files...", end=" ")
    moved_paths: dict[str, str] = {}
    if tqdm is not None:
        mv_iter = tqdm(converted_records, desc="Moving files", unit="file", dynamic_ncols=True, ncols=80, ascii=True)
        for record in mv_iter:
            converted_path = record.get("converted_path")
            if not converted_path or record.get("skipped"):
                mv_iter.set_description("Moving skipped")
                continue
            classification = classifications.get(converted_path)
            if classification is None:
                continue
            try:
                if classification["bucket"] == "burst":
                    # Move selected representative into its QC-derived bucket (clean/review),
                    # otherwise move non-selected burst members into the burst folder.
                    if str(Path(converted_path).resolve()) in selected_burst_representatives:
                        qc_res = qc_results.get(converted_path, {
                            "duration_check": "pass",
                            "blur_check": "pass",
                            "exposure_check": "pass",
                            "shake_check": "pass",
                            "reasons": [],
                        })
                        try:
                            move_bucket = classifier._bucket_from_qc(qc_res)
                        except Exception:
                            move_bucket = "clean"
                        mv_iter.set_description(f"Moving burst representative {Path(converted_path).name} -> {move_bucket}")
                    else:
                        move_bucket = "burst"
                        mv_iter.set_description(f"Moving burst member {Path(converted_path).name} -> burst")
                else:
                    move_bucket = classification["bucket"]
                    mv_iter.set_description(f"Moving {Path(converted_path).name}")
                destination = mover.move_file(
                    converted_path,
                    move_bucket,
                    record["detected_type"],
                    output_root,
                )
                moved_paths[converted_path] = str(destination.relative_to(output_root).as_posix())
            except Exception:
                logger.exception("Moving failed for %s", converted_path)
    else:
        for record in converted_records:
            converted_path = record.get("converted_path")
            if not converted_path or record.get("skipped"):
                continue
            classification = classifications.get(converted_path)
            if classification is None:
                continue
            try:
                if classification["bucket"] == "burst":
                    if str(Path(converted_path).resolve()) in selected_burst_representatives:
                        qc_res = qc_results.get(converted_path, {
                            "duration_check": "pass",
                            "blur_check": "pass",
                            "exposure_check": "pass",
                            "shake_check": "pass",
                            "reasons": [],
                        })
                        try:
                            move_bucket = classifier._bucket_from_qc(qc_res)
                        except Exception:
                            move_bucket = "clean"
                    else:
                        move_bucket = "burst"
                else:
                    move_bucket = classification["bucket"]
                destination = mover.move_file(
                    converted_path,
                    move_bucket,
                    record["detected_type"],
                    output_root,
                )
                moved_paths[converted_path] = str(destination.relative_to(output_root).as_posix())
            except Exception:
                logger.exception("Moving failed for %s", converted_path)
    print("Done")

    report_data = {
        "run_date": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        "source_folder": str(source_folder),
        "output_folder": str(output_root),
        "total_files_found": total_files_found,
        "files_processed": files_processed,
        "files_skipped": files_skipped,
        "skipped_note": "unsupported type" if files_skipped else "",
        "converted_counts": {
            "mp4": sum(1 for r in converted_records if r.get("converted_path") and Path(r["converted_path"]).suffix.lower() == ".mp4"),
            "jpg": sum(1 for r in converted_records if r.get("converted_path") and Path(r["converted_path"]).suffix.lower() in {".jpg", ".jpeg"}),
            "mp3": sum(1 for r in converted_records if r.get("converted_path") and Path(r["converted_path"]).suffix.lower() == ".mp3"),
        },
        "results": {
            "clean": sum(1 for bucket in classifications.values() if bucket["bucket"] == "clean"),
            "review": sum(1 for bucket in classifications.values() if bucket["bucket"] == "review"),
            "burst": sum(1 for bucket in classifications.values() if bucket["bucket"] == "burst"),
            "rejected": sum(1 for bucket in classifications.values() if bucket["bucket"] == "rejected"),
            "usable": sum(1 for final_path in moved_paths.values() if final_path.startswith("usable/")),
            "defects": sum(1 for final_path in moved_paths.values() if final_path.startswith("defects/")),
        },
        "entries": _build_report_entries(
            source_folder,
            output_root,
            converted_records,
            qc_results,
            classifications,
            unsupported_entries,
            moved_paths,
        ),
    }

    print("Writing report...", end=" ")
    if tqdm is not None:
        with tqdm(total=1, desc="Writing report", dynamic_ncols=True, ncols=80, ascii=True) as rep_bar:
            report_path = reporter.write_report(output_root, report_data)
            rep_bar.update(1)
    else:
        report_path = reporter.write_report(output_root, report_data)
    print("Done")

    print("")
    print("========================================")
    print("DONE")
    print(f"  Usable:    {report_data['results']['usable']} files")
    print(f"  Review:    {report_data['results']['review']} files")
    print(f"  Defects:   {report_data['results']['defects']} files")
    print(f"Report saved to: {report_path}")
    print("========================================")

    return 0


def main(argv: list[str] | None = None) -> int:
    parser = _build_arg_parser()
    args = parser.parse_args(argv)

    target_folder = Path(args.target_folder)
    if not target_folder.exists() or not target_folder.is_dir():
        parser.error(f"Target folder does not exist or is not a directory: {target_folder}")

    try:
        return _run_pipeline(target_folder, Path(args.config_path) if args.config_path else None, args.verbose)
    except Exception as exc:
        logger.exception("Unexpected error during pipeline")
        print(f"Error: {exc}")
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
