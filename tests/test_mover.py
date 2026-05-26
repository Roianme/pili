"""Tests for mover (Step 9)."""

from __future__ import annotations

import shutil
from datetime import datetime
from pathlib import Path

import pytest

from mover import BUCKETS, TYPE_SUBFOLDERS, move_file, setup_output_folder


def test_setup_output_folder_creates_bucket_tree(tmp_path: Path) -> None:
    target = tmp_path / "TargetFolder"
    target.mkdir()

    output = setup_output_folder(target)

    assert output.name == "TargetFolder_sorted"
    assert output.parent == tmp_path.resolve()
    assert (output / "usable" / "videos").is_dir()
    assert (output / "usable" / "photos").is_dir()
    assert (output / "usable" / "audio").is_dir()
    assert (output / "review" / "videos").is_dir()
    assert (output / "review" / "photos").is_dir()
    assert (output / "review" / "audio").is_dir()
    assert (output / "usable" / "burst").is_dir()
    assert (output / "defects" / "videos").is_dir()
    assert (output / "defects" / "photos").is_dir()
    assert (output / "defects" / "audio").is_dir()


def test_setup_output_folder_timestamp_when_exists(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    target = tmp_path / "MyMedia"
    target.mkdir()
    existing = tmp_path / "MyMedia_sorted"
    existing.mkdir()

    fixed = datetime(2026, 5, 17, 14, 32, 1)

    class FixedDatetime(datetime):
        @classmethod
        def now(cls, tz=None):
            return fixed

    monkeypatch.setattr("mover.datetime", FixedDatetime)

    output = setup_output_folder(target)
    assert output.name == "MyMedia_sorted_20260517_143201"
    assert output.is_dir()
    assert (output / "usable" / "videos").is_dir()


def test_move_file_to_correct_bucket_and_type(tmp_path: Path) -> None:
    target = tmp_path / "TargetFolder"
    target.mkdir()
    output = setup_output_folder(target)

    work = tmp_path / "work"
    work.mkdir()
    converted = work / "clip.mp4"
    converted.write_bytes(b"video-bytes")

    final = move_file(converted, "clean", "video", output)
    assert final == output / "usable" / "videos" / "clip.mp4"
    assert final.is_file()
    assert not converted.exists()


def test_move_file_collision_appends_suffix(tmp_path: Path) -> None:
    output = tmp_path / "TargetFolder_sorted"
    (output / "review" / "photos").mkdir(parents=True)

    work = tmp_path / "work"
    work.mkdir()
    existing = output / "review" / "photos" / "shot.jpg"
    existing.write_bytes(b"existing")

    converted = work / "shot.jpg"
    converted.write_bytes(b"new")

    final = move_file(converted, "review", "photo", output)
    assert final.name == "shot_1.jpg"
    assert final.is_file()
    assert existing.read_bytes() == b"existing"


def test_move_file_never_touches_original_target_folder(tmp_path: Path) -> None:
    target = tmp_path / "TargetFolder"
    target.mkdir()
    original = target / "raw.mov"
    original.write_bytes(b"original-content")

    output = setup_output_folder(target)
    work = tmp_path / "clipsorter_work"
    work.mkdir()
    converted = work / "raw.mp4"
    converted.write_bytes(b"converted-content")

    move_file(converted, "rejected", "video", output)

    assert original.is_file()
    assert original.read_bytes() == b"original-content"
    assert (output / "defects" / "videos" / "raw.mp4").is_file()


def test_move_file_all_detected_types(tmp_path: Path) -> None:
    output = tmp_path / "out_sorted"
    for bucket in BUCKETS:
        if bucket == "burst":
            (output / bucket).mkdir(parents=True)
            continue
        for sub in TYPE_SUBFOLDERS.values():
            (output / bucket / sub).mkdir(parents=True)

    work = tmp_path / "work"
    work.mkdir()

    cases = [
        ("a.mp4", "clean", "video", "videos", "usable"),
        ("b.jpg", "review", "photo", "photos", "review"),
        ("c.mp3", "rejected", "audio", "audio", "defects"),
    ]
    for name, bucket, detected_type, sub, folder in cases:
        src = work / name
        src.write_bytes(b"x")
        dest = move_file(src, bucket, detected_type, output)
        assert dest.parent == output / folder / sub


def test_move_file_to_burst_photo_folder(tmp_path: Path) -> None:
    target = tmp_path / "TargetFolder"
    target.mkdir()
    output = setup_output_folder(target)

    work = tmp_path / "work"
    work.mkdir()
    converted = work / "burst.jpg"
    converted.write_bytes(b"image-bytes")

    final = move_file(converted, "burst", "photo", output)
    assert final == output / "usable" / "burst" / "burst.jpg"
    assert final.is_file()


def test_move_file_missing_source_raises(tmp_path: Path) -> None:
    output = tmp_path / "TargetFolder_sorted"
    (output / "usable" / "videos").mkdir(parents=True)
    missing = tmp_path / "work" / "gone.mp4"

    with pytest.raises(FileNotFoundError):
        move_file(missing, "clean", "video", output)


def test_move_file_invalid_bucket_raises(tmp_path: Path) -> None:
    output = tmp_path / "TargetFolder_sorted"
    (output / "usable" / "videos").mkdir(parents=True)
    src = tmp_path / "clip.mp4"
    src.write_bytes(b"x")

    with pytest.raises(ValueError):
        move_file(src, "invalid", "video", output)  # type: ignore[arg-type]
