"""Tests for scanner (Step 2)."""

from __future__ import annotations

import logging
from pathlib import Path

import pytest

from scanner import FileRecord, classify_file, scan_folder

# Minimal valid file headers for libmagic classification.
MINIMAL_JPEG = b"\xff\xd8\xff\xd9"
MINIMAL_PNG = (
    b"\x89PNG\r\n\x1a\n\x00\x00\x00\rIHDR\x00\x00\x00\x01"
    b"\x00\x00\x00\x01\x08\x02\x00\x00\x00\x90wS\xde"
    b"\x00\x00\x00\x0cIDATx\x9cc\x00\x01\x00\x00\x05\x00\x01\r\n-\xdb"
    b"\x00\x00\x00\x00IEND\xaeB`\x82"
)
MINIMAL_MP4 = (
    b"\x00\x00\x00\x18ftypmp41\x00\x00\x00\x00mp41isom"
    b"\x00\x00\x00\x08free"
)
MINIMAL_WAV = (
    b"RIFF$\x00\x00\x00WAVEfmt \x10\x00\x00\x00\x01\x00\x01\x00"
    b"\x44\xac\x00\x00\x88X\x01\x00\x02\x00\x10\x00data\x00\x00\x00\x00"
)
MINIMAL_MP3_FRAME = b"\xff\xfb\x90\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00"


@pytest.fixture
def sample_media_tree(tmp_path: Path) -> Path:
    """Temp folder with mixed supported and unsupported files."""
    root = tmp_path / "TargetFolder"
    subdir = root / "107_FUJI"
    subdir.mkdir(parents=True)

    (root / "photo_001.jpg").write_bytes(MINIMAL_JPEG)
    (root / "photo_002.png").write_bytes(MINIMAL_PNG)
    (subdir / "clip_001.mp4").write_bytes(MINIMAL_MP4)
    (subdir / "clip_renamed.mov").write_bytes(MINIMAL_MP4)
    (root / "brolls" / "broll_001.mp3").parent.mkdir(parents=True, exist_ok=True)
    (root / "brolls" / "broll_001.mp3").write_bytes(MINIMAL_MP3_FRAME)
    (root / "interview.wav").write_bytes(MINIMAL_WAV)
    (root / "documents" / "brief.pdf").parent.mkdir(parents=True, exist_ok=True)
    (root / "documents" / "brief.pdf").write_bytes(b"%PDF-1.4\n%\xe2\xe3\xcf\xd3\n")
    (root / "notes.txt").write_text("shoot list", encoding="utf-8")
    (root / "fake.jpg").write_text("not an image", encoding="utf-8")

    return root


def test_scan_folder_finds_supported_media(sample_media_tree: Path):
    records = scan_folder(sample_media_tree)

    assert len(records) == 6

    by_name = {record["filename"]: record for record in records}
    assert set(by_name) == {
        "photo_001.jpg",
        "photo_002.png",
        "clip_001.mp4",
        "clip_renamed.mov",
        "broll_001.mp3",
        "interview.wav",
    }

    assert by_name["clip_renamed.mov"]["detected_type"] == "video"
    assert by_name["photo_001.jpg"]["detected_type"] == "photo"
    assert by_name["broll_001.mp3"]["detected_type"] == "audio"


def test_scan_folder_file_record_shape(sample_media_tree: Path):
    records = scan_folder(sample_media_tree)
    record = next(r for r in records if r["filename"] == "clip_001.mp4")

    assert set(record.keys()) == {"original_path", "detected_type", "extension", "filename"}
    assert Path(record["original_path"]).is_absolute()
    assert record["extension"] == ".mp4"
    assert record["detected_type"] == "video"


def test_scan_folder_skips_unsupported_and_logs(sample_media_tree: Path, caplog):
    with caplog.at_level(logging.INFO):
        records = scan_folder(sample_media_tree)

    skipped_names = {"brief.pdf", "notes.txt", "fake.jpg"}
    found_names = {record["filename"] for record in records}
    assert skipped_names.isdisjoint(found_names)

    messages = " ".join(record.message for record in caplog.records).lower()
    assert "brief.pdf" in messages
    assert "notes.txt" in messages
    assert "fake.jpg" in messages


def test_scan_folder_recursive_subfolders(sample_media_tree: Path):
    records = scan_folder(sample_media_tree)
    paths = [record["original_path"] for record in records]

    assert any("107_FUJI" in path for path in paths)
    assert any("brolls" in path for path in paths)


def test_classify_mp4_with_mov_extension(sample_media_tree: Path):
    mov_path = sample_media_tree / "107_FUJI" / "clip_renamed.mov"
    assert classify_file(mov_path) == "video"


def test_classify_ignored_extension_without_reading_magic(tmp_path: Path, monkeypatch):
    path = tmp_path / "brief.pdf"
    path.write_bytes(b"%PDF-1.4\n")

    def fail_magic(_path: str, mime: bool = True) -> str:
        raise AssertionError("magic should not be called for ignored extensions")

    monkeypatch.setattr("scanner.magic.from_file", fail_magic)
    assert classify_file(path) == "unknown"


def test_scan_folder_invalid_path_raises(tmp_path: Path):
    missing = tmp_path / "does_not_exist"
    with pytest.raises(NotADirectoryError):
        scan_folder(missing)


def test_file_record_typing(sample_media_tree: Path):
    records: list[FileRecord] = scan_folder(sample_media_tree)
    assert records
