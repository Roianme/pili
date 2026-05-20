"""Tests for converter (Step 3)."""

from __future__ import annotations

import logging
import shutil
import subprocess
from pathlib import Path
from typing import Any
from unittest.mock import MagicMock

import pytest
from PIL import Image

from config_loader import DEFAULT_CONFIG
from converter import (
    ConvertedFileRecord,
    convert_file,
    get_work_dir,
)
from scanner import FileRecord

MINIMAL_PNG = (
    b"\x89PNG\r\n\x1a\n\x00\x00\x00\rIHDR\x00\x00\x00\x01"
    b"\x00\x00\x00\x01\x08\x02\x00\x00\x00\x90wS\xde"
    b"\x00\x00\x00\x0cIDATx\x9cc\x00\x01\x00\x00\x05\x00\x01\r\n-\xdb"
    b"\x00\x00\x00\x00IEND\xaeB`\x82"
)


@pytest.fixture
def config() -> dict[str, Any]:
    return dict(DEFAULT_CONFIG)


@pytest.fixture
def work_dir(tmp_path: Path) -> Path:
    return tmp_path / "clipsorter_work"


def _record(
    path: Path,
    detected_type: str,
    extension: str | None = None,
) -> FileRecord:
    ext = extension if extension is not None else path.suffix.lower()
    return FileRecord(
        original_path=str(path.resolve()),
        detected_type=detected_type,  # type: ignore[typeddict-item]
        extension=ext,
        filename=path.name,
    )


@pytest.fixture
def require_ffmpeg() -> None:
    if shutil.which("ffmpeg") is None or shutil.which("ffprobe") is None:
        pytest.skip("ffmpeg/ffprobe not in PATH")


def test_get_work_dir_under_system_temp():
    import tempfile

    work = get_work_dir()
    assert work == Path(tempfile.gettempdir()) / "clipsorter_work"
    assert work.is_dir()


def test_canonical_jpg_is_copied(config: dict[str, Any], work_dir: Path, tmp_path: Path):
    source = tmp_path / "photo.jpg"
    Image.new("RGB", (8, 8), color="red").save(source, "JPEG")

    result = convert_file(_record(source, "photo"), config, work_dir=work_dir)

    assert result.get("skipped") is not True
    assert "converted_path" in result
    converted = Path(result["converted_path"])
    assert converted.parent == work_dir
    assert converted.suffix == ".jpg"
    assert converted.read_bytes() == source.read_bytes()


def test_png_converted_to_jpg(config: dict[str, Any], work_dir: Path, tmp_path: Path):
    source = tmp_path / "frame.png"
    source.write_bytes(MINIMAL_PNG)

    result = convert_file(_record(source, "photo"), config, work_dir=work_dir)

    assert "converted_path" in result
    converted = Path(result["converted_path"])
    assert converted.suffix == ".jpg"
    with Image.open(converted) as image:
        assert image.format == "JPEG"


def test_raw_rawpy_failure_uses_pillow_fallback(tmp_path: Path):
    """JPEG bytes in a .arw file: rawpy fails, Pillow fallback succeeds."""
    from converter import _convert_photo_raw

    source = tmp_path / "shot.arw"
    dest = tmp_path / "shot.jpg"
    Image.new("RGB", (4, 4), color="blue").save(source, "JPEG")

    assert _convert_photo_raw(source, dest) is True
    assert dest.exists()
    with Image.open(dest) as image:
        assert image.format == "JPEG"


def test_raw_both_decoders_fail_marks_skipped(
    config: dict[str, Any],
    work_dir: Path,
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
):
    source = tmp_path / "broken.arw"
    source.write_bytes(b"not a raw file")

    monkeypatch.setattr("converter._convert_photo_raw", lambda *_: False)

    result = convert_file(_record(source, "photo", ".arw"), config, work_dir=work_dir)

    assert result.get("skipped") is True
    assert "converted_path" not in result


def test_video_h264_uses_stream_copy(
    config: dict[str, Any],
    work_dir: Path,
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
):
    source = tmp_path / "clip.mov"
    source.write_bytes(b"placeholder")
    commands: list[list[str]] = []

    def fake_run(cmd: list[str]) -> MagicMock:
        commands.append(cmd)
        if cmd[0] == "ffprobe":
            return MagicMock(returncode=0, stdout="h264\n", stderr="")
        Path(cmd[-1]).write_bytes(b"converted-mp4")
        return MagicMock(returncode=0, stdout="", stderr="")

    monkeypatch.setattr("converter._run_command", fake_run)
    monkeypatch.setattr("converter._ffmpeg_available", lambda: True)
    monkeypatch.setattr("converter._ffprobe_available", lambda: True)

    result = convert_file(_record(source, "video", ".mov"), config, work_dir=work_dir)

    assert "converted_path" in result
    ffmpeg_cmds = [cmd for cmd in commands if cmd and cmd[0] == "ffmpeg"]
    assert ffmpeg_cmds
    assert "-c" in ffmpeg_cmds[0] and "copy" in ffmpeg_cmds[0]


def test_video_non_h264_reencodes_with_config_crf(
    config: dict[str, Any],
    work_dir: Path,
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
):
    source = tmp_path / "clip.mkv"
    source.write_bytes(b"placeholder")
    commands: list[list[str]] = []

    def fake_run(cmd: list[str]) -> MagicMock:
        commands.append(cmd)
        if cmd[0] == "ffprobe":
            return MagicMock(returncode=0, stdout="hevc\n", stderr="")
        Path(cmd[-1]).write_bytes(b"converted-mp4")
        return MagicMock(returncode=0, stdout="", stderr="")

    monkeypatch.setattr("converter._run_command", fake_run)
    monkeypatch.setattr("converter._ffmpeg_available", lambda: True)
    monkeypatch.setattr("converter._ffprobe_available", lambda: True)

    convert_file(_record(source, "video", ".mkv"), config, work_dir=work_dir)

    ffmpeg_cmds = [cmd for cmd in commands if cmd and cmd[0] == "ffmpeg"]
    assert ffmpeg_cmds
    joined = " ".join(ffmpeg_cmds[0])
    assert "-crf" in joined
    assert str(config["video_output_crf"]) in joined


def test_ffmpeg_failure_marks_skipped(
    config: dict[str, Any],
    work_dir: Path,
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    caplog: pytest.LogCaptureFixture,
):
    source = tmp_path / "bad.mov"
    source.write_bytes(b"not a real video")

    def fail_ffmpeg(_source: Path, _dest: Path, _config: dict[str, Any]) -> bool:
        return False

    monkeypatch.setattr("converter._convert_video", fail_ffmpeg)

    with caplog.at_level(logging.ERROR):
        result = convert_file(_record(source, "video", ".mov"), config, work_dir=work_dir)

    assert result.get("skipped") is True
    assert "converted_path" not in result


def test_audio_wav_to_mp3(
    config: dict[str, Any],
    work_dir: Path,
    tmp_path: Path,
    require_ffmpeg: None,
):
    source = tmp_path / "tone.wav"
    subprocess.run(
        [
            "ffmpeg",
            "-y",
            "-f",
            "lavfi",
            "-i",
            "sine=frequency=440:duration=0.2",
            str(source),
        ],
        check=True,
        capture_output=True,
    )

    result = convert_file(_record(source, "audio", ".wav"), config, work_dir=work_dir)

    assert "converted_path" in result
    converted = Path(result["converted_path"])
    assert converted.suffix == ".mp3"
    assert converted.stat().st_size > 0


def test_canonical_mp3_is_copied(
    config: dict[str, Any],
    work_dir: Path,
    tmp_path: Path,
    require_ffmpeg: None,
):
    source = tmp_path / "voice.mp3"
    subprocess.run(
        [
            "ffmpeg",
            "-y",
            "-f",
            "lavfi",
            "-i",
            "sine=frequency=220:duration=0.2",
            "-codec:a",
            "libmp3lame",
            str(source),
        ],
        check=True,
        capture_output=True,
    )

    result = convert_file(_record(source, "audio"), config, work_dir=work_dir)

    assert result.get("skipped") is not True
    assert Path(result["converted_path"]).read_bytes() == source.read_bytes()


def test_missing_source_marks_skipped(config: dict[str, Any], work_dir: Path, tmp_path: Path):
    missing = tmp_path / "gone.jpg"
    record = _record(missing, "photo")

    result = convert_file(record, config, work_dir=work_dir)

    assert result.get("skipped") is True


def test_converted_file_record_typing(config: dict[str, Any], work_dir: Path, tmp_path: Path):
    source = tmp_path / "still.png"
    source.write_bytes(MINIMAL_PNG)

    result: ConvertedFileRecord = convert_file(_record(source, "photo"), config, work_dir=work_dir)
    assert result["detected_type"] == "photo"
