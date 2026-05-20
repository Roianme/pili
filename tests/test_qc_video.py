"""Tests for qc_video (Step 4)."""

from __future__ import annotations

import subprocess
from pathlib import Path
from typing import Any

import numpy as np
import pytest

from config_loader import DEFAULT_CONFIG
from qc_video import QCResult, _ratio_check, analyze_video

RNG = np.random.default_rng(42)


@pytest.fixture
def config() -> dict[str, Any]:
    """Defaults with small sample count for speed."""
    cfg = dict(DEFAULT_CONFIG)
    cfg["frame_sample_count"] = 5
    cfg["min_video_duration_sec"] = 5.0
    return cfg


def _make_mp4_with_ffmpeg(path: Path, duration_sec: float) -> None:
    subprocess.run(
        [
            "ffmpeg",
            "-y",
            "-f",
            "lavfi",
            "-i",
            f"color=c=blue:s=64x64:d={duration_sec}",
            "-c:v",
            "libx264",
            "-pix_fmt",
            "yuv420p",
            str(path),
        ],
        check=True,
        capture_output=True,
    )


@pytest.fixture
def require_fftools() -> None:
    try:
        subprocess.run(["ffprobe", "-version"], check=True, capture_output=True)
        subprocess.run(["ffmpeg", "-version"], check=True, capture_output=True)
    except (subprocess.CalledProcessError, FileNotFoundError):
        pytest.skip("ffmpeg/ffprobe not available")


def test_ratio_check_reject_review_pass() -> None:
    assert _ratio_check(0.71, 0.7, 0.3) == "rejected"
    assert _ratio_check(0.5, 0.7, 0.3) == "review"
    assert _ratio_check(0.2, 0.7, 0.3) == "pass"


def test_short_clip_duration_rejected(tmp_path: Path, config: dict[str, Any], require_fftools: None) -> None:
    path = tmp_path / "short.mp4"
    _make_mp4_with_ffmpeg(path, 2.0)
    result = analyze_video(path, config)
    assert result["duration_check"] == "rejected"
    assert any("below minimum" in r.lower() for r in result["reasons"])


def test_long_clip_duration_pass(tmp_path: Path, config: dict[str, Any], require_fftools: None) -> None:
    path = tmp_path / "long.mp4"
    _make_mp4_with_ffmpeg(path, 6.0)
    result = analyze_video(path, config)
    assert result["duration_check"] == "pass"


def test_opencv_open_failure_sets_visual_review(
    tmp_path: Path,
    config: dict[str, Any],
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    path = tmp_path / "missing.mp4"
    path.write_bytes(b"")

    monkeypatch.setattr(
        "qc_video._run_ffprobe_duration_seconds",
        lambda _p: 10.0,
    )
    monkeypatch.setattr(
        "qc_video._read_sampled_frames",
        lambda *_a, **_k: (None, "OpenCV cannot open video"),
    )

    result = analyze_video(path, config)
    assert result["blur_check"] == "review"
    assert result["exposure_check"] == "review"
    assert result["shake_check"] == "review"
    assert result["duration_check"] == "pass"
    assert any("opencv" in r.lower() or "open" in r.lower() for r in result["reasons"])


def test_blur_mostly_rejected(monkeypatch: pytest.MonkeyPatch, config: dict[str, Any], tmp_path: Path) -> None:
    path = tmp_path / "fake.mp4"
    path.touch()
    monkeypatch.setattr("qc_video._run_ffprobe_duration_seconds", lambda _p: 10.0)

    flat = np.full((100, 100, 3), 128, dtype=np.uint8)
    frames = [flat.copy() for _ in range(5)]

    monkeypatch.setattr("qc_video._read_sampled_frames", lambda *_a, **_k: (frames, None))

    result = analyze_video(path, config)
    assert result["blur_check"] == "rejected"


def test_blur_pass_sharp(monkeypatch: pytest.MonkeyPatch, config: dict[str, Any], tmp_path: Path) -> None:
    path = tmp_path / "fake.mp4"
    path.touch()
    monkeypatch.setattr("qc_video._run_ffprobe_duration_seconds", lambda _p: 10.0)

    frames = []
    for _ in range(5):
        frames.append(RNG.integers(0, 256, (120, 120, 3), dtype=np.uint8))

    monkeypatch.setattr("qc_video._read_sampled_frames", lambda *_a, **_k: (frames, None))

    result = analyze_video(path, config)
    assert result["blur_check"] == "pass"


def test_exposure_rejected_mostly_dark(
    monkeypatch: pytest.MonkeyPatch,
    config: dict[str, Any],
    tmp_path: Path,
) -> None:
    path = tmp_path / "fake.mp4"
    path.touch()
    monkeypatch.setattr("qc_video._run_ffprobe_duration_seconds", lambda _p: 10.0)

    dark = np.zeros((80, 80, 3), dtype=np.uint8)
    frames = [dark.copy() for _ in range(5)]

    monkeypatch.setattr("qc_video._read_sampled_frames", lambda *_a, **_k: (frames, None))

    result = analyze_video(path, config)
    assert result["exposure_check"] == "rejected"


def test_exposure_review_mid_band(
    monkeypatch: pytest.MonkeyPatch,
    config: dict[str, Any],
    tmp_path: Path,
) -> None:
    """Exactly 40% bad frames: above review 0.3, below reject 0.7."""
    path = tmp_path / "fake.mp4"
    path.touch()
    monkeypatch.setattr("qc_video._run_ffprobe_duration_seconds", lambda _p: 10.0)

    dark = np.zeros((80, 80, 3), dtype=np.uint8)
    ok = np.full((80, 80, 3), 128, dtype=np.uint8)
    frames = [dark, dark, ok, ok, ok]

    monkeypatch.setattr("qc_video._read_sampled_frames", lambda *_a, **_k: (frames, None))

    result = analyze_video(path, config)
    assert result["exposure_check"] == "review"


def test_shake_high_flow_is_review_only(
    monkeypatch: pytest.MonkeyPatch,
    config: dict[str, Any],
    tmp_path: Path,
) -> None:
    path = tmp_path / "fake.mp4"
    path.touch()
    monkeypatch.setattr("qc_video._run_ffprobe_duration_seconds", lambda _p: 10.0)

    frames = [np.zeros((64, 64, 3), dtype=np.uint8) for _ in range(3)]
    monkeypatch.setattr("qc_video._read_sampled_frames", lambda *_a, **_k: (frames, None))
    monkeypatch.setattr("qc_video._shake_mean_flow_magnitude", lambda _f: 100.0)

    result = analyze_video(path, config)
    assert result["shake_check"] == "review"
    assert result["shake_check"] != "rejected"


def test_ffprobe_failure_duration_review(
    monkeypatch: pytest.MonkeyPatch,
    config: dict[str, Any],
    tmp_path: Path,
) -> None:
    path = tmp_path / "x.mp4"
    path.touch()
    monkeypatch.setattr("qc_video._run_ffprobe_duration_seconds", lambda _p: None)

    sharp = RNG.integers(0, 256, (64, 64, 3), dtype=np.uint8)
    good_exp = np.full((64, 64, 3), 128, dtype=np.uint8)
    frames = [sharp, good_exp, good_exp, good_exp, good_exp]
    monkeypatch.setattr("qc_video._read_sampled_frames", lambda *_a, **_k: (frames, None))
    monkeypatch.setattr("qc_video._shake_mean_flow_magnitude", lambda _f: 0.1)

    result: QCResult = analyze_video(path, config)
    assert result["duration_check"] == "review"
    assert any("duration" in r.lower() and "ffprobe" in r.lower() for r in result["reasons"])


def test_qc_result_typing(tmp_path: Path, config: dict[str, Any], require_fftools: None) -> None:
    path = tmp_path / "typed.mp4"
    _make_mp4_with_ffmpeg(path, 6.0)
    result: QCResult = analyze_video(path, config)
    for key in ("duration_check", "blur_check", "exposure_check", "shake_check"):
        assert result[key] in ("pass", "review", "rejected")
    assert isinstance(result["reasons"], list)
