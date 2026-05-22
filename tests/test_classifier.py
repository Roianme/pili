"""Tests for classifier (Step 8)."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import pytest

from classifier import ClassifierResult, classify_file
from config_loader import DEFAULT_CONFIG
from duplicate import DuplicatePair
from qc_video import QCResult


def _qc(
    *,
    duration: str = "pass",
    blur: str = "pass",
    exposure: str = "pass",
    shake: str = "pass",
    reasons: list[str] | None = None,
) -> QCResult:
    return QCResult(
        duration_check=duration,  # type: ignore[typeddict-item]
        blur_check=blur,  # type: ignore[typeddict-item]
        exposure_check=exposure,  # type: ignore[typeddict-item]
        shake_check=shake,  # type: ignore[typeddict-item]
        reasons=reasons or [],
    )


@pytest.fixture
def config() -> dict[str, Any]:
    return dict(DEFAULT_CONFIG)


@pytest.fixture
def file_path(tmp_path: Path) -> Path:
    path = tmp_path / "clip.mp4"
    path.touch()
    return path


def test_all_pass_is_clean(file_path: Path, config: dict[str, Any]) -> None:
    result = classify_file(_qc(), [], file_path, config)
    assert result["bucket"] == "clean"
    assert result["reasons"] == []


def test_one_review_is_review(file_path: Path, config: dict[str, Any]) -> None:
    qc = _qc(shake="review", reasons=["Shake detected"])
    result = classify_file(qc, [], file_path, config)
    assert result["bucket"] == "review"
    assert "Shake detected" in result["reasons"]


def test_one_rejected_is_rejected(file_path: Path, config: dict[str, Any]) -> None:
    qc = _qc(duration="rejected", reasons=["Too short"])
    result = classify_file(qc, [], file_path, config)
    assert result["bucket"] == "rejected"
    assert "Too short" in result["reasons"]


def test_rejected_overrides_review(file_path: Path, config: dict[str, Any]) -> None:
    qc = _qc(duration="rejected", blur="review", reasons=["Short", "Blurry"])
    result = classify_file(qc, [], file_path, config)
    assert result["bucket"] == "rejected"


def test_duplicate_forces_review_when_all_pass(file_path: Path, config: dict[str, Any]) -> None:
    other = file_path.parent / "other.mp4"
    other.touch()
    pair = DuplicatePair(
        file_a=str(file_path.resolve()),
        file_b=str(other.resolve()),
        match_type="video_keyframe",
        confidence=0.85,
    )

    result = classify_file(_qc(), [pair], file_path, config)
    assert result["bucket"] == "review"
    assert any("DUPLICATE of" in reason for reason in result["reasons"])


def test_duplicate_does_not_override_rejected(file_path: Path, config: dict[str, Any]) -> None:
    other = file_path.parent / "other.mp4"
    other.touch()
    pair = DuplicatePair(
        file_a=str(file_path.resolve()),
        file_b=str(other.resolve()),
        match_type="image_hash",
        confidence=2.0,
    )
    qc = _qc(duration="rejected", reasons=["Too short"])

    result = classify_file(qc, [pair], file_path, config)
    assert result["bucket"] == "rejected"
    assert "Too short" in result["reasons"]
    assert any("DUPLICATE of" in reason for reason in result["reasons"])


def test_duplicate_on_other_path_in_pair(file_path: Path, config: dict[str, Any]) -> None:
    other = file_path.parent / "dup.mp4"
    other.touch()
    pair = DuplicatePair(
        file_a=str(other.resolve()),
        file_b=str(file_path.resolve()),
        match_type="audio_fingerprint",
        confidence=0.99,
    )

    result = classify_file(_qc(), [pair], other, config)
    assert result["bucket"] == "review"
    assert any("dup.mp4" not in reason or "DUPLICATE" in reason for reason in result["reasons"])


def test_multiple_review_checks_still_review(file_path: Path, config: dict[str, Any]) -> None:
    qc = _qc(blur="review", exposure="review", reasons=["Blur", "Exposure"])
    result = classify_file(qc, [], file_path, config)
    assert result["bucket"] == "review"
    assert len(result["reasons"]) == 2


def test_classifier_result_typing(file_path: Path, config: dict[str, Any]) -> None:
    result: ClassifierResult = classify_file(_qc(), [], file_path, config)
    assert result["bucket"] in ("clean", "review", "rejected")
    assert isinstance(result["reasons"], list)
