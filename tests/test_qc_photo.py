"""Tests for qc_photo (Step 5)."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np
import pytest
from PIL import Image

from config_loader import DEFAULT_CONFIG
from qc_photo import analyze_photo
from qc_video import QCResult


@pytest.fixture
def config() -> dict[str, Any]:
    return dict(DEFAULT_CONFIG)


def _save_array_as_image(path: Path, arr: np.ndarray) -> None:
    Image.fromarray(arr, mode="RGB").save(path)


def test_photo_qc_structure_and_not_applicable_fields(tmp_path: Path, config: dict[str, Any]) -> None:
    path = tmp_path / "ok.jpg"
    sharp = np.random.default_rng(7).integers(0, 256, (128, 128, 3), dtype=np.uint8)
    _save_array_as_image(path, sharp)

    result: QCResult = analyze_photo(path, config)
    assert set(result.keys()) == {
        "duration_check",
        "blur_check",
        "content_check",
        "saturation_check",
        "entropy_check",
        "exposure_check",
        "shake_check",
        "reasons",
    }
    assert result["duration_check"] == "pass"
    assert result["shake_check"] == "pass"


def test_no_subject_detected_is_rejected(tmp_path: Path, config: dict[str, Any], monkeypatch: pytest.MonkeyPatch) -> None:
    path = tmp_path / "no_subject.jpg"
    sharp = np.random.default_rng(97).integers(0, 256, (160, 160, 3), dtype=np.uint8)
    _save_array_as_image(path, sharp)

    monkeypatch.setattr(
        "qc_photo._subject_detection_status",
        lambda frame, config: ("rejected", ["No subject detected"], []),
    )

    result = analyze_photo(path, config)
    assert result["content_check"] == "rejected"
    assert any("no subject" in reason.lower() for reason in result["reasons"])


def test_person_detected_is_pass(tmp_path: Path, config: dict[str, Any], monkeypatch: pytest.MonkeyPatch) -> None:
    path = tmp_path / "person.jpg"
    sharp = np.random.default_rng(32).integers(0, 256, (160, 160, 3), dtype=np.uint8)
    _save_array_as_image(path, sharp)

    monkeypatch.setattr(
        "qc_photo._subject_detection_status",
        lambda frame, config: (
            "pass",
            [],
            [{"name": "person", "conf": 0.9, "xyxy": (0, 0, 160, 160), "area_ratio": 1.0}],
        ),
    )

    result = analyze_photo(path, config)
    assert result["content_check"] == "pass"
    assert result["blur_check"] == "pass"


def test_center_subject_is_preferred_for_blur_check(tmp_path: Path, config: dict[str, Any], monkeypatch: pytest.MonkeyPatch) -> None:
    path = tmp_path / "center_subject.jpg"
    sharp = np.random.default_rng(11).integers(0, 256, (160, 160, 3), dtype=np.uint8)
    _save_array_as_image(path, sharp)

    selected_boxes: list[tuple[int, int, int, int]] = []

    def fake_subject_detection(frame, config):
        return (
            "pass",
            [],
            [
                {"name": "person", "conf": 0.9, "xyxy": (0, 0, 80, 160), "area_ratio": 0.5},
                {"name": "person", "conf": 0.8, "xyxy": (40, 40, 120, 120), "area_ratio": 0.25},
            ],
        )

    def fake_subject_crop_blur(frame, xyxy):
        selected_boxes.append(xyxy)
        return 100.0

    config["subject_blur_expand_px"] = 0
    monkeypatch.setattr("qc_photo._subject_detection_status", fake_subject_detection)
    monkeypatch.setattr("qc_photo._subject_crop_blur", fake_subject_crop_blur)

    result = analyze_photo(path, config)
    assert result["content_check"] == "pass"
    assert result["blur_check"] == "pass"
    assert selected_boxes == [(40, 40, 120, 120)]


def test_subject_detected_but_subject_crop_is_blurry_rejected(tmp_path: Path, config: dict[str, Any], monkeypatch: pytest.MonkeyPatch) -> None:
    path = tmp_path / "blurry_subject.jpg"
    flat = np.full((160, 160, 3), 128, dtype=np.uint8)
    _save_array_as_image(path, flat)

    monkeypatch.setattr(
        "qc_photo._subject_detection_status",
        lambda frame, config: (
            "pass",
            [],
            [{"name": "person", "conf": 0.9, "xyxy": (0, 0, 160, 160), "area_ratio": 1.0}],
        ),
    )

    result = analyze_photo(path, config)
    assert result["blur_check"] == "rejected"
    assert result["content_check"] == "pass"
    assert any("subject blur" in reason.lower() for reason in result["reasons"])


def test_fallback_object_detected_is_review(tmp_path: Path, config: dict[str, Any], monkeypatch: pytest.MonkeyPatch) -> None:
    path = tmp_path / "fallback.jpg"
    sharp = np.random.default_rng(24).integers(0, 256, (160, 160, 3), dtype=np.uint8)
    _save_array_as_image(path, sharp)

    monkeypatch.setattr(
        "qc_photo._subject_detection_status",
        lambda frame, config: ("review", ["Fallback object detected"], []),
    )

    result = analyze_photo(path, config)
    assert result["content_check"] == "review"
    assert any("fallback object" in reason.lower() for reason in result["reasons"])


def test_blurry_image_with_no_subject_is_rejected(tmp_path: Path, config: dict[str, Any], monkeypatch: pytest.MonkeyPatch) -> None:
    path = tmp_path / "blurry.jpg"
    flat = np.full((160, 160, 3), 128, dtype=np.uint8)
    _save_array_as_image(path, flat)

    monkeypatch.setattr(
        "qc_photo._subject_detection_status",
        lambda frame, config: ("rejected", ["No subject detected"], []),
    )

    result = analyze_photo(path, config)
    assert result["blur_check"] == "rejected"
    assert result["content_check"] == "rejected"


def test_underexposed_image_with_person_detected_is_pass(tmp_path: Path, config: dict[str, Any], monkeypatch: pytest.MonkeyPatch) -> None:
    path = tmp_path / "underexposed.jpg"
    rng = np.random.default_rng(3)
    dark = np.clip(rng.normal(loc=40, scale=40, size=(160, 160, 3)), 0, 255).astype(np.uint8)
    _save_array_as_image(path, dark)

    monkeypatch.setattr(
        "qc_photo._subject_detection_status",
        lambda frame, config: (
            "pass",
            [],
            [{"name": "person", "conf": 0.9, "xyxy": (0, 0, 160, 160), "area_ratio": 1.0}],
        ),
    )

    result = analyze_photo(path, config)
    assert result["content_check"] == "pass"
    assert result["exposure_check"] == "pass"


def test_unreadable_photo_sets_review_for_checks(tmp_path: Path, config: dict[str, Any]) -> None:
    path = tmp_path / "broken.jpg"
    path.write_text("not an image", encoding="utf-8")

    result = analyze_photo(path, config)
    assert result["blur_check"] == "review"
    assert result["content_check"] == "review"
    assert any("cannot open photo" in reason.lower() for reason in result["reasons"])
