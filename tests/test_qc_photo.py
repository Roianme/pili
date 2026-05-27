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


def test_blur_below_threshold_rejected(tmp_path: Path, config: dict[str, Any]) -> None:
    path = tmp_path / "blurry.jpg"
    flat = np.full((160, 160, 3), 128, dtype=np.uint8)
    _save_array_as_image(path, flat)

    result = analyze_photo(path, config)
    assert result["blur_check"] == "rejected"
    assert any("laplacian variance" in reason.lower() for reason in result["reasons"])


def test_blur_above_threshold_pass(tmp_path: Path, config: dict[str, Any]) -> None:
    path = tmp_path / "sharp.jpg"
    sharp = np.random.default_rng(99).integers(0, 256, (160, 160, 3), dtype=np.uint8)
    _save_array_as_image(path, sharp)

    result = analyze_photo(path, config)
    assert result["blur_check"] == "pass"


def test_exposure_dark_is_review(tmp_path: Path, config: dict[str, Any]) -> None:
    path = tmp_path / "dark.jpg"
    dark = np.zeros((128, 128, 3), dtype=np.uint8)
    _save_array_as_image(path, dark)

    result = analyze_photo(path, config)
    assert result["exposure_check"] == "review"
    assert any("exposure" in reason.lower() for reason in result["reasons"])


def test_exposure_bright_is_review(tmp_path: Path, config: dict[str, Any]) -> None:
    path = tmp_path / "bright.jpg"
    bright = np.full((128, 128, 3), 255, dtype=np.uint8)
    _save_array_as_image(path, bright)

    result = analyze_photo(path, config)
    assert result["exposure_check"] == "review"
    assert any("exposure" in reason.lower() for reason in result["reasons"])


def test_exposure_normal_is_pass(tmp_path: Path, config: dict[str, Any]) -> None:
    path = tmp_path / "normal.jpg"
    normal = np.full((128, 128, 3), 128, dtype=np.uint8)
    _save_array_as_image(path, normal)

    result = analyze_photo(path, config)
    assert result["exposure_check"] == "pass"


def test_contrast_flat_grey_rejected(tmp_path: Path, config: dict[str, Any]) -> None:
    path = tmp_path / "flat_grey.jpg"
    flat = np.full((128, 128, 3), 128, dtype=np.uint8)
    _save_array_as_image(path, flat)

    result = analyze_photo(path, config)
    assert result["blur_check"] == "rejected"
    assert any("contrast" in reason.lower() or "featureless" in reason.lower() for reason in result["reasons"])


def test_contrast_lens_cap_rejected(tmp_path: Path, config: dict[str, Any]) -> None:
    path = tmp_path / "lens_cap.jpg"
    lens_cap = np.full((160, 160, 3), 5, dtype=np.uint8)
    _save_array_as_image(path, lens_cap)

    result = analyze_photo(path, config)
    assert result["blur_check"] == "rejected"
    assert any("contrast" in reason.lower() for reason in result["reasons"])


def test_subjectless_low_contrast_image_rejected(tmp_path: Path, config: dict[str, Any]) -> None:
    path = tmp_path / "subjectless.jpg"
    rng = np.random.default_rng(123)
    base = np.full((160, 160), 120, dtype=np.int16)
    noise = rng.integers(-4, 5, size=base.shape, dtype=np.int16)
    gray = np.clip(base + noise, 0, 255).astype(np.uint8)
    textured = np.dstack([gray] * 3)
    _save_array_as_image(path, textured)

    result = analyze_photo(path, config)
    assert result["blur_check"] == "rejected"
    assert any("contrast" in reason.lower() for reason in result["reasons"])


def test_low_saturation_image_rejected(tmp_path: Path, config: dict[str, Any]) -> None:
    path = tmp_path / "low_sat.jpg"
    gray = np.full((160, 160, 3), 128, dtype=np.uint8)
    _save_array_as_image(path, gray)

    result = analyze_photo(path, config)
    assert result["saturation_check"] == "rejected"
    assert any("saturation" in reason.lower() for reason in result["reasons"])


def test_low_entropy_image_rejected(tmp_path: Path, config: dict[str, Any]) -> None:
    path = tmp_path / "low_entropy.jpg"
    low_entropy = np.full((160, 160, 3), 128, dtype=np.uint8)
    _save_array_as_image(path, low_entropy)

    result = analyze_photo(path, config)
    assert result["entropy_check"] == "rejected"
    assert any("entropy" in reason.lower() for reason in result["reasons"])


def test_contrast_high_variance_passes(tmp_path: Path, config: dict[str, Any]) -> None:
    path = tmp_path / "high_contrast.jpg"
    rng = np.random.default_rng(42)
    varied = rng.integers(0, 256, (160, 160, 3), dtype=np.uint8)
    _save_array_as_image(path, varied)

    result = analyze_photo(path, config)
    assert result["blur_check"] == "pass"
    assert not any("contrast" in reason.lower() for reason in result["reasons"])


def test_unreadable_photo_sets_review_for_checks(tmp_path: Path, config: dict[str, Any]) -> None:
    path = tmp_path / "broken.jpg"
    path.write_text("not an image", encoding="utf-8")

    result = analyze_photo(path, config)
    assert result["blur_check"] == "review"
    assert result["exposure_check"] == "review"
    assert any("cannot open photo" in reason.lower() for reason in result["reasons"])
