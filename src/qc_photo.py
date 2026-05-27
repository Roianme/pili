"""Photo quality checks: blur, contrast, and exposure."""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Any

import cv2
import numpy as np

from qc_video import QCResult, _laplacian_variance_gray, _mean_brightness_bgr

logger = logging.getLogger(__name__)


def _std_dev_gray(gray: np.ndarray) -> float:
    """Compute standard deviation of pixel values in grayscale image."""
    if gray.size == 0:
        return 0.0
    return float(gray.std())


def _low_content_patch_ratio(gray: np.ndarray, threshold: float, patch_size: int = 16) -> float:
    """Return the ratio of low-variance patches in the image."""
    if gray.size == 0:
        return 0.0

    h, w = gray.shape
    low_count = 0
    total = 0
    for y in range(0, h, patch_size):
        for x in range(0, w, patch_size):
            patch = gray[y : min(y + patch_size, h), x : min(x + patch_size, w)]
            if patch.size == 0:
                continue
            total += 1
            if float(patch.var()) < threshold:
                low_count += 1
    return float(low_count) / total if total else 0.0


def _mean_saturation(frame: np.ndarray) -> float:
    hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
    saturation = hsv[..., 1].astype(np.float32) / 255.0
    return float(saturation.mean())


def _histogram_entropy(gray: np.ndarray) -> float:
    histogram = cv2.calcHist([gray], [0], None, [256], [0, 256]).flatten()
    total = float(histogram.sum())
    if total <= 0.0:
        return 0.0
    probabilities = histogram / total
    probabilities = probabilities[probabilities > 0.0]
    return float(-(probabilities * np.log2(probabilities)).sum())


def analyze_photo(path: Path | str, config: dict[str, Any]) -> QCResult:
    """
    Run photo QC checks.

    duration_check and shake_check are always pass for photos.
    """
    image_path = Path(path)
    reasons: list[str] = []

    frame = cv2.imread(str(image_path), cv2.IMREAD_COLOR)
    if frame is None:
        reasons.append("OpenCV cannot open photo")
        return QCResult(
            duration_check="pass",
            blur_check="review",
            content_check="review",
            saturation_check="review",
            entropy_check="review",
            exposure_check="review",
            shake_check="pass",
            reasons=reasons,
        )

    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    blur_value = _laplacian_variance_gray(gray)
    blur_threshold = float(config["blur_threshold"])
    if blur_value < blur_threshold:
        blur_check = "rejected"
        reasons.append(
            f"Blur: Laplacian variance {blur_value:.2f} below threshold {blur_threshold}",
        )
    else:
        blur_check = "pass"

    contrast_value = _std_dev_gray(gray)
    contrast_threshold = float(config["contrast_threshold"])
    if contrast_value < contrast_threshold:
        blur_check = "rejected"
        reasons.append(
            f"Contrast: standard deviation {contrast_value:.2f} below threshold {contrast_threshold} (featureless image)",
        )

    content_ratio = _low_content_patch_ratio(
        gray,
        float(config["content_variance_threshold"]),
    )
    content_threshold = float(config["content_variance_reject_ratio"])
    if content_ratio >= content_threshold:
        content_check = "rejected"
        reasons.append(
            f"Content: {100.0 * content_ratio:.1f}% of patches low variance below "
            f"{config['content_variance_threshold']}",
        )
    else:
        content_check = "pass"

    mean_saturation = _mean_saturation(frame)
    sat_threshold = float(config["saturation_threshold"])
    sat_reject_ratio = float(config["saturation_reject_ratio"])
    low_saturation_ratio = float(np.mean(cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)[..., 1].astype(np.float32) / 255.0 < sat_threshold))
    if low_saturation_ratio > sat_reject_ratio:
        saturation_check = "rejected"
        reasons.append(
            f"Saturation: {100.0 * low_saturation_ratio:.1f}% of pixels below "
            f"threshold {sat_threshold}",
        )
    else:
        saturation_check = "pass"

    entropy_value = _histogram_entropy(gray)
    entropy_threshold = float(config["histogram_entropy_threshold"])
    if bool(config.get("histogram_entropy_reject", False)) and entropy_value < entropy_threshold:
        entropy_check = "rejected"
        reasons.append(
            f"Entropy: histogram entropy {entropy_value:.2f} below threshold {entropy_threshold}",
        )
    else:
        entropy_check = "pass"

    mean_brightness = _mean_brightness_bgr(frame)
    low = int(config["exposure_low_threshold"])
    high = int(config["exposure_high_threshold"])
    if mean_brightness < low or mean_brightness > high:
        exposure_check = "review"
        reasons.append(
            f"Exposure: mean brightness {mean_brightness:.2f} outside {low}-{high}",
        )
    else:
        exposure_check = "pass"

    return QCResult(
        duration_check="pass",
        blur_check=blur_check,
        content_check=content_check,
        saturation_check=saturation_check,
        entropy_check=entropy_check,
        exposure_check=exposure_check,
        shake_check="pass",
        reasons=reasons,
    )
