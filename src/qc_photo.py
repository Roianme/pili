"""Photo quality checks: blur and exposure."""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Any

import cv2

from qc_video import QCResult, _laplacian_variance_gray, _mean_brightness_bgr

logger = logging.getLogger(__name__)


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
        exposure_check=exposure_check,
        shake_check="pass",
        reasons=reasons,
    )
