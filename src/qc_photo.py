"""Photo quality checks: blur and subject detection."""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Any

import cv2
import numpy as np
from ultralytics import YOLO

from qc_video import QCResult, _laplacian_variance_gray

logger = logging.getLogger(__name__)

_YOLO_MODELS: dict[str, YOLO] = {}


def _load_yolo_model(model_name: str) -> YOLO:
    model_name = str(model_name)
    model = _YOLO_MODELS.get(model_name)
    if model is None:
        model = YOLO(model_name)
        _YOLO_MODELS[model_name] = model
    return model


def _expand_box(x1: int, y1: int, x2: int, y2: int, frame_shape: tuple[int, int, int], expand_px: int) -> tuple[int, int, int, int]:
    height, width = frame_shape[:2]
    return (
        max(0, x1 - expand_px),
        max(0, y1 - expand_px),
        min(width, x2 + expand_px),
        min(height, y2 + expand_px),
    )


def _choose_primary_subject(subjects: list[dict[str, Any]], frame_shape: tuple[int, int, int]) -> dict[str, Any]:
    height, width = frame_shape[:2]
    image_center = np.array([width / 2.0, height / 2.0], dtype=float)
    max_distance = np.linalg.norm(image_center)

    def subject_priority(candidate: dict[str, Any]) -> tuple[float, float, float]:
        x1, y1, x2, y2 = candidate["xyxy"]
        box_center = np.array([(x1 + x2) / 2.0, (y1 + y2) / 2.0], dtype=float)
        distance = np.linalg.norm(box_center - image_center)
        centrality = 1.0 - min(distance / max_distance, 1.0)
        return centrality, candidate["conf"], candidate["area_ratio"]

    return max(subjects, key=subject_priority)


def _subject_detection_status(
    frame: np.ndarray,
    config: dict[str, Any],
) -> tuple[str, list[str], list[dict[str, Any]]]:
    if not bool(config.get("subject_detection_enabled", True)):
        return "pass", [], []

    model_name = config["subject_detection_model"]
    min_confidence = float(config["subject_detection_min_confidence"])
    subject_classes = {str(item).lower() for item in config["subject_detection_classes"]}
    fallback_classes = {str(item).lower() for item in config["subject_detection_fallback_classes"]}
    min_area_ratio = float(config["subject_detection_min_area_ratio"])

    model = _load_yolo_model(model_name)
    results = model.predict(frame, conf=min_confidence, verbose=False)
    if not results:
        return "rejected", ["Subject detection failed to produce results"], []

    result = results[0]
    boxes = result.boxes
    if len(boxes) == 0:
        return "rejected", ["Subject detection found no objects"], []

    image_area = float(frame.shape[0] * frame.shape[1]) or 1.0
    valid_subjects: list[dict[str, Any]] = []
    fallback_found = False

    for class_idx, conf, xyxy in zip(
        boxes.cls.cpu().numpy(),
        boxes.conf.cpu().numpy(),
        boxes.xyxy.cpu().numpy(),
    ):
        class_name = str(result.names.get(int(class_idx), "")).lower()
        x1, y1, x2, y2 = [float(value) for value in xyxy]
        box_area = max(0.0, x2 - x1) * max(0.0, y2 - y1)
        if box_area / image_area < min_area_ratio:
            continue

        if class_name in subject_classes:
            valid_subjects.append(
                {
                    "name": class_name,
                    "conf": float(conf),
                    "xyxy": (int(round(x1)), int(round(y1)), int(round(x2)), int(round(y2))),
                    "area_ratio": box_area / image_area,
                }
            )
        elif class_name in fallback_classes:
            fallback_found = True

    if valid_subjects:
        return "pass", [], valid_subjects
    if fallback_found:
        return "review", [
            "Subject detection found only fallback objects; review required",
        ], []
    return "rejected", ["Subject detection found no valid subject"], []


def _subject_crop_blur(frame: np.ndarray, xyxy: tuple[int, int, int, int]) -> float:
    x1, y1, x2, y2 = xyxy
    crop = frame[y1:y2, x1:x2]
    if crop.size == 0:
        return 0.0
    gray_crop = cv2.cvtColor(crop, cv2.COLOR_BGR2GRAY)
    return _laplacian_variance_gray(gray_crop)


def analyze_photo(path: Path | str | None = None, config: dict[str, Any] | None = None, frame: np.ndarray | None = None) -> QCResult:
    """
    Run photo QC checks.

    duration_check and shake_check are always pass for photos.
    
    Args:
        path: File path to photo (optional if frame is provided)
        config: Configuration dict
        frame: Pre-loaded numpy array (BGR format), skips file reading
    """
    if config is None:
        raise ValueError("config must be provided")
    
    if frame is None:
        if path is None:
            raise ValueError("Either path or frame must be provided")
        image_path = Path(path)
        reasons: list[str] = []
        frame = cv2.imread(str(image_path), cv2.IMREAD_COLOR)
        if frame is None:
            reasons.append("OpenCV cannot open photo")
            return QCResult(
                duration_check="pass",
                blur_check="review",
                content_check="review",
                saturation_check="pass",
                entropy_check="pass",
                exposure_check="pass",
                shake_check="pass",
                reasons=reasons,
            )
    else:
        reasons: list[str] = []

    subject_check, subject_reasons, subject_boxes = _subject_detection_status(frame, config)
    reasons.extend(subject_reasons)

    if subject_check == "pass" and subject_boxes:
        subject = _choose_primary_subject(subject_boxes, frame.shape)
        expand_px = int(config.get("subject_blur_expand_px", 16))
        x1, y1, x2, y2 = _expand_box(
            *subject["xyxy"],
            frame.shape,
            expand_px,
        )
        blur_value = _subject_crop_blur(frame, (x1, y1, x2, y2))
        blur_threshold = float(config.get("subject_blur_threshold", float(config["blur_threshold"])))
        if blur_value < blur_threshold:
            blur_check = "rejected"
            reasons.append(
                f"Subject blur: Laplacian variance {blur_value:.2f} below threshold {blur_threshold}",
            )
        else:
            blur_check = "pass"
    else:
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

    return QCResult(
        duration_check="pass",
        blur_check=blur_check,
        content_check=subject_check,
        saturation_check="pass",
        entropy_check="pass",
        exposure_check="pass",
        shake_check="pass",
        reasons=reasons,
    )
