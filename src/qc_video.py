"""Video quality checks: duration, blur, exposure, shake."""

from __future__ import annotations

import logging
import subprocess
from pathlib import Path
from typing import Any, Literal, TypedDict

import cv2
import numpy as np

logger = logging.getLogger(__name__)

QCLevel = Literal["pass", "review", "rejected"]


class QCResult(TypedDict):
    duration_check: QCLevel
    blur_check: QCLevel
    exposure_check: QCLevel
    shake_check: QCLevel
    reasons: list[str]


def _run_ffprobe_duration_seconds(path: Path) -> float | None:
    cmd = [
        "ffprobe",
        "-v",
        "error",
        "-show_entries",
        "format=duration",
        "-of",
        "default=noprint_wrappers=1:nokey=1",
        str(path),
    ]
    try:
        result = subprocess.run(cmd, capture_output=True, text=True, check=False, timeout=120)
    except (subprocess.SubprocessError, OSError) as exc:
        logger.warning("ffprobe failed for %s: %s", path, exc)
        return None

    if result.returncode != 0:
        logger.warning("ffprobe returned %s for %s: %s", result.returncode, path, result.stderr.strip())
        return None

    text = result.stdout.strip()
    if not text:
        return None
    try:
        return float(text)
    except ValueError:
        logger.warning("ffprobe gave non-numeric duration for %s: %r", path, text)
        return None


def _laplacian_variance_gray(gray: np.ndarray) -> float:
    if gray.size == 0:
        return 0.0
    lap = cv2.Laplacian(gray, cv2.CV_64F)
    return float(lap.var())


def _sample_frame_timestamps(duration_sec: float, sample_count: int) -> list[float]:
    if sample_count <= 0:
        return []
    if duration_sec <= 0:
        return [0.0] * sample_count
    if sample_count == 1:
        return [min(duration_sec / 2.0, max(duration_sec - 1e-3, 0.0))]
    return [duration_sec * i / (sample_count - 1) for i in range(sample_count)]


def _read_sampled_frames(
    path: Path,
    duration_sec: float | None,
    sample_count: int,
) -> tuple[list[np.ndarray] | None, str | None]:
    """Return list of BGR frames or None if OpenCV cannot read."""
    cap = cv2.VideoCapture(str(path))
    if not cap.isOpened():
        logger.warning("OpenCV cannot open video: %s", path)
        return None, "OpenCV cannot open video"

    frames: list[np.ndarray] = []
    try:
        if duration_sec is not None and duration_sec > 0:
            timestamps = _sample_frame_timestamps(duration_sec, sample_count)
            for t_sec in timestamps:
                cap.set(cv2.CAP_PROP_POS_MSEC, t_sec * 1000.0)
                ok, frame = cap.read()
                if not ok or frame is None:
                    cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
                    ok, frame = cap.read()
                if not ok or frame is None:
                    logger.warning("Failed to read frame at %.3fs in %s", t_sec, path)
                    return None, "Failed to read sampled frames"
                frames.append(frame)
        else:
            for _ in range(sample_count):
                ok, frame = cap.read()
                if not ok or frame is None:
                    if not frames:
                        return None, "Failed to read sampled frames"
                    frames.append(frames[-1])
                    continue
                frames.append(frame)
    finally:
        cap.release()

    return frames, None


def _mean_brightness_bgr(frame: np.ndarray) -> float:
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    return float(gray.mean())


def _blur_ratio(frames: list[np.ndarray], blur_threshold: float) -> float:
    blurry = 0
    for frame in frames:
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        if _laplacian_variance_gray(gray) < blur_threshold:
            blurry += 1
    return blurry / len(frames) if frames else 0.0


def _exposure_fail_ratio(
    frames: list[np.ndarray],
    low: int,
    high: int,
) -> float:
    failed = 0
    for frame in frames:
        mean = _mean_brightness_bgr(frame)
        if mean < low or mean > high:
            failed += 1
    return failed / len(frames) if frames else 0.0


def _shake_mean_flow_magnitude(frames: list[np.ndarray]) -> float | None:
    """Mean optical-flow magnitude averaged over consecutive frame pairs; None if not computable."""
    if len(frames) < 2:
        return None

    pair_means: list[float] = []
    prev_gray = cv2.cvtColor(frames[0], cv2.COLOR_BGR2GRAY)
    for next_frame in frames[1:]:
        gray = cv2.cvtColor(next_frame, cv2.COLOR_BGR2GRAY)
        flow = cv2.calcOpticalFlowFarneback(
            prev_gray,
            gray,
            None,
            pyr_scale=0.5,
            levels=3,
            winsize=15,
            iterations=3,
            poly_n=5,
            poly_sigma=1.2,
            flags=0,
        )
        mag = np.sqrt(flow[..., 0] ** 2 + flow[..., 1] ** 2)
        pair_means.append(float(mag.mean()))
        prev_gray = gray

    return float(np.mean(pair_means)) if pair_means else None


def _ratio_check(
    ratio: float,
    reject_ratio: float,
    review_ratio: float,
) -> QCLevel:
    if ratio > reject_ratio:
        return "rejected"
    if ratio > review_ratio:
        return "review"
    return "pass"


def analyze_video(path: Path | str, config: dict[str, Any]) -> QCResult:
    """
    Run all video QC checks on a file (typically converted .mp4).

    OpenCV open/read failures set blur, exposure, and shake to review (per project rules).
    """
    video_path = Path(path)
    reasons: list[str] = []

    duration_sec = _run_ffprobe_duration_seconds(video_path)
    min_sec = float(config["min_video_duration_sec"])

    if duration_sec is None:
        duration_check: QCLevel = "review"
        reasons.append("Could not read duration (ffprobe failed or missing metadata)")
    elif duration_sec < min_sec:
        duration_check = "rejected"
        reasons.append(
            f"Duration {duration_sec:.2f}s is below minimum {min_sec}s",
        )
    else:
        duration_check = "pass"

    sample_count = int(config["frame_sample_count"])
    frames, frame_error = _read_sampled_frames(video_path, duration_sec, sample_count)

    blur_check: QCLevel
    exposure_check: QCLevel
    shake_check: QCLevel

    if frames is None:
        blur_check = exposure_check = shake_check = "review"
        if frame_error:
            reasons.append(frame_error)
    else:
        blur_r = _blur_ratio(frames, float(config["blur_threshold"]))
        blur_check = _ratio_check(
            blur_r,
            float(config["blur_reject_ratio"]),
            float(config["blur_review_ratio"]),
        )
        if blur_check == "rejected":
            reasons.append(
                f"Blur: {100.0 * blur_r:.1f}% of sampled frames below Laplacian threshold "
                f"{config['blur_threshold']}",
            )
        elif blur_check == "review":
            reasons.append(
                f"Blur: {100.0 * blur_r:.1f}% of frames blurry (review band)",
            )

        exp_r = _exposure_fail_ratio(
            frames,
            int(config["exposure_low_threshold"]),
            int(config["exposure_high_threshold"]),
        )
        exposure_check = _ratio_check(
            exp_r,
            float(config["exposure_reject_ratio"]),
            float(config["exposure_review_ratio"]),
        )
        if exposure_check == "rejected":
            reasons.append(
                f"Exposure: {100.0 * exp_r:.1f}% of frames under/over thresholds "
                f"({config['exposure_low_threshold']}/{config['exposure_high_threshold']})",
            )
        elif exposure_check == "review":
            reasons.append(f"Exposure: {100.0 * exp_r:.1f}% of frames fail (review band)")

        flow_mean = _shake_mean_flow_magnitude(frames)
        if flow_mean is None:
            shake_check = "pass"
        elif flow_mean > float(config["shake_threshold"]):
            shake_check = "review"
            reasons.append(
                f"Shake: mean optical-flow magnitude {flow_mean:.2f} exceeds threshold "
                f"{config['shake_threshold']}",
            )
        else:
            shake_check = "pass"

    return QCResult(
        duration_check=duration_check,
        blur_check=blur_check,
        exposure_check=exposure_check,
        shake_check=shake_check,
        reasons=reasons,
    )
