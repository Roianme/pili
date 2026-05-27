"""Load ClipSorter configuration from config.json with defaults fallback."""

from __future__ import annotations

import json
import logging
from copy import deepcopy
from pathlib import Path
from typing import Any

logger = logging.getLogger(__name__)

DEFAULT_CONFIG: dict[str, Any] = {
    "min_video_duration_sec": 0,
    "min_audio_duration_sec": 0,
    "blur_threshold": 60.0,
    "blur_reject_ratio": 0.6,
    "blur_review_ratio": 0.3,
    "blur_max_threshold": 1500.0,
    "contrast_threshold": 10.0,
    "exposure_low_threshold": 30,
    "exposure_high_threshold": 225,
    "exposure_reject_ratio": 0.85,
    "exposure_review_ratio": 0.3,
    "content_variance_threshold": 15.0,
    "content_variance_reject_ratio": 0.6,
    "saturation_threshold": 0.1,
    "saturation_reject_ratio": 0.8,
    "histogram_entropy_threshold": 4.2,
    "histogram_entropy_reject": True,
    "shake_threshold": 30.0,
    "frame_sample_count": 10,
    "duplicate_hash_threshold": 10,
    "burst_hash_distance_threshold": 20,
    "burst_min_group_size": 2,
    "duplicate_video_frame_match_ratio": 0.7,
    "duplicate_audio_similarity_threshold": 0.95,
    "silence_rms_threshold": 0.01,
    "silence_ratio_threshold": 0.8,
    "video_output_codec": "libx264",
    "video_output_crf": 18,
    "audio_output_bitrate": "192k",
    "raw_conversion_timeout_sec": 30,
    "raw_conversion_strategy": "auto",
}

MODE_PRESETS: dict[str, dict[str, Any]] = {
    "normal": {},
    "aggressive": {
        "min_video_duration_sec": 4,
        "min_audio_duration_sec": 2,
        "blur_threshold": 70.0,
        "blur_reject_ratio": 0.75,
        "blur_review_ratio": 0.2,
        "exposure_low_threshold": 25,
        "exposure_high_threshold": 230,
        "exposure_reject_ratio": 0.8,
        "exposure_review_ratio": 0.2,
        "shake_threshold": 20.0,
        "duplicate_hash_threshold": 8,
        "burst_hash_distance_threshold": 25,
    },
    "conservative": {
        "min_video_duration_sec": 6,
        "min_audio_duration_sec": 4,
        "blur_threshold": 90.0,
        "blur_reject_ratio": 0.5,
        "blur_review_ratio": 0.4,
        "exposure_low_threshold": 35,
        "exposure_high_threshold": 220,
        "exposure_reject_ratio": 0.6,
        "exposure_review_ratio": 0.4,
        "shake_threshold": 10.0,
        "duplicate_hash_threshold": 12,
        "burst_hash_distance_threshold": 15,
    },
}


def _project_root() -> Path:
    """Directory containing sort.py (parent of src/)."""
    return Path(__file__).resolve().parent.parent


def _apply_mode_preset(config: dict[str, Any], mode: str) -> dict[str, Any]:
    preset = MODE_PRESETS.get(mode)
    if preset is None:
        logger.warning("Unknown mode %s; using normal preset", mode)
        return config
    config.update(preset)
    return config


def load_config(config_path: Path | str | None = None, mode: str = "normal") -> dict[str, Any]:
    """
    Load configuration from JSON, merging over defaults and preset mode.

    If config_path is None, reads config.json next to sort.py.
    Missing or malformed files fall back to defaults with a warning.
    """
    if config_path is None:
        path = _project_root() / "config.json"
    else:
        path = Path(config_path)

    config = deepcopy(DEFAULT_CONFIG)

    if not path.is_file():
        logger.warning("config.json not found at %s; using defaults", path)
        config = _apply_mode_preset(config, mode)
        return config

    try:
        with path.open(encoding="utf-8") as handle:
            user_config = json.load(handle)
    except json.JSONDecodeError as exc:
        logger.warning("Malformed config at %s (%s); using defaults", path, exc)
        return config

    if not isinstance(user_config, dict):
        logger.warning("Config at %s must be a JSON object; using defaults", path)
        return config

    for key, value in user_config.items():
        if key in DEFAULT_CONFIG:
            config[key] = value
        else:
            logger.warning("Unknown config key ignored: %s", key)

    config = _apply_mode_preset(config, mode)
    return config
