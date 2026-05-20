"""Load ClipSorter configuration from config.json with defaults fallback."""

from __future__ import annotations

import json
import logging
from copy import deepcopy
from pathlib import Path
from typing import Any

logger = logging.getLogger(__name__)

DEFAULT_CONFIG: dict[str, Any] = {
    "min_video_duration_sec": 5,
    "min_audio_duration_sec": 3,
    "blur_threshold": 80.0,
    "blur_reject_ratio": 0.6,
    "blur_review_ratio": 0.3,
    "exposure_low_threshold": 30,
    "exposure_high_threshold": 225,
    "exposure_reject_ratio": 0.7,
    "exposure_review_ratio": 0.3,
    "shake_threshold": 15.0,
    "frame_sample_count": 10,
    "duplicate_hash_threshold": 10,
    "duplicate_video_frame_match_ratio": 0.7,
    "duplicate_audio_similarity_threshold": 0.95,
    "silence_rms_threshold": 0.01,
    "silence_ratio_threshold": 0.80,
    "video_output_codec": "libx264",
    "video_output_crf": 18,
    "audio_output_bitrate": "192k",
}


def _project_root() -> Path:
    """Directory containing sort.py (parent of src/)."""
    return Path(__file__).resolve().parent.parent


def load_config(config_path: Path | str | None = None) -> dict[str, Any]:
    """
    Load configuration from JSON, merging over defaults.

    If config_path is None, reads config.json next to sort.py.
    Missing or malformed files fall back to defaults with a warning.
    """
    if config_path is None:
        path = _project_root() / "config.json"
    else:
        path = Path(config_path)

    if not path.is_file():
        logger.warning("config.json not found at %s; using defaults", path)
        return deepcopy(DEFAULT_CONFIG)

    try:
        with path.open(encoding="utf-8") as handle:
            user_config = json.load(handle)
    except json.JSONDecodeError as exc:
        logger.warning("Malformed config at %s (%s); using defaults", path, exc)
        return deepcopy(DEFAULT_CONFIG)

    if not isinstance(user_config, dict):
        logger.warning("Config at %s must be a JSON object; using defaults", path)
        return deepcopy(DEFAULT_CONFIG)

    config = deepcopy(DEFAULT_CONFIG)
    for key, value in user_config.items():
        if key in DEFAULT_CONFIG:
            config[key] = value
        else:
            logger.warning("Unknown config key ignored: %s", key)

    return config
