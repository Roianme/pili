"""Tests for config_loader (Step 1)."""

from __future__ import annotations

import json
import logging

import pytest

from config_loader import DEFAULT_CONFIG, load_config


def test_load_config_missing_file_uses_defaults(tmp_path, caplog):
    missing = tmp_path / "config.json"

    with caplog.at_level(logging.WARNING):
        config = load_config(missing)

    assert config == DEFAULT_CONFIG
    assert any("not found" in record.message.lower() for record in caplog.records)


def test_load_config_valid_file_merges_over_defaults(tmp_path, caplog):
    config_path = tmp_path / "config.json"
    config_path.write_text(
        json.dumps({"min_video_duration_sec": 10, "blur_threshold": 50.0}),
        encoding="utf-8",
    )

    with caplog.at_level(logging.WARNING):
        config = load_config(config_path)

    assert config["min_video_duration_sec"] == 10
    assert config["blur_threshold"] == 50.0
    assert config["min_audio_duration_sec"] == DEFAULT_CONFIG["min_audio_duration_sec"]
    assert not caplog.records


def test_load_config_malformed_json_uses_defaults(tmp_path, caplog):
    config_path = tmp_path / "config.json"
    config_path.write_text("{ not valid json", encoding="utf-8")

    with caplog.at_level(logging.WARNING):
        config = load_config(config_path)

    assert config == DEFAULT_CONFIG
    assert any("malformed" in record.message.lower() for record in caplog.records)


def test_load_config_non_object_json_uses_defaults(tmp_path, caplog):
    config_path = tmp_path / "config.json"
    config_path.write_text("[1, 2, 3]", encoding="utf-8")

    with caplog.at_level(logging.WARNING):
        config = load_config(config_path)

    assert config == DEFAULT_CONFIG
    assert any("json object" in record.message.lower() for record in caplog.records)


def test_load_config_unknown_keys_are_ignored(tmp_path, caplog):
    config_path = tmp_path / "config.json"
    config_path.write_text(
        json.dumps({"min_video_duration_sec": 7, "extra_setting": True}),
        encoding="utf-8",
    )

    with caplog.at_level(logging.WARNING):
        config = load_config(config_path)

    assert config["min_video_duration_sec"] == 7
    assert "extra_setting" not in config
    assert any("unknown config key" in record.message.lower() for record in caplog.records)


def test_load_config_does_not_mutate_defaults(tmp_path):
    config_path = tmp_path / "config.json"
    config_path.write_text(json.dumps({"blur_threshold": 1.0}), encoding="utf-8")

    config = load_config(config_path)
    config["blur_threshold"] = 999.0

    assert DEFAULT_CONFIG["blur_threshold"] == 35.0


def test_load_config_mode_preset_changes_defaults(tmp_path):
    missing = tmp_path / "config.json"

    config = load_config(missing, mode="aggressive")

    assert config["min_video_duration_sec"] == 4
    assert config["blur_threshold"] == 70.0
    assert config["duplicate_hash_threshold"] == 8


def test_load_config_mode_preset_can_override_user(tmp_path):
    config_path = tmp_path / "config.json"
    config_path.write_text(
        json.dumps({"min_video_duration_sec": 8, "blur_threshold": 60.0}),
        encoding="utf-8",
    )

    config = load_config(config_path, mode="conservative")

    assert config["min_video_duration_sec"] == 6
    assert config["blur_threshold"] == 90.0
    assert config["shake_threshold"] == 10.0
