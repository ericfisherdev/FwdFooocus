"""Tests for new_ui.app FastAPI endpoints."""

import sys
from pathlib import Path

import pytest

# Ensure project root is on sys.path
sys.path.insert(0, str(Path(__file__).parent.parent))

# args_manager calls parse_args() at import time, which chokes on pytest's
# argv.  Patch sys.argv before any project modules are imported.
_original_argv = sys.argv
sys.argv = [sys.argv[0]]

from fastapi.testclient import TestClient  # noqa: E402
from new_ui.app import app  # noqa: E402

sys.argv = _original_argv

client = TestClient(app)


class TestIndexPage:
    def test_returns_html(self):
        r = client.get("/")
        assert r.status_code == 200
        assert "text/html" in r.headers["content-type"]

    def test_contains_title(self):
        r = client.get("/")
        assert "FwdFooocus" in r.text


class TestConfigAPI:
    def test_returns_config(self):
        r = client.get("/api/config")
        assert r.status_code == 200
        data = r.json()
        assert "default_model" in data
        assert "default_performance" in data
        assert "available_aspect_ratios" in data
        assert isinstance(data["available_aspect_ratios"], list)

    def test_config_has_lora_settings(self):
        r = client.get("/api/config")
        data = r.json()
        assert "default_loras_min_weight" in data
        assert "default_loras_max_weight" in data
        assert "default_max_lora_number" in data


class TestModelsAPI:
    def test_returns_model_lists(self):
        r = client.get("/api/models")
        assert r.status_code == 200
        data = r.json()
        assert "checkpoints" in data
        assert "loras" in data
        assert isinstance(data["checkpoints"], list)
        assert isinstance(data["loras"], list)


class TestStylesAPI:
    def test_returns_styles(self):
        r = client.get("/api/styles")
        assert r.status_code == 200
        data = r.json()
        assert "styles" in data
        assert isinstance(data["styles"], list)
        assert len(data["styles"]) > 0


class TestLoRALibraryAPI:
    def test_library_data_returns_list(self):
        r = client.get("/api/lora-library-data")
        assert r.status_code == 200
        assert isinstance(r.json(), list)

    def test_scan_status_returns_dict(self):
        r = client.get("/api/lora-library-scan-status")
        assert r.status_code == 200
        data = r.json()
        assert "is_scanning" in data
        assert "total_indexed" in data

    def test_trigger_words_requires_filename(self):
        r = client.get("/api/lora-trigger-words")
        assert r.status_code == 422  # FastAPI validation error

    def test_trigger_words_with_filename(self):
        r = client.get("/api/lora-trigger-words", params={"filename": "nonexistent.safetensors"})
        assert r.status_code == 200
        data = r.json()
        assert data["filename"] == "nonexistent.safetensors"
        assert isinstance(data["trigger_words"], list)


class TestHeartbeatAPI:
    def test_heartbeat_returns_ok(self):
        r = client.post("/api/heartbeat")
        assert r.status_code == 200
        assert r.json() == {"ok": True}

    def test_heartbeat_rejects_get(self):
        r = client.get("/api/heartbeat")
        assert r.status_code == 405
