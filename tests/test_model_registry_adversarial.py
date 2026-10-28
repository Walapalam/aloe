"""Adversarial stress-test suite for Stage 2 / R2: Model Acquisition & Registry.

Stress-tests:
1. Zero-network offline loading for configs, tokenizers, and meta-models.
2. Meta-device parameter instantiation and memory containment (zero RAM allocation).
3. Alias lookup resilience (casing, whitespace, slugs, canonical IDs).
4. Loud failures on missing snapshot paths, corrupted JSON, missing config files, and unknown models.
5. Markdown generator response to FAIL status.
"""
from __future__ import annotations

import json
import socket
import tempfile
from pathlib import Path
from typing import Any

import psutil
import pytest
from pydantic import ValidationError

from aloe.datasets.model_registry import ModelRegistry

RUNS_MODELS_DIR = Path("runs/models")
PILOT_MODELS = [
    "google/gemma-4-E2B-it",
    "tiiuae/Falcon-H1-1.5B-Instruct",
]


@pytest.fixture(scope="module")
def registry() -> ModelRegistry:
    """Fixture providing initialized ModelRegistry."""
    return ModelRegistry(RUNS_MODELS_DIR)


def test_offline_loading_with_socket_blocked(registry: ModelRegistry, monkeypatch: pytest.MonkeyPatch) -> None:
    """Verify configs and tokenizers load cleanly with zero network access and socket connect blocked."""
    if not all(Path(m.model.local_snapshot_path).is_dir() for m in registry.list_models()):
        pytest.skip("Local HF cache snapshots from acquisition.json are not mounted on this host")
    def _blocked_connect(*args: Any, **kwargs: Any) -> None:
        raise OSError("Network access blocked by adversarial harness")

    monkeypatch.setattr(socket.socket, "connect", _blocked_connect)
    monkeypatch.setenv("HF_HUB_OFFLINE", "1")
    monkeypatch.setenv("TRANSFORMERS_OFFLINE", "1")

    for model_id in PILOT_MODELS:
        # Load config offline
        cfg = registry.load_config(model_id, local_files_only=True)
        assert cfg is not None

        # Load tokenizer offline
        tok = registry.load_tokenizer(model_id, local_files_only=True)
        assert tok is not None
        assert len(tok) > 10000

        # Load meta model offline
        meta_model = registry.load_model_meta(model_id, local_files_only=True)
        assert meta_model is not None


@pytest.mark.parametrize("model_id", PILOT_MODELS)
def test_meta_device_strict_containment_and_zero_ram(registry: ModelRegistry, model_id: str) -> None:
    """Verify 100% of parameters and buffers are on meta device with negligible RSS change."""
    if not Path(registry.get(model_id).model.local_snapshot_path).is_dir():
        pytest.skip(f"Local HF cache snapshot for {model_id} is not mounted on this host")
    proc = psutil.Process()
    rss_before_mb = proc.memory_info().rss / (1024 * 1024)

    model = registry.load_model_meta(model_id)

    rss_after_mb = proc.memory_info().rss / (1024 * 1024)
    rss_delta_mb = rss_after_mb - rss_before_mb

    params = list(model.parameters())
    buffers = list(model.buffers())

    assert len(params) > 0, f"Model {model_id} has no parameters"
    non_meta_params = [p for p in params if p.device.type != "meta"]
    non_meta_buffers = [b for b in buffers if b.device.type != "meta"]

    assert len(non_meta_params) == 0, f"Found {len(non_meta_params)} non-meta parameters"
    assert len(non_meta_buffers) == 0, f"Found {len(non_meta_buffers)} non-meta buffers"

    # Multi-billion parameter models in RAM would consume > 2000 MB.
    # Meta device must consume < 300 MB (only Python object graph overhead).
    assert rss_delta_mb < 300.0, f"Excessive memory delta {rss_delta_mb:.2f} MB on meta load"


def test_alias_lookup_matrix(registry: ModelRegistry) -> None:
    """Stress-test alias normalization across permutations of casing and whitespace."""
    aliases_a = [
        "google/gemma-4-E2B-it",
        "google--gemma-4-E2B-it",
        "gemma-4-E2B-it",
        "gemma-4-e2b-it",
        "gemma4",
        "gemma",
        "GEMMA",
        "  gemma  ",
        "model_a",
        "MODEL_A",
    ]
    for alias in aliases_a:
        rec = registry.get(alias)
        assert rec.model.id == "google/gemma-4-E2B-it", f"Failed to resolve alias: {alias}"

    aliases_b = [
        "tiiuae/Falcon-H1-1.5B-Instruct",
        "tiiuae--Falcon-H1-1.5B-Instruct",
        "Falcon-H1-1.5B-Instruct",
        "falcon-h1-1.5b-instruct",
        "falcon-h1-1.5b",
        "falcon-h1",
        "falcon",
        "FALCON",
        "  falcon  ",
        "model_b",
        "MODEL_B",
    ]
    for alias in aliases_b:
        rec = registry.get(alias)
        assert rec.model.id == "tiiuae/Falcon-H1-1.5B-Instruct", f"Failed to resolve alias: {alias}"


def test_unknown_models_fail_loud(registry: ModelRegistry) -> None:
    """Verify unknown model IDs fail loud with descriptive KeyError."""
    unknown_keys = ["", "  ", "nonexistent-model", "llama-3-8b", "gpt-4"]
    for key in unknown_keys:
        with pytest.raises(KeyError) as exc_info:
            registry.get(key)
        assert "not found in registry" in str(exc_info.value)
        assert "Available models" in str(exc_info.value)


def test_missing_snapshot_directory_fails_loud() -> None:
    """Verify missing snapshot path raises FileNotFoundError across all accessors."""
    with tempfile.TemporaryDirectory() as tmpdir:
        slug_dir = Path(tmpdir) / "test--model"
        slug_dir.mkdir()
        fake_rec = {
            "schema_version": "1.0",
            "run_id": "test-uuid",
            "timestamp_utc": "2026-09-17T00:00:00Z",
            "script": "download_models.py",
            "status": "PASS",
            "model": {
                "id": "test/model",
                "revision": "a" * 40,
                "precision": "bf16",
                "quantisation": "none",
                "local_snapshot_path": "/tmp/non_existent_snapshot_path_987654321",
                "total_size_bytes": 1000,
                "total_size_gb": 0.001,
                "files": [],
                "chat_template": "template",
                "has_chat_template": True,
                "config": {},
            },
        }
        (slug_dir / "acquisition.json").write_text(json.dumps(fake_rec), encoding="utf-8")
        reg = ModelRegistry(tmpdir)

        with pytest.raises(FileNotFoundError, match="does not exist on disk"):
            reg.get_snapshot_path("test/model")

        with pytest.raises(FileNotFoundError, match="does not exist on disk"):
            reg.load_config("test/model")

        with pytest.raises(FileNotFoundError, match="does not exist on disk"):
            reg.load_tokenizer("test/model")

        with pytest.raises(FileNotFoundError, match="does not exist on disk"):
            reg.load_model_meta("test/model")


def test_corrupted_json_fails_loud() -> None:
    """Verify corrupted JSON or schema violations fail loud during reload."""
    with tempfile.TemporaryDirectory() as tmpdir:
        slug_dir = Path(tmpdir) / "corrupted--model"
        slug_dir.mkdir()
        (slug_dir / "acquisition.json").write_text("{ corrupt json ...", encoding="utf-8")
        with pytest.raises(json.JSONDecodeError):
            ModelRegistry(tmpdir)

    with tempfile.TemporaryDirectory() as tmpdir:
        slug_dir = Path(tmpdir) / "schema-violating--model"
        slug_dir.mkdir()
        (slug_dir / "acquisition.json").write_text(json.dumps({"schema_version": "1.0"}), encoding="utf-8")
        with pytest.raises(ValidationError):
            ModelRegistry(tmpdir)


def test_registry_markdown_gate_failure_handling() -> None:
    """Verify registry markdown generator switches Gate 2 status to FAIL on non-PASS records."""
    with tempfile.TemporaryDirectory() as tmpdir:
        runs_dir = Path(tmpdir)
        m_dir = runs_dir / "test--model"
        m_dir.mkdir()
        (m_dir / "acquisition.json").write_text(
            json.dumps({
                "schema_version": "1.0",
                "run_id": "r-fail",
                "timestamp_utc": "2026-09-17T00:00:00Z",
                "script": "download_models.py",
                "status": "FAIL",
                "model": {
                    "id": "test/model",
                    "revision": "0" * 40,
                    "precision": "bf16",
                    "quantisation": "none",
                    "local_snapshot_path": str(runs_dir),
                    "total_size_bytes": 0,
                    "total_size_gb": 0.0,
                    "files": [],
                    "chat_template": "",
                    "has_chat_template": False,
                    "config": {},
                },
            }),
            encoding="utf-8",
        )
        reg = ModelRegistry(runs_dir)
        md = reg.generate_registry_markdown()
        assert "Gate 2 Evaluation: FAIL" in md
        assert "- [ ] Both models successfully downloaded" in md
