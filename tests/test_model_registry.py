"""Tests for Stage 2 / R2: Model Acquisition & Registry.

Validates that:
1. Both pilot models (Gemma-4-E2B-it and Falcon-H1-1.5B-Instruct) exist in the registry.
2. Both acquisition.json records adhere to schema_version "1.0" and status "PASS".
3. Revision commit SHAs are pinned 40-character hex strings.
4. AutoConfig and AutoTokenizer load cleanly via ModelRegistry.
5. Model lookup supports repo IDs, slugs, and aliases, and fails loud on unknown models.
6. Programmatic REGISTRY.md matches generate_registry_markdown() verbatim.
"""
from __future__ import annotations

import re
from pathlib import Path

import pytest
from transformers import PretrainedConfig, PreTrainedTokenizerBase

from aloe.datasets.model_registry import (
    ModelAcquisitionRecord,
    ModelRegistry,
)

RUNS_MODELS_DIR = Path("runs/models")
PILOT_MODELS = [
    "google/gemma-4-E2B-it",
    "tiiuae/Falcon-H1-1.5B-Instruct",
]
HEX_40_REGEX = re.compile(r"^[0-9a-f]{40}$")


@pytest.fixture(scope="module")
def registry() -> ModelRegistry:
    """Module-level fixture providing the ModelRegistry loaded from runs/models."""
    assert RUNS_MODELS_DIR.is_dir(), f"{RUNS_MODELS_DIR} must exist before running tests"
    reg = ModelRegistry(RUNS_MODELS_DIR)
    return reg


def test_registry_contains_both_pilot_models(registry: ModelRegistry) -> None:
    """Verify both pilot models are registered."""
    models = registry.list_models()
    registered_ids = {m.model.id for m in models}
    for expected_id in PILOT_MODELS:
        assert expected_id in registered_ids, f"Model {expected_id} missing from registry"


@pytest.mark.parametrize("model_id", PILOT_MODELS)
def test_model_acquisition_record_schema_and_status(
    registry: ModelRegistry, model_id: str
) -> None:
    """Validate acquisition envelope conforms to schema_version 1.0 and Gate 2 status."""
    record = registry.get(model_id)
    assert isinstance(record, ModelAcquisitionRecord)
    assert record.schema_version == "1.0"
    assert record.status == "PASS"
    assert record.script == "download_models.py"
    assert record.run_id != ""
    assert record.timestamp_utc != ""

    meta = record.model
    assert meta.id == model_id
    assert meta.precision == "bf16"
    assert meta.quantisation == "none"
    assert meta.total_size_bytes > 0
    assert meta.total_size_gb > 0
    assert len(meta.files) >= 5
    assert meta.has_chat_template is True
    assert "NOT PRESENT" not in meta.chat_template
    assert "ERROR" not in meta.chat_template

    # Revision SHA must be pinned 40-char hex string
    assert HEX_40_REGEX.match(meta.revision), (
        f"Revision '{meta.revision}' for {model_id} is not a pinned 40-char hex commit SHA"
    )

    # Local snapshot directory must exist on disk
    snapshot_path = Path(meta.local_snapshot_path)
    assert snapshot_path.is_dir(), f"Snapshot directory {snapshot_path} does not exist"

    # All recorded files must exist within snapshot directory
    for f in meta.files:
        p = snapshot_path / f.path
        assert p.exists(), f"File {f.path} recorded in acquisition.json does not exist at {p}"


@pytest.mark.parametrize("model_id", PILOT_MODELS)
def test_model_config_loading(registry: ModelRegistry, model_id: str) -> None:
    """Verify AutoConfig loads from local snapshot and preserves architecture metadata."""
    cfg = registry.load_config(model_id)
    assert isinstance(cfg, PretrainedConfig)

    if model_id == "google/gemma-4-E2B-it":
        assert "Gemma4ForConditionalGeneration" in cfg.architectures
        assert cfg.text_config.num_hidden_layers == 35
        assert cfg.text_config.hidden_size == 1536
        assert hasattr(cfg.text_config, "layer_types")
        assert len(cfg.text_config.layer_types) == 35
        assert cfg.text_config.num_kv_shared_layers == 20
    elif model_id == "tiiuae/Falcon-H1-1.5B-Instruct":
        assert "FalconH1ForCausalLM" in cfg.architectures
        assert cfg.num_hidden_layers == 24
        assert cfg.hidden_size == 2048
        assert cfg.mamba_d_ssm == 3072


@pytest.mark.parametrize("model_id", PILOT_MODELS)
def test_model_tokenizer_loading(registry: ModelRegistry, model_id: str) -> None:
    """Verify AutoTokenizer loads from local snapshot and has chat_template configured."""
    tok = registry.load_tokenizer(model_id)
    assert isinstance(tok, PreTrainedTokenizerBase)
    assert tok.chat_template is not None
    assert len(tok.chat_template) > 100
    assert len(tok) > 10000


@pytest.mark.parametrize("model_id", PILOT_MODELS)
def test_model_meta_loading(registry: ModelRegistry, model_id: str) -> None:
    """Verify models instantiate on PyTorch meta-device without allocating memory."""
    meta_model = registry.load_model_meta(model_id)
    assert meta_model is not None
    # Confirm parameters are on meta device
    first_param = next(meta_model.parameters())
    assert first_param.device.type == "meta"


def test_registry_lookup_aliases_and_slugs(registry: ModelRegistry) -> None:
    """Verify lookup flexibility: repo_id, slug, and recognized aliases."""
    # Gemma lookups
    rec_gemma_full = registry.get("google/gemma-4-E2B-it")
    assert registry.get("google--gemma-4-E2B-it").model.id == rec_gemma_full.model.id
    assert registry.get("gemma-4-E2B-it").model.id == rec_gemma_full.model.id
    assert registry.get("gemma").model.id == rec_gemma_full.model.id
    assert registry.get("model_a").model.id == rec_gemma_full.model.id

    # Falcon lookups
    rec_falcon_full = registry.get("tiiuae/Falcon-H1-1.5B-Instruct")
    assert registry.get("tiiuae--Falcon-H1-1.5B-Instruct").model.id == rec_falcon_full.model.id
    assert registry.get("falcon-h1-1.5b").model.id == rec_falcon_full.model.id
    assert registry.get("falcon").model.id == rec_falcon_full.model.id
    assert registry.get("model_b").model.id == rec_falcon_full.model.id

    # Unknown model must fail loud with KeyError
    with pytest.raises(KeyError, match="Model 'nonexistent-model' not found"):
        registry.get("nonexistent-model")


def test_registry_markdown_generation_reproducibility(registry: ModelRegistry) -> None:
    """Verify REGISTRY.md is generated deterministically from JSON acquisition data."""
    registry_md_path = RUNS_MODELS_DIR / "REGISTRY.md"
    assert registry_md_path.is_file(), f"{registry_md_path} must exist"

    expected_markdown = registry.generate_registry_markdown()
    actual_markdown = registry_md_path.read_text(encoding="utf-8")

    assert "Gate 2 Evaluation: PASS" in actual_markdown
    assert "google/gemma-4-E2B-it" in actual_markdown
    assert "tiiuae/Falcon-H1-1.5B-Instruct" in actual_markdown
    assert actual_markdown == expected_markdown, (
        "REGISTRY.md on disk does not match output of generate_registry_markdown()"
    )
