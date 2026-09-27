"""Tests for Gate C: TrialConfig serialization, roundtrip, and PEFT translation."""
import json
from pathlib import Path
import pytest

from aloe.execution.trial_config import GroupTrialSetting, TrialConfig
from aloe.search_space.schema import SearchGrammar
from aloe.tracking.trial_store import TrialMetrics, TrialResultEnvelope, TrialStore


@pytest.fixture
def gemma_grammar():
    grammar_file = Path("runs/grammars/gemma_e2b.json")
    if not grammar_file.exists():
        pytest.skip(f"Grammar file missing: {grammar_file}")
    return SearchGrammar.model_validate_json(grammar_file.read_text())


def test_trial_config_json_roundtrip(tmp_path):
    """Assert TrialConfig serializes to and deserializes from JSON losslessly."""
    config = TrialConfig(
        trial_id="test_trial_001",
        model_id="google/gemma-4-E2B-it",
        seed=1337,
        group_settings={
            "local_attention": GroupTrialSetting(enabled=True, rank=8, alpha=16),
            "global_attention": GroupTrialSetting(enabled=True, rank=16, alpha=32),
            "ffn": GroupTrialSetting(enabled=False, rank=4, alpha=8),
        },
        description="Lossless serialization verification test",
    )

    # In-memory JSON roundtrip
    json_str = config.to_json()
    reloaded = TrialConfig.from_json(json_str)
    assert reloaded == config
    assert reloaded.trial_id == "test_trial_001"
    assert reloaded.group_settings["local_attention"].rank == 8
    assert reloaded.group_settings["global_attention"].alpha == 32
    assert reloaded.group_settings["ffn"].enabled is False

    # Disk file roundtrip
    file_path = tmp_path / "trial_config.json"
    config.save(file_path)
    from_disk = TrialConfig.from_file(file_path)
    assert from_disk == config


def test_peft_lora_config_translation(gemma_grammar):
    """Assert TrialConfig translates into exact PEFT rank_pattern and alpha_pattern per group."""
    config = TrialConfig(
        trial_id="test_peft_trans",
        model_id="google/gemma-4-E2B-it",
        group_settings={
            "local_attention": GroupTrialSetting(enabled=True, rank=8, alpha=16),
            "global_attention": GroupTrialSetting(enabled=True, rank=16, alpha=32),
            "ffn": GroupTrialSetting(enabled=False, rank=4, alpha=8),
        },
    )

    peft_kwargs, lora_cfg = config.to_peft_lora_config(gemma_grammar)

    # Gemma E2B has 80 local targets and 20 global targets = 100 total
    assert len(lora_cfg.target_modules) == 100
    assert len(peft_kwargs["target_modules"]) == 100

    # Ensure FFN modules are NOT present
    for target in lora_cfg.target_modules:
        assert "gate_proj" not in target
        assert "up_proj" not in target
        assert "down_proj" not in target

    # Check rank_pattern and alpha_pattern mappings
    rank_pattern = lora_cfg.rank_pattern
    alpha_pattern = lora_cfg.alpha_pattern

    groups_by_type = {g.component_type.value: g for g in gemma_grammar.groups}
    for local_t in groups_by_type["local_attention"].targets:
        assert rank_pattern[local_t] == 8
        assert alpha_pattern[local_t] == 16

    for global_t in groups_by_type["global_attention"].targets:
        assert rank_pattern[global_t] == 16
        assert alpha_pattern[global_t] == 32


def test_trial_store_envelope_roundtrip(tmp_path):
    """Assert TrialStore records and retrieves Result Schema Envelope 1.0."""
    store = TrialStore(runs_dir=tmp_path)

    metrics = TrialMetrics(
        peak_memory_gb=10.28,
        peak_rss_gb=10.28,
        mps_driver_allocated_gb=1.31,
        train_seconds=42.5,
        latency_per_step_sec=0.71,
        trainable_params=1234567,
        total_params=2000000000,
        trainable_ratio=0.000617,
        adapter_size_mb=4.85,
        initial_train_loss=3.21,
        final_train_loss=1.45,
        fresh_process_reload=True,
    )

    envelope = TrialResultEnvelope(
        trial_id="pilot_test_001",
        git_commit="f438acf",
        dataset_manifest_sha256="c035d607acb183fd893bb9fe427125275c24a18687cf31a9524e17778f749bb9",
        hardware="Apple M1 Max (64 GB unified memory), macOS, MPS",
        library_versions={"torch": "2.4.0", "peft": "0.12.0"},
        model_id="google/gemma-4-E2B-it",
        backend="peft_mps",
        trial_config={"seed": 42, "model_id": "google/gemma-4-E2B-it"},
        metrics=metrics,
        status="PASS",
    )

    saved_path = store.save(envelope)
    assert saved_path.exists()

    loaded = store.get("pilot_test_001")
    assert loaded.schema_version == "1.0"
    assert loaded.trial_id == "pilot_test_001"
    assert loaded.metrics.peak_memory_gb == 10.28
    assert loaded.metrics.mps_driver_allocated_gb == 1.31
    assert loaded.status == "PASS"
