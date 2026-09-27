"""Tests for Gate C: Invalid Configuration and Falcon conv1d rejection BEFORE training."""
from pathlib import Path
from unittest.mock import MagicMock
import pytest

from aloe.execution.backends.peft_mps import PeftMPSBackend
from aloe.execution.trial_config import GroupTrialSetting, InvalidConfigError, TrialConfig
from aloe.search_space.schema import SearchGrammar
from aloe.tracking.trial_store import TrialStore


@pytest.fixture
def falcon_grammar():
    grammar_file = Path("runs/grammars/falcon_h1.json")
    if not grammar_file.exists():
        pytest.skip(f"Grammar file missing: {grammar_file}")
    return SearchGrammar.model_validate_json(grammar_file.read_text())


def test_zero_or_negative_rank_rejected():
    """Assert non-positive rank raises InvalidConfigError or ValueError."""
    with pytest.raises((InvalidConfigError, ValueError), match="rank must be positive"):
        GroupTrialSetting(rank=0)

    with pytest.raises((InvalidConfigError, ValueError), match="rank must be positive"):
        GroupTrialSetting(rank=-4)


def test_zero_or_negative_alpha_rejected():
    """Assert non-positive alpha raises InvalidConfigError or ValueError."""
    with pytest.raises((InvalidConfigError, ValueError), match="alpha must be positive"):
        GroupTrialSetting(alpha=0)

    with pytest.raises((InvalidConfigError, ValueError), match="alpha must be positive"):
        GroupTrialSetting(alpha=-16)


def test_all_groups_disabled_rejected():
    """Assert config with all groups disabled raises InvalidConfigError or ValueError."""
    with pytest.raises((InvalidConfigError, ValueError), match="At least one architectural module group must be enabled"):
        TrialConfig(
            model_id="google/gemma-4-E2B-it",
            group_settings={
                "local_attention": GroupTrialSetting(enabled=False),
                "global_attention": GroupTrialSetting(enabled=False),
            },
        )


def test_falcon_conv1d_rejected_by_grammar_before_gpu(falcon_grammar):
    """Assert config targeting Falcon conv1d is rejected by grammar validation before touching MPS/GPU."""
    # Attempting to target Falcon conv1d explicitly
    config = TrialConfig(
        trial_id="falcon_invalid_conv1d",
        model_id="tiiuae/Falcon-H1-1.5B-Instruct",
        explicit_target_modules=["model.layers.0.mamba.conv1d"],
    )

    with pytest.raises(InvalidConfigError, match="excluded"):
        config.to_peft_lora_config(falcon_grammar)


def test_gradient_missing_aborts_before_training(tmp_path):
    """Assert run_trial aborts with INVALID_CONFIG when verify_gradients detects GRADIENT_MISSING.

    Proves compute is saved by rejecting inert modules in one backward pass BEFORE training.
    """
    backend = PeftMPSBackend()
    store = TrialStore(runs_dir=tmp_path)

    config = TrialConfig(
        trial_id="inert_module_trial",
        model_id="google/gemma-4-E2B-it",
        group_settings={
            "local_attention": GroupTrialSetting(enabled=True, rank=8, alpha=16),
        },
    )

    # Mock verify_gradients to simulate a module that silently receives no gradient (like Falcon conv1d)
    mock_grad_check = {
        "success": False,
        "verdict": "FAIL",
        "target_module_count": 1,
        "adapter_param_count": 2,
        "gradient_ok_count": 0,
        "gradient_zero_count": 0,
        "gradient_missing_count": 2,
        "param_details": [
            {"param_name": "base_model.model.model.layers.0.mamba.conv1d.lora_A", "norm": None, "classification": "GRADIENT_MISSING"},
            {"param_name": "base_model.model.model.layers.0.mamba.conv1d.lora_B", "norm": None, "classification": "GRADIENT_MISSING"},
        ],
    }

    backend.verify_gradients = MagicMock(return_value=mock_grad_check)

    # Supply dummy base_model mock
    mock_base_model = MagicMock()
    mock_tokenizer = MagicMock()

    grammar_file = Path("runs/grammars/gemma_e2b.json")
    grammar = SearchGrammar.model_validate_json(grammar_file.read_text())

    with pytest.raises(InvalidConfigError, match="Pre-train gradient check failed"):
        backend.run_trial(
            config=config,
            runs_trials_dir=tmp_path,
            grammar=grammar,
            base_model=mock_base_model,
            tokenizer=mock_tokenizer,
        )

    # Assert trial was recorded in store as INVALID_CONFIG
    record = store.get("inert_module_trial")
    assert record.status == "INVALID_CONFIG"
    assert len(record.failures) > 0
    assert "conv1d" in record.failures[0]
    assert record.metrics is None  # Zero training ran
