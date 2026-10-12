"""Tests for EvalCard reproducibility reporting."""
import json
import tempfile
from pathlib import Path
import pytest
from pydantic import ValidationError

from aloe.evaluation.eval_card import (
    EvalCard,
    EvalCardMetrics,
    EvalCardOptimizer,
)


@pytest.fixture
def sample_metrics() -> EvalCardMetrics:
    return EvalCardMetrics(
        tool_accuracy_mean=0.625,
        tool_accuracy_std=0.018,
        ci_95_bootstrap=[0.035, 0.125],
        p_value_vs_uniform=0.012,
        peak_memory_rss_gb=11.95,
        memory_metric="process_rss_unified",
    )


@pytest.fixture
def sample_optimizer() -> EvalCardOptimizer:
    return EvalCardOptimizer(
        impl="mobo_botorch_matern52",
        kernel="ScaleKernel(MaternKernel(nu=2.5))",
        acquisition="qLogNEHVI",
        ref_point=[0.0, -100_000_000.0, -24_000.0],
        n_init=6,
        bo_seed=42,
    )


def test_eval_card_creation_and_integrity(
    sample_metrics: EvalCardMetrics,
    sample_optimizer: EvalCardOptimizer,
) -> None:
    card = EvalCard.create(
        model_id="google/gemma-4-E2B-it",
        parameters_total=2_500_000_000,
        trainable_parameters=10_057_728,
        group_settings={
            "local_attention": {"enabled": True, "rank": 16, "alpha": 32},
            "global_attention": {"enabled": True, "rank": 16, "alpha": 32},
            "ffn": {"enabled": True, "rank": 4, "alpha": 8},
        },
        metrics=sample_metrics,
        optimizer=sample_optimizer,
        hardware_tier="apple_silicon_mps",
        backend="peft_mps",
        seeds=[42, 1337, 2026],
        baseline_trial_id="uniform_baseline_001",
    )

    assert card.eval_card_version == "1.0"
    assert card.model.model_id == "google/gemma-4-E2B-it"
    assert card.model.parameter_fraction == pytest.approx(10_057_728 / 2_500_000_000)
    assert len(card.eval_card_sha256) == 64
    assert card.environment.seeds_evaluated == [42, 1337, 2026]


def test_eval_card_save_load_roundtrip(
    sample_metrics: EvalCardMetrics,
    sample_optimizer: EvalCardOptimizer,
) -> None:
    card = EvalCard.create(
        model_id="tiiuae/falcon-h1-1.5b",
        parameters_total=1_500_000_000,
        trainable_parameters=8_000_000,
        group_settings={
            "global_attention": {"enabled": True, "rank": 8, "alpha": 16},
            "ssm": {"enabled": True, "rank": 8, "alpha": 16},
            "ffn": {"enabled": False, "rank": 4, "alpha": 8},
        },
        metrics=sample_metrics,
        optimizer=sample_optimizer,
        hardware_tier="colab_a100",
        backend="peft_mps",
        seeds=[42],
    )

    with tempfile.TemporaryDirectory() as tmpdir:
        card_path = Path(tmpdir) / "eval_card.json"
        card.save(card_path)

        loaded = EvalCard.load(card_path)
        assert loaded.model.model_id == "tiiuae/falcon-h1-1.5b"
        assert loaded.metrics.tool_accuracy_mean == 0.625
        assert loaded.eval_card_sha256 == card.eval_card_sha256

        # Tamper test
        content = json.loads(card_path.read_text(encoding="utf-8"))
        content["model"]["model_id"] = "tampered_model_id"
        card_path.write_text(json.dumps(content), encoding="utf-8")

        with pytest.raises(ValueError, match="integrity check failed"):
            EvalCard.load(card_path)


def test_eval_card_validation_errors(
    sample_metrics: EvalCardMetrics,
    sample_optimizer: EvalCardOptimizer,
) -> None:
    with pytest.raises(ValidationError):
        # Empty seeds list must fail validation
        EvalCard.create(
            model_id="test_model",
            parameters_total=1000,
            trainable_parameters=10,
            group_settings={},
            metrics=sample_metrics,
            optimizer=sample_optimizer,
            hardware_tier="apple_silicon_mps",
            backend="peft_mps",
            seeds=[],  # Invalid!
        )
