"""Tests for BoTorch search space adapter."""
import torch
import pytest

from aloe.architecture.schema import ComponentType
from aloe.execution.trial_config import GroupTrialSetting
from aloe.optimization.bo_adapter import BOSearchAdapter
from aloe.search_space.schema import GroupDimensions, SearchGrammar


@pytest.fixture
def gemma_grammar() -> SearchGrammar:
    """Fixture producing a 3-group SearchGrammar matching Gemma 4-E2B."""
    return SearchGrammar(
        model_id="google/gemma-4-E2B-it",
        family="gemma",
        topology="dense_transformer",
        backend="peft_mps",
        groups=[
            GroupDimensions(
                component_type=ComponentType.LOCAL_ATTENTION,
                targets=["model.layers.0.self_attn.q_proj", "model.layers.0.self_attn.v_proj"],
                rank=[4, 8, 16, 32],
                alpha=[8, 16, 32, 64, 128],
            ),
            GroupDimensions(
                component_type=ComponentType.GLOBAL_ATTENTION,
                targets=["model.layers.1.self_attn.q_proj", "model.layers.1.self_attn.v_proj"],
                rank=[4, 8, 16, 32],
                alpha=[8, 16, 32, 64, 128],
            ),
            GroupDimensions(
                component_type=ComponentType.FFN,
                targets=["model.layers.0.mlp.gate_proj", "model.layers.0.mlp.up_proj"],
                rank=[4, 8, 16, 32],
                alpha=[8, 16, 32, 64, 128],
            ),
        ],
    )


def test_adapter_dimensions_and_bounds(gemma_grammar: SearchGrammar) -> None:
    adapter = BOSearchAdapter(gemma_grammar)
    # 3 groups * 2 params (rank, alpha) = 6 dimensions
    assert adapter.dim == 6
    assert adapter.bounds.shape == (2, 6)
    assert torch.all(adapter.bounds[0] == 0.0)
    assert torch.all(adapter.bounds[1] == 1.0)


def test_encode_decode_roundtrip(gemma_grammar: SearchGrammar) -> None:
    adapter = BOSearchAdapter(gemma_grammar)

    # Active settings
    settings = {
        "local_attention": GroupTrialSetting(enabled=True, rank=16, alpha=32),
        "global_attention": GroupTrialSetting(enabled=True, rank=8, alpha=16),
        "ffn": GroupTrialSetting(enabled=True, rank=4, alpha=64),
    }

    vec = adapter.encode(settings)
    assert vec.shape == (6,)
    assert torch.all(vec >= 0.0) and torch.all(vec <= 1.0)

    decoded = adapter.decode(vec, trial_id="test_roundtrip")
    assert decoded.trial_id == "test_roundtrip"
    for name, s in settings.items():
        dec_s = decoded.group_settings[name]
        assert dec_s.enabled == s.enabled
        assert dec_s.rank == s.rank
        assert dec_s.alpha == s.alpha


def test_canonical_collapse_disabled_group(gemma_grammar: SearchGrammar) -> None:
    adapter = BOSearchAdapter(gemma_grammar)

    # Group with enabled=False
    settings = {
        "local_attention": GroupTrialSetting(enabled=False, rank=32, alpha=128),
        "global_attention": GroupTrialSetting(enabled=True, rank=8, alpha=16),
        "ffn": GroupTrialSetting(enabled=False, rank=16, alpha=64),
    }

    vec = adapter.encode(settings)
    decoded = adapter.decode(vec)

    # Disabled groups must decode with enabled=False and canonical fallback
    assert decoded.group_settings["local_attention"].enabled is False
    assert decoded.group_settings["global_attention"].enabled is True
    assert decoded.group_settings["global_attention"].rank == 8
    assert decoded.group_settings["global_attention"].alpha == 16
    assert decoded.group_settings["ffn"].enabled is False


def test_cardinality_and_uniform_sampling(gemma_grammar: SearchGrammar) -> None:
    adapter = BOSearchAdapter(gemma_grammar)

    # Distinct states per group: 1 (disabled) + 4 (ranks) * 5 (alphas) = 21
    # Total non-trivial points: 21^3 - 1 = 9260
    assert gemma_grammar.cardinality(include_all_disabled=False) == 9260

    # Draw uniform sample of 6 initial points
    samples = adapter.generate_uniform_sample(n_samples=6, seed=42)
    assert len(samples) == 6

    # Verify no duplicates
    keys = []
    for s in samples:
        sig = tuple((k, s[k].enabled, s[k].rank if s[k].enabled else 0, s[k].alpha if s[k].enabled else 0) for k in sorted(s.keys()))
        keys.append(sig)
    assert len(set(keys)) == 6


def test_parameter_estimation(gemma_grammar: SearchGrammar) -> None:
    adapter = BOSearchAdapter(gemma_grammar, d_model_est=2048)

    # All disabled
    empty_settings = {k: GroupTrialSetting(enabled=False, rank=4, alpha=8) for k in adapter.group_names}
    assert adapter.estimate_trainable_parameters(empty_settings) == 0

    # Local attention enabled with r=8, 2 targets -> 2 * 8 * 2048 * 2 = 65,536 params
    active_settings = {
        "local_attention": GroupTrialSetting(enabled=True, rank=8, alpha=16),
        "global_attention": GroupTrialSetting(enabled=False, rank=4, alpha=8),
        "ffn": GroupTrialSetting(enabled=False, rank=4, alpha=8),
    }
    assert adapter.estimate_trainable_parameters(active_settings) == 65536
