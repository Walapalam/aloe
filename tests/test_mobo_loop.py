"""Tests for Multi-Objective Bayesian Optimization (MOBO) loop."""
import pytest
import torch

from aloe.architecture.schema import ComponentType
from aloe.optimization.bo_adapter import BOSearchAdapter
from aloe.optimization.mobo import MOBOptimizer
from aloe.search_space.schema import GroupDimensions, SearchGrammar


@pytest.fixture
def test_grammar() -> SearchGrammar:
    return SearchGrammar(
        model_id="google/gemma-4-E2B-it",
        family="gemma",
        topology="dense_transformer",
        backend="peft_mps",
        groups=[
            GroupDimensions(
                component_type=ComponentType.LOCAL_ATTENTION,
                targets=["model.layers.0.self_attn.q_proj"],
                rank=[4, 8, 16],
                alpha=[8, 16, 32],
            ),
            GroupDimensions(
                component_type=ComponentType.GLOBAL_ATTENTION,
                targets=["model.layers.1.self_attn.q_proj"],
                rank=[4, 8, 16],
                alpha=[8, 16, 32],
            ),
            GroupDimensions(
                component_type=ComponentType.FFN,
                targets=["model.layers.0.mlp.gate_proj"],
                rank=[4, 8, 16],
                alpha=[8, 16, 32],
            ),
        ],
    )


def test_mobo_import_error_without_fallback(test_grammar: SearchGrammar) -> None:
    adapter = BOSearchAdapter(test_grammar)

    # If BoTorch is absent, initializing without allow_fallback must raise ImportError
    if not MOBOptimizer._check_botorch_available():
        with pytest.raises(ImportError, match="BoTorch and GPyTorch are required"):
            MOBOptimizer(adapter, allow_fallback=False)


def test_mobo_initial_design_and_observe_loop(test_grammar: SearchGrammar) -> None:
    adapter = BOSearchAdapter(test_grammar)
    opt = MOBOptimizer(adapter, n_init=3, allow_fallback=True, seed=123)

    assert opt.n_init == 3
    assert len(opt.initial_configs) == 3

    # Step 1: Suggest first batch
    configs = opt.suggest(q=3)
    assert len(configs) == 3
    assert configs[0].trial_id == "trial_mobo_0000"
    assert configs[1].trial_id == "trial_mobo_0001"
    assert configs[2].trial_id == "trial_mobo_0002"

    # Step 2: Observe evaluations
    opt.observe(configs[0], {"tool_accuracy": 0.55, "peak_process_rss_mb": 8500.0})
    opt.observe(configs[1], {"tool_accuracy": 0.62, "peak_process_rss_mb": 9200.0})
    opt.observe(configs[2], {"tool_accuracy": 0.48, "peak_process_rss_mb": 7800.0})

    assert len(opt.observed_X) == 3
    assert len(opt.observed_Y) == 3

    # Verify maximization convention [acc, -params, -rss]
    assert opt.observed_Y[0][0] == 0.55
    assert opt.observed_Y[0][2] == -8500.0

    # Step 3: Suggest next candidate (active phase)
    next_cand = opt.suggest(q=1)
    assert len(next_cand) == 1
    assert next_cand[0].trial_id == "trial_mobo_0003"

    # Step 4: Verify serialization
    state = opt.state_dict()
    assert state["trial_counter"] == 4
    assert len(state["observed_X"]) == 3
    assert len(state["observed_Y"]) == 3


def test_mobo_observe_missing_metric_raises(test_grammar: SearchGrammar) -> None:
    adapter = BOSearchAdapter(test_grammar)
    opt = MOBOptimizer(adapter, n_init=1, allow_fallback=True)
    cand = opt.suggest(q=1)[0]

    with pytest.raises(KeyError, match="Missing required metric 'tool_accuracy'"):
        opt.observe(cand, {"peak_process_rss_mb": 8000.0})

    with pytest.raises(KeyError, match="Missing required metric 'peak_process_rss_mb'"):
        opt.observe(cand, {"tool_accuracy": 0.5})
