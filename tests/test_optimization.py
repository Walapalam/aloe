"""Unit tests for ALOE multi-objective optimization and Pareto filtering."""
from __future__ import annotations

from aloe.architecture.schema import ComponentType
from aloe.optimization.controller import OptimizationController
from aloe.optimization.objectives import MultiObjectivePoint, ObjectiveConfig
from aloe.optimization.pareto import calculate_hypervolume_2d, extract_pareto_frontier, is_dominated
from aloe.optimization.random_search import RandomSearchSampler
from aloe.search_space.schema import GroupDimensions, SearchGrammar


def test_is_dominated() -> None:
    directions = {"accuracy": False, "memory": True}  # Maximize acc, minimize mem

    # a: acc=0.8, mem=500
    # b: acc=0.9, mem=400 (b is strictly better on both -> dominates a)
    assert is_dominated({"accuracy": 0.8, "memory": 500}, {"accuracy": 0.9, "memory": 400}, directions)

    # c: acc=0.9, mem=600 (c has higher acc, but higher mem -> neither dominates)
    assert not is_dominated({"accuracy": 0.8, "memory": 500}, {"accuracy": 0.9, "memory": 600}, directions)
    assert not is_dominated({"accuracy": 0.9, "memory": 600}, {"accuracy": 0.8, "memory": 500}, directions)


def test_extract_pareto_frontier() -> None:
    pts = [
        MultiObjectivePoint(point_id="p1", objectives={"tool_accuracy": 0.70, "trainable_parameters": 1e6, "peak_process_rss_mb": 4000}),
        MultiObjectivePoint(point_id="p2", objectives={"tool_accuracy": 0.85, "trainable_parameters": 2e6, "peak_process_rss_mb": 5000}),
        MultiObjectivePoint(point_id="p3", objectives={"tool_accuracy": 0.65, "trainable_parameters": 3e6, "peak_process_rss_mb": 6000}),  # Dominated by p1 and p2
        MultiObjectivePoint(point_id="p4", objectives={"tool_accuracy": 0.90, "trainable_parameters": 4e6, "peak_process_rss_mb": 7000}),
    ]
    frontier = extract_pareto_frontier(pts)
    frontier_ids = {p.point_id for p in frontier}
    assert "p1" in frontier_ids
    assert "p2" in frontier_ids
    assert "p4" in frontier_ids
    assert "p3" not in frontier_ids


def test_calculate_hypervolume_2d() -> None:
    # 2 non-dominated points in (acc, inverse_mem) where both are to be maximized
    # Point 1: (2, 4)
    # Point 2: (4, 2)
    # Reference nadir: (0, 0)
    # Hypervolume area = (2 * 4) + (4 - 2) * 2 = 8 + 4 = 12
    hv = calculate_hypervolume_2d([(2.0, 4.0), (4.0, 2.0)], (0.0, 0.0))
    assert hv == 12.0


def test_random_search_sampler() -> None:
    grammar = SearchGrammar(
        model_id="test_model",
        family="test_family",
        topology="test_topology",
        backend="peft_mps",
        groups=[
            GroupDimensions(
                component_type=ComponentType.GLOBAL_ATTENTION,
                targets=["q_proj", "k_proj", "v_proj"],
                rank=[2, 4, 8],
                alpha=[4, 8, 16],
            ),
            GroupDimensions(
                component_type=ComponentType.FFN,
                targets=["gate_proj", "up_proj"],
                rank=[4, 8, 16],
                alpha=[8, 16],
            ),
        ],
    )
    sampler = RandomSearchSampler(grammar, seed=123)
    cand = sampler.sample_candidate("trial_001")
    assert cand.trial_id == "trial_001"
    assert cand.model_id == "test_model"
    assert len(cand.group_settings) == 2
    assert cand.group_settings[ComponentType.GLOBAL_ATTENTION.value].rank in [2, 4, 8]


def test_optimization_controller() -> None:
    ctrl = OptimizationController()
    p1 = ctrl.add_evaluation("t1", {"tool_accuracy": 0.75, "trainable_parameters": 1e6, "peak_process_rss_mb": 4500})
    p2 = ctrl.add_evaluation("t2", {"tool_accuracy": 0.85, "trainable_parameters": 2e6, "peak_process_rss_mb": 5200})
    p3 = ctrl.add_evaluation("t3", {"tool_accuracy": 0.70, "trainable_parameters": 3e6, "peak_process_rss_mb": 6000})

    assert len(ctrl.pareto_frontier) == 2
    summary = ctrl.get_summary()
    assert summary["total_evaluated"] == 3
    assert summary["pareto_frontier_size"] == 2
    best = ctrl.get_best_per_objective()
    assert best["tool_accuracy"].point_id == "t2"
    assert best["trainable_parameters"].point_id == "t1"
