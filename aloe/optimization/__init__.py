"""ALOE Optimization Engine.

Provides multi-objective evaluation metrics, Pareto frontier identification,
random search sampling with budget matching, and search controllers.
"""
from __future__ import annotations

from aloe.optimization.objectives import MultiObjectivePoint, ObjectiveConfig
from aloe.optimization.pareto import calculate_hypervolume_2d, extract_pareto_frontier, is_dominated
from aloe.optimization.random_search import RandomSearchSampler

__all__ = [
    "MultiObjectivePoint",
    "ObjectiveConfig",
    "RandomSearchSampler",
    "calculate_hypervolume_2d",
    "extract_pareto_frontier",
    "is_dominated",
]
