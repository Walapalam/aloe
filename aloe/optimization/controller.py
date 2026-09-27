"""Optimization controller orchestrating multi-objective search campaigns."""
from __future__ import annotations

from typing import Any

from aloe.optimization.objectives import DEFAULT_ALOE_OBJECTIVES, MultiObjectivePoint, ObjectiveConfig
from aloe.optimization.pareto import calculate_hypervolume_2d, extract_pareto_frontier


class OptimizationController:
    """Manages multi-objective trial evaluations and maintains the Pareto frontier."""

    def __init__(
        self,
        objectives: list[ObjectiveConfig] | None = None,
    ) -> None:
        self.objectives = objectives or DEFAULT_ALOE_OBJECTIVES
        self.points: list[MultiObjectivePoint] = []
        self._pareto_frontier: list[MultiObjectivePoint] = []
        self._directions = {obj.name: obj.minimize for obj in self.objectives}

    def add_evaluation(
        self,
        point_id: str,
        objectives: dict[str, float],
        metadata: dict[str, Any] | None = None,
    ) -> MultiObjectivePoint:
        """Record an evaluation outcome and update the Pareto frontier."""
        point = MultiObjectivePoint(
            point_id=point_id,
            objectives=objectives,
            metadata=metadata or {},
        )
        self.points.append(point)
        self._update_pareto()
        return point

    def _update_pareto(self) -> None:
        """Re-compute the non-dominated Pareto frontier."""
        self._pareto_frontier = extract_pareto_frontier(self.points, self._directions)

    @property
    def pareto_frontier(self) -> list[MultiObjectivePoint]:
        """Return the current non-dominated points."""
        return list(self._pareto_frontier)

    def get_best_per_objective(self) -> dict[str, MultiObjectivePoint]:
        """Return the best individual point for each objective."""
        best: dict[str, MultiObjectivePoint] = {}
        if not self.points:
            return best

        for obj in self.objectives:
            name = obj.name
            minimize = obj.minimize
            valid_pts = [p for p in self.points if name in p.objectives]
            if not valid_pts:
                continue
            if minimize:
                best_pt = min(valid_pts, key=lambda p: p.objectives[name])
            else:
                best_pt = max(valid_pts, key=lambda p: p.objectives[name])
            best[name] = best_pt

        return best

    def get_summary(self) -> dict[str, Any]:
        """Return a structured summary of the search state."""
        best_pts = self.get_best_per_objective()
        return {
            "total_evaluated": len(self.points),
            "pareto_frontier_size": len(self._pareto_frontier),
            "pareto_points": [p.model_dump() for p in self._pareto_frontier],
            "best_per_objective": {k: v.model_dump() for k, v in best_pts.items()},
        }
