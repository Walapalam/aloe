"""Objective definitions and data models for ALOE multi-objective optimization."""
from __future__ import annotations

from typing import Any

from pydantic import BaseModel, Field


class ObjectiveConfig(BaseModel):
    """Specification of an optimization objective."""

    name: str
    minimize: bool = True
    unit: str = ""
    description: str = ""


class MultiObjectivePoint(BaseModel):
    """A single evaluation point in the multi-objective objective space."""

    point_id: str
    objectives: dict[str, float]
    metadata: dict[str, Any] = Field(default_factory=dict)

    def get_value(self, name: str) -> float:
        """Get the value for an objective, raising KeyError if absent."""
        if name not in self.objectives:
            raise KeyError(f"Objective '{name}' not found in point '{self.point_id}'.")
        return self.objectives[name]


# Standard ALOE 3-Objective Specification
DEFAULT_ALOE_OBJECTIVES: list[ObjectiveConfig] = [
    ObjectiveConfig(
        name="tool_accuracy",
        minimize=False,
        unit="proportion",
        description="Argument-level exact match tool-calling accuracy (0.0 to 1.0)",
    ),
    ObjectiveConfig(
        name="trainable_parameters",
        minimize=True,
        unit="count",
        description="Total trainable adapter parameter count",
    ),
    ObjectiveConfig(
        name="peak_process_rss_mb",
        minimize=True,
        unit="MB",
        description="Peak process resident set size memory during training and inference",
    ),
]
