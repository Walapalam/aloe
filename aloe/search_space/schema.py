"""The search grammar. This object is EMITTED from an ArchitectureDescription —
never hand-written, never hardcoded per model. That is Gap 3."""
from __future__ import annotations

from pydantic import BaseModel, Field

from aloe.architecture.schema import ComponentType


class GroupDimensions(BaseModel):
    component_type: ComponentType
    targets: list[str]                       # resolved dotted paths, VALID only
    enabled: list[bool] = [True, False]
    rank: list[int] = [4, 8, 16, 32]
    alpha: list[int] = [8, 16, 32, 64, 128]  # deliberately wider than the 2r heuristic (Gap 2)
    notes: str = ""


class SearchGrammar(BaseModel):
    model_id: str
    family: str
    topology: str
    backend: str
    groups: list[GroupDimensions] = Field(default_factory=list)
    excluded: dict[str, str] = Field(default_factory=dict)  # module -> reason

    def cardinality(self, include_all_disabled: bool = False) -> int:
        """Calculates the exact distinct configurations in the search space.
        When enabled=False for a group, rank and alpha choices are irrelevant (1 distinct state).
        When enabled=True, there are |rank| * |alpha| states.
        Per group: 1 + (|rank| * |alpha|).
        If include_all_disabled is False, subtracts 1 (the trivial all-off configuration).
        """
        if not self.groups:
            return 0
        n = 1
        for g in self.groups:
            if False in g.enabled:
                group_states = 1 + (len(g.rank) * len(g.alpha))
            else:
                group_states = len(g.rank) * len(g.alpha)
            n *= group_states
        return n if include_all_disabled else max(1, n - 1)

