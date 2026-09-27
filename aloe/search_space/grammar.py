"""Stage 4: ArchitectureDescription + validity -> SearchGrammar."""
from __future__ import annotations

from aloe.architecture.schema import ArchitectureDescription, Validity
from aloe.search_space.schema import GroupDimensions, SearchGrammar


def generate(desc: ArchitectureDescription, backend: str) -> SearchGrammar:
    """One group per adaptable component type. Excluded modules are recorded with
    their reason so the grammar is auditable."""
    groups = []
    for ct in desc.adaptable_types():
        targets = [m.name for m in desc.by_type(ct)
                   if m.validity in (Validity.VALID, Validity.VALID_WITH_SPECIAL_HANDLER)]
        if targets:
            groups.append(GroupDimensions(component_type=ct, targets=targets))
    excluded = {m.name: m.validity_reason for m in desc.modules
                if m.validity in (Validity.EXCLUDED, Validity.UNSUPPORTED)}
    return SearchGrammar(model_id=desc.model_id, family=desc.family,
                         topology=desc.topology, backend=backend,
                         groups=groups, excluded=excluded)
