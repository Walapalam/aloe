"""Stage 2: assign architectural MEANING to modules. A ModelAdapter per family.

Adding a new architecture must mean writing one adapter here and nothing else.
If you find yourself editing the search space, executor or optimiser to support a
new model, the abstraction has failed — that is the reusability claim (SQ2)."""
from __future__ import annotations

from typing import Protocol

from aloe.architecture.schema import ArchitectureDescription, ComponentType


class ModelAdapter(Protocol):
    family: str
    topology: str

    def matches(self, model_id: str, config) -> bool: ...
    def classify(self, desc: ArchitectureDescription, config=...) -> ArchitectureDescription: ...


_REGISTRY: list[ModelAdapter] = []


def register(adapter: ModelAdapter) -> ModelAdapter:
    _REGISTRY.append(adapter)
    return adapter


def _ensure_adapters_loaded() -> None:
    if not _REGISTRY:
        import importlib

        importlib.import_module("aloe.architecture.adapters.falcon_h1")
        importlib.import_module("aloe.architecture.adapters.gemma4")


def resolve(model_id: str, config) -> ModelAdapter:
    _ensure_adapters_loaded()
    for a in _REGISTRY:
        if a.matches(model_id, config):
            return a
    raise LookupError(
        f"No ModelAdapter matches {model_id!r}. Write one in aloe/architecture/adapters/ "
        f"rather than special-casing elsewhere."
    )


def classify(desc: ArchitectureDescription, model_id: str, config) -> ArchitectureDescription:
    adapter = resolve(model_id, config)
    desc.family, desc.topology = adapter.family, adapter.topology
    try:
        return adapter.classify(desc, config=config)
    except TypeError:
        return adapter.classify(desc)


__all__ = ["ComponentType", "ModelAdapter", "classify", "register", "resolve"]
