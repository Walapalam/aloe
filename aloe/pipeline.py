"""High-level inspection and grammar generation pipeline for ALOE (Prompt 8 / Gate B).

Implements inspect_to_grammar(model_id, backend) which:
1. Loads model on PyTorch meta device (no weights loaded).
2. Performs structural inspection -> ArchitectureDescription.
3. Performs architectural classification -> ComponentType.
4. Applies backend-aware validity rules -> Validity.
5. Emits architecture-typed SearchGrammar.
"""
from __future__ import annotations

from pathlib import Path
from typing import Any

from aloe.architecture.constraints import apply_validity
from aloe.architecture.inspector import inspect_model
from aloe.architecture.schema import ArchitectureDescription, ComponentType, Validity
from aloe.architecture.typing import classify
from aloe.datasets.model_registry import ModelRegistry
from aloe.search_space.grammar import generate
from aloe.search_space.schema import SearchGrammar


def inspect_to_grammar(
    model_id: str,
    backend: str = "auto",
    runs_models_dir: Path | str = "runs/models",
) -> tuple[ArchitectureDescription, SearchGrammar]:
    """Execute the full inspection-to-grammar chain on a model."""
    if backend == "auto":
        backend = "peft_mps"

    reg = ModelRegistry(runs_models_dir)
    model_meta = reg.load_model_meta(model_id)
    config = reg.load_config(model_id)
    record = reg.get(model_id)
    canonical_id = record.model.id

    # 1. Structural inspection on meta tensors
    desc = inspect_model(model_meta, model_id=canonical_id)

    # 2. Architectural typing & role classification
    desc = classify(desc, model_id=canonical_id, config=config)

    # 3. Apply backend-aware validity constraints
    desc = apply_validity(desc, backend=backend)

    # Ensure auxiliary OTHER modules in language models are excluded
    for m in desc.modules:
        if m.component_type == ComponentType.OTHER and m.validity != Validity.EXCLUDED:
            m.validity = Validity.EXCLUDED
            m.validity_reason = "Auxiliary module; excluded from LoRA search space."

    # 4. Generate search space grammar
    grammar = generate(desc, backend=backend)

    return desc, grammar
