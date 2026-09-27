"""Core data types. Backend-agnostic by design — nothing here may import torch-CUDA,
bitsandbytes, peft or mlx. See docs/DECISIONS.md ADR-002."""
from __future__ import annotations

from enum import Enum
from typing import Literal

from pydantic import BaseModel, Field


class ComponentType(str, Enum):
    """Architectural role of a module. This is the semantic layer that makes ALOE
    'architecture-typed' rather than 'module-name matched'. Extend per family, but
    every new member must be justified in DECISIONS.md."""

    LOCAL_ATTENTION = "local_attention"     # sliding-window attention (Gemma 4)
    GLOBAL_ATTENTION = "global_attention"   # full softmax attention
    LINEAR_ATTENTION = "linear_attention"   # GatedDeltaNet etc. (Qwen3.5)
    SSM = "ssm"                             # Mamba / Mamba-2 recurrent path (Falcon-H1)
    FFN = "ffn"
    ROUTER = "router"                       # MoE gate
    EXPERT = "expert"                       # MoE expert FFN
    EMBEDDING = "embedding"
    NORM = "norm"
    OTHER = "other"


class Validity(str, Enum):
    """Whether a module may legally host a LoRA adapter under a given backend.

    The distinction that carries Gap 3: VALID means 'can receive a gradient', not
    'the adapter attached successfully'. Borobia et al. (2604.22127) Sec. 3.2 verify
    the latter; they discovered the former by hitting training instability."""

    VALID = "valid"
    VALID_WITH_SPECIAL_HANDLER = "valid_with_special_handler"
    UNSUPPORTED = "unsupported"   # backend cannot adapt this module type at all
    EXCLUDED = "excluded"         # structurally present but provably inert / unsafe


class ModuleRecord(BaseModel):
    name: str                       # full dotted path
    module_class: str
    layer_index: int | None = None
    shape: tuple[int, ...] | None = None
    param_count: int = 0
    parent: str | None = None
    is_linear: bool = False
    component_type: ComponentType = ComponentType.OTHER
    validity: Validity = Validity.UNSUPPORTED
    validity_reason: str = ""


class ArchitectureDescription(BaseModel):
    model_id: str
    family: str
    num_layers: int
    total_params: int
    topology: Literal["uniform", "interleaved", "sequential_hybrid", "parallel_hybrid", "moe"]
    modules: list[ModuleRecord] = Field(default_factory=list)

    def by_type(self, t: ComponentType) -> list[ModuleRecord]:
        return [m for m in self.modules if m.component_type == t]

    def adaptable_types(self) -> list[ComponentType]:
        """Component types with at least one VALID module — these become search groups."""
        seen: dict[ComponentType, None] = {}
        for m in self.modules:
            if m.validity in (Validity.VALID, Validity.VALID_WITH_SPECIAL_HANDLER):
                seen.setdefault(m.component_type, None)
        return list(seen)
