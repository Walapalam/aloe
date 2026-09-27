"""Falcon-H1 — PARALLEL hybrid: attention branch and Mamba-2 SSM branch run on the
same input, outputs summed before the MLP.

Borobia et al. (2604.22127) Sec. 3.2 exclude conv1d and out_proj. Encoded as data in
constraints.py, not as an if-statement here."""
from __future__ import annotations

from typing import Any

from aloe.architecture.constraints import KNOWN_EXCLUSIONS
from aloe.architecture.schema import ArchitectureDescription, ComponentType, Validity
from aloe.architecture.typing import register

if not any(x[0] == "lm_head" for x in KNOWN_EXCLUSIONS.get("falcon_h1", [])):
    KNOWN_EXCLUSIONS.setdefault("falcon_h1", []).append(
        ("lm_head", Validity.EXCLUDED, "Language model head; excluded by convention.")
    )


class FalconH1Adapter:
    family = "falcon_h1"
    topology = "parallel_hybrid"

    def matches(self, model_id: str, config: Any) -> bool:
        return "falcon-h1" in model_id.lower() or "falcon_h1" in model_id.lower()

    def classify(
        self, desc: ArchitectureDescription, config: Any = None
    ) -> ArchitectureDescription:
        for m in desc.modules:
            if "embed_tokens" in m.name:
                m.component_type = ComponentType.EMBEDDING
            elif "layernorm" in m.name or m.module_class.endswith("Norm"):
                m.component_type = ComponentType.NORM
            elif "feed_forward" in m.name or "mlp" in m.name:
                m.component_type = ComponentType.FFN
            elif "self_attn" in m.name:
                m.component_type = ComponentType.GLOBAL_ATTENTION
            elif "mamba" in m.name:
                m.component_type = ComponentType.SSM
            else:
                m.component_type = ComponentType.OTHER
                m.validity = Validity.EXCLUDED
                if m.name == "lm_head":
                    m.validity_reason = "Language model head; excluded by convention."
                else:
                    m.validity_reason = "Auxiliary component; excluded from LoRA search space."

        return desc


register(FalconH1Adapter())
