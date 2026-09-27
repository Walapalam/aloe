"""Gemma 4 E2B — interleaved local/global attention.

Expected: 35 layers in a repeating 4-local + 1-global pattern (7 repeats = 28 local,
7 global). VERIFY THIS AGAINST THE LOADED CONFIG — do not trust the docstring.
KV-cache sharing means k_proj/v_proj may be absent on some layers; a grammar that
assumes all seven projections on all 35 layers is invalid."""
from __future__ import annotations

from typing import Any

from aloe.architecture.constraints import KNOWN_EXCLUSIONS
from aloe.architecture.schema import ArchitectureDescription, ComponentType, Validity
from aloe.architecture.typing import register

_GEMMA4_EXCLUSIONS: list[tuple[str, Validity, str]] = [
    ("vision_tower", Validity.EXCLUDED, "Auxiliary multimodal component; excluded from language LoRA search space."),
    ("audio_tower", Validity.EXCLUDED, "Auxiliary multimodal component; excluded from language LoRA search space."),
    ("embed_vision", Validity.EXCLUDED, "Auxiliary multimodal component; excluded from language LoRA search space."),
    ("embed_audio", Validity.EXCLUDED, "Auxiliary multimodal component; excluded from language LoRA search space."),
    ("lm_head", Validity.EXCLUDED, "Language model head; excluded by convention."),
    ("per_layer", Validity.EXCLUDED, "Per-layer embedding gating/projection; excluded from language LoRA search space."),
]

existing_gemma = {x[0] for x in KNOWN_EXCLUSIONS.get("gemma4", [])}
for needle, v, why in _GEMMA4_EXCLUSIONS:
    if needle not in existing_gemma:
        KNOWN_EXCLUSIONS.setdefault("gemma4", []).append((needle, v, why))


class Gemma4Adapter:
    family = "gemma4"
    topology = "interleaved"

    def __init__(self) -> None:
        self._last_config: Any = None

    def matches(self, model_id: str, config: Any) -> bool:
        self._last_config = config
        return "gemma" in model_id.lower()

    def is_global_layer(self, layer_index: int, config: Any = None) -> bool:
        """Derive whether layer_index is global from config (text_config.layer_types).

        Raise KeyError if layer_types field is missing — FAIL LOUD.
        """
        cfg = config if config is not None else getattr(self, "_last_config", None)
        if cfg is None:
            raise KeyError("Gemma 4 config required to determine local/global attention pattern.")

        tc = getattr(cfg, "text_config", cfg)
        if isinstance(tc, dict):
            if "layer_types" not in tc:
                raise KeyError("Gemma 4 config missing 'layer_types' field in text_config (FAIL LOUD).")
            layer_types = tc["layer_types"]
        else:
            if not hasattr(tc, "layer_types"):
                raise KeyError("Gemma 4 config missing 'layer_types' field in text_config (FAIL LOUD).")
            layer_types = tc.layer_types

        if layer_index < 0 or layer_index >= len(layer_types):
            raise IndexError(
                f"Layer index {layer_index} out of bounds for layer_types (len={len(layer_types)})."
            )

        return layer_types[layer_index] == "full_attention"

    def classify(
        self, desc: ArchitectureDescription, config: Any = None
    ) -> ArchitectureDescription:
        cfg = config if config is not None else getattr(self, "_last_config", None)

        for m in desc.modules:
            if "language_model" in m.name:
                if "embed_tokens" in m.name or "embed_tokens_per_layer" in m.name:
                    m.component_type = ComponentType.EMBEDDING
                elif "norm" in m.name or m.module_class.endswith("Norm") or "layernorm" in m.name:
                    m.component_type = ComponentType.NORM
                elif "mlp" in m.name:
                    m.component_type = ComponentType.FFN
                elif "self_attn" in m.name:
                    if m.layer_index is not None and self.is_global_layer(m.layer_index, cfg):
                        m.component_type = ComponentType.GLOBAL_ATTENTION
                    else:
                        m.component_type = ComponentType.LOCAL_ATTENTION
                else:
                    m.component_type = ComponentType.OTHER
                    m.validity = Validity.EXCLUDED
                    m.validity_reason = (
                        "Auxiliary per-layer component; excluded from language LoRA search space."
                    )
            else:
                m.component_type = ComponentType.OTHER
                m.validity = Validity.EXCLUDED
                if m.name == "lm_head":
                    m.validity_reason = "Language model head; excluded by convention."
                else:
                    m.validity_reason = (
                        "Multimodal auxiliary component; excluded from language LoRA search space."
                    )

        return desc


register(Gemma4Adapter())
