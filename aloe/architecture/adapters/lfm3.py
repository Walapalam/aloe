"""LiquidAI LFM3-1.7B — SEQUENTIAL tri-hybrid: gated short-convolution (LIV),
sliding-window local attention, and full-context global attention interleaved
across 24 decoder layers with SwiGLU feed-forward blocks (w1, w2, w3).

Expected (from config.layer_types — VERIFY AGAINST LOADED CONFIG, never hardcode):
- 13 'conv' layers (indices 0, 1, 3, 5, 7, 9, 11, 13, 15, 17, 19, 21, 23):
  Lfm3ShortConv with Linear in_proj, Linear out_proj, and depthwise Conv1d conv.
- 5 'sliding_attention' layers (indices 4, 8, 12, 16, 20; sliding_window=256):
  Lfm3Attention with q_proj, k_proj, v_proj, o_proj, q_norm, k_norm, and gate_proj.
- 6 'full_attention' layers (indices 2, 6, 10, 14, 18, 22; partial RoPE 0.5):
  Lfm3Attention with q_proj, k_proj, v_proj, o_proj, q_norm, k_norm, and gate_proj.
"""
from __future__ import annotations

from typing import Any

from aloe.architecture.constraints import KNOWN_EXCLUSIONS
from aloe.architecture.schema import ArchitectureDescription, ComponentType, Validity
from aloe.architecture.typing import register

_LFM3_EXCLUSIONS: list[tuple[str, Validity, str]] = [
    (
        "conv.conv",
        Validity.EXCLUDED,
        "Width-7 depthwise short-convolution kernel (shape [2048, 1, 7], groups=2048); "
        "requires r % groups == 0 under PEFT, cannot be merged, and is bypassed via direct "
        "self.conv.weight.squeeze(1) access in _segmented_conv / causal_conv1d_fn kernels.",
    ),
    (
        "self_attn.gate_proj",
        Validity.EXCLUDED,
        "Per-head scalar sigmoid attention gate (shape [16, 2048], out_features=num_attention_heads=16); "
        "rank r>=16 exceeds full matrix rank min(d_in, d_out)=16 and leaf name 'gate_proj' "
        "collides with standard SwiGLU FFN naming.",
    ),
    ("lm_head", Validity.EXCLUDED, "Language model head; excluded by convention."),
]

existing_lfm3 = {x[0] for x in KNOWN_EXCLUSIONS.get("lfm3", [])}
for needle, v, why in _LFM3_EXCLUSIONS:
    if needle not in existing_lfm3:
        KNOWN_EXCLUSIONS.setdefault("lfm3", []).append((needle, v, why))

_ALLOWED_LAYER_TYPES = {"conv", "sliding_attention", "full_attention"}


class Lfm3Adapter:
    family = "lfm3"
    topology = "sequential_hybrid"

    def __init__(self) -> None:
        self._last_config: Any = None

    def matches(self, model_id: str, config: Any) -> bool:
        self._last_config = config
        mid = model_id.lower()
        if "lfm3" in mid or "lfm-3" in mid:
            return True
        model_type = getattr(config, "model_type", None) if config is not None else None
        if isinstance(config, dict):
            model_type = config.get("model_type", model_type)
        return model_type == "lfm3"

    def layer_type(self, layer_index: int, config: Any = None) -> str:
        """Extract exact per-layer operator type from config.layer_types.

        Raises KeyError if config or layer_types is missing — FAIL LOUD.
        """
        cfg = config if config is not None else getattr(self, "_last_config", None)
        if cfg is None:
            raise KeyError("LFM3 config required to determine per-layer operator types (layer_types).")

        tc = getattr(cfg, "text_config", None) or cfg
        if isinstance(tc, dict):
            if "layer_types" not in tc or tc["layer_types"] is None:
                raise KeyError("LFM3 config missing 'layer_types' field (FAIL LOUD).")
            layer_types = tc["layer_types"]
        else:
            if not hasattr(tc, "layer_types") or tc.layer_types is None:
                raise KeyError("LFM3 config missing 'layer_types' field (FAIL LOUD).")
            layer_types = tc.layer_types

        if layer_index < 0 or layer_index >= len(layer_types):
            raise IndexError(
                f"Layer index {layer_index} out of bounds for layer_types (len={len(layer_types)})."
            )

        lt = str(layer_types[layer_index])
        if lt not in _ALLOWED_LAYER_TYPES:
            raise ValueError(
                f"Unexpected LFM3 layer_type {lt!r} at layer {layer_index}; expected one of {sorted(_ALLOWED_LAYER_TYPES)}."
            )
        return lt

    def classify(
        self, desc: ArchitectureDescription, config: Any = None
    ) -> ArchitectureDescription:
        cfg = config if config is not None else getattr(self, "_last_config", None)

        for m in desc.modules:
            if "embed_tokens" in m.name:
                m.component_type = ComponentType.EMBEDDING
            elif "norm" in m.name or m.module_class.endswith("Norm") or "layernorm" in m.name:
                m.component_type = ComponentType.NORM
            elif "feed_forward" in m.name or "mlp" in m.name:
                m.component_type = ComponentType.FFN
            elif "self_attn" in m.name:
                if "gate_proj" in m.name:
                    m.component_type = ComponentType.OTHER
                    m.validity = Validity.EXCLUDED
                    m.validity_reason = (
                        "Per-head scalar sigmoid attention gate (shape [16, 2048], "
                        "out_features=num_attention_heads=16); rank r>=16 exceeds full matrix rank "
                        "min(d_in, d_out)=16 and leaf name 'gate_proj' collides with standard SwiGLU FFN naming."
                    )
                else:
                    if m.layer_index is None:
                        raise KeyError(f"Attention module {m.name!r} missing layer_index in LFM3.")
                    lt = self.layer_type(m.layer_index, cfg)
                    if lt == "full_attention":
                        m.component_type = ComponentType.GLOBAL_ATTENTION
                    elif lt == "sliding_attention":
                        m.component_type = ComponentType.LOCAL_ATTENTION
                    else:
                        raise ValueError(
                            f"Module {m.name!r} has self_attn on non-attention layer {m.layer_index} ({lt!r})."
                        )
            elif ".conv." in m.name or m.name.endswith(".conv"):
                m.component_type = ComponentType.SSM
            else:
                m.component_type = ComponentType.OTHER
                m.validity = Validity.EXCLUDED
                if m.name == "lm_head":
                    m.validity_reason = "Language model head; excluded by convention."
                else:
                    m.validity_reason = "Auxiliary component; excluded from LoRA search space."

        return desc


register(Lfm3Adapter())
