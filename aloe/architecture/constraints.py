"""Stage 3: validity. THIS FILE IS THE RESEARCH CONTRIBUTION — see Gap 3.

Validity is a function of (architecture, backend), not of architecture alone.
A module that is a legal LoRA target under HuggingFace PEFT may be unsupported
under MLX, and vice versa. Every rule must carry a citation or an experiment ID
in its reason string. Never add a silent rule."""
from __future__ import annotations

from aloe.architecture.schema import ArchitectureDescription, ComponentType, Validity

# Rules are (family, module-name substring) -> (Validity, reason).
# Each reason must name its evidence. Rules without evidence do not go in.
KNOWN_EXCLUSIONS: dict[str, list[tuple[str, Validity, str]]] = {
    "falcon_h1": [
        ("conv1d", Validity.EXCLUDED,
         "Feeds a custom autograd Mamba kernel that bypasses the LoRA forward pass; "
         "adapter is silently inert. Borobia et al. (arXiv:2604.22127) Sec. 3.2 exclude "
         "it after observing training instability."),
        ("out_proj", Validity.EXCLUDED,
         "Same custom-kernel path as conv1d. Excluded in Borobia et al. Sec. 3.2."),
    ],
    "lfm3": [
        ("conv.conv", Validity.EXCLUDED,
         "Width-7 depthwise short-convolution kernel (shape [2048, 1, 7], groups=2048); "
         "requires r % groups == 0 under PEFT, cannot be merged, and is bypassed via direct "
         "self.conv.weight.squeeze(1) access in _segmented_conv / causal_conv1d_fn kernels."),
        ("self_attn.gate_proj", Validity.EXCLUDED,
         "Per-head scalar sigmoid attention gate (shape [16, 2048], out_features=num_attention_heads=16); "
         "rank r>=16 exceeds full matrix rank min(d_in, d_out)=16 and leaf name 'gate_proj' "
         "collides with standard SwiGLU FFN naming."),
    ],
}


def apply_validity(desc: ArchitectureDescription, backend: str) -> ArchitectureDescription:
    """Annotate every module with a Validity verdict and a reason.

    backend: 'cuda_peft' | 'mlx'
    """
    rules = KNOWN_EXCLUSIONS.get(desc.family, [])
    for m in desc.modules:
        verdict, reason = _default_verdict(m, backend)
        for needle, v, why in rules:
            if needle in m.name:
                verdict, reason = v, why
                break
        m.validity, m.validity_reason = verdict, reason
    return desc


def _default_verdict(m, backend: str) -> tuple[Validity, str]:
    if m.component_type in (ComponentType.NORM, ComponentType.EMBEDDING):
        return Validity.EXCLUDED, "Not a low-rank adaptation target by convention."
    if m.component_type == ComponentType.SSM and not m.is_linear:
        if "peft" in backend:
            return (
                Validity.EXCLUDED,
                "Non-linear recurrent module is not a standard PyTorch Linear layer; "
                "HuggingFace PEFT cannot inject LoRA adapter without custom kernel support.",
            )
        return (Validity.VALID_WITH_SPECIAL_HANDLER,
                "Recurrent-path tensor; requires backend-specific handling. Verify "
                "gradient flow with tests/test_gradient_reaches_adapter.py before trusting.")
    if m.component_type == ComponentType.EXPERT and backend == "cuda_peft":
        return (Validity.VALID_WITH_SPECIAL_HANDLER,
                "bitsandbytes 4-bit replaces only nn.Linear and skips MoE experts, so the "
                "bulk of an MoE stays in full precision. Quantisation must be verified, "
                "not assumed.")
    if m.is_linear:
        return Validity.VALID, "Standard nn.Linear target."
    return Validity.UNSUPPORTED, f"No rule for module class {m.module_class}."
