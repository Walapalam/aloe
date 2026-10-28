"""Unit tests for pre-training trainable parameter calculation and +/-5% budget parity (FR04)."""
from __future__ import annotations

import json
from pathlib import Path
import pytest

from aloe.architecture.schema import ArchitectureDescription
from aloe.execution.trial_config import GroupTrialSetting, InvalidConfigError, TrialConfig
from aloe.search_space.schema import SearchGrammar

BASE_DIR = Path(__file__).resolve().parent.parent


def _load_arch_and_grammar(model_slug: str, insp_dir: str) -> tuple[ArchitectureDescription, SearchGrammar]:
    insp_path = BASE_DIR / "runs" / "architecture" / insp_dir / "inspection.json"
    gram_path = BASE_DIR / "runs" / "grammars" / f"{model_slug}.json"
    insp_raw = json.loads(insp_path.read_text(encoding="utf-8"))["architecture"]
    arch = ArchitectureDescription.model_validate(insp_raw)
    grammar = SearchGrammar.model_validate_json(gram_path.read_text(encoding="utf-8"))
    return arch, grammar


def test_gemma_exact_trainable_parameter_counts() -> None:
    """Verify analytical parameter counts match PEFT runtime counts on Gemma 4 E2B."""
    arch, grammar = _load_arch_and_grammar("gemma_e2b", "google--gemma-4-E2B-it")

    # P1: Uniform r=8 across Local Attn, Global Attn, FFN -> 12,079,104
    p1 = TrialConfig(
        model_id="google/gemma-4-E2B-it",
        group_settings={
            "local_attention": GroupTrialSetting(enabled=True, rank=8, alpha=16),
            "global_attention": GroupTrialSetting(enabled=True, rank=8, alpha=16),
            "ffn": GroupTrialSetting(enabled=True, rank=8, alpha=16),
        },
    )
    assert p1.compute_trainable_params(arch, grammar) == 12_079_104

    # Unmatched FFN-pruned ablation (r_local=4, r_global=32, r_ffn=0) -> 3,891,200 (-67.79%)
    ablation = TrialConfig(
        model_id="google/gemma-4-E2B-it",
        group_settings={
            "local_attention": GroupTrialSetting(enabled=True, rank=4, alpha=16),
            "global_attention": GroupTrialSetting(enabled=True, rank=32, alpha=128),
            "ffn": GroupTrialSetting(enabled=False, rank=4, alpha=8),
        },
    )
    assert ablation.compute_trainable_params(arch, grammar) == 3_891_200
    with pytest.raises(InvalidConfigError, match="Parameter budget parity violation"):
        ablation.validate_parameter_budget_parity(arch, baseline_params=12_079_104, tolerance=0.05, grammar=grammar)

    # Budget-matched (+/-5%) Architecture-Typed Allocation (r_local=16, r_global=32, r_ffn=4) -> 11,515,904 (-4.66%)
    p4_matched = TrialConfig(
        model_id="google/gemma-4-E2B-it",
        group_settings={
            "local_attention": GroupTrialSetting(enabled=True, rank=16, alpha=32),
            "global_attention": GroupTrialSetting(enabled=True, rank=32, alpha=64),
            "ffn": GroupTrialSetting(enabled=True, rank=4, alpha=8),
        },
    )
    parity = p4_matched.validate_parameter_budget_parity(
        arch, baseline_params=12_079_104, tolerance=0.05, grammar=grammar
    )
    assert parity["candidate_params"] == 11_515_904
    assert parity["within_budget"] is True
    assert parity["relative_delta_pct"] == -4.66


def test_falcon_exact_trainable_parameter_counts() -> None:
    """Verify analytical parameter counts and +/-5% parity on Falcon-H1 1.5B."""
    arch, grammar = _load_arch_and_grammar("falcon_h1", "tiiuae--Falcon-H1-1.5B-Instruct")

    # P1: Uniform r=8 across Global Attn, SSM, FFN -> 7,578,624
    p1 = TrialConfig(
        model_id="tiiuae/Falcon-H1-1.5B-Instruct",
        group_settings={
            "global_attention": GroupTrialSetting(enabled=True, rank=8, alpha=16),
            "ssm": GroupTrialSetting(enabled=True, rank=8, alpha=16),
            "ffn": GroupTrialSetting(enabled=True, rank=8, alpha=16),
        },
    )
    assert p1.compute_trainable_params(arch, grammar) == 7_578_624

    # P2: Budget-matched (+/-5%) Typed A (r_global=16, r_ssm=8, r_ffn=4) -> 7,726,080 (+1.95%)
    p2_matched = TrialConfig(
        model_id="tiiuae/Falcon-H1-1.5B-Instruct",
        group_settings={
            "global_attention": GroupTrialSetting(enabled=True, rank=16, alpha=32),
            "ssm": GroupTrialSetting(enabled=True, rank=8, alpha=16),
            "ffn": GroupTrialSetting(enabled=True, rank=4, alpha=8),
        },
    )
    parity = p2_matched.validate_parameter_budget_parity(
        arch, baseline_params=7_578_624, tolerance=0.05, grammar=grammar
    )
    assert parity["candidate_params"] == 7_726_080
    assert parity["within_budget"] is True
    assert parity["relative_delta_pct"] == 1.95

    # P4_SSM_Matched: Budget-matched (+/-5%) SSM-Prioritized (r_global=8, r_ssm=16, r_ffn=4) -> 7,342,080 (-3.12%)
    p4_ssm_matched = TrialConfig(
        model_id="tiiuae/Falcon-H1-1.5B-Instruct",
        group_settings={
            "global_attention": GroupTrialSetting(enabled=True, rank=8, alpha=16),
            "ssm": GroupTrialSetting(enabled=True, rank=16, alpha=32),
            "ffn": GroupTrialSetting(enabled=True, rank=4, alpha=8),
        },
    )
    ssm_parity = p4_ssm_matched.validate_parameter_budget_parity(
        arch, baseline_params=7_578_624, tolerance=0.05, grammar=grammar
    )
    assert ssm_parity["candidate_params"] == 7_342_080
    assert ssm_parity["within_budget"] is True
    assert ssm_parity["relative_delta_pct"] == -3.12


def test_analytic_counts_match_recorded_peft_runtime_trials() -> None:
    """Verify compute_trainable_params matches actual PEFT runtime counts saved in trial.json records."""
    gemma_arch, gemma_gram = _load_arch_and_grammar("gemma_e2b", "google--gemma-4-E2B-it")
    falcon_arch, falcon_gram = _load_arch_and_grammar("falcon_h1", "tiiuae--Falcon-H1-1.5B-Instruct")

    for slug, arch, gram, cfgs in [
        ("gemma_e2b", gemma_arch, gemma_gram, ["P1", "P4_Matched", "P2", "P3"]),
        ("falcon_h1", falcon_arch, falcon_gram, ["P1", "P2", "P3"]),
    ]:
        for cid in cfgs:
            tpath = BASE_DIR / "runs" / "pilot" / slug / cid / "trial.json"
            if tpath.is_file():
                rec = json.loads(tpath.read_text(encoding="utf-8"))
                cfg = TrialConfig.model_validate(rec["trial_config"])
                runtime_params = rec["metrics"]["trainable_params"]
                assert cfg.compute_trainable_params(arch, gram) == runtime_params


def test_explicit_target_modules_non_default_rank_matches_peft() -> None:
    """Verify explicit_target_modules at r != 8 uses identical rank/alpha in both compute_trainable_params and PEFT."""
    import torch.nn as nn
    from peft import get_peft_model
    from aloe.architecture.schema import ComponentType, ModuleRecord, Validity

    class ToyModel(nn.Module):
        def __init__(self) -> None:
            super().__init__()
            self.q_proj = nn.Linear(64, 128, bias=False)
            self.v_proj = nn.Linear(64, 32, bias=False)

        def forward(self, x):
            return self.q_proj(x), self.v_proj(x)

        def prepare_inputs_for_generation(self, *args, **kwargs):
            return {}

    arch = ArchitectureDescription(
        model_id="toy/explicit-test",
        family="toy",
        num_layers=1,
        total_params=10240,
        topology="uniform",
        modules=[
            ModuleRecord(
                name="q_proj",
                module_class="Linear",
                layer_idx=0,
                component_type=ComponentType.GLOBAL_ATTENTION,
                validity=Validity.VALID,
                shape=[128, 64],
            ),
            ModuleRecord(
                name="v_proj",
                module_class="Linear",
                layer_idx=0,
                component_type=ComponentType.GLOBAL_ATTENTION,
                validity=Validity.VALID,
                shape=[32, 64],
            ),
        ],
    )
    cfg = TrialConfig(
        model_id="toy/explicit-test",
        group_settings={"global_attention": GroupTrialSetting(enabled=True, rank=16, alpha=32)},
        explicit_target_modules=["q_proj", "v_proj"],
    )
    analytic_params = cfg.compute_trainable_params(arch)
    assert analytic_params == 16 * (64 + 128) + 16 * (64 + 32)

    peft_kwargs, lora_cfg = cfg.to_peft_lora_config()
    assert peft_kwargs["rank_pattern"] == {"q_proj": 16, "v_proj": 16}
    assert peft_kwargs["alpha_pattern"] == {"q_proj": 32, "v_proj": 32}

    peft_model = get_peft_model(ToyModel(), lora_cfg)
    actual_trainable = sum(p.numel() for p in peft_model.parameters() if p.requires_grad)
    assert analytic_params == actual_trainable


def test_lfm3_exact_trainable_parameter_counts_and_budget_parity() -> None:
    """Verify analytical parameter counts and +/-5% parity across LFM3-1.7B's 4 groups."""
    arch, grammar = _load_arch_and_grammar("lfm3_1_7b", "LiquidAI--LFM3-1.7B")

    # P1: Uniform r=8 across Local Attn, Global Attn, SSM, FFN -> 8,052,736
    p1 = TrialConfig(
        model_id="LiquidAI/LFM3-1.7B",
        group_settings={
            "local_attention": GroupTrialSetting(enabled=True, rank=8, alpha=16),
            "global_attention": GroupTrialSetting(enabled=True, rank=8, alpha=16),
            "ssm": GroupTrialSetting(enabled=True, rank=8, alpha=16),
            "ffn": GroupTrialSetting(enabled=True, rank=8, alpha=16),
        },
    )
    assert p1.compute_trainable_params(arch, grammar) == 8_052_736

    # P2: Exact 0.00% budget-matched Global-Attention Prioritized (r_local=8, r_global=16, r_ssm=4, r_ffn=8)
    p2_exact = TrialConfig(
        model_id="LiquidAI/LFM3-1.7B",
        group_settings={
            "local_attention": GroupTrialSetting(enabled=True, rank=8, alpha=16),
            "global_attention": GroupTrialSetting(enabled=True, rank=16, alpha=32),
            "ssm": GroupTrialSetting(enabled=True, rank=4, alpha=8),
            "ffn": GroupTrialSetting(enabled=True, rank=8, alpha=16),
        },
    )
    parity_p2 = p2_exact.validate_parameter_budget_parity(
        arch, baseline_params=8_052_736, tolerance=0.05, grammar=grammar
    )
    assert parity_p2["candidate_params"] == 8_052_736
    assert parity_p2["within_budget"] is True
    assert parity_p2["relative_delta_pct"] == 0.0

    # P3: Budget-matched (-4.37%) Operator-Prioritized (r_local=16, r_global=16, r_ssm=16, r_ffn=4)
    p3_operators = TrialConfig(
        model_id="LiquidAI/LFM3-1.7B",
        group_settings={
            "local_attention": GroupTrialSetting(enabled=True, rank=16, alpha=32),
            "global_attention": GroupTrialSetting(enabled=True, rank=16, alpha=32),
            "ssm": GroupTrialSetting(enabled=True, rank=16, alpha=32),
            "ffn": GroupTrialSetting(enabled=True, rank=4, alpha=8),
        },
    )
    parity_p3 = p3_operators.validate_parameter_budget_parity(
        arch, baseline_params=8_052_736, tolerance=0.05, grammar=grammar
    )
    assert parity_p3["candidate_params"] == 7_700_480
    assert parity_p3["within_budget"] is True
    assert parity_p3["relative_delta_pct"] == -4.37

    # Excluded conv.conv and self_attn.gate_proj must be rejected by validate_against_grammar
    bad_conv = TrialConfig(
        model_id="LiquidAI/LFM3-1.7B",
        explicit_target_modules=["model.layers.0.conv.conv"],
    )
    with pytest.raises(InvalidConfigError, match="excluded by architecture grammar"):
        bad_conv.validate_against_grammar(grammar)

    bad_gate = TrialConfig(
        model_id="LiquidAI/LFM3-1.7B",
        explicit_target_modules=["model.layers.2.self_attn.gate_proj"],
    )
    with pytest.raises(InvalidConfigError, match="excluded by architecture grammar"):
        bad_gate.validate_against_grammar(grammar)

