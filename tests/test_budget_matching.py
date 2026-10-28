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
