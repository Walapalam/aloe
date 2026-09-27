"""Tests for Gate B: Search Grammar Differentiation across model topologies."""
import json
from pathlib import Path
import pytest
from aloe.search_space.schema import SearchGrammar
from aloe.architecture.schema import ComponentType


@pytest.fixture
def grammars():
    base_dir = Path(__file__).resolve().parent.parent
    grammar_dir = base_dir / "runs" / "grammars"
    
    files = {
        "e2b": grammar_dir / "gemma_e2b.json",
        "e4b": grammar_dir / "gemma_e4b.json",
        "falcon": grammar_dir / "falcon_h1.json",
    }
    for k, p in files.items():
        if not p.exists():
            pytest.skip(f"Grammar file missing: {p}")
            
    return {
        k: SearchGrammar.model_validate_json(p.read_text())
        for k, p in files.items()
    }


def test_attention_differentiation(grammars):
    """Assert (a): Gemma contains LOCAL_ATTENTION & GLOBAL_ATTENTION; Falcon does not."""
    e2b_types = {g.component_type for g in grammars["e2b"].groups}
    e4b_types = {g.component_type for g in grammars["e4b"].groups}
    falcon_types = {g.component_type for g in grammars["falcon"].groups}

    assert ComponentType.LOCAL_ATTENTION in e2b_types
    assert ComponentType.GLOBAL_ATTENTION in e2b_types
    assert ComponentType.LOCAL_ATTENTION in e4b_types
    assert ComponentType.GLOBAL_ATTENTION in e4b_types

    assert ComponentType.LOCAL_ATTENTION not in falcon_types
    assert ComponentType.GLOBAL_ATTENTION in falcon_types


def test_ssm_differentiation(grammars):
    """Assert (b): Falcon contains SSM; Gemma does not."""
    e2b_types = {g.component_type for g in grammars["e2b"].groups}
    e4b_types = {g.component_type for g in grammars["e4b"].groups}
    falcon_types = {g.component_type for g in grammars["falcon"].groups}

    assert ComponentType.SSM in falcon_types
    assert ComponentType.SSM not in e2b_types
    assert ComponentType.SSM not in e4b_types


def test_falcon_mamba_conv1d_excluded(grammars):
    """Assert (c): Falcon excluded contains 24 conv1d naming Mamba kernel."""
    falcon_excluded = grammars["falcon"].excluded
    conv1d_entries = {k: v for k, v in falcon_excluded.items() if "conv1d" in k.lower()}

    assert len(conv1d_entries) == 24, f"Expected 24 conv1d excluded entries, found {len(conv1d_entries)}"
    
    # Verify reason names Mamba autograd bypass
    for module_path, reason in conv1d_entries.items():
        assert "mamba" in module_path.lower()
        assert "mamba" in reason.lower() or "kernel" in reason.lower()
        assert "lora" in reason.lower()


def test_gemma_e2b_attention_target_count(grammars):
    """Assert (d): Gemma E2B attention count reflects k/v proj on layers 0-14 only (100 total = 80 local + 20 global)."""
    groups_by_type = {g.component_type: g for g in grammars["e2b"].groups}
    local_targets = groups_by_type[ComponentType.LOCAL_ATTENTION].targets
    global_targets = groups_by_type[ComponentType.GLOBAL_ATTENTION].targets

    assert len(local_targets) == 80
    assert len(global_targets) == 20
    assert len(local_targets) + len(global_targets) == 100

    # Verify that k_proj and v_proj only appear on layers 0..14
    all_attn_targets = local_targets + global_targets
    kv_targets = [t for t in all_attn_targets if "k_proj" in t or "v_proj" in t]
    
    # 15 layers * 2 (k, v) = 30 kv targets
    assert len(kv_targets) == 30
    for target in kv_targets:
        layer_idx = int(target.split(".layers.")[1].split(".")[0])
        assert 0 <= layer_idx <= 14, f"Target {target} on layer {layer_idx} should not have k/v projection"


def test_gemma_local_global_ratio_differentiation(grammars):
    """Assert (e): E2B and E4B produce different local:global ratios (4:1 vs 5:1)."""
    e2b_groups = {g.component_type: g for g in grammars["e2b"].groups}
    e4b_groups = {g.component_type: g for g in grammars["e4b"].groups}

    e2b_local_count = len(e2b_groups[ComponentType.LOCAL_ATTENTION].targets)
    e2b_global_count = len(e2b_groups[ComponentType.GLOBAL_ATTENTION].targets)
    assert e2b_local_count / e2b_global_count == 4.0  # 80 / 20 = 4:1

    e4b_local_count = len(e4b_groups[ComponentType.LOCAL_ATTENTION].targets)
    e4b_global_count = len(e4b_groups[ComponentType.GLOBAL_ATTENTION].targets)
    assert e4b_local_count / e4b_global_count == 5.0  # 110 / 22 = 5:1
