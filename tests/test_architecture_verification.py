"""Tests for Stage 3 / R3: Architecture Verification & Claims Testing."""
from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import pytest

from aloe.architecture.adapters.falcon_h1 import FalconH1Adapter
from aloe.architecture.adapters.gemma4 import Gemma4Adapter
from aloe.architecture.constraints import apply_validity
from aloe.architecture.inspector import inspect_model
from aloe.architecture.schema import ArchitectureDescription, ComponentType, ModuleRecord, Validity
from aloe.architecture.typing import classify, resolve
from aloe.datasets.model_registry import ModelRegistry
from aloe.search_space.grammar import generate


class DummyConfig:
    """Mock configuration object for unit testing adapter logic without transformers."""

    def __init__(self, text_config: Any = None, **kwargs: Any) -> None:
        self.text_config = text_config
        for k, v in kwargs.items():
            setattr(self, k, v)


class DummyTextConfig:
    """Mock text configuration for Gemma 4."""

    def __init__(
        self,
        num_hidden_layers: int = 35,
        layer_types: list[str] | None = None,
        num_kv_shared_layers: int = 20,
        sliding_window: int = 512,
    ) -> None:
        self.num_hidden_layers = num_hidden_layers
        self.layer_types = (
            layer_types
            if layer_types is not None
            else ["sliding_attention", "sliding_attention", "sliding_attention", "sliding_attention", "full_attention"] * 7
        )
        self.num_kv_shared_layers = num_kv_shared_layers
        self.sliding_window = sliding_window


def test_gemma4_adapter_pattern_extraction_success() -> None:
    adapter = Gemma4Adapter()
    config = DummyConfig(text_config=DummyTextConfig())

    assert adapter.matches("google/gemma-4-E2B-it", config)
    # Check 4 local + 1 global pattern for 7 cycles
    for i in range(35):
        is_global = adapter.is_global_layer(i, config)
        if i % 5 == 4:
            assert is_global is True, f"Layer {i} should be global"
        else:
            assert is_global is False, f"Layer {i} should be local"


def test_gemma4_adapter_missing_config_fails_loud() -> None:
    adapter = Gemma4Adapter()

    # Missing config entirely
    with pytest.raises(KeyError, match="Gemma 4 config required"):
        adapter.is_global_layer(0, config=None)

    # Config missing layer_types attribute
    empty_tc_config = DummyConfig(text_config=object())
    with pytest.raises(KeyError, match="missing 'layer_types' field"):
        adapter.is_global_layer(0, config=empty_tc_config)

    # Config as dict missing layer_types
    dict_config = {"text_config": {"num_hidden_layers": 35}}
    with pytest.raises(KeyError, match="missing 'layer_types' field"):
        adapter.is_global_layer(0, config=dict_config)

    # Out of bounds index
    valid_config = DummyConfig(text_config=DummyTextConfig())
    with pytest.raises(IndexError, match="out of bounds"):
        adapter.is_global_layer(35, config=valid_config)

    with pytest.raises(IndexError, match="out of bounds"):
        adapter.is_global_layer(-1, config=valid_config)


def test_gemma4_adapter_classification_synthetic() -> None:
    adapter = Gemma4Adapter()
    config = DummyConfig(text_config=DummyTextConfig())

    desc = ArchitectureDescription(
        model_id="test-gemma",
        family="gemma4",
        num_layers=5,
        total_params=1000,
        topology="interleaved",
        modules=[
            ModuleRecord(name="model.language_model.embed_tokens", module_class="Embedding", param_count=100),
            ModuleRecord(name="model.language_model.layers.0.self_attn.q_proj", module_class="Linear", layer_index=0, is_linear=True, param_count=50),
            ModuleRecord(name="model.language_model.layers.4.self_attn.q_proj", module_class="Linear", layer_index=4, is_linear=True, param_count=50),
            ModuleRecord(name="model.language_model.layers.0.mlp.gate_proj", module_class="Linear", layer_index=0, is_linear=True, param_count=50),
            ModuleRecord(name="model.language_model.layers.0.input_layernorm", module_class="RMSNorm", layer_index=0, param_count=10),
            ModuleRecord(name="model.vision_tower.encoder.layers.0.self_attn.q_proj", module_class="Linear", layer_index=0, is_linear=True, param_count=50),
            ModuleRecord(name="lm_head", module_class="Linear", is_linear=True, param_count=100),
        ],
    )

    classified = adapter.classify(desc, config=config)

    m_by_name = {m.name: m for m in classified.modules}
    assert m_by_name["model.language_model.embed_tokens"].component_type == ComponentType.EMBEDDING
    assert m_by_name["model.language_model.layers.0.self_attn.q_proj"].component_type == ComponentType.LOCAL_ATTENTION
    assert m_by_name["model.language_model.layers.4.self_attn.q_proj"].component_type == ComponentType.GLOBAL_ATTENTION
    assert m_by_name["model.language_model.layers.0.mlp.gate_proj"].component_type == ComponentType.FFN
    assert m_by_name["model.language_model.layers.0.input_layernorm"].component_type == ComponentType.NORM

    # Multimodal auxiliary module excluded
    assert m_by_name["model.vision_tower.encoder.layers.0.self_attn.q_proj"].component_type == ComponentType.OTHER
    assert m_by_name["model.vision_tower.encoder.layers.0.self_attn.q_proj"].validity == Validity.EXCLUDED

    # lm_head excluded
    assert m_by_name["lm_head"].component_type == ComponentType.OTHER
    assert m_by_name["lm_head"].validity == Validity.EXCLUDED


def test_falcon_h1_adapter_classification_synthetic() -> None:
    adapter = FalconH1Adapter()
    config = DummyConfig(num_hidden_layers=24)

    desc = ArchitectureDescription(
        model_id="test-falcon",
        family="falcon_h1",
        num_layers=1,
        total_params=500,
        topology="parallel_hybrid",
        modules=[
            ModuleRecord(name="model.embed_tokens", module_class="Embedding", param_count=100),
            ModuleRecord(name="model.layers.0.self_attn.q_proj", module_class="Linear", layer_index=0, is_linear=True, param_count=50),
            ModuleRecord(name="model.layers.0.mamba.in_proj", module_class="Linear", layer_index=0, is_linear=True, param_count=50),
            ModuleRecord(name="model.layers.0.mamba.conv1d", module_class="Conv1d", layer_index=0, is_linear=False, param_count=10),
            ModuleRecord(name="model.layers.0.mamba.out_proj", module_class="Linear", layer_index=0, is_linear=True, param_count=50),
            ModuleRecord(name="model.layers.0.feed_forward.gate_proj", module_class="Linear", layer_index=0, is_linear=True, param_count=50),
            ModuleRecord(name="model.layers.0.input_layernorm", module_class="RMSNorm", layer_index=0, param_count=10),
            ModuleRecord(name="lm_head", module_class="Linear", is_linear=True, param_count=100),
        ],
    )

    classified = adapter.classify(desc, config=config)
    m_by_name = {m.name: m for m in classified.modules}

    assert m_by_name["model.embed_tokens"].component_type == ComponentType.EMBEDDING
    assert m_by_name["model.layers.0.self_attn.q_proj"].component_type == ComponentType.GLOBAL_ATTENTION
    assert m_by_name["model.layers.0.mamba.in_proj"].component_type == ComponentType.SSM
    assert m_by_name["model.layers.0.mamba.conv1d"].component_type == ComponentType.SSM
    assert m_by_name["model.layers.0.mamba.out_proj"].component_type == ComponentType.SSM
    assert m_by_name["model.layers.0.feed_forward.gate_proj"].component_type == ComponentType.FFN
    assert m_by_name["model.layers.0.input_layernorm"].component_type == ComponentType.NORM
    assert m_by_name["lm_head"].validity == Validity.EXCLUDED


def test_falcon_h1_adr005_exclusions_under_validity() -> None:
    adapter = resolve("tiiuae/Falcon-H1-1.5B-Instruct", None)
    desc = ArchitectureDescription(
        model_id="tiiuae/Falcon-H1-1.5B-Instruct",
        family=adapter.family,
        num_layers=1,
        total_params=500,
        topology=adapter.topology,
        modules=[
            ModuleRecord(name="model.layers.0.mamba.in_proj", module_class="Linear", layer_index=0, is_linear=True, param_count=50),
            ModuleRecord(name="model.layers.0.mamba.conv1d", module_class="Conv1d", layer_index=0, is_linear=False, param_count=10),
            ModuleRecord(name="model.layers.0.mamba.out_proj", module_class="Linear", layer_index=0, is_linear=True, param_count=50),
        ],
    )
    desc = adapter.classify(desc)
    desc = apply_validity(desc, backend="peft_mps")

    m_by_name = {m.name: m for m in desc.modules}
    assert m_by_name["model.layers.0.mamba.in_proj"].validity == Validity.VALID
    assert m_by_name["model.layers.0.mamba.conv1d"].validity == Validity.EXCLUDED
    assert "Borobia et al." in m_by_name["model.layers.0.mamba.conv1d"].validity_reason
    assert m_by_name["model.layers.0.mamba.out_proj"].validity == Validity.EXCLUDED
    assert "Borobia et al." in m_by_name["model.layers.0.mamba.out_proj"].validity_reason


def test_meta_device_inspection_full_models() -> None:
    reg = ModelRegistry("runs/models")

    # Verify Model A meta loading & inspection
    model_a = reg.load_model_meta("google/gemma-4-E2B-it")
    param_a = next(model_a.parameters())
    assert param_a.is_meta or str(param_a.device) == "meta"

    cfg_a = reg.load_config("google/gemma-4-E2B-it")
    desc_a = inspect_model(model_a, "google/gemma-4-E2B-it")
    desc_a = classify(desc_a, "google/gemma-4-E2B-it", cfg_a)
    desc_a = apply_validity(desc_a, backend="peft_mps")

    assert desc_a.family == "gemma4"
    assert desc_a.topology == "interleaved"
    assert desc_a.num_layers == 35

    # Check search grammar groups
    grammar_a = generate(desc_a, backend="peft_mps")
    group_types_a = {g.component_type for g in grammar_a.groups}
    assert group_types_a == {ComponentType.LOCAL_ATTENTION, ComponentType.FFN, ComponentType.GLOBAL_ATTENTION}

    # Verify Model B meta loading & inspection
    model_b = reg.load_model_meta("tiiuae/Falcon-H1-1.5B-Instruct")
    param_b = next(model_b.parameters())
    assert param_b.is_meta or str(param_b.device) == "meta"

    cfg_b = reg.load_config("tiiuae/Falcon-H1-1.5B-Instruct")
    desc_b = inspect_model(model_b, "tiiuae/Falcon-H1-1.5B-Instruct")
    desc_b = classify(desc_b, "tiiuae/Falcon-H1-1.5B-Instruct", cfg_b)
    desc_b = apply_validity(desc_b, backend="peft_mps")

    assert desc_b.family == "falcon_h1"
    assert desc_b.topology == "parallel_hybrid"
    assert desc_b.num_layers == 24

    grammar_b = generate(desc_b, backend="peft_mps")
    group_types_b = {g.component_type for g in grammar_b.groups}
    assert group_types_b == {ComponentType.FFN, ComponentType.SSM, ComponentType.GLOBAL_ATTENTION}


def test_gate_3_claims_artifacts_on_disk() -> None:
    runs_dir = Path("runs/architecture")
    assert runs_dir.is_dir(), "runs/architecture must exist"

    report_path = runs_dir / "ARCHITECTURE_REPORT.md"
    assert report_path.is_file(), "ARCHITECTURE_REPORT.md must exist"
    report_text = report_path.read_text(encoding="utf-8")
    assert "Gate 3 Evaluation: PASS" in report_text
    assert "google/gemma-4-E2B-it" in report_text
    assert "tiiuae/Falcon-H1-1.5B-Instruct" in report_text

    # Model A artifacts
    dir_a = runs_dir / "google--gemma-4-E2B-it"
    insp_a_path = dir_a / "inspection.json"
    claims_a_path = dir_a / "claims.json"
    assert insp_a_path.is_file()
    assert claims_a_path.is_file()

    with open(claims_a_path, encoding="utf-8") as f:
        data_a = json.load(f)
    assert data_a["schema_version"] == "1.0"
    assert data_a["gate_3"]["status"] == "PASS"
    assert all(c["status"] == "PASS" for c in data_a["claims"])

    # Model B artifacts
    dir_b = runs_dir / "tiiuae--Falcon-H1-1.5B-Instruct"
    insp_b_path = dir_b / "inspection.json"
    claims_b_path = dir_b / "claims.json"
    assert insp_b_path.is_file()
    assert claims_b_path.is_file()

    with open(claims_b_path, encoding="utf-8") as f:
        data_b = json.load(f)
    assert data_b["schema_version"] == "1.0"
    assert data_b["gate_3"]["status"] == "PASS"
    assert all(c["status"] == "PASS" for c in data_b["claims"])
