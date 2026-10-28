"""Tests for Stage 4 / R4: Adapter Attachment & Empirical Gradient Verification.

Tests unit behavior of PeftMPSBackend, verifies schema envelope and Gate 4 invariants
on generated gradient reports, and tests empirical detection of inert targets.
"""
from __future__ import annotations

import json
from pathlib import Path

import pytest
import torch
from torch import nn

from aloe.architecture.schema import ComponentType
from aloe.execution.backends.peft_mps import PeftMPSBackend
from aloe.search_space.schema import GroupDimensions, SearchGrammar


@pytest.fixture
def mps_device() -> torch.device:
    if not torch.backends.mps.is_available():
        pytest.skip("Apple Silicon MPS backend is not available.")
    return torch.device("mps")


class ToyLinearModel(nn.Module):
    """Simple 2-layer model for unit testing PeftMPSBackend on MPS."""

    def __init__(self, in_features: int = 16, hidden: int = 32, out_features: int = 8) -> None:
        super().__init__()
        self.in_features = in_features
        self.fc1 = nn.Linear(in_features, hidden, bias=False)
        self.fc2 = nn.Linear(hidden, out_features, bias=False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.fc2(torch.relu(self.fc1(x)))


class ToyInertModuleModel(nn.Module):
    """Model with an active linear layer and an inert linear layer bypassed in forward()."""

    def __init__(self, in_features: int = 16, out_features: int = 8) -> None:
        super().__init__()
        self.in_features = in_features
        self.active_layer = nn.Linear(in_features, out_features, bias=False)
        self.inert_layer = nn.Linear(in_features, out_features, bias=False)  # Bypassed!

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Bypasses self.inert_layer completely
        return self.active_layer(x)


def test_peft_mps_backend_protocol_conformance() -> None:
    """Verify that PeftMPSBackend conforms to the ExecutionBackend Protocol."""
    backend = PeftMPSBackend()
    assert backend.name == "peft_mps"
    assert callable(backend.available)
    assert callable(backend.supports)
    assert callable(backend.run_trial)
    assert callable(backend.verify_gradients)


def test_peft_mps_backend_availability() -> None:
    """Verify backend.available() returns boolean without raising."""
    backend = PeftMPSBackend()
    avail = backend.available()
    assert isinstance(avail, bool)
    assert avail is bool(hasattr(torch.backends, "mps") and torch.backends.mps.is_available())


def test_peft_mps_backend_supports_validation() -> None:
    """Verify grammar validation logic in PeftMPSBackend.supports."""
    backend = PeftMPSBackend()

    # Valid grammar
    valid_grammar = SearchGrammar(
        model_id="test/model",
        family="test",
        topology="standard",
        backend="peft_mps",
        groups=[
            GroupDimensions(
                component_type=ComponentType.GLOBAL_ATTENTION,
                targets=["model.layers.0.q_proj"],
                rank=[8, 16],
                alpha=[16, 32],
            )
        ],
    )
    ok, reason = backend.supports(valid_grammar)
    assert ok is True
    assert "supported" in reason.lower()

    # Empty groups
    empty_grammar = SearchGrammar(
        model_id="test/model",
        family="test",
        topology="standard",
        backend="peft_mps",
        groups=[],
    )
    ok, reason = backend.supports(empty_grammar)
    assert ok is False
    assert "no groups" in reason.lower()

    # Rank <= 0
    invalid_rank_grammar = SearchGrammar(
        model_id="test/model",
        family="test",
        topology="standard",
        backend="peft_mps",
        groups=[
            GroupDimensions(
                component_type=ComponentType.GLOBAL_ATTENTION,
                targets=["model.layers.0.q_proj"],
                rank=[0, 8],
                alpha=[16],
            )
        ],
    )
    ok, reason = backend.supports(invalid_rank_grammar)
    assert ok is False
    assert "rank <= 0" in reason.lower()

    # Incompatible conv1d target
    conv1d_grammar = SearchGrammar(
        model_id="test/model",
        family="test",
        topology="standard",
        backend="peft_mps",
        groups=[
            GroupDimensions(
                component_type=ComponentType.SSM,
                targets=["model.layers.0.mamba.conv1d"],
                rank=[4],
                alpha=[8],
            )
        ],
    )
    ok, reason = backend.supports(conv1d_grammar)
    assert ok is False
    assert "conv1d" in reason.lower()


def test_peft_mps_backend_run_trial_rejects_none_config() -> None:
    """run_trial() is implemented (Gate C) but must reject a None config cleanly."""
    backend = PeftMPSBackend()
    with pytest.raises((AttributeError, ValueError)):
        backend.run_trial(config=None)


def test_peft_mps_verify_gradients_toy_active(mps_device: torch.device) -> None:
    """Verify verify_gradients on active target module achieves GRADIENT_OK."""
    backend = PeftMPSBackend()
    model = ToyLinearModel().to(mps_device)

    res = backend.verify_gradients(
        model=model,
        target_modules=["fc1"],
        r=4,
        lora_alpha=8,
        init_lora_weights=False,
    )

    assert res["success"] is True
    assert res["verdict"] == "PASS"
    assert res["target_module_count"] == 1
    assert res["adapter_param_count"] == 2  # lora_A and lora_B
    assert res["gradient_ok_count"] == 2
    assert res["gradient_zero_count"] == 0
    assert res["gradient_missing_count"] == 0
    assert res["min_gradient_norm"] is not None
    assert res["min_gradient_norm"] > 1e-8
    assert res["target_status"]["fc1"] is True

    # Check that returned unloaded model is clean
    unloaded = res["unloaded_model"]
    assert all("lora_" not in name for name, _ in unloaded.named_parameters())


def test_peft_mps_verify_gradients_toy_inert_detection(mps_device: torch.device) -> None:
    """Verify verify_gradients catches bypassed/inert layer as GRADIENT_MISSING."""
    backend = PeftMPSBackend()
    model = ToyInertModuleModel().to(mps_device)

    res = backend.verify_gradients(
        model=model,
        target_modules=["inert_layer"],
        r=4,
        lora_alpha=8,
        init_lora_weights=False,
    )

    # Inert module is bypassed in forward(), so no gradients should reach its adapter
    assert res["success"] is False
    assert res["verdict"] == "FAIL"
    assert res["adapter_param_count"] == 2
    assert res["gradient_ok_count"] == 0
    assert res["gradient_missing_count"] == 2
    assert res["target_status"]["inert_layer"] is False


def test_gradient_report_json_schema_envelope() -> None:
    """Verify that both generated gradient reports adhere to Schema Version 1.0."""
    runs_dir = Path("runs/gradients")
    slugs = ["google--gemma-4-E2B-it", "tiiuae--Falcon-H1-1.5B-Instruct"]

    for slug in slugs:
        json_path = runs_dir / slug / "gradient_report.json"
        assert json_path.exists(), f"Missing gradient report at {json_path}"

        with open(json_path, encoding="utf-8") as f:
            data = json.load(f)

        # Top-level required keys
        required_keys = [
            "schema_version",
            "run_id",
            "timestamp_utc",
            "script",
            "model_id",
            "model_slug",
            "backend",
            "device",
            "dtype",
            "gate_4",
            "components",
            "status",
        ]
        for key in required_keys:
            assert key in data, f"Missing required key '{key}' in {json_path}"

        assert data["schema_version"] == "1.0"
        assert data["script"] == "verify_gradients.py"
        assert data["backend"] == "peft_mps"
        assert data["device"] == "mps"
        assert data["dtype"] == "torch.bfloat16"
        assert data["status"] == "PASS"

        # Gate 4 structure
        gate_4 = data["gate_4"]
        assert gate_4["gate_id"] == "Gate 4"
        assert gate_4["status"] == "PASS"
        assert gate_4["negative_control_passed"] is True
        assert len(gate_4["summary_sentence"]) > 20

        # Components validation
        components = data["components"]
        for comp_data in components.values():
            assert comp_data["verdict"] == "PASS"
            assert comp_data["gradient_ok_count"] == comp_data["adapter_param_count"]
            assert comp_data["gradient_zero_count"] == 0
            assert comp_data["gradient_missing_count"] == 0
            assert comp_data["min_gradient_norm"] > 1e-8


def test_model_a_exact_module_counts() -> None:
    """Verify Model A exact counts: 80 local_attn, 20 global_attn, 105 ffn = 205 modules (410 params)."""
    json_path = Path("runs/gradients/google--gemma-4-E2B-it/gradient_report.json")
    with open(json_path, encoding="utf-8") as f:
        data = json.load(f)

    comps = data["components"]
    assert comps["local_attention"]["target_module_count"] == 80
    assert comps["local_attention"]["adapter_param_count"] == 160
    assert comps["local_attention"]["gradient_ok_count"] == 160

    assert comps["global_attention"]["target_module_count"] == 20
    assert comps["global_attention"]["adapter_param_count"] == 40
    assert comps["global_attention"]["gradient_ok_count"] == 40

    assert comps["ffn"]["target_module_count"] == 105
    assert comps["ffn"]["adapter_param_count"] == 210
    assert comps["ffn"]["gradient_ok_count"] == 210

    total_modules = sum(c["target_module_count"] for c in comps.values())
    total_params = sum(c["adapter_param_count"] for c in comps.values())
    assert total_modules == 205
    assert total_params == 410


def test_model_b_exact_module_counts_and_negative_control() -> None:
    """Verify Model B exact counts: 96 global_attn, 72 ffn, 24 ssm = 192 modules, plus inert conv1d."""
    json_path = Path("runs/gradients/tiiuae--Falcon-H1-1.5B-Instruct/gradient_report.json")
    with open(json_path, encoding="utf-8") as f:
        data = json.load(f)

    comps = data["components"]
    assert comps["global_attention"]["target_module_count"] == 96
    assert comps["global_attention"]["adapter_param_count"] == 192
    assert comps["global_attention"]["gradient_ok_count"] == 192

    assert comps["ffn"]["target_module_count"] == 72
    assert comps["ffn"]["adapter_param_count"] == 144
    assert comps["ffn"]["gradient_ok_count"] == 144

    assert comps["ssm"]["target_module_count"] == 24
    assert comps["ssm"]["adapter_param_count"] == 48
    assert comps["ssm"]["gradient_ok_count"] == 48

    total_modules = sum(c["target_module_count"] for c in comps.values())
    total_params = sum(c["adapter_param_count"] for c in comps.values())
    assert total_modules == 192
    assert total_params == 384

    # Mandatory Negative Control assertions
    nc = data["negative_control"]
    assert nc is not None
    assert nc["target"] == "conv1d"
    assert nc["module_count"] == 24
    assert nc["adapter_param_count"] == 48
    assert nc["gradient_ok_count"] == 0
    assert nc["gradient_zero_count"] == 0
    assert nc["gradient_missing_count"] == 48
    assert nc["verdict"] == "CONFIRMED_INERT"
    assert nc["headline_sentence"] == "The negative control DID correctly detect an inert adapter target."


def test_markdown_report_headline_and_structure() -> None:
    """Verify GRADIENT_REPORT.md contains the mandatory headline and key sections."""
    md_path = Path("runs/gradients/GRADIENT_REPORT.md")
    assert md_path.exists(), "GRADIENT_REPORT.md missing!"

    content = md_path.read_text(encoding="utf-8")
    mandatory_headline = "The negative control DID correctly detect an inert adapter target."
    assert mandatory_headline in content, f"Headline missing from {md_path}"

    assert "Gate 4 Evaluation: PASS" in content
    assert "Comprehensive Verification Summary" in content
    assert "Model A Breakdown: `google/gemma-4-E2B-it`" in content
    assert "Model B Breakdown: `tiiuae/Falcon-H1-1.5B-Instruct`" in content
    assert "Deep-Dive: Mandatory Negative Control (`conv1d`) & Autograd Bypass" in content
    assert "Execution & Memory Telemetry" in content
