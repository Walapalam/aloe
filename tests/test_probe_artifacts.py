"""Tests for Stage 1 / R1 probe scripts and artifacts.

Validates that runs/probe/environment.json, runs/probe/backends.json, and runs/probe/REPORT.md
adhere strictly to Schema Envelope 1.0, Gate 1 criteria, and programmatic generation rules.
"""
from __future__ import annotations

import sys
from pathlib import Path

# Ensure repository root is on sys.path for scripts import
REPO_ROOT = Path(__file__).resolve().parent.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from scripts.generate_probe_report import format_report, load_json


def test_environment_json_schema_and_telemetry():
    env_path = Path("runs/probe/environment.json")
    assert env_path.is_file(), "runs/probe/environment.json must exist"

    data = load_json(env_path)
    assert data["schema_version"] == "1.0"
    assert data["script"] == "probe_environment.py"
    assert data["status"] == "PASS"

    env = data["environment"]
    assert "platform" in env
    assert env["chip"] == "Apple M1 Max"
    assert env["unified_memory_gb"] == 64.0
    assert env["backend"] == "torch_mps"

    results = data["results"]
    assert results["system"]["chip"] == "Apple M1 Max"
    assert results["memory"]["memory_type"] == "unified"
    assert results["memory"]["total_gb"] == 64.0
    assert results["torch"]["mps_available"] is True
    assert results["torch"]["mps_built"] is True
    assert results["torch"]["cuda_available"] is False
    assert results["disk"]["sufficient_for_pilot_models"] is True

    bnb = results["bitsandbytes"]
    assert bnb["tested"] is True
    assert bnb["installed"] is False
    assert bnb["functional"] is False
    assert "ModuleNotFoundError" in bnb["failure_mode"]


def test_backends_json_schema_and_gate_1():
    backends_path = Path("runs/probe/backends.json")
    assert backends_path.is_file(), "runs/probe/backends.json must exist"

    data = load_json(backends_path)
    assert data["schema_version"] == "1.0"
    assert data["script"] == "probe_backends.py"
    assert data["status"] == "PASS"

    gate_1 = data["gate_1"]
    assert gate_1["gate_id"] == "Gate 1"
    assert gate_1["status"] == "PASS"
    assert gate_1["passing_backend"] == "peft_mps"

    expected_summary = (
        "Per-group rank and alpha assignment IS possible on this machine, via PEFT on MPS, "
        "evidenced by runs/probe/backends.json:backends.peft_mps.empirical_test.status PASS."
    )
    assert gate_1["summary_sentence"] == expected_summary

    peft = data["backends"]["peft_mps"]
    assert peft["supports_per_group_rank_alpha"] is True
    assert peft["status"] == "PASS"
    emp = peft["empirical_test"]
    assert emp["test_passed"] is True
    assert emp["distinct_ranks_count"] >= 2
    assert emp["distinct_alphas_count"] >= 2
    assert emp["gradient_flow"]["backward_pass_success"] is True
    for grad in emp["gradient_flow"]["adapter_param_gradients"]:
        assert grad["norm"] > 1e-8

    mlx = data["backends"]["mlx_lm"]
    assert mlx["supports_per_group_rank_alpha"] is False
    assert mlx["status"] == "UNIFORM_ONLY"
    assert mlx["source_inspection"]["keys_filtering_supported"] is True
    assert mlx["source_inspection"]["per_module_rank_supported"] is False
    assert "TypeError" in mlx["empirical_test"]["per_module_dict_rank_lora"]


def test_probe_report_markdown():
    report_path = Path("runs/probe/REPORT.md")
    assert report_path.is_file(), "runs/probe/REPORT.md must exist"

    content = report_path.read_text(encoding="utf-8")
    expected_header_sentence = (
        "> **Per-group rank and alpha assignment IS possible on this machine, via PEFT on MPS, "
        "evidenced by runs/probe/backends.json:backends.peft_mps.empirical_test.status PASS.**"
    )
    assert expected_header_sentence in content

    # Test programmatic generator output matches existing file
    env_data = load_json(Path("runs/probe/environment.json"))
    backends_data = load_json(Path("runs/probe/backends.json"))
    generated = format_report(env_data, backends_data)
    assert content.strip() == generated.strip()
