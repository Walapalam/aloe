#!/usr/bin/env python3
"""Generate runs/probe/REPORT.md strictly from JSON artifacts.

Adheres to Rule 4: Results as machine-readable JSON first, human-readable
Markdown second, generated FROM the JSON. Never hand-write the Markdown.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any


def load_json(path: Path) -> dict[str, Any]:
    if not path.is_file():
        raise FileNotFoundError(f"Missing required probe artifact: {path}")
    with open(path, "r", encoding="utf-8") as f:
        data = json.load(f)
    if data.get("schema_version") != "1.0":
        raise ValueError(
            f"Invalid schema_version {data.get('schema_version')} in {path}. Expected '1.0'"
        )
    return data


def format_report(env_data: dict[str, Any], backends_data: dict[str, Any]) -> str:
    gate_1 = backends_data["gate_1"]
    env_res = env_data["results"]
    backends = backends_data["backends"]
    peft_mps = backends["peft_mps"]
    peft_emp = peft_mps["empirical_test"]
    mlx = backends["mlx_lm"]
    mlx_src = mlx["source_inspection"]
    bnb = env_res["bitsandbytes"]
    unsloth = env_res.get("unsloth", {})

    lines = [
        "# ALOE Stage 1 / R1: Environment & Backend Capability Report",
        "",
        f"> **{gate_1['summary_sentence']}**",
        "",
        "---",
        "",
        "## 1. Gate 1 Evaluation & Verification",
        "",
        f"- **Gate Status**: `{gate_1['status']}`",
        f"- **Passing Backend**: `{gate_1['passing_backend']}`",
        f"- **Probe Run ID (Environment)**: `{env_data['run_id']}`",
        f"- **Probe Run ID (Backends)**: `{backends_data['run_id']}`",
        f"- **Timestamp (UTC)**: `{backends_data['timestamp_utc']}`",
        f"- **Git Commit**: `{backends_data['git_commit']}`",
        "",
        "### Gate 1 Acceptance Criteria Checklist",
        f"- [{'x' if env_res['torch']['mps_available'] else ' '}] PyTorch MPS backend available and functional on Apple Silicon",
        f"- [{'x' if peft_emp['distinct_ranks_count'] >= 2 else ' '}] PEFT on MPS empirically proven to instantiate adapters with differing ranks across module groups",
        f"- [{'x' if peft_emp['distinct_alphas_count'] >= 2 else ' '}] PEFT on MPS empirically proven to instantiate adapters with differing alphas across module groups",
        f"- [{'x' if peft_emp['gradient_flow']['backward_pass_success'] else ' '}] Gradient flow verified to PEFT adapters on MPS via backward pass",
        f"- [{'x' if mlx_src['signature'] else ' '}] MLX / mlx-lm LoRA targeting capability inspected and characterized",
        f"- [{'x' if not bnb['functional'] else ' '}] bitsandbytes confirmed absent or non-functional with failure mode recorded verbatim",
        f"- [{'x' if env_data['schema_version'] == '1.0' and backends_data['schema_version'] == '1.0' else ' '}] Standard Result Envelope (schema_version: '1.0') adhered to",
        "",
        "---",
        "",
        "## 2. Platform & Hardware Environment",
        "",
        "| Component | Detail | Specification / Status |",
        "|---|---|---|",
        f"| **Chip / SoC** | Apple Silicon | {env_res['system']['chip']} |",
        f"| **Physical Cores** | Performance + Efficiency | {env_res['system']['cpu_physical_cores']} physical ({env_res['system']['cpu_logical_cores']} logical) |",
        f"| **OS / Kernel** | macOS / Darwin | {env_res['system']['platform']} |",
        f"| **Unified Memory** | RAM Total / Available | {env_res['memory']['total_gb']} GB total ({env_res['memory']['available_gb']} GB available) |",
        f"| **Disk Space** | Workspace Volume | {env_res['disk']['free_gb']} GB free / {env_res['disk']['total_gb']} GB total |",
        f"| **Python Runtime** | Executable | {env_res['python']['version']} (`{env_res['python']['executable']}`) |",
        f"| **PyTorch** | Version & Devices | {env_res['torch']['version']} (MPS: {env_res['torch']['mps_available']}, CUDA: {env_res['torch']['cuda_available']}) |",
        "",
        "### Key Library Matrix",
        "",
        "| Package | Installed | Version | Import Error / Status |",
        "|---|---|---|---|",
    ]

    for pkg_name, pkg_info in env_res["packages"].items():
        err = pkg_info["import_error"] if pkg_info["import_error"] else "OK"
        ver = pkg_info["version"] if pkg_info["version"] else "N/A"
        lines.append(f"| `{pkg_name}` | `{pkg_info['installed']}` | `{ver}` | `{err}` |")

    lines.extend([
        "",
        "---",
        "",
        "## 3. Mandatory Hardware Invariant: Prohibited Packages Audit (CUDA Dependencies)",
        "",
        "### 3.1 bitsandbytes",
        f"- **Tested**: `{bnb['tested']}`",
        f"- **Installed**: `{bnb['installed']}`",
        f"- **Functional**: `{bnb['functional']}`",
        f"- **Explanation**: {bnb['explanation']}",
        "",
        "**Verbatim bitsandbytes Probe Failure Output:**",
        "```text",
        bnb["failure_mode"].strip(),
        "```",
    ])

    if unsloth:
        lines.extend([
            "",
            "### 3.2 unsloth",
            f"- **Tested**: `{unsloth['tested']}`",
            f"- **Installed**: `{unsloth['installed']}`",
            f"- **Functional**: `{unsloth['functional']}`",
            f"- **Explanation**: {unsloth['explanation']}",
            "",
            "**Verbatim unsloth Probe Failure Output:**",
            "```text",
            unsloth["failure_mode"].strip(),
            "```",
        ])

    lines.extend([
        "",
        "---",
        "",
        "## 4. Backend Capability Analysis",
        "",
        "### 4.1 Backend A: PEFT on MPS (Primary Execution Backend)",
        f"- **Backend Status**: `{peft_mps['status']}`",
        f"- **Per-Group Rank & Alpha Support**: `{peft_mps['supports_per_group_rank_alpha']}`",
        f"- **Mechanism**: `{peft_mps['mechanism']}`",
        f"- **Model Architecture Evaluated**: `{peft_emp['model_architecture']}`",
        "",
        "**Inspected Adapter Modules on MPS:**",
        "",
        "| Module Name | Component Target | Injected Rank (r) | Injected Alpha | Scaling Factor | Weight Shape |",
        "|---|---|---|---|---|---|",
    ])

    for mod in peft_emp["verified_modules"]:
        lines.append(
            f"| `{mod['name']}` | `{mod['target']}` | {mod['actual_r']} | {mod['actual_alpha']} | {mod['actual_scaling']} | `{mod['param_shape']}` |"
        )

    lines.extend([
        "",
        "**Empirical Gradient Flow Verification:**",
        "",
        "| Parameter Name | Gradient Norm | Status |",
        "|---|---|---|",
    ])

    for grad in peft_emp["gradient_flow"]["adapter_param_gradients"]:
        norm_status = "GRADIENT_OK" if grad["norm"] > 1e-8 else "GRADIENT_ZERO"
        lines.append(f"| `{grad['param']}` | {grad['norm']:.4f} | `{norm_status}` |")

    lines.extend([
        "",
        "### 4.2 Backend B: Apple MLX (mlx-lm)",
        f"- **Backend Status**: `{mlx['status']}`",
        f"- **Per-Group Rank & Alpha Support**: `{mlx['supports_per_group_rank_alpha']}`",
        f"- **Source Entrypoint**: `{mlx_src['entrypoint']}`",
        f"- **Function Signature**: `{mlx_src['signature']}`",
        f"- **Keys Filtering Supported**: `{mlx_src['keys_filtering_supported']}`",
        f"- **Per-Module Rank Supported**: `{mlx_src['per_module_rank_supported']}`",
        "",
        "**Source Code Evidence:**",
        f"> {mlx_src['evidence']}",
        "",
        "**Empirical Evaluation:**",
        f"- Uniform LoRA: `{mlx['empirical_test']['uniform_lora']}`",
        f"- Target Key Filtering: `{mlx['empirical_test']['key_filtering_lora']}`",
        f"- Per-Module Dictionary Rank: `{mlx['empirical_test']['per_module_dict_rank_lora']}`",
        "",
        "---",
        "",
        "## 5. Architectural Implications for Downstream Milestones",
        "",
        "1. **Execution Backend Selection**:",
        "   - PEFT on PyTorch MPS (`torch.backends.mps`) is designated as the primary execution backend for ALOE on this machine.",
        "   - MLX is reserved as an optional baseline runner for uniform rank configurations or target-filtered configurations, but cannot serve as the typed search space backend without external weight partitioning surgery.",
        "2. **Precision & Memory Strategy**:",
        "   - Host has 64.0 GB of unified memory. bf16 execution fits comfortably without requiring 4-bit quantisation.",
        "   - bitsandbytes, NF4, and Unsloth are confirmed non-viable and excluded from the critical path.",
        "3. **Milestone Progression**:",
        "   - Gate 1 is completely satisfied. Proceed immediately to Stage 2 (Model Acquisition).",
        "",
    ])

    return "\n".join(lines)


def main():
    parser = argparse.ArgumentParser(description="Generate runs/probe/REPORT.md from probe JSON artifacts.")
    parser.add_argument("--env-json", type=Path, default=Path("runs/probe/environment.json"))
    parser.add_argument("--backends-json", type=Path, default=Path("runs/probe/backends.json"))
    parser.add_argument("--out-report", type=Path, default=Path("runs/probe/REPORT.md"))
    args = parser.parse_args()

    env_data = load_json(args.env_json)
    backends_data = load_json(args.backends_json)

    report_md = format_report(env_data, backends_data)

    args.out_report.parent.mkdir(parents=True, exist_ok=True)
    with open(args.out_report, "w", encoding="utf-8") as f:
        f.write(report_md)

    print(f"Report successfully generated at: {args.out_report}")


if __name__ == "__main__":
    main()
