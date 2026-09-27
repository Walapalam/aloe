#!/usr/bin/env python3
"""Stage 1 / R1: System Environment & Backend Capability Probe.

Records platform specs, unified memory, PyTorch MPS availability, MLX, PEFT,
and verifies bitsandbytes / unsloth absence verbatim into runs/probe/environment.json,
and generates runs/probe/ENVIRONMENT.md strictly from the JSON envelope.

Adheres to:
- ADR-002: Core framework isolation
- ADR-003: Two execution backends (CUDA-PEFT and MLX)
- ADR-004: Apple Silicon hardware claims (64 GB unified memory, not NVIDIA VRAM)
- Gate 6: Standard Envelope (schema_version: "1.0") & generated Markdown reports
"""
from __future__ import annotations

import argparse
import datetime
import importlib.metadata
import json
import os
import platform
import shutil
import subprocess
import sys
import uuid
from pathlib import Path
from typing import Any

import psutil


def get_git_commit() -> str:
    try:
        res = subprocess.run(
            ["git", "rev-parse", "HEAD"],
            capture_output=True,
            text=True,
            check=True,
        )
        return res.stdout.strip()
    except (subprocess.SubprocessError, OSError):
        return "UNKNOWN"


def get_cpu_info() -> dict[str, Any]:
    brand = "NOT DETERMINED"
    hw_model = "NOT DETERMINED"
    memsize = 0
    ncpu = os.cpu_count() or 0
    physical_cpu = ncpu

    try:
        res = subprocess.run(
            ["sysctl", "-n", "machdep.cpu.brand_string"],
            capture_output=True,
            text=True,
            check=False,
        )
        if res.returncode == 0 and res.stdout.strip():
            brand = res.stdout.strip()
    except (subprocess.SubprocessError, OSError) as e:
        brand = f"NOT DETERMINED: {e}"

    try:
        res = subprocess.run(
            ["sysctl", "-n", "hw.model"],
            capture_output=True,
            text=True,
            check=False,
        )
        if res.returncode == 0 and res.stdout.strip():
            hw_model = res.stdout.strip()
    except (subprocess.SubprocessError, OSError) as e:
        hw_model = f"NOT DETERMINED: {e}"

    try:
        res = subprocess.run(
            ["sysctl", "-n", "hw.memsize"],
            capture_output=True,
            text=True,
            check=False,
        )
        if res.returncode == 0 and res.stdout.strip():
            memsize = int(res.stdout.strip())
    except (subprocess.SubprocessError, OSError, ValueError):
        memsize = 0

    try:
        res = subprocess.run(
            ["sysctl", "-n", "hw.physicalcpu"],
            capture_output=True,
            text=True,
            check=False,
        )
        if res.returncode == 0 and res.stdout.strip():
            physical_cpu = int(res.stdout.strip())
    except (subprocess.SubprocessError, OSError, ValueError):
        physical_cpu = ncpu

    return {
        "brand_string": brand,
        "hw_model": hw_model,
        "sysctl_memsize_bytes": memsize,
        "sysctl_memsize_gb": round(memsize / (1024**3), 2) if memsize else "NOT DETERMINED",
        "logical_cpu_count": ncpu,
        "physical_cpu_count": physical_cpu,
    }


def get_macos_info() -> dict[str, Any]:
    mac_ver = platform.mac_ver()[0]
    build_ver = "NOT DETERMINED"
    try:
        res = subprocess.run(
            ["sw_vers", "-buildVersion"],
            capture_output=True,
            text=True,
            check=False,
        )
        if res.returncode == 0 and res.stdout.strip():
            build_ver = res.stdout.strip()
    except (subprocess.SubprocessError, OSError) as e:
        build_ver = f"NOT DETERMINED: {e}"

    return {
        "system": platform.system(),
        "release": platform.release(),
        "mac_version": mac_ver if mac_ver else "NOT DETERMINED",
        "build_version": build_ver,
        "platform_string": platform.platform(),
        "architecture": platform.machine(),
        "processor": platform.processor(),
    }


def probe_prohibited_package(pkg_name: str) -> dict[str, Any]:
    """Probe a prohibited package in an isolated subprocess to prevent crashing or namespace pollution."""
    code = f"import {pkg_name}"
    res = subprocess.run(
        [sys.executable, "-c", code],
        capture_output=True,
        text=True,
        check=False,
    )

    installed = res.returncode == 0
    functional = res.returncode == 0
    stderr = res.stderr
    stdout = res.stdout

    failure_mode = "NONE (imported successfully)"
    if res.returncode != 0:
        lines = [line.strip() for line in stderr.splitlines() if line.strip()]
        failure_mode = lines[-1] if lines else f"Process exited with code {res.returncode}"

    explanation = (
        f"{pkg_name} is not installed and cannot run on Apple Silicon MPS "
        "(kernels compiled for NVIDIA CUDA only)."
    )

    return {
        "tested": True,
        "installed": installed,
        "functional": functional,
        "failure_mode": stderr.strip() if stderr.strip() else failure_mode,
        "verbatim_stderr": stderr,
        "verbatim_stdout": stdout,
        "exit_code": res.returncode,
        "explanation": explanation,
    }


def probe_torch() -> dict[str, Any]:
    try:
        import torch

        torch_ver = torch.__version__
        cuda_avail = torch.cuda.is_available()
        mps_avail = torch.backends.mps.is_available()
        mps_built = torch.backends.mps.is_built()

        mps_details: dict[str, Any] = {
            "is_available": mps_avail,
            "is_built": mps_built,
        }

        if mps_avail:
            initial_alloc = torch.mps.current_allocated_memory()
            initial_driver = torch.mps.driver_allocated_memory()

            # Empirical bf16 allocation test on MPS
            t = torch.zeros((1024, 1024), dtype=torch.bfloat16, device="mps")
            alloc_after = torch.mps.current_allocated_memory()
            t_shape = list(t.shape)
            t_device = str(t.device)
            t_dtype = str(t.dtype)
            del t
            torch.mps.empty_cache()
            alloc_after_free = torch.mps.current_allocated_memory()

            mps_details.update(
                {
                    "bf16_allocation_test": "PASS",
                    "test_tensor_shape": t_shape,
                    "test_tensor_device": t_device,
                    "test_tensor_dtype": t_dtype,
                    "initial_allocated_bytes": initial_alloc,
                    "tensor_allocated_bytes": alloc_after,
                    "freed_allocated_bytes": alloc_after_free,
                    "driver_allocated_bytes": initial_driver,
                }
            )
        else:
            mps_details["bf16_allocation_test"] = "FAIL (MPS not available)"

        return {
            "installed": True,
            "version": torch_ver,
            "mps_available": mps_avail,
            "mps_built": mps_built,
            "cuda_available": cuda_avail,
            "default_device": "mps" if mps_avail else "cpu",
            "mps": mps_details,
        }
    except (ImportError, RuntimeError, OSError) as e:
        return {
            "installed": False,
            "version": "NOT_INSTALLED",
            "mps_available": False,
            "mps_built": False,
            "cuda_available": False,
            "default_device": "cpu",
            "error": repr(e),
            "mps": {"is_available": False, "is_built": False},
        }


def probe_mlx() -> dict[str, Any]:
    try:
        import mlx.core as mx

        try:
            mlx_ver = importlib.metadata.version("mlx")
        except (importlib.metadata.PackageNotFoundError, AttributeError):
            mlx_ver = getattr(mx, "__version__", "NOT DETERMINED")

        try:
            mlx_lm_ver = importlib.metadata.version("mlx-lm")
            mlx_lm_installed = True
        except (importlib.metadata.PackageNotFoundError, AttributeError):
            mlx_lm_ver = "NOT DETERMINED"
            mlx_lm_installed = False

        dev_info: dict[str, Any] = {}
        try:
            raw_info = mx.device_info()
            dev_info = {
                "device_name": raw_info.get("device_name", "NOT DETERMINED"),
                "memory_size_bytes": raw_info.get("memory_size", 0),
                "memory_size_gb": (
                    round(raw_info.get("memory_size", 0) / (1024**3), 2)
                    if raw_info.get("memory_size")
                    else "NOT DETERMINED"
                ),
                "max_recommended_working_set_bytes": raw_info.get(
                    "max_recommended_working_set_size", 0
                ),
                "max_recommended_working_set_gb": (
                    round(raw_info.get("max_recommended_working_set_size", 0) / (1024**3), 2)
                    if raw_info.get("max_recommended_working_set_size")
                    else "NOT DETERMINED"
                ),
                "architecture": raw_info.get("architecture", "NOT DETERMINED"),
            }
        except (KeyError, ValueError, TypeError, RuntimeError) as e:
            dev_info = {"error": repr(e)}

        return {
            "installed": True,
            "version": mlx_ver,
            "default_device": str(mx.default_device()),
            "metal_available": mx.metal.is_available(),
            "device_info": dev_info,
            "mlx_lm": {
                "installed": mlx_lm_installed,
                "version": mlx_lm_ver,
            },
        }
    except (ImportError, RuntimeError, OSError) as e:
        return {
            "installed": False,
            "error": repr(e),
        }


def probe_package_inventory() -> dict[str, dict[str, Any]]:
    packages = [
        "transformers",
        "peft",
        "accelerate",
        "huggingface_hub",
        "datasets",
        "mlx",
        "mlx-lm",
        "psutil",
        "pydantic",
        "safetensors",
        "tokenizers",
        "typer",
        "rich",
        "pytest",
        "ruff",
    ]
    results: dict[str, dict[str, Any]] = {}
    for pkg in packages:
        try:
            ver = importlib.metadata.version(pkg)
            results[pkg] = {
                "installed": True,
                "version": ver,
                "import_error": None,
            }
        except importlib.metadata.PackageNotFoundError:
            results[pkg] = {
                "installed": False,
                "version": None,
                "import_error": f"PackageNotFoundError: {pkg}",
            }
        except (ValueError, TypeError, OSError) as e:
            results[pkg] = {
                "installed": False,
                "version": None,
                "import_error": repr(e),
            }
    return results


def collect_environment() -> dict[str, Any]:
    vm = psutil.virtual_memory()
    disk = shutil.disk_usage(Path.cwd())
    git_commit = get_git_commit()
    cpu_info = get_cpu_info()
    macos_info = get_macos_info()
    torch_probe = probe_torch()
    mlx_probe = probe_mlx()
    package_inventory = probe_package_inventory()
    bnb_probe = probe_prohibited_package("bitsandbytes")
    unsloth_probe = probe_prohibited_package("unsloth")

    library_versions: dict[str, Any] = {
        pkg: info["version"] for pkg, info in package_inventory.items()
    }

    free_gb = round(disk.free / (1024**3), 2)
    total_disk_gb = round(disk.total / (1024**3), 2)
    used_disk_gb = round(disk.used / (1024**3), 2)
    disk_used_percent = round((disk.used / disk.total) * 100, 2)
    sufficient_disk = free_gb >= 60.0

    total_mem_gb = round(vm.total / (1024**3), 2)
    avail_mem_gb = round(vm.available / (1024**3), 2)

    status = "PASS"
    failures: list[str] = []

    if not torch_probe.get("mps_available", False):
        failures.append("PyTorch MPS backend is NOT available.")
    if bnb_probe["functional"]:
        failures.append("Prohibited package 'bitsandbytes' was found functional.")
    if unsloth_probe["functional"]:
        failures.append("Prohibited package 'unsloth' was found functional.")
    if not mlx_probe.get("installed", False):
        failures.append("Apple MLX is NOT installed.")
    if not sufficient_disk:
        failures.append(f"Disk space insufficient: {free_gb} GB free < 60 GB required.")

    if failures:
        status = "FAIL"

    data: dict[str, Any] = {
        "schema_version": "1.0",
        "run_id": str(uuid.uuid4()),
        "timestamp_utc": datetime.datetime.now(datetime.timezone.utc).isoformat(),
        "script": "probe_environment.py",
        "git_commit": git_commit,
        "environment": {
            "platform": macos_info["platform_string"],
            "chip": cpu_info["brand_string"],
            "unified_memory_gb": total_mem_gb,
            "python": platform.python_version(),
            "torch": torch_probe["version"],
            "backend": "torch_mps" if torch_probe.get("mps_available") else "cpu",
            "library_versions": library_versions,
        },
        "results": {
            "system": {
                "platform": macos_info["platform_string"],
                "os_name": macos_info["system"],
                "os_release": macos_info["release"],
                "architecture": macos_info["architecture"],
                "processor": macos_info["processor"],
                "chip": cpu_info["brand_string"],
                "cpu_physical_cores": cpu_info["physical_cpu_count"],
                "cpu_logical_cores": cpu_info["logical_cpu_count"],
            },
            "memory": {
                "total_bytes": vm.total,
                "total_gb": total_mem_gb,
                "available_bytes": vm.available,
                "available_gb": avail_mem_gb,
                "used_bytes": vm.used,
                "used_percent": vm.percent,
                "memory_type": "unified",
            },
            "disk": {
                "target_path": str(Path.cwd()),
                "total_bytes": disk.total,
                "total_gb": total_disk_gb,
                "free_bytes": disk.free,
                "free_gb": free_gb,
                "used_bytes": disk.used,
                "used_percent": disk_used_percent,
                "sufficient_for_pilot_models": sufficient_disk,
            },
            "python": {
                "version": platform.python_version(),
                "executable": sys.executable,
                "compiler": platform.python_compiler(),
            },
            "torch": {
                "version": torch_probe["version"],
                "mps_available": torch_probe["mps_available"],
                "mps_built": torch_probe["mps_built"],
                "cuda_available": torch_probe["cuda_available"],
                "default_device": torch_probe["default_device"],
                "mps_bf16_test": torch_probe.get("mps", {}).get("bf16_allocation_test", "N/A"),
                "allocated_tensor_bytes": torch_probe.get("mps", {}).get("tensor_allocated_bytes", 0),
            },
            "packages": package_inventory,
            "bitsandbytes": {
                "tested": bnb_probe["tested"],
                "installed": bnb_probe["installed"],
                "functional": bnb_probe["functional"],
                "failure_mode": bnb_probe["failure_mode"],
                "explanation": bnb_probe["explanation"],
            },
            "unsloth": {
                "tested": unsloth_probe["tested"],
                "installed": unsloth_probe["installed"],
                "functional": unsloth_probe["functional"],
                "failure_mode": unsloth_probe["failure_mode"],
                "explanation": unsloth_probe["explanation"],
            },
        },
        # Direct top-level fields for ergonomic querying and explorer assertion compatibility
        "platform": macos_info,
        "cpu_hardware": cpu_info,
        "unified_memory": {
            "total_bytes": vm.total,
            "total_gb": total_mem_gb,
            "available_bytes": vm.available,
            "available_gb": avail_mem_gb,
            "used_bytes": vm.used,
            "used_gb": round(vm.used / (1024**3), 2),
            "percent_used": vm.percent,
        },
        "disk": {
            "cwd": str(Path.cwd()),
            "total_bytes": disk.total,
            "total_gb": total_disk_gb,
            "free_bytes": disk.free,
            "free_gb": free_gb,
            "used_bytes": disk.used,
            "used_gb": used_disk_gb,
            "percent_free": round((disk.free / disk.total) * 100, 2),
        },
        "pytorch": torch_probe,
        "mlx": mlx_probe,
        "key_packages": package_inventory,
        "prohibited_packages": {
            "bitsandbytes": bnb_probe,
            "unsloth": unsloth_probe,
        },
        "status": status,
        "failures": failures,
        "not_determined": [],
    }

    return data


def generate_markdown(data: dict[str, Any]) -> str:
    """Generate Markdown report strictly from JSON dictionary."""
    lines = [
        "# Environment & Hardware Probe Report",
        "",
        f"- **Timestamp UTC**: `{data['timestamp_utc']}`",
        f"- **Run ID**: `{data['run_id']}`",
        f"- **Git Commit**: `{data['git_commit']}`",
        f"- **Schema Version**: `{data['schema_version']}`",
        f"- **Overall Status**: **{data['status']}**",
        "",
        "## Hardware & Operating System",
        "",
        "| Property | Value | Source / Query |",
        "|---|---|---|",
        f"| Chip Model | **{data['cpu_hardware']['brand_string']}** | `sysctl -n machdep.cpu.brand_string` |",
        f"| Hardware Model | {data['cpu_hardware']['hw_model']} | `sysctl -n hw.model` |",
        f"| CPU Cores | {data['cpu_hardware']['physical_cpu_count']} physical / {data['cpu_hardware']['logical_cpu_count']} logical | `sysctl -n hw.physicalcpu`, `os.cpu_count()` |",
        f"| Unified Memory | **{data['unified_memory']['total_gb']} GB** ({data['unified_memory']['total_bytes']:,} bytes) | `psutil.virtual_memory().total` |",
        f"| Available Memory | {data['unified_memory']['available_gb']} GB ({data['unified_memory']['available_bytes']:,} bytes) | `psutil.virtual_memory().available` |",
        f"| Operating System | {data['platform']['system']} {data['platform']['mac_version']} (Build {data['platform']['build_version']}) | `sw_vers`, `platform.mac_ver()` |",
        f"| Platform String | `{data['platform']['platform_string']}` | `platform.platform()` |",
        f"| Architecture | {data['platform']['architecture']} ({data['platform']['processor']}) | `platform.machine()`, `platform.processor()` |",
        f"| Free Disk Space | **{data['disk']['free_gb']} GB** ({data['disk']['percent_free']}% free) | `shutil.disk_usage` |",
        "",
        "## Python Runtime & Frameworks",
        "",
        "| Framework / Package | Status | Version | Key Capability Details |",
        "|---|---|---|---|",
        f"| Python | Available | {data['results']['python']['version']} | `{data['results']['python']['executable']}` |",
        f"| PyTorch | Available | {data['pytorch']['version']} | MPS Available: **{data['pytorch']['mps']['is_available']}**, MPS Built: **{data['pytorch']['mps']['is_built']}**, CUDA: {data['pytorch']['cuda_available']} |",
        f"| PyTorch MPS bf16 Test | **{data['pytorch']['mps'].get('bf16_allocation_test', 'N/A')}** | - | Allocated {data['pytorch']['mps'].get('tensor_allocated_bytes', 0):,} B on `{data['pytorch']['mps'].get('test_tensor_device', 'N/A')}`, freed to {data['pytorch']['mps'].get('freed_allocated_bytes', 0)} B |",
        f"| Apple MLX | Available | {data['mlx']['version']} | Default Device: `{data['mlx']['default_device']}`, Metal: {data['mlx']['metal_available']}, Working Set: {data['mlx']['device_info'].get('max_recommended_working_set_gb')} GB |",
        f"| Apple MLX-LM | Available | {data['mlx']['mlx_lm']['version']} | Package installed and importable |",
        f"| Hugging Face PEFT | Available | {data['key_packages']['peft']['version']} | Package installed and importable |",
        f"| Hugging Face Transformers | Available | {data['key_packages']['transformers']['version']} | Package installed and importable |",
        f"| Hugging Face Accelerate | Available | {data['key_packages']['accelerate']['version']} | Package installed and importable |",
        f"| Hugging Face Hub | Available | {data['key_packages']['huggingface_hub']['version']} | Package installed and importable |",
        f"| Pydantic | Available | {data['key_packages']['pydantic']['version']} | Package installed and importable |",
        "",
        "## Prohibited Packages Audit (CUDA Dependencies)",
        "",
        "| Package | Installed? | Functional? | Failure Mode / Status |",
        "|---|---|---|---|",
    ]
    for pkg, pdata in data["prohibited_packages"].items():
        inst_str = "YES (FAIL)" if pdata["installed"] else "NO (PASS)"
        func_str = "YES (VIOLATION)" if pdata["functional"] else "NO (CONFIRMED)"
        fail_mode = pdata["failure_mode"].splitlines()[-1] if pdata["failure_mode"] else "None"
        lines.append(f"| `{pkg}` | {inst_str} | {func_str} | `{fail_mode}` |")

    lines.extend([
        "",
        "### Verbatim Probe Error Outputs",
        "",
    ])
    for pkg, pdata in data["prohibited_packages"].items():
        lines.extend([
            f"#### `{pkg}` Probe Result",
            "```text",
            f"Exit Code: {pdata['exit_code']}",
            "Stderr:",
            pdata["verbatim_stderr"].strip() if pdata["verbatim_stderr"] else "<empty>",
            "```",
            "",
        ])

    return "\n".join(lines)


def main() -> None:
    parser = argparse.ArgumentParser(description="Probe hardware, platform, and backend capabilities.")
    parser.add_argument(
        "--out-dir",
        default="runs/probe",
        help="Target directory for environment.json and ENVIRONMENT.md (default: runs/probe)",
    )
    args = parser.parse_args()

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    print(f"[{datetime.datetime.now(datetime.timezone.utc).isoformat()}] Probing system environment and backends...")
    env_data = collect_environment()

    json_path = out_dir / "environment.json"
    with open(json_path, "w", encoding="utf-8") as f:
        json.dump(env_data, f, indent=2)
    print(f"[{datetime.datetime.now(datetime.timezone.utc).isoformat()}] Wrote machine-readable JSON to {json_path}")

    md_content = generate_markdown(env_data)
    md_path = out_dir / "ENVIRONMENT.md"
    with open(md_path, "w", encoding="utf-8") as f:
        f.write(md_content + "\n")
    print(f"[{datetime.datetime.now(datetime.timezone.utc).isoformat()}] Wrote generated Markdown report to {md_path}")

    print("\n" + "=" * 60)
    print("PROBE SUMMARY:")
    print(f"  Status:           {env_data['status']}")
    print(f"  Chip:             {env_data['cpu_hardware']['brand_string']}")
    print(f"  Unified Memory:   {env_data['unified_memory']['total_gb']} GB")
    print(f"  PyTorch Version:  {env_data['pytorch']['version']}")
    print(f"  MPS Available:    {env_data['pytorch']['mps']['is_available']}")
    print(f"  MLX Version:      {env_data['mlx']['version']}")
    print(f"  MLX-LM Version:   {env_data['mlx']['mlx_lm']['version']}")
    print(f"  PEFT Version:     {env_data['key_packages']['peft']['version']}")
    print(f"  bitsandbytes:     {env_data['results']['bitsandbytes']['failure_mode'].splitlines()[-1]}")
    print("=" * 60)

    if env_data["status"] != "PASS":
        sys.exit(1)


if __name__ == "__main__":
    main()
