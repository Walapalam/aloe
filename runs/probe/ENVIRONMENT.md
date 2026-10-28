# Environment & Hardware Probe Report

- **Timestamp UTC**: `2026-09-17T13:05:56.786104+00:00`
- **Run ID**: `3b1b845d-0242-42a6-8ddf-98fb6a64b5b3`
- **Git Commit**: `ac96ece0eae4b1bcbc991e0d3e3f96c7760bc8a4`
- **Schema Version**: `1.0`
- **Overall Status**: **PASS**

## Hardware & Operating System

| Property | Value | Source / Query |
|---|---|---|
| Chip Model | **Apple M1 Max** | `sysctl -n machdep.cpu.brand_string` |
| Hardware Model | MacBookPro18,2 | `sysctl -n hw.model` |
| CPU Cores | 10 physical / 10 logical | `sysctl -n hw.physicalcpu`, `os.cpu_count()` |
| Unified Memory | **64.0 GB** (68,719,476,736 bytes) | `psutil.virtual_memory().total` |
| Available Memory | 40.34 GB (43,320,016,896 bytes) | `psutil.virtual_memory().available` |
| Operating System | Darwin 26.6.2 (Build 25G83) | `sw_vers`, `platform.mac_ver()` |
| Platform String | `macOS-26.6.2-arm64-arm-64bit` | `platform.platform()` |
| Architecture | arm64 (arm) | `platform.machine()`, `platform.processor()` |
| Free Disk Space | **751.57 GB** (81.13% free) | `shutil.disk_usage` |

## Python Runtime & Frameworks

| Framework / Package | Status | Version | Key Capability Details |
|---|---|---|---|
| Python | Available | 3.12.14 | `/Users/walapalam/Development/fyp/aloe/.venv/bin/python` |
| PyTorch | Available | 2.14.0 | MPS Available: **True**, MPS Built: **True**, CUDA: False |
| PyTorch MPS bf16 Test | **PASS** | - | Allocated 2,097,152 B on `mps:0`, freed to 0 B |
| Apple MLX | Available | 0.32.2 | Default Device: `Device(gpu, 0)`, Metal: True, Working Set: 51.84 GB |
| Apple MLX-LM | Available | 0.31.3 | Package installed and importable |
| Hugging Face PEFT | Available | 0.21.0 | Package installed and importable |
| Hugging Face Transformers | Available | 5.17.0 | Package installed and importable |
| Hugging Face Accelerate | Available | 1.15.0 | Package installed and importable |
| Hugging Face Hub | Available | 1.32.0 | Package installed and importable |
| Pydantic | Available | 2.13.5 | Package installed and importable |

## Prohibited Packages Audit (CUDA Dependencies)

| Package | Installed? | Functional? | Failure Mode / Status |
|---|---|---|---|
| `bitsandbytes` | NO (PASS) | NO (CONFIRMED) | `ModuleNotFoundError: No module named 'bitsandbytes'` |
| `unsloth` | NO (PASS) | NO (CONFIRMED) | `ModuleNotFoundError: No module named 'unsloth'` |

### Verbatim Probe Error Outputs

#### `bitsandbytes` Probe Result
```text
Exit Code: 1
Stderr:
Traceback (most recent call last):
  File "<string>", line 1, in <module>
ModuleNotFoundError: No module named 'bitsandbytes'
```

#### `unsloth` Probe Result
```text
Exit Code: 1
Stderr:
Traceback (most recent call last):
  File "<string>", line 1, in <module>
ModuleNotFoundError: No module named 'unsloth'
```

