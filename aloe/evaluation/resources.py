"""Resource profiling & fail-closed hardware isolation for ALOE (ADR-004 / ADR-012 / ADR-015 / ADR-017).

Supports strict metric isolation between:
- Apple Silicon unified memory: 100ms sampled Process RSS (`process_rss_sampled_100ms`).
- NVIDIA CUDA discrete VRAM (L4 / A100): allocator peak (`cuda_max_memory_allocated`).
Never mixes or compares RSS against CUDA VRAM across hardware tiers.
"""
from __future__ import annotations

import os
import platform
import subprocess
import threading
import time
from typing import Any

import psutil
import torch


def has_causal_conv1d() -> bool:
    """Return True if CUDA fast-path `causal_conv1d` kernel can be imported."""
    try:
        import causal_conv1d  # type: ignore  # noqa: F401
        return True
    except Exception:
        return False


def has_fast_mamba_kernels() -> bool:
    """Return True if CUDA fast-path Mamba-2 + causal-conv1d kernels can be imported."""
    try:
        import causal_conv1d  # type: ignore  # noqa: F401
        import mamba_ssm  # type: ignore  # noqa: F401
        return True
    except Exception:
        return False


def detect_hardware_summary() -> dict[str, Any]:
    """Detect runtime OS, CPU, GPU SKU, and CUDA/MPS availability without hardcoded strings."""
    os_desc = platform.platform()
    machine = platform.machine()
    total_ram_gb = round(psutil.virtual_memory().total / (1024**3), 2)

    if torch.cuda.is_available():
        gpu_name = torch.cuda.get_device_name(0)
        props = torch.cuda.get_device_properties(0)
        vram_gb = round(props.total_memory / (1024**3), 2)
        cuda_ver = torch.version.cuda or "unknown"
        driver_ver = "unknown"
        try:
            nv = subprocess.run(
                ["nvidia-smi", "--query-gpu=driver_version", "--format=csv,noheader"],
                capture_output=True,
                text=True,
                timeout=3,
                check=False,
            )
            if nv.returncode == 0 and nv.stdout.strip():
                driver_ver = nv.stdout.strip().splitlines()[0].strip()
        except Exception:
            pass
        return {
            "device": "cuda",
            "gpu_name": gpu_name,
            "gpu_vram_gb": vram_gb,
            "cuda_version": cuda_ver,
            "nvidia_driver": driver_ver,
            "fast_mamba_kernels": has_fast_mamba_kernels(),
            "fast_causal_conv1d": has_causal_conv1d(),
            "host_ram_gb": total_ram_gb,
            "hardware_label": f"{gpu_name} ({vram_gb} GB VRAM, CUDA {cuda_ver}, driver {driver_ver})",
            "os": os_desc,
        }

    if hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
        cpu_brand = platform.processor() or machine
        try:
            sc = subprocess.run(
                ["sysctl", "-n", "machdep.cpu.brand_string"],
                capture_output=True,
                text=True,
                timeout=2,
                check=False,
            )
            if sc.returncode == 0 and sc.stdout.strip():
                cpu_brand = sc.stdout.strip()
        except Exception:
            pass
        return {
            "device": "mps",
            "gpu_name": cpu_brand,
            "gpu_vram_gb": None,
            "cuda_version": None,
            "nvidia_driver": None,
            "fast_mamba_kernels": False,
            "host_ram_gb": total_ram_gb,
            "hardware_label": f"{cpu_brand} ({total_ram_gb} GB unified memory), macOS, MPS",
            "os": os_desc,
        }

    return {
        "device": "cpu",
        "gpu_name": None,
        "gpu_vram_gb": None,
        "cuda_version": None,
        "nvidia_driver": None,
        "fast_mamba_kernels": False,
        "host_ram_gb": total_ram_gb,
        "hardware_label": f"CPU ({machine}, {total_ram_gb} GB RAM), {os_desc}",
        "os": os_desc,
    }


def enforce_hardware_contract(
    model_id: str,
    device: str,
    *,
    pinned_gpu: str | None = None,
    exact_gpu_name: str | None = None,
    official_campaign: bool = False,
    allow_slow_kernels: bool = False,
) -> dict[str, Any]:
    """Fail-closed enforcement of GPU SKU pinning and hybrid kernel requirements (ADR-012 / ADR-017 / ADR-018)."""
    hw = detect_hardware_summary()
    mid = model_id.lower()

    if pinned_gpu:
        if device != "cuda" or not hw.get("gpu_name"):
            raise RuntimeError(
                f"Hardware contract violation: pinned_gpu='{pinned_gpu}' requires CUDA GPU, "
                f"but active device is '{device}' ({hw['hardware_label']})."
            )
        actual_gpu = str(hw["gpu_name"])
        if pinned_gpu.lower() not in actual_gpu.lower():
            raise RuntimeError(
                f"GPU SKU mismatch: expected '{pinned_gpu}', got '{actual_gpu}'. "
                "Mixing GPU SKUs within a campaign invalidates VRAM comparisons."
            )

    if exact_gpu_name:
        actual_gpu = str(hw.get("gpu_name") or hw["hardware_label"])
        if actual_gpu != exact_gpu_name:
            raise RuntimeError(
                f"Exact GPU SKU lock violation: campaign locked to '{exact_gpu_name}', got '{actual_gpu}'. "
                "Mixing e.g. 40GB and 80GB variants within a campaign is forbidden."
            )

    if official_campaign:
        is_hybrid = ("falcon-h1" in mid) or ("lfm3" in mid)
        if is_hybrid and device == "mps":
            raise RuntimeError(
                f"ADR-012 / ADR-017 violation: official campaign for hybrid model '{model_id}' "
                "cannot run on Apple MPS (unfused scan fallback + incompatible RSS metric). "
                "Execute on pinned CUDA GPU (NVIDIA A100 / L4)."
            )
        if not is_hybrid and "gemma" in mid and device == "cuda" and not allow_slow_kernels:
            raise RuntimeError(
                f"ADR-012 hardware-tier violation: official campaign for local-tier model '{model_id}' "
                "belongs on Apple Silicon MPS (`process_rss_sampled_100ms`), not CUDA."
            )
        if is_hybrid and device == "cuda" and not allow_slow_kernels:
            if "falcon-h1" in mid and not hw.get("fast_mamba_kernels"):
                raise RuntimeError(
                    "Falcon-H1 CUDA fast-path check failed: 'mamba_ssm' and/or 'causal_conv1d' "
                    "are not installed. Refusing silent 9x fallback to Python scan; install "
                    "prebuilt wheels or pass --allow-slow-kernels explicitly for non-campaign tests."
                )
            if "lfm3" in mid and not hw.get("fast_causal_conv1d", hw.get("fast_mamba_kernels")):
                raise RuntimeError(
                    "LFM3 CUDA fast-path check failed: 'causal_conv1d' is not installed for "
                    "ShortConv acceleration. Install prebuilt wheels or pass --allow-slow-kernels "
                    "explicitly for non-campaign tests."
                )

    return hw


class ResourceProfiler:
    """Periodic sampler for Process RSS, Apple MPS allocation, and CUDA peak VRAM."""

    def __init__(self, sample_interval_ms: int = 100, *, device: str | None = None) -> None:
        self.sample_interval_ms = sample_interval_ms
        self.sample_interval_sec = sample_interval_ms / 1000.0
        self._stop_event = threading.Event()
        self._thread: threading.Thread | None = None
        self._samples_mps: list[int] = []
        self._samples_rss: list[int] = []
        self._process = psutil.Process(os.getpid())
        if device is not None:
            self._has_cuda = device == "cuda"
            self._has_mps = device == "mps"
        else:
            self._has_mps = bool(hasattr(torch.backends, "mps") and torch.backends.mps.is_available())
            self._has_cuda = bool(torch.cuda.is_available())

    def _sample_loop(self) -> None:
        while not self._stop_event.is_set():
            if self._has_mps:
                try:
                    alloc = torch.mps.current_allocated_memory()
                    self._samples_mps.append(alloc)
                except Exception:
                    pass
            try:
                rss = self._process.memory_info().rss
                self._samples_rss.append(rss)
            except Exception:
                pass
            time.sleep(self.sample_interval_sec)

    def start(self) -> None:
        self._samples_mps.clear()
        self._samples_rss.clear()
        self._stop_event.clear()
        if self._has_cuda:
            try:
                torch.cuda.synchronize()
                torch.cuda.reset_peak_memory_stats()
            except Exception:
                pass
        self._thread = threading.Thread(target=self._sample_loop, daemon=True)
        self._thread.start()

    def snapshot(self) -> dict[str, float]:
        """Sample current running peak memory without stopping the background sampler."""
        try:
            self._samples_rss.append(self._process.memory_info().rss)
        except Exception:
            pass
        if self._has_mps:
            try:
                self._samples_mps.append(torch.mps.current_allocated_memory())
            except Exception:
                pass
        peak_rss = max(self._samples_rss) if self._samples_rss else 0
        peak_mps = max(self._samples_mps) if self._samples_mps else 0
        cuda_alloc = 0
        cuda_res = 0
        if self._has_cuda:
            cuda_alloc = int(torch.cuda.max_memory_allocated())
            cuda_res = int(torch.cuda.max_memory_reserved())
        search_mb = round((cuda_alloc if self._has_cuda else peak_rss) / (1024**2), 3)
        peak_gb = round((cuda_alloc if self._has_cuda else peak_rss) / (1024**3), 4)
        return {
            "peak_memory_gb": peak_gb,
            "peak_rss_gb": round(peak_rss / (1024**3), 4),
            "mps_driver_allocated_gb": round(peak_mps / (1024**3), 4),
            "cuda_max_memory_allocated_gb": round(cuda_alloc / (1024**3), 4),
            "cuda_max_memory_reserved_gb": round(cuda_res / (1024**3), 4),
            "search_memory_mb": search_mb,
        }

    def stop(self) -> dict[str, Any]:
        self._stop_event.set()
        if self._thread is not None:
            self._thread.join(timeout=2.0)

        if self._has_mps:
            try:
                self._samples_mps.append(torch.mps.current_allocated_memory())
            except Exception:
                pass
        try:
            self._samples_rss.append(self._process.memory_info().rss)
        except Exception:
            pass

        cuda_alloc_bytes = 0
        cuda_reserved_bytes = 0
        if self._has_cuda:
            torch.cuda.synchronize()
            cuda_alloc_bytes = int(torch.cuda.max_memory_allocated())
            cuda_reserved_bytes = int(torch.cuda.max_memory_reserved())
            if cuda_alloc_bytes <= 0:
                raise RuntimeError(
                    "CUDA profiler recorded 0 bytes for max_memory_allocated() on active CUDA run; "
                    "refusing fail-open 0 MB Pareto memory observation."
                )

        peak_mps = max(self._samples_mps) if self._samples_mps else 0
        peak_rss = max(self._samples_rss) if self._samples_rss else 0
        sample_count = max(len(self._samples_mps), len(self._samples_rss), 1 if self._has_cuda else 0)

        peak_rss_gb = round(peak_rss / (1024**3), 4)
        mps_driver_allocated_gb = round(peak_mps / (1024**3), 4)
        cuda_alloc_gb = round(cuda_alloc_bytes / (1024**3), 4)
        cuda_reserved_gb = round(cuda_reserved_bytes / (1024**3), 4)

        if self._has_cuda:
            primary_bytes = cuda_alloc_bytes
            primary_gb = cuda_alloc_gb
            search_memory_mb = round(cuda_alloc_bytes / (1024**2), 3)
            metric_name = "cuda_max_memory_allocated"
            method_name = "cuda_allocator_peak"
        else:
            primary_bytes = peak_rss
            primary_gb = peak_rss_gb
            search_memory_mb = round(peak_rss / (1024**2), 3)
            metric_name = "process_rss_sampled_100ms"
            method_name = "sampled"

        return {
            "peak_memory_bytes": primary_bytes,
            "peak_memory_gb": primary_gb,
            "search_memory_mb": search_memory_mb,
            "peak_rss_bytes": peak_rss,
            "peak_rss_gb": peak_rss_gb,
            "mps_driver_allocated_bytes": peak_mps,
            "mps_driver_allocated_gb": mps_driver_allocated_gb,
            "cuda_max_memory_allocated_bytes": cuda_alloc_bytes,
            "cuda_max_memory_allocated_gb": cuda_alloc_gb,
            "cuda_max_memory_allocated_mb": round(cuda_alloc_bytes / (1024**2), 3),
            "cuda_max_memory_reserved_bytes": cuda_reserved_bytes,
            "cuda_max_memory_reserved_gb": cuda_reserved_gb,
            "memory_metric": metric_name,
            "memory_method": method_name,
            "sample_interval_ms": self.sample_interval_ms,
            "sample_count": sample_count,
        }

    def __enter__(self) -> ResourceProfiler:
        self.start()
        return self

    def __exit__(self, exc_type: Any, exc_val: Any, exc_tb: Any) -> None:
        self.stop()
