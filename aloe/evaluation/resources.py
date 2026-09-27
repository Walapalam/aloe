"""Resource profiling for ALOE (Stage 5 / R5).

Auditable periodic background sampling of torch.mps allocated memory and process RSS
to record true peak memory during training and inference.
"""
from __future__ import annotations

import os
import threading
import time
from typing import Any

import psutil
import torch


class ResourceProfiler:
    """Periodic sampler for MPS memory and process RSS."""

    def __init__(self, sample_interval_ms: int = 100) -> None:
        self.sample_interval_ms = sample_interval_ms
        self.sample_interval_sec = sample_interval_ms / 1000.0
        self._stop_event = threading.Event()
        self._thread: threading.Thread | None = None
        self._samples_mps: list[int] = []
        self._samples_rss: list[int] = []
        self._process = psutil.Process(os.getpid())
        self._has_mps = bool(hasattr(torch.backends, "mps") and torch.backends.mps.is_available())

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
        self._thread = threading.Thread(target=self._sample_loop, daemon=True)
        self._thread.start()

    def stop(self) -> dict[str, Any]:
        self._stop_event.set()
        if self._thread is not None:
            self._thread.join(timeout=2.0)

        # Final reading
        if self._has_mps:
            try:
                self._samples_mps.append(torch.mps.current_allocated_memory())
            except Exception:
                pass
        try:
            self._samples_rss.append(self._process.memory_info().rss)
        except Exception:
            pass

        peak_mps = max(self._samples_mps) if self._samples_mps else 0
        peak_rss = max(self._samples_rss) if self._samples_rss else 0
        sample_count = max(len(self._samples_mps), len(self._samples_rss))

        # Per ADR-004: on Apple Silicon, peak_memory_gb is defined as peak RSS.
        # mps_driver_allocated_gb is recorded alongside for diagnostic purposes.
        peak_rss_gb = round(peak_rss / (1024**3), 4)
        mps_driver_allocated_gb = round(peak_mps / (1024**3), 4)

        return {
            "peak_memory_bytes": peak_rss,
            "peak_memory_gb": peak_rss_gb,
            "peak_rss_bytes": peak_rss,
            "peak_rss_gb": peak_rss_gb,
            "mps_driver_allocated_bytes": peak_mps,
            "mps_driver_allocated_gb": mps_driver_allocated_gb,
            "memory_metric": "process_rss_sampled_100ms",
            "memory_method": "sampled",
            "sample_interval_ms": self.sample_interval_ms,
            "sample_count": sample_count,
        }

    def __enter__(self) -> ResourceProfiler:
        self.start()
        return self

    def __exit__(self, exc_type: Any, exc_val: Any, exc_tb: Any) -> None:
        self.stop()
