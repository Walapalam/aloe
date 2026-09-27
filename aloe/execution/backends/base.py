"""Execution backend contract.

ALOE runs on CUDA (HF PEFT + bitsandbytes) and on Apple Silicon (MLX). Nothing above
this file may know which. See DECISIONS.md ADR-003 — bitsandbytes has no production
Apple Silicon support, so the Mac path is MLX, not QLoRA-via-PEFT."""
from __future__ import annotations

from typing import Protocol

from aloe.search_space.schema import SearchGrammar


class TrialResult(Protocol):
    adapter_path: str
    peak_memory_gb: float
    train_seconds: float
    trainable_params: int
    adapter_size_mb: float


class ExecutionBackend(Protocol):
    name: str

    def available(self) -> bool:
        """True if this backend can run on the current machine. Never raise."""

    def supports(self, grammar: SearchGrammar) -> tuple[bool, str]:
        """Can this backend execute every group in the grammar? Return (ok, reason)."""

    def run_trial(self, config) -> TrialResult:
        """Train one configuration and return measured cost. Must be deterministic
        given a seed, and must record peak memory as MEASURED, not estimated."""

    def verify_gradients(self, config) -> dict[str, bool]:
        """CRITICAL. For each adapter target, confirm a non-zero gradient actually
        reaches it after one backward pass. This is what separates ALOE's validity
        layer from 'the adapter attached successfully'. A target that attaches but
        never receives gradient is the silent failure Gap 3 exists to catch."""
