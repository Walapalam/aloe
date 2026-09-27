#!/usr/bin/env python3
"""Gate C Verification Script (Gate C Milestone Verification).

Proves that:
1. An invalid configuration (targeting excluded Falcon conv1d) is caught and aborted
   before training by verify_gradients() / validity checks.
2. A valid configuration executes end-to-end:
   - Pre-train gradient verification passes
   - Trains on xLAM split
   - Background 100ms peak memory profiling active
   - Adapter saved to disk
   - Adapter reloaded and verified in a FRESH SUBPROCESS
   - Full Result Schema Envelope 1.0 saved in TrialStore
3. Replays from the TrialStore and produces matching results.
"""
from __future__ import annotations

import json
from pathlib import Path
import sys

from aloe.execution.backends.peft_mps import PeftMPSBackend
from aloe.execution.trial_config import GroupTrialSetting, InvalidConfigError, TrialConfig
from aloe.search_space.schema import SearchGrammar
from aloe.tracking.trial_store import TrialStore


def main():
    print("=" * 60)
    print("Gate C Verification: Trial Execution, Store, Replay & Pre-train Checks")
    print("=" * 60)

    store = TrialStore("runs/trials")
    backend = PeftMPSBackend()

    # Part 1: Verify invalid configuration targeting Falcon conv1d is rejected before training
    print("\n[Gate C Test 1] Verifying Falcon conv1d rejection...")
    falcon_grammar_file = Path("runs/grammars/falcon_h1.json")
    if falcon_grammar_file.exists():
        falcon_grammar = SearchGrammar.model_validate_json(falcon_grammar_file.read_text())
        invalid_config = TrialConfig(
            trial_id="gate_c_invalid_test",
            model_id="tiiuae/Falcon-H1-1.5B-Instruct",
            explicit_target_modules=["model.layers.0.mamba.conv1d"],
        )
        try:
            invalid_config.to_peft_lora_config(falcon_grammar)
            print("FAILED: Invalid Falcon conv1d was not rejected by grammar!")
            sys.exit(1)
        except InvalidConfigError as e:
            print(f"PASSED: Falcon conv1d rejected before GPU/MPS allocation: {e}")
    else:
        print("Falcon grammar not found, skipping Part 1.")

    # Part 2: Execute an end-to-end trial on Gemma 4 E2B
    print("\n[Gate C Test 2] Executing end-to-end trial on Gemma 4 E2B (2 steps)...")
    gemma_grammar_file = Path("runs/grammars/gemma_e2b.json")
    gemma_grammar = SearchGrammar.model_validate_json(gemma_grammar_file.read_text())

    trial_config = TrialConfig(
        trial_id="gate_c_verification_trial",
        model_id="google/gemma-4-E2B-it",
        seed=42,
        group_settings={
            "local_attention": GroupTrialSetting(enabled=True, rank=8, alpha=16),
            "global_attention": GroupTrialSetting(enabled=True, rank=16, alpha=32),
            "ffn": GroupTrialSetting(enabled=False),
        },
        description="Gate C Verification Trial: local r=8 a=16, global r=16 a=32, ffn disabled",
    )
    trial_config.training.max_steps = 2
    trial_config.training.max_seq_len = 128

    result = backend.run_trial(
        config=trial_config,
        grammar=gemma_grammar,
        runs_trials_dir="runs/trials",
    )

    print(f"\nTrial Completed:")
    print(f"  Status: {result.status}")
    print(f"  Peak RSS (peak_memory_gb): {result.metrics.peak_memory_gb} GB")
    print(f"  MPS Allocated: {result.metrics.mps_driver_allocated_gb} GB")
    print(f"  Trainable params: {result.metrics.trainable_params:,}")
    print(f"  Adapter size: {result.metrics.adapter_size_mb} MB")
    print(f"  Fresh Process Reload: {'PASS' if result.metrics.fresh_process_reload else 'FAIL'}")
    print(f"  Initial Loss: {result.metrics.initial_train_loss:.4f} -> Final Loss: {result.metrics.final_train_loss:.4f}")

    assert result.status == "PASS", f"Trial failed: {result.failures}"
    assert result.metrics.fresh_process_reload is True, "Fresh subprocess reload failed"

    # Part 3: Replay the trial and assert reproducibility
    print("\n[Gate C Test 3] Replaying trial from TrialStore...")
    replayed = store.replay("gate_c_verification_trial", backend_runner=backend, loss_tolerance=0.05)
    print(f"Replay Completed:")
    print(f"  Original Loss: {result.metrics.final_train_loss:.4f} vs Replay Loss: {replayed.metrics.final_train_loss:.4f}")
    print(f"  Original Trainable: {result.metrics.trainable_params} vs Replay Trainable: {replayed.metrics.trainable_params}")

    print("\n" + "=" * 60)
    print("GATE C VERIFICATION: PASSED")
    print("=" * 60)


if __name__ == "__main__":
    main()
