"""Isolated single-trial worker subprocess entry point for CampaignRunner.

Running each trial in its own short-lived Python process guarantees 100% release
of CUDA VRAM / Apple Silicon unified memory, PEFT adapter hooks, and RNG state
before the next trial starts.
"""
from __future__ import annotations

import argparse
from pathlib import Path
import sys

from aloe.execution.backends.peft_mps import PeftMPSBackend
from aloe.execution.trial_config import TrialConfig
from aloe.pipeline import inspect_to_grammar


def main() -> int:
    parser = argparse.ArgumentParser(description="ALOE isolated trial worker")
    parser.add_argument("--config", required=True, help="Path to TrialConfig JSON")
    parser.add_argument("--trials-dir", required=True, help="Directory for trial outputs")
    parser.add_argument("--eval-split", default="validation", help="Evaluation split name")
    parser.add_argument("--parity-tol", type=float, default=0.05)
    parser.add_argument("--baseline-params", type=int, default=0)
    parser.add_argument("--enforce-parity", action="store_true")
    parser.add_argument("--pinned-gpu", default=None)
    parser.add_argument("--exact-gpu-name", default=None)
    parser.add_argument("--official-campaign", action="store_true")
    parser.add_argument("--allow-slow-kernels", action="store_true")
    parser.add_argument("--forbid-dirty-git", action="store_true")
    parser.add_argument("--mirror-dir", default=None)
    parser.add_argument("--device", default=None)
    parser.add_argument("--skip-subprocess-reload", action="store_true")
    parser.add_argument("--final", action="store_true", help="Unlock sealed test split for final evaluation")
    parser.add_argument("--seal-file", default=None, help="Path to test_unsealed.json for official --final runs")
    args = parser.parse_args()

    cfg = TrialConfig.from_file(Path(args.config))
    arch = None
    grammar = None
    if cfg.explicit_target_modules is None:
        arch, grammar = inspect_to_grammar(cfg.model_id, backend=cfg.backend)
    backend = PeftMPSBackend()

    envelope = backend.run_trial(
        config=cfg,
        trial_id=cfg.trial_id,
        runs_trials_dir=Path(args.trials_dir),
        grammar=grammar,
        arch=arch,
        enforce_budget_parity=args.enforce_parity,
        parity_tolerance=args.parity_tol,
        baseline_params=args.baseline_params if args.baseline_params > 0 else None,
        eval_split=args.eval_split if args.eval_split != "none" else None,
        pinned_gpu=args.pinned_gpu,
        exact_gpu_name=args.exact_gpu_name,
        official_campaign=args.official_campaign,
        allow_slow_kernels=args.allow_slow_kernels,
        forbid_dirty_git=args.forbid_dirty_git,
        resume=True,
        skip_subprocess_reload=args.skip_subprocess_reload,
        mirror_dir=Path(args.mirror_dir) if args.mirror_dir else None,
        device_override=args.device,
        allow_sealed_test=args.final,
        seal_file=Path(args.seal_file) if args.seal_file else None,
    )
    return 0 if envelope.status == "PASS" else 1


if __name__ == "__main__":
    sys.exit(main())
