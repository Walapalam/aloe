# ALOE — Scientific & Engineering Claims Traceability (`CLAIMS.md`)

Every headline methodological and engineering claim maps directly to executable source code, automated conformance tests, and zero-dependency stdlib verification scripts:

| Claim | Source Module | Automated Conformance Test | Offline Stdlib Verification |
| :--- | :--- | :--- | :--- |
| **3-Model Role-Typed Search Grammars (`205 / 192 / 142` valid targets)** | `aloe/architecture/*`, `aloe/search_space/grammar.py` | `tests/test_architecture_verification.py`, `tests/test_differentiation.py` | `aloe inspect <model_id>` |
| **Exact Shape-Derived `±5%` Parameter Parity (`550 / 430 / 11,375` states)** | `aloe/execution/trial_config.py`, `aloe/execution/campaign_runner.py` | `tests/test_budget_matching.py`, `tests/test_cli_and_coverage.py` | `aloe search --model <id>` |
| **3-Way Stratified Tool-Disjoint Split (`1,000 / 200 / 400`, `0` overlap)** | `scripts/build_xlam_split.py`, `data/manifests/xlam_ipd_v1.json` | `tests/test_xlam_split.py`, `tests/test_protocol_v1_conformance.py` | `python3 scripts/verify_campaign.py` |
| **5 Scrambled-Sobol + 15 Discrete `qLogNEHVI` (`allow_fallback=False`)** | `aloe/optimization/bo_adapter.py`, `aloe/optimization/mobo.py` | `tests/test_mobo_loop.py`, `tests/test_protocol_v1_conformance.py` | `python3 scripts/verify_campaign.py --campaign-dir <dir>` |
| **Strict Multi-Call & Schema-Aware AST Evaluation (`v2.0.0-strict`)** | `aloe/evaluation/tool_calling.py`, `aloe/evaluation/prediction_store.py` | `tests/test_tool_calling_evaluator.py`, `tests/test_protocol_v1_conformance.py` | `python3 scripts/recompute_metrics.py <dir> --check-trial-json` |
| **Paired Bootstrap (`B=10,000`, BCa) + Step-Down Holm-Bonferroni + McNemar** | `aloe/evaluation/bootstrap.py`, `aloe/cli/main.py` | `tests/test_bootstrap.py`, `tests/test_protocol_v1_conformance.py` | `aloe evaluate --run-dir <dir> --split validation` |
| **Hardware Metric Isolation (`MPS 100ms RSS` vs Pinned `CUDA max_memory_allocated`)** | `aloe/evaluation/resources.py`, `docs/DECISIONS.md` (`ADR-004, 012, 015, 017`) | `tests/test_protocol_v1_conformance.py` | Recorded in `trial.json` `memory_metric_name` & `gpu_name` |
| **Atomic Persistence & Mid-Campaign / Mid-Trial Crash Recovery** | `aloe/tracking/trial_store.py`, `aloe/execution/backends/peft_mps.py` | `tests/test_protocol_v1_conformance.py` | `campaign_ledger.json` + `trial.json` atomic `os.replace` |
