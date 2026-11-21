"""Integration and unit tests for ALOE CLI commands, pipeline, resources, xLAM, and search."""
from __future__ import annotations

import json
from pathlib import Path
from typing import Any
import pytest
from typer.testing import CliRunner

from aloe.cli.main import _run_kernel_autograd_sanity_check, app, compute_parity_space
from aloe.datasets.xlam import XLAMExample, XLAMToolCall, XLAMToolDefinition, load_xlam_file, parse_xlam_record
from aloe.evaluation.resources import ResourceProfiler
from aloe.execution.backends.base import ExecutionBackend, TrialResult
from aloe.execution.trial_config import GroupTrialSetting, TrialConfig
from aloe.optimization.random_search import RandomSearchSampler
from aloe.pipeline import inspect_to_grammar
from aloe.tracking.trial_store import (
    TrialMetrics,
    TrialResultEnvelope,
    TrialStore,
    get_environment_metadata,
    get_file_sha256,
    get_git_commit,
)

runner = CliRunner()


def test_cli_probe_falcon_and_gemma_and_lfm3(tmp_path: Path) -> None:
    """Verify probe CLI command executes autograd sanity check and prints verified gradient norms."""
    out_file = tmp_path / "falcon_probe.json"
    res = runner.invoke(app, ["probe", "--model", "tiiuae/Falcon-H1-1.5B-Instruct", "--out", str(out_file)])
    assert res.exit_code == 0, res.output
    assert "tiiuae/Falcon-H1-1.5B-Instruct" in res.output
    assert "48/48 conv1d GRADIENT_MISSING" in res.output
    assert out_file.is_file()
    data = json.loads(out_file.read_text(encoding="utf-8"))
    assert data["valid_modules"] == 192
    assert data["ok_tensors"] == 384

    res_gemma = runner.invoke(app, ["probe", "--model", "google/gemma-4-E2B-it"])
    assert res_gemma.exit_code == 0, res_gemma.output
    assert "205/205 valid modules" in res_gemma.output

    res_lfm3 = runner.invoke(app, ["probe", "--model", "LiquidAI/LFM3-1.7B"])
    assert res_lfm3.exit_code == 0, res_lfm3.output
    assert "26/26 conv.conv GRADIENT_MISSING" in res_lfm3.output


def test_cli_inspect_and_search(tmp_path: Path) -> None:
    """Verify inspect and search CLI commands enumerate parity states and write TrialConfigs."""
    gram_out = tmp_path / "grammar.json"
    res_insp = runner.invoke(app, ["inspect", "tiiuae/Falcon-H1-1.5B-Instruct", "--out", str(gram_out)])
    assert res_insp.exit_code == 0, res_insp.output
    assert gram_out.is_file()

    search_out = tmp_path / "search_gemma"
    res_search = runner.invoke(
        app,
        [
            "search",
            "--model",
            "google/gemma-4-E2B-it",
            "--trials",
            "6",
            "--parity-tol",
            "0.05",
            "--out",
            str(search_out),
        ],
    )
    assert res_search.exit_code == 0, res_search.output
    assert "6 parity rank vectors (550 coupled states)" in res_search.output
    assert len(list(search_out.glob("trial_mobo_*.json"))) == 6
    summary = json.loads((search_out / "search_summary.json").read_text(encoding="utf-8"))
    assert summary["parity_coupled_states"] == 550
    assert summary["trials_generated"] == 6


def test_cli_evaluate_and_run_errors(tmp_path: Path) -> None:
    """Verify evaluate CLI computes AST metrics and paired bootstrap CIs on pilot trials."""
    res_eval = runner.invoke(
        app,
        ["evaluate", "--run-dir", "runs/pilot", "--split", "test", "--bootstrap", "500"],
    )
    assert res_eval.exit_code == 0, res_eval.output
    assert "google/gemma-4-E2B-it" in res_eval.output
    assert "tiiuae/Falcon-H1-1.5B-Instruct" in res_eval.output
    assert "BOOTSTRAP P4_SSM_Matched vs P1" in res_eval.output

    res_missing = runner.invoke(app, ["evaluate", "--run-dir", str(tmp_path / "empty")])
    assert res_missing.exit_code == 1

    res_run_missing = runner.invoke(app, ["run", str(tmp_path / "nonexistent.json")])
    assert res_run_missing.exit_code == 1


def test_compute_parity_space_all_three_models() -> None:
    """Verify compute_parity_space matches exact 430 / 550 / 11,375 coupled states."""
    desc_f, gram_f = inspect_to_grammar("tiiuae/Falcon-H1-1.5B-Instruct")
    pf = compute_parity_space(desc_f, gram_f, parity_tol=0.05)
    assert pf["baseline_params"] == 7_578_624
    assert pf["parity_rank_vectors"] == 6
    assert pf["coupled_states"] == 430

    desc_g, gram_g = inspect_to_grammar("google/gemma-4-E2B-it")
    pg = compute_parity_space(desc_g, gram_g, parity_tol=0.05)
    assert pg["baseline_params"] == 12_079_104
    assert pg["parity_rank_vectors"] == 6
    assert pg["coupled_states"] == 550

    desc_l, gram_l = inspect_to_grammar("LiquidAI/LFM3-1.7B")
    pl = compute_parity_space(desc_l, gram_l, parity_tol=0.05)
    assert pl["baseline_params"] == 8_052_736
    assert pl["parity_rank_vectors"] == 35
    assert pl["coupled_states"] == 11_375


def test_resource_profiler_context_manager() -> None:
    """Verify ResourceProfiler starts, samples RSS/MPS, and returns ADR-004 dictionary."""
    with ResourceProfiler(sample_interval_ms=20) as prof:
        _ = [i * i for i in range(1000)]
    metrics = prof.stop()
    assert metrics["peak_rss_bytes"] > 0
    assert metrics["peak_memory_gb"] >= 0.0
    assert metrics["memory_metric"] == "process_rss_sampled_100ms"
    assert metrics["sample_count"] >= 1


def test_xlam_dataset_loader_and_categories(tmp_path: Path) -> None:
    """Verify XLAMExample properties, categories, and JSON loader."""
    tc = XLAMToolCall(name="get_weather", arguments={"city": "Colombo"})
    td = XLAMToolDefinition(name="get_weather", description="Fetch weather")
    assert tc.name == "get_weather"
    assert td.name == "get_weather"

    ex_none = parse_xlam_record({"id": 1, "query": "hi", "tools": "[]", "answers": "[]"})
    assert ex_none.category == "no_tool_needed"
    assert ex_none.all_referenced_tool_names == set()

    ex_single = parse_xlam_record(
        {
            "id": 2,
            "query": "weather",
            "tools": json.dumps([{"name": "w"}]),
            "answers": json.dumps([{"name": "w", "arguments": {}}]),
        }
    )
    assert ex_single.category == "obvious_single_tool"
    assert ex_single.all_referenced_tool_names == {"w"}

    ex_multi = parse_xlam_record(
        {
            "id": 3,
            "query": "weather",
            "tools": [{"name": "w1"}, {"name": "w2"}],
            "answers": [{"name": "w1"}],
        }
    )
    assert ex_multi.category == "multi_plausible_tools"

    ex_complex = XLAMExample(id="4", query="q", tools=[], answers=[{"name": "hidden_tool"}])
    assert ex_complex.category == "complex_tool_invocation"

    ex_bad_json = parse_xlam_record({"query": "bad", "tools": "{bad", "answers": "{bad"}, index=9)
    assert ex_bad_json.id == "9"
    assert ex_bad_json.tools == []

    json_file = tmp_path / "xlam_sample.json"
    json_file.write_text(json.dumps([{"id": "10", "query": "test", "tools": [], "answers": []}]), encoding="utf-8")
    loaded = load_xlam_file(json_file)
    assert len(loaded) == 1
    assert loaded[0].id == "10"

    bad_file = tmp_path / "bad.json"
    bad_file.write_text(json.dumps({"not": "a list"}), encoding="utf-8")
    with pytest.raises(ValueError, match="Expected list"):
        load_xlam_file(bad_file)

    with pytest.raises(FileNotFoundError):
        load_xlam_file(tmp_path / "missing.json")


def test_random_search_sampler() -> None:
    """Verify RandomSearchSampler estimates parameters and samples batches within tolerance."""
    _, gram = inspect_to_grammar("tiiuae/Falcon-H1-1.5B-Instruct")
    sampler = RandomSearchSampler(gram, seed=123)
    unconstrained = sampler.sample_candidate("t0")
    est = sampler.estimate_trainable_parameters(unconstrained.group_settings)
    assert est > 0

    batch = sampler.sample_batch(count=4, prefix="rs", target_param_count=est, param_tolerance=0.25)
    assert len(batch) == 4
    assert batch[0].trial_id == "rs_001"


def test_trial_store_list_and_replay(tmp_path: Path) -> None:
    """Verify TrialStore save, get, list_trials, replay, and helper functions."""
    assert get_file_sha256(tmp_path / "missing.txt") == "UNKNOWN"
    assert get_git_commit(tmp_path / "not_a_git_repo") == "UNKNOWN"
    env_meta = get_environment_metadata()
    assert "torch" in env_meta["library_versions"]

    store = TrialStore(runs_dir=tmp_path / "trials")
    cfg = TrialConfig(
        trial_id="t_replay_01",
        model_id="google/gemma-4-E2B-it",
        group_settings={"ffn": GroupTrialSetting(enabled=True, rank=8, alpha=16)},
    )
    metrics = TrialMetrics(
        peak_memory_gb=4.5,
        peak_rss_gb=4.5,
        mps_driver_allocated_gb=1.5,
        train_seconds=10.0,
        latency_per_step_sec=0.5,
        trainable_params=1000,
        total_params=100000,
        trainable_ratio=0.01,
        adapter_size_mb=4.0,
        initial_train_loss=2.0,
        final_train_loss=1.0,
    )
    env = TrialResultEnvelope(
        trial_id="t_replay_01",
        git_commit="abc",
        dataset_manifest_sha256="def",
        hardware="M1 Max",
        library_versions={"torch": "2.14.0"},
        model_id="google/gemma-4-E2B-it",
        backend="peft_mps",
        trial_config=cfg.model_dump(mode="json"),
        metrics=metrics,
    )
    store.save(env)
    # Write a corrupt trial.json in another folder to test list_trials resilience
    corrupt_dir = tmp_path / "trials" / "corrupt"
    corrupt_dir.mkdir(parents=True)
    (corrupt_dir / "trial.json").write_text("{corrupt", encoding="utf-8")

    listed = store.list_trials()
    assert len(listed) == 1
    assert listed[0].trial_id == "t_replay_01"

    with pytest.raises(FileNotFoundError):
        store.get("nonexistent_trial")

    class DummyRunner:
        def run_trial(self, config: Any, trial_id: str = "") -> TrialResultEnvelope:
            return env

    replayed = store.replay("t_replay_01", backend_runner=DummyRunner())
    assert replayed.trial_id == "t_replay_01"


def test_cli_five_commands_registered() -> None:
    """Verify the CLI exposes the exact five commands: inspect, probe, search, evaluate, run."""
    res = runner.invoke(app, ["--help"])
    assert res.exit_code == 0, res.output
    for cmd in ("inspect", "probe", "search", "evaluate", "run"):
        assert cmd in res.output


def test_bootstrap_exact_normal_quantiles() -> None:
    """Verify _norm_ppf returns exact standard normal quantiles at 95% two-sided tails."""
    from aloe.evaluation.bootstrap import _norm_ppf

    assert abs(_norm_ppf(0.025) - (-1.95996398)) < 1e-6
    assert abs(_norm_ppf(0.975) - 1.95996398) < 1e-6
    assert _norm_ppf(0.5) == 0.0

