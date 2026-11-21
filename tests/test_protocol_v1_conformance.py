"""Comprehensive Protocol-v1 Conformance, Adversarial Evaluator, Crash-Resume & Offline Audit Tests."""
from __future__ import annotations

import json
from pathlib import Path
import subprocess
import sys
import pytest
from typer.testing import CliRunner

from aloe.cli.main import app
from aloe.datasets.xlam import XLAMExample
from aloe.evaluation.bootstrap import holm_bonferroni, mcnemar_exact, paired_bootstrap
from aloe.evaluation.prediction_store import (
    PredictionRecord,
    PredictionStore,
    build_paired_outcome_matrix,
)
from aloe.evaluation.resources import enforce_hardware_contract
from aloe.evaluation.tool_calling import ASTToolCallEvaluator
from aloe.execution.backends.peft_mps import tokenize_xlam_sft_example
from aloe.execution.campaign_runner import CampaignRunner
from aloe.execution.trial_config import (
    InvalidConfigError,
    TrialConfig,
    solve_borobia_ablation_configs,
)
from aloe.optimization.random_search import RandomSearchSampler
from aloe.pipeline import inspect_to_grammar
from aloe.tracking.trial_store import TrialMetrics, TrialResultEnvelope

runner = CliRunner()


def test_adversarial_strict_tool_calling_matrix() -> None:
    """Verify every adversarial tool-calling edge case identified in Quad-Agent review."""
    ev = ASTToolCallEvaluator(strict=True)
    schema = [
        {
            "name": "get_weather",
            "parameters": {
                "city": {"type": "str"},
                "days": {"type": "int"},
                "metric": {"type": "bool, optional"},
            },
        },
        {
            "name": "get_time",
            "parameters": {"tz": {"type": "str"}},
        },
    ]
    gt_single = [{"name": "get_weather", "arguments": {"city": "London", "days": 3}}]

    # 1. Underscore & case sensitivity in tool names
    assert ev.evaluate_single('[{"name":"getweather","arguments":{"city":"London","days":3}}]', gt_single).exact_match is False
    assert ev.evaluate_single('[{"name":"GET_WEATHER","arguments":{"city":"London","days":3}}]', gt_single).exact_match is False

    # 2. String vs int ("001" vs 1) and bool vs int (True vs 1)
    assert ev.evaluate_single('[{"name":"get_weather","arguments":{"city":"London","days":"3"}}]', gt_single).exact_match is False
    assert ev.evaluate_single('[{"name":"get_weather","arguments":{"city":"London","days":true}}]', [{"name": "get_weather", "arguments": {"city": "London", "days": 1}}]).exact_match is False

    # 3. Integer-equivalent float (3.0 == 3) passes when schema isn't enforcing int, fails when schema requires integer
    assert ev.evaluate_single('[{"name":"get_weather","arguments":{"city":"London","days":3.0}}]', gt_single).exact_match is True
    assert ev.evaluate_single('[{"name":"get_weather","arguments":{"city":"London","days":3.0}}]', gt_single, available_tools=schema).exact_match is False
    assert ev.evaluate_single('[{"name":"get_weather","arguments":{"city":"London","days":3}}]', gt_single, available_tools=schema).exact_match is True

    # 4. Extra unrequested second call MUST fail exact_match
    extra_call = '[{"name":"get_weather","arguments":{"city":"London","days":3}},{"name":"get_time","arguments":{"tz":"UTC"}}]'
    assert ev.evaluate_single(extra_call, gt_single).exact_match is False

    # 5. Missing required second call MUST fail exact_match
    gt_multi = [
        {"name": "get_weather", "arguments": {"city": "London", "days": 3}},
        {"name": "get_time", "arguments": {"tz": "UTC"}},
    ]
    assert ev.evaluate_single('[{"name":"get_weather","arguments":{"city":"London","days":3}}]', gt_multi).exact_match is False
    assert ev.evaluate_single(extra_call, gt_multi, available_tools=schema).exact_match is True

    # 6. Valid first call + malformed second call MUST fail parse_success
    malformed_second = '[{"name":"get_weather","arguments":{"city":"London","days":3}}, "not_an_object"]'
    res_mal = ev.evaluate_single(malformed_second, gt_single)
    assert res_mal.parse_success is False
    assert res_mal.exact_match is False

    # 7. Duplicate JSON keys & NaN rejected
    dup_key = '[{"name":"get_weather","arguments":{"city":"London","city":"Paris","days":3}}]'
    assert ev.evaluate_single(dup_key, gt_single).parse_success is False
    nan_val = '[{"name":"get_weather","arguments":{"city":"London","days":NaN}}]'
    assert ev.evaluate_single(nan_val, gt_single).parse_success is False

    # 8. Abstention: [] vs [] -> parse_success=True, exact_match=True; [] vs non-empty -> exact_match=False
    res_abs_ok = ev.evaluate_single("[]", [])
    assert res_abs_ok.parse_success is True
    assert res_abs_ok.exact_match is True
    res_abs_miss = ev.evaluate_single("[]", gt_single)
    assert res_abs_miss.parse_success is True
    assert res_abs_miss.exact_match is False


def test_holm_bonferroni_and_mcnemar_and_seed_consistency() -> None:
    """Verify exact step-down Holm-Bonferroni math, McNemar test, and per-query seed validation."""
    holm = holm_bonferroni({"b0": 0.008, "b1": 0.020, "b2": 0.045}, alpha=0.05)
    # Sorted: b0 (3 * 0.008 = 0.024), b1 (max(0.024, 2 * 0.020) = 0.040), b2 (max(0.040, 1 * 0.045) = 0.045)
    assert pytest.approx(holm["b0"].adjusted_p, abs=1e-6) == 0.024
    assert pytest.approx(holm["b1"].adjusted_p, abs=1e-6) == 0.040
    assert pytest.approx(holm["b2"].adjusted_p, abs=1e-6) == 0.045
    assert holm["b0"].reject_null is True

    mcn = mcnemar_exact(
        {"q1": [1], "q2": [1], "q3": [1], "q4": [0]},
        {"q1": [0], "q2": [0], "q3": [0], "q4": [0]},
    )
    assert mcn.discordant_pairs == 3
    assert pytest.approx(mcn.exact_p_value, abs=1e-6) == 0.25

    # Inconsistent seed count on q2 must raise ValueError
    with pytest.raises(ValueError, match="Inconsistent seed count"):
        paired_bootstrap({"q1": [1, 1], "q2": [1]}, {"q1": [0, 0], "q2": [0]})


def test_tokenize_preserves_eos_when_pad_equals_eos() -> None:
    """Verify EOS token is NOT masked with -100 when pad_token_id == eos_token_id."""

    class DummyTokenizer:
        bos_token_id = 1
        eos_token_id = 2
        pad_token_id = 2  # pad == eos!
        eos_token = "<EOS>"
        chat_template = None

        def encode(self, text: str, add_special_tokens: bool = False) -> list[int]:
            del add_special_tokens
            # Map "<EOS>" at the end to token 2, all other chars to >10
            has_eos = text.endswith("<EOS>")
            core = text[:-5] if has_eos else text
            ids = [10 + (ord(c) % 50) for c in core]
            if has_eos:
                ids.append(self.eos_token_id)
            return ids

    tok = DummyTokenizer()
    ex = XLAMExample(
        id="t1",
        query="weather?",
        tools=[{"name": "w", "parameters": {}}],
        answers=[{"name": "w", "arguments": {}}],
    )
    out = tokenize_xlam_sft_example(ex, tok, max_seq_len=512, reject_overlength=True, pad_to_max=True)
    # Last non-pad token is EOS (2) and its label MUST be 2 (supervised), while trailing pads are -100!
    supervised = out["labels"][out["labels"] != -100].tolist()
    assert len(supervised) > 1
    assert supervised[-1] == tok.eos_token_id

    # Overlength rejection check
    with pytest.raises(ValueError, match="exceeds max_seq_len"):
        tokenize_xlam_sft_example(ex, tok, max_seq_len=20, reject_overlength=True)


def test_prediction_store_torn_tail_recovery_and_paired_matrix(tmp_path: Path) -> None:
    """Verify PredictionStore cleans torn tail lines on crash recovery and builds multi-seed paired matrices."""
    p_s1 = PredictionStore(tmp_path / "arm_a_s42.jsonl")
    p_s2 = PredictionStore(tmp_path / "arm_a_s1337.jsonl")

    for sid, match_s1, match_s2 in [("10", True, True), ("11", False, True)]:
        p_s1.append(
            PredictionRecord(
                sample_id=sid,
                split="validation",
                model_id="m",
                model_revision="r",
                arm="aloe",
                trial_id="t1",
                seed=42,
                prompt_sha256="abc",
                raw_output="[]",
                ground_truth=[],
                parse_success=True,
                tool_match=match_s1,
                arg_names_match=match_s1,
                arg_values_match=match_s1,
                exact_match=match_s1,
            )
        )
        p_s2.append(
            PredictionRecord(
                sample_id=sid,
                split="validation",
                model_id="m",
                model_revision="r",
                arm="aloe",
                trial_id="t2",
                seed=1337,
                prompt_sha256="abc",
                raw_output="[]",
                ground_truth=[],
                parse_success=True,
                tool_match=match_s2,
                arg_names_match=match_s2,
                arg_values_match=match_s2,
                exact_match=match_s2,
            )
        )

    # Simulate power-loss torn line at end of p_s1
    with open(p_s1.path, "a", encoding="utf-8") as f:
        f.write('{"sample_id": "12", "truncated_json...')

    done_ids = p_s1.completed_sample_ids()
    assert done_ids == {"10", "11"}
    assert len(p_s1.load_records(strict=True)) == 2

    mat = build_paired_outcome_matrix({42: p_s1, 1337: p_s2})
    assert mat["10"] == [1, 1]
    assert mat["11"] == [0, 1]


def test_borobia_exact_integer_ranks_and_strict_random_sampler() -> None:
    """Verify exact integer-rank solver for single-operator controls and strict RandomSearchSampler rejection."""
    desc_g, gram_g = inspect_to_grammar("google/gemma-4-E2B-it")
    solved_g = solve_borobia_ablation_configs(desc_g, gram_g, baseline_params=12_079_104)
    assert set(solved_g.keys()) == {"borobia_attn", "borobia_ffn"}  # no SSM on Gemma
    assert solved_g["borobia_attn"]["solved_rank"] == 36
    assert solved_g["borobia_attn"]["parity"]["within_budget"] is True
    assert solved_g["borobia_ffn"]["solved_rank"] == 10
    assert solved_g["borobia_ffn"]["parity"]["within_budget"] is True

    desc_l, gram_l = inspect_to_grammar("LiquidAI/LFM3-1.7B")
    solved_l = solve_borobia_ablation_configs(desc_l, gram_l, baseline_params=8_052_736)
    assert set(solved_l.keys()) == {"borobia_attn", "borobia_ssm", "borobia_ffn"}
    assert solved_l["borobia_attn"]["solved_rank"] == 55
    assert solved_l["borobia_ssm"]["solved_rank"] == 50

    # RandomSearchSampler with impossible target budget MUST raise InvalidConfigError, never return nearest!
    sampler = RandomSearchSampler(gram_g, arch=desc_g, seed=42)
    with pytest.raises(InvalidConfigError, match="No feasible"):
        sampler.sample_candidate("impossible", target_param_count=999_999_999, param_tolerance=0.01)


def test_hardware_contract_fails_closed_on_hybrid_mps_or_wrong_gpu() -> None:
    """Verify enforce_hardware_contract rejects hybrid official campaigns on MPS and GPU SKU mismatches."""
    with pytest.raises(RuntimeError, match="ADR-012"):
        enforce_hardware_contract(
            "tiiuae/Falcon-H1-1.5B-Instruct",
            "mps",
            official_campaign=True,
        )
    with pytest.raises(RuntimeError, match="pinned_gpu"):
        enforce_hardware_contract(
            "LiquidAI/LFM3-1.7B",
            "cpu",
            pinned_gpu="A100",
        )


def test_campaign_runner_crash_resume_and_stdlib_offline_verifiers(tmp_path: Path) -> None:
    """End-to-end test: Sobol(5) + discrete qLogNEHVI(2) campaign -> simulated crash at trial 3 -> resume -> stdlib verify."""
    desc, grammar = inspect_to_grammar("google/gemma-4-E2B-it")
    campaign_dir = tmp_path / "gemma_aloe"
    mirror_dir = tmp_path / "drive_mirror"

    call_count = {"n": 0}

    def _mock_executor(cfg: TrialConfig, trials_dir: Path) -> TrialResultEnvelope:
        call_count["n"] += 1
        # Simulate unexpected worker crash on the 4th execution call
        if call_count["n"] == 4:
            raise RuntimeError("Simulated Colab session preemption!")

        tdir = trials_dir / cfg.trial_id
        tdir.mkdir(parents=True, exist_ok=True)
        pstore = PredictionStore(tdir / "predictions_validation.jsonl")
        done_ids = pstore.completed_sample_ids()
        for i in range(4):
            sid = f"val_{i}"
            if sid in done_ids:
                continue
            exact = (i < 3) if cfg.group_settings["global_attention"].rank >= 8 else (i < 2)
            pstore.append(
                PredictionRecord(
                    sample_id=sid,
                    split="validation",
                    model_id=cfg.model_id,
                    model_revision="3e22461f65e89153144f8adb70e3b8c2cc9845a7",
                    arm=cfg.arm,
                    trial_id=cfg.trial_id,
                    seed=cfg.seed,
                    prompt_sha256="sha",
                    raw_output='[{"name":"f","arguments":{}}]',
                    ground_truth=[{"name": "f", "arguments": {}}],
                    parse_success=True,
                    tool_match=exact,
                    arg_names_match=exact,
                    arg_values_match=exact,
                    exact_match=exact,
                )
            )
        recs = pstore.load_records(strict=True)
        summ = PredictionStore.summarize(recs)
        cand_params = cfg.compute_trainable_params(desc, grammar)
        metrics = TrialMetrics(
            peak_memory_gb=11.2,
            peak_rss_gb=11.2,
            search_memory_mb=11468.8,
            train_seconds=1.5,
            latency_per_step_sec=0.1,
            trainable_params=cand_params,
            total_params=2_000_000_000,
            trainable_ratio=0.006,
            adapter_size_mb=12.0,
            eval_samples=summ.total_samples,
            predictions_jsonl_path=str(pstore.path),
            predictions_sha256=pstore.sha256(),
            tool_accuracy=summ.exact_match_accuracy,
            ast_parse_success_rate=summ.parse_success_rate,
            ast_exact_match_accuracy=summ.exact_match_accuracy,
            is_budget_matched_5pct=True,
            fresh_process_reload=True,
        )
        return TrialResultEnvelope(
            trial_id=cfg.trial_id,
            arm=cfg.arm,
            git_commit="clean_sha",
            dataset_manifest_sha256="manifest_sha",
            model_revision="3e22461f65e89153144f8adb70e3b8c2cc9845a7",
            hardware="Test Hardware",
            library_versions={"torch": "2.4"},
            model_id=cfg.model_id,
            backend="peft_cpu",
            trial_config=cfg.model_dump(mode="json"),
            metrics=metrics,
            status="PASS",
        )

    # First run crashes on attempt 1 of trial 3; max_attempts_per_trial=1 forces crash out to caller!
    runner1 = CampaignRunner(
        arch=desc,
        grammar=grammar,
        campaign_dir=campaign_dir,
        arm="aloe",
        target_trials=7,
        official_campaign=False,
        use_subprocess_worker=False,
        mirror_dir=mirror_dir,
        trial_executor_override=_mock_executor,
    )
    with pytest.raises(RuntimeError, match="Simulated Colab session preemption"):
        runner1.run_campaign(max_attempts_per_trial=1)

    # Reset failed entry to RUNNING (simulating mid-trial kill -9 before status update) and resume!
    ledger_state = runner1.ledger.load()
    assert sum(1 for e in ledger_state.entries if e.status == "DONE") == 3
    ledger_state.entries[-1].status = "RUNNING"
    ledger_state.entries[-1].attempts = 0
    runner1.ledger.save(ledger_state)

    runner2 = CampaignRunner(
        arch=desc,
        grammar=grammar,
        campaign_dir=campaign_dir,
        arm="aloe",
        target_trials=7,
        official_campaign=False,
        use_subprocess_worker=False,
        mirror_dir=mirror_dir,
        trial_executor_override=_mock_executor,
    )
    summary = runner2.run_campaign(max_attempts_per_trial=2)
    assert summary["completed_trials"] == 7
    assert (mirror_dir / "campaign_summary.json").is_file()

    # Run BOTH zero-dependency stdlib audit scripts (ChatGPT-container simulation!)
    repo_root = Path(__file__).resolve().parent.parent
    ver_proc = subprocess.run(
        [
            sys.executable,
            str(repo_root / "scripts/verify_campaign.py"),
            "--manifest",
            str(repo_root / "data/manifests/xlam_ipd_v1.json"),
            "--campaign-dir",
            str(campaign_dir),
            "--allow-smoke-budget",
        ],
        capture_output=True,
        text=True,
        check=True,
    )
    ver_json = json.loads(ver_proc.stdout)
    assert ver_json["manifest_verification"]["split_counts"] == {
        "train": 1000,
        "validation": 200,
        "test": 400,
    }
    assert ver_json["campaigns"][0]["verified_done_trials"] == 7
    assert ver_json["campaigns"][0]["unique_signatures"] == 7

    recomp_proc = subprocess.run(
        [
            sys.executable,
            str(repo_root / "scripts/recompute_metrics.py"),
            str(campaign_dir),
            "--check-trial-json",
        ],
        capture_output=True,
        text=True,
        check=True,
    )
    recomp_json = json.loads(recomp_proc.stdout)
    assert recomp_json["verified_files"] == 7


def _create_tiny_llama_and_dataset(tmp_path: Path) -> tuple[Path, Path, Path]:
    """Create a tiny offline LlamaForCausalLM + ByteLevel BPE tokenizer + mini xLAM manifest."""
    from tokenizers import Tokenizer
    from tokenizers.decoders import ByteLevel as ByteLevelDecoder
    from tokenizers.models import BPE
    from tokenizers.pre_tokenizers import ByteLevel
    from transformers import LlamaConfig, LlamaForCausalLM, PreTrainedTokenizerFast

    model_dir = tmp_path / "tiny_llama"
    model_dir.mkdir(parents=True, exist_ok=True)

    alphabet = ByteLevel.alphabet()
    vocab = {"<pad>": 0, "<bos>": 1, "<eos>": 2}
    for ch in alphabet:
        if ch not in vocab:
            vocab[ch] = len(vocab)

    tok_obj = Tokenizer(BPE(vocab=vocab, merges=[]))
    tok_obj.pre_tokenizer = ByteLevel(add_prefix_space=False)
    tok_obj.decoder = ByteLevelDecoder()
    fast_tok = PreTrainedTokenizerFast(
        tokenizer_object=tok_obj,
        bos_token="<bos>",
        eos_token="<eos>",
        pad_token="<eos>",  # Intentionally test pad_token == eos_token EOS-preservation!
    )
    fast_tok.save_pretrained(model_dir)

    cfg = LlamaConfig(
        vocab_size=len(fast_tok),
        hidden_size=32,
        intermediate_size=64,
        num_hidden_layers=2,
        num_attention_heads=4,
        num_key_value_heads=2,
        max_position_embeddings=512,
        bos_token_id=1,
        eos_token_id=2,
        pad_token_id=2,
    )
    model = LlamaForCausalLM(cfg)
    model.save_pretrained(model_dir)

    raw_items = []
    for idx in range(6):
        fn_name = f"tool_t{idx}" if idx < 4 else f"tool_v{idx}"
        raw_items.append(
            {
                "id": idx,
                "query": f"Invoke {fn_name} with x={idx}",
                "tools": json.dumps(
                    [
                        {
                            "name": fn_name,
                            "description": "Test tool",
                            "parameters": {"x": {"type": "int", "default": 0}},
                        }
                    ]
                ),
                "answers": json.dumps([{"name": fn_name, "arguments": {"x": idx}}]),
            }
        )
    raw_path = tmp_path / "mini_xlam.json"
    raw_path.write_text(json.dumps(raw_items), encoding="utf-8")

    manifest_path = tmp_path / "mini_manifest.json"
    manifest_path.write_text(
        json.dumps(
            {
                "dataset": "mini_xlam",
                "version": "ipd_v1_mini",
                "splits": {
                    "train": {"count": 4, "ids": [0, 1, 2, 3]},
                    "validation": {"count": 2, "ids": [4, 5]},
                    "test": {"count": 0, "ids": []},
                },
            }
        ),
        encoding="utf-8",
    )
    return model_dir, manifest_path, raw_path


def test_real_cpu_tiny_llama_lora_adalora_checkpoint_resume_and_worker(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Exercise real PeftMPSBackend.run_trial() on CPU: LoRA, EOS-label masking, step-checkpoint resume, AdaLoRA, rescore, and worker_entry."""
    from aloe.datasets.xlam import load_xlam_file
    from aloe.execution import worker_entry
    from aloe.execution.backends.peft_mps import PeftMPSBackend, tokenize_xlam_sft_example
    from aloe.execution.trial_config import TrainingParameters
    from transformers import AutoTokenizer

    model_dir, manifest_path, raw_path = _create_tiny_llama_and_dataset(tmp_path)
    trials_dir = tmp_path / "trials"
    mirror_dir = tmp_path / "mirror"

    # 1. Verify EOS token is NOT masked (-100) even though pad_token_id == eos_token_id!
    tok = AutoTokenizer.from_pretrained(model_dir)
    ex0 = load_xlam_file(raw_path)[0]
    tok_item = tokenize_xlam_sft_example(ex0, tok, max_seq_len=512, reject_overlength=True)
    assert tok_item["labels"][-1].item() == tok.eos_token_id
    assert (tok_item["labels"] != -100).sum().item() >= 5

    backend = PeftMPSBackend()

    # 2. Run 2 steps first so we can capture a real mid-trial checkpoint and test deterministic step resume to 4 steps!
    cfg_step2 = TrialConfig(
        trial_id="cpu_lora_resume",
        arm="aloe",
        model_id=str(model_dir),
        seed=42,
        dataset_manifest_path=str(manifest_path),
        dataset_raw_path=str(raw_path),
        explicit_target_modules=["q_proj", "v_proj"],
        explicit_rank=4,
        explicit_alpha=8,
        training=TrainingParameters(
            learning_rate=2e-4,
            batch_size=2,
            gradient_accumulation_steps=2,
            max_steps=2,
            warmup_steps=1,
            max_seq_len=512,
            max_new_tokens=16,
            checkpoint_every_steps=1,
            precision="fp32",
        ),
    )
    # Save a real step-2 checkpoint and then resume from step 2 -> step 4!
    env2 = backend.run_trial(
        config=cfg_step2,
        runs_trials_dir=trials_dir,
        enforce_budget_parity=False,
        eval_split="validation",
        official_campaign=False,
        skip_subprocess_reload=True,
        mirror_dir=mirror_dir,
        device_override="cpu",
        keep_step_checkpoint=True,
    )
    assert env2.status == "PASS"
    assert env2.metrics is not None
    assert env2.metrics.eval_samples == 2
    assert env2.metrics.examples_seen == 8  # 2 steps * grad_accum(2) * batch_size(2)
    assert (mirror_dir / "cpu_lora_resume" / "predictions_validation.jsonl").is_file()
    assert (trials_dir / "cpu_lora_resume" / "checkpoint_state.pt").is_file()

    # Test PredictionStore.rescore() offline without model inference
    pstore = PredictionStore(Path(env2.metrics.predictions_jsonl_path))
    rescored = pstore.rescore()
    assert rescored.total_samples == 2

    # Resume the exact same trial_id ("cpu_lora_resume") from step 2 -> max_steps=4!
    cfg_step4 = cfg_step2.model_copy(
        update={
            "training": cfg_step2.training.model_copy(update={"max_steps": 4, "checkpoint_every_steps": 2}),
        }
    )
    env4 = backend.run_trial(
        config=cfg_step4,
        runs_trials_dir=trials_dir,
        enforce_budget_parity=False,
        eval_split="validation",
        official_campaign=False,
        skip_subprocess_reload=True,
        device_override="cpu",
        resume=True,
    )
    assert env4.status == "PASS"
    assert env4.metrics is not None and env4.metrics.optimizer_steps_completed == 4
    assert env4.metrics.examples_seen == 16  # 8 resumed + 8 new!
    assert not (trials_dir / "cpu_lora_resume" / "checkpoint_state.pt").exists()

    # 3. Test real PEFT AdaLoRA (`update_and_allocate`) on CPU!
    cfg_adalora = TrialConfig(
        trial_id="cpu_adalora",
        arm="b2",
        peft_method="adalora",
        adalora_init_r=6,
        adalora_target_r=4,
        model_id=str(model_dir),
        seed=42,
        dataset_manifest_path=str(manifest_path),
        dataset_raw_path=str(raw_path),
        explicit_target_modules=["q_proj", "v_proj"],
        training=TrainingParameters(
            learning_rate=2e-4,
            batch_size=2,
            gradient_accumulation_steps=1,
            max_steps=6,
            warmup_steps=1,
            max_seq_len=512,
            max_new_tokens=8,
            checkpoint_every_steps=3,
            precision="fp32",
        ),
    )
    env_ada = backend.run_trial(
        config=cfg_adalora,
        runs_trials_dir=trials_dir,
        enforce_budget_parity=False,
        eval_split="validation",
        official_campaign=False,
        skip_subprocess_reload=True,
        device_override="cpu",
    )
    assert env_ada.status == "PASS"
    assert env_ada.metrics is not None
    assert env_ada.metrics.adalora_init_params is not None

    # 4. Exercise isolated worker_entry.main() CLI entry point!
    worker_cfg_path = tmp_path / "worker_cfg.json"
    cfg_worker = cfg_step2.model_copy(update={"trial_id": "cpu_worker_cli"})
    cfg_worker.to_file(worker_cfg_path)
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "worker_entry",
            "--config",
            str(worker_cfg_path),
            "--trials-dir",
            str(trials_dir),
            "--eval-split",
            "validation",
            "--device",
            "cpu",
            "--skip-subprocess-reload",
        ],
    )
    rc = worker_entry.main()
    assert rc == 0
    assert (trials_dir / "cpu_worker_cli" / "trial.json").is_file()


def test_campaign_runner_all_baseline_arms_and_cli_evaluate(tmp_path: Path) -> None:
    """Exercise CampaignRunner schedules for b0, b1, b2, borobia_ffn and CLI `aloe evaluate` + `aloe search`."""
    from typer.testing import CliRunner
    from aloe.cli.main import app
    from aloe.tracking.trial_store import TrialStore

    desc, grammar = inspect_to_grammar("google/gemma-4-E2B-it")
    eval_root = tmp_path / "multi_arm_eval"

    def _fast_exec(cfg: TrialConfig, trials_dir: Path) -> TrialResultEnvelope:
        tdir = trials_dir / cfg.trial_id
        tdir.mkdir(parents=True, exist_ok=True)
        pstore = PredictionStore(tdir / "predictions_validation.jsonl")
        done = pstore.completed_sample_ids()
        for i in range(6):
            sid = f"q_{i}"
            if sid in done:
                continue
            ok = (i < 5) if cfg.arm in ("aloe", "borobia_ffn") else (i < 3)
            pstore.append(
                PredictionRecord(
                    sample_id=sid,
                    split="validation",
                    model_id=cfg.model_id,
                    model_revision="3e22461f65e89153144f8adb70e3b8c2cc9845a7",
                    arm=cfg.arm,
                    trial_id=cfg.trial_id,
                    seed=cfg.seed,
                    prompt_sha256="sha",
                    raw_output="[]",
                    ground_truth=[],
                    parse_success=True,
                    tool_match=ok,
                    arg_names_match=ok,
                    arg_values_match=ok,
                    exact_match=ok,
                )
            )
        recs = pstore.load_records(strict=True)
        summ = PredictionStore.summarize(recs)
        cand_params = cfg.compute_trainable_params(desc, grammar)
        metrics = TrialMetrics(
            peak_memory_gb=10.5,
            peak_rss_gb=10.5,
            search_memory_mb=10752.0,
            train_seconds=1.0,
            latency_per_step_sec=0.1,
            trainable_params=cand_params,
            total_params=2_000_000_000,
            trainable_ratio=0.006,
            adapter_size_mb=12.0,
            eval_samples=summ.total_samples,
            predictions_jsonl_path=str(pstore.path),
            predictions_sha256=pstore.sha256(),
            tool_accuracy=summ.exact_match_accuracy,
            ast_parse_success_rate=1.0,
            ast_exact_match_accuracy=summ.exact_match_accuracy,
            is_budget_matched_5pct=True,
            fresh_process_reload=True,
        )
        env = TrialResultEnvelope(
            trial_id=cfg.trial_id,
            arm=cfg.arm,
            git_commit="clean_sha",
            dataset_manifest_sha256="m_sha",
            model_revision="3e22461f65e89153144f8adb70e3b8c2cc9845a7",
            hardware="Test",
            library_versions={"torch": "2.4"},
            model_id=cfg.model_id,
            backend="peft_cpu",
            trial_config=cfg.model_dump(mode="json"),
            metrics=metrics,
            status="PASS",
        )
        TrialStore(trials_dir).save(env)
        return env

    for arm_name in ("b0", "b1", "b2", "borobia_ffn"):
        runner = CampaignRunner(
            arch=desc,
            grammar=grammar,
            campaign_dir=eval_root / arm_name,
            arm=arm_name,
            target_trials=2,
            official_campaign=False,
            use_subprocess_worker=False,
            trial_executor_override=_fast_exec,
        )
        res = runner.run_campaign()
        expected_n = 1 if arm_name.startswith("borobia_") else 2
        assert res["completed_trials"] == expected_n

    cli = CliRunner()
    # Sealed test split MUST refuse without --final
    bad_test = cli.invoke(app, ["evaluate", "--run-dir", str(eval_root), "--split", "test"])
    assert bad_test.exit_code != 0

    # Validation split runs paired bootstrap + Holm-Bonferroni + McNemar cleanly
    ok_val = cli.invoke(
        app,
        ["evaluate", "--run-dir", str(eval_root), "--split", "validation", "--bootstrap", "200"],
    )
    assert ok_val.exit_code == 0, ok_val.output
    assert "p_holm=" in ok_val.output
    assert "mcnemar_p=" in ok_val.output

    # Plan-only search outputs honest Sobol initial design without fake objective numbers
    plan_out = tmp_path / "sobol_plan"
    plan_res = cli.invoke(
        app,
        ["search", "--model", "google/gemma-4-E2B-it", "--trials", "3", "--out", str(plan_out)],
    )
    assert plan_res.exit_code == 0, plan_res.output
    assert (plan_out / "search_summary.json").is_file()


def test_xlam_ground_truth_oracle_self_consistency_and_fail_closed_schemas() -> None:
    """Verify 100% oracle exact_match across all 1,600 xLAM manifest items (when raw data present) + strict schema edge cases."""
    from aloe.datasets.xlam import load_xlam_file

    ev = ASTToolCallEvaluator(strict=True)

    # 1. Fail-closed unknown schema type + additionalProperties=false + malformed ground truth + unordered lenient
    strict_tools = [
        {
            "name": "strict_fn",
            "parameters": {
                "type": "object",
                "properties": {"x": {"type": "integer"}},
                "required": ["x"],
                "additionalProperties": False,
            },
        },
        {
            "name": "weird_type_fn",
            "parameters": {
                "type": "object",
                "properties": {"x": {"type": "custom_quantum_tensor"}},
                "required": ["x"],
            },
        },
    ]
    res_extra = ev.evaluate_single(
        '[{"name":"strict_fn","arguments":{"x":1,"undeclared":2}}]',
        [{"name": "strict_fn", "arguments": {"x": 1}}],
        available_tools=strict_tools,
    )
    assert res_extra.schema_valid is False
    assert res_extra.exact_match is False

    res_unknown_type = ev.evaluate_single(
        '[{"name":"weird_type_fn","arguments":{"x":"abc"}}]',
        [{"name": "weird_type_fn", "arguments": {"x": "abc"}}],
        available_tools=strict_tools,
    )
    assert res_unknown_type.schema_valid is False

    with pytest.raises(ValueError, match="missing valid tool 'name'"):
        ev.evaluate_single("[]", [{"arguments": {"x": 1}}])
    with pytest.raises(ValueError, match="Malformed ground truth arguments"):
        ev.evaluate_single("[]", [{"name": "strict_fn", "arguments": "{bad_json"}])
    with pytest.raises(ValueError, match="sample_ids length"):
        ev.evaluate_batch(["[]", "[]"], [[], []], sample_ids=["only_one"])

    # Unordered multi-call lenient >= strict invariant
    res_unord = ev.evaluate_single(
        '[{"name":"b_fn","arguments":{"k":2}},{"name":"a_fn","arguments":{"k":1}}]',
        [{"name": "a_fn", "arguments": {"k": 1}}, {"name": "b_fn", "arguments": {"k": 2}}],
    )
    assert res_unord.exact_match is True
    assert res_unord.lenient_exact_match is True

    # 2. Full 1,600-item xLAM oracle self-consistency check if raw file is present locally
    repo_root = Path(__file__).resolve().parent.parent
    raw_file = repo_root / "data/raw/xlam/xlam_function_calling_60k.json"
    manifest_file = repo_root / "data/manifests/xlam_ipd_v1.json"
    if raw_file.is_file() and manifest_file.is_file():
        manifest = json.loads(manifest_file.read_text(encoding="utf-8"))
        raw_map = {str(ex.id): ex for ex in load_xlam_file(raw_file)}
        checked = 0
        for sbody in manifest["splits"].values():
            for sid in sbody["ids"]:
                ex = raw_map[str(sid)]
                oracle_str = json.dumps(ex.answers, separators=(",", ":"))
                r = ev.evaluate_single(oracle_str, ex.answers, sample_id=str(sid), available_tools=ex.tools)
                assert r.exact_match is True, f"Oracle self-consistency failed on #{sid}: {r.parse_error}"
                checked += 1
        assert checked == 1600


def test_resume_incompatibility_and_hardware_drift_guards(tmp_path: Path) -> None:
    """Verify CampaignRunner rejects GPU SKU changes, seed/manifest/training mismatches, and mid-campaign GPU drift."""
    from aloe.execution.trial_config import TrainingParameters

    desc, grammar = inspect_to_grammar("google/gemma-4-E2B-it")
    cdir = tmp_path / "resume_guard"
    gpu_holder = {"gpu": "NVIDIA A100-SXM4-40GB"}

    def _exec(cfg: TrialConfig, trials_dir: Path) -> TrialResultEnvelope:
        tdir = trials_dir / cfg.trial_id
        tdir.mkdir(parents=True, exist_ok=True)
        pstore = PredictionStore(tdir / "predictions_validation.jsonl")
        if "v0" not in pstore.completed_sample_ids():
            pstore.append(
                PredictionRecord(
                    sample_id="v0",
                    split="validation",
                    model_id=cfg.model_id,
                    model_revision="3e22461f65e89153144f8adb70e3b8c2cc9845a7",
                    arm=cfg.arm,
                    trial_id=cfg.trial_id,
                    seed=cfg.seed,
                    prompt_sha256="s",
                    raw_output="[]",
                    ground_truth=[],
                    parse_success=True,
                    tool_match=True,
                    arg_names_match=True,
                    arg_values_match=True,
                    exact_match=True,
                )
            )
        return TrialResultEnvelope(
            trial_id=cfg.trial_id,
            arm=cfg.arm,
            git_commit="sha",
            dataset_manifest_sha256="m",
            model_revision="3e22461f65e89153144f8adb70e3b8c2cc9845a7",
            hardware=gpu_holder["gpu"],
            gpu_name=gpu_holder["gpu"],
            library_versions={"torch": "2.4"},
            model_id=cfg.model_id,
            backend="peft_cuda",
            trial_config=cfg.model_dump(mode="json"),
            metrics=TrialMetrics(
                peak_memory_gb=8.0,
                peak_rss_gb=4.0,
                search_memory_mb=8192.0,
                memory_metric_name="cuda_max_memory_allocated",
                train_seconds=1.0,
                latency_per_step_sec=0.1,
                trainable_params=cfg.compute_trainable_params(desc, grammar),
                total_params=2_000_000_000,
                trainable_ratio=0.006,
                adapter_size_mb=10.0,
                eval_samples=1,
                predictions_jsonl_path=str(pstore.path),
                predictions_sha256=pstore.sha256(),
                tool_accuracy=1.0,
                ast_exact_match_accuracy=1.0,
            ),
            status="PASS",
        )

    repo_manifest = str(Path(__file__).resolve().parent.parent / "data/manifests/xlam_ipd_v1.json")
    r1 = CampaignRunner(
        arch=desc,
        grammar=grammar,
        campaign_dir=cdir,
        arm="b0",
        target_trials=2,
        manifest_path=repo_manifest,
        pinned_gpu="A100",
        official_campaign=False,
        use_subprocess_worker=False,
        trial_executor_override=_exec,
    )
    # Simulate mid-campaign SKU switch from A100-40GB on trial 0 -> L4 on trial 1!
    calls = {"n": 0}

    def _drifting_exec(cfg: TrialConfig, trials_dir: Path) -> TrialResultEnvelope:
        calls["n"] += 1
        if calls["n"] >= 2:
            gpu_holder["gpu"] = "NVIDIA L4"
        return _exec(cfg, trials_dir)

    r1.trial_executor_override = _drifting_exec
    with pytest.raises(RuntimeError, match="Mid-campaign hardware drift|Cross-arm GPU variant mismatch"):
        r1.run_campaign(max_attempts_per_trial=1)

    # Attempting to resume with pinned_gpu="L4", different search_seed, or altered max_steps MUST fail closed!
    for bad_kwargs in (
        {"pinned_gpu": "L4"},
        {"search_seed": 999},
        {"base_training": TrainingParameters(max_steps=999)},
    ):
        kw = {
            "arch": desc,
            "grammar": grammar,
            "campaign_dir": cdir,
            "arm": "b0",
            "target_trials": 2,
            "manifest_path": repo_manifest,
            "pinned_gpu": "A100",
            "official_campaign": False,
            "use_subprocess_worker": False,
            "trial_executor_override": _exec,
        }
        kw.update(bad_kwargs)
        bad_runner = CampaignRunner(**kw)  # type: ignore[arg-type]
        with pytest.raises(RuntimeError, match="incompatible frozen settings"):
            bad_runner.run_campaign()


def test_sealed_test_backend_guard_missing_ids_and_ephemeral_drive_checkpoint_restore(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Verify sealed test split rejection in run_trial/worker_entry, missing ID rejection, and full ephemeral Drive restore."""
    import shutil
    from aloe.execution import worker_entry
    from aloe.execution.backends.peft_mps import PeftMPSBackend
    from aloe.execution.trial_config import TrainingParameters

    model_dir, manifest_path, raw_path = _create_tiny_llama_and_dataset(tmp_path)
    backend = PeftMPSBackend()

    cfg = TrialConfig(
        trial_id="ephemeral_colab_trial",
        arm="aloe",
        model_id=str(model_dir),
        seed=42,
        dataset_manifest_path=str(manifest_path),
        dataset_raw_path=str(raw_path),
        explicit_target_modules=["q_proj", "v_proj"],
        explicit_rank=4,
        explicit_alpha=8,
        training=TrainingParameters(
            learning_rate=2e-4,
            batch_size=2,
            gradient_accumulation_steps=1,
            max_steps=2,
            warmup_steps=1,
            max_seq_len=512,
            max_new_tokens=8,
            checkpoint_every_steps=1,
            precision="fp32",
        ),
    )

    # 1. Direct backend & worker_entry access to eval_split="test" MUST raise PermissionError without allow_sealed_test!
    with pytest.raises(PermissionError, match="sealed"):
        backend.run_trial(
            config=cfg,
            runs_trials_dir=tmp_path / "t_sealed",
            enforce_budget_parity=False,
            eval_split="test",
            official_campaign=False,
            skip_subprocess_reload=True,
            device_override="cpu",
        )

    w_cfg = tmp_path / "w_sealed.json"
    cfg.to_file(w_cfg)
    monkeypatch.setattr(
        sys,
        "argv",
        ["worker_entry", "--config", str(w_cfg), "--trials-dir", str(tmp_path / "t_sealed"), "--eval-split", "test", "--device", "cpu"],
    )
    with pytest.raises(PermissionError, match="sealed"):
        worker_entry.main()

    # 2. Missing raw IDs in manifest MUST raise ValueError instead of silently dropping!
    bad_manifest = tmp_path / "bad_manifest.json"
    bad_manifest.write_text(
        json.dumps(
            {
                "splits": {
                    "train": {"count": 2, "ids": [0, 999999]},
                    "validation": {"count": 1, "ids": [4]},
                    "test": {"count": 0, "ids": []},
                }
            }
        ),
        encoding="utf-8",
    )
    with pytest.raises(ValueError, match="absent from raw dataset"):
        backend.run_trial(
            config=cfg.model_copy(update={"dataset_manifest_path": str(bad_manifest)}),
            runs_trials_dir=tmp_path / "t_missing",
            enforce_budget_parity=False,
            eval_split="validation",
            official_campaign=False,
            skip_subprocess_reload=True,
            device_override="cpu",
        )

    # 3. True Colab preemption test: run 2 steps with mirror, DELETE the entire local trials dir, and resume to 4 steps from Drive mirror!
    local_trials = tmp_path / "ephemeral_vm_trials"
    drive_mirror = tmp_path / "persistent_drive_trials"
    backend.run_trial(
        config=cfg,
        runs_trials_dir=local_trials,
        enforce_budget_parity=False,
        eval_split=None,
        official_campaign=False,
        skip_subprocess_reload=True,
        mirror_dir=drive_mirror,
        device_override="cpu",
        keep_step_checkpoint=True,
    )
    assert (drive_mirror / "ephemeral_colab_trial" / "checkpoint_state.pt").is_file()

    # Simulate total Colab VM wipe!
    shutil.rmtree(local_trials)
    assert not local_trials.exists()

    resumed_env = backend.run_trial(
        config=cfg.model_copy(
            update={"training": cfg.training.model_copy(update={"max_steps": 4})}
        ),
        runs_trials_dir=local_trials,
        enforce_budget_parity=False,
        eval_split="validation",
        official_campaign=False,
        skip_subprocess_reload=True,
        mirror_dir=drive_mirror,
        device_override="cpu",
        resume=True,
    )
    assert resumed_env.status == "PASS"
    assert resumed_env.metrics is not None
    assert resumed_env.metrics.examples_seen == 8  # 4 restored from Drive + 4 new!
    assert not (drive_mirror / "ephemeral_colab_trial" / "checkpoint_state.pt").exists()


def test_lfm3_fast_kernel_guard_cuda_profiler_borobia_strict_and_verifier_guards(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Verify LFM3 CUDA fast-kernel check, Gemma CUDA rejection, 0-byte CUDA profiler abort, strict Borobia failure, and verifier rejections."""
    import aloe.evaluation.resources as res_mod
    from scripts.verify_campaign import verify_campaign_dir

    # 1. LFM3 without causal_conv1d on CUDA official campaign MUST fail closed; Gemma on CUDA MUST fail closed!
    monkeypatch.setattr(
        res_mod,
        "detect_hardware_summary",
        lambda: {
            "device": "cuda",
            "gpu_name": "NVIDIA A100-SXM4-40GB",
            "gpu_vram_gb": 40.0,
            "cuda_version": "12.4",
            "nvidia_driver": "550.54",
            "fast_mamba_kernels": False,
            "fast_causal_conv1d": False,
            "host_ram_gb": 85.0,
            "hardware_label": "NVIDIA A100-SXM4-40GB",
            "os": "Linux",
        },
    )
    with pytest.raises(RuntimeError, match="LFM3 CUDA fast-path"):
        enforce_hardware_contract("LiquidAI/LFM3-1.7B", "cuda", pinned_gpu="A100", official_campaign=True)
    with pytest.raises(RuntimeError, match="ADR-012 hardware-tier"):
        enforce_hardware_contract("google/gemma-4-E2B-it", "cuda", official_campaign=True)
    with pytest.raises(RuntimeError, match="Exact GPU SKU lock"):
        enforce_hardware_contract(
            "LiquidAI/LFM3-1.7B",
            "cuda",
            pinned_gpu="A100",
            exact_gpu_name="NVIDIA A100-SXM4-80GB",
            official_campaign=False,
        )

    # 2. Mocked CUDA ResourceProfiler: 0 bytes MUST raise RuntimeError; positive bytes returns cuda_max_memory_allocated
    import torch

    monkeypatch.setattr(torch.cuda, "synchronize", lambda: None)
    monkeypatch.setattr(torch.cuda, "reset_peak_memory_stats", lambda: None)
    monkeypatch.setattr(torch.cuda, "max_memory_allocated", lambda: 0)
    monkeypatch.setattr(torch.cuda, "max_memory_reserved", lambda: 0)
    prof_zero = res_mod.ResourceProfiler(sample_interval_ms=20, device="cuda")
    prof_zero.start()
    with pytest.raises(RuntimeError, match="0 bytes"):
        prof_zero.stop()

    monkeypatch.setattr(torch.cuda, "max_memory_allocated", lambda: 512 * 1024 * 1024)
    monkeypatch.setattr(torch.cuda, "max_memory_reserved", lambda: 768 * 1024 * 1024)
    prof_ok = res_mod.ResourceProfiler(sample_interval_ms=20, device="cuda")
    prof_ok.start()
    stats_ok = prof_ok.stop()
    assert stats_ok["memory_metric"] == "cuda_max_memory_allocated"
    assert stats_ok["search_memory_mb"] == 512.0

    # 3. Borobia with ultra-tight ±0.001% tolerance where no integer rank fits MUST raise InvalidConfigError!
    desc_g, gram_g = inspect_to_grammar("google/gemma-4-E2B-it")
    with pytest.raises(InvalidConfigError, match="Parameter budget parity violation"):
        solve_borobia_ablation_configs(desc_g, gram_g, baseline_params=12_079_105, tolerance=1e-7)

    # 4. verify_campaign_dir MUST reject partial campaigns and UNKNOWN predictions_sha256!
    partial_dir = tmp_path / "partial_camp"
    partial_dir.mkdir(parents=True)
    (partial_dir / "campaign_ledger.json").write_text(
        json.dumps(
            {
                "campaign_id": "c1",
                "model_id": "m",
                "arm": "b0",
                "target_trials": 2,
                "entries": [{"trial_id": "t0", "config_signature": "s0", "status": "DONE"}],
            }
        ),
        encoding="utf-8",
    )
    with pytest.raises(AssertionError, match="Non-protocol target_trials"):
        verify_campaign_dir(partial_dir, allow_smoke_budget=False)
    with pytest.raises(AssertionError, match="Incomplete campaign"):
        verify_campaign_dir(partial_dir, allow_smoke_budget=True)

    unk_dir = tmp_path / "unk_sha_camp"
    (unk_dir / "trials" / "t0").mkdir(parents=True)
    (unk_dir / "trials" / "t0" / "predictions_validation.jsonl").write_text('{"a":1}\n', encoding="utf-8")
    (unk_dir / "trials" / "t0" / "trial.json").write_text(
        json.dumps(
            {
                "status": "PASS",
                "trial_config": {"peft_method": "lora"},
                "metrics": {
                    "ast_exact_match_accuracy": 0.5,
                    "tool_accuracy": 0.5,
                    "search_memory_mb": 100.0,
                    "trainable_params": 1000,
                    "is_budget_matched_5pct": True,
                    "predictions_sha256": "UNKNOWN",
                    "eval_samples": 1,
                },
            }
        ),
        encoding="utf-8",
    )
    (unk_dir / "campaign_ledger.json").write_text(
        json.dumps(
            {
                "campaign_id": "c2",
                "model_id": "m",
                "arm": "b0",
                "target_trials": 1,
                "entries": [
                    {
                        "trial_id": "t0",
                        "config_signature": "s0",
                        "status": "DONE",
                        "metrics": {"tool_accuracy": 0.5, "search_memory_mb": 100.0},
                    }
                ],
            }
        ),
        encoding="utf-8",
    )
    with pytest.raises(AssertionError, match="UNKNOWN predictions_sha256"):
        verify_campaign_dir(unk_dir, allow_smoke_budget=True)


def test_cli_eval_isolation_mobo_calibration_exhaustion_and_stats_boundaries(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Verify CLI evaluate hardware/seed/config guards, MOBO Sobol calibration + chunked/exhausted/GP-fail paths, and p==alpha / NaN stats."""
    import math
    from typer.testing import CliRunner
    from aloe.cli.main import app
    from aloe.execution.campaign_runner import compute_exact_feasible_pool
    from aloe.optimization.bo_adapter import BOSearchAdapter
    from aloe.optimization.mobo import MOBOptimizer
    from aloe.tracking.trial_store import TrialStore

    # 1. Statistical boundary checks: p == alpha rejects; NaN raises; even-seed tie raises!
    h_boundary = holm_bonferroni({"h1": 0.05}, alpha=0.05)
    assert h_boundary["h1"].reject_null is True
    with pytest.raises(ValueError, match="Invalid p-value"):
        holm_bonferroni({"bad": float("nan")})
    with pytest.raises(ValueError, match="alpha"):
        holm_bonferroni({"h1": 0.01}, alpha=1.5)
    with pytest.raises(ValueError, match="0.5 majority-vote tie"):
        mcnemar_exact({"q1": [1, 0]}, {"q1": [1, 1]})

    # 2. MOBO Sobol-nadir calibration + chunked discrete qLogNEHVI + exhausted pool + official GP-fit failure
    desc, grammar = inspect_to_grammar("google/gemma-4-E2B-it")
    pool_info = compute_exact_feasible_pool(desc, grammar, parity_tol=0.05)
    small_adapter = BOSearchAdapter(
        grammar=grammar,
        arch=desc,
        feasible_configs=pool_info["feasible_configs"][:3],
    )
    opt = MOBOptimizer(adapter=small_adapter, seed=42, n_init=2, allow_fallback=False, discrete_chunk_size=1)
    assert opt.ref_point_calibrated is False
    c0, c1 = opt.suggest(q=2)
    opt.observe(c0, {"tool_accuracy": 0.50, "trainable_parameters": 12_000_000, "search_memory_mb": 8100.0})
    opt.observe(c1, {"tool_accuracy": 0.60, "trainable_parameters": 11_900_000, "search_memory_mb": 8000.0})
    assert opt.ref_point_calibrated is True
    assert opt.ref_point[0] == pytest.approx(0.485)
    assert opt.ref_point[2] == pytest.approx(-8124.3)

    # 3rd candidate uses chunked discrete qLogNEHVI (`discrete_chunk_size=1`)
    c2 = opt.suggest(q=1)[0]
    opt.observe(c2, {"tool_accuracy": 0.65, "trainable_parameters": 11_950_000, "search_memory_mb": 8050.0})

    # 4th candidate exhausts the 3-config feasible pool -> MUST raise RuntimeError (no continuous fallback!)
    with pytest.raises(RuntimeError, match="Exhausted all unique candidates"):
        opt.suggest(q=1)

    # Mismatched load_state_dict MUST raise ValueError
    opt_bad = MOBOptimizer(adapter=small_adapter, seed=999, n_init=2, allow_fallback=False)
    with pytest.raises(ValueError, match="mismatched settings"):
        opt_bad.load_state_dict(opt.state_dict())

    # Official GP fitting failure MUST raise RuntimeError when allow_fallback=False
    import botorch.fit as bf_mod

    opt_gp_fail = MOBOptimizer(
        adapter=BOSearchAdapter(grammar=grammar, arch=desc, feasible_configs=pool_info["feasible_configs"][:5]),
        seed=42,
        n_init=2,
        allow_fallback=False,
    )
    f0, f1 = opt_gp_fail.suggest(q=2)
    opt_gp_fail.observe(f0, {"tool_accuracy": 0.4, "trainable_parameters": 12_000_000, "search_memory_mb": 8100.0})
    opt_gp_fail.observe(f1, {"tool_accuracy": 0.6, "trainable_parameters": 11_900_000, "search_memory_mb": 8000.0})
    monkeypatch.setattr(bf_mod, "fit_gpytorch_mll", lambda *_a, **_kw: (_ for _ in ()).throw(RuntimeError("boom")))
    with pytest.raises(RuntimeError, match="Official MOBO GP MLL fitting failed"):
        opt_gp_fail.suggest(q=1)

    # 3. CLI evaluate isolation: reject mixed memory_metric_name and mismatched seed sets!
    cli = CliRunner()
    mix_dir = tmp_path / "mixed_hw"
    for idx, (arm_n, mem_m, seed_v) in enumerate(
        [("b0", "process_rss_sampled_100ms", 42), ("aloe", "cuda_max_memory_allocated", 42)]
    ):
        tdir = mix_dir / f"t{idx}"
        pstore = PredictionStore(tdir / "predictions_validation.jsonl")
        pstore.append(
            PredictionRecord(
                sample_id="q0",
                split="validation",
                model_id="google/gemma-4-E2B-it",
                model_revision="rev",
                arm=arm_n,
                trial_id=f"t{idx}",
                seed=seed_v,
                prompt_sha256="s",
                raw_output="[]",
                ground_truth=[],
                parse_success=True,
                tool_match=True,
                arg_names_match=True,
                arg_values_match=True,
                exact_match=True,
            )
        )
        TrialStore(mix_dir).save(
            TrialResultEnvelope(
                trial_id=f"t{idx}",
                arm=arm_n,
                git_commit="sha",
                dataset_manifest_sha256="m",
                model_revision="rev",
                hardware="HW",
                gpu_name="HW",
                library_versions={"torch": "2.4"},
                model_id="google/gemma-4-E2B-it",
                backend="peft_cpu",
                trial_config={"model_id": "google/gemma-4-E2B-it", "arm": arm_n, "seed": seed_v},
                metrics=TrialMetrics(
                    peak_memory_gb=8.0,
                    peak_rss_gb=8.0,
                    search_memory_mb=8192.0,
                    memory_metric_name=mem_m,
                    train_seconds=1.0,
                    latency_per_step_sec=0.1,
                    trainable_params=12_079_104,
                    total_params=2_000_000_000,
                    trainable_ratio=0.006,
                    adapter_size_mb=10.0,
                    eval_samples=1,
                    predictions_jsonl_path=str(pstore.path),
                    predictions_sha256=pstore.sha256(),
                    tool_accuracy=1.0,
                    ast_exact_match_accuracy=1.0,
                ),
                status="PASS",
            )
        )

    res_mix = cli.invoke(app, ["evaluate", "--run-dir", str(mix_dir), "--split", "validation"])
    assert res_mix.exit_code != 0
    assert "Cross-arm hardware/metric pooling prohibited" in res_mix.output

    # Different seed labels ({42} vs {1337}) MUST fail closed!
    seed_mismatch_dir = tmp_path / "seed_mismatch"
    for idx, (arm_n, seed_v) in enumerate([("b0", 42), ("aloe", 1337)]):
        tdir = seed_mismatch_dir / f"s{idx}"
        pstore = PredictionStore(tdir / "predictions_validation.jsonl")
        pstore.append(
            PredictionRecord(
                sample_id="q0",
                split="validation",
                model_id="google/gemma-4-E2B-it",
                model_revision="rev",
                arm=arm_n,
                trial_id=f"s{idx}",
                seed=seed_v,
                prompt_sha256="s",
                raw_output="[]",
                ground_truth=[],
                parse_success=True,
                tool_match=True,
                arg_names_match=True,
                arg_values_match=True,
                exact_match=True,
            )
        )
        TrialStore(seed_mismatch_dir).save(
            TrialResultEnvelope(
                trial_id=f"s{idx}",
                arm=arm_n,
                git_commit="sha",
                dataset_manifest_sha256="m",
                model_revision="rev",
                hardware="SameHW",
                gpu_name="SameHW",
                library_versions={"torch": "2.4"},
                model_id="google/gemma-4-E2B-it",
                backend="peft_cpu",
                trial_config={"model_id": "google/gemma-4-E2B-it", "arm": arm_n, "seed": seed_v},
                metrics=TrialMetrics(
                    peak_memory_gb=8.0,
                    peak_rss_gb=8.0,
                    search_memory_mb=8192.0,
                    memory_metric_name="process_rss_sampled_100ms",
                    train_seconds=1.0,
                    latency_per_step_sec=0.1,
                    trainable_params=12_079_104,
                    total_params=2_000_000_000,
                    trainable_ratio=0.006,
                    adapter_size_mb=10.0,
                    eval_samples=1,
                    predictions_jsonl_path=str(pstore.path),
                    predictions_sha256=pstore.sha256(),
                    tool_accuracy=1.0,
                    ast_exact_match_accuracy=1.0,
                ),
                status="PASS",
            )
        )
    res_seeds = cli.invoke(app, ["evaluate", "--run-dir", str(seed_mismatch_dir), "--split", "validation"])
    assert res_seeds.exit_code != 0
    assert "identical seed labels" in res_seeds.output


def test_round2_preemption_memory_cross_arm_lock_sobol_abort_and_colab_gates(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Verify Opus H1-H3 + M1-M4 + Hermes F-01..F-10: eval-resume peak memory, cross-arm GPU lock, Sobol failure abort, --retry-failed, stale prediction rejection, and Colab phase gates."""
    import torch
    from aloe.execution.backends.peft_mps import PeftMPSBackend
    from aloe.execution.trial_config import TrainingParameters
    import scripts.run_colab_a100_campaign as colab_mod

    model_dir, manifest_path, raw_path = _create_tiny_llama_and_dataset(tmp_path)
    backend = PeftMPSBackend()

    # 1. Opus H1 + Hermes F-04: Resuming at completed_step == max_steps preserves high training-phase peak memory!
    t_trials = tmp_path / "h1_trials"
    cfg = TrialConfig(
        trial_id="h1_mem_trial",
        arm="aloe",
        model_id=str(model_dir),
        seed=42,
        dataset_manifest_path=str(manifest_path),
        dataset_raw_path=str(raw_path),
        explicit_target_modules=["q_proj", "v_proj"],
        explicit_rank=4,
        explicit_alpha=8,
        training=TrainingParameters(
            learning_rate=2e-4,
            batch_size=2,
            gradient_accumulation_steps=1,
            max_steps=2,
            warmup_steps=1,
            max_seq_len=512,
            max_new_tokens=8,
            checkpoint_every_steps=2,
            precision="fp32",
        ),
    )
    backend.run_trial(
        config=cfg,
        runs_trials_dir=t_trials,
        enforce_budget_parity=False,
        eval_split=None,
        official_campaign=False,
        skip_subprocess_reload=True,
        device_override="cpu",
        keep_step_checkpoint=True,
    )
    ckpt_file = t_trials / "h1_mem_trial" / "checkpoint_state.pt"
    ckpt_data = torch.load(ckpt_file, map_location="cpu", weights_only=True)
    # Inject a high 9999.0 MB training peak into checkpoint to verify evaluation-only resume preserves it!
    ckpt_data["peak_memory_so_far"]["search_memory_mb"] = 9999.0
    torch.save(ckpt_data, ckpt_file)

    resumed = backend.run_trial(
        config=cfg,
        runs_trials_dir=t_trials,
        enforce_budget_parity=False,
        eval_split="validation",
        official_campaign=False,
        skip_subprocess_reload=True,
        device_override="cpu",
        resume=True,
        keep_step_checkpoint=True,
    )
    assert resumed.metrics is not None
    assert resumed.metrics.resumed_from_checkpoint is True
    assert resumed.metrics.search_memory_mb == pytest.approx(9999.0)

    # Hermes F-03: Injecting a foreign/stale sample_id into predictions_validation.jsonl MUST fail closed on resume!
    pred_file = t_trials / "h1_mem_trial" / "predictions_validation.jsonl"
    pstore = PredictionStore(pred_file)
    pstore.append(
        PredictionRecord(
            sample_id="999_foreign",
            split="validation",
            model_id=str(model_dir),
            model_revision="main",
            arm="aloe",
            trial_id="h1_mem_trial",
            seed=42,
            prompt_sha256="bad",
            raw_output="[]",
            ground_truth=[],
            parse_success=True,
            tool_match=True,
            arg_names_match=True,
            arg_values_match=True,
            exact_match=True,
        )
    )
    with pytest.raises(ValueError, match="Stale/foreign sample_id"):
        backend.run_trial(
            config=cfg,
            runs_trials_dir=t_trials,
            enforce_budget_parity=False,
            eval_split="validation",
            official_campaign=False,
            skip_subprocess_reload=True,
            device_override="cpu",
            resume=True,
        )

    # Hermes F-04: Corrupting checkpoint missing adapter key MUST fail closed!
    bad_ckpt = dict(ckpt_data)
    bad_ckpt["adapter_state"] = {}
    torch.save(bad_ckpt, ckpt_file)
    with pytest.raises(RuntimeError, match="Adapter checkpoint key mismatch"):
        backend.run_trial(
            config=cfg,
            runs_trials_dir=t_trials,
            enforce_budget_parity=False,
            eval_split=None,
            official_campaign=False,
            skip_subprocess_reload=True,
            device_override="cpu",
            resume=True,
        )

    # 2. Opus H2: Cross-arm GPU lock (`hardware_lock.json`) rejects switching from A100-40GB on aloe -> A100-80GB on b0!
    desc, grammar = inspect_to_grammar("google/gemma-4-E2B-it")
    repo_manifest = str(Path(__file__).resolve().parent.parent / "data/manifests/xlam_ipd_v1.json")
    model_camp_root = tmp_path / "cross_arm_lock"
    sku = {"name": "NVIDIA A100-SXM4-40GB"}

    def _sku_exec(c: TrialConfig, tdir_root: Path) -> TrialResultEnvelope:
        td = tdir_root / c.trial_id
        td.mkdir(parents=True, exist_ok=True)
        ps = PredictionStore(td / "predictions_validation.jsonl")
        if "q0" not in ps.completed_sample_ids():
            ps.append(
                PredictionRecord(
                    sample_id="q0",
                    split="validation",
                    model_id=c.model_id,
                    model_revision="3e22461f65e89153144f8adb70e3b8c2cc9845a7",
                    arm=c.arm,
                    trial_id=c.trial_id,
                    seed=c.seed,
                    prompt_sha256="s",
                    raw_output="[]",
                    ground_truth=[],
                    parse_success=True,
                    tool_match=True,
                    arg_names_match=True,
                    arg_values_match=True,
                    exact_match=True,
                )
            )
        return TrialResultEnvelope(
            trial_id=c.trial_id,
            arm=c.arm,
            git_commit="sha",
            dataset_manifest_sha256="m",
            model_revision="3e22461f65e89153144f8adb70e3b8c2cc9845a7",
            hardware=sku["name"],
            gpu_name=sku["name"],
            library_versions={"torch": "2.4"},
            model_id=c.model_id,
            backend="peft_cuda",
            trial_config=c.model_dump(mode="json"),
            metrics=TrialMetrics(
                peak_memory_gb=8.0,
                peak_rss_gb=4.0,
                search_memory_mb=8192.0,
                memory_metric_name="cuda_max_memory_allocated",
                train_seconds=1.0,
                latency_per_step_sec=0.1,
                trainable_params=c.compute_trainable_params(desc, grammar),
                total_params=2_000_000_000,
                trainable_ratio=0.006,
                adapter_size_mb=10.0,
                eval_samples=1,
                predictions_jsonl_path=str(ps.path),
                predictions_sha256=ps.sha256(),
                tool_accuracy=0.75,
                ast_exact_match_accuracy=0.75,
                is_budget_matched_5pct=True,
            ),
            status="PASS",
        )

    r_arm1 = CampaignRunner(
        arch=desc,
        grammar=grammar,
        campaign_dir=model_camp_root / "borobia_ffn",
        arm="borobia_ffn",
        target_trials=1,
        manifest_path=repo_manifest,
        pinned_gpu="A100",
        official_campaign=False,
        use_subprocess_worker=False,
        trial_executor_override=_sku_exec,
    )
    s_arm1 = r_arm1.run_campaign()
    assert s_arm1["protocol_tag"] == "smoke-non-official"
    assert (model_camp_root / "hardware_lock.json").is_file()

    sku["name"] = "NVIDIA A100-SXM4-80GB"
    r_arm2 = CampaignRunner(
        arch=desc,
        grammar=grammar,
        campaign_dir=model_camp_root / "b0",
        arm="b0",
        target_trials=1,
        manifest_path=repo_manifest,
        pinned_gpu="A100",
        official_campaign=False,
        use_subprocess_worker=False,
        trial_executor_override=_sku_exec,
    )
    with pytest.raises(RuntimeError, match="Cross-arm GPU variant mismatch"):
        r_arm2.run_campaign(max_attempts_per_trial=1)

    # 3. Opus H3: Failed Sobol trial stops ALOE before BO, and --retry-failed recovers FAILED entries!
    sku["name"] = "NVIDIA A100-SXM4-40GB"
    with pytest.raises(RuntimeError, match="Incomplete campaign"):
        r_arm2.run_campaign(max_attempts_per_trial=1, retry_failed=False)
    recovered = r_arm2.run_campaign(max_attempts_per_trial=1, retry_failed=True)
    assert recovered["completed_trials"] == 1

    sobol_fail_dir = tmp_path / "sobol_fail"
    r_aloe = CampaignRunner(
        arch=desc,
        grammar=grammar,
        campaign_dir=sobol_fail_dir / "aloe",
        arm="aloe",
        target_trials=6,
        manifest_path=repo_manifest,
        official_campaign=False,
        use_subprocess_worker=False,
        trial_executor_override=lambda _c, _d: (_ for _ in ()).throw(RuntimeError("transient OOM")),
    )
    with pytest.raises(RuntimeError, match="transient OOM"):
        r_aloe.run_campaign(max_attempts_per_trial=1)
    # Rerun without retry_failed: must refuse to advance past failed Sobol point!
    r_aloe.trial_executor_override = _sku_exec
    with pytest.raises(RuntimeError, match="Sobol initial design trial"):
        r_aloe.run_campaign(max_attempts_per_trial=1, retry_failed=False)

    # 4. Hermes F-01 + Opus M1: Colab orchestrator rejects 1-seed final-test, non-20 search, and active lease!
    lease_p = tmp_path / "lease.json"
    with colab_mod.acquire_model_lease(lease_p, phase="search"):
        with pytest.raises(RuntimeError, match="Active model campaign lease"):
            with colab_mod.acquire_model_lease(lease_p, phase="confirm"):
                pass

    monkeypatch.setattr(sys, "argv", ["run_colab_a100_campaign.py", "--model", "tiiuae/Falcon-H1-1.5B-Instruct", "--phase", "final-test", "--seeds", "42", "--confirm-sealed-test"])
    with pytest.raises(ValueError, match="42,1337,2026"):
        colab_mod.main()

    monkeypatch.setattr(sys, "argv", ["run_colab_a100_campaign.py", "--model", "tiiuae/Falcon-H1-1.5B-Instruct", "--phase", "search", "--trials", "3"])
    with pytest.raises(ValueError, match="fixed at --trials 20"):
        colab_mod.main()


def test_round3_lease_heartbeat_seal_hash_dirty_git_rescore_pareto_and_traversal(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Verify Opus B1/B2/M-b + Hermes #1/#2/#3: background lease heartbeat, full winner seal hash, fail-closed git status, rescore Pareto propagation, and path traversal guard."""
    import time
    from aloe.execution.trial_config import TrainingParameters
    from aloe.tracking.trial_store import TrialStore, get_git_commit
    import scripts.run_colab_a100_campaign as colab_mod
    from scripts.verify_campaign import verify_campaign_dir

    # 1. Opus B1: 0.15s TTL lease stays alive across 0.32s via 0.04s background heartbeat thread!
    hb_lease = tmp_path / "hb_lease.json"
    with colab_mod.acquire_model_lease(
        hb_lease,
        phase="search",
        ttl_seconds=0.15,
        heartbeat_interval_seconds=0.04,
    ):
        time.sleep(0.32)
        assert float(json.loads(hb_lease.read_text())["expires_epoch"]) > time.time()
        with pytest.raises(RuntimeError, match="Active model campaign lease"):
            with colab_mod.acquire_model_lease(hb_lease, phase="confirm"):
                pass

    # 2. Hermes #1: Full winner hash changes when max_steps or explicit_target_modules changes, while seed changes preserve it!
    w_base = TrialConfig(
        trial_id="t0",
        arm="aloe",
        model_id="tiiuae/Falcon-H1-1.5B-Instruct",
        seed=42,
        explicit_target_modules=["q_proj"],
        training=TrainingParameters(max_steps=125),
    )
    h_base = colab_mod.compute_full_winner_sha256(w_base)
    assert colab_mod.compute_full_winner_sha256(w_base.model_copy(update={"seed": 1337, "trial_id": "t9"})) == h_base
    assert colab_mod.compute_full_winner_sha256(
        w_base.model_copy(update={"training": TrainingParameters(max_steps=50)})
    ) != h_base
    assert colab_mod.compute_full_winner_sha256(
        w_base.model_copy(update={"explicit_target_modules": ["q_proj", "v_proj"]})
    ) != h_base

    # 3. Hermes #2: git rev-parse succeeds + git status returns non-zero -> forbid_dirty=True MUST raise!
    def _fake_subproc(cmd: list[str], **_kw: object) -> subprocess.CompletedProcess[str]:
        if "rev-parse" in cmd:
            return subprocess.CompletedProcess(cmd, 0, stdout="abc1234\n", stderr="")
        return subprocess.CompletedProcess(cmd, 128, stdout="", stderr="index lock error")

    monkeypatch.setattr("aloe.tracking.trial_store.subprocess.run", _fake_subproc)
    with pytest.raises(RuntimeError, match="clean committed Git tree"):
        get_git_commit(tmp_path, forbid_dirty=True)

    # 4. Hermes #3: rescore() updates trial.json, campaign_ledger.json, winner, AND pareto_trial_ids!
    desc, grammar = inspect_to_grammar("google/gemma-4-E2B-it")
    repo_manifest = str(Path(__file__).resolve().parent.parent / "data/manifests/xlam_ipd_v1.json")
    rc_dir = tmp_path / "rescore_pareto"

    def _two_trial_exec(cfg: TrialConfig, t_root: Path) -> TrialResultEnvelope:
        is_t0 = cfg.trial_id.endswith("0000")
        td = t_root / cfg.trial_id
        td.mkdir(parents=True, exist_ok=True)
        ps = PredictionStore(td / "predictions_validation.jsonl")
        # Initially t0 has raw_output="[]" vs gt=[{"name":"f","arguments":{}}] (0.0), while t1 has exact match (1.0) but higher memory!
        raw_out = "[]" if is_t0 else '[{"name":"f","arguments":{}}]'
        gt_val = [{"name": "f", "arguments": {}}]
        ok = not is_t0
        ps.append(
            PredictionRecord(
                sample_id="q0",
                split="validation",
                model_id=cfg.model_id,
                model_revision="3e22461f65e89153144f8adb70e3b8c2cc9845a7",
                arm=cfg.arm,
                trial_id=cfg.trial_id,
                seed=cfg.seed,
                prompt_sha256="s",
                raw_output=raw_out,
                ground_truth=gt_val,
                parse_success=True,
                tool_match=ok,
                arg_names_match=ok,
                arg_values_match=ok,
                exact_match=ok,
            )
        )
        mem_mb = 7000.0 if is_t0 else 9000.0
        env = TrialResultEnvelope(
            trial_id=cfg.trial_id,
            arm=cfg.arm,
            git_commit="sha",
            dataset_manifest_sha256="m",
            model_revision="3e22461f65e89153144f8adb70e3b8c2cc9845a7",
            hardware="MPS",
            gpu_name="MPS",
            library_versions={"torch": "2.4"},
            model_id=cfg.model_id,
            backend="peft_mps",
            trial_config=cfg.model_dump(mode="json"),
            metrics=TrialMetrics(
                peak_memory_gb=mem_mb / 1024.0,
                peak_rss_gb=mem_mb / 1024.0,
                search_memory_mb=mem_mb,
                memory_metric_name="process_rss_sampled_100ms",
                train_seconds=1.0,
                latency_per_step_sec=0.1,
                trainable_params=12_079_104,
                total_params=2_000_000_000,
                trainable_ratio=0.006,
                adapter_size_mb=10.0,
                eval_samples=1,
                predictions_jsonl_path=str(ps.path),
                predictions_sha256=ps.sha256(),
                tool_accuracy=1.0 if ok else 0.0,
                ast_exact_match_accuracy=1.0 if ok else 0.0,
                is_budget_matched_5pct=True,
            ),
            status="PASS",
        )
        TrialStore(t_root).save(env)
        return env

    runner = CampaignRunner(
        arch=desc,
        grammar=grammar,
        campaign_dir=rc_dir,
        arm="b0",
        target_trials=2,
        manifest_path=repo_manifest,
        official_campaign=False,
        use_subprocess_worker=False,
        trial_executor_override=_two_trial_exec,
    )
    s_init = runner.run_campaign()
    assert s_init["selected_winner_trial_id"] == "trial_b0_0001"
    assert sorted(s_init["pareto_trial_ids"]) == ["trial_b0_0000", "trial_b0_0001"]

    # Now update t0's stored raw_output to match gt (1.0 accuracy at lower 7000 MB memory -> t0 strictly dominates t1!)
    p0 = PredictionStore(rc_dir / "trials" / "trial_b0_0000" / "predictions_validation.jsonl")
    recs0 = p0.load_records()
    p0._rewrite_atomic([recs0[0].model_copy(update={"raw_output": '[{"name":"f","arguments":{}}]'})])
    p0.rescore()

    s_after = json.loads((rc_dir / "campaign_summary.json").read_text())
    assert s_after["selected_winner_trial_id"] == "trial_b0_0000"
    assert s_after["pareto_trial_ids"] == ["trial_b0_0000"]
    v_res = verify_campaign_dir(rc_dir, allow_smoke_budget=True)
    assert v_res["status"] == "PASS"

    # 5. Opus M-b: verify_campaign_dir rejects ../ path traversal in ledger trial_id!
    trav_dir = tmp_path / "trav_camp"
    (trav_dir / "trials").mkdir(parents=True)
    (trav_dir / "campaign_ledger.json").write_text(
        json.dumps(
            {
                "campaign_id": "trav",
                "model_id": "m",
                "arm": "borobia_ffn",
                "target_trials": 1,
                "entries": [{"trial_id": "../escape", "config_signature": "s", "status": "DONE"}],
            }
        ),
        encoding="utf-8",
    )
    with pytest.raises(AssertionError, match="Path traversal"):
        verify_campaign_dir(trav_dir, allow_smoke_budget=True)

    # 6. Opus Round-3 Blocker fix: verify_campaign_dir on a Google Drive mirror directory MUST succeed
    # even when `trial.json` still records the local VM's `predictions_jsonl_path` and that local file still exists!
    local_vm_dir = tmp_path / "local_vm_camp"
    drive_mir_dir = tmp_path / "drive_mir_camp"
    r_mir = CampaignRunner(
        arch=desc,
        grammar=grammar,
        campaign_dir=local_vm_dir,
        arm="borobia_ffn",
        target_trials=1,
        manifest_path=repo_manifest,
        official_campaign=False,
        use_subprocess_worker=False,
        mirror_dir=drive_mir_dir,
        trial_executor_override=_two_trial_exec,
    )
    r_mir.run_campaign()
    assert (local_vm_dir / "trials" / "borobia_ffn_s42" / "predictions_validation.jsonl").is_file()
    assert (drive_mir_dir / "trials" / "borobia_ffn_s42" / "predictions_validation.jsonl").is_file()
    mir_ver = verify_campaign_dir(drive_mir_dir, allow_smoke_budget=True)
    assert mir_ver["status"] == "PASS"
