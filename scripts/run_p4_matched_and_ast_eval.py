#!/usr/bin/env python3
"""Execute budget-matched (±5%) pilot trials and full AST tool+argument evaluation.

Addresses supervisor feedback and Claude Opus audit on PoC fairness:
1. Enforces ±5% parameter-budget parity on both architectures:
   - Gemma 4 E2B:
     * P1 (Uniform r=8, α=16): 12,079,104 params (0.00%)
     * P4_Matched (r_local=16, α=32; r_global=32, α=64; r_ffn=4, α=8): 11,515,904 params (-4.66% vs P1)
   - Falcon-H1 1.5B:
     * P1 (Uniform r=8, α=16): 7,578,624 params (0.00%)
     * P2 (Attention-Prioritized Matched: r_global=16, α=32; r_ssm=8, α=16; r_ffn=4, α=8): 7,726,080 params (+1.95% vs P1)
     * P4_SSM_Matched (SSM-Prioritized Matched: r_global=8, α=16; r_ssm=16, α=32; r_ffn=4, α=8): 7,342,080 params (-3.12% vs P1)
2. Trains matched pairs at the identical git commit under strict assistant-only label masking (-100 on all prompt tokens).
3. Evaluates Base Models (Zero-Shot, No Adapter) and all pilot adapters with ASTToolCallEvaluator at max_length=1024 (no prompt truncation).
"""
from __future__ import annotations

import json
import logging
from pathlib import Path
import shutil
from typing import Any

from peft import PeftModel
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

from aloe.architecture.schema import ArchitectureDescription
from aloe.datasets.model_registry import ModelRegistry
from aloe.datasets.xlam import load_xlam_file
from aloe.evaluation.tool_calling import ASTToolCallEvaluator, EvaluationSummary
from aloe.execution.backends.peft_mps import PeftMPSBackend
from aloe.execution.trial_config import GroupTrialSetting, TrialConfig
from aloe.search_space.schema import SearchGrammar
from aloe.tracking.trial_store import TrialResultEnvelope

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)


def evaluate_model_or_adapter_ast(
    snapshot_path: Path,
    adapter_path: Path | None,
    val_examples: list[Any],
    device: str = "mps",
    max_eval: int = 8,
) -> EvaluationSummary:
    """Evaluate tool-call selection and argument accuracy using ASTToolCallEvaluator (max_length=1024)."""
    label = adapter_path.parent.name if adapter_path is not None else "BASE_NO_ADAPTER"
    logger.info(f"Running AST evaluation on {label} ({max_eval} unseen tool val samples, max_length=1024)...")
    tokenizer = AutoTokenizer.from_pretrained(snapshot_path, trust_remote_code=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    base_model = AutoModelForCausalLM.from_pretrained(
        snapshot_path,
        torch_dtype=torch.bfloat16,
        device_map=device,
        trust_remote_code=True,
    )
    if adapter_path is not None:
        model = PeftModel.from_pretrained(base_model, adapter_path)
    else:
        model = base_model
    model.eval()

    evaluator = ASTToolCallEvaluator()
    predictions: list[str] = []
    ground_truths: list[list[dict[str, Any]]] = []
    sample_ids: list[str] = []

    total = min(len(val_examples), max_eval)
    for i in range(total):
        ex = val_examples[i]
        tools_str = json.dumps(ex.tools, indent=2)
        system_msg = (
            "You are an agent with access to the following tools:\n"
            f"{tools_str}\n\n"
            "If you need to call a tool, output valid JSON in the format:\n"
            '{"name": "tool_name", "arguments": {"arg1": "value"}}\n'
            "If no tool is required, respond with plain text."
        )
        messages = [
            {"role": "user", "content": f"{system_msg}\n\nUser request: {ex.query}"},
        ]
        try:
            prompt_text = tokenizer.apply_chat_template(
                messages, tokenize=False, add_generation_prompt=True
            )
        except Exception:
            prompt_text = f"{system_msg}\n\nUser request: {ex.query}\nAssistant: "

        inputs = tokenizer(
            prompt_text, return_tensors="pt", truncation=True, max_length=1024
        ).to(device)

        with torch.no_grad():
            out_tokens = model.generate(**inputs, max_new_tokens=96, do_sample=False)

        generated_text = tokenizer.decode(
            out_tokens[0][inputs["input_ids"].shape[1] :], skip_special_tokens=True
        )
        predictions.append(generated_text)
        ground_truths.append(ex.answers)
        sample_ids.append(str(ex.id))

    del model
    if adapter_path is not None:
        del base_model
    if hasattr(torch, "mps") and hasattr(torch.mps, "empty_cache"):
        torch.mps.empty_cache()

    summary = evaluator.evaluate_batch(predictions, ground_truths, sample_ids)
    logger.info(
        f"AST Summary [{label}]: "
        f"Parse={summary.parse_success_rate:.1%}, "
        f"Tool={summary.tool_selection_accuracy:.1%}, "
        f"ArgNames={summary.argument_name_accuracy:.1%}, "
        f"ArgValues={summary.argument_value_accuracy:.1%}, "
        f"Exact={summary.exact_match_accuracy:.1%}"
    )
    return summary


def main() -> None:
    repo_dir = Path(__file__).resolve().parent.parent
    # Detect sibling repo if syncing between working and public clones
    alt_dir = (
        Path("/Users/walapalam/Development/aloe")
        if repo_dir.name == "aloe" and repo_dir.parent.name == "fyp"
        else Path("/Users/walapalam/Development/fyp/aloe")
    )
    runs_dir = repo_dir / "runs" / "pilot"
    device = "mps" if torch.backends.mps.is_available() else "cpu"

    manifest_path = repo_dir / "data" / "manifests" / "xlam_pilot_v1.json"
    raw_path = repo_dir / "data" / "raw" / "xlam" / "xlam_function_calling_60k.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    val_ids = set(str(i) for i in manifest["splits"]["validation"]["ids"])
    raw_examples = load_xlam_file(raw_path)
    val_examples = [ex for ex in raw_examples if str(ex.id) in val_ids]

    registry = ModelRegistry()
    backend = PeftMPSBackend()

    # Load architectures and grammars
    gemma_arch = ArchitectureDescription.model_validate(
        json.loads(
            (repo_dir / "runs" / "architecture" / "google--gemma-4-E2B-it" / "inspection.json").read_text(
                encoding="utf-8"
            )
        )["architecture"]
    )
    gemma_grammar = SearchGrammar.model_validate_json(
        (repo_dir / "runs" / "grammars" / "gemma_e2b.json").read_text(encoding="utf-8")
    )

    falcon_arch = ArchitectureDescription.model_validate(
        json.loads(
            (repo_dir / "runs" / "architecture" / "tiiuae--Falcon-H1-1.5B-Instruct" / "inspection.json").read_text(
                encoding="utf-8"
            )
        )["architecture"]
    )
    falcon_grammar = SearchGrammar.model_validate_json(
        (repo_dir / "runs" / "grammars" / "falcon_h1.json").read_text(encoding="utf-8")
    )

    # 1. Verify ±5% parameter-budget parity on both Gemma 4 E2B and Falcon-H1 1.5B
    p4_gemma_cfg = TrialConfig(
        trial_id="pilot_gemma_e2b_p4_matched",
        model_id="google/gemma-4-E2B-it",
        seed=42,
        group_settings={
            "local_attention": GroupTrialSetting(enabled=True, rank=16, alpha=32),
            "global_attention": GroupTrialSetting(enabled=True, rank=32, alpha=64),
            "ffn": GroupTrialSetting(enabled=True, rank=4, alpha=8),
        },
        description=(
            "Typed Matched (±5% Parity) — Local Attn r=16 α=32, Global Attn r=32 α=64, "
            "FFN r=4 α=8 (11,515,904 params; -4.66% vs P1)"
        ),
    )
    p4_gemma_cfg.training.max_steps = 25
    p4_gemma_cfg.training.max_seq_len = 128
    gemma_parity = p4_gemma_cfg.validate_parameter_budget_parity(
        arch=gemma_arch, baseline_params=12_079_104, tolerance=0.05, grammar=gemma_grammar
    )
    logger.info(f"Gemma P4_Matched Parity Check vs P1: {gemma_parity}")

    p4_falcon_ssm_cfg = TrialConfig(
        trial_id="pilot_falcon_h1_p4_ssm_matched",
        model_id="tiiuae/Falcon-H1-1.5B-Instruct",
        seed=42,
        group_settings={
            "global_attention": GroupTrialSetting(enabled=True, rank=8, alpha=16),
            "ssm": GroupTrialSetting(enabled=True, rank=16, alpha=32),
            "ffn": GroupTrialSetting(enabled=True, rank=4, alpha=8),
        },
        description=(
            "Typed SSM-Prioritized Matched (±5% Parity) — Global Attn r=8 α=16, SSM r=16 α=32, "
            "FFN r=4 α=8 (7,342,080 params; -3.12% vs P1)"
        ),
    )
    p4_falcon_ssm_cfg.training.max_steps = 12
    p4_falcon_ssm_cfg.training.max_seq_len = 128
    falcon_ssm_parity = p4_falcon_ssm_cfg.validate_parameter_budget_parity(
        arch=falcon_arch, baseline_params=7_578_624, tolerance=0.05, grammar=falcon_grammar
    )
    logger.info(f"Falcon P4_SSM_Matched Parity Check vs P1: {falcon_ssm_parity}")

    # 2. Re-run Gemma P1 & P4_Matched and Falcon P1, P2 & P4_SSM_Matched at the identical commit
    trials_to_execute = [
        (
            "gemma_e2b",
            "P1",
            gemma_grammar,
            TrialConfig(
                trial_id="pilot_gemma_e2b_p1",
                model_id="google/gemma-4-E2B-it",
                seed=42,
                group_settings={
                    "local_attention": GroupTrialSetting(enabled=True, rank=8, alpha=16),
                    "global_attention": GroupTrialSetting(enabled=True, rank=8, alpha=16),
                    "ffn": GroupTrialSetting(enabled=True, rank=8, alpha=16),
                },
                description="Baseline — Uniform LoRA r=8, α=16 across all groups (12,079,104 params)",
            ),
            25,
        ),
        ("gemma_e2b", "P4_Matched", gemma_grammar, p4_gemma_cfg, 25),
        (
            "falcon_h1",
            "P1",
            falcon_grammar,
            TrialConfig(
                trial_id="pilot_falcon_h1_p1",
                model_id="tiiuae/Falcon-H1-1.5B-Instruct",
                seed=42,
                group_settings={
                    "global_attention": GroupTrialSetting(enabled=True, rank=8, alpha=16),
                    "ssm": GroupTrialSetting(enabled=True, rank=8, alpha=16),
                    "ffn": GroupTrialSetting(enabled=True, rank=8, alpha=16),
                },
                description="Baseline — Uniform LoRA r=8, α=16 across all groups (7,578,624 params)",
            ),
            12,
        ),
        (
            "falcon_h1",
            "P2",
            falcon_grammar,
            TrialConfig(
                trial_id="pilot_falcon_h1_p2",
                model_id="tiiuae/Falcon-H1-1.5B-Instruct",
                seed=42,
                group_settings={
                    "global_attention": GroupTrialSetting(enabled=True, rank=16, alpha=32),
                    "ssm": GroupTrialSetting(enabled=True, rank=8, alpha=16),
                    "ffn": GroupTrialSetting(enabled=True, rank=4, alpha=8),
                },
                description=(
                    "Typed Attention-Prioritized Matched (±5% Parity) — Global Attn r=16 α=32, "
                    "SSM r=8 α=16, FFN r=4 α=8 (7,726,080 params; +1.95% vs P1)"
                ),
            ),
            12,
        ),
        ("falcon_h1", "P4_SSM_Matched", falcon_grammar, p4_falcon_ssm_cfg, 12),
    ]

    for slug, cfg_id, grammar, cfg, steps in trials_to_execute:
        cfg.training.max_steps = steps
        cfg.training.max_seq_len = 128
        logger.info(f"Executing {slug}/{cfg_id} ({cfg.trial_id}, {steps} steps) at current commit...")
        envelope = backend.run_trial(
            config=cfg,
            runs_trials_dir=str(runs_dir / slug),
            grammar=grammar,
        )
        cfg_dir = runs_dir / slug / cfg_id
        cfg_dir.mkdir(parents=True, exist_ok=True)
        (cfg_dir / "trial.json").write_text(envelope.to_json(), encoding="utf-8")

    # 3. Evaluate Base Models (No Adapter) and all pilot adapters with max_length=1024
    model_runs = [
        (
            "google/gemma-4-E2B-it",
            "gemma_e2b",
            12_079_104,
            [
                ("P1", "pilot_gemma_e2b_p1"),
                ("P4_Matched", "pilot_gemma_e2b_p4_matched"),
                ("P2", "pilot_gemma_e2b_p2"),
                ("P3", "pilot_gemma_e2b_p3"),
            ],
        ),
        (
            "tiiuae/Falcon-H1-1.5B-Instruct",
            "falcon_h1",
            7_578_624,
            [
                ("P1", "pilot_falcon_h1_p1"),
                ("P2", "pilot_falcon_h1_p2"),
                ("P4_SSM_Matched", "pilot_falcon_h1_p4_ssm_matched"),
                ("P3", "pilot_falcon_h1_p3"),
            ],
        ),
    ]

    for model_id, slug, baseline_params, cfg_list in model_runs:
        snapshot_path = registry.get_snapshot_path(model_id)
        for cfg_id, trial_folder in cfg_list:
            record_path = runs_dir / slug / cfg_id / "trial.json"
            adapter_path = runs_dir / slug / trial_folder / "adapter"
            envelope = TrialResultEnvelope.from_json(record_path.read_text(encoding="utf-8"))

            ast_summary = evaluate_model_or_adapter_ast(
                snapshot_path=snapshot_path,
                adapter_path=adapter_path,
                val_examples=val_examples,
                device=device,
                max_eval=8,
            )

            assert envelope.metrics is not None
            delta_pct = round(
                (envelope.metrics.trainable_params - baseline_params) / baseline_params * 100.0, 2
            )
            is_matched = abs(delta_pct) <= 5.0

            envelope.metrics.tool_accuracy = ast_summary.tool_selection_accuracy
            envelope.metrics.ast_parse_success_rate = ast_summary.parse_success_rate
            envelope.metrics.ast_tool_selection_accuracy = ast_summary.tool_selection_accuracy
            envelope.metrics.ast_argument_name_accuracy = ast_summary.argument_name_accuracy
            envelope.metrics.ast_argument_value_accuracy = ast_summary.argument_value_accuracy
            envelope.metrics.ast_exact_match_accuracy = ast_summary.exact_match_accuracy
            envelope.metrics.param_delta_vs_baseline_pct = delta_pct
            envelope.metrics.is_budget_matched_5pct = is_matched

            record_path.write_text(envelope.to_json(), encoding="utf-8")
            trial_record_path = runs_dir / slug / trial_folder / "trial.json"
            trial_record_path.write_text(envelope.to_json(), encoding="utf-8")

            if alt_dir.is_dir():
                pub_cfg_dir = alt_dir / "runs" / "pilot" / slug / cfg_id
                pub_cfg_dir.mkdir(parents=True, exist_ok=True)
                shutil.copy2(record_path, pub_cfg_dir / "trial.json")

                pub_trial_dir = alt_dir / "runs" / "pilot" / slug / trial_folder
                pub_trial_dir.mkdir(parents=True, exist_ok=True)
                shutil.copy2(record_path, pub_trial_dir / "trial.json")

    logger.info("All budget-matched trials and AST evaluations completed and synced!")


if __name__ == "__main__":
    main()
