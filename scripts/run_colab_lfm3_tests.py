#!/usr/bin/env python3
"""ALOE Engine Verification & Empirical Benchmark Suite for LiquidAI/LFM3-1.7B.

Designed to run both locally (CPU / Apple Silicon MPS) and on Google Colab (CUDA GPU):
  1. Stage 1-4 Zero-RAM Meta-Device Inspection & 4-Group SearchGrammar Derivation
  2. Analytical vs PEFT Runtime Trainable Parameter Count & +/-5% Budget Parity Verification
  3. Empirical Pre-Flight Gradient Verification (142 valid targets = 284 GRADIENT_OK)
     + Mandatory Negative Control on LFM3 conv.conv under packed SFT (26/26 GRADIENT_MISSING)
  4. Optional Full-Weight GPU SFT Trial & AST Tool-Calling Evaluation (--run-sft-pilot)
"""
from __future__ import annotations

import argparse
import datetime
import gc
import json
import platform
import sys
import time
from pathlib import Path
from typing import Any

import torch
from peft import get_peft_model
from transformers import AutoConfig, AutoModelForCausalLM, AutoTokenizer

from aloe.architecture.constraints import apply_validity
from aloe.architecture.inspector import inspect_model
from aloe.architecture.typing import classify
from aloe.datasets.model_registry import ModelRegistry
from aloe.datasets.xlam import XLAMExample
from aloe.evaluation.tool_calling import ASTToolCallEvaluator
from aloe.execution.backends.peft_mps import PeftMPSBackend, set_seed, tokenize_xlam_sft_example
from aloe.execution.trial_config import GroupTrialSetting, InvalidConfigError, TrialConfig
from aloe.pipeline import inspect_to_grammar
from aloe.search_space.grammar import generate

LFM3_MODEL_ID = "LiquidAI/LFM3-1.7B"
LFM3_REVISION = "40ebe0aa6662c9d40389c2f5d565b6a2c6aebacd"


def detect_device() -> str:
    if torch.cuda.is_available():
        return "cuda"
    if hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
        return "mps"
    return "cpu"


def evaluate_lfm3_ast(
    model: torch.nn.Module,
    tokenizer: Any,
    val_examples: list[XLAMExample],
    device: str,
    max_eval: int = 8,
) -> dict[str, Any]:
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
        messages = [{"role": "user", "content": f"{system_msg}\n\nUser request: {ex.query}"}]
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

    summary = evaluator.evaluate_batch(predictions, ground_truths, sample_ids)
    return {
        "total_samples": summary.total_samples,
        "parse_success_rate": summary.parse_success_rate,
        "tool_selection_accuracy": summary.tool_selection_accuracy,
        "argument_name_accuracy": summary.argument_name_accuracy,
        "argument_value_accuracy": summary.argument_value_accuracy,
        "exact_match_accuracy": summary.exact_match_accuracy,
    }


def run_lfm3_sft_pilot(
    model: torch.nn.Module,
    tokenizer: Any,
    grammar: Any,
    configs: list[tuple[str, TrialConfig, dict[str, Any]]],
    device: str,
    steps: int = 25,
    max_eval: int = 8,
) -> dict[str, Any]:
    mini_path = Path(__file__).resolve().parent.parent / "data" / "manifests" / "xlam_mini_pilot.json"
    raw = json.loads(mini_path.read_text(encoding="utf-8"))
    train_examples = [XLAMExample.model_validate(x) for x in raw["train"]]
    val_examples = [XLAMExample.model_validate(x) for x in raw["validation"]]

    print("-" * 76)
    print(f"Step 4: Running LFM3-1.7B Zero-Shot & Budget-Matched SFT Pilot ({steps} steps, {max_eval} eval samples)")
    print("-" * 76)

    pilot_results: dict[str, Any] = {}
    base_ast = evaluate_lfm3_ast(model, tokenizer, val_examples, device=device, max_eval=max_eval)
    pilot_results["base_zero_shot"] = {
        "trainable_params": 0,
        "relative_delta_pct": -100.0,
        "ast_eval": base_ast,
    }
    print(
        f"[PILOT] Base Zero-Shot -> Parse={base_ast['parse_success_rate']:.1%} | "
        f"Tool={base_ast['tool_selection_accuracy']:.1%} | "
        f"ArgNames={base_ast['argument_name_accuracy']:.1%} | "
        f"Exact={base_ast['exact_match_accuracy']:.1%}"
    )

    tokenized_train = [
        tokenize_xlam_sft_example(train_examples[i % len(train_examples)], tokenizer, max_seq_len=256)
        for i in range(steps)
    ]

    for label, tcfg, parity in configs:
        set_seed(42)
        _, lora_cfg = tcfg.to_peft_lora_config(grammar)
        peft_model = get_peft_model(model, lora_cfg)
        peft_model.train()
        trainable_params = sum(p.numel() for p in peft_model.parameters() if p.requires_grad)

        optimizer = torch.optim.AdamW(peft_model.parameters(), lr=2e-4, weight_decay=0.01)
        losses: list[float] = []
        t_sft0 = time.perf_counter()
        for step in range(steps):
            optimizer.zero_grad()
            item = tokenized_train[step]
            out = peft_model(
                input_ids=item["input_ids"].unsqueeze(0).to(device),
                attention_mask=item["attention_mask"].unsqueeze(0).to(device),
                labels=item["labels"].unsqueeze(0).to(device),
            )
            loss = out.loss
            loss.backward()
            optimizer.step()
            losses.append(float(loss.item()))
        train_sec = round(time.perf_counter() - t_sft0, 2)

        ast_res = evaluate_lfm3_ast(peft_model, tokenizer, val_examples, device=device, max_eval=max_eval)
        model = peft_model.unload()
        if device == "cuda":
            torch.cuda.empty_cache()

        pilot_results[label] = {
            "trainable_params": trainable_params,
            "relative_delta_pct": parity["relative_delta_pct"],
            "within_5pct_budget": parity["within_budget"],
            "initial_loss": round(losses[0], 4),
            "final_loss": round(losses[-1], 4),
            "train_seconds": train_sec,
            "ast_eval": ast_res,
        }
        print(
            f"[PILOT] {label:28s} ({trainable_params:,} params, {parity['relative_delta_pct']:+.2f}%) -> "
            f"Loss {losses[0]:.3f}->{losses[-1]:.3f} ({train_sec:.1f}s) | "
            f"Parse={ast_res['parse_success_rate']:.1%} | "
            f"Tool={ast_res['tool_selection_accuracy']:.1%} | "
            f"ArgNames={ast_res['argument_name_accuracy']:.1%} | "
            f"Exact={ast_res['exact_match_accuracy']:.1%}"
        )

    return pilot_results


def run_lfm3_verification(
    full_weights: bool = False,
    run_sft_pilot: bool = False,
    steps: int = 25,
    output_json: Path | None = None,
) -> dict[str, Any]:
    t0 = time.perf_counter()
    device = detect_device()
    gpu_name = torch.cuda.get_device_name(0) if device == "cuda" else platform.processor() or platform.machine()

    print("=" * 76)
    print(f"ALOE x LiquidAI/LFM3-1.7B Verification Suite | Device: {device} ({gpu_name})")
    print("=" * 76)

    # -------------------------------------------------------------------------
    # Step 1: Zero-RAM Meta-Device Architecture Inspection & Grammar Derivation
    # -------------------------------------------------------------------------
    t_insp0 = time.perf_counter()
    desc, grammar = inspect_to_grammar(LFM3_MODEL_ID, backend="peft_mps")
    insp_ms = round((time.perf_counter() - t_insp0) * 1000.0, 2)

    groups_by_type = {g.component_type.value: len(g.targets) for g in grammar.groups}
    conv_excluded = [k for k in grammar.excluded if k.endswith(".conv.conv")]
    gate_excluded = [k for k in grammar.excluded if k.endswith(".self_attn.gate_proj")]

    assert desc.family == "lfm3", f"Expected family='lfm3', got {desc.family}"
    assert desc.topology == "sequential_hybrid", f"Expected topology='sequential_hybrid', got {desc.topology}"
    assert desc.num_layers == 24, f"Expected 24 layers, got {desc.num_layers}"
    assert desc.total_params == 1_990_945_536, f"Expected 1,990,945,536 params, got {desc.total_params}"
    assert groups_by_type == {
        "ssm": 26,
        "ffn": 72,
        "global_attention": 24,
        "local_attention": 20,
    }, f"Unexpected group target counts: {groups_by_type}"
    assert len(conv_excluded) == 13, f"Expected 13 excluded conv.conv modules, got {len(conv_excluded)}"
    assert len(gate_excluded) == 11, f"Expected 11 excluded self_attn.gate_proj modules, got {len(gate_excluded)}"
    assert grammar.cardinality() == 194_480, f"Expected cardinality 194,480, got {grammar.cardinality()}"

    print(f"[PASS] Step 1: Meta-Device Inspection & 4-Group Grammar ({insp_ms:.1f} ms)")
    print(f"       Total Params: {desc.total_params:,} | Valid Targets: 142 | Excluded: {len(grammar.excluded)}")
    print(f"       Groups: {groups_by_type} | Search Space Cardinality: {grammar.cardinality():,}")

    # -------------------------------------------------------------------------
    # Step 2: +/-5% Parameter-Budget Parity & Exclusion Gate Verification
    # -------------------------------------------------------------------------
    p1_uniform = TrialConfig(
        trial_id="lfm3_P1_Uniform_r8",
        model_id=LFM3_MODEL_ID,
        group_settings={
            "local_attention": GroupTrialSetting(enabled=True, rank=8, alpha=16),
            "global_attention": GroupTrialSetting(enabled=True, rank=8, alpha=16),
            "ssm": GroupTrialSetting(enabled=True, rank=8, alpha=16),
            "ffn": GroupTrialSetting(enabled=True, rank=8, alpha=16),
        },
    )
    p1_params = p1_uniform.compute_trainable_params(desc, grammar)
    assert p1_params == 8_052_736
    parity_p1 = p1_uniform.validate_parameter_budget_parity(
        desc, baseline_params=p1_params, tolerance=0.05, grammar=grammar
    )

    p2_global_exact = TrialConfig(
        trial_id="lfm3_P2_Global_Exact_Parity",
        model_id=LFM3_MODEL_ID,
        group_settings={
            "local_attention": GroupTrialSetting(enabled=True, rank=8, alpha=16),
            "global_attention": GroupTrialSetting(enabled=True, rank=16, alpha=32),
            "ssm": GroupTrialSetting(enabled=True, rank=4, alpha=8),
            "ffn": GroupTrialSetting(enabled=True, rank=8, alpha=16),
        },
    )
    parity_p2 = p2_global_exact.validate_parameter_budget_parity(
        desc, baseline_params=p1_params, tolerance=0.05, grammar=grammar
    )
    assert parity_p2["candidate_params"] == 8_052_736
    assert parity_p2["relative_delta_pct"] == 0.0

    p3_operators = TrialConfig(
        trial_id="lfm3_P3_Operator_Prioritized",
        model_id=LFM3_MODEL_ID,
        group_settings={
            "local_attention": GroupTrialSetting(enabled=True, rank=16, alpha=32),
            "global_attention": GroupTrialSetting(enabled=True, rank=16, alpha=32),
            "ssm": GroupTrialSetting(enabled=True, rank=16, alpha=32),
            "ffn": GroupTrialSetting(enabled=True, rank=4, alpha=8),
        },
    )
    parity_p3 = p3_operators.validate_parameter_budget_parity(
        desc, baseline_params=p1_params, tolerance=0.05, grammar=grammar
    )
    assert parity_p3["candidate_params"] == 7_700_480
    assert parity_p3["relative_delta_pct"] == -4.37

    # Verify excluded modules raise InvalidConfigError
    for bad_mod in ["model.layers.0.conv.conv", "model.layers.2.self_attn.gate_proj"]:
        bad_cfg = TrialConfig(model_id=LFM3_MODEL_ID, explicit_target_modules=[bad_mod])
        try:
            bad_cfg.validate_against_grammar(grammar)
            raise AssertionError(f"Expected InvalidConfigError for excluded module {bad_mod}")
        except InvalidConfigError:
            pass

    print("[PASS] Step 2: +/-5% Parameter-Budget Parity & Exclusion Gate Verification")
    print(f"       P1 (Uniform r=8):            {p1_params:,} params (baseline)")
    print(f"       P2 (r_loc=8,r_glb=16,r_ssm=4,r_ffn=8): {parity_p2['candidate_params']:,} params ({parity_p2['relative_delta_pct']:+.2f}%)")
    print(f"       P3 (r_loc=16,r_glb=16,r_ssm=16,r_ffn=4): {parity_p3['candidate_params']:,} params ({parity_p3['relative_delta_pct']:+.2f}%)")

    # -------------------------------------------------------------------------
    # Step 3: Empirical Pre-Flight Gradient Verification & Negative Control
    # -------------------------------------------------------------------------
    ModelRegistry._ensure_lfm3_rope_compat()
    cfg = AutoConfig.from_pretrained(LFM3_MODEL_ID, trust_remote_code=True)

    if full_weights or run_sft_pilot:
        print(f"[INFO] Loading full 1.99B checkpoint {LFM3_MODEL_ID} on {device} (bfloat16)...")
        model = AutoModelForCausalLM.from_pretrained(
            LFM3_MODEL_ID,
            revision=LFM3_REVISION,
            torch_dtype=torch.bfloat16,
            trust_remote_code=True,
        ).to(device)
        test_desc, test_grammar = desc, grammar
        conv_rank = cfg.hidden_size
    else:
        # Dimension-scaled 24-layer LFM3 preserving exact 24-layer topology (13 conv, 5 sliding, 6 full)
        cfg.hidden_size = 64
        cfg.intermediate_size = 192
        cfg.num_attention_heads = 4
        cfg.num_key_value_heads = 2
        cfg.head_dim = 16
        cfg.vocab_size = 512
        cfg.pad_token_id = 0
        cfg.bos_token_id = 1
        cfg.eos_token_id = 2
        model = AutoModelForCausalLM.from_config(cfg, trust_remote_code=True).to(device)
        test_desc = inspect_model(model, model_id=LFM3_MODEL_ID)
        test_desc = classify(test_desc, model_id=LFM3_MODEL_ID, config=cfg)
        test_desc = apply_validity(test_desc, backend="peft_mps")
        test_grammar = generate(test_desc, backend="peft_mps")
        conv_rank = cfg.hidden_size

        # Verify PEFT runtime parameter count matches analytical formula on scaled 24-layer model
        for trial_cfg in (p1_uniform, p2_global_exact, p3_operators):
            analytic = trial_cfg.compute_trainable_params(test_desc, test_grammar)
            _, peft_lora_cfg = trial_cfg.to_peft_lora_config(test_grammar)
            pm = get_peft_model(model, peft_lora_cfg)
            runtime_trainable = sum(p.numel() for p in pm.parameters() if p.requires_grad)
            model = pm.unload()
            assert analytic == runtime_trainable, f"Mismatch: analytic={analytic} vs runtime={runtime_trainable}"

    backend = PeftMPSBackend()
    packed_inputs = {
        "input_ids": torch.tensor([[10, 11, 12, 13, 14, 15]], device=device),
        "position_ids": torch.tensor([[0, 1, 2, 0, 1, 2]], device=device),
    }

    component_grad_results: dict[str, Any] = {}
    for g in test_grammar.groups:
        cname = g.component_type.value
        t_g0 = time.perf_counter()
        res = backend.verify_gradients(
            model=model,
            target_modules=g.targets,
            r=8,
            lora_alpha=16,
            inputs=packed_inputs,
        )
        elapsed_ms = round((time.perf_counter() - t_g0) * 1000.0, 2)
        model = res["unloaded_model"]
        assert res["verdict"] == "PASS", f"Gradient check failed for {cname}: {res}"
        assert res["gradient_ok_count"] == len(g.targets) * 2
        assert res["gradient_missing_count"] == 0
        component_grad_results[cname] = {
            "target_module_count": res["target_module_count"],
            "adapter_param_count": res["adapter_param_count"],
            "gradient_ok_count": res["gradient_ok_count"],
            "gradient_missing_count": res["gradient_missing_count"],
            "min_gradient_norm": res["min_gradient_norm"],
            "max_gradient_norm": res["max_gradient_norm"],
            "elapsed_ms": elapsed_ms,
            "verdict": res["verdict"],
        }
        print(
            f"[PASS] Step 3a: Gradient Check [{cname:16s}] -> "
            f"{res['gradient_ok_count']}/{res['adapter_param_count']} GRADIENT_OK "
            f"(min={res['min_gradient_norm']:.2e}, max={res['max_gradient_norm']:.2e}, {elapsed_ms:.1f} ms)"
        )

    # Mandatory Negative Control: 13 conv.conv depthwise kernels under packed SFT
    conv_targets = [k for k in test_grammar.excluded if k.endswith(".conv.conv")]
    res_neg = backend.verify_gradients(
        model=model,
        target_modules=conv_targets,
        r=conv_rank,
        lora_alpha=conv_rank,
        inputs=packed_inputs,
    )
    model = res_neg["unloaded_model"]
    assert res_neg["verdict"] == "FAIL", "Negative control on conv.conv must FAIL!"
    assert res_neg["gradient_ok_count"] == 0
    assert res_neg["gradient_missing_count"] == 26  # 13 conv layers x 2 (lora_A, lora_B)

    print(
        f"[PASS] Step 3b: Mandatory Negative Control [conv.conv] -> "
        f"{res_neg['gradient_missing_count']}/{res_neg['adapter_param_count']} GRADIENT_MISSING "
        f"(CONFIRMED_INERT under packed SFT)"
    )

    pilot_results = None
    if run_sft_pilot:
        tokenizer = AutoTokenizer.from_pretrained(
            LFM3_MODEL_ID, revision=LFM3_REVISION, trust_remote_code=True
        )
        if tokenizer.pad_token is None:
            tokenizer.pad_token = tokenizer.eos_token
        pilot_results = run_lfm3_sft_pilot(
            model=model,
            tokenizer=tokenizer,
            grammar=grammar,
            configs=[
                ("P1_Uniform_r8", p1_uniform, parity_p1),
                ("P2_Global_Exact_Parity", p2_global_exact, parity_p2),
                ("P3_Operator_Prioritized", p3_operators, parity_p3),
            ],
            device=device,
            steps=steps,
        )

    del model
    gc.collect()
    if device == "cuda":
        torch.cuda.empty_cache()

    summary = {
        "schema_version": "1.0",
        "timestamp_utc": datetime.datetime.now(datetime.timezone.utc).isoformat(),
        "model_id": LFM3_MODEL_ID,
        "revision": LFM3_REVISION,
        "device": device,
        "hardware": gpu_name,
        "full_weights_loaded": bool(full_weights or run_sft_pilot),
        "architecture": {
            "family": desc.family,
            "topology": desc.topology,
            "num_layers": desc.num_layers,
            "total_params": desc.total_params,
            "valid_targets_by_group": groups_by_type,
            "total_valid_targets": 142,
            "excluded_count": len(grammar.excluded),
            "conv_excluded_count": len(conv_excluded),
            "gate_excluded_count": len(gate_excluded),
            "search_space_cardinality": grammar.cardinality(),
        },
        "budget_parity": {
            "P1_Uniform_r8_params": p1_params,
            "P2_Global_Exact_Parity": parity_p2,
            "P3_Operator_Prioritized": parity_p3,
        },
        "gradient_verification": {
            "components": component_grad_results,
            "negative_control_conv_conv": {
                "target_module_count": len(conv_targets),
                "adapter_param_count": res_neg["adapter_param_count"],
                "gradient_ok_count": res_neg["gradient_ok_count"],
                "gradient_missing_count": res_neg["gradient_missing_count"],
                "verdict": "CONFIRMED_INERT",
            },
        },
        "sft_pilot": pilot_results,
        "total_elapsed_sec": round(time.perf_counter() - t0, 3),
        "status": "PASS",
    }

    if output_json is not None:
        output_json.parent.mkdir(parents=True, exist_ok=True)
        output_json.write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")
        print(f"[INFO] Saved verification report to {output_json}")

    print("=" * 76)
    print(f"ALL LFM3-1.7B CHECKS PASSED IN {summary['total_elapsed_sec']:.2f}s")
    print("=" * 76)
    return summary


def main() -> int:
    parser = argparse.ArgumentParser(description="Run ALOE verification suite on LiquidAI/LFM3-1.7B.")
    parser.add_argument(
        "--full-weights",
        action="store_true",
        help="Download and load full 1.99B bfloat16 weights for gradient check (recommended on Colab GPU).",
    )
    parser.add_argument(
        "--run-sft-pilot",
        action="store_true",
        help="Also run 25-step budget-matched SFT + AST tool-calling evaluation on LiquidAI/LFM3-1.7B.",
    )
    parser.add_argument(
        "--steps",
        type=int,
        default=25,
        help="Number of SFT training steps per configuration when --run-sft-pilot is enabled.",
    )
    parser.add_argument(
        "--out",
        type=Path,
        default=Path("runs/gradients/LiquidAI--LFM3-1.7B/gradient_report.json"),
        help="Path to write JSON verification report.",
    )
    args = parser.parse_args()
    run_lfm3_verification(
        full_weights=args.full_weights,
        run_sft_pilot=args.run_sft_pilot,
        steps=args.steps,
        output_json=args.out,
    )
    return 0


if __name__ == "__main__":
    sys.exit(main())
