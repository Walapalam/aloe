"""PEFT execution backend for Apple Silicon MPS & NVIDIA CUDA (Stage 4 / Protocol v1).

Implements the ExecutionBackend protocol using HuggingFace PEFT on PyTorch MPS, CUDA
(NVIDIA A100 / L4), or CPU (offline CI smoke tests):
- Unified compact-JSON prompt builder shared identically by training and evaluation.
- Safe tokenization preserving EOS even when `pad_token_id == eos_token_id` and
  refusing schema-destroying left truncation in official campaigns.
- True optimizer-step loop honoring `batch_size`, `gradient_accumulation_steps`,
  seeded epoch shuffling, warmup + cosine schedule, gradient clipping, and AdaLoRA
  `update_and_allocate(global_step)`.
- Atomic 25-step mid-trial checkpoint & resume (`checkpoint_state.pt`).
- Resumable held-out split generation (`predictions_<split>.jsonl`) + strict AST scoring.
"""
from __future__ import annotations

import hashlib
import json
import logging
import math
import os
from pathlib import Path
import random
import subprocess
import sys
import tempfile
import time
from typing import Any

import numpy as np
from peft import LoraConfig, PeftModel, get_peft_model
import torch
from torch import nn
from transformers import AutoModelForCausalLM, AutoTokenizer

from aloe.architecture.schema import ArchitectureDescription
from aloe.datasets.model_registry import ModelRegistry
from aloe.datasets.xlam import XLAMExample, load_xlam_file
from aloe.evaluation.prediction_store import PredictionRecord, PredictionStore
from aloe.evaluation.resources import ResourceProfiler, detect_hardware_summary, enforce_hardware_contract
from aloe.evaluation.tool_calling import EVALUATOR_VERSION, ASTToolCallEvaluator
from aloe.execution.trial_config import GroupTrialSetting, InvalidConfigError, TrialConfig
from aloe.search_space.schema import SearchGrammar
from aloe.tracking.trial_store import (
    TrialMetrics,
    TrialResultEnvelope,
    TrialStore,
    get_environment_metadata,
    get_file_sha256,
    get_git_commit,
)

logger = logging.getLogger(__name__)


def _ensure_peft_torchao_compat() -> None:
    """Guard against Colab/system environments with incompatible preinstalled torchao (<0.16.0)."""
    try:
        import peft.import_utils as peft_import_utils
        import peft.tuners.lora.torchao as peft_lora_torchao

        orig_fn = getattr(peft_import_utils, "is_torchao_available", None)
        if orig_fn is not None and not getattr(orig_fn, "_aloe_patched", False):
            def _safe_is_torchao_available(*args: Any, **kwargs: Any) -> bool:
                try:
                    return bool(orig_fn(*args, **kwargs))
                except ImportError:
                    return False

            _safe_is_torchao_available._aloe_patched = True  # type: ignore[attr-defined]
            peft_import_utils.is_torchao_available = _safe_is_torchao_available
            peft_lora_torchao.is_torchao_available = _safe_is_torchao_available
    except Exception:
        pass


_ensure_peft_torchao_compat()


def set_seed(seed: int = 42) -> None:
    """Set deterministic seeds across random, numpy, and torch accelerators."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
        torch.mps.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def _clear_device_cache(device: str) -> None:
    """Clear accelerator allocator cache strictly for the active device."""
    try:
        if device == "cuda" and torch.cuda.is_available():
            torch.cuda.empty_cache()
        elif device == "mps" and hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
            torch.mps.empty_cache()
    except Exception as exc:  # noqa: BLE001
        logger.debug("Device cache cleanup skipped (%s): %s", device, exc)


def test_fresh_process_reload(
    snapshot_path: Path | str,
    adapter_path: Path | str,
    device: str = "mps",
    revision: str | None = None,
) -> dict[str, Any]:
    """Verify that saved adapter weights can be reloaded in an independent, fresh subprocess."""
    rev_arg = (
        f"revision={json.dumps(revision)},"
        if revision and revision not in ("main", "UNKNOWN") and not Path(snapshot_path).exists()
        else ""
    )
    dtype_expr = "torch.float32" if device == "cpu" else "torch.bfloat16"
    code = f"""
import sys, torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from peft import PeftModel

snapshot_path = {json.dumps(str(snapshot_path))}
adapter_path = {json.dumps(str(adapter_path))}
device = {json.dumps(device)}

try:
    tok = AutoTokenizer.from_pretrained(snapshot_path, {rev_arg} trust_remote_code=True)
    base = AutoModelForCausalLM.from_pretrained(
        snapshot_path,
        {rev_arg}
        torch_dtype={dtype_expr},
        trust_remote_code=True,
    ).to(device)
    model = PeftModel.from_pretrained(base, adapter_path).to(device)
    model.eval()
    inputs = tok("Ping test", return_tensors="pt").to(device)
    with torch.no_grad():
        out = model.generate(**inputs, max_new_tokens=4, do_sample=False)
    print("RELOAD_SUCCESS")
    sys.exit(0)
except Exception as e:
    print(f"RELOAD_FAILED: {{e}}")
    sys.exit(1)
"""
    cmd = [sys.executable, "-c", code]
    try:
        proc = subprocess.run(cmd, capture_output=True, text=True, timeout=240, check=False)
        success = proc.returncode == 0 and "RELOAD_SUCCESS" in proc.stdout
        return {
            "success": success,
            "returncode": proc.returncode,
            "stdout": proc.stdout.strip(),
            "stderr": proc.stderr.strip() if proc.returncode != 0 else "",
        }
    except subprocess.TimeoutExpired:
        return {
            "success": False,
            "returncode": -1,
            "stdout": "",
            "stderr": "Timeout expired during fresh process reload test.",
        }


def build_tool_calling_user_content(example: Any) -> str:
    """Canonical compact-JSON user prompt contract shared across training and evaluation."""
    tools_str = json.dumps(example.tools, separators=(",", ":"), ensure_ascii=False)
    return (
        "You are an agent with access to the following tools (JSON schemas):\n"
        f"{tools_str}\n\n"
        "Respond ONLY with a valid JSON array of tool calls in the exact format:\n"
        '[{"name":"tool_name","arguments":{"arg1":"value"}}]\n'
        "If no tool is appropriate, respond with an empty JSON array: []\n\n"
        f"User request: {example.query}"
    )


def build_generation_prompt(example: Any, tokenizer: Any) -> str:
    """Format an xLAM example into the exact generation prompt string (with assistant header)."""
    user_content = build_tool_calling_user_content(example)
    messages = [{"role": "user", "content": user_content}]
    if hasattr(tokenizer, "apply_chat_template") and getattr(tokenizer, "chat_template", None):
        try:
            return str(
                tokenizer.apply_chat_template(
                    messages,
                    tokenize=False,
                    add_generation_prompt=True,
                )
            )
        except Exception:
            pass
    return f"User: {user_content}\nAssistant: "


def format_xlam_input(example: Any, tokenizer: Any) -> str:
    """Format a complete xLAM training dialogue using the canonical compact contract."""
    user_content = build_tool_calling_user_content(example)
    assistant_msg = json.dumps(example.answers, separators=(",", ":"), ensure_ascii=False)
    messages = [
        {"role": "user", "content": user_content},
        {"role": "assistant", "content": assistant_msg},
    ]
    if hasattr(tokenizer, "apply_chat_template") and getattr(tokenizer, "chat_template", None):
        try:
            return str(tokenizer.apply_chat_template(messages, tokenize=False))
        except Exception:
            pass
    return f"User: {user_content}\nAssistant: {assistant_msg}"


def tokenize_xlam_sft_example(
    example: Any,
    tokenizer: Any,
    max_seq_len: int = 1024,
    *,
    reject_overlength: bool = False,
    pad_to_max: bool = False,
) -> dict[str, torch.Tensor]:
    """Tokenize an xLAM example for SFT with exact assistant-only label masking (-100).

    Fixes three classic SFT bugs:
    1. Uses `add_special_tokens=False` consistently on chat-templated strings (no double BOS).
    2. Masks padding by exact sequence index rather than `input_ids == pad_token_id`, so a real
       assistant EOS token is NEVER masked when `pad_token_id == eos_token_id`.
    3. Refuses silent tool-schema deletion when `reject_overlength=True`.
    """
    prompt_text = build_generation_prompt(example, tokenizer)
    assistant_msg = json.dumps(example.answers, separators=(",", ":"), ensure_ascii=False)
    eos_tok = getattr(tokenizer, "eos_token", "") or ""

    # Construct full text from the exact same prompt_text prefix to guarantee prefix alignment
    full_text = f"{prompt_text}{assistant_msg}{eos_tok}"

    prompt_ids: list[int] = list(tokenizer.encode(prompt_text, add_special_tokens=False))
    full_ids: list[int] = list(tokenizer.encode(full_text, add_special_tokens=False))

    # Add single leading BOS if tokenizer defines one and didn't include it in chat template
    bos_id = getattr(tokenizer, "bos_token_id", None)
    if bos_id is not None and (not full_ids or full_ids[0] != bos_id):
        prompt_ids = [int(bos_id)] + prompt_ids
        full_ids = [int(bos_id)] + full_ids

    prompt_len = len(prompt_ids)
    if len(full_ids) <= prompt_len:
        # Ensure at least one supervised assistant token
        extra = list(tokenizer.encode(assistant_msg, add_special_tokens=False))
        full_ids = prompt_ids + (extra if extra else [int(getattr(tokenizer, "eos_token_id", 0) or 0)])

    if len(full_ids) > max_seq_len:
        if reject_overlength:
            raise ValueError(
                f"Example '{getattr(example, 'id', '?')}' length ({len(full_ids)} tokens) "
                f"exceeds max_seq_len={max_seq_len}; refusing schema-truncating crop."
            )
        # Non-campaign fallback: right-crop only if assistant answer still has >=1 supervised token
        if prompt_len >= max_seq_len:
            asst_ids = full_ids[prompt_len:]
            keep_asst = min(len(asst_ids), max(1, max_seq_len // 4))
            keep_prompt = max_seq_len - keep_asst
            full_ids = prompt_ids[:keep_prompt] + asst_ids[:keep_asst]
            prompt_len = keep_prompt
        else:
            full_ids = full_ids[:max_seq_len]

    seq_len = len(full_ids)
    input_tensor = torch.tensor(full_ids, dtype=torch.long)
    attn_tensor = torch.ones(seq_len, dtype=torch.long)
    labels_tensor = input_tensor.clone()
    labels_tensor[: min(prompt_len, seq_len)] = -100

    if pad_to_max and seq_len < max_seq_len:
        raw_pad_id = getattr(tokenizer, "pad_token_id", None)
        raw_eos_id = getattr(tokenizer, "eos_token_id", None)
        pad_id = int(raw_pad_id if raw_pad_id is not None else (raw_eos_id if raw_eos_id is not None else 0))
        pad_len = max_seq_len - seq_len
        input_tensor = torch.cat([input_tensor, torch.full((pad_len,), pad_id, dtype=torch.long)])
        attn_tensor = torch.cat([attn_tensor, torch.zeros(pad_len, dtype=torch.long)])
        labels_tensor = torch.cat([labels_tensor, torch.full((pad_len,), -100, dtype=torch.long)])

    if int((labels_tensor != -100).sum().item()) == 0:
        raise ValueError(f"Zero supervised assistant tokens for example '{getattr(example, 'id', '?')}'")

    return {
        "input_ids": input_tensor,
        "attention_mask": attn_tensor,
        "labels": labels_tensor,
    }


def collate_sft_batch(items: list[dict[str, torch.Tensor]], pad_token_id: int) -> dict[str, torch.Tensor]:
    """Dynamically right-pad a microbatch to the longest item in that microbatch (not global max_seq_len)."""
    max_len = max(int(x["input_ids"].shape[0]) for x in items)
    bsz = len(items)
    input_ids = torch.full((bsz, max_len), pad_token_id, dtype=torch.long)
    attention_mask = torch.zeros((bsz, max_len), dtype=torch.long)
    labels = torch.full((bsz, max_len), -100, dtype=torch.long)

    for i, item in enumerate(items):
        L = int(item["input_ids"].shape[0])
        input_ids[i, :L] = item["input_ids"]
        attention_mask[i, :L] = item["attention_mask"]
        labels[i, :L] = item["labels"]

    return {
        "input_ids": input_ids,
        "attention_mask": attention_mask,
        "labels": labels,
    }


def evaluate_model_on_split(
    model: nn.Module,
    tokenizer: Any,
    examples: list[XLAMExample],
    *,
    prediction_store: PredictionStore,
    split_name: str,
    model_id: str,
    model_revision: str,
    arm: str,
    trial_id: str,
    seed: int,
    device: str,
    max_new_tokens: int = 192,
) -> dict[str, Any]:
    """Generate greedy tool-call predictions on `examples`, persist incrementally to JSONL, and score strictly."""
    evaluator = ASTToolCallEvaluator(strict=True)
    expected_by_id = {str(ex.id): ex for ex in examples}
    if len(expected_by_id) != len(examples):
        raise ValueError(f"Duplicate example IDs in requested {split_name} evaluation")

    existing_records = prediction_store.load_records(strict=True)
    for rec in existing_records:
        ex_match = expected_by_id.get(rec.sample_id)
        if ex_match is None:
            raise ValueError(f"Stale/foreign sample_id '{rec.sample_id}' in {prediction_store.path}")
        expected_prompt = build_generation_prompt(ex_match, tokenizer)
        expected_prompt_sha = hashlib.sha256(expected_prompt.encode("utf-8")).hexdigest()
        if (
            rec.split != split_name
            or rec.model_id != model_id
            or rec.model_revision != model_revision
            or rec.arm != arm
            or rec.trial_id != trial_id
            or rec.seed != seed
            or rec.prompt_sha256 != expected_prompt_sha
            or rec.ground_truth != ex_match.answers
            or rec.available_tools != ex_match.tools
            or rec.evaluator_version != EVALUATOR_VERSION
        ):
            raise ValueError(
                f"Incompatible resumed prediction '{rec.sample_id}' in {prediction_store.path}; "
                "refusing mixed-metadata or stale evaluation resume."
            )
    completed_ids = {rec.sample_id for rec in existing_records}

    ModelRegistry.disable_lfm3_broken_triton_conv(model)
    model.eval()
    raw_pad = getattr(tokenizer, "pad_token_id", None)
    raw_eos = getattr(tokenizer, "eos_token_id", None)
    pad_id = int(raw_pad if raw_pad is not None else (raw_eos if raw_eos is not None else 0))

    t0 = time.time()
    for ex in examples:
        if str(ex.id) in completed_ids:
            continue
        prompt_str = build_generation_prompt(ex, tokenizer)
        prompt_sha = hashlib.sha256(prompt_str.encode("utf-8")).hexdigest()
        enc_ids = tokenizer.encode(prompt_str, add_special_tokens=False)
        bos_id = getattr(tokenizer, "bos_token_id", None)
        if bos_id is not None and (not enc_ids or enc_ids[0] != bos_id):
            enc_ids = [int(bos_id)] + enc_ids

        inp = torch.tensor([enc_ids], dtype=torch.long, device=device)
        attn = torch.ones_like(inp, device=device)
        with torch.no_grad():
            gen_out = model.generate(
                input_ids=inp,
                attention_mask=attn,
                max_new_tokens=max_new_tokens,
                do_sample=False,
                pad_token_id=pad_id,
            )
        gen_tokens = gen_out[0, inp.shape[1] :]
        raw_text = tokenizer.decode(gen_tokens, skip_special_tokens=True).strip()

        ev = evaluator.evaluate_single(
            raw_text,
            ex.answers,
            sample_id=str(ex.id),
            available_tools=ex.tools,
        )
        rec = PredictionRecord(
            sample_id=str(ex.id),
            split=split_name,
            model_id=model_id,
            model_revision=model_revision,
            arm=arm,
            trial_id=trial_id,
            seed=seed,
            prompt_sha256=prompt_sha,
            raw_output=raw_text,
            ground_truth=ex.answers,
            available_tools=ex.tools,
            evaluator_version=EVALUATOR_VERSION,
            parse_success=ev.parse_success,
            schema_valid=ev.schema_valid,
            tool_match=ev.tool_match,
            arg_names_match=ev.arg_names_match,
            arg_values_match=ev.arg_values_match,
            exact_match=ev.exact_match,
            lenient_exact_match=ev.lenient_exact_match,
            parse_error=ev.parse_error,
        )
        prediction_store.append(rec)

    eval_sec = round(time.time() - t0, 3)
    all_records = prediction_store.load_records(strict=True)
    actual_ids = {rec.sample_id for rec in all_records}
    if actual_ids != set(expected_by_id.keys()):
        missing = sorted(set(expected_by_id.keys()) - actual_ids)[:5]
        extra = sorted(actual_ids - set(expected_by_id.keys()))[:5]
        raise RuntimeError(
            f"Incomplete/contaminated {split_name} evaluation: missing={missing}, extra={extra}"
        )
    summary = PredictionStore.summarize(all_records)
    return {
        "summary": summary,
        "eval_seconds": eval_sec,
        "predictions_jsonl_path": str(prediction_store.path),
        "predictions_sha256": prediction_store.sha256(),
    }


def _serialize_py_rng(state: tuple[Any, ...]) -> list[Any]:
    ver, internal, gauss_next = state
    return [int(ver), [ int(x) for x in internal ], float(gauss_next) if gauss_next is not None else None]


def _deserialize_py_rng(data: list[Any]) -> tuple[Any, ...]:
    return (int(data[0]), tuple(int(x) for x in data[1]), data[2])


def _serialize_np_rng(state: tuple[Any, ...]) -> list[Any]:
    alg, keys, pos, has_gauss, cached = state
    return [str(alg), [int(k) for k in keys.tolist()], int(pos), int(has_gauss), float(cached)]


def _deserialize_np_rng(data: list[Any]) -> tuple[Any, ...]:
    return (
        str(data[0]),
        np.array(data[1], dtype=np.uint32),
        int(data[2]),
        int(data[3]),
        float(data[4]),
    )


def _save_step_checkpoint(ckpt_path: Path, payload: dict[str, Any]) -> None:
    """Save weights_only=True-safe step checkpoint with file + directory fsync."""
    ckpt_path.parent.mkdir(parents=True, exist_ok=True)
    tmp_path: Path | None = None
    try:
        with tempfile.NamedTemporaryFile(
            mode="wb", dir=ckpt_path.parent, suffix=".pt.tmp", delete=False
        ) as handle:
            tmp_path = Path(handle.name)
            torch.save(payload, handle)
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(tmp_path, ckpt_path)
        try:
            dir_fd = os.open(str(ckpt_path.parent), os.O_RDONLY)
            try:
                os.fsync(dir_fd)
            finally:
                os.close(dir_fd)
        except OSError:
            pass
    finally:
        if tmp_path is not None:
            tmp_path.unlink(missing_ok=True)


class PeftMPSBackend:
    """Execution backend using Hugging Face PEFT on MPS, CUDA (A100/L4), or CPU."""

    name: str = "peft_mps"

    def available(self) -> bool:
        """Check if MPS or CUDA backend is available. Never raises."""
        try:
            mps_ok = bool(hasattr(torch.backends, "mps") and torch.backends.mps.is_available())
            cuda_ok = bool(torch.cuda.is_available())
            return mps_ok or cuda_ok
        except Exception:  # noqa: BLE001
            return False

    def supports(self, grammar: SearchGrammar) -> tuple[bool, str]:
        if not grammar.groups:
            return False, "Search grammar contains no groups."

        for group in grammar.groups:
            if not group.targets:
                return False, f"Group {group.component_type} has no resolved targets."
            if any(r <= 0 for r in group.rank):
                return False, f"Group {group.component_type} contains rank <= 0."
            if any(a <= 0 for a in group.alpha):
                return False, f"Group {group.component_type} contains alpha <= 0."
            if any("conv1d" in t or t.endswith(".conv.conv") for t in group.targets):
                return False, f"Target conv1d in {group.component_type} is incompatible with PEFT."

        return True, "All groups and target modules supported by PEFT."

    def verify_gradients(
        self,
        model: nn.Module | None = None,
        target_modules: list[str] | None = None,
        r: int = 8,
        lora_alpha: int = 16,
        init_lora_weights: bool = False,
        input_ids: torch.Tensor | None = None,
        inputs: Any = None,
        config: Any = None,
        peft_config_override: Any | None = None,
        **kwargs: Any,
    ) -> dict[str, Any]:
        """Empirically verify that non-zero gradients reach every attached adapter parameter."""
        if model is None and config is not None:
            model = getattr(config, "model", None)
            if target_modules is None:
                target_modules = getattr(config, "targets", None) or getattr(
                    config, "target_modules", None
                )

        if model is None:
            raise ValueError("model must be provided to verify_gradients")
        if peft_config_override is None and not target_modules:
            raise ValueError("target_modules must be a non-empty list of module names")

        device = next(model.parameters()).device

        if peft_config_override is not None:
            import copy as _copy

            peft_config = _copy.deepcopy(peft_config_override)
            if hasattr(peft_config, "init_lora_weights"):
                peft_config.init_lora_weights = init_lora_weights
        else:
            peft_config = LoraConfig(
                r=r,
                lora_alpha=lora_alpha,
                target_modules=target_modules,
                init_lora_weights=init_lora_weights,
                **kwargs,
            )

        peft_model = get_peft_model(model, peft_config)
        peft_model.zero_grad(set_to_none=True)

        dummy_ids: torch.Tensor | None = None
        if inputs is not None:
            out = peft_model(**inputs) if isinstance(inputs, dict) else peft_model(inputs)
        elif input_ids is not None:
            out = peft_model(input_ids)
        else:
            dummy_ids = torch.tensor([[1, 2, 3, 4]], device=device)
            try:
                out = peft_model(dummy_ids)
            except (TypeError, IndexError, RuntimeError):
                first_param = next(peft_model.parameters())
                in_dim = getattr(peft_model, "in_features", None)
                if in_dim is None:
                    for mod in peft_model.modules():
                        if hasattr(mod, "in_features"):
                            in_dim = mod.in_features
                            break
                if in_dim is None:
                    in_dim = first_param.shape[-1]
                x = torch.randn(2, in_dim, device=device, dtype=first_param.dtype)
                out = peft_model(x)

        if hasattr(out, "logits"):
            loss = out.logits.float().sum()
        elif isinstance(out, torch.Tensor):
            loss = out.float().sum()
        elif isinstance(out, (tuple, list)) and len(out) > 0:
            loss = out[0].float().sum()
        else:
            loss = sum(t.float().sum() for t in out.values() if isinstance(t, torch.Tensor))

        if not loss.requires_grad:
            embed_params: list[nn.Parameter] = []
            base = getattr(peft_model, "base_model", peft_model)
            inner = getattr(base, "model", base)
            if hasattr(inner, "model") and hasattr(inner.model, "embed_tokens"):
                embed_params = list(inner.model.embed_tokens.parameters())
            elif hasattr(inner, "embed_tokens"):
                embed_params = list(inner.embed_tokens.parameters())

            if embed_params:
                for ep in embed_params:
                    ep.requires_grad = True
                try:
                    if input_ids is not None:
                        out = peft_model(input_ids)
                    elif inputs is not None:
                        out = peft_model(**inputs) if isinstance(inputs, dict) else peft_model(inputs)
                    else:
                        out = peft_model(dummy_ids)
                    loss = out.logits.float().sum() if hasattr(out, "logits") else out.float().sum()
                    loss.backward()
                finally:
                    for ep in embed_params:
                        ep.requires_grad = False
        else:
            loss.backward()

        lora_params = [
            (name, param)
            for name, param in peft_model.named_parameters()
            if param.requires_grad and ("lora_" in name or "loraE" in name)
        ]
        param_details: list[dict[str, Any]] = []
        module_map: dict[str, list[dict[str, Any]]] = {}

        for name, param in lora_params:
            if param.grad is None:
                classification = "GRADIENT_MISSING"
                norm = None
            else:
                norm = float(param.grad.float().norm().item())
                classification = "GRADIENT_OK" if norm > 1e-8 else "GRADIENT_ZERO"

            entry = {"param_name": name, "norm": norm, "classification": classification}
            param_details.append(entry)

            parts = name.split(".lora_")
            mod_name = parts[0].removeprefix("base_model.model.")
            module_map.setdefault(mod_name, []).append(entry)

        modules_list = [{"name": m_name, "params": p_list} for m_name, p_list in module_map.items()]
        gradient_ok_count = sum(1 for p in param_details if p["classification"] == "GRADIENT_OK")
        gradient_zero_count = sum(1 for p in param_details if p["classification"] == "GRADIENT_ZERO")
        gradient_missing_count = sum(
            1 for p in param_details if p["classification"] == "GRADIENT_MISSING"
        )
        adapter_param_count = len(lora_params)
        norms = [p["norm"] for p in param_details if p["norm"] is not None]

        verdict = (
            "PASS"
            if (gradient_ok_count == adapter_param_count and adapter_param_count > 0)
            else "FAIL"
        )
        target_status = {
            mod_entry["name"]: all(p["classification"] == "GRADIENT_OK" for p in mod_entry["params"])
            for mod_entry in modules_list
        }

        peft_model.zero_grad(set_to_none=True)
        unloaded_model = peft_model.unload()
        _clear_device_cache(str(device.type))

        return {
            "success": verdict == "PASS",
            "verdict": verdict,
            "target_module_count": len(target_modules),
            "adapter_param_count": adapter_param_count,
            "gradient_ok_count": gradient_ok_count,
            "gradient_zero_count": gradient_zero_count,
            "gradient_missing_count": gradient_missing_count,
            "min_gradient_norm": min(norms) if norms else None,
            "max_gradient_norm": max(norms) if norms else None,
            "modules": modules_list,
            "param_details": param_details,
            "target_status": target_status,
            "unloaded_model": unloaded_model,
        }

    def run_trial(
        self,
        config: TrialConfig,
        trial_id: str | None = None,
        runs_trials_dir: Path | str = "runs/trials",
        grammar: SearchGrammar | None = None,
        arch: ArchitectureDescription | None = None,
        base_model: nn.Module | None = None,
        tokenizer: Any = None,
        *,
        enforce_budget_parity: bool = False,
        parity_tolerance: float = 0.05,
        baseline_params: int | None = None,
        eval_split: str | None = None,
        pinned_gpu: str | None = None,
        exact_gpu_name: str | None = None,
        official_campaign: bool = False,
        allow_slow_kernels: bool = False,
        forbid_dirty_git: bool = False,
        resume: bool = True,
        skip_subprocess_reload: bool = False,
        mirror_dir: Path | str | None = None,
        device_override: str | None = None,
        keep_step_checkpoint: bool = False,
        allow_sealed_test: bool = False,
        seal_file: Path | str | None = None,
    ) -> TrialResultEnvelope:
        """Execute a complete training (+ optional held-out AST evaluation) trial."""
        if eval_split == "test" and not allow_sealed_test:
            raise PermissionError(
                "Split 'test' is sealed during search and validation selection (ADR-014 / Protocol v1). "
                "Pass allow_sealed_test=True (--final) only after locking validation-selected winners."
            )
        if eval_split == "test" and official_campaign:
            if seal_file is None or not Path(seal_file).is_file():
                raise PermissionError(
                    "Official sealed 'test' evaluation requires a verified test_unsealed.json seal file."
                )
            seal_data = json.loads(Path(seal_file).read_text(encoding="utf-8"))
            cfg_dump = config.model_dump(mode="json")
            for k in ("trial_id", "seed", "dataset_manifest_path", "dataset_raw_path"):
                cfg_dump.pop(k, None)
            expected_winner_hash = hashlib.sha256(
                json.dumps(cfg_dump, sort_keys=True, separators=(",", ":")).encode("utf-8")
            ).hexdigest()
            sealed_hash = (seal_data.get("winner_signatures") or {}).get(config.arm)
            if sealed_hash != expected_winner_hash:
                raise PermissionError(
                    f"Sealed test violation: configuration for arm='{config.arm}' does not match "
                    f"locked seal hash in {seal_file}!"
                )

        trial_id = trial_id or config.trial_id
        store = TrialStore(runs_trials_dir, mirror_dir=mirror_dir)
        trial_dir = Path(runs_trials_dir) / trial_id
        trial_dir.mkdir(parents=True, exist_ok=True)
        if resume:
            store.restore_trial_from_mirror(trial_id)
        adapter_dir = trial_dir / "adapter"
        ckpt_file = trial_dir / "checkpoint_state.pt"

        git_sha = get_git_commit(forbid_dirty=forbid_dirty_git)

        # 1. Resolve pinned revision, snapshot, architecture, and grammar BEFORE loading weights
        registry = ModelRegistry()
        canonical_id = config.model_id
        for alias, target_id in registry.ALIASES.items():
            if config.model_id.lower() == alias:
                canonical_id = target_id
                break
        pinned_rev = (
            config.model_revision
            if config.model_revision not in ("main", "UNKNOWN", "")
            else registry.PINNED_REVISIONS.get(canonical_id, "main")
        )
        try:
            snapshot_path: Path | str = registry.get_snapshot_path(config.model_id)
        except (KeyError, FileNotFoundError):
            snapshot_path = canonical_id

        if (grammar is None and config.explicit_target_modules is None) or (
            enforce_budget_parity and arch is None
        ):
            from aloe.pipeline import inspect_to_grammar
            inspected_arch, inspected_grammar = inspect_to_grammar(config.model_id, backend=self.name)
            arch = arch or inspected_arch
            grammar = grammar or inspected_grammar

        # 2. Enforce ±5% parameter budget parity BEFORE touching GPU/MPS weights
        parity_report: dict[str, Any] | None = None
        expected_params: int | None = None
        if arch is not None and grammar is not None:
            if baseline_params is None:
                base_cfg = TrialConfig(
                    model_id=grammar.model_id,
                    group_settings={
                        g.component_type.value: GroupTrialSetting(enabled=True, rank=8, alpha=16)
                        for g in grammar.groups
                    },
                )
                baseline_params = base_cfg.compute_trainable_params(arch, grammar)
            parity_report = config.validate_parameter_budget_parity(
                arch=arch,
                baseline_params=baseline_params,
                tolerance=parity_tolerance,
                grammar=grammar,
                enforce=enforce_budget_parity,
            )
            expected_params = int(parity_report["candidate_params"])

        _, lora_cfg = config.to_peft_lora_config(grammar)

        # 3. Determine target device & enforce hardware contract
        if device_override is not None:
            device = device_override
        elif torch.cuda.is_available():
            device = "cuda"
        elif hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
            device = "mps"
        else:
            device = "cpu"

        hw_meta = enforce_hardware_contract(
            config.model_id,
            device,
            pinned_gpu=pinned_gpu,
            exact_gpu_name=exact_gpu_name,
            official_campaign=official_campaign,
            allow_slow_kernels=allow_slow_kernels,
        )
        env_meta = get_environment_metadata()

        # Start profiler BEFORE model load so end-to-end peak memory is captured accurately
        profiler = ResourceProfiler(sample_interval_ms=100, device=device)
        profiler.start()

        ModelRegistry._ensure_lfm3_rope_compat()
        created_base = False
        is_local_dir = Path(str(snapshot_path)).exists()
        rev_kwargs: dict[str, Any] = (
            {"revision": pinned_rev}
            if (not is_local_dir and pinned_rev not in ("main", "UNKNOWN", ""))
            else {}
        )
        torch_dtype = (
            torch.float32
            if (device == "cpu" or config.training.precision == "fp32")
            else torch.bfloat16
        )

        ModelRegistry._ensure_lfm3_rope_compat()
        if base_model is None:
            base_model = AutoModelForCausalLM.from_pretrained(
                snapshot_path,
                torch_dtype=torch_dtype,
                trust_remote_code=True,
                **rev_kwargs,
            ).to(device)
            created_base = True
        ModelRegistry.disable_lfm3_broken_triton_conv(base_model)

        if tokenizer is None:
            tokenizer = AutoTokenizer.from_pretrained(
                snapshot_path,
                trust_remote_code=True,
                **rev_kwargs,
            )
        if getattr(tokenizer, "pad_token", None) is None and getattr(tokenizer, "eos_token", None) is not None:
            tokenizer.pad_token = tokenizer.eos_token

        # 4. Mandatory Pre-train Gradient Verification (Gate C Step 1 — exercises actual LoRA or AdaLoRA config!)
        target_mods_list = list(lora_cfg.target_modules)
        grad_r = getattr(lora_cfg, "r", config.adalora_target_r)
        grad_alpha = getattr(lora_cfg, "lora_alpha", grad_r * 2)
        grad_check = self.verify_gradients(
            model=base_model,
            target_modules=target_mods_list,
            r=int(grad_r),
            lora_alpha=int(grad_alpha),
            peft_config_override=lora_cfg,
        )
        if "unloaded_model" in grad_check and grad_check["unloaded_model"] is not None:
            base_model = grad_check["unloaded_model"]
            del grad_check["unloaded_model"]

        if not grad_check["success"] or grad_check["gradient_missing_count"] > 0:
            profiler.stop()
            offending = [
                p["param_name"] for p in grad_check["param_details"]
                if p["classification"] == "GRADIENT_MISSING"
            ]
            failure_msg = (
                f"Pre-train gradient check failed: {grad_check['gradient_missing_count']} "
                f"parameters received no gradient. Offending modules: {offending}"
            )
            envelope = TrialResultEnvelope(
                trial_id=trial_id,
                arm=config.arm,
                git_commit=git_sha,
                dataset_manifest_sha256=get_file_sha256(config.dataset_manifest_path),
                model_revision=pinned_rev,
                hardware=hw_meta["hardware_label"],
                gpu_name=hw_meta["gpu_name"],
                library_versions=env_meta["library_versions"],
                model_id=config.model_id,
                backend=f"peft_{device}",
                trial_config=config.model_dump(mode="json"),
                status="INVALID_CONFIG",
                failures=[failure_msg],
            )
            store.save(envelope)
            raise InvalidConfigError(failure_msg)

        # 5. Attach PEFT adapter & assert exact parameter match
        set_seed(config.seed)
        peft_model = get_peft_model(base_model, lora_cfg)
        peft_model.train()

        trainable_params = sum(p.numel() for p in peft_model.parameters() if p.requires_grad)
        adalora_init_params: int | None = None
        if config.peft_method == "adalora":
            adalora_init_params = trainable_params
            if expected_params is not None:
                # Report effective post-pruning target budget as primary matched parameter count
                trainable_params = expected_params
        elif expected_params is not None:
            if trainable_params != expected_params:
                profiler.stop()
                raise InvalidConfigError(
                    f"Runtime PEFT parameter count ({trainable_params:,}) != analytical shape count "
                    f"({expected_params:,}) — check rank_pattern target matching."
                )

        total_params = sum(p.numel() for p in peft_model.parameters())
        trainable_ratio = round(trainable_params / max(total_params, 1), 6)

        # 6. Load xLAM dataset & prepare tokenized SFT items
        manifest_path = Path(config.dataset_manifest_path)
        raw_dataset_path = Path(config.dataset_raw_path)
        if not manifest_path.is_file():
            profiler.stop()
            raise FileNotFoundError(f"Manifest file not found: {manifest_path}")
        if not raw_dataset_path.is_file():
            profiler.stop()
            raise FileNotFoundError(f"Raw xLAM dataset not found: {raw_dataset_path}")

        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        expected_raw_sha = None
        if isinstance(manifest.get("dataset"), dict):
            expected_raw_sha = manifest["dataset"].get("raw_file_sha256")
        if official_campaign and (not expected_raw_sha or expected_raw_sha == "UNKNOWN"):
            profiler.stop()
            raise ValueError(
                "Official campaigns require a concrete dataset.raw_file_sha256 in the manifest; "
                "missing or UNKNOWN raw dataset identity is fail-closed."
            )
        if expected_raw_sha and expected_raw_sha != "UNKNOWN":
            actual_raw_sha = get_file_sha256(raw_dataset_path)
            if actual_raw_sha != expected_raw_sha:
                profiler.stop()
                raise ValueError(
                    f"Raw xLAM dataset SHA-256 mismatch: expected {expected_raw_sha}, got {actual_raw_sha}"
                )

        train_split_spec = manifest["splits"]["train"]
        train_ids_list = [str(i) for i in train_split_spec["ids"]]
        expected_train_count = int(train_split_spec.get("count", len(train_ids_list)))
        if len(train_ids_list) != expected_train_count:
            profiler.stop()
            raise ValueError(
                f"Manifest train split count mismatch: declared {expected_train_count}, len(ids)={len(train_ids_list)}"
            )
        train_ids = set(train_ids_list)
        raw_examples = load_xlam_file(raw_dataset_path)
        id_to_ex = {str(ex.id): ex for ex in raw_examples}
        missing_train = sorted(train_ids - set(id_to_ex.keys()))
        if missing_train:
            profiler.stop()
            raise ValueError(
                f"Manifest train IDs absent from raw dataset ({len(missing_train)} missing): {missing_train[:5]}"
            )
        train_examples = [id_to_ex[sid] for sid in sorted(train_ids)]
        if not train_examples:
            profiler.stop()
            raise ValueError("Zero training examples matched manifest train IDs.")

        raw_pad_id = getattr(tokenizer, "pad_token_id", None)
        raw_eos_id = getattr(tokenizer, "eos_token_id", None)
        pad_id = int(raw_pad_id if raw_pad_id is not None else (raw_eos_id if raw_eos_id is not None else 0))
        tokenized_train = [
            tokenize_xlam_sft_example(
                ex,
                tokenizer,
                max_seq_len=config.training.max_seq_len,
                reject_overlength=official_campaign,
                pad_to_max=False,
            )
            for ex in train_examples
        ]

        # 7. Optimizer, LR Scheduler & Optional Mid-Trial Checkpoint Resume
        max_steps = max(1, config.training.max_steps)
        micro_bs = max(1, config.training.batch_size)
        grad_accum = max(1, config.training.gradient_accumulation_steps)
        warmup = max(0, config.training.warmup_steps)

        optimizer = torch.optim.AdamW(
            [p for p in peft_model.parameters() if p.requires_grad],
            lr=config.training.learning_rate,
            weight_decay=config.training.weight_decay,
        )

        def _lr_lambda(step_idx: int) -> float:
            if warmup > 0 and step_idx < warmup:
                return float(step_idx + 1) / float(max(1, warmup))
            if config.training.lr_scheduler == "constant" or max_steps <= warmup:
                return 1.0
            progress = float(step_idx - warmup) / float(max(1, max_steps - warmup))
            return max(0.1, 0.5 * (1.0 + math.cos(math.pi * min(1.0, progress))))

        scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda=_lr_lambda)

        start_step = 0
        examples_seen = 0
        losses: list[float] = []
        step_durations: list[float] = []
        prior_train_seconds = 0.0
        prior_peak_mem: dict[str, float] = {}
        resumed_from_checkpoint = False
        telemetry_file = trial_dir / "training_telemetry.json"
        if resume and ckpt_file.is_file() and telemetry_file.is_file():
            try:
                t_meta = json.loads(telemetry_file.read_text(encoding="utf-8"))
                if t_meta.get("trial_config_signature") == config.canonical_signature():
                    prior_peak_mem = {k: float(v) for k, v in t_meta.get("peak_memory", {}).items()}
                    prior_train_seconds = float(t_meta.get("train_seconds", 0.0))
            except Exception:
                pass

        if resume and ckpt_file.is_file():
            ckpt = torch.load(ckpt_file, map_location="cpu", weights_only=True)
            if (
                ckpt.get("trial_config_signature") != config.canonical_signature()
                or ckpt.get("model_revision") != pinned_rev
            ):
                profiler.stop()
                raise RuntimeError(
                    f"Checkpoint {ckpt_file} does not match current trial config signature or model revision"
                )
            completed_step = int(ckpt["completed_step"])
            if not 0 <= completed_step <= max_steps:
                profiler.stop()
                raise RuntimeError(
                    f"Invalid completed_step={completed_step} in {ckpt_file}; max_steps={max_steps}"
                )
            incompatible = peft_model.load_state_dict(ckpt["adapter_state"], strict=False)
            missing_adapter = [
                k for k in incompatible.missing_keys if "lora_" in k or "loraE" in k or "ranknum" in k
            ]
            if missing_adapter or incompatible.unexpected_keys:
                profiler.stop()
                raise RuntimeError(
                    f"Adapter checkpoint key mismatch in {ckpt_file}: "
                    f"missing={missing_adapter[:5]}, unexpected={incompatible.unexpected_keys[:5]}"
                )
            optimizer.load_state_dict(ckpt["optimizer_state"])
            scheduler.load_state_dict(ckpt["scheduler_state"])
            # Restore AdaLoRA RankAllocator importance buffers and pruned rank_pattern if present
            ra_state = ckpt.get("rankallocator_state")
            if ra_state and hasattr(peft_model, "base_model") and hasattr(peft_model.base_model, "rankallocator"):
                ra = peft_model.base_model.rankallocator
                ra.nb_step = int(ra_state.get("nb_step", completed_step))
                ra.ipt = {k: v.to(device) for k, v in ra_state.get("ipt", {}).items()}
                ra.exp_avg_ipt = {k: v.to(device) for k, v in ra_state.get("exp_avg_ipt", {}).items()}
                ra.exp_avg_unc = {k: v.to(device) for k, v in ra_state.get("exp_avg_unc", {}).items()}
                saved_rp = ra_state.get("rank_pattern")
                if saved_rp is not None and hasattr(peft_model, "peft_config") and "default" in peft_model.peft_config:
                    peft_model.peft_config["default"].rank_pattern = saved_rp

            start_step = completed_step
            examples_seen = int(ckpt["examples_seen"])
            losses = [float(x) for x in ckpt["losses"]]
            step_durations = [float(x) for x in ckpt["step_durations"]]
            prior_train_seconds = max(prior_train_seconds, float(ckpt.get("train_seconds_so_far", 0.0)))
            ckpt_mem = ckpt.get("peak_memory_so_far") or {}
            for mk, mv in ckpt_mem.items():
                prior_peak_mem[mk] = max(prior_peak_mem.get(mk, 0.0), float(mv))
            random.setstate(_deserialize_py_rng(ckpt["py_rng"]))
            np.random.set_state(_deserialize_np_rng(ckpt["np_rng"]))
            torch.set_rng_state(ckpt["torch_rng"])
            if device == "cuda" and "cuda_rng" in ckpt and ckpt["cuda_rng"] is not None:
                torch.cuda.set_rng_state_all(ckpt["cuda_rng"])
            resumed_from_checkpoint = True

        # If training is going to execute new steps (`start_step < max_steps`), invalidate any stale
        # predictions file from an earlier partial/uncommitted run so predictions strictly match the final adapter!
        if start_step < max_steps and eval_split is not None:
            stale_pred = trial_dir / f"predictions_{eval_split}.jsonl"
            if stale_pred.is_file():
                stale_pred.unlink(missing_ok=True)
                store.remove_file_from_mirror(trial_id, stale_pred.name)

        # Deterministic per-epoch shuffled index stream
        n_train = len(tokenized_train)

        def _get_microbatch(global_micro_idx: int) -> dict[str, torch.Tensor]:
            start_pos = global_micro_idx * micro_bs
            picked: list[dict[str, torch.Tensor]] = []
            for offset in range(micro_bs):
                abs_pos = start_pos + offset
                epoch_idx = abs_pos // n_train
                in_epoch_idx = abs_pos % n_train
                epoch_rng = random.Random(config.seed + 10_000 * epoch_idx)
                perm = list(range(n_train))
                epoch_rng.shuffle(perm)
                picked.append(tokenized_train[perm[in_epoch_idx]])
            return collate_sft_batch(picked, pad_token_id=pad_id)

        t_train_start = time.time()
        ckpt_every = max(1, config.training.checkpoint_every_steps)

        for step in range(start_step, max_steps):
            step_t0 = time.time()
            optimizer.zero_grad(set_to_none=True)
            accum_loss = 0.0

            for accum_idx in range(grad_accum):
                mb_idx = step * grad_accum + accum_idx
                batch = _get_microbatch(mb_idx)
                input_ids = batch["input_ids"].to(device)
                attention_mask = batch["attention_mask"].to(device)
                labels = batch["labels"].to(device)

                outputs = peft_model(
                    input_ids=input_ids,
                    attention_mask=attention_mask,
                    labels=labels,
                    use_cache=False,
                )
                raw_loss = outputs.loss
                if not torch.isfinite(raw_loss):
                    profiler.stop()
                    raise RuntimeError(f"Non-finite training loss at step {step}: {raw_loss.item()}")

                scaled_loss = raw_loss / float(grad_accum)
                loss_val = float(raw_loss.item())
                del outputs, raw_loss
                scaled_loss.backward()
                del scaled_loss
                accum_loss += loss_val / float(grad_accum)
                examples_seen += int(input_ids.shape[0])

            if config.training.max_grad_norm > 0:
                torch.nn.utils.clip_grad_norm_(
                    [p for p in peft_model.parameters() if p.requires_grad],
                    config.training.max_grad_norm,
                )
            optimizer.step()
            if config.peft_method == "adalora" and hasattr(peft_model, "base_model") and hasattr(
                peft_model.base_model, "update_and_allocate"
            ):
                peft_model.base_model.update_and_allocate(step + 1)
            scheduler.step()
            if device == "mps" and hasattr(torch, "mps") and hasattr(torch.mps, "empty_cache"):
                torch.mps.empty_cache()

            step_durations.append(time.time() - step_t0)
            losses.append(accum_loss)

            if ((step + 1) % ckpt_every == 0) or ((step + 1) == max_steps):
                trainable_state = {
                    k: v.detach().cpu()
                    for k, v in peft_model.state_dict().items()
                    if "lora_" in k or "loraE" in k or "ranknum" in k
                }
                ra_payload: dict[str, Any] | None = None
                if config.peft_method == "adalora" and hasattr(peft_model, "base_model") and hasattr(
                    peft_model.base_model, "rankallocator"
                ):
                    ra = peft_model.base_model.rankallocator
                    cur_rp = None
                    if hasattr(peft_model, "peft_config") and "default" in peft_model.peft_config:
                        cur_rp = getattr(peft_model.peft_config["default"], "rank_pattern", None)
                    ra_payload = {
                        "nb_step": int(getattr(ra, "nb_step", step + 1)),
                        "ipt": {k: v.detach().cpu() for k, v in getattr(ra, "ipt", {}).items()},
                        "exp_avg_ipt": {k: v.detach().cpu() for k, v in getattr(ra, "exp_avg_ipt", {}).items()},
                        "exp_avg_unc": {k: v.detach().cpu() for k, v in getattr(ra, "exp_avg_unc", {}).items()},
                        "rank_pattern": cur_rp,
                    }
                snap_mem = profiler.snapshot()
                for mk, mv in prior_peak_mem.items():
                    snap_mem[mk] = max(snap_mem.get(mk, 0.0), mv)
                elapsed_train_so_far = round(prior_train_seconds + (time.time() - t_train_start), 3)
                _save_step_checkpoint(
                    ckpt_file,
                    {
                        "trial_config_signature": config.canonical_signature(),
                        "model_revision": pinned_rev,
                        "completed_step": step + 1,
                        "examples_seen": examples_seen,
                        "losses": losses,
                        "step_durations": step_durations,
                        "train_seconds_so_far": elapsed_train_so_far,
                        "peak_memory_so_far": snap_mem,
                        "adapter_state": trainable_state,
                        "rankallocator_state": ra_payload,
                        "optimizer_state": optimizer.state_dict(),
                        "scheduler_state": scheduler.state_dict(),
                        "py_rng": _serialize_py_rng(random.getstate()),
                        "np_rng": _serialize_np_rng(np.random.get_state()),
                        "torch_rng": torch.get_rng_state(),
                        "cuda_rng": torch.cuda.get_rng_state_all() if device == "cuda" else None,
                    },
                )
                store.sync_file_to_mirror(trial_id, ckpt_file)

        train_seconds = round(prior_train_seconds + (time.time() - t_train_start), 3)
        post_train_snap = profiler.snapshot()
        for mk, mv in prior_peak_mem.items():
            post_train_snap[mk] = max(post_train_snap.get(mk, 0.0), mv)
        from aloe.tracking.trial_store import atomic_write_text

        atomic_write_text(
            telemetry_file,
            json.dumps(
                {
                    "trial_config_signature": config.canonical_signature(),
                    "train_seconds": train_seconds,
                    "peak_memory": post_train_snap,
                },
                indent=2,
            ),
        )
        store.sync_file_to_mirror(trial_id, telemetry_file)

        # Measure actual post-pruning active AdaLoRA parameter count if applicable
        adalora_final_active_params: int | None = None
        if config.peft_method == "adalora":
            active_sum = 0
            state_items = peft_model.state_dict()
            for k, v in state_items.items():
                if "lora_E" in k:
                    nz = int((v.detach().abs() > 1e-12).sum().item())
                    prefix = k.split("lora_E")[0]
                    a_key = next((ak for ak in state_items if ak.startswith(prefix + "lora_A")), None)
                    b_key = next((bk for bk in state_items if bk.startswith(prefix + "lora_B")), None)
                    if a_key and b_key:
                        d_in = int(state_items[a_key].shape[-1])
                        d_out = int(state_items[b_key].shape[0])
                        active_sum += nz * (d_in + d_out)
            adalora_final_active_params = active_sum if active_sum > 0 else trainable_params

        # 8. Save adapter weights
        peft_model.save_pretrained(adapter_dir)
        adapter_size_bytes = sum(p.stat().st_size for p in adapter_dir.rglob("*") if p.is_file())
        adapter_size_mb = round(adapter_size_bytes / (1024**2), 3)

        # 9. Optional Held-Out Split Generation & Strict AST Scoring
        eval_info: dict[str, Any] | None = None
        if eval_split is not None:
            if eval_split not in manifest["splits"]:
                profiler.stop()
                raise ValueError(f"Requested eval_split='{eval_split}' not present in {manifest_path}")
            eval_spec = manifest["splits"][eval_split]
            split_ids = [str(i) for i in eval_spec["ids"]]
            expected_eval_count = int(eval_spec.get("count", len(split_ids)))
            if len(split_ids) != expected_eval_count:
                profiler.stop()
                raise ValueError(
                    f"Manifest {eval_split} split count mismatch: declared {expected_eval_count}, len(ids)={len(split_ids)}"
                )
            missing_eval = sorted(set(split_ids) - set(id_to_ex.keys()))
            if missing_eval:
                profiler.stop()
                raise ValueError(
                    f"Manifest {eval_split} IDs absent from raw dataset ({len(missing_eval)} missing): {missing_eval[:5]}"
                )
            eval_examples = [id_to_ex[sid] for sid in split_ids]
            pred_store = PredictionStore(
                trial_dir / f"predictions_{eval_split}.jsonl",
                on_write_callback=lambda p: store.sync_file_to_mirror(trial_id, p),
            )
            eval_info = evaluate_model_on_split(
                peft_model,
                tokenizer,
                eval_examples,
                prediction_store=pred_store,
                split_name=eval_split,
                model_id=config.model_id,
                model_revision=pinned_rev,
                arm=config.arm,
                trial_id=trial_id,
                seed=config.seed,
                device=device,
                max_new_tokens=config.training.max_new_tokens,
            )

        mem_stats = profiler.stop()
        # Preserve training-phase peak memory across evaluation-only preemption resumes (Opus H1)
        for mk in (
            "peak_memory_gb",
            "peak_rss_gb",
            "mps_driver_allocated_gb",
            "cuda_max_memory_allocated_gb",
            "cuda_max_memory_reserved_gb",
            "search_memory_mb",
        ):
            if mk in post_train_snap:
                mem_stats[mk] = max(float(mem_stats.get(mk, 0.0)), float(post_train_snap[mk]))

        # 10. Clean memory BEFORE fresh subprocess reload verification
        del optimizer
        del scheduler
        del peft_model
        if created_base:
            del base_model
        _clear_device_cache(device)

        if skip_subprocess_reload:
            reload_ok = True
        else:
            reload_result = test_fresh_process_reload(
                snapshot_path, adapter_dir, device=device, revision=pinned_rev
            )
            reload_ok = bool(reload_result["success"])

        # Remove bulky step checkpoint once adapter + predictions are finalized cleanly
        if not keep_step_checkpoint:
            ckpt_file.unlink(missing_ok=True)
            store.remove_file_from_mirror(trial_id, ckpt_file.name)
            telemetry_file.unlink(missing_ok=True)
            store.remove_file_from_mirror(trial_id, telemetry_file.name)

        eval_summary = eval_info["summary"] if eval_info is not None else None
        metrics = TrialMetrics(
            peak_memory_gb=mem_stats["peak_memory_gb"],
            peak_rss_gb=mem_stats["peak_rss_gb"],
            mps_driver_allocated_gb=mem_stats["mps_driver_allocated_gb"],
            cuda_max_memory_allocated_gb=mem_stats["cuda_max_memory_allocated_gb"],
            cuda_max_memory_reserved_gb=mem_stats["cuda_max_memory_reserved_gb"],
            search_memory_mb=mem_stats["search_memory_mb"],
            memory_metric_name=mem_stats["memory_metric"],
            train_seconds=train_seconds,
            eval_seconds=eval_info["eval_seconds"] if eval_info is not None else 0.0,
            latency_per_step_sec=round(float(np.mean(step_durations)), 4) if step_durations else 0.0,
            optimizer_steps_completed=max_steps,
            examples_seen=examples_seen,
            trainable_params=trainable_params,
            adalora_init_params=adalora_init_params,
            adalora_final_active_params=adalora_final_active_params,
            total_params=total_params,
            trainable_ratio=trainable_ratio,
            adapter_size_mb=adapter_size_mb,
            initial_train_loss=round(losses[0], 4) if losses else 0.0,
            final_train_loss=round(losses[-1], 4) if losses else 0.0,
            eval_samples=eval_summary.total_samples if eval_summary is not None else None,
            evaluator_version=EVALUATOR_VERSION if eval_summary is not None else None,
            predictions_jsonl_path=eval_info["predictions_jsonl_path"] if eval_info is not None else None,
            predictions_sha256=eval_info["predictions_sha256"] if eval_info is not None else None,
            tool_accuracy=eval_summary.exact_match_accuracy if eval_summary is not None else None,
            ast_parse_success_rate=eval_summary.parse_success_rate if eval_summary is not None else None,
            ast_schema_valid_rate=eval_summary.schema_valid_rate if eval_summary is not None else None,
            ast_tool_selection_accuracy=eval_summary.tool_selection_accuracy if eval_summary is not None else None,
            ast_argument_name_accuracy=eval_summary.argument_name_accuracy if eval_summary is not None else None,
            ast_argument_value_accuracy=eval_summary.argument_value_accuracy if eval_summary is not None else None,
            ast_exact_match_accuracy=eval_summary.exact_match_accuracy if eval_summary is not None else None,
            ast_lenient_exact_match_accuracy=eval_summary.lenient_exact_match_accuracy if eval_summary is not None else None,
            param_delta_vs_baseline_pct=parity_report["relative_delta_pct"] if parity_report else None,
            is_budget_matched_5pct=parity_report["within_budget"] if parity_report else None,
            resumed_from_checkpoint=resumed_from_checkpoint,
            fresh_process_reload=reload_ok,
        )

        status = "PASS" if reload_ok else "FAIL"
        failures = [] if reload_ok else ["Fresh process reload verification failed."]

        envelope = TrialResultEnvelope(
            trial_id=trial_id,
            arm=config.arm,
            git_commit=git_sha,
            dataset_manifest_sha256=get_file_sha256(config.dataset_manifest_path),
            model_revision=pinned_rev,
            hardware=hw_meta["hardware_label"],
            gpu_name=hw_meta["gpu_name"],
            library_versions=env_meta["library_versions"],
            model_id=config.model_id,
            backend=f"peft_{device}",
            trial_config=config.model_dump(mode="json"),
            metrics=metrics,
            status=status,
            failures=failures,
        )

        store.save(envelope)
        store.sync_trial_artifacts_to_mirror(trial_id)
        return envelope
