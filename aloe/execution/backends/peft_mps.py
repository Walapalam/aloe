"""PEFT execution backend for Apple Silicon MPS (Stage 4 / Prompt 9 / Gate C).

Implements the ExecutionBackend protocol for Apple Silicon using HuggingFace PEFT
and PyTorch's MPS backend. Provides empirical gradient verification across
architectural module groups to validate adapter trainability and executes reproducible
trials on tool-calling datasets with true 100ms peak memory profiling.
"""
from __future__ import annotations

import datetime
import json
import logging
import os
from pathlib import Path
import random
import subprocess
import sys
import time
from typing import Any

import numpy as np
from peft import LoraConfig, get_peft_model
import torch
from torch import nn
from transformers import AutoModelForCausalLM, AutoTokenizer

from aloe.datasets.model_registry import ModelRegistry
from aloe.datasets.xlam import load_xlam_file
from aloe.evaluation.resources import ResourceProfiler
from aloe.execution.trial_config import InvalidConfigError, TrialConfig
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


def set_seed(seed: int = 42) -> None:
    """Set deterministic seeds across random, numpy, and torch MPS."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if hasattr(torch, "mps") and torch.backends.mps.is_available():
        torch.mps.manual_seed(seed)


def test_fresh_process_reload(
    snapshot_path: Path | str,
    adapter_path: Path | str,
    device: str = "mps",
) -> dict[str, Any]:
    """Verify that saved adapter weights can be reloaded in an independent, fresh subprocess."""
    code = f"""
import sys, torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from peft import PeftModel

snapshot_path = r"{snapshot_path}"
adapter_path = r"{adapter_path}"
device = "{device}"

try:
    tok = AutoTokenizer.from_pretrained(snapshot_path, trust_remote_code=True)
    base = AutoModelForCausalLM.from_pretrained(
        snapshot_path,
        torch_dtype=torch.bfloat16,
        device_map=device,
        trust_remote_code=True,
    )
    model = PeftModel.from_pretrained(base, adapter_path)
    inputs = tok("Ping test", return_tensors="pt").to(device)
    with torch.no_grad():
        out = model.generate(**inputs, max_new_tokens=5)
    print("RELOAD_SUCCESS")
    sys.exit(0)
except Exception as e:
    print(f"RELOAD_FAILED: {{e}}")
    sys.exit(1)
"""
    cmd = [sys.executable, "-c", code]
    try:
        proc = subprocess.run(cmd, capture_output=True, text=True, timeout=180)
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


def format_xlam_input(example: Any, tokenizer: Any) -> str:
    """Format an xLAM example into chat messages using the tokenizer's chat template."""
    tools_str = json.dumps(example.tools, indent=2)
    system_msg = (
        "You are an agent with access to the following tools:\n"
        f"{tools_str}\n\n"
        "If you need to call a tool, output valid JSON in the format:\n"
        '{"name": "tool_name", "arguments": {"arg1": "value"}}\n'
        "If no tool is required, respond with plain text."
    )
    assistant_msg = json.dumps(example.answers)
    messages = [
        {"role": "user", "content": f"{system_msg}\n\nUser request: {example.query}"},
        {"role": "assistant", "content": assistant_msg},
    ]
    try:
        return tokenizer.apply_chat_template(messages, tokenize=False)
    except Exception:
        # Fallback if system role is rejected by chat template
        combined_prompt = f"{system_msg}\n\nUser request: {example.query}"
        return tokenizer.apply_chat_template(
            [{"role": "user", "content": combined_prompt}, {"role": "assistant", "content": assistant_msg}],
            tokenize=False,
        )


def tokenize_xlam_sft_example(
    example: Any,
    tokenizer: Any,
    max_seq_len: int = 512,
) -> dict[str, torch.Tensor]:
    """Tokenize an xLAM example for SFT with prompt label masking (-100).

    Masks all prompt/instruction tokens and padding tokens with -100 so that
    loss is computed strictly over assistant tool call responses.
    """
    tools_str = json.dumps(example.tools, indent=2)
    system_msg = (
        "You are an agent with access to the following tools:\n"
        f"{tools_str}\n\n"
        "If you need to call a tool, output valid JSON in the format:\n"
        '{"name": "tool_name", "arguments": {"arg1": "value"}}\n'
        "If no tool is required, respond with plain text."
    )
    user_content = f"{system_msg}\n\nUser request: {example.query}"
    assistant_msg = json.dumps(example.answers)

    # 1. Prompt-only text to determine prompt token length
    try:
        prompt_text = tokenizer.apply_chat_template(
            [{"role": "user", "content": user_content}],
            tokenize=False,
            add_generation_prompt=True,
        )
    except Exception:
        prompt_text = f"User: {user_content}\nAssistant: "

    prompt_ids = tokenizer.encode(prompt_text, add_special_tokens=False)
    prompt_len = len(prompt_ids)

    # 2. Full dialogue text
    messages = [
        {"role": "user", "content": user_content},
        {"role": "assistant", "content": assistant_msg},
    ]
    try:
        full_text = tokenizer.apply_chat_template(messages, tokenize=False)
    except Exception:
        full_text = f"User: {user_content}\nAssistant: {assistant_msg}"

    full_ids = tokenizer.encode(full_text, add_special_tokens=False)
    if len(full_ids) > max_seq_len:
        # Left-truncate the long tool-schema prefix so the user query and assistant
        # JSON tool-call response at the end of the sequence are preserved.
        asst_len = max(1, len(full_ids) - prompt_len)
        kept_ids = full_ids[-max_seq_len:]
        kept_prompt_len = max(0, max_seq_len - asst_len)
        input_ids = torch.tensor(kept_ids, dtype=torch.long)
        attention_mask = torch.ones(max_seq_len, dtype=torch.long)
        labels = input_ids.clone()
        labels[:kept_prompt_len] = -100
    else:
        tokens = tokenizer(
            full_text,
            max_length=max_seq_len,
            padding="max_length",
            truncation=True,
            return_tensors="pt",
        )
        input_ids = tokens["input_ids"].squeeze(0)
        attention_mask = tokens["attention_mask"].squeeze(0)
        labels = input_ids.clone()
        mask_len = min(prompt_len, max_seq_len)
        labels[:mask_len] = -100
        if tokenizer.pad_token_id is not None:
            labels[input_ids == tokenizer.pad_token_id] = -100

    return {
        "input_ids": input_ids,
        "attention_mask": attention_mask,
        "labels": labels,
    }


class PeftMPSBackend:
    """Execution backend using Hugging Face PEFT on Apple Silicon (MPS)."""

    name: str = "peft_mps"

    def available(self) -> bool:
        """Check if Apple Silicon MPS backend is available. Never raises."""
        try:
            return bool(hasattr(torch.backends, "mps") and torch.backends.mps.is_available())
        except Exception:  # noqa: BLE001
            return False

    def supports(self, grammar: SearchGrammar) -> tuple[bool, str]:
        """Check whether PEFT on MPS can execute every group in the grammar."""
        if not grammar.groups:
            return False, "Search grammar contains no groups."

        for group in grammar.groups:
            if not group.targets:
                return False, f"Group {group.component_type} has no resolved targets."
            if any(r <= 0 for r in group.rank):
                return False, f"Group {group.component_type} contains rank <= 0."
            if any(a <= 0 for a in group.alpha):
                return False, f"Group {group.component_type} contains alpha <= 0."
            # Check for known incompatible modules
            if any("conv1d" in t for t in group.targets):
                return False, f"Target conv1d in {group.component_type} is incompatible with PEFT."

        return True, "All groups and target modules supported by PEFT on MPS."

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
        **kwargs: Any,
    ) -> dict[str, Any]:
        """CRITICAL: Empirically verify that non-zero gradients reach adapter targets.

        Attaches a LoRA adapter to the specified target modules, executes a forward pass
        and backward(), and classifies every adapter parameter into:
        - GRADIENT_OK: grad exists and norm > 1e-8
        - GRADIENT_ZERO: grad exists and norm <= 1e-8
        - GRADIENT_MISSING: grad is None (parameter bypassed by computational graph)

        Unloads the adapter cleanly prior to returning.
        """
        if model is None and config is not None:
            model = getattr(config, "model", None)
            if target_modules is None:
                target_modules = getattr(config, "targets", None) or getattr(
                    config, "target_modules", None
                )

        if model is None:
            raise ValueError("model must be provided to verify_gradients")
        if not target_modules:
            raise ValueError("target_modules must be a non-empty list of module names")

        device = next(model.parameters()).device

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
            if isinstance(inputs, dict):
                out = peft_model(**inputs)
            else:
                out = peft_model(inputs)
        elif input_ids is not None:
            out = peft_model(input_ids)
        else:
            dummy_ids = torch.tensor([[1, 2, 3, 4]], device=device)
            try:
                out = peft_model(dummy_ids)
            except (TypeError, IndexError, RuntimeError):
                # Fallback for models with custom input requirements
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
            loss = out.logits.sum()
        elif isinstance(out, torch.Tensor):
            loss = out.sum()
        elif isinstance(out, (tuple, list)) and len(out) > 0:
            loss = out[0].sum()
        else:
            loss = sum(t.sum() for t in out.values() if isinstance(t, torch.Tensor))

        if not loss.requires_grad:
            # Model output does not require grad. Attempt to enable grad on embeddings.
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
                        out = (
                            peft_model(**inputs)
                            if isinstance(inputs, dict)
                            else peft_model(inputs)
                        )
                    else:
                        out = peft_model(dummy_ids)
                    loss = out.logits.sum() if hasattr(out, "logits") else out.sum()
                    loss.backward()
                finally:
                    for ep in embed_params:
                        ep.requires_grad = False
        else:
            loss.backward()

        # Classify all adapter parameters
        lora_params = [
            (name, param) for name, param in peft_model.named_parameters() if "lora_" in name
        ]
        param_details: list[dict[str, Any]] = []
        module_map: dict[str, list[dict[str, Any]]] = {}

        for name, param in lora_params:
            if param.grad is None:
                classification = "GRADIENT_MISSING"
                norm = None
            else:
                norm = float(param.grad.norm().item())
                if norm > 1e-8:
                    classification = "GRADIENT_OK"
                else:
                    classification = "GRADIENT_ZERO"

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
        min_norm = min(norms) if norms else None
        max_norm = max(norms) if norms else None

        verdict = (
            "PASS"
            if (gradient_ok_count == adapter_param_count and adapter_param_count > 0)
            else "FAIL"
        )

        target_status: dict[str, bool] = {}
        for mod_entry in modules_list:
            target_status[mod_entry["name"]] = all(
                p["classification"] == "GRADIENT_OK" for p in mod_entry["params"]
            )

        peft_model.zero_grad(set_to_none=True)
        unloaded_model = peft_model.unload()
        if hasattr(torch, "mps") and hasattr(torch.mps, "empty_cache"):
            try:
                torch.mps.empty_cache()
            except Exception as exc:  # noqa: BLE001
                logger.debug("MPS empty_cache failed: %s", exc)

        return {
            "success": verdict == "PASS",
            "verdict": verdict,
            "target_module_count": len(target_modules),
            "adapter_param_count": adapter_param_count,
            "gradient_ok_count": gradient_ok_count,
            "gradient_zero_count": gradient_zero_count,
            "gradient_missing_count": gradient_missing_count,
            "min_gradient_norm": min_norm,
            "max_gradient_norm": max_norm,
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
        base_model: nn.Module | None = None,
        tokenizer: Any = None,
    ) -> TrialResultEnvelope:
        """Execute a complete training trial under a given TrialConfig and record to TrialStore.

        Follows mandatory Prompt 9 Gate C execution protocol:
        1. Pre-train gradient verification check via verify_gradients(). Aborts if GRADIENT_MISSING.
        2. Train on the disjoint xLAM split.
        3. Periodic 100ms background memory sampling (process RSS + MPS driver allocation).
        4. Adapter persistence and fresh subprocess verification.
        5. Empirical cost measurements recorded under Result Schema Envelope 1.0.
        """
        trial_id = trial_id or config.trial_id
        store = TrialStore(runs_trials_dir)
        trial_dir = Path(runs_trials_dir) / trial_id
        trial_dir.mkdir(parents=True, exist_ok=True)
        adapter_dir = trial_dir / "adapter"

        # 1. Resolve grammar and snapshot
        registry = ModelRegistry()
        snapshot_path = registry.get_snapshot_path(config.model_id)

        if grammar is None:
            from aloe.pipeline import inspect_to_grammar
            _, grammar = inspect_to_grammar(config.model_id, backend=self.name)

        # 2. Translate TrialConfig to PEFT LoraConfig (and validate against grammar)
        peft_kwargs, lora_cfg = config.to_peft_lora_config(grammar)

        device = "mps" if (hasattr(torch.backends, "mps") and torch.backends.mps.is_available()) else "cpu"

        # 3. Load base model if not supplied
        created_base = False
        if base_model is None:
            base_model = AutoModelForCausalLM.from_pretrained(
                snapshot_path,
                torch_dtype=torch.bfloat16,
                device_map=device,
                trust_remote_code=True,
            )
            created_base = True

        if tokenizer is None:
            tokenizer = AutoTokenizer.from_pretrained(snapshot_path, trust_remote_code=True)
            if tokenizer.pad_token is None:
                tokenizer.pad_token = tokenizer.eos_token

        # 4. Mandatory Pre-train Gradient Verification (Prompt 9 Step 1)
        logger.info(f"[{trial_id}] Running pre-train gradient check on {len(lora_cfg.target_modules)} targets...")
        grad_check = self.verify_gradients(
            model=base_model,
            target_modules=lora_cfg.target_modules,
            r=lora_cfg.r,
            lora_alpha=lora_cfg.lora_alpha,
        )

        if not grad_check["success"] or grad_check["gradient_missing_count"] > 0:
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
                git_commit=get_git_commit(),
                dataset_manifest_sha256=get_file_sha256(config.dataset_manifest_path),
                hardware="Apple M1 Max (64 GB unified memory), macOS, MPS",
                library_versions=get_environment_metadata()["library_versions"],
                model_id=config.model_id,
                backend=self.name,
                trial_config=config.model_dump(),
                status="INVALID_CONFIG",
                failures=[failure_msg],
            )
            store.save(envelope)
            raise InvalidConfigError(failure_msg)

        # 5. Attach PEFT model
        set_seed(config.seed)
        peft_model = get_peft_model(base_model, lora_cfg)
        peft_model.train()

        trainable_params = sum(p.numel() for p in peft_model.parameters() if p.requires_grad)
        total_params = sum(p.numel() for p in peft_model.parameters())
        trainable_ratio = round(trainable_params / total_params, 6)

        # 6. Load xLAM training split
        manifest_path = Path(config.dataset_manifest_path)
        raw_dataset_path = Path(config.dataset_raw_path)
        if not manifest_path.is_file():
            raise FileNotFoundError(f"Manifest file not found: {manifest_path}")
        if not raw_dataset_path.is_file():
            raise FileNotFoundError(f"Raw xLAM dataset not found: {raw_dataset_path}")

        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        train_ids = set(str(i) for i in manifest["splits"]["train"]["ids"])
        raw_examples = load_xlam_file(raw_dataset_path)
        train_examples = [ex for ex in raw_examples if str(ex.id) in train_ids]

        # Prepare tokenized training items with SFT prompt label masking (-100)
        max_steps = config.training.max_steps
        train_batch = []
        for i in range(max_steps):
            ex = train_examples[i % len(train_examples)]
            batch_item = tokenize_xlam_sft_example(
                ex, tokenizer, max_seq_len=config.training.max_seq_len
            )
            train_batch.append(batch_item)

        # 7. Start Resource Profiler (Prompt 9 Step 3)
        profiler = ResourceProfiler(sample_interval_ms=100)
        profiler.start()
        t_start = time.time()

        # Optimizer
        optimizer = torch.optim.AdamW(
            peft_model.parameters(),
            lr=config.training.learning_rate,
            weight_decay=config.training.weight_decay,
        )

        losses: list[float] = []
        step_durations: list[float] = []

        for step in range(max_steps):
            step_t0 = time.time()
            optimizer.zero_grad()
            batch_item = train_batch[step]
            input_ids = batch_item["input_ids"].unsqueeze(0).to(device)
            attention_mask = batch_item["attention_mask"].unsqueeze(0).to(device)
            labels = batch_item["labels"].unsqueeze(0).to(device)

            outputs = peft_model(
                input_ids=input_ids,
                attention_mask=attention_mask,
                labels=labels,
            )
            loss = outputs.loss
            loss.backward()
            optimizer.step()

            step_durations.append(time.time() - step_t0)
            losses.append(float(loss.item()))

        t_end = time.time()
        mem_stats = profiler.stop()

        train_seconds = round(t_end - t_start, 3)
        latency_per_step_sec = round(float(np.mean(step_durations)), 4) if step_durations else 0.0

        # 8. Save adapter (Prompt 9 Step 4)
        peft_model.save_pretrained(adapter_dir)
        adapter_size_bytes = sum(p.stat().st_size for p in adapter_dir.rglob("*") if p.is_file())
        adapter_size_mb = round(adapter_size_bytes / (1024**2), 3)

        # Free memory before fresh process test
        del peft_model
        if created_base:
            del base_model
        if hasattr(torch, "mps") and hasattr(torch.mps, "empty_cache"):
            torch.mps.empty_cache()

        # 9. Fresh Subprocess Reload Verification (Prompt 9 Step 4)
        reload_result = test_fresh_process_reload(snapshot_path, adapter_dir, device=device)

        # 10. Assemble Result Envelope (Prompt 9 Step 5)
        metrics = TrialMetrics(
            peak_memory_gb=mem_stats["peak_memory_gb"],
            peak_rss_gb=mem_stats["peak_rss_gb"],
            mps_driver_allocated_gb=mem_stats["mps_driver_allocated_gb"],
            train_seconds=train_seconds,
            latency_per_step_sec=latency_per_step_sec,
            trainable_params=trainable_params,
            total_params=total_params,
            trainable_ratio=trainable_ratio,
            adapter_size_mb=adapter_size_mb,
            initial_train_loss=round(losses[0], 4) if losses else 0.0,
            final_train_loss=round(losses[-1], 4) if losses else 0.0,
            fresh_process_reload=reload_result["success"],
        )

        envelope = TrialResultEnvelope(
            trial_id=trial_id,
            timestamp_utc=datetime.datetime.now(datetime.timezone.utc).isoformat(),
            git_commit=get_git_commit(),
            dataset_manifest_sha256=get_file_sha256(config.dataset_manifest_path),
            hardware="Apple M1 Max (64 GB unified memory), macOS, MPS",
            library_versions=get_environment_metadata()["library_versions"],
            model_id=config.model_id,
            backend=self.name,
            trial_config=config.model_dump(),
            metrics=metrics,
            status="PASS" if reload_result["success"] else "FAIL",
            failures=[] if reload_result["success"] else [reload_result.get("stderr", "Fresh reload failed")],
        )

        store.save(envelope)
        return envelope
