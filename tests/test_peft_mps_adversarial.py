"""Adversarial stress-test suite for PEFT on Apple Silicon MPS backend.

Authored by Challenger 1 (challenger_r1_1) to empirically probe:
1. Extreme disparate rank & alpha pairs (r=1 vs r=32 vs r=64, asymmetric alpha scaling).
2. Hierarchical module targeting (same submodule name under different parent blocks).
3. Mathematical initialization dynamics (init_lora_weights=True step 0 vs step 1 gradient flow).
4. Multi-step gradient tracking and genuine in-place weight updates on MPS.
5. Adapter serialization, fresh reload, and exact numerical output equivalence.
6. Mixed-precision bfloat16 gradient flow and optimizer stepping on MPS.
7. MPS memory stability and device synchronization across multiple steps.
8. Invalid rank rejection (r=0 raises ValueError during layer instantiation).
"""
from __future__ import annotations

import copy
import shutil
import tempfile

import pytest
import torch
from peft import LoraConfig, PeftModel, get_peft_model
from torch import nn


class DeepHierarchicalToyModel(nn.Module):
    """Multi-layer model with diverse module names and varying dimensions."""

    def __init__(self, in_features: int = 32, hidden_features: int = 64, out_features: int = 16):
        super().__init__()
        self.in_features = in_features
        self.hidden_features = hidden_features
        self.out_features = out_features

        # Backbone with hierarchical blocks
        self.backbone = nn.ModuleDict({
            "block_0": nn.ModuleDict({
                "dense_attn": nn.Linear(in_features, hidden_features, bias=False),
                "dense_gate": nn.Linear(hidden_features, hidden_features, bias=False),
            }),
            "block_1": nn.ModuleDict({
                "dense_attn": nn.Linear(hidden_features, hidden_features, bias=False),
                "dense_proj": nn.Linear(hidden_features, hidden_features, bias=False),
            }),
        })
        self.output_head = nn.Linear(hidden_features, out_features, bias=False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        h0_attn = torch.relu(self.backbone["block_0"]["dense_attn"](x))
        h0_gate = self.backbone["block_0"]["dense_gate"](h0_attn)
        h1_attn = torch.relu(self.backbone["block_1"]["dense_attn"](h0_gate))
        h1_proj = self.backbone["block_1"]["dense_proj"](h1_attn)
        out = self.output_head(h1_proj)
        return out


@pytest.fixture
def mps_device() -> torch.device:
    if not torch.backends.mps.is_available():
        pytest.skip("Apple Silicon MPS backend is not available.")
    return torch.device("mps")


def test_adversarial_disparate_ranks_and_alphas(mps_device: torch.device):
    """Challenge PEFT with extreme rank disparity: r=1 (vector outer product), r=2, r=32, r=64."""
    model = DeepHierarchicalToyModel().to(mps_device)

    target_modules = ["dense_attn", "dense_gate", "dense_proj", "output_head"]
    rank_pattern = {
        "dense_attn": 1,    # Extreme minimal rank (rank 1 matrix = vector outer product)
        "dense_gate": 2,    # Low rank
        "dense_proj": 32,   # Standard high rank
        "output_head": 64,  # Very high rank
    }
    alpha_pattern = {
        "dense_attn": 1,    # scaling = 1 / 1 = 1.0
        "dense_gate": 16,   # scaling = 16 / 2 = 8.0
        "dense_proj": 64,   # scaling = 64 / 32 = 2.0
        "output_head": 128, # scaling = 128 / 64 = 2.0
    }

    config = LoraConfig(
        r=8,
        lora_alpha=16,
        target_modules=target_modules,
        rank_pattern=rank_pattern,
        alpha_pattern=alpha_pattern,
        init_lora_weights=False,
    )

    peft_model = get_peft_model(model, config)

    # Inspect all LoRA modules and assert exact ranks and scalings
    found_ranks: dict[str, int] = {}
    found_alphas: dict[str, int] = {}
    found_scalings: dict[str, float] = {}

    for name, module in peft_model.named_modules():
        if hasattr(module, "lora_A") and "default" in module.lora_A:
            r_val = module.r["default"] if isinstance(module.r, dict) else module.r
            alpha_val = module.lora_alpha["default"] if isinstance(module.lora_alpha, dict) else module.lora_alpha
            scaling_val = module.scaling["default"] if isinstance(module.scaling, dict) else module.scaling
            target_key = name.split(".")[-1]
            found_ranks[target_key] = r_val
            found_alphas[target_key] = alpha_val
            found_scalings[target_key] = float(scaling_val)

            # Check parameter shapes
            weight_a = module.lora_A["default"].weight
            weight_b = module.lora_B["default"].weight
            assert weight_a.shape[0] == r_val, f"lora_A first dim should be r={r_val}, got {weight_a.shape}"
            assert weight_b.shape[1] == r_val, f"lora_B second dim should be r={r_val}, got {weight_b.shape}"

    assert found_ranks["dense_attn"] == 1
    assert found_ranks["dense_gate"] == 2
    assert found_ranks["dense_proj"] == 32
    assert found_ranks["output_head"] == 64

    assert found_alphas["dense_attn"] == 1
    assert found_alphas["dense_gate"] == 16
    assert found_alphas["dense_proj"] == 64
    assert found_alphas["output_head"] == 128

    assert found_scalings["dense_attn"] == pytest.approx(1.0)
    assert found_scalings["dense_gate"] == pytest.approx(8.0)
    assert found_scalings["dense_proj"] == pytest.approx(2.0)
    assert found_scalings["output_head"] == pytest.approx(2.0)

    # Forward pass on MPS
    x = torch.randn(4, 32, device=mps_device)
    out = peft_model(x)
    assert out.shape == (4, 16)
    assert not torch.isnan(out).any(), "Forward pass produced NaNs on MPS"
    assert not torch.isinf(out).any(), "Forward pass produced Infs on MPS"


def test_adversarial_hierarchical_subpath_targeting(mps_device: torch.device):
    """Verify that PEFT can disambiguate the SAME submodule name ('dense_attn') across different parent blocks."""
    model = DeepHierarchicalToyModel().to(mps_device)

    # block_0.dense_attn -> r=1, block_1.dense_attn -> r=32
    config = LoraConfig(
        r=4,
        lora_alpha=8,
        target_modules=["dense_attn"],
        rank_pattern={"block_0.dense_attn": 1, "block_1.dense_attn": 32},
        alpha_pattern={"block_0.dense_attn": 2, "block_1.dense_attn": 64},
        init_lora_weights=False,
    )
    peft_model = get_peft_model(model, config)

    mod_0 = peft_model.base_model.model.backbone["block_0"]["dense_attn"]
    mod_1 = peft_model.base_model.model.backbone["block_1"]["dense_attn"]

    r_0 = mod_0.r["default"] if isinstance(mod_0.r, dict) else mod_0.r
    r_1 = mod_1.r["default"] if isinstance(mod_1.r, dict) else mod_1.r
    alpha_0 = mod_0.lora_alpha["default"] if isinstance(mod_0.lora_alpha, dict) else mod_0.lora_alpha
    alpha_1 = mod_1.lora_alpha["default"] if isinstance(mod_1.lora_alpha, dict) else mod_1.lora_alpha

    assert r_0 == 1, f"Expected block_0.dense_attn to have r=1, got {r_0}"
    assert r_1 == 32, f"Expected block_1.dense_attn to have r=32, got {r_1}"
    assert alpha_0 == 2, f"Expected block_0.dense_attn to have alpha=2, got {alpha_0}"
    assert alpha_1 == 64, f"Expected block_1.dense_attn to have alpha=64, got {alpha_1}"


def test_adversarial_init_weights_dynamics(mps_device: torch.device):
    """Verify the mathematical consequence of init_lora_weights=True:
    Step 0: lora_B is initialized to 0, so grad(lora_A) is 0 by chain rule dL/dA = B^T * dL/d(BA).
    Step 1: after one optimizer step, lora_B is non-zero, so grad(lora_A) > 0.
    """
    model = DeepHierarchicalToyModel().to(mps_device)
    config = LoraConfig(
        r=4,
        lora_alpha=8,
        target_modules=["output_head"],
        init_lora_weights=True,  # Default PEFT init: A is kaiming, B is zeros
    )
    peft_model = get_peft_model(model, config)
    optimizer = torch.optim.AdamW(peft_model.parameters(), lr=0.01)

    x = torch.randn(4, 32, device=mps_device)

    # Step 0
    loss_0 = peft_model(x).sum()
    loss_0.backward()

    head_lora_a = peft_model.base_model.model.output_head.lora_A["default"].weight
    head_lora_b = peft_model.base_model.model.output_head.lora_B["default"].weight

    # Mathematically, B is zero, so A gradient MUST be zero at step 0
    assert head_lora_a.grad is not None
    assert head_lora_b.grad is not None
    assert head_lora_a.grad.norm().item() == pytest.approx(0.0, abs=1e-7)
    assert head_lora_b.grad.norm().item() > 1e-4

    # Perform optimizer step: B is now non-zero!
    optimizer.step()
    optimizer.zero_grad()

    # Step 1
    loss_1 = peft_model(x).sum()
    loss_1.backward()

    # Now both A and B MUST receive non-zero gradients
    assert head_lora_a.grad.norm().item() > 1e-4
    assert head_lora_b.grad.norm().item() > 1e-4


def test_adversarial_genuine_gradient_flow_and_weight_updates(mps_device: torch.device):
    """Empirically prove:
    1. Base model weights stay frozen and receive NO gradients.
    2. LoRA parameters receive valid gradients (> 1e-6) on MPS.
    3. Multi-step optimization updates LoRA weights in-place on MPS.
    4. Loss strictly decreases over optimization trajectory.
    5. optimizer.zero_grad() clears MPS gradient tensors without stale memory.
    """
    torch.manual_seed(42)
    model = DeepHierarchicalToyModel().to(mps_device)
    config = LoraConfig(
        r=1,
        lora_alpha=2,
        target_modules=["dense_attn", "dense_gate", "dense_proj", "output_head"],
        rank_pattern={"dense_attn": 1, "dense_proj": 32},
        alpha_pattern={"dense_attn": 2, "dense_proj": 64},
        init_lora_weights=False,
    )
    peft_model = get_peft_model(model, config)

    # Freeze base model check
    for name, param in peft_model.named_parameters():
        if "lora_" not in name:
            assert not param.requires_grad, f"Base parameter {name} should NOT require grad!"
        else:
            assert param.requires_grad, f"LoRA parameter {name} MUST require grad!"

    # Snapshot initial parameter values (deep cloned on device)
    initial_params = {name: p.clone().detach() for name, p in peft_model.named_parameters()}

    optimizer = torch.optim.AdamW(peft_model.parameters(), lr=0.01)
    loss_fn = nn.MSELoss()

    # Fixed training target to evaluate loss decrease
    x_train = torch.randn(16, 32, device=mps_device)
    y_target = torch.randn(16, 16, device=mps_device)

    losses: list[float] = []
    num_steps = 15

    for step in range(num_steps):
        optimizer.zero_grad()

        # Verify zero_grad cleared all gradients
        for name, param in peft_model.named_parameters():
            assert param.grad is None or (param.grad == 0).all(), (
                f"Step {step}: param {name} grad not cleared by zero_grad!"
            )

        preds = peft_model(x_train)
        loss = loss_fn(preds, y_target)
        losses.append(loss.item())

        loss.backward()

        # Check gradients before stepping
        for name, param in peft_model.named_parameters():
            if "lora_" in name:
                assert param.grad is not None, f"Step {step}: LoRA param {name} missing gradient!"
                grad_norm = param.grad.norm().item()
                assert grad_norm > 1e-6, f"Step {step}: LoRA param {name} gradient norm too small ({grad_norm})!"
                assert not torch.isnan(param.grad).any(), f"Step {step}: NaN in gradient for {name}"
            else:
                assert param.grad is None, f"Step {step}: Base model param {name} received unexpected gradient!"

        optimizer.step()
        torch.mps.synchronize()

    # Verify loss decreased significantly
    assert losses[-1] < losses[0], f"Optimization failed to reduce loss: initial={losses[0]}, final={losses[-1]}"
    loss_reduction_pct = (losses[0] - losses[-1]) / losses[0] * 100
    assert loss_reduction_pct > 25.0, f"Expected >25% loss reduction, got {loss_reduction_pct:.1f}%"

    # Verify in-place parameter updates:
    # 1. Base parameters must be IDENTICAL to initial values
    for name, param in peft_model.named_parameters():
        if "lora_" not in name:
            diff = (param - initial_params[name]).abs().max().item()
            assert diff == 0.0, f"Base parameter {name} was modified by optimizer (diff={diff})!"
        else:
            diff = (param - initial_params[name]).abs().max().item()
            assert diff > 1e-3, f"LoRA parameter {name} was NOT updated (max_diff={diff})!"


def test_adversarial_serialization_and_exact_inference_parity(mps_device: torch.device):
    """Verify that saving and reloading heterogeneous LoRA adapters on MPS preserves:
    1. Exact parameter shapes and ranks.
    2. Identical forward pass outputs (torch.allclose).
    """
    torch.manual_seed(123)
    base_model = DeepHierarchicalToyModel().to(mps_device)
    fresh_base = copy.deepcopy(base_model)

    config = LoraConfig(
        r=2,
        lora_alpha=4,
        target_modules=["dense_attn", "dense_proj", "output_head"],
        rank_pattern={"dense_attn": 1, "dense_proj": 32, "output_head": 16},
        alpha_pattern={"dense_attn": 2, "dense_proj": 64, "output_head": 32},
        init_lora_weights=False,
    )
    peft_model = get_peft_model(base_model, config)
    peft_model.eval()

    test_input = torch.randn(8, 32, device=mps_device)
    with torch.no_grad():
        original_output = peft_model(test_input).clone()

    tmp_dir = tempfile.mkdtemp(prefix="peft_mps_adv_")
    try:
        peft_model.save_pretrained(tmp_dir)

        # Reload into fresh base model instance
        reloaded_model = PeftModel.from_pretrained(fresh_base, tmp_dir)
        reloaded_model.eval()

        with torch.no_grad():
            reloaded_output = reloaded_model(test_input)

        # Assert numerical equivalence
        assert torch.allclose(original_output, reloaded_output, atol=1e-5), (
            "Reloaded model outputs differ from original outputs on MPS!"
        )

        # Assert rank preservation
        reloaded_ranks: dict[str, int] = {}
        for name, mod in reloaded_model.named_modules():
            if hasattr(mod, "lora_A") and "default" in mod.lora_A:
                r = mod.r["default"] if isinstance(mod.r, dict) else mod.r
                reloaded_ranks[name.split(".")[-1]] = r

        assert reloaded_ranks["dense_attn"] == 1
        assert reloaded_ranks["dense_proj"] == 32
        assert reloaded_ranks["output_head"] == 16
    finally:
        shutil.rmtree(tmp_dir)


def test_adversarial_bfloat16_precision_on_mps(mps_device: torch.device):
    """Verify LoRA training in bfloat16 mixed-precision on Apple Silicon MPS."""
    model = DeepHierarchicalToyModel().to(device=mps_device, dtype=torch.bfloat16)
    config = LoraConfig(
        r=4,
        lora_alpha=8,
        target_modules=["dense_attn", "output_head"],
        rank_pattern={"dense_attn": 1, "output_head": 16},
        init_lora_weights=False,
    )
    peft_model = get_peft_model(model, config).to(torch.bfloat16)

    optimizer = torch.optim.AdamW(peft_model.parameters(), lr=0.01)
    x = torch.randn(4, 32, device=mps_device, dtype=torch.bfloat16)
    target = torch.randn(4, 16, device=mps_device, dtype=torch.bfloat16)

    out = peft_model(x)
    assert out.dtype == torch.bfloat16
    loss = nn.functional.mse_loss(out, target)
    assert not torch.isnan(loss), "bfloat16 loss is NaN on MPS"

    loss.backward()
    for name, p in peft_model.named_parameters():
        if "lora_" in name:
            assert p.grad is not None
            assert p.grad.dtype == torch.bfloat16
            assert not torch.isnan(p.grad).any()
            assert p.grad.norm().item() > 1e-6

    optimizer.step()
    torch.mps.synchronize()


def test_adversarial_mps_memory_stability(mps_device: torch.device):
    """Stress test memory stability: 50 consecutive optimization steps on MPS.
    Verifies that MPS allocated memory does not monotonically explode (leak).
    """
    model = DeepHierarchicalToyModel().to(mps_device)
    config = LoraConfig(
        r=4,
        lora_alpha=8,
        target_modules=["dense_attn", "dense_gate", "dense_proj", "output_head"],
        rank_pattern={"dense_attn": 1, "dense_proj": 32},
        init_lora_weights=False,
    )
    peft_model = get_peft_model(model, config)
    optimizer = torch.optim.AdamW(peft_model.parameters(), lr=0.01)

    torch.mps.empty_cache()
    torch.mps.synchronize()

    mem_samples: list[int] = []
    x = torch.randn(16, 32, device=mps_device)
    target = torch.randn(16, 16, device=mps_device)

    for _ in range(50):
        optimizer.zero_grad()
        out = peft_model(x)
        loss = nn.functional.mse_loss(out, target)
        loss.backward()
        optimizer.step()
        torch.mps.synchronize()
        mem_samples.append(torch.mps.current_allocated_memory())

    # Memory after warm-up (last 20 steps) should be stable
    recent_mem = mem_samples[-20:]
    max_recent = max(recent_mem)
    min_recent = min(recent_mem)
    # Fluctuation during steady-state should be minimal (less than 2 MB)
    diff_mb = (max_recent - min_recent) / (1024 * 1024)
    assert diff_mb < 2.0, f"MPS memory leaked or drifted significantly: diff={diff_mb:.2f} MB"


def test_adversarial_zero_rank_rejected(mps_device: torch.device):
    """Empirically prove that PEFT rejects r=0 with ValueError during get_peft_model."""
    model = DeepHierarchicalToyModel().to(mps_device)
    config = LoraConfig(
        r=4,
        lora_alpha=8,
        target_modules=["output_head"],
        rank_pattern={"output_head": 0},
    )
    with pytest.raises(ValueError, match=r"should be a positive integer"):
        get_peft_model(model, config)


if __name__ == "__main__":
    pytest.main(["-v", __file__])
