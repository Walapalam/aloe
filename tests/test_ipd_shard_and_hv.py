"""Unit tests for ADR-019 3-D Normalized Hypervolume & CAS-Hardened HF Shard Syncer."""
from __future__ import annotations

import ast
import json
from pathlib import Path
import random
import re
from typing import Any

import pytest
import torch
from botorch.utils.multi_objective.box_decompositions.dominated import DominatedPartitioning

from scripts.compute_3d_hypervolume import (
    calculate_hypervolume_3d_exact,
    evaluate_model_campaigns_3d_hv,
)
from scripts.run_colab_shard import (
     HfShardSyncer,
    LeaseLostError,
    verify_arm_git_commits,
)


def test_exact_3d_hypervolume_matches_botorch() -> None:
    rng = random.Random(20261123)
    ref = (-0.15, -0.15, -0.15)
    ref_t = torch.tensor(ref, dtype=torch.float64)

    for _ in range(25):
        n_pts = rng.randint(1, 20)
        pts = [
            (rng.uniform(-0.2, 1.0), rng.uniform(-0.2, 1.0), rng.uniform(-0.2, 1.0))
            for _ in range(n_pts)
        ]
        exact_stdlib = calculate_hypervolume_3d_exact(pts, ref)

        valid = [p for p in pts if p[0] > ref[0] and p[1] > ref[1] and p[2] > ref[2]]
        if not valid:
            assert exact_stdlib == 0.0
            continue
        bd = DominatedPartitioning(ref_point=ref_t, Y=torch.tensor(valid, dtype=torch.float64))
        bt_hv = float(bd.compute_hypervolume().item())
        assert abs(exact_stdlib - bt_hv) < 1e-12


def test_evaluate_model_campaigns_3d_hv_shared_reference_and_sku_guard(tmp_path: Path) -> None:
    model_dir = tmp_path / "campaigns" / "synthetic--model"
    for arm, acc_base, mem_mb in (("aloe", 0.80, 13500.0), ("b0", 0.78, 15400.0), ("b1", 0.77, 13600.0)):
        arm_dir = model_dir / arm
        arm_dir.mkdir(parents=True)
        entries = [
            {
                "trial_id": f"t_{arm}_{i}",
                "status": "DONE",
                "metrics": {
                    "tool_accuracy": acc_base + 0.005 * i,
                    "exact_match_accuracy": acc_base + 0.005 * i,
                    "trainable_parameters": 7500000 + 1000 * i,
                    "search_memory_mb": mem_mb + 10.0 * i,
                },
            }
            for i in range(4)
        ]
        (arm_dir / "campaign_ledger.json").write_text(
            json.dumps(
                {
                    "model_id": "synthetic/model",
                    "arm": arm,
                    "locked_gpu_name": "NVIDIA L4",
                    "locked_memory_metric_name": "cuda_max_memory_allocated",
                    "entries": entries,
                }
            ),
            encoding="utf-8",
        )

    rep = evaluate_model_campaigns_3d_hv(model_dir)
    assert rep["locked_gpu_name"] == "NVIDIA L4"
    assert rep["arms"]["aloe"]["normalized_3d_hv"] > rep["arms"]["b1"]["normalized_3d_hv"]
    assert rep["arms"]["b1"]["normalized_3d_hv"] > rep["arms"]["b0"]["normalized_3d_hv"]
    assert rep["contribution_disentanglement"]["C1_typed_space_b1_vs_b0_hv_delta"] > 0
    assert rep["contribution_disentanglement"]["C2_qlognehvi_vs_random_hv_delta"] > 0

    bad_ledger = json.loads((model_dir / "b0" / "campaign_ledger.json").read_text(encoding="utf-8"))
    bad_ledger["locked_gpu_name"] = "NVIDIA A100-SXM4-40GB"
    (model_dir / "b0" / "campaign_ledger.json").write_text(json.dumps(bad_ledger), encoding="utf-8")
    with pytest.raises(RuntimeError, match="mixed GPU SKUs"):
        evaluate_model_campaigns_3d_hv(model_dir)


class _FakeSibling:
    def __init__(self, rfilename: str) -> None:
        self.rfilename = rfilename


class _FakeInfo:
    def __init__(self, sha: str, files: list[str]) -> None:
        self.sha = sha
        self.siblings = [_FakeSibling(f) for f in files]


class _FakeHfApi:
    def __init__(self) -> None:
        self.token = "fake_test_token"
        self.head_sha = "sha_0"
        self.commits: list[dict[str, Any]] = []
        self.files: dict[str, bytes] = {}
        self.fail_first_precondition = False

    def dataset_info(self, repo_id: str) -> _FakeInfo:
        return _FakeInfo(self.head_sha, sorted(self.files.keys()))

    def create_commit(
        self,
        *,
        repo_id: str,
        repo_type: str,
        operations: list[Any],
        commit_message: str,
        parent_commit: str | None = None,
    ) -> None:
        assert parent_commit is not None, "All shard commits must pass CAS parent_commit!"
        if self.fail_first_precondition:
            self.fail_first_precondition = False
            self.head_sha = "sha_bumped_by_sibling"
            raise RuntimeError("412 Precondition Failed: parent_commit outdated")
        if parent_commit != self.head_sha:
            raise RuntimeError("412 Precondition Failed")
        for op in operations:
            src = op.path_or_fileobj
            if isinstance(src, (str, Path)):
                self.files[op.path_in_repo] = Path(src).read_bytes()
            else:
                src.seek(0)
                self.files[op.path_in_repo] = src.read()
        self.head_sha = f"sha_{len(self.commits) + 1}"
        self.commits.append({"message": commit_message, "ops": [o.path_in_repo for o in operations]})


def test_hf_shard_syncer_cas_ownership_zombie_fence_and_git_pin(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    fake_api = _FakeHfApi()
    fake_api.fail_first_precondition = True
    fail_download = False

    def _fake_download(*, filename: str, **kwargs: Any) -> str:
        if fail_download:
            raise OSError("Simulated Hub 503 during lease read")
        out = tmp_path / "_dl" / filename
        out.parent.mkdir(parents=True, exist_ok=True)
        out.write_bytes(fake_api.files[filename])
        return str(out)

    import huggingface_hub
    monkeypatch.setattr(huggingface_hub, "hf_hub_download", _fake_download)

    s1 = HfShardSyncer(
        api=fake_api,
        hf_repo="Walapalam/aloe-ipd-v1-runs",
        shard_id="L1_lfm3_aloe",
        model_id="LiquidAI/LFM3-1.7B",
        arms=["aloe"],
        phases=["search", "confirm"],
    )
    s1.acquire_lease()
    assert "leases/L1_lfm3_aloe.json" in fake_api.files

    # Duplicate tab blocked
    s_dup = HfShardSyncer(
        api=fake_api,
        hf_repo="Walapalam/aloe-ipd-v1-runs",
        shard_id="L1_lfm3_aloe",
        model_id="LiquidAI/LFM3-1.7B",
        arms=["aloe"],
        phases=["search", "confirm"],
    )
    with pytest.raises(RuntimeError, match="DUPLICATE SHARD BLOCKED"):
        s_dup.acquire_lease()

    # Fail-closed when lease read throws Hub error (Opus B2 / Sol P1 #1)
    fail_download = True
    s_err = HfShardSyncer(
        api=fake_api,
        hf_repo="Walapalam/aloe-ipd-v1-runs",
        shard_id="L1_lfm3_aloe",
        model_id="LiquidAI/LFM3-1.7B",
        arms=["aloe"],
        phases=["search", "confirm"],
    )
    monkeypatch.setattr("time.sleep", lambda _s: None)
    with pytest.raises(RuntimeError, match="Fail-closed: unable to read remote lease"):
        s_err.acquire_lease()
    fail_download = False

    # Replacement VM steals lease via FORCE_STEAL_LEASE -> zombie VM s1 must immediately fence & abort!
    s_replacement = HfShardSyncer(
        api=fake_api,
        hf_repo="Walapalam/aloe-ipd-v1-runs",
        shard_id="L1_lfm3_aloe",
        model_id="LiquidAI/LFM3-1.7B",
        arms=["aloe"],
        phases=["search", "confirm"],
        force_steal_lease=True,
    )
    s_replacement.acquire_lease()

    arm_dir = tmp_path / "campaigns" / "LiquidAI--LFM3-1.7B" / "aloe"
    (arm_dir / "trials" / "t0" / "adapter").mkdir(parents=True)
    (arm_dir / "campaign_ledger.json").write_text("{}", encoding="utf-8")
    (arm_dir / "trials" / "t0" / "trial.json").write_text(
        json.dumps({"status": "PASS", "git_commit": "8187255cae076f583dce455acaf5019b20bad1e4"}),
        encoding="utf-8",
    )
    (arm_dir / "trials" / "t0" / "adapter" / "adapter_model.safetensors").write_bytes(b"w0")

    with pytest.raises(LeaseLostError, match="lease ownership lost"):
        s1.push_arm_delta(arm_dir, "campaigns/LiquidAI--LFM3-1.7B/aloe", commit_label="zombie_attempt")

    # Replacement VM succeeds and persists leader adapter
    s_replacement.push_arm_delta(
        arm_dir,
        "campaigns/LiquidAI--LFM3-1.7B/aloe",
        commit_label="replacement_ok",
        include_winner_adapter_tid="t0",
        strict=True,
    )
    assert "campaigns/LiquidAI--LFM3-1.7B/aloe/trials/t0/adapter/adapter_model.safetensors" in fake_api.files

    # Git pin guard rejects mismatched commit SHA (Sol P1 #3)
    verify_arm_git_commits(arm_dir, "8187255cae076f583dce455acaf5019b20bad1e4")
    with pytest.raises(RuntimeError, match="Git pin violation"):
        verify_arm_git_commits(arm_dir, "f0d0ea2b56b441e18d475366ea249617afe65883")

    s1.release_lease(status="FAILED")  # Must no-op safely without overwriting replacement's lease!
    latest_lease = json.loads(fake_api.files["leases/L1_lfm3_aloe.json"].decode("utf-8"))
    assert latest_lease["token"] == s_replacement.token_id
    assert latest_lease["status"] == "ACTIVE"
    s_replacement.release_lease(status="COMPLETED")


def test_generated_colab_shard_notebooks_valid_ast_and_zero_secrets() -> None:
    from scripts.generate_colab_shard_notebooks import OUT_DIR, SHARDS

    home_prefix = "/" + "Users/"
    assert len(SHARDS) == 16
    for spec in SHARDS:
        nb_path = OUT_DIR / str(spec["filename"])
        assert nb_path.is_file(), f"Missing {nb_path}"
        raw_txt = nb_path.read_text(encoding="utf-8")
        assert home_prefix not in raw_txt
        assert re.search(r"hf_[A-Za-z0-9]{15,}", raw_txt) is None

        nb = json.loads(raw_txt)
        assert nb["metadata"]["accelerator"] == "GPU"
        assert nb["metadata"]["colab"]["gpuType"] == "L4"
        assert nb["metadata"]["colab"]["background_execution"] == "on"
        code_cells = [c for c in nb["cells"] if c["cell_type"] == "code"]
        assert len(code_cells) == 1
        code_src = "".join(code_cells[0]["source"])
        ast.parse(code_src)
        assert "runtime.unassign()" in code_src
        assert "FORCE_STEAL_LEASE" in code_src
        assert '"clean"' not in code_src
        assert str(spec["pinned_sha"]) in code_src
        assert str(spec["shard_id"]) in code_src


def test_wheel_installer_never_uses_no_index_and_passes_no_deps_and_no_build_isolation(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    import subprocess as _sp
    import sys as _sys
    import types
    import scripts.run_colab_shard as rcs

    py_tag = f"py{_sys.version_info.major}{_sys.version_info.minor}"
    torch_tag = torch.__version__.split("+")[0]
    cu_tag = (torch.version.cuda or "cpu").replace(".", "")
    abi_dir = tmp_path / "wheels_cache" / f"{py_tag}_torch{torch_tag}_cu{cu_tag}"
    abi_dir.mkdir(parents=True)
    cc_whl = abi_dir / "causal_conv1d-1.7.0-fake.whl"
    mb_whl = abi_dir / "mamba_ssm-2.3.2.post1-fake.whl"
    cc_whl.write_bytes(b"fake_cc")
    mb_whl.write_bytes(b"fake_mb")

    recorded_cmds: list[list[str]] = []
    installed_ready = False

    def _fake_run(cmd: list[str], *args: Any, **kwargs: Any) -> _sp.CompletedProcess[str]:
        nonlocal installed_ready
        recorded_cmds.append(list(cmd))
        if "-c" in cmd and "import causal_conv1d" in cmd[-1]:
            if not installed_ready:
                raise _sp.CalledProcessError(1, cmd, stderr="not installed yet")
            return _sp.CompletedProcess(cmd, 0, "", "")
        if "pip" in cmd and "install" in cmd and any(x.endswith(".whl") for x in cmd):
            installed_ready = True
            # Simulate freshly installed modules in sys.modules
            monkeypatch.setitem(_sys.modules, "causal_conv1d", types.ModuleType("causal_conv1d"))
            mamba_mod = types.ModuleType("mamba_ssm")
            ops_mod = types.ModuleType("mamba_ssm.ops")
            ssi_mod = types.ModuleType("mamba_ssm.ops.selective_scan_interface")
            setattr(ssi_mod, "selective_scan_fn", lambda *a, **k: None)
            monkeypatch.setitem(_sys.modules, "mamba_ssm", mamba_mod)
            monkeypatch.setitem(_sys.modules, "mamba_ssm.ops", ops_mod)
            monkeypatch.setitem(_sys.modules, "mamba_ssm.ops.selective_scan_interface", ssi_mod)
        return _sp.CompletedProcess(cmd, 0, "", "")

    monkeypatch.setattr(rcs.subprocess, "run", _fake_run)
    fake_syncer = HfShardSyncer(
        api=_FakeHfApi(),
        hf_repo="Walapalam/aloe-ipd-v1-runs",
        shard_id="F1_test",
        model_id="tiiuae/Falcon-H1-1.5B-Instruct",
        arms=["b2"],
        phases=["search"],
    )
    rcs._install_or_build_cuda_wheels(
        tmp_path / "wheels_cache",
        need_mamba=True,
        need_causal_conv=True,
        syncer=fake_syncer,
    )

    pip_cmds = [c for c in recorded_cmds if "pip" in c]
    assert len(pip_cmds) == 2
    assert pip_cmds[0][-3:] == ["packaging", "ninja", "einops"]
    assert "--no-index" not in pip_cmds[0]
    assert "--no-index" not in pip_cmds[1]
    assert "--no-deps" in pip_cmds[1]
    assert str(cc_whl) in pip_cmds[1]
    assert str(mb_whl) in pip_cmds[1]

    # Also verify source avoids --no-index anywhere and uses --no-build-isolation on pip download
    src_txt = Path(rcs.__file__).read_text(encoding="utf-8")
    assert "--no-index" not in src_txt
    assert '"--no-build-isolation"' in src_txt


def test_disable_lfm3_broken_triton_conv_patches_shortconv_globals() -> None:
    from aloe.datasets.model_registry import ModelRegistry

    reg = ModelRegistry()
    cfg = reg.load_config_from_hub_or_path("LiquidAI/LFM3-1.7B")
    meta_model = reg.load_model_meta_from_config(cfg)
    count = ModelRegistry.disable_lfm3_broken_triton_conv(meta_model)
    assert count > 0
    for sub in meta_model.modules():
        if type(sub).__name__ == "Lfm3ShortConv":
            fwd = getattr(sub.__class__, "forward")
            while hasattr(fwd, "__wrapped__"):
                fwd = fwd.__wrapped__
            assert fwd.__globals__["can_use_lfm3_conv"](None, None) is False

    import aloe.execution.backends.peft_mps as pm_mod
    pm_src = Path(pm_mod.__file__).read_text(encoding="utf-8")
    assert 'gen_use_cache = False if "lfm3" in model_id.lower() else True' in pm_src
    assert "use_cache=gen_use_cache" in pm_src


