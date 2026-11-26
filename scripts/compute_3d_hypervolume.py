#!/usr/bin/env python3
"""Exact 3-D Normalized Pareto Hypervolume & Trajectory Evaluator (`compute_3d_hypervolume.py`).

Implements ADR-019 cross-arm shared-reference 3-D hypervolume in pure Python stdlib
(zero torch/botorch runtime dependency, exact via 2-D z-slab sweeping) to disentangle:
  - Contribution C1 (Typed Operator Grammar): {ALOE, B1} vs {B0, B2, Borobia-*}
  - Contribution C2 (qLogNEHVI vs Random):    ALOE vs B1 across trials t = 1..20

Supports both per-directory normalization and fixed cross-seed `shared_bounds` so multi-seed
`ALOE` vs `B1` hypervolume comparisons are evaluated in a single invariant coordinate system.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys
from typing import Any

REPO_ROOT = Path(__file__).resolve().parent.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))


def _exact_hv_2d_unrounded(
    points: list[tuple[float, float]],
    reference_point: tuple[float, float],
) -> float:
    """Unrounded float64 2-D hypervolume (avoids 6-decimal rounding accumulation across z-slabs)."""
    rx, ry = reference_point
    valid = [(x, y) for (x, y) in points if x > rx and y > ry]
    if not valid:
        return 0.0
    sorted_pts = sorted(valid, key=lambda p: (p[0], p[1]), reverse=True)
    non_dom: list[tuple[float, float]] = []
    max_y = ry
    for x, y in sorted_pts:
        if y > max_y:
            non_dom.append((x, y))
            max_y = y
    non_dom.sort(key=lambda p: p[0])
    area = 0.0
    for idx, (xc, yc) in enumerate(non_dom):
        xp = rx if idx == 0 else non_dom[idx - 1][0]
        area += (xc - xp) * (yc - ry)
    return area


def calculate_hypervolume_3d_exact(
    points: list[tuple[float, float, float]],
    reference_point: tuple[float, float, float],
) -> float:
    """Compute exact 3-D dominated hypervolume above `reference_point` (all coordinates maximized).

    Uses exact orthogonal z-slab decomposition over unrounded 2-D hypervolume:
      HV_3D(P, r) = sum_{k=1..m} (z_(k) - z_(k-1)) * HV_2D({(x, y) : (x, y, z) in P, z >= z_(k)}, (r_x, r_y))
    """
    rx, ry, rz = reference_point
    valid = [(float(x), float(y), float(z)) for (x, y, z) in points if x > rx and y > ry and z > rz]
    if not valid:
        return 0.0

    z_levels = sorted({p[2] for p in valid})
    total_vol = 0.0
    z_prev = rz
    for z_curr in z_levels:
        dz = z_curr - z_prev
        if dz > 0.0:
            xy_slice = [(p[0], p[1]) for p in valid if p[2] >= z_curr]
            area = _exact_hv_2d_unrounded(xy_slice, (rx, ry))
            total_vol += dz * area
        z_prev = z_curr
    return float(total_vol)


def _extract_trial_max_coords(entry: dict[str, Any]) -> tuple[float, float, float]:
    """Return maximization vector (exact_match_accuracy, -trainable_parameters, -search_memory_mb)."""
    m = entry["metrics"]
    acc = float(m.get("exact_match_accuracy", m["tool_accuracy"]))
    neg_params = -float(m["trainable_parameters"])
    neg_mem = -float(m["search_memory_mb"])
    return (acc, neg_params, neg_mem)


def compute_union_raw_max_bounds(
    model_campaign_dirs: list[Path],
    *,
    arms_filter: set[str] | None = None,
) -> tuple[list[float], list[float]]:
    """Compute invariant maximization coordinate `(mins, maxs)` across multiple seed directories of ONE model."""
    all_raw: list[tuple[float, float, float]] = []
    locked_gpus: set[str] = set()
    locked_mem_metrics: set[str] = set()
    model_ids: set[str] = set()

    for mdir in model_campaign_dirs:
        if not mdir.is_dir():
            continue
        for ledger_path in sorted(mdir.glob("*/campaign_ledger.json")):
            arm = ledger_path.parent.name
            if arms_filter is not None and arm not in arms_filter:
                continue
            ldata = json.loads(ledger_path.read_text(encoding="utf-8"))
            done = [e for e in ldata.get("entries", []) if e.get("status") == "DONE"]
            if not done:
                continue
            if ldata.get("model_id"):
                model_ids.add(str(ldata["model_id"]))
            if ldata.get("locked_gpu_name"):
                locked_gpus.add(str(ldata["locked_gpu_name"]))
            if ldata.get("locked_memory_metric_name"):
                locked_mem_metrics.add(str(ldata["locked_memory_metric_name"]))
            all_raw.extend(_extract_trial_max_coords(e) for e in done)

    if not all_raw:
        raise FileNotFoundError(f"No completed trials found across {model_campaign_dirs}")
    if len(model_ids) > 1:
        raise RuntimeError(f"Refusing union HV bounds across mixed models: {sorted(model_ids)}")
    if len(locked_gpus) > 1:
        raise RuntimeError(f"Refusing union HV bounds on mixed GPU SKUs: {sorted(locked_gpus)}")
    if len(locked_mem_metrics) > 1:
        raise RuntimeError(f"Refusing union HV bounds on mixed memory metrics: {sorted(locked_mem_metrics)}")

    mins = [min(p[k] for p in all_raw) for k in range(3)]
    maxs = [max(p[k] for p in all_raw) for k in range(3)]
    return mins, maxs


def evaluate_model_campaigns_3d_hv(
    model_campaigns_dir: Path,
    *,
    nadir_pad: float = 0.15,
    shared_bounds: tuple[list[float], list[float]] | None = None,
    normalization_scope: str = "directory_local",
    arms_filter: set[str] | None = None,
) -> dict[str, Any]:
    """Compute shared-reference normalized 3-D HV and anytime trajectories across arms in a model directory."""
    arm_entries: dict[str, list[dict[str, Any]]] = {}
    locked_gpus: set[str] = set()
    locked_mem_metrics: set[str] = set()
    model_id = "UNKNOWN"

    for ledger_path in sorted(model_campaigns_dir.glob("*/campaign_ledger.json")):
        arm = ledger_path.parent.name
        if arms_filter is not None and arm not in arms_filter:
            continue
        ldata = json.loads(ledger_path.read_text(encoding="utf-8"))
        done = [e for e in ldata.get("entries", []) if e.get("status") == "DONE"]
        if not done:
            continue
        model_id = str(ldata.get("model_id", model_id))
        if ldata.get("locked_gpu_name"):
            locked_gpus.add(str(ldata["locked_gpu_name"]))
        if ldata.get("locked_memory_metric_name"):
            locked_mem_metrics.add(str(ldata["locked_memory_metric_name"]))
        arm_entries[arm] = done

    if not arm_entries:
        raise FileNotFoundError(f"No completed campaign ledgers found in {model_campaigns_dir}")
    if len(locked_gpus) > 1:
        raise RuntimeError(f"Refusing cross-arm HV on mixed GPU SKUs: {sorted(locked_gpus)}")
    if len(locked_mem_metrics) > 1:
        raise RuntimeError(f"Refusing cross-arm HV on mixed memory metrics: {sorted(locked_mem_metrics)}")

    all_raw = [_extract_trial_max_coords(e) for entries in arm_entries.values() for e in entries]
    if shared_bounds is None:
        mins = [min(p[k] for p in all_raw) for k in range(3)]
        maxs = [max(p[k] for p in all_raw) for k in range(3)]
    else:
        mins = [float(x) for x in shared_bounds[0]]
        maxs = [float(x) for x in shared_bounds[1]]
        if len(mins) != 3 or len(maxs) != 3 or any(maxs[k] < mins[k] for k in range(3)):
            raise ValueError(f"Invalid shared_bounds={shared_bounds}")
    spans = [max(maxs[k] - mins[k], 1e-12) for k in range(3)]

    def _norm(pt: tuple[float, float, float]) -> tuple[float, float, float]:
        return (
            (pt[0] - mins[0]) / spans[0],
            (pt[1] - mins[1]) / spans[1],
            (pt[2] - mins[2]) / spans[2],
        )

    ref_norm = (-nadir_pad, -nadir_pad, -nadir_pad)
    max_possible_vol = (1.0 + nadir_pad) ** 3
    raw_ref = (
        mins[0] - nadir_pad * spans[0],
        mins[1] - nadir_pad * spans[1],
        mins[2] - nadir_pad * spans[2],
    )

    arms_report: dict[str, Any] = {}
    for arm, entries in sorted(arm_entries.items()):
        raw_pts = [_extract_trial_max_coords(e) for e in entries]
        norm_pts = [_norm(p) for p in raw_pts]
        trajectory: list[float] = []
        for t in range(1, len(norm_pts) + 1):
            hv_t = calculate_hypervolume_3d_exact(norm_pts[:t], ref_norm) / max_possible_vol
            trajectory.append(round(hv_t, 6))
        raw_hv = calculate_hypervolume_3d_exact(raw_pts, raw_ref)

        # Non-dominated count in 3D maximization space
        pareto_ids: list[str] = []
        raw_pareto_front: list[dict[str, Any]] = []
        for i, e_i in enumerate(entries):
            pi = raw_pts[i]
            dom = False
            for j, _ in enumerate(entries):
                if i == j:
                    continue
                pj = raw_pts[j]
                if (
                    pj[0] >= pi[0]
                    and pj[1] >= pi[1]
                    and pj[2] >= pi[2]
                    and (pj[0] > pi[0] or pj[1] > pi[1] or pj[2] > pi[2])
                ):
                    dom = True
                    break
            if not dom:
                tid = str(e_i["trial_id"])
                pareto_ids.append(tid)
                raw_pareto_front.append(
                    {
                        "trial_id": tid,
                        "exact_match": round(pi[0], 4),
                        "trainable_parameters": int(-pi[1]),
                        "search_memory_mb": round(-pi[2], 3),
                    }
                )

        accs = sorted(p[0] for p in raw_pts)
        mems_mb = [-p[2] for p in raw_pts]
        params_list = [-p[1] for p in raw_pts]
        mid = len(accs) // 2
        median_acc = accs[mid] if len(accs) % 2 == 1 else 0.5 * (accs[mid - 1] + accs[mid])

        arms_report[arm] = {
            "completed_trials": len(entries),
            "normalized_3d_hv": trajectory[-1],
            "raw_3d_hv": round(raw_hv, 3),
            "anytime_hv_trajectory": trajectory,
            "pareto_count": len(pareto_ids),
            "pareto_trial_ids": pareto_ids,
            "raw_pareto_front": raw_pareto_front,
            "best_exact_match": round(max(accs), 4),
            "median_exact_match": round(median_acc, 4),
            "min_memory_mb": round(min(mems_mb), 3),
            "min_trainable_params": int(min(params_list)),
        }

    disentanglement: dict[str, Any] = {}
    if "b0" in arms_report and "b1" in arms_report:
        b0_hv = arms_report["b0"]["normalized_3d_hv"]
        b1_hv = arms_report["b1"]["normalized_3d_hv"]
        disentanglement["C1_typed_space_b1_vs_b0_hv_delta"] = round(b1_hv - b0_hv, 6)
        disentanglement["C1_typed_space_min_memory_saving_mb"] = round(
            arms_report["b0"]["min_memory_mb"] - arms_report["b1"]["min_memory_mb"], 3
        )
    if "aloe" in arms_report and "b1" in arms_report:
        aloe_hv = arms_report["aloe"]["normalized_3d_hv"]
        b1_hv = arms_report["b1"]["normalized_3d_hv"]
        disentanglement["C2_qlognehvi_vs_random_hv_delta"] = round(aloe_hv - b1_hv, 6)
        disentanglement["C2_qlognehvi_vs_random_rel_hv_pct"] = (
            round(100.0 * (aloe_hv - b1_hv) / b1_hv, 2) if b1_hv > 0 else None
        )
        disentanglement["C2_median_em_delta"] = round(
            arms_report["aloe"]["median_exact_match"] - arms_report["b1"]["median_exact_match"], 4
        )

    return {
        "model_id": model_id,
        "locked_gpu_name": next(iter(locked_gpus), None),
        "locked_memory_metric_name": next(iter(locked_mem_metrics), None),
        "normalization_scope": normalization_scope,
        "nadir_pad": nadir_pad,
        "shared_raw_bounds": {
            "exact_match": [round(mins[0], 6), round(maxs[0], 6)],
            "trainable_params": [int(-maxs[1]), int(-mins[1])],
            "search_memory_mb": [round(-maxs[2], 3), round(-mins[2], 3)],
        },
        "physical_reference_point": {
            "exact_match_floor": round(raw_ref[0], 6),
            "trainable_params_ceiling": round(-raw_ref[1], 1),
            "search_memory_mb_ceiling": round(-raw_ref[2], 3),
        },
        "arms": arms_report,
        "contribution_disentanglement": disentanglement,
    }


def main() -> int:
    parser = argparse.ArgumentParser(description="Compute ADR-019 cross-arm 3D normalized Pareto hypervolume")
    parser.add_argument("--model-campaigns-dir", required=True, help="Path to campaigns/<model_slug>")
    parser.add_argument("--nadir-pad", type=float, default=0.15)
    parser.add_argument("--out", default="", help="Optional JSON output path")
    args = parser.parse_args()

    report = evaluate_model_campaigns_3d_hv(Path(args.model_campaigns_dir), nadir_pad=args.nadir_pad)
    encoded = json.dumps(report, indent=2)
    if args.out:
        out_p = Path(args.out)
        out_p.parent.mkdir(parents=True, exist_ok=True)
        out_p.write_text(encoded + "\n", encoding="utf-8")
    print(encoded)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
