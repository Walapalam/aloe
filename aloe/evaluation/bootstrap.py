"""Non-parametric paired bootstrap, Holm-Bonferroni, and McNemar statistical engine for ALOE.

Implements ADR-014 and Protocol v1 statistical evaluation:
- Complete per-query seed-count verification across all paired items.
- Paired bootstrap resampling (B=10,000 iterations) with BCa and percentile CIs.
- Step-down Holm-Bonferroni family-wise multiple-comparison adjustment.
- Exact binomial & continuity-corrected McNemar discordant-pair test.
- Cohen's d effect size computation.
"""
from __future__ import annotations

from dataclasses import dataclass
import math
from statistics import NormalDist
from typing import Literal, Mapping, Sequence
import numpy as np

_STD_NORM = NormalDist()


@dataclass(frozen=True)
class PairedBootstrapResult:
    """Rigorous statistical result of paired bootstrap hypothesis test."""

    mean_diff: float
    ci_low: float
    ci_high: float
    ci_method: Literal["bca", "percentile"]
    p_value: float
    p_value_bonferroni: float | None
    n_comparisons: int
    cohens_d: float | None
    n_queries: int
    n_seeds: int
    B: int
    rng_seed: int
    superior: bool  # True if effective_p < alpha AND ci_low > 0.0


@dataclass(frozen=True)
class HolmAdjustmentResult:
    """Family-wise step-down Holm-Bonferroni multiple-comparison result."""

    name: str
    raw_p: float
    adjusted_p: float
    alpha: float
    reject_null: bool


@dataclass(frozen=True)
class McNemarResult:
    """Paired discordant-pair McNemar test result."""

    n_both_correct: int
    n_a_only: int
    n_b_only: int
    n_both_wrong: int
    discordant_pairs: int
    exact_p_value: float


def _norm_cdf(x: float) -> float:
    """Standard normal cumulative distribution function."""
    return 0.5 * (1.0 + math.erf(x / math.sqrt(2.0)))


def _norm_ppf(p: float) -> float:
    """Standard normal percent point function (exact inverse CDF)."""
    if p <= 0.0:
        return -8.0
    if p >= 1.0:
        return 8.0
    if p == 0.5:
        return 0.0
    return float(_STD_NORM.inv_cdf(p))


def holm_bonferroni(
    p_values: Mapping[str, float],
    *,
    alpha: float = 0.05,
) -> dict[str, HolmAdjustmentResult]:
    """Apply step-down Holm-Bonferroni correction across a pre-declared hypothesis family (ADR-018).

    For sorted raw p-values p_(1) <= p_(2) <= ... <= p_(m):
      p_holm_(i) = min(1.0, max_{j <= i} (m - j + 1) * p_(j))
    Rejects while p_holm_(i) <= alpha; stops rejecting from the first p_holm_(k) > alpha onward.
    """
    if not (math.isfinite(alpha) and 0.0 < alpha < 1.0):
        raise ValueError(f"alpha must be in (0, 1), got {alpha}")
    if not p_values:
        return {}

    validated: list[tuple[str, float]] = []
    for name, raw_p in p_values.items():
        rp = float(raw_p)
        if not math.isfinite(rp) or rp < 0.0 or rp > 1.0:
            raise ValueError(f"Invalid p-value for '{name}': {raw_p} (must be finite in [0, 1])")
        validated.append((name, rp))

    items = sorted(validated, key=lambda kv: (kv[1], kv[0]))
    m = len(items)
    results: dict[str, HolmAdjustmentResult] = {}
    running_max = 0.0
    still_rejecting = True

    for idx, (name, rp) in enumerate(items):
        scaled = (m - idx) * rp
        adj = min(1.0, max(running_max, scaled))
        running_max = adj
        if adj > alpha:
            still_rejecting = False
        results[name] = HolmAdjustmentResult(
            name=name,
            raw_p=rp,
            adjusted_p=adj,
            alpha=alpha,
            reject_null=still_rejecting,
        )
    return results


def mcnemar_exact(
    a: Mapping[str, Sequence[int | float | bool]],
    b: Mapping[str, Sequence[int | float | bool]],
) -> McNemarResult:
    """Exact two-sided binomial McNemar test on odd-seed majority-vote / single-seed binary outcomes."""
    if set(a.keys()) != set(b.keys()):
        raise ValueError("Query ID keys for model A and model B must match exactly.")
    qids = sorted(a.keys())
    if not qids:
        raise ValueError("Cannot perform McNemar test on empty query set.")

    expected_seeds = len(a[qids[0]])
    if expected_seeds == 0:
        raise ValueError("Each query must have at least one seed observation.")

    n11 = n10 = n01 = n00 = 0
    for q in qids:
        if len(a[q]) != expected_seeds or len(b[q]) != expected_seeds:
            raise ValueError(f"Inconsistent seed count at query '{q}' in mcnemar_exact")
        vals_a = [float(x) for x in a[q]]
        vals_b = [float(x) for x in b[q]]
        if not (all(math.isfinite(x) for x in vals_a) and all(math.isfinite(x) for x in vals_b)):
            raise ValueError(f"Non-finite outcome value at query '{q}' in mcnemar_exact")
        mean_a = float(np.mean(vals_a))
        mean_b = float(np.mean(vals_b))
        if expected_seeds % 2 == 0 and (abs(mean_a - 0.5) < 1e-12 or abs(mean_b - 0.5) < 1e-12):
            raise ValueError(
                f"Even seed count ({expected_seeds}) produced a 0.5 majority-vote tie at query '{q}'; "
                "use odd seed counts (e.g. 1 or 3 seeds per ADR-014/018)."
            )
        va = mean_a > 0.5
        vb = mean_b > 0.5
        if va and vb:
            n11 += 1
        elif va and not vb:
            n10 += 1
        elif not va and vb:
            n01 += 1
        else:
            n00 += 1

    n_disc = n10 + n01
    if n_disc == 0:
        p_exact = 1.0
    else:
        k = min(n10, n01)
        # Exact two-sided binomial tail: 2 * sum_{i=0}^{k} C(n_disc, i) * 0.5^n_disc
        tail = sum(math.comb(n_disc, i) for i in range(k + 1)) * (0.5**n_disc)
        p_exact = min(1.0, 2.0 * tail)

    return McNemarResult(
        n_both_correct=n11,
        n_a_only=n10,
        n_b_only=n01,
        n_both_wrong=n00,
        discordant_pairs=n_disc,
        exact_p_value=float(p_exact),
    )


def paired_bootstrap(
    a: Mapping[str, Sequence[int | float | bool]],
    b: Mapping[str, Sequence[int | float | bool]],
    *,
    B: int = 10_000,
    alpha: float = 0.05,
    ci_method: Literal["bca", "percentile"] = "bca",
    n_comparisons: int = 1,
    rng_seed: int = 42,
) -> PairedBootstrapResult:
    """Perform non-parametric paired query bootstrap test between candidate A and baseline B.

    Note (ADR-018): Resamples queries with replacement conditional on the fixed seed set.
    Family-wise multiplicity across multiple arms is controlled via `holm_bonferroni()`.
    """
    if not (math.isfinite(alpha) and 0.0 < alpha < 1.0):
        raise ValueError(f"alpha must be in (0, 1), got {alpha}")
    if B < 100:
        raise ValueError(f"Bootstrap resamples B must be >= 100, got {B}")
    if set(a.keys()) != set(b.keys()):
        raise ValueError("Query ID keys for model A and model B must match exactly.")

    query_ids = sorted(list(a.keys()))
    M = len(query_ids)
    if M == 0:
        raise ValueError("Cannot perform bootstrap on empty query set.")

    n_seeds_a = len(a[query_ids[0]])
    n_seeds_b = len(b[query_ids[0]])
    if n_seeds_a == 0 or n_seeds_b == 0:
        raise ValueError("Each query must have at least one seed observation.")
    if n_seeds_a != n_seeds_b:
        raise ValueError(f"Seed count mismatch: model A has {n_seeds_a}, model B has {n_seeds_b}")

    # Enforce uniform seed count across ALL queries (not just query_ids[0])
    d = np.zeros(M, dtype=np.float64)
    for i, qid in enumerate(query_ids):
        seq_a = a[qid]
        seq_b = b[qid]
        if len(seq_a) != n_seeds_a or len(seq_b) != n_seeds_b:
            raise ValueError(
                f"Inconsistent seed count at query '{qid}': "
                f"len(a)={len(seq_a)} (expected {n_seeds_a}), len(b)={len(seq_b)} (expected {n_seeds_b})"
            )
        mean_a = float(np.mean([float(v) for v in seq_a]))
        mean_b = float(np.mean([float(v) for v in seq_b]))
        if not (math.isfinite(mean_a) and math.isfinite(mean_b)):
            raise ValueError(f"Non-finite outcome value at query '{qid}'")
        d[i] = mean_a - mean_b

    mean_diff = float(np.mean(d))

    if np.all(d == 0.0):
        return PairedBootstrapResult(
            mean_diff=0.0,
            ci_low=0.0,
            ci_high=0.0,
            ci_method=ci_method,
            p_value=1.0,
            p_value_bonferroni=1.0 if n_comparisons > 1 else None,
            n_comparisons=n_comparisons,
            cohens_d=None,
            n_queries=M,
            n_seeds=n_seeds_a,
            B=B,
            rng_seed=rng_seed,
            superior=False,
        )

    d_std = float(np.std(d, ddof=1)) if M > 1 else 0.0
    cohens_d = (mean_diff / d_std) if d_std > 1e-12 else None

    rng = np.random.default_rng(rng_seed)
    indices = rng.integers(0, M, size=(B, M))
    boot_means = np.mean(d[indices], axis=1)

    boot_centered = boot_means - mean_diff
    extreme_count = np.sum(np.abs(boot_centered) >= np.abs(mean_diff))
    p_val = float((extreme_count + 1) / (B + 1))
    p_val = min(1.0, max(p_val, 1.0 / (B + 1)))

    p_bonf = min(1.0, p_val * n_comparisons) if n_comparisons > 1 else None

    if ci_method == "bca" and d_std > 1e-12 and M > 2:
        try:
            prop_less = float(np.mean(boot_means < mean_diff))
            prop_less = min(max(prop_less, 1e-6), 1.0 - 1e-6)
            z0 = _norm_ppf(prop_less)

            jack_means = np.zeros(M, dtype=np.float64)
            d_sum = float(np.sum(d))
            for i in range(M):
                jack_means[i] = (d_sum - d[i]) / (M - 1)

            jack_mean_bar = float(np.mean(jack_means))
            diff_jack = jack_mean_bar - jack_means
            num = float(np.sum(diff_jack**3))
            denom = 6.0 * (float(np.sum(diff_jack**2)) ** 1.5)
            a_acc = (num / denom) if denom > 1e-12 else 0.0

            z_alpha_low = _norm_ppf(alpha / 2.0)
            z_alpha_high = _norm_ppf(1.0 - alpha / 2.0)

            w1 = z0 + (z0 + z_alpha_low) / (1.0 - a_acc * (z0 + z_alpha_low))
            w2 = z0 + (z0 + z_alpha_high) / (1.0 - a_acc * (z0 + z_alpha_high))

            pct_low = min(max(_norm_cdf(w1) * 100.0, 0.0), 100.0)
            pct_high = min(max(_norm_cdf(w2) * 100.0, 0.0), 100.0)

            ci_low = float(np.percentile(boot_means, pct_low))
            ci_high = float(np.percentile(boot_means, pct_high))
            actual_ci_method: Literal["bca", "percentile"] = "bca"
        except Exception:
            ci_low = float(np.percentile(boot_means, 100.0 * (alpha / 2.0)))
            ci_high = float(np.percentile(boot_means, 100.0 * (1.0 - alpha / 2.0)))
            actual_ci_method = "percentile"
    else:
        ci_low = float(np.percentile(boot_means, 100.0 * (alpha / 2.0)))
        ci_high = float(np.percentile(boot_means, 100.0 * (1.0 - alpha / 2.0)))
        actual_ci_method = "percentile"

    effective_p = p_bonf if p_bonf is not None else p_val
    superior = bool((effective_p < alpha) and (ci_low > 0.0))

    return PairedBootstrapResult(
        mean_diff=mean_diff,
        ci_low=ci_low,
        ci_high=ci_high,
        ci_method=actual_ci_method,
        p_value=p_val,
        p_value_bonferroni=p_bonf,
        n_comparisons=n_comparisons,
        cohens_d=cohens_d,
        n_queries=M,
        n_seeds=n_seeds_a,
        B=B,
        rng_seed=rng_seed,
        superior=superior,
    )
