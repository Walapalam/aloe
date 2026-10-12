"""Non-parametric paired bootstrap statistical evaluation engine for ALOE.

Implements ADR-014 and October Gate E validation protocol:
- Multi-seed aggregation across matching query sets.
- Paired bootstrap resampling (B=10,000 iterations).
- Bias-corrected and accelerated (BCa) and percentile confidence intervals.
- Null-centered two-tailed empirical p-values and Bonferroni adjustments.
- Cohen's d effect size computation.
"""
from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Literal, Mapping, Sequence
import numpy as np


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
    superior: bool  # True if p < alpha AND ci_low > 0.0


def _norm_cdf(x: float) -> float:
    """Standard normal cumulative distribution function."""
    return 0.5 * (1.0 + math.erf(x / math.sqrt(2.0)))


def _norm_ppf(p: float) -> float:
    """Standard normal percent point function (inverse CDF) using rational approximation."""
    if p <= 0.0:
        return -8.0
    if p >= 1.0:
        return 8.0
    if p == 0.5:
        return 0.0

    # Beasley-Springer-Moro approximation
    q = p - 0.5
    if abs(q) <= 0.42:
        r = q * q
        val = (
            q
            * (
                (
                    (-25.44106049637 * r + 41.39119773534) * r
                    - 18.61500062529
                )
                * r
                + 2.50662823884
            )
            / (
                (
                    ((3.13082909833 * r - 21.06224101826) * r + 23.08336743743)
                    * r
                    - 8.47351093090
                )
                * r
                + 1.0
            )
        )
        return val

    r = p if q < 0.0 else 1.0 - p
    r = math.log(-math.log(r))
    val = 0.322232431088 + r * (
        1.0
        + r
        * (
            0.342242088547
            + r * (0.0204231210245 + r * 0.0000453642210148)
        )
    )
    val /= (
        0.0993484626060
        + r
        * (
            0.588581570495
            + r * (0.531103462366 + r * (0.103537752850 + r * 0.0038560700634))
        )
    )
    return -val if q < 0.0 else val


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
    """Perform non-parametric paired bootstrap test between candidate A and baseline B.

    Args:
        a: Mapping of query_id -> sequence of binary correctness indicators for model A across seeds.
        b: Mapping of query_id -> sequence of binary correctness indicators for model B across seeds.
        B: Number of bootstrap resamples (default 10,000 per ADR-014).
        alpha: Significance level (default 0.05).
        ci_method: 'bca' for bias-corrected and accelerated, or 'percentile'.
        n_comparisons: Number of hypothesis comparisons for Bonferroni correction.
        rng_seed: Deterministic random seed for reproducibility.

    Returns:
        PairedBootstrapResult instance containing CIs, p-values, and significance flag.
    """
    if set(a.keys()) != set(b.keys()):
        raise ValueError("Query ID keys for model A and model B must match exactly.")

    query_ids = sorted(list(a.keys()))
    M = len(query_ids)
    if M == 0:
        raise ValueError("Cannot perform bootstrap on empty query set.")

    # Validate seed counts
    n_seeds_a = len(a[query_ids[0]])
    n_seeds_b = len(b[query_ids[0]])
    if n_seeds_a != n_seeds_b:
        raise ValueError(f"Seed count mismatch: model A has {n_seeds_a}, model B has {n_seeds_b}")

    # Compute paired differences d_i = mean_seed(a_i) - mean_seed(b_i)
    d = np.zeros(M, dtype=np.float64)
    for i, qid in enumerate(query_ids):
        mean_a = float(np.mean([float(v) for v in a[qid]]))
        mean_b = float(np.mean([float(v) for v in b[qid]]))
        d[i] = mean_a - mean_b

    mean_diff = float(np.mean(d))

    # Degenerate case: identical predictions across all queries
    if np.all(d == 0.0):
        return PairedBootstrapResult(
            mean_diff=0.0,
            ci_low=0.0,
            ci_high=0.0,
            ci_method=ci_method,
            p_value=1.0,
            p_value_bonferroni=1.0,
            n_comparisons=n_comparisons,
            cohens_d=None,
            n_queries=M,
            n_seeds=n_seeds_a,
            B=B,
            rng_seed=rng_seed,
            superior=False,
        )

    # Effect size (Cohen's d on paired differences)
    d_std = float(np.std(d, ddof=1))
    cohens_d = (mean_diff / d_std) if d_std > 1e-12 else None

    # Resampling
    rng = np.random.default_rng(rng_seed)
    indices = rng.integers(0, M, size=(B, M))
    boot_means = np.mean(d[indices], axis=1)

    # Null-centered empirical two-tailed p-value
    # H0: mean_diff = 0
    boot_centered = boot_means - mean_diff
    extreme_count = np.sum(np.abs(boot_centered) >= np.abs(mean_diff))
    p_val = float((extreme_count + 1) / (B + 1))
    p_val = min(1.0, max(p_val, 1.0 / (B + 1)))

    p_bonf = min(1.0, p_val * n_comparisons) if n_comparisons > 1 else None

    # Confidence Interval calculation
    if ci_method == "bca" and d_std > 1e-12:
        try:
            # 1. Bias-correction z0
            prop_less = np.mean(boot_means < mean_diff)
            # Clip probability to avoid infinite normal inverse
            prop_less = min(max(prop_less, 1e-6), 1.0 - 1e-6)
            z0 = _norm_ppf(prop_less)

            # 2. Acceleration parameter a via leave-one-out jackknife
            jack_means = np.zeros(M, dtype=np.float64)
            d_sum = np.sum(d)
            for i in range(M):
                jack_means[i] = (d_sum - d[i]) / (M - 1)

            jack_mean_bar = np.mean(jack_means)
            diff_jack = jack_mean_bar - jack_means
            num = np.sum(diff_jack**3)
            denom = 6.0 * (np.sum(diff_jack**2) ** 1.5)
            a_acc = (num / denom) if denom > 1e-12 else 0.0

            # 3. Adjusted percentiles
            z_alpha_low = _norm_ppf(alpha / 2.0)
            z_alpha_high = _norm_ppf(1.0 - alpha / 2.0)

            w1 = z0 + (z0 + z_alpha_low) / (1.0 - a_acc * (z0 + z_alpha_low))
            w2 = z0 + (z0 + z_alpha_high) / (1.0 - a_acc * (z0 + z_alpha_high))

            pct_low = _norm_cdf(w1) * 100.0
            pct_high = _norm_cdf(w2) * 100.0

            # Safety bounds
            pct_low = min(max(pct_low, 0.0), 100.0)
            pct_high = min(max(pct_high, 0.0), 100.0)

            ci_low = float(np.percentile(boot_means, pct_low))
            ci_high = float(np.percentile(boot_means, pct_high))
            actual_ci_method = "bca"
        except Exception:
            # Fallback to percentile if BCa calculation encounters numerical issues
            ci_low = float(np.percentile(boot_means, 100.0 * (alpha / 2.0)))
            ci_high = float(np.percentile(boot_means, 100.0 * (1.0 - alpha / 2.0)))
            actual_ci_method = "percentile"
    else:
        ci_low = float(np.percentile(boot_means, 100.0 * (alpha / 2.0)))
        ci_high = float(np.percentile(boot_means, 100.0 * (1.0 - alpha / 2.0)))
        actual_ci_method = "percentile"

    # Decision rule: p < alpha AND lower bound of CI > 0.0
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
