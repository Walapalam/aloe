"""Tests for non-parametric paired bootstrap statistical evaluation engine."""
import pytest
import numpy as np

from aloe.evaluation.bootstrap import paired_bootstrap, PairedBootstrapResult


def test_bootstrap_identical_models() -> None:
    """When models are identical, mean diff is 0, p-val is 1.0, superior is False."""
    a = {f"q_{i}": [1, 1, 0] for i in range(100)}
    b = {f"q_{i}": [1, 1, 0] for i in range(100)}

    res = paired_bootstrap(a, b, B=1000, rng_seed=42)
    assert res.mean_diff == 0.0
    assert res.ci_low == 0.0
    assert res.ci_high == 0.0
    assert res.p_value == 1.0
    assert res.superior is False
    assert res.cohens_d is None


def test_bootstrap_strictly_superior_model() -> None:
    """When model A clearly beats model B, p-val is small, CI > 0, superior is True."""
    # Model A gets 90% correct, Model B gets 40% correct
    rng = np.random.default_rng(123)
    a = {}
    b = {}
    for i in range(150):
        a[f"q_{i}"] = [int(rng.random() < 0.90) for _ in range(3)]
        b[f"q_{i}"] = [int(rng.random() < 0.40) for _ in range(3)]

    res = paired_bootstrap(a, b, B=2000, rng_seed=42, ci_method="bca")
    assert res.mean_diff > 0.35
    assert res.ci_low > 0.20
    assert res.ci_high > res.ci_low
    assert res.p_value < 0.001
    assert res.superior is True
    assert res.cohens_d is not None and res.cohens_d > 0


def test_bootstrap_key_and_seed_mismatches() -> None:
    """Verifies that key or seed mismatches raise clear errors."""
    a = {"q_1": [1, 1], "q_2": [1, 0]}
    b = {"q_1": [1, 1], "q_3": [1, 0]}  # Mismatched key

    with pytest.raises(ValueError, match="Query ID keys"):
        paired_bootstrap(a, b)

    # Seed count mismatch
    c = {"q_1": [1, 1, 1], "q_2": [1, 0, 1]}
    with pytest.raises(ValueError, match="Seed count mismatch"):
        paired_bootstrap(a, c)


def test_bootstrap_bonferroni_adjustment() -> None:
    """Verifies Bonferroni scaling on p-values."""
    a = {f"q_{i}": [1, 1] for i in range(50)}
    b = {f"q_{i}": [0, 1] for i in range(50)}

    res_single = paired_bootstrap(a, b, B=1000, n_comparisons=1, rng_seed=42)
    res_multi = paired_bootstrap(a, b, B=1000, n_comparisons=3, rng_seed=42)

    assert res_single.p_value_bonferroni is None
    assert res_multi.p_value_bonferroni is not None
    assert pytest.approx(res_multi.p_value_bonferroni) == min(1.0, res_single.p_value * 3)


def test_bootstrap_reproducibility() -> None:
    """Verifies that rng_seed guarantees deterministic output."""
    rng = np.random.default_rng(999)
    a = {f"q_{i}": [int(rng.random() < 0.7) for _ in range(3)] for i in range(80)}
    b = {f"q_{i}": [int(rng.random() < 0.6) for _ in range(3)] for i in range(80)}

    res1 = paired_bootstrap(a, b, B=1000, rng_seed=1234)
    res2 = paired_bootstrap(a, b, B=1000, rng_seed=1234)

    assert res1.mean_diff == res2.mean_diff
    assert res1.ci_low == res2.ci_low
    assert res1.ci_high == res2.ci_high
    assert res1.p_value == res2.p_value
