"""Characterization tests for mv_post_train.stats.

Goal: not to re-prove the underlying scipy/statsmodels implementations, but
to pin the contract (orientation, NaN handling, degenerate-input behaviour,
reproducibility under fixed seeds) so a future refactor that breaks any of
them fails loudly.
"""
from __future__ import annotations

import numpy as np

from mv_post_train.stats.core import (
    bh_adjust,
    newcombe_diff_ci,
    paired_bootstrap_ci,
    paired_perm_test,
    paired_tost,
    paired_wilcoxon,
    suite_average_bootstrap,
    unpaired_tost,
    wilson_ci,
)


def test_wilson_ci_empty_returns_nan():
    r = wilson_ci(0, 0)
    assert np.isnan(r.point) and np.isnan(r.lo) and np.isnan(r.hi)


def test_wilson_ci_brackets_point():
    r = wilson_ci(50, 100)
    assert r.lo < r.point < r.hi
    assert abs(r.point - 0.5) < 1e-9


def test_newcombe_diff_ci_sign_matches_orientation():
    # Group 1 better than group 2 => positive point estimate.
    r = newcombe_diff_ci(80, 100, 60, 100)
    assert r.point > 0
    assert r.lo < r.point < r.hi


def test_paired_bootstrap_ci_reproducible_under_seed():
    rng = np.random.default_rng(0)
    ya = rng.binomial(1, 0.4, 200).astype(float)
    yb = rng.binomial(1, 0.5, 200).astype(float)
    r1 = paired_bootstrap_ci(ya, yb, n_resamples=500, seed=42)
    r2 = paired_bootstrap_ci(ya, yb, n_resamples=500, seed=42)
    assert (r1.point, r1.lo, r1.hi) == (r2.point, r2.lo, r2.hi)


def test_paired_bootstrap_ci_handles_empty():
    r = paired_bootstrap_ci(np.array([]), np.array([]))
    assert np.isnan(r.point) and np.isnan(r.lo) and np.isnan(r.hi)


def test_paired_perm_test_identical_inputs_p_one():
    y = np.array([0.0, 1.0, 0.0, 1.0])
    obs, p = paired_perm_test(y, y, n_resamples=200)
    assert obs == 0.0 and p == 1.0


def test_paired_perm_test_clear_signal_low_p():
    # A always 1, B always 0 => signal is unambiguous.
    ya = np.ones(50)
    yb = np.zeros(50)
    _, p = paired_perm_test(ya, yb, n_resamples=500)
    assert p < 0.05


def test_paired_tost_zero_diff_equivalent():
    y = np.array([0.0, 1.0, 0.0, 1.0])
    _, rej = paired_tost(y, y, margin=0.05)
    assert rej is True


def test_unpaired_tost_runs():
    p, rej = unpaired_tost(50, 100, 49, 100, margin=0.10)
    assert isinstance(rej, bool)
    assert 0.0 <= p <= 1.0


def test_paired_wilcoxon_nan_on_zero_diff():
    y = np.array([0.0, 1.0, 0.0])
    assert np.isnan(paired_wilcoxon(y, y))


def test_bh_adjust_preserves_order_and_handles_nan():
    pvals = [0.01, float("nan"), 0.04, 0.5]
    q = bh_adjust(pvals)
    assert len(q) == 4
    assert np.isnan(q[1])  # NaN stays NaN
    # BH is monotone (within the non-NaN entries), so the smallest p has the
    # smallest q.
    nonnan = [(p, qi) for p, qi in zip(pvals, q) if not np.isnan(p)]
    assert nonnan[0][1] <= nonnan[1][1] <= nonnan[2][1]


def test_bh_adjust_all_nan_returns_all_nan():
    q = bh_adjust([float("nan"), float("nan")])
    assert all(np.isnan(x) for x in q)


def test_suite_average_bootstrap_paired_zero_diff_centred_at_zero():
    rng = np.random.default_rng(0)
    a = rng.binomial(1, 0.3, 50).astype(float)
    per_task_a = {"t1": a, "t2": a}
    per_task_b = {"t1": a, "t2": a}
    r = suite_average_bootstrap(per_task_a, per_task_b, n_resamples=300, seed=1)
    assert r.point == 0.0
    # CI collapses to zero when A == B identically.
    assert r.lo == 0.0 and r.hi == 0.0


def test_suite_average_bootstrap_single_sample_brackets_point():
    arr = np.array([0.0, 1.0, 1.0, 0.0, 1.0])
    r = suite_average_bootstrap({"t": arr}, n_resamples=300, seed=2)
    assert r.lo <= r.point <= r.hi
