"""Thin wrappers around scipy / statsmodels for the analyses we need.

Everything here defers to library implementations:
  - Wilson CI            -> statsmodels.stats.proportion.proportion_confint
  - Newcombe diff CI     -> statsmodels.stats.proportion.confint_proportions_2indep
  - Paired bootstrap     -> scipy.stats.bootstrap (paired=True)
  - Permutation test     -> scipy.stats.permutation_test (permutation_type='samples')
  - BH FDR adjustment    -> statsmodels.stats.multitest.multipletests
  - Paired binomial MDE  -> simulation using statsmodels.stats.contingency_tables.mcnemar
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from scipy import stats
from statsmodels.stats.contingency_tables import mcnemar
from statsmodels.stats.multitest import multipletests
from statsmodels.stats.proportion import (
    confint_proportions_2indep,
    proportion_confint,
)
from statsmodels.stats.weightstats import ttost_ind, ttost_paired


@dataclass
class CIResult:
    point: float
    lo: float
    hi: float
    method: str


def wilson_ci(n_correct: float, n: int, alpha: float = 0.05) -> CIResult:
    """Wilson score CI on a proportion. Accepts fractional successes (rounded for the CI)."""
    if n == 0:
        return CIResult(float("nan"), float("nan"), float("nan"), "wilson")
    # statsmodels accepts integer counts; for fractional successes (multi-attempt
    # mean-per-item) we round and report — mostly used as a cross-reference to
    # the bootstrap CI.
    k = int(round(n_correct))
    lo, hi = proportion_confint(k, n, alpha=alpha, method="wilson")
    return CIResult(n_correct / n, float(lo), float(hi), "wilson")


def newcombe_diff_ci(c1: int, n1: int, c2: int, n2: int, alpha: float = 0.05) -> CIResult:
    """Newcombe hybrid score CI for the difference of two independent proportions."""
    if n1 == 0 or n2 == 0:
        return CIResult(float("nan"), float("nan"), float("nan"), "newcombe")
    # statsmodels names this method 'newcomb' (single 'b' is intentional)
    lo, hi = confint_proportions_2indep(c1, n1, c2, n2, method="newcomb", alpha=alpha)
    point = c1 / n1 - c2 / n2
    return CIResult(point, float(lo), float(hi), "newcombe")


def paired_bootstrap_ci(y_a: np.ndarray, y_b: np.ndarray, n_resamples: int = 10000,
                        alpha: float = 0.05, seed: int = 0) -> CIResult:
    """Percentile bootstrap CI for the paired mean difference y_a - y_b.

    Uses scipy.stats.bootstrap with paired=True (resamples row indices).
    """
    y_a = np.asarray(y_a, dtype=float)
    y_b = np.asarray(y_b, dtype=float)
    if len(y_a) == 0 or len(y_a) != len(y_b):
        return CIResult(float("nan"), float("nan"), float("nan"), "bootstrap")

    def _stat(a, b, axis):
        return a.mean(axis=axis) - b.mean(axis=axis)

    res = stats.bootstrap(
        (y_a, y_b),
        statistic=_stat,
        paired=True,
        vectorized=True,
        n_resamples=n_resamples,
        confidence_level=1 - alpha,
        method="percentile",
        random_state=np.random.default_rng(seed),
    )
    point = float(y_a.mean() - y_b.mean())
    return CIResult(point, float(res.confidence_interval.low), float(res.confidence_interval.high), "bootstrap")


def paired_perm_test(y_a: np.ndarray, y_b: np.ndarray, n_resamples: int = 10000,
                     seed: int = 0) -> tuple[float, float]:
    """Two-sided paired sign-flip permutation test on the mean difference.

    Returns (observed_diff, p_value).
    """
    y_a = np.asarray(y_a, dtype=float)
    y_b = np.asarray(y_b, dtype=float)
    if len(y_a) == 0 or len(y_a) != len(y_b):
        return float("nan"), float("nan")
    if np.allclose(y_a, y_b):
        return 0.0, 1.0

    def _stat(a, b, axis):
        return (a - b).mean(axis=axis)

    # permutation_type='samples' performs sign-flips on paired observations.
    res = stats.permutation_test(
        (y_a, y_b),
        statistic=_stat,
        permutation_type="samples",
        vectorized=True,
        alternative="two-sided",
        n_resamples=n_resamples,
        random_state=np.random.default_rng(seed),
    )
    return float(y_a.mean() - y_b.mean()), float(res.pvalue)


def paired_tost(y_a: np.ndarray, y_b: np.ndarray, margin: float = 0.05) -> tuple[float, bool]:
    """Two one-sided test (TOST) for equivalence of paired means at ±margin.

    Wraps `statsmodels.stats.weightstats.ttost_paired`. Returns
    (pvalue, reject_null_of_difference). The TOST null is "|mean(y_a - y_b)| >= margin";
    rejecting it (pvalue < 0.05) supports "the two means are equivalent within
    ±margin".

    Returns (NaN, False) when the difference vector is degenerate.
    """
    y_a = np.asarray(y_a, dtype=float)
    y_b = np.asarray(y_b, dtype=float)
    if len(y_a) == 0 or len(y_a) != len(y_b):
        return float("nan"), False
    if np.allclose(y_a, y_b):
        # zero difference — trivially equivalent at any positive margin
        return 0.0, True
    try:
        pvalue, _t1, _t2 = ttost_paired(y_a, y_b, low=-margin, upp=+margin)
        return float(pvalue), bool(pvalue < 0.05)
    except (ValueError, ZeroDivisionError):
        return float("nan"), False


def unpaired_tost(c1: int, n1: int, c2: int, n2: int, margin: float = 0.05) -> tuple[float, bool]:
    """TOST equivalence for two independent binomial proportions at ±margin.

    Uses `statsmodels.stats.weightstats.ttost_ind` on the implied 0/1 vectors.
    """
    if n1 == 0 or n2 == 0:
        return float("nan"), False
    x1 = np.concatenate([np.ones(c1), np.zeros(n1 - c1)])
    x2 = np.concatenate([np.ones(c2), np.zeros(n2 - c2)])
    try:
        pvalue, _t1, _t2 = ttost_ind(x1, x2, low=-margin, upp=+margin, usevar="unequal")
        return float(pvalue), bool(pvalue < 0.05)
    except (ValueError, ZeroDivisionError):
        return float("nan"), False


def paired_wilcoxon(y_a: np.ndarray, y_b: np.ndarray) -> float:
    """Wilcoxon signed-rank p-value for paired differences (sanity-check companion
    to the sign-flip permutation test).

    Returns NaN if the difference vector is all zeros (Wilcoxon is undefined
    when no nonzero differences exist).
    """
    y_a = np.asarray(y_a, dtype=float)
    y_b = np.asarray(y_b, dtype=float)
    if len(y_a) == 0 or len(y_a) != len(y_b):
        return float("nan")
    d = y_a - y_b
    if np.allclose(d, 0):
        return float("nan")
    try:
        res = stats.wilcoxon(d, alternative="two-sided", zero_method="wilcox", correction=False)
        return float(res.pvalue)
    except ValueError:
        return float("nan")


def bh_adjust(pvals: list[float]) -> list[float]:
    """Benjamini–Hochberg FDR-adjusted q-values."""
    arr = np.asarray(pvals, dtype=float)
    mask = ~np.isnan(arr)
    out = np.full_like(arr, np.nan, dtype=float)
    if mask.sum() == 0:
        return out.tolist()
    _, q, _, _ = multipletests(arr[mask], method="fdr_bh")
    out[mask] = q
    return out.tolist()


# ---------------------------------------------------------------------------
# Stratified suite-average bootstrap

def suite_average_bootstrap(
    per_task_a: dict[str, np.ndarray],
    per_task_b: dict[str, np.ndarray] | None = None,
    n_resamples: int = 10000,
    alpha: float = 0.05,
    seed: int = 0,
) -> CIResult:
    """Stratified bootstrap of an unweighted task-mean average.

    For each replicate:
      - resample item-level outcomes within each task with replacement,
      - compute the task mean,
      - average across tasks (unweighted).

    If per_task_b is provided, the statistic is the *difference* of unweighted
    suite averages between A and B (paired within each task — same item indices
    are used to resample both A and B).
    """
    rng = np.random.default_rng(seed)
    tasks = list(per_task_a.keys())
    if not tasks:
        return CIResult(float("nan"), float("nan"), float("nan"), "stratified_bootstrap")
    arrs_a = {t: np.asarray(per_task_a[t], dtype=float) for t in tasks}
    arrs_b = {t: np.asarray(per_task_b[t], dtype=float) for t in tasks} if per_task_b else None
    n_per = {t: len(arrs_a[t]) for t in tasks}

    obs_a = np.mean([arrs_a[t].mean() for t in tasks])
    if arrs_b is not None:
        obs_b = np.mean([arrs_b[t].mean() for t in tasks])
        observed = obs_a - obs_b
    else:
        observed = obs_a

    samples = np.empty(n_resamples, dtype=float)
    for r in range(n_resamples):
        if arrs_b is None:
            means = []
            for t in tasks:
                idx = rng.integers(0, n_per[t], n_per[t])
                means.append(arrs_a[t][idx].mean())
            samples[r] = np.mean(means)
        else:
            ma = []; mb = []
            for t in tasks:
                idx = rng.integers(0, n_per[t], n_per[t])
                ma.append(arrs_a[t][idx].mean())
                mb.append(arrs_b[t][idx].mean())
            samples[r] = np.mean(ma) - np.mean(mb)
    lo, hi = np.quantile(samples, [alpha / 2, 1 - alpha / 2])
    return CIResult(float(observed), float(lo), float(hi), "stratified_bootstrap")


# ---------------------------------------------------------------------------
# Power / MDE for paired binomial via simulation
#
# We match the test that analyze.py reports: paired sign-flip permutation
# on per-item differences. This keeps MDE and reported p-values on the same
# footing. McNemar is asymptotically equivalent for binary paired data but
# differs slightly at small N.

def _simulate_paired_bernoulli(n: int, p_a: float, p_b: float, rho: float, rng: np.random.Generator) -> tuple[np.ndarray, np.ndarray]:
    """Generate n pairs of correlated Bernoulli(p_a, p_b) outcomes via Gaussian copula.

    rho is the *latent* (Gaussian) correlation, not the achieved Bernoulli
    Pearson correlation. The two coincide only at p=0.5; at p=0.05 with rho=0.3
    the achieved Bernoulli correlation is roughly 0.10. mde_paired_binomial
    accepts an `achieved_rho` parameter to back-calibrate.
    """
    cov = np.array([[1.0, rho], [rho, 1.0]])
    z = rng.multivariate_normal([0.0, 0.0], cov, size=n)
    a = (z[:, 0] < stats.norm.ppf(p_a)).astype(int)
    b = (z[:, 1] < stats.norm.ppf(p_b)).astype(int)
    return a, b


def latent_rho_for_target(p_a: float, p_b: float, target_bernoulli_rho: float,
                          n_calibration: int = 50000, seed: int = 0) -> float:
    """Find the Gaussian-copula latent rho that yields the requested Bernoulli rho.

    Bisection over latent rho in [-0.99, 0.99]. ~0.01 precision is enough.
    """
    if target_bernoulli_rho == 0:
        return 0.0
    rng = np.random.default_rng(seed)
    lo, hi = -0.99, 0.99
    for _ in range(15):
        mid = (lo + hi) / 2
        a, b = _simulate_paired_bernoulli(n_calibration, p_a, p_b, mid, rng)
        achieved = float(np.corrcoef(a, b)[0, 1]) if a.std() > 0 and b.std() > 0 else 0.0
        if achieved < target_bernoulli_rho:
            lo = mid
        else:
            hi = mid
    return (lo + hi) / 2


def _power_at(n: int, p_a: float, p_b: float, latent_rho: float, alpha: float,
              n_trials: int, seed: int, n_perm: int) -> float:
    """Empirical power of paired sign-flip permutation test for binary paired data.

    Fully vectorized: simulate all n_trials of paired Bernoullis at once, then
    apply the sign-flip permutation in one batched matrix operation.
    """
    rng = np.random.default_rng(seed)
    cov = np.array([[1.0, latent_rho], [latent_rho, 1.0]])
    z = rng.multivariate_normal([0.0, 0.0], cov, size=(n_trials, n))  # (T, n, 2)
    qa = stats.norm.ppf(p_a); qb = stats.norm.ppf(p_b)
    a = (z[..., 0] < qa).astype(np.float32)
    b = (z[..., 1] < qb).astype(np.float32)
    d = a - b  # (T, n)
    obs = d.mean(axis=1)  # (T,)
    abs_obs = np.abs(obs)
    rejects = 0
    # process trials in chunks to keep memory bounded for the (T, n_perm, n) sign tensor
    chunk = max(1, min(n_trials, 4_000_000 // (n_perm * max(n, 1))))
    for s in range(0, n_trials, chunk):
        e = min(s + chunk, n_trials)
        d_chunk = d[s:e]                                          # (c, n)
        signs = rng.choice([-1.0, 1.0], size=(e - s, n_perm, n)).astype(np.float32)  # (c, P, n)
        null = (signs * d_chunk[:, None, :]).mean(axis=2)         # (c, P)
        # two-sided p-value per trial
        p_vals = (np.abs(null) >= abs_obs[s:e, None]).mean(axis=1)
        rejects += int((p_vals <= alpha).sum())
    return rejects / n_trials


def mde_paired_binomial(p_baseline: float, n: int, target_bernoulli_rho: float = 0.3,
                        alpha: float = 0.05, target_power: float = 0.8,
                        n_trials: int = 1000, n_perm: int = 1000, seed: int = 0,
                        delta_grid: np.ndarray | None = None) -> tuple[float, float]:
    """Smallest |delta| at which power >= target_power for paired binary outcomes.

    Uses paired sign-flip permutation as the rejection rule (matches analyze.py).
    `target_bernoulli_rho` is the *Bernoulli Pearson correlation* between paired
    item-level outcomes; the function calibrates the latent Gaussian-copula rho
    to achieve it.

    Returns (mde, latent_rho_used).
    """
    if delta_grid is None:
        delta_grid = np.linspace(0.005, 0.5, 50)
    p_baseline = float(np.clip(p_baseline, 1e-3, 1 - 1e-3))
    # Calibrate latent rho once at midpoint of grid (rho is roughly invariant in delta)
    p_mid = float(np.clip(p_baseline + 0.1, 1e-3, 1 - 1e-3))
    latent_rho = latent_rho_for_target(p_baseline, p_mid, target_bernoulli_rho, seed=seed)
    for d in delta_grid:
        p_b = float(np.clip(p_baseline + d, 1e-3, 1 - 1e-3))
        pwr = _power_at(n, p_baseline, p_b, latent_rho, alpha, n_trials, seed, n_perm)
        if pwr >= target_power:
            return float(d), latent_rho
    return float("nan"), latent_rho
