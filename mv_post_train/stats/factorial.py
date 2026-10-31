"""Summaries and contrasts for the Protocol A factorial and Phase 1.

Aggregation order, which every table and figure shares:

  item scores -> benchmark accuracy per (cell, seed)
              -> suite average per (cell, seed), unweighted over 11 benchmarks
              -> mean and sample SD over the three decoding seeds

The SD therefore measures decoding-seed variation of one trained checkpoint,
not variation across training runs.

Contrasts use a paired item-level bootstrap. Scores are first averaged over
seeds per item, items are resampled within each benchmark (which keeps the
suite's benchmark composition fixed in every replicate), and the suite average
is recomputed per replicate. Each family of contrasts gets its own
Benjamini-Hochberg correction. The seed, replicate count and draw order match
the paper's generator, so the printed intervals reproduce exactly.
"""
from __future__ import annotations

from dataclasses import asdict, dataclass

import numpy as np
import pandas as pd
from statsmodels.stats.multitest import multipletests

from ..data.protocol_a import (
    BENCHMARKS,
    CONTEXTS,
    DOMAIN,
    DOMAINS,
    PERMISSIVE,
    POOLS,
    STAGES,
)

N_BOOT = 2000
BOOT_SEED = 20260904


def _mean_sd(frame: pd.DataFrame, keys: list[str], col: str) -> pd.DataFrame:
    return (frame.groupby(keys, observed=True)[col]
            .agg(mean="mean", sd=lambda s: s.std(ddof=1))
            .reset_index())


def benchmark_scores(df: pd.DataFrame, keys: list[str]) -> pd.DataFrame:
    """Accuracy in percent per (cell, seed, benchmark), with its domain."""
    out = (df.groupby(keys + ["seed", "benchmark"], observed=True)["correct"]
           .mean().mul(100.0).reset_index(name="score"))
    out["domain"] = out["benchmark"].map(DOMAIN)
    return out


def suite_average(scores: pd.DataFrame, keys: list[str]) -> pd.DataFrame:
    """Unweighted mean over the benchmarks present, per seed, then over seeds."""
    per_seed = (scores.groupby(keys + ["seed"], observed=True)["score"]
                .mean().reset_index(name="suite"))
    return _mean_sd(per_seed, keys, "suite")


def benchmark_summary(scores: pd.DataFrame, keys: list[str]) -> pd.DataFrame:
    """Per-benchmark mean and SD over seeds, plus the suite average as the
    pseudo-benchmark `__suite__`."""
    per_bench = _mean_sd(scores, keys + ["benchmark"], "score")
    suite = suite_average(scores, keys).assign(benchmark="__suite__")
    return pd.concat([per_bench, suite], ignore_index=True)


def equal_domain_average(scores: pd.DataFrame, keys: list[str]) -> pd.DataFrame:
    """Mean of the four domain means, so math (5 of 11 benchmarks) does not
    dominate."""
    dom = (scores.groupby(keys + ["seed", "domain"], observed=True)["score"]
           .mean().reset_index(name="dom"))
    per_seed = (dom.groupby(keys + ["seed"], observed=True)["dom"]
                .mean().reset_index(name="suite"))
    return _mean_sd(per_seed, keys, "suite")


def leave_one_domain_out(scores: pd.DataFrame, keys: list[str]) -> pd.DataFrame:
    """Suite average with each domain removed in turn (column `dropped`)."""
    return pd.concat(
        [suite_average(scores[scores["domain"] != d], keys).assign(dropped=d)
         for d in DOMAINS],
        ignore_index=True)


def domain_scores(scores: pd.DataFrame, keys: list[str]) -> pd.DataFrame:
    """Domain mean per seed, then mean over seeds."""
    return (scores.groupby(keys + ["seed", "domain"], observed=True)["score"].mean()
            .groupby(keys + ["domain"], observed=True).mean()
            .reset_index(name="score"))


def language_average(language: pd.DataFrame) -> pd.DataFrame:
    """Unweighted mean over the eleven language tasks per checkpoint."""
    return (language.groupby(["model_group", "front_loading", "context_length"],
                             observed=True)["performance"]
            .mean().reset_index(name="language"))


# ---------------------------------------------------------------------------
# Paired item-level bootstrap


@dataclass(frozen=True)
class Contrast:
    family: str           # "front-loading" | "permissiveness" | "interaction" | "permissive pair"
    label: str            # substrate, or "MV - <substrate>"
    context_length: int
    stage: str
    estimate: float
    lo: float
    hi: float
    p: float
    q: float = float("nan")


class _ArmMatrix:
    """Items x arms matrix of seed-averaged scores (percent) for one
    (context, stage) cell of the factorial. Arms are `<substrate>|FL` and
    `<substrate>|noFL`, columns sorted by name; rows are sorted by
    (benchmark, sample_id)."""

    def __init__(self, df: pd.DataFrame, context_length: int, stage: str):
        cell = df[(df["context_length"] == context_length) & (df["stage"] == stage)]
        item = (cell.groupby(["model_group", "front_loading", "benchmark", "sample_id"],
                             observed=True)["correct"].mean().reset_index())
        item["arm"] = item["model_group"] + np.where(item["front_loading"], "|FL", "|noFL")
        wide = item.pivot_table(index=["benchmark", "sample_id"], columns="arm",
                                values="correct")
        if wide.isna().any().any():
            raise ValueError(f"missing item scores at {context_length}/{stage}")
        self.col = {arm: i for i, arm in enumerate(wide.columns)}
        self.values = wide.to_numpy() * 100.0
        bench = wide.index.get_level_values("benchmark").to_numpy()
        self.blocks = [np.flatnonzero(bench == b) for b in BENCHMARKS]

    def _suite(self, blocks: list[np.ndarray]) -> np.ndarray:
        return np.mean([self.values[idx].mean(axis=0) for idx in blocks], axis=0)

    def observed(self) -> np.ndarray:
        return self._suite(self.blocks)

    def replicates(self, rng: np.random.Generator, n: int) -> np.ndarray:
        out = np.empty((n, self.values.shape[1]))
        for i in range(n):
            out[i] = self._suite([rng.choice(idx, size=idx.size, replace=True)
                                  for idx in self.blocks])
        return out


def _two_sided_p(draws: np.ndarray) -> float:
    """Bootstrap p-value for a zero contrast, with the +1/n floor."""
    below = float(np.mean(draws <= 0.0))
    above = float(np.mean(draws >= 0.0))
    return min(1.0, 2.0 * min(below, above) + 1.0 / draws.size)


def factorial_contrasts(df: pd.DataFrame, *, n_boot: int = N_BOOT, seed: int = BOOT_SEED,
                        pools: tuple[str, ...] = POOLS,
                        permissive: tuple[str, ...] = PERMISSIVE) -> pd.DataFrame:
    """Every Protocol A contrast on the suite average, per (context, stage):

      front-loading   FL minus no-FL within each substrate
      permissiveness  each permissive substrate's FL minus each non-permissive
                      substrate's FL (MV|FL minus Nemotron|FL and FineWeb-Edu|FL)
      interaction     each permissive substrate's front-loading effect minus
                      the mean effect of the non-permissive substrates

    The replicates resample items only, so adding substrates leaves every
    other contrast's estimate and interval unchanged. Its q-value can move,
    because the families grow.
    """
    others_ = [p for p in pools if p not in permissive]
    rng = np.random.default_rng(seed)
    rows: list[Contrast] = []
    for ctx in CONTEXTS:
        for stage in STAGES:
            mat = _ArmMatrix(df, ctx, stage)
            obs, rep, c = mat.observed(), mat.replicates(rng, n_boot), mat.col

            def diff(x: np.ndarray, a: str, b: str) -> np.ndarray:
                return x[..., c[a]] - x[..., c[b]]

            def add(family: str, label: str, est: float, draws: np.ndarray) -> None:
                lo, hi = np.percentile(draws, [2.5, 97.5])
                rows.append(Contrast(family, label, ctx, stage, float(est),
                                     float(lo), float(hi), _two_sided_p(draws)))

            for pool in pools:
                add("front-loading", pool, diff(obs, f"{pool}|FL", f"{pool}|noFL"),
                    diff(rep, f"{pool}|FL", f"{pool}|noFL"))
            for perm in permissive:
                for pool in others_:
                    add("permissiveness", f"{perm} - {pool}", diff(obs, f"{perm}|FL", f"{pool}|FL"),
                        diff(rep, f"{perm}|FL", f"{pool}|FL"))

            def interaction(x: np.ndarray, perm: str) -> np.ndarray:
                others = sum(diff(x, f"{p}|FL", f"{p}|noFL") for p in others_) / len(others_)
                return diff(x, f"{perm}|FL", f"{perm}|noFL") - others

            for perm in permissive:
                add("interaction", f"{perm} - non-permissive", interaction(obs, perm),
                    interaction(rep, perm))

    out = pd.DataFrame([asdict(r) for r in rows])
    for _, block in out.groupby("family", observed=True):
        out.loc[block.index, "q"] = multipletests(block["p"].to_numpy(),
                                                  method="fdr_bh")[1]
    return out


def permissive_pair_contrasts(df: pd.DataFrame, *, n_boot: int = N_BOOT, seed: int = BOOT_SEED,
                              permissive: tuple[str, ...] = PERMISSIVE) -> pd.DataFrame:
    """The front-loaded permissive substrates against each other on the suite
    average, per (context, stage): the first minus each later one (MV|FL minus
    Comma|FL). It draws the replicates exactly as factorial_contrasts does, in
    the same order from the same seed, so these intervals rest on the same
    resampled items as Table 28's. Its contrasts are one Benjamini-Hochberg
    family of their own, so they leave Table 28's q-values alone. Empty with
    fewer than two permissive substrates."""
    rng = np.random.default_rng(seed)
    rows: list[Contrast] = []
    pairs = [(a, b) for k, a in enumerate(permissive) for b in permissive[k + 1:]]
    for ctx in CONTEXTS:
        for stage in STAGES:
            mat = _ArmMatrix(df, ctx, stage)
            obs, rep, c = mat.observed(), mat.replicates(rng, n_boot), mat.col
            for a, b in pairs:
                draws = rep[..., c[f"{a}|FL"]] - rep[..., c[f"{b}|FL"]]
                lo, hi = np.percentile(draws, [2.5, 97.5])
                rows.append(Contrast("permissive pair", f"{a} - {b}", ctx, stage,
                                     float(obs[c[f"{a}|FL"]] - obs[c[f"{b}|FL"]]),
                                     float(lo), float(hi), _two_sided_p(draws)))
    out = pd.DataFrame([asdict(r) for r in rows])
    if len(out):
        out["q"] = multipletests(out["p"].to_numpy(), method="fdr_bh")[1]
    return out
