"""Paired contrasts between two Phase 1 checkpoints under Protocol A.

The method is the one `stats/factorial.py` uses for the front-loading study,
applied to any two checkpoints of Phase 1:

  - each item is scored as its mean over the three decoding seeds;
  - a replicate resamples the items of each benchmark with replacement, with
    the same draw for every checkpoint, so the comparison stays paired;
  - a benchmark contrast is the difference of the two benchmark means, and the
    suite contrast is the difference of the two eleven-benchmark averages;
  - a reweighted view (`domain_views`) is the same difference with other
    weights on the benchmark means, from the same replicates;
  - the interval is the 2.5th to 97.5th percentile of the replicates, the
    p-value is the two-sided bootstrap p-value, and each family of contrasts
    gets one Benjamini-Hochberg correction.

The seed and replicate count are those of the factorial, so the numbers are
reproducible. All checkpoints of one call share one set of draws.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd
from statsmodels.stats.multitest import multipletests

from ..data.protocol_a import BENCHMARKS, DOMAIN, DOMAINS, ZOO_KEYS
from .factorial import BOOT_SEED, N_BOOT, _two_sided_p

SUITE = "__suite__"


def domain_views() -> dict[str, np.ndarray]:
    """Weights of the benchmarks, in BENCHMARKS order, for the suite average
    reweighted by domain: `equal_domain` gives each of the four domains a
    quarter, split evenly over its benchmarks, and `drop_<domain>` is the
    plain mean of the benchmarks of the other three. These are the views of
    the front-loading study's robustness table (`stats/factorial.py`)."""
    domain = np.array([DOMAIN[b] for b in BENCHMARKS])
    views = {"equal_domain": np.array([1.0 / (len(DOMAINS) * (domain == d).sum())
                                       for d in domain])}
    for d in DOMAINS:
        keep = (domain != d).astype(float)
        views[f"drop_{d}"] = keep / keep.sum()
    return views


@dataclass(frozen=True)
class Cell:
    """One checkpoint of Phase 1: model family, context length and stage."""
    group: str
    context_length: int
    stage: str

    def select(self, df: pd.DataFrame) -> pd.DataFrame:
        return df[(df["model_group"] == self.group)
                  & (df["context_length"] == self.context_length)
                  & (df["stage"] == self.stage)]


@dataclass(frozen=True)
class PairedContrast:
    a: Cell
    b: Cell
    benchmark: str        # a benchmark, SUITE for the eleven-benchmark average, or a view
    level_a: float        # percent
    level_b: float
    estimate: float       # level_a - level_b, in points
    lo: float
    hi: float
    p: float
    q: float = float("nan")


class _ItemMatrix:
    """Items x checkpoints matrix of seed-averaged scores in percent. Rows are
    sorted by (benchmark, sample_id) and must be complete for every checkpoint."""

    def __init__(self, df: pd.DataFrame, cells: list[Cell]):
        cols = []
        for c in cells:
            sub = c.select(df)
            if sub.empty:
                raise ValueError(f"no records for {c}")
            if sub["model_id"].nunique() != 1:
                raise ValueError(f"{c} maps to several checkpoints")
            if sub["seed"].nunique() != 3:
                raise ValueError(f"{c} does not have three seeds")
            cols.append(sub.groupby(["benchmark", "sample_id"])["correct"].mean().rename(c))
        wide = pd.concat(cols, axis=1)
        if wide.isna().any().any():
            raise ValueError("the checkpoints were not scored on the same items")
        missing = set(BENCHMARKS) - set(wide.index.get_level_values("benchmark"))
        if missing:
            raise ValueError(f"benchmarks without records: {sorted(missing)}")
        self.col = {c: i for i, c in enumerate(cells)}
        self.values = wide.to_numpy() * 100.0
        bench = wide.index.get_level_values("benchmark").to_numpy()
        self.blocks = [np.flatnonzero(bench == b) for b in BENCHMARKS]

    def means(self, blocks: list[np.ndarray]) -> np.ndarray:
        """Benchmarks x checkpoints means."""
        return np.stack([self.values[idx].mean(axis=0) for idx in blocks])

    def replicates(self, rng: np.random.Generator, n: int) -> np.ndarray:
        """n x benchmarks x checkpoints replicate means."""
        return np.stack([self.means([rng.choice(idx, size=idx.size, replace=True)
                                     for idx in self.blocks]) for _ in range(n)])


def paired_contrasts(df: pd.DataFrame, pairs: list[tuple[Cell, Cell]], *,
                     n_boot: int = N_BOOT, seed: int = BOOT_SEED,
                     views: dict[str, np.ndarray] | None = None) -> list[PairedContrast]:
    """For each (a, b) pair: one contrast per benchmark, then the suite
    contrast, then one per view of `views` (name -> benchmark weights, as
    `domain_views` gives). The views use the same replicates, so they leave
    the other contrasts unchanged. q-values are left for the caller, who
    knows the families."""
    for col in ZOO_KEYS:
        if col not in df:
            raise ValueError(f"records lack column {col}")
    cells = list(dict.fromkeys(c for pair in pairs for c in pair))
    m = _ItemMatrix(df, cells)
    observed = m.means(m.blocks)                         # benchmarks x checkpoints
    draws = m.replicates(np.random.default_rng(seed), n_boot)
    out = []
    for a, b in pairs:
        i, j = m.col[a], m.col[b]
        for k, bench in enumerate([*BENCHMARKS, SUITE]):
            if bench == SUITE:
                la, lb = observed[:, i].mean(), observed[:, j].mean()
                d = draws[:, :, i].mean(axis=1) - draws[:, :, j].mean(axis=1)
            else:
                la, lb = observed[k, i], observed[k, j]
                d = draws[:, k, i] - draws[:, k, j]
            lo, hi = np.percentile(d, [2.5, 97.5])
            out.append(PairedContrast(a, b, bench, float(la), float(lb), float(la - lb),
                                      float(lo), float(hi), _two_sided_p(d)))
        for name, w in (views or {}).items():
            la, lb = float(observed[:, i] @ w), float(observed[:, j] @ w)
            d = draws[:, :, i] @ w - draws[:, :, j] @ w
            lo, hi = np.percentile(d, [2.5, 97.5])
            out.append(PairedContrast(a, b, name, la, lb, la - lb, float(lo), float(hi),
                                      _two_sided_p(d)))
    return out


def with_q(contrasts: list[PairedContrast], family: list[int]) -> list[PairedContrast]:
    """Benjamini-Hochberg q-values within the family given by the indices."""
    q = multipletests([contrasts[i].p for i in family], method="fdr_bh")[1]
    out = list(contrasts)
    for i, qv in zip(family, q):
        c = out[i]
        out[i] = PairedContrast(c.a, c.b, c.benchmark, c.level_a, c.level_b, c.estimate,
                                c.lo, c.hi, c.p, float(qv))
    return out
