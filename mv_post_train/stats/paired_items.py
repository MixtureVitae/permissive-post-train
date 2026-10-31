"""Paired question-level intervals for Phase 2 changes.

The manifest holds per-task means. An interval for a change needs the
per-question records, which `manifest/phase2.py` loads at the pinned dataset
revision. Each question's score is its mean over the three runs. A replicate
resamples the questions with replacement, using the same draw for both
checkpoints so the pairing is kept, and recomputes the difference of the
means. Each family of changes gets one Benjamini-Hochberg correction. The seed
and replicate count are those of the Protocol A contrasts
(`stats/factorial.py`), and changes are drawn in the order given, so the
intervals are reproducible.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd
from statsmodels.stats.multitest import multipletests

from .factorial import BOOT_SEED, N_BOOT, _two_sided_p

RUNS = 3


def item_scores(meta: pd.DataFrame, base_model: str, stage: str, task: str) -> pd.Series:
    """Percent score per question (indexed by `example_id`), mean over the runs."""
    rows = meta[(meta["base_model"] == base_model) & (meta["stage"] == stage)
                & (meta["task"] == task)]
    if rows.empty:
        raise ValueError(f"No {task} records for {base_model} {stage}.")
    if rows.duplicated(["example_id", "run_idx"]).any():
        raise ValueError(f"{base_model} {stage} {task}: a question has two answers in one run.")
    per = rows.groupby("example_id")["is_correct"].agg(["mean", "size"])
    if (per["size"] != RUNS).any():
        raise ValueError(f"{base_model} {stage} {task}: some questions lack {RUNS} runs.")
    return (per["mean"] * 100.0).sort_index()


def run_scores(meta: pd.DataFrame, base_model: str, stage: str, task: str) -> list[float]:
    """Percent score of each run, sorted (the runs' order is not tied to a seed)."""
    rows = meta[(meta["base_model"] == base_model) & (meta["stage"] == stage)
                & (meta["task"] == task)]
    return sorted(float(v) * 100.0 for v in rows.groupby("run_idx")["is_correct"].mean())


@dataclass(frozen=True)
class PairedChange:
    estimate: float
    lo: float
    hi: float
    p: float
    q: float


def paired_changes(pairs: list[tuple[pd.Series, pd.Series]], *, n_boot: int = N_BOOT,
                   seed: int = BOOT_SEED) -> list[PairedChange]:
    """One family of (child, parent) question scores -> changes with 95% intervals."""
    rng = np.random.default_rng(seed)
    rows = []
    for child, parent in pairs:
        if set(child.index) != set(parent.index):
            raise ValueError("The two checkpoints were scored on different questions.")
        diff = (child - parent.reindex(child.index)).to_numpy()
        draws = diff[rng.integers(0, diff.size, size=(n_boot, diff.size))].mean(axis=1)
        lo, hi = np.percentile(draws, [2.5, 97.5])
        rows.append((float(diff.mean()), float(lo), float(hi), _two_sided_p(draws)))
    q = multipletests([r[3] for r in rows], method="fdr_bh")[1] if rows else []
    return [PairedChange(*r, float(qv)) for r, qv in zip(rows, q)]
