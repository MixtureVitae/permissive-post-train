"""Paired question-level intervals for changes on the language suite.

The method is the one `appendix/statistics.tex` describes for Protocol A, and
`stats/factorial.py` implements: each question gets one score per checkpoint,
a replicate resamples the questions with replacement within each task, with
the same draw for every checkpoint so the pairing is kept, and the 11-task
average is recomputed as the plain mean of the task means. Two-sided p-values
come from the bootstrap distribution. Each family of changes gets one
Benjamini-Hochberg correction. The replicate count and seed are those of
`stats/factorial.py`.

A question's score is 0 or 1 under one prompt-example seed (`default`, the
seed the paper's tables use), or its mean over the three seeds. The seeds only
change the few-shot examples in the prompt, so an interval describes the
uncertainty from the sample of questions, not from training.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd
from statsmodels.stats.multitest import multipletests

from ..data.benchmarks import (
    COPA, LAMBADA_OPENAI, LANGUAGE_SUITE_TASKS, MMLU, OPENBOOKQA, WINOGRANDE,
)
from .factorial import BOOT_SEED, N_BOOT, _two_sided_p

AVG = "avg"
#: Tasks run with 0 shots (Table 25), so the prompt-example seed cannot change
#: their answers.
SEED_INVARIANT = (COPA, LAMBADA_OPENAI, MMLU, OPENBOOKQA, WINOGRANDE)


def check_same_questions(items: pd.DataFrame) -> None:
    """Questions are paired by their item key (`doc_id`, with the subject for
    MMLU). Stop unless every key names the same question, by `doc_hash`, in
    every folder."""
    hashes = items.groupby(["task", "item"])["doc_hash"].nunique()
    bad = hashes[hashes > 1]
    if not bad.empty:
        task, item = bad.index[0]
        raise ValueError(f"{len(bad)} question keys name different questions in different "
                         f"folders, for example {task} {item}.")


#: Folders run in different sessions may answer this share of the questions
#: no seed can change differently, pooled over those tasks. R23 ran seed 1234
#: in one session and seeds 42 and 2025 in another, and the two sessions
#: differ on 1.3 to 2.5% of these questions, while two different checkpoints
#: of R23 differ on 10.9% or more, even a 4K base and its 16K extension.
SESSION_TOLERANCE = 0.05


def check_seed_invariant_tasks(items: pd.DataFrame, folders: list[str], name: str,
                               sessions: list[list[str]] | None = None) -> None:
    """The seed folders of one checkpoint must give the same answers on the
    tasks the seed cannot change. Otherwise they may hold different models.
    `sessions` groups the folders that were run together: those must agree
    on every such question, and folders of different sessions on all but
    SESSION_TOLERANCE of them."""
    rows = items[items["folder"].isin(folders) & items["task"].isin(SEED_INVARIANT)]
    wide = rows.pivot_table(index=["task", "item"], columns="folder", values="score")
    for group in sessions or [folders]:
        sub = wide.reindex(columns=group)
        differ = sub.nunique(axis=1) > 1
        if differ.any() or sub.isna().any().any():
            task = sub.index[differ.to_numpy() | sub.isna().any(axis=1).to_numpy()][0][0]
            raise ValueError(f"{name}: the seed folders {group} disagree on {task}, which no "
                             "seed can change, so they may not hold the same model.")
    share = float((wide.nunique(axis=1) > 1).mean())
    if share > SESSION_TOLERANCE:
        raise ValueError(f"{name}: the runs of {folders} disagree on {share:.1%} of the "
                         f"questions no seed can change, more than two sessions of one model "
                         f"do ({SESSION_TOLERANCE:.0%}), so they may not hold the same model.")


def question_scores(items: pd.DataFrame, folders: list[str]) -> pd.Series:
    """Percent score per (task, item), the mean over `folders` (one folder per
    seed used). Every folder must hold the same questions."""
    parts = []
    for folder in folders:
        rows = items[items["folder"] == folder]
        if rows.empty:
            raise ValueError(f"No language records for {folder}.")
        parts.append(rows.set_index(["task", "item"])["score"].sort_index())
    for other, folder in zip(parts[1:], folders[1:]):
        if not other.index.equals(parts[0].index):
            raise ValueError(f"{folder} and {folders[0]} hold different questions.")
    return pd.concat(parts, axis=1).mean(axis=1) * 100.0


@dataclass(frozen=True)
class Matrix:
    """Questions x checkpoints of one family, with one block of rows per task."""
    columns: list[str]
    values: np.ndarray
    blocks: list[np.ndarray]

    @classmethod
    def build(cls, scores: dict[str, pd.Series]) -> "Matrix":
        wide = pd.concat(scores, axis=1)
        if wide.isna().any().any():
            bad = wide.columns[wide.isna().any()].tolist()
            raise ValueError(f"Checkpoints {bad} were not scored on the same questions.")
        tasks = wide.index.get_level_values("task").to_numpy()
        missing = sorted(set(LANGUAGE_SUITE_TASKS) - set(tasks))
        if missing:
            raise ValueError(f"No questions for {missing}.")
        return cls(list(wide.columns), wide.to_numpy(),
                   [np.flatnonzero(tasks == t) for t in LANGUAGE_SUITE_TASKS])

    def task_means(self) -> np.ndarray:
        """Tasks x checkpoints."""
        return np.stack([self.values[idx].mean(axis=0) for idx in self.blocks])

    def replicates(self, n_boot: int, seed: int) -> np.ndarray:
        """Replicates x tasks x checkpoints, one shared draw per replicate."""
        rng = np.random.default_rng(seed)
        out = np.empty((n_boot, len(self.blocks), self.values.shape[1]))
        for i in range(n_boot):
            for t, idx in enumerate(self.blocks):
                out[i, t] = self.values[idx[rng.integers(0, idx.size, idx.size)]].mean(axis=0)
        return out


def contrasts(matrix: Matrix, specs: list[tuple[str, str, dict[str, float]]], *,
              n_boot: int = N_BOOT, seed: int = BOOT_SEED) -> pd.DataFrame:
    """One row per (contrast, task), the 11-task average as task `avg`. A
    contrast is (family, label, weights), the weighted sum of its checkpoints.
    Columns: family, label, task, estimate, lo, hi, p, and q with one
    correction over the averages of each family and one over its per-task
    contrasts."""
    obs = matrix.task_means()
    draws = matrix.replicates(n_boot, seed)
    col = {c: i for i, c in enumerate(matrix.columns)}
    rows = []
    for family, label, weights in specs:
        diff_obs = sum(w * obs[:, col[k]] for k, w in weights.items())
        diff = sum(w * draws[:, :, col[k]] for k, w in weights.items())
        series = [(t, diff_obs[k], diff[:, k]) for k, t in enumerate(LANGUAGE_SUITE_TASKS)]
        series.append((AVG, diff_obs.mean(), diff.mean(axis=1)))
        for task, est, d in series:
            lo, hi = np.percentile(d, [2.5, 97.5])
            rows.append(dict(family=family, label=label, task=task, estimate=float(est),
                             lo=float(lo), hi=float(hi), p=_two_sided_p(d)))
    out = pd.DataFrame(rows)
    out["q"] = np.nan
    for family in out["family"].unique():
        for is_avg in (True, False):
            sel = (out["family"] == family) & ((out["task"] == AVG) == is_avg)
            if sel.any():
                out.loc[sel, "q"] = multipletests(out.loc[sel, "p"], method="fdr_bh")[1]
    return out


def changes(matrix: Matrix, pairs: list[tuple[str, str]], *, n_boot: int = N_BOOT,
            seed: int = BOOT_SEED) -> pd.DataFrame:
    """One row per (child, parent, task), the 11-task average as task `avg`.
    Columns: estimate, lo, hi, p, and q with one correction over the averages
    of the family and one over its per-task changes."""
    specs = [("changes", f"{child}|{parent}", {child: 1.0, parent: -1.0})
             for child, parent in pairs]
    out = contrasts(matrix, specs, n_boot=n_boot, seed=seed)
    out[["child", "parent"]] = out["label"].str.split("|", expand=True)
    return out[["child", "parent", "task", "estimate", "lo", "hi", "p", "q"]]
