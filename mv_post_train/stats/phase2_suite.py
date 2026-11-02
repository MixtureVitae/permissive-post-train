"""Paired question-level intervals for Phase 2 changes on averages of several tasks.

`paired_items.py` gives one task's change an interval (Table 4). This module
does the same for the suite average, for the average of one domain, and for
the difference between two changes, such as the OT3 exclusion's two gains.

A contrast is a weighted sum of checkpoints, for example child minus parent,
or (child minus parent) minus (child minus parent) of another model, on the
mean of some tasks. Each question's score is its mean over the runs. A
replicate resamples the questions of every task with replacement, using the
same draw for every checkpoint of the contrast, recomputes each task's score
and then the mean over the tasks. IFEval's score is the share of instructions
followed, so its prompts are resampled with their instruction counts and the
score is the followed instructions over all instructions of the draw.

The code tasks (HumanEval, LiveCodeBench, MBPP) have no per-question verdicts
in the records. A code task's part of a replicate is its observed change plus
a normal draw with the standard deviation

    sum over the contrast's checkpoints of |weight| * sqrt(p (1 - p) / n),

with p the checkpoint's score as a share and n the task's questions. That is
an upper bound on the standard deviation the question resampling would give:
for scores between 0 and 1 with mean p, the variance of one question's score
is at most p (1 - p), and the standard deviation of a sum is at most the sum
of the standard deviations, whatever the correlation between checkpoints. So a
code task widens the interval at least as much as its verdicts would, never
less. The decision was taken on 2026-10-24, for the analyses N1 to N3 of
docs/phase2_analyses.md.

Seed and replicate count are those of the Protocol A contrasts
(`stats/factorial.py`), so the intervals are reproducible. Each family gets one
Benjamini-Hochberg correction. Like Table 4's, the intervals cover the choice
of questions and the sampling of answers, not differences between training
runs, and not run 0's repeats of the May answers (docs/phase2_analyses.md).
"""
from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
import pandas as pd
from statsmodels.stats.multitest import multipletests

from ..manifest.reader import Manifest, ManifestError
from .factorial import BOOT_SEED, N_BOOT, _two_sided_p

#: The tasks without per-question verdicts, as the manifest names them.
CODE_TASKS = ("HumanEval", "LiveCodeBench", "MBPP")
IFEVAL = "IFEval"
#: The paper's domains (§2.4), as the manifest names the tasks.
DOMAINS = {
    "math": ("AIME24", "AIME25", "AMC23", "MATH500", "GSM8K"),
    "code": CODE_TASKS,
    "sci": ("GPQA-Diamond", "JEEBench"),
    "if": (IFEVAL,),
}
DOMAIN_NAMES = {"math": "Mathematics", "code": "Code", "sci": "Science and QA",
                "if": "Instruction following", "nonmath": "All but mathematics"}
ALL_RUNS = (0, 1, 2)
#: The column that names a question. JEEBench's `example_id` is a subject
#: label shared by many questions, so the answer rows use `example_idx`,
#: which maps one to one onto `example_id` for the other tasks. The IFEval
#: counts have `example_id` only, one per prompt.
QUESTION = "example_idx"
PROMPT = "example_id"


def domain_tasks(m: Manifest) -> dict[str, tuple[str, ...]]:
    """The four domains, and every task but mathematics as "nonmath"."""
    tasks = set(m.tasks())
    out = {d: tuple(t for t in ts if t in tasks) for d, ts in DOMAINS.items()}
    out["nonmath"] = tuple(t for t in m.tasks() if t not in DOMAINS["math"])
    return out


@dataclass(frozen=True)
class Contrast:
    """sum of weight * score over `weights`, each score the mean over `tasks`."""
    family: str
    label: str
    weights: tuple[tuple[str, float], ...]     # (checkpoint key, weight)
    tasks: tuple[str, ...]


@dataclass(frozen=True)
class Interval:
    contrast: Contrast
    estimate: float
    lo: float
    hi: float
    p: float
    q: float
    code_sd: float      # the bound's share of the replicates' spread, 0 without code tasks


@dataclass
class Scores:
    """Per-question scores of the Phase 2 checkpoints, checked against the
    manifest: {(key, task): Series by question} in percent, IFEval as
    {key: (followed, instructions) by prompt}, and the code tasks' scores.
    `exclude` names questions (QUESTION values) to leave out of every
    checkpoint, by task; the scores are then not the manifest's."""
    m: Manifest
    meta: pd.DataFrame
    ifeval: pd.DataFrame
    runs: tuple[int, ...] = ALL_RUNS
    exclude: dict = field(default_factory=dict)
    _items: dict = field(default_factory=dict)

    def _rows(self, key: str, task: str) -> pd.DataFrame:
        src = self.m.result(key, task)["source"]
        frame = self.ifeval if task == IFEVAL else self.meta
        rows = frame[(frame["base_model"] == src["base_model"]) & (frame["stage"] == src["stage"])
                     & frame["run_idx"].isin(self.runs)]
        if task != IFEVAL:
            rows = rows[rows["task"] == src["task"]]
            rows = rows[~rows[QUESTION].isin(self.exclude.get(task, ()))]
        if rows.empty:
            raise ManifestError(f"No {task} records for {key}.")
        if rows.duplicated([PROMPT if task == IFEVAL else QUESTION, "run_idx"]).any():
            raise ManifestError(f"{key} {task}: a question has two answers in one run.")
        return rows

    def _check(self, key: str, task: str, per_run: pd.Series) -> None:
        """The records' per-run scores must be the manifest's, in any order."""
        if self.runs != ALL_RUNS or self.exclude:
            return
        ours = sorted(per_run.round(6).tolist())
        theirs = sorted(self.m.result(key, task)["per_seed"].values())
        if len(ours) != len(theirs) or any(abs(a - b) > 1e-3 for a, b in zip(ours, theirs)):
            raise ManifestError(f"{key} {task}: the per-question records give runs {ours}, the "
                                f"manifest {theirs}.")

    def items(self, key: str, task: str):
        """Series of question scores in percent, or for IFEval a frame of
        `followed` and `instructions` per prompt."""
        if (key, task) in self._items:
            return self._items[(key, task)]
        rows = self._rows(key, task)
        q = PROMPT if task == IFEVAL else QUESTION
        counts = rows.groupby(q)["run_idx"].nunique()
        if (counts != len(self.runs)).any():
            raise ManifestError(f"{key} {task}: some questions lack {len(self.runs)} runs.")
        if task == IFEVAL:
            per_run = rows.groupby("run_idx").apply(
                lambda r: 100.0 * r["n_strict"].sum() / r["n_total"].sum(), include_groups=False)
            out = rows.groupby(q).agg(followed=("n_strict", "mean"),
                                                 instructions=("n_total", "first")).sort_index()
        else:
            per_run = rows.groupby("run_idx")["is_correct"].mean() * 100.0
            out = (rows.groupby(q)["is_correct"].mean() * 100.0).sort_index()
        self._check(key, task, per_run)
        self._items[(key, task)] = out
        return out

    def code(self, key: str, task: str) -> float:
        """A code task's score in percent: the manifest's mean, or the mean of
        the chosen runs' released scores."""
        r = self.m.result(key, task)
        if self.runs == ALL_RUNS:
            return r["mean"]
        return float(np.mean([r["per_seed"][str(k)] for k in self.runs]))

    def score(self, key: str, task: str) -> float:
        if task in CODE_TASKS:
            return self.code(key, task)
        x = self.items(key, task)
        if task == IFEVAL:
            return 100.0 * x["followed"].sum() / x["instructions"].sum()
        return float(x.mean())


def _matrix(scores: Scores, keys: list[str], task: str):
    """Questions x checkpoints, with the same questions for every checkpoint."""
    parts = [scores.items(k, task) for k in keys]
    index = parts[0].index
    if any(not p.index.equals(index) for p in parts[1:]):
        raise ManifestError(f"{task}: the checkpoints {keys} were scored on different questions.")
    if task == IFEVAL:
        if any(not (p["instructions"] == parts[0]["instructions"]).all() for p in parts[1:]):
            raise ManifestError("IFEval: the checkpoints count different instructions per prompt.")
        return (np.column_stack([p["followed"].to_numpy() for p in parts]),
                parts[0]["instructions"].to_numpy())
    return np.column_stack([p.to_numpy() for p in parts]), None


def evaluate(scores: Scores, contrasts: list[Contrast], *, n_boot: int = N_BOOT,
             seed: int = BOOT_SEED) -> list[Interval]:
    """Every contrast with its 95% interval and p-value, and q-values corrected
    within each family. Contrasts are drawn in the order given, one generator
    per family, so a family's intervals do not depend on the others."""
    out: list[Interval] = []
    families = list(dict.fromkeys(c.family for c in contrasts))
    for family in families:
        rng = np.random.default_rng(seed)
        rows = []
        for c in (c for c in contrasts if c.family == family):
            keys = [k for k, _ in c.weights]
            w = np.array([wt for _, wt in c.weights])
            draws = np.zeros(n_boot)
            estimate, code_var = 0.0, 0.0
            for task in c.tasks:
                if task in CODE_TASKS:
                    p = np.array([scores.code(k, task) for k in keys]) / 100.0
                    n = scores.m.task_info(task)["n_items"]
                    sd = 100.0 * float(np.sum(np.abs(w) * np.sqrt(p * (1 - p) / n)))
                    obs = 100.0 * float(w @ p)
                    draws += obs + sd * rng.standard_normal(n_boot)
                    estimate += obs
                    code_var += sd ** 2
                    continue
                x, inst = _matrix(scores, keys, task)
                idx = rng.integers(0, x.shape[0], size=(n_boot, x.shape[0]))
                if inst is None:
                    draws += x[idx].mean(axis=1) @ w
                    estimate += float(x.mean(axis=0) @ w)
                else:
                    draws += 100.0 * (x[idx].sum(axis=1) / inst[idx].sum(axis=1)[:, None]) @ w
                    estimate += float(100.0 * x.sum(axis=0) / inst.sum() @ w)
            k = len(c.tasks)
            draws /= k
            estimate /= k
            lo, hi = np.percentile(draws, [2.5, 97.5])
            share = float(np.sqrt(code_var) / k / draws.std()) if code_var and draws.std() else 0.0
            rows.append((c, estimate, float(lo), float(hi), _two_sided_p(draws), share))
        q = multipletests([r[4] for r in rows], method="fdr_bh")[1] if rows else []
        out += [Interval(c, est, lo, hi, p, float(qv), share)
                for (c, est, lo, hi, p, share), qv in zip(rows, q)]
    return out


def change(family: str, label: str, child: str, parent: str, tasks) -> Contrast:
    return Contrast(family, label, ((child, 1.0), (parent, -1.0)), tuple(tasks))


def gain_difference(family: str, label: str, a: tuple[str, str], b: tuple[str, str],
                    tasks) -> Contrast:
    """(a's child minus a's parent) minus (b's child minus b's parent)."""
    return Contrast(family, label, ((a[0], 1.0), (a[1], -1.0), (b[0], -1.0), (b[1], 1.0)),
                    tuple(tasks))
