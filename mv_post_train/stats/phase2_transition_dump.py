"""Phase 1 -> OT30K transitions with **both ends scored from the HF dump**.

Why this exists alongside `phase2_transition.py`
------------------------------------------------
`phase2_transition.py` pairs a Phase 1 start against a Phase 2 endpoint drawn
from whatever source carries that task at item level. For gsm8k (via
`phase1_dump_items.parquet`) and for IFEval / MBPP (via
`recovered_items.parquet`) both ends come from the dump, so the pairing is
single-source. For the four contest-math tasks it is **not**: the Phase 1 side
comes from logs.zip (`items_long.parquet`) while the Phase 2 side comes from
the dump. Any difference in generation run or answer extraction between the two
then lands entirely in the delta.

It is not hypothetical. logs.zip scores MV/base/16k MATH500 at 48.6, the dump
scores the same cell at 34.8, and the submitted table says 33.6 -- so the
cross-source `base -> OT30K` delta was understated by ~14 pp against a start
that matches neither the dump nor the paper.

This module recomputes the four math transitions with both ends read from
`eval_responses.parquet` and scored by the same `math-verify` scorer, which is
the only configuration in which the delta means what it says.

Pooled checkpoints
------------------
Exactly two of the dump's 794 (base_model, stage, task) cells hold generations
from two different checkpoints, and both are MV `OT3+SFT`: AIME25 (30 rows from
the intended OT30K-on-SFT checkpoint, 270 from a differently named January run)
and AMC23 (40 vs 360). Averaging across them reports a model that was never
trained, so foreign rows are dropped and the drop is logged.

Which checkpoint is canonical is decided at the (base_model, stage) level, by
how many distinct *tasks* a checkpoint covers -- the same rule
`artifacts/audit/checkpoint_identity.csv` uses. Counting rows within the cell
instead gets it exactly backwards: the intruder carries ten greedy repetitions
per item to the canonical run's one, so it wins any row-count majority 270-30
while covering two tasks to the canonical run's four.
See `docs/contest-math-dump-inconsistencies.md`.

Degenerate generations
----------------------
`math-verify` can spend ~30 s parsing a `model_answer` that has degenerated
into thousands of repeated tokens. 99 of the 22,890 rows in scope are like
that, which is the difference between a 2-minute run and an hour. Answers
longer than `MAX_ANSWER_CHARS` are scored 0 without being parsed; every one
inspected is runaway repetition, and none is a correct answer.

Writes `phase2_transition_dump_paired.csv` (the paired tests) and
`dump_cell_levels.csv` (per-cell accuracy, for cross-checking against the
released `eval_scores_wide.csv`).
"""
from __future__ import annotations

import collections
from pathlib import Path

import numpy as np
import pandas as pd
import pyarrow.parquet as pq

from ..data.benchmarks import AIME24, AIME25, AMC23, MATH500, SPECS
from ..data.scorers import CompareScorer
from .core import bh_adjust, paired_bootstrap_ci, paired_perm_test, paired_wilcoxon

TRANSITION_CSV = "phase2_transition_dump_paired.csv"
LEVELS_CSV = "dump_cell_levels.csv"

# The four tasks whose Phase 1 side would otherwise come from logs.zip. gsm8k
# is deliberately absent: it is already single-source via phase1_dump_items.
SCORED_TASKS = [MATH500, AIME24, AIME25, AMC23]

# (Phase 1 stage, Phase 2 stage) in the dump's own stage vocabulary.
PAIRS = [
    ("base", "OT3"),
    ("SFT-Tulu3", "OT3+SFT"),
    ("DPO-Tulu3", "OT3+DPO"),
]
STAGES = {s for pair in PAIRS for s in pair}

# See "Degenerate generations" above. The longest correct answer across every
# cell in scope is far below this.
MAX_ANSWER_CHARS = 400

# Small enough that a batch of `model_answer` (full generations, in a single
# row group) fits alongside everything else on a modest machine.
BATCH_ROWS = 5000

_READ_COLUMNS = [
    "base_model", "stage", "task", "example_idx",
    "gold_answer", "model_answer", "source_dir",
]


def canonical_checkpoints(counts: dict) -> dict:
    """(base_model, stage, task) -> the checkpoint to keep.

    `counts` maps each cell to a Counter of source_dir -> row count. The
    canonical checkpoint for a (base_model, stage) group is the one covering
    the most distinct tasks; ties fall back to total rows. A cell that does not
    carry the group's checkpoint at all keeps its own -- dropping it outright
    would silently delete a benchmark rather than report it.
    """
    by_group: dict[tuple, collections.Counter] = collections.defaultdict(collections.Counter)
    rows_by_group: dict[tuple, collections.Counter] = collections.defaultdict(collections.Counter)
    for (model, stage, _task), srcs in counts.items():
        for src, n in srcs.items():
            by_group[(model, stage)][src] += 1      # tasks covered
            rows_by_group[(model, stage)][src] += n  # tie-break
    group_pick = {
        g: max(c, key=lambda s: (c[s], rows_by_group[g][s]))
        for g, c in by_group.items()
    }
    keep = {}
    for cell, srcs in counts.items():
        pick = group_pick[(cell[0], cell[1])]
        keep[cell] = pick if pick in srcs else srcs.most_common(1)[0][0]
    return keep


def _checkpoints(parquet: Path) -> tuple[dict, list[tuple]]:
    """Canonical `source_dir` per (base_model, stage, task), plus the cells
    that hold more than one. Reads only small columns, so it is cheap."""
    counts: dict[tuple, collections.Counter] = collections.defaultdict(collections.Counter)
    pf = pq.ParquetFile(parquet)
    for batch in pf.iter_batches(batch_size=BATCH_ROWS,
                                 columns=["base_model", "stage", "task", "source_dir"]):
        d = batch.to_pydict()
        for model, stage, task, src in zip(d["base_model"], d["stage"],
                                           d["task"], d["source_dir"]):
            if task in SCORED_TASKS and stage in STAGES:
                counts[(model, stage, task)][str(src)] += 1
    pooled = [(cell, dict(c)) for cell, c in counts.items() if len(c) > 1]
    return canonical_checkpoints(counts), pooled


def _score_cells(parquet: Path, keep: dict) -> dict:
    """(model, stage, task) -> {example_idx: [scores]}, canonical checkpoint only."""
    cache: dict[tuple, int | None] = {}

    def score(task: str, gold, pred) -> int | None:
        if pred is None or len(pred) > MAX_ANSWER_CHARS:
            return 0
        key = (task, gold, pred)
        if key not in cache:
            scorer = SPECS[task].scorer
            if not isinstance(scorer, CompareScorer):
                cache[key] = None
            else:
                try:
                    cache[key] = scorer.compare(gold, pred)
                except Exception:
                    cache[key] = None
        return cache[key]

    recs: dict = collections.defaultdict(lambda: collections.defaultdict(list))
    pf = pq.ParquetFile(parquet)
    for batch in pf.iter_batches(batch_size=BATCH_ROWS, columns=_READ_COLUMNS):
        d = batch.to_pydict()
        for i in range(len(d["task"])):
            task, stage = d["task"][i], d["stage"][i]
            if task not in SCORED_TASKS or stage not in STAGES:
                continue
            cell = (d["base_model"][i], stage, task)
            if str(d["source_dir"][i]) != keep.get(cell):
                continue  # pooled foreign checkpoint
            s = score(task, d["gold_answer"][i], d["model_answer"][i])
            if s is not None:
                recs[cell][d["example_idx"][i]].append(s)
        del d, batch
    return recs


def transitions(recs: dict, n_resamples: int = 10000) -> pd.DataFrame:
    """Paired bootstrap / permutation tests over the scored cells.

    The paired unit is the problem, not the generation: cells carrying ten
    greedy repetitions are collapsed to a per-item mean first, so they are
    comparable with cells carrying one.
    """
    rows = []
    for model in sorted({cell[0] for cell in recs}):
        for start, end in PAIRS:
            for task in SCORED_TASKS:
                a = recs.get((model, end, task))
                b = recs.get((model, start, task))
                if not a or not b:
                    continue
                shared = sorted(set(a) & set(b))
                if len(shared) < 2:
                    continue
                y_a = np.array([np.mean(a[i]) for i in shared], dtype=float)
                y_b = np.array([np.mean(b[i]) for i in shared], dtype=float)
                common = {
                    "model": model, "transition": f"{start}->{end}", "task": task,
                    "n_items": len(shared),
                    "acc_start": round(float(y_b.mean() * 100), 2),
                    "acc_end": round(float(y_a.mean() * 100), 2),
                    "delta": round(float((y_a - y_b).mean() * 100), 2),
                }
                # Both ends identical on every item: nothing to test. Kept
                # rather than dropped -- "every reference scores 0 on every
                # contest-math item at both ends" is the actual finding.
                if np.allclose(y_a, y_b):
                    rows.append({**common, "ci_lo": 0.0, "ci_hi": 0.0,
                                 "p_perm": 1.0, "p_wilcoxon": 1.0,
                                 "degenerate": True})
                    continue
                ci = paired_bootstrap_ci(y_a, y_b, n_resamples=n_resamples)
                _, p = paired_perm_test(y_a, y_b, n_resamples=n_resamples)
                rows.append({**common,
                             "ci_lo": round(float(ci.lo) * 100, 2),
                             "ci_hi": round(float(ci.hi) * 100, 2),
                             "p_perm": float(p),
                             "p_wilcoxon": float(paired_wilcoxon(y_a, y_b)),
                             "degenerate": False})

    out = pd.DataFrame(rows)
    if out.empty:
        return out
    live = out.index[~out.degenerate]
    out["q_value"] = 1.0
    out.loc[live, "q_value"] = bh_adjust(out.loc[live, "p_perm"].tolist())
    for col in ("q_value", "p_perm", "p_wilcoxon"):
        out[col] = out[col].round(4)
    return out


def cell_levels(recs: dict) -> pd.DataFrame:
    """Per-cell accuracy, for cross-checking against `eval_scores_wide.csv`."""
    rows = [{"model": model, "stage": stage, "task": task,
             "acc": round(float(np.mean([np.mean(v) for v in items.values()]) * 100), 2),
             "n_items": len(items)}
            for (model, stage, task), items in recs.items()]
    return pd.DataFrame(rows).sort_values(["model", "task", "stage"])


def main(responses: Path, art_dir: Path) -> int:
    responses, art_dir = Path(responses), Path(art_dir)
    if not responses.exists():
        print(f"missing {responses}")
        return 1
    art_dir.mkdir(parents=True, exist_ok=True)

    keep, pooled = _checkpoints(responses)
    if pooled:
        print(f"{len(pooled)} pooled cell(s); keeping the canonical checkpoint only:")
        for (model, stage, task), counts in sorted(pooled):
            print(f"    {model} / {stage} / {task}:")
            for src, n in sorted(counts.items(), key=lambda kv: -kv[1]):
                mark = "keep" if src == keep[(model, stage, task)] else "DROP"
                print(f"        [{mark}] {n:>5} rows  {src.split('__checkpoints__')[-1]}")

    recs = _score_cells(responses, keep)
    out = transitions(recs)
    if out.empty:
        print("no paired cells found")
        return 1

    out_csv = art_dir / TRANSITION_CSV
    out.to_csv(out_csv, index=False)
    levels_csv = art_dir / LEVELS_CSV
    cell_levels(recs).to_csv(levels_csv, index=False)

    print(out.to_string(index=False))
    print(f"\n{len(out)} cells: {int((~out.degenerate).sum())} tested, "
          f"{int(out.degenerate.sum())} degenerate (both ends identical)")
    print(f"wrote {out_csv}")
    print(f"wrote {levels_csv}")
    return 0
