"""Decoding variance on AIME24 / AIME25 from the temperature-0.7 replicates.

The manuscript's contest-math cells are greedy-decoded, so between-generation
variance is zero by construction and the released `run_stats` blocks show it:
ten repetitions, identical `num_solved`, `accuracy_std_err = 0.0`. That answers
"how many generations" but not "how much would the number move under a decoding
policy that *can* move" — which is the question behind the reviewer's concern.

The 2026-07-27 revision of `mv_aime24_amc23_best_responses.parquet` adds three
independent temperature-0.7 samples of AIME24 and AIME25 for six MixtureVitae
stages, tagged by a `decoding` column (null on the original greedy rows). This
module scores them with the repo's own math-verify scorer and reports the
dispersion across those three samples.

Two caveats the output carries, and any use of it must repeat:

  * These are temperature-0.7 samples. They are a *different decoding policy*
    from the manuscript's greedy cells and score well below them. They bound
    run-to-run movement; they do not restate the reported numbers.
  * n = 30. One problem is 3.33 pp, so the observed SD is pinned to the
    granularity floor and should be read as "at most one problem", not as a
    precise dispersion estimate.

Writes `artifacts/aime_temp07_variance.csv` (per-cell dispersion) and
`artifacts/aime_temp07_items.csv` (the scored item x run grid).
"""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

from ..data.benchmarks import SPECS
from ..data.scorers import CompareScorer

DECODING_TAG = "sample_temp0.7"

# Pipeline order, so the output table reads base -> SFT -> DPO then the
# OT30K endpoints rather than alphabetically.
STAGE_ORDER = ["base", "SFT-Tulu3", "DPO-Tulu3", "OT3", "OT3+SFT", "OT3+DPO"]

VARIANCE_CSV = "aime_temp07_variance.csv"
ITEMS_CSV = "aime_temp07_items.csv"


def _score(task: str, gold: str, pred: str) -> int | None:
    """Score one response with the same scorer the logs.zip path uses."""
    scorer = SPECS[task].scorer
    if not isinstance(scorer, CompareScorer):
        return None
    try:
        return scorer.compare(gold, pred)
    except Exception:
        # math-verify occasionally times out on a pathological expression;
        # such rows are dropped and counted, never silently scored 0.
        return None


def load_scored(mv_best: Path) -> pd.DataFrame:
    """Return the scored item x run grid for the temp-0.7 replicates."""
    df = pd.read_parquet(mv_best)
    if "decoding" not in df.columns:
        raise ValueError(
            f"{mv_best.name} has no `decoding` column — it predates the "
            "2026-07-27 revision that added the temperature-0.7 replicates"
        )
    df = df[df.decoding == DECODING_TAG].copy()
    if df.empty:
        raise ValueError(f"no rows tagged decoding={DECODING_TAG!r} in {mv_best.name}")

    df["score"] = [
        _score(t, g, p)
        for t, g, p in zip(df.task, df.gold_answer, df.model_answer)
    ]
    n_unscored = int(df.score.isna().sum())
    if n_unscored:
        print(f"note: {n_unscored} of {len(df)} rows could not be scored, dropped")
    df = df.dropna(subset=["score"])
    df["score"] = df.score.astype(int)
    return df


def summarize(df: pd.DataFrame) -> pd.DataFrame:
    """Per-(stage, task) dispersion across the independent samples."""
    per_run = (
        df.groupby(["stage", "task", "run_idx"]).score.mean().mul(100)
        .reset_index(name="acc")
    )

    rows = []
    for (stage, task), g in per_run.groupby(["stage", "task"]):
        accs = g.sort_values("run_idx").acc.to_numpy()
        cell = df[(df.stage == stage) & (df.task == task)]
        n_items = int(cell.example_idx.nunique())
        grid = cell.pivot_table(index="example_idx", columns="run_idx", values="score").dropna()
        n_solved = grid.sum(axis=1)
        rows.append({
            "stage": stage,
            "task": task,
            "n_runs": len(accs),
            "n_items": n_items,
            "run_accs": " / ".join(f"{a:.1f}" for a in accs),
            "mean": round(float(np.mean(accs)), 2),
            "sd": round(float(np.std(accs, ddof=1)), 2),
            "min": round(float(accs.min()), 2),
            "max": round(float(accs.max()), 2),
            "range": round(float(accs.max() - accs.min()), 2),
            # One problem is 100/n_items pp; the SD cannot resolve below it.
            "one_item_pp": round(100.0 / n_items, 2),
            "items_all_runs_agree": int((grid.nunique(axis=1) == 1).sum()),
            "items_solved_always": int((n_solved == len(accs)).sum()),
            "items_solved_sometimes": int(((n_solved > 0) & (n_solved < len(accs))).sum()),
            "items_solved_never": int((n_solved == 0).sum()),
        })

    out = pd.DataFrame(rows)
    order = {s: i for i, s in enumerate(STAGE_ORDER)}
    out["_o"] = out.stage.map(order).fillna(len(order))
    return out.sort_values(["task", "_o"]).drop(columns="_o").reset_index(drop=True)


def main(mv_best: Path, art_dir: Path) -> int:
    mv_best, art_dir = Path(mv_best), Path(art_dir)
    if not mv_best.exists():
        print(f"missing source: {mv_best}")
        return 1
    art_dir.mkdir(parents=True, exist_ok=True)

    scored = load_scored(mv_best)
    summary = summarize(scored)

    scored[["stage", "task", "example_idx", "run_idx", "score"]].to_csv(
        art_dir / ITEMS_CSV, index=False
    )
    summary.to_csv(art_dir / VARIANCE_CSV, index=False)

    print(summary.to_string(index=False))
    print(f"\nwrote {art_dir / VARIANCE_CSV}")
    print(f"wrote {art_dir / ITEMS_CSV}")
    return 0
