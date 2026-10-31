"""Significance of the MV contest-math advantage under sampling.

The reviewer's comment singles out the contest-math benchmarks because they are
scored from few generations. The three-seed temperature-0.7 replicate runs ship
per-sample scores for AIME24, AIME25 and AMC23, so for the first time these
comparisons can be tested at item level rather than eyeballed from a table.

Each item is averaged over its three seeds before testing, so a problem
contributes one paired observation instead of three correlated ones.

Two intervals are reported for every comparison, deliberately:

  * **paired** bootstrap CI plus a sign-flip permutation test -- the sharper
    test, but it assumes the item identities are real.
  * **unpaired** Newcombe interval on the marginal totals -- weaker, and valid
    regardless of item identity.

The reason for the redundancy is that the release marks these three benchmarks'
per-sample rows `reconstructed` rather than `stored`. We checked that the
correct items are scattered rather than a prefix block, so they are not
obviously synthesized, but we have no statement of what the reconstruction did.
Reporting both means the conclusion does not rest on the answer: where the two
agree, the question is moot.

Multiplicity is likewise reported twice, because the choice of family changes
the headline and neither choice is obviously right:

  * **within benchmark** (12 tests) -- defensible on the grounds that the AIME
    comparisons are structurally incapable of detecting anything at these
    accuracy levels (see `mde_paired_binomial`: 27.8 pp at n=30), so folding
    them in only inflates the denominator.
  * **across all contest-math tests** (36) -- the conservative reading.

Writes `artifacts/contest_math_sampled_sig.csv`.
"""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

from . import seed_variance
from .core import (bh_adjust, newcombe_diff_ci, paired_bootstrap_ci,
                   paired_perm_test)

OUT_CSV = "contest_math_sampled_sig.csv"
PER_SAMPLE = "per_sample_scores.csv"

TASKS = ("AIME24", "AIME25", "AMC23")
REFERENCES = ("Comma", "FineWeb-Edu", "Nemotron", "SmolLM2")
FOCAL = "MV"
STAGES = ("base", "sft", "dpo")
N_SEEDS = 3


def load_items(data_dir: Path, tier: str = "16k") -> pd.Series:
    """(family, stage, benchmark, sample_index) -> accuracy in [0, 1].

    Averaged over seeds, so one value per problem per model.
    """
    p = pd.read_csv(Path(data_dir) / PER_SAMPLE)
    p = p[p.benchmark.isin(TASKS)]
    if p.empty:
        raise ValueError(f"no contest-math rows in {data_dir / PER_SAMPLE}")
    meta = seed_variance.classify(sorted(p.huggingface_model_id.unique()))
    p = p.merge(meta, on="huggingface_model_id")
    p = p[p.tier == tier]
    return (p.groupby(["family", "stage", "benchmark", "sample_index"])
              .sample_score_percent.mean() / 100.0)


def test_all(cell: pd.Series, n_resamples: int = 10000) -> pd.DataFrame:
    rows = []
    for task in TASKS:
        for stage in STAGES:
            try:
                mv = cell.loc[(FOCAL, stage, task)]
            except KeyError:
                continue
            for ref in REFERENCES:
                try:
                    rv = cell.loc[(ref, stage, task)]
                except KeyError:
                    continue
                idx = mv.index.intersection(rv.index)
                if len(idx) == 0:
                    continue
                a = mv.loc[idx].to_numpy()
                b = rv.loc[idx].to_numpy()
                ci = paired_bootstrap_ci(a, b, n_resamples=n_resamples)
                _, pval = paired_perm_test(a, b, n_resamples=n_resamples)
                # Marginal successes over items x seeds for the unpaired bound.
                n = len(idx) * N_SEEDS
                nc = newcombe_diff_ci(int(round(a.sum() * N_SEEDS)), n,
                                      int(round(b.sum() * N_SEEDS)), n)
                rows.append({
                    "task": task, "stage": stage, "reference": ref,
                    "n_items": len(idx),
                    "mv": round(a.mean() * 100, 2), "ref": round(b.mean() * 100, 2),
                    "diff": round((a.mean() - b.mean()) * 100, 2),
                    "ci_lo": round(ci.lo * 100, 2), "ci_hi": round(ci.hi * 100, 2),
                    "p": pval,
                    "nc_lo": round(nc.lo * 100, 2), "nc_hi": round(nc.hi * 100, 2),
                })
    df = pd.DataFrame(rows)
    if df.empty:
        return df
    df["q_global"] = bh_adjust(df.p.tolist())
    df["q_within_task"] = np.nan
    for task, g in df.groupby("task"):
        df.loc[g.index, "q_within_task"] = bh_adjust(g.p.tolist())
    df["sig_within_task"] = df.q_within_task < 0.05
    df["sig_global"] = df.q_global < 0.05
    # If the paired and unpaired lower bounds disagree about excluding zero,
    # the conclusion depends on item identity being real -- and item identity
    # is exactly what `reconstructed` leaves unspecified.
    df["pairing_matters"] = (df.ci_lo > 0) != (df.nc_lo > 0)
    return df


def main(data_dir: Path, art_dir: Path) -> int:
    data_dir, art_dir = Path(data_dir), Path(art_dir)
    if not (data_dir / PER_SAMPLE).exists():
        print(f"missing {data_dir / PER_SAMPLE}")
        return 1
    art_dir.mkdir(parents=True, exist_ok=True)

    df = test_all(load_items(data_dir))
    if df.empty:
        print("no comparisons available")
        return 1
    df.to_csv(art_dir / OUT_CSV, index=False)

    for task in TASKS:
        t = df[df.task == task]
        if t.empty:
            continue
        print(f"\n=== {task} (n={t.n_items.iloc[0]} items x {N_SEEDS} seeds) ===")
        print(t[["stage", "reference", "mv", "ref", "diff", "ci_lo", "ci_hi",
                 "p", "q_within_task", "sig_within_task"]].to_string(index=False))
        print(f"  significant within-benchmark: {int(t.sig_within_task.sum())}/{len(t)}")

    print(f"\nsignificant, BH within benchmark: {int(df.sig_within_task.sum())}/{len(df)}")
    print(f"significant, BH across all {len(df)} tests: {int(df.sig_global.sum())}/{len(df)}")
    n_dep = int(df.pairing_matters.sum())
    print(f"comparisons whose verdict depends on item pairing: {n_dep}/{len(df)}"
          + ("  (paired and unpaired agree throughout)" if n_dep == 0 else ""))
    print(f"\nwrote {art_dir / OUT_CSV}")
    return 0
