"""Seed-to-seed variance from the three-seed Phase 1 replicate runs.

This is the one dataset in the project that answers the reviewer's question
directly rather than by proxy. Every other uncertainty estimate here is either
item-level (a bootstrap over a single set of generations) or a two-run
comparison; this is three independent seeds across the whole Phase 1 grid --
39 models x 11 benchmarks x 3 seeds, complete, no missing cells.

Source: `ali-elganzory/mv-posttraining-scores`, which publishes **two** such
grids. Pass whichever you mean via `--seed-dir`; they are not interchangeable
and the outputs must not be pooled:

  * `phase1/paper-decoding-3-seeds` -- the **submitted** decoding configuration.
    Levels reproduce the paper (MV 16k suite average 28.12 / 24.88 / 25.37
    against a submitted 27.9 / 24.0 / 25.0), so both the spread *and* the level
    attach to the published figures. Conventionally written to
    `artifacts/greedy_seeds/`.
  * `phase1/sampling-0.7` -- stochastic decoding, a robustness check.

Two things to keep straight when reading the output:

  * The T=0.7 grid is **sampled**; the paper's tables are **greedy**. Its
    numbers are systematically lower wherever a model is genuinely capable (MV
    loses ~11-13 pp on AMC23 and MATH500) while barely moving for models pinned
    at the floor. What transfers to the paper is the *spread*, not the level.
  * Greedy is not seed-invariant on the small benchmarks. Under
    `paper-decoding-3-seeds` all 39 cells agree on gsm8k but only 23 of 39 on
    AMC23, worst spread 15.00 pp. "Greedy therefore deterministic" holds for
    repetitions inside one evaluation process, not across independent runs.
  * The suite average is the mean over the 11 benchmarks, matching the paper's
    aggregation. Per-benchmark aggregation also matches: HumanEval is the mean
    of `python_pass@1` and `sh_pass@1`, IFEval the mean of prompt- and
    instruction-level loose. Verified to rounding against `metric_scores.csv`.

    That is the **submission** metric convention, and it is fixed here: these
    grids are read as published, so this module cannot re-derive HumanEval or
    IFEval under a different definition. If the parsers are switched to the
    arXiv convention (`data/conventions.py`), the numbers this module reports
    stop being comparable with `aggregates.parquet`, and `main` refuses to run
    rather than writing two conventions into one `artifacts/` directory.

Writes `artifacts/seed_variance_per_task.csv`, `seed_variance_suite.csv` and
`seed_variance_gaps.csv`.
"""
from __future__ import annotations

import re
from pathlib import Path

import numpy as np
import pandas as pd

from ..data import conventions

#: The convention the published replicate grids were scored under. Not a
#: choice this module makes -- it is a property of the input CSVs.
GRID_CONVENTION = conventions.MetricConvention.SUBMISSION

PER_TASK_CSV = "seed_variance_per_task.csv"
SUITE_CSV = "seed_variance_suite.csv"
GAPS_CSV = "seed_variance_gaps.csv"
LEAVE_OUT_CSV = "seed_variance_leave_out.csv"
LEAVE_OUT_SUMMARY_CSV = "seed_variance_leave_out_summary.csv"

FINAL_SCORES = "final_scores.csv"
METRIC_SCORES = "metric_scores.csv"

SEEDS = (42, 1234, 2025)

# The 11-benchmark reasoning suite, in the paper's order.
TASKS = ("AIME24", "AIME25", "AMC23", "MATH500", "GPQADiamond", "JEEBench",
         "LiveCodeBench", "HumanEval", "MBPP", "IFEval", "gsm8k")

# GPQA-Diamond is 4-choice, so 25% is chance. A cell well below that is not a
# weak model, it is a model whose answers are not being parsed -- worth
# surfacing rather than averaging in silently.
GPQA_CHANCE = 25.0


def _family(name: str) -> str | None:
    if "MixtureVitae" in name:
        return "MV"
    if "Comma0.1" in name:
        return "Comma"
    if "fineweb-edu" in name:
        return "FineWeb-Edu"
    if "nemotron-hq" in name:
        return "Nemotron"
    if "dclm" in name:
        return "DCLM"
    if "SmolLM2" in name:
        return "SmolLM2"
    if "Qwen2.5" in name:
        return "Qwen2.5"
    if "Qwen3" in name:
        return "Qwen3"
    return None


def _stage(name: str) -> str:
    # Order matters: the DPO checkpoints also carry the SFT suffix upstream of
    # it in some names, so test DPO first.
    if re.search(r"-DPO-Tulu3", name):
        return "dpo"
    if re.search(r"-SFT-Tulu3", name):
        return "sft"
    return "base"


def _tier(name: str) -> str:
    """16K vs the short-context tier.

    The 16K long-SFT checkpoints are spelled three different ways upstream
    (`longsft_16k`, `long_sft_16k`, and a plain `-16k`), which is why this is a
    regex over several spellings rather than a suffix test.
    """
    if re.search(r"long_?sft_16k|-16k(-|$)|1\.7B-16k", name):
        return "16k"
    return "4k"


def classify(model_ids: list[str]) -> pd.DataFrame:
    rows = []
    for mid in model_ids:
        short = mid.split("/")[-1]
        fam = _family(short)
        rows.append({"huggingface_model_id": mid, "model": short,
                     "family": fam, "stage": _stage(short), "tier": _tier(short)})
    df = pd.DataFrame(rows)
    unknown = df[df.family.isna()]
    if not unknown.empty:
        raise ValueError(f"unclassified models: {unknown.model.tolist()}")
    return df


def load(data_dir: Path) -> pd.DataFrame:
    f = pd.read_csv(data_dir / FINAL_SCORES)
    meta = classify(sorted(f.huggingface_model_id.unique()))
    return f.merge(meta, on="huggingface_model_id")


def per_task(df: pd.DataFrame) -> pd.DataFrame:
    """Mean / SD / range across seeds for every (model, benchmark) cell."""
    g = df.groupby(["family", "stage", "tier", "model", "benchmark"])
    out = g.final_score_percent.agg(
        mean="mean", sd=lambda s: s.std(ddof=1), lo="min", hi="max",
        n_seeds="size").reset_index()
    out["range"] = out.hi - out.lo
    return out.round(3)


def suite(df: pd.DataFrame) -> pd.DataFrame:
    """Suite average per seed, then mean and SD of that average.

    The SD here is the quantity Figure 2's bars were missing: how much a whole
    bar moves when only the sampling seed changes.
    """
    per_seed = (df.groupby(["family", "stage", "tier", "seed"])
                  .final_score_percent.agg(suite_avg="mean", n_tasks="size")
                  .reset_index())
    bad = per_seed[per_seed.n_tasks != len(TASKS)]
    if not bad.empty:
        raise ValueError(f"incomplete suite for {len(bad)} (family, stage, tier, seed) "
                         f"groups: {bad.head().to_dict('records')}")
    g = per_seed.groupby(["family", "stage", "tier"])
    out = g.suite_avg.agg(mean="mean", sd=lambda s: s.std(ddof=1),
                          lo="min", hi="max").reset_index()
    wide = per_seed.pivot_table(index=["family", "stage", "tier"],
                                columns="seed", values="suite_avg")
    wide.columns = [f"seed_{c}" for c in wide.columns]
    return out.merge(wide.reset_index(), on=["family", "stage", "tier"]).round(3)


def gaps(df: pd.DataFrame, tier: str = "16k", focal: str = "MV") -> pd.DataFrame:
    """MV-minus-reference suite-average gap, computed within each seed.

    Differencing inside a seed and only then aggregating is the point: the two
    models saw the same sampling seed, so any shared run-level effect cancels
    instead of being counted twice.
    """
    per_seed = (df[df.tier == tier].groupby(["family", "stage", "seed"])
                  .final_score_percent.mean().reset_index(name="suite_avg"))
    rows = []
    for stage in ("base", "sft", "dpo"):
        s = per_seed[per_seed.stage == stage]
        mv = s[s.family == focal].set_index("seed").suite_avg
        if mv.empty:
            continue
        for fam in sorted(set(s.family) - {focal}):
            ref = s[s.family == fam].set_index("seed").suite_avg
            common = sorted(set(mv.index) & set(ref.index))
            if len(common) < 2:
                continue
            d = np.array([mv[k] - ref[k] for k in common], dtype=float)
            rows.append({
                "stage": stage, "reference": fam, "tier": tier,
                "n_seeds": len(d), "gap_mean": d.mean(), "gap_sd": d.std(ddof=1),
                "gap_lo": d.min(), "gap_hi": d.max(),
                # How many multiples of the seed SD the gap sits at. Reported
                # instead of a p-value: with 3 seeds a t-test has 2 degrees of
                # freedom and is not worth quoting.
                "gap_over_sd": (d.mean() / d.std(ddof=1)
                                if d.std(ddof=1) > 0 else np.inf),
            })
    return pd.DataFrame(rows).round(3)


DROP_SETS = {
    "full suite (11)": (),
    "drop AIME24+25": ("AIME24", "AIME25"),
    "drop all contest math": ("AIME24", "AIME25", "AMC23"),
    "drop every floor-bound task": ("AIME24", "AIME25", "AMC23",
                                    "LiveCodeBench"),
}


def leave_out(df: pd.DataFrame, tier: str = "16k",
              focal: str = "MV") -> pd.DataFrame:
    """The MV-minus-reference gap with subsets of benchmarks removed.

    Answers the specific objection that the separation is carried by AIME24/25
    -- small sets where the references sit at the floor, so MV's few solved
    items look decisive. If the gap survives deleting them outright, the
    separation is not an artifact of those benchmarks; if it collapses, the
    objection is correct and the headline needs rewording.

    Also runs a jackknife dropping each benchmark singly, which catches the
    weaker version of the same problem: no *one* task should be able to move
    the gap much on its own.
    """
    rows = []
    for label, dropped in DROP_SETS.items():
        kept = df[~df.benchmark.isin(dropped)]
        g = gaps(kept, tier=tier, focal=focal)
        for _, r in g.iterrows():
            rows.append({"subset": label, "n_tasks": len(TASKS) - len(dropped),
                         **r.to_dict()})
    for task in TASKS:
        kept = df[df.benchmark != task]
        g = gaps(kept, tier=tier, focal=focal)
        for _, r in g.iterrows():
            rows.append({"subset": f"drop {task}", "n_tasks": len(TASKS) - 1,
                         **r.to_dict()})
    return pd.DataFrame(rows).round(3)


def leave_out_summary(lo: pd.DataFrame) -> pd.DataFrame:
    """Worst case over references for each subset -- the number that matters.

    A separation that survives on average but collapses against the strongest
    reference has not survived.
    """
    rows = []
    for (subset, stage), g in lo.groupby(["subset", "stage"], sort=False):
        rows.append({
            "subset": subset, "stage": stage,
            "n_tasks": int(g.n_tasks.iloc[0]),
            "min_gap": g.gap_mean.min(),
            "max_gap": g.gap_mean.max(),
            "worst_reference": g.loc[g.gap_mean.idxmin(), "reference"],
            "max_seed_sd": g.gap_sd.max(),
            "min_gap_over_sd": g.gap_over_sd.min(),
            "all_positive": bool((g.gap_lo > 0).all()),
        })
    return pd.DataFrame(rows).round(3)


def audit(df: pd.DataFrame) -> list[str]:
    """Data-quality flags worth reporting back to whoever collected the runs."""
    notes = []

    counts = df.groupby(["huggingface_model_id", "benchmark"]).seed.nunique()
    if (counts != len(SEEDS)).any():
        n = int((counts != len(SEEDS)).sum())
        notes.append(f"{n} (model, benchmark) cells do not have all {len(SEEDS)} seeds")

    g = df[df.benchmark == "GPQADiamond"]
    sub = g[g.final_score_percent < GPQA_CHANCE * 0.72]
    for _, r in sub.sort_values("final_score_percent").iterrows():
        notes.append(f"GPQA-Diamond below chance: {r.model} seed={r.seed} "
                     f"= {r.final_score_percent:.2f}% (4-choice chance is {GPQA_CHANCE:.0f}%)")

    # A seed that is systematically high or low would mean the three runs are
    # not exchangeable and cannot be pooled as replicates.
    try:
        from scipy import stats
        for b, gg in df.groupby("benchmark"):
            w = gg.pivot_table(index="huggingface_model_id", columns="seed",
                               values="final_score_percent").dropna()
            if w.shape[0] < 5 or w.shape[1] < 3:
                continue
            _, p = stats.friedmanchisquare(*[w[c] for c in w.columns])
            if p < 0.05:
                notes.append(f"systematic seed effect on {b}: Friedman p={p:.4f}")
    except ImportError:
        notes.append("scipy unavailable; skipped the seed-exchangeability test")

    return notes


def main(data_dir: Path, art_dir: Path) -> int:
    data_dir, art_dir = Path(data_dir), Path(art_dir)
    src = data_dir / FINAL_SCORES
    if not src.exists():
        print(f"missing {src}")
        return 1

    # The replicate grids are published under the submission convention and are
    # consumed as-is. Emitting them next to artifacts built under a different
    # convention would put two definitions of HumanEval and IFEval in one
    # directory under the same column names, so stop instead.
    active = conventions.active_convention()
    if active is not GRID_CONVENTION:
        print(
            f"refusing to run: the replicate grids are scored under the "
            f"'{GRID_CONVENTION.value}' metric convention, but this code emits "
            f"'{active.value}'. HumanEval and IFEval are not comparable across "
            f"the two (see docs/statistics.md); re-scored grids would be needed "
            f"to report seed variance under '{active.value}'."
        )
        return 1

    art_dir.mkdir(parents=True, exist_ok=True)

    df = load(data_dir)
    print(f"{df.huggingface_model_id.nunique()} models x "
          f"{df.benchmark.nunique()} benchmarks x {df.seed.nunique()} seeds "
          f"= {len(df)} rows")

    pt = per_task(df)
    pt.to_csv(art_dir / PER_TASK_CSV, index=False)
    print("\n=== seed SD by benchmark (across all models) ===")
    by_task = (pt.groupby("benchmark")
                 .agg(mean_sd=("sd", "mean"), median_sd=("sd", "median"),
                      max_sd=("sd", "max"), n_cells=("sd", "size"))
                 .sort_values("mean_sd", ascending=False).round(2))
    print(by_task.to_string())

    su = suite(df)
    su.to_csv(art_dir / SUITE_CSV, index=False)
    print("\n=== suite average +- seed SD, 16K tier ===")
    s16 = su[su.tier == "16k"].sort_values(["family", "stage"])
    print(s16[["family", "stage", "mean", "sd", "lo", "hi"]].to_string(index=False))

    gp = gaps(df)
    gp.to_csv(art_dir / GAPS_CSV, index=False)
    if not gp.empty:
        print("\n=== MV - reference suite gap, differenced within seed (16K) ===")
        print(gp.to_string(index=False))

    lo = leave_out(df)
    lo.to_csv(art_dir / LEAVE_OUT_CSV, index=False)
    if not lo.empty:
        summ = leave_out_summary(lo)
        summ.to_csv(art_dir / LEAVE_OUT_SUMMARY_CSV, index=False)
        print("\n=== does the separation depend on AIME? "
              "worst-case gap over references (16K) ===")
        named = summ[summ.subset.isin(DROP_SETS)]
        print(named.to_string(index=False))
        print("\n=== jackknife: drop each benchmark singly, worst case ===")
        jk = summ[~summ.subset.isin(DROP_SETS)]
        print(jk.sort_values("min_gap").head(12).to_string(index=False))

    notes = audit(df)
    print("\n=== audit ===")
    for n in notes:
        print(f"  * {n}")
    if not notes:
        print("  (no flags)")

    for name in (PER_TASK_CSV, SUITE_CSV, GAPS_CSV, LEAVE_OUT_CSV,
             LEAVE_OUT_SUMMARY_CSV):
        print(f"wrote {art_dir / name}")
    return 0
