"""One math claim, tested once: MixtureVitae vs each reference across all five
math benchmarks pooled.

Why pool
--------
`phase2_vs_baselines.py` runs 40 separate contrasts, and the contest-math ones
are individually hopeless: AIME24 and AIME25 carry 30 items each, so their
minimum detectable effect is around 28-30 pp (`core.mde_paired_binomial`) and
they cannot clear a multiplicity correction no matter how the model behaves.
Testing them one at a time answers "is MV better on AIME24?", which the data
cannot support -- but the claim the paper actually makes is "MV is stronger at
math", and that claim has 1,919 paired items behind it, not 30.

Pooling is legitimate here for three specific reasons:

1. **One vintage.** Every math cell at every OT30K stage was generated 3-7 May
   2026 (`artifacts/audit/response_vintage.csv`), so no benchmark enters the
   composite on a different side of any configuration change.
2. **One scorer.** MATH500 and contest math are scored by `math-verify`; GSM8K
   carries the release's own verdicts. Both are exact-answer matching on a
   numeric or symbolic final answer.
3. **Paired throughout.** Every item is answered by both models being compared,
   so the composite is a mean of paired differences, not a difference of two
   independently pooled rates.

Two weightings, both reported
-----------------------------
They answer different questions and the honest thing is to show both rather
than pick the flattering one.

- **Macro** -- unweighted mean over the five benchmark-level differences. Each
  benchmark gets 20 %, so contest math carries 60 % of the composite despite
  being 5 % of the items. This is the weighting that treats "math" as the five
  benchmarks the paper chose.
- **Micro** -- every item is one unit across 1,919 items. GSM8K alone is 69 %
  of them, so this is close to a GSM8K + MATH500 result with contest math as a
  rounding term.

Both use the same bootstrap resamples, so the two intervals are directly
comparable rather than being separate simulations that happen to be adjacent.

Leave-one-benchmark-out
-----------------------
A composite that only survives because of its largest component is not a
composite. Every contrast is therefore re-run five times with one benchmark
removed, and the **worst** of those five is reported alongside the headline. If
dropping GSM8K or MATH500 collapses the result, that shows up in the table
instead of being buried.

Both phases
-----------
The same contrast runs at Phase 1 (`base`, `SFT-Tulu3`, `DPO-Tulu3`) and Phase 2
(`OT3`, `OT3+SFT`, `OT3+DPO`), from the same source with the same scorer, so the
two tables are comparable. Coverage is not symmetric and the gaps are real:

- `base` yields **no** contrast at all -- the release carries MixtureVitae's
  base checkpoint on these benchmarks but no reference's.
- Phase 1 `SFT-Tulu3` is the richest cell in the release: all five references,
  including DCLM, which has no Phase 2 stages anywhere.
- Nemotron-CC-HQ and SmolLM2 have no Phase 1 GSM8K generations, so their
  Phase 1 composites run on four benchmarks and 600 items, not five and 1,919.

Vintage holds across both: every math cell used here was generated 3-8 May 2026
(`artifacts/audit/response_vintage.csv`).

Writes `math_composite.csv`.
"""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

from ..data.benchmarks import AIME24, AIME25, AMC23, MATH500
from .core import bh_adjust
from .phase2_vs_baselines import (
    GSM8K,
    MV,
    PHASE1_STAGES,
    PHASE2_STAGES,
    REFERENCES,
    _checkpoints,
    dump_scores,
)

COMPOSITE_CSV = "math_composite.csv"
LOO_CSV = "math_composite_loo.csv"

# All five, in the order the response reads them. GSM8K is the largest and
# easiest; AIME24/25 the smallest and hardest.
MATH_TASKS = [GSM8K, MATH500, AMC23, AIME24, AIME25]

# The three benchmarks that cannot be tested individually: 30 + 30 + 40 items,
# MDEs of 28-30 pp apiece. Pooled they are 100 paired items, which is a
# different proposition -- so the composite is also reported restricted to
# them, to show whether pooling rescues the contest-math claim on its own or
# only carries it on MATH500's and GSM8K's back.
CONTEST_TASKS = [AMC23, AIME24, AIME25]

# Both phases, run through one code path so the two tables are comparable.
# Order matters: it is the order the emitted table reads in.
STAGE_PHASE = {s: "Phase 1" for s in PHASE1_STAGES} | {s: "Phase 2" for s in PHASE2_STAGES}
ALL_STAGES = [*PHASE1_STAGES, *PHASE2_STAGES]


def _paired_diffs(recs: dict, stage: str, ref: str) -> dict[str, np.ndarray]:
    """task -> per-item (MV - reference) differences, over shared items only."""
    out: dict[str, np.ndarray] = {}
    for task in MATH_TASKS:
        mv = recs.get((MV, stage, task))
        other = recs.get((ref, stage, task))
        if not mv or not other:
            continue
        shared = sorted(set(mv) & set(other))
        if len(shared) < 2:
            continue
        y_mv = np.array([np.mean(mv[i]) for i in shared], dtype=float)
        y_rf = np.array([np.mean(other[i]) for i in shared], dtype=float)
        out[task] = y_mv - y_rf
    return out


def _macro(diffs: dict[str, np.ndarray]) -> float:
    return float(np.mean([d.mean() for d in diffs.values()]))


def _micro(diffs: dict[str, np.ndarray]) -> float:
    return float(np.concatenate(list(diffs.values())).mean())


def _bootstrap(diffs: dict[str, np.ndarray], n_resamples: int, seed: int,
               alpha: float = 0.05) -> tuple[tuple[float, float], tuple[float, float]]:
    """Stratified paired bootstrap CIs for (macro, micro) from shared resamples.

    Items are resampled *within* each benchmark: an AIME24 item can only be
    replaced by another AIME24 item. Resampling across the pooled 1,919 would
    let benchmark sizes wander between replicates and silently reweight the
    macro statistic.
    """
    rng = np.random.default_rng(seed)
    tasks = list(diffs)
    ns = {t: len(diffs[t]) for t in tasks}
    macro = np.empty(n_resamples)
    micro = np.empty(n_resamples)
    for r in range(n_resamples):
        means, pooled = [], []
        for t in tasks:
            d = diffs[t][rng.integers(0, ns[t], ns[t])]
            means.append(d.mean())
            pooled.append(d)
        macro[r] = np.mean(means)
        micro[r] = np.concatenate(pooled).mean()
    q = [alpha / 2, 1 - alpha / 2]
    return tuple(np.quantile(macro, q)), tuple(np.quantile(micro, q))


def _perm_p(diffs: dict[str, np.ndarray], stat, n_resamples: int,
            seed: int) -> float:
    """Two-sided sign-flip permutation p for a composite statistic.

    The null is that the two models are exchangeable on each item, so each
    item's paired difference independently flips sign. Benchmark structure is
    preserved: the statistic is recomputed from the flipped per-benchmark
    arrays, which is what makes this valid for the macro weighting too.
    """
    rng = np.random.default_rng(seed)
    observed = abs(stat(diffs))
    tasks = list(diffs)
    hits = 0
    for _ in range(n_resamples):
        flipped = {t: diffs[t] * rng.choice([-1.0, 1.0], size=len(diffs[t]))
                   for t in tasks}
        if abs(stat(flipped)) >= observed:
            hits += 1
    return (hits + 1) / (n_resamples + 1)


def survives(row) -> bool:
    """Did this composite survive dropping its worst benchmark?

    The criterion, stated once so no table has to imply one: **the worst
    leave-one-out subset's 95 % bootstrap interval excludes zero, under both
    weightings.** A bare point estimate is not enough -- SmolLM2's Phase 1
    composite leaves +1.1 pp macro after dropping MATH500, which looks positive
    and is indistinguishable from zero over 100 contest-math items.

    This is a robustness *diagnostic*, not a hypothesis test. The subset is
    chosen by being the worst of five, so the interval is selected post hoc and
    does not carry 95 % coverage; it is here to stop "survives" from resting on
    an unstated threshold, not to license a second significance claim.
    """
    return bool(row.worst_loo_macro_ci_lo > 0 and row.worst_loo_micro_ci_lo > 0)


def composite_contrasts(recs: dict, n_resamples: int = 10000,
                        n_perm: int = 10000, seed: int = 0,
                        loo_out: list | None = None) -> pd.DataFrame:
    """One pooled math contrast per (stage, reference).

    `loo_out`, if given, is extended with one row per leave-one-out subset --
    the long-format detail behind the `worst_loo_*` columns.
    """
    rows = []
    all_loo = loo_out if loo_out is not None else []
    for stage in ALL_STAGES:
        for ref in REFERENCES:
            diffs = _paired_diffs(recs, stage, ref)
            if len(diffs) < 2:
                continue
            macro, micro = _macro(diffs), _micro(diffs)
            (mac_lo, mac_hi), (mic_lo, mic_hi) = _bootstrap(diffs, n_resamples, seed)

            # Leave-one-benchmark-out, under BOTH weightings. The worst subset
            # is not always the same benchmark for the two: micro is driven by
            # whichever benchmark carries the most items, macro by whichever
            # carries the largest gap, and those differ.
            #
            # Each subset gets a bootstrap interval so that "the composite
            # survives dropping any one benchmark" is a stated criterion rather
            # than an eyeballed point estimate. See `_SURVIVES` below.
            loo_macro, loo_micro, loo_rows = {}, {}, []
            if len(diffs) > 2:
                for drop in diffs:
                    kept = {t: d for t, d in diffs.items() if t != drop}
                    loo_macro[drop] = _macro(kept)
                    loo_micro[drop] = _micro(kept)
                    (lo_a, hi_a), (lo_i, hi_i) = _bootstrap(kept, n_resamples, seed)
                    loo_rows.append({
                        "phase": STAGE_PHASE[stage], "stage": stage,
                        "reference": ref, "dropped": drop,
                        "n_tasks": len(kept),
                        "n_items": int(sum(len(d) for d in kept.values())),
                        "macro_delta": round(loo_macro[drop] * 100, 2),
                        "macro_ci_lo": round(lo_a * 100, 2),
                        "macro_ci_hi": round(hi_a * 100, 2),
                        "micro_delta": round(loo_micro[drop] * 100, 2),
                        "micro_ci_lo": round(lo_i * 100, 2),
                        "micro_ci_hi": round(hi_i * 100, 2),
                    })
            worst_macro = min(loo_macro, key=loo_macro.get) if loo_macro else None
            worst_micro = min(loo_micro, key=loo_micro.get) if loo_micro else None
            worst_rows = {r["dropped"]: r for r in loo_rows}
            all_loo.extend(loo_rows)

            # The same composite restricted to the three benchmarks that are
            # individually underpowered, again under both weightings. Here the
            # two are close by construction -- AMC23's 40 items against
            # AIME's 30 + 30 is nearly balanced -- but reporting only one would
            # still leave the reader to assume which.
            contest = {t: d for t, d in diffs.items() if t in CONTEST_TASKS}
            if len(contest) >= 2:
                (cma_lo, cma_hi), (cmi_lo, cmi_hi) = _bootstrap(contest, n_resamples, seed)
                contest_cols = {
                    "contest_n_items": int(sum(len(d) for d in contest.values())),
                    "contest_macro_delta": round(_macro(contest) * 100, 2),
                    "contest_macro_ci_lo": round(cma_lo * 100, 2),
                    "contest_macro_ci_hi": round(cma_hi * 100, 2),
                    "contest_macro_p": _perm_p(contest, _macro, n_perm, seed),
                    "contest_micro_delta": round(_micro(contest) * 100, 2),
                    "contest_micro_ci_lo": round(cmi_lo * 100, 2),
                    "contest_micro_ci_hi": round(cmi_hi * 100, 2),
                    "contest_micro_p": _perm_p(contest, _micro, n_perm, seed),
                }
            else:
                contest_cols = dict.fromkeys(
                    ["contest_macro_delta", "contest_macro_ci_lo",
                     "contest_macro_ci_hi", "contest_macro_p",
                     "contest_micro_delta", "contest_micro_ci_lo",
                     "contest_micro_ci_hi", "contest_micro_p"], np.nan)
                contest_cols["contest_n_items"] = 0

            rows.append({
                "phase": STAGE_PHASE[stage], "stage": stage, "reference": ref,
                "n_tasks": len(diffs),
                "n_items": int(sum(len(d) for d in diffs.values())),
                **contest_cols,
                "macro_delta": round(macro * 100, 2),
                "macro_ci_lo": round(mac_lo * 100, 2),
                "macro_ci_hi": round(mac_hi * 100, 2),
                "macro_p": _perm_p(diffs, _macro, n_perm, seed),
                "micro_delta": round(micro * 100, 2),
                "micro_ci_lo": round(mic_lo * 100, 2),
                "micro_ci_hi": round(mic_hi * 100, 2),
                "micro_p": _perm_p(diffs, _micro, n_perm, seed),
                "worst_loo_macro": round(loo_macro[worst_macro] * 100, 2) if worst_macro else np.nan,
                "worst_loo_macro_dropped": worst_macro,
                "worst_loo_macro_ci_lo": worst_rows[worst_macro]["macro_ci_lo"] if worst_macro else np.nan,
                "worst_loo_macro_ci_hi": worst_rows[worst_macro]["macro_ci_hi"] if worst_macro else np.nan,
                "worst_loo_micro": round(loo_micro[worst_micro] * 100, 2) if worst_micro else np.nan,
                "worst_loo_micro_dropped": worst_micro,
                "worst_loo_micro_ci_lo": worst_rows[worst_micro]["micro_ci_lo"] if worst_micro else np.nan,
                "worst_loo_micro_ci_hi": worst_rows[worst_micro]["micro_ci_hi"] if worst_micro else np.nan,
                "tasks": "+".join(diffs),
            })

    out = pd.DataFrame(rows)
    if out.empty:
        return out
    # One BH family per (benchmark set x weighting). Each is declared by the
    # code -- all contrasts this command runs -- not chosen after seeing which
    # came out large.
    out["macro_q"] = bh_adjust(out.macro_p.tolist())
    out["micro_q"] = bh_adjust(out.micro_p.tolist())
    for weighting in ("macro", "micro"):
        p_col, q_col = f"contest_{weighting}_p", f"contest_{weighting}_q"
        live = out.index[out[p_col].notna()]
        out[q_col] = np.nan
        if len(live):
            out.loc[live, q_col] = bh_adjust(out.loc[live, p_col].tolist())
    for col in ("macro_p", "micro_p", "macro_q", "micro_q",
                "contest_macro_p", "contest_macro_q",
                "contest_micro_p", "contest_micro_q"):
        out[col] = out[col].round(5)
    out["survives_loo"] = [survives(r) for r in out.itertuples()]
    return out


def main(responses: Path, art_dir: Path) -> int:
    responses, art_dir = Path(responses), Path(art_dir)
    if not responses.exists():
        print(f"missing {responses}")
        return 1
    art_dir.mkdir(parents=True, exist_ok=True)

    keep, pooled = _checkpoints(responses, ALL_STAGES)
    if pooled:
        print(f"{len(pooled)} pooled cell(s); canonical checkpoint only:")
        for (model, stage, task), counts in sorted(pooled):
            print(f"    {model} / {stage} / {task}: "
                  f"{len(counts)} checkpoints, keeping "
                  f"{keep[(model, stage, task)].split('__checkpoints__')[-1][:60]}")

    recs = dump_scores(responses, keep, ALL_STAGES)
    loo_rows: list = []
    out = composite_contrasts(recs, loo_out=loo_rows)
    if out.empty:
        print("no composite contrasts found")
        return 1

    out_csv = art_dir / COMPOSITE_CSV
    out.to_csv(out_csv, index=False)
    loo_csv = art_dir / LOO_CSV
    pd.DataFrame(loo_rows).to_csv(loo_csv, index=False)

    show = ["stage", "reference", "n_tasks", "n_items",
            "macro_delta", "macro_ci_lo", "macro_ci_hi", "macro_q",
            "micro_delta", "micro_ci_lo", "micro_ci_hi", "micro_q"]
    contest_show = ["stage", "reference", "contest_n_items",
                    "contest_macro_delta", "contest_macro_ci_lo",
                    "contest_macro_ci_hi", "contest_macro_q",
                    "contest_micro_delta", "contest_micro_ci_lo",
                    "contest_micro_ci_hi", "contest_micro_q"]
    loo_show = ["stage", "reference",
                "worst_loo_macro_dropped", "worst_loo_macro",
                "worst_loo_macro_ci_lo", "worst_loo_macro_ci_hi",
                "worst_loo_micro", "worst_loo_micro_ci_lo",
                "worst_loo_micro_ci_hi", "survives_loo"]
    for phase in ("Phase 1", "Phase 2"):
        block = out[out.phase == phase]
        if block.empty:
            print(f"\n=== {phase}: no contrasts ===")
            continue
        print(f"\n=== {phase}: all five math benchmarks pooled ===")
        print(block[show].to_string(index=False))
        print(f"\n=== {phase}: contest math alone (AMC23 + AIME24 + AIME25) ===")
        print(block[contest_show].to_string(index=False))
        print(f"\n=== {phase}: leave-one-benchmark-out (worst subset) ===")
        print(block[loo_show].to_string(index=False))

    print(f"\n{len(out)} composite contrasts over {sorted(MATH_TASKS)} "
          f"({int((out.phase == 'Phase 1').sum())} Phase 1, "
          f"{int((out.phase == 'Phase 2').sum())} Phase 2)")
    print(f"significant at q<0.05 of {len(out)}: "
          f"macro {int((out.macro_q < 0.05).sum())}, "
          f"micro {int((out.micro_q < 0.05).sum())}, "
          f"contest-only macro {int((out.contest_macro_q < 0.05).sum())}, "
          f"contest-only micro {int((out.contest_micro_q < 0.05).sum())}")
    print(f"survives leave-one-out (worst subset's 95% CI excludes zero under "
          f"both weightings): {int(out.survives_loo.sum())} of {len(out)} "
          f"-- Phase 1 {int(out[out.phase == 'Phase 1'].survives_loo.sum())}"
          f"/{int((out.phase == 'Phase 1').sum())}, "
          f"Phase 2 {int(out[out.phase == 'Phase 2'].survives_loo.sum())}"
          f"/{int((out.phase == 'Phase 2').sum())}")
    print(f"wrote {out_csv}")
    print(f"wrote {loo_csv}")
    return 0
