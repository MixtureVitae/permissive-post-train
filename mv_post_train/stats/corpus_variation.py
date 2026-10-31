"""Does the *base* condition the response, or does the *corpus* drive it?

Two claims about Phase 2 are really one experiment read along two axes:

- **Fix the corpus, vary the base.** The same OT30K update lifts MixtureVitae
  and does nothing for the matched references.
- **Fix the base, vary the corpus.** Holding the MixtureVitae SFT checkpoint
  fixed, three other reasoning corpora reproduce most of the OT3 gain, so the
  effect is not an artifact of OT3 overlapping MixtureVitae's pretraining.

The release carries the full grid needed for both: four reasoning corpora
(OT3 / OpenThoughts3, AR / AceReason, NC / Nemotron-Cascade, SR /
Superior-Reasoning) applied on top of both the SFT and the DPO checkpoint, for
MixtureVitae and every matched reference. That is 4 corpora x 2 routes x 6
bases, and every cell is item-level.

This module runs the paired transition for all of them, so the corpus-variation
table stops being bare point estimates: each arm gets the same 95 % paired
bootstrap and sign-flip permutation as the OT3 arm already had, over the same
starting items.

Why this is worth doing rather than quoting released-grid subtractions
---------------------------------------------------------------------
A difference of two published cells has no interval, and -- as
`phase2_transition_dump` documents -- can silently cross evaluation runs. Here
both ends of every arm come from the same release, the same scorer and the same
starting checkpoint, so the delta means what it says and carries an interval.

Writes `corpus_variation.csv` (per task) and `corpus_variation_pooled.csv`
(the five math benchmarks pooled, macro and micro).
"""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

from .core import bh_adjust, paired_bootstrap_ci, paired_perm_test
from .math_composite import MATH_TASKS, _bootstrap, _macro, _micro, _perm_p
from .phase2_vs_baselines import MV, REFERENCES, _checkpoints, dump_scores

PER_TASK_CSV = "corpus_variation.csv"
POOLED_CSV = "corpus_variation_pooled.csv"

# The dump's stage prefix -> the corpus name the paper uses.
CORPORA = {
    "OT3": "OpenThoughts3",
    "AR": "AceReason",
    "NC": "Nemotron-Cascade",
    "SR": "Superior-Reasoning",
}

# (Phase 1 starting stage, suffix of the Phase 2 stage).
ROUTES = [("SFT-Tulu3", "+SFT"), ("DPO-Tulu3", "+DPO")]

MODELS = [MV, *REFERENCES]

ALL_STAGES = sorted(
    {start for start, _ in ROUTES}
    | {f"{c}{suffix}" for c in CORPORA for _, suffix in ROUTES}
)


def _pairs(recs: dict, model: str, corpus: str, start: str, suffix: str):
    """task -> (start scores, end scores) aligned on shared items."""
    end_stage = f"{corpus}{suffix}"
    out = {}
    for task in MATH_TASKS:
        a = recs.get((model, start, task))
        b = recs.get((model, end_stage, task))
        if not a or not b:
            continue
        shared = sorted(set(a) & set(b))
        if len(shared) < 2:
            continue
        out[task] = (
            np.array([np.mean(a[i]) for i in shared], dtype=float),
            np.array([np.mean(b[i]) for i in shared], dtype=float),
        )
    return out


def per_task(recs: dict, n_resamples: int = 10000) -> pd.DataFrame:
    """One paired transition per (model, corpus, route, task)."""
    rows = []
    for model in MODELS:
        for corpus in CORPORA:
            for start, suffix in ROUTES:
                for task, (y0, y1) in _pairs(recs, model, corpus, start, suffix).items():
                    common = {
                        "model": model, "corpus": CORPORA[corpus],
                        "route": f"{start}->{corpus}{suffix}",
                        "task": task, "n_items": len(y0),
                        "acc_start": round(float(y0.mean() * 100), 2),
                        "acc_end": round(float(y1.mean() * 100), 2),
                        "delta": round(float((y1 - y0).mean() * 100), 2),
                    }
                    # Both ends identical on every item -- the references on
                    # contest math, which sit at 0 before and after. Reported,
                    # not dropped: that is the finding.
                    if np.allclose(y0, y1):
                        rows.append({**common, "ci_lo": 0.0, "ci_hi": 0.0,
                                     "p_perm": 1.0, "degenerate": True})
                        continue
                    ci = paired_bootstrap_ci(y1, y0, n_resamples=n_resamples)
                    _, p = paired_perm_test(y1, y0, n_resamples=n_resamples)
                    rows.append({**common,
                                 "ci_lo": round(float(ci.lo) * 100, 2),
                                 "ci_hi": round(float(ci.hi) * 100, 2),
                                 "p_perm": float(p), "degenerate": False})

    out = pd.DataFrame(rows)
    if out.empty:
        return out
    live = out.index[~out.degenerate]
    out["q_value"] = 1.0
    out.loc[live, "q_value"] = bh_adjust(out.loc[live, "p_perm"].tolist())
    for col in ("p_perm", "q_value"):
        out[col] = out[col].round(5)
    return out


def pooled(recs: dict, n_resamples: int = 10000, n_perm: int = 10000,
           seed: int = 0) -> pd.DataFrame:
    """The five math benchmarks pooled, per (model, corpus, route).

    Same macro / micro weightings as `math_composite`, applied to the
    within-model transition rather than the between-model contrast.
    """
    rows = []
    for model in MODELS:
        for corpus in CORPORA:
            for start, suffix in ROUTES:
                pairs = _pairs(recs, model, corpus, start, suffix)
                if len(pairs) < 2:
                    continue
                diffs = {t: y1 - y0 for t, (y0, y1) in pairs.items()}
                (ma_lo, ma_hi), (mi_lo, mi_hi) = _bootstrap(diffs, n_resamples, seed)
                rows.append({
                    "model": model, "corpus": CORPORA[corpus],
                    "route": f"{start}->{corpus}{suffix}",
                    "n_tasks": len(diffs),
                    "n_items": int(sum(len(d) for d in diffs.values())),
                    "macro_delta": round(_macro(diffs) * 100, 2),
                    "macro_ci_lo": round(ma_lo * 100, 2),
                    "macro_ci_hi": round(ma_hi * 100, 2),
                    "macro_p": _perm_p(diffs, _macro, n_perm, seed),
                    "micro_delta": round(_micro(diffs) * 100, 2),
                    "micro_ci_lo": round(mi_lo * 100, 2),
                    "micro_ci_hi": round(mi_hi * 100, 2),
                    "micro_p": _perm_p(diffs, _micro, n_perm, seed),
                })
    out = pd.DataFrame(rows)
    if out.empty:
        return out
    out["macro_q"] = bh_adjust(out.macro_p.tolist())
    out["micro_q"] = bh_adjust(out.micro_p.tolist())
    for col in ("macro_p", "micro_p", "macro_q", "micro_q"):
        out[col] = out[col].round(5)
    return out


def main(responses: Path, art_dir: Path) -> int:
    responses, art_dir = Path(responses), Path(art_dir)
    if not responses.exists():
        print(f"missing {responses}")
        return 1
    art_dir.mkdir(parents=True, exist_ok=True)

    keep, pooled_cells = _checkpoints(responses, ALL_STAGES)
    if pooled_cells:
        print(f"{len(pooled_cells)} pooled cell(s); canonical checkpoint only:")
        for (model, stage, task), counts in sorted(pooled_cells):
            print(f"    {model} / {stage} / {task}: {len(counts)} checkpoints")

    recs = dump_scores(responses, keep, ALL_STAGES)

    pt = per_task(recs)
    if pt.empty:
        print("no transitions found")
        return 1
    pt.to_csv(art_dir / PER_TASK_CSV, index=False)

    pl = pooled(recs)
    pl.to_csv(art_dir / POOLED_CSV, index=False)

    m500 = pt[(pt.task == "MATH500") & (pt.route.str.endswith("+SFT"))]
    print("\n=== MATH500, SFT route, by base and corpus ===")
    print(m500.pivot_table(index="model", columns="corpus", values="delta").to_string())
    print("\n=== pooled math (macro), SFT route ===")
    print(pl[pl.route.str.endswith("+SFT")]
          .pivot_table(index="model", columns="corpus", values="macro_delta").to_string())
    print(f"\n{len(pt)} per-task transitions, {len(pl)} pooled")
    print(f"wrote {art_dir / PER_TASK_CSV}")
    print(f"wrote {art_dir / POOLED_CSV}")
    return 0
