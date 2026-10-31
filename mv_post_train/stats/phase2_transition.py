"""Within-model Phase 2 transitions: does OT30K help, per family?

Figure 3's claim is not "MV beats the references after OT30K" — that is
Figure 2's claim carried forward. It is that **MV is the only model that gains
from Phase 2**, everything else loses reasoning. That is a within-model,
across-phase comparison, and none of the matched-compute machinery in
`analyze.py` tests it: that code compares models to each other inside a fixed
(phase, context) cell.

The transition is genuinely paired. A Phase 1 checkpoint and its OT30K
descendant are evaluated on the same benchmark items, so `item_index` lines up
and the paired bootstrap / sign-flip tests apply directly to the per-item
difference.

Transitions tested (Phase 1 start -> Phase 2 endpoint):
    sft  -> ot30k_on_sft
    dpo  -> ot30k_on_dpo
    base -> ot30k_on_base

Phase 2 comes from the HF dump (`phase2_items.parquet`) and only from there.
`logs.zip` carries the same OT30K checkpoints across more benchmarks, and
parsing it is worth doing for provenance work, but its cells do not reproduce
the submission's Table 1 on AIME25 or MATH500 (see
docs/phase2-aime-provenance.md). Broader coverage is worth nothing if the
numbers are not the ones the paper reports, so the wider table is not an
alternative this module offers.

That leaves four pairable tasks: AIME24, AIME25, AMC23, MATH500. The dump's
fifth task, gsm8k, has no item-level Phase 1 counterpart to pair against, and
the remaining six benchmarks of the suite are absent from the dump's Phase 2
export entirely.

Writes `artifacts/phase2_transition.csv`.
"""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

from .. import paths
from .core import bh_adjust, paired_bootstrap_ci, paired_perm_test, paired_wilcoxon

TRANSITION_CSV = "phase2_transition.csv"

TRANSITIONS = [
    ("base", "ot30k_on_base"),
    ("sft", "ot30k_on_sft"),
    ("dpo", "ot30k_on_dpo"),
]

def _per_item(items: pd.DataFrame, family: str, phase: str, task: str,
              tier: str) -> pd.Series | None:
    """Mean score per item for one cell, indexed by item_index.

    Averaging over `attempt_index` first is what makes cells with ten greedy
    repetitions comparable to cells with one: the paired unit is the problem,
    not the generation.
    """
    cell = items[
        (items.family == family)
        & (items.phase == phase)
        & (items.task == task)
        & (items.context_tier == tier)
    ]
    if cell.empty:
        return None
    return cell.groupby("item_index").score.mean()


def transitions(items: pd.DataFrame, n_resamples: int = 10000) -> pd.DataFrame:
    rows = []
    for family in sorted(items.family.unique()):
        # Context tier is a property of the family, not a constant. The
        # MV-style families evaluate at 16k, but Qwen2.5 evaluates at 131k and
        # Qwen3 at 32k -- their native lengths. Hardcoding one tier silently
        # dropped both, which is why this table used to show four families.
        #
        # The transition is within-model, so the tier does not need to be any
        # particular value; it only has to be the SAME at both ends, or the
        # difference confounds post-training with context length. Families
        # carrying several tiers (MV and Comma have 4k and 16k in Phase 1) get
        # one row per tier, and tiers with no Phase 2 counterpart drop out on
        # their own.
        fam_tiers = sorted(items[items.family == family].context_tier.unique())
        for tier in fam_tiers:
            for start, end in TRANSITIONS:
                for task in sorted(items.task.unique()):
                    a = _per_item(items, family, end, task, tier)
                    b = _per_item(items, family, start, task, tier)
                    if a is None or b is None:
                        continue
                    shared = a.index.intersection(b.index)
                    if len(shared) < 2:
                        continue
                    y_a = a.loc[shared].to_numpy(dtype=float)
                    y_b = b.loc[shared].to_numpy(dtype=float)

                    delta = float((y_a - y_b).mean() * 100)
                    common = {
                        "family": family, "context_tier": tier,
                        "transition": f"{start}->{end}", "task": task,
                        "n_items": len(shared),
                        "acc_start": round(float(y_b.mean() * 100), 2),
                        "acc_end": round(float(y_a.mean() * 100), 2),
                        "delta": round(delta, 2),
                    }
                    # Both cells constant and equal -> nothing to test. Reported
                    # rather than dropped: "both score 0 on every item" is the
                    # actual finding for the references on contest math.
                    if np.allclose(y_a, y_b):
                        rows.append({
                            **common,
                            "ci_lo": 0.0, "ci_hi": 0.0,
                            "p_perm": 1.0, "p_wilcoxon": 1.0,
                            "degenerate": True,
                        })
                        continue

                    ci = paired_bootstrap_ci(y_a, y_b, n_resamples=n_resamples)
                    _, p = paired_perm_test(y_a, y_b, n_resamples=n_resamples)
                    rows.append({
                        **common,
                        "ci_lo": round(float(ci.lo) * 100, 2),
                        "ci_hi": round(float(ci.hi) * 100, 2),
                        "p_perm": float(p),
                        "p_wilcoxon": float(paired_wilcoxon(y_a, y_b)),
                        "degenerate": False,
                    })

    out = pd.DataFrame(rows)
    if out.empty:
        return out
    # One BH family across the whole table — every transition test reported
    # together is one family of hypotheses.
    live = out.index[~out.degenerate]
    out["q_value"] = 1.0
    out.loc[live, "q_value"] = bh_adjust(out.loc[live, "p_perm"].tolist())
    out["q_value"] = out.q_value.round(4)
    out["p_perm"] = out.p_perm.round(4)
    out["p_wilcoxon"] = out.p_wilcoxon.round(4)
    return out


def main(data_dir: Path, art_dir: Path) -> int:
    data_dir, art_dir = Path(data_dir), Path(art_dir)
    phase1 = data_dir / paths.ITEMS_LONG
    phase2 = data_dir / paths.PHASE2_ITEMS
    for p in (phase1, phase2):
        if not p.exists():
            print(f"missing {p}")
            return 1
    art_dir.mkdir(parents=True, exist_ok=True)

    frames = [pd.read_parquet(phase1), pd.read_parquet(phase2)]
    names = [phase1.name, phase2.name]

    # logs.zip has no item-level gsm8k, so without this the Phase 2 gsm8k
    # cells have nothing to pair against and the benchmark silently vanishes
    # from the table. `parse_phase1_dump` emits only tasks items_long.parquet
    # lacks, so this cannot duplicate an item.
    # `recovered_items` carries both phases for IFEval / MBPP, which the dump
    # ships without verdicts; `phase1_dump_items` carries only the Phase 1 side
    # of gsm8k, whose Phase 2 side is already in phase2_items.parquet. Sharing
    # a task name across sources is therefore normal and not an error -- what
    # must never happen is the same cell arriving twice, which would inflate n
    # and narrow every interval.
    for extra_path in (data_dir / paths.PHASE1_DUMP_ITEMS,
                       data_dir / paths.RECOVERED_ITEMS):
        if not extra_path.exists():
            continue
        extra = pd.read_parquet(extra_path)
        if not extra.empty:
            frames.append(extra)
            names.append(extra_path.name)

    items = pd.concat(frames, ignore_index=True)
    print(f"reading items from {' + '.join(names)}")

    key = ["model", "task", "item_index", "attempt_index"]
    dup = items.duplicated(subset=key)
    if dup.any():
        d = items[dup].groupby(["family", "phase", "task"]).size()
        print(f"ERROR: {int(dup.sum())} duplicated item rows across sources:")
        print(d.to_string())
        return 1

    # `_per_item` pools every model in a (family, phase, tier) cell. That is
    # right for one checkpoint evaluated once, and wrong when a cell holds two
    # different checkpoints -- SmolLM2/base/16k carries both
    # `ali-elganzory__SmolLM2-1.7B-16k` and `ontocord__SmolLM2-1.7B-16k`, whose
    # scores would be averaged into a model that does not exist. Warn rather
    # than fail: no such cell currently takes part in a transition, and dropping
    # them silently would be its own bug.
    per_cell = (items.groupby(["family", "phase", "context_tier"])
                     .model.nunique())
    pooled = per_cell[per_cell > 1]
    if not pooled.empty:
        print(f"warning: {len(pooled)} (family, phase, tier) cell(s) hold "
              f"several checkpoints and will be pooled:")
        for (fam, ph, tier), n in pooled.items():
            print(f"    {fam} / {ph} / {tier}: {n} models")
    out = transitions(items)
    if out.empty:
        print("no paired Phase 1 -> Phase 2 cells found")
        return 1
    out_csv = art_dir / TRANSITION_CSV
    out.to_csv(out_csv, index=False)
    print(out.to_string(index=False))
    print(f"\n{len(out)} cells: {int((~out.degenerate).sum())} tested, "
          f"{int(out.degenerate.sum())} degenerate (both ends identical)")
    print(f"wrote {out_csv}")
    return 0
