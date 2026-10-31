"""MixtureVitae vs the matched-compute references **at the OT30K endpoint**.

Why this exists
---------------
`phase2_transition_dump.py` answers "does this model gain from Phase 2?" --
a within-model, across-stage question. This module answers the other one the
paper actually claims in Figure 3: **after OT30K, is MixtureVitae ahead of the
references, and on what?** That is a between-model contrast at a fixed stage,
paired on the benchmark item, which is a different test with a different
denominator.

The honest answer is that only part of the 11-task suite can carry it, and the
limit is data availability rather than statistics. Within the released dump,
at an OT30K stage, with a matched-compute reference on the other side:

===============  ==========================================================
GSM8K            testable -- 1,319 items, and the only task the dump ships
                 with the authors' own `is_correct` verdicts.
MATH500          testable -- 500 items, re-scored here with `math-verify`.
AIME24/25/AMC23  testable but nearly uninformative: 30/30/40 items, and
                 every reference scores 0 on every item, so the contrast is
                 "some vs none" rather than a measured difference.
IFEval           testable -- 500 prompts, via the official checker
                 (`recovered_items.parquet`; see `data/recover_items.py`).
GPQADiamond      **not** scorable from the dump: `gold_answer` holds answer
                 text while `model_answer` holds a letter, and the prompt
                 carries no options to map between them.
JEEBench         **not** scorable from the dump: `model_answer` is empty in
                 82% of the OT30K reference rows.
HumanEval        no reference rows at any OT30K stage. The dump carries
MBPP             MixtureVitae `OT3+DPO` and the Qwen/SmolLM2 externals only,
LiveCodeBenchv5  so the released 0.0 cells for Comma/FineWeb-Edu/Nemotron
                 cannot be reproduced or tested from the dump at all.
===============  ==========================================================

So the between-model claim rests on GSM8K and MATH500 as measured results,
IFEval as a small measured result, and contest math as a floor effect. The
six remaining columns of Figure 3 are asserted from the released table and
are not backed by anything item-level here.

Multiplicity
------------
The BH family is every live contrast this module runs -- all (stage, task,
reference) triples with both ends present and not jointly degenerate -- and
q-values are reported alongside p. Declaring the family after seeing which
contrasts are large would be the thing that invalidates them.

Writes `phase2_vs_baselines.csv` (the contrasts) and reuses
`dump_cell_levels.csv`-style levels via the `acc_mv` / `acc_ref` columns.
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
from .phase2_transition_dump import BATCH_ROWS, MAX_ANSWER_CHARS, canonical_checkpoints

CONTRAST_CSV = "phase2_vs_baselines.csv"

MV = "MixtureVitae-1.7B"

# The matched-compute references: same architecture, same token budget, same
# post-training recipe, different pretraining corpus. Qwen and the SmolLM2
# *Instruct* checkpoints are external comparators on a different compute
# footing and are deliberately out of scope here.
REFERENCES = [
    "Comma0.1-1.7B",
    "DCLM-1.7B",
    "FineWebEdu-1.7B",
    "NemotronHQ-1.7B",
    "SmolLM2-1.7B-16k-Base",
]

PHASE2_STAGES = ["OT3", "OT3+SFT", "OT3+DPO"]

# Phase 1, in the dump's stage vocabulary. Only `SFT-Tulu3` carries a full set
# of references; `base` carries MixtureVitae alone, so no `base` contrast
# exists. Used by `math_composite`, which runs the same comparison at both
# phases.
PHASE1_STAGES = ["base", "SFT-Tulu3", "DPO-Tulu3"]

# Scored here with `math-verify`, exactly as in `phase2_transition_dump`.
MATH_TASKS = [MATH500, AIME24, AIME25, AMC23]

# The one task the dump ships pre-scored. `is_correct` is null for all ten
# other benchmarks, so this is not a stylistic choice.
GSM8K = "GSM8K"

# Column order for the emitted table; `task` order drives how the response reads.
TASK_ORDER = [GSM8K, MATH500, "IFEval", AMC23, AIME24, AIME25]

_READ_COLUMNS = [
    "base_model", "stage", "task", "example_idx",
    "gold_answer", "model_answer", "is_correct", "source_dir",
]

_MODELS = {MV, *REFERENCES}


def _checkpoints(parquet: Path, stages: list[str] | None = None) -> tuple[dict, list[tuple]]:
    """Canonical `source_dir` per (base_model, stage, task) over this module's
    scope. Same group-level task-coverage rule as `phase2_transition_dump`;
    counting rows inside a cell instead picks the intruder (see that module)."""
    stages = set(stages if stages is not None else PHASE2_STAGES)
    tasks = set(MATH_TASKS) | {GSM8K}
    counts: dict[tuple, collections.Counter] = collections.defaultdict(collections.Counter)
    pf = pq.ParquetFile(parquet)
    for batch in pf.iter_batches(batch_size=BATCH_ROWS,
                                 columns=["base_model", "stage", "task", "source_dir"]):
        d = batch.to_pydict()
        for model, stage, task, src in zip(d["base_model"], d["stage"],
                                           d["task"], d["source_dir"]):
            if task in tasks and stage in stages and model in _MODELS:
                counts[(model, stage, task)][str(src)] += 1
    pooled = [(cell, dict(c)) for cell, c in counts.items() if len(c) > 1]
    return canonical_checkpoints(counts), pooled


def dump_scores(parquet: Path, keep: dict, stages: list[str] | None = None) -> dict:
    """(model, stage, task) -> {example_idx: [scores]}, canonical checkpoint only.

    GSM8K uses the dump's own `is_correct`; the math tasks are re-scored, since
    `is_correct` is null for every one of their rows.
    """
    stages = set(stages if stages is not None else PHASE2_STAGES)
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
            task, stage, model = d["task"][i], d["stage"][i], d["base_model"][i]
            if model not in _MODELS or stage not in stages:
                continue
            cell = (model, stage, task)
            if task == GSM8K:
                ok = d["is_correct"][i]
                if ok is None or str(d["source_dir"][i]) != keep.get(cell):
                    continue
                recs[cell][d["example_idx"][i]].append(float(ok))
            elif task in MATH_TASKS:
                if str(d["source_dir"][i]) != keep.get(cell):
                    continue
                s = score(task, d["gold_answer"][i], d["model_answer"][i])
                if s is not None:
                    recs[cell][d["example_idx"][i]].append(float(s))
        del d, batch
    return recs


def ifeval_scores(dump: Path, prompts: Path) -> dict:
    """IFEval verdicts straight from the dump, keyed like `dump_scores`.

    Deliberately **not** read from `recovered_items.parquet`. That file is
    gated on `parse_phase2.FAMILY`, which lists the five matched-compute
    corpora under their internal names and does not include SmolLM2 -- and
    SmolLM2 is the one reference that beats MixtureVitae on IFEval after
    OT30K. Sourcing the contrast from the gated file would have dropped the
    only losing comparison in the table, which is exactly the failure mode a
    reviewer should assume by default.

    Scoring is `ifeval_score.check_response` under the **prompt-level loose**
    criterion, the manuscript's stated convention (Appendix Tab. 18). Cells
    whose checkpoint contradicts their stage label are dropped, not scored.
    """
    from ..data import recover_items
    from . import ifeval_score

    dump, prompts = Path(dump), Path(prompts)
    if not prompts.exists():
        return {}
    scored = ifeval_score.score(dump, ifeval_score.load_prompt_metadata(prompts))
    if scored.empty:
        return {}
    excluded = recover_items.excluded_cells(dump)

    recs: dict = collections.defaultdict(lambda: collections.defaultdict(list))
    for row in scored.itertuples():
        if row.base_model not in _MODELS or row.stage not in PHASE2_STAGES:
            continue
        if (row.base_model, row.stage, "IFEval") in excluded:
            continue
        cell = (row.base_model, row.stage, "IFEval")
        recs[cell][int(row.example_id)].append(float(row.loose_all))
    return recs


def contrasts(recs: dict, n_resamples: int = 10000) -> pd.DataFrame:
    """Paired MV - reference contrasts at each Phase 2 stage.

    The paired unit is the benchmark item, so cells carrying ten greedy
    repetitions are collapsed to a per-item mean before pairing -- otherwise a
    ten-repeat cell would be weighted ten times against a one-repeat cell.
    """
    tasks = [t for t in TASK_ORDER if any(cell[2] == t for cell in recs)]
    rows = []
    for stage in PHASE2_STAGES:
        for task in tasks:
            mv = recs.get((MV, stage, task))
            if not mv:
                continue
            for ref in REFERENCES:
                other = recs.get((ref, stage, task))
                if not other:
                    continue
                shared = sorted(set(mv) & set(other))
                if len(shared) < 2:
                    continue
                y_mv = np.array([np.mean(mv[i]) for i in shared], dtype=float)
                y_rf = np.array([np.mean(other[i]) for i in shared], dtype=float)
                common = {
                    "stage": stage, "task": task, "reference": ref,
                    "n_items": len(shared),
                    "acc_mv": round(float(y_mv.mean() * 100), 2),
                    "acc_ref": round(float(y_rf.mean() * 100), 2),
                    "delta": round(float((y_mv - y_rf).mean() * 100), 2),
                }
                # Both models identical on every item -- on contest math this
                # means both scored 0 throughout. Reported, not dropped: it is
                # the finding for the references, and a test would be a
                # division by zero dressed up as p = 1.
                if np.allclose(y_mv, y_rf):
                    rows.append({**common, "ci_lo": 0.0, "ci_hi": 0.0,
                                 "p_perm": 1.0, "p_wilcoxon": 1.0,
                                 "degenerate": True})
                    continue
                ci = paired_bootstrap_ci(y_mv, y_rf, n_resamples=n_resamples)
                _, p = paired_perm_test(y_mv, y_rf, n_resamples=n_resamples)
                rows.append({**common,
                             "ci_lo": round(float(ci.lo) * 100, 2),
                             "ci_hi": round(float(ci.hi) * 100, 2),
                             "p_perm": float(p),
                             "p_wilcoxon": float(paired_wilcoxon(y_mv, y_rf)),
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


def main(responses: Path, prompts: Path, art_dir: Path) -> int:
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

    recs = dump_scores(responses, keep)
    ife = ifeval_scores(responses, prompts)
    if ife:
        recs.update(ife)
    else:
        print(f"note: {prompts} absent -- IFEval contrasts skipped")

    out = contrasts(recs)
    if out.empty:
        print("no paired contrasts found")
        return 1

    out_csv = art_dir / CONTRAST_CSV
    out.to_csv(out_csv, index=False)
    print(out.to_string(index=False))
    live = out[~out.degenerate]
    print(f"\n{len(out)} contrasts: {len(live)} tested, "
          f"{int(out.degenerate.sum())} degenerate (identical on every item)")
    print(f"significant at q<0.05: {int((live.q_value < 0.05).sum())} of {len(live)}")
    print(f"wrote {out_csv}")
    return 0
