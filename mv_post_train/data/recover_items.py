"""Item rows for benchmarks the dump ships without verdicts.

IFEval and MBPP are scored cell-by-cell elsewhere (`stats/ifeval_score.py` via
the official checker, `stats/code_score.py` via execution). Both compute a
per-item verdict on the way to their headline number and then discard it:
`ifeval_score.summarize` collapses a per-`example_id` frame, and the execution
harness reduces `results[task_id]` to pass@1. That discard is the only reason
neither benchmark can take part in a paired Phase 1 -> OT30K test.

This module keeps the per-item verdicts and writes them in the canonical item
schema, so `stats/phase2_transition.py` can pair them like any other benchmark.

Unlike gsm8k (see `parse_phase1_dump.py`), *neither* end exists at item level
anywhere else, so both Phase 1 and Phase 2 stages are emitted here.

Cells are gated on `checkpoint_guard`: the dump's `stage` label is per-task
unreliable, and for MixtureVitae `base` the IFEval and MBPP responses come from
a Superior-Reasoning SFT checkpoint rather than the base model. Pairing those
would silently compare the wrong checkpoint, so they are dropped with a note
rather than scored.

Writes `recovered_items.parquet`, merging per task: re-running one benchmark
leaves the other's rows intact.
"""
from __future__ import annotations

from pathlib import Path

import pandas as pd

from .. import paths
from .checkpoint_guard import collect, flag, inconsistent_cells
from .io_logs import ItemRow
from .parse_phase2 import FAMILY

# Phase 1 and Phase 2 stages together, keyed as the dump spells them.
STAGE = {
    "base":      ("base", ""),
    "SFT-Tulu3": ("sft",  "-SFT-Tulu3-decontaminated"),
    "DPO-Tulu3": ("dpo",  "-DPO-Tulu3-decontaminated"),
    "OT3":       ("ot30k_on_base", ""),
    "OT3+SFT":   ("ot30k_on_sft",  "-SFT-Tulu3-decontaminated"),
    "OT3+DPO":   ("ot30k_on_dpo",  "-DPO-Tulu3-decontaminated"),
}

_PHASE2 = {"OT3", "OT3+SFT", "OT3+DPO"}


def _model_name(base_model: str, stage: str) -> str:
    _, root = FAMILY[base_model]
    _, suffix = STAGE[stage]
    if stage in _PHASE2:
        return (f"open-sci__sft__ot30k_"
                f"{root.removeprefix('ali-elganzory__')}{suffix}")
    return f"{root}{suffix}"


def to_item_rows(verdicts: pd.DataFrame, task: str,
                 excluded: set[tuple[str, str, str]] | None = None,
                 dump_task: str | None = None) -> list[ItemRow]:
    """Canonical item rows from a per-item verdict frame.

    `verdicts` needs `base_model`, `stage`, `item_index`, `attempt_index` and a
    `score` in [0, 1]. `dump_task` is the task's spelling in the dump when it
    differs from the canonical `task` (it does not, for IFEval or MBPP).
    """
    excluded = excluded or set()
    key_task = dump_task or task
    rows: list[ItemRow] = []
    for _, r in verdicts.iterrows():
        bm, st = r.base_model, r.stage
        if bm not in FAMILY or st not in STAGE:
            continue
        if (bm, st, key_task) in excluded:
            continue
        family, _ = FAMILY[bm]
        phase, _ = STAGE[st]
        rows.append(ItemRow(
            model=_model_name(bm, st),
            family=family,
            phase=phase,
            context_tier="16k",
            tokens_b=300.0,
            size_b=1.7,
            task=task,
            item_index=int(r.item_index),
            attempt_index=int(r.attempt_index),
            score=float(r.score),
        ))
    return rows


def merge(data_dir: Path, task: str, rows: list[ItemRow],
          keep_models: set[str] | None = None) -> pd.DataFrame:
    """Replace `task`'s rows in recovered_items.parquet, keep every other task.

    Each benchmark costs a separate expensive pass (execution, in MBPP's case),
    so re-running one must not blank the other.

    `keep_models` names models whose existing rows this call is *not*
    authoritative for -- the cells a resumed run skipped because they were
    already scored. Without it, resuming would silently delete every item row
    the interrupted run had already produced, which is the opposite of what
    resuming is for. Models absent from both `rows` and `keep_models` are
    dropped, so a cell the checkpoint guard now rejects does not survive as
    stale data from an earlier run.
    """
    data_dir = Path(data_dir)
    out = data_dir / paths.RECOVERED_ITEMS
    new = pd.DataFrame([r.model_dump() for r in rows])
    if out.exists():
        prior = pd.read_parquet(out)
        keep = prior.task != task
        if keep_models:
            keep |= (prior.task == task) & prior.model.isin(keep_models)
        prior = prior[keep]
        new = pd.concat([prior, new], ignore_index=True) if not new.empty else prior
    data_dir.mkdir(parents=True, exist_ok=True)
    new.to_parquet(out, index=False)
    return new


def excluded_cells(dump: Path) -> set[tuple[str, str, str]]:
    """Cells whose checkpoint contradicts their stage label."""
    return inconsistent_cells(flag(collect(Path(dump))))


def report(rows: list[ItemRow], task: str,
           excluded: set[tuple[str, str, str]]) -> None:
    dropped = sorted(c for c in excluded if c[2] == task)
    if dropped:
        print(f"\ndropped {len(dropped)} {task} cell(s) whose checkpoint "
              f"contradicts the stage label:")
        for bm, st, _ in dropped:
            print(f"    {bm} / {st}")
    if not rows:
        print(f"no {task} item rows emitted")
        return
    cov = (pd.DataFrame([r.model_dump() for r in rows])
             .groupby(["family", "phase"]).item_index.nunique()
             .unstack("phase", fill_value=0))
    print(f"\n{task} items per (family, phase):")
    print(cov.to_string())
