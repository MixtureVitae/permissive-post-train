"""Phase 1 item rows recovered from the HF response dump.

`parse_phase1.py` sources Phase 1 from logs.zip, where gsm8k exists only as an
aggregate scalar — evalchemy writes no per-example verdicts for it. That is why
the Phase 1 -> OT30K transition silently drops gsm8k: `phase2_items.parquet`
carries 1,319 scored Phase 2 items per cell, and there is nothing on the Phase 1
side to pair them against.

The dump does carry per-example gsm8k for Phase 1 checkpoints, with a populated
`is_correct` column -- the one task in the dump where verdicts survive rather
than needing to be re-derived. This module emits those rows in the canonical
item schema so the transition machinery can pair them.

Both ends of a gsm8k transition therefore come from the dump. That is a
deliberate consequence of the project rule that Phase 2 analysis uses the HF
dump only: the alternative -- logs.zip on the Phase 1 side -- has no item-level
gsm8k to offer, so there is no same-source-per-phase option to prefer.

Validated before use: for the six (family, phase) cells where the dump and
logs.zip both carry gsm8k, per-cell accuracy agrees within 0.07-1.28 pp,
including MixtureVitae `base` (58.38 vs 58.53). That last cell matters --
`stats/released_scores.KNOWN_MISLABELLED` flags MV/base as carrying responses
from a Superior-Reasoning SFT checkpoint on IFEval and MBPP. Whatever that is,
it does not extend to gsm8k, where the cell reproduces the base checkpoint.
Re-run `mvpt parse-phase1-dump --validate` to check that again after any dump
revision.

Writes `phase1_dump_items.parquet`, schema-compatible with
`items_long.parquet`. Only tasks absent from `items_long.parquet` are emitted,
so concatenating the two can never double-count an item.
"""
from __future__ import annotations

from pathlib import Path

import pandas as pd
from tqdm import tqdm

from .. import paths
from .benchmarks import GSM8K
from .io_logs import ItemRow
from .parse_phase2 import FAMILY, _PHASE2_RAW_GSM8K
from .parser import EvalParser, ParseResult

# Only tasks that logs.zip cannot supply at item level belong here. Adding a
# task that `items_long.parquet` already carries would duplicate items when the
# two frames are concatenated.
SCORED_TASKS = [_PHASE2_RAW_GSM8K]

TASK_RENAME = {_PHASE2_RAW_GSM8K: GSM8K}

# Phase 1 stage -> (canonical phase, model_basename suffix). The mirror of
# `parse_phase2.STAGE`, which maps the OT3* stages. Deliberately excludes the
# ablation stages (AR / NC / SR and their +SFT / +DPO variants): those are
# Phase 1 checkpoints, but no OT30K descendant exists for them, so they could
# never take part in a transition.
STAGE = {
    "base":      ("base", ""),
    "SFT-Tulu3": ("sft",  "-SFT-Tulu3-decontaminated"),
    "DPO-Tulu3": ("dpo",  "-DPO-Tulu3-decontaminated"),
}

_SCORING_COLUMNS = [
    "base_model", "stage", "task", "example_idx", "run_idx", "is_correct",
]


def _model_name(base_model: str, stage: str) -> str:
    """Filesystem id for a Phase 1 row, matching what logs.zip emits."""
    _, root = FAMILY[base_model]
    _, suffix = STAGE[stage]
    return f"{root}{suffix}"


class Phase1DumpParser(EvalParser):
    parser_name = "phase1-dump"
    items_filename = paths.PHASE1_DUMP_ITEMS
    emits_aggregates = False
    audit_filename = None

    def __init__(self, hf_main: Path) -> None:
        self.hf_main = Path(hf_main)

    def parse(self) -> ParseResult:
        if not self.hf_main.exists():
            raise FileNotFoundError(f"missing dump: {self.hf_main}")

        # Same memory discipline as Phase2Parser: the dump's `model_output`
        # and `problem` columns hold full generations and dominate the file.
        df = pd.read_parquet(
            self.hf_main,
            columns=_SCORING_COLUMNS,
            filters=[
                ("task", "in", SCORED_TASKS),
                ("base_model", "in", list(FAMILY)),
                ("stage", "in", list(STAGE)),
            ],
        )
        # `filters` prunes at row-group granularity, so re-apply exactly.
        df = df[
            df.task.isin(SCORED_TASKS)
            & df.base_model.isin(FAMILY)
            & df.stage.isin(STAGE)
        ].copy()
        print(f"Phase 1 dump rows after filter: {len(df)}")

        item_rows: list[ItemRow] = []
        n_null = 0
        for (base_model, stage, task), g in tqdm(
            df.groupby(["base_model", "stage", "task"]), desc=self.parser_name
        ):
            family, _ = FAMILY[base_model]
            phase, _ = STAGE[stage]
            model = _model_name(base_model, stage)
            canonical_task = TASK_RENAME.get(task, task)
            for _, r in g.iterrows():
                # gsm8k is the one dump task with verdicts already computed.
                # A null here means the cell was never scored upstream; it is
                # an absence, not a zero, and must not become a failed item.
                if r.is_correct is None or pd.isna(r.is_correct):
                    n_null += 1
                    continue
                item_rows.append(ItemRow(
                    model=model,
                    family=family,
                    phase=phase,
                    context_tier="16k",
                    tokens_b=300.0,
                    size_b=1.7,
                    task=canonical_task,
                    item_index=int(r.example_idx),
                    attempt_index=int(r.run_idx),
                    score=float(int(bool(r.is_correct))),
                ))
        if n_null:
            print(f"skipped {n_null} rows with no verdict")

        if item_rows:
            cov = (pd.DataFrame([r.model_dump() for r in item_rows])
                     .groupby(["family", "phase", "task"]).size()
                     .unstack("task", fill_value=0))
            print("coverage (rows per (family, phase, task)):")
            print(cov.to_string())

        return ParseResult(item_rows=tuple(item_rows))


def validate(data_dir: Path) -> pd.DataFrame:
    """Per-cell accuracy from the dump against logs.zip's gsm8k aggregates.

    The two are independent recordings of the same evaluation, so a cell that
    disagrees by more than a rounding difference means one of them is not the
    checkpoint its label claims. Returns an empty frame if either side is
    absent rather than reporting vacuous agreement.
    """
    data_dir = Path(data_dir)
    items = data_dir / paths.PHASE1_DUMP_ITEMS
    aggs = data_dir / paths.AGGREGATES
    if not items.exists() or not aggs.exists():
        return pd.DataFrame()

    dump = pd.read_parquet(items)
    dump = dump[dump.task == GSM8K]
    if dump.empty:
        return pd.DataFrame()
    got = (dump.groupby(["family", "phase"]).score.mean().mul(100).round(2)
              .rename("dump_acc").reset_index())

    ag = pd.read_parquet(aggs)
    ag = ag[(ag.task == GSM8K) & (ag.context_tier == "16k")]
    ag = (ag.assign(logs_acc=(ag.accuracy * 100).round(2))
            [["family", "phase", "logs_acc"]])

    out = got.merge(ag, on=["family", "phase"], how="inner")
    out["delta"] = (out.dump_acc - out.logs_acc).round(2)
    return out.sort_values("delta")


def main(hf_main: Path, data_dir: Path, do_validate: bool = True) -> int:
    if not Path(hf_main).exists():
        print(f"missing source: {hf_main}")
        return 1
    rc = Phase1DumpParser(hf_main).write_to_data_dir(data_dir)
    if rc != 0 or not do_validate:
        return rc

    cmp = validate(data_dir)
    if cmp.empty:
        print("\nno overlapping cells to validate against logs.zip")
        return 0
    print("\n=== dump vs logs.zip, same cells ===")
    print(cmp.to_string(index=False))
    worst = cmp.delta.abs().max()
    print(f"\nlargest disagreement: {worst:.2f} pp")
    if worst > 2.0:
        print("WARNING: a cell disagrees by more than 2 pp — the dump and the "
              "logs may not be the same checkpoint. Investigate before pairing.")
    return 0
