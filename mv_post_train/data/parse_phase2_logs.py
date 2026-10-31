"""Phase 2 parser for `logs.zip` — the OT30K model dirs that
`parse_phase1.py` deliberately skips.

Why this exists alongside `parse_phase2.py`: the LAION HF dump is a
per-response export covering seven tasks, four of which
(HumanEval, MBPP, IFEval, LiveCodeBench) it does not carry at all. The
same checkpoints' evalchemy `results_*.json` files in `logs.zip` carry all
eleven benchmarks of the reasoning suite, so sourcing Phase 2 from the logs
gives the OT30K cells the same task coverage the Phase 1 cells already
have — which is what a paired 11-task suite-average bootstrap needs.

This module is the exact complement of `Phase1Parser`: same directory
walk, same `select_results_files` / `emit_*` helpers, same output schema —
only the phase filter is inverted (keep `PHASE2_PHASES`, skip everything
else). Outputs go to their own parquets so neither the Phase 1 pipeline
nor the HF-sourced Phase 2 pipeline changes shape.

`ot30k_on_base_alt` is included: it is a real OT30K checkpoint, and
callers that want only the three canonical cells can filter on `phase`.
"""
from __future__ import annotations

from pathlib import Path

from tqdm import tqdm

from .. import paths
from .benchmarks import REASONING_AGG_TASKS
from .io_logs import (
    PER_ITEM_TASKS,
    emit_aggregate_row,
    emit_item_rows,
    select_results_files,
)
from .models import PHASE2_PHASES, lookup
from .parser import EvalParser, ParseResult


class Phase2LogsParser(EvalParser):
    """Walks the same log root as `Phase1Parser` but keeps only the model
    dirs whose registry phase is in `PHASE2_PHASES`."""

    parser_name = "phase2-logs"
    items_filename = paths.PHASE2_LOGS_ITEMS
    emits_aggregates = True
    aggregates_filename = paths.PHASE2_LOGS_AGGREGATES
    audit_filename = paths.PHASE2_LOGS_FILE_SELECTION_LOG

    def __init__(self, logs_dir: Path) -> None:
        self.logs_dir = Path(logs_dir)

    def parse(self) -> ParseResult:
        item_rows = []
        aggregate_rows = []
        audit_rows = []

        model_dirs = [d for d in sorted(self.logs_dir.iterdir()) if d.is_dir()]
        n_phase1_skipped = 0
        for d in tqdm(model_dirs, desc=self.parser_name):
            model = d.name
            try:
                family_meta = lookup(model)
            except KeyError as e:
                print(f"WARNING unmapped: {e}")
                continue
            # Inverse of Phase1Parser's filter: Phase 1 and reference
            # checkpoints belong to items_long.parquet, not here.
            if family_meta[1] not in PHASE2_PHASES:
                n_phase1_skipped += 1
                continue
            chosen = select_results_files(d)
            for task, (path, payload) in chosen.items():
                audit_rows.append({
                    "model": model,
                    "task": task,
                    "selected_file": path.name,
                    "size_bytes": path.stat().st_size,
                })
                if task in PER_ITEM_TASKS:
                    item_rows.extend(emit_item_rows(model, family_meta, task, payload))
                elif task in REASONING_AGG_TASKS:
                    aggregate_rows.extend(emit_aggregate_row(model, family_meta, task, payload))

        print(f"non-Phase-2 model dirs skipped (live in items_long.parquet): "
              f"{n_phase1_skipped}")

        return ParseResult(
            item_rows=tuple(item_rows),
            aggregate_rows=tuple(aggregate_rows),
            audit_rows=tuple(audit_rows),
        )


def main(logs_dir: Path, data_dir: Path, art_dir: Path | None = None) -> int:
    """CLI entry point. Thin wrapper around `Phase2LogsParser.write_to_data_dir`."""
    return Phase2LogsParser(logs_dir).write_to_data_dir(data_dir, art_dir)
