"""Logs.zip parser: walks per-checkpoint log directories, selects the best
`results_*.json` per (model, task), and emits the tidy parquets the rest of
the pipeline consumes.

**Phase 1 (and reference checkpoints) only.** Model directories whose
registry phase is in `PHASE2_PHASES` (OpenThoughts3-30K post-training) are
skipped — Phase 2 cells come from the Phase 2 HF parquet via
`data/parse_phase2.py`, not from logs.zip. Clean source separation;
no merge step.

`Phase1Parser` is an `EvalParser` subclass (`data/parser.py`); the
module-level `main` is a thin CLI wrapper that constructs the parser and
calls `write_to_data_dir`.
"""
from __future__ import annotations

import json
from pathlib import Path

from tqdm import tqdm

from .. import paths
from .benchmarks import GSM8K, REASONING_AGG_TASKS, AggregateComponent
from .io_logs import (
    PER_ITEM_TASKS,
    emit_aggregate_row,
    emit_item_rows,
    results_runs,
    select_results_files,
)
from .run_rule import broken, run_value
from .models import PHASE2_PHASES, lookup
from .parser import EvalParser, ParseResult

#: GSM8K scored with the flexible-extract parser, as the older setup's tables
#: (Tables 7 and 9) score it. The default is strict matching.
GSM8K_FLEXIBLE = "exact_match,flexible-extract"
GSM8K_FLEXIBLE_COMPONENTS = (AggregateComponent(metric_field=GSM8K_FLEXIBLE, n=1319),)
#: Attempt indices of a checkpoint's k-th kept run start at k * RUN_STRIDE, so
#: per-item means average over the runs.
RUN_STRIDE = 1000


class Phase1Parser(EvalParser):
    """Per-checkpoint log-directory parser. Produces both item rows
    (per-item benchmarks) and aggregate rows (HumanEval / MBPP / IFEval /
    gsm8k) plus an audit log of which `results_*.json` was selected per
    (model, task)."""

    parser_name = "phase1"
    items_filename = paths.ITEMS_LONG
    emits_aggregates = True
    audit_filename = paths.FILE_SELECTION_LOG

    def __init__(self, logs_dir: Path, latest_run: bool = False,
                 mean_unbroken_runs: bool = False, gsm8k_flexible: bool = False) -> None:
        self.logs_dir = Path(logs_dir)
        self.latest_run = latest_run
        self.mean_unbroken_runs = mean_unbroken_runs
        self.gsm8k_flexible = gsm8k_flexible

    def _aggregate_kwargs(self, task: str) -> dict:
        if self.gsm8k_flexible and task == GSM8K:
            return dict(components=GSM8K_FLEXIBLE_COMPONENTS, source_metric=GSM8K_FLEXIBLE)
        return {}

    def _unbroken_rows(self, model: str, family_meta: tuple, d: Path):
        """(item rows, aggregate rows, audit rows) from the mean of each task's
        complete runs that are not broken (`data/run_rule.py`)."""
        items, aggs, audit = [], [], []
        for task, candidates in sorted(results_runs(d, latest_run=True).items()):
            if task not in PER_ITEM_TASKS and task not in REASONING_AGG_TASKS:
                continue
            most = max((c[2], c[3]) for c in candidates)
            complete = sorted((c for c in candidates if (c[2], c[3]) == most),
                              key=lambda c: c[4])
            flags = broken([run_value(c[1]) for c in complete])
            kept = [c for c, bad in zip(complete, flags) if not bad]
            audit.append({"model": model, "task": task,
                          "selected_file": ";".join(c[0].name for c in kept),
                          "size_bytes": sum(c[0].stat().st_size for c in kept)})
            if task in PER_ITEM_TASKS:
                for k, (_, payload, *_) in enumerate(kept):
                    items.extend(row.model_copy(update={
                        "attempt_index": row.attempt_index + k * RUN_STRIDE})
                        for row in emit_item_rows(model, family_meta, task, payload))
                continue
            rows = [r for _, payload, *_ in kept for r in emit_aggregate_row(
                model, family_meta, task, payload, **self._aggregate_kwargs(task))]
            accuracy = sum(r.accuracy for r in rows) / len(rows)
            fields = json.loads(rows[0].components)
            aggs.append(rows[0].model_copy(update={
                "accuracy": accuracy,
                "accuracy_mean": sum(r.accuracy_mean for r in rows) / len(rows),
                "n_correct": int(round(accuracy * rows[0].n)),
                "components": json.dumps({f: sum(json.loads(r.components)[f] for r in rows)
                                          / len(rows) for f in fields}),
            }))
        return items, aggs, audit

    def parse(self) -> ParseResult:
        item_rows = []
        aggregate_rows = []
        audit_rows = []

        model_dirs = [d for d in sorted(self.logs_dir.iterdir()) if d.is_dir()]
        n_phase2_skipped = 0
        for d in tqdm(model_dirs, desc=self.parser_name):
            model = d.name
            try:
                family_meta = lookup(model)
            except KeyError as e:
                print(f"WARNING unmapped: {e}")
                continue
            # Phase 2 (OT30K) lives in the Phase 2 HF parquet, not
            # logs.zip. Skip silently; the count is reported below.
            if family_meta[1] in PHASE2_PHASES:
                n_phase2_skipped += 1
                continue
            if self.mean_unbroken_runs:
                items, aggs, audit = self._unbroken_rows(model, family_meta, d)
                item_rows.extend(items)
                aggregate_rows.extend(aggs)
                audit_rows.extend(audit)
                continue
            chosen = select_results_files(d, latest_run=self.latest_run)
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
                    # Narrow to REASONING_AGG_TASKS rather than the union
                    # AGG_TASKS — language tasks share SPECS but are read
                    # from their own logs (data/language_samples.py and
                    # data/language_records.py). Without this narrowing, any stray
                    # language task key in a reasoning results_*.json would
                    # silently leak into aggregates.parquet.
                    aggregate_rows.extend(emit_aggregate_row(
                        model, family_meta, task, payload, **self._aggregate_kwargs(task)))

        if n_phase2_skipped:
            print(f"Phase 2 model dirs skipped (live in Phase 2 HF parquet): "
                  f"{n_phase2_skipped}")

        return ParseResult(
            item_rows=tuple(item_rows),
            aggregate_rows=tuple(aggregate_rows),
            audit_rows=tuple(audit_rows),
        )


def main(logs_dir: Path, data_dir: Path, art_dir: Path | None = None,
         latest_run: bool = False, mean_unbroken_runs: bool = False,
         gsm8k_flexible: bool = False) -> int:
    """CLI entry point. Thin wrapper around `Phase1Parser.write_to_data_dir`.
    `latest_run` keeps HumanEval's latest run instead of its best one.
    `mean_unbroken_runs` averages each task's complete runs that are not broken
    instead of keeping one, and `gsm8k_flexible` scores GSM8K with the
    flexible-extract parser: Table 7's rules, which Table 27 uses."""
    return Phase1Parser(logs_dir, latest_run, mean_unbroken_runs,
                        gsm8k_flexible).write_to_data_dir(data_dir, art_dir)
