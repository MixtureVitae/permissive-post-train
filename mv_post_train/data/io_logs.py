"""File-selection and per-task adapters: lm-eval-harness JSON -> tidy rows.

Each model directory contains many results_*.json (one task per file, often
with duplicates and small failed-run stubs). We pick the best file per
(model, task) and convert each to (item rows) and/or (aggregate row).

Tasks that emit per-item correctness:
  AIME24, AIME25, AMC23, MATH500, GPQADiamond, JEEBench, LiveCodeBench

Tasks that emit aggregates only (no per-item data in this dataset):
  HumanEval, MBPP, IFEval, gsm8k

A few directories also contain "longsft_16k_eval" or similar files; those
have multiple tasks per file (typical lm-eval format with mmlu/etc.). They
are not present in this dataset (verified) — guard for it anyway.
"""
from __future__ import annotations

import json
import re
import sys
from pathlib import Path
from typing import Iterable

from pydantic import BaseModel, ConfigDict, ValidationError

from .benchmarks import (
    AGG_TASKS,
    FileSelectionRule,
    PER_ITEM_TASKS,
    SCORE_METRICS,
    SPECS,
)
from .conventions import NOT_APPLICABLE, active_convention, stamp_for

# PER_ITEM_TASKS, AGG_TASKS, SCORE_METRICS, and the per-task SPECS dict are
# owned by data/benchmarks.py — single source of truth. Re-exported here
# for backwards-compatibility with existing call sites.

# Task-key fixups (none currently — every key is consistent in this dataset)
TASK_RENAME: dict[str, str] = {}


# ---------------------------------------------------------------------------
# Output row schemas (pydantic models). Both emit_* functions yield instances
# of these classes; parse_phase1.main calls .model_dump() to build the parquet
# DataFrame. The model_config=frozen prevents accidental mutation downstream.

class _BaseModelRow(BaseModel):
    """Columns shared by every emitted row — identifies the (model, task) cell."""
    model_config = ConfigDict(frozen=True)
    model: str
    family: str
    phase: str
    context_tier: str
    tokens_b: float | None  # None for Baguettotron / ablation-model-fineweb-edu
    size_b: float
    task: str


class ItemRow(_BaseModelRow):
    """One per (model, task, item_index, attempt_index) — a single attempt
    score on a single benchmark item. Emitted by `emit_item_rows`.
    Persisted to items_long.parquet."""
    item_index: int
    attempt_index: int
    score: float


class AggregateRow(_BaseModelRow):
    """One per (model, task) for aggregate-only benchmarks (HumanEval, MBPP,
    IFEval, gsm8k). Emitted by `emit_aggregate_row`. Persisted to
    aggregates.parquet.

    `accuracy` is combined-counts (k_total / n_total) — used by the stat-sig
    pipeline for Newcombe diff CIs. `accuracy_mean` is mean-of-fractions —
    used by the paper-table renderer. `components` is a JSON-serialised dict
    of raw metric values for downstream re-pooling."""
    n: int
    n_correct: int
    accuracy: float
    accuracy_mean: float
    components: str   # JSON-encoded {metric_field: float}
    source_metric: str
    # Which HumanEval/IFEval definition produced this row: "submission",
    # "arxiv", or "n/a" for tasks defined identically under both. Defaulted so
    # rows built in tests stay constructible; parsers always set it explicitly.
    # See data/conventions.py.
    metric_convention: str = NOT_APPLICABLE


def _warn(msg: str) -> None:
    """One-line stderr warning. Centralised so test fixtures can capture."""
    print(f"WARNING io_logs: {msg}", file=sys.stderr)


RESULTS_FILE_PATTERN: str = "results_*.json"
RESULTS_TS_REGEX:     str = r"results_(\d{4}-\d{2}-\d{2}T\d{2}-\d{2}-\d{2})"

# lm-eval-harness language-suite layout: one file per task named
# `<10char-hash>_<timestamp>.json` (no `results_` prefix). Used by the
# language-eval logs zips; passed in by `data/language_samples.py`.
LANG_FILE_PATTERN: str = "*.json"
LANG_TS_REGEX:     str = r"_(\d{4}-\d{2}-\d{2}T\d{2}-\d{2}-\d{2}\.\d+)\.json$"


def _ts_in_name(p: Path, regex: str = RESULTS_TS_REGEX) -> str:
    m = re.search(regex, p.name)
    return m.group(1) if m else ""


def _is_valid_record(task: str, payload: dict, *,
                      source: str | None = None) -> tuple[bool, int, int]:
    """Returns (is_valid, n_items, n_aggregate). n_items=0 for agg-only.

    Validity is determined by attempting to construct the task's
    `payload_model` (a pydantic BaseModel subclass declared in
    data/benchmarks.SPECS). A ValidationError on a known task is surfaced as
    a stderr warning before returning False — silently dropping a malformed
    HumanEval payload was a real data-quality footgun. Unknown tasks (not in
    SPECS) stay silent: the harness may emit benchmarks we don't track and
    that's expected.

    `source` is an optional file path used in the warning message so the user
    can grep the offending record.
    """
    if not isinstance(payload, dict):
        return False, 0, 0
    spec = SPECS.get(task)
    if spec is None:
        # Unknown task — silently ignored (the harness emits more benchmarks
        # than we declare in SPECS, and that's not an error).
        return False, 0, 0
    try:
        validated = spec.payload_model.model_validate(payload)
    except ValidationError as exc:
        # Known task with a bad payload — surface, don't swallow.
        where = f" in {source}" if source else ""
        _warn(f"{task}{where} rejected by payload schema: "
              f"{exc.error_count()} validation error(s) "
              f"(first: {exc.errors()[0].get('msg', 'unknown')})")
        return False, 0, 0
    if spec.kind == "per_item":
        return True, len(validated.examples), 0
    if spec.file_selection == FileSelectionRule.BY_MAX_METRIC:
        rank = int(float(payload[spec.selection_metric]) * 1_000_000)
    elif spec.file_selection == FileSelectionRule.BY_NUM_EXAMPLES:
        rank = int(payload.get("num_examples", 0) or 0)
    else:  # BY_TIMESTAMP_ONLY
        rank = 0
    return True, 0, rank


def select_results_files(
    model_dir: Path,
    *,
    pattern: str = RESULTS_FILE_PATTERN,
    ts_regex: str = RESULTS_TS_REGEX,
    latest_run: bool = False,
) -> dict[str, tuple[Path, dict]]:
    """For each task seen in any matching JSON under model_dir, pick the best file.

    Returns dict[task_name] = (file_path, results_dict_for_that_task).

    Default `pattern` matches the Phase-1 reasoning logs (`results_*.json`).
    Language-eval logs use `<hash>_<timestamp>.json` instead — callers in
    `data/language_samples.py` pass `pattern=LANG_FILE_PATTERN` + `ts_regex=LANG_TS_REGEX`.

    Files that can't be opened or parsed as JSON, or whose top-level
    `results` is not a dict, are surfaced as stderr warnings before being
    skipped. Malformed payloads for known tasks are also surfaced (inside
    `_is_valid_record`). Silent drops were a real data-quality footgun.

    HumanEval's rule keeps the run with the highest `python_pass@1`, the old
    paper convention. With `latest_run`, it keeps the latest run instead, like
    every other task, so no cell is the best of several runs.
    """
    chosen: dict[str, tuple[Path, dict]] = {}
    for task, candidates in results_runs(model_dir, pattern=pattern, ts_regex=ts_regex,
                                         latest_run=latest_run).items():
        # Sort by (n_items desc, n_aggregate desc, latest timestamp desc)
        candidates.sort(key=lambda t: (t[2], t[3], t[4]), reverse=True)
        f, payload, *_ = candidates[0]
        chosen[task] = (f, payload)
    return chosen


def results_runs(
    model_dir: Path,
    *,
    pattern: str = RESULTS_FILE_PATTERN,
    ts_regex: str = RESULTS_TS_REGEX,
    latest_run: bool = False,
) -> dict[str, list[tuple[Path, dict, int, int, str]]]:
    """Every valid run of every task under model_dir, as
    (file, payload, n_items, n_aggregate, timestamp). `n_items` is the number of
    per-item records, and `n_aggregate` the rank `_is_valid_record` gives an
    aggregate-only task. With `latest_run`, HumanEval's rank is 0 instead of its
    `python_pass@1`, so for every task a higher rank means a more complete run."""
    by_task: dict[str, list[tuple[Path, dict, int, int, str]]] = {}
    for f in sorted(model_dir.glob(pattern)):
        try:
            with f.open() as fh:
                data = json.load(fh)
        except json.JSONDecodeError as exc:
            _warn(f"{f}: corrupted JSON ({exc.msg} at line {exc.lineno})")
            continue
        except OSError as exc:
            _warn(f"{f}: cannot read ({exc})")
            continue
        results = data.get("results")
        if not isinstance(results, dict):
            _warn(f"{f}: top-level `results` is missing or not a dict; skipping")
            continue
        for task_key, payload in results.items():
            task = TASK_RENAME.get(task_key, task_key)
            ok, n_items, n_agg = _is_valid_record(task, payload, source=str(f))
            if not ok:
                continue
            if latest_run and SPECS[task].file_selection == FileSelectionRule.BY_MAX_METRIC:
                n_agg = 0
            ts = _ts_in_name(f, ts_regex)
            by_task.setdefault(task, []).append((f, payload, n_items, n_agg, ts))
    return by_task


def emit_item_rows(model: str, family_meta: tuple, task: str, payload: dict) -> Iterable[ItemRow]:
    """Yield one ItemRow per (item_index, attempt_index).

    Each example dict is handed to the task's `spec.scorer`, which
    validates it against its declared `Example` pydantic schema and then
    computes the per-attempt scores. A bad example record (one that fails
    schema validation) silently contributes zero rows — the harness
    occasionally produces partial records.
    """
    family, phase, context_tier, tokens, size = family_meta
    scorer = SPECS[task].scorer
    if scorer is None:
        return
    examples = payload.get("examples", [])
    for i, ex in enumerate(examples):
        scores = scorer.score_raw(ex)
        for j, s in enumerate(scores):
            yield ItemRow(
                model=model, family=family, phase=phase,
                context_tier=context_tier, tokens_b=tokens, size_b=size,
                task=task, item_index=i, attempt_index=j, score=float(s),
            )


def emit_aggregate_row(model: str, family_meta: tuple, task: str, payload: dict, *,
                       components: tuple | None = None,
                       source_metric: str | None = None) -> Iterable[AggregateRow]:
    """Yield one AggregateRow per aggregate-only task. Driven by the task's
    spec in `data/benchmarks.SPECS` — no per-task code paths or inline sizes.

    Schema (declared by the AggregateRow pydantic model above):
      accuracy        — combined-counts fraction (k_total / n_total). Used by
                        render/statistics.py for Newcombe diff CIs and score
                        tests; reconstructing integer counts via
                        int(round(accuracy * n)) is exact for any n.
      accuracy_mean   — mean of the component fractions (paper's "Avg"
                        convention for HumanEval/IFEval; equal to `accuracy`
                        for single-component tasks).
      components      — JSON-serialised dict of the raw metrics that fed this
                        row, e.g. {"python_pass@1": 0.34, "sh_pass@1": 0.32}.
      source_metric   — the task's canonical source-metric label, copied
                        verbatim from spec.source_metric_label.
      metric_convention — the HumanEval/IFEval definition this row was computed
                        under ("submission" / "arxiv"), or "n/a" for tasks that
                        are convention-independent. Read back by
                        data/conventions.check_frame before any analysis.

    Sizes (164/158 for HE, 500/769 for IFE, 1319 for gsm8k, 500 for MBPP)
    are read from `spec.n` and `spec.components[*].n` — never inlined here.

    `components` and `source_metric` replace the spec's, for a task scored from
    another field of the same payload (GSM8K's flexible-extract score).
    """
    spec = SPECS.get(task)
    if spec is None or spec.kind != "aggregate":
        return

    family, phase, context_tier, tokens, size = family_meta
    # Read each component's metric value from the payload (validated upstream).
    parts = components or spec.components
    vals: dict[str, float] = {
        c.metric_field: float(payload[c.metric_field]) for c in parts
    }
    # combined-counts: sum of round(metric * n_subset) over components.
    n_correct = sum(
        int(round(vals[c.metric_field] * c.n)) for c in parts
    )
    accuracy_combined = n_correct / spec.n
    # mean-of-fractions: unweighted mean of the metric values.
    accuracy_mean = sum(vals.values()) / len(vals)

    yield AggregateRow(
        model=model, family=family, phase=phase,
        context_tier=context_tier, tokens_b=tokens, size_b=size,
        task=task, n=spec.n, n_correct=n_correct,
        accuracy=accuracy_combined, accuracy_mean=accuracy_mean,
        components=json.dumps(vals),
        source_metric=source_metric or spec.source_metric_label,
        metric_convention=stamp_for(task, active_convention()),
    )
