"""Tests for the analyze.py helpers that do not require full pipeline data.

We test:
  - per_item_y collapses repeated attempts as expected
  - _seed_offset is deterministic
  - _collect_pair_task_cells builds paired-vs-beta cells correctly
  - _bootstrap_suite_diff signal direction is correct on a constructed case
  - run_pairwise emits the new audit columns (n_a, n_b, alignment_mismatch)
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from mv_post_train.stats import analyze
from mv_post_train.stats.analyze import (
    _bootstrap_suite_diff,
    _collect_pair_task_cells,
    _seed_offset,
    per_item_y,
    run_pairwise,
)


def _items_row(model, family, phase, ctx, task, item, attempt, score):
    return {
        "model": model, "family": family, "phase": phase,
        "context_tier": ctx, "tokens_b": 100.0, "size_b": 1.7,
        "task": task, "item_index": item, "attempt_index": attempt,
        "score": float(score),
    }


def test_per_item_y_averages_attempts():
    rows = [
        _items_row("m", "F", "sft", "16k", "T", 0, 0, 1.0),
        _items_row("m", "F", "sft", "16k", "T", 0, 1, 0.0),
        _items_row("m", "F", "sft", "16k", "T", 1, 0, 1.0),
    ]
    df = pd.DataFrame(rows)
    y = per_item_y(df)
    assert len(y) == 2
    assert sorted(y.y.tolist()) == [0.5, 1.0]


def test_seed_offset_deterministic_and_within_u32():
    a = _seed_offset("foo")
    b = _seed_offset("foo")
    c = _seed_offset("bar")
    assert a == b
    assert a != c
    assert 0 <= a < 2**32


def test_collect_pair_task_cells_paired_path():
    rows = []
    for m in ("A", "B"):
        for item in range(5):
            rows.append(_items_row(m, "F", "sft", "16k", "T", item, 0,
                                    1.0 if m == "A" else 0.0))
    items = pd.DataFrame(rows)
    p = per_item_y(items)
    rng = np.random.default_rng(0)
    cells = _collect_pair_task_cells(p, pd.DataFrame(), "A", "B", ["T"], rng,
                                     n_resamples=10)
    assert set(cells.keys()) == {"T"}
    c = cells["T"]
    assert c["kind"] == "paired"
    assert c["obs_a"] == 1.0 and c["obs_b"] == 0.0
    assert len(c["y_a"]) == 5 and len(c["y_b"]) == 5


def test_collect_pair_task_cells_beta_path():
    # No per-item data; aggregate only.
    agg = pd.DataFrame([
        {"model": "A", "family": "F", "phase": "sft", "context_tier": "16k",
         "task": "HumanEval", "n": 100, "n_correct": 80, "accuracy": 0.8,
         "source_metric": "x"},
        {"model": "B", "family": "F", "phase": "sft", "context_tier": "16k",
         "task": "HumanEval", "n": 100, "n_correct": 50, "accuracy": 0.5,
         "source_metric": "x"},
    ])
    rng = np.random.default_rng(0)
    cells = _collect_pair_task_cells(pd.DataFrame(columns=["model", "task", "item_index", "y"]),
                                     agg, "A", "B", ["HumanEval"], rng, n_resamples=200)
    c = cells["HumanEval"]
    assert c["kind"] == "beta"
    # Beta posterior means should be near k/n.
    assert abs(c["obs_a"] - 0.8) < 0.05
    assert abs(c["obs_b"] - 0.5) < 0.05
    assert len(c["draws_a"]) == 200


def test_bootstrap_suite_diff_sign_matches_input():
    # Two tasks, A always 1, B always 0 => diff distribution should be ~+1.
    cells = {
        "T1": {"kind": "paired",
               "y_a": np.ones(10), "y_b": np.zeros(10),
               "obs_a": 1.0, "obs_b": 0.0},
        "T2": {"kind": "paired",
               "y_a": np.ones(10), "y_b": np.zeros(10),
               "obs_a": 1.0, "obs_b": 0.0},
    }
    rng = np.random.default_rng(1)
    diffs = _bootstrap_suite_diff(cells, rng, n_resamples=300)
    # Both models constant => diffs all 1.0 exactly.
    assert np.all(diffs == 1.0)


def _meta_row(model, family, phase, ctx):
    return {"model": model, "family": family, "phase": phase, "context_tier": ctx}


def test_run_pairwise_emits_audit_columns():
    # Two families, same (phase, ctx), one per-item task. Both have 10 items.
    rows = []
    for m, fam, val in (("a1", "MV", 1.0), ("b1", "Comma", 0.0)):
        for i in range(10):
            rows.append(_items_row(m, fam, "sft", "16k", "AIME24", i, 0, val))
    items = pd.DataFrame(rows)
    per_item = per_item_y(items)
    family_meta = pd.DataFrame([
        _meta_row("a1", "MV", "sft", "16k"),
        _meta_row("b1", "Comma", "sft", "16k"),
    ])
    pf, audit = run_pairwise(per_item, pd.DataFrame(), family_meta,
                              ["MV", "Comma"], [("sft", "16k")])
    assert not pf.empty
    assert {"n_a", "n_b", "alignment_mismatch"}.issubset(pf.columns)
    assert pf.iloc[0]["n_a"] == 10
    assert pf.iloc[0]["n_b"] == 10
    assert pf.iloc[0]["alignment_mismatch"] is False or pf.iloc[0]["alignment_mismatch"] == False
    # Audit should reflect the same alignment.
    assert {"n_a", "n_b", "n_paired", "alignment_mismatch"}.issubset(audit.columns)
    assert audit.iloc[0]["n_paired"] == 10


def test_run_pairwise_flags_mismatched_pair():
    # A has 10 items, B has only 8 items (item_index 2..9). Pair = 8.
    rows = []
    for i in range(10):
        rows.append(_items_row("a1", "MV", "sft", "16k", "AIME24", i, 0, 1.0))
    for i in range(2, 10):
        rows.append(_items_row("b1", "Comma", "sft", "16k", "AIME24", i, 0, 0.0))
    items = pd.DataFrame(rows)
    per_item = per_item_y(items)
    family_meta = pd.DataFrame([
        _meta_row("a1", "MV", "sft", "16k"),
        _meta_row("b1", "Comma", "sft", "16k"),
    ])
    pf, _ = run_pairwise(per_item, pd.DataFrame(), family_meta,
                         ["MV", "Comma"], [("sft", "16k")])
    r = pf.iloc[0]
    assert r["n_a"] == 10 and r["n_b"] == 8
    assert bool(r["alignment_mismatch"]) is True
