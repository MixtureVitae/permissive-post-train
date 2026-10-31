"""Tests for the Phase 1 gsm8k recovery from the HF dump.

The failure this guards against is silent in both directions: too few tasks and
gsm8k drops out of the transition table (the original bug), too many and items
get double-counted when the frame is concatenated with items_long.parquet.
"""
from __future__ import annotations

import pandas as pd
import pytest

from mv_post_train import paths
from mv_post_train.data import parse_phase1_dump as p1d


def _dump(tmp_path, rows: list[dict]):
    base = dict(base_model="MixtureVitae-1.7B", stage="SFT-Tulu3", task="GSM8K",
                example_idx=0, run_idx=0, is_correct=True)
    p = tmp_path / "eval_responses.parquet"
    pd.DataFrame([{**base, **r} for r in rows]).to_parquet(p)
    return p


def test_only_tasks_missing_from_items_long_are_emitted():
    """Emitting a task logs.zip already has would double-count on concat."""
    assert p1d.SCORED_TASKS == ["GSM8K"]


def test_stage_map_excludes_ablation_stages():
    """AR / NC / SR checkpoints have no OT30K descendant to pair with."""
    assert set(p1d.STAGE) == {"base", "SFT-Tulu3", "DPO-Tulu3"}
    for bad in ("SR+SFT", "AR", "NC+DPO", "OT3+SFT"):
        assert bad not in p1d.STAGE


def test_phase_labels_match_the_canonical_names():
    assert [p1d.STAGE[s][0] for s in ("base", "SFT-Tulu3", "DPO-Tulu3")] == \
        ["base", "sft", "dpo"]


def test_rows_without_a_verdict_are_skipped_not_scored_zero(tmp_path):
    """A null is_correct is an absence. Scoring it 0 would invent failures."""
    rows = [{"example_idx": i, "is_correct": None if i < 4 else True}
            for i in range(10)]
    out = p1d.Phase1DumpParser(_dump(tmp_path, rows)).parse()
    assert len(out.item_rows) == 6
    assert all(r.score == 1.0 for r in out.item_rows)


def test_scores_are_zero_or_one(tmp_path):
    rows = [{"example_idx": i, "is_correct": bool(i % 2)} for i in range(6)]
    out = p1d.Phase1DumpParser(_dump(tmp_path, rows)).parse()
    assert sorted(r.score for r in out.item_rows) == [0.0, 0.0, 0.0, 1.0, 1.0, 1.0]


def test_task_is_renamed_to_the_canonical_lowercase_key(tmp_path):
    out = p1d.Phase1DumpParser(_dump(tmp_path, [{}])).parse()
    assert out.item_rows[0].task == "gsm8k"


def test_unmapped_stages_and_models_are_filtered_out(tmp_path):
    rows = [
        {"example_idx": 0, "stage": "SR+SFT"},
        {"example_idx": 1, "base_model": "Qwen3-1.7B-Base"},
        {"example_idx": 2, "task": "MATH500"},
        {"example_idx": 3},
    ]
    out = p1d.Phase1DumpParser(_dump(tmp_path, rows)).parse()
    assert len(out.item_rows) == 1
    assert out.item_rows[0].item_index == 3


def test_families_map_to_canonical_names(tmp_path):
    rows = [{"example_idx": 0, "base_model": "Comma0.1-1.7B"},
            {"example_idx": 1, "base_model": "FineWebEdu-1.7B"}]
    out = p1d.Phase1DumpParser(_dump(tmp_path, rows)).parse()
    assert {r.family for r in out.item_rows} == {"Comma", "FineWeb-Edu"}


# --- the cross-check against logs.zip ---------------------------------------

def test_validate_returns_empty_rather_than_claiming_vacuous_agreement(tmp_path):
    assert p1d.validate(tmp_path).empty


def test_validate_reports_the_signed_gap_per_cell(tmp_path):
    pd.DataFrame([
        {"model": "m", "family": "MV", "phase": "base", "context_tier": "16k",
         "tokens_b": 300.0, "size_b": 1.7, "task": "gsm8k", "item_index": i,
         "attempt_index": 0, "score": 1.0 if i < 6 else 0.0}
        for i in range(10)
    ]).to_parquet(tmp_path / paths.PHASE1_DUMP_ITEMS)
    pd.DataFrame([
        {"model": "m", "family": "MV", "phase": "base", "context_tier": "16k",
         "tokens_b": 300.0, "size_b": 1.7, "task": "gsm8k", "n": 10,
         "n_correct": 7, "accuracy": 0.70, "accuracy_mean": 0.70,
         "components": None, "source_metric": "x"}
    ]).to_parquet(tmp_path / paths.AGGREGATES)

    out = p1d.validate(tmp_path)
    assert len(out) == 1
    assert out.iloc[0].dump_acc == pytest.approx(60.0)
    assert out.iloc[0].logs_acc == pytest.approx(70.0)
    assert out.iloc[0].delta == pytest.approx(-10.0)
