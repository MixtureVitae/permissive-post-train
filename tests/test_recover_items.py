"""Tests for turning recovered per-item verdicts into canonical item rows.

Two failures here are silent and expensive: emitting a cell whose checkpoint
contradicts its label (you pair the wrong model and get a plausible number),
and blanking the other benchmark's rows on a partial re-run (each MBPP pass
costs minutes of sandboxed execution).
"""
from __future__ import annotations

import pandas as pd
import pytest

from mv_post_train import paths
from mv_post_train.data import recover_items as ri
from mv_post_train.stats import code_score as cs


def _verdicts(rows):
    base = dict(base_model="MixtureVitae-1.7B", stage="SFT-Tulu3",
                item_index=0, attempt_index=0, score=1.0)
    return pd.DataFrame([{**base, **r} for r in rows])


def test_both_phases_are_emitted():
    """Unlike gsm8k, neither end of IFEval exists at item level elsewhere."""
    rows = ri.to_item_rows(_verdicts([
        {"stage": "base"}, {"stage": "SFT-Tulu3"}, {"stage": "DPO-Tulu3"},
        {"stage": "OT3"}, {"stage": "OT3+SFT"}, {"stage": "OT3+DPO"},
    ]), "IFEval")
    assert [r.phase for r in rows] == [
        "base", "sft", "dpo", "ot30k_on_base", "ot30k_on_sft", "ot30k_on_dpo"]


def test_a_cell_whose_checkpoint_contradicts_its_label_is_dropped():
    """MV `base` IFEval is a Superior-Reasoning SFT checkpoint, not the base."""
    excluded = {("MixtureVitae-1.7B", "base", "IFEval")}
    rows = ri.to_item_rows(_verdicts([
        {"stage": "base"}, {"stage": "SFT-Tulu3"},
    ]), "IFEval", excluded)
    assert [r.phase for r in rows] == ["sft"]


def test_the_exclusion_is_task_scoped():
    """The same (model, stage) is fine for gsm8k and wrong for IFEval."""
    excluded = {("MixtureVitae-1.7B", "base", "IFEval")}
    rows = ri.to_item_rows(_verdicts([{"stage": "base"}]), "MBPP", excluded)
    assert len(rows) == 1


def test_unmapped_models_and_stages_are_skipped():
    rows = ri.to_item_rows(_verdicts([
        {"base_model": "Qwen3-1.7B-Base"}, {"stage": "SR+SFT"}, {},
    ]), "IFEval")
    assert len(rows) == 1


def test_phase2_models_get_the_ot30k_basename():
    rows = ri.to_item_rows(_verdicts([{"stage": "OT3+SFT"}]), "IFEval")
    assert rows[0].model.startswith("open-sci__sft__ot30k_")


def test_phase1_models_do_not():
    rows = ri.to_item_rows(_verdicts([{"stage": "SFT-Tulu3"}]), "IFEval")
    assert "ot30k" not in rows[0].model


def test_scores_pass_through_as_fractions():
    rows = ri.to_item_rows(_verdicts([{"score": 0.5}]), "MBPP")
    assert rows[0].score == pytest.approx(0.5)


# --- merge semantics --------------------------------------------------------

def test_merging_one_task_keeps_the_others_rows(tmp_path):
    ri.merge(tmp_path, "IFEval", ri.to_item_rows(_verdicts([{}]), "IFEval"))
    ri.merge(tmp_path, "MBPP", ri.to_item_rows(_verdicts([{}]), "MBPP"))
    out = pd.read_parquet(tmp_path / paths.RECOVERED_ITEMS)
    assert sorted(out.task.unique()) == ["IFEval", "MBPP"]


def test_re_merging_a_task_replaces_rather_than_appends(tmp_path):
    ri.merge(tmp_path, "IFEval", ri.to_item_rows(
        _verdicts([{"item_index": i} for i in range(5)]), "IFEval"))
    ri.merge(tmp_path, "IFEval", ri.to_item_rows(
        _verdicts([{"item_index": i} for i in range(2)]), "IFEval"))
    out = pd.read_parquet(tmp_path / paths.RECOVERED_ITEMS)
    assert len(out) == 2


# --- the code-score side ----------------------------------------------------

@pytest.mark.parametrize("task_id,expected", [
    ("Python/0", 0), ("Python/163", 163), ("11", 11), (11, 11),
    ("nonsense", None), ("Python/abc", None),
])
def test_problem_number_extraction(task_id, expected):
    assert cs._problem_number(task_id) == expected


def test_an_unscorable_cell_contributes_no_items(tmp_path, monkeypatch):
    """HumanEval's cells are all `no_extractable_code`; their per-problem
    verdicts are the same fiction as their pass@1."""
    monkeypatch.setattr(cs, "collect", lambda dump, task: {
        ("MixtureVitae-1.7B", "SFT-Tulu3"): [{"example_id": "x", "output": ""}]})

    def fake_score_cell(task, rows, problems, tmp, timeout, n_workers,
                        detail_out=None):
        if detail_out is not None:
            detail_out["Python/0"] = 1.0     # computed, but the cell is bad
        return None, 164, 164, "no_extractable_code"

    monkeypatch.setattr(cs, "score_cell", fake_score_cell)
    dump = tmp_path / "dump.parquet"
    dump.write_bytes(b"")
    monkeypatch.setattr("mv_post_train.data.recover_items.excluded_cells",
                        lambda d: set())

    assert cs.main(dump, tmp_path, tasks=("HumanEval",), data_dir=tmp_path) == 0
    assert not (tmp_path / paths.RECOVERED_ITEMS).exists()


# --- resume / checkpoint semantics -----------------------------------------

def _rows(models, task="MBPP", n=3):
    """Item rows for several models, bypassing the base_model/stage mapping."""
    from mv_post_train.data.io_logs import ItemRow
    return [ItemRow(model=m, family="MV", phase="sft", context_tier="16k",
                    tokens_b=300.0, size_b=1.7, task=task, item_index=i,
                    attempt_index=0, score=1.0)
            for m in models for i in range(n)]


def test_a_resumed_merge_keeps_the_skipped_cells_rows(tmp_path):
    """Without keep_models, resuming deletes what the interrupted run made."""
    ri.merge(tmp_path, "MBPP", _rows(["done-a", "done-b"]))
    ri.merge(tmp_path, "MBPP", _rows(["fresh"]), keep_models={"done-a", "done-b"})
    out = pd.read_parquet(tmp_path / paths.RECOVERED_ITEMS)
    assert set(out.model) == {"done-a", "done-b", "fresh"}


def test_a_full_rerun_drops_models_no_longer_emitted(tmp_path):
    """A cell the checkpoint guard now rejects must not survive as stale data."""
    ri.merge(tmp_path, "MBPP", _rows(["keep", "now-excluded"]))
    ri.merge(tmp_path, "MBPP", _rows(["keep"]))
    out = pd.read_parquet(tmp_path / paths.RECOVERED_ITEMS)
    assert set(out.model) == {"keep"}


def test_keep_models_does_not_leak_across_tasks(tmp_path):
    ri.merge(tmp_path, "IFEval", _rows(["m"], task="IFEval"))
    ri.merge(tmp_path, "MBPP", _rows(["m2"]), keep_models={"m"})
    out = pd.read_parquet(tmp_path / paths.RECOVERED_ITEMS)
    assert set(out[out.task == "IFEval"].model) == {"m"}
    assert set(out[out.task == "MBPP"].model) == {"m2"}


def test_resume_skips_scored_cells_but_retries_errors(tmp_path, monkeypatch):
    """An errored cell is why you are resuming; it must not count as done."""
    pd.DataFrame([
        {"base_model": "MixtureVitae-1.7B", "stage": "SFT-Tulu3", "task": "MBPP",
         "n_rows": 500, "n_problems_matched": 500, "n_problems_expected": 500,
         "pass@1": 40.6, "table1": None, "delta": None, "status": "scored"},
        {"base_model": "Comma0.1-1.7B", "stage": "SFT-Tulu3", "task": "MBPP",
         "n_rows": 500, "n_problems_matched": -1, "n_problems_expected": 500,
         "pass@1": None, "table1": None, "delta": None, "status": "error"},
    ]).to_csv(tmp_path / cs.RESCORED_CSV, index=False)

    scored = []
    monkeypatch.setattr(cs, "collect", lambda dump, task: {
        ("MixtureVitae-1.7B", "SFT-Tulu3"): [], ("Comma0.1-1.7B", "SFT-Tulu3"): []})

    def fake(task, rows, problems, tmp, timeout, n_workers, detail_out=None):
        scored.append(task)
        return 10.0, 500, 500, "scored"

    monkeypatch.setattr(cs, "score_cell", fake)
    dump = tmp_path / "dump.parquet"
    dump.write_bytes(b"")

    assert cs.main(dump, tmp_path, tasks=("MBPP",), resume=True) == 0
    assert len(scored) == 1          # only the errored cell was recomputed
    out = pd.read_csv(tmp_path / cs.RESCORED_CSV)
    assert len(out) == 2
    assert set(out.status) == {"scored"}


def test_the_csv_is_written_after_every_cell(tmp_path, monkeypatch):
    """A crash at the last cell must not discard the earlier ones."""
    seen = []
    monkeypatch.setattr(cs, "collect", lambda dump, task: {
        ("MixtureVitae-1.7B", "SFT-Tulu3"): [],
        ("Comma0.1-1.7B", "SFT-Tulu3"): [],
        ("NemotronHQ-1.7B", "SFT-Tulu3"): []})

    def fake(task, rows, problems, tmp, timeout, n_workers, detail_out=None):
        # Record how many rows were already durable when this cell started.
        p = tmp_path / cs.RESCORED_CSV
        seen.append(len(pd.read_csv(p)) if p.exists() else 0)
        return 10.0, 500, 500, "scored"

    monkeypatch.setattr(cs, "score_cell", fake)
    dump = tmp_path / "dump.parquet"
    dump.write_bytes(b"")

    assert cs.main(dump, tmp_path, tasks=("MBPP",)) == 0
    assert seen == [0, 1, 2]
