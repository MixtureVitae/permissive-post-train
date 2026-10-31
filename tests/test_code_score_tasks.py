"""The --tasks filter must not destroy the other benchmark's rows.

Each cell costs minutes of sandboxed execution, so a partial re-run silently
blanking the rows it did not compute would be an expensive mistake to notice
late.
"""
from __future__ import annotations

import pandas as pd
import pytest

from mv_post_train.stats import code_score as cs


@pytest.fixture
def prior_csv(tmp_path):
    pd.DataFrame([
        {"base_model": "M", "stage": "base", "task": "MBPP", "n_rows": 500,
         "n_problems_matched": 500, "n_problems_expected": 500,
         "pass@1": 40.6, "table1": 40.8, "delta": -0.2, "status": "scored"},
        {"base_model": "M", "stage": "base", "task": "HumanEval", "n_rows": 164,
         "n_problems_matched": 6, "n_problems_expected": 164,
         "pass@1": None, "table1": 48.2, "delta": None,
         "status": "insufficient_coverage"},
    ]).to_csv(tmp_path / cs.RESCORED_CSV, index=False)
    return tmp_path


def test_scoring_one_task_keeps_the_others_rows(prior_csv, monkeypatch):
    """Re-scoring HumanEval must leave the existing MBPP row intact."""
    monkeypatch.setattr(cs, "collect", lambda dump, task: {})
    dump = prior_csv / "dump.parquet"
    dump.write_bytes(b"")

    assert cs.main(dump, prior_csv, tasks=("HumanEval",)) == 0

    out = pd.read_csv(prior_csv / cs.RESCORED_CSV)
    mbpp = out[out.task == "MBPP"]
    assert len(mbpp) == 1
    assert mbpp.iloc[0]["pass@1"] == pytest.approx(40.6)
    # The stale HumanEval row is replaced, not duplicated.
    assert (out.task == "HumanEval").sum() == 0


def test_scoring_both_tasks_replaces_the_file(prior_csv, monkeypatch):
    monkeypatch.setattr(cs, "collect", lambda dump, task: {})
    dump = prior_csv / "dump.parquet"
    dump.write_bytes(b"")

    assert cs.main(dump, prior_csv, tasks=("HumanEval", "MBPP")) == 0
    assert pd.read_csv(prior_csv / cs.RESCORED_CSV).empty


def test_a_missing_prior_file_is_not_an_error(tmp_path, monkeypatch):
    monkeypatch.setattr(cs, "collect", lambda dump, task: {})
    dump = tmp_path / "dump.parquet"
    dump.write_bytes(b"")
    assert cs.main(dump, tmp_path, tasks=("HumanEval",)) == 0
