"""Tests for the released-vs-recovered score comparison."""
from __future__ import annotations

import pandas as pd
import pytest

from mv_post_train.stats import released_scores as rs


def _wide(tmp_path, rows: list[dict]) -> object:
    p = tmp_path / "eval_scores_wide.csv"
    pd.DataFrame(rows).to_csv(p, index=False)
    return p


def test_load_released_melts_only_the_recoverable_columns(tmp_path):
    p = _wide(tmp_path, [{"base_model": "M", "stage": "base", "IFEval": 42.6,
                          "MBPP": 41.0, "HumanEval": 48.2,
                          "AIME24": 0.0, "Avg": 27.9}])
    out = rs.load_released(p)
    # AIME24 and Avg are not columns we recover; they must not appear.
    assert set(out.task) == {"IFEval", "MBPP", "HumanEval"}
    assert out[out.task == "IFEval"].released.iloc[0] == pytest.approx(42.6)


def test_load_released_drops_blank_cells(tmp_path):
    """The release leaves some cells empty; those are absences, not zeros."""
    p = _wide(tmp_path, [{"base_model": "M", "stage": "base", "IFEval": None,
                          "MBPP": 10.0}])
    out = rs.load_released(p)
    assert set(out.task) == {"MBPP"}


def test_compare_computes_signed_delta_and_flags_mislabelled():
    released = pd.DataFrame([
        {"base_model": "MixtureVitae-1.7B", "stage": "OT3+DPO",
         "task": "IFEval", "released": 43.0},
        {"base_model": "MixtureVitae-1.7B", "stage": "base",
         "task": "IFEval", "released": 42.6},
    ])
    recovered = pd.DataFrame([
        {"base_model": "MixtureVitae-1.7B", "stage": "OT3+DPO",
         "task": "IFEval", "recovered": 43.0},
        {"base_model": "MixtureVitae-1.7B", "stage": "base",
         "task": "IFEval", "recovered": 26.6},
    ])
    out = rs.compare(released, recovered)
    dpo = out[out.stage == "OT3+DPO"].iloc[0]
    assert dpo.delta == pytest.approx(0.0)
    assert not dpo.mislabelled
    assert out[out.stage == "base"].iloc[0].mislabelled


def test_summarize_excludes_the_mislabelled_cell():
    df = pd.DataFrame([
        {"task": "IFEval", "delta": 0.0, "mislabelled": False},
        {"task": "IFEval", "delta": 1.0, "mislabelled": False},
        {"task": "IFEval", "delta": -16.0, "mislabelled": True},
    ])
    row = rs.summarize(df).iloc[0]
    assert row.n_cells == 2
    assert row.max_abs_delta == pytest.approx(1.0)
    assert row.within_1pp == 2


def test_load_recovered_ignores_unscorable_code_cells(tmp_path):
    """Cells the execution scorer refused must not be compared as if scored."""
    pd.DataFrame([
        {"base_model": "M", "stage": "base", "task": "HumanEval",
         "pass@1": None, "status": "insufficient_coverage"},
        {"base_model": "M", "stage": "base", "task": "MBPP",
         "pass@1": 40.6, "status": "scored"},
    ]).to_csv(tmp_path / "code_rescored.csv", index=False)
    out = rs.load_recovered(tmp_path)
    assert list(out.task) == ["MBPP"]


def test_load_recovered_raises_when_nothing_has_been_scored(tmp_path):
    with pytest.raises(FileNotFoundError, match="ifeval-rescore"):
        rs.load_recovered(tmp_path)


# --- the run-to-run leg -----------------------------------------------------

def test_released_vs_table1_flags_only_rerun_cells(tmp_path, monkeypatch):
    """Cells matching Table 1 to the digit were not re-run; the rest were."""
    monkeypatch.setattr("mv_post_train.stats.ifeval_score.TABLE1_IFE",
                        {("M", "base"): 36.5, ("M", "sft"): 20.0})
    monkeypatch.setattr("mv_post_train.stats.code_score.TABLE1",
                        {("M", "base"): {"MBPP": 0.2}})
    p = _wide(tmp_path, [
        {"base_model": "M", "stage": "base", "IFEval": 30.6, "MBPP": 4.8},
        {"base_model": "M", "stage": "sft", "IFEval": 20.0},
    ])
    out = rs.released_vs_table1(p)
    by = {(r.stage, r.task): r for _, r in out.iterrows()}
    assert by[("base", "IFEval")].delta == pytest.approx(-5.9)
    assert by[("base", "IFEval")].rerun
    assert by[("base", "MBPP")].delta == pytest.approx(4.6)
    # An untouched cell must not be counted as movement.
    assert not by[("sft", "IFEval")].rerun
    assert by[("sft", "IFEval")].delta == pytest.approx(0.0)


def test_decomposition_names_both_sources():
    vs_rec = pd.DataFrame([{"delta": 0.2, "mislabelled": False},
                           {"delta": -0.4, "mislabelled": False},
                           {"delta": -25.4, "mislabelled": True}])
    vs_t1 = pd.DataFrame([{"delta": 0.0, "rerun": False},
                          {"delta": -5.9, "rerun": True}])
    text = rs.decomposition(vs_rec, vs_t1)
    assert "scoring convention" in text
    assert "run-to-run" in text
    assert "identical=1/2" in text
    # The mislabelled cell must not inflate the convention figure.
    assert "max=0.40 pp" in text
