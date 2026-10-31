"""Tests for Tables 7, 8, 9 and 15, the older experiments (render-older-tables).

They pin that Tables 7 to 9 print Protocol A means with their SDs and bold the
printed column maximum, that Table 15 prints its masked and unmasked rows under
Protocol A with their changes as macros, and, for the older evaluations that
the manifest still reads, the run rule (the mean of a checkpoint's runs,
leaving out runs that score near zero where another run does not) and that a
checkpoint whose runs all score near zero keeps them. They also pin that the older pipeline's `--mean-unbroken-runs
--gsm8k-flexible` options use the same rule, and that without them it is
unchanged.
"""
from __future__ import annotations

import json

import pytest

from mv_post_train.data import run_rule
from mv_post_train.paper_check import table_rows
from mv_post_train.render import older_tables as ot


def _run(ts: str, results: dict) -> tuple[str, bytes]:
    return (f"logs/x__model/results_{ts}.json", json.dumps({"results": results}).encode())


def _humaneval(py: float, sh: float) -> dict:
    return {"HumanEval": {"python_pass@1": py, "sh_pass@1": sh}}


def test_a_near_zero_run_is_broken_only_if_another_run_works():
    assert run_rule.broken([7.6, 7.3, 1.8, 0.3]) == [False, False, False, True]
    assert run_rule.broken([0.0, 0.0, 0.6, 0.0]) == [False] * 4      # the model scores near zero


def test_unbroken_rule_averages_the_runs_that_did_not_fail():
    files = [_run("2026-04-15T17-40-06", _humaneval(0.140, 0.013)),
             _run("2026-04-15T17-42-43", _humaneval(0.140, 0.006)),
             _run("2026-04-15T17-45-53", _humaneval(0.000, 0.006))]
    # HumanEval is the mean of its Python and shell scores: 7.65, 7.3 and 0.3
    assert ot.scores_unbroken(files)["HumanEval"] == pytest.approx((7.65 + 7.3) / 2)


def test_metadata_fields_are_not_scores():
    files = [_run("2026-04-15T17-49-27", {"gsm8k": {
        "alias": "gsm8k", "exact_match,strict-match": 0.01, "exact_match,flexible-extract": 0.08,
        "exact_match_stderr,flexible-extract": 0.01}})]
    assert ot.scores_unbroken(files)["gsm8k"] == pytest.approx(8.0)


def _pa(value=10.0, sd=0.5):
    cks = ot.checkpoints()
    return ot.ProtocolA({ck: {t: value for t in ot.TASKS} for ck in cks},
                        {ck: {t: sd for t in ot.TASKS} for ck in cks},
                        {ck: (value, sd) for ck in cks})


def test_protocol_a_cells_print_the_sd_and_no_bold():
    pa = _pa()
    first = next(ot.TABLES[1]["groups"][0].rows())[2]
    pa.mean[first]["AIME24"] = 21.04
    pa.avg[first] = (11.0, 0.123)
    tex = ot.table_tex(ot.TABLES[1], pa)
    rows = {r.label: r.cells for r in table_rows(tex)}      # the two panels, merged
    assert rows["Baguettotron"][0] == ["21.0", "0.50"] and rows["Baguettotron"][-1] == ["11.0", "0.12"]
    assert r"21.0$_{\pm0.50}$" in tex and r"\textbf{2" not in tex
    assert rows["Baguettotron 16k"][0] == ["10.0", "0.50"]
    assert "Protocol" in tex and "--" not in tex and r"\resizebox" not in tex


def _write_runs(model_dir, runs):
    model_dir.mkdir(parents=True)
    for ts, results in runs:
        (model_dir / f"results_{ts}.json").write_text(json.dumps({"results": results}))


def _math(answers: list[str]) -> dict:
    return {"MATH500": {"accuracy": 0.0, "examples": [
        {"answer": "4", "model_answer": a} for a in answers]}}


def test_older_pipeline_options_use_the_run_rule(tmp_path):
    from mv_post_train.data.parse_phase1 import Phase1Parser
    d = tmp_path / "ali-elganzory__1.7b-MixtureVitae-100BT-longsft_16k-DPO-Tulu3-decontaminated"
    gsm8k = {"gsm8k": {"exact_match,strict-match": 0.10, "exact_match,flexible-extract": 0.30}}
    _write_runs(d, [("2026-04-15T17-40-06", {**_humaneval(0.140, 0.013), **_math(["4", "5"])}),
                    ("2026-04-15T17-42-43", {**_humaneval(0.140, 0.006), **_math(["4", "4"]),
                                             **gsm8k}),
                    ("2026-04-15T17-45-53", _humaneval(0.000, 0.006))])
    out = Phase1Parser(tmp_path, mean_unbroken_runs=True, gsm8k_flexible=True).parse()
    agg = {r.task: r for r in out.aggregate_rows}
    # the near-zero HumanEval run is left out, the other two are averaged
    assert agg["HumanEval"].accuracy_mean == pytest.approx((0.0765 + 0.073) / 2)
    assert agg["gsm8k"].accuracy_mean == pytest.approx(0.30)
    assert agg["gsm8k"].source_metric == "exact_match,flexible-extract"
    # MATH500's per-item scores keep both runs, as separate attempts
    items = [r for r in out.item_rows if r.task == "MATH500"]
    assert sorted((r.item_index, r.attempt_index, r.score) for r in items) == [
        (0, 0, 1.0), (0, 1000, 1.0), (1, 0, 0.0), (1, 1000, 1.0)]
    # without the options, the parser keeps one run and strict GSM8K, as before
    old = {r.task: r for r in Phase1Parser(tmp_path, latest_run=True).parse().aggregate_rows}
    assert old["HumanEval"].accuracy_mean == pytest.approx(0.003)
    assert old["gsm8k"].accuracy_mean == pytest.approx(0.10)


def _decon_scores(value=10.0):
    return {ck: {t: value for t in ot.TASKS} for ck in ot.checkpoints()}


def test_table8_has_two_sections_and_no_bold():
    pa = _pa()
    clean16_dpo = ot.DECON[1][2][1][4]
    pa.mean[clean16_dpo]["AIME24"] = 20.0
    pa.avg[clean16_dpo] = (10.9, 0.2)
    tex = ot.decontamination_tex(pa)
    rows = table_rows(tex)                                   # the two panels, merged
    assert len(rows) == 12 and tex.count(r"\multirow{6}{*}") == 4 and tex.count(r"\cmidrule{2-") == 4
    assert rows[-1].cells[0] == ["20.0", "0.50"] and rows[-1].cells[-1] == ["10.9", "0.20"]


def _pa_masked(masked=12.0, unmasked=10.0):
    pa = _pa(unmasked)
    for ck in ot.MASKED.values():
        pa.mean[ck] = {t: masked for t in ot.TASKS}
        pa.sd[ck] = {t: 0.5 for t in ot.TASKS}
        pa.avg[ck] = (masked, 0.5)
    return pa


def test_table15_prints_both_arms_under_protocol_a():
    tex = ot.masking_tex(_pa_masked())
    assert "With masking" in tex and "transcribed" not in tex and r"$^\dagger$" not in tex
    rows = table_rows(tex)                                   # the two panels, merged
    assert len(rows) == 8
    assert all(r.cells[0] == ["12.0", "0.50"] for r in rows[:4])
    assert all(r.cells[0] == ["10.0", "0.50"] for r in rows[4:])
    assert "raises the suite average by 2.0 to 2.0 points" in tex


def test_masking_keys_give_each_cell_and_the_change():
    out = ot.macros(_pa_masked(masked=9.0))
    assert out["mask.masked.4k.sft.avg"] == "9.0" and out["mask.unmasked.16k.dpo.he"] == "10.0"
    assert out["mask.masked.16k.dpo.jee.sd"] == "0.50"
    assert out["mask.change.4k.sft.avg"] == "\\ensuremath{-1.0}"
    assert out["mask.change.ife.min"] == out["mask.change.ife.max"] == "\\ensuremath{-1.0}"
    assert out["decon.change.4k.base.avg"] == "\\ensuremath{+0.0}"
    assert out["small.mv04.4k.dpo.he"] == "10.0" and out["small.mv04.4k.dpo.he.sd"] == "0.50"
