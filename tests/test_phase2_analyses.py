"""Tests for the Phase 2 analyses (mvpt render-phase2-analyses, measure-phase2-answers).

They pin the rules the new numbers rest on: a contrast's estimate is the
manifest's change; the code tasks' bound is never narrower than question
resampling; IFEval is scored as the share of instructions; JEEBench questions
are told apart by `example_idx`; records that do not give the manifest's runs
are refused; an answer within the tolerance of its limit counts as cut off;
and only answers long enough to reach the limit are re-encoded. The last
tests resolve every run of the real tables against the committed manifest.
"""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq
import pytest

from mv_post_train.manifest import answer_lengths as al
from mv_post_train.manifest.reader import Manifest, ManifestError
from mv_post_train.render import phase2_analyses as pa2
from mv_post_train.stats import answer_endings as ae
from mv_post_train.stats import phase2_suite as ps

P2 = "phase2_three_seed"
TASKS = {"MATH500": 4, "HumanEval": 4, "IFEval": 2}


def _records(correct: dict[str, list[list[float]]]):
    """Answer rows for MATH500 (4 questions), one per checkpoint, question and
    run, and IFEval counts for two prompts with 1 and 3 instructions."""
    meta, ifeval = [], []
    for stage, runs in correct.items():
        for r, row in enumerate(runs):
            meta += [dict(base_model="M", stage=stage, task="MATH500", example_idx=i,
                          example_id=f"q{i}", run_idx=r, is_correct=v) for i, v in enumerate(row)]
            ifeval += [dict(base_model="M", stage=stage, run_idx=r, example_id="p0", n_total=1,
                            n_strict=row[0]),
                       dict(base_model="M", stage=stage, run_idx=r, example_id="p1", n_total=3,
                            n_strict=int(sum(row[1:])))]
    return pd.DataFrame(meta), pd.DataFrame(ifeval)


def _manifest(meta: pd.DataFrame, ifeval: pd.DataFrame, code: dict[str, float],
              parents: dict[str, str]) -> Manifest:
    """A manifest whose results are the records' own scores."""
    checkpoints, results, changes = {}, [], []
    for stage in meta["stage"].unique():
        checkpoints[stage] = {"parent": parents.get(stage), "hf_repos": [],
                              "parent_status": "verified" if stage in parents else "none recorded",
                              "labels": {P2: {"base_model": "M", "stage": stage}}}
        m = meta[meta["stage"] == stage].groupby("run_idx")["is_correct"].mean() * 100
        i = ifeval[ifeval["stage"] == stage].groupby("run_idx")
        f = i["n_strict"].sum() / i["n_total"].sum() * 100
        for task, runs in (("MATH500", m), ("IFEval", f),
                           ("HumanEval", pd.Series([code[stage]] * 3))):
            results.append({"checkpoint": stage, "protocol": P2, "task": task,
                            "mean": float(runs.mean()), "sd": 0.0,
                            "per_seed": {str(k): float(v) for k, v in enumerate(runs)},
                            "source": {"base_model": "M", "stage": stage, "task": task}})
    for stage, parent in parents.items():
        changes.append({"checkpoint": stage, "protocol": P2, "parent": parent, "change": 0.0})
    return Manifest({
        "protocols": {P2: {"tasks": {t: {"n_items": n, "metric": "accuracy"}
                                     for t, n in TASKS.items()}, "temperature": 0.7}},
        "checkpoints": checkpoints, "results": results, "suite_averages": [],
        "changes": changes, "rejected_changes": []})


CORRECT = {"base": [[1, 0, 0, 1], [1, 0, 1, 1], [0, 0, 0, 1]],
           "ot3": [[1, 1, 1, 1], [1, 1, 0, 1], [1, 1, 1, 0]]}


def _scores(code=None, runs=ps.ALL_RUNS):
    meta, ifeval = _records(CORRECT)
    m = _manifest(meta, ifeval, code or {"base": 25.0, "ot3": 50.0}, {"ot3": "base"})
    return ps.Scores(m, meta, ifeval, runs=runs)


def test_the_estimate_is_the_change_of_the_task_means():
    s = _scores()
    tasks = ("MATH500", "HumanEval", "IFEval")
    iv = ps.evaluate(s, [ps.change("f", "x", "ot3", "base", tasks)], n_boot=200)[0]
    expected = np.mean([s.score("ot3", t) - s.score("base", t) for t in tasks])
    assert iv.estimate == pytest.approx(expected)
    assert iv.lo <= iv.estimate <= iv.hi
    assert iv == ps.evaluate(s, [ps.change("f", "x", "ot3", "base", tasks)], n_boot=200)[0]


def test_ifeval_is_the_share_of_instructions_followed():
    s = _scores()
    # base, followed of 4 instructions: run 0, 1 + 1; run 1, 1 + 2; run 2, 0 + 1
    assert s.score("base", "IFEval") == pytest.approx(100 * (2 + 3 + 1) / 12)


def test_the_code_bound_is_never_narrower_than_question_resampling():
    rng = np.random.default_rng(1)
    for _ in range(20):
        n = 200
        a = rng.integers(0, 4, n) / 3       # a question's mean over three runs
        b = np.clip(a + rng.integers(-1, 2, n) / 3, 0, 1)
        idx = rng.integers(0, n, size=(4000, n))
        observed = (a[idx].mean(1) - b[idx].mean(1)).std()
        bound = np.sqrt(a.mean() * (1 - a.mean()) / n) + np.sqrt(b.mean() * (1 - b.mean()) / n)
        assert observed <= bound


def test_a_code_task_alone_gives_the_normal_interval_of_its_bound():
    s = _scores({"base": 25.0, "ot3": 50.0})
    iv = ps.evaluate(s, [ps.change("f", "x", "ot3", "base", ["HumanEval"])], n_boot=20000)[0]
    sd = 100 * (np.sqrt(0.25 * 0.75 / 4) + np.sqrt(0.5 * 0.5 / 4))
    assert iv.estimate == pytest.approx(25.0)
    assert (iv.hi - iv.lo) / 2 == pytest.approx(1.96 * sd, rel=0.03)
    assert iv.code_sd == pytest.approx(1.0, rel=0.02)


def test_a_difference_of_gains_is_the_first_change_minus_the_second():
    meta, ifeval = _records({**CORRECT, "b0": CORRECT["ot3"], "b1": CORRECT["base"]})
    m = _manifest(meta, ifeval, {"base": 25.0, "ot3": 50.0, "b0": 50.0, "b1": 25.0},
                  {"ot3": "base", "b1": "b0"})
    s = ps.Scores(m, meta, ifeval)
    ivs = ps.evaluate(s, [ps.gain_difference("f", "d", ("ot3", "base"), ("b1", "b0"), ["MATH500"]),
                          ps.change("f", "a", "ot3", "base", ["MATH500"])], n_boot=200)
    assert ivs[0].estimate == pytest.approx(2 * ivs[1].estimate)


def test_records_that_do_not_give_the_manifest_runs_are_refused():
    meta, ifeval = _records(CORRECT)
    m = _manifest(meta, ifeval, {"base": 25.0, "ot3": 50.0}, {"ot3": "base"})
    meta.loc[0, "is_correct"] = 0.0
    with pytest.raises(ManifestError, match="per-question records give runs"):
        ps.Scores(m, meta, ifeval).items("base", "MATH500")


def test_runs_1_and_2_leave_out_run_0():
    s = _scores(runs=(1, 2))
    assert s.score("base", "MATH500") == pytest.approx(100 * (3 + 1) / 8)
    assert s.code("base", "HumanEval") == 25.0


def test_questions_are_told_apart_by_example_idx():
    # JEEBench's example_id is a subject label that many questions share.
    meta, ifeval = _records(CORRECT)
    meta["example_id"] = "physics"
    s = _scores()
    s.meta = meta
    assert len(s.items("base", "MATH500")) == 4


def test_a_cut_off_answer_reaches_its_share_of_the_limit():
    edge = ae.LIMIT_SHARE * 32000
    lengths = pd.DataFrame({"max_tokens": [32000.0] * 4,
                            "n_tokens": [32000, edge, edge - 1, None]})
    assert ae.cut_off(lengths).tolist() == [True, True, False, False]


def test_length_shares_count_each_percent_of_each_limit():
    lengths = pd.DataFrame({"max_tokens": [32000.0, 32000.0, 16000.0, 16000.0],
                            "n_tokens": [32000, 31700, 15000, None]})
    out = ae.length_shares(lengths).set_index(["max_tokens", "share_from"])["answers"]
    assert out[(32000, "1.00")] == 1 and out[(32000, "0.99")] == 1
    assert out[(16000, "0.93")] == 1 and out.sum() == 3


def test_shares_are_per_cell_and_only_where_they_apply():
    lengths = pd.DataFrame({
        "base_model": "M", "stage": "s",
        "task": ["MATH500", "MATH500", "IFEval", "MBPP", "MBPP", "HumanEval"],
        "max_tokens": [32000.0, 32000.0, 16000.0, 16000.0, 16000.0, 16000.0],
        "n_tokens": [32000, None, None, None, None, None],
        "no_answer": [False, True, True, False, False, False],
        "no_extracted": [False, False, True, True, False, True]})
    out = ae.shares(lengths).set_index("task")
    assert out.loc["MATH500", ["cut_off", "no_answer"]].tolist() == [50.0, 50.0]
    assert np.isnan(out.loc["IFEval", "no_answer"]) and np.isnan(out.loc["MATH500", "no_code"])
    assert out.loc["MBPP", "no_code"] == 50.0
    assert np.isnan(out.loc["HumanEval", "no_code"])   # no answer has its code: not recorded


class _Encoding:
    def __init__(self, n):
        self.ids = [0] * n


class _ByteTokenizer:
    """One token per byte, so a long answer re-encodes to its length."""
    def encode_batch_fast(self, texts, add_special_tokens=False):
        return [_Encoding(len(t.encode())) for t in texts]


def test_only_answers_that_can_reach_the_limit_are_re_encoded(tmp_path, monkeypatch):
    rows = [dict(base_model="M", stage=s, task="MATH500", example_idx=i, example_id=f"q{i}",
                 run_idx=0, max_tokens=100.0, decoding=d, model_output=o, model_answer=a,
                 model_answer_extracted=a)
            for i, (s, d, o, a) in enumerate([("ot3", "sampling", "x" * 95, "4"),
                                              ("ot3", "sampling", "x" * 50, ""),
                                              ("ot3", "greedy", "x" * 95, "4"),
                                              ("other", "sampling", "x" * 95, "4")])]
    path = tmp_path / "responses.parquet"
    table = pa.Table.from_pylist(rows)
    table = table.cast(pa.schema([pa.field(f.name, pa.large_string() if f.type == pa.string()
                                           else f.type) for f in table.schema]))
    pq.write_table(table, path)        # the released file stores its text as large_string
    meta, ifeval = _records(CORRECT)
    m = _manifest(meta, ifeval, {"base": 25.0, "ot3": 50.0}, {"ot3": "base"})
    monkeypatch.setattr(al, "tokenizer_sources",
                        lambda m, snap, cells: {c: ("r", "v", ("sha1", "h")) for c in cells})
    monkeypatch.setattr(al, "load_tokenizer", lambda *a: _ByteTokenizer())
    out = al.measure(path, m, {"repos": {}}, tmp_path).to_pandas()
    assert out["n_bytes"].tolist() == [95, 50]          # sampled answers of labelled cells only
    assert out["n_tokens"].tolist()[0] == 95 and pd.isna(out["n_tokens"].tolist()[1])
    assert out["no_answer"].tolist() == [False, True]
    assert out["tokenizer"].tolist() == ["r@v", "r@v"]


def test_a_checkpoint_without_a_tokenizer_uses_its_parents(monkeypatch):
    meta, ifeval = _records(CORRECT)
    m = _manifest(meta, ifeval, {"base": 25.0, "ot3": 50.0}, {"ot3": "base"})
    m.data["checkpoints"]["base"]["hf_repos"] = ["org/base"]
    snap = {"repos": {"org/base": {"revision": "abc"}}}
    monkeypatch.setattr(al, "_repo_file", lambda repo, rev, name: ("sha1", "h"))
    assert al.tokenizer_sources(m, snap, [("M", "ot3")]) == {("M", "ot3"): ("org/base", "abc",
                                                                            ("sha1", "h"))}


ROOT = Path(__file__).resolve().parents[1]


def test_every_run_of_the_real_tables_resolves_against_the_committed_manifest():
    m = Manifest.load(ROOT / "results" / "manifest.json")
    runs = pa2.all_runs(m)
    assert len({r.stem for r in runs}) == len(runs)
    assert {r.stem for r in pa2.table2_runs(m)} <= {r.stem for r in runs}
    with_, without = pa2.exclusion_runs(m)
    assert (with_.base_model, without.base_model) == (pa2.pt.MV, pa2.pt.MV_NO_OT3)
    assert set(ps.domain_tasks(m)["nonmath"]) | set(ps.DOMAINS["math"]) == set(m.tasks())


def test_the_widening_is_the_gap_after_minus_the_gap_before():
    meta, ifeval = _records({"mv_sft": CORRECT["base"], "mv_ot3": CORRECT["ot3"],
                             "ref_sft": CORRECT["ot3"], "ref_ot3": CORRECT["base"]})
    m = _manifest(meta, ifeval, {k: 25.0 for k in ("mv_sft", "mv_ot3", "ref_sft", "ref_ot3")},
                  {"mv_ot3": "mv_sft", "ref_ot3": "ref_sft"})
    s = ps.Scores(m, meta, ifeval)
    pr = pa2.Pair("Qwen3-1.7B-Base", "mv_sft", "mv_ot3", "ref_sft", "ref_ot3")
    tasks = ("MATH500", "IFEval")
    ivs = {iv.contrast.label: iv for iv in ps.evaluate(s, [
        pa2._gap("f", "sft", "mv_sft", "ref_sft", tasks),
        pa2._gap("f", "ot3", "mv_ot3", "ref_ot3", tasks),
        ps.gain_difference("f", "w", (pr.mv_ot3, pr.mv_sft), (pr.ref_ot3, pr.ref_sft), tasks)],
        n_boot=200)}
    assert ivs["w"].estimate == pytest.approx(ivs["ot3"].estimate - ivs["sft"].estimate)


def test_leave_one_out_drops_each_mathematics_task_once():
    pr = pa2.Pair("Qwen3-1.7B-Base", "a", "b", "c", "d")
    cs = pa2.leave_one_out_contrasts([pr])
    math = ps.DOMAINS["math"]
    assert len(cs) == len(math)
    assert all(len(c.tasks) == len(math) - 1 for c in cs)
    assert {t for c in cs for t in math if t not in c.tasks} == set(math)


def test_every_reference_has_both_endpoints_in_the_committed_manifest():
    m = Manifest.load(ROOT / "results" / "manifest.json")
    prs = pa2.pairs(m)
    assert [p.base_model for p in prs] == list(pa2.pt.REFERENCES)
    for p in prs:
        assert m.parent(p.ref_ot3)[0] == p.ref_sft and m.parent(p.mv_ot3)[0] == p.mv_sft


def test_a_cell_whose_empty_answers_score_correct_has_no_share():
    lengths = pd.DataFrame({
        "base_model": "M", "stage": ["a", "a", "b", "b"], "task": "MATH500",
        "example_idx": [0, 1, 0, 1], "run_idx": 0, "max_tokens": 32000.0, "n_tokens": None,
        "no_answer": [True, False, True, True], "no_extracted": False})
    verdicts = lengths[["base_model", "stage", "task", "example_idx", "run_idx"]].assign(
        is_correct=[0.0, 1.0, 1.0, 0.0])        # in b, an answer with no extracted answer is right
    out = ae.shares(lengths, verdicts).set_index("stage")["no_answer"]
    assert out["a"] == 50.0 and np.isnan(out["b"])
    lengths["no_answer"] = [True, True, False, True]          # every answer of a is empty
    out = ae.shares(lengths, verdicts.assign(is_correct=0.0)).set_index("stage")["no_answer"]
    assert np.isnan(out["a"]) and out["b"] == 50.0


def test_a_range_is_the_extremes_of_its_values_and_empty_without_values():
    assert pa2._span("k", [3.0, -1.0, 2.0], str) == {"k.min": "-1.0", "k.max": "3.0"}
    assert pa2._span("k", [], str) == {}


def test_the_committed_share_ranges_are_the_extremes_of_the_printed_cells():
    import re
    tex = (ROOT / "results" / "phase2_analyses" / "tables" / "phase2_analyses_numbers.tex").read_text()
    values = dict(re.findall(r"\\csname ptwoan@(\S+?)\\endcsname\{([^}]*)\}", tex))
    matched = ("fineweb", "nemotron", "comma")
    cells = [float(v) for k, v in values.items()
             if k.split(".")[0] in matched and ".sft-ot3.cutoff." in k
             and k.count(".") == 3 and not k.endswith(".gsm8k")]
    assert len(cells) == 3 * 10
    assert float(values["cutoff.matched.sft-ot3.but-gsm8k.min"]) == min(cells)
    assert float(values["cutoff.matched.sft-ot3.but-gsm8k.max"]) == max(cells)
    math = [float(values[f"qwen3.sft-ot3.noanswer.{t}"]) for t in ("a24", "a25", "amc", "m500")]
    assert "qwen3.sft-ot3.noanswer.gsm8k" not in values
    assert float(values["noanswer.qwen3.sft-ot3.math.min"]) == min(math)
