"""Matching the other reasoning corpora's questions against OpenThoughts3."""
from __future__ import annotations

import pandas as pd
import pytest

from mv_post_train.stats import question_overlap as qo

# The matching needs the `overlap` extra.
pytest.importorskip("corpus_assay")

PROBLEMS = [
    "Given an array of n integers, find the length of the longest strictly increasing "
    "subsequence whose sum of elements is divisible by the number k given on the second line.",
    "A frog sits on the first of n stones placed along a river and may jump to any stone "
    "at most d positions ahead, paying the absolute height difference each time it lands.",
    "Compute the number of ordered pairs of positive integers a and b such that the least "
    "common multiple of a and b equals one thousand and their sum is a prime number.",
]
WRAPPER = ("\n\nWrite Python code to solve the problem. Please place the solution code in "
           "the following format: your solution code here between two markers")


def _grams(text):
    return {h for h, _ in qo._hashes(qo.tokens(text))}


def test_the_question_is_the_first_user_turn():
    assert qo._question({"conversations": [{"from": "human", "value": "q"},
                                           {"from": "gpt", "value": "a"}]}) == "q"
    assert qo._question({"messages": [{"role": "system", "content": "s"},
                                      {"role": "user", "content": "q"},
                                      {"role": "assistant", "content": "a"}]}) == "q"
    assert qo._question({"input": "q", "output": "a"}) == "q"


def test_boilerplate_needs_a_share_and_a_floor():
    sets = [_grams(p + WRAPPER) for p in PROBLEMS]
    drop = qo.boilerplate(sets)
    assert drop and drop <= _grams(WRAPPER)
    # In two questions only, the wrapper is below BOILERPLATE_MIN.
    assert not qo.boilerplate(sets[:2])


def test_a_wrapper_is_cut_out_whole():
    """No stretch keeps a token of the wrapper, so no N-gram straddles it."""
    prep = qo.prepare([p + WRAPPER for p in PROBLEMS])
    for p, (_, row) in zip(PROBLEMS, prep.iterrows()):
        assert row.text == " ".join(qo.tokens(p))
        assert row.n_grams == len(_grams(p))


def test_stretches_shorter_than_n_are_dropped():
    toks = [f"w{i}" for i in range(qo.N + 5)]
    drop = {qo._hashes(toks)[3][0]}
    # The dropped N-gram covers tokens 3 to 3+N-1, leaving 3 and 2 tokens.
    assert qo.stretches(toks, drop) == []
    assert qo.stretches(toks, set()) == [toks]


def test_the_best_item_sums_hits_over_sub_documents():
    parts = [
        {"match_count": 5, "item_hits": [
            {"item_id": 7, "hits": 4, "boilerplate_hits": 1, "longest_run_tokens": 16},
            {"item_id": 2, "hits": 3, "boilerplate_hits": 0, "longest_run_tokens": 15}]},
        {"match_count": 3, "item_hits": [
            {"item_id": 2, "hits": 3, "boilerplate_hits": 0, "longest_run_tokens": 15}]},
    ]
    assert qo.best_item(parts) == dict(n_shared=8, best_shared=6, best_boilerplate=0,
                                       best_longest_run=15, best_qid=2)
    assert qo.best_item([])["best_qid"] == -1


def test_levels_and_rates():
    df = pd.DataFrame({"corpus": "X", "subset": "all", "cluster": [0, 0, 1, 1],
                       "n_grams": [10, 10, 10, 0], "n_shared": [10, 3, 0, 0],
                       "best_shared": [10, 3, 0, 0], "best_in_phase2": [True, False, False, False],
                       "whole_match": [None, None, None, True]})
    hits = qo.levels(df)
    assert hits["any"].tolist() == [True, True, False, True]
    assert hits["fifth"].tolist() == [True, True, False, True]
    assert hits["half"].tolist() == [True, False, False, True]
    assert hits["most"].tolist() == [True, False, False, True]
    assert hits["half_phase2"].tolist() == [True, False, False, False]
    r = qo.rates(df).iloc[0]
    assert (r.questions, r.clusters, r.no_grams) == (4, 2, 1)
    assert (r["any"], r.half, r.half_phase2) == (75.0, 50.0, 25.0)
    assert r.half_lo <= 50.0 <= r.half_hi


def test_corpus_assay_finds_a_wrapped_copy(tmp_path):
    """End to end: index three questions, scan wrapped copies and an unrelated one."""
    ref = pd.DataFrame({"qid": range(3), "question": PROBLEMS, "source": "src",
                        "domain": "code", "n_rows": 1, "in_phase2": [True, False, False]})
    ref.to_parquet(tmp_path / "ref.parquet", index=False)
    qo.build_index(tmp_path / "ref.parquet", tmp_path / "index" / "ot3.native")
    other = ("Explain in two paragraphs why the sky looks blue during the day and red at "
             "sunset, and name the physical effect responsible for each one of them.")
    samples = pd.DataFrame({"corpus": "X", "subset": "all", "dataset": "d", "revision": "r",
                            "cluster": range(4), "file": "f", "position": 0, "domain": "code",
                            "question": [p + WRAPPER for p in PROBLEMS[::-1]] + [other + WRAPPER]})
    df = qo.match(samples, ref, tmp_path / "index" / "ot3.native", tmp_path / "scan")
    assert df.best_qid.tolist() == [2, 1, 0, -1]
    assert (df.best_shared[:3] == df.n_grams[:3]).all()
    assert df.best_in_phase2.tolist() == [False, False, True, False]
    assert qo.levels(df)["most"].tolist() == [True, True, True, False]


def test_trained_questions_read_every_record_with_its_stage(tmp_path, monkeypatch):
    import pyarrow as pa
    import pyarrow.parquet as pq
    monkeypatch.setattr(qo, "TRAINED_ROWS", 2)
    conv = lambda q: [{"from": "human", "value": q}, {"from": "gpt", "value": "a"}]
    tables = {
        "AceReason": {"conversations": [conv("q1"), conv("q2")], "category": ["math", "code"]},
        "Nemotron-Cascade": {"conversations": [conv("q3"), conv("q4")], "category": ["science", "math"]},
        "Superior-Reasoning": {"conversations": [conv("q5"), conv("q6")], "domain": ["code", "math"],
                               "meta": [{"training_stage": "stage2"}, {"training_stage": "stage1"}]},
    }
    for corpus, cols in tables.items():
        d = tmp_path / qo.TRAINED[corpus] / "data"
        d.mkdir(parents=True)
        pq.write_table(pa.table(cols), d / "train-00000-of-00001.parquet")
    df = qo.trained_questions(tmp_path)
    assert df.question.tolist() == ["q1", "q2", "q3", "q4", "q5", "q6"]
    assert df.subset.tolist() == ["math", "code", "science", "math", "code", "math"]
    assert df.stage.fillna("-").tolist() == ["-", "-", "stage1", "stage1", "stage2", "stage1"]
    assert df.cluster.tolist() == [0, 1, 0, 1, 0, 1]
    monkeypatch.setattr(qo, "TRAINED_ROWS", 3)
    with pytest.raises(ValueError, match="not 3"):
        qo.trained_questions(tmp_path)
