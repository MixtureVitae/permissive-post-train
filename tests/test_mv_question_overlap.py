"""Matching the 30K reasoning samples' questions against MixtureVitae's reasoning shards."""
from __future__ import annotations

import gzip
import json

import pandas as pd
import pyarrow.parquet as pq
import pytest

from mv_post_train.stats import mv_question_overlap as mq

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


def test_items_cut_each_subset_and_number_every_stretch():
    q = pd.DataFrame(dict(corpus="C", subset=["code"] * 3 + ["math"],
                          position=[0, 1, 2, 3],
                          question=[p + WRAPPER for p in PROBLEMS] + ["What is 2 + 2?"]))
    qs, its = mq.items(q)
    assert qs["qid"].tolist() == [0, 1, 2, 3]
    assert its["item"].tolist() == list(range(len(its)))
    # The wrapper is the code subset's boilerplate, so it is no item's text.
    assert not its["question"].str.contains("solution code").any()
    assert (qs["n_grams"].iloc[:3] > 0).all() and qs["n_grams"].iloc[3] == 0


def test_chunks_read_lines_longer_than_a_megabyte(tmp_path, monkeypatch):
    src = tmp_path / "s.jsonl.gz"
    rows = [{"id": "a", "text": "x" * (3 << 20), "meta": {"k": 1}}, {"id": 7, "text": "short"}]
    with gzip.open(src, "wt") as f:
        f.writelines(json.dumps(r) + "\n" for r in rows)
    monkeypatch.setattr(mq, "CHUNK_BYTES", 1 << 20)
    assert mq.chunks(src, tmp_path / "in") == 2
    files = sorted((tmp_path / "in").glob("*.parquet"))
    assert len(files) == 2
    t = pq.read_table(files).to_pydict()
    assert t["id"] == ["a", "7"] and len(t["text"][0]) == 3 << 20


def test_per_question_sums_items_and_counts_levels():
    item_qid = [0, 0, 1]
    q_grams = pd.Series({0: 10, 1: 4})
    hits = [{"doc_id": "d1-part-0", "item_hits": [{"item_id": 0, "hits": 3},
                                                 {"item_id": 1, "hits": 6}]},
            {"doc_id": "d2-part-1", "item_hits": [{"item_id": 0, "hits": 5},
                                                 {"item_id": 2, "hits": 1}]}]
    rule = [{"doc_id": "d1-part-0"}]
    d = mq.per_question(hits, rule, item_qid, q_grams).set_index("qid")
    assert d.loc[0, "best_hits"] == 9 and d.loc[0, "best_doc"] == "d1-part-0"
    assert bool(d.loc[0, "best_flagged"])
    assert d.loc[0, "docs_half"] == 2 and d.loc[0, "docs_most"] == 1
    assert d.loc[1, "best_hits"] == 1 and d.loc[1, "docs_half"] == 0
    # On a tie, the document the rule flags, wherever it comes in the records.
    tie = [{"doc_id": "a", "item_hits": [{"item_id": 2, "hits": 2}]},
           {"doc_id": "b", "item_hits": [{"item_id": 2, "hits": 2}]},
           {"doc_id": "c", "item_hits": [{"item_id": 2, "hits": 2}]}]
    d = mq.per_question(tie, [{"doc_id": "b"}], item_qid, q_grams).set_index("qid")
    assert d.loc[1, "best_doc"] == "b" and bool(d.loc[1, "best_flagged"])
    assert d.loc[1, "docs_half"] == 3


def test_merge_takes_the_best_shard_and_exact_only_without_grams():
    qs = pd.DataFrame(dict(qid=[0, 1, 2], corpus="C", subset="math", position=[0, 1, 2],
                           n_tokens=20, n_grams=[10, 10, 0]))
    cols = ["qid", "best_hits", "q_grams", "best_doc", "best_flagged", "docs_half", "docs_most"]
    ex_cols = ["qid", "exact_docs", "exact_examples"]
    per_shard = {
        "a": (pd.DataFrame([(0, 4, 10, "x", False, 0, 0), (1, 10, 10, "y", False, 1, 1)], columns=cols),
              pd.DataFrame([(2, 1, "z-part-0"), (1, 1, "y-part-0")], columns=ex_cols)),
        "b": (pd.DataFrame([(0, 6, 10, "u", False, 1, 0), (1, 10, 10, "v", True, 1, 1)], columns=cols),
              pd.DataFrame([(2, 2, "w-part-3")], columns=ex_cols)),
    }
    d = mq.merge_frames(qs, per_shard).set_index("qid")
    assert d.loc[0, "best_shard"] == "b" and d.loc[0, "share"] == 0.6
    assert bool(d.loc[0, "half"]) and not d.loc[0, "most"]
    # A tie goes to the document the rule flags.
    assert d.loc[1, "best_doc"] == "v" and bool(d.loc[1, "flagged"])
    assert d.loc[1, "docs_half"] == 2
    # The exact pass counts only for a question with no 13-gram.
    assert not d.loc[1, "exact"] and bool(d.loc[2, "exact"]) and d.loc[2, "exact_docs"] == 3
    assert d["found"].tolist() == [True, True, True]


def test_exact_splits_examples_and_needs_the_whole_question(tmp_path):
    pytest.importorskip("ahocorasick")
    (tmp_path / "in").mkdir()
    import pyarrow as pa
    pq.write_table(pa.table({"text": ["Intro<|endoftext|>Q: What is TWO plus  two?\nA: 4",
                                      "what is two plus"],
                             "id": ["m", "n"]}), tmp_path / "in" / "c_0000.parquet")
    no_grams = pd.DataFrame(dict(qid=[5, 6], question=["What is two plus two?", "Short"]))
    e = mq.exact(tmp_path / "in", no_grams, 1)
    assert e.to_dict("records") == [{"qid": 5, "exact_docs": 1, "exact_examples": "m-part-1"}]


def test_ot3_questions_read_the_first_human_turn(tmp_path, monkeypatch):
    import pyarrow as pa
    monkeypatch.setattr(mq.qo, "TRAINED_ROWS", 2)
    conv = [[{"from": "human", "value": "Q1"}, {"from": "gpt", "value": "A1"}],
            [{"from": "human", "value": "Q2"}, {"from": "gpt", "value": "A2"}]]
    pq.write_table(pa.table({"domain": ["math", None], "conversations": conv}),
                   tmp_path / "train-00000-of-00001.parquet")
    q = mq.ot3_questions(tmp_path)
    assert q[["subset", "position", "question"]].values.tolist() == [["math", 0, "Q1"],
                                                                     ["none", 1, "Q2"]]
    assert set(q["corpus"]) == {"OpenThoughts3"}
