"""Matching the post-training data against the paper's benchmarks."""
from __future__ import annotations

import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq
import pytest

from mv_post_train.stats import training_data_overlap as to

# The matching needs the `overlap` extra.
pytest.importorskip("corpus_assay")

QUESTION = ("A bag holds seven red marbles and five blue marbles. Two marbles are drawn "
            "without replacement. What is the probability that both marbles drawn are red, "
            "expressed as a common fraction in lowest terms?")
OTHER = ("Every morning a baker sells three dozen rolls at two dollars each and half as many "
         "loaves of rye bread at five dollars each. How much money does the baker take in "
         "over a week of seven days of business?")


def test_fields_are_joined_like_the_mixturevitae_index():
    row = {"question": "Which gas?", "choices": {"text": ["oxygen", "neon"], "label": ["A", "B"]},
           "test_list": ["assert f(1) == 2", "assert f(2) == 3"], "none": None}
    assert to._text(row, ["question", "choices"]) == "Which gas? oxygen neon"
    assert to._text(row, ["test_list", "none"]) == "assert f(1) == 2 assert f(2) == 3"


def test_each_message_is_a_turn_and_a_shared_prompt_counts_once():
    chat = {"messages": [{"role": "user", "content": "q"}, {"role": "assistant", "content": None}]}
    assert to._turns(chat, "chat") == [("0", "q"), ("1", "")]
    pref = {"chosen": [{"role": "user", "content": "q"}, {"role": "assistant", "content": "good"}],
            "rejected": [{"role": "user", "content": "q"}, {"role": "assistant", "content": "bad"}]}
    assert to._turns(pref, "preference") == [("0", "q"), ("c1", "good"), ("r1", "bad")]


def test_a_copied_question_is_flagged_and_found(tmp_path, monkeypatch):
    monkeypatch.setattr(to, "BENCHMARKS", ("MATH500", "GSM8K"))
    items = pd.DataFrame(dict(benchmark=["MATH500", "GSM8K"], item=["test/x.json", "7"],
                              text=[QUESTION, OTHER]))
    to.build_index(items, tmp_path / "items", tmp_path / "index" / "b.native")
    table = to.item_table(tmp_path / "index" / "b.native", items)
    assert table["item"].tolist() == ["test/x.json", "7"] and (table["item_grams"] > 0).all()

    raw = tmp_path / "raw" / "data"
    raw.mkdir(parents=True)
    unrelated = "Tell me about the history of the printing press in Europe, please."
    pq.write_table(pa.Table.from_pylist([
        {"messages": [{"role": "user", "content": "Solve this: " + QUESTION},
                      {"role": "assistant", "content": "The answer is 7/22."}]},
        {"messages": [{"role": "user", "content": unrelated},
                      {"role": "assistant", "content": "It began with Gutenberg."}]},
    ]), raw / "train-00000-of-00001.parquet")
    d = to.Dataset("toy", "x/y", "0" * 40, "chat", True)
    rows = to.messages(d, tmp_path / "raw", tmp_path / "in")
    assert rows["messages"].tolist() == [2, 2]
    rule = to.scan(tmp_path / "in", tmp_path / "index" / "b.native", tmp_path / "rule",
                   to.MIN_HITS, to.MIN_COVERAGE)
    hits = to.scan(tmp_path / "in", tmp_path / "index" / "b.native", tmp_path / "hits", 1, 0.0)
    df = to.overlap(d, rule, hits, table, rows)
    assert df[["row", "turn", "benchmark", "item"]].values.tolist() == [[0, "0", "MATH500", "test/x.json"]]
    assert df["flagged"].all() and (df["hits"] == df["item_grams"]).all()
    r = to.rates(df, items)
    assert r[["benchmark", "rows_flagged", "items_found", "items"]].values.tolist() == [["MATH500", 1, 1, 1]]


def test_an_exact_match_counts_the_choices_shown_near_the_question(monkeypatch):
    monkeypatch.setattr(to, "EXACT_WINDOW", 40)
    items = pd.DataFrame(dict(benchmark=["commonsense_qa", "MATH500"], item=["a", "b"],
                              question=["Where would you put a plate after washing it?",
                                        "Compute $\\dbinom{8}{4}$."],
                              choices=[["cupboard", "table"], []]))
    docs = [("0:0", "Q: Where would you  put a plate after washing it? Options: - cupboard - table"),
            ("1:0", "Where would you put a plate after washing it? " + "x" * 80 + " cupboard table"),
            ("2:1", "First, COMPUTE $\\dbinom{8}{4}$. Then add one.")]
    df = to.exact(items, docs)
    assert df[["doc", "item", "choices", "choices_found"]].values.tolist() == [
        ["0:0", "a", 2, 2], ["1:0", "a", 2, 0], ["2:1", "b", 0, 0]]


def test_only_strong_evidence_is_left_out():
    found = pd.DataFrame(dict(where=["long_sft", "long_sft", "tulu3_sft_decontaminated", "AceReason"],
                              benchmark=["MATH500", "mmlu", "commonsense_qa", "MATH500"],
                              item=["a", "b", "c", "d"],
                              evidence=["13-gram", "question only", "exact with choices", "exact"]))
    assert to.excluded(found) == {"MATH500": {"a", "d"}, "commonsense_qa": {"c"}}
    assert to.excluded(found, where=["long_sft"]) == {"MATH500": {"a"}}
