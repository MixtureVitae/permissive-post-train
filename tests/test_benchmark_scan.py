"""Scanning the training data only a cluster holds against the benchmarks."""
from __future__ import annotations

import gzip
import json
import shutil

import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq
import pytest

from mv_post_train.stats import benchmark_scan as bs
from mv_post_train.stats import training_data_overlap as to

# The matching needs the `overlap` extra.
pytest.importorskip("corpus_assay")

QUESTION = ("A bag holds seven red marbles and five blue marbles. Two marbles are drawn "
            "without replacement. What is the probability that both marbles drawn are red, "
            "expressed as a common fraction in lowest terms?")
PASSAGE = ("The river rises in the northern hills and flows south through three counties, "
           "past the old mill towns and the wide marshes, before it reaches the estuary and "
           "the sea beyond the harbour wall.")


@pytest.fixture
def work(tmp_path, monkeypatch):
    monkeypatch.setattr(to, "BENCHMARKS", ("MATH500", "commonsense_qa", "boolq"))
    items = pd.DataFrame(dict(
        benchmark=["MATH500", "commonsense_qa", "boolq"], item=["test/x.json", "q1", "7"],
        text=[QUESTION, "Where would you put a plate after washing it? cupboard table",
              PASSAGE + " does the river reach the sea"],
        question=[QUESTION, "Where would you put a plate after washing it?",
                  "does the river reach the sea"],
        choices=[[], ["cupboard", "table"], []]))
    to.build_index(items, tmp_path / "items", tmp_path / "index" / "benchmarks.native")
    items.to_parquet(tmp_path / "benchmark_items.parquet", index=False)
    return tmp_path


def test_a_sample_and_a_shard_are_scanned_like_the_released_data(work, monkeypatch):
    monkeypatch.setattr(bs.qo, "TRAINED_ROWS", 3)
    folder = work / "samples" / bs.qo.TRAINED["AceReason"] / "data"
    folder.mkdir(parents=True)
    pq.write_table(pa.Table.from_pylist([
        {"conversations": [{"from": "human", "value": "Solve: " + QUESTION},
                           {"from": "gpt", "value": "7/22"}]},
        {"conversations": [{"from": "human", "value": "Q: Where would you put a plate after "
                                                      "washing it? - cupboard - table"},
                           {"from": "gpt", "value": "cupboard"}]},
        {"conversations": [{"from": "human", "value": "Tell me about the printing press."},
                           {"from": "gpt", "value": PASSAGE}]},
    ]), folder / "train-0.parquet")
    assert bs.scan("sample:AceReason", work, work / "samples", workers=2) == 0
    s = pd.read_parquet(work / "sources" / "sample:AceReason.parquet")
    by = s.set_index(["doc", "item"])
    assert by.loc[("0:0", "test/x.json"), "hits"] == by.loc[("0:0", "test/x.json"), "item_grams"]
    assert by.loc[("0:0", "test/x.json"), "role"] == "user"
    assert by.loc[("1:0", "q1"), "exact"] and by.loc[("1:0", "q1"), "choices_found"] == 2
    # The passage without its question is not BoolQ's item.
    assert not by.loc[("2:1", "7"), "boolq_question"]

    # A shard: two documents packed into one record.
    name = bs.mq.shard_name(bs.mq.SHARDS[0][0])
    shard = work / f"{name}.jsonl.gz"
    with gzip.open(shard, "wt") as f:
        f.write(json.dumps({"id": "r0", "text": "Intro.<|endoftext|>" + PASSAGE
                            + " Does the river reach the sea? Yes."}) + "\n")
    monkeypatch.setattr(bs.mq, "SHARDS", ((bs.mq.SHARDS[0][0], shard.stat().st_size),))
    assert bs.scan(f"mv:{name}", work, shard, workers=2) == 0
    m = pd.read_parquet(work / "sources" / f"mv:{name}.parquet")
    assert m[["doc", "item", "role"]].values.tolist() == [["r0-part-1", "7", "pretraining"]]
    assert m["boolq_question"].tolist() == [True]

    monkeypatch.setattr(bs, "SOURCES", ["sample:AceReason", f"mv:{name}"])
    assert bs.merge(work, work) == 0
    f = to.cluster_found(pd.read_parquet(work / to.CLUSTER_SCAN))
    got = f[["where", "row", "turn", "benchmark", "evidence"]].values.tolist()
    assert sorted(got, key=str) == sorted([
        ["sample:AceReason", 0, "0", "MATH500", "13-gram"],
        ["sample:AceReason", 0, "0", "MATH500", "exact"],
        ["sample:AceReason", 1, "0", "commonsense_qa", "exact with choices"],
        ["mv_reasoning", -1, "r0-part-1", "boolq", "13-gram"]], key=str)
    assert to.excluded(f) == {"MATH500": {"test/x.json"}, "commonsense_qa": {"q1"},
                              "boolq": {"7"}}


def test_a_cache_is_a_prompt_and_an_answer_per_sample(work):
    texts = work / "texts"
    texts.mkdir()
    pq.write_table(pa.table({"row": [0, 1], "prompt": ["<|user|>\n" + QUESTION, "<|user|>\nHi"],
                             "answer": ["<think>easy</think>7/22", "Hello."]}),
                   texts / "AceReason.parquet")
    assert bs.scan("phase2:AceReason", work, texts / "AceReason.parquet", workers=2) == 0
    s = pd.read_parquet(work / "sources" / "phase2:AceReason.parquet")
    assert s[["doc", "role", "item"]].values.tolist() == [["0:prompt", "prompt", "test/x.json"]]
    shutil.rmtree(work / "sources")
