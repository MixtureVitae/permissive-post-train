"""Counting the training tokens of a Phase 2 dataset from its tokenized cache."""
from __future__ import annotations

import pyarrow as pa

from mv_post_train.stats import phase2_tokens as pt

IGN = pt.LABEL_IGNORE


def _cache(path, rows):
    """A save_to_disk-style cache: train/*.arrow, Arrow IPC streams."""
    (path / "train").mkdir(parents=True)
    for i, part in enumerate(rows):
        t = pa.table({"input_ids": [list(range(len(lab))) for lab in part],
                      "labels": part})
        with pa.OSFile(str(path / "train" / f"data-{i:05d}-of-{len(rows):05d}.arrow"), "wb") as f:
            with pa.ipc.new_stream(f, t.schema) as w:
                w.write_table(t)


def test_lengths_count_tokens_and_supervised_tokens(tmp_path):
    _cache(tmp_path / "c", [[[IGN, IGN, 5, 6], [IGN, 7]], [[IGN] * 3 + [1] * 5]])
    df = pt.lengths(tmp_path / "c")
    assert df.seq.tolist() == [4, 2, 8]
    assert df.sup.tolist() == [2, 1, 5]


def test_summary_marks_samples_at_the_cutoff(tmp_path, monkeypatch):
    monkeypatch.setattr(pt, "CUTOFF", 4)
    _cache(tmp_path / "c", [[[IGN, IGN, 5, 6], [IGN, 7]]])
    s = pt.summary("x", pt.lengths(tmp_path / "c"))
    assert s["samples"] == 2 and s["at_cutoff"] == 50.0
    assert s["sup_epoch"] == 3 and s["sup_run"] == 3 * pt.EPOCHS


def _cache_ids(path, samples):
    """A cache from (input_ids, labels) pairs."""
    (path / "train").mkdir(parents=True)
    t = pa.table({"input_ids": [i for i, _ in samples], "labels": [lab for _, lab in samples]})
    with pa.OSFile(str(path / "train" / "data-00000-of-00001.arrow"), "wb") as f:
        with pa.ipc.new_stream(f, t.schema) as w:
            w.write_table(t)


def test_a_sample_at_the_cutoff_is_not_trained_to_stop(tmp_path, monkeypatch):
    monkeypatch.setattr(pt, "CUTOFF", 6)
    eot, te = pt.END_OF_TURN, list(pt.THINK_END)
    _cache_ids(tmp_path / "c", [
        ([1, 2, 9] + te[:2] + [7], [IGN, IGN, 9] + te[:2] + [7]),   # cut inside "</think>"
        ([1, 2] + te + [eot], [IGN, IGN] + te + [eot]),                       # at the cutoff, ended
        ([1, 5, eot], [IGN, 5, eot]),                                         # short, ended
        ([1, 5, eot], [IGN, 5, IGN]),                                         # end not supervised
    ])
    df = pt.lengths(tmp_path / "c")
    assert df.ended.tolist() == [False, True, True, False]
    assert df.think_end.tolist() == [False, True, False, False]
    s = pt.summary("x", df)
    assert s["at_cutoff"] == 50.0 and s["ended"] == 50.0 and s["cap_think_end"] == 50.0


def test_release_summary_counts_what_the_cut_removes(monkeypatch):
    import pandas as pd
    monkeypatch.setattr(pt, "CUTOFF", 10)
    df = pd.DataFrame(dict(domain=["math", "math", "code"], tokens=[8, 14, 20],
                           think_end=[True, False, False], boxed=[True, False, False]))
    s = pt.release_summary(df).set_index(["domain", "over"])
    assert s.loc[("math", True), "removed_median"] == 4
    assert s.loc[("all", True), "samples"] == 2 and s.loc[("all", True), "think_end"] == 0
    assert s.loc[("all", False), "think_end"] == 100
    assert pt.templated("q", "a") == "<|user|>\nq\n<|assistant|>\na<end_of_turn>"
