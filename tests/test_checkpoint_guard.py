"""Tests for recovering checkpoint identity from `source_dir`.

The bug this replaces was a hand-maintained exclusion set. The failure mode is
silent in both directions: miss a swapped checkpoint and you pair the wrong
model, over-flag and you discard good cells.
"""
from __future__ import annotations

import pandas as pd

from mv_post_train.data import checkpoint_guard as cg

BASE = "__e__scratch__jureap59__ot__data__checkpoints__1.7b-MixtureVitae-300BT-v1-decontaminated-16k"
SR = ("__p__scratch__laionize__marianna__checkpoints__sft__Superior-Reasoning-"
      "SFT-gpt-oss-120b-30k-sharegpt-hf__1-7b-MixtureVitae-30__sdpa-open-sci-st")


def test_the_same_checkpoint_under_two_hosts_normalizes_equal():
    """IFEval ran from a different filesystem root than the math tasks."""
    a = "__e__scratch__jureap59__ot__data__checkpoints__sft__ot30k_1.7b-MV-16k"
    b = "__p__scratch__laionize__marianna__checkpoints__sft__ot30k_1.7b-MV-16k"
    assert cg.checkpoint_id(a) == cg.checkpoint_id(b) == "sft__ot30k_1.7b-MV-16k"


def test_the_models_layout_is_recognized():
    s = "__p__scratch__transfernetx__marianna__models__Qwen3-1.7B-Base-SFT-Tulu3"
    assert cg.checkpoint_id(s) == "Qwen3-1.7B-Base-SFT-Tulu3"


def test_a_hub_snapshot_reduces_to_its_hash():
    """So a hub-evaluated cell matches the same model recorded by hash."""
    hub = ("__e__scratch__jureap59__ot__data__hub__models--Qwen--Qwen3-1.7B__"
           "snapshots__70d244cc86ccca08cf5af4e1e306ecf908b1ad5e")
    ckpt = "__p__scratch__x__checkpoints__70d244cc86ccca08cf5af4e1e306ecf908b1ad5e"
    assert cg.checkpoint_id(hub) == cg.checkpoint_id(ckpt)


def test_trailing_separators_are_stripped():
    a = "__x__snapshots__70d244cc__"
    assert cg.checkpoint_id(a) == "70d244cc"


def test_unrecognizable_paths_return_none():
    for s in (None, "", "no-markers-here"):
        assert cg.checkpoint_id(s) is None


def _ident(rows):
    return pd.DataFrame([
        {"base_model": "MixtureVitae-1.7B", "stage": "base",
         "n_distinct_in_cell": 1, **r} for r in rows])


def test_a_swapped_checkpoint_is_flagged():
    out = cg.flag(_ident([
        {"task": "GSM8K", "checkpoint": cg.checkpoint_id(BASE)},
        {"task": "MATH500", "checkpoint": cg.checkpoint_id(BASE)},
        {"task": "AIME24", "checkpoint": cg.checkpoint_id(BASE)},
        {"task": "IFEval", "checkpoint": cg.checkpoint_id(SR)},
    ]))
    by = dict(zip(out.task, out.consistent))
    assert by["GSM8K"] and by["MATH500"]
    assert not by["IFEval"]
    assert cg.inconsistent_cells(out) == {
        ("MixtureVitae-1.7B", "base", "IFEval")}


def test_a_truncated_path_is_not_a_disagreement():
    """`sft_Qwen3_open-thoughts` is the same run as its full spelling."""
    full = "sft_Qwen3-1-7B_open-thoughts_OpenThoughts3-1-2M_30000_samples"
    out = cg.flag(_ident([
        {"task": "AIME24", "checkpoint": full},
        {"task": "MATH500", "checkpoint": full},
        {"task": "AMC23", "checkpoint": "sft_Qwen3-1-7B_open-thoughts"},
    ]))
    assert out.consistent.all()


def test_the_longest_spelling_becomes_the_group_reference():
    """A truncated majority must not make the complete path look wrong."""
    short, full = "sft_Qwen3_open", "sft_Qwen3_open-thoughts_3-1-2M_30000"
    out = cg.flag(_ident([
        {"task": "AIME24", "checkpoint": short},
        {"task": "AMC23", "checkpoint": short},
        {"task": "MATH500", "checkpoint": full},
    ]))
    assert out.consistent.all()
    assert set(out.group_checkpoint) == {full}


def test_an_abbreviated_rename_is_still_flagged():
    """Not a prefix: a genuinely different run name must not slip through."""
    out = cg.flag(_ident([
        {"task": "AIME24", "checkpoint": "sft__ot30k_1.7b-MixtureVitae-16k-SFT"},
        {"task": "MATH500", "checkpoint": "sft__ot30k_1.7b-MixtureVitae-16k-SFT"},
        {"task": "AMC23", "checkpoint": "sft_1-7b-Mixt-300B-v1-16k-SFT-Tulu_step_1-"},
    ]))
    assert cg.inconsistent_cells(out) == {
        ("MixtureVitae-1.7B", "base", "AMC23")}
