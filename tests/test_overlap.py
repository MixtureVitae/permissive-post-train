"""The benchmark-overlap audit in the paper (mv_post_train/render/overlap.py)."""
from __future__ import annotations

from types import SimpleNamespace

import pandas as pd
import pytest

from mv_post_train import inputs
from mv_post_train.render import overlap as o
from mv_post_train.render import phase2_tables as pt


def test_a_q_equal_to_the_threshold_up_to_rounding_is_not_significant():
    # 0.0025 x 20 in floating point: Table 4's Comma0.1 cell under Nemotron-Cascade
    assert not pt.significant(0.049999999999999996)
    assert not pt.significant(0.05)
    assert pt.significant(0.0499)


def test_changed_counts_a_call_that_crosses_the_threshold_either_way():
    s = pd.DataFrame({"q": [0.01, 0.2, 0.01, 0.049999999999999996],
                      "q_cut": [0.2, 0.01, 0.02, 0.2]})
    assert o.changed(s).tolist() == [True, True, False, False]


@pytest.mark.parametrize("analysis,family,label,expected", [
    ("phase2", "table2.m500", "mv.dpo-ot3", r"MV DPO$\to$OT30K, MATH500"),
    ("phase2", "exclusion", "difference.m500", "OT3-source exclusion, difference of gains, MATH500"),
    ("table4", "table4.m500", "FineWebEdu-1.7B | AceReason", r"FineWeb-Edu SFT$\to$AceReason, MATH500"),
    ("factorial", "permissiveness", "MV - Nemotron | 4096 | sft",
     "Source pools, MV $-$ Nemotron-CC-HQ, 4K, SFT"),
    ("language", "change.avg", "mv | mv.sft-ot3 | mv.base",
     r"MV SFT$\to$OT30K, from 16K base, language"),
    ("language", "frontloading.avg", "interaction | 16384 | MV - non-permissive",
     "Interaction, MV $-$ non-permissive, 16K base, language"),
])
def test_comparisons_are_named_in_words(analysis, family, label, expected):
    assert o.name(SimpleNamespace(analysis=analysis, family=family, label=label)) == expected


def _found() -> pd.DataFrame:
    try:
        path = inputs.path("O3")
    except inputs.InputError:
        pytest.skip("input O3 (the audit's items found) is not here")
    return pd.read_parquet(path)


def test_the_items_table_reproduces_the_audit():
    f = _found()
    o.check_found(f, o.excluded_items())
    it = o.items_table(f).set_index("benchmark")
    assert it["found"].sum() == 1720 and len(it) == 17
    assert (it.loc["MATH500", ["found", "left", "mv_reasoning"]] == [213, 287, 211]).all()
    assert it.loc["commonsense_qa", "tulu3_sft_decontaminated"] == 650
    assert (it.loc["AIME24", ["found", "left"]] == [25, 5]).all()
    assert it.loc["LiveCodeBenchv5", "mv_reasoning"] == 199


def test_the_share_of_ot3_answers_that_never_close_is_weighted_by_samples():
    release = pd.DataFrame({"domain": ["all", "all", "math"], "samples": [100, 300, 50],
                            "think_end": [80.0, 20.0, 99.0]})
    assert o.ot3_unfinished(release) == pytest.approx(100 - (80 * 100 + 20 * 300) / 400)


def test_billions_prints_two_decimals():
    assert o.billions(2195234965) == "2.20B" and o.billions(881775120) == "0.88B"
