"""Varying the reasoning corpus with the base fixed, and vice versa."""
from __future__ import annotations

import numpy as np
import pandas as pd

from mv_post_train.stats.corpus_variation import (
    ALL_STAGES,
    CORPORA,
    ROUTES,
    per_task,
    pooled,
)
from mv_post_train.stats.phase2_vs_baselines import MV

SMOL = "SmolLM2-1.7B-16k-Base"


def _cell(scores) -> dict:
    return {i: [float(s)] for i, s in enumerate(scores)}


def test_all_four_corpora_and_both_routes():
    assert set(CORPORA) == {"OT3", "AR", "NC", "SR"}
    assert [r[0] for r in ROUTES] == ["SFT-Tulu3", "DPO-Tulu3"]
    # Every Phase 2 stage the grid needs, and no Phase 1 stage beyond the two
    # starting points.
    assert "AR+SFT" in ALL_STAGES and "SR+DPO" in ALL_STAGES
    assert len(ALL_STAGES) == 2 + len(CORPORA) * len(ROUTES)


def test_transition_is_end_minus_start():
    recs = {
        (MV, "SFT-Tulu3", "MATH500"): _cell([1.0] * 2 + [0.0] * 8),
        (MV, "AR+SFT", "MATH500"): _cell([1.0] * 7 + [0.0] * 3),
    }
    row = per_task(recs, n_resamples=500).iloc[0]
    assert (row.acc_start, row.acc_end, row.delta) == (20.0, 70.0, 50.0)
    assert row.corpus == "AceReason"
    assert row.route == "SFT-Tulu3->AR+SFT"


def test_a_regression_keeps_its_sign():
    recs = {
        (MV, "SFT-Tulu3", "MATH500"): _cell([1.0] * 8 + [0.0] * 2),
        (MV, "SR+SFT", "MATH500"): _cell([1.0] * 3 + [0.0] * 7),
    }
    row = per_task(recs, n_resamples=500).iloc[0]
    assert row.delta == -50.0
    assert row.ci_hi < 0


def test_flat_reference_cells_are_degenerate_not_dropped():
    """Every reference sits at 0 on contest math before and after every corpus.
    That is the finding; it must reach the table."""
    recs = {
        ("Comma0.1-1.7B", "SFT-Tulu3", "AIME24"): _cell([0.0] * 30),
        ("Comma0.1-1.7B", "OT3+SFT", "AIME24"): _cell([0.0] * 30),
    }
    out = per_task(recs, n_resamples=500)
    assert len(out) == 1
    assert out.iloc[0].degenerate and out.iloc[0].q_value == 1.0


def test_pooled_needs_more_than_one_benchmark():
    recs = {
        (MV, "SFT-Tulu3", "MATH500"): _cell([0.0] * 10),
        (MV, "OT3+SFT", "MATH500"): _cell([1.0] * 10),
    }
    assert pooled(recs, n_resamples=200, n_perm=200).empty
    assert not per_task(recs, n_resamples=200).empty


def _artifact(name):
    try:
        return pd.read_csv(f"artifacts/{name}")
    except FileNotFoundError:
        import pytest
        pytest.skip("artifacts not present")


def test_mixturevitae_responds_to_every_corpus():
    t = _artifact("corpus_variation.csv")
    mv = t[(t.model == MV) & (t.task == "MATH500") & t.route.str.endswith("+SFT")]
    assert len(mv) == 4
    assert (mv.q_value < 0.01).all()
    assert (mv.ci_lo > 10).all()


def test_acereason_and_cascade_are_indistinguishable_from_ot3():
    """The claim the response makes: the gain is not an OT3 artifact. It rests
    on the intervals overlapping, so pin that rather than the point estimates."""
    t = _artifact("corpus_variation.csv")
    mv = (t[(t.model == MV) & (t.task == "MATH500") & t.route.str.endswith("+SFT")]
          .set_index("corpus"))
    ot3 = mv.loc["OpenThoughts3"]
    for other in ("AceReason", "Nemotron-Cascade"):
        r = mv.loc[other]
        assert r.ci_lo < ot3.ci_hi and r.ci_hi > ot3.ci_lo, other
    # Superior-Reasoning is the one that does NOT reach OT3 -- it is the mixed
    # control, and presenting it as equivalent would be wrong.
    sr = mv.loc["Superior-Reasoning"]
    assert sr.ci_hi < ot3.ci_lo


def test_smollm2_responds_to_three_of_four_corpora():
    """The honest boundary on "the base conditions the response".

    SmolLM2 is a matched reference that *does* gain on AceReason,
    Nemotron-Cascade and Superior-Reasoning -- only OT3 leaves it flat. Any
    wording that says no reference responds to any corpus is refuted here.
    """
    t = _artifact("corpus_variation.csv")
    s = (t[(t.model == SMOL) & (t.task == "MATH500") & t.route.str.endswith("+SFT")]
         .set_index("corpus"))
    responds = s[s.q_value < 0.05]
    assert set(responds.index) == {"AceReason", "Nemotron-Cascade", "Superior-Reasoning"}
    assert (responds.ci_lo > 0).all()
    assert s.loc["OpenThoughts3"].q_value > 0.05
    # ...but at a quarter of MixtureVitae's magnitude.
    mv = t[(t.model == MV) & (t.task == "MATH500")
           & (t.corpus == "AceReason") & t.route.str.endswith("+SFT")].iloc[0]
    assert responds.loc["AceReason"].delta < mv.delta / 3


def test_the_other_three_references_stay_flat():
    t = _artifact("corpus_variation.csv")
    flat = t[(t.task == "MATH500") & t.route.str.endswith("+SFT")
             & t.model.isin(["Comma0.1-1.7B", "FineWebEdu-1.7B", "NemotronHQ-1.7B"])]
    assert len(flat) == 12
    # Nothing positive-significant; the only significant cell is a *decline*.
    assert not ((flat.q_value < 0.05) & (flat.delta > 0)).any()
    assert (flat.delta.abs() <= 2.2).all()


def test_pooled_agrees_with_per_task_on_direction():
    p = _artifact("corpus_variation_pooled.csv")
    mv = p[(p.model == MV) & p.route.str.endswith("+SFT")]
    assert len(mv) == 4
    assert (mv.macro_ci_lo > 0).all() and (mv.micro_ci_lo > 0).all()
    assert (mv.macro_q < 0.01).all()
