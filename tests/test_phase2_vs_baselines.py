"""Between-model MV-minus-reference contrasts at the OT30K endpoint."""
from __future__ import annotations

import numpy as np
import pandas as pd

from mv_post_train.stats.phase2_vs_baselines import (
    MV,
    REFERENCES,
    contrasts,
)

COMMA = "Comma0.1-1.7B"
SMOL = "SmolLM2-1.7B-16k-Base"


def _cell(scores: list[float]) -> dict:
    return {i: [s] for i, s in enumerate(scores)}


def test_smollm2_is_a_reference():
    """The one reference that BEATS MixtureVitae on IFEval after OT30K.

    It is absent from `parse_phase2.FAMILY`, so sourcing IFEval from
    `recovered_items.parquet` drops it -- and dropping it would leave a table
    in which MixtureVitae wins every single contrast. Guards the scope, not
    the numbers.
    """
    assert SMOL in REFERENCES


def test_external_comparators_are_out_of_scope():
    """Qwen sits on a different compute footing; a win over it is not the
    matched-compute claim Figure 3 makes."""
    assert not any(r.startswith("Qwen") for r in REFERENCES)
    assert "SmolLM2-1.7B-Instruct-16k" not in REFERENCES


def test_pairs_on_items_and_reports_both_ends():
    recs = {
        (MV, "OT3+SFT", "GSM8K"): _cell([1.0] * 8 + [0.0] * 2),
        (COMMA, "OT3+SFT", "GSM8K"): _cell([1.0] * 3 + [0.0] * 7),
    }
    row = contrasts(recs, n_resamples=500).iloc[0]
    assert (row.acc_mv, row.acc_ref, row.delta) == (80.0, 30.0, 50.0)
    assert row.n_items == 10
    assert not row.degenerate


def test_a_reference_ahead_of_mv_is_reported_as_negative():
    """Nothing in the pipeline may quietly orient the contrast MV-favourably."""
    recs = {
        (MV, "OT3+SFT", "IFEval"): _cell([1.0] * 2 + [0.0] * 8),
        (SMOL, "OT3+SFT", "IFEval"): _cell([1.0] * 6 + [0.0] * 4),
    }
    row = contrasts(recs, n_resamples=500).iloc[0]
    assert row.delta == -40.0
    assert row.ci_hi < 0


def test_repetitions_collapse_to_one_item_before_pairing():
    """Ten greedy repeats of one problem are one paired unit, not ten -- and a
    ten-repeat cell must not outweigh a one-repeat cell on the other side."""
    recs = {
        (MV, "OT3+SFT", "AMC23"): {i: [1.0] * 10 for i in range(4)},
        (COMMA, "OT3+SFT", "AMC23"): {i: [0.0] for i in range(4)},
    }
    row = contrasts(recs, n_resamples=500).iloc[0]
    assert row.n_items == 4
    assert row.delta == 100.0


def test_only_shared_items_are_paired():
    recs = {
        (MV, "OT3+SFT", "AIME24"): {0: [1.0], 1: [1.0], 2: [0.0]},
        (COMMA, "OT3+SFT", "AIME24"): {1: [0.0], 2: [0.0], 3: [1.0]},
    }
    row = contrasts(recs, n_resamples=500).iloc[0]
    assert row.n_items == 2
    assert (row.acc_mv, row.acc_ref) == (50.0, 0.0)


def test_both_models_identical_is_degenerate_not_dropped():
    """Both at 0 on every contest-math item is a real, reportable state; a
    permutation test on an all-zero difference vector is not."""
    recs = {
        (MV, "OT3+SFT", "AIME24"): _cell([0.0] * 30),
        (COMMA, "OT3+SFT", "AIME24"): _cell([0.0] * 30),
    }
    out = contrasts(recs, n_resamples=500)
    assert len(out) == 1
    row = out.iloc[0]
    assert row.degenerate
    assert (row.delta, row.p_perm, row.q_value) == (0.0, 1.0, 1.0)


def test_missing_reference_cell_is_skipped():
    recs = {(MV, "OT3+SFT", "GSM8K"): _cell([1.0] * 10)}
    assert contrasts(recs, n_resamples=500).empty


def test_q_values_only_span_live_tests():
    """Degenerate rows carry no p-value; letting them into the BH family would
    dilute every real q toward 1."""
    recs = {
        (MV, "OT3+SFT", "AIME24"): _cell([0.0] * 20),
        (COMMA, "OT3+SFT", "AIME24"): _cell([0.0] * 20),
        (MV, "OT3+SFT", "GSM8K"): _cell([1.0] * 20),
        ("NemotronHQ-1.7B", "OT3+SFT", "GSM8K"): _cell([0.0] * 20),
    }
    out = contrasts(recs, n_resamples=2000)
    live = out[~out.degenerate]
    assert len(live) == 1
    assert np.isclose(live.iloc[0].q_value, live.iloc[0].p_perm)
    assert (out[out.degenerate].q_value == 1.0).all()


def test_artifact_tracks_the_released_scores():
    """`phase2_vs_baselines.csv` is only usable in a response if its levels
    match the authors' own released grid. Skips when either file is absent."""
    try:
        got = pd.read_csv("artifacts/phase2_vs_baselines.csv")
        wide = pd.read_csv("data/laion/eval_scores_wide.csv")
    except FileNotFoundError:
        import pytest
        pytest.skip("artifacts or dump not present")

    released = {(r.base_model, r.stage): r for r in wide.itertuples()}

    # GSM8K carries the dump's own `is_correct`, so it should reproduce the
    # released cell almost exactly -- this is the check that the pairing,
    # checkpoint filter and repetition collapse did not perturb the levels.
    gsm = got[got.task == "GSM8K"]
    assert len(gsm) >= 6
    checked = 0
    for r in gsm.itertuples():
        assert abs(r.acc_mv - released[(MV, r.stage)].GSM8K) < 1.0
        ref_cell = released[(r.reference, r.stage)].GSM8K
        # FineWeb-Edu / OT3 is blank in the released grid but fully present in
        # the dump (1,319 scored rows, 8.79). Recomputing a cell the table
        # leaves empty is a result, not a mismatch -- but it cannot be
        # cross-checked, so it is skipped rather than asserted against NaN.
        if pd.isna(ref_cell):
            continue
        assert abs(r.acc_ref - ref_cell) < 1.0
        checked += 1
    assert checked >= 5

    # The headline non-MATH500 result the response quotes.
    sft = gsm[(gsm.stage == "OT3+SFT")].set_index("reference")
    assert sft.loc["NemotronHQ-1.7B"].delta > 40
    assert sft.loc["NemotronHQ-1.7B"].ci_lo > 0
    assert (sft.q_value < 0.01).all()

    # AIME24 cannot resolve its own effect at n=30, and must not be quoted as
    # if it could.
    aime = got[(got.task == "AIME24") & (~got.degenerate)]
    assert (aime.q_value > 0.05).all()
