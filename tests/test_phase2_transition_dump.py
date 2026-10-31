"""Canonical-checkpoint selection and paired transitions from the dump alone."""
from __future__ import annotations

import collections

import numpy as np
import pandas as pd

from mv_post_train.stats.phase2_transition_dump import (
    canonical_checkpoints,
    cell_levels,
    transitions,
)

CANON = "sft__ot30k_1.7b-MixtureVitae-16k-SFT-Tulu3"
INTRUDER = "sft_1-7b-Mixt-300B-v1-16k-SFT-Tulu_open-thou_Open-1-2M"


def _counts(spec: dict) -> dict:
    return {cell: collections.Counter(srcs) for cell, srcs in spec.items()}


def test_row_count_majority_would_pick_the_intruder():
    """The real failure mode: the foreign run carries ten greedy repetitions
    per item to the canonical run's one, so it wins any within-cell row-count
    majority 270-30 while covering fewer tasks. Guards the rule, not the data.
    """
    counts = _counts({
        ("MV", "OT3+SFT", "MATH500"): {CANON: 500},
        ("MV", "OT3+SFT", "AIME24"): {CANON: 30},
        ("MV", "OT3+SFT", "AIME25"): {CANON: 30, INTRUDER: 270},
        ("MV", "OT3+SFT", "AMC23"): {CANON: 40, INTRUDER: 360},
    })
    keep = canonical_checkpoints(counts)

    assert keep[("MV", "OT3+SFT", "AIME25")] == CANON
    assert keep[("MV", "OT3+SFT", "AMC23")] == CANON
    # ... and a naive within-cell majority would have gone the other way.
    assert counts[("MV", "OT3+SFT", "AIME25")].most_common(1)[0][0] == INTRUDER


def test_cell_without_the_group_checkpoint_keeps_its_own():
    """A benchmark evaluated only from a stray checkpoint is reported, not
    silently deleted -- dropping it would remove the task from the table."""
    counts = _counts({
        ("MV", "OT3", "MATH500"): {CANON: 500},
        ("MV", "OT3", "AIME24"): {CANON: 30},
        ("MV", "OT3", "AMC23"): {INTRUDER: 40},
    })
    keep = canonical_checkpoints(counts)
    assert keep[("MV", "OT3", "AMC23")] == INTRUDER
    assert keep[("MV", "OT3", "MATH500")] == CANON


def test_groups_are_independent():
    """One stage's checkpoint choice must not leak into another stage."""
    other = "sft__ot30k_1.7b-MixtureVitae-16k"
    counts = _counts({
        ("MV", "OT3+SFT", "MATH500"): {CANON: 500},
        ("MV", "OT3+SFT", "AIME24"): {CANON: 30},
        ("MV", "OT3", "MATH500"): {other: 500},
        ("MV", "OT3", "AIME24"): {other: 30},
    })
    keep = canonical_checkpoints(counts)
    assert keep[("MV", "OT3+SFT", "MATH500")] == CANON
    assert keep[("MV", "OT3", "MATH500")] == other


def _cell(scores: list[float]) -> dict:
    return {i: [s] for i, s in enumerate(scores)}


def test_transition_pairs_on_items_and_reports_both_ends():
    recs = {
        ("MV", "SFT-Tulu3", "MATH500"): _cell([0.0] * 6 + [1.0] * 4),
        ("MV", "OT3+SFT", "MATH500"): _cell([1.0] * 9 + [0.0]),
    }
    out = transitions(recs, n_resamples=200)
    row = out[out.task == "MATH500"].iloc[0]
    assert row.acc_start == 40.0
    assert row.acc_end == 90.0
    assert row.delta == 50.0
    assert not row.degenerate


def test_repetitions_collapse_to_one_item_before_pairing():
    """Ten greedy repetitions of one problem are one paired unit, not ten."""
    start = {i: [0.0] * 10 for i in range(4)}
    end = {i: [1.0] * 10 for i in range(4)}
    out = transitions({("MV", "SFT-Tulu3", "AMC23"): start,
                       ("MV", "OT3+SFT", "AMC23"): end}, n_resamples=200)
    row = out.iloc[0]
    assert row.n_items == 4
    assert row.delta == 100.0


def test_both_ends_identical_is_degenerate_not_dropped():
    """Every reference scoring 0 on every contest-math item at both ends is the
    finding; it must survive into the table rather than vanish."""
    zeros = _cell([0.0] * 30)
    out = transitions({("Comma", "SFT-Tulu3", "AIME24"): dict(zeros),
                       ("Comma", "OT3+SFT", "AIME24"): dict(zeros)},
                      n_resamples=200)
    assert len(out) == 1
    row = out.iloc[0]
    assert row.degenerate
    assert (row.delta, row.ci_lo, row.ci_hi, row.p_perm) == (0.0, 0.0, 0.0, 1.0)


def test_unpaired_cells_are_skipped():
    out = transitions({("MV", "OT3+SFT", "MATH500"): _cell([1.0] * 10)},
                      n_resamples=200)
    assert out.empty


def test_only_shared_items_are_paired():
    recs = {
        ("MV", "SFT-Tulu3", "AIME24"): {0: [0.0], 1: [0.0], 2: [1.0]},
        ("MV", "OT3+SFT", "AIME24"): {1: [1.0], 2: [1.0], 3: [1.0]},
    }
    row = transitions(recs, n_resamples=200).iloc[0]
    assert row.n_items == 2
    assert row.acc_start == 50.0
    assert row.acc_end == 100.0


def test_cell_levels_average_over_repetitions():
    recs = {("MV", "OT3+SFT", "AIME24"): {0: [1.0, 0.0], 1: [0.0, 0.0]}}
    lv = cell_levels(recs)
    assert lv.iloc[0].acc == 25.0
    assert lv.iloc[0].n_items == 2


def test_q_values_only_span_live_tests():
    """Degenerate rows carry no p-value and must not enter the BH family --
    including them would dilute every real q toward 1."""
    zeros = _cell([0.0] * 20)
    recs = {
        ("Comma", "SFT-Tulu3", "AIME24"): dict(zeros),
        ("Comma", "OT3+SFT", "AIME24"): dict(zeros),
        ("MV", "SFT-Tulu3", "MATH500"): _cell([0.0] * 20),
        ("MV", "OT3+SFT", "MATH500"): _cell([1.0] * 20),
    }
    out = transitions(recs, n_resamples=2000)
    live = out[~out.degenerate]
    assert len(live) == 1
    # One live test, so BH leaves it alone.
    assert np.isclose(live.iloc[0].q_value, live.iloc[0].p_perm)
    assert (out[out.degenerate].q_value == 1.0).all()


def test_artifact_matches_the_released_scores_where_both_exist():
    """`dump_cell_levels.csv` is only useful if it tracks the authors' own
    released per-cell scores. Skips when either file is absent."""
    try:
        lv = pd.read_csv("artifacts/dump_cell_levels.csv")
        wide = pd.read_csv("data/laion/eval_scores_wide.csv")
    except FileNotFoundError:
        import pytest
        pytest.skip("artifacts or dump not present")
    tasks = ["MATH500", "AIME24", "AIME25", "AMC23"]
    long = wide.melt(id_vars=["base_model", "stage"], value_vars=tasks,
                     var_name="task", value_name="released")
    merged = lv.merge(long, left_on=["model", "stage", "task"],
                      right_on=["base_model", "stage", "task"])
    assert len(merged) > 50
    mv = merged[merged.model == "MixtureVitae-1.7B"]
    # MATH500 differs by a bounded scorer offset (math-verify vs the harness's
    # string match), the same sign at every stage.
    m500 = mv[mv.task == "MATH500"]
    assert ((m500.acc - m500.released).between(1.0, 5.0)).all()
    # The corrected contest-math row, which is what the response quotes.
    row = lv[(lv.model == "MixtureVitae-1.7B") & (lv.stage == "OT3+SFT")]
    got = {r.task: r.acc for r in row.itertuples()}
    assert (got["AIME24"], got["AIME25"], got["AMC23"]) == (6.67, 10.0, 20.0)
