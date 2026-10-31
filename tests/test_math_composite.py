"""Pooling the five math benchmarks into one contrast per reference."""
from __future__ import annotations

import numpy as np
import pandas as pd

from mv_post_train.stats.core import suite_average_bootstrap
from mv_post_train.stats.math_composite import (
    MATH_TASKS,
    _bootstrap,
    _macro,
    _micro,
    _paired_diffs,
    _perm_p,
    composite_contrasts,
)
from mv_post_train.stats.phase2_vs_baselines import MV

COMMA = "Comma0.1-1.7B"
NEMO = "NemotronHQ-1.7B"


def _cell(scores) -> dict:
    return {i: [float(s)] for i, s in enumerate(scores)}


def test_all_five_math_benchmarks_are_pooled():
    assert set(MATH_TASKS) == {"GSM8K", "MATH500", "AMC23", "AIME24", "AIME25"}


def test_macro_and_micro_differ_when_benchmarks_are_unbalanced():
    """The whole reason both are reported. A big benchmark where MV is flat
    and a small one where MV dominates must not produce the same number."""
    diffs = {
        "GSM8K": np.zeros(1000),
        "AIME24": np.ones(30),
    }
    assert _macro(diffs) == 0.5           # 20% weight each, unweighted mean
    assert abs(_micro(diffs) - 30 / 1030) < 1e-12   # items dominate
    assert _macro(diffs) > _micro(diffs)


def test_macro_matches_the_shared_suite_average_primitive():
    """`_bootstrap`'s macro arm must agree with core.suite_average_bootstrap,
    which is the tested primitive for stratified paired suite averages."""
    rng = np.random.default_rng(7)
    a = {"GSM8K": rng.random(300).round(), "AMC23": rng.random(40).round()}
    b = {"GSM8K": rng.random(300).round(), "AMC23": rng.random(40).round()}
    diffs = {t: a[t] - b[t] for t in a}
    ref = suite_average_bootstrap(a, b, n_resamples=4000, seed=0)
    assert abs(_macro(diffs) - ref.point) < 1e-12
    (lo, hi), _ = _bootstrap(diffs, n_resamples=4000, seed=0)
    # Independent RNG streams, so agreement is to Monte-Carlo error, not exact.
    assert abs(lo - ref.lo) < 0.03 and abs(hi - ref.hi) < 0.03


def test_bootstrap_resamples_within_benchmark_not_across():
    """Pooling the 1,919 items before resampling would let benchmark sizes
    wander between replicates and silently reweight the macro statistic. If
    strata were ignored, a benchmark with zero variance could not stay fixed."""
    diffs = {"GSM8K": np.zeros(500), "AIME24": np.ones(30)}
    (lo, hi), _ = _bootstrap(diffs, n_resamples=500, seed=0)
    # GSM8K contributes exactly 0 every replicate; AIME24 exactly 1. Macro is
    # pinned at 0.5 only if the two strata keep their sizes.
    assert lo == hi == 0.5


def test_perm_p_is_two_sided_and_bounded():
    """A composite with no signal must not look significant; the +1/+1
    correction keeps p away from an unattainable 0."""
    rng = np.random.default_rng(3)
    null = {"GSM8K": rng.choice([-1.0, 0.0, 1.0], 400),
            "AMC23": rng.choice([-1.0, 0.0, 1.0], 40)}
    p = _perm_p(null, _macro, n_resamples=2000, seed=0)
    assert 0.0 < p <= 1.0
    assert p > 0.05

    strong = {"GSM8K": np.ones(400), "AMC23": np.ones(40)}
    assert _perm_p(strong, _macro, n_resamples=2000, seed=0) < 0.001


def test_paired_diffs_uses_shared_items_only():
    recs = {
        (MV, "OT3+SFT", "AIME24"): {0: [1.0], 1: [1.0], 2: [0.0]},
        (COMMA, "OT3+SFT", "AIME24"): {1: [0.0], 2: [0.0], 3: [1.0]},
    }
    d = _paired_diffs(recs, "OT3+SFT", COMMA)
    assert len(d["AIME24"]) == 2
    assert d["AIME24"].tolist() == [1.0, 0.0]


def test_a_single_benchmark_does_not_make_a_composite():
    """One task is not a pooled claim; it is the per-task test that already
    exists in phase2_vs_baselines."""
    recs = {
        (MV, "OT3+SFT", "GSM8K"): _cell([1.0] * 20),
        (COMMA, "OT3+SFT", "GSM8K"): _cell([0.0] * 20),
    }
    assert composite_contrasts(recs, n_resamples=200, n_perm=200).empty


def test_leave_one_out_reports_the_worst_subset():
    """A composite carried entirely by GSM8K must expose that, not hide it."""
    recs = {
        (MV, "OT3+SFT", "GSM8K"): _cell([1.0] * 100),
        (COMMA, "OT3+SFT", "GSM8K"): _cell([0.0] * 100),
        (MV, "OT3+SFT", "MATH500"): _cell([0.0] * 100),
        (COMMA, "OT3+SFT", "MATH500"): _cell([0.0] * 100),
        (MV, "OT3+SFT", "AMC23"): _cell([0.0] * 40),
        (COMMA, "OT3+SFT", "AMC23"): _cell([0.0] * 40),
    }
    row = composite_contrasts(recs, n_resamples=200, n_perm=200).iloc[0]
    assert row.n_tasks == 3
    assert abs(row.macro_delta - 100 / 3) < 0.005  # emitted rounded to 2 dp
    # Dropping GSM8K is what hurts, and it takes the composite to zero under
    # both weightings.
    assert row.worst_loo_macro_dropped == "GSM8K"
    assert row.worst_loo_macro == 0.0
    assert row.worst_loo_micro_dropped == "GSM8K"
    assert row.worst_loo_micro == 0.0


def test_leave_one_out_weightings_can_disagree_on_the_worst_subset():
    """Micro is driven by whichever benchmark carries the most items, macro by
    whichever carries the largest gap. Reporting only one hides the other."""
    recs = {
        # Highest per-benchmark gap but only 40 items: removing it costs macro
        # a full third of its three benchmark means, and costs micro almost
        # nothing.
        (MV, "OT3+SFT", "AMC23"): _cell([1.0] * 40),
        (COMMA, "OT3+SFT", "AMC23"): _cell([0.0] * 40),
        # Slightly smaller gap over 1,000 items: it *is* the micro statistic,
        # so removing it collapses micro from .82 to .29.
        (MV, "OT3+SFT", "GSM8K"): _cell([1.0] * 900 + [0.0] * 100),
        (COMMA, "OT3+SFT", "GSM8K"): _cell([0.0] * 1000),
        (MV, "OT3+SFT", "MATH500"): _cell([0.0] * 100),
        (COMMA, "OT3+SFT", "MATH500"): _cell([0.0] * 100),
    }
    row = composite_contrasts(recs, n_resamples=200, n_perm=200).iloc[0]
    assert row.worst_loo_macro_dropped == "AMC23"
    assert row.worst_loo_micro_dropped == "GSM8K"
    # macro: mean(1.0, .9, 0) = .633 -> .45 without AMC23
    # micro: 940/1140 = .825      -> 40/140 = .286 without GSM8K
    assert abs(row.worst_loo_macro - 45.0) < 0.005
    assert abs(row.worst_loo_micro - 28.57) < 0.005


def test_perm_p_matches_the_closed_form_when_one_exists():
    """The contest-math composite has an exact answer, so the simulation can be
    checked rather than trusted. Every reference scores 0 on every contest-math
    item, so the paired differences are 13 ones among 100 zeros (2/30 + 3/30 +
    8/40). Flipping any one of the 13 strictly shrinks the statistic, so
    |macro| is maximal only when all 13 share a sign: exact two-sided
    p = 2 / 2**13 = 2.44e-4.
    """
    diffs = {
        "AIME24": np.array([1.0] * 2 + [0.0] * 28),
        "AIME25": np.array([1.0] * 3 + [0.0] * 27),
        "AMC23": np.array([1.0] * 8 + [0.0] * 32),
    }
    assert abs(_macro(diffs) - 0.12222222) < 1e-6
    exact = 2 / 2 ** 13
    got = _perm_p(diffs, _macro, n_resamples=20000, seed=0)
    # The +1/+1 correction floors the estimate slightly above `exact`.
    assert exact <= got < exact + 3e-4


def _artifact():
    try:
        return pd.read_csv("artifacts/math_composite.csv")
    except FileNotFoundError:
        import pytest
        pytest.skip("artifacts not present")


def test_artifact_covers_both_phases():
    got = _artifact()
    assert set(got.phase) == {"Phase 1", "Phase 2"}
    # `base` yields no contrast: the release carries MV's base checkpoint on
    # these benchmarks but no reference's.
    assert "base" not in set(got.stage)


def test_pooled_math_is_significant_at_both_phases():
    """Under both weightings, at every stage, against every reference."""
    got = _artifact()
    assert len(got) >= 13
    assert (got.macro_q < 0.05).all()
    assert (got.micro_q < 0.05).all()
    assert (got.macro_ci_lo > 0).all()
    assert (got.micro_ci_lo > 0).all()
    # Contest math is 5% of items but 60% of the macro weight, so macro must
    # not simply reproduce micro; and since contest math carries the smallest
    # gaps, macro is the conservative weighting.
    assert not np.allclose(got.macro_delta, got.micro_delta)
    assert (got.macro_delta < got.micro_delta).all()


def test_contest_math_separates_only_after_phase_2():
    """The finding that distinguishes the two tables, and the one most likely
    to be misread if only Phase 2 is shown.

    Pooling rescues contest math at Phase 2 (q < .001 everywhere) but at
    Phase 1 there is nothing to rescue: MV scores 0.0 / 0.0 / 5.0 on
    AIME24 / AIME25 / AMC23, level with every reference, so the pooled contest
    composite is flat and one contrast is even negative.
    """
    got = _artifact()
    assert (got.contest_n_items == 100).all()

    p2 = got[got.phase == "Phase 2"]
    p1 = got[got.phase == "Phase 1"]
    # The conclusion must not depend on the weighting.
    for w in ("macro", "micro"):
        assert (p2[f"contest_{w}_q"] < 0.001).all()
        assert (p2[f"contest_{w}_ci_lo"] > 0).all()
        assert (p1[f"contest_{w}_q"] > 0.5).all()
        # Every Phase 1 contest interval straddles zero, in both directions.
        assert (p1[f"contest_{w}_ci_lo"] <= 0).all()
        assert (p1[f"contest_{w}_ci_hi"] >= 0).all()
        assert p1[f"contest_{w}_delta"].min() < 0


def test_survives_is_a_stated_criterion_not_a_point_estimate():
    """A positive worst-subset point estimate is not enough to "survive".

    This is the exact case that made the term ambiguous: a subset whose macro
    delta is small but positive, over few enough items that the interval covers
    zero, must count as NOT surviving.
    """
    from mv_post_train.stats.math_composite import survives

    class Row:
        worst_loo_macro_ci_lo = 0.4
        worst_loo_micro_ci_lo = 0.6
    assert survives(Row())

    Row.worst_loo_macro_ci_lo = -0.9   # point estimate could still be > 0
    assert not survives(Row())

    Row.worst_loo_macro_ci_lo, Row.worst_loo_micro_ci_lo = 0.4, -0.1
    assert not survives(Row())  # must hold under BOTH weightings


def test_loo_detail_is_emitted_for_every_subset():
    recs = {
        (MV, "OT3+SFT", "GSM8K"): _cell([1.0] * 100),
        (COMMA, "OT3+SFT", "GSM8K"): _cell([0.0] * 100),
        (MV, "OT3+SFT", "MATH500"): _cell([1.0] * 50 + [0.0] * 50),
        (COMMA, "OT3+SFT", "MATH500"): _cell([0.0] * 100),
        (MV, "OT3+SFT", "AMC23"): _cell([0.0] * 40),
        (COMMA, "OT3+SFT", "AMC23"): _cell([0.0] * 40),
    }
    loo: list = []
    composite_contrasts(recs, n_resamples=300, n_perm=300, loo_out=loo)
    assert {r["dropped"] for r in loo} == {"GSM8K", "MATH500", "AMC23"}
    assert all(r["n_tasks"] == 2 for r in loo)
    assert all(r["macro_ci_lo"] <= r["macro_delta"] <= r["macro_ci_hi"] for r in loo)


def test_phase1_leave_one_out_exposes_math500_dependence():
    """Phase 2 composites survive dropping any benchmark; two Phase 1 ones do
    not. Nemotron and SmolLM2 have no Phase 1 GSM8K, so their composites run on
    MATH500 plus three flat contest benchmarks -- remove MATH500 and nothing is
    left. This must stay visible rather than be averaged away.
    """
    got = _artifact()
    p2 = got[got.phase == "Phase 2"]
    assert p2.survives_loo.all()
    assert (p2.worst_loo_macro_ci_lo > 0).all()

    p1 = got[got.phase == "Phase 1"]
    thin = p1[p1.n_tasks == 4]
    assert len(thin) >= 2
    assert (thin.worst_loo_macro_dropped == "MATH500").all()
    assert (thin.worst_loo_micro_dropped == "MATH500").all()
    # None of the three thin Phase 1 composites survives, including the one
    # whose point estimate is positive (+1.1) but whose interval covers zero.
    assert not thin.survives_loo.any()
    assert thin.worst_loo_macro.max() > 0
