"""The metric-convention guard.

HumanEval and IFEval are defined differently in the NeurIPS submission and the
arXiv revision, and the gap is large (HumanEval moves +7.9 pp on average,
IFEval -6.1 pp on every model). These tests pin the three properties that stop
the two definitions being mixed under one column name:

  - the active convention is DERIVED from the specs, so a spec edit relabels
    artifacts instead of leaving a stale label behind,
  - an unrecognised spec combination raises rather than defaulting,
  - a frame stamped with the other convention is refused at load time.
"""
from __future__ import annotations

import copy

import pandas as pd
import pytest

from mv_post_train.data import conventions
from mv_post_train.data.benchmarks import SPECS
from mv_post_train.data.conventions import (
    MetricConvention,
    NOT_APPLICABLE,
    STAMP_COLUMN,
    check_frame,
    detect_convention,
    stamp_for,
)


def _arxiv_specs() -> dict:
    """SPECS as PR #1 ("HumanEval and IFEval fixes") would leave them:
    HumanEval python-only at n=164, IFEval prompt-level-only at n=500."""
    specs = copy.deepcopy(SPECS)
    he, ife = specs["HumanEval"], specs["IFEval"]
    specs["HumanEval"] = he.model_copy(update={
        "n": 164, "components": (he.components[0],)})
    specs["IFEval"] = ife.model_copy(update={
        "n": 500, "components": (ife.components[0],)})
    return specs


def test_current_specs_are_the_submission_convention():
    assert detect_convention(SPECS) is MetricConvention.SUBMISSION


def test_arxiv_specs_are_detected_without_touching_a_label():
    """Switching the specs must switch the detected convention on its own —
    this is what keeps a stamp from drifting away from the code."""
    assert detect_convention(_arxiv_specs()) is MetricConvention.ARXIV


def test_unknown_spec_combination_raises_rather_than_defaulting():
    specs = copy.deepcopy(SPECS)
    he = specs["HumanEval"]
    specs["HumanEval"] = he.model_copy(update={"n": 999})
    with pytest.raises(ValueError, match="no known metric convention"):
        detect_convention(specs)


def test_only_convention_sensitive_tasks_are_stamped():
    assert stamp_for("HumanEval", MetricConvention.SUBMISSION) == "submission"
    assert stamp_for("IFEval", MetricConvention.ARXIV) == "arxiv"
    # MBPP and gsm8k are identical under both conventions.
    assert stamp_for("MBPP", MetricConvention.SUBMISSION) == NOT_APPLICABLE
    assert stamp_for("gsm8k", MetricConvention.ARXIV) == NOT_APPLICABLE


def _frame(*stamps: str) -> pd.DataFrame:
    return pd.DataFrame({"task": ["HumanEval"] * len(stamps),
                         STAMP_COLUMN: list(stamps)})


def test_check_frame_accepts_matching_convention():
    got = check_frame(_frame("submission", "submission"), "aggregates.parquet",
                      expected=MetricConvention.SUBMISSION)
    assert got is MetricConvention.SUBMISSION


def test_check_frame_rejects_the_other_convention():
    """The core protection: analysing an arXiv-convention parquet with
    submission-convention code must fail loudly, not silently report a mixture."""
    with pytest.raises(ValueError, match="built under the 'arxiv'"):
        check_frame(_frame("arxiv"), "aggregates.parquet",
                    expected=MetricConvention.SUBMISSION)


def test_check_frame_rejects_a_mixed_file():
    with pytest.raises(ValueError, match="mixes metric conventions"):
        check_frame(_frame("submission", "arxiv"), "aggregates.parquet",
                    expected=MetricConvention.SUBMISSION)


def test_check_frame_rejects_an_unstamped_frame():
    """A parquet written before the guard existed cannot have its HumanEval
    and IFEval columns attributed after the fact."""
    df = pd.DataFrame({"task": ["HumanEval"], "accuracy": [0.4]})
    with pytest.raises(ValueError, match="predates the convention guard"):
        check_frame(df, "aggregates.parquet")


def test_convention_independent_rows_alone_are_accepted():
    """A frame carrying only MBPP/gsm8k rows says nothing about the convention
    and must not be rejected for it."""
    df = pd.DataFrame({"task": ["MBPP", "gsm8k"],
                       STAMP_COLUMN: [NOT_APPLICABLE, NOT_APPLICABLE]})
    assert check_frame(df, "aggregates.parquet",
                       expected=MetricConvention.ARXIV) is MetricConvention.ARXIV


def test_emitted_rows_carry_the_active_stamp():
    from mv_post_train.data.io_logs import emit_aggregate_row
    meta = ("MV", "sft", "16k", 300.0, 1.7)
    he = list(emit_aggregate_row("m", meta, "HumanEval",
                                 {"python_pass@1": 0.5, "sh_pass@1": 0.3}))[0]
    mbpp = list(emit_aggregate_row("m", meta, "MBPP", {"pass@1": 0.3}))[0]
    assert he.metric_convention == conventions.active_convention().value
    assert mbpp.metric_convention == NOT_APPLICABLE


def test_changing_specs_without_recording_the_decision_raises(monkeypatch):
    """PR #1's exact change: redefining HumanEval and IFEval in SPECS without
    setting DECLARED_CONVENTION must fail loudly. This is the forcing function
    that turns a silent numeric shift into an explicit decision."""
    import mv_post_train.data.benchmarks as benchmarks
    monkeypatch.setattr(benchmarks, "SPECS", _arxiv_specs())
    with pytest.raises(ValueError, match="has to be a recorded decision"):
        conventions.active_convention()


def test_declared_and_specs_agree_on_this_branch():
    """The invariant itself: intent and implementation must not drift apart."""
    assert detect_convention(SPECS) is conventions.DECLARED_CONVENTION
