"""Per-Scorer contract tests.

Each Scorer subclass must:
  - validate its nested `Example` schema (accept well-formed inputs,
    reject malformed ones),
  - produce the expected per-attempt score list from a validated example,
  - implement `compare(gold, pred)` if and only if it inherits CompareScorer.

These tests pin each scorer's behaviour so the schema + scoring decisions
can't drift apart.
"""
from __future__ import annotations

import pytest
from pydantic import ValidationError

from mv_post_train.data.scorers import (
    CompareScorer,
    JEEBenchScorer,
    LiveCodeBenchScorer,
    MathSingleScorer,
    MultiAttemptScorer,
    Scorer,
)


# ----------------------------------------------------------------------
# Hierarchy

def test_compare_scorer_is_a_scorer():
    assert issubclass(CompareScorer, Scorer)


def test_only_compare_scorers_have_compare_method():
    """JEEBench and LiveCodeBench read pre-computed scores — they can't
    compare gold vs pred, so they're NOT CompareScorers."""
    assert not isinstance(JEEBenchScorer(), CompareScorer)
    assert not isinstance(LiveCodeBenchScorer(), CompareScorer)
    assert isinstance(MathSingleScorer(), CompareScorer)
    assert isinstance(MultiAttemptScorer("answer", "int_or_math"), CompareScorer)


# ----------------------------------------------------------------------
# JEEBenchScorer

def test_jeebench_reads_score_list_with_partial_credit():
    s = JEEBenchScorer()
    ex = s.Example.model_validate({"score": [1.0, 0.5, None, 0.25]})
    assert s.score(ex) == [1.0, 0.5, 0.0, 0.25]


def test_jeebench_rejects_missing_score_field():
    s = JEEBenchScorer()
    with pytest.raises(ValidationError):
        s.Example.model_validate({"other": "stuff"})


def test_jeebench_score_raw_returns_empty_on_validation_error():
    """score_raw must NOT raise on a bad record — it returns []."""
    s = JEEBenchScorer()
    assert s.score_raw({"not_a_score_field": True}) == []


# ----------------------------------------------------------------------
# LiveCodeBenchScorer

def test_livecodebench_accepts_list_of_bools():
    s = LiveCodeBenchScorer()
    ex = s.Example.model_validate({"correctness": [True, False, True]})
    assert s.score(ex) == [1.0, 0.0, 1.0]


def test_livecodebench_accepts_single_bool_fallback():
    s = LiveCodeBenchScorer()
    ex = s.Example.model_validate({"correctness": True})
    assert s.score(ex) == [1.0]


# ----------------------------------------------------------------------
# MathSingleScorer (MATH500)

def test_math_single_uses_model_answer_when_present():
    s = MathSingleScorer()
    ex = s.Example.model_validate({"answer": "42", "model_answer": "42"})
    assert s.score(ex) == [1.0]


def test_math_single_falls_back_to_model_answers_list():
    s = MathSingleScorer()
    ex = s.Example.model_validate({"answer": "42", "model_answers": ["42", "0", "42"]})
    assert s.score(ex) == [1.0, 0.0, 1.0]


def test_math_single_compare_handles_string_equality():
    s = MathSingleScorer()
    assert s.compare("12", "12") == 1
    assert s.compare("12", "13") == 0


# ----------------------------------------------------------------------
# MultiAttemptScorer

def test_multi_attempt_aliases_gold_field_per_task():
    """AIME24's gold lives at `expected_answer`; other AIME tasks at `answer`.
    The scorer must populate `.gold` from the right field via alias."""
    s_aime24 = MultiAttemptScorer("expected_answer", "int_or_math")
    ex = s_aime24.Example.model_validate(
        {"expected_answer": "42", "model_answers": ["42", "0"]}
    )
    assert ex.gold == "42"
    assert s_aime24.score(ex) == [1.0, 0.0]


def test_multi_attempt_int_or_math_strips_leading_zeros():
    """AIME-style integer compare must treat '07' == '7'."""
    s = MultiAttemptScorer("answer", "int_or_math")
    assert s.compare("7", "07") == 1
    assert s.compare("7", "8") == 0


def test_multi_attempt_letter_equal_strict():
    """GPQADiamond: single-letter string equality, no math fallback."""
    s = MultiAttemptScorer("answer", "letter_equal")
    assert s.compare("A", "A") == 1
    assert s.compare("A", "a") == 0   # case-sensitive
    assert s.compare("A", "B") == 0


def test_multi_attempt_score_iterates_model_answers():
    s = MultiAttemptScorer("answer", "int_or_math")
    ex = s.Example.model_validate({"answer": "5", "model_answers": ["5", "5", "0"]})
    assert s.score(ex) == [1.0, 1.0, 0.0]


def test_multi_attempt_rejects_missing_gold():
    s = MultiAttemptScorer("answer", "int_or_math")
    with pytest.raises(ValidationError):
        s.Example.model_validate({"model_answers": ["5"]})  # no 'answer'


# ----------------------------------------------------------------------
# score_raw: end-to-end validate + score in one call

def test_score_raw_validates_then_scores():
    s = JEEBenchScorer()
    assert s.score_raw({"score": [1.0, 0.0]}) == [1.0, 0.0]


def test_score_raw_silently_drops_unrecognised_records():
    """A record that fails validation contributes zero rows — the harness
    occasionally produces partial records and we don't want one bad row
    to crash a whole parse."""
    s = MultiAttemptScorer("answer", "int_or_math")
    assert s.score_raw({"completely": "wrong"}) == []
