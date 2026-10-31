"""Pin the pydantic invariants on BenchmarkSpec.

These tests guarantee the spec registry can't drift: aggregate tasks must
have components that sum to the declared `n`, per-item tasks must declare
a `scorer`, file-selection rules are mutually consistent. A misconfigured
spec must raise at module load — these tests pin that contract.
"""
from __future__ import annotations

import pytest
from pydantic import BaseModel, ValidationError

from mv_post_train.data.benchmarks import (
    AggregateComponent,
    BenchmarkSpec,
    FileSelectionRule,
    SCORE_METRICS,
    SPECS,
)
from mv_post_train.data.scorers import (
    JEEBenchScorer,
    LiveCodeBenchScorer,
    MathSingleScorer,
    MultiAttemptScorer,
    Scorer,
)


class _DummyPayload(BaseModel):
    pass


# ----------------------------------------------------------------------
# SPECS registry contract

def test_specs_contains_reasoning_and_language_benchmarks():
    reasoning = {
        "AIME24", "AIME25", "AMC23", "MATH500", "GPQADiamond", "JEEBench",
        "LiveCodeBench", "HumanEval", "MBPP", "IFEval", "gsm8k",
    }
    language = {
        "arc_challenge", "arc_easy", "boolq", "commonsense_qa", "copa",
        "hellaswag", "lambada_openai", "mmlu", "openbookqa", "piqa",
        "winogrande", "social_iqa",
    }
    assert set(SPECS.keys()) == reasoning | language
    assert len(SPECS) == 23


def test_specs_self_name_consistent():
    """Each spec's `name` field must match its key in SPECS."""
    for key, spec in SPECS.items():
        assert spec.name == key, f"SPECS[{key!r}].name == {spec.name!r}"


def test_aggregate_components_sum_to_n():
    for spec in SPECS.values():
        if spec.kind == "aggregate":
            total = sum(c.n for c in spec.components)
            assert total == spec.n, (
                f"{spec.name}: components sum to {total}, declared n={spec.n}"
            )


def test_humaneval_sizes_pinned():
    he = SPECS["HumanEval"]
    assert he.n == 322
    sizes = {c.metric_field: c.n for c in he.components}
    assert sizes == {"python_pass@1": 164, "sh_pass@1": 158}


def test_ifeval_sizes_pinned():
    ife = SPECS["IFEval"]
    assert ife.n == 1269
    sizes = {c.metric_field: c.n for c in ife.components}
    assert sizes == {"prompt-level": 500, "instruction-level": 769}


def test_gsm8k_sizes_pinned():
    gs = SPECS["gsm8k"]
    assert gs.n == 1319


def test_score_metrics_derived_from_specs():
    """SCORE_METRICS is a view over SPECS; spot-check the four aggregates."""
    assert SCORE_METRICS["HumanEval"] == ("python_pass@1", "sh_pass@1")
    assert SCORE_METRICS["IFEval"] == ("prompt-level", "instruction-level")
    assert SCORE_METRICS["MBPP"] == ("pass@1",)
    assert SCORE_METRICS["gsm8k"] == ("exact_match,strict-match",)


# ----------------------------------------------------------------------
# BenchmarkSpec validator contract — constructing a bad spec must raise

def test_aggregate_without_components_rejected():
    with pytest.raises(ValidationError, match="aggregate task needs components"):
        BenchmarkSpec(
            name="X", display_name="X", abbreviation="X",
            kind="aggregate", n=100, source_metric_label="x",
            payload_model=_DummyPayload,
        )


def test_aggregate_components_must_sum_to_n():
    with pytest.raises(ValidationError, match=r"component n sums to .*not declared n"):
        BenchmarkSpec(
            name="X", display_name="X", abbreviation="X",
            kind="aggregate", n=100, source_metric_label="x",
            components=(AggregateComponent(metric_field="m", n=50),),
            payload_model=_DummyPayload,
        )


def test_by_max_metric_requires_selection_metric():
    with pytest.raises(ValidationError, match="BY_MAX_METRIC requires selection_metric"):
        BenchmarkSpec(
            name="X", display_name="X", abbreviation="X",
            kind="aggregate", n=10, source_metric_label="x",
            components=(AggregateComponent(metric_field="m", n=10),),
            file_selection=FileSelectionRule.BY_MAX_METRIC,
            payload_model=_DummyPayload,
        )


def test_per_item_without_scorer_rejected():
    with pytest.raises(ValidationError, match="per-item task needs a scorer"):
        BenchmarkSpec(
            name="X", display_name="X", abbreviation="X",
            kind="per_item", n=10, source_metric_label="x",
            payload_model=_DummyPayload,
        )


def test_aggregate_must_not_declare_scorer():
    """Aggregate tasks (HumanEval/MBPP/IFEval/gsm8k) have no per-example
    records; declaring a scorer on them indicates a config mistake."""
    with pytest.raises(ValidationError, match="aggregate task must not declare a scorer"):
        BenchmarkSpec(
            name="X", display_name="X", abbreviation="X",
            kind="aggregate", n=10, source_metric_label="x",
            components=(AggregateComponent(metric_field="m", n=10),),
            scorer=LiveCodeBenchScorer(),
            payload_model=_DummyPayload,
        )


def test_per_item_must_not_declare_components():
    with pytest.raises(ValidationError, match="per-item task must not declare components"):
        BenchmarkSpec(
            name="X", display_name="X", abbreviation="X",
            kind="per_item", n=10, source_metric_label="x",
            scorer=LiveCodeBenchScorer(),
            components=(AggregateComponent(metric_field="m", n=10),),
            payload_model=_DummyPayload,
        )


def test_per_item_must_use_by_n_items_selection():
    with pytest.raises(ValidationError, match="per-item task must use file_selection=BY_N_ITEMS"):
        BenchmarkSpec(
            name="X", display_name="X", abbreviation="X",
            kind="per_item", n=10, source_metric_label="x",
            scorer=LiveCodeBenchScorer(),
            file_selection=FileSelectionRule.BY_NUM_EXAMPLES,
            payload_model=_DummyPayload,
        )


# ----------------------------------------------------------------------
# Each per-item spec is wired to the right Scorer type.

def test_each_per_item_spec_has_a_scorer():
    """Every per-item benchmark must carry a Scorer instance; every
    aggregate-only benchmark must have scorer=None."""
    for name, spec in SPECS.items():
        if spec.kind == "per_item":
            assert isinstance(spec.scorer, Scorer), f"{name}: missing scorer"
        else:
            assert spec.scorer is None, f"{name}: aggregate should have scorer=None"


def test_specs_wire_expected_scorer_types():
    """Spot-check the (task -> Scorer subclass) wiring is correct."""
    assert isinstance(SPECS["AIME24"].scorer, MultiAttemptScorer)
    assert isinstance(SPECS["GPQADiamond"].scorer, MultiAttemptScorer)
    assert isinstance(SPECS["MATH500"].scorer, MathSingleScorer)
    assert isinstance(SPECS["JEEBench"].scorer, JEEBenchScorer)
    assert isinstance(SPECS["LiveCodeBench"].scorer, LiveCodeBenchScorer)
    # AIME24 uses "expected_answer" as the gold field; others use "answer".
    assert SPECS["AIME24"].scorer.gold_field == "expected_answer"
    assert SPECS["AIME25"].scorer.gold_field == "answer"
    assert SPECS["AMC23"].scorer.gold_field == "answer"
    assert SPECS["GPQADiamond"].scorer.gold_field == "answer"
    # GPQADiamond uses letter_equal; the integer-or-math tasks use int_or_math.
    assert SPECS["GPQADiamond"].scorer.compare_kind == "letter_equal"
    for t in ("AIME24", "AIME25", "AMC23"):
        assert SPECS[t].scorer.compare_kind == "int_or_math"


# ----------------------------------------------------------------------
# Payload model contract — pydantic accepts a valid payload, rejects junk

def test_humaneval_payload_accepts_well_formed_dict():
    spec = SPECS["HumanEval"]
    obj = spec.payload_model.model_validate(
        {"python_pass@1": 0.5, "sh_pass@1": 0.3}
    )
    assert obj.python_pass_at_1 == 0.5
    assert obj.sh_pass_at_1 == 0.3


def test_humaneval_payload_rejects_missing_field():
    spec = SPECS["HumanEval"]
    with pytest.raises(ValidationError):
        spec.payload_model.model_validate({"python_pass@1": 0.5})  # sh_pass@1 missing


def test_per_item_payload_rejects_empty_examples_list():
    spec = SPECS["AIME24"]
    with pytest.raises(ValidationError, match="examples list is empty"):
        spec.payload_model.model_validate({"examples": []})


def test_gsm8k_payload_rejects_missing_field():
    spec = SPECS["gsm8k"]
    with pytest.raises(ValidationError):
        spec.payload_model.model_validate({"some_other_field": 0.5})
