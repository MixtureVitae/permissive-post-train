"""Shared benchmark identity and per-benchmark metric allowlist.

This module is the canonical source for:
  - The set of benchmark names the pipeline understands (PER_ITEM_TASKS,
    AGG_TASKS, REASONING_TASKS).
  - The per-benchmark allowlist of which raw lm-eval-harness metric fields
    feed each benchmark's score (SCORE_METRICS).
  - The HF id <-> filesystem id bridge (hf_id_normalize, hf_id_to_path).

Both the stat-sig pipeline (mv_post_train/io_logs.py) and the paper-table
renderer consume these definitions, so they cannot drift. New harness noise
fields added upstream cannot leak into the score average — they simply
aren't in SCORE_METRICS and are ignored.

Long benchmark names are canonical at the data layer. The short-name display
aliases (A24, A25, AMC, ...) live in the render layer (render/common.py),
where they belong.
"""
from __future__ import annotations

# ---------------------------------------------------------------------------
# Canonical benchmark names.
#
# Every other module that needs to refer to a benchmark by name imports one
# of these constants — direct string literals are a refactor hazard (typos,
# capitalisation drift, the gsm8k-vs-GSM8K trap). The constant's value IS
# the string the lm-eval-harness JSON / parquets / CSVs use.

# Per-item benchmarks: harness emits per-example records under `examples`.
AIME24:        str = "AIME24"
AIME25:        str = "AIME25"
AMC23:         str = "AMC23"
MATH500:       str = "MATH500"
GPQA_DIAMOND:  str = "GPQADiamond"
JEEBENCH:      str = "JEEBench"
LIVECODEBENCH: str = "LiveCodeBench"

# Aggregate-only benchmarks: harness records a single scalar per (model, task).
HUMANEVAL:     str = "HumanEval"
MBPP:          str = "MBPP"
IFEVAL:        str = "IFEval"
GSM8K:         str = "gsm8k"  # lowercase is the canonical lm-eval-harness key

# Standard lm-eval-harness language-understanding suite. Names are the
# canonical task keys the harness emits in `results[task]`.
ARC_CHALLENGE:   str = "arc_challenge"
ARC_EASY:        str = "arc_easy"
BOOLQ:           str = "boolq"
COMMONSENSE_QA:  str = "commonsense_qa"
COPA:            str = "copa"
HELLASWAG:       str = "hellaswag"
LAMBADA_OPENAI:  str = "lambada_openai"
MMLU:            str = "mmlu"
OPENBOOKQA:      str = "openbookqa"
PIQA:            str = "piqa"
WINOGRANDE:      str = "winogrande"
SOCIAL_IQA:      str = "social_iqa"

# Raw harness metric-field names (the values inside `results[task]`).
ACCURACY_AVG:           str = "accuracy_avg"
PYTHON_PASS_AT_1:       str = "python_pass@1"
SH_PASS_AT_1:           str = "sh_pass@1"
PASS_AT_1:              str = "pass@1"
PROMPT_LEVEL:           str = "prompt-level"
INSTRUCTION_LEVEL:      str = "instruction-level"
EXACT_MATCH_STRICT:     str = "exact_match,strict-match"
ACC_NONE:               str = "acc,none"       # lm-eval-harness aggregate accuracy field
ACC_NORM_NONE:          str = "acc_norm,none"  # length-normalised accuracy (preferred for ARC/HellaS/OBQA/PIQA)


# Tasks for which the harness emits per-example records under "examples".
# Their per-item score arrays drive the paired item-level bootstrap.
PER_ITEM_TASKS: frozenset[str] = frozenset(
    {AIME24, AIME25, AMC23, MATH500, GPQA_DIAMOND, JEEBENCH, LIVECODEBENCH}
)

# Tasks where the harness records only a single scalar per (model, task).
# These get Newcombe diff CIs at the stat-sig layer and mean-of-fractions
# at the paper-table layer.
REASONING_AGG_TASKS: frozenset[str] = frozenset({HUMANEVAL, MBPP, IFEVAL, GSM8K})

# Standard lm-eval-harness language suite (decontaminated). All are
# aggregate-only: the harness records a single `acc,none` scalar per
# (model, task). MMLU is the macro average across 57 subtask accuracies
# that the harness bundles into one results JSON; we read the top-level
# `mmlu` row directly and ignore the per-subtask rows (they're not in SPECS,
# so `_is_valid_record` drops them silently).
LANGUAGE_TASKS: frozenset[str] = frozenset({
    ARC_CHALLENGE, ARC_EASY, BOOLQ, COMMONSENSE_QA, COPA, HELLASWAG,
    LAMBADA_OPENAI, MMLU, OPENBOOKQA, PIQA, WINOGRANDE, SOCIAL_IQA,
})

AGG_TASKS: frozenset[str] = REASONING_AGG_TASKS | LANGUAGE_TASKS

# The 11-task reasoning suite that defines the main-paper "Avg" column.
# HumanEval and IFEval enter as combined metrics (see SCORE_METRICS).
REASONING_TASKS: tuple[str, ...] = (
    AIME24, AIME25, AMC23, MATH500, GPQA_DIAMOND, JEEBENCH,
    HUMANEVAL, LIVECODEBENCH, MBPP, IFEVAL, GSM8K,
)

# The 11-task language suite that defines the paper's language "Avg" column.
# `social_iqa` is collected by the harness but not used by the paper.
LANGUAGE_SUITE_TASKS: tuple[str, ...] = (
    ARC_CHALLENGE, ARC_EASY, BOOLQ, COMMONSENSE_QA, COPA, HELLASWAG,
    LAMBADA_OPENAI, MMLU, OPENBOOKQA, PIQA, WINOGRANDE,
)

# Short column-header aliases (the paper's table convention).
LANGUAGE_BENCHMARK_ABBREVIATIONS: dict[str, str] = {
    ARC_CHALLENGE:  "ARC-C",
    ARC_EASY:       "ARC-E",
    BOOLQ:          "BoolQ",
    COMMONSENSE_QA: "CSQA",
    COPA:           "COPA",
    HELLASWAG:      "HellaS",
    LAMBADA_OPENAI: "LAMBADA",
    MMLU:           "MMLU",
    OPENBOOKQA:     "OBQA",
    PIQA:           "PIQA",
    WINOGRANDE:     "WinoG",
}

# ---------------------------------------------------------------------------
# Per-task descriptors: schema + sizes + scoring in one place.
#
# Each benchmark gets a single BenchmarkSpec entry below. Everything that
# depends on per-task knowledge (validity check in io_logs, aggregate row
# construction, per-item scoring, display label + n_items in the statistics
# appendix) reads from SPECS.
#
# Per-item scoring lives on `spec.scorer` (a `Scorer` instance from
# `data/scorers.py`). The scorer owns its example schema (a pydantic
# class) and its score function, so io_logs / parse_phase2 never
# guess field names or branch on a task-kind enum.
#
# Pydantic enforces the invariants at module load: aggregate tasks must
# have components whose `n` sums to the benchmark's canonical `n`; per-item
# tasks must declare a `scorer`. A misconfigured spec raises ValidationError
# before any data is read.

from enum import Enum
from typing import Literal

from pydantic import BaseModel, ConfigDict, Field, field_validator, model_validator

from .scorers import (
    JEEBenchScorer,
    LiveCodeBenchScorer,
    MathSingleScorer,
    MultiAttemptScorer,
    Scorer,
)


class FileSelectionRule(str, Enum):
    """Per-task best-file selection rule (consumed by io_logs.select_results_files)."""
    BY_N_ITEMS = "n_items"                       # per-item: largest examples list wins
    BY_MAX_METRIC = "max_metric"                 # HumanEval: max selection_metric value
    BY_NUM_EXAMPLES = "num_examples"             # MBPP, IFEval: largest num_examples wins
    BY_TIMESTAMP_ONLY = "ts_only"                # gsm8k: latest timestamp only


class AggregateComponent(BaseModel):
    """One sub-metric of an aggregate-only benchmark.

    For HumanEval: (python_pass@1, n=164) and (sh_pass@1, n=158).
    """
    model_config = ConfigDict(frozen=True)
    metric_field: str
    n: int


# ---------------------------------------------------------------------------
# Per-task raw-harness payload models (pydantic).
#
# Each one describes the expected shape of `results[task]` in an
# lm-eval-harness JSON file. _is_valid_record in io_logs.py validates by
# `model_validate(payload)`; ValidationError == invalid record.

class _HumanEvalPayload(BaseModel):
    model_config = ConfigDict(extra="allow", populate_by_name=True)
    python_pass_at_1: float = Field(alias=PYTHON_PASS_AT_1)
    sh_pass_at_1: float = Field(alias=SH_PASS_AT_1)


class _MBPPPayload(BaseModel):
    model_config = ConfigDict(extra="allow", populate_by_name=True)
    pass_at_1: float = Field(alias=PASS_AT_1)
    num_examples: int | None = None


class _IFEvalPayload(BaseModel):
    model_config = ConfigDict(extra="allow", populate_by_name=True)
    prompt_level: float = Field(alias=PROMPT_LEVEL)
    instruction_level: float = Field(alias=INSTRUCTION_LEVEL)
    num_examples: int | None = None


class _GSM8KPayload(BaseModel):
    model_config = ConfigDict(extra="allow", populate_by_name=True)
    exact_match_strict: float = Field(alias=EXACT_MATCH_STRICT)


class _LangAggregatePayload(BaseModel):
    """Standard lm-eval-harness language-suite results row.

    Every entry in the harness's `results[task]` for ARC / BoolQ / MMLU / etc.
    carries an `acc,none` scalar (plus a stderr); we only consume the accuracy.
    """
    model_config = ConfigDict(extra="allow", populate_by_name=True)
    acc_none: float = Field(alias=ACC_NONE)


class _PerItemPayload(BaseModel):
    """Per-item benchmarks all share the same outer payload schema: a
    non-empty `examples` list. Per-example shape lives on the task's
    `Scorer.Example` (in `data/scorers.py`), not here."""
    model_config = ConfigDict(extra="allow")
    examples: list[dict]

    @field_validator("examples")
    @classmethod
    def _nonempty(cls, v: list) -> list:
        if not v:
            raise ValueError("examples list is empty")
        return v


class BenchmarkSpec(BaseModel):
    """Single source of truth per benchmark."""
    model_config = ConfigDict(frozen=True, arbitrary_types_allowed=True)

    name: str
    display_name: str          # paper-table label
    abbreviation: str          # column header
    kind: Literal["per_item", "aggregate"]
    n: int                     # canonical sample size
    source_metric_label: str   # the `source_metric` value stored on aggregate rows

    # per-item-only: the Scorer carries the per-example pydantic schema +
    # the score function (and, for tasks that compare gold vs pred, the
    # compare() method used by parse_phase2).
    scorer: Scorer | None = None

    # aggregate-only
    components: tuple[AggregateComponent, ...] | None = None
    file_selection: FileSelectionRule = FileSelectionRule.BY_N_ITEMS
    selection_metric: str | None = None  # required when file_selection == BY_MAX_METRIC

    # payload validator: must be a BaseModel subclass
    payload_model: type[BaseModel]

    @model_validator(mode="after")
    def _consistency(self) -> "BenchmarkSpec":
        if self.kind == "aggregate":
            if not self.components:
                raise ValueError(f"{self.name}: aggregate task needs components")
            total = sum(c.n for c in self.components)
            if total != self.n:
                raise ValueError(
                    f"{self.name}: component n sums to {total}, not declared n={self.n}"
                )
            if self.file_selection == FileSelectionRule.BY_MAX_METRIC and not self.selection_metric:
                raise ValueError(
                    f"{self.name}: file_selection=BY_MAX_METRIC requires selection_metric"
                )
            if self.scorer is not None:
                raise ValueError(
                    f"{self.name}: aggregate task must not declare a scorer"
                )
        else:  # per_item
            if self.scorer is None:
                raise ValueError(f"{self.name}: per-item task needs a scorer")
            if self.components:
                raise ValueError(f"{self.name}: per-item task must not declare components")
            if self.file_selection != FileSelectionRule.BY_N_ITEMS:
                raise ValueError(
                    f"{self.name}: per-item task must use file_selection=BY_N_ITEMS"
                )
        return self


# The registry. One entry per benchmark; pydantic enforces invariants at
# module load — a typo in `n` or a missing gold_field raises ValidationError
# before any data is read.
SPECS: dict[str, BenchmarkSpec] = {
    AIME24: BenchmarkSpec(
        name=AIME24, display_name="AIME24", abbreviation="A24",
        kind="per_item", n=30, source_metric_label=ACCURACY_AVG,
        scorer=MultiAttemptScorer(gold_field="expected_answer", compare_kind="int_or_math"),
        payload_model=_PerItemPayload,
    ),
    AIME25: BenchmarkSpec(
        name=AIME25, display_name="AIME25", abbreviation="A25",
        kind="per_item", n=30, source_metric_label=ACCURACY_AVG,
        scorer=MultiAttemptScorer(gold_field="answer", compare_kind="int_or_math"),
        payload_model=_PerItemPayload,
    ),
    AMC23: BenchmarkSpec(
        name=AMC23, display_name="AMC23", abbreviation="AMC",
        kind="per_item", n=40, source_metric_label=ACCURACY_AVG,
        scorer=MultiAttemptScorer(gold_field="answer", compare_kind="int_or_math"),
        payload_model=_PerItemPayload,
    ),
    MATH500: BenchmarkSpec(
        name=MATH500, display_name="MATH500", abbreviation="M500",
        kind="per_item", n=500, source_metric_label=ACCURACY_AVG,
        scorer=MathSingleScorer(),
        payload_model=_PerItemPayload,
    ),
    GPQA_DIAMOND: BenchmarkSpec(
        name=GPQA_DIAMOND, display_name="GPQA-Diamond", abbreviation="GPQA",
        kind="per_item", n=198, source_metric_label=ACCURACY_AVG,
        scorer=MultiAttemptScorer(gold_field="answer", compare_kind="letter_equal"),
        payload_model=_PerItemPayload,
    ),
    JEEBENCH: BenchmarkSpec(
        name=JEEBENCH, display_name="JEEBench", abbreviation="JEE",
        kind="per_item", n=515, source_metric_label=ACCURACY_AVG,
        scorer=JEEBenchScorer(),
        payload_model=_PerItemPayload,
    ),
    LIVECODEBENCH: BenchmarkSpec(
        name=LIVECODEBENCH, display_name="LiveCodeBench", abbreviation="LCB",
        kind="per_item", n=511, source_metric_label=ACCURACY_AVG,
        scorer=LiveCodeBenchScorer(),
        payload_model=_PerItemPayload,
    ),
    HUMANEVAL: BenchmarkSpec(
        name=HUMANEVAL, display_name="HumanEval", abbreviation="HE",
        kind="aggregate", n=322,
        source_metric_label="combined_python_sh_pass@1",
        components=(
            AggregateComponent(metric_field=PYTHON_PASS_AT_1, n=164),
            AggregateComponent(metric_field=SH_PASS_AT_1, n=158),
        ),
        file_selection=FileSelectionRule.BY_MAX_METRIC,
        selection_metric=PYTHON_PASS_AT_1,
        payload_model=_HumanEvalPayload,
    ),
    MBPP: BenchmarkSpec(
        name=MBPP, display_name="MBPP", abbreviation="MBPP",
        kind="aggregate", n=500, source_metric_label=PASS_AT_1,
        components=(AggregateComponent(metric_field=PASS_AT_1, n=500),),
        file_selection=FileSelectionRule.BY_NUM_EXAMPLES,
        payload_model=_MBPPPayload,
    ),
    IFEVAL: BenchmarkSpec(
        name=IFEVAL, display_name="IFEval", abbreviation="IFE",
        kind="aggregate", n=1269,
        source_metric_label="mean_prompt_inst_loose",
        components=(
            AggregateComponent(metric_field=PROMPT_LEVEL, n=500),
            AggregateComponent(metric_field=INSTRUCTION_LEVEL, n=769),
        ),
        file_selection=FileSelectionRule.BY_NUM_EXAMPLES,
        payload_model=_IFEvalPayload,
    ),
    GSM8K: BenchmarkSpec(
        name=GSM8K, display_name="GSM8K", abbreviation="GSM8K",
        kind="aggregate", n=1319, source_metric_label=EXACT_MATCH_STRICT,
        components=(AggregateComponent(metric_field=EXACT_MATCH_STRICT, n=1319),),
        file_selection=FileSelectionRule.BY_TIMESTAMP_ONLY,
        payload_model=_GSM8KPayload,
    ),
    # ---- Standard lm-eval-harness language suite ------------------------
    # Each task has a single `acc,none` metric per benchmark; `n` is taken
    # from the harness `n-samples.<task>.effective` field (confirmed by
    # inspecting an actual results JSON). MMLU has no `n-samples` row because
    # it's a group of 57 subtasks; the standard MMLU test-set size is 14042
    # (sum of subtask sizes) and the harness reports the macro-mean accuracy.
    # ARC-Challenge / ARC-Easy / HellaSwag / OpenBookQA / PIQA: the
    # lm-eval-harness default-and-paper-preferred metric is `acc_norm,none`
    # (length-normalised accuracy), not raw `acc,none`. Both are present in
    # every results JSON; we read the normalised one.
    ARC_CHALLENGE: BenchmarkSpec(
        name=ARC_CHALLENGE, display_name="ARC-C", abbreviation="ARC-C",
        kind="aggregate", n=1172, source_metric_label=ACC_NORM_NONE,
        components=(AggregateComponent(metric_field=ACC_NORM_NONE, n=1172),),
        file_selection=FileSelectionRule.BY_TIMESTAMP_ONLY,
        payload_model=_LangAggregatePayload,
    ),
    ARC_EASY: BenchmarkSpec(
        name=ARC_EASY, display_name="ARC-E", abbreviation="ARC-E",
        kind="aggregate", n=2376, source_metric_label=ACC_NORM_NONE,
        components=(AggregateComponent(metric_field=ACC_NORM_NONE, n=2376),),
        file_selection=FileSelectionRule.BY_TIMESTAMP_ONLY,
        payload_model=_LangAggregatePayload,
    ),
    BOOLQ: BenchmarkSpec(
        name=BOOLQ, display_name="BoolQ", abbreviation="BoolQ",
        kind="aggregate", n=3270, source_metric_label=ACC_NONE,
        components=(AggregateComponent(metric_field=ACC_NONE, n=3270),),
        file_selection=FileSelectionRule.BY_TIMESTAMP_ONLY,
        payload_model=_LangAggregatePayload,
    ),
    COMMONSENSE_QA: BenchmarkSpec(
        name=COMMONSENSE_QA, display_name="CSQA", abbreviation="CSQA",
        kind="aggregate", n=1221, source_metric_label=ACC_NONE,
        components=(AggregateComponent(metric_field=ACC_NONE, n=1221),),
        file_selection=FileSelectionRule.BY_TIMESTAMP_ONLY,
        payload_model=_LangAggregatePayload,
    ),
    COPA: BenchmarkSpec(
        name=COPA, display_name="COPA", abbreviation="COPA",
        kind="aggregate", n=100, source_metric_label=ACC_NONE,
        components=(AggregateComponent(metric_field=ACC_NONE, n=100),),
        file_selection=FileSelectionRule.BY_TIMESTAMP_ONLY,
        payload_model=_LangAggregatePayload,
    ),
    HELLASWAG: BenchmarkSpec(
        name=HELLASWAG, display_name="HellaSwag", abbreviation="HellaS",
        kind="aggregate", n=10042, source_metric_label=ACC_NORM_NONE,
        components=(AggregateComponent(metric_field=ACC_NORM_NONE, n=10042),),
        file_selection=FileSelectionRule.BY_TIMESTAMP_ONLY,
        payload_model=_LangAggregatePayload,
    ),
    LAMBADA_OPENAI: BenchmarkSpec(
        name=LAMBADA_OPENAI, display_name="LAMBADA", abbreviation="LAMBADA",
        kind="aggregate", n=5153, source_metric_label=ACC_NONE,
        components=(AggregateComponent(metric_field=ACC_NONE, n=5153),),
        file_selection=FileSelectionRule.BY_TIMESTAMP_ONLY,
        payload_model=_LangAggregatePayload,
    ),
    MMLU: BenchmarkSpec(
        name=MMLU, display_name="MMLU", abbreviation="MMLU",
        kind="aggregate", n=14042, source_metric_label=ACC_NONE,
        components=(AggregateComponent(metric_field=ACC_NONE, n=14042),),
        file_selection=FileSelectionRule.BY_TIMESTAMP_ONLY,
        payload_model=_LangAggregatePayload,
    ),
    OPENBOOKQA: BenchmarkSpec(
        name=OPENBOOKQA, display_name="OBQA", abbreviation="OBQA",
        kind="aggregate", n=500, source_metric_label=ACC_NORM_NONE,
        components=(AggregateComponent(metric_field=ACC_NORM_NONE, n=500),),
        file_selection=FileSelectionRule.BY_TIMESTAMP_ONLY,
        payload_model=_LangAggregatePayload,
    ),
    PIQA: BenchmarkSpec(
        name=PIQA, display_name="PIQA", abbreviation="PIQA",
        kind="aggregate", n=1838, source_metric_label=ACC_NORM_NONE,
        components=(AggregateComponent(metric_field=ACC_NORM_NONE, n=1838),),
        file_selection=FileSelectionRule.BY_TIMESTAMP_ONLY,
        payload_model=_LangAggregatePayload,
    ),
    WINOGRANDE: BenchmarkSpec(
        name=WINOGRANDE, display_name="WinoG", abbreviation="WinoG",
        kind="aggregate", n=1267, source_metric_label=ACC_NONE,
        components=(AggregateComponent(metric_field=ACC_NONE, n=1267),),
        file_selection=FileSelectionRule.BY_TIMESTAMP_ONLY,
        payload_model=_LangAggregatePayload,
    ),
    SOCIAL_IQA: BenchmarkSpec(
        name=SOCIAL_IQA, display_name="SocialIQA", abbreviation="SocialIQA",
        kind="aggregate", n=1954, source_metric_label=ACC_NONE,
        components=(AggregateComponent(metric_field=ACC_NONE, n=1954),),
        file_selection=FileSelectionRule.BY_TIMESTAMP_ONLY,
        payload_model=_LangAggregatePayload,
    ),
}


# Sanity check: SPECS must cover exactly PER_ITEM_TASKS ∪ AGG_TASKS.
def _check_spec_coverage() -> None:
    declared = set(SPECS.keys())
    expected = PER_ITEM_TASKS | AGG_TASKS
    if declared != expected:
        missing = expected - declared
        extra = declared - expected
        raise RuntimeError(
            f"SPECS/PER_ITEM_TASKS/AGG_TASKS disagree — "
            f"missing in SPECS: {missing}; extra in SPECS: {extra}"
        )


_check_spec_coverage()


# Per-benchmark metric-field allowlist. Derived from SPECS so the allowlist
# and the spec can't drift. For per-item tasks the harness records the score
# under ACCURACY_AVG (the result of example-level computation in the harness).
SCORE_METRICS: dict[str, tuple[str, ...]] = {
    name: (tuple(c.metric_field for c in spec.components)
            if spec.components else (ACCURACY_AVG,))
    for name, spec in SPECS.items()
}


def hf_id_normalize(hf_id: str) -> str:
    """Return the filesystem form of an HF id (`/` -> `__`).

    The eval logs directory uses `__` as the section separator because `/`
    is reserved on every filesystem; HuggingFace canonical ids use `/`.
    Use this when crossing from one convention to the other.
    """
    return hf_id.replace("/", "__")


def hf_id_denormalize(filesystem_id: str) -> str:
    """Inverse of hf_id_normalize: `__` -> `/`. Idempotent if the id has no `__`."""
    return filesystem_id.replace("__", "/", 1)
