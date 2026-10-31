"""Per-task scorer classes — one source for each benchmark's `(example
schema, score function)` pair.

Each `BenchmarkSpec` in `data/benchmarks.py:SPECS` references a `Scorer`
instance. Two consumers call into it:

  - `io_logs.emit_item_rows` calls `spec.scorer.score_raw(raw_example_dict)`
    on each entry in the harness's `examples` list. The scorer validates
    the dict against its declared `Example` pydantic schema, then computes
    the per-attempt score list.

  - `parse_phase2._score_row` calls `spec.scorer.compare(gold, pred)`
    on the Phase 2 HF parquet's flat (gold_answer, model_answer) rows.
    Only `CompareScorer` subclasses (the ones that fundamentally compare
    gold vs predicted) implement this.

Replaces the old `score_kind` enum + giant `if/elif` switch in
`example_scores`. Now each task carries its own scoring strategy in a
typed object.
"""
from __future__ import annotations

import re
from abc import ABC, abstractmethod
from functools import lru_cache
from typing import Any, Literal

from pydantic import BaseModel, ConfigDict, Field, ValidationError


# ---------------------------------------------------------------------------
# Internal helpers (private)

_mv_parse = None
_mv_verify = None


def _load_math_verify():
    """Lazy-import math-verify (it's slow to load)."""
    global _mv_parse, _mv_verify
    if _mv_parse is None:
        from math_verify import parse as _p, verify as _v
        _mv_parse, _mv_verify = _p, _v
    return _mv_parse, _mv_verify


def _norm(s: object) -> str:
    """Strip whitespace + enclosing `$...$` or `\\[...\\]` LaTeX wrappers."""
    if s is None:
        return ""
    s = str(s).strip()
    s = re.sub(r"^\$+|\$+$", "", s).strip()
    s = re.sub(r"^\\\[|\\\]$", "", s).strip()
    return s


@lru_cache(maxsize=8192)
def _math_equiv(gold: str, pred: str) -> bool:
    """Symbolic math equivalence via math-verify (cached)."""
    p, v = _load_math_verify()
    try:
        g = p(r"\boxed{" + gold + r"}")
        q = p(r"\boxed{" + pred + r"}")
        return bool(v(g, q))
    except Exception:
        return False


# ---------------------------------------------------------------------------
# Abstract base

class Scorer(ABC):
    """Per-task scoring strategy. Each subclass owns a nested `Example`
    pydantic schema and an implementation of `score(example)`."""

    @property
    @abstractmethod
    def Example(self) -> type[BaseModel]:
        """The per-example pydantic schema. Validates one entry of the
        harness's `examples` list."""
        ...

    @abstractmethod
    def score(self, example: BaseModel) -> list[float]:
        """Return per-attempt correctness scores for one validated example."""
        ...

    def score_raw(self, raw_example: dict) -> list[float]:
        """Validate `raw_example` against the Example schema, then score.
        Returns `[]` on validation failure — bad records are dropped, not
        crashed (the harness occasionally produces partial records)."""
        try:
            ex = self.Example.model_validate(raw_example)
        except ValidationError:
            return []
        return self.score(ex)


class CompareScorer(Scorer):
    """Scorers that fundamentally compare a single (gold, pred) pair.

    Used by `parse_phase2._score_row`, which sees flat per-attempt
    rows from the HF parquet (one `gold_answer` + one `model_answer` per
    row). Not all scorers can do this — `JEEBenchScorer` and
    `LiveCodeBenchScorer` read pre-computed per-attempt scores, so there's
    no gold vs pred comparison to perform.
    """

    @abstractmethod
    def compare(self, gold: Any, pred: Any) -> int:
        """0/1 correctness for a single (gold, pred) pair."""
        ...


# ---------------------------------------------------------------------------
# Concrete scorers

class JEEBenchScorer(Scorer):
    """JEEBench stores per-attempt scores directly (with partial-credit
    fractions for multi-correct items). No gold/pred comparison happens
    here — we just read the list."""

    class Example(BaseModel):
        model_config = ConfigDict(extra="allow")
        score: list[float | None]

    def score(self, example: "JEEBenchScorer.Example") -> list[float]:
        return [float(x) if x is not None else 0.0 for x in example.score]


class LiveCodeBenchScorer(Scorer):
    """LiveCodeBench stores per-attempt boolean correctness."""

    class Example(BaseModel):
        model_config = ConfigDict(extra="allow")
        # The harness sometimes emits a single bool instead of a list.
        correctness: bool | list[bool] = False

    def score(self, example: "LiveCodeBenchScorer.Example") -> list[float]:
        c = example.correctness
        if isinstance(c, list):
            return [float(bool(x)) for x in c]
        return [float(bool(c))]


class MathSingleScorer(CompareScorer):
    """MATH500: single attempt per item under `model_answer`, with fallback
    to `model_answers` when the singular field is missing. Scoring uses
    math-verify for symbolic equivalence."""

    class Example(BaseModel):
        model_config = ConfigDict(extra="allow")
        answer: Any
        model_answer: str | None = None
        model_answers: list[str] = []

    def compare(self, gold: Any, pred: Any) -> int:
        g, p = _norm(gold), _norm(pred)
        if g == p:
            return 1
        return int(_math_equiv(g, p))

    def score(self, example: "MathSingleScorer.Example") -> list[float]:
        if example.model_answer is not None:
            return [float(self.compare(example.answer, example.model_answer))]
        return [float(self.compare(example.answer, a)) for a in example.model_answers]


class MultiAttemptScorer(CompareScorer):
    """Multi-attempt scoring: each example has a `gold` value (under a
    task-specific field name) and a list of `model_answers`. Used by
    AIME24 (`expected_answer`, integer-or-math), AIME25 + AMC23 (`answer`,
    integer-or-math), and GPQADiamond (`answer`, letter-equal).

    The per-instance `Example` is built at `__init__` time with the right
    pydantic alias for the gold field, so validation maps the harness's
    actual field name to a uniform `.gold` attribute on the scored object.
    """

    CompareKind = Literal["int_or_math", "letter_equal"]

    def __init__(self, gold_field: str, compare_kind: CompareKind) -> None:
        self.gold_field = gold_field
        self.compare_kind: "MultiAttemptScorer.CompareKind" = compare_kind

        # Build a per-instance Example schema that aliases the gold field
        # to a uniform `gold` attribute.
        gold_alias = gold_field

        class _Example(BaseModel):
            model_config = ConfigDict(populate_by_name=True, extra="allow")
            gold: Any = Field(alias=gold_alias)
            model_answers: list[str] = []

        self._Example = _Example

    @property
    def Example(self) -> type[BaseModel]:
        return self._Example

    def compare(self, gold: Any, pred: Any) -> int:
        g, p = _norm(gold), _norm(pred)
        if self.compare_kind == "int_or_math":
            # Strip leading zeros; integer compare; fall back to math-verify.
            try:
                return int(int(g) == int(p))
            except (ValueError, TypeError):
                pass
            if g == p:
                return 1
            return int(_math_equiv(g, p))
        # letter_equal (e.g., GPQADiamond): single letter A/B/C/D string equality
        return int(g == p)

    def score(self, example: BaseModel) -> list[float]:
        return [float(self.compare(example.gold, a)) for a in example.model_answers]
