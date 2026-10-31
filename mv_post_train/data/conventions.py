"""Which metric convention is this code in, and which one built that file?

HumanEval and IFEval are reported under two different definitions in this
project's history, and the choice moves the numbers a long way:

  submission (NeurIPS)  HumanEval = python_pass@1 + sh_pass@1 pooled, n=322
                        IFEval    = prompt-level + instruction-level, n=1269
  arxiv                 HumanEval = python_pass@1 only,             n=164
                        IFEval    = prompt-level (loose) only,      n=500

Switching from the first to the second moves HumanEval by +7.9 pp on average
(up to +29.8) and IFEval by -6.1 pp on every model, which shifts the 11-task
suite Avg by >=0.5 pp for 54 of 93 Phase 1 rows. `docs/neurips-vs-arxiv-diff.md`
documents the change as it appears in the two manuscripts.

Two numbers under the same column name is the failure this module exists to
prevent. It does three things:

1. **Derives** the active convention from `SPECS` rather than storing it
   separately, so editing HumanEval's or IFEval's spec re-labels every artifact
   automatically and a label cannot drift away from the code that produced it.
2. **Stamps** it onto each aggregate row, so a parquet carries the definition it
   was built under.
3. **Refuses** to analyse rows whose stamp disagrees with the running code, or a
   file that mixes stamps -- see `check_frame`.

An unrecognised spec combination raises rather than defaulting: a new
convention is a decision that has to be named here and reflected in the docs,
not something to infer from whatever the specs happen to say.
"""
from __future__ import annotations

from enum import Enum

import pandas as pd


class MetricConvention(str, Enum):
    """The metric definition a HumanEval / IFEval number was computed under."""
    SUBMISSION = "submission"   # NeurIPS submission: combined HE, combined IFEval
    ARXIV = "arxiv"             # arXiv revision: python-only HE, prompt-level IFEval


#: Tasks whose reported value depends on the convention. Everything else
#: (MBPP, gsm8k, per-item reasoning tasks, the whole language suite) is defined
#: identically under both, and its rows are stamped `NOT_APPLICABLE`.
CONVENTION_SENSITIVE_TASKS: frozenset[str] = frozenset({"HumanEval", "IFEval"})

#: Stamp for rows whose task is defined the same way under every convention.
NOT_APPLICABLE: str = "n/a"

#: Column carrying the stamp on `aggregates.parquet` and the artifact CSVs.
STAMP_COLUMN: str = "metric_convention"

# (task -> (declared n, component metric fields)). Identity is the metric
# composition: that is what changes the meaning of the number. File-selection
# policy (max vs. mean across rerun files) rides along with the arXiv change but
# is a separable scoring choice, so it is deliberately not part of the key.
_SIGNATURES: dict[tuple, MetricConvention] = {
    (("HumanEval", 322, ("python_pass@1", "sh_pass@1")),
     ("IFEval", 1269, ("prompt-level", "instruction-level"))): MetricConvention.SUBMISSION,
    (("HumanEval", 164, ("python_pass@1",)),
     ("IFEval", 500, ("prompt-level",))): MetricConvention.ARXIV,
}


def _signature(specs: dict) -> tuple:
    """Fingerprint the convention-sensitive specs, in a stable order."""
    out = []
    for task in sorted(CONVENTION_SENSITIVE_TASKS):
        spec = specs[task]
        fields = tuple(c.metric_field for c in (spec.components or ()))
        out.append((task, spec.n, fields))
    return tuple(out)


def detect_convention(specs: dict) -> MetricConvention:
    """The convention `specs` encodes.

    Raises ValueError on an unrecognised combination: a third convention has to
    be named here (and in docs/statistics.md) before anything will emit under
    it, which is what stops a spec edit from silently redefining a column.
    """
    sig = _signature(specs)
    try:
        return _SIGNATURES[sig]
    except KeyError:
        raise ValueError(
            "HumanEval/IFEval specs match no known metric convention.\n"
            f"  got: {sig}\n"
            "If this is a deliberate new convention, add it to _SIGNATURES in "
            "data/conventions.py and document it in docs/statistics.md. If it "
            "is not, the specs have drifted -- check benchmarks.py."
        ) from None


def stamp_for(task: str, convention: MetricConvention) -> str:
    """The stamp a row for `task` should carry."""
    return convention.value if task in CONVENTION_SENSITIVE_TASKS else NOT_APPLICABLE


#: The convention this project has decided to report under. This is the
#: *intent*; `SPECS` is the *implementation*. Keeping both and checking they
#: agree is deliberate: it means switching convention is an explicit, greppable
#: edit here plus a spec change, and never a side effect of touching a spec.
#:
#: To switch (e.g. to adopt the arXiv definitions): set this to
#: `MetricConvention.ARXIV`, change the HumanEval / IFEval entries in
#: `benchmarks.SPECS` to match, update `docs/statistics.md`, and re-run
#: `mvpt parse-phase1` so the parquets are rebuilt and re-stamped. Note that
#: `mvpt seed-variance` cannot follow: its inputs are published grids scored
#: under `SUBMISSION`.
DECLARED_CONVENTION: MetricConvention = MetricConvention.SUBMISSION


def active_convention() -> MetricConvention:
    """The convention the running code emits under.

    Raises ValueError if `SPECS` and `DECLARED_CONVENTION` disagree — that
    means someone redefined HumanEval or IFEval without recording the decision,
    which is precisely the change that silently moves every reported number.
    Imported lazily so this module stays importable from `benchmarks` itself.
    """
    from .benchmarks import SPECS
    detected = detect_convention(SPECS)
    if detected is not DECLARED_CONVENTION:
        raise ValueError(
            f"HumanEval/IFEval specs encode the '{detected.value}' metric "
            f"convention, but DECLARED_CONVENTION is "
            f"'{DECLARED_CONVENTION.value}'.\n"
            "Changing these specs changes every reported HumanEval, IFEval and "
            "suite-Avg number, so it has to be a recorded decision: set "
            "DECLARED_CONVENTION in data/conventions.py, update "
            "docs/statistics.md, and re-run `mvpt parse-phase1`."
        )
    return detected


def check_frame(df: pd.DataFrame, source: str,
                expected: MetricConvention | None = None) -> MetricConvention:
    """Verify a loaded aggregate frame was built under the running convention.

    Returns the convention the frame carries. Raises ValueError if the frame
    mixes conventions, or disagrees with `expected` (default: the running
    code's). An unstamped frame -- one written before this guard existed --
    also raises, because its HumanEval and IFEval columns cannot be attributed
    to either definition after the fact.
    """
    expected = expected or active_convention()

    if STAMP_COLUMN not in df.columns:
        raise ValueError(
            f"{source}: no `{STAMP_COLUMN}` column, so its HumanEval / IFEval "
            "numbers cannot be attributed to a metric convention. It predates "
            "the convention guard -- re-run the parse step to rebuild it."
        )

    found = {s for s in df[STAMP_COLUMN].dropna().unique() if s != NOT_APPLICABLE}
    if len(found) > 1:
        raise ValueError(
            f"{source}: mixes metric conventions {sorted(found)} in one file. "
            "Rows computed under different HumanEval / IFEval definitions "
            "cannot be pooled or compared -- rebuild from a single source."
        )
    if not found:
        return expected  # only convention-independent tasks present

    got = MetricConvention(found.pop())
    if got != expected:
        raise ValueError(
            f"{source}: built under the '{got.value}' metric convention, but "
            f"this code emits '{expected.value}'. HumanEval and IFEval mean "
            "different things in the two (see docs/statistics.md), so the "
            "numbers are not comparable. Re-run the parse step to rebuild the "
            "parquet under the current specs."
        )
    return got
