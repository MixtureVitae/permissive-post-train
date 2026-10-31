"""How the older evaluations combine a checkpoint's repeated runs.

Several checkpoints of the older evaluations (logs.zip) were evaluated more
than once. The paper's old tables kept each score at its highest over the runs.
Tables 7, 9 and 27 now use the mean of the runs that are not broken
(`render/older_tables.py` and `mvpt parse-phase1 --mean-unbroken-runs`).

A run's value is computed as the paper's older script computes a task's score:
the mean of the run's score fields, leaving out METADATA_FIELDS. For HumanEval
that is the mean of its Python and shell scores, for IFEval the mean of its
prompt-level and instruction-level scores, and for GSM8K the flexible-extract
score.

A run is broken if its value is below BROKEN_BELOW points while another run of
the same checkpoint and task scores at least WORKING_AT_LEAST. Only HumanEval
has such runs: in the batch of 2026-04-15, some runs score near zero where the
same checkpoint's other runs do not, which points to failed code execution. A
checkpoint whose runs all score near zero keeps them all.
"""
from __future__ import annotations

#: Fields of a task's results that are not scores, as the paper's older script lists them.
METADATA_FIELDS = frozenset({
    "num_examples", "completion_rate", "examples", "num_total", "solved_avg", "run_stats",
    "accuracy_std_err", "num_repeat", "accuracy_easy_avg", "accuracy_easy_std_err",
    "accuracy_medium_avg", "accuracy_medium_std_err", "accuracy_hard_avg",
    "accuracy_hard_std_err", "raw_metrics", "num_solved", "alias",
    "exact_match,strict-match", "exact_match_stderr,strict-match",
    "exact_match_stderr,flexible-extract",
})

BROKEN_BELOW = 1.0
WORKING_AT_LEAST = 3.0


def run_value(payload: dict) -> float:
    """A run's value for one task, in points: the mean of its score fields."""
    scores = [float(v) for k, v in payload.items() if k not in METADATA_FIELDS]
    return 100.0 * sum(scores) / len(scores)


def broken(values: list[float]) -> list[bool]:
    """Which of a checkpoint's runs of one task are broken."""
    working = max(values) >= WORKING_AT_LEAST
    return [working and v < BROKEN_BELOW for v in values]
