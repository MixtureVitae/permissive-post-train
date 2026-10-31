"""Vendored HumanEval / MBPP execution harness.

Copied from evalchemy, whose execution path is in turn OpenAI's `human_eval`
harness (`human_eval/execution.py`, `data.py`, `evaluation.py`) plus
evalchemy's own `utils.extract_generation_code`. Imports are rewritten as
relative ones so the package is self-contained; the scoring and extraction
logic is unmodified.

It is vendored rather than reimplemented because the extraction conventions
are where the numbers come from — see the `stats/code_score.py` docstring for
which quirks are deliberately preserved.

`data/` holds the benchmark problem files (`humaneval-python.jsonl`,
`mbpp.jsonl`, `mbpp_test.jsonl`) the harness scores against.

**Running this executes model-generated code.** `human_eval.execution` forks a
process per candidate under `reliability_guard()`, but that is the benchmark's
standard sandbox, not a substitute for a container.
"""
