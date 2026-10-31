"""Vendored IFEval instruction checker.

Verbatim copy of `instructions.py`, `instructions_registry.py` and
`instructions_util.py` from google-research/instruction_following_eval, with
one mechanical change: the absolute imports (`from instruction_following_eval
import instructions_util`) are rewritten as relative ones so the package is
self-contained. No checking logic is modified.

Vendored rather than pip-installed because the upstream project ships no
release to PyPI, and the analyses need a scoring path that is pinned and
reproducible from this repo alone. `evaluation_lib.py` is deliberately *not*
vendored — its strict/loose drivers key responses off a
`prompt -> response` dict, and the prompt text in the released response dump
does not always match the dataset prompt byte-for-byte. `stats/ifeval_score.py`
reimplements those two functions against a response string instead, preserving
their semantics exactly (see the comments there).
"""
from __future__ import annotations

from . import instructions, instructions_registry, instructions_util

__all__ = ["instructions", "instructions_registry", "instructions_util"]
