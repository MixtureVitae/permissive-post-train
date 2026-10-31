"""Tests for Table 1, Phase 1 for all models (mv_post_train/render/zoo.py).

They pin that the claims the abstract makes in words about Table 1 hold for the
committed numbers, that the check stops when one of them fails, and that each
row of the committed Table 1 prints the macros of its family at the context
the caption names.
"""
from __future__ import annotations

from pathlib import Path

import pytest

from mv_post_train.paper_check import table_rows
from mv_post_train.render import zoo
from mv_post_train.render.macros import parse_macros

ROOT = Path(__file__).resolve().parents[1]
TABLES = ROOT / "results" / "zoo" / "tables"


def committed() -> dict[str, str]:
    return parse_macros(TABLES / "zoo_numbers.tex")[1]


def test_abstract_claims_hold_for_the_committed_numbers():
    zoo.check_claims(committed())


def test_check_stops_when_a_reference_scores_more_than_half_of_mixturevitae():
    values = committed()
    mv = float(values["mv.16k.dpo.avg"])
    values["comma.16k.dpo.avg"] = f"{mv / 2 + 1:.1f}"
    with pytest.raises(ValueError, match="more than twice"):
        zoo.check_claims(values)


def test_check_stops_when_qwen3_is_five_points_ahead():
    values = committed()
    key = f"qwen3.{zoo.table1_context(values, 'Qwen3')}.dpo.avg"
    values[key] = f"{float(values['mv.16k.dpo.avg']) + 5:.1f}"
    with pytest.raises(ValueError, match="within five"):
        zoo.check_claims(values)


def test_table1_rows_print_their_family_at_its_context():
    values = committed()
    rows = table_rows((TABLES / "phase1_zoo.tex").read_text())
    names = [name for _, block in zoo.TABLE1_BLOCKS for _, name in block]
    families = [family for _, block in zoo.TABLE1_BLOCKS for family, _ in block]
    assert [r.label for r in rows] == names
    abbr = ["a24", "a25", "amc", "m500", "gsm8k", "he", "lcb", "mbpp", "gpqa", "jee", "ife"]
    for family, row in zip(families, rows):
        stem = f"{zoo.FAMILY_KEY[family]}.{zoo.table1_context(values, family)}"
        expected = [values[f"{stem}.dpo.{t}"] for t in abbr]
        expected += [values[f"{stem}.{stage}.avg"] for stage in ("base", "sft", "dpo")]
        assert [cell[0] for cell in row.cells] == expected, family
