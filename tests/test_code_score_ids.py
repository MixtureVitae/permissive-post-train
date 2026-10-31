"""Tests for the HumanEval id join.

The dump and the vendored problem file spell the same 164 problems two
different ways. Getting this wrong does not raise -- it silently drops rows
until the coverage gate reports the data as missing, which is exactly what
happened before this join existed.
"""
from __future__ import annotations

import pytest

from mv_post_train.stats import code_score as cs


@pytest.fixture(scope="module")
def problems():
    return cs._load_problems(cs.HUMANEVAL_PROBLEMS)


def test_the_problem_file_uses_python_slash_keys(problems):
    assert "Python/0" in problems
    assert len(problems) == 164


@pytest.mark.parametrize("raw,expected", [
    ("HumanEval_0_has_close_elements", "Python/0"),
    ("HumanEval_52_below_threshold", "Python/52"),
    ("HumanEval_139_special_factorial", "Python/139"),
    ("HumanEval_163_generate_integers", "Python/163"),
])
def test_dump_ids_map_onto_problem_keys(problems, raw, expected):
    assert cs.normalize_humaneval_id(raw, problems) == expected


def test_a_key_already_in_problem_form_passes_through(problems):
    assert cs.normalize_humaneval_id("Python/3", problems) == "Python/3"


@pytest.mark.parametrize("raw", ["nonsense", "MBPP_3", "", None, "HumanEval_9999_x"])
def test_unmappable_ids_return_none(problems, raw):
    assert cs.normalize_humaneval_id(raw, problems) is None


def test_a_name_mismatch_raises_rather_than_scoring_the_wrong_problem(problems):
    """A renumbered release must fail loudly, not produce a plausible number."""
    with pytest.raises(ValueError, match="id mismatch"):
        cs.normalize_humaneval_id("HumanEval_0_some_other_function", problems)


def test_every_dump_style_id_maps_and_agrees_on_the_function_name(problems):
    """All 164 map, and each one's embedded name matches the problem text."""
    mapped = set()
    for i in range(164):
        key = f"Python/{i}"
        name = cs._function_name(problems[key]["prompt"])
        assert name, f"no function found in {key}"
        assert cs.normalize_humaneval_id(f"HumanEval_{i}_{name}", problems) == key
        mapped.add(key)
    assert len(mapped) == 164


# --- the extractable-code gate ---------------------------------------------

def _rows(problems, n, with_code):
    """n HumanEval rows, `with_code` of them containing an actual program."""
    out = []
    for i in range(n):
        body = (f"```python\ndef f{i}():\n    return {i}\n```"
                if i < with_code else
                "Let me think about this. " * 50)
        out.append({"example_id": f"HumanEval_{i}_x", "output": body})
    return out


def _fallback_extractor(sample, lang_code):
    """evalchemy's real behaviour when it finds no ```python block.

    It returns `prompt + output`. That matters: the prompt carries the
    problem's own `def`, so a gate that inspects the *generation* sees a
    program even when the model wrote none. Mocking this as a passthrough
    hides the bug it is meant to catch.
    """
    return {"generation": sample["prompt"] + sample["output"]}


def test_a_cell_of_reasoning_traces_is_not_scored(tmp_path, problems, monkeypatch):
    """Coverage can pass while the responses contain no code at all.

    This is the HumanEval failure in the release: every problem present, almost
    no programs. Scoring it yields ~0.6%, which reads as a real measurement.
    """
    monkeypatch.setattr(cs, "normalize_humaneval_id",
                        lambda raw, probs: f"Python/{str(raw).split('_')[1]}")
    monkeypatch.setattr(cs, "extract_generation_code", _fallback_extractor)
    rows = _rows(problems, 164, with_code=7)
    acc, matched, expected, status = cs.score_cell(
        "HumanEval", rows, problems, tmp_path, 10.0, 1)
    assert acc is None
    assert status == "no_extractable_code"
    assert matched == 164          # coverage was never the problem


def test_a_cell_of_real_programs_passes_the_gate(tmp_path, problems, monkeypatch):
    monkeypatch.setattr(cs, "normalize_humaneval_id",
                        lambda raw, probs: f"Python/{str(raw).split('_')[1]}")
    monkeypatch.setattr(cs, "extract_generation_code",
                        lambda sample, lang_code: {"generation": sample["output"]})
    monkeypatch.setattr(cs, "evaluate_functional_correctness",
                        lambda **kw: {"pass@1": 0.42})
    rows = _rows(problems, 164, with_code=164)
    acc, _, _, status = cs.score_cell("HumanEval", rows, problems, tmp_path, 10.0, 1)
    assert status == "scored"
    assert acc == pytest.approx(42.0)
