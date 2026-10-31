"""Pin the unified EvalParser contract.

Both upstream parsers (`Phase1Parser` for logs.zip, `Phase2Parser` for
the LAION HF dump) must conform to the same shape: an `EvalParser` subclass
that returns a `ParseResult` from `.parse()`, and a `write_to_data_dir`
that persists the result via three class attributes.

These tests ensure a future eval source can be added as a third subclass
without modifying any downstream code.
"""
from __future__ import annotations

import json
from pathlib import Path

import pandas as pd

from mv_post_train.data.io_logs import AggregateRow, ItemRow
from mv_post_train.data.parse_phase2 import Phase2Parser
from mv_post_train.data.parse_phase1 import Phase1Parser
from mv_post_train.data.parser import EvalParser, ParseResult


# --------------------------------------------------------------------------
# Contract-shape tests: both subclasses must conform to the same interface.

def test_both_parsers_inherit_eval_parser_base():
    assert issubclass(Phase1Parser, EvalParser)
    assert issubclass(Phase2Parser, EvalParser)


def test_both_parsers_declare_required_class_attributes():
    """Each subclass must declare parser_name / items_filename / emits_aggregates."""
    for cls in (Phase1Parser, Phase2Parser):
        assert isinstance(cls.parser_name, str) and cls.parser_name
        assert isinstance(cls.items_filename, str) and cls.items_filename.endswith(".parquet")
        assert isinstance(cls.emits_aggregates, bool)


def test_parsers_use_distinct_items_filenames():
    """Phase 1 / Phase 2 source separation: they must not write to the same file."""
    assert Phase1Parser.items_filename != Phase2Parser.items_filename


def test_logs_zip_parser_emits_aggregates_but_phase2_does_not():
    """logs.zip has the harness aggregate scalars (HumanEval / MBPP / IFE /
    gsm8k); the Phase 2 dump doesn't carry them."""
    assert Phase1Parser.emits_aggregates is True
    assert Phase2Parser.emits_aggregates is False


# --------------------------------------------------------------------------
# ParseResult schema

def test_parse_result_default_is_empty_tuples():
    r = ParseResult()
    assert r.item_rows == () and r.aggregate_rows == () and r.audit_rows == ()


def test_parse_result_accepts_typed_rows():
    item = ItemRow(model="m", family="f", phase="sft", context_tier="16k",
                   tokens_b=300.0, size_b=1.7,
                   task="AIME24", item_index=0, attempt_index=0, score=1.0)
    agg = AggregateRow(model="m", family="f", phase="sft", context_tier="16k",
                       tokens_b=300.0, size_b=1.7,
                       task="HumanEval", n=322, n_correct=129,
                       accuracy=0.4006, accuracy_mean=0.40,
                       components='{"python_pass@1": 0.5, "sh_pass@1": 0.3}',
                       source_metric="combined_python_sh_pass@1")
    r = ParseResult(item_rows=(item,), aggregate_rows=(agg,))
    assert len(r.item_rows) == 1 and len(r.aggregate_rows) == 1


def test_parse_result_is_frozen():
    """ParseResult should be immutable so a downstream consumer can't
    mutate the parser's output."""
    import pydantic
    r = ParseResult()
    with __import__("pytest").raises(pydantic.ValidationError):
        r.item_rows = ()  # type: ignore[misc]


# --------------------------------------------------------------------------
# End-to-end: parse() returns a valid ParseResult for both subclasses.

FIXTURE_MODEL = "ali-elganzory__1.7b-Comma0.1-300BT-WithChatTemplate"


def test_logs_zip_parser_parse_returns_parse_result(tmp_path: Path):
    """End-to-end: build a fixture logs/ tree, call parser.parse(), assert
    the returned object is a ParseResult with typed rows."""
    logs = tmp_path / "logs"
    md = logs / FIXTURE_MODEL
    md.mkdir(parents=True)
    (md / "results_2026-01-01T00-00-00.json").write_text(json.dumps({
        "model_name": "x",
        "results": {
            "AIME24": {"examples": [
                {"expected_answer": "12", "model_answers": ["12", "12"]},
            ]},
            "HumanEval": {"python_pass@1": 0.5, "sh_pass@1": 0.3},
        },
    }))
    result = Phase1Parser(logs).parse()
    assert isinstance(result, ParseResult)
    assert all(isinstance(r, ItemRow) for r in result.item_rows)
    assert all(isinstance(r, AggregateRow) for r in result.aggregate_rows)
    assert len(result.item_rows) == 2  # 2 AIME24 attempts
    assert len(result.aggregate_rows) == 1  # HumanEval


def test_write_to_data_dir_persists_via_class_attributes(tmp_path: Path):
    """Persistence layer is in the base class; subclasses just declare
    *where* to write via items_filename / emits_aggregates / audit_filename.
    Verify both file kinds land at the right basenames for Phase1Parser."""
    logs = tmp_path / "logs"
    md = logs / FIXTURE_MODEL
    md.mkdir(parents=True)
    (md / "results_2026-01-01T00-00-00.json").write_text(json.dumps({
        "model_name": "x",
        "results": {
            "AIME24": {"examples": [{"expected_answer": "1", "model_answers": ["1"]}]},
            "MBPP": {"pass@1": 0.20, "num_examples": 500},
        },
    }))
    data_dir = tmp_path / "data"
    art_dir = tmp_path / "art"
    rc = Phase1Parser(logs).write_to_data_dir(data_dir, art_dir)
    assert rc == 0
    assert (data_dir / "items_long.parquet").exists()
    assert (data_dir / "aggregates.parquet").exists()
    assert (art_dir / "audit" / "file_selection_log.csv").exists()
    # phase2_items.parquet must NOT be written by the logs.zip parser.
    assert not (data_dir / "phase2_items.parquet").exists()


def test_phase2_parser_writes_to_phase2_items(tmp_path: Path):
    """Build a minimal Phase 2 HF parquet and verify Phase2Parser
    routes its output to phase2_items.parquet (not items_long.parquet),
    and does NOT write aggregates.parquet."""
    hf = tmp_path / "eval_responses.parquet"
    # One MV AIME24 OT3+SFT row.
    df = pd.DataFrame([{
        "base_model": "MixtureVitae-1.7B",
        "stage": "OT3+SFT",
        "task": "AIME24",
        "example_idx": 0,
        "run_idx": 0,
        "gold_answer": "12",
        "model_answer": "12",
        "is_correct": True,
    }])
    df.to_parquet(hf, index=False)
    data_dir = tmp_path / "data"
    rc = Phase2Parser(hf).write_to_data_dir(data_dir)
    assert rc == 0
    assert (data_dir / "phase2_items.parquet").exists()
    # The Phase-2 parser doesn't emit aggregates; aggregates.parquet must
    # not be created by it (logs.zip parse is responsible for that).
    assert not (data_dir / "aggregates.parquet").exists()
    # And it must not collide with the Phase 1 items file.
    assert not (data_dir / "items_long.parquet").exists()

    out = pd.read_parquet(data_dir / "phase2_items.parquet")
    assert len(out) == 1
    assert out.iloc[0].phase == "ot30k_on_sft"
    assert out.iloc[0].task == "AIME24"
