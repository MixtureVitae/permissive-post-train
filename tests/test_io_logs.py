"""Fixture-based tests for io_logs.py + parse_phase1.py.

Pins the lm-eval-harness results_*.json → tidy parquet contract so the Phase C
refactor cannot silently change the per-task selection rule or aggregate
emission. The selection conventions tested here:

- Per-item tasks: highest n_items wins; ties broken by latest timestamp.
- HumanEval: highest python_pass@1 wins (paper convention) — overrides timestamp.
- MBPP / IFEval: highest num_examples wins; ties broken by latest timestamp.
- gsm8k: latest timestamp wins (n_aggregate is always 0 for gsm8k).
- Malformed / empty payloads are dropped.
"""
from __future__ import annotations

import json
from pathlib import Path

import pandas as pd
import pytest

from mv_post_train.data import io_logs, parse_phase1


# A real registered model from mv_post_train/data/models.py — using its name so
# parse_phase1' lookup() succeeds end-to-end without registry surgery.
FIXTURE_MODEL = "ali-elganzory__1.7b-Comma0.1-300BT-WithChatTemplate"


def _write_results_json(model_dir: Path, ts: str, results: dict) -> Path:
    """Write a results_<ts>.json file in lm-eval-harness format."""
    path = model_dir / f"results_{ts}.json"
    path.write_text(json.dumps({"model_name": "fixture", "results": results}))
    return path


@pytest.fixture
def logs_root(tmp_path: Path) -> Path:
    """Build a synthetic logs/ tree for one model with full task coverage and
    a few intentional duplicate-file scenarios to exercise selection rules."""
    logs = tmp_path / "logs"
    model_dir = logs / FIXTURE_MODEL
    model_dir.mkdir(parents=True)

    # ---- Per-item task: AIME24. Two files, same n_items=3, different ts.
    # Latest timestamp must win.
    _write_results_json(model_dir, "2026-01-01T00-00-00", {
        "AIME24": {
            "examples": [
                {"expected_answer": "12", "model_answers": ["12", "11", "12"]},
                {"expected_answer": "7",  "model_answers": ["7", "7", "7"]},
                {"expected_answer": "5",  "model_answers": ["0", "0", "5"]},
            ],
        },
    })
    _write_results_json(model_dir, "2026-02-01T00-00-00", {
        "AIME24": {
            "examples": [
                {"expected_answer": "12", "model_answers": ["12", "12", "12"]},  # higher score
                {"expected_answer": "7",  "model_answers": ["7", "7", "7"]},
                {"expected_answer": "5",  "model_answers": ["5", "5", "5"]},
            ],
        },
    })

    # ---- HumanEval: max python_pass@1 wins, even if timestamp is older.
    _write_results_json(model_dir, "2026-03-01T00-00-00", {
        "HumanEval": {"python_pass@1": 0.10, "sh_pass@1": 0.05},
    })
    _write_results_json(model_dir, "2026-01-15T00-00-00", {
        "HumanEval": {"python_pass@1": 0.50, "sh_pass@1": 0.30},  # winner
    })

    # ---- MBPP: highest num_examples wins.
    _write_results_json(model_dir, "2026-01-10T00-00-00", {
        "MBPP": {"pass@1": 0.20, "num_examples": 500},  # winner (max num_examples)
    })
    _write_results_json(model_dir, "2026-02-10T00-00-00", {
        "MBPP": {"pass@1": 0.30, "num_examples": 100},
    })

    # ---- gsm8k: latest timestamp wins.
    _write_results_json(model_dir, "2026-01-20T00-00-00", {
        "gsm8k": {"exact_match,strict-match": 0.10},
    })
    _write_results_json(model_dir, "2026-04-20T00-00-00", {
        "gsm8k": {"exact_match,strict-match": 0.40},  # winner (latest ts)
    })

    # ---- Empty / malformed payloads — must be dropped.
    _write_results_json(model_dir, "2026-05-01T00-00-00", {
        "AIME25": {"examples": []},        # empty examples list
        "AMC23":  {"not_a_list": "oops"},  # examples missing
    })

    return logs


# ----------------------------------------------------------------------
# select_results_files

def test_select_per_item_prefers_latest_timestamp_on_tie(logs_root: Path):
    model_dir = logs_root / FIXTURE_MODEL
    chosen = io_logs.select_results_files(model_dir)
    assert "AIME24" in chosen
    f, payload = chosen["AIME24"]
    assert "2026-02-01" in f.name, f"expected later-ts AIME24 file, got {f.name}"
    # The winner has all 3 items correct on every attempt.
    assert payload["examples"][0]["model_answers"] == ["12", "12", "12"]


def test_select_humaneval_prefers_max_python_pass(logs_root: Path):
    model_dir = logs_root / FIXTURE_MODEL
    chosen = io_logs.select_results_files(model_dir)
    assert "HumanEval" in chosen
    f, payload = chosen["HumanEval"]
    # The Jan file has python_pass@1 = 0.50, the Mar file has 0.10.
    # Max-python rule must pick Jan even though Mar is later.
    assert "2026-01-15" in f.name, f"max-python_pass@1 should win, got {f.name}"
    assert payload["python_pass@1"] == 0.50


def test_select_humaneval_latest_run_ignores_the_best_score(logs_root: Path):
    chosen = io_logs.select_results_files(logs_root / FIXTURE_MODEL, latest_run=True)
    f, payload = chosen["HumanEval"]
    assert "2026-03" in f.name, f"latest run should win, got {f.name}"
    assert payload["python_pass@1"] == 0.10
    # Every other task is chosen as without the option.
    default = io_logs.select_results_files(logs_root / FIXTURE_MODEL)
    assert {t: p.name for t, (p, _) in chosen.items() if t != "HumanEval"} == \
        {t: p.name for t, (p, _) in default.items() if t != "HumanEval"}


def test_select_mbpp_prefers_max_num_examples(logs_root: Path):
    model_dir = logs_root / FIXTURE_MODEL
    chosen = io_logs.select_results_files(model_dir)
    f, payload = chosen["MBPP"]
    assert payload["num_examples"] == 500


def test_select_gsm8k_prefers_latest_timestamp(logs_root: Path):
    model_dir = logs_root / FIXTURE_MODEL
    chosen = io_logs.select_results_files(model_dir)
    f, payload = chosen["gsm8k"]
    assert "2026-04-20" in f.name


def test_select_drops_empty_or_malformed(logs_root: Path):
    model_dir = logs_root / FIXTURE_MODEL
    chosen = io_logs.select_results_files(model_dir)
    # AIME25 had empty `examples`, AMC23 had no `examples` — both must be absent.
    assert "AIME25" not in chosen
    assert "AMC23" not in chosen


def test_is_valid_record_per_item():
    ok, n_items, n_agg = io_logs._is_valid_record(
        "AIME24", {"examples": [{"x": 1}, {"x": 2}]}
    )
    assert ok and n_items == 2 and n_agg == 0


def test_is_valid_record_humaneval_requires_both_subsets():
    ok, _, _ = io_logs._is_valid_record(
        "HumanEval", {"python_pass@1": 0.5}
    )
    assert not ok, "HumanEval payload missing sh_pass@1 must be rejected"
    ok, _, _ = io_logs._is_valid_record(
        "HumanEval", {"python_pass@1": 0.5, "sh_pass@1": 0.4}
    )
    assert ok


def test_is_valid_record_rejects_non_dict():
    assert io_logs._is_valid_record("AIME24", "not a dict") == (False, 0, 0)


def test_is_valid_record_rejects_humaneval_missing_sh_pass():
    """HumanEval payload model requires both python_pass@1 and sh_pass@1;
    a payload with only python_pass@1 must be rejected via pydantic
    ValidationError (caught inside _is_valid_record)."""
    ok, n_items, n_agg = io_logs._is_valid_record(
        "HumanEval", {"python_pass@1": 0.5}
    )
    assert ok is False


def test_is_valid_record_rejects_mbpp_missing_pass_at_1():
    ok, _, _ = io_logs._is_valid_record("MBPP", {"num_examples": 500})
    assert ok is False


def test_is_valid_record_rejects_unknown_task():
    ok, _, _ = io_logs._is_valid_record("DefinitelyNotABenchmark", {"x": 1})
    assert ok is False


# ----------------------------------------------------------------------
# Row schemas: ItemRow + AggregateRow

def test_item_row_validates_required_fields():
    """ItemRow must declare every column persisted to items_long.parquet."""
    row = io_logs.ItemRow(
        model="m", family="f", phase="sft", context_tier="16k",
        tokens_b=300.0, size_b=1.7,
        task="AIME24", item_index=0, attempt_index=0, score=1.0,
    )
    d = row.model_dump()
    assert set(d.keys()) == {
        "model", "family", "phase", "context_tier", "tokens_b", "size_b",
        "task", "item_index", "attempt_index", "score",
    }
    assert isinstance(d["score"], float)
    assert isinstance(d["item_index"], int)


def test_item_row_allows_null_tokens_b():
    """Baguettotron / FW-Edu-ablation rows have tokens_b=None — must validate."""
    row = io_logs.ItemRow(
        model="m", family="Baguettotron", phase="base", context_tier="4k",
        tokens_b=None, size_b=0.32,
        task="AIME24", item_index=0, attempt_index=0, score=0.0,
    )
    assert row.tokens_b is None


def test_aggregate_row_validates_required_fields():
    row = io_logs.AggregateRow(
        model="m", family="f", phase="sft", context_tier="16k",
        tokens_b=300.0, size_b=1.7,
        task="HumanEval", n=322, n_correct=129,
        accuracy=0.4006, accuracy_mean=0.40,
        components='{"python_pass@1": 0.5, "sh_pass@1": 0.3}',
        source_metric="combined_python_sh_pass@1",
    )
    d = row.model_dump()
    assert set(d.keys()) == {
        "model", "family", "phase", "context_tier", "tokens_b", "size_b",
        "task", "n", "n_correct", "accuracy", "accuracy_mean",
        "components", "source_metric", "metric_convention",
    }
    assert isinstance(d["n"], int) and isinstance(d["accuracy"], float)
    # Defaulted when not passed, so existing construction sites keep working;
    # the parsers always set it explicitly (see data/conventions.py).
    assert d["metric_convention"] == "n/a"


def test_aggregate_row_rejects_missing_field():
    """A row missing `accuracy_mean` must fail validation, not silently pass."""
    import pytest
    from pydantic import ValidationError
    with pytest.raises(ValidationError):
        io_logs.AggregateRow(
            model="m", family="f", phase="sft", context_tier="16k",
            tokens_b=300.0, size_b=1.7,
            task="HumanEval", n=322, n_correct=129,
            accuracy=0.4006,
            # accuracy_mean missing
            components='{}',
            source_metric="x",
        )


def test_emit_item_rows_yields_itemrow_instances():
    family_meta = ("Comma", "base", "4k", 300.0, 1.7)
    payload = {"examples": [
        {"expected_answer": "12", "model_answers": ["12", "0"]},
    ]}
    rows = list(io_logs.emit_item_rows("test_model", family_meta, "AIME24", payload))
    assert len(rows) == 2  # 2 attempts
    assert all(isinstance(r, io_logs.ItemRow) for r in rows)
    assert rows[0].score == 1.0 and rows[1].score == 0.0


def test_emit_aggregate_row_yields_aggregaterow_instance():
    family_meta = ("MV", "sft", "16k", 300.0, 1.7)
    payload = {"python_pass@1": 0.5, "sh_pass@1": 0.3}
    rows = list(io_logs.emit_aggregate_row("test_model", family_meta, "HumanEval", payload))
    assert len(rows) == 1
    assert isinstance(rows[0], io_logs.AggregateRow)
    assert rows[0].n == 322
    assert rows[0].accuracy_mean == 0.40


# ----------------------------------------------------------------------
# Silent-error surfacing

def test_corrupt_json_emits_warning(tmp_path: Path, capsys):
    """A results_*.json with malformed JSON must produce a stderr warning,
    not be silently skipped."""
    model_dir = tmp_path / "logs" / FIXTURE_MODEL
    model_dir.mkdir(parents=True)
    (model_dir / "results_2026-01-01T00-00-00.json").write_text("{this is not json}")
    io_logs.select_results_files(model_dir)
    captured = capsys.readouterr()
    assert "corrupted JSON" in captured.err
    assert "results_2026-01-01T00-00-00.json" in captured.err


def test_top_level_results_not_dict_emits_warning(tmp_path: Path, capsys):
    """A JSON file with `results` missing or not-a-dict must warn."""
    model_dir = tmp_path / "logs" / FIXTURE_MODEL
    model_dir.mkdir(parents=True)
    (model_dir / "results_2026-01-01T00-00-00.json").write_text(
        json.dumps({"model_name": "x", "results": "not a dict"})
    )
    io_logs.select_results_files(model_dir)
    captured = capsys.readouterr()
    assert "results` is missing or not a dict" in captured.err


def test_known_task_with_bad_payload_emits_warning(tmp_path: Path, capsys):
    """A HumanEval payload missing sh_pass@1 is a real data problem —
    must warn before being rejected, not silently dropped."""
    model_dir = tmp_path / "logs" / FIXTURE_MODEL
    model_dir.mkdir(parents=True)
    (model_dir / "results_2026-01-01T00-00-00.json").write_text(json.dumps({
        "model_name": "x",
        "results": {"HumanEval": {"python_pass@1": 0.5}},  # sh_pass@1 missing
    }))
    io_logs.select_results_files(model_dir)
    captured = capsys.readouterr()
    assert "HumanEval" in captured.err and "rejected by payload schema" in captured.err


def test_unknown_task_stays_silent(tmp_path: Path, capsys):
    """An unknown task in the harness output (e.g., an extra benchmark we
    don't track) must NOT generate a warning — only known-task failures do."""
    model_dir = tmp_path / "logs" / FIXTURE_MODEL
    model_dir.mkdir(parents=True)
    (model_dir / "results_2026-01-01T00-00-00.json").write_text(json.dumps({
        "model_name": "x",
        "results": {"SomeFutureBench": {"score": 0.5}},
    }))
    io_logs.select_results_files(model_dir)
    captured = capsys.readouterr()
    assert "SomeFutureBench" not in captured.err
    assert "rejected by payload schema" not in captured.err


# ----------------------------------------------------------------------
# parse_phase1 end-to-end

def test_parse_phase1_emits_expected_parquets(logs_root: Path, tmp_path: Path):
    data_dir = tmp_path / "data"
    rc = parse_phase1.main(logs_root, data_dir)
    assert rc == 0
    items = pd.read_parquet(data_dir / "items_long.parquet")
    agg = pd.read_parquet(data_dir / "aggregates.parquet")

    # Items: 3 AIME24 items × 3 attempts = 9 rows (only the chosen Feb file).
    aime_rows = items[items.task == "AIME24"]
    assert len(aime_rows) == 9
    # All AIME24 attempts in the winning file are correct.
    assert aime_rows["score"].sum() == 9.0

    # Aggregates: HumanEval combined n = 164+158 = 322, k computed from
    # python_pass@1 * 164 + sh_pass@1 * 158 from the WINNING file (0.50/0.30).
    he = agg[agg.task == "HumanEval"].iloc[0]
    assert int(he.n) == 322
    assert int(he.n_correct) == round(0.50 * 164) + round(0.30 * 158)
    assert he.source_metric == "combined_python_sh_pass@1"

    # MBPP from the n_examples=500 file (winner), accuracy = 0.20.
    mb = agg[agg.task == "MBPP"].iloc[0]
    assert int(mb.n) == 500
    assert int(mb.n_correct) == round(0.20 * 500)

    # gsm8k from the April file (winner), accuracy = 0.40, n=1319 (canonical).
    gs = agg[agg.task == "gsm8k"].iloc[0]
    assert int(gs.n) == 1319
    assert int(gs.n_correct) == round(0.40 * 1319)


def test_aggregates_schema_has_dual_accuracy_and_components(logs_root: Path, tmp_path: Path):
    """aggregates.parquet must carry both `accuracy` (combined-counts) and
    `accuracy_mean` (mean-of-fractions), plus a `components` JSON column.

    Pins the C1 schema bump so each reader can pick its own convention
    without re-walking logs.
    """
    import json
    data_dir = tmp_path / "data"
    parse_phase1.main(logs_root, data_dir)
    agg = pd.read_parquet(data_dir / "aggregates.parquet")
    assert {"accuracy", "accuracy_mean", "components"}.issubset(agg.columns)

    # HumanEval: combined-counts vs mean-of-fractions differ for the winning
    # file (python_pass@1=0.50, sh_pass@1=0.30).
    he = agg[agg.task == "HumanEval"].iloc[0]
    assert abs(he.accuracy_mean - 0.40) < 1e-9  # (0.50 + 0.30) / 2
    expected_combined = (round(0.50 * 164) + round(0.30 * 158)) / 322
    assert abs(he.accuracy - expected_combined) < 1e-9
    he_components = json.loads(he.components)
    assert he_components == {"python_pass@1": 0.50, "sh_pass@1": 0.30}

    # MBPP: single-metric, accuracy and accuracy_mean must agree.
    mb = agg[agg.task == "MBPP"].iloc[0]
    assert mb.accuracy == mb.accuracy_mean == 0.20
    assert json.loads(mb.components) == {"pass@1": 0.20}


def test_parse_phase1_skips_phase2_model_dirs(tmp_path: Path, capsys):
    """parse-phase1 is Phase 1 only. Model directories whose registry phase
    is in PHASE2_PHASES (OT30K) must be skipped — Phase 2 cells come from
    the Phase 2 HF parquet, not from logs.zip."""
    logs = tmp_path / "logs"
    phase2_dir = logs / "open-sci__sft__ot30k_1.7b-MixtureVitae-300BT-v1-decontaminated-16k-SFT-Tulu3-decontaminated"
    phase2_dir.mkdir(parents=True)
    _write_results_json(phase2_dir, "2026-01-01T00-00-00", {
        "AIME24": {"examples": [{"expected_answer": "12", "model_answers": ["12"]}]},
    })
    data_dir = tmp_path / "data"
    rc = parse_phase1.main(logs, data_dir)
    assert rc == 0

    items = pd.read_parquet(data_dir / "items_long.parquet")
    # No rows from the Phase 2 model dir.
    assert items.empty
    # The summary line should announce the skip.
    captured = capsys.readouterr()
    assert "Phase 2 model dirs skipped" in captured.out


def test_parse_phase1_skips_unmapped_models(tmp_path: Path):
    logs = tmp_path / "logs"
    unmapped = logs / "definitely__not__in__registry"
    unmapped.mkdir(parents=True)
    _write_results_json(unmapped, "2026-01-01T00-00-00", {
        "AIME24": {"examples": [{"expected_answer": "1", "model_answers": ["1"]}]},
    })
    data_dir = tmp_path / "data"
    rc = parse_phase1.main(logs, data_dir)
    assert rc == 0
    # No registered models -> empty outputs (still written).
    items = pd.read_parquet(data_dir / "items_long.parquet")
    assert items.empty
