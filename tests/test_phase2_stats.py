"""Tests for the Phase 2 statistics modules.

Two contracts are pinned here:

- `phase2_transition` pairs a Phase 1 checkpoint with its OT30K descendant by
  `item_index`, averages over repeated generations first, and reports
  both-constant cells as degenerate rather than dropping them.
- `aime_decoding_variance` reads only the temperature-0.7 rows and reports
  dispersion across the independent samples.
"""
from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from mv_post_train.stats import (
    aime_decoding_variance,
    phase2_transition,
)


def _items(rows: list[dict]) -> pd.DataFrame:
    """Build an items_long-shaped frame from partial specs."""
    base = dict(model="m", family="MV", phase="sft", context_tier="16k",
                tokens_b=300.0, size_b=1.7, task="MATH500", item_index=0,
                attempt_index=0, score=0.0)
    return pd.DataFrame([{**base, **r} for r in rows])


# --------------------------------------------------------------------------
# phase2_transition


def test_transition_pairs_on_item_index_and_reports_gain():
    """A clean +50 pp gain on shared items is recovered with a CI above zero."""
    rows = []
    for i in range(100):
        rows.append({"family": "MV", "phase": "sft", "task": "MATH500",
                     "item_index": i, "score": 1.0 if i < 20 else 0.0})
        rows.append({"family": "MV", "phase": "ot30k_on_sft", "task": "MATH500",
                     "item_index": i, "score": 1.0 if i < 70 else 0.0})
    out = phase2_transition.transitions(_items(rows), n_resamples=500)

    cell = out[out.transition == "sft->ot30k_on_sft"].iloc[0]
    assert cell.delta == pytest.approx(50.0)
    assert cell.acc_start == pytest.approx(20.0)
    assert cell.acc_end == pytest.approx(70.0)
    assert cell.ci_lo > 0
    assert not cell.degenerate


def test_transition_averages_over_attempts_before_pairing():
    """Ten repetitions of one problem are one paired unit, not ten.

    The OT30K cell solves item 0 in half its repetitions; the paired
    difference must be +50 pp on that item, not +100 or 0.
    """
    rows = [{"family": "MV", "phase": "sft", "task": "AIME24",
             "item_index": 0, "attempt_index": 0, "score": 0.0}]
    rows += [{"family": "MV", "phase": "ot30k_on_sft", "task": "AIME24",
              "item_index": 0, "attempt_index": a,
              "score": 1.0 if a < 5 else 0.0} for a in range(10)]
    # A second, always-wrong item so the cell is not degenerate.
    rows += [{"family": "MV", "phase": p, "task": "AIME24",
              "item_index": 1, "attempt_index": 0, "score": 0.0}
             for p in ("sft", "ot30k_on_sft")]

    out = phase2_transition.transitions(_items(rows), n_resamples=200)
    cell = out[out.task == "AIME24"].iloc[0]
    assert cell.n_items == 2
    assert cell.delta == pytest.approx(25.0)  # (0.5 + 0.0) / 2


def test_transition_reports_all_zero_cells_as_degenerate():
    """References scoring 0 before and after is a finding, not missing data."""
    rows = [{"family": "Comma", "phase": p, "task": "AIME24",
             "item_index": i, "score": 0.0}
            for p in ("sft", "ot30k_on_sft") for i in range(30)]
    out = phase2_transition.transitions(_items(rows), n_resamples=200)

    assert len(out) == 1
    cell = out.iloc[0]
    assert bool(cell.degenerate)
    assert cell.delta == 0.0
    assert cell.p_perm == 1.0
    assert cell.q_value == 1.0


def test_transition_skips_cells_without_a_paired_counterpart():
    """An OT30K cell with no Phase 1 start produces no row."""
    rows = [{"family": "MV", "phase": "ot30k_on_sft", "task": "MATH500",
             "item_index": i, "score": 1.0} for i in range(50)]
    out = phase2_transition.transitions(_items(rows), n_resamples=200)
    assert out.empty


def test_transition_intersects_items_rather_than_assuming_alignment():
    """Only items present in both phases are paired."""
    rows = [{"family": "MV", "phase": "sft", "task": "MATH500",
             "item_index": i, "score": 0.0} for i in range(100)]
    rows += [{"family": "MV", "phase": "ot30k_on_sft", "task": "MATH500",
              "item_index": i, "score": 1.0} for i in range(40)]
    out = phase2_transition.transitions(_items(rows), n_resamples=200)
    assert out.iloc[0].n_items == 40


def test_transition_covers_families_evaluated_outside_the_16k_tier():
    """Qwen2.5 lives at 131k and Qwen3 at 32k; a 16k-only filter loses them.

    Nothing raises when this regresses -- the families just vanish from the
    table, which is how they went unreported in the first place.
    """
    rows = []
    for fam, tier in (("Qwen2.5", "131k"), ("Qwen3", "32k"), ("MV", "16k")):
        for i in range(50):
            rows.append({"family": fam, "context_tier": tier, "phase": "sft",
                         "task": "MATH500", "item_index": i, "score": 0.0})
            rows.append({"family": fam, "context_tier": tier,
                         "phase": "ot30k_on_sft", "task": "MATH500",
                         "item_index": i, "score": 1.0 if i < 25 else 0.0})
    out = phase2_transition.transitions(_items(rows), n_resamples=200)

    assert set(out.family) == {"Qwen2.5", "Qwen3", "MV"}
    assert set(out.context_tier) == {"131k", "32k", "16k"}
    assert out.delta.tolist() == pytest.approx([50.0] * 3)


def test_transition_never_pairs_across_context_tiers():
    """A 4k start and a 16k endpoint would confound post-training with length."""
    rows = [{"family": "MV", "context_tier": "4k", "phase": "sft",
             "task": "MATH500", "item_index": i, "score": 0.0}
            for i in range(50)]
    rows += [{"family": "MV", "context_tier": "16k", "phase": "ot30k_on_sft",
              "task": "MATH500", "item_index": i, "score": 1.0}
             for i in range(50)]
    out = phase2_transition.transitions(_items(rows), n_resamples=200)
    assert out.empty


def test_transition_reports_each_tier_separately_when_a_family_has_several():
    rows = []
    for tier in ("4k", "16k"):
        for i in range(50):
            rows.append({"family": "MV", "context_tier": tier, "phase": "sft",
                         "task": "MATH500", "item_index": i, "score": 0.0})
            rows.append({"family": "MV", "context_tier": tier,
                         "phase": "ot30k_on_sft", "task": "MATH500",
                         "item_index": i, "score": 1.0})
    out = phase2_transition.transitions(_items(rows), n_resamples=200)
    assert len(out) == 2
    assert set(out.context_tier) == {"4k", "16k"}


# --------------------------------------------------------------------------
# aime_decoding_variance


def _replicates(accs_by_run: dict[int, float], task: str = "AIME24",
                stage: str = "OT3+SFT", n: int = 30) -> pd.DataFrame:
    """Build a best-responses-shaped frame with the requested per-run accuracy.

    `gold_answer` / `model_answer` are integers so math-verify scores them
    without needing a real generation.
    """
    rows = []
    for run, acc in accs_by_run.items():
        n_correct = int(round(acc * n))
        for i in range(n):
            rows.append({
                "base_model": "MixtureVitae-1.7B", "stage": stage, "task": task,
                "example_idx": i, "run_idx": run,
                "gold_answer": "42",
                "model_answer": "42" if i < n_correct else "7",
                "decoding": aime_decoding_variance.DECODING_TAG,
            })
    # A greedy row that must be excluded by the decoding filter.
    rows.append({
        "base_model": "MixtureVitae-1.7B", "stage": stage, "task": task,
        "example_idx": 0, "run_idx": 0,
        "gold_answer": "42", "model_answer": "42", "decoding": None,
    })
    return pd.DataFrame(rows)


def test_summarize_reports_per_run_dispersion(tmp_path):
    src = tmp_path / "best.parquet"
    _replicates({0: 0.0, 1: 1 / 30, 2: 1 / 30}).to_parquet(src)

    scored = aime_decoding_variance.load_scored(src)
    # The greedy row is filtered out: 3 runs x 30 items.
    assert len(scored) == 90

    out = aime_decoding_variance.summarize(scored)
    cell = out.iloc[0]
    assert cell.n_runs == 3
    assert cell.n_items == 30
    assert cell["range"] == pytest.approx(100 / 30, abs=1e-2)
    assert cell.one_item_pp == pytest.approx(100 / 30, abs=1e-2)
    # Two runs solve item 0, one does not -> exactly one disputed item.
    assert cell.items_solved_sometimes == 1
    assert cell.items_solved_always == 0
    assert cell.items_all_runs_agree == 29


def test_summarize_reports_zero_sd_when_runs_agree(tmp_path):
    src = tmp_path / "best.parquet"
    _replicates({0: 2 / 30, 1: 2 / 30, 2: 2 / 30}).to_parquet(src)
    out = aime_decoding_variance.summarize(aime_decoding_variance.load_scored(src))
    cell = out.iloc[0]
    assert cell.sd == pytest.approx(0.0)
    assert cell.items_all_runs_agree == 30
    assert cell.items_solved_always == 2


def test_load_scored_rejects_a_file_without_the_decoding_column(tmp_path):
    """Pre-2026-07-27 revisions must fail loudly, not silently score greedy runs."""
    src = tmp_path / "old.parquet"
    df = _replicates({0: 0.0}).drop(columns=["decoding"])
    df.to_parquet(src)
    with pytest.raises(ValueError, match="decoding"):
        aime_decoding_variance.load_scored(src)


def test_load_scored_rejects_a_file_with_no_replicate_rows(tmp_path):
    src = tmp_path / "greedy_only.parquet"
    df = _replicates({0: 0.0})
    df["decoding"] = None
    df.to_parquet(src)
    with pytest.raises(ValueError, match="no rows tagged"):
        aime_decoding_variance.load_scored(src)
