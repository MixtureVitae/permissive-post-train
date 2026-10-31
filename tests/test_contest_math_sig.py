"""Tests for the sampled contest-math significance analysis."""
from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from mv_post_train.stats import contest_math_sig as cms

MV16 = "x/1.7b-MixtureVitae-300BT-v1-decontaminated-16k"
COMMA16 = "x/1.7b-Comma0.1-300BT-longsft_16k"


def _per_sample(tmp_path, spec: dict[str, list[float]], task="AMC23", n=40):
    """{model_id: per-item accuracy in [0,1]} -> a per_sample_scores.csv."""
    rows = []
    for mid, acc in spec.items():
        assert len(acc) == n
        for seed in (42, 1234, 2025):
            for i, a in enumerate(acc):
                rows.append({"seed": seed, "huggingface_model_id": mid,
                             "benchmark": task, "sample_index": i,
                             "sample_id": f"{task}-{i}",
                             "sample_score_percent": a * 100.0,
                             "score_origin": "reconstructed"})
    pd.DataFrame(rows).to_csv(tmp_path / cms.PER_SAMPLE, index=False)
    return tmp_path


def test_load_items_averages_over_seeds(tmp_path):
    rows = []
    for seed, val in zip((42, 1234, 2025), (0.0, 100.0, 100.0)):
        rows.append({"seed": seed, "huggingface_model_id": MV16,
                     "benchmark": "AMC23", "sample_index": 0,
                     "sample_id": "a", "sample_score_percent": val,
                     "score_origin": "stored"})
    pd.DataFrame(rows).to_csv(tmp_path / cms.PER_SAMPLE, index=False)
    cell = cms.load_items(tmp_path)
    assert cell.loc[("MV", "base", "AMC23", 0)] == pytest.approx(2 / 3)


def test_load_items_rejects_a_file_without_contest_math(tmp_path):
    pd.DataFrame([{"seed": 42, "huggingface_model_id": MV16,
                   "benchmark": "MATH500", "sample_index": 0, "sample_id": "a",
                   "sample_score_percent": 0.0, "score_origin": "stored"}]
                 ).to_csv(tmp_path / cms.PER_SAMPLE, index=False)
    with pytest.raises(ValueError, match="no contest-math rows"):
        cms.load_items(tmp_path)


def test_a_clear_separation_is_significant(tmp_path):
    """MV solves the first 16 of 40 problems, the reference solves none."""
    mv = [1.0] * 16 + [0.0] * 24
    ref = [0.0] * 40
    cell = cms.load_items(_per_sample(tmp_path, {MV16: mv, COMMA16: ref}))
    out = cms.test_all(cell, n_resamples=2000)
    row = out[out.reference == "Comma"].iloc[0]
    assert row["diff"] == pytest.approx(40.0)
    assert row.ci_lo > 0
    assert row.sig_within_task


def test_identical_models_are_not_significant(tmp_path):
    same = [1.0] * 8 + [0.0] * 32
    cell = cms.load_items(_per_sample(tmp_path, {MV16: same, COMMA16: same}))
    out = cms.test_all(cell, n_resamples=2000)
    row = out[out.reference == "Comma"].iloc[0]
    assert row["diff"] == pytest.approx(0.0)
    assert row.p == pytest.approx(1.0)
    assert not row.sig_within_task


def test_within_task_correction_is_never_stricter_than_global():
    """The global family is a superset, so its q-values can only be larger."""
    cell = pd.Series(dtype=float)
    df = pd.DataFrame({
        "task": ["AMC23"] * 3 + ["AIME24"] * 3,
        "p": [0.001, 0.02, 0.30, 1.0, 1.0, 1.0],
    })
    from mv_post_train.stats.core import bh_adjust
    q_global = bh_adjust(df.p.tolist())
    q_within = []
    for _, g in df.groupby("task", sort=False):
        q_within.extend(bh_adjust(g.p.tolist()))
    assert all(w <= g + 1e-12 for w, g in zip(q_within, q_global))


def test_pairing_matters_flags_only_boundary_disagreements():
    """The flag must fire when paired and unpaired disagree about zero."""
    df = pd.DataFrame({
        "ci_lo": [1.0, 0.0, -1.0],
        "nc_lo": [2.0, 1.0, -2.0],
    })
    matters = (df.ci_lo > 0) != (df.nc_lo > 0)
    assert list(matters) == [False, True, False]


def test_reported_columns_survive_a_full_run(tmp_path):
    mv = [1.0] * 10 + [0.0] * 30
    ref = [1.0] * 2 + [0.0] * 38
    cell = cms.load_items(_per_sample(tmp_path, {MV16: mv, COMMA16: ref}))
    out = cms.test_all(cell, n_resamples=2000)
    for col in ("diff", "ci_lo", "ci_hi", "p", "nc_lo", "nc_hi",
                "q_within_task", "q_global", "pairing_matters"):
        assert col in out.columns
        assert out[col].notna().all()
    assert (out.n_items == 40).all()
