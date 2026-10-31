"""Tests for the 3-seed temperature-0.7 replicate analysis."""
from __future__ import annotations

import pandas as pd
import pytest

from mv_post_train.stats import seed_variance


def _grid(scores: dict[tuple[str, int], list[float]]) -> pd.DataFrame:
    """{(model_id, seed): [one score per task]} -> a final_scores-shaped frame."""
    rows = []
    for (mid, seed), vals in scores.items():
        assert len(vals) == len(seed_variance.TASKS)
        for task, v in zip(seed_variance.TASKS, vals):
            rows.append({"seed": seed, "huggingface_model_id": mid,
                         "benchmark": task, "final_score_percent": v})
    return pd.DataFrame(rows)


MV16 = "x/1.7b-MixtureVitae-300BT-v1-decontaminated-16k"
MV16_SFT = "x/1.7b-MixtureVitae-300BT-v1-decontaminated-16k-SFT-Tulu3-decontaminated"
MV16_DPO = "x/1.7b-MixtureVitae-300BT-v1-decontaminated-16k-DPO-Tulu3-decontaminated"
COMMA16 = "x/1.7b-Comma0.1-300BT-longsft_16k"


def _classified(df: pd.DataFrame) -> pd.DataFrame:
    meta = seed_variance.classify(sorted(df.huggingface_model_id.unique()))
    return df.merge(meta, on="huggingface_model_id")


# --- classification ---------------------------------------------------------

@pytest.mark.parametrize("model,family,stage,tier", [
    (MV16, "MV", "base", "16k"),
    (MV16_SFT, "MV", "sft", "16k"),
    (MV16_DPO, "MV", "dpo", "16k"),
    ("x/1.7b-MixtureVitae-300BT-v1-decontaminated", "MV", "base", "4k"),
    ("x/1.7b-Comma0.1-300BT-WithChatTemplate", "Comma", "base", "4k"),
    (COMMA16, "Comma", "base", "16k"),
    ("x/open-sci-ref-v0.02-1.7b-nemotron-hq-300B-4096-long_sft_16k",
     "Nemotron", "base", "16k"),
    ("x/open-sci-ref-v0.02-1.7b-nemotron-hq-300B-16k-DPO-Tulu3-decontaminated",
     "Nemotron", "dpo", "16k"),
    ("x/open-sci-ref-v0.02-1.7b-fineweb-edu-1.4t-300B-4096-4096-longsft_16k",
     "FineWeb-Edu", "base", "16k"),
    ("x/open-sci-ref-v0.02-1.7b-dclm-300B-4096-SFT-Tulu3-decontaminated",
     "DCLM", "sft", "4k"),
    ("x/SmolLM2-1.7B-16k-DPO-Tulu3-decontaminated", "SmolLM2", "dpo", "16k"),
    ("x/SmolLM2-1.7B-SFT-Tulu3-decontaminated", "SmolLM2", "sft", "4k"),
    ("Qwen/Qwen3-1.7B-Base", "Qwen3", "base", "4k"),
])
def test_classify_assigns_family_stage_and_tier(model, family, stage, tier):
    row = seed_variance.classify([model]).iloc[0]
    assert (row.family, row.stage, row.tier) == (family, stage, tier)


def test_classify_rejects_an_unrecognized_model():
    """A new family must fail loudly rather than be silently dropped."""
    with pytest.raises(ValueError, match="unclassified"):
        seed_variance.classify(["x/some-brand-new-model-300BT"])


def test_dpo_is_not_misread_as_sft():
    """The DPO checkpoints are trained from SFT and can carry both tokens."""
    assert seed_variance._stage("mv-16k-SFT-Tulu3-decontaminated-DPO-Tulu3") == "dpo"


# --- per-task and suite statistics ------------------------------------------

def test_per_task_reports_sd_and_range_across_seeds():
    df = _classified(_grid({
        (MV16, 42): [10.0] * 11,
        (MV16, 1234): [20.0] * 11,
        (MV16, 2025): [30.0] * 11,
    }))
    out = seed_variance.per_task(df)
    row = out[out.benchmark == "MATH500"].iloc[0]
    assert row["mean"] == pytest.approx(20.0)
    assert row.sd == pytest.approx(10.0)       # ddof=1 over 10/20/30
    assert row["range"] == pytest.approx(20.0)
    assert row.n_seeds == 3


def test_suite_average_is_the_mean_over_all_eleven_tasks():
    vals = [0.0, 10.0, 20.0, 30.0, 40.0, 50.0, 60.0, 70.0, 80.0, 90.0, 100.0]
    df = _classified(_grid({(MV16, s): vals for s in (42, 1234, 2025)}))
    out = seed_variance.suite(df)
    assert out.iloc[0]["mean"] == pytest.approx(sum(vals) / 11)
    assert out.iloc[0].sd == pytest.approx(0.0)   # identical seeds -> no spread


def test_suite_rejects_an_incomplete_benchmark_set():
    """A dropped benchmark would silently change the average, so it must raise."""
    df = _classified(_grid({(MV16, s): [10.0] * 11 for s in (42, 1234, 2025)}))
    df = df[df.benchmark != "gsm8k"]
    with pytest.raises(ValueError, match="incomplete suite"):
        seed_variance.suite(df)


def test_gaps_are_differenced_within_each_seed():
    """Pairing inside a seed cancels a shared run-level shift; averaging first would not."""
    df = _classified(_grid({
        (MV16, 42): [30.0] * 11, (COMMA16, 42): [20.0] * 11,
        (MV16, 1234): [50.0] * 11, (COMMA16, 1234): [40.0] * 11,
        (MV16, 2025): [70.0] * 11, (COMMA16, 2025): [60.0] * 11,
    }))
    out = seed_variance.gaps(df)
    row = out[(out.stage == "base") & (out.reference == "Comma")].iloc[0]
    # Each seed's gap is exactly 10 even though the levels swing by 40 pp.
    assert row.gap_mean == pytest.approx(10.0)
    assert row.gap_sd == pytest.approx(0.0)
    assert row.n_seeds == 3


# --- audit ------------------------------------------------------------------

def test_audit_flags_a_sub_chance_gpqa_cell():
    df = _classified(_grid({(MV16, s): [50.0] * 11 for s in (42, 1234, 2025)}))
    df.loc[(df.benchmark == "GPQADiamond") & (df.seed == 1234),
           "final_score_percent"] = 7.0
    notes = seed_variance.audit(df)
    assert any("GPQA-Diamond below chance" in n and "7.0" in n for n in notes)


def test_audit_flags_a_missing_seed():
    df = _classified(_grid({(MV16, s): [50.0] * 11 for s in (42, 1234)}))
    assert any("do not have all 3 seeds" in n for n in seed_variance.audit(df))


def test_audit_is_quiet_on_clean_data():
    df = _classified(_grid({
        (MV16, s): [40.0] * 11 for s in (42, 1234, 2025)
    } | {
        (COMMA16, s): [30.0] * 11 for s in (42, 1234, 2025)
    }))
    assert seed_variance.audit(df) == []


# --- leave-benchmark-out ----------------------------------------------------

def _lo_grid(scores):
    """{(family, benchmark): value} -> a classified frame, three seeds.

    A per-seed tuple gives that cell a different value in each seed. Named
    apart from `_grid` above, which builds the raw pre-classification shape.
    """
    rows = []
    for (fam, bench), v in scores.items():
        vals = v if isinstance(v, tuple) else (v, v, v)
        for seed, val in zip(seed_variance.SEEDS, vals):
            rows.append({"family": fam, "stage": "sft", "tier": "16k",
                         "model": fam, "huggingface_model_id": f"x/{fam}",
                         "benchmark": bench, "seed": seed,
                         "final_score_percent": float(val)})
    return pd.DataFrame(rows)


def test_dropping_a_benchmark_removes_it_from_the_average():
    """MV leads only on AIME24; dropping it must erase the gap."""
    df = _lo_grid({("MV", "AIME24"): 30.0, ("MV", "MATH500"): 10.0,
                   ("Comma", "AIME24"): 0.0, ("Comma", "MATH500"): 10.0})
    lo = seed_variance.leave_out(df)
    full = lo[lo.subset == "full suite (11)"].iloc[0]
    dropped = lo[lo.subset == "drop AIME24"].iloc[0]
    assert full.gap_mean == pytest.approx(15.0)   # (30-0 + 0)/2
    assert dropped.gap_mean == pytest.approx(0.0)


def test_a_gap_carried_by_other_tasks_survives_dropping_aime():
    df = _lo_grid({("MV", "AIME24"): 0.0, ("MV", "MATH500"): 40.0,
                   ("Comma", "AIME24"): 0.0, ("Comma", "MATH500"): 10.0})
    lo = seed_variance.leave_out(df)
    assert lo[lo.subset == "drop AIME24"].iloc[0].gap_mean == pytest.approx(30.0)


def test_the_summary_reports_the_worst_reference_not_the_average():
    """A separation that holds on average but fails against the strongest
    reference has not held."""
    df = _lo_grid({("MV", "MATH500"): 30.0,
                   ("Comma", "MATH500"): 25.0, ("Nemotron", "MATH500"): 5.0})
    summ = seed_variance.leave_out_summary(seed_variance.leave_out(df))
    row = summ[summ.subset == "full suite (11)"].iloc[0]
    assert row.min_gap == pytest.approx(5.0)
    assert row.worst_reference == "Comma"
    assert row.max_gap == pytest.approx(25.0)


def test_all_positive_is_false_when_any_seed_flips_the_sign():
    """Judged per seed, not on the mean: a gap that inverts in one seed of
    three is not a stable separation."""
    df = _lo_grid({("MV", "MATH500"): (10.0, 10.0, 1.0),
                   ("Comma", "MATH500"): 5.0})
    summ = seed_variance.leave_out_summary(seed_variance.leave_out(df))
    assert not bool(summ[summ.subset == "full suite (11)"].iloc[0].all_positive)


def test_every_named_subset_and_a_jackknife_entry_per_task():
    df = _lo_grid({**{("MV", b): 10.0 for b in seed_variance.TASKS},
                   **{("Comma", b): 5.0 for b in seed_variance.TASKS}})
    lo = seed_variance.leave_out(df)
    assert set(seed_variance.DROP_SETS) <= set(lo.subset)
    assert {f"drop {t}" for t in seed_variance.TASKS} <= set(lo.subset)
