"""Tests for the statistics appendix (mvpt render-statistics).

They pin what makes Table 26 comparable with the rest of the paper: each
contrast is the difference of the two checkpoints' scores on the same items,
the suite contrast is over the eleven benchmarks, the interval comes from
paired draws and is reproducible, q-values are corrected within the family
the caller names, the suite levels equal the Protocol A suite averages of
the Phase 1 tables, and the reweighted suite averages use the same draws.
"""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from mv_post_train.data import protocol_a as pa
from mv_post_train.render import statistics as st
from mv_post_train.stats import factorial as fs
from mv_post_train.stats.pairwise_a import (SUITE, Cell, domain_views, paired_contrasts,
                                            with_q)

A, B = Cell("MV", 16384, "sft"), Cell("Ref", 16384, "sft")


def _records(score_a: float, score_b: float, n: int = 20, seed: int = 0) -> pd.DataFrame:
    """Two checkpoints on n items of every benchmark, three seeds, with the
    given chance of a correct answer."""
    rng = np.random.default_rng(seed)
    rows = []
    for cell, p in ((A, score_a), (B, score_b)):
        for b in pa.BENCHMARKS:
            for i in range(n):
                for s in pa.SEEDS:
                    rows.append(dict(model_id=cell.group, model_group=cell.group,
                                     context_length=cell.context_length, stage=cell.stage,
                                     seed=s, benchmark=b, sample_id=f"{b}-{i}",
                                     correct=float(rng.random() < p)))
    return pd.DataFrame(rows)


def test_contrasts_are_differences_on_the_same_items():
    df = _records(0.7, 0.3)
    res = paired_contrasts(df, [(A, B)], n_boot=200)
    assert [c.benchmark for c in res] == [*pa.BENCHMARKS, SUITE]
    item = df.groupby(["model_group", "benchmark", "sample_id"])["correct"].mean() * 100
    for c in res[:-1]:
        mean = item.xs(c.benchmark, level="benchmark").groupby(level="model_group").mean()
        assert c.estimate == pytest.approx(mean["MV"] - mean["Ref"])
        assert c.lo <= c.estimate <= c.hi
    suite = res[-1]
    assert suite.estimate == pytest.approx(np.mean([c.estimate for c in res[:-1]]))
    assert res == paired_contrasts(df, [(A, B)], n_boot=200)


def test_checkpoints_scored_on_different_items_are_refused():
    df = _records(0.5, 0.5)
    df = df[~((df["model_group"] == "Ref") & (df["sample_id"] == "MATH500-3"))]
    with pytest.raises(ValueError, match="same items"):
        paired_contrasts(df, [(A, B)], n_boot=10)


def test_a_checkpoint_without_three_seeds_is_refused():
    df = _records(0.5, 0.5)
    df = df[~((df["model_group"] == "Ref") & (df["seed"] == 42))]
    with pytest.raises(ValueError, match="three seeds"):
        paired_contrasts(df, [(A, B)], n_boot=10)


def test_q_values_are_corrected_within_the_named_family_only():
    res = paired_contrasts(_records(0.8, 0.2), [(A, B)], n_boot=200)
    fam = [0, 1, 2]
    out = with_q(res, fam)
    assert all(not np.isnan(out[i].q) for i in fam)
    assert all(np.isnan(out[i].q) for i in range(3, len(out)))
    assert all(out[i].q >= out[i].p for i in fam)


def test_q_values_print_as_in_table_28():
    assert st.qfmt(0.0004) == r"$<0.001$"
    assert st.qfmt(0.0011) == "$0.001$"
    assert st.qfmt(float("nan")) == "--"


def test_a_benchmark_both_models_fail_entirely_prints_dashes():
    assert st.cells(0.0, 0.0, 0.0, "x") == ["---", "---", "---"]
    assert st.cells(3.3, 0.0, 10.0, "x")[0] == "$+3.3$"


def _zoo_records(zero: str = "AIME24") -> pd.DataFrame:
    """MV, Comma0.1 and Nemotron at 16K after SFT and DPO, with every model
    scoring zero on the `zero` benchmark."""
    frames = []
    for i, (group, p) in enumerate((("MV", 0.6), ("Comma0.1", 0.2), ("Nemotron", 0.1))):
        for stage in ("sft", "dpo"):
            df = _records(p, p, n=12, seed=i)
            df = df[df["model_group"] == "MV"].assign(model_group=group, model_id=f"{group}-{stage}",
                                                      stage=stage)
            df.loc[df["benchmark"] == zero, "correct"] = 0.0
            frames.append(df)
    return pd.concat(frames, ignore_index=True)


def test_untested_benchmarks_get_no_q_and_leave_the_family():
    per_task, suite, views = st.matched_compute(_zoo_records(), n_boot=200)
    for c in per_task:
        assert np.isnan(c.q) == (c.benchmark == "AIME24")
    from statsmodels.stats.multitest import multipletests
    comma = [c for c in per_task if c.b.group == "Comma0.1" and c.benchmark != "AIME24"]
    expect = multipletests([c.p for c in comma], method="fdr_bh")[1]
    assert [c.q for c in comma] == pytest.approx(list(expect))
    tex = st.matched_compute_table(per_task, suite, views)
    assert r"AIME24 ($n{=}30$) & --- & --- & ---" in tex


def test_power_rows_are_reused_only_with_the_same_settings(tmp_path, monkeypatch):
    calls = []
    monkeypatch.setattr(st, "mde_paired_binomial",
                        lambda base, n, **kw: calls.append((base, n)) or (0.25, 0.3))
    per_task, _, _ = st.matched_compute(_zoo_records(), n_boot=50)
    cache = tmp_path / "power.csv"
    st.power(per_task, cache).to_csv(cache, index=False)
    assert len(calls) == len(pa.BENCHMARKS)
    st.power(per_task, cache)
    assert len(calls) == len(pa.BENCHMARKS)              # all reused
    monkeypatch.setitem(st.MDE_SETTINGS, "target_bernoulli_rho", 0.5)
    st.power(per_task, cache)
    assert len(calls) == 2 * len(pa.BENCHMARKS)          # a setting changed


RECORDS = Path(__file__).resolve().parents[1] / "data" / "workshop" / pa.REASONING_CSV


def test_suite_levels_equal_the_protocol_a_suite_averages():
    if not RECORDS.exists():
        pytest.skip("Protocol A records not present")
    zoo = pa.zoo(pa.load_reasoning(RECORDS))
    per_task, suite, _ = st.matched_compute(zoo, n_boot=200)
    avg = fs.suite_average(fs.benchmark_scores(zoo, pa.ZOO_KEYS), pa.ZOO_KEYS)
    avg = avg.set_index(pa.ZOO_KEYS)["mean"]
    for c in suite:
        for cell, level in ((c.a, c.level_a), (c.b, c.level_b)):
            assert level == pytest.approx(avg[(cell.group, cell.context_length, cell.stage)])
    assert len(per_task) == len(pa.BENCHMARKS) * len(st.OPPONENTS)
    assert all(not np.isnan(c.q) for c in per_task + suite)


def _ablation_records() -> pd.DataFrame:
    """MV-100, MV-noweb and MV-noinst in the six settings of Table 27."""
    frames = []
    for i, (group, p) in enumerate(((st.FULL_100B, 0.5), ("MV-noweb", 0.6), ("MV-noinst", 0.2))):
        for stage, ctx, _ in st.ABLATION_CELLS:
            df = _records(p, p, n=10, seed=i)
            frames.append(df[df["model_group"] == "MV"].assign(
                model_group=group, model_id=f"{group}-{ctx}-{stage}", context_length=ctx,
                stage=stage))
    return pd.concat(frames, ignore_index=True)


def test_table27_contrasts_each_ablation_with_the_full_mixture():
    suite, per_task = st.mixture_ablation(_ablation_records(), n_boot=200)
    assert len(suite) == 12 and len(per_task) == 2 * len(pa.BENCHMARKS)
    assert {c.b.group for c in suite + per_task} == {st.FULL_100B}
    assert all(c.a.stage == "sft" and c.a.context_length == st.CTX for c in per_task)
    # the twelve suite rows are one family, the per-task rows one per column
    assert [c.q for c in suite] == pytest.approx(
        [c.q for c in with_q(suite, list(range(len(suite))))])
    tex = st.mixture_ablation_table(suite, per_task)
    rows = [line.split(" & ")[0] for line in tex.splitlines() if line.count(" & ") == 6]
    assert rows[-len(st.ABLATION_CELLS) - len(pa.BENCHMARKS):] == (
        [label for _, _, label in st.ABLATION_CELLS]
        + [f"{st.TASK_NAME[b]} ($n{{=}}{pa.BENCH_ITEMS[b]}$)" for b in pa.BENCHMARKS])
    keys = st.mixture_ablation_macros(suite, per_task)
    assert {"abl.noweb.sft16k.suite.q", "abl.noinst.he", "abl.noweb.suite.max"} <= set(keys)


def test_reweighted_views_reuse_the_draws_and_weight_the_benchmark_means():
    views = domain_views()
    assert all(abs(w.sum() - 1) < 1e-12 for w in views.values())
    assert set(views) == {"equal_domain"} | {f"drop_{d}" for d in pa.DOMAINS}
    df = _records(0.6, 0.3)
    plain = paired_contrasts(df, [(A, B)], n_boot=100)
    both = paired_contrasts(df, [(A, B)], n_boot=100, views=views)
    assert both[:len(plain)] == plain                      # the other contrasts do not move
    means = {c.benchmark: c.estimate for c in plain if c.benchmark != SUITE}
    eq = next(c for c in both if c.benchmark == "equal_domain")
    by_domain = [np.mean([means[b] for b in pa.BENCHMARKS if pa.DOMAIN[b] == d])
                 for d in pa.DOMAINS]
    assert eq.estimate == pytest.approx(np.mean(by_domain))
    assert eq.lo < eq.estimate < eq.hi


def test_table26_prints_the_reweighted_block_as_one_family():
    per_task, suite, views = st.matched_compute(_zoo_records(), n_boot=200)
    assert len(views) == len(st.VIEW_LABEL) * len(st.OPPONENTS)
    assert all(c.a.stage == "sft" for c in views)
    assert [c.q for c in views] == pytest.approx(
        [c.q for c in with_q(views, list(range(len(views))))])
    tex = st.matched_compute_table(per_task, suite, views)
    rows = [line.split(" & ")[0] for line in tex.splitlines() if line.count(" & ") == 6]
    assert rows[-len(st.VIEW_LABEL):] == list(st.VIEW_LABEL.values())
    keys = st.matched_compute_macros(per_task, suite, views)
    assert {"mc.sft.comma.equal_domain", "mc.sft.nemotron.drop_math.q",
            "mc.sft.reweighted.min", "mc.sft.reweighted.lo.min"} <= set(keys)
