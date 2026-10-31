"""Tests for the workshop-paper pipeline (render-workshop / check-workshop).

The real inputs are 830k rows and are not bundled, so these tests shrink every
benchmark to two items and build the full 3 x 2 x 2 x 3 factorial
synthetically. They pin the properties the paper's tables depend on: the
design validation, the aggregation order, single rounding, the printed delta
being the difference of printed cells, the bootstrap's determinism, and the
checker catching a changed cell.
"""
from __future__ import annotations

import itertools

import numpy as np
import pandas as pd
import pytest

from mv_post_train.data import protocol_a as pa
from mv_post_train.data import protocol_b as pb
from mv_post_train.render import workshop_check, workshop_tables
from mv_post_train.stats import factorial as fs

ITEMS = 2


@pytest.fixture(autouse=True)
def _two_items_per_benchmark(monkeypatch):
    monkeypatch.setattr(pa, "BENCH_ITEMS", {b: ITEMS for b in pa.BENCHMARKS})


def _factorial_rows(seed: int = 0) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    rows = []
    for pool, fl, ctx, stage in itertools.product(pa.POOLS, (False, True), pa.CONTEXTS, pa.STAGES):
        mid = f"x/{pool}-{fl}-{ctx}-{stage}"
        for s, bench, item in itertools.product(pa.SEEDS, pa.BENCHMARKS, range(ITEMS)):
            p = 0.7 if fl else 0.3
            rows.append(dict(model_id=mid, stage=stage, model_group=pool, front_loading=fl,
                             context_length=ctx, seed=s, benchmark=bench,
                             sample_id=f"{bench}-{item}", correct=float(rng.random() < p),
                             in_factorial=True, in_zoo=False))
    return pd.DataFrame(rows)


# --- design validation ------------------------------------------------------

def test_complete_factorial_loads(tmp_path):
    path = tmp_path / "r.csv"
    _factorial_rows().to_csv(path, index=False)
    df = pa.factorial(pa.load_reasoning(path))
    assert df.groupby(pa.FACTORIAL_KEYS).ngroups == 36


def test_a_missing_item_stops_the_pipeline(tmp_path):
    path = tmp_path / "r.csv"
    _factorial_rows().iloc[1:].to_csv(path, index=False)
    with pytest.raises(pa.DesignError, match="incomplete"):
        pa.load_reasoning(path)


def test_a_missing_arm_stops_the_pipeline():
    df = _factorial_rows()
    df = df[~((df.model_group == "MV") & ~df.front_loading & (df.context_length == 4096))]
    with pytest.raises(pa.DesignError, match="not fully crossed"):
        pa.factorial(df)


def test_one_checkpoint_in_two_cells_is_refused(tmp_path):
    df = _factorial_rows()
    first, second = df.model_id.unique()[:2]
    df.loc[df.model_id == second, "model_id"] = first
    path = tmp_path / "r.csv"
    df.to_csv(path, index=False)
    with pytest.raises(pa.DesignError, match="several"):
        pa.load_reasoning(path)


def _strict_verdicts(df: pd.DataFrame) -> pd.DataFrame:
    """Strict IFEval verdicts for every checkpoint of `df`: the loose ones
    flipped, so that a swapped value is easy to see."""
    ife = df[df.benchmark == "IFEval"]
    return ife[["model_id", "stage", "seed", "benchmark", "sample_id"]].assign(
        correct=1.0 - ife.correct)


def test_strict_ifeval_replaces_only_ifeval_and_keeps_the_row_order():
    df = _factorial_rows()
    out = pa.with_strict_ifeval(df, _strict_verdicts(df))
    ife = df.benchmark == "IFEval"
    assert out.drop(columns="correct").equals(df.drop(columns="correct"))
    assert out.loc[~ife, "correct"].equals(df.loc[~ife, "correct"])
    assert (out.loc[ife, "correct"] == 1.0 - df.loc[ife, "correct"]).all()


def test_a_checkpoint_without_strict_verdicts_stops_the_pipeline():
    df = _factorial_rows()
    strict = _strict_verdicts(df)
    first = df.model_id.iloc[0]
    with pytest.raises(pa.DesignError, match="no strict verdicts"):
        pa.with_strict_ifeval(df, strict[strict.model_id != first])


def test_a_listed_checkpoint_without_strict_verdicts_loses_only_ifeval(monkeypatch):
    df = _factorial_rows()
    strict = _strict_verdicts(df)
    first = df.model_id.iloc[0]
    monkeypatch.setattr(pa, "IFEVAL_STRICT_MISSING", {first: "test"})
    out = pa.with_strict_ifeval(df, strict[strict.model_id != first])
    gone = (df.model_id == first) & (df.benchmark == "IFEval")
    assert len(out) == len(df) - gone.sum()
    assert set(out.loc[out.model_id == first, "benchmark"]) == set(pa.BENCHMARKS) - {"IFEval"}


def test_strict_verdicts_are_validated(tmp_path):
    strict = _strict_verdicts(_factorial_rows())
    path = tmp_path / "s.csv"
    strict.to_csv(path, index=False)
    assert len(pa.load_ifeval_strict(path)) == len(strict)
    strict.iloc[1:].to_csv(path, index=False)
    with pytest.raises(pa.DesignError, match="instructions"):
        pa.load_ifeval_strict(path)


def test_language_names_map_to_factor_levels(tmp_path):
    names = {
        ("MV", True): "mixturevitae-300B-4096", ("MV", False): "mixturevitae-noinstruct-300B-4096",
        ("Nemotron", True): "nemotron-hq-mv_reasoning-300B-4096",
        ("Nemotron", False): "nemotron-hq-300B-4096",
        ("FineWeb-Edu", True): "fineweb-edu-1.4t-mv_reasoning-300B-4096",
        ("FineWeb-Edu", False): "fineweb-edu-1.4t-300B-4096",
    }
    rows = [dict(model_name=f"open-sci/open-sci-ref-v0.02-1.7b-{stem}{suffix}", task=t,
                 n_shot=0, performance=0.5, metric_name="acc,none")
            for stem in names.values() for suffix in ("", "-longsft_16k")
            for t in pa.LANGUAGE_TASKS + ("social_iqa",)]
    path = tmp_path / "lang.csv"
    pd.DataFrame(rows).to_csv(path, index=False)
    lang = pa.load_language(path)
    levels = lang.groupby(["model_group", "front_loading", "context_length"]).size()
    assert len(levels) == 12 and (levels == len(pa.LANGUAGE_TASKS)).all()
    assert set(lang.task) == set(pa.LANGUAGE_TASKS)        # social_iqa dropped
    assert lang.performance.eq(50.0).all()                  # percent


# --- aggregation ------------------------------------------------------------

def test_suite_average_is_mean_over_benchmarks_then_over_seeds():
    scores = pd.DataFrame({
        "g": "a", "seed": [1, 1, 2, 2], "benchmark": ["x", "y", "x", "y"],
        "score": [10.0, 30.0, 20.0, 60.0],
    })
    out = fs.suite_average(scores, ["g"]).iloc[0]
    assert out["mean"] == pytest.approx(30.0)                # (20 + 40) / 2
    assert out["sd"] == pytest.approx(np.std([20.0, 40.0], ddof=1))


def test_equal_domain_average_weights_each_domain_once():
    scores = pd.DataFrame({
        "g": "a", "seed": 1, "benchmark": ["m1", "m2", "m3", "c1"],
        "domain": ["math", "math", "math", "code"], "score": [0.0, 0.0, 0.0, 100.0],
    })
    assert fs.equal_domain_average(scores, ["g"]).iloc[0]["mean"] == pytest.approx(50.0)
    assert fs.suite_average(scores, ["g"]).iloc[0]["mean"] == pytest.approx(25.0)


# --- rounding ---------------------------------------------------------------

@pytest.mark.parametrize("exact,printed", [
    (50.9477, "50.9"),   # rounding to 2 decimals first would give 50.95 -> 51.0
    (46.2505, "46.3"),   # ... and 46.25 -> 46.2
    (0.0485, "0.0"),
])
def test_cells_are_rounded_once(exact, printed):
    assert workshop_tables._ms(exact, 0.115).startswith(printed + "$")


def test_delta_row_is_the_difference_of_the_printed_cells(tmp_path):
    keys = list(itertools.product(pa.POOLS, (False, True), pa.CONTEXTS, pa.STAGES))
    summary = pd.DataFrame(
        [dict(model_group=p, front_loading=fl, context_length=c, stage=s,
              benchmark="__suite__", mean=17.66 if fl else 4.24, sd=0.1)
         for p, fl, c, s in keys])
    lang = pd.DataFrame([dict(model_group=p, front_loading=fl, context_length=c, language=50.0)
                         for p, fl, c in itertools.product(pa.POOLS, (False, True), pa.CONTEXTS)])
    tex = workshop_tables.factorial_summary(summary, lang, tmp_path).read_text()
    # 17.7 - 4.2 = 13.5 as printed, although 17.66 - 4.24 = 13.42.
    assert "$\\Delta$ & $+13.5$" in tex


# --- contrasts ----------------------------------------------------------------

def test_contrasts_match_the_suite_averages_and_are_deterministic():
    df = _factorial_rows()
    a = fs.factorial_contrasts(df, n_boot=50)
    b = fs.factorial_contrasts(df, n_boot=50)
    pd.testing.assert_frame_equal(a, b)
    assert len(a) == 2 * 3 * 6
    suite = fs.suite_average(fs.benchmark_scores(df, pa.FACTORIAL_KEYS), pa.FACTORIAL_KEYS)
    level = suite.set_index(pa.FACTORIAL_KEYS)["mean"]
    row = a[(a.family == "front-loading") & (a.label == "MV")
            & (a.context_length == 16384) & (a.stage == "sft")].iloc[0]
    expected = level[("MV", True, 16384, "sft")] - level[("MV", False, 16384, "sft")]
    assert row.estimate == pytest.approx(expected)
    assert row.lo <= row.estimate <= row.hi
    for _, fam in a.groupby("family"):
        assert (fam.q >= fam.p - 1e-12).all()


# --- Protocol B constants ---------------------------------------------------

def test_protocol_b_constants_are_consistent():
    assert pb.check_constants() == []


def test_a_mistyped_protocol_b_average_is_caught(monkeypatch):
    rows = [list(r) for r in pb.PHASE2_ENDPOINTS]
    rows[0][3] = rows[0][3][:11] + [99.9]
    monkeypatch.setattr(pb, "PHASE2_ENDPOINTS", [tuple(r) for r in rows])
    assert any("printed Avg 99.9" in p for p in pb.check_constants())


# --- check-workshop -----------------------------------------------------------

TABLE = r"""\begin{tabular}{lrr}
\textbf{Model} & \textbf{A24} & \textbf{Avg} \\
MixtureVitae-16K & 1.1$_{\pm1.92}$ & 30.7$_{\pm0.72}$ \\
\quad $\to$ SFT & \textbf{3.3}$_{\pm0.00}$ & $+27.3$ \\
\end{tabular}
"""


def test_identical_tables_pass_and_formatting_is_ignored():
    paper = TABLE.replace(r"\textbf{3.3}", "3.3").replace("$+27.3$", "27.3")
    report = workshop_check.compare_tables(TABLE, paper, "t.tex")
    assert report.ok and report.rows_compared == 3


def test_a_changed_cell_is_reported_with_its_column():
    paper = TABLE.replace("30.7$_{\\pm0.72}$", "30.8$_{\\pm0.72}$")
    report = workshop_check.compare_tables(TABLE, paper, "t.tex")
    assert not report.ok
    assert report.mismatches == ["MixtureVitae-16K: Avg: ours 30.7/0.72 vs paper 30.8/0.72"]


def test_labels_with_digits_are_not_read_as_values():
    rows = workshop_check.parse_rows(TABLE)
    assert rows[0].cells == [[], []]                          # header: A24, Avg
    assert rows[1].cells == [["1.1", "1.92"], ["30.7", "0.72"]]


def test_a_dropped_row_is_reported():
    paper = TABLE.replace("\\end{tabular}", "Extra row & 1.0 & 2.0 \\\\\n\\end{tabular}")
    report = workshop_check.compare_tables(TABLE, paper, "t.tex")
    assert report.mismatches == ["only in paper: Extra row"]
