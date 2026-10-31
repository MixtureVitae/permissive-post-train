"""Run the full pairwise analysis.

Outputs (to artifacts/):
  per_task_accuracy.csv     model x task accuracy with Wilson CI + bootstrap CI
  pairwise_full_matrix.csv  every model_a, model_b pair (within (phase,ctx)) x task
  pairwise_mv_vs_comma.csv  filtered subset
  pairwise_mv_vs_nemotron.csv
  pairwise_comma_vs_nemotron.csv
  suite_average_ci.csv      per (model) reasoning-suite avg with stratified bootstrap CI
  pairing_audit.csv         which (pair, task) cells were paired vs unpaired
"""
from __future__ import annotations

import hashlib
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import stats as _scipy_stats
from tqdm import tqdm

from .. import paths
from ..data import conventions
from .core import (
    bh_adjust,
    newcombe_diff_ci,
    paired_bootstrap_ci,
    paired_perm_test,
    paired_tost,
    paired_wilcoxon,
    suite_average_bootstrap,
    unpaired_tost,
    wilson_ci,
)

# Reasoning suite ordering for tables / plots
REASONING_TASKS = [
    "AIME24", "AIME25", "AMC23", "MATH500", "GPQADiamond", "JEEBench",
    "HumanEval", "LiveCodeBench", "MBPP", "IFEval", "gsm8k",
]

# The 11-task reasoning suite that defines the main-paper "Avg" column
# (Table 2 / Figure 3). HumanEval and IFEval are stored as combined-metric
# rows in aggregates.parquet, matching the paper's column definitions.
SUITE_TASKS = REASONING_TASKS

# Which families participate in the matched-compute headline
MATCHED_FAMILIES = ["MV", "Comma", "Nemotron", "FineWeb-Edu", "DCLM"]
# Phase x context cells we sweep
COMPARISON_CELLS = [
    ("base", "4k"), ("sft", "4k"), ("dpo", "4k"),
    ("base", "16k"), ("sft", "16k"), ("dpo", "16k"),
    ("ot30k_on_base", "16k"), ("ot30k_on_sft", "16k"), ("ot30k_on_dpo", "16k"),
]

# Cells that get a matched-compute suite-average difference. Every bar in the
# submission's Figures 2 and 3 needs one, so this spans all three Phase 1
# stages and all three Phase 2 stages rather than just the SFT/DPO endpoints.
SUITE_DIFF_CELLS = [
    ("base", "16k"), ("sft", "16k"), ("dpo", "16k"),
    ("ot30k_on_base", "16k"), ("ot30k_on_sft", "16k"), ("ot30k_on_dpo", "16k"),
]

# §3.3 mixture-composition ablation at 100B-token budget
ABLATION_FAMILIES = ["MV-100", "MV-noweb", "MV-noinst"]
ABLATION_CELLS = [
    ("base", "4k"), ("sft", "4k"), ("dpo", "4k"),
    ("base", "16k"), ("sft", "16k"), ("dpo", "16k"),
]
TOST_MARGINS_PP = [0.02, 0.05, 0.10]  # ±2, ±5, ±10 percentage points

N_BOOT = 10000
N_PERM = 10000
SEED = 20260505


# ---------------------------------------------------------------------------
# helpers

def _seed_offset(key: str) -> int:
    """Stable 32-bit integer offset derived from a string. Used so that
    independent bootstrap calls within one run get independent RNG sequences
    without depending on PYTHONHASHSEED."""
    return int.from_bytes(hashlib.md5(key.encode()).digest()[:4], "little")


def per_item_y(items: pd.DataFrame) -> pd.DataFrame:
    """Collapse multiple attempts per (model, task, item_index) to mean score."""
    return (items.groupby(["model", "family", "phase", "context_tier", "task", "item_index"], as_index=False)
            ["score"].mean().rename(columns={"score": "y"}))


def per_task_accuracy(per_item: pd.DataFrame, agg: pd.DataFrame) -> pd.DataFrame:
    """One row per (model, task) with n, accuracy, Wilson CI, bootstrap CI."""
    rows = []
    for (model, family, phase, ctx, task), g in per_item.groupby(
        ["model", "family", "phase", "context_tier", "task"], dropna=False, sort=False
    ):
        y = g["y"].to_numpy()
        n = len(y)
        acc = float(y.mean())
        w = wilson_ci(y.sum(), n)
        # one-sample bootstrap on the mean — per-row deterministic seed so each
        # (model, task) draws an independent resample sequence, instead of every
        # row sharing the same RNG state (which would correlate their CIs).
        row_seed = SEED + _seed_offset(f"per_task|{model}|{task}")
        rng = np.random.default_rng(row_seed)
        bs = _scipy_stats.bootstrap((y,), statistic=lambda a, axis: a.mean(axis=axis),
                                    vectorized=True, n_resamples=N_BOOT, method="percentile",
                                    random_state=rng)
        rows.append({
            "model": model, "family": family, "phase": phase, "context_tier": ctx,
            "task": task, "n_items": n, "accuracy": acc,
            "wilson_lo": w.lo, "wilson_hi": w.hi,
            "boot_lo": float(bs.confidence_interval.low),
            "boot_hi": float(bs.confidence_interval.high),
            "metric": "item_mean",
        })
    if not agg.empty:
        for _, r in agg.iterrows():
            w = wilson_ci(r["n_correct"], r["n"])
            rows.append({
                "model": r["model"], "family": r["family"], "phase": r["phase"],
                "context_tier": r["context_tier"], "task": r["task"],
                "n_items": int(r["n"]), "accuracy": float(r["accuracy"]),
                "wilson_lo": w.lo, "wilson_hi": w.hi,
                "boot_lo": float("nan"), "boot_hi": float("nan"),
                "metric": r["source_metric"],
            })
    return pd.DataFrame(rows)


def _pivot_per_item(per_item: pd.DataFrame, task: str) -> pd.DataFrame:
    """Wide table: rows=item_index, cols=model, values=y, for a single task."""
    sub = per_item[per_item["task"] == task]
    if sub.empty:
        return pd.DataFrame()
    return sub.pivot_table(index="item_index", columns="model", values="y")


def _pairwise_for_task(piv: pd.DataFrame, agg_for_task: pd.DataFrame | None,
                       models_a: list[str], models_b: list[str], task: str,
                       tost_margins_pp: list[float] | None = None) -> list[dict]:
    """For each (a, b) pair, compute paired bootstrap or Newcombe diff.

    If `tost_margins_pp` is provided, also adds a TOST equivalence column per
    margin. For paired tasks uses ttost_paired; for unpaired (Newcombe) uses
    ttost_ind.
    """
    rows = []
    pairs_seen: set[tuple[str, str]] = set()
    for a in models_a:
        for b in models_b:
            if a == b:
                continue
            key = tuple(sorted([a, b]))
            if key in pairs_seen:
                continue
            pairs_seen.add(key)
            if not piv.empty and a in piv.columns and b in piv.columns:
                col_a = piv[a]
                col_b = piv[b]
                n_a = int(col_a.notna().sum())
                n_b = int(col_b.notna().sum())
                ya = col_a.to_numpy()
                yb = col_b.to_numpy()
                mask = ~(np.isnan(ya) | np.isnan(yb))
                ya, yb = ya[mask], yb[mask]
                n_paired = int(len(ya))
                # If the two models have very different item counts (more than
                # one item discarded by the pair-wise NaN mask), the alignment
                # is suspect — the harness may have emitted items in a
                # different order or with a different gold-set. We still run
                # the test on the intersection but flag it for the audit.
                alignment_mismatch = (n_paired < min(n_a, n_b)) or (n_a != n_b)
                if n_paired > 0:
                    if alignment_mismatch:
                        print(f"WARNING pairing mismatch for task={task} "
                              f"a={a} (n={n_a}) b={b} (n={n_b}) paired={n_paired}")
                    ci = paired_bootstrap_ci(ya, yb, n_resamples=N_BOOT, seed=SEED)
                    _, p_perm = paired_perm_test(ya, yb, n_resamples=N_PERM, seed=SEED)
                    p_wilcoxon = paired_wilcoxon(ya, yb)
                    degenerate = bool(np.allclose(ya, yb))
                    row = {
                        "task": task, "model_a": a, "model_b": b, "n": n_paired,
                        "n_a": n_a, "n_b": n_b, "alignment_mismatch": alignment_mismatch,
                        "delta": ci.point, "ci_lo": ci.lo, "ci_hi": ci.hi,
                        "perm_p": p_perm, "wilcoxon_p": p_wilcoxon,
                        "method": "paired_bootstrap",
                        "degenerate_ya_eq_yb": degenerate,
                    }
                    if tost_margins_pp is not None:
                        for m in tost_margins_pp:
                            p, rej = paired_tost(ya, yb, margin=m)
                            row[f"tost_p_{int(m*100):d}pp"] = p
                            row[f"tost_reject_{int(m*100):d}pp"] = rej
                    rows.append(row)
                    continue
            # fallback: Newcombe unpaired
            if agg_for_task is not None:
                ar = agg_for_task[agg_for_task.model == a]
                br = agg_for_task[agg_for_task.model == b]
                if len(ar) and len(br):
                    ar = ar.iloc[0]; br = br.iloc[0]
                    ci = newcombe_diff_ci(int(ar.n_correct), int(ar.n), int(br.n_correct), int(br.n))
                    row = {
                        "task": task, "model_a": a, "model_b": b,
                        "n": int(min(ar.n, br.n)),
                        "n_a": int(ar.n), "n_b": int(br.n),
                        "alignment_mismatch": False,
                        "delta": ci.point, "ci_lo": ci.lo, "ci_hi": ci.hi,
                        "perm_p": float("nan"), "wilcoxon_p": float("nan"),
                        "method": "newcombe_unpaired",
                        "degenerate_ya_eq_yb": False,
                    }
                    if tost_margins_pp is not None:
                        for m in tost_margins_pp:
                            p, rej = unpaired_tost(int(ar.n_correct), int(ar.n),
                                                   int(br.n_correct), int(br.n), margin=m)
                            row[f"tost_p_{int(m*100):d}pp"] = p
                            row[f"tost_reject_{int(m*100):d}pp"] = rej
                    rows.append(row)
    return rows


def run_pairwise(per_item: pd.DataFrame, agg: pd.DataFrame, family_meta: pd.DataFrame,
                 families: list[str], cells: list[tuple[str, str]],
                 tost_margins_pp: list[float] | None = None) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Generic pairwise sweep over arbitrary family list and (phase, ctx) cells."""
    out: list[dict] = []
    audit: list[dict] = []
    n_rows_by_model = per_item.groupby("model").size()
    for phase, ctx in cells:
        cell_models = family_meta[
            (family_meta.phase == phase) & (family_meta.context_tier == ctx)
            & (family_meta.family.isin(families))
        ]
        if len(cell_models) < 2:
            continue
        cell_models = (cell_models.assign(_n=cell_models.model.map(n_rows_by_model).fillna(0))
                       .sort_values(["_n", "model"], ascending=[False, True])
                       .drop_duplicates("family")
                       .drop(columns=["_n"]))
        models = cell_models.model.tolist()
        for task in REASONING_TASKS:
            agg_task = agg[(agg.task == task) & (agg.model.isin(models))] if not agg.empty else None
            piv = _pivot_per_item(per_item[per_item.model.isin(models)], task)
            new_rows = _pairwise_for_task(piv, agg_task, models, models, task,
                                          tost_margins_pp=tost_margins_pp)
            for r in new_rows:
                r.update({"phase": phase, "context_tier": ctx,
                          "family_a": cell_models[cell_models.model == r["model_a"]].family.iloc[0],
                          "family_b": cell_models[cell_models.model == r["model_b"]].family.iloc[0]})
                out.append(r)
                audit.append({
                    "phase": phase, "context_tier": ctx, "task": task,
                    "model_a": r["model_a"], "model_b": r["model_b"],
                    "method": r["method"], "n_a": r["n_a"], "n_b": r["n_b"],
                    "n_paired": r["n"], "alignment_mismatch": r["alignment_mismatch"],
                })
    df = pd.DataFrame(out)
    if not df.empty:
        df["q_bh"] = float("nan")
        for _, idx in df.groupby(["phase", "context_tier", "family_a", "family_b"]).groups.items():
            sub = df.loc[idx]
            df.loc[idx, "q_bh"] = bh_adjust(sub["perm_p"].tolist())
    audit_df = pd.DataFrame(audit)
    return df, audit_df


def _collect_pair_task_cells(per_item: pd.DataFrame, agg: pd.DataFrame,
                              a: str, b: str, tasks: list[str],
                              rng: np.random.Generator, n_resamples: int) -> dict:
    """For a single (a, b) pair, build one dict[task] -> cell describing how to
    bootstrap that task's contribution to the suite-average diff.

    Cell shape (mutually exclusive):
      {"kind": "paired", "y_a": np.ndarray, "y_b": np.ndarray,
       "obs_a": float, "obs_b": float}                          # per-item, paired
      {"kind": "beta", "draws_a": np.ndarray, "draws_b": np.ndarray,
       "obs_a": float, "obs_b": float}                          # aggregate-only

    Aggregate Beta draws are pre-generated up front (n_resamples each) so the
    inner bootstrap loop just indexes by k.
    """
    out: dict[str, dict] = {}
    for t in tasks:
        sub_a = per_item[(per_item.model == a) & (per_item.task == t)]
        sub_b = per_item[(per_item.model == b) & (per_item.task == t)]
        if len(sub_a) > 0 and len(sub_b) > 0:
            merged = (sub_a[["item_index", "y"]].rename(columns={"y": "y_a"})
                      .merge(sub_b[["item_index", "y"]].rename(columns={"y": "y_b"}),
                             on="item_index"))
            if len(merged) > 0:
                y_a = merged["y_a"].to_numpy()
                y_b = merged["y_b"].to_numpy()
                out[t] = {"kind": "paired", "y_a": y_a, "y_b": y_b,
                          "obs_a": float(y_a.mean()), "obs_b": float(y_b.mean())}
                continue
        if not agg.empty:
            ag_a = agg[(agg.model == a) & (agg.task == t)]
            ag_b = agg[(agg.model == b) & (agg.task == t)]
            if len(ag_a) and len(ag_b):
                ka, na = int(ag_a.n_correct.iloc[0]), int(ag_a.n.iloc[0])
                kb, nb = int(ag_b.n_correct.iloc[0]), int(ag_b.n.iloc[0])
                draws_a = rng.beta(ka + 0.5, na - ka + 0.5, size=n_resamples)
                draws_b = rng.beta(kb + 0.5, nb - kb + 0.5, size=n_resamples)
                out[t] = {"kind": "beta", "draws_a": draws_a, "draws_b": draws_b,
                          "obs_a": float(draws_a.mean()), "obs_b": float(draws_b.mean())}
    return out


def _bootstrap_suite_diff(tcells: dict, rng: np.random.Generator,
                          n_resamples: int) -> np.ndarray:
    """Stratified paired bootstrap of the unweighted suite-average difference.

    For "paired" cells, resample item indices (same draw for A and B). For
    "beta" cells, index into the pre-drawn Beta sample sequence. Returns the
    array of length n_resamples of (mean_a - mean_b) replicates.
    """
    tasks = sorted(tcells.keys())
    out = np.empty(n_resamples, dtype=float)
    for k in range(n_resamples):
        ma = np.empty(len(tasks))
        mb = np.empty(len(tasks))
        for ti, t in enumerate(tasks):
            c = tcells[t]
            if c["kind"] == "paired":
                ya, yb = c["y_a"], c["y_b"]
                idx = rng.integers(0, len(ya), len(ya))
                ma[ti] = ya[idx].mean()
                mb[ti] = yb[idx].mean()
            else:  # "beta"
                ma[ti] = c["draws_a"][k]
                mb[ti] = c["draws_b"][k]
        out[k] = float(ma.mean() - mb.mean())
    return out


def suite_avg_diff(per_item: pd.DataFrame, agg: pd.DataFrame, family_meta: pd.DataFrame,
                   families: list[str], cells: list[tuple[str, str]],
                   tost_margins_pp: list[float], suite_tasks: list[str] | None = None,
                   n_resamples: int = 5000) -> pd.DataFrame:
    """Pairwise stratified-bootstrap diff of the unweighted suite average.

    For per-item tasks, resample item indices within each task (same indices for
    A and B — paired). For aggregate-only tasks, draw Beta(k+0.5, n-k+0.5)
    independently for A and B (unpaired). The TOST equivalence column at each
    margin is computed on the bootstrap distribution of the diff:
      tost_reject_<m>pp = True iff the (1 - 2*0.05) = 90% bootstrap CI lies
      strictly inside (-margin, +margin). This is the bootstrap-CI-based TOST
      analog and matches the convention "claim equivalence iff CI fits inside
      the margin band".
    """
    if suite_tasks is None:
        suite_tasks = REASONING_TASKS
    n_rows_by_model = per_item.groupby("model").size()
    rows = []
    for phase, ctx in cells:
        cell_models = family_meta[
            (family_meta.phase == phase) & (family_meta.context_tier == ctx)
            & (family_meta.family.isin(families))
        ]
        if len(cell_models) < 2:
            continue
        cell_models = (cell_models.assign(_n=cell_models.model.map(n_rows_by_model).fillna(0))
                       .sort_values(["_n", "model"], ascending=[False, True])
                       .drop_duplicates("family")
                       .drop(columns=["_n"]))
        models = cell_models.model.tolist()
        # Build per_task_a/b dicts for each pair
        for i in range(len(models)):
            for j in range(i + 1, len(models)):
                a, b = models[i], models[j]
                fam_a = cell_models[cell_models.model == a].family.iloc[0]
                fam_b = cell_models[cell_models.model == b].family.iloc[0]
                avail = set(per_item.task.unique())
                if not agg.empty:
                    avail |= set(agg.task.unique())
                tasks = sorted(set(suite_tasks) & avail)
                rng = np.random.default_rng(SEED + _seed_offset(f"suite_diff|{phase}|{ctx}|{a}|{b}"))
                tcells = _collect_pair_task_cells(per_item, agg, a, b, tasks, rng, n_resamples)
                if not tcells:
                    continue
                obs_a = float(np.mean([c["obs_a"] for c in tcells.values()]))
                obs_b = float(np.mean([c["obs_b"] for c in tcells.values()]))
                obs_diff = obs_a - obs_b
                boot_diffs = _bootstrap_suite_diff(tcells, rng, n_resamples)
                lo95, hi95 = np.quantile(boot_diffs, [0.025, 0.975])
                lo90, hi90 = np.quantile(boot_diffs, [0.05, 0.95])
                row = {
                    "phase": phase, "context_tier": ctx,
                    "family_a": fam_a, "family_b": fam_b,
                    "model_a": a, "model_b": b,
                    "n_tasks": len(tcells),
                    "suite_avg_a": obs_a, "suite_avg_b": obs_b,
                    "diff": obs_diff,
                    "boot_lo": float(lo95), "boot_hi": float(hi95),
                    "boot_lo_90": float(lo90), "boot_hi_90": float(hi90),
                }
                for m in tost_margins_pp:
                    # bootstrap-CI TOST: 90% CI inside (-margin, +margin)
                    inside = (lo90 > -m) and (hi90 < m)
                    row[f"tost_reject_{int(m*100):d}pp"] = bool(inside)
                rows.append(row)
    return pd.DataFrame(rows)


def suite_avg_ci(per_item: pd.DataFrame, agg: pd.DataFrame, family_meta: pd.DataFrame) -> pd.DataFrame:
    """For every (model), compute a stratified bootstrap CI of the suite average.

    Per-item tasks contribute their own per-item arrays. Aggregate-only tasks
    contribute a Beta(n_correct + 1/2, n - n_correct + 1/2) draw to stand in
    for missing per-item resampling (Jeffreys prior beta-binomial).
    """
    rows = []
    rng = np.random.default_rng(SEED)
    n_resamples = 5000
    models = sorted(per_item.model.unique().tolist())
    for model in tqdm(models, desc="suite_avg"):
        per_task: dict[str, np.ndarray] = {}
        # per-item tasks
        sub = per_item[per_item.model == model]
        for task, g in sub.groupby("task"):
            if task not in SUITE_TASKS:
                continue
            per_task[task] = g["y"].to_numpy()
        # aggregate-only tasks: parametric Beta substitute
        for _, r in agg[agg.model == model].iterrows():
            if r.task not in SUITE_TASKS:
                continue
            n = int(r.n); k = int(r.n_correct)
            # draw a single Beta sample of n outcomes per replicate? We need a
            # vector — use n IID Bernoullis with p drawn from Beta(k+0.5, n-k+0.5).
            # Cheaper: draw resample mean from Beta directly, equivalent in
            # expectation. We'll fold it into a "fake" per-task array of length 1
            # whose bootstrap variability we override below.
            # Simpler: pre-draw n_resamples Beta means for this task.
            beta_draws = rng.beta(k + 0.5, n - k + 0.5, size=n_resamples)
            per_task[r.task] = ("beta", beta_draws)
        if not per_task:
            continue
        # custom suite bootstrap: handles tuple-as-beta-draws
        tasks = sorted(per_task.keys())
        boot_means = np.empty(n_resamples)
        for r_i in range(n_resamples):
            ms = []
            for t in tasks:
                v = per_task[t]
                if isinstance(v, tuple) and v[0] == "beta":
                    ms.append(v[1][r_i])
                else:
                    arr = v
                    idx = rng.integers(0, len(arr), len(arr))
                    ms.append(arr[idx].mean())
            boot_means[r_i] = float(np.mean(ms))
        # observed
        obs_parts = []
        for t in tasks:
            v = per_task[t]
            if isinstance(v, tuple):
                obs_parts.append(float(v[1].mean()))  # Beta posterior mean ≈ k+0.5/(n+1)
            else:
                obs_parts.append(float(v.mean()))
        obs = float(np.mean(obs_parts))
        lo, hi = np.quantile(boot_means, [0.025, 0.975])
        meta = family_meta[family_meta.model == model].iloc[0]
        rows.append({
            "model": model, "family": meta.family, "phase": meta.phase,
            "context_tier": meta.context_tier, "n_tasks": len(tasks),
            "suite_avg": obs, "boot_lo": float(lo), "boot_hi": float(hi),
        })
    return pd.DataFrame(rows)


def main(data_dir: Path, art_dir: Path, phase2_source: str = "hf") -> int:
    if phase2_source not in {"hf", "logs"}:
        raise ValueError(f"phase2_source must be 'hf' or 'logs', got {phase2_source!r}")
    art_dir.mkdir(exist_ok=True, parents=True)
    # Create the per-section subdirs all CSV writers below will land into
    # (paths constants embed `matched_compute/`, `ablation/`, `audit/`).
    for sub in ("matched_compute", "ablation", "audit"):
        (art_dir / sub).mkdir(exist_ok=True)
    # Read Phase 1 (items_long.parquet) and Phase 2 (phase2_items.parquet,
    # if present) and concatenate. Clean source separation — there's no merged
    # parquet on disk. The two have disjoint phase values, so concat is safe.
    phase1_path = data_dir / paths.ITEMS_LONG
    if not phase1_path.exists():
        raise FileNotFoundError(
            f"{phase1_path.name} not found — run `mvpt parse-phase1` first"
        )
    items_frames = [pd.read_parquet(phase1_path)]
    agg_path = data_dir / paths.AGGREGATES
    agg = pd.read_parquet(agg_path) if agg_path.exists() else pd.DataFrame()
    if not agg.empty:
        # HumanEval and IFEval mean different things under the two metric
        # conventions, so a parquet built under the other one must not be
        # analysed by this code. Fails loudly rather than reporting a mixture.
        conventions.check_frame(agg, agg_path.name)

    # Phase 2 has two possible sources and they must not both be loaded —
    # they describe the same (model, phase) cells and would double-count.
    #
    #   HF dump  (phase2_items)      5 scorable tasks, but reproduces the
    #                                manuscript's Table 1: AIME24 and AMC23
    #                                match every MV cell exactly, MATH500 and
    #                                GSM8K within 3.2 pp, references within
    #                                0.6 pp.
    #   logs.zip (phase2_logs_*)     all 11 reasoning benchmarks, but does
    #                                *not* reproduce Table 1 — 8 of 35 cells
    #                                agree within 1.0 pp, HumanEval is off by
    #                                50.6 pp. See docs/phase2-aime-provenance.md.
    #
    # Default to the HF dump: broader task coverage is worth nothing if the
    # cells disagree with the paper they are meant to quantify. Pass
    # `--phase2-source logs` to get the 11-task view, understanding that its
    # numbers are not the manuscript's.
    phase2_logs_path = data_dir / paths.PHASE2_LOGS_ITEMS
    phase2_hf_path = data_dir / paths.PHASE2_ITEMS
    if phase2_source == "logs" and phase2_logs_path.exists():
        items_frames.append(pd.read_parquet(phase2_logs_path))
        sources = f"{phase1_path.name} + {phase2_logs_path.name}"
        phase2_agg_path = data_dir / paths.PHASE2_LOGS_AGGREGATES
        if phase2_agg_path.exists() and not agg.empty:
            agg = pd.concat([agg, pd.read_parquet(phase2_agg_path)],
                            ignore_index=True)
            sources += f" + {phase2_agg_path.name}"
        print(f"reading items from {sources}")
        print("warning: Phase 2 sourced from logs.zip, which does not reproduce "
              "submission Table 1 — see docs/phase2-aime-provenance.md")
    elif phase2_source == "hf" and phase2_hf_path.exists():
        items_frames.append(pd.read_parquet(phase2_hf_path))
        print(f"reading items from {phase1_path.name} + {phase2_hf_path.name}")
        if phase2_logs_path.exists():
            print(f"note: {phase2_logs_path.name} present but not loaded — "
                  f"pass --phase2-source logs to use it instead")
    else:
        print(f"reading items from {phase1_path.name} (no Phase 2)")
    items = pd.concat(items_frames, ignore_index=True) if len(items_frames) > 1 else items_frames[0]

    per_item = per_item_y(items)

    # one row per model with family metadata
    family_meta = (per_item[["model", "family", "phase", "context_tier"]]
                   .drop_duplicates("model").reset_index(drop=True))
    if not agg.empty:
        family_meta = pd.concat([
            family_meta,
            agg[["model", "family", "phase", "context_tier"]].drop_duplicates("model"),
        ]).drop_duplicates("model").reset_index(drop=True)

    print(f"models: {len(family_meta)}, per_item rows: {len(per_item)}, agg rows: {len(agg)}")

    # Warn about families that exist in the data but never enter any analysis
    # output. Common culprit: Nemotron-altpath / Nemotron-rope1M, which the
    # registry separates from the canonical Nemotron family on purpose. The
    # warning makes the silent drop visible at run time.
    analysis_families = set(MATCHED_FAMILIES) | set(ABLATION_FAMILIES)
    data_families = set(family_meta.family.unique())
    unused = sorted(data_families - analysis_families)
    if unused:
        print(f"NOTE families present in data but not in any analysis: {unused}")

    # 1) per_task_accuracy
    print("\n[1/4] per_task_accuracy ...")
    pta = per_task_accuracy(per_item, agg)
    pta.to_csv(art_dir / paths.PER_TASK_ACCURACY, index=False)

    # named-subset orientation helper
    def _filt(df, fa, fb, tost_cols=()):
        m = (((df.family_a == fa) & (df.family_b == fb)) |
             ((df.family_a == fb) & (df.family_b == fa)))
        out = df[m].copy()
        flip = out.family_a == fb
        for col_a, col_b in [("family_a", "family_b"), ("model_a", "model_b")]:
            out.loc[flip, [col_a, col_b]] = out.loc[flip, [col_b, col_a]].values
        out.loc[flip, "delta"] = -out.loc[flip, "delta"]
        # ci_lo/hi need to be negated AND swapped on flip
        new_lo = np.where(flip, -out["ci_hi"].values, out["ci_lo"].values)
        new_hi = np.where(flip, -out["ci_lo"].values, out["ci_hi"].values)
        out["ci_lo"] = new_lo
        out["ci_hi"] = new_hi
        return out.sort_values(["phase", "context_tier", "task"]).reset_index(drop=True)

    # 2) pairwise — matched compute
    print("\n[2/5] pairwise full matrix (matched compute) ...")
    pf, audit = run_pairwise(per_item, agg, family_meta, MATCHED_FAMILIES, COMPARISON_CELLS)
    if not pf.empty:
        pf = pf[["phase", "context_tier", "task", "family_a", "family_b",
                 "model_a", "model_b", "n", "n_a", "n_b", "alignment_mismatch",
                 "delta", "ci_lo", "ci_hi",
                 "perm_p", "wilcoxon_p", "q_bh",
                 "degenerate_ya_eq_yb", "method"]]
        pf.to_csv(art_dir / paths.PAIRWISE_FULL_MATRIX, index=False)
        _filt(pf, "MV", "Comma").to_csv(art_dir / paths.PAIRWISE_MV_VS_COMMA, index=False)
        _filt(pf, "MV", "Nemotron").to_csv(art_dir / paths.PAIRWISE_MV_VS_NEMOTRON, index=False)
        _filt(pf, "MV", "FineWeb-Edu").to_csv(art_dir / paths.PAIRWISE_MV_VS_FINEWEB, index=False)
        _filt(pf, "Comma", "Nemotron").to_csv(art_dir / paths.PAIRWISE_COMMA_VS_NEMOTRON, index=False)
    audit.to_csv(art_dir / paths.PAIRING_AUDIT, index=False)

    # 3) pairwise — §3.3 ablation (TOST equivalence columns added)
    print("\n[3/5] pairwise full matrix (ablation, with TOST) ...")
    pf_ab, audit_ab = run_pairwise(per_item, agg, family_meta,
                                   ABLATION_FAMILIES, ABLATION_CELLS,
                                   tost_margins_pp=TOST_MARGINS_PP)
    if not pf_ab.empty:
        tost_cols = []
        for m in TOST_MARGINS_PP:
            tost_cols += [f"tost_p_{int(m*100):d}pp", f"tost_reject_{int(m*100):d}pp"]
        pf_ab = pf_ab[["phase", "context_tier", "task", "family_a", "family_b",
                       "model_a", "model_b", "n", "n_a", "n_b", "alignment_mismatch",
                       "delta", "ci_lo", "ci_hi",
                       "perm_p", "wilcoxon_p", "q_bh",
                       "degenerate_ya_eq_yb", "method"] + tost_cols]
        pf_ab.to_csv(art_dir / paths.PAIRWISE_ABLATION_FULL_MATRIX, index=False)
        _filt(pf_ab, "MV-noweb", "MV-noinst").to_csv(art_dir / paths.PAIRWISE_MVNOWEB_VS_MVNOINST, index=False)
        _filt(pf_ab, "MV-noweb", "MV-100").to_csv(art_dir / paths.PAIRWISE_MVNOWEB_VS_MV100, index=False)
        _filt(pf_ab, "MV-noinst", "MV-100").to_csv(art_dir / paths.PAIRWISE_MVNOINST_VS_MV100, index=False)

    # 4) suite-average diff for ablation pairs (with bootstrap-CI TOST)
    print("\n[4/6] suite_avg_diff_ablation ...")
    sad = suite_avg_diff(per_item, agg, family_meta,
                         ABLATION_FAMILIES, ABLATION_CELLS,
                         tost_margins_pp=TOST_MARGINS_PP, suite_tasks=SUITE_TASKS)
    if not sad.empty:
        sad.to_csv(art_dir / paths.SUITE_AVG_DIFF_ABLATION, index=False)

    # 5) suite-average diff for matched-compute pairs (Phase 1 SFT/DPO, 16K).
    # HumanEval is substituted with the combined python+sh metric so the
    # 11-task suite reproduces the main-paper Table 2 "Avg" column.
    print("\n[5/6] suite_avg_diff_matched ...")
    # No longer overriding: io.py now emits HumanEval as the combined
    # python+sh metric and IFEval as the combined prompt+inst metric, so the
    # raw aggregates match the paper's column definitions directly.
    agg_suite = agg
    sadm = suite_avg_diff(per_item, agg_suite, family_meta,
                          MATCHED_FAMILIES, SUITE_DIFF_CELLS,
                          tost_margins_pp=TOST_MARGINS_PP, suite_tasks=SUITE_TASKS)
    if not sadm.empty:
        sadm.to_csv(art_dir / paths.SUITE_AVG_DIFF_MATCHED, index=False)

    # 6) suite-average per-model CI
    print("\n[6/6] suite_average_ci ...")
    sav = suite_avg_ci(per_item, agg, family_meta)
    sav.to_csv(art_dir / paths.SUITE_AVERAGE_CI, index=False)

    print("\ndone. artifacts/:")
    for f in sorted(p for p in art_dir.rglob("*") if p.is_file()):
        print(" ", f.relative_to(art_dir))
    return 0

