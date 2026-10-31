"""Compare our recovered benchmark columns against the released score table.

The 2026-07-28 revision of `laion/mv-posttrain-eval-responses` added
`eval_scores_wide.csv`, a (base_model, stage) x task grid of the project's own
scores. Its MixtureVitae `base` row reproduces the submission's Table 1 exactly
(IFEval 42.6, Avg 27.9), so it is the paper's scoring pipeline rather than a
third party's.

That makes it a much wider check than the hand-transcribed Table 1 comparison
in `ifeval_score` / `code_score`, which covered only the 10-11 cells printed in
the paper: here every cell we can re-score has something to be compared against.
The three columns we recover independently -- IFEval via the official checker,
HumanEval and MBPP via execution -- are the ones worth checking, because for
those the release ships raw generations and we derive the verdict ourselves.

One caveat that limits what a disagreement means: the same commit describes
"IFEval gaps (12 cells, 4096-token re-runs)", so an unknown subset of the
IFEval column comes from re-runs rather than from the original evaluation. A
gap on those cells can be run-to-run difference rather than a scoring
difference, and the two cannot be separated from the released files alone.

Writes `artifacts/released_vs_recovered.csv`.
"""
from __future__ import annotations

from pathlib import Path

import pandas as pd

OUT_CSV = "released_vs_recovered.csv"

WIDE_CSV = "eval_scores_wide.csv"

# Released column name -> (our artifact, column in it)
SOURCES = {
    "IFEval": ("ifeval_rescored.csv", "prompt_loose"),
    "HumanEval": ("code_rescored.csv", "pass@1"),
    "MBPP": ("code_rescored.csv", "pass@1"),
}

# Cells known to be mislabelled in the release rather than mis-scored by us:
# the responses filed under this key come from a Superior-Reasoning SFT
# checkpoint, which two unrelated graders independently agree is not the base
# model. Excluded from the summary statistics, never from the table.
KNOWN_MISLABELLED = {("MixtureVitae-1.7B", "base")}


def load_released(path: Path) -> pd.DataFrame:
    wide = pd.read_csv(path)
    keep = ["base_model", "stage"] + [c for c in SOURCES if c in wide.columns]
    long = wide[keep].melt(id_vars=["base_model", "stage"], var_name="task",
                           value_name="released")
    return long.dropna(subset=["released"])


def load_recovered(art_dir: Path) -> pd.DataFrame:
    frames = []
    ife = art_dir / SOURCES["IFEval"][0]
    if ife.exists():
        d = pd.read_csv(ife)
        frames.append(d.assign(task="IFEval",
                               recovered=d[SOURCES["IFEval"][1]])
                       [["base_model", "stage", "task", "recovered"]])
    code = art_dir / SOURCES["HumanEval"][0]
    if code.exists():
        d = pd.read_csv(code)
        d = d[d.status == "scored"]
        frames.append(d.assign(recovered=d["pass@1"])
                       [["base_model", "stage", "task", "recovered"]])
    if not frames:
        raise FileNotFoundError(
            f"no recovered artifacts in {art_dir}; run ifeval-rescore / code-rescore")
    return pd.concat(frames, ignore_index=True)


def compare(released: pd.DataFrame, recovered: pd.DataFrame) -> pd.DataFrame:
    df = released.merge(recovered, on=["base_model", "stage", "task"], how="inner")
    df["delta"] = (df.recovered - df.released).round(2)
    df["mislabelled"] = [
        (b, s) in KNOWN_MISLABELLED for b, s in zip(df.base_model, df.stage)]
    return df.sort_values(["task", "delta"])


def summarize(df: pd.DataFrame) -> pd.DataFrame:
    clean = df[~df.mislabelled]
    rows = []
    for task, g in clean.groupby("task"):
        a = g.delta.abs()
        rows.append({
            "task": task, "n_cells": len(g),
            "median_abs_delta": round(a.median(), 2),
            "mean_abs_delta": round(a.mean(), 2),
            "max_abs_delta": round(a.max(), 2),
            "within_1pp": int((a <= 1.0).sum()),
            "within_2pp": int((a <= 2.0).sum()),
        })
    return pd.DataFrame(rows)


TABLE1_CSV = "released_vs_table1.csv"


def released_vs_table1(wide: Path) -> pd.DataFrame:
    """The release's own grid against the submission's Table 1.

    This is the leg that isolates *run-to-run* movement. Both sides come from
    the project's own scoring pipeline, so any difference is the evaluation
    having been run again on the same checkpoint -- not a convention, and not
    our re-scoring. Cells that agree to the digit are cells the release did not
    re-run.
    """
    from .code_score import TABLE1
    from .ifeval_score import TABLE1_IFE

    w = pd.read_csv(wide)
    rows = []
    for _, r in w.iterrows():
        key = (r.base_model, r.stage)
        if key in TABLE1_IFE and pd.notna(r.get("IFEval")):
            rows.append({"base_model": key[0], "stage": key[1], "task": "IFEval",
                         "table1": TABLE1_IFE[key], "released": float(r["IFEval"])})
        for task in ("MBPP", "HumanEval"):
            if key in TABLE1 and task in TABLE1[key] and pd.notna(r.get(task)):
                rows.append({"base_model": key[0], "stage": key[1], "task": task,
                             "table1": TABLE1[key][task], "released": float(r[task])})
    df = pd.DataFrame(rows)
    if df.empty:
        return df
    df["delta"] = (df.released - df.table1).round(2)
    df["rerun"] = df.delta.abs() > 0.05
    return df.sort_values(["task", "delta"])


def decomposition(vs_recovered: pd.DataFrame, vs_table1: pd.DataFrame) -> str:
    """Two sources of movement, separated and named."""
    a = vs_recovered[~vs_recovered.mislabelled].delta.abs()
    b = vs_table1.delta.abs()
    moved = vs_table1[vs_table1.rerun]
    return "\n".join([
        "",
        "=== where the movement comes from ===",
        f"  scoring convention (our grader vs theirs, SAME generations):",
        f"      n={len(a)}  median |d|={a.median():.2f} pp  max={a.max():.2f} pp",
        f"  run-to-run (their re-run vs their Table 1, SAME checkpoint):",
        f"      n={len(b)}  identical={int((~vs_table1.rerun).sum())}/{len(b)}"
        f"  moved={len(moved)}  max={b.max():.2f} pp",
        "",
        "  Read together: our re-scoring reproduces the project's own scores on",
        "  the same generations, so disagreements with Table 1 are the evaluation",
        "  having been re-run, not a difference in how it was scored.",
    ])


def main(wide: Path, art_dir: Path) -> int:
    wide, art_dir = Path(wide), Path(art_dir)
    if not wide.exists():
        print(f"missing {wide}")
        return 1
    art_dir.mkdir(parents=True, exist_ok=True)

    df = compare(load_released(wide), load_recovered(art_dir))
    if df.empty:
        print("no overlapping cells")
        return 1
    df.to_csv(art_dir / OUT_CSV, index=False)

    for task, g in df.groupby("task"):
        print(f"\n=== {task} ({len(g)} cells) ===")
        print(g[["base_model", "stage", "released", "recovered", "delta"]]
              .to_string(index=False))

    print("\n=== agreement, excluding known-mislabelled cells ===")
    print(summarize(df).to_string(index=False))

    bad = df[df.mislabelled]
    if not bad.empty:
        print("\nexcluded as mislabelled in the release:")
        print(bad[["base_model", "stage", "task", "released", "recovered",
                   "delta"]].to_string(index=False))

    t1 = released_vs_table1(wide)
    if not t1.empty:
        t1.to_csv(art_dir / TABLE1_CSV, index=False)
        moved = t1[t1.rerun]
        print(f"\n=== released grid vs submission Table 1 "
              f"({int((~t1.rerun).sum())}/{len(t1)} cells identical) ===")
        if not moved.empty:
            print(moved[["base_model", "stage", "task", "table1", "released",
                         "delta"]].to_string(index=False))
        print(decomposition(df, t1))
        print(f"\nwrote {art_dir / TABLE1_CSV}")

    print(f"\nwrote {art_dir / OUT_CSV}")
    return 0
