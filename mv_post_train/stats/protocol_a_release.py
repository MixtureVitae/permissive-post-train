"""Compare the Protocol A per-sample records with the released three-seed grid.

The workshop tables are computed from `canonical_reasoning_per_sample.csv`
(`data/protocol_a.py`). Part of the same grid was released earlier as
`phase1/paper-decoding-3-seeds/per_sample_scores.csv` in
`ali-elganzory/mv-posttraining-scores` (README input 5): seven benchmarks, the
Phase 1, no front-loading arms. This command lines the two up item by item
wherever both have a (checkpoint, seed, benchmark, item) and reports, per
checkpoint and benchmark, how many items disagree and the two accuracies.

A benchmark that agrees everywhere shows the paper's number is the released
one. A benchmark that disagrees broadly marks a re-run whose records exist
only in the newer file. Checkpoints present in only one file are listed too,
because a renamed checkpoint and a different checkpoint look the same here.

Writes `<art-dir>/release_comparison.csv` and `<art-dir>/release_only_in_one.csv`.
"""
from __future__ import annotations

from pathlib import Path

import pandas as pd

from ..data import protocol_a as pa

KEY = ["model_id", "seed", "benchmark", "sample_id"]


def compare(reasoning: pd.DataFrame, release: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    rel = release.rename(columns={"huggingface_model_id": "model_id"})
    rel = rel.assign(released=rel["sample_score_percent"] / 100.0)[KEY + ["released"]]
    both = reasoning[KEY + ["correct"]].merge(rel, on=KEY, how="inner")
    both["differs"] = both["correct"] != both["released"]
    summary = (both.groupby(["model_id", "benchmark"])
               .agg(items=("differs", "size"), items_differ=("differs", "sum"),
                    paper_input=("correct", "mean"), released=("released", "mean"))
               .reset_index())
    summary[["paper_input", "released"]] *= 100.0
    rel_benches = set(rel["benchmark"])
    ours = set(reasoning.loc[reasoning["benchmark"].isin(rel_benches), "model_id"])
    theirs = set(rel["model_id"])
    only = pd.DataFrame(
        [("paper input only", m) for m in sorted(ours - theirs)]
        + [("release only", m) for m in sorted(theirs - ours)],
        columns=["where", "model_id"])
    return summary, only


def main(data_dir: Path, release_csv: Path, art_dir: Path) -> int:
    reasoning = pa.load_reasoning(data_dir / pa.REASONING_CSV)
    summary, only = compare(reasoning, pd.read_csv(release_csv))
    art_dir.mkdir(parents=True, exist_ok=True)
    summary.to_csv(art_dir / "release_comparison.csv", index=False)
    only.to_csv(art_dir / "release_only_in_one.csv", index=False)

    per_bench = (summary.assign(cell_differs=summary["items_differ"] > 0)
                 .groupby("benchmark")
                 .agg(checkpoints=("model_id", "size"), checkpoints_differ=("cell_differs", "sum"),
                      items=("items", "sum"), items_differ=("items_differ", "sum")))
    print("Paper input vs released three-seed grid, items present in both:")
    print(per_bench.to_string())
    print(f"\n{(only['where'] == 'paper input only').sum()} checkpoints only in the paper input, "
          f"{(only['where'] == 'release only').sum()} only in the release "
          f"(see release_only_in_one.csv).")
    return 0
