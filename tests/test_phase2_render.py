"""Tests for the Phase 2 tables rendered from the manifest (mvpt render-phase2).

They pin the rules that make the tables trustworthy: a printed change is
against the checkpoint's recorded parent, printed above it; a change the
manifest rejected is refused; an excluded task has no column and the caption
says why; a printed change is the difference of the printed averages; and the
per-question records must reproduce the manifest before an interval is drawn
from them. The last test resolves every row of the real tables against the
committed manifest.
"""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from mv_post_train.manifest.reader import Manifest, ManifestError
from mv_post_train.render import phase2_tables as pt
from mv_post_train.stats.paired_items import item_scores, paired_changes

P2 = "phase2_three_seed"
TASKS = ("AIME24", "MATH500")


def _manifest(avgs: dict[str, float], parents: dict[str, str], *, rejected=(),
              runs: dict[str, list[float]] | None = None) -> Manifest:
    """Checkpoints named after their stage, all of base model M, scoring
    `avgs[stage]` on every task."""
    checkpoints, results, suites, changes = {}, [], [], []
    for stage, avg in avgs.items():
        parent = parents.get(stage)
        checkpoints[stage] = {"parent": parent, "parent_status": "verified" if parent else
                              "none recorded", "labels": {P2: {"base_model": "M", "stage": stage}}}
        per_run = (runs or {}).get(stage, [avg] * 3)
        for t in TASKS:
            results.append({"checkpoint": stage, "protocol": P2, "task": t, "mean": avg, "sd": 0.1,
                            "per_seed": {str(i): v for i, v in enumerate(per_run)},
                            "source": {"base_model": "M", "stage": stage, "task": t}})
        suites.append({"checkpoint": stage, "protocol": P2, "complete": True, "mean": avg,
                       "sd": 0.1})
        if parent and stage not in rejected:
            changes.append({"checkpoint": stage, "protocol": P2, "parent": parent,
                            "change": avg - avgs[parent]})
    return Manifest({
        "protocols": {P2: {"tasks": {t: {"n_items": 2, "metric": "accuracy"} for t in TASKS},
                           "excluded_tasks": {"GSM8K": "being re-run"}, "temperature": 0.7}},
        "checkpoints": checkpoints, "results": results, "suite_averages": suites,
        "changes": changes,
        "rejected_changes": [{"checkpoint": s, "protocol": P2, "reason": "wrong parent"}
                             for s in rejected],
    })


def _block(*stages, roots=1):
    return [(None, [pt.Row(s, "M", s, change=i >= roots) for i, s in enumerate(stages)])]


def test_a_change_needs_its_parent_printed_above():
    m = _manifest({"base": 20.0, "sft": 25.0, "ot3": 30.0}, {"sft": "base", "ot3": "sft"})
    with pytest.raises(ManifestError, match="not a row printed above"):
        pt.resolve(m, _block("base", "ot3"))
    rows = pt.resolve(m, _block("base", "sft", "ot3"))[0][1]
    assert [r.changes["avg"] for r in rows[1:]] == [5.0, 5.0]


def test_a_rejected_change_is_refused():
    m = _manifest({"base": 20.0, "ot3": 30.0}, {"ot3": "base"}, rejected=("ot3",))
    with pytest.raises(ManifestError, match="rejected: wrong parent"):
        pt.resolve(m, _block("base", "ot3"))


def test_the_printed_change_is_computed_from_unrounded_averages():
    m = _manifest({"base": 20.14, "ot3": 31.16}, {"ot3": "base"})
    row = pt.resolve(m, _block("base", "ot3"))[0][1][1]
    assert row.changes["avg"] == 11.0          # 31.16 - 20.14, although 31.2 - 20.1 = 11.1


def test_an_excluded_task_has_no_column_and_the_caption_says_why():
    m = _manifest({"base": 20.0, "ot3": 30.0}, {"ot3": "base"})
    before, after = pt.resolve(m, _block("base", "ot3"))[0][1]
    ups = [pt.Update("Main", None, "M", before, after)]
    tex = "\n".join(pt._full_lines(m, ups, pt._note(m), "tab:x"))
    assert r"\textbf{Stage} & \textbf{A24} & \textbf{M500} \\" in tex and "GSM8K}" not in tex
    assert "GSM8K is left out while it is re-evaluated" in tex
    assert "M & Before & 20.0$_{\\pm0.10}$ & 20.0$_{\\pm0.10}$ \\\\" in tex
    assert " & After & 30.0$_{\\pm0.10}$ & 30.0$_{\\pm0.10}$ \\\\" in tex


def _meta(stage: str, correct_by_run: list[list[float]]) -> pd.DataFrame:
    return pd.DataFrame([dict(base_model="M", stage=stage, task="MATH500", example_id=f"q{i}",
                              run_idx=r, is_correct=v)
                         for r, row in enumerate(correct_by_run) for i, v in enumerate(row)])


def test_records_that_do_not_reproduce_the_manifest_are_refused():
    m = _manifest({"sft": 50.0}, {}, runs={"sft": [50.0, 50.0, 50.0]})
    meta = _meta("sft", [[1, 0], [1, 0], [1, 1]])        # runs 50, 50, 100
    with pytest.raises(ManifestError, match="per-question records give runs"):
        pt._records_match(m, meta, "sft", "MATH500")
    pt._records_match(m, _meta("sft", [[1, 0], [1, 0], [0, 1]]), "sft", "MATH500")


def test_paired_intervals_are_reproducible_and_bracket_the_estimate():
    rng = np.random.default_rng(0)
    child = pd.Series(rng.random(200) * 100, index=[f"q{i}" for i in range(200)])
    parent = pd.Series(rng.random(200) * 100, index=child.index[::-1])
    a = paired_changes([(child, parent)], n_boot=200)
    assert a == paired_changes([(child, parent)], n_boot=200)
    assert a[0].lo <= a[0].estimate <= a[0].hi
    assert a[0].estimate == pytest.approx((child - parent.reindex(child.index)).mean())


def test_item_scores_need_every_run():
    meta = _meta("sft", [[1, 0], [1, 0], [0, 1]])
    assert item_scores(meta, "M", "sft", "MATH500").tolist() == pytest.approx([200 / 3, 100 / 3])
    with pytest.raises(ValueError, match="lack 3 runs"):
        item_scores(meta.iloc[1:], "M", "sft", "MATH500")


@pytest.mark.parametrize("stage,key", [("base", "base"), ("SFT-Tulu3", "sft"),
                                       ("OT3+SFT", "sft-ot3"), ("AR+DPO", "dpo-ar"),
                                       ("OT3", "ot3")])
def test_macro_keys_name_the_checkpoint(stage, key):
    assert pt.stage_id(stage) == key


def test_every_row_of_the_real_tables_resolves_against_the_committed_manifest():
    m = Manifest.load(Path(__file__).resolve().parents[1] / "results" / "manifest.json")
    for blocks in (pt.endpoint_blocks(), pt.ot3_ablation_blocks(), pt.mv_grid_blocks()):
        for _, rows in pt.resolve(m, blocks):
            for r in rows:
                assert r.row.change == (r.changes is not None)
