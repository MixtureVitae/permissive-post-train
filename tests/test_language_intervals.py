"""Tests for the language-suite changes with question-level intervals
(mvpt render-language-intervals).

They pin the rules: the per-question scores must reproduce the results files,
the run chosen is the one `LanguageParser` uses, the bootstrap is
reproducible, a multi-step change is labelled as one, and a change from a
checkpoint that is not on its child's line in the results manifest is
refused. R23's folders are read per checkpoint and seed, and the front-loading
study's language contrasts are those of Table 27. The last tests read the
committed outputs.
"""
from __future__ import annotations

import json
import re
import zipfile
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from mv_post_train.data import language_samples as ls
from mv_post_train.data.benchmarks import LANGUAGE_SUITE_TASKS
from mv_post_train.render import language_intervals as li
from mv_post_train.stats import language_changes as lc

ROOT = "logs"
TS1, TS2 = "2026-05-06T12-00-00.000001", "2026-05-07T12-00-00.000001"


def _line(doc_id: int, acc: float, acc_norm: float | None = None) -> str:
    rec = {"doc_id": doc_id, "doc": {"q": "x"}, "target": "1", "doc_hash": f"h{doc_id}",
           "prompt_hash": "p", "target_hash": "t", "acc": acc}
    if acc_norm is not None:
        rec["acc_norm"] = acc_norm
    return json.dumps(rec)


def _zip(path: Path, runs: dict[str, dict]) -> Path:
    """`runs`: timestamp -> {task: (results dict, samples lines)}; MMLU's
    samples are {subject: lines}."""
    with zipfile.ZipFile(path, "w") as z:
        for k, (ts, tasks) in enumerate(runs.items()):
            results = {task: res for task, (res, _) in tasks.items()}
            z.writestr(f"{ROOT}/M-default/{k:010x}_{ts}.json", json.dumps({"results": results}))
            for task, (_, lines) in tasks.items():
                if isinstance(lines, dict):
                    for subject, sub in lines.items():
                        z.writestr(f"{ROOT}/M-default/samples_mmlu_{subject}_{ts}.jsonl",
                                   "\n".join(sub) + "\n")
                else:
                    z.writestr(f"{ROOT}/M-default/samples_{task}_{ts}.jsonl",
                               "\n".join(lines) + "\n")
    return path


def test_samples_reproduce_the_results_file_with_the_papers_metric(tmp_path):
    arc = [_line(0, 1.0, 0.0), _line(1, 0.0, 1.0), _line(2, 0.0, 1.0)]
    mmlu = {"anatomy": [_line(0, 1.0), _line(1, 0.0)], "virology": [_line(0, 1.0)]}
    path = _zip(tmp_path / "z.zip", {TS1: {
        "arc_challenge": ({"acc,none": 1 / 3, "acc_norm,none": 2 / 3}, arc),
        "mmlu": ({"acc,none": 2 / 3}, mmlu),
    }})
    items = ls.read_items(path, ["M-default"])
    arc_scores = items[items["task"] == "arc_challenge"]["score"]
    assert arc_scores.tolist() == [0.0, 1.0, 1.0]                 # acc_norm, not acc
    assert sorted(items[items["task"] == "mmlu"]["item"]) == ["anatomy/0", "anatomy/1",
                                                              "virology/0"]


def test_samples_that_do_not_reproduce_the_results_file_stop_the_load(tmp_path):
    path = _zip(tmp_path / "z.zip", {TS1: {
        "copa": ({"acc,none": 0.9}, [_line(0, 1.0), _line(1, 0.0)])}})
    with pytest.raises(ValueError, match="results file 0.900000"):
        ls.read_items(path, ["M-default"])


def test_the_later_of_two_runs_is_used_as_the_language_parser_does(tmp_path):
    path = _zip(tmp_path / "z.zip", {
        TS1: {"copa": ({"acc,none": 0.0}, [_line(0, 0.0), _line(1, 0.0)])},
        TS2: {"copa": ({"acc,none": 1.0}, [_line(0, 1.0), _line(1, 1.0)])},
    })
    assert ls.read_items(path, ["M-default"])["score"].tolist() == [1.0, 1.0]


def test_a_log_tree_stored_twice_is_read_once_and_must_match(tmp_path):
    path = tmp_path / "z.zip"
    results = json.dumps({"results": {"copa": {"acc,none": 0.5}}})
    lines = _line(0, 1.0) + "\n" + _line(1, 0.0) + "\n"
    with zipfile.ZipFile(path, "w") as z:
        for root in ("a/logs", "logs"):
            z.writestr(f"{root}/M-default/0000000001_{TS1}.json", results)
            z.writestr(f"{root}/M-default/samples_copa_{TS1}.jsonl", lines)
    assert ls.read_items(path, ["M-default"])["score"].tolist() == [1.0, 0.0]
    with zipfile.ZipFile(path, "a") as z:
        z.writestr(f"b/M-default/samples_copa_{TS1}.jsonl", lines + _line(2, 1.0) + "\n")
    with pytest.raises(ValueError, match="differ"):
        ls.read_items(path, ["M-default"])


def _matrix(rng: np.random.Generator, shift: float) -> lc.Matrix:
    scores = {}
    for name, extra in (("parent", 0.0), ("child", shift)):
        idx = pd.MultiIndex.from_tuples([(t, str(i)) for t in LANGUAGE_SUITE_TASKS
                                         for i in range(40)], names=["task", "item"])
        p = np.clip(0.5 + extra, 0, 1)
        scores[name] = pd.Series((rng.random(len(idx)) < p) * 100.0, index=idx)
    return lc.Matrix.build(scores)


def test_bootstrap_is_reproducible_and_the_average_is_the_mean_of_the_tasks():
    m = _matrix(np.random.default_rng(0), 0.2)
    a = lc.changes(m, [("child", "parent")], n_boot=200)
    assert a.equals(lc.changes(m, [("child", "parent")], n_boot=200))
    avg = a[a["task"] == lc.AVG].iloc[0]
    assert avg["estimate"] == pytest.approx(a[a["task"] != lc.AVG]["estimate"].mean())
    assert (a["lo"] <= a["estimate"]).all() and (a["estimate"] <= a["hi"]).all()
    assert avg["lo"] > 0                                          # a 20-point shift is found


def test_checkpoints_scored_on_different_questions_are_refused():
    idx = pd.MultiIndex.from_tuples([("copa", "0")], names=["task", "item"])
    with pytest.raises(ValueError, match="same questions"):
        lc.Matrix.build({"a": pd.Series([1.0], index=idx),
                         "b": pd.Series([1.0], index=idx.set_levels(["boolq"], level=0))})


def _items(rows: list[tuple[str, str, str, str, float]]) -> pd.DataFrame:
    return pd.DataFrame(rows, columns=["folder", "task", "item", "doc_hash", "score"])


def test_a_question_key_that_names_two_questions_stops_the_render():
    ok = _items([("A", "copa", "0", "h0", 1.0), ("B", "copa", "0", "h0", 0.0)])
    lc.check_same_questions(ok)
    bad = _items([("A", "copa", "0", "h0", 1.0), ("B", "copa", "0", "h1", 0.0)])
    with pytest.raises(ValueError, match="different questions"):
        lc.check_same_questions(bad)


def test_seed_folders_must_agree_on_the_tasks_no_seed_can_change():
    folders = ["M-default", "M-1", "M-2"]
    few_shot = _items([(f, "arc_challenge", "0", "h", float(k == 0))
                       for k, f in enumerate(folders)])
    lc.check_seed_invariant_tasks(few_shot, folders, "M")        # 10 shots: may differ
    zero_shot = _items([(f, "copa", "0", "h", float(k == 0)) for k, f in enumerate(folders)])
    with pytest.raises(ValueError, match="may not hold the same model"):
        lc.check_seed_invariant_tasks(zero_shot, folders, "M")


def test_runs_of_two_sessions_may_differ_on_a_few_zero_shot_questions():
    folders = ["M-seed1234", "M-seed42", "M-seed2025"]
    sessions = [["M-seed1234"], ["M-seed42", "M-seed2025"]]

    def items(n_differ: int) -> pd.DataFrame:
        return _items([(f, "copa", str(i), f"h{i}", float(f == "M-seed1234" and i < n_differ))
                       for f in folders for i in range(100)])
    lc.check_seed_invariant_tasks(items(5), folders, "M", sessions)      # 5% of the questions
    with pytest.raises(ValueError, match="6.0% of the questions"):
        lc.check_seed_invariant_tasks(items(6), folders, "M", sessions)
    with pytest.raises(ValueError, match="disagree on copa"):           # within one session
        lc.check_seed_invariant_tasks(items(1), folders, "M", [folders])


def test_multi_step_changes_are_labelled_and_keyed_as_such():
    multi = [c for cs in li.CHANGES.values() for c in cs if c.multi_step]
    assert {(c.child, c.parent) for c in multi} >= {("mv.dpo-ot3", "mv.base"),
                                                    ("mv.sft-ot3", "mv.base"),
                                                    ("mv.dpo", "mv.base"),
                                                    ("comma.dpo", "comma.base"),
                                                    ("fineweb.dpo", "fineweb.base")}
    row = pd.Series({"child": "mv.dpo-ot3", "parent": "mv.base", "multi_step": True})
    assert r"\Rightarrow" in li._change_label(row)


def test_changes_follow_the_manifests_lineage():
    from mv_post_train.manifest.reader import Manifest
    m = Manifest.load(Path(__file__).resolve().parents[1] / "results" / "manifest.json")
    keys = li.manifest_keys(m)
    # R8's 16K Nemotron-CC-HQ base is not on the 16K SFT model's line.
    with pytest.raises(ValueError, match="refused"):
        li.check(li.Change("nemotron.sft", "nemotron.base", li.SFT), m, keys)
    # A multi-step change must not be a single step, and the other way round.
    with pytest.raises(ValueError, match="step 1"):
        li.check(li.Change("mv.dpo", "mv.sft", li.DPO, True), m, keys)
    with pytest.raises(ValueError, match="step 2"):
        li.check(li.Change("mv.dpo", "mv.base", li.DPO), m, keys)
    # R8's FineWeb-Edu 16K run holds the April weights, the SFT model's parent.
    assert keys["fineweb.base"].endswith("-4096-4096-longsft_16k")
    for changes in li.CHANGES.values():
        for c in changes:
            li.check(c, m, keys)                                  # none of the listed ones
            assert c.child in li.CHECKPOINTS and c.parent in li.CHECKPOINTS
            assert c.child.split(".")[0] == c.parent.split(".")[0]


def test_every_checkpoint_has_one_folder_per_seed():
    for c in li.CHECKPOINTS.values():
        assert len(c.folders) == len(ls.SEEDS) == len(set(c.folders))
        assert c.default.endswith("-seed1234" if c.zip == li.R23 else "-default")


def _r23_zip(path: Path, seeds: dict[str, dict[str, tuple[dict, list[str]]]]) -> Path:
    """`seeds`: seed -> {task: (results dict, samples lines)}, for checkpoint M."""
    with zipfile.ZipFile(path, "w") as z:
        for seed, tasks in seeds.items():
            for task, (res, lines) in tasks.items():
                folder = f"root/M/{task}/seed{seed}/M"
                z.writestr(f"{folder}/results_{TS1}.json", json.dumps({"results": {task: res}}))
                z.writestr(f"{folder}/samples_{task}_{TS1}.jsonl", "\n".join(lines) + "\n")
        z.writestr(f"root/M/seed1234/copa_results_{TS1}.json", "{}")   # a results file only
    return path


def test_r23_is_read_as_one_folder_per_checkpoint_and_seed(tmp_path):
    path = tmp_path / ls.R23
    _r23_zip(path, {"1234": {"copa": ({"acc,none": 0.5}, [_line(0, 1.0), _line(1, 0.0)]),
                             "boolq": ({"acc,none": 1.0}, [_line(0, 1.0)])},
                    "42": {"copa": ({"acc,none": 0.5}, [_line(0, 1.0), _line(1, 0.0)])}})
    with zipfile.ZipFile(path) as z:
        assert sorted(ls.r23_folders(z)) == ["M-seed1234", "M-seed42"]
    items = ls.read_items(path, ["M-seed1234", "M-seed42"])
    assert items.groupby("folder")["task"].nunique().to_dict() == {"M-seed1234": 2, "M-seed42": 1}
    with zipfile.ZipFile(path, "a") as z:
        z.writestr(f"root/M/copa/seed42/M/results_{TS2}.json",
                   json.dumps({"results": {"copa": {"acc,none": 0.5}}}))
    with pytest.raises(ValueError, match="second results file"):
        ls.read_items(path, ["M-seed42"])


def test_a_weighted_contrast_of_two_checkpoints_is_their_change():
    m = _matrix(np.random.default_rng(1), 0.1)
    a = lc.changes(m, [("child", "parent")], n_boot=200)
    b = lc.contrasts(m, [("f", "x", {"child": 1.0, "parent": -1.0})], n_boot=200)
    assert a[["task", "estimate", "lo", "hi", "p", "q"]].equals(
        b[["task", "estimate", "lo", "hi", "p", "q"]])


def test_the_front_loading_contrasts_are_those_of_table_27():
    specs = li.frontloading_specs()
    assert len(li.FL_CELLS) == 16 and len({(p, c) for p, _, c in li.FL_CELLS}) == 8
    assert {k for k in li.FL_CELLS.values()} <= set(li.CHECKPOINTS)
    by_family = {}
    for family, label, weights in specs:
        by_family.setdefault(family, []).append(label)
        assert sum(weights.values()) == pytest.approx(0.0)       # a contrast of the arms
    assert {f: len(v) for f, v in by_family.items()} == {
        "front-loading": 8, "permissiveness": 8, "interaction": 4}
    (_, _, inter), = [s for s in specs if s[1] == "16384|Comma - non-permissive"]
    assert inter == {"Comma|FL|16384": 1.0, "Comma|noFL|16384": -1.0,
                     "Nemotron|FL|16384": -0.5, "Nemotron|noFL|16384": 0.5,
                     "FineWeb-Edu|FL|16384": -0.5, "FineWeb-Edu|noFL|16384": 0.5}


def test_the_committed_outputs_are_consistent():
    root = Path(__file__).resolve().parents[1] / "results" / "language"
    table = pd.read_csv(root / "language_changes.csv")
    assert set(table["mode"]) == {"default", "three-seed mean"}
    assert ((table["lo"] <= table["estimate"] + 1e-9) & (table["estimate"] <= table["hi"] + 1e-9)).all()
    for (_, sel) in table.groupby(["mode", "child", "parent"]):
        avg = sel[sel["task"] == lc.AVG]["estimate"].iloc[0]
        assert avg == pytest.approx(sel[sel["task"] != lc.AVG]["estimate"].mean(), abs=1e-5)
    assert not ((table["parent"] == "nemotron.base")).any()
    numbers = dict(re.findall(r"csname langci@([^\\]+)\\endcsname\{(.*)\}$",
                              (root / "tables" / "language_numbers.tex").read_text(), re.M))
    row = table[(table["mode"] == "default") & (table["child"] == "mv.dpo-ot3")
                & (table["parent"] == "mv.base")].set_index("task")
    # One task: the rounded estimate. The average: the difference of the printed averages.
    assert numbers["mv.dpo-ot3.from-base.arc-c"] == f"\\ensuremath{{{row.loc['arc_challenge', 'estimate']:+.1f}}}"
    avg = round(row.loc["avg", "child_value"], 2) - round(row.loc["avg", "parent_value"], 2)
    assert numbers["mv.dpo-ot3.from-base.avg"] == f"\\ensuremath{{{avg:+.2f}}}"
    assert "mv.dpo.from-base.avg" in numbers


def test_the_committed_front_loading_contrasts_are_consistent():
    root = Path(__file__).resolve().parents[1] / "results" / "language"
    fl = pd.read_csv(root / "language_frontloading.csv")
    assert set(fl["mode"]) == {"default", "three-seed mean"}
    avg = fl[fl["task"] == lc.AVG]
    assert avg.groupby(["mode", "family"]).size().to_dict() == {
        (m, f): n for m in ("default", "three-seed mean")
        for f, n in (("front-loading", 8), ("interaction", 4), ("permissiveness", 8))}
    assert ((fl["lo"] <= fl["estimate"] + 1e-9) & (fl["estimate"] <= fl["hi"] + 1e-9)).all()
    for _, sel in fl.groupby(["mode", "family", "context_length", "label"]):
        assert sel[sel["task"] == lc.AVG]["estimate"].iloc[0] == pytest.approx(
            sel[sel["task"] != lc.AVG]["estimate"].mean(), abs=1e-5)
    numbers = dict(re.findall(r"csname langci@([^\\]+)\\endcsname\{(.*)\}$",
                              (root / "tables" / "language_numbers.tex").read_text(), re.M))
    row = avg[(avg["mode"] == "default") & (avg["label"] == "MV")
              & (avg["context_length"] == 16384)].iloc[0]
    assert numbers["fl.mv.16k.avg"] == f"\\ensuremath{{{row['estimate']:+.2f}}}"
