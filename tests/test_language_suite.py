"""Tests for Tables 10 to 14 and 16, the language suite
(mv_post_train/render/language_suite.py and data/language_records.py).

They pin that every row names one run, that a run with other shot counts or
another prompt-example seed is refused, and how a table prints (one decimal,
Avg from the unrounded scores, bold column maxima).
"""
from __future__ import annotations

import json
import zipfile
from pathlib import Path

import pytest

from mv_post_train.data import language_records as lr
from mv_post_train.render import language_suite as ls

ROOT = Path(__file__).resolve().parents[1]
SETTINGS = {t: (10, "acc,none") for t in ls.TASKS}


def test_every_row_names_one_run():
    sources = {}
    for table in ls.TABLES:
        for group in table.groups:
            for row in group.rows:
                if row.source is not None:
                    assert row.source.split(":", 1)[0] in ("R8", "R8-fineweb", "R16", "R17")
                    sources[f"{table.key}.{row.key}"] = row.source
    # the two rows that no run reproduced until 2026-10-27 print the runs that exist for them
    assert sources["small.bag.base"] == "R16:ali-elganzory/Baguettotron"
    assert sources["phase2.mv16k.ot3"] == (
        "R8:open-sci__sft__ot30k_1.7b-MixtureVitae-300BT-v1-decontaminated-16k")


def _results(model: str, tasks: list[str], shots: int = 10, seed: int = 1234,
             pretrained: str | None = None) -> bytes:
    return json.dumps({
        "model_name": model,
        "config": {"model_args": {"pretrained": pretrained or model}, "fewshot_seed": seed},
        "n-shot": {t: shots for t in tasks},
        "results": {t: {"acc,none": 0.5 + i / 100} for i, t in enumerate(tasks)},
    }).encode()


def _zip(path: Path, files: dict[str, bytes]) -> Path:
    with zipfile.ZipFile(path, "w") as z:
        for name, data in files.items():
            z.writestr(name, data)
    return path


def test_a_run_needs_every_task_once_with_its_shots_and_seed(tmp_path):
    tasks = list(ls.TASKS)
    good = _zip(tmp_path / "r16.zip", {
        "raw_json/a_2026-10-10T10-00-00.1.json": _results("org/m", tasks[:5]),
        "raw_json/b_2026-10-15T10-00-00.1.json": _results("org/m", tasks[5:],
                                                          pretrained="/x/patched_models/m"),
        "raw_json/c_2026-10-10T10-00-00.1.json": _results("org/other", tasks)})
    run = lr.load(["R16:org/m"], {"R16": good}, SETTINGS)["R16:org/m"]
    assert run.scores["mmlu"] == pytest.approx(100 * (0.5 + tasks[5:].index("mmlu") / 100))
    assert run.patched and run.dates == ("2026-10-10", "2026-10-15")
    for bad in (_results("org/m", tasks, shots=0), _results("org/m", tasks, seed=42)):
        path = _zip(tmp_path / "bad.zip", {"raw_json/a_2026-10-10T10-00-00.1.json": bad})
        with pytest.raises(ValueError, match="shots|seed"):
            lr.load(["R16:org/m"], {"R16": path}, SETTINGS)
    twice = _zip(tmp_path / "twice.zip", {
        "raw_json/a_2026-10-10T10-00-00.1.json": _results("org/m", tasks),
        "raw_json/b_2026-10-11T10-00-00.1.json": _results("org/m", tasks[:1])})
    with pytest.raises(ValueError, match="2 results files"):
        lr.load(["R16:org/m"], {"R16": twice}, SETTINGS)


def _run(source: str, base: float) -> lr.Run:
    return lr.Run(source, source, {t: base + i / 10 for i, t in enumerate(ls.TASKS)}, (), False, ())


TABLE = ls.Table("t.tex", "Table 0", "t", "tab:t", "Caption.",
                 (ls.chain("MV", "mv", "R16:mv", "R16:sft", "R16:dpo"),))


def test_a_table_prints_one_decimal_avg_from_unrounded_and_no_bold():
    runs = {"R16:mv": _run("R16:mv", 40.04), "R16:sft": _run("R16:sft", 50.06),
            "R16:dpo": _run("R16:dpo", 60.0)}
    # ten cells print 40.0 and one 40.5: their mean is 40.05, the unrounded mean 40.09
    runs["R16:mv"].scores.update({t: 40.04 for t in ls.TASKS[:-1]} | {ls.TASKS[-1]: 40.54})
    tex = ls.table_tex(TABLE, ls.printed_rows(TABLE, runs))
    rows = [line for line in tex.splitlines() if " & " in line and "textbf{Model}" not in line]
    assert rows[0] == "MV & " + "40.0 & " * 10 + "40.5 & 40.1 \\\\"
    assert rows[2].startswith("\\quad $\\to$SFT $\\to$DPO & 60.0 & ")
    assert not any("\\textbf" in r for r in rows)


def test_ranges_keep_each_end_as_printed():
    out = {}
    ls._span(out, "k", ["50", "51.4"])
    assert out == {"k.min": "50", "k.max": "51.4", "k.range": "50 to 51.4"}
    ls._span(out, "g", ["-0.3", "1.6"])
    assert out["g.range"] == "\\ensuremath{-0.3} to 1.6"
    ls._span(out, "one", ["68", "68"])
    assert out["one.range"] == "68"


def test_every_row_reads_its_run_from_the_committed_manifest():
    from mv_post_train.manifest.reader import Manifest
    m = Manifest.load(ROOT / "results" / "manifest.json")
    runs = ls.runs_from_manifest(m)
    assert set(ls.sources()) <= set(runs)
    ls.check_design(m, runs)
