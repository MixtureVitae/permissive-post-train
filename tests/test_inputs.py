"""Tests for the pinned inputs (mv_post_train/inputs.json and inputs.py).

They pin that every input has a SHA-256 and a place to fetch it from, that
the old module constants read their pins from inputs.json, that no SHA-256 is
written in the code any more, that a copy which is not the pinned version is
refused, that a download falls back to the next source, and that
docs/inputs.md describes every input.
"""
from __future__ import annotations

import hashlib
import json
import re
from pathlib import Path

import pytest

from mv_post_train import inputs

ROOT = Path(__file__).resolve().parents[1]


def test_every_input_is_pinned_and_fetchable():
    data = json.loads(inputs.INPUTS_JSON.read_text())["inputs"]
    assert {"R6", "R8", "R8-fineweb", "R13", "R13-language", "R14-scores", "R14-meta", "R15",
            "R16", "R18", "R19", "R20", "S1"} <= {k for k, v in data.items() if v["paper"]}
    for key, v in data.items():
        assert re.fullmatch(r"[0-9a-f]{64}", v["sha256"]), key
        assert v["bytes"] > 0 and Path(v["local"]).name == v["file"], key
        assert v["sources"] and all(len(s) == 1 and set(s) <= {"url", "gcs", "command", "repo"}
                                    for s in v["sources"]), key
        for s in v["sources"]:
            if "url" in s:
                # a fixed revision or commit, never a branch
                assert "/main/" not in s["url"] and re.search(r"/[0-9a-f]{40}/", s["url"]), key


def test_module_constants_come_from_inputs_json():
    from mv_post_train.data import language_samples as ls
    from mv_post_train.data import protocol_a as pa
    from mv_post_train.manifest import phase2 as p2
    from mv_post_train.render import older_tables as ot
    get = inputs.get
    assert pa.REASONING_CSV_SHA256 == get("R13").sha256
    assert pa.COMMA_CSV_SHA256 == get("R15").sha256 and pa.COMMA_CSV_URI == get("R15").gcs
    assert pa.LANGUAGE_RESULTS_ZIP == get("R16").file
    assert pa.LANGUAGE_LOGS_MEMBERS_SHA256 == get("R8").member_digests["comma_language"]
    assert ls.LOG_ZIPS[0][1] == get("R8").sha256 and ls.LOG_ZIPS[1][1] == get("R8-fineweb").sha256
    assert ls.LOG_ZIPS[2][1] == get("R23").sha256 and ls.R23 == get("R23").file
    assert ot.LOGS_ZIP_SHA256 == get("R6").sha256
    assert ot.LOGS_MEMBERS_SHA256 == get("R6").member_digests["older_tables"]
    assert p2.SCORES_SHA256 == get("R14-scores").sha256
    assert [i for _, i in pa.RECORD_FILES] == ["R13", "R15", "R18", "R19", "R20", "R24"]


def test_no_sha256_is_written_in_the_code():
    hits = [f"{p.relative_to(ROOT)}:{n}" for p in (ROOT / "mv_post_train").rglob("*.py")
            for n, line in enumerate(p.read_text().splitlines(), 1)
            if re.search(r"[\"'][0-9a-f]{64}[\"']", line)]
    assert not hits, "pin these in mv_post_train/inputs.json: " + ", ".join(hits)


def _fake(monkeypatch, tmp_path, content: bytes, sources) -> inputs.Input:
    inp = inputs.Input(id="X", file="x.csv", local=str(tmp_path / "x.csv"),
                       sha256=hashlib.sha256(content).hexdigest(), bytes=len(content),
                       sources=tuple(sources), paper=True, what="test input")
    monkeypatch.setattr(inputs, "registry", lambda: {"X": inp})
    return inp


def test_a_copy_that_is_not_the_pinned_version_is_refused(monkeypatch, tmp_path):
    _fake(monkeypatch, tmp_path, b"records", [{"command": "mvpt make-x"}])
    (tmp_path / "x.csv").write_bytes(b"records")
    assert inputs.path("X") == tmp_path / "x.csv"
    assert inputs.check("X") == "ok"
    (tmp_path / "x.csv").write_bytes(b"recorDs")
    with pytest.raises(inputs.InputError, match="not the pinned version"):
        inputs.path("X")
    assert inputs.check("X") == "differs"


def test_a_download_falls_back_to_the_next_source(monkeypatch, tmp_path):
    good, bad = tmp_path / "good.csv", tmp_path / "bad.csv"
    good.write_bytes(b"records")
    bad.write_bytes(b"other records")
    _fake(monkeypatch, tmp_path / "data", b"records",
          [{"url": bad.as_uri()}, {"url": (tmp_path / "gone.csv").as_uri()},
           {"url": good.as_uri()}])
    assert inputs.path("X").read_bytes() == b"records"
    assert not list((tmp_path / "data").glob("*.part"))


def test_a_missing_input_names_what_to_run(monkeypatch, tmp_path):
    _fake(monkeypatch, tmp_path, b"records", [{"command": "mvpt make-x"}])
    assert inputs.check("X") == "missing"
    with pytest.raises(inputs.InputError, match="run `mvpt make-x`"):
        inputs.path("X")


def test_docs_describe_every_input():
    doc = (ROOT / "docs" / "inputs.md").read_text()
    missing = [k for k in inputs.registry() if f"`{k}`" not in doc]
    assert not missing, f"docs/inputs.md does not describe {missing}"
