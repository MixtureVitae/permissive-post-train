"""Tests for the results manifest (mvpt build-manifest).

They pin the rules the manifest exists to enforce: a parent comes from
training evidence, copies are recognized by their weights, a change is only
computed against the checkpoint a run was trained from, over the same tasks,
under one protocol, and excluded tasks stay out. Several Protocol A files
must cover the same questions and must not share a checkpoint.
"""
from __future__ import annotations

import hashlib
import http.server
import threading
from pathlib import Path

import pandas as pd
import pyarrow as pa_
import pyarrow.parquet as pq
import pytest

from mv_post_train.manifest import build as mb
from mv_post_train.manifest import gcs, hf_snapshot
from mv_post_train.manifest.http import open_url
from mv_post_train.manifest.lineage import Snapshot, find_parent, name_rule, resolve_prefix, to_key


# --- name rules and parents -------------------------------------------------

@pytest.mark.parametrize("name,parent", [
    ("sft__ot30k_Qwen3-1.7B-Base-SFT-Tulu3-decontaminated", "Qwen3-1.7B-Base-SFT-Tulu3-decontaminated"),
    ("Qwen3-1.7B-Base-SFT-Tulu3-decontaminated-sr30k", "Qwen3-1.7B-Base-SFT-Tulu3-decontaminated"),
    ("X-16k-DPO-Tulu3-decontaminated", "X-16k-SFT-Tulu3-decontaminated"),
    ("X-16k-SFT-Tulu3-decontaminated", "X-16k"),
    ("X-16k", "X"),
    ("open-sci-ref-v0.02-1.7b-fineweb-edu-1.4t-300B-4096-4096-longsft_16k",
     "open-sci-ref-v0.02-1.7b-fineweb-edu-1.4t-300B-4096"),
    ("open-sci-ref-v0.02-1.7b-nemotron-hq-300B-4096-long_sft_16k",
     "open-sci-ref-v0.02-1.7b-nemotron-hq-300B-4096"),
    ("SmolLM2-1.7B-WithChatTemplate", "SmolLM2-1.7B"),
])
def test_name_rules(name, parent):
    assert name_rule(name)[0] == parent


def test_external_base_has_no_name_parent():
    assert name_rule("Qwen2.5-1.5B") is None


def test_cut_short_names_resolve_to_the_full_name():
    known = {"X-16k-DPO-Tulu3-decontaminated", "Y"}
    assert resolve_prefix("X-16k-DPO-Tulu3-decontaminate", known) == "X-16k-DPO-Tulu3-decontaminated"
    assert resolve_prefix("Z", known) == "Z"
    # A sibling run from the same cut folder name is not the parent's full name.
    known.add("X-16k-DPO-Tulu3-decontaminate-ot30k-rope8-32k")
    assert resolve_prefix("X-16k-DPO-Tulu3-decontaminate", known) == "X-16k-DPO-Tulu3-decontaminated"


def _snap(**repos) -> Snapshot:
    names = {}
    for repo in repos:
        names.setdefault(repo.split("/", 1)[1], []).append(repo)
    return Snapshot(names, repos)


def _rec(weights, card=None, summary=None):
    return {"weights": [weights], "card_base_model": card, "run_summary_base_model": summary}


def test_training_summary_verifies_the_parent():
    snap = _snap(**{"o/child": _rec("c", card="o/parent", summary="o/parent"),
                    "o/parent": _rec("p")})
    p = find_parent("child", ["o/child"], snap, {"child", "parent"})
    assert (p.parent, p.status) == ("o/parent", "verified")


def test_disagreeing_evidence_is_a_conflict():
    snap = _snap(**{"o/child": _rec("c", card="o/a", summary="o/b"),
                    "o/a": _rec("1"), "o/b": _rec("2")})
    assert find_parent("child", ["o/child"], snap, set()).status == "conflict"


def test_copies_with_the_same_weights_are_one_model():
    """A card naming the original release maps to the copy that has results."""
    snap = _snap(**{"H/SmolLM2-1.7B": _rec("w"), "o/SmolLM2-1.7B-WithChatTemplate": _rec("w"),
                    "o/SmolLM2-1.7B-SFT": _rec("s", card="H/SmolLM2-1.7B")})
    keys = {"SmolLM2-1.7B-WithChatTemplate", "SmolLM2-1.7B-SFT"}
    p = find_parent("SmolLM2-1.7B-SFT", ["o/SmolLM2-1.7B-SFT"], snap, keys)
    assert to_key(p.parent, snap, keys) == "SmolLM2-1.7B-WithChatTemplate"


def test_a_card_naming_the_same_weights_is_a_copy_not_a_parent():
    """A `-WithChatTemplate` copy whose card names the repo it copies keeps the
    name rule's parent, not itself."""
    snap = _snap(**{"o/X-16k-WithChatTemplate": _rec("w", card="p/X-long"),
                    "p/X-long": _rec("w")})
    p = find_parent("X-16k-WithChatTemplate", ["o/X-16k-WithChatTemplate"], snap,
                    {"X-16k-WithChatTemplate", "X-16k"})
    assert (p.parent, p.status) == ("X-16k", "name only")


def test_a_stripped_suffix_is_not_matched_to_a_longer_name():
    """`X-Instruct-16k` extends `X-Instruct`, not the one known name that
    starts with `X-Instruct`, which is the checkpoint itself."""
    snap = _snap()
    p = find_parent("X-Instruct-16k", [], snap, {"X-Instruct-16k"})
    assert p.parent == "X-Instruct"


def test_cut_short_parent_inside_a_reasoning_run_name_is_completed():
    snap = _snap()
    known = {"P-SFT-Tulu3-decontaminated", "sft__ot30k_P-SFT-Tulu3-decontaminate"}
    p = find_parent("sft__ot30k_P-SFT-Tulu3-decontaminate", [], snap, known)
    assert p.parent == "P-SFT-Tulu3-decontaminated"


def test_name_only_parent_is_marked():
    snap = _snap()
    p = find_parent("A-ar30k", [], snap, {"A", "A-ar30k"})
    assert (p.parent, p.status) == ("A", "name only")


# --- suite averages and changes ---------------------------------------------

def _results(ck, protocol, scores, seeds=("0", "1", "2")):
    return [{"checkpoint": ck, "protocol": protocol, "task": t, "mean": v,
             "per_seed": {s: v for s in seeds}} for t, v in scores.items()]


def _full(value, protocol=mb.PHASE2):
    return {t: value for t in mb._task_set(protocol)}


def test_suite_average_needs_every_task():
    res = _results("a", mb.PHASE2, _full(10.0)) + _results("b", mb.PHASE2, {"AIME24": 5.0})
    suites, complete = mb.suite_averages(res)
    by = {s["checkpoint"]: s for s in suites}
    assert by["a"]["complete"] and by["a"]["mean"] == 10.0
    assert not by["b"]["complete"] and "AIME25" in by["b"]["missing_tasks"]
    assert (mb.PHASE2, "b") not in complete


def test_an_excluded_task_leaves_the_task_set(monkeypatch):
    assert "GSM8K" in mb._task_set(mb.PHASE2)       # back since the re-run of 2026-10-15
    monkeypatch.setattr(mb, "EXCLUDED_TASKS", {mb.PHASE2: {"GSM8K": "cut off"}})
    assert "GSM8K" not in mb._task_set(mb.PHASE2)
    assert "GSM8K" in mb._task_set(mb.PROTOCOL_A)


def _ck(parent, status="verified"):
    return {"parent": parent, "parent_status": status, "relation": "Tulu3 SFT"}


def test_change_is_against_the_actual_parent():
    snap = _snap()
    res = _results("child", mb.PHASE2, _full(30.0)) + _results("parent", mb.PHASE2, _full(20.0))
    _, complete = mb.suite_averages(res)
    accepted, rejected = mb.changes({"child": _ck("parent"), "parent": _ck(None, "none recorded")},
                                    complete, {(mb.PHASE2, "child"): "parent"}, snap)
    assert rejected == []
    assert accepted[0]["change"] == 10.0 and accepted[0]["parent"] == "parent"


def test_pairing_with_the_wrong_parent_is_rejected():
    """The labels pair `child` with `row_above`, but it was trained from `other`."""
    snap = _snap()
    res = (_results("child", mb.PHASE2, _full(30.0)) + _results("row_above", mb.PHASE2, _full(20.0))
           + _results("other", mb.PHASE2, _full(25.0)))
    _, complete = mb.suite_averages(res)
    cks = {"child": _ck("other"), "row_above": _ck(None, "none recorded"), "other": _ck(None, "none recorded")}
    accepted, rejected = mb.changes(cks, complete, {(mb.PHASE2, "child"): "row_above"}, snap)
    assert accepted == []
    assert rejected[0]["table_parent"] == "row_above" and rejected[0]["parent"] == "other"


def test_no_change_across_protocols():
    snap = _snap()
    res = (_results("child", mb.PHASE2, _full(30.0))
           + _results("parent", mb.PROTOCOL_A, _full(20.0, mb.PROTOCOL_A), seeds=("42", "1234", "2025")))
    _, complete = mb.suite_averages(res)
    accepted, rejected = mb.changes({"child": _ck("parent"), "parent": _ck(None, "none recorded")},
                                    complete, {}, snap)
    assert accepted == [] and rejected == []


# --- several Protocol A files -------------------------------------------------

def _records(model_id, questions=("q1", "q2"), bench="AIME24"):
    return pd.DataFrame([dict(model_id=model_id, benchmark=bench, sample_id=q, seed=s,
                              correct=1.0, model_group="Comma", front_loading=True,
                              context_length=4096, stage="base", in_zoo=False, in_factorial=True)
                         for s in (42, 1234, 2025) for q in questions])


def test_a_second_file_must_cover_the_same_questions():
    differ, overlap = mb.protocol_a_consistency(
        {"main": _records("o/a"), "extra": _records("o/b", questions=("q1", "q3"))})
    assert differ == ["extra AIME24"] and overlap == []


def test_a_checkpoint_in_two_files_is_caught_across_organizations():
    differ, overlap = mb.protocol_a_consistency({"main": _records("o/a"), "extra": _records("p/a")})
    assert differ == [] and overlap == ["a"]


def test_results_name_the_file_they_come_from():
    results, labels = mb.protocol_a_results(_records("o/a"), "protocol_a_comma_records")
    assert {r["source"]["input"] for r in results} == {"protocol_a_comma_records"}
    assert results[0]["mean"] == 100.0 and labels["a"]["experiments"] == ["front_loading_study"]


# --- inputs kept in the bucket ---------------------------------------------------

def _fake_gcloud(monkeypatch, content: bytes):
    calls = []

    def run(cmd, **kwargs):
        calls.append(cmd)
        Path(cmd[-1]).write_bytes(content)

    monkeypatch.setattr(gcs.subprocess, "run", run)
    return calls


def test_a_missing_bucket_input_is_downloaded_once(tmp_path, monkeypatch):
    calls = _fake_gcloud(monkeypatch, b"records")
    pinned = hashlib.sha256(b"records").hexdigest()
    dest = tmp_path / "workshop" / "r.csv"
    assert gcs.local_copy("gs://b/r.csv", dest, pinned).read_bytes() == b"records"
    gcs.local_copy("gs://b/r.csv", dest, pinned)
    assert calls == [["gcloud", "storage", "cp", "gs://b/r.csv", str(dest) + ".part"]]


def test_a_download_with_the_wrong_hash_is_not_kept(tmp_path, monkeypatch):
    _fake_gcloud(monkeypatch, b"changed")
    dest = tmp_path / "r.csv"
    with pytest.raises(gcs.FetchError, match="not the pinned"):
        gcs.local_copy("gs://b/r.csv", dest, hashlib.sha256(b"records").hexdigest())
    assert list(tmp_path.iterdir()) == []


# --- snapshot updates ---------------------------------------------------------

def test_an_update_fetches_only_new_checkpoints(tmp_path, monkeypatch):
    fetched = []

    def record(repo):
        fetched.append(repo)
        return {**_rec("new"), "created": "2026-10-10"}

    def no_listing(org):
        raise AssertionError("no new folder names, so the organizations are not listed")

    monkeypatch.setattr(hf_snapshot, "repo_record", record)
    monkeypatch.setattr(hf_snapshot, "list_org", no_listing)
    previous = {"fetched": "2026-10-01", "names": {"M": ["o/M"]},
                "repos": {"o/M": _rec("old"), "o/a": _rec("old")}}
    data = hf_snapshot.snapshot({"M"}, {"o/a", "o/b"}, tmp_path / "s.json", previous=previous)
    assert fetched == ["o/b"]
    assert data["repos"]["o/a"] == _rec("old") and data["fetched"] == "2026-10-01"
    assert data["updates"][-1]["added"] == ["o/b"]


# --- Phase 2 results ----------------------------------------------------------

def _scores(rows):
    base = dict(score_std=0.0, num_total=30, metric="accuracy_avg", num_repeat=3, decoding="sampling")
    return pd.DataFrame([{**base, **r, "score_seed0": r["score"], "score_seed1": r["score"],
                          "score_seed2": r["score"]} for r in rows])


def _meta(rows):
    return pd.DataFrame([{"max_tokens": 32000.0, "decoding": "sampling", **r} for r in rows])


def test_phase2_results_name_each_cells_checkpoint_and_skip_excluded_tasks(monkeypatch):
    scores = _scores([dict(base_model="M", stage="base", task="AIME24", score=0.1),
                      dict(base_model="M", stage="base", task="GSM8K", score=0.5)])
    meta = _meta([dict(base_model="M", stage="base", task="AIME24",
                       source_dir="__e__scratch__x__checkpoints__M-16k"),
                  dict(base_model="M", stage="base", task="GSM8K",
                       source_dir="__e__scratch__x__checkpoints__other")])
    checks = []
    results, labels = mb.phase2_results(scores, meta, checks)
    # A task run on another checkpoint is that checkpoint's result, not M-16k's.
    assert [(r["checkpoint"], r["task"]) for r in results] == [("M-16k", "AIME24"),
                                                               ("other", "GSM8K")]
    assert results[0]["mean"] == 10.0 and labels["M-16k"]["stage"] == "base"
    assert checks[0]["status"] == "pass"
    monkeypatch.setattr(mb, "EXCLUDED_TASKS", {mb.PHASE2: {"GSM8K": "cut off"}})
    results, _ = mb.phase2_results(scores, meta, [])
    assert [(r["checkpoint"], r["task"]) for r in results] == [("M-16k", "AIME24")]


def test_phase2_cell_with_two_checkpoints_is_left_out():
    scores = _scores([dict(base_model="M", stage="base", task="AIME24", score=0.1)])
    meta = _meta([dict(base_model="M", stage="base", task="AIME24",
                       source_dir="__e__scratch__x__checkpoints__A"),
                  dict(base_model="M", stage="base", task="AIME24",
                       source_dir="__e__scratch__x__checkpoints__B")])
    checks = []
    results, _ = mb.phase2_results(scores, meta, checks)
    assert results == [] and checks[0]["status"] == "fail"


# --- range reads ----------------------------------------------------------------

class _RangeHandler(http.server.BaseHTTPRequestHandler):
    payload = b""
    range_header = "Content-Range"

    def do_GET(self):
        rng = self.headers.get("Range")
        data, status = self.payload, 200
        if rng:
            start, end = (int(x) for x in rng.split("=")[1].split("-"))
            data, status = self.payload[start:end + 1], 206
        self.send_response(status)
        if rng:
            self.send_header(self.range_header, f"bytes {start}-{end}/{len(self.payload)}")
        self.send_header("Content-Length", str(len(data)))
        self.end_headers()
        self.wfile.write(data)

    def log_message(self, *args):
        pass


@pytest.mark.parametrize("range_header", ["Content-Range", "content-range"])
def test_range_reads_let_pyarrow_read_selected_columns(tmp_path, monkeypatch, range_header):
    # Hugging Face's CDN sends header names in lower case since October 2026.
    monkeypatch.setattr(_RangeHandler, "range_header", range_header)
    table = pa_.table({"a": list(range(1000)), "b": ["x" * 50] * 1000})
    path = tmp_path / "t.parquet"
    pq.write_table(table, path, row_group_size=100)
    _RangeHandler.payload = path.read_bytes()
    server = http.server.HTTPServer(("127.0.0.1", 0), _RangeHandler)
    threading.Thread(target=server.serve_forever, daemon=True).start()
    try:
        url = f"http://127.0.0.1:{server.server_port}/t.parquet"
        got = pq.ParquetFile(open_url(url, buffer_size=4096)).read(columns=["a"])
        assert got.column("a").to_pylist() == list(range(1000))
    finally:
        server.shutdown()


# ---------------------------------------------------------------------------
# The older evaluations, the language suite and the designs (manifest/suites.py)

def _committed():
    from mv_post_train.manifest.reader import Manifest
    return Manifest.load(Path(__file__).resolve().parents[1] / "results" / "manifest.json")


def test_every_table_row_has_a_result_for_every_task():
    from mv_post_train.manifest import suites
    m = _committed()
    older = m.design("older_tables")
    assert older["protocols"]["tab:evalchemy_table1"] == "protocol_a"
    assert set(older["protocols"].values()) == {"protocol_a"}
    language = m.design("language_tables")
    assert language["protocol"] == suites.LANGUAGE
    for design, protocol_of, tasks in (
            (older, older["protocols"].get, None),
            (language, lambda _: suites.LANGUAGE, m.tasks(suites.LANGUAGE))):
        for label, rows in design["tables"].items():
            protocol = protocol_of(label)
            for row in rows:
                have = {r["task"] for r in m.results_for(row["checkpoint"], protocol)}
                want = tasks or (m.tasks(protocol) if protocol == "protocol_a" else None)
                assert have and (want is None or set(want) <= have), (label, row)
    for cell in m.design("front_loading_study")["language_cells"]:
        assert len(m.results_for(cell["checkpoint"], suites.LANGUAGE)) == 11


def test_older_results_list_every_run_and_the_rule():
    from mv_post_train.data.run_rule import broken
    from mv_post_train.manifest import suites
    m = _committed()
    older = [r for r in m.data["results"] if r["protocol"] == suites.OLDER]
    assert older and all(r["rule"] == suites.RULE for r in older)
    for r in older:
        values = [run["value"] for run in r["runs"]]
        used = [v for v, bad in zip(values, broken(values)) if not bad]
        assert [run["used"] for run in r["runs"]] == [not b for b in broken(values)]
        assert r["mean"] == pytest.approx(sum(used) / len(used), abs=1e-12)


def test_a_language_run_on_changed_weights_is_keyed_by_the_weights_it_used():
    from mv_post_train.manifest import suites
    m = _committed()
    (source, (key, repo, _)), = suites.RUN_IDENTITY.items()
    r = m.results_for(key, suites.LANGUAGE)[0]
    assert f"{r['source']['input']}:{r['source']['run']}" == source
    assert r["source"]["weights_from"] == repo and repo in m.checkpoints[key]["hf_repos"]


def test_masked_records_are_keyed_by_the_laion_repos_they_copy():
    from mv_post_train.data import protocol_a as pa
    from mv_post_train.manifest.lineage import bare
    m = _committed()
    for records_name, repo in pa.RECORD_IDENTITY.items():
        results = m.results_for(bare(repo), "protocol_a")
        assert len(results) == 11 and all(r["source"]["records_name"] == records_name
                                           for r in results)
        # the language runs of Table 16 name the same checkpoint
        assert len(m.results_for(bare(repo), "language_suite")) == 11


def test_phase2_ifeval_is_instruction_level_strict():
    from mv_post_train.manifest import phase2 as p2
    m = _committed()
    ife = [r for r in m.data["results"] if r["protocol"] == "phase2_three_seed" and r["task"] == "IFEval"]
    assert p2.IFEVAL_METRIC == "inst-level,strict" and len(ife) == 93
    assert all(r["metric"] == "instruction-level accuracy, strict" and r["n_items"] == 769
               for r in ife)
    check = next(c for c in m.data["checks"] if c["name"] == "phase2_scores_reproduce_from_answers")
    assert check["status"] == "pass" and "IFEval" in check["detail"]


def test_protocol_a_ifeval_is_instruction_level_strict():
    from mv_post_train.data import protocol_a as pa
    from mv_post_train.manifest.lineage import bare
    m = _committed()
    ife = [r for r in m.data["results"] if r["protocol"] == "protocol_a" and r["task"] == "IFEval"]
    # R24 holds its DCLM 16K checkpoints' strict verdicts itself (pa.OWN_VERDICTS)
    own = {"protocol_a_dclm_16k_records"}
    assert len(ife) == 106 and all(r["n_items"] == 769 for r in ife)
    assert sum(r["source"]["input"] == mb.IFEVAL_STRICT for r in ife) == 103
    assert all(r["source"]["input"] in own for r in ife if r["source"]["input"] != mb.IFEVAL_STRICT)
    assert m.data["protocols"]["protocol_a"]["tasks"]["IFEval"]["metric"] == mb.PROTOCOL_A_METRICS["IFEval"]
    assert "strict" in mb.PROTOCOL_A_METRICS["IFEval"]
    # the one checkpoint R21 lacks has no IFEval result and no suite average
    (august,) = pa.IFEVAL_STRICT_MISSING
    assert not [r for r in ife if r["checkpoint"] == bare(august)]
    avg = next(a for a in m.data["suite_averages"]
               if a["checkpoint"] == bare(august) and a["protocol"] == "protocol_a")
    assert not avg["complete"] and "IFEval" in avg["missing_tasks"]
    check = next(c for c in m.data["checks"] if c["name"] == "protocol_a_ifeval_strict")
    assert check["status"] == "pass"


def test_protocol_a_livecodebench_is_v5():
    from mv_post_train.data import protocol_a as pa
    from mv_post_train.manifest.lineage import bare
    m = _committed()
    lcb = [r for r in m.data["results"]
           if r["protocol"] == "protocol_a" and r["task"] == "LiveCodeBench"]
    own = {"protocol_a_dclm_16k_records"}
    assert len(lcb) == 106 and all(r["n_items"] == 369 for r in lcb)
    assert sum(r["source"]["input"] == mb.LCB_V5 for r in lcb) == 103
    assert all(r["source"]["input"] in own for r in lcb if r["source"]["input"] != mb.LCB_V5)
    assert m.data["protocols"]["protocol_a"]["tasks"]["LiveCodeBench"]["n_items"] == 369
    # the one checkpoint R22 lacks has no LiveCodeBench result and no suite average
    (august,) = pa.LCB_V5_MISSING
    assert not [r for r in lcb if r["checkpoint"] == bare(august)]
    avg = next(a for a in m.data["suite_averages"]
               if a["checkpoint"] == bare(august) and a["protocol"] == "protocol_a")
    assert avg["missing_tasks"] == ["LiveCodeBench", "IFEval"]
    check = next(c for c in m.data["checks"] if c["name"] == "protocol_a_lcb_v5")
    assert check["status"] == "pass"


def test_the_april_fineweb_base_replaces_the_august_model_in_its_cell():
    from mv_post_train.data import protocol_a as pa
    from mv_post_train.manifest.lineage import bare
    m = _committed()
    (input_id, august), = pa.REPLACEMENTS.items()
    april = "open-sci-ref-v0.02-1.7b-fineweb-edu-1.4t-300B-4096-4096-longsft_16k"
    zoo = {(c["model_group"], c["context_length"], c["stage"]): c["checkpoint"]
           for c in m.design("phase1_zoo")["cells"]}
    assert input_id == "R20" and zoo[("FineWeb-Edu", 16384, "base")] == april
    assert m.data["checkpoints"][bare(august)]["labels"]["protocol_a"]["experiments"] == []
    # the SFT row now has the base it was trained from, and its change is accepted
    sft = bare(august) + "-SFT-Tulu3-decontaminated"
    change = next(c for c in m.data["changes"] if c["checkpoint"] == sft and c["protocol"] == "protocol_a")
    assert change["parent"] == april
    lang = {(c["substrate"], c["front_loaded"], c["context_length"]): c["checkpoint"]
            for c in m.design("front_loading_study")["language_cells"]}
    assert lang[("FineWeb-Edu", False, 16384)] == april
