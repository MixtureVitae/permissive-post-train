"""Build the results manifest: one JSON file with every reasoning result the
paper can draw on, each tied to a named checkpoint, its parent, its protocol
and its source records.

    mvpt build-manifest

Inputs (all pinned, see docs/manifest.md):

  Protocol A records   Ali Elganzory's per-question three-seed records behind
                       the Phase 1 tables and the front-loading study
                       (data/workshop/, `data/protocol_a.py`)
  Phase 2 records      marianna13's three-seed Phase 2 records at a fixed
                       dataset revision (data/phase2_3seed/, `manifest/phase2.py`)
  checkpoint snapshot  Hugging Face model records for every checkpoint and the
                       parents they name (data/manifest/hf_checkpoints.json,
                       `manifest/hf_snapshot.py`)

The manifest holds results, suite averages and single-step changes. Every
change is a checkpoint minus the checkpoint it was actually trained from,
under one protocol, over the same tasks on both sides. A change the paper's
tables would make with the wrong parent, a missing task or a mixed protocol is
not computed. It is listed under `rejected_changes` with the reason.

It also holds the results of the older evaluations (Tables 7, 8, 9 and 15)
and of the language suite (Tables 10 to 14 and 16), and which result fills
each row of those tables (`manifest/suites.py`). Suite averages and changes are
computed for the two reasoning protocols only.

Out of scope for now: the Protocol B values that are only copied by hand.
"""
from __future__ import annotations

import hashlib
import json
import math
from collections import defaultdict
from pathlib import Path

import numpy as np
import pandas as pd

from .. import inputs
from ..data import protocol_a as pa
from ..data.checkpoint_guard import checkpoint_id
from . import phase2 as p2
from . import suites
from .lineage import Snapshot, bare, find_parent, to_key

SCHEMA_VERSION = 2
PROTOCOL_A = "protocol_a"
PHASE2 = "phase2_three_seed"

#: Suite order and the canonical task names used in the manifest.
TASKS = ("AIME24", "AIME25", "AMC23", "MATH500", "GSM8K", "HumanEval", "LiveCodeBench",
         "MBPP", "GPQA-Diamond", "JEEBench", "IFEval")
_A_TASK = {"gsm8k": "GSM8K", "GPQADiamond": "GPQA-Diamond"}
_P2_TASK = {"LiveCodeBenchv5": "LiveCodeBench", "GPQADiamond": "GPQA-Diamond"}

#: Tasks left out of a protocol, with the reason. Their results are not in the
#: manifest, and suite averages and changes for that protocol use the others.
#: Phase 2 GSM8K was left out until its re-run of 2026-10-15, because the
#: earlier responses were cut off at the evaluation tool's default of 256 new
#: tokens.
EXCLUDED_TASKS: dict[str, dict[str, str]] = {}

#: What each Protocol A task score is, from the records and their documentation.
PROTOCOL_A_METRICS = {
    "AIME24": "accuracy", "AIME25": "accuracy", "AMC23": "accuracy", "MATH500": "accuracy",
    "GSM8K": "accuracy", "HumanEval": "pass@1, Python problems", "LiveCodeBench": "pass@1",
    "MBPP": "pass@1", "GPQA-Diamond": "accuracy",
    "JEEBench": "score with partial credit (0, 0.25, 0.5, 0.75 or 1 per question)",
    "IFEval": "instruction-level accuracy, strict, over the 500-prompt subset",
}
#: The source name of R21, the strict IFEval verdicts that replace the record
#: files' loose ones (data/protocol_a.with_strict_ifeval).
IFEVAL_STRICT = "protocol_a_ifeval_strict"
#: The source name of R22, the LiveCodeBench v5 verdicts that replace the
#: record files' v2 ones.
LCB_V5 = "protocol_a_lcb_v5"
#: Protocol A tasks whose verdicts come from another input than the record
#: files, by the records' task name.
TASK_INPUT = {"IFEval": IFEVAL_STRICT, "LiveCodeBench": LCB_V5}
#: The released metric name for each Phase 2 task, in plain words.
PHASE2_METRICS = {
    "accuracy_avg": "accuracy, mean over the seeds", "accuracy": "accuracy",
    "python_pass@1": "pass@1, Python problems", "pass@1": "pass@1",
    "prompt-level": "prompt-level accuracy, loose",
    "inst-level,strict": "instruction-level accuracy, strict",
    "exact_match,flexible-extract": "exact match, flexible answer extraction",
}
#: Tasks whose per-question correctness in eval_responses should reproduce the
#: released per-seed scores. The code tasks have no per-question correctness.
REPRODUCIBLE = ("AIME24", "AIME25", "AMC23", "MATH500", "GPQADiamond", "JEEBench", "IFEval")


def _r(x) -> float | None:
    if x is None or (isinstance(x, float) and math.isnan(x)):
        return None
    return round(float(x), 6)


def _sha256(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(2**20), b""):
            h.update(chunk)
    return h.hexdigest()


def _local(input_id: str, dest: Path) -> Path:
    """`dest`, fetched first if it is missing. Its SHA-256 is not checked
    here: the build records that as a check."""
    return dest if dest.exists() else inputs.fetch(inputs.get(input_id), dest)


def _check(checks: list, name: str, ok: bool | None, detail: str) -> None:
    status = "note" if ok is None else ("pass" if ok else "fail")
    checks.append({"name": name, "status": status, "detail": detail})


# ---------------------------------------------------------------------------
# Results from each source


def protocol_a_results(reasoning: pd.DataFrame, source: str = "protocol_a_records",
                       records_name: dict[str, str] | None = None,
                       own_verdicts: bool = False) -> tuple[list[dict], dict]:
    """Results and labels of one record file. `records_name` maps a model_id
    that replaced the records' own name (pa.RECORD_IDENTITY) back to it, and
    that name is kept in the source and the labels. With `own_verdicts` (a file
    of pa.OWN_VERDICTS), the IFEval and LiveCodeBench results come from the file
    itself, not from R21 and R22."""
    records_name = records_name or {}
    per_seed = (reasoning.groupby(["model_id", "benchmark", "seed"])["correct"]
                .agg(["mean", "size"]).reset_index())
    results = []
    for (mid, bench), g in per_seed.groupby(["model_id", "benchmark"]):
        scores = (g.set_index("seed")["mean"] * 100.0).sort_index()
        results.append({
            "checkpoint": bare(mid), "protocol": PROTOCOL_A,
            "task": _A_TASK.get(bench, bench),
            "mean": _r(scores.mean()), "sd": _r(scores.std(ddof=1)),
            "per_seed": {str(s): _r(v) for s, v in scores.items()},
            "n_items": int(g["size"].iloc[0]),
            "source": {"input": source if own_verdicts else TASK_INPUT.get(bench, source),
                       "model_id": mid, "benchmark": bench,
                       **({"records_name": records_name[mid]} if mid in records_name else {})},
        })
    labels = {}
    cols = ["model_group", "front_loading", "context_length", "stage", "in_zoo", "in_factorial"]
    for mid, row in reasoning.drop_duplicates("model_id").set_index("model_id")[cols].iterrows():
        labels[bare(mid)] = {
            "model_id": mid,
            **({"records_name": records_name[mid]} if mid in records_name else {}),
            "model_group": row.model_group,
            "front_loading": bool(row.front_loading), "context_length": int(row.context_length),
            "stage": row.stage,
            "experiments": [e for e, on in (("phase1_zoo", row.in_zoo),
                                            ("front_loading_study", row.in_factorial)) if on],
        }
    return results, labels


def protocol_a_consistency(frames: dict[str, pd.DataFrame]) -> tuple[list[str], list[str]]:
    """(tasks whose question set differs from the first file's, checkpoints that
    appear in more than one file). Checkpoints are compared by bare name, which
    is the manifest's key, so one name under two organizations counts too."""
    sources = list(frames)
    ref = frames[sources[0]].groupby("benchmark")["sample_id"].agg(set)
    differ = []
    for source in sources[1:]:
        got = frames[source].groupby("benchmark")["sample_id"].agg(set)
        differ += [f"{source} {b}" for b in ref.index if got.get(b, set()) != ref[b]]
    seen: dict[str, str] = {}
    overlap = []
    for source in sources:
        for key in {bare(m) for m in frames[source]["model_id"].unique()}:
            if key in seen:
                overlap.append(key)
            seen.setdefault(key, source)
    return differ, sorted(overlap)


def front_loading_design(reasoning: pd.DataFrame, comma: pd.DataFrame) -> dict:
    """Which checkpoint fills each cell of the revision's front-loading study
    (`protocol_a.frontloading`), and why two of its arms come from earlier runs.
    `reasoning` is R13 with its replacements (`protocol_a.with_replacements`)."""
    df = pa.frontloading(reasoning, comma)
    cells = df.drop_duplicates("model_id")
    order = {p: i for i, p in enumerate(pa.FRONTLOADING_POOLS)}
    return {
        "protocol": PROTOCOL_A,
        "substrates": list(pa.FRONTLOADING_POOLS),
        "permissive": list(pa.FRONTLOADING_PERMISSIVE),
        "cells": [{"substrate": r.model_group, "front_loaded": bool(r.front_loading),
                   "context_length": int(r.context_length), "stage": r.stage,
                   "checkpoint": bare(r.model_id)}
                  for r in sorted(cells.itertuples(), key=lambda r: (
                      order[r.model_group], r.front_loading, r.context_length,
                      pa.STAGES.index(r.stage)))],
        "notes": [
            "Comma0.1 has no open-sci-ref v0.02 run without front-loading. Its arm without "
            "front-loading is the Comma0.1 model of Phase 1 (ontocord/1.7b-Comma0.1-"
            "300BT, uploaded 2025-09-17), trained with the same recorded open-sci-ref "
            "configuration. Its data was not affected by the corruption that led to the v0.02 "
            "re-runs. Decided on 2026-10-13.",
            "MixtureVitae's front-loaded arm is the main MixtureVitae model "
            "(1.7b-MixtureVitae-300BT-v1-decontaminated, uploaded 2026-01-22), not the "
            "open-sci-ref v0.02 MixtureVitae run (open-sci-ref-v0.02-1.7b-mixturevitae-300B-4096), "
            "whose weights differ.",
            "The other arms are open-sci-ref v0.02 runs.",
        ],
    }


def phase2_results(scores: pd.DataFrame, meta: pd.DataFrame, checks: list,
                   n_items: dict[str, int] | None = None) -> tuple[list[dict], dict]:
    """(results, labels by checkpoint folder name). `n_items` replaces the
    released item count of a task: eval_scores counts IFEval's 500 prompts,
    and the instruction-level score is over its instructions."""
    n_items = n_items or {}
    meta = meta.assign(checkpoint=meta["source_dir"].map(checkpoint_id))
    cell = meta.groupby(["base_model", "stage", "task"])
    folders = cell["checkpoint"].agg(lambda s: sorted(set(s.dropna())))
    max_tokens = cell["max_tokens"].agg(lambda s: sorted({int(x) for x in s.dropna()}))
    excluded = EXCLUDED_TASKS.get(PHASE2, {})
    results, labels, ambiguous = [], {}, []
    for row in scores.itertuples():
        task = _P2_TASK.get(row.task, row.task)
        if task in excluded:
            continue
        found = folders.get((row.base_model, row.stage, row.task), [])
        if len(found) != 1:
            ambiguous.append(f"{row.base_model} {row.stage} {row.task}: {found or 'no answers'}")
            continue
        name = found[0]
        seeds = [getattr(row, f"score_seed{k}") for k in range(3)]
        results.append({
            "checkpoint": name, "protocol": PHASE2, "task": task,
            "mean": _r(row.score * 100), "sd": _r(row.score_std * 100),
            "per_seed": {str(k): _r(v * 100) for k, v in enumerate(seeds)},
            "n_items": n_items.get(row.task, int(row.num_total)),
            "metric": PHASE2_METRICS.get(row.metric, row.metric),
            "max_new_tokens": max_tokens.get((row.base_model, row.stage, row.task)) or None,
            "source": {"input": "phase2_records", "base_model": row.base_model,
                       "stage": row.stage, "task": row.task},
        })
        labels.setdefault(name, {"base_model": row.base_model, "stage": row.stage})
    _check(checks, "phase2_one_checkpoint_per_cell", not ambiguous,
           "Every included (model, stage, task) cell of the Phase 2 records was evaluated "
           "on exactly one checkpoint." if not ambiguous else
           "Cells left out because their answers name no or several checkpoints: "
           + "; ".join(ambiguous))
    _check(checks, "phase2_three_seeds", all(None not in r["per_seed"].values() for r in results),
           "Every Phase 2 result has three per-seed scores.")
    return results, labels


def phase2_reproduction(scores: pd.DataFrame, meta: pd.DataFrame, checks: list,
                        ifeval: pd.DataFrame) -> None:
    """Released per-seed scores against the per-question correctness. IFEval's
    instruction-level score is the share of instructions followed, from the
    per-prompt counts (`ifeval`), not a mean over prompts.

    A cell can match exactly, match with its seeds in a different order (same
    three values, so the mean and SD are unaffected), or not match.
    """
    graded = meta[meta["task"].isin(REPRODUCIBLE) & (meta["task"] != "IFEval")]
    items = graded.groupby(["base_model", "stage", "task", "run_idx"])["is_correct"].mean()
    counts = ifeval.assign(task="IFEval").groupby(["base_model", "stage", "task", "run_idx"])
    followed = counts["n_strict"].sum() / counts["n_total"].sum()
    items = pd.concat([items, followed]).unstack("run_idx")
    released = (scores[scores["task"].isin(REPRODUCIBLE)]
                .set_index(["base_model", "stage", "task"])
                [["score_seed0", "score_seed1", "score_seed2"]])
    cells = items.index.intersection(released.index)
    a, b = items.loc[cells].to_numpy(), released.loc[cells].to_numpy()
    exact = np.all(np.abs(a - b) < 1e-9, axis=1)
    permuted = ~exact & np.array([np.allclose(np.sort(x), np.sort(y), atol=1e-9)
                                  for x, y in zip(a, b)])
    other = [cells[i] for i in np.flatnonzero(~exact & ~permuted)]
    _check(checks, "phase2_scores_reproduce_from_answers", not other,
           f"In {len(cells) - len(other)} of {len(cells)} cells on {', '.join(REPRODUCIBLE)}, "
           "the three released per-seed scores equal the means of the per-question "
           f"correctness, {int(permuted.sum())} of them in a different seed order "
           "(see phase2_seed_order)." + ("" if not other else
                             " Cells that do not: " + "; ".join(map(str, other[:20]))))
    by_task = pd.Series([c[2] for c in cells[permuted]]).value_counts().sort_index()
    _check(checks, "phase2_seed_order", None if permuted.any() else True,
           "The released per-seed scores and the answer file agree on the seed order."
           if not permuted.any() else
           f"In {int(permuted.sum())} cells the two files hold the same three per-seed values "
           "in a different seed order (" + ", ".join(f"{t} {n}" for t, n in by_task.items())
           + "). Means and SDs are unaffected. The manifest keeps the released order, so a "
           "per-seed suite average pairs the k-th released score of each task.")


# ---------------------------------------------------------------------------
# Checkpoints and parents


def _expected_parents(labels_a: dict, labels_p2: dict) -> dict[tuple[str, str], str]:
    """The parent each table row is paired with, from the records' own labels:
    {(protocol, checkpoint): checkpoint}."""
    out = {}
    by_cell = {(v["model_group"], v["front_loading"], v["context_length"], v["stage"]): k
               for k, v in labels_a.items()}
    for key, v in labels_a.items():
        g, fl, ctx, stage = v["model_group"], v["front_loading"], v["context_length"], v["stage"]
        want = {"sft": (g, fl, ctx, "base"), "dpo": (g, fl, ctx, "sft")}.get(stage)
        if stage == "base" and ctx == 16384:
            want = (g, fl, 4096, "base")
        if want in by_cell:
            out[(PROTOCOL_A, key)] = by_cell[want]
    by_stage = {(v["base_model"], v["stage"]): k for k, v in labels_p2.items()}
    for key, v in labels_p2.items():
        bm, stage = v["base_model"], v["stage"]
        if stage == "base":
            continue
        if stage == "SFT-Tulu3":
            want = "base"
        elif stage == "DPO-Tulu3":
            want = "SFT-Tulu3"
        elif stage.endswith("+SFT"):
            want = "SFT-Tulu3"
        elif stage.endswith("+DPO"):
            want = "DPO-Tulu3"
        else:
            want = "base"          # OT3, AR, NC, SR trained from the base
        if (bm, want) in by_stage:
            out[(PHASE2, key)] = by_stage[(bm, want)]
    return out


def checkpoints(keys: set[str], repos_by_key: dict, record_name: dict, snap: Snapshot,
                labels: dict) -> dict:
    out = {}
    for key in sorted(keys):
        repos = repos_by_key.get(key, [])
        parent = find_parent(record_name.get(key, key), repos, snap, keys)
        recs = [snap.repos[r] for r in repos if r in snap.repos and not snap.repos[r].get("missing")]
        weights = sorted({w for r in recs for w in r.get("weights", [])})
        config = next((r["config"] for r in recs if r.get("config")), None)
        out[key] = {
            "record_name": record_name.get(key, key),
            "hf_repos": repos,
            "repos_differ": len({tuple(r.get("weights", [])) for r in recs}) > 1,
            "created": min((r["created"] for r in recs if r.get("created")), default=None),
            "weights_sha256": weights,
            "context": config,
            "training_data": next((r["run_summary_dataset"] for r in recs
                                   if r.get("run_summary_dataset")), None),
            "parent": to_key(parent.parent, snap, keys),
            "parent_status": parent.status,
            "relation": parent.relation,
            "parent_evidence": parent.evidence,
            "labels": labels.get(key, {}),
        }
    return out


# ---------------------------------------------------------------------------
# Suite averages and single-step changes


def _task_set(protocol: str) -> list[str]:
    return [t for t in TASKS if t not in EXCLUDED_TASKS.get(protocol, {})]


def suite_averages(results: list[dict]) -> tuple[list[dict], dict]:
    by = defaultdict(dict)
    for r in results:
        by[(r["protocol"], r["checkpoint"])][r["task"]] = r
    out, complete = [], {}
    for (protocol, ck), tasks in sorted(by.items()):
        want = _task_set(protocol)
        missing = [t for t in want if t not in tasks]
        if missing:
            out.append({"checkpoint": ck, "protocol": protocol, "complete": False,
                        "missing_tasks": missing})
            continue
        seeds = list(tasks[want[0]]["per_seed"])
        per_seed = {s: float(np.mean([tasks[t]["per_seed"][s] for t in want])) for s in seeds}
        vals = np.array(list(per_seed.values()))
        out.append({"checkpoint": ck, "protocol": protocol, "complete": True, "tasks": want,
                    "mean": _r(vals.mean()), "sd": _r(vals.std(ddof=1)),
                    "per_seed": {s: _r(v) for s, v in per_seed.items()}})
        complete[(protocol, ck)] = {t: tasks[t]["mean"] for t in want}
    return out, complete


def changes(cks: dict, complete: dict, expected: dict, snap: Snapshot
            ) -> tuple[list[dict], list[dict]]:
    """Single-step changes, and the pairings that had to be rejected."""
    accepted, rejected = [], []
    protocols_of = defaultdict(set)
    for protocol, ck in complete:
        protocols_of[ck].add(protocol)
    for ck, info in cks.items():
        for protocol in sorted(protocols_of.get(ck, ())):
            parent = info["parent"]
            table_parent = expected.get((protocol, ck))
            base = {"checkpoint": ck, "protocol": protocol, "parent": parent,
                    "parent_status": info["parent_status"]}
            if info["parent_status"] == "conflict":
                rejected.append({**base, "reason": "The training summary and the model card "
                                 "name different parents."})
                continue
            if table_parent and parent and not snap.same_model(table_parent, parent) \
                    and table_parent != parent:
                if info["parent_status"] == "name only":
                    reason = (f"The records' labels pair this checkpoint with {table_parent}, "
                              f"but its name implies it was trained from {parent}. No training "
                              "summary or model card settles which.")
                else:
                    reason = (f"The records' labels pair this checkpoint with {table_parent}, "
                              f"but it was trained from {parent}, a different model.")
                rejected.append({**base, "table_parent": table_parent, "reason": reason})
                continue
            if table_parent and parent is None:
                rejected.append({**base, "table_parent": table_parent,
                                 "reason": "No evidence of which model this checkpoint was "
                                 "trained from, so the labels' pairing cannot be confirmed."})
                continue
            if parent is None or parent == ck or (protocol, parent) not in complete:
                continue
            child, par = complete[(protocol, ck)], complete[(protocol, parent)]
            accepted.append({
                **base, "relation": info["relation"], "tasks": list(child),
                "checkpoint_mean": _r(np.mean(list(child.values()))),
                "parent_mean": _r(np.mean(list(par.values()))),
                "change": _r(np.mean(list(child.values())) - np.mean(list(par.values()))),
                "per_task": {t: _r(child[t] - par[t]) for t in child},
            })
    return accepted, rejected


# ---------------------------------------------------------------------------


def _input_source(input_id: str, what: str) -> dict:
    inp = inputs.get(input_id)
    src = {"what": what, "input": input_id, "file": inp.file, "sha256": inp.sha256}
    return src | ({"location": inp.gcs} if inp.gcs else {})


def build(workshop_dir: Path, phase2_dir: Path, snapshot_path: Path) -> tuple[dict, bool]:
    checks: list[dict] = []
    frames, file_sha, renamed = {}, {}, {}
    strict_path = _local("R21", workshop_dir / pa.IFEVAL_STRICT_CSV)
    file_sha[IFEVAL_STRICT] = _sha256(strict_path)
    _check(checks, "protocol_a_ifeval_strict_input_sha256",
           file_sha[IFEVAL_STRICT] == inputs.get("R21").sha256,
           f"{pa.IFEVAL_STRICT_CSV} has SHA-256 {file_sha[IFEVAL_STRICT]}.")
    strict = pa.load_ifeval_strict(strict_path)
    lcb_path = _local("R22", workshop_dir / pa.LCB_V5_CSV)
    file_sha[LCB_V5] = _sha256(lcb_path)
    _check(checks, "protocol_a_lcb_v5_input_sha256",
           file_sha[LCB_V5] == inputs.get("R22").sha256,
           f"{pa.LCB_V5_CSV} has SHA-256 {file_sha[LCB_V5]}.")
    lcb_v5 = pa.load_lcb_v5(lcb_path)
    for source, input_id in pa.RECORD_FILES:
        inp = inputs.get(input_id)
        path = _local(input_id, workshop_dir / inp.file)
        file_sha[source] = _sha256(path)
        _check(checks, f"{source.removesuffix('_records')}_input_sha256",
               file_sha[source] == inp.sha256,
               f"{inp.file} has SHA-256 {file_sha[source]}.")
        frame = pa.revised_frame(input_id, path, strict, lcb_v5)
        renamed[source] = {pa.RECORD_IDENTITY[m]: m for m in frame["model_id"].unique()
                           if m in pa.RECORD_IDENTITY}
        frames[source] = frame.assign(model_id=frame["model_id"].replace(pa.RECORD_IDENTITY))
    # R20 replaces R13's August FineWeb-Edu 16K base in Phase 1 and the
    # factorial. The August model keeps its results, but IFEval, and belongs to
    # no experiment. R24's DCLM 16K checkpoints join Phase 1.
    source_of = {i: src for src, i in pa.RECORD_FILES}
    r13 = frames["protocol_a_records"]
    revised = pa.with_additions(
        pa.with_replacements(r13, {i: frames[source_of[i]] for i in pa.REPLACEMENTS}),
        {i: frames[source_of[i]] for i in pa.ADDITIONS})
    frames["protocol_a_records"] = revised.iloc[:len(r13)]
    _check(checks, "protocol_a_replacements", True,
           "In Phase 1 and the front-loading study, " + "; ".join(
               f"{i} replaces {m}" for i, m in pa.REPLACEMENTS.items())
           + ". The replaced checkpoint keeps its results, "
           "apart from IFEval. In Phase 1, " + "; ".join(
               f"{i} adds " + ", ".join(sorted(bare(m) for m in frames[source_of[i]]["model_id"]
                                               .unique()))
               for i in pa.ADDITIONS) + ".")
    own = {source_of[i] for i in pa.OWN_VERDICTS}
    used = {m for s, f in frames.items() if s not in own
            for m in f.loc[f["benchmark"] == "IFEval", "model_id"]}
    used_own = {m for s in own for m in frames[s].loc[frames[s]["benchmark"] == "IFEval",
                                                      "model_id"]}
    _check(checks, "protocol_a_ifeval_strict", True,
           f"Every Protocol A IFEval result is instruction-level accuracy with the strict "
           f"checker, from R21 ({pa.IFEVAL_STRICT_CSV}), for {len(used)} checkpoints. The "
           "record files score IFEval with the loose checker, and their IFEval verdicts are not "
           "used, except in " + ", ".join(pa.OWN_VERDICTS) + f", which holds the verdicts of its "
           f"{len(used_own)} checkpoints, which R21 does not have, and is read as strict. R21 has "
           "no verdicts for " + "; ".join(
               f"{bare(m)} ({why})" for m, why in pa.IFEVAL_STRICT_MISSING.items())
           + ", which has no IFEval result.")
    named = {v: k for k, v in pa.RECORD_IDENTITY.items()}
    used = {named.get(m, m) for s, f in frames.items() if s not in own
            for m in f.loc[f["benchmark"] == "LiveCodeBench", "model_id"]}
    unused = sorted(bare(m) for m in set(lcb_v5["model_id"]) - used)
    _check(checks, "protocol_a_lcb_v5", True,
           f"Every Protocol A LiveCodeBench result is LiveCodeBench v5, on its "
           f"{pa.BENCH_ITEMS['LiveCodeBench']} problems, from R22 ({pa.LCB_V5_CSV}), for "
           f"{len(used)} checkpoints. The record files run LiveCodeBench v2 "
           f"({pa.RECORD_ITEMS['LiveCodeBench']} problems), and their LiveCodeBench verdicts are "
           "not used. " + ", ".join(pa.OWN_VERDICTS) + f" also holds the v5 verdicts of its "
           f"{len(used_own)} checkpoints, which R22 does not have, and they are used. "
           "R22 has no verdicts for " + "; ".join(
               f"{bare(m)} ({why})" for m, why in pa.LCB_V5_MISSING.items())
           + ", which has no LiveCodeBench result. R22's verdicts for " + ", ".join(unused)
           + ", which no record file holds, are not used.")
    _check(checks, "protocol_a_design", True, "Every Protocol A record file passes the design "
           "checks of data/protocol_a.py: three seeds, every question of every task once per "
           "seed, and one checkpoint per cell, except in the older experiments' and the masked "
           "checkpoints' files, whose checkpoints are told apart by name.")
    same, overlap = protocol_a_consistency(frames)
    _check(checks, "protocol_a_same_questions", not same,
           "Every Protocol A record file covers the same questions on every task."
           if not same else "Question sets differ: " + "; ".join(same))
    _check(checks, "protocol_a_one_file_per_checkpoint", not overlap,
           "No checkpoint appears in more than one Protocol A record file."
           if not overlap else "In several files: " + ", ".join(overlap))
    scores_sha = _sha256(_local("R14-scores", phase2_dir / p2.SCORES))
    _check(checks, "phase2_input_sha256", scores_sha == p2.SCORES_SHA256,
           f"{p2.SCORES} has SHA-256 {scores_sha}.")
    meta_sha = _sha256(_local("R14-meta", phase2_dir / p2.RESPONSES_META))
    _check(checks, "phase2_meta_sha256", meta_sha == inputs.get("R14-meta").sha256,
           f"{p2.RESPONSES_META} has SHA-256 {meta_sha}.")
    ifeval_sha = _sha256(_local("R14-ifeval", phase2_dir / p2.IFEVAL))
    _check(checks, "phase2_ifeval_sha256", ifeval_sha == inputs.get("R14-ifeval").sha256,
           f"{p2.IFEVAL} has SHA-256 {ifeval_sha}.")
    snapshot_sha = _sha256(_local("S1", snapshot_path))
    _check(checks, "checkpoint_snapshot_sha256", snapshot_sha == inputs.get("S1").sha256,
           f"{snapshot_path.name} has SHA-256 {snapshot_sha}. After `mvpt snapshot-checkpoints "
           "--update`, upload the new snapshot and pin it as input S1.")
    scores, meta = p2.load(phase2_dir)
    snap_data = json.loads(snapshot_path.read_text())
    snap = Snapshot(snap_data["names"], snap_data["repos"])

    res_a, labels_a = [], {}
    for source, frame in frames.items():
        res, lab = protocol_a_results(frame, source, renamed[source],
                                      dict(pa.RECORD_FILES)[source] in pa.OWN_VERDICTS)
        res_a += res
        labels_a.update(lab)
    counts = p2.ifeval_counts(phase2_dir)
    instructions = counts.groupby(["base_model", "stage", "run_idx"])["n_total"].sum().unique()
    _check(checks, "phase2_ifeval_instructions", len(instructions) == 1,
           f"Every IFEval run counts {', '.join(map(str, instructions))} instructions over its "
           "prompts, the denominator of the instruction-level score.")
    res_p2, labels_p2 = phase2_results(scores, meta, checks, {"IFEval": int(instructions[0])})
    phase2_reproduction(scores, meta, checks, counts)

    # Checkpoint keys: the bare repo name for Protocol A, the folder name for
    # Phase 2, except hash-named folders, which take the name of their repo.
    repos_by_key, record_name = {}, {}
    for key, lab in labels_a.items():
        repos_by_key[key] = [lab["model_id"]]
    rename = {}
    for name in labels_p2:
        repos = [r for r in snap.names.get(name, []) if r in snap.repos]
        key = bare(repos[0]) if repos and len(name) == 40 and name.isalnum() else name
        rename[name] = key
        record_name[key] = name
        repos_by_key[key] = sorted(set(repos_by_key.get(key, [])) | set(repos))
    for r in res_p2:
        r["checkpoint"] = rename[r["checkpoint"]]
    labels_p2 = {rename[k]: v for k, v in labels_p2.items()}
    for key, repos in repos_by_key.items():
        snap.register(key, repos)

    keys = set(repos_by_key)
    labels = {k: {**({PROTOCOL_A: labels_a[k]} if k in labels_a else {}),
                  **({PHASE2: labels_p2[k]} if k in labels_p2 else {})} for k in keys}
    cks = checkpoints(keys, repos_by_key, record_name, snap, labels)
    status = defaultdict(int)
    for c in cks.values():
        status[c["parent_status"]] += 1
    _check(checks, "parents", status.get("conflict", 0) == 0,
           "Parent evidence: " + ", ".join(f"{n} {s}" for s, n in sorted(status.items()))
           + ". A parent is verified when the training summary or the model card names it.")
    missing_repo = sorted(k for k, c in cks.items() if not c["hf_repos"])
    _check(checks, "checkpoints_on_hugging_face", None,
           f"{len(keys) - len(missing_repo)} of {len(keys)} checkpoints were found on Hugging "
           f"Face. Not found: {', '.join(missing_repo) if missing_repo else 'none'}.")

    results = sorted(res_a + res_p2, key=lambda r: (r["protocol"], r["checkpoint"],
                                                     TASKS.index(r["task"])))
    suite_avgs, complete = suite_averages(results)
    # The older experiments' records put several checkpoints in one cell, so
    # their labels pair no rows.
    expected = _expected_parents({k: v[PROTOCOL_A] for k, v in labels.items()
                                  if PROTOCOL_A in v and v[PROTOCOL_A]["experiments"]},
                                 {k: v[PHASE2] for k, v in labels.items() if PHASE2 in v})
    accepted, rejected = changes(cks, complete, expected, snap)

    # The older evaluations and the language suite. Their checkpoints get
    # entries like the others, built after them, so that the reasoning
    # checkpoints' parents do not depend on them.
    res_old = suites.older_results(inputs.path("R6"))
    _check(checks, "older_member_digest", True,
           "The results files read from logs.zip (R6) are the pinned ones (member digest "
           "older_tables).")
    runs = suites.language_runs(suites.language_sources())
    res_lang = suites.language_results(runs)
    res_lang += suites.frontloading_language(res_lang)
    _check(checks, "language_settings", True,
           f"Every one of the {len(runs)} language runs has each task in one results file, with "
           "the shot count and metric of R13-language and prompt-example seed 1234.")
    patched = sorted(r.source for r in runs.values() if r.patched)
    _check(checks, "language_patched_copies", None,
           f"{len(patched)} language runs load the model from a local `patched_models` copy, and "
           "what was changed in those copies is not recorded: " + ", ".join(patched) + ".")
    new_repos: dict[str, set] = defaultdict(set)
    for r in res_old + res_lang:
        new_repos[r["checkpoint"]].add(r["source"].get("weights_from", r["source"]["model_id"]))
    extra = {k: sorted(v) for k, v in new_repos.items() if k not in cks}
    for key, repos in extra.items():
        snap.register(key, repos)
    labels_new = defaultdict(dict)
    for r in res_old:
        labels_new[r["checkpoint"]][suites.OLDER] = {"model_id": r["source"]["model_id"]}
    for r in res_lang:
        labels_new[r["checkpoint"]][suites.LANGUAGE] = {
            "model_id": r["source"]["model_id"],
            **({"run": r["source"]["run"]} if "run" in r["source"] else {})}
    # A reasoning checkpoint keeps its repos and parent. The repo each new result
    # was read from is in its labels and its source.
    for key, lab in labels_new.items():
        if key in cks:
            cks[key]["labels"].update(lab)
    cks.update(checkpoints(set(extra), extra, {}, snap, labels_new) if extra else {})
    cks = dict(sorted(cks.items()))
    results += sorted(res_old, key=lambda r: (r["checkpoint"], TASKS.index(r["task"])))
    results += sorted(res_lang, key=lambda r: (r["checkpoint"], pa.LANGUAGE_TASKS.index(r["task"])))
    settings = pa.language_settings(inputs.path("R13-language"))

    manifest = {
        "schema_version": SCHEMA_VERSION,
        "about": {
            "what": "Every reasoning result the paper can draw on, each tied to a named "
                    "checkpoint, its parent, its protocol and its source records.",
            "built_by": "mvpt build-manifest (mv_post_train/manifest/build.py)",
            "rules": [
                "A result comes from one source record, named in its `source` field.",
                "A change is a checkpoint minus the checkpoint it was trained from, under one "
                "protocol, over the same tasks on both sides.",
                "A pairing with the wrong parent, a missing task or mixed protocols is not "
                "computed. It is listed in `rejected_changes` with the reason.",
            ],
            "out_of_scope": [
                "Protocol B values that exist only as numbers copied by hand "
                "(data/protocol_b.py).",
            ],
        },
        "sources": {
            "protocol_a_records": {
                "repo": pa.SOURCE_REPO, "commit": pa.SOURCE_COMMIT,
                "file": f"artifacts/{pa.REASONING_CSV}",
                "sha256": file_sha["protocol_a_records"],
            },
            "protocol_a_comma_records": {
                "what": "The front-loaded Comma arm (comma-mv_reasoning), in the same format "
                        "and on the same questions as protocol_a_records",
                "location": pa.COMMA_CSV_URI, "received": "2026-10-12",
                "sha256": file_sha["protocol_a_comma_records"],
            },
            "protocol_a_older_records": {
                "what": "The older experiments' checkpoints (100B mixture ablation, "
                        "non-decontaminated MV, 0.4B comparison), in the same format and on "
                        "the same questions as protocol_a_records",
                "location": inputs.get("R18").gcs, "received": "2026-10-18",
                "sha256": file_sha["protocol_a_older_records"],
            },
            "protocol_a_fineweb_april_records": {
                "what": "The April FineWeb-Edu 16K base, in the same format and on the same "
                        "questions as protocol_a_records. It replaces that file's August 16K "
                        "model in Phase 1 and the front-loading study",
                "location": inputs.get("R20").gcs, "received": "2026-10-23",
                "sha256": file_sha["protocol_a_fineweb_april_records"],
            },
            "protocol_a_dclm_16k_records": {
                "what": "The DCLM 16K checkpoints (base, Tulu3 SFT and DPO), in the same format "
                        "and on the same questions as protocol_a_records, with their strict "
                        "IFEval verdicts and their LiveCodeBench v5 verdicts. They join Phase 1",
                "location": inputs.get("R24").gcs, "received": "2026-10-28",
                "sha256": file_sha["protocol_a_dclm_16k_records"],
            },
            IFEVAL_STRICT: {
                "what": "Protocol A IFEval verdicts with the strict checker, in the same format "
                        "and on the same instructions as protocol_a_records. Every Protocol A "
                        "IFEval result is computed from them, in place of the record files' "
                        "loose verdicts",
                "location": inputs.get("R21").gcs, "received": "2026-10-24",
                "sha256": file_sha[IFEVAL_STRICT],
            },
            LCB_V5: {
                "what": "Protocol A LiveCodeBench v5 verdicts, in the same format as "
                        "protocol_a_records, on the 369 problems Protocol B runs. Every Protocol "
                        "A LiveCodeBench result is computed from them, in place of the record "
                        "files' LiveCodeBench v2 verdicts",
                "location": inputs.get("R22").gcs, "received": "2026-10-26",
                "sha256": file_sha[LCB_V5],
            },
            "protocol_a_masked_records": {
                "what": "The four masked MV checkpoints of Table 15, in the same format and on "
                        "the same questions as protocol_a_records. The records name copies of "
                        "the laion masked models with the same weights, and the results are "
                        "keyed by the laion repos, with the records' name in their source",
                "location": inputs.get("R19").gcs, "received": "2026-10-19",
                "sha256": file_sha["protocol_a_masked_records"],
            },
            "phase2_records": {
                "dataset": p2.DATASET, "revision": p2.REVISION,
                "files": {p2.SCORES: scores_sha,
                          "eval_responses.parquet": "columns " + ", ".join(p2.META_COLUMNS),
                          p2.IFEVAL: ifeval_sha},
                "rows": "decoding == sampling (and num_repeat == 3 in eval_scores). IFEval: "
                        f"metric == {p2.IFEVAL_METRIC}",
            },
            "checkpoint_snapshot": {
                "what": "Hugging Face model records: repos, weight hashes, model cards, "
                        "training summaries and context settings",
                "fetched": snap_data.get("fetched"), "organizations": snap_data.get("orgs"),
                "updates": snap_data.get("updates", []),
                "sha256": snapshot_sha,
            },
            "older_evaluations": _input_source(
                "R6", "lm-eval-harness results files of the older evaluations, January to "
                "May 2026, which no table of the paper prints any more"),
            "language_logs": _input_source(
                "R8", "Harsh Raj's language runs of May 2026, three prompt-example seeds, of "
                "which the `default` folder is read"),
            "language_logs_fineweb": _input_source("R8-fineweb", "The same for FineWeb-Edu"),
            "language_results": _input_source(
                "R16", "Harsh Raj's language results files of 2026-10-16, one seed"),
            "language_masked": _input_source(
                "R17", "Language results files of the masked checkpoints, run on 2026-10-12"),
            "language_settings": _input_source(
                "R13-language", "The shot count and metric of each language task"),
        },
        "protocols": {
            PROTOCOL_A: {
                "used_for": "Phase 1, the front-loading study, the older "
                            "experiments and masking",
                "decoding": "greedy", "seeds": list(pa.SEEDS), "answers_per_question_per_seed": 1,
                "settings": "the paper's Protocol A table (appendix, evaluation settings)",
                "tasks": {t: {"n_items": pa.BENCH_ITEMS[b], "metric": PROTOCOL_A_METRICS[t]}
                          for b in pa.BENCHMARKS for t in [_A_TASK.get(b, b)]},
            },
            PHASE2: {
                "used_for": "Phase 2 (OT30K and the other reasoning datasets), and the "
                            "removal of OpenThoughts3 from pretraining",
                "decoding": "sampling", "temperature": 0.7,
                "seeds": "three runs, run_idx 0, 1 and 2", "answers_per_question_per_seed": 1,
                "settings": "marianna13's note of 2026-10-07 and the records' max_tokens column",
                "tasks": {t: {"n_items": int(g["num_total"].iloc[0]),
                              "metric": PHASE2_METRICS.get(g["metric"].iloc[0], g["metric"].iloc[0])}
                          for raw, g in scores.groupby("task")
                          for t in [_P2_TASK.get(raw, raw)]
                          if t not in EXCLUDED_TASKS.get(PHASE2, {})},
                "excluded_tasks": EXCLUDED_TASKS.get(PHASE2, {}),
            },
            suites.OLDER: {
                "used_for": "Tables 7, 8, 9 and 15: the 100B mixture ablation, decontamination, "
                            "the 0.4B comparison and masking",
                "decoding": "the older evaluation setup, one run per evaluation",
                "scoring": "lm-eval-harness accuracy_avg (accuracy for MATH500), GSM8K with "
                           "the flexible-extract parser, HumanEval as the mean of its Python and "
                           "shell scores, IFEval as the mean of its prompt-level and "
                           "instruction-level scores",
                "rule": suites.RULE,
                "tasks": {t: {} for t in TASKS},
            },
            suites.LANGUAGE: {
                "used_for": "Tables 10 to 14 and 16",
                "harness": "lm-eval-harness 0.4.10, no chat template",
                "prompt_example_seed": pa.FEWSHOT_SEED,
                "runs_per_checkpoint": 1,
                "tasks": {t: {"shots": settings[t][0], "metric": settings[t][1].split(",")[0]}
                          for t in pa.LANGUAGE_TASKS},
            },
        },
        "checkpoints": cks,
        "results": results,
        "suite_averages": suite_avgs,
        "changes": sorted(accepted, key=lambda c: (c["protocol"], c["checkpoint"])),
        "rejected_changes": sorted(rejected, key=lambda c: (c["protocol"], c["checkpoint"])),
        "designs": {"front_loading_study": {**front_loading_design(
                        revised, frames["protocol_a_comma_records"]),
                                            "language_cells":
                                                suites.frontloading_language_cells(runs)},
                    "phase1_zoo": suites.zoo_design(revised),
                    "older_tables": suites.older_design(),
                    "language_tables": suites.language_design(runs)},
        "checks": checks,
    }
    hard_fail = any(c["status"] == "fail" and c["name"].endswith(
        ("sha256", "design", "same_questions", "one_file_per_checkpoint")) for c in checks)
    return manifest, not hard_fail


def main(workshop_dir: Path, phase2_dir: Path, snapshot_path: Path, out: Path) -> int:
    try:
        manifest, ok = build(workshop_dir, phase2_dir, snapshot_path)
    except inputs.InputError as exc:
        print(f"error: {exc}")
        return 1
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(manifest, indent=1, sort_keys=False, ensure_ascii=False) + "\n")
    print(f"wrote {out}: {len(manifest['results'])} results, {len(manifest['checkpoints'])} "
          f"checkpoints, {len(manifest['changes'])} changes, "
          f"{len(manifest['rejected_changes'])} rejected")
    for c in manifest["checks"]:
        print(f"  {c['status']:4s}  {c['name']}")
    return 0 if ok else 1
