"""The results of the older evaluations and of the language suite, and the
table designs, for results/manifest.json.

  older_evalchemy  the older evaluations in logs.zip (input R6) of the
                   checkpoints of Tables 7, 8, 9 and 15, scored by the rules of
                   render/older_tables.py. Each result lists every run, its
                   value and whether the rule uses it. No table prints them
                   since R18 and R19: all four tables print Protocol A.
  language_suite   Tables 10 to 14 and 16: one lm-eval run per checkpoint, from
                   R8, R8-fineweb, R16 and R17 (data/language_records.py), with
                   the shot counts and metrics of R13-language.

Values are stored unrounded, so a table printed from the manifest rounds once,
as the renderers do. The table layouts stay in the renderers; the designs here
record which checkpoint or run fills each row.
"""
from __future__ import annotations

from pathlib import Path

from .. import inputs
from ..data import language_records as lr
from ..data import protocol_a as pa
from ..data.run_rule import broken
from .lineage import bare

OLDER = "older_evalchemy"
LANGUAGE = "language_suite"
#: Task names of the older evaluations -> the manifest's.
OLDER_TASK = {"gsm8k": "GSM8K", "GPQADiamond": "GPQA-Diamond"}
RULE = ("mean of the runs that are not broken: a run is broken if it scores below 1 point "
        "while another run of the same checkpoint and task scores at least 3 "
        "(data/run_rule.py)")


def older_results(logs_zip: Path) -> list[dict]:
    from ..render import older_tables as ot
    digests: list = []
    files = ot.read_folders(logs_zip, ot.checkpoints(), digests)
    if ot.members_digest(digests) != ot.LOGS_MEMBERS_SHA256:
        raise ValueError(f"The results files read from {logs_zip} are not the pinned ones.")
    out = []
    for ck, f in sorted(files.items()):
        values = ot.scores_unbroken(f)
        for task, rs in ot.runs(f).items():
            bad = broken([v for _, v in rs])
            out.append({
                "checkpoint": bare(ck), "protocol": OLDER, "task": OLDER_TASK.get(task, task),
                "mean": values[task], "rule": RULE,
                "runs": [{"file": m.rsplit("/", 1)[1], "value": v, "used": not b}
                         for (m, v), b in zip(rs, bad)],
                "source": {"input": "R6", "model_id": ck},
            })
    return out


#: Language runs whose results files name a repo that has changed its weights
#: since: run -> (checkpoint key, the repo that holds the weights the run used,
#: the reason).
RUN_IDENTITY = {
    "R8-fineweb:open-sci__open-sci-ref-v0.02-1_7b-fineweb-edu-1.4t-300B-4096-longsft_16k": (
        "open-sci-ref-v0.02-1.7b-fineweb-edu-1.4t-300B-4096-4096-longsft_16k",
        "ali-elganzory/open-sci-ref-v0.02-1.7b-fineweb-edu-1.4t-300B-4096-4096-longsft_16k",
        "Run in May 2026 on open-sci/open-sci-ref-v0.02-1.7b-fineweb-edu-1.4t-300B-4096-"
        "longsft_16k, which held the April 16K weights (d5a55bab) until it switched to the "
        "August model on 2026-08-18. Its scores differ from the August model's in 8 of 11 "
        "tasks."),
}


def language_runs(sources: list[str]) -> dict[str, lr.Run]:
    settings = pa.language_settings(inputs.path("R13-language"))
    paths = lr.input_paths({s.split(":", 1)[0] for s in sources})
    return lr.load(sources, paths, settings)


def language_results(runs: dict[str, lr.Run]) -> list[dict]:
    """One result per checkpoint and task. A checkpoint with two runs stops the
    build, since a table must name which one it prints."""
    seen: dict[str, str] = {}
    out = []
    for source, run in sorted(runs.items()):
        key, repo, why = RUN_IDENTITY.get(source, (bare(run.checkpoint), run.checkpoint, None))
        if key in seen:
            raise ValueError(f"language: {key} has two runs, {seen[key]} and {source}")
        seen[key] = source
        input_id, name = source.split(":", 1)
        for task in pa.LANGUAGE_TASKS:
            out.append({
                "checkpoint": key, "protocol": LANGUAGE, "task": task,
                "mean": run.scores[task],
                "source": {"input": input_id, "run": name, "model_id": run.checkpoint,
                           "results_files": list(run.members), "patched_copy": run.patched,
                           "run_dates": list(run.dates),
                           **({"weights_from": repo, "identity": why} if why else {})},
            })
    return out


def frontloading_language(results: list[dict]) -> list[dict]:
    """The front-loading study's language scores that are not yet among
    `results`: the twelve checkpoints of R13-language, one score per task. A
    checkpoint that a run already gives must have the same scores, and that
    result notes R13-language as a second source, unless R13_LANGUAGE_REPLACED
    names it."""
    lang = pa.load_language(inputs.path("R13-language"))
    have = {(r["checkpoint"], r["task"]): r for r in results}
    out, differ = [], set()
    for r in lang.itertuples():
        key = bare(r.model_name)
        if (key, r.task) in have:
            mine = have[(key, r.task)]
            if key in R13_LANGUAGE_REPLACED:
                if abs(mine["mean"] - r.performance) > 1e-9:
                    differ.add(key)
                mine["source"]["replaces"] = {"input": "R13-language", "mean": float(r.performance),
                                              "why": R13_LANGUAGE_REPLACED[key]}
                continue
            if abs(mine["mean"] - r.performance) > 1e-9:
                raise ValueError(f"language: {key} {r.task} is {r.performance} in R13-language "
                                 f"and {mine['mean']} in {mine['source']['input']}")
            mine["source"].setdefault("also_in", []).append("R13-language")
            continue
        out.append({"checkpoint": key, "protocol": LANGUAGE, "task": r.task,
                    "mean": float(r.performance),
                    "source": {"input": "R13-language", "model_id": r.model_name}})
    agree = sorted(set(R13_LANGUAGE_REPLACED) - differ)
    if agree:
        raise ValueError(f"language: R13-language and the run now agree for {agree}. "
                         "Delete them from R13_LANGUAGE_REPLACED.")
    return out


# ---------------------------------------------------------------------------
# Designs: which result fills each row of a table


def zoo_design(reasoning) -> dict:
    df = pa.zoo(reasoning).drop_duplicates("model_id")
    order = {g: i for i, g in enumerate(pa.ZOO_ORDER)}
    return {
        "protocol": "protocol_a", "items": ["Table 1", "Table 6"],
        "cells": [{"model_group": r.model_group, "context_length": int(r.context_length),
                   "stage": r.stage, "checkpoint": bare(r.model_id)}
                  for r in sorted(df.itertuples(), key=lambda r: (
                      order[r.model_group], r.context_length, pa.STAGES.index(r.stage)))],
    }


def older_design() -> dict:
    from ..render import older_tables as ot
    tables = {}
    for t in ot.TABLES:
        tables[t["label"]] = [{"row": f"{t['key']}.{key}", "checkpoint": bare(ck)}
                              for g in t["groups"] for _, key, ck in g.rows()]
    rows = []
    for _, skey, blocks in ot.DECON:
        for _, ctx, *cks in blocks:
            rows += [{"row": f"decon.{skey}.{ctx}.{stage}", "checkpoint": bare(ck)}
                     for (stage, _), ck in zip(ot.STAGE_ROWS, cks)]
    tables["tab:evalchemy_table3"] = rows
    tables["tab:evalchemy_masking"] = [
        {"row": f"mask.{kind}.{ctx}.{stage}", "checkpoint": bare(ck)}
        for kind, cks in (("masked", ot.MASKED), ("unmasked", ot.unmasked()))
        for (ctx, stage), ck in cks.items()]
    # Every table is under Protocol A: R13, R18 for the older experiments, and
    # R19 for Table 15's rows with masking.
    return {"protocols": {label: "protocol_a" for label in tables}, "tables": tables}


def language_design(runs: dict[str, lr.Run]) -> dict:
    from ..render import language_suite as ls
    tables = {}
    for t in ls.TABLES:
        rows = []
        for g in t.groups:
            for r in g.rows:
                if r.source is None:
                    continue
                rows.append({"row": f"{t.key}.{r.key}",
                             "checkpoint": bare(runs[r.source].checkpoint)})
        tables[t.label] = rows
    return {"protocol": LANGUAGE, "tables": tables}


_R23 = "R23:open-sci/open-sci-ref-v0.02-1.7b-"
#: The front-loading study's language cells that no R8 log covers read Harsh
#: Raj's runs of 2026-10-26 (R23), the one input with per-question files for
#: them, so that each printed score and its interval in render-language-
#: intervals come from one run: (substrate, front_loaded, context) -> run.
R23_CELLS = {
    ("MV", False, 4096): _R23 + "mixturevitae-noinstruct-300B-4096",
    ("MV", False, 16384): _R23 + "mixturevitae-noinstruct-300B-4096-longsft_16k",
    ("Comma", True, 4096): _R23 + "comma-mv_reasoning-300B-4096",
    ("Comma", True, 16384): _R23 + "comma-mv_reasoning-300B-4096-longsft_16k",
    ("Nemotron", False, 16384): _R23 + "nemotron-hq-300B-4096-longsft_16k",
    ("Nemotron", True, 4096): _R23 + "nemotron-hq-mv_reasoning-300B-4096",
    ("Nemotron", True, 16384): _R23 + "nemotron-hq-mv_reasoning-300B-4096-longsft_16k",
    ("FineWeb-Edu", True, 4096): _R23 + "fineweb-edu-1.4t-mv_reasoning-300B-4096",
    ("FineWeb-Edu", True, 16384): _R23 + "fineweb-edu-1.4t-mv_reasoning-300B-4096-longsft_16k",
}

#: The runs of the front-loading study's Comma0.1 language cells, as
#: language_records names them: R8's for the arm without front-loading
#: (protocol_a.COMMA_LANGUAGE_CELLS), R23's for the front-loaded arm, whose
#: scores equal R16's.
COMMA_SOURCES = {cell: (R23_CELLS[("Comma",) + cell] if cell[0]
                        else "R8:" + folder.rstrip("/").rsplit("/", 1)[1].removesuffix("-default"))
                 for cell, (_, folder, _, _) in pa.COMMA_LANGUAGE_CELLS.items()}


def interval_sources() -> dict[str, str]:
    """Checkpoint of render/language_intervals.py -> its `default` run, as
    language_records names it."""
    from ..render import language_intervals as li
    return {c.key: c.run for c in li.CHECKPOINTS.values()}


def language_sources() -> list[str]:
    """Every run the language tables print, the front-loading study's Comma0.1
    runs, and the `default` runs of the language changes (Tables 33 and 34)."""
    from ..render import language_suite as ls
    return sorted(set(ls.sources()) | set(COMMA_SOURCES.values())
                  | set(interval_sources().values()) | set(LANGUAGE_REPLACEMENTS.values()))


#: Language cells of the front-loading study that read another run than
#: R13-language's: (substrate, front_loaded, context) -> run. The FineWeb-Edu
#: 16K base without front-loading is the April model (protocol_a.REPLACEMENTS),
#: whose language run is R8's of May 2026 (RUN_IDENTITY). The front-loaded
#: MixtureVitae is the main MixtureVitae model, as in the reasoning cells, so
#: its language cells are R8's runs that Table 11 prints. R13-language scored
#: the open-sci-ref v0.02 MixtureVitae run instead (decided on 2026-10-26). The other cells without R8 logs read R23
#: (R23_CELLS).
LANGUAGE_REPLACEMENTS = {
    ("FineWeb-Edu", False, 16384):
        "R8-fineweb:open-sci__open-sci-ref-v0.02-1_7b-fineweb-edu-1.4t-300B-4096-longsft_16k",
    ("MV", True, 4096): "R8:ontocord__1.7b-MixtureVitae-300BT-v1-decontaminated",
    ("MV", True, 16384): "R8:ontocord__1.7b-MixtureVitae-300BT-v1-decontaminated-16k",
    **{cell: run for cell, run in R23_CELLS.items() if cell[0] != "Comma"},
}

#: R13-language scores that a run replaces although the two differ:
#: checkpoint -> why. The build stops when they come to agree.
R13_LANGUAGE_REPLACED = {
    "open-sci-ref-v0.02-1.7b-fineweb-edu-1.4t-mv_reasoning-300B-4096-longsft_16k": (
        "R23's run answers one OpenBookQA question differently from the run that "
        "R13-language scored (41.8 against 42.0). The tables print R23's, whose per-question "
        "files give the intervals."),
}


def _run_key(runs: dict[str, lr.Run], source: str) -> str:
    return RUN_IDENTITY.get(source, (bare(runs[source].checkpoint),))[0]


def frontloading_language_cells(runs: dict[str, lr.Run]) -> list[dict]:
    """Which checkpoint gives each language cell of the front-loading study
    (Table 31): R13-language's, or the run LANGUAGE_REPLACEMENTS names."""
    lang = pa.load_language(inputs.path("R13-language")).drop_duplicates("model_name")
    cells = []
    for r in lang.itertuples():
        cell = (r.model_group, bool(r.front_loading), int(r.context_length))
        source = LANGUAGE_REPLACEMENTS.get(cell)
        cells.append({"substrate": cell[0], "front_loaded": cell[1], "context_length": cell[2],
                      "checkpoint": _run_key(runs, source) if source else bare(r.model_name)})
    cells += [{"substrate": "Comma", "front_loaded": fl, "context_length": ctx,
               "checkpoint": bare(runs[src].checkpoint)}
              for (fl, ctx), src in COMMA_SOURCES.items()]
    order = {p: i for i, p in enumerate(pa.FRONTLOADING_POOLS)}
    return sorted(cells, key=lambda c: (order[c["substrate"]], c["front_loaded"],
                                        c["context_length"]))
