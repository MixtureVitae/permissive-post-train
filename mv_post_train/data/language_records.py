"""Language-suite scores from lm-eval results files: one score per checkpoint
and task, for the language tables of the revision (Tables 10 to 14 and 16).

Each run is named by the input that holds it and where it is in that zip:
  R8, R8-fineweb  `<folder>`: the `default` folder of the May 2026 rerun, the
                  one run with prompt-example seed 1234 (the other two folders
                  of each checkpoint use other seeds)
  R16             `<checkpoint>`: the results files under `raw_json/` whose
                  model_name is the checkpoint
  R17             `<checkpoint>`: the same under `results/`, for the masked
                  checkpoints
  R23             `<checkpoint>`: the results files of Harsh Raj's runs of
                  2026-10-26 whose model_name is the checkpoint, in the
                  `seed1234` folder of each task
Each task must be in exactly one results file of the run, and every score must
use the shot count and metric of LANGUAGE_CSV's runs (input R13-language,
which Table 25 describes) and the prompt-example seed 1234.

R16 and R17 are small and their SHA-256 is checked on every read. The three
large zips are checked by a digest of the results files read from them
(`member_digests.language_suite` in inputs.json), as `protocol_a` does.
"""
from __future__ import annotations

import json
import re
import zipfile
from dataclasses import dataclass
from pathlib import Path

from .. import inputs
from . import language_samples as ls
from . import protocol_a as pa

TASKS = pa.LANGUAGE_TASKS
#: Where each input keeps its results files, and how a source names a run in it.
FOLDERS = {"R16": "raw_json/", "R17": "masked_models_lang_evals/results/"}
LARGE = ("R8", "R8-fineweb", "R23")
#: A results file of the May rerun: <hash>_<timestamp>.json.
_RESULTS = re.compile(r"/[0-9a-f]{10}_\d{4}-\d{2}-\d{2}T[\d.-]+\.json$")
#: A results file of R23 with the paper's prompt-example seed.
_R23_RESULTS = re.compile(r"^[^/]+/[^/]+/[^/]+/seed1234/[^/]+/results_[^/]+\.json$")


@dataclass(frozen=True)
class Run:
    source: str                  # "<input>:<folder or checkpoint>"
    checkpoint: str              # the model_name of its results files
    scores: dict[str, float]     # task -> percent
    members: tuple[str, ...]     # the results files read, in name order
    patched: bool                # loaded from a local `patched_models` copy
    dates: tuple[str, ...]       # the runs' dates, from the file names


def _pretrained(result: dict) -> str:
    args = result.get("config", {}).get("model_args", "")
    if isinstance(args, dict):
        return str(args.get("pretrained", ""))
    m = re.search(r"pretrained=([^,]+)", str(args))
    return m.group(1) if m else ""


def _run(source: str, files: list[tuple[str, bytes, dict]],
         settings: dict[str, tuple[int, str]]) -> Run:
    if not files:
        raise ValueError(f"language: no results file for {source}")
    names = {r.get("model_name") for _, _, r in files}
    if len(names) != 1:
        raise ValueError(f"language: {source} holds results of {sorted(map(str, names))}")
    scores = {}
    for task in TASKS:
        n_shot, metric = settings[task]
        hits = [r for _, _, r in files if task in r.get("results", {})]
        if len(hits) != 1:
            raise ValueError(f"language: {source} has {len(hits)} results files for {task}")
        result = hits[0]
        if result["config"].get("fewshot_seed") != pa.FEWSHOT_SEED:
            raise ValueError(f"language: {source} {task} was run with prompt-example seed "
                             f"{result['config'].get('fewshot_seed')}, not {pa.FEWSHOT_SEED}")
        if pa._shots(result, task) != {n_shot}:
            raise ValueError(f"language: {source} {task} was run with "
                             f"{sorted(pa._shots(result, task))} shots, not {n_shot}")
        scores[task] = float(result["results"][task][metric]) * 100.0
    used = sorted((m, r) for m, _, r in files if set(r.get("results", {})) & set(TASKS))
    dates = sorted({re.search(r"(\d{4}-\d{2}-\d{2})T", m).group(1) for m, _ in used
                    if re.search(r"(\d{4}-\d{2}-\d{2})T", m)})
    return Run(source, names.pop(), scores, tuple(m for m, _ in used),
               any("patched_models" in _pretrained(r) for _, r in used), tuple(dates))


def _files(z: zipfile.ZipFile, members: list[str]) -> list[tuple[str, bytes, dict]]:
    out = []
    for member in sorted(members):
        if member.endswith(".json"):
            raw = z.read(member)
            result = json.loads(raw)
            if isinstance(result.get("results"), dict):
                out.append((member, raw, result))
    return out


def load(sources: list[str], paths: dict[str, Path],
         settings: dict[str, tuple[int, str]]) -> dict[str, Run]:
    """source -> Run for every source. `paths` gives each input's local copy,
    already checked by `inputs.path`."""
    by_input: dict[str, list[str]] = {}
    for s in sources:
        by_input.setdefault(s.split(":", 1)[0], []).append(s.split(":", 1)[1])
    out = {}
    for input_id, names in sorted(by_input.items()):
        with zipfile.ZipFile(paths[input_id]) as z:
            if input_id in LARGE:
                read = []
                folders = {} if input_id == "R23" else ls.zip_folders(z)
                every = (_files(z, [m for m in z.namelist() if _R23_RESULTS.match(m)])
                         if input_id == "R23" else [])
                for name in sorted(set(names)):
                    if input_id == "R23":
                        files = [f for f in every if f[2].get("model_name") == name]
                    else:
                        folder = f"{name}-default"
                        if folder not in folders:
                            raise ValueError(f"language: {input_id} has no folder {folder}")
                        files = _files(z, [m for m in folders[folder] if _RESULTS.search(m)])
                    read += files
                    out[f"{input_id}:{name}"] = _run(f"{input_id}:{name}", files, settings)
                want = inputs.member_digest(input_id, "language_suite")
                got = pa.logs_members_digest(read)
                if got != want:
                    raise ValueError(f"language: the results files read from {input_id} have "
                                     f"digest {got}, not the pinned {want}")
            else:
                prefix = FOLDERS[input_id]
                every = _files(z, [m for m in z.namelist() if m.startswith(prefix)])
                for name in sorted(set(names)):
                    files = [f for f in every if f[2].get("model_name") == name]
                    out[f"{input_id}:{name}"] = _run(f"{input_id}:{name}", files, settings)
    return out


def input_paths(ids) -> dict[str, Path]:
    """The checked local copy of each input in `ids`, downloaded if missing."""
    return {i: inputs.path(i) for i in sorted(set(ids))}
