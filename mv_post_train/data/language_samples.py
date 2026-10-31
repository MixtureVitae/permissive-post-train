"""Per-question language scores from the lm-eval logs (R8 and R23).

Three zips hold the logs. `language_eval-logs.zip` has 19 checkpoints and
`logs-fineweb.zip` the 7 FineWeb-Edu checkpoints. Each checkpoint was run with
three prompt-example seeds, one folder per seed (`-1765181304`, `-1819856179`
and `-default`). A folder holds one results file per run
(`<hash>_<timestamp>.json`) and one `samples_<task>_<timestamp>.jsonl` per
task of that run, with one line per question.

R23, Harsh Raj's runs of 2026-10-26, holds one folder per checkpoint, task and
seed (`<root>/<checkpoint>/<task>/seed<N>/<checkpoint>/`), each with one
`results_<timestamp>.json` and the samples files of that run. Its seeds are
1234, R8's `default`, and 42 and 2025, which are not R8's other two. Its
folders are read as one folder per checkpoint and seed, `<checkpoint>-seed<N>`.

The run of each task is the one `io_logs.select_results_files` chooses.
Every folder of these zips holds one run per task. `logs-fineweb.zip` holds its log tree twice, under two root folders,
with identical files, so each file is read once. The samples files with the
chosen run's timestamp give one score per question, with the metric the
paper's language tables use. Their mean must reproduce the results file for
every folder and task, or loading stops. MMLU is the harness's overall score,
the mean over all 14,042 questions of the 57 subjects.

The zips are pinned by SHA-256 and kept in the project bucket. The parsed
scores of each zip are cached as a Parquet file next to it, keyed by its hash
and the folders read.
"""
from __future__ import annotations

import json
import re
import tempfile
import zipfile
from pathlib import Path

import pandas as pd

from .. import inputs
from ..inputs import sha256
from .benchmarks import LANGUAGE_SUITE_TASKS, MMLU, SPECS
from .io_logs import LANG_FILE_PATTERN, LANG_TS_REGEX, select_results_files

BUCKET = "gs://mayvic-cc-gating/projects/mv-post-train"
#: The input IDs of the log zips (mv_post_train/inputs.json).
LOG_ZIP_IDS = ("R8", "R8-fineweb", "R23")
#: (file name, SHA-256, bucket URI) of each log zip, as inputs.json pins them.
LOG_ZIPS = tuple((inputs.get(i).file, inputs.get(i).sha256, inputs.get(i).gcs)
                 for i in LOG_ZIP_IDS)
SEEDS = ("default", "1765181304", "1819856179")
#: R23's seeds, the one the paper's tables use first.
R23_SEEDS = ("1234", "42", "2025")
#: R23's seeds by the session that ran them: 1234 in the morning of
#: 2026-10-26, 42 and 2025 in the evening.
R23_SESSIONS = (("1234",), ("42", "2025"))
R23 = LOG_ZIPS[2][0]
#: Part of every cache key, with METRIC. Raise it whenever the way scores or
#: question keys are read changes (the samples-file patterns, `_question`,
#: or the `subject/doc_id` key for MMLU).
CACHE_VERSION = 2

#: The metric each paper language table prints:
#: normalized accuracy for ARC-C, ARC-E, HellaSwag, OBQA and PIQA, accuracy
#: otherwise, as `data/benchmarks.SPECS` records it.
METRIC = {t: SPECS[t].source_metric_label.split(",")[0] for t in LANGUAGE_SUITE_TASKS}

_RESULTS = re.compile(r"/([0-9a-f]{10}_\d{4}-\d{2}-\d{2}T\d{2}-\d{2}-\d{2}\.\d+\.json)$")
#: A file of R23: (checkpoint, task, seed, file name).
_R23_MEMBER = re.compile(r"^[^/]+/([^/]+)/([^/]+)/seed(\d+)/[^/]+/([^/]+)$")
_R23_RESULTS = re.compile(r"^results_(\d{4}-\d{2}-\d{2}T\d{2}-\d{2}-\d{2}\.\d+)\.json$")
_DOC_ID = re.compile(r'^\{"doc_id": (\d+)')


def zip_folders(z: zipfile.ZipFile) -> dict[str, list[str]]:
    """Folder name -> member names of the files directly inside it. A file
    stored twice under different roots is listed once, and only if both
    copies are the same."""
    seen: dict[tuple[str, str], zipfile.ZipInfo] = {}
    for info in z.infolist():
        parts = info.filename.split("/")
        if info.is_dir() or len(parts) < 2:
            continue
        key = (parts[-2], parts[-1])
        first = seen.setdefault(key, info)
        if (first.CRC, first.file_size) != (info.CRC, info.file_size):
            raise ValueError(f"{first.filename} and {info.filename} differ.")
    out: dict[str, list[str]] = {}
    for (folder, _), info in seen.items():
        out.setdefault(folder, []).append(info.filename)
    return out


def r23_folders(z: zipfile.ZipFile) -> dict[str, list[str]]:
    """`<checkpoint>-seed<N>` -> member names of that checkpoint and seed, over
    all its task folders. R23's results files without samples, which sit
    directly in a checkpoint's folder, are left out."""
    out: dict[str, list[str]] = {}
    for name in z.namelist():
        m = _R23_MEMBER.match(name)
        if m:
            out.setdefault(f"{m.group(1)}-seed{m.group(3)}", []).append(name)
    return out


def r23_runs(z: zipfile.ZipFile, members: list[str]) -> dict[str, tuple[str, dict]]:
    """Task -> (run timestamp, results payload) for one R23 folder. Each task
    folder holds one run, so a second results file stops the read."""
    out: dict[str, tuple[str, dict]] = {}
    for name in members:
        _, task, _, file = _R23_MEMBER.match(name).groups()
        m = _R23_RESULTS.match(file)
        if not m or task not in METRIC:
            continue
        if task in out:
            raise ValueError(f"{name}: a second results file for {task}.")
        out[task] = (m.group(1), json.loads(z.read(name))["results"][task])
    return out


def chosen_runs(z: zipfile.ZipFile, members: list[str]) -> dict[str, tuple[str, dict]]:
    """Task -> (run timestamp, results payload) for one folder: the results
    files are written to a temporary folder and passed to
    `select_results_files`, which keeps the most complete and latest run."""
    with tempfile.TemporaryDirectory() as tmp:
        for name in members:
            m = _RESULTS.search(name)
            if m:
                (Path(tmp) / m.group(1)).write_bytes(z.read(name))
        chosen = select_results_files(Path(tmp), pattern=LANG_FILE_PATTERN,
                                      ts_regex=LANG_TS_REGEX)
    out = {}
    for task, (path, payload) in chosen.items():
        if task in METRIC:
            out[task] = (re.search(LANG_TS_REGEX, path.name).group(1), payload)
    return out


def _question(line: str) -> tuple[int, str, dict]:
    """(doc_id, doc_hash, metrics) of one samples line. The metrics are the
    last fields of the line, so only that part is decoded. The full line is
    decoded only when that fails."""
    start = line.rfind('"doc_hash"')
    try:
        tail = json.loads("{" + line[start:])
        return int(_DOC_ID.match(line).group(1)), tail["doc_hash"], tail
    except (ValueError, AttributeError, KeyError):
        rec = json.loads(line)
        return int(rec["doc_id"]), rec["doc_hash"], rec


def folder_items(z: zipfile.ZipFile, folder: str, members: list[str],
                 runs=chosen_runs) -> pd.DataFrame:
    """One row per (task, question) of one folder: `item`, `doc_hash`, `score`
    (0 or 1). Stops if a task's mean differs from its results file."""
    rows = []
    for task, (ts, payload) in sorted(runs(z, members).items()):
        metric = METRIC[task]
        if task == MMLU:
            pat = re.compile(rf"/samples_mmlu_([a-z_]+)_{re.escape(ts)}\.jsonl$")
        else:
            pat = re.compile(rf"/samples_{re.escape(task)}_{re.escape(ts)}\.jsonl$")
        files = [(n, pat.search(n)) for n in members if pat.search(n)]
        if not files:
            raise ValueError(f"{folder}: no samples file for {task} at {ts}.")
        scores = []
        for name, m in sorted(files):
            subject = m.group(1) + "/" if task == MMLU else ""
            with z.open(name) as f:
                for raw in f:
                    doc_id, doc_hash, rec = _question(raw.decode("utf-8"))
                    scores.append(float(rec[metric]))
                    rows.append((folder, task, f"{subject}{doc_id}", doc_hash, scores[-1]))
        want = payload[f"{metric},none"]
        got = sum(scores) / len(scores)
        if abs(got - want) > 1e-9:
            raise ValueError(f"{folder} {task}: the samples give {metric} {got:.6f}, "
                             f"the results file {want:.6f}.")
    out = pd.DataFrame(rows, columns=["folder", "task", "item", "doc_hash", "score"])
    if out.duplicated(["task", "item"]).any():
        raise ValueError(f"{folder}: a question appears twice.")
    return out


def read_items(zip_path: Path, folders: list[str]) -> pd.DataFrame:
    """Per-question scores of the given folders of one zip."""
    r23 = zip_path.name == R23
    with zipfile.ZipFile(zip_path) as z:
        by_folder = r23_folders(z) if r23 else zip_folders(z)
        missing = sorted(set(folders) - set(by_folder))
        if missing:
            raise ValueError(f"{zip_path.name} has no folder {missing}.")
        runs = r23_runs if r23 else chosen_runs
        return pd.concat([folder_items(z, f, by_folder[f], runs) for f in folders],
                         ignore_index=True)


def load(log_dir: Path, folders_by_zip: dict[str, list[str]]) -> pd.DataFrame:
    """Per-question scores for every requested folder. Each zip's scores come
    from its cache when that was built from the same zip and folders. A
    missing zip is downloaded from the bucket and must match its pinned
    SHA-256."""
    frames = []
    for input_id, (name, pinned, uri) in zip(LOG_ZIP_IDS, LOG_ZIPS):
        folders = sorted(folders_by_zip.get(name, []))
        if not folders:
            continue
        cache = log_dir / f"{name}.items.parquet"
        stamp = cache.with_suffix(".key")
        key = json.dumps([CACHE_VERSION, METRIC, pinned, folders], sort_keys=True)
        if cache.exists() and stamp.exists() and stamp.read_text() == key:
            frames.append(pd.read_parquet(cache))
            continue
        path = inputs.path(input_id, log_dir / name)
        got = sha256(path)
        if got != pinned:
            raise ValueError(f"{path} has SHA-256 {got}, not the pinned {pinned}.")
        frame = read_items(path, folders)
        frame.to_parquet(cache, index=False)
        stamp.write_text(key)
        frames.append(frame)
    return pd.concat(frames, ignore_index=True)
