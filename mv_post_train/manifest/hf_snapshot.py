"""Snapshot the Hugging Face records that identify each checkpoint and its parent.

For every checkpoint that has results, and every parent those records name,
this saves: which repos carry the name, upload and modification dates, the
weight files' SHA-256, the parent the model card declares (`base_model`), the
parent the training summary records (`run_summary.json`), and the context
settings in `config.json`. The manifest build reads only this snapshot, so the
build is reproducible offline and does not drift when a repo is overwritten.

Requests are made one at a time, with retries and back-off on rate limits.

With `update`, the records already saved are kept as they are and only
checkpoints that are new to the sources are fetched. A new source then adds
records without changing the ones the manifest already reads. Each update is
logged under `updates` with its date and the repos it added.
"""
from __future__ import annotations

import datetime as dt
import json
import re
import time
from pathlib import Path

from .http import NotFound, get
from .lineage import bare

API = "https://huggingface.co/api/models"
#: Organizations whose repos the records' checkpoint folders are looked up in.
ORGS = ("ali-elganzory", "open-sci", "laion", "ontocord", "marianna13")
#: External releases that hash-named folders (Hugging Face snapshot ids) may belong to.
EXTERNAL = ("Qwen/Qwen2.5-1.5B-Instruct", "Qwen/Qwen3-1.7B", "Qwen/Qwen2.5-1.5B",
            "Qwen/Qwen3-1.7B-Base", "HuggingFaceTB/SmolLM2-1.7B",
            "HuggingFaceTB/SmolLM2-1.7B-Instruct")
PAUSE = 1.0
_HASH = re.compile(r"^[0-9a-f]{40}$")
_NEXT = re.compile(r'<([^>]+)>;\s*rel="next"')


def _json(url: str):
    body, headers = get(url)
    time.sleep(PAUSE)
    return json.loads(body), headers


def list_org(org: str) -> list[str]:
    repos, url = [], f"{API}?author={org}&limit=1000"
    while url:
        page, headers = _json(url)
        repos += [m["id"] for m in page]
        m = _NEXT.search(headers.get("Link", ""))
        url = m.group(1) if m else None
    return repos


def _parent_from_run_summary(value: str | None) -> str | None:
    """A hub cache path (`.../models--ORG--NAME/snapshots/<sha>`) or a repo id
    becomes `ORG/NAME`. Any other local path becomes its last folder name."""
    if not value:
        return None
    m = re.search(r"models--([^/]+?)--([^/]+)", value)
    if m:
        return f"{m.group(1)}/{m.group(2)}"
    if re.fullmatch(r"[\w.-]+/[\w.-]+", value):
        return value
    return value.rstrip("/").rsplit("/", 1)[-1]


def _dataset_name(value: str | None) -> str | None:
    """The training data a run used, as its folder or dataset name."""
    if not value:
        return None
    parts = [p for p in value.rstrip("/").split("/") if p]
    named = [p for p in parts if "samples" in p]
    if named:
        return named[0]
    if "datasets" in parts[:-1]:
        return parts[parts.index("datasets") + 1]
    return parts[-1]


def repo_record(repo: str) -> dict:
    info, _ = _json(f"{API}/{repo}?blobs=true")
    files = {s["rfilename"]: s for s in info.get("siblings", [])}
    rec = {
        "created": (info.get("createdAt") or "")[:10],
        "last_modified": (info.get("lastModified") or "")[:10],
        "revision": info.get("sha"),
        "card_base_model": (info.get("cardData") or {}).get("base_model"),
        "weights": sorted(f"{name}:{(s.get('lfs') or {}).get('sha256', '')}"
                          for name, s in files.items() if name.endswith(".safetensors")),
        "run_summary_base_model": None,
        "run_summary_dataset": None,
        "config": None,
    }
    if isinstance(rec["card_base_model"], list):
        rec["card_base_model"] = rec["card_base_model"][0] if rec["card_base_model"] else None
    for name, key in (("run_summary.json", "run_summary"), ("config.json", "config")):
        if name not in files:
            continue
        try:
            body, _ = get(f"https://huggingface.co/{repo}/resolve/{info.get('sha')}/{name}")
            time.sleep(PAUSE)
            data = json.loads(body)
        except (NotFound, ValueError):
            continue
        if key == "run_summary":
            rec["run_summary_base_model"] = _parent_from_run_summary(data.get("base_model_name"))
            rec["run_summary_dataset"] = _dataset_name(data.get("dataset_name"))
        else:
            rec["config"] = {k: data.get(k) for k in
                             ("max_position_embeddings", "rope_theta", "rope_scaling")}
    return rec


def resolve_hash(snapshot_id: str) -> str | None:
    for repo in EXTERNAL:
        try:
            _json(f"{API}/{repo}/revision/{snapshot_id}")
            return repo
        except NotFound:
            continue
    return None


def snapshot(names: set[str], repo_ids: set[str], out: Path, max_depth: int = 4,
             previous: dict | None = None) -> dict:
    """`names`: bare checkpoint folder names. `repo_ids`: full repo ids.
    `previous`: a saved snapshot whose records are kept and not fetched again. A
    name it found no repository for is looked up again, since the repository may
    have been uploaded since."""
    resolved = dict(previous["names"]) if previous else {}
    repos: dict[str, dict] = dict(previous["repos"]) if previous else {}
    by_name: dict[str, list[str]] = {}
    new_names = sorted(n for n in names if not resolved.get(n))
    if new_names:
        for org in ORGS:
            for repo in list_org(org):
                by_name.setdefault(repo.split("/", 1)[1], []).append(repo)
    for name in new_names:
        if _HASH.match(name):
            repo = resolve_hash(name)
            resolved[name] = [repo] if repo else []
        else:
            # Folder names are sometimes cut short (`...-decontaminate`), so a
            # name with no exact match is matched to the one repo it starts.
            exact = sorted(by_name.get(name, []))
            prefix = sorted({r for n, rs in by_name.items() if n.startswith(name) for r in rs})
            resolved[name] = exact or (prefix if len({bare(r) for r in prefix}) == 1 else [])
    queue = [(r, 0) for r in sorted({r for v in resolved.values() for r in v} | set(repo_ids))
             if r not in repos]
    while queue:
        repo, depth = queue.pop(0)
        if repo in repos:
            continue
        try:
            repos[repo] = repo_record(repo)
        except NotFound:
            repos[repo] = {"missing": True}
            continue
        if depth < max_depth:
            for parent in (repos[repo]["card_base_model"], repos[repo]["run_summary_base_model"]):
                if parent and "/" in parent and parent not in repos:
                    queue.append((parent, depth + 1))
    today = dt.date.today().isoformat()
    data = {"fetched": previous["fetched"] if previous else today, "orgs": list(ORGS),
            "names": dict(sorted(resolved.items())), "repos": dict(sorted(repos.items()))}
    if previous:
        added = sorted(set(repos) - set(previous["repos"]))
        data["updates"] = previous.get("updates", []) + [{"date": today, "added": added}]
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(data, indent=1, sort_keys=True) + "\n")
    return data


def main(workshop_dir: Path, phase2_dir: Path, out: Path, update: bool = False) -> int:
    """Snapshot every checkpoint that has results in the manifest's sources."""
    import pandas as pd

    from .. import inputs
    from ..data import protocol_a as pa
    from ..data.checkpoint_guard import checkpoint_id
    from . import phase2 as p2

    repo_ids = set()
    for _, input_id in pa.RECORD_FILES:
        try:
            path = inputs.path(input_id, workshop_dir / inputs.get(input_id).file)
        except inputs.InputError as exc:
            print(f"error: {exc}")
            return 1
        repo_ids |= set(pd.read_csv(path, usecols=["model_id"])["model_id"])
    _, meta = p2.load(phase2_dir)
    names = {n for n in meta["source_dir"].map(checkpoint_id).dropna()}
    previous = json.loads(out.read_text()) if update and out.exists() else None
    data = snapshot(names, repo_ids, out, previous=previous)
    found = sum(1 for v in data["names"].values() if v)
    print(f"wrote {out}: {len(data['repos'])} repos, {found} of {len(names)} Phase 2 "
          f"checkpoint names found on Hugging Face")
    return 0
