"""Read-only access to results/manifest.json, for the commands that render
tables and figures from it.

Every number a renderer prints comes through here, and every change keeps the
manifest's own rules:

  - a suite-average change is one the manifest computed (its `changes`), so it
    is against the checkpoint's recorded parent, over the same tasks, under
    one protocol;
  - a single-task change needs the checkpoint's recorded parent, a result for
    that task on both sides under the same protocol, and no rejected pairing.

Anything else raises `ManifestError` with the reason, so a table cannot
quietly print a change the manifest would not make.
"""
from __future__ import annotations

import json
from pathlib import Path

import pandas as pd

from .build import _A_TASK, PHASE2, PROTOCOL_A, TASKS

#: The manifest's task names -> the Protocol A records' (data/protocol_a.py).
RECORD_TASK = {v: k for k, v in _A_TASK.items()}


class ManifestError(Exception):
    """The manifest has no such result, or does not allow the change asked for."""


class Manifest:
    def __init__(self, data: dict):
        self.data = data
        self.checkpoints: dict[str, dict] = data["checkpoints"]
        self._results = {(r["checkpoint"], r["protocol"], r["task"]): r for r in data["results"]}
        self._suites = {(s["checkpoint"], s["protocol"]): s for s in data["suite_averages"]}
        self._changes = {(c["checkpoint"], c["protocol"]): c for c in data["changes"]}
        self._rejected = {(c["checkpoint"], c["protocol"]): c for c in data["rejected_changes"]}
        self._by_label: dict[tuple, list[str]] = {}
        for key, ck in self.checkpoints.items():
            for protocol, lab in ck["labels"].items():
                if "base_model" in lab:
                    self._by_label.setdefault((protocol, lab["base_model"], lab["stage"]),
                                              []).append(key)

    @classmethod
    def load(cls, path: Path) -> "Manifest":
        return cls(json.loads(Path(path).read_text()))

    # --- protocol --------------------------------------------------------

    def tasks(self, protocol: str = PHASE2) -> list[str]:
        """The protocol's tasks in the paper's column order, without excluded ones."""
        have = self.data["protocols"][protocol]["tasks"]
        return [t for t in TASKS if t in have]

    def excluded(self, protocol: str = PHASE2) -> dict[str, str]:
        return dict(self.data["protocols"][protocol].get("excluded_tasks", {}))

    def task_info(self, task: str, protocol: str = PHASE2) -> dict:
        return self.data["protocols"][protocol]["tasks"][task]

    # --- checkpoints and results ------------------------------------------

    def labelled(self, base_model: str, stage: str, protocol: str = PHASE2) -> bool:
        """Whether any checkpoint is labelled (base_model, stage)."""
        return bool(self._by_label.get((protocol, base_model, stage)))

    def checkpoint(self, base_model: str, stage: str, protocol: str = PHASE2) -> str:
        """The checkpoint the records label (base_model, stage)."""
        keys = self._by_label.get((protocol, base_model, stage), [])
        if len(keys) != 1:
            raise ManifestError(f"{len(keys)} checkpoints are labelled {base_model} {stage} "
                                f"under {protocol}, expected one.")
        return keys[0]

    def result(self, key: str, task: str, protocol: str = PHASE2) -> dict:
        try:
            return self._results[(key, protocol, task)]
        except KeyError:
            raise ManifestError(f"No {task} result for {key} under {protocol}.") from None

    def suite(self, key: str, protocol: str = PHASE2) -> dict:
        s = self._suites.get((key, protocol))
        if s is None or not s["complete"]:
            missing = ", ".join(s["missing_tasks"]) if s else "every task"
            raise ManifestError(f"{key} has no suite average under {protocol}: missing {missing}.")
        return s

    def results_for(self, key: str, protocol: str) -> list[dict]:
        """Every result of one checkpoint under one protocol, in the manifest's order."""
        return [r for (k, p, _), r in self._results.items() if k == key and p == protocol]

    def value(self, key: str, task: str, protocol: str) -> float:
        return self.result(key, task, protocol)["mean"]

    def design(self, name: str) -> dict:
        try:
            return self.data["designs"][name]
        except KeyError:
            raise ManifestError(f"The manifest has no design {name!r}.") from None

    def seed_scores(self, cells: list[dict], keys: list[str],
                    protocol: str = PROTOCOL_A) -> pd.DataFrame:
        """One row per cell, seed and task: the cell's `keys`, `seed`,
        `benchmark` (the records' task name) and `score` in percent, from the
        results of each cell's checkpoint."""
        rows = []
        for cell in cells:
            for task in self.tasks(protocol):
                r = self.result(cell["checkpoint"], task, protocol)
                for seed, score in r["per_seed"].items():
                    rows.append({**{k: cell[k] for k in keys}, "seed": int(seed),
                                 "benchmark": RECORD_TASK.get(task, task), "score": score})
        return pd.DataFrame(rows)

    def check_records(self, records: pd.DataFrame, cells: list[dict], keys: list[str],
                      protocol: str = PROTOCOL_A) -> None:
        """Stop unless per-question records (the columns of
        `data/protocol_a.load_reasoning`) hold, for every cell, the checkpoint
        the cell names, with the manifest's per-seed score on every task. The
        manifest keeps six decimals."""
        from .lineage import bare
        used = records.drop_duplicates("model_id").set_index(keys)["model_id"].map(bare)
        for cell in cells:
            got = used.get(tuple(cell[k] for k in keys))
            if got != cell["checkpoint"]:
                raise ManifestError(f"The records' checkpoint of {cell} is {got}.")
        per_seed = (records.groupby(keys + ["seed", "benchmark"], observed=True)["correct"]
                    .mean().mul(100.0).reset_index(name="score"))
        both = per_seed.merge(self.seed_scores(cells, keys, protocol),
                              on=keys + ["seed", "benchmark"], how="right")
        if both["score_x"].isna().any() or (both["score_x"] - both["score_y"]).abs().max() > 1e-5:
            raise ManifestError("The per-question records do not give the manifest's "
                                "per-seed scores.")

    def same_model(self, a: str, b: str) -> bool:
        """The same checkpoint key, or two checkpoints with the same weights."""
        if a == b:
            return True
        wa = set(self.checkpoints.get(a, {}).get("weights_sha256") or [])
        wb = set(self.checkpoints.get(b, {}).get("weights_sha256") or [])
        return bool(wa and wb and wa & wb)

    def line(self, key: str, steps: int = 8) -> list[str]:
        """The checkpoints `key` was trained from, nearest first, as far as the
        manifest's parents go."""
        out = []
        while len(out) < steps:
            ck = self.checkpoints.get(key)
            if ck is None or ck["parent"] is None or ck["parent_status"] == "conflict":
                break
            key = ck["parent"]
            out.append(key)
        return out

    def parent(self, key: str) -> tuple[str | None, str]:
        ck = self.checkpoints[key]
        return ck["parent"], ck["parent_status"]

    # --- changes ---------------------------------------------------------------

    def change(self, key: str, protocol: str = PHASE2) -> dict:
        """The manifest's suite-average change for `key`, against its parent."""
        c = self._changes.get((key, protocol))
        if c is not None:
            return c
        if (key, protocol) in self._rejected:
            raise ManifestError(f"The change for {key} was rejected: "
                                f"{self._rejected[(key, protocol)]['reason']}")
        raise ManifestError(f"The manifest has no change for {key} under {protocol}.")

    def task_change(self, key: str, task: str, protocol: str = PHASE2) -> tuple[float, str, str]:
        """(child minus parent on `task`, parent, parent status)."""
        if (key, protocol) in self._rejected:
            raise ManifestError(f"The change for {key} was rejected: "
                                f"{self._rejected[(key, protocol)]['reason']}")
        parent, status = self.parent(key)
        if parent is None or status in ("none recorded", "conflict"):
            raise ManifestError(f"{key} has no usable parent ({status}).")
        child = self.result(key, task, protocol)["mean"]
        return child - self.result(parent, task, protocol)["mean"], parent, status
