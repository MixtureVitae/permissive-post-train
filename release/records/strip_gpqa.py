"""Write the copy of logs.zip (input R6) that the records dataset publishes:
the same zip without the GPQA Diamond items.

    python release/records/strip_gpqa.py <bucket logs.zip> <out.zip>

GPQA asks that its items not be published in plain text. Evalchemy saves every
item it scores, with its question, choices, explanation, validator notes and
canary string, under `results.GPQADiamond.examples` of each results file. The
copy keeps of each item only its GPQA record ID, the letter of the correct
choice and the answer the model gave in each repetition if it is a letter (A to
D, as the pipeline's GPQA scorer reads it), or an empty string, since the other
answers can quote a choice. Every answer scores as before. Everything
else is unchanged: the other members byte for byte, each results file's scores
and settings, and the nested zips of the ot30k evaluations, whose results files
get the same change. The script checks that each repetition's solved count is
still in what is kept.
"""
from __future__ import annotations

import io
import json
import sys
import zipfile
from pathlib import Path

from mv_post_train.data.scorers import _norm

LETTERS = {"A", "B", "C", "D"}


def strip_item(item: dict) -> dict:
    return {"Record ID": item["Record ID"], "answer": item["answer"],
            "model_answers": [a if _norm(a) in LETTERS else "" for a in item["model_answers"]]}


def strip_results(raw: bytes) -> bytes:
    """A results file without GPQA item text, serialised as Evalchemy writes it."""
    data = json.loads(raw)
    if json.dumps(data, indent=2, ensure_ascii=False).encode() != raw:
        raise SystemExit("a results file is not in Evalchemy's layout; the copy would differ")
    gpqa = data["results"]["GPQADiamond"]
    gpqa["examples"] = [strip_item(e) for e in gpqa["examples"]]
    for r, stats in enumerate(gpqa["run_stats"]):
        solved = sum(e["model_answers"][r] == e["answer"] for e in gpqa["examples"])
        if solved != stats["num_solved"]:
            raise SystemExit(f"repetition {r + 1}: {solved} solved, not {stats['num_solved']}")
    return json.dumps(data, indent=2, ensure_ascii=False).encode()


def copy_zip(src: zipfile.ZipFile, dst: zipfile.ZipFile, depth: int = 0) -> int:
    """Copy every member of `src` to `dst` in order, stripping the GPQA results
    files, and nested zips one level down. Returns the number stripped."""
    stripped = 0
    for info in src.infolist():
        raw = src.read(info)
        if info.filename.endswith(".json") and b'"GPQADiamond"' in raw:
            raw = strip_results(raw)
            stripped += 1
        elif info.filename.endswith(".zip") and depth == 0:
            buf = io.BytesIO()
            with zipfile.ZipFile(io.BytesIO(raw)) as inner, zipfile.ZipFile(buf, "w") as out:
                n = copy_zip(inner, out, depth + 1)
            if n:
                raw, stripped = buf.getvalue(), stripped + n
        elif b"GPQA" in raw.upper():
            raise SystemExit(f"{info.filename} mentions GPQA and is not a results file")
        dst.writestr(info, raw)
    return stripped


def main(src: str, out: str) -> None:
    with zipfile.ZipFile(src) as z, zipfile.ZipFile(Path(out), "w") as o:
        n = copy_zip(z, o)
    print(f"{n} results files stripped -> {out}")


if __name__ == "__main__":
    main(*sys.argv[1:])
