"""The three-seed Phase 2 records (marianna13, released 2026-10-07, GSM8K re-run
2026-10-15, IFEval scored by all four definitions 2026-10-19).

Source: the Hugging Face dataset `laion/mv-posttrain-eval-responses` at a
fixed revision. Three files are used:

  eval_scores.parquet     one released score per (base_model, stage, task),
                          with the three per-seed scores. IFEval has one row
                          per definition, and the paper uses IFEVAL_METRIC
  eval_responses.parquet  one row per answer. Only its small columns are
                          fetched: model, stage, task, question, seed,
                          correctness, checkpoint folder and token limit.
                          The answer text (most of its 10 GB) is skipped
  ifeval_allmetrics_persample.parquet
                          one row per IFEval prompt and seed, with its count
                          of instructions and of those followed, strict and
                          loose. eval_responses gives IFEval one strict
                          prompt-level verdict only

The three-seed rows are those with `decoding == "sampling"` (and
`num_repeat == 3` in eval_scores). They are sampled at temperature 0.7, once
per question per seed, for three seeds (`run_idx` 0, 1 and 2).
"""
from __future__ import annotations

import hashlib
from pathlib import Path

import pandas as pd
import pyarrow.parquet as pq

from .. import inputs as _inputs
from .http import get, open_url

DATASET = "laion/mv-posttrain-eval-responses"
REVISION = "e820fb3461f01d330a8f278892cbd9e39e8ba710"
SCORES = "eval_scores.parquet"
IFEVAL = "ifeval_allmetrics_persample.parquet"
#: The IFEval definition of the paper's Phase 2 results: instruction-level
#: accuracy with the strict checker, which marianna13's revision of 2026-10-19
#: names as Protocol A's, so both protocols use one definition (decided on
#: 2026-10-20).
IFEVAL_METRIC = "inst-level,strict"
RESPONSES_META = "eval_responses_meta.parquet"
META_COLUMNS = ["base_model", "stage", "task", "example_idx", "run_idx", "example_id",
                "is_correct", "source_dir", "source_file", "decoding", "max_tokens"]

#: SHA-256 of eval_scores.parquet at REVISION, input R14-scores of
#: mv_post_train/inputs.json. RESPONSES_META is input R14-meta.
SCORES_SHA256 = _inputs.get("R14-scores").sha256


def url(name: str, revision: str = REVISION) -> str:
    return f"https://huggingface.co/datasets/{DATASET}/resolve/{revision}/{name}"


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(2**20), b""):
            h.update(chunk)
    return h.hexdigest()


def fetch(out_dir: Path, revision: str = REVISION) -> list[Path]:
    """Download eval_scores.parquet, the IFEval counts and the small columns of
    eval_responses.parquet."""
    out_dir.mkdir(parents=True, exist_ok=True)
    body, _ = get(url(SCORES, revision))
    scores = out_dir / SCORES
    scores.write_bytes(body)
    body, _ = get(url(IFEVAL, revision))
    (out_dir / IFEVAL).write_bytes(body)
    pf = pq.ParquetFile(open_url(url("eval_responses.parquet", revision)))
    meta = out_dir / RESPONSES_META
    pq.write_table(pf.read(columns=META_COLUMNS), meta)
    return [scores, out_dir / IFEVAL, meta]


def load(data_dir: Path) -> tuple[pd.DataFrame, pd.DataFrame]:
    """(three-seed released scores, three-seed answer rows). IFEval keeps the
    rows of IFEVAL_METRIC only."""
    scores = pd.read_parquet(data_dir / SCORES)
    scores = scores[(scores["decoding"] == "sampling") & (scores["num_repeat"] == 3)
                    & ((scores["task"] != "IFEval") | (scores["metric"] == IFEVAL_METRIC))].copy()
    meta = pd.read_parquet(data_dir / RESPONSES_META)
    meta = meta[meta["decoding"] == "sampling"].copy()
    return scores, meta


def ifeval_counts(data_dir: Path) -> pd.DataFrame:
    """The per-prompt IFEval counts: base_model, stage, run_idx, example_id, the
    number of instructions (n_total) and of those followed strictly (n_strict)
    and loosely (n_loose)."""
    return pd.read_parquet(data_dir / IFEVAL)


def main(out_dir: Path, revision: str) -> int:
    """Fetch the records and say whether they are the pinned inputs. At another
    revision, or with other package versions, they are not, and
    build-manifest records that as a failed check."""
    for path, input_id in zip(fetch(out_dir, revision), ("R14-scores", "R14-ifeval", "R14-meta")):
        got = sha256(path)
        pinned = "the pinned input" if got == _inputs.get(input_id).sha256 else (
            f"NOT the pinned input {input_id} (mv_post_train/inputs.json)")
        print(f"  wrote {path}  sha256 {got}, {pinned}")
    return 0
