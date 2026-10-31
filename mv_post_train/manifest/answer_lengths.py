"""The length of every sampled Phase 2 answer, and whether the grader found an answer in it.

    mvpt measure-phase2-answers --responses <eval_responses.parquet>

Reads the whole eval_responses.parquet of the pinned revision (input
R14-responses, 10 GB, of which the answer text is most), one batch at a time,
and writes one row per sampled answer without its text (input R14-lengths):

  base_model, stage, task, example_idx, example_id, run_idx, max_tokens
                the answer's cell, as in R14-meta
  n_bytes       UTF-8 length of the answer (`model_output`)
  n_tokens      the answer re-encoded with its checkpoint's own tokenizer, for
                answers of at least TOKENIZE_FROM times the token limit in
                bytes, otherwise null. Every tokenizer here is a byte-level
                BPE, so an answer has at most as many tokens as bytes, and a
                shorter answer cannot have reached the limit
  no_answer     `model_answer` is empty: the answer the grader compares with
                the reference was not found
  no_extracted  `model_answer_extracted` is empty. For the code tasks this is
                the code block the checker runs
  tokenizer     the tokenizer.json used, as <repo>@<revision>, or null for a
                checkpoint whose tokenizer is not found (n_tokens is then null)

Only the cells of a checkpoint the results manifest labels are kept, since the
tokenizer comes from the checkpoint's Hugging Face repo, at the revision the
checkpoint snapshot (input S1) recorded. A checkpoint without a repo uses the
tokenizer of its nearest ancestor in the manifest that has one. Each tokenizer.json is checked
against the hash Hugging Face lists for it and kept in TOKENIZER_DIR.

Re-encoding the text does not always give back the generated tokens, so
`stats/answer_endings.py` decides from n_tokens which answers reached the
limit, with a tolerance.
"""
from __future__ import annotations

import ctypes
import hashlib
import json
import time
from pathlib import Path

import pyarrow as pa
import pyarrow.compute as pc
import pyarrow.parquet as pq

from .http import get
from .phase2 import DATASET, REVISION
from .reader import Manifest, ManifestError

RESPONSES = "eval_responses.parquet"
LENGTHS = "answer_lengths.parquet"
TOKENIZER_DIR = "tokenizers"
#: Where each row group's rows wait until the file is written, so that a run
#: that is stopped (it takes about an hour) starts again where it stopped.
PARTS_DIR = "answer_lengths.parts"
#: Answers shorter than this share of the token limit, in bytes, are not
#: re-encoded.
TOKENIZE_FROM = 0.9
#: Seconds between Hugging Face API calls, which rate-limit bursts of a few
#: hundred unauthenticated requests.
API_PAUSE = 1.0
BATCH_ROWS = 500
#: Answers encoded in one call, which bounds the memory the encodings take.
ENCODE_ROWS = 32
KEYS = ["base_model", "stage", "task", "example_idx", "example_id", "run_idx", "max_tokens"]
READ = KEYS + ["decoding", "model_output", "model_answer", "model_answer_extracted"]


def _blob_sha1(body: bytes) -> str:
    return hashlib.sha1(b"blob %d\0" % len(body) + body).hexdigest()


def _repo_file(repo: str, revision: str, name: str) -> tuple[str, str] | None:
    """(hash kind, hash) that Hugging Face lists for one file of a model repo,
    or None if the repo has no such file."""
    time.sleep(API_PAUSE)
    body, _ = get(f"https://huggingface.co/api/models/{repo}/tree/{revision}")
    for f in json.loads(body):
        if f["path"] == name:
            return ("sha256", f["lfs"]["oid"]) if f.get("lfs") else ("sha1", f["oid"])
    return None


def tokenizer_sources(m: Manifest, snapshot: dict, cells: list[tuple[str, str]]
                      ) -> dict[tuple[str, str], tuple[str, str, tuple[str, str]] | None]:
    """(base_model, stage) -> (repo, revision, (hash kind, hash)) of the
    tokenizer.json each cell's answers are re-encoded with, or None if neither
    the checkpoint nor an ancestor in the manifest has one."""
    found: dict[str, tuple[str, str, tuple[str, str]] | None] = {}

    def of_checkpoint(key: str | None, seen: tuple = ()):
        if key is None or key in seen or key not in m.data["checkpoints"]:
            return None
        if key not in found:
            found[key] = None
            for repo in m.data["checkpoints"][key]["hf_repos"]:
                revision = snapshot["repos"].get(repo, {}).get("revision")
                listed = revision and _repo_file(repo, revision, "tokenizer.json")
                if listed:
                    found[key] = (repo, revision, listed)
                    break
        if found[key] is None:
            return of_checkpoint(m.parent(key)[0], seen + (key,))
        return found[key]

    return {cell: of_checkpoint(m.checkpoint(*cell)) for cell in cells}


def load_tokenizer(repo: str, revision: str, listed: tuple[str, str], cache: Path):
    kind, digest = listed
    path = cache / f"{digest}.json"
    if not path.exists():
        body, _ = get(f"https://huggingface.co/{repo}/resolve/{revision}/tokenizer.json")
        got = hashlib.sha256(body).hexdigest() if kind == "sha256" else _blob_sha1(body)
        if got != digest:
            raise ValueError(f"{repo}@{revision} tokenizer.json has {kind} {got}, Hugging Face "
                             f"lists {digest}.")
        cache.mkdir(parents=True, exist_ok=True)
        path.write_bytes(body)
    text = path.read_text()
    spec = json.loads(text)
    if "ByteLevel" not in json.dumps(spec.get("pre_tokenizer")) + json.dumps(spec.get("decoder")):
        raise ValueError(f"{repo}@{revision}: not a byte-level tokenizer, so an answer can have "
                         "more tokens than bytes and the TOKENIZE_FROM shortcut does not hold.")
    return _Fresh(text)


class _Fresh:
    """A tokenizer that is built again for every batch. A BPE tokenizer caches
    the words it has split, and the degenerate answers hold long words, so
    with some forty tokenizers the caches outgrow this machine's memory."""

    def __init__(self, spec: str):
        self.spec = spec

    def encode_batch_fast(self, texts: list[str], add_special_tokens: bool = False):
        from tokenizers import Tokenizer

        return Tokenizer.from_str(self.spec).encode_batch_fast(
            texts, add_special_tokens=add_special_tokens)


def _one_checkpoint(m: Manifest, cell: tuple[str, str]) -> bool:
    """Whether the manifest labels exactly one checkpoint with this cell. It
    leaves out cells whose answers name several."""
    try:
        m.checkpoint(*cell)
    except ManifestError:
        return False
    return True


def _trim() -> None:
    """Give freed memory back to the system. An encoding of a long answer holds
    tens of thousands of tokens, and without this the process keeps the peak
    of every batch."""
    try:
        ctypes.CDLL("libc.so.6").malloc_trim(0)
    except OSError:
        pass
    pa.default_memory_pool().release_unused()


def _empty(col: pa.ChunkedArray) -> pa.Array:
    return pc.equal(pc.fill_null(col, ""), "")


def _group_rows(pf: pq.ParquetFile) -> list[range]:
    """The row numbers of each row group."""
    out, at = [], 0
    for g in range(pf.metadata.num_row_groups):
        n = pf.metadata.row_group(g).num_rows
        out.append(range(at, at + n))
        at += n
    return out


def _measure_group(pf: pq.ParquetFile, g: int, keep: dict, sources: dict,
                   tokenizers: dict) -> pa.Table:
    parts = []
    for batch in pf.iter_batches(batch_size=BATCH_ROWS, columns=READ, row_groups=[g],
                                 use_threads=False):
        t = pa.Table.from_batches([batch])
        t = t.filter(pc.equal(t["decoding"], "sampling"))
        cell = list(zip(t["base_model"].to_pylist(), t["stage"].to_pylist()))
        mask = [c in keep for c in cell]
        t = t.filter(pa.array(mask, pa.bool_()))
        cell = [c for c, k in zip(cell, mask) if k]
        if not t.num_rows:
            continue
        n_bytes = pc.binary_length(pc.cast(pc.fill_null(t["model_output"], ""), pa.binary()))
        limit = pc.cast(t["max_tokens"], pa.float64())
        long_ = pc.greater_equal(pc.cast(n_bytes, pa.float64()),
                                 pc.multiply(limit, TOKENIZE_FROM)).to_pylist()
        n_tokens: list[int | None] = [None] * t.num_rows
        text = t["model_output"]
        by_tok: dict[tuple[str, str], list[int]] = {}
        for i, (is_long, c) in enumerate(zip(long_, cell)):
            if is_long and sources[c] is not None:
                by_tok.setdefault(sources[c][:2], []).append(i)
        for src, rows in by_tok.items():
            for at in range(0, len(rows), ENCODE_ROWS):
                chunk = rows[at:at + ENCODE_ROWS]
                enc = tokenizers[src].encode_batch_fast([text[i].as_py() for i in chunk],
                                                        add_special_tokens=False)
                for i, e in zip(chunk, enc):
                    n_tokens[i] = len(e.ids)
                del enc
        _trim()
        parts.append(pa.table({
            **{k: t[k] for k in KEYS},
            "n_bytes": n_bytes,
            "n_tokens": pa.array(n_tokens, pa.int64()),
            "no_answer": _empty(t["model_answer"]),
            "no_extracted": _empty(t["model_answer_extracted"]),
            "tokenizer": pa.array([keep[c] for c in cell], pa.string()),
        }))
    return pa.concat_tables(parts).combine_chunks()


def measure(responses: Path, m: Manifest, snapshot: dict, cache: Path,
            row_groups: list[int] | None = None, parts_dir: Path | None = None) -> pa.Table:
    """The R14-lengths rows, from every row group of `responses` or the ones
    named. With `parts_dir`, each row group's rows are kept there as they are
    done, and a run that was stopped starts again where it stopped."""
    pf = pq.ParquetFile(responses, buffer_size=1 << 20, pre_buffer=False)
    meta = pf.read(columns=["base_model", "stage", "decoding"])
    label = pc.binary_join_element_wise(pc.cast(meta["base_model"], pa.string()),
                                        pc.cast(meta["stage"], pa.string()), "\x00")
    sampled = pc.equal(meta["decoding"], "sampling")
    found = pc.unique(pc.filter(label, sampled)).to_pylist()
    cells = sorted(c for c in (tuple(x.split("\x00")) for x in found) if _one_checkpoint(m, c))
    sources = tokenizer_sources(m, snapshot, cells)
    for c in sorted(c for c, src in sources.items() if src is None):
        print(f"  no tokenizer for {c[0]} {c[1]}: its answers are kept without token counts")
    tokenizers = {src[:2]: load_tokenizer(*src, cache)
                  for src in set(sources.values()) if src is not None}
    keep = {c: f"{src[0]}@{src[1]}" if src else None for c, src in sources.items()}

    n = pf.metadata.num_row_groups
    kept = pc.and_(sampled, pc.is_in(label, pa.array(["\x00".join(c) for c in keep])))
    rows = _group_rows(pf)
    wanted = [g for g in (range(n) if row_groups is None else row_groups)
              if pc.any(kept.slice(rows[g].start, len(rows[g]))).as_py()]
    del meta, label, sampled, kept

    tables = []
    for g in wanted:
        part = parts_dir / f"row_group_{g:02d}.parquet" if parts_dir else None
        if part is not None and part.exists():
            tables.append(pq.read_table(part))
            continue
        print(f"  row group {g + 1} of {n}", flush=True)
        table = _measure_group(pf, g, keep, sources, tokenizers)
        pa.default_memory_pool().release_unused()
        if part is not None:
            parts_dir.mkdir(parents=True, exist_ok=True)
            pq.write_table(table, part)
            table = pq.read_table(part)
        tables.append(table)
    return pa.concat_tables(tables).combine_chunks()


def main(responses: Path, manifest: Path, snapshot: Path, out_dir: Path) -> int:
    from .. import inputs

    got = inputs.sha256(responses)
    pinned = inputs.get("R14-responses").sha256
    if got != pinned:
        print(f"error: {responses} has SHA-256 {got}, not the pinned R14-responses {pinned} "
              f"({DATASET} at {REVISION[:8]}).")
        return 1
    parts = out_dir / PARTS_DIR
    table = measure(responses, Manifest.load(manifest), json.loads(snapshot.read_text()),
                    out_dir / TOKENIZER_DIR, parts_dir=parts)
    out = out_dir / LENGTHS
    pq.write_table(table, out)
    for part in sorted(parts.glob("row_group_*.parquet")):
        part.unlink()
    parts.rmdir()
    print(f"  wrote {out}  {table.num_rows} answers, sha256 {inputs.sha256(out)}")
    return 0

