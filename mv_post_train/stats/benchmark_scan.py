"""The paper's benchmarks against the training data that only a cluster holds.

    mvpt scan-benchmarks texts --tokenized-dir DIR --tokenizer DIR   (where the caches are)
    mvpt scan-benchmarks scan SOURCE --file PATH                     (once per SOURCES entry)
    mvpt scan-benchmarks merge

(or `python -m mv_post_train.stats.benchmark_scan texts|scan ...`, below, where
the package's other dependencies are missing.)

training_data_overlap.py scans the post-training data with a public release
and the questions of the reasoning corpora. This scans the rest of what the
models trained on, with its index, its two 13-gram scans and its exact pass
(SOURCES):

  phase2:<dataset>  the four Phase 2 30K samples as trained: LLaMA-Factory's
                    tokenized caches (phase2_tokens.CACHES) decoded by `texts`,
                    each sample's prompt (its tokens without a label: the
                    template and the question) and its answer (the supervised
                    tokens), after the cut at 16,384 tokens. PATH is
                    `texts/<dataset>.parquet`
  sample:<corpus>   the 30K samples of the three other reasoning corpora as
                    written (question_overlap.TRAINED), every message, before
                    the template and the cut. PATH is the folder holding them.
                    OT3's 30K is read only as trained
  mv:<shard>        MixtureVitae's reasoning data other than `open_thoughts`
                    (mv_question_overlap.SHARDS, 183 GB), each record split on
                    `<|endoftext|>` into the documents the MixtureVitae
                    decontamination scanned. PATH is the shard's .jsonl.gz.
                    This is the pretraining pool: whether a 300B-token run drew
                    a document depends on the sampling of its source

`scan` needs `benchmark_items.parquet` and `index/` from
measure-training-data-overlap's output folder, copied to the work folder;
both stay private on the cluster, since the items include the GPQA-Diamond
questions. It writes `sources/<SOURCE>.parquet`, one row per document and
item:

  doc               the document: `row:prompt` or `row:answer` (phase2),
                    `row:turn` (sample), corpus-assay's `<record>-part-<k>` (mv)
  role              prompt or answer; the message's role; pretraining
  benchmark, item   as in training_data_overlap
  hits, item_grams, longest_run, flagged
                    the 13-gram scans, as in its `overlap.parquet`
  exact, question_chars, choices, choices_found
                    the exact pass, as in its `exact.parquet`
  boolq_question    for BoolQ, whether the document holds the question too

A shard keeps the rows that reach FIFTH of the item's 13-grams, the exact
matches `found` can use, and the rule's flags; the smaller sources keep every
row. `merge` joins the sources into `benchmark_scan.parquet` (with `where`:
`phase2:<dataset>`, `sample:<corpus>` or `mv_reasoning`, and `source`, the
shard), which `training_data_overlap.found` reads.
"""
from __future__ import annotations

import hashlib
import json
import os
import shutil
from multiprocessing import Pool
from pathlib import Path

import numpy as np
import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq

from . import mv_question_overlap as mq
from . import phase2_tokens as p2t
from . import question_overlap as qo
from . import training_data_overlap as tdo

SOURCES = ([f"phase2:{d}" for d in p2t.CACHES] + [f"sample:{c}" for c in qo.TRAINED]
           + [f"mv:{mq.shard_name(f)}" for f, _ in mq.SHARDS])
FIFTH = qo.LEVELS["fifth"]
FILES = 32
ROLES = {"human": "user", "user": "user", "gpt": "assistant", "assistant": "assistant",
         "system": "system"}
COLUMNS = ["doc", "role", "benchmark", "item", "hits", "item_grams", "longest_run", "flagged",
           "exact", "question_chars", "choices", "choices_found", "boolq_question"]


# ---------------------------------------------------------------------------
# The caches, decoded


def _sha256(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for block in iter(lambda: f.read(1 << 24), b""):
            h.update(block)
    return h.hexdigest()


def texts(tokenized_dir: Path, tokenizer: Path, out_dir: Path) -> int:
    """`texts/<dataset>.parquet` for each cache of phase2_tokens.CACHES: a row
    per sample (`row`, its place in the cache), its `prompt` and its `answer`,
    decoded with the open-sci tokenizer in `tokenizer`."""
    from transformers import AutoTokenizer
    tok = AutoTokenizer.from_pretrained(str(tokenizer))
    dest = out_dir / "texts"
    dest.mkdir(parents=True, exist_ok=True)
    meta = dict(tokenizer=str(tokenizer), tokenizer_json=_sha256(tokenizer / "tokenizer.json"),
                caches={})
    for name, cache in p2t.CACHES.items():
        prompts, answers = [], []
        files = sorted((tokenized_dir / cache / "train").glob("*.arrow"))
        for f in files:
            with pa.memory_map(str(f)) as src:
                t = pa.ipc.open_stream(src).read_all()
            p, a = [], []
            for ids_chunk, lab_chunk in zip(t["input_ids"].chunks, t["labels"].chunks):
                ids_off, ids = ids_chunk.offsets.to_numpy(), ids_chunk.values.to_numpy()
                lab_off, lab = lab_chunk.offsets.to_numpy(), lab_chunk.values.to_numpy()
                for i in range(len(ids_off) - 1):
                    x = ids[ids_off[i]:ids_off[i + 1]]
                    y = lab[lab_off[i]:lab_off[i + 1]]
                    p.append(x[y == p2t.LABEL_IGNORE].tolist())
                    a.append(x[y != p2t.LABEL_IGNORE].tolist())
            prompts += tok.batch_decode(p, skip_special_tokens=False)
            answers += tok.batch_decode(a, skip_special_tokens=False)
        pq.write_table(pa.table({"row": pa.array(range(len(prompts)), pa.int64()),
                                 "prompt": pa.array(prompts, pa.string()),
                                 "answer": pa.array(answers, pa.string())}),
                       dest / f"{name}.parquet", compression="zstd")
        meta["caches"][name] = dict(cache=cache, files=len(files), samples=len(prompts),
                                    bytes=sum(f.stat().st_size for f in files))
        print(f"  {name}: {len(prompts):,} samples")
    (dest / "texts.json").write_text(json.dumps(meta, indent=1) + "\n")
    return 0


# ---------------------------------------------------------------------------
# One source


def _write(ids: list[str], text: list[str], out: Path) -> None:
    """The scan input, in FILES files: corpus-assay's workers each take one."""
    out.mkdir(parents=True)
    step = -(-len(ids) // FILES)
    for k in range(0, len(ids), step):
        pq.write_table(pa.table({"id": pa.array(ids[k:k + step], pa.string()),
                                 "text": pa.array(text[k:k + step], pa.string())}),
                       out / f"d_{k // step:04d}.parquet")


def _cache_docs(path: Path, out: Path) -> dict[str, str]:
    t = pq.read_table(path)
    rows = t["row"].to_pylist()
    ids = [f"{r}:{part}" for r in rows for part in ("prompt", "answer")]
    text = [x for pair in zip(t["prompt"].to_pylist(), t["answer"].to_pylist()) for x in pair]
    _write(ids, text, out)
    return {i: i.split(":", 1)[1] for i in ids}


def _sample_docs(folder: Path, out: Path) -> dict[str, str]:
    """Every message of a sample, `row:turn`, with `row` as
    question_overlap.trained_questions counts it (`position`)."""
    records = pq.read_table(sorted((folder / "data").glob("*.parquet"))).to_pylist()
    if len(records) != qo.TRAINED_ROWS:
        raise ValueError(f"{folder}: {len(records):,} rows, not {qo.TRAINED_ROWS:,}")
    ids, text, roles = [], [], {}
    for r, rec in enumerate(records):
        for k, m in enumerate(rec["conversations"]):
            ids.append(f"{r}:{k}")
            text.append(m["value"] or "")
            roles[ids[-1]] = ROLES.get(m["from"], m["from"])
    _write(ids, text, out)
    return roles


def _docs(path: Path, split: bool):
    """(doc, text) of a scan input file; with `split`, each record's
    documents, named as corpus-assay names them."""
    for batch in pq.ParquetFile(path).iter_batches(batch_size=50_000):
        yield from _split(batch, split)


def _split(batch, split: bool):
    for doc, text in zip(batch.column("id").to_pylist(), batch.column("text").to_pylist()):
        if not split:
            yield doc, text
            continue
        for k, part in enumerate((text or "").replace(mq.SEPS[1], mq.SEPS[0]).split(mq.SEPS[0])):
            yield f"{doc}-part-{k}", part


_ITEMS = None


def _exact_init(items: pd.DataFrame) -> None:
    global _ITEMS
    _ITEMS = items


def _exact_file(args: tuple[Path, bool]) -> pd.DataFrame:
    return tdo.exact(_ITEMS, _docs(*args))


def exact(in_dir: Path, items: pd.DataFrame, split: bool, workers: int) -> pd.DataFrame:
    """training_data_overlap.exact over every file of `in_dir`, one per worker."""
    files = [(f, split) for f in sorted(in_dir.glob("*.parquet"))]
    with Pool(min(workers, len(files)), initializer=_exact_init, initargs=(items,)) as pool:
        parts = list(pool.imap_unordered(_exact_file, files))
    return pd.concat(parts, ignore_index=True)


def _doc(record: dict, split: bool) -> str:
    return record["doc_id"] if split else tdo._message(record)


def scan(source: str, work: Path, path: Path, workers: int | None = None) -> int:
    kind, name = source.split(":", 1)
    if source not in SOURCES:
        raise SystemExit(f"{source} is not a SOURCES entry")
    workers = workers or os.cpu_count() or 1
    items = pd.read_parquet(work / "benchmark_items.parquet")
    index = work / "index" / "benchmarks.native"
    table = tdo.item_table(index, items)
    tmp = work / "tmp" / source.replace(":", "_")
    shutil.rmtree(tmp, ignore_errors=True)
    split = kind == "mv"
    if kind == "phase2":
        roles = _cache_docs(path, tmp / "in")
    elif kind == "sample":
        roles = _sample_docs(path / qo.TRAINED[name], tmp / "in")
    else:
        file, size = next(s for s in mq.SHARDS if mq.shard_name(s[0]) == name)
        if path.stat().st_size != size:
            raise SystemExit(f"{path} is not {mq.MV[0]}@{mq.MV[1][:8]}:{file}: "
                             f"{size:,} bytes expected")
        n = mq.chunks(path, tmp / "in")
        print(f"  {name}: {n:,} records")
        roles = None
    rule = tdo.scan(tmp / "in", index, tmp / "rule", tdo.MIN_HITS, tdo.MIN_COVERAGE, workers, split)
    hits = tdo.scan(tmp / "in", index, tmp / "items", 1, 0.0, workers, split)
    flagged = {_doc(r, split) for r in rule}
    # A shard keeps the rows from FIFTH of the item's 13-grams up.
    least = (FIFTH * table["item_grams"].clip(lower=1)).to_numpy() if split \
        else np.zeros(len(table))
    grams = pd.DataFrame([(_doc(r, split), it["item_id"], it["hits"], it["longest_run_tokens"])
                          for r in hits for it in r.get("item_hits", [])
                          if it["hits"] >= least[it["item_id"]]],
                         columns=["doc", "item_index", "hits", "longest_run"])
    grams = grams.join(table, on="item_index").drop(columns="item_index")
    grams["flagged"] = grams["doc"].isin(flagged)
    ex = exact(tmp / "in", items, split, workers)
    if split:
        ex = ex[tdo._kinds(ex) != "question only"]
    df = grams.merge(ex.assign(exact=True), on=["doc", "benchmark", "item"], how="outer")
    df["exact"] = df["exact"].fillna(False).astype(bool)
    df["flagged"] = df["flagged"].fillna(df["doc"].isin(flagged)).astype(bool)
    # BoolQ reuses its passages across splits: the question has to be there too.
    boolq = df[(df["benchmark"] == "boolq") & (df["hits"] >= tdo.HALF * df["item_grams"])]
    want = set(boolq["doc"])
    text = {d: t for f in sorted((tmp / "in").glob("*.parquet")) for d, t in _docs(f, split)
            if d in want} if want else {}
    question = items.set_index(["benchmark", "item"])["question"]
    df["boolq_question"] = pd.array([pd.NA] * len(df), dtype="boolean")
    df.loc[boolq.index, "boolq_question"] = [
        tdo._plain(question[("boolq", i)]) in tdo._plain(text[d])
        for d, i in zip(boolq["doc"], boolq["item"])]
    df["role"] = df["doc"].map(roles) if roles is not None else "pretraining"
    out = work / "sources" / f"{source}.parquet"
    out.parent.mkdir(parents=True, exist_ok=True)
    df[COLUMNS].to_parquet(out, index=False)
    (work / "sources" / f"{source}.json").write_text(json.dumps(dict(
        source=source, path=str(path), documents_flagged=len(flagged),
        documents_with_a_hit=len({_doc(r, split) for r in hits}), rows=len(df))) + "\n")
    shutil.rmtree(tmp)
    print(f"  {source}: {len(df):,} rows, {len(flagged):,} documents flagged by the rule")
    return 0


# ---------------------------------------------------------------------------
# All sources


def _where(source: str) -> str:
    return "mv_reasoning" if source.startswith("mv:") else source


def merge(work: Path, out_dir: Path) -> int:
    missing = [s for s in SOURCES if not (work / "sources" / f"{s}.parquet").exists()]
    if missing:
        raise SystemExit(f"run `scan` first for: {' '.join(missing)}")
    parts = []
    for s in SOURCES:
        d = pd.read_parquet(work / "sources" / f"{s}.parquet")
        mv = s.startswith("mv:")
        row = (np.full(len(d), -1) if mv else d["doc"].str.split(":").str[0].astype(int))
        turn = d["doc"] if mv else d["doc"].str.split(":", n=1).str[1]
        parts.append(d.drop(columns="doc").assign(
            where=_where(s), source=s.split(":", 1)[1], row=row, turn=turn))
    df = pd.concat(parts, ignore_index=True)
    df.to_parquet(out_dir / tdo.CLUSTER_SCAN, index=False)
    meta = {s: json.loads((work / "sources" / f"{s}.json").read_text()) for s in SOURCES}
    (out_dir / tdo.CLUSTER_SCAN).with_suffix(".json").write_text(json.dumps(meta, indent=1) + "\n")
    print(f"  wrote {out_dir / tdo.CLUSTER_SCAN}: {len(df):,} rows from {len(SOURCES)} sources")
    return 0


if __name__ == "__main__":
    # Where the package's other dependencies are missing, as on a cluster:
    #   python -m mv_post_train.stats.benchmark_scan texts TOKENIZED_DIR TOKENIZER WORK
    #   python -m mv_post_train.stats.benchmark_scan scan SOURCE FILE WORK [WORKERS]
    import sys
    step, *a = sys.argv[1:]
    if step == "texts":
        sys.exit(texts(Path(a[0]), Path(a[1]), Path(a[2])))
    sys.exit(scan(a[0], Path(a[2]), Path(a[1]), int(a[3]) if len(a) > 3 else None))
