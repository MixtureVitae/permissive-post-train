"""How many questions of the 30K reasoning samples are in MixtureVitae's other reasoning data.

    mvpt measure-mv-question-overlap prepare --trained-dir DIR
    mvpt measure-mv-question-overlap shard NAME --shard-file FILE   (once per SHARDS entry)
    mvpt measure-mv-question-overlap merge

question_overlap.py matches the questions of the 30K samples of AceReason,
Nemotron-Cascade and Superior-Reasoning that Phase 2 trained on against
OpenThoughts3, the source of MixtureVitae's `open_thoughts` data. This matches
them against the rest of MixtureVitae's reasoning data: the 37 shards (SHARDS)
of `ontocord/MixtureVitae-v1-decontaminated` (MV, at a pinned revision) that
make up the reasoning subset of M1 other than `open_thoughts` (formatted_text,
math and synthetic_instruct; 183 GB compressed).

  prepare  the questions (question_overlap.trained_questions), cut as
           question_overlap.prepare cuts them (each corpus and domain's
           boilerplate out), and a corpus-assay index with one item per
           stretch left (`items.parquet`, `questions.parquet`, `index/`)
  shard    one shard, scanned by corpus-assay twice, with the settings of
           training_data_overlap.py: every document with one 13-gram shared
           with an item (`hits`), and the documents the MixtureVitae
           decontamination rule flags (MIN_HITS, MIN_COVERAGE). Then the
           exact pass for the questions with no 13-gram left (`exact`), and
           per question its best document: `shards/NAME/per_question.parquet`
           and `exact.parquet`. A shard takes 2 to 30 minutes on 32 cores;
           on a cluster each can be one array task
  merge    every shard's results into one row per question
           (`mv_question_overlap.parquet`) and the rates by corpus and domain
           (`mv_question_overlap_rates.csv`)

A document is one example of a shard split on `<|endoftext|>`, corpus-assay's
default and how the MixtureVitae decontamination split them (MV packs several
examples into one record). A question's share is the most of its 13-grams that
one document holds, summed over the question's stretches. The levels are those
of question_overlap.LEVELS, and a question is **found** in MV, as in
training_data_overlap.py, when one document holds at least half of its 13-grams
(HALF), or, for a question with no 13-gram left after the cut, its text
verbatim after `_plain` (at least EXACT_MIN_CHARS characters).
"""
from __future__ import annotations

import gzip
import json
import os
import shutil
from multiprocessing import Pool
from pathlib import Path

import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq

from . import question_overlap as qo
from .training_data_overlap import BOILERPLATE_ITEMS, EXACT_MIN_CHARS, HALF, MIN_COVERAGE, \
    MIN_HITS, _plain

MV = ("ontocord/MixtureVitae-v1-decontaminated", "0b412b40468bb1806290a6e8ea2db45073de7192")
#: (file in MV, bytes). The Hub keeps each file's SHA-256 at MV's revision.
SHARDS = (
    ("data/formatted_text/formatted_text-0.jsonl.gz", 8666050878),
    ("data/formatted_text/formatted_text-1.jsonl.gz", 6863351478),
    ("data/math/math-0.jsonl.gz", 2605100356),
    ("data/math/math_word_problems-0.jsonl.gz", 338173479),
    ("data/synthetic_instruct/dart-math.jsonl.gz", 681749207),
    ("data/synthetic_instruct/few-shot-0.jsonl.gz", 5786994241),
    ("data/synthetic_instruct/few-shot-1.jsonl.gz", 5719669154),
    ("data/synthetic_instruct/few-shot-2.jsonl.gz", 4698819808),
    ("data/synthetic_instruct/glaiveai-reasoning-0.jsonl.gz", 6004205375),
    ("data/synthetic_instruct/glaiveai-reasoning-1.jsonl.gz", 6000298801),
    ("data/synthetic_instruct/glaiveai-reasoning-2.jsonl.gz", 6006919872),
    ("data/synthetic_instruct/glaiveai-reasoning-3.jsonl.gz", 6003377179),
    ("data/synthetic_instruct/glaiveai-reasoning-4.jsonl.gz", 5483470243),
    ("data/synthetic_instruct/glaiveai-reasoning-5.jsonl.gz", 5998193847),
    ("data/synthetic_instruct/glaiveai-reasoning-6.jsonl.gz", 6000998800),
    ("data/synthetic_instruct/glaiveai-reasoning-7.jsonl.gz", 6001759622),
    ("data/synthetic_instruct/glaiveai-reasoning-8.jsonl.gz", 5275300149),
    ("data/synthetic_instruct/ling-coder-dpo-0.jsonl.gz", 326023988),
    ("data/synthetic_instruct/ling-coder-sft-0.jsonl.gz", 4572666265),
    ("data/synthetic_instruct/ling-coder-sft-1.jsonl.gz", 830306574),
    ("data/synthetic_instruct/ling-coder-syntheticqa-0.jsonl.gz", 6779244258),
    ("data/synthetic_instruct/ling-coder-syntheticqa-1.jsonl.gz", 6778533572),
    ("data/synthetic_instruct/ling-coder-syntheticqa-2.jsonl.gz", 6777208829),
    ("data/synthetic_instruct/ling-coder-syntheticqa-3.jsonl.gz", 2754747682),
    ("data/synthetic_instruct/misc_instruct.jsonl.gz", 6360496617),
    ("data/synthetic_instruct/nemo_science_math-0.jsonl.gz", 5371270452),
    ("data/synthetic_instruct/nemo_science_math-1.jsonl.gz", 13141685758),
    ("data/synthetic_instruct/nemo_science_math-2.jsonl.gz", 1589442026),
    ("data/synthetic_instruct/os-q2-0.jsonl.gz", 5751824427),
    ("data/synthetic_instruct/os-q2-1.jsonl.gz", 5587766129),
    ("data/synthetic_instruct/os-q2-2.jsonl.gz", 5633462487),
    ("data/synthetic_instruct/os-q2-3.jsonl.gz", 3941713487),
    ("data/synthetic_instruct/prism_math.jsonl.gz", 4211244691),
    ("data/synthetic_instruct/pyedu_reasoning.jsonl.gz", 2302966203),
    ("data/synthetic_instruct/ring-lite-sft-0.jsonl.gz", 5566953616),
    ("data/synthetic_instruct/ring-lite-sft-1.jsonl.gz", 5688788809),
    ("data/synthetic_instruct/ring-lite-sft-2.jsonl.gz", 1284026687),
)
SEPS = ("<|endoftext|>", "<|endoftext}>")
#: Text per chunk file a shard is cut into: the unit corpus-assay's workers take.
CHUNK_BYTES = 400 << 20
MOST = 0.9


def shard_name(file: str) -> str:
    return Path(file).name.removesuffix(".jsonl.gz")


# ---------------------------------------------------------------------------
# Questions and index


def items(questions: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    """questions (qid, corpus, subset, position, n_tokens, n_grams) and items
    (item, qid, question: one stretch), each corpus and subset cut on its own
    boilerplate as question_overlap.prepare cuts it."""
    qs, its = [], []
    for (corpus, subset), part in questions.groupby(["corpus", "subset"], sort=False):
        toks = [qo.tokens(q) for q in part["question"]]
        distinct = {}
        for q, t in zip(part["question"], toks):
            distinct.setdefault(q, t)
        drop = qo.boilerplate([{h for h, _ in qo._hashes(t)} for t in distinct.values()])
        for pos, t in zip(part["position"], toks):
            parts = qo.stretches(t, drop)
            qid = len(qs)
            qs.append(dict(qid=qid, corpus=corpus, subset=subset, position=pos, n_tokens=len(t),
                           n_grams=len({h for p in parts for h, _ in qo._hashes(p)})))
            its += [dict(item=len(its) + k, qid=qid, question=" ".join(p))
                    for k, p in enumerate(parts)]
    return pd.DataFrame(qs), pd.DataFrame(its)


def ot3_questions(release_dir: Path) -> pd.DataFrame:
    """One row per record of the OT3 30K that Phase 2 trained on, as released
    (phase2_tokens.OT3_RELEASE): corpus, subset (its domain label), position
    (its row) and question."""
    from .phase2_tokens import OT3_RELEASE
    files = sorted(release_dir.glob("*.parquet"))
    if not files:
        raise FileNotFoundError(f"no parquet files of {OT3_RELEASE[0]} in {release_dir}")
    rows = pq.read_table(files, columns=["domain", "conversations"]).to_pylist()
    if len(rows) != qo.TRAINED_ROWS:
        raise ValueError(f"{release_dir}: {len(rows):,} rows, not {qo.TRAINED_ROWS:,}")
    return pd.DataFrame([dict(corpus="OpenThoughts3", subset=r["domain"] or "none", position=i,
                              question=next(m["value"] for m in r["conversations"]
                                            if m["from"] == "human"))
                         for i, r in enumerate(rows)])


def prepare(trained_dir: Path | None, out_dir: Path, ot3_release_dir: Path | None = None) -> int:
    """The questions of the three TRAINED samples, or with `ot3_release_dir`
    those of the OT3 30K, cut and indexed (each in its own `out_dir`)."""
    from corpus_assay import build_index_hf
    from corpus_assay.benchmark_spec import BenchmarkSpec
    from corpus_assay.config import BuildIndexConfig
    t = ot3_questions(ot3_release_dir) if ot3_release_dir else qo.trained_questions(trained_dir)
    qs, its = items(t)
    out_dir.mkdir(parents=True, exist_ok=True)
    qs.to_parquet(out_dir / "questions.parquet", index=False)
    its.to_parquet(out_dir / "items.parquet", index=False)
    # The exact pass needs the full texts of the questions with no 13-gram.
    (qs[qs["n_grams"] == 0].merge(t[["corpus", "position", "question"]], on=["corpus", "position"])
     [["qid", "question"]].to_parquet(out_dir / "no_grams.parquet", index=False))
    spec = BenchmarkSpec(name="Phase2-30K", dataset="parquet",
                         data_files={"train": str((out_dir / "items.parquet").resolve())},
                         splits=["train"], fields=["question"])
    # The items are pinned by the trained samples, not by a Hub revision.
    build_index_hf(BuildIndexConfig(out_path=str(out_dir / "index" / "phase2.native"), ngram=qo.N,
                                    benchmarks=[spec], allow_unpinned=True))
    print(f"  wrote {out_dir}: {len(qs):,} questions, {len(its):,} items, "
          f"{int((qs['n_grams'] == 0).sum()):,} with no 13-gram")
    return 0


# ---------------------------------------------------------------------------
# One shard


def chunks(src: Path, out: Path) -> int:
    """The shard as Parquet files of `id` and `text` (UTF-8 strings), what
    corpus-assay makes of a JSONL file itself. It reads JSONL with pyarrow's
    1 MB block, which fails on the longer lines some shards have."""
    out.mkdir(parents=True, exist_ok=True)
    ids, texts, size, n, k = [], [], 0, 0, 0

    def flush():
        nonlocal ids, texts, size, k
        pq.write_table(pa.table({"text": pa.array(texts, pa.string()),
                                 "id": pa.array(ids, pa.string())}), out / f"c_{k:04d}.parquet")
        ids, texts, size, k = [], [], 0, k + 1

    with gzip.open(src, "rt", encoding="utf-8") as f:
        for line in f:
            r = json.loads(line)
            ids.append(str(r["id"]))
            texts.append(str(r["text"]))
            size += len(line)
            n += 1
            if size >= CHUNK_BYTES:
                flush()
    if ids:
        flush()
    return n


def scan(in_dir: Path, index_path: Path, out_dir: Path, min_hits: int, min_coverage: float,
         workers: int) -> list[dict]:
    """corpus-assay's record of every document it flags (documents split on
    its default separator)."""
    from corpus_assay import do_scan
    from corpus_assay.config import ScanConfig
    from corpus_assay.scanner import OutputLayout, write_run_outputs
    shutil.rmtree(out_dir, ignore_errors=True)
    cfg = ScanConfig(inputs=[str(in_dir)], text_key="text", id_key="id",
                     index_path=str(index_path), out_dir=str(out_dir), ngram=qo.N,
                     min_hits=min_hits, min_coverage=min_coverage, gate_mode="union",
                     max_return_records=None, boilerplate_threshold=BOILERPLATE_ITEMS,
                     workers=workers)
    run = do_scan(cfg, print_progress=False)
    write_run_outputs(layout=OutputLayout(Path(cfg.out_dir)), run=run, cfg=cfg)
    if run.failed_files:
        errors = sorted({s.error for s in run.per_file if s.error})
        raise RuntimeError(f"corpus-assay could not scan {run.failed_files} files of {in_dir}: "
                           f"{errors[:3]}")
    return [json.loads(line) for f in sorted((out_dir / "contaminated_docs").glob("*.ndjson"))
            for line in f.read_text().splitlines()]


_AUTO = None


def _exact_init(patterns: dict[str, list[int]]) -> None:
    global _AUTO
    import ahocorasick
    _AUTO = ahocorasick.Automaton()
    for p, qids in patterns.items():
        _AUTO.add_word(p, qids)
    _AUTO.make_automaton()


def _exact_file(path: Path) -> dict[int, list]:
    found: dict[int, list] = {}
    for batch in pq.ParquetFile(path).iter_batches(batch_size=50_000):
        for doc, text in zip(batch.column("id").to_pylist(), batch.column("text").to_pylist()):
            for k, part in enumerate((text or "").replace(SEPS[1], SEPS[0]).split(SEPS[0])):
                hit = {q for _, qids in _AUTO.iter(_plain(part)) for q in qids}
                for q in hit:
                    e = found.setdefault(q, [0, []])
                    e[0] += 1
                    if len(e[1]) < 3:
                        e[1].append(f"{doc}-part-{k}")
    return found


def exact(in_dir: Path, no_grams: pd.DataFrame, workers: int) -> pd.DataFrame:
    """The questions with no 13-gram that a document holds verbatim after
    `_plain`: qid, exact_docs and up to three of the documents."""
    patterns: dict[str, list[int]] = {}
    for qid, text in zip(no_grams["qid"], no_grams["question"]):
        p = _plain(text)
        if len(p) >= EXACT_MIN_CHARS:
            patterns.setdefault(p, []).append(int(qid))
    total: dict[int, list] = {}
    with Pool(workers, initializer=_exact_init, initargs=(patterns,)) as pool:
        for found in pool.imap_unordered(_exact_file, sorted(in_dir.glob("*.parquet"))):
            for q, (n, docs) in found.items():
                e = total.setdefault(q, [0, []])
                e[0] += n
                e[1] = (e[1] + docs)[:3]
    return pd.DataFrame([(q, n, ";".join(d)) for q, (n, d) in total.items()],
                        columns=["qid", "exact_docs", "exact_examples"])


def per_question(hits: list[dict], rule: list[dict], item_qid, q_grams: pd.Series) -> pd.DataFrame:
    """Per question with a hit: its best document (most hits, summed over the
    question's items; on a tie, one the rule flags), q_grams, whether the rule flags that document, and how
    many documents hold half and MOST of its 13-grams."""
    flagged = {r["doc_id"] for r in rule}
    best, nhalf, nmost = {}, {}, {}
    for r in hits:
        per: dict[int, int] = {}
        for it in r.get("item_hits", []):
            q = int(item_qid[int(it["item_id"])])
            per[q] = per.get(q, 0) + it["hits"]
        for q, h in per.items():
            s = h / q_grams[q]
            if s >= HALF:
                nhalf[q] = nhalf.get(q, 0) + 1
            if s >= MOST:
                nmost[q] = nmost.get(q, 0) + 1
            fl = r["doc_id"] in flagged
            b = best.get(q, (0, "", False))
            if h > b[0] or (h == b[0] and fl and not b[2]):   # on a tie, a flagged document
                best[q] = (h, r["doc_id"], fl)
    rows = [(q, h, int(q_grams[q]), doc, fl, nhalf.get(q, 0), nmost.get(q, 0))
            for q, (h, doc, fl) in best.items()]
    return pd.DataFrame(rows, columns=["qid", "best_hits", "q_grams", "best_doc", "best_flagged",
                                       "docs_half", "docs_most"])


def shard(name: str, shard_file: Path, out_dir: Path, workers: int | None = None) -> int:
    file, size = next((s for s in SHARDS if shard_name(s[0]) == name), (None, 0))
    if file is None:
        raise SystemExit(f"{name} is not a SHARDS entry")
    if shard_file.stat().st_size != size:
        raise SystemExit(f"{shard_file} is not {MV[0]}@{MV[1][:8]}:{file}: {size:,} bytes expected")
    workers = workers or os.cpu_count() or 1
    index = out_dir / "index" / "phase2.native"
    its = pd.read_parquet(out_dir / "items.parquet")
    meta = json.loads(Path(f"{index}.items.meta.json").read_text())["items"]
    item_qid = its["qid"].to_numpy()
    q_grams = pd.Series([m["n_protected_hashes"] for m in meta]).groupby(item_qid).sum()
    work = out_dir / "shards" / name
    shutil.rmtree(work, ignore_errors=True)
    n = chunks(shard_file, work / "in")
    hits = scan(work / "in", index, work / "hits", 1, 0.0, workers)
    rule = scan(work / "in", index, work / "rule", MIN_HITS, MIN_COVERAGE, workers)
    ex = exact(work / "in", pd.read_parquet(out_dir / "no_grams.parquet"), workers)
    pq_ = per_question(hits, rule, item_qid, q_grams)
    pq_.to_parquet(work / "per_question.parquet", index=False)
    ex.to_parquet(work / "exact.parquet", index=False)
    shutil.rmtree(work / "in")
    print(f"  {name}: {n:,} records, {len(hits):,} documents with a hit, {len(rule):,} flagged, "
          f"{len(pq_):,} questions with a hit, {len(ex):,} exact")
    return 0


# ---------------------------------------------------------------------------
# All shards


def merge_frames(questions: pd.DataFrame, per_shard: dict[str, tuple[pd.DataFrame, pd.DataFrame]]
                 ) -> pd.DataFrame:
    """One row per question: share (best document over all shards), best_doc,
    best_shard, best_flagged (the rule flags that document), docs_half and
    docs_most (over all shards),
    exact_docs and exact_shards, and the levels."""
    pq_ = pd.concat([p.assign(shard=s) for s, (p, _) in per_shard.items()])
    pq_["share"] = pq_["best_hits"] / pq_["q_grams"]
    # On a tie, a document the rule flags, then the first shard.
    best = (pq_.sort_values(["share", "best_flagged", "shard"], ascending=[False, False, True])
            .drop_duplicates("qid").set_index("qid"))
    counts = pq_.groupby("qid")[["docs_half", "docs_most"]].sum()
    ex = pd.concat([e.assign(shard=s) for s, (_, e) in per_shard.items()])
    ex = ex.groupby("qid").agg(exact_docs=("exact_docs", "sum"),
                               exact_shards=("shard", lambda s: ";".join(sorted(s))))
    d = (questions.set_index("qid")
         .join(best[["share", "best_doc", "best_flagged", "shard"]]
               .rename(columns={"shard": "best_shard"}))
         .join(counts).join(ex).reset_index())
    d = d.fillna({"share": 0.0, "best_flagged": False, "docs_half": 0, "docs_most": 0,
                  "exact_docs": 0})
    for c in ("docs_half", "docs_most", "exact_docs"):
        d[c] = d[c].astype(int)
    for name, cut in qo.LEVELS.items():
        if cut is not None:
            d[name] = d["share"] >= cut
    d["flagged"] = d["best_flagged"].astype(bool) & d["half"]
    d["exact"] = (d["n_grams"] == 0) & (d["exact_docs"] > 0)
    d["found"] = d["half"] | d["exact"]
    return d


def rates(d: pd.DataFrame, by=("corpus", "subset")) -> pd.DataFrame:
    """Percent of questions at each level (of those with a 13-gram), found
    (of all) and, where the OT3 match is joined, found in MV or OT3."""
    g = d[d["n_grams"] > 0]
    lv = [c for c in ("fifth", "half", "most", "flagged") if c in d]
    out = g.groupby(list(by))[lv].mean().mul(100)
    out.insert(0, "with_grams", g.groupby(list(by)).size())
    out.insert(0, "questions", d.groupby(list(by)).size())
    out["no_grams_exact"] = d.groupby(list(by))["exact"].sum()
    out["found"] = d.groupby(list(by))["found"].mean() * 100
    if "found_or_ot3" in d:
        out["ot3_half"] = d.groupby(list(by))["ot3_half"].mean() * 100
        out["found_or_ot3"] = d.groupby(list(by))["found_or_ot3"].mean() * 100
    return out.reset_index()


def merge(out_dir: Path, ot3_dir: Path | None = None) -> int:
    missing = [shard_name(f) for f, _ in SHARDS
               if not (out_dir / "shards" / shard_name(f) / "per_question.parquet").exists()]
    if missing:
        raise SystemExit(f"run `shard` first for: {' '.join(missing)}")
    per_shard = {shard_name(f): (pd.read_parquet(out_dir / "shards" / shard_name(f) / "per_question.parquet"),
                                 pd.read_parquet(out_dir / "shards" / shard_name(f) / "exact.parquet"))
                 for f, _ in SHARDS}
    d = merge_frames(pd.read_parquet(out_dir / "questions.parquet"), per_shard)
    trained = (ot3_dir or Path()) / "question_overlap_trained.parquet"
    if ot3_dir is not None and trained.exists() and "OpenThoughts3" not in set(d["corpus"]):
        t = pd.read_parquet(trained)
        t = t[["corpus", "position"]].assign(ot3_half=qo.levels(t)["half"].to_numpy())
        d = d.merge(t, on=["corpus", "position"], how="left", validate="one_to_one")
        d["found_or_ot3"] = d["found"] | d["ot3_half"]
    d.to_parquet(out_dir / "mv_question_overlap.parquet", index=False)
    r = pd.concat([rates(d), rates(d, by=("corpus",)).assign(subset="all")])
    r.to_csv(out_dir / "mv_question_overlap_rates.csv", index=False, float_format="%.2f")
    with pd.option_context("display.width", 200, "display.max_columns", 20):
        print(r.round(2).to_string(index=False))
    print(f"  wrote {out_dir / 'mv_question_overlap.parquet'} and the rates")
    return 0
