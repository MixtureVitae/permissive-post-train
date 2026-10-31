"""How many questions of the other reasoning corpora are also OpenThoughts3 questions.

    mvpt measure-question-overlap

The paper calls AceReason, Nemotron-Cascade and Superior-Reasoning "not named
sources" of any pretraining mixture. A corpus that is not a named source can
still hold the same questions, and the three draw on some of the public pools
OpenThoughts3 draws on (OpenCodeReasoning, for one). This measures, for a
random sample of each corpus's questions, how much of the question's text is
also in a question of OpenThoughts3-1.2M (REFERENCE), the release that
MixtureVitae's `open_thoughts` source comes from, and whether that question is
one of the 30K that Phase 2 trains on (PHASE2_OT3).

The 30K samples of the three corpora that Phase 2 trained on are not public,
so each corpus is sampled from its public release, one SUBSETS entry at a time
(a release's category or training stage), with a fixed seed:

  jsonl releases    CLUSTERS byte offsets, uniform over the subset's files,
                    and the first ROWS_PER_CLUSTER complete lines after each
  AceReason         ACE_SHARDS of its Arrow shards, and ROWS_PER_CLUSTER
                    consecutive rows at ACE_CLUSTERS random places in each

A cluster is the unit that the intervals of `rates` resample.

Matching uses corpus-assay (CORPUS_ASSAY, https://github.com/mrcabbage972/corpus-assay),
the 13-gram overlap detector that grew out of the MixtureVitae decontamination,
with its bundled normalization (NFKC, lower case, word tokens, 40 stop words
dropped, multiple-choice scaffolding rejected). The distinct OT3 questions are
its "benchmark", one item per question, and the sampled questions are the
corpus it scans, in item-gate mode so that each question's hits are counted
against one OT3 question at a time. Before the scan, the N-grams found in more
than BOILERPLATE of one corpus's own questions (and in at least
BOILERPLATE_MIN) are taken as instructions it adds to many questions ("Write
Python code to solve the problem...", "Let's think step by step...") or
templates of its sources. The tokens they cover are cut out, and the
stretches left are scanned as separate sub-documents, so that no N-gram
straddles a question and an instruction. Each sampled
question gets one row:

  corpus, subset          SUBSETS entry
  dataset, revision, file where it comes from
  cluster, position       its cluster, and its place in it (byte offset of the
                          line, or row of the shard)
  domain                  the record's own label (math, code, science, ...)
  n_tokens, n_grams       its normalized tokens, and its distinct N-grams
                          after cutting
  n_shared                how many of those are N-grams of some OT3 question
  best_shared             how many the one OT3 question holding most of them
                          holds (corpus-assay's item hits)
  best_boilerplate        how many of those are in BOILERPLATE_ITEMS or more
                          OT3 questions
  best_longest_run        the longest run of tokens shared with that question
  best_qid, best_source   that question (an item and row of the reference
                          file) and its OT3 source
  best_in_phase2          that question is one of the Phase 2 30K
  whole_match             a question with no N-gram left equals an OT3 question
                          after normalization (null for the others)
  question, best_question both texts, cut to TEXT_CHARS

The reference file (`ot3_questions.parquet`) has one row per distinct OT3
question: qid, source, domain, n_rows (rows of OT3 that ask it),
in_phase2, question.
"""
from __future__ import annotations

import hashlib
import io
import json
import shutil
from collections import defaultdict
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd
import pyarrow as pa
import pyarrow.compute as pc
import pyarrow.parquet as pq

from ..manifest.http import get

#: The release the matching was run with; its normalization config is recorded
#: in the index and scan manifests.
CORPUS_ASSAY = "0.1.0"

N = 13
#: Share of a corpus's own questions above which an N-gram is boilerplate.
BOILERPLATE = 0.01
#: And the fewest questions a boilerplate N-gram is in, whatever the share.
BOILERPLATE_MIN = 3
#: corpus-assay's `boilerplate_threshold`: an item hit on an N-gram that this
#: many OT3 questions hold counts as boilerplate (best_boilerplate).
BOILERPLATE_ITEMS = 50
SCAN_WORKERS = 4
CLUSTERS = 200
ROWS_PER_CLUSTER = 10
ACE_SHARDS = 20
ACE_CLUSTERS = 10
SEED = 20261025
TEXT_CHARS = 2000
#: Bytes read at a sampled offset, doubled until ROWS_PER_CLUSTER lines are
#: complete or READ_MAX is reached.
READ_FIRST = 2**19
READ_MAX = 2**25
WORKERS = 8

HF = "https://huggingface.co/datasets"
REFERENCE = ("open-thoughts/OpenThoughts3-1.2M", "61bcf9d4eb38b30295efc2021227a63cc5bb34c8",
             [f"data/train-{i:05d}-of-00120.parquet" for i in range(120)])
PHASE2_OT3 = ("marianna13/open-thoughts3-30k", "283c172385569ff111f4ba6b166fa35dda92d70d",
              [f"data/train-{i:05d}-of-00003.parquet" for i in range(3)])

NC1 = ("nvidia/Nemotron-Cascade-SFT-Stage-1", "e59a356b107737562f064ec252e6adacbc737899")
NC2 = ("nvidia/Nemotron-Cascade-SFT-Stage-2", "54cfa23ba28c9d94aeae15142e73bde6f8d14d8b")
SR = ("Alibaba-Apsara/Superior-Reasoning-SFT-gpt-oss-120b", "21b55a649f05ac110bea5a64a9c064a5100ff554")
ACE = ("nvidia/AceReason-1.1-SFT", "5ac692742de9f2481f8274d022dc78d5fc96c249", 187)


@dataclass(frozen=True)
class Subset:
    corpus: str
    subset: str
    dataset: str
    revision: str
    files: tuple[str, ...]


SUBSETS = (
    Subset("AceReason", "all", ACE[0], ACE[1],
           tuple(f"sft_data.parquet/data-{i:05d}-of-{ACE[2]:05d}.arrow" for i in range(ACE[2]))),
    Subset("Nemotron-Cascade", "stage-1 math", *NC1, tuple(f"math/math_{i}.jsonl" for i in range(1, 7))),
    Subset("Nemotron-Cascade", "stage-1 code", *NC1, tuple(f"code/code_{i}.jsonl" for i in range(1, 5))),
    Subset("Nemotron-Cascade", "stage-1 science", *NC1, ("science.jsonl",)),
    Subset("Nemotron-Cascade", "stage-1 general", *NC1, ("general.jsonl",)),
    Subset("Nemotron-Cascade", "stage-2 math", *NC2, tuple(f"math/math_{i}.jsonl" for i in range(1, 9))),
    Subset("Nemotron-Cascade", "stage-2 code", *NC2, tuple(f"code/code_{i}.jsonl" for i in range(1, 15))),
    Subset("Nemotron-Cascade", "stage-2 science", *NC2, ("science.jsonl",)),
    Subset("Nemotron-Cascade", "stage-2 general", *NC2,
           tuple(f"general/general_{i}.jsonl" for i in range(1, 5))),
    Subset("Superior-Reasoning", "stage 1", *SR,
           ("Superior-Reasoning-SFT-gpt-oss-120b-stage1-train-data.jsonl",)),
    Subset("Superior-Reasoning", "stage 2", *SR,
           ("Superior-Reasoning-SFT-gpt-oss-120b-stage2-train-data.jsonl",)),
)

SEP = "<|endoftext|>"


def url(dataset: str, revision: str, file: str) -> str:
    return f"{HF}/{dataset}/resolve/{revision}/{file}"


def tokens(text: str) -> list[str]:
    """corpus-assay's normalized tokens."""
    from corpus_assay._native import normalize_text
    return normalize_text(text)


def _hashes(toks: list[str]) -> list[tuple[int, int]]:
    """(hash, start) of every allowed N-gram, as corpus-assay forms them."""
    from corpus_assay.text_utils import iter_hashed_ngrams_with_positions
    return list(iter_hashed_ngrams_with_positions(toks, N))


def whole(toks: list[str]) -> int:
    """A hash of the whole token sequence."""
    return int.from_bytes(hashlib.blake2b(" ".join(toks).encode(), digest_size=8).digest(), "little")


def boilerplate(gram_sets: list[set[int]], share: float = BOILERPLATE) -> set[int]:
    """The N-grams in more than `share` of the questions, and in at least BOILERPLATE_MIN."""
    counts: dict[int, int] = defaultdict(int)
    for g in gram_sets:
        for h in g:
            counts[h] += 1
    return {h for h, c in counts.items()
            if c > share * len(gram_sets) and c >= BOILERPLATE_MIN}


def stretches(toks: list[str], drop: set[int]) -> list[list[str]]:
    """The runs of at least N tokens that no N-gram of `drop` covers."""
    covered = np.zeros(len(toks), dtype=bool)
    for h, pos in _hashes(toks):
        if h in drop:
            covered[pos:pos + N] = True
    edges = np.flatnonzero(np.diff(np.r_[False, ~covered, False].astype(int)))
    return [toks[a:b] for a, b in zip(edges[::2], edges[1::2]) if b - a >= N]


def prepare(questions: list[str]) -> pd.DataFrame:
    """What corpus-assay scans for one subset's questions: `text`, the
    stretches left after cutting the subset's boilerplate, joined by SEP, and
    `n_grams`, their distinct allowed N-grams."""
    toks = [tokens(q) for q in questions]
    distinct = {}
    for q, t in zip(questions, toks):
        distinct.setdefault(q, t)
    drop = boilerplate([{h for h, _ in _hashes(t)} for t in distinct.values()])
    rows = []
    for t in toks:
        parts = stretches(t, drop)
        grams = {h for part in parts for h, _ in _hashes(part)}
        rows.append(dict(text=SEP.join(" ".join(p) for p in parts), n_tokens=len(t),
                         n_grams=len(grams), whole=whole(t)))
    return pd.DataFrame(rows)


def build_index(ref_path: Path, out_path: Path) -> None:
    """A corpus-assay index of the reference questions, one item per row."""
    from corpus_assay import build_index_hf
    from corpus_assay.benchmark_spec import BenchmarkSpec
    from corpus_assay.config import BuildIndexConfig
    spec = BenchmarkSpec(name="OpenThoughts3", dataset="parquet",
                         data_files={"train": str(ref_path.resolve())},
                         splits=["train"], fields=["question"])
    # The reference file is pinned by its SHA-256, not by a Hub revision.
    build_index_hf(BuildIndexConfig(out_path=str(out_path), ngram=N, benchmarks=[spec],
                                    allow_unpinned=True))


def scan(texts: pd.DataFrame, index_path: Path, work_dir: Path) -> dict[str, list[dict]]:
    """corpus-assay's records of every sub-document with a hit, by question id."""
    from corpus_assay import do_scan
    from corpus_assay.config import ScanConfig
    from corpus_assay.scanner import OutputLayout, write_run_outputs
    shutil.rmtree(work_dir, ignore_errors=True)
    (work_dir / "in").mkdir(parents=True)
    # corpus-assay reads plain `string` columns only, not `large_string`.
    pq.write_table(pa.table({"id": pa.array(texts["id"].astype(str).tolist(), pa.string()),
                             "text": pa.array(texts["text"].tolist(), pa.string())}),
                   work_dir / "in" / "questions.parquet")
    cfg = ScanConfig(inputs=[str(work_dir / "in")], text_key="text", id_key="id",
                     index_path=str(index_path), out_dir=str(work_dir / "out"), ngram=N,
                     min_hits=1, min_coverage=0.0, gate_mode="item", max_return_records=None,
                     boilerplate_threshold=BOILERPLATE_ITEMS, workers=SCAN_WORKERS)
    run = do_scan(cfg, print_progress=False)
    write_run_outputs(layout=OutputLayout(Path(cfg.out_dir)), run=run, cfg=cfg)
    if run.failed_files:
        raise RuntimeError(f"corpus-assay could not scan {work_dir / 'in'}: see {cfg.out_dir}")
    records: dict[str, list[dict]] = defaultdict(list)
    for f in sorted((work_dir / "out" / "contaminated_docs").glob("*.ndjson")):
        for line in f.read_text().splitlines():
            r = json.loads(line)
            records[r["doc_id"].rsplit("-part-", 1)[0]].append(r)
    return records


def best_item(parts: list[dict]) -> dict:
    """Sum each OT3 question's hits over a question's sub-documents and keep
    the one with most (the lowest id on a tie)."""
    items: dict[int, list[int]] = defaultdict(lambda: [0, 0, 0])
    for r in parts:
        for it in r.get("item_hits", []):
            e = items[it["item_id"]]
            e[0] += it["hits"]
            e[1] += it["boilerplate_hits"]
            e[2] = max(e[2], it["longest_run_tokens"])
    n_shared = sum(r["match_count"] for r in parts)
    if not items:
        return dict(n_shared=n_shared, best_shared=0, best_boilerplate=0,
                    best_longest_run=0, best_qid=-1)
    qid, (hits, bp, run) = min(items.items(), key=lambda kv: (-kv[1][0], kv[0]))
    return dict(n_shared=n_shared, best_shared=hits, best_boilerplate=bp,
                best_longest_run=run, best_qid=qid)


# ---------------------------------------------------------------------------
# Reading the corpora


def _question(record: dict) -> str:
    """The first user turn of a record, in any of the formats here."""
    if "conversations" in record:
        return next(t["value"] for t in record["conversations"] if t["from"] in ("human", "user"))
    if "messages" in record:
        return next(t["content"] for t in record["messages"] if t["role"] == "user")
    return record["input"]


def _domain(record: dict) -> str | None:
    """The record's own domain label (`category` or `domain`)."""
    return record.get("category") or record.get("domain")


def reference(raw_dir: Path, out: Path, phase2_dir: Path) -> pd.DataFrame:
    """The distinct OT3 questions, from the downloaded REFERENCE shards."""
    rows: dict[str, list] = {}
    for f in REFERENCE[2]:
        t = pq.read_table(raw_dir / Path(f).name, columns=["source", "domain", "conversations"])
        first = pc.list_element(t["conversations"], 0)
        if not pc.all(pc.equal(pc.struct_field(first, "from"), "human")).as_py():
            raise ValueError(f"{f}: a conversation does not start with the question")
        questions = pc.struct_field(first, "value").to_pylist()
        for src, dom, q in zip(t["source"].to_pylist(), t["domain"].to_pylist(), questions):
            r = rows.get(q)
            if r is None:
                rows[q] = [src, dom, 1]
            else:
                r[2] += 1
    phase2 = set()
    for f in PHASE2_OT3[2]:
        t = pq.read_table(phase2_dir / Path(f).name, columns=["conversations"])
        phase2.update(whole(tokens(_question({"conversations": c})))
                      for c in t["conversations"].to_pylist())
    df = pd.DataFrame([(q, *r) for q, r in rows.items()],
                      columns=["question", "source", "domain", "n_rows"])
    df.insert(0, "qid", np.arange(len(df), dtype=np.int64))
    df["in_phase2"] = [whole(tokens(q)) in phase2 for q in df["question"]]
    out.parent.mkdir(parents=True, exist_ok=True)
    df.to_parquet(out, index=False)
    found = len(phase2 & {whole(tokens(q)) for q in df["question"]})
    print(f"  {out}: {len(df):,} distinct questions from {df.n_rows.sum():,} rows; "
          f"{found:,} of the {len(phase2):,} distinct Phase 2 questions are among them")
    return df


def _sizes(dataset: str, revision: str) -> dict[str, int]:
    body, _ = get(f"https://huggingface.co/api/datasets/{dataset}/tree/{revision}?recursive=true")
    return {f["path"]: f["size"] for f in json.loads(body) if f["type"] == "file"}


def _lines_at(u: str, offset: int, size: int) -> list[tuple[int, dict]]:
    """The first ROWS_PER_CLUSTER complete lines that start after `offset`."""
    n = READ_FIRST
    while True:
        end = min(size, offset + n) - 1
        body, _ = get(u, headers={"Range": f"bytes={offset}-{end}"})
        start = body.find(b"\n") + 1 if offset > 0 else 0
        out, pos = [], start
        while len(out) < ROWS_PER_CLUSTER:
            nl = body.find(b"\n", pos)
            if nl < 0:
                break
            out.append((offset + pos, json.loads(body[pos:nl])))
            pos = nl + 1
        if len(out) == ROWS_PER_CLUSTER or end == size - 1 or n >= READ_MAX:
            if end == size - 1 and pos < len(body) and len(out) < ROWS_PER_CLUSTER:
                out.append((offset + pos, json.loads(body[pos:])))
            return out
        n *= 2


def sample_jsonl(s: Subset, rng: np.random.Generator) -> list[dict]:
    sizes = _sizes(s.dataset, s.revision)
    files = list(s.files)
    cum = np.cumsum([sizes[f] for f in files])
    offsets = np.sort(rng.integers(0, cum[-1], CLUSTERS))
    jobs = []
    for c, o in enumerate(offsets):
        i = int(np.searchsorted(cum, o, "right"))
        jobs.append((c, files[i], int(o - (cum[i - 1] if i else 0)), sizes[files[i]]))

    def one(job):
        c, f, o, size = job
        return [dict(cluster=c, file=f, position=p, domain=_domain(r), question=_question(r))
                for p, r in _lines_at(url(s.dataset, s.revision, f), o, size)]

    with ThreadPoolExecutor(WORKERS) as ex:
        return [r for rows in ex.map(one, jobs) for r in rows]


def sample_arrow(s: Subset, rng: np.random.Generator) -> list[dict]:
    """ACE_CLUSTERS runs of ROWS_PER_CLUSTER rows in each of ACE_SHARDS shards,
    read as an Arrow stream (the shards are written by `datasets`)."""
    shards = sorted(rng.choice(len(s.files), ACE_SHARDS, replace=False).tolist())

    def one(k: int, shard: int) -> list[dict]:
        f = s.files[shard]
        body, _ = get(url(s.dataset, s.revision, f), timeout=900.0)
        t = pa.ipc.open_stream(io.BytesIO(body)).read_all().select(["input", "category"])
        r = np.random.default_rng([SEED, shard])
        starts = np.sort(r.choice(t.num_rows - ROWS_PER_CLUSTER, ACE_CLUSTERS, replace=False))
        out = []
        for j, st in enumerate(starts):
            for p in range(st, st + ROWS_PER_CLUSTER):
                out.append(dict(cluster=k * ACE_CLUSTERS + j, file=f, position=int(p),
                                domain=t["category"][int(p)].as_py(),
                                question=t["input"][int(p)].as_py()))
        return out

    with ThreadPoolExecutor(4) as ex:
        return [r for rows in ex.map(lambda a: one(*a), enumerate(shards)) for r in rows]


def sample(s: Subset) -> pd.DataFrame:
    rng = np.random.default_rng([SEED, *s.subset.encode(), *s.dataset.encode()])
    rows = sample_arrow(s, rng) if s.files[0].endswith(".arrow") else sample_jsonl(s, rng)
    df = pd.DataFrame(rows)
    df.insert(0, "corpus", s.corpus)
    df.insert(1, "subset", s.subset)
    df.insert(2, "dataset", s.dataset)
    df.insert(3, "revision", s.revision)
    return df


# ---------------------------------------------------------------------------
# Matching


def match(samples: pd.DataFrame, ref: pd.DataFrame, index_path: Path,
          work_dir: Path) -> pd.DataFrame:
    wholes: dict[int, int] = {}
    for qid, q in zip(ref["qid"], ref["question"]):
        wholes.setdefault(whole(tokens(q)), int(qid))
    out = []
    for (corpus, subset), part in samples.groupby(["corpus", "subset"], sort=False):
        prep = prepare(part["question"].tolist())
        prep.insert(0, "id", np.arange(len(prep)))
        records = scan(prep[prep["text"] != ""], index_path, work_dir)
        for (_, row), (_, p) in zip(part.iterrows(), prep.iterrows()):
            m = best_item(records.get(str(p["id"]), []))
            whole_match = None
            if p["n_grams"] == 0:
                m["best_qid"] = wholes.get(p["whole"], -1)
                whole_match = m["best_qid"] >= 0
            out.append({**row.to_dict(), "n_tokens": p["n_tokens"], "n_grams": p["n_grams"],
                        **m, "whole_match": whole_match})
        print(f"  matched {corpus} {subset}")
    df = pd.DataFrame(out)
    best = ref.set_index("qid").reindex(df["best_qid"])
    df["best_source"] = best["source"].to_numpy()
    df["best_in_phase2"] = best["in_phase2"].fillna(False).astype(bool).to_numpy()
    df["best_question"] = best["question"].str.slice(0, TEXT_CHARS).to_numpy()
    df["question"] = df["question"].str.slice(0, TEXT_CHARS)
    return df


def main(raw_dir: Path, out_dir: Path, subsets: list[str] | None = None) -> int:
    """Build the reference from the OT3 shards in `raw_dir`, sample each
    subset (samples are kept in `out_dir` and reused), and match them."""
    out_dir.mkdir(parents=True, exist_ok=True)
    ref_path = out_dir / "ot3_questions.parquet"
    if ref_path.exists():
        ref = pd.read_parquet(ref_path)
    else:
        phase2_dir = out_dir / "phase2_ot3"
        phase2_dir.mkdir(exist_ok=True)
        for f in PHASE2_OT3[2]:
            dest = phase2_dir / Path(f).name
            if not dest.exists():
                body, _ = get(url(*PHASE2_OT3[:2], f), timeout=900.0)
                dest.write_bytes(body)
        ref = reference(raw_dir, ref_path, phase2_dir)
    parts = []
    for s in SUBSETS:
        if subsets and s.subset not in subsets and s.corpus not in subsets:
            continue
        p = out_dir / "samples" / f"{s.corpus}_{s.subset.replace(' ', '_')}.parquet"
        if not p.exists():
            p.parent.mkdir(exist_ok=True)
            sample(s).to_parquet(p, index=False)
            print(f"  sampled {s.corpus} {s.subset}")
        parts.append(pd.read_parquet(p))
    index_path = out_dir / "index" / "ot3.native"
    if not index_path.exists():
        build_index(ref_path, index_path)
    df = match(pd.concat(parts, ignore_index=True), ref, index_path, out_dir / "scan")
    dest = out_dir / "question_overlap.parquet"
    df.to_parquet(dest, index=False)
    print(f"  wrote {dest}: {len(df):,} sampled questions")
    return 0


# ---------------------------------------------------------------------------
# Rates

#: Overlap levels: the share of a question's N-grams that one OT3 question holds.
#: "any" is the decontamination rule, one N-gram shared with any OT3 question,
#: which common phrasing of programming problems ("the first line contains")
#: also meets. From "half" up the pairs are the same problem; "fifth" also
#: takes the same problem written with different notation, and variants of
#: one problem with other numbers, so it bounds the rate from above.
LEVELS = {"any": None, "fifth": 0.2, "half": 0.5, "most": 0.9}
RESAMPLES = 2000


def levels(df: pd.DataFrame) -> pd.DataFrame:
    """Whether each sampled question reaches each level. A question with no
    N-gram counts at every level if it equals an OT3 question."""
    whole_hit = df["whole_match"].fillna(False).astype(bool)
    share = df["best_shared"] / df["n_grams"].where(df["n_grams"] > 0)
    out = pd.DataFrame(index=df.index)
    for name, cut in LEVELS.items():
        hit = df["n_shared"] > 0 if cut is None else share >= cut
        out[name] = hit.fillna(False) | whole_hit
    out["half_phase2"] = out["half"] & df["best_in_phase2"]
    return out


def rates(df: pd.DataFrame, by=("corpus", "subset")) -> pd.DataFrame:
    """Percent of sampled questions at each level, with 95% intervals that
    resample clusters."""
    hits = levels(df)
    rows = []
    for key, part in df.groupby(list(by), sort=False):
        h = hits.loc[part.index]
        clusters = part["cluster"].to_numpy()
        ids, inv = np.unique(clusters, return_inverse=True)
        sums = np.zeros((len(ids), h.shape[1]))
        np.add.at(sums, inv, h.to_numpy(float))
        size = np.bincount(inv).astype(float)
        rng = np.random.default_rng(SEED)
        draw = rng.integers(0, len(ids), (RESAMPLES, len(ids)))
        boot = sums[draw].sum(axis=1) / size[draw].sum(axis=1)[:, None] * 100
        lo, hi = np.percentile(boot, [2.5, 97.5], axis=0)
        row = dict(zip(by, key if isinstance(key, tuple) else (key,)))
        row.update(questions=len(part), clusters=len(ids),
                   no_grams=int((part["n_grams"] == 0).sum()))
        for j, c in enumerate(h.columns):
            row[c] = h[c].mean() * 100
            row[f"{c}_lo"], row[f"{c}_hi"] = lo[j], hi[j]
        rows.append(row)
    return pd.DataFrame(rows)
