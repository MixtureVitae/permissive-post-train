"""How much of the paper's post-training data overlaps the benchmarks it reports.

    mvpt measure-training-data-overlap

Section 2.4 says that all post-training data and the MixtureVitae pretraining
data are decontaminated against every benchmark the paper reports, with the
13-gram protocol of the MixtureVitae paper. This measures the overlap of the
post-training data that has a public release (DATASETS):

  long_sft            the 16K context extension data (SeaLong, LongAlign and
                      SmolTalk), which every 16K model is trained on
  Tulu3 SFT and DPO   ontocord's decontaminated copies, which Phase 1 trains
                      on, and AI2's releases they were made from. The
                      difference is what the decontamination removed

The benchmarks (BENCHMARKS) are the 22 the paper reports, each on the split
and the items its evaluations ran: the language suite's validation splits
where lm-eval scores those, LiveCodeBench's 369 problems of Protocol B, and
for AIME24, AIME25, AMC23, MATH500, JEEBench and GPQA-Diamond the items of the
Protocol B logs (LOGS_MODEL in logs.zip; GPQA-Diamond's questions are only in
the copy with them, not in input R6). An item is the text the model is shown,
without the answer: the question, and the choices where it has them.

Matching uses corpus-assay (question_overlap.CORPUS_ASSAY) with the rule of the
MixtureVitae decontamination, which corpus-assay keeps as its defaults: a
document is flagged when it shares at least MIN_HITS distinct 13-grams with
the benchmark index and they are at least MIN_COVERAGE of its distinct
13-grams. Each message of a conversation is one document, as in the
decontamination of the Tulu3 data (its scan counts 2.1M documents for 0.94M
conversations), and a conversation counts as flagged when one of its messages
is. Two scans of each dataset:

  rule     MIN_HITS and MIN_COVERAGE: the messages the decontamination removes
  items    one shared 13-gram: every message that shares any with an item,
           with its hits on each item

`overlap.parquet` has one row per message and benchmark item with a hit:

  dataset, row, turn  the message (`row` counts rows across the files in
                      order; `turn` is its place in the conversation, with a
                      prefix `c` or `r` for the chosen and rejected answers of
                      the preference data)
  benchmark, item     the item (BENCHMARKS name, and the item id the
                      evaluation records use)
  hits                distinct 13-grams of the message that the item holds
  item_grams          the item's distinct 13-grams
  longest_run         longest run of tokens shared with the item
  boilerplate_hits    hits on 13-grams that BOILERPLATE_ITEMS or more items hold
  source              the row's source, where the dataset has one
  flagged             the message meets the rule

and `rows.parquet` one row per dataset row: dataset, row, its own id and
source where the dataset has them, and its number of messages.

The 13-gram scans cannot see an item with no 13-gram, and the rule not one
with fewer than MIN_HITS, so an exact pass (`exact`) matches every item's
question verbatim, after a normalization that keeps numbers, operators and
LaTeX, and counts the choices shown near it: `exact.parquet`, one row per
message and item. With `--reasoning-questions` the same scans and pass run
over the reasoning questions question_overlap.py holds
(`reasoning_questions.parquet`). `found` turns all of these into the items
found, by rules fixed before any model's answers are looked at, and
stats/overlap_sensitivity.py recomputes the paper's comparisons without them.
`benchmark_items.parquet` holds the items' texts, and stays local: it has the
GPQA-Diamond questions.
"""
from __future__ import annotations

import json
import os
import shutil
import unicodedata
import urllib.request
import zipfile
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq

from .question_overlap import N

MIN_HITS = 3
MIN_COVERAGE = 0.001
#: An item counts as in a message from this share of its 13-grams up.
HALF = 0.5
BOILERPLATE_ITEMS = 50
SCAN_WORKERS = 8
#: Never in the data, so that corpus-assay does not split a message at
#: `<|endoftext|>`.
NO_SPLIT = "\x00no-split\x00"


@dataclass(frozen=True)
class Dataset:
    name: str
    dataset: str
    revision: str
    kind: str          # "chat": `messages`; "preference": `chosen` and `rejected`
    trained: bool      # a model of the paper trains on it


DATASETS = (
    Dataset("long_sft", "Harsh1729/long_sft", "bef20b8a103939b5d894835c7e8786beda3e7968",
            "chat", True),
    Dataset("tulu3_sft_decontaminated", "ontocord/tulu-3-sft-mixture-decontaminated",
            "c88950cf7919b32e751baf0336c65bdc0363aa8f", "chat", True),
    Dataset("tulu3_dpo_decontaminated",
            "ontocord/llama-3.1-tulu-3-8b-preference-mixture-decontaminated",
            "5546de87d5f053a0c44c0416ae46cab805f774e7", "preference", True),
    Dataset("tulu3_sft", "allenai/tulu-3-sft-mixture",
            "b14afda60f1bbebe55d5d2fa1e4df5042f97f8be", "chat", False),
    Dataset("tulu3_dpo", "allenai/llama-3.1-tulu-3-8b-preference-mixture",
            "78a6f00785946cd24276c5dd075f83a143a3b1e6", "preference", False),
)

#: The Protocol B logs the items of these tasks are read from: task -> (id
#: fields, joined by "/", and text fields, the question first). JEEBench
#: numbers its questions within each exam paper.
LOGS_MODEL = ("logs/open-sci__sft__ot30k_open-sci-ref-v0.02-1.7b-fineweb-edu-1.4t-300B-4096-"
              "longsft_16k-SFT-Tulu3/")
FROM_LOGS = {
    "AIME24": (["id"], ["problem"]),
    "AIME25": (["id"], ["problem"]),
    "AMC23": (["id"], ["question"]),
    "MATH500": (["unique_id"], ["problem"]),
    "JEEBench": (["description", "index"], ["question"]),
    "GPQADiamond": (["Record ID"], ["Question", "Correct Answer", "Incorrect Answer 1",
                                  "Incorrect Answer 2", "Incorrect Answer 3"]),
}


@dataclass(frozen=True)
class Source:
    """A benchmark file on the Hub, the split the evaluation scores."""
    dataset: str
    revision: str
    file: str
    fields: tuple[str, ...]
    id: str | None = None   # the id field; the row number if None


FROM_HUB = {
    "GSM8K": Source("openai/gsm8k", "740312add88f781978c0658806c59bc2815b9866",
                    "main/test-00000-of-00001.parquet", ("question",)),
    "HumanEval": Source("openai/openai_humaneval", "7dce6050a7d6d172f3cc5c32aa97f52fa1a2e544",
                        "openai_humaneval/test-00000-of-00001.parquet", ("prompt",), "task_id"),
    "MBPP": Source("google-research-datasets/mbpp", "4bb6404fdc6cacfda99d4ac4205087b89d32030c",
                   "full/test-00000-of-00001.parquet", ("text", "test_list"), "task_id"),
    "IFEval": Source("google/IFEval", "966cd89545d6b6acfd7638bc708b98261ca58e84",
                     "ifeval_input_data.jsonl", ("prompt",), "key"),
    "mmlu": Source("cais/mmlu", "c30699e8356da336a370243923dbaf21066bb9fe",
                   "all/test-00000-of-00001.parquet", ("question", "choices")),
    "hellaswag": Source("Rowan/hellaswag", "218ec52e09a7e7462a5400043bb9a69a41d06b76",
                        "data/validation-00000-of-00001.parquet", ("ctx", "endings")),
    "commonsense_qa": Source("tau/commonsense_qa", "94630fe30dad47192a8546eb75f094926d47e155",
                             "data/validation-00000-of-00001.parquet", ("question", "choices"), "id"),
    "arc_challenge": Source("allenai/ai2_arc", "210d026faf9955653af8916fad021475a3f00453",
                            "ARC-Challenge/test-00000-of-00001.parquet", ("question", "choices"), "id"),
    "arc_easy": Source("allenai/ai2_arc", "210d026faf9955653af8916fad021475a3f00453",
                       "ARC-Easy/test-00000-of-00001.parquet", ("question", "choices"), "id"),
    "piqa": Source("baber/piqa", "142f6d7367fd9877f0fb3b5734ea6a545f54cdd1",
                   "piqa_validation.parquet", ("goal", "sol1", "sol2")),
    "boolq": Source("aps/super_glue", "3de24cf8022e94f4ee4b9d55a6f539891524d646",
                    "boolq/validation-00000-of-00001.parquet", ("passage", "question")),
    "winogrande": Source("allenai/winogrande", "01e74176c63542e6b0bcb004dcdea22d94fb67b5",
                         "winogrande_xl/validation-00000-of-00001.parquet",
                         ("sentence", "option1", "option2")),
    "openbookqa": Source("allenai/openbookqa", "388097ea7776314e93a529163e0fea805b8a6454",
                         "main/test-00000-of-00001.parquet", ("question_stem", "choices"), "id"),
    "copa": Source("aps/super_glue", "3de24cf8022e94f4ee4b9d55a6f539891524d646",
                   "copa/validation-00000-of-00001.parquet", ("premise", "choice1", "choice2")),
    "lambada_openai": Source("EleutherAI/lambada_openai", "900124bf3b8235c6daf21033af9948b3f07346c4",
                             "default/test/default.parquet", ("text",)),
}
#: Where the item's answer is, for the items that have a short one.
ANSWERS = {"AIME24": "expected_answer", "AIME25": "answer", "AMC23": "answer",
           "MATH500": "answer", "JEEBench": "gold", "GPQADiamond": "Correct Answer"}
#: The question field where it is not the first of the text fields.
QUESTION = {"boolq": "question"}
LCB = ("livecodebench/code_generation_lite", "0fe84c3912ea0c4d4a78037083943e8f0c4dd505",
       [f"test{i}.jsonl" if i > 1 else "test.jsonl" for i in range(1, 6)])
#: The per-question records whose problem ids are the 369 that Protocol B runs.
LCB_RECORDS = "data/workshop/phase1_livecodebenchv5_per_sample.csv"

#: Every benchmark, in the order of the paper's tables.
BENCHMARKS = ("AIME24", "AIME25", "AMC23", "MATH500", "GSM8K", "HumanEval", "LiveCodeBenchv5",
              "MBPP", "GPQADiamond", "JEEBench", "IFEval", "mmlu", "hellaswag",
              "commonsense_qa", "arc_challenge", "arc_easy", "piqa", "boolq", "winogrande",
              "openbookqa", "copa", "lambada_openai")


def _flatten(v) -> list[str]:
    """A field's text pieces, as MixtureVitae's `_concat_fields` takes them."""
    if v is None:
        return []
    if isinstance(v, dict):
        return [str(x) for x in v.get("text", [])]
    if isinstance(v, (list, tuple, np.ndarray)):
        return [str(x) for x in v]
    return [str(v)]


def _text(row: dict, fields) -> str:
    return " ".join(p for f in fields for p in _flatten(row.get(f)) if p).strip()


def _item(benchmark: str, item: str, row: dict, fields, answer=None) -> dict:
    """An item: its text, and the question, choices and answer apart."""
    q = QUESTION.get(benchmark, fields[0])
    return dict(benchmark=benchmark, item=item, text=_text(row, fields), question=_text(row, [q]),
                choices=[p for f in fields if f != q for p in _flatten(row.get(f)) if p],
                answer=None if answer is None else str(answer))


def _hub_file(dataset: str, revision: str, file: str, cache: Path) -> Path:
    from huggingface_hub import hf_hub_download
    return Path(hf_hub_download(dataset, file, repo_type="dataset", revision=revision,
                                local_dir=cache / dataset.replace("/", "__")))


def _rows(path: Path) -> list[dict]:
    if path.suffix == ".jsonl":
        return [json.loads(line) for line in path.read_text().splitlines() if line.strip()]
    return pq.read_table(path).to_pylist()


def _lcb(cache: Path) -> list[dict]:
    """The LiveCodeBench problems Protocol B runs, streamed from the release
    files (4.4 GB, mostly test cases), and kept in `cache`."""
    keep = cache / "livecodebench_problems.parquet"
    if keep.exists():
        return pq.read_table(keep).to_pylist()
    wanted = set(pd.read_csv(LCB_RECORDS, dtype=str)["sample_id"])
    out = []
    for f in LCB[2]:
        u = f"https://huggingface.co/datasets/{LCB[0]}/resolve/{LCB[1]}/{f}"
        with urllib.request.urlopen(u, timeout=600) as resp:
            for line in resp:
                r = json.loads(line)
                if r["question_id"] in wanted:
                    out.append(dict(item=r["question_id"], text=r["question_content"], file=f))
    if {r["item"] for r in out} != wanted:
        raise ValueError(f"LiveCodeBench: found {len(out)} of the {len(wanted)} problems")
    cache.mkdir(parents=True, exist_ok=True)
    pq.write_table(pa.Table.from_pylist(out), keep)
    return out


def benchmark_items(logs: Path, cache: Path) -> pd.DataFrame:
    """One row per benchmark item: benchmark, item, text, and its question,
    choices and answer (where ANSWERS has its field, and GSM8K's after
    `####`)."""
    rows = []
    with zipfile.ZipFile(logs) as z:
        for name in z.namelist():
            if not (name.startswith(LOGS_MODEL) and name.endswith(".json")):
                continue
            if z.getinfo(name).file_size < 1_000_000:   # scores only, no items
                continue
            task, result = next(iter(json.load(z.open(name))["results"].items()))
            if task not in FROM_LOGS:
                continue
            keys, fields = FROM_LOGS[task]
            rows += [_item(task, "/".join(str(e[k]) for k in keys), e, fields,
                           e.get(ANSWERS.get(task, ""), None))
                     for e in result["examples"]]
    for bench, s in FROM_HUB.items():
        for i, r in enumerate(_rows(_hub_file(s.dataset, s.revision, s.file, cache))):
            answer = r["answer"].split("####")[-1].strip() if bench == "GSM8K" else None
            rows.append(_item(bench, str(r[s.id]) if s.id else str(i), r, s.fields, answer))
    rows += [_item("LiveCodeBenchv5", r["item"], r, ["text"]) for r in _lcb(cache)]
    df = pd.DataFrame(rows)
    missing = set(BENCHMARKS) - set(df["benchmark"])
    if missing:
        raise ValueError(f"no items for {sorted(missing)} in {logs}")
    if df.duplicated(["benchmark", "item"]).any():
        raise ValueError("an item id repeats within a benchmark")
    order = {b: i for i, b in enumerate(BENCHMARKS)}
    return df.sort_values("benchmark", key=lambda s: s.map(order), kind="stable",
                          ignore_index=True)


def build_index(items: pd.DataFrame, work: Path, out_path: Path) -> None:
    """A corpus-assay index of the items, one benchmark per BENCHMARKS entry,
    one item per row (in the order of `items`)."""
    from corpus_assay import build_index_hf
    from corpus_assay.benchmark_spec import BenchmarkSpec
    from corpus_assay.config import BuildIndexConfig
    work.mkdir(parents=True, exist_ok=True)
    specs = []
    for bench in BENCHMARKS:
        p = work / f"{bench}.parquet"
        texts = items.loc[items["benchmark"] == bench, "text"].tolist()
        pq.write_table(pa.table({"text": pa.array(texts, pa.string())}), p)
        specs.append(BenchmarkSpec(name=bench, dataset="parquet",
                                   data_files={"test": str(p.resolve())},
                                   splits=["test"], fields=["text"]))
    out_path.parent.mkdir(parents=True, exist_ok=True)
    # The item files are pinned by the revisions above, not by a Hub revision.
    build_index_hf(BuildIndexConfig(out_path=str(out_path), ngram=N, benchmarks=specs,
                                    allow_unpinned=True))


def item_table(index_path: Path, items: pd.DataFrame) -> pd.DataFrame:
    """benchmark, item and item_grams by corpus-assay's item index, which
    numbers the items in the order they were indexed."""
    meta = pd.DataFrame(json.loads(Path(f"{index_path}.items.meta.json").read_text())["items"])
    if meta["benchmark"].tolist() != items["benchmark"].tolist():
        raise ValueError(f"{index_path} does not index the items of benchmark_items.parquet")
    return items[["benchmark", "item"]].assign(item_grams=meta["n_protected_hashes"].to_numpy())


# ---------------------------------------------------------------------------
# Messages


def _turns(record: dict, kind: str) -> list[tuple[str, str]]:
    """(turn, text) of each message of a row."""
    if kind == "chat":
        return [(str(i), m["content"] or "") for i, m in enumerate(record["messages"])]
    chosen, rejected = record["chosen"], record["rejected"]
    shared = 0
    while (shared < min(len(chosen), len(rejected))
           and chosen[shared]["content"] == rejected[shared]["content"]):
        shared += 1
    return ([(str(i), m["content"] or "") for i, m in enumerate(chosen[:shared])]
            + [(f"c{i}", m["content"] or "") for i, m in enumerate(chosen) if i >= shared]
            + [(f"r{i}", m["content"] or "") for i, m in enumerate(rejected) if i >= shared])


def messages(d: Dataset, raw: Path, out: Path) -> pd.DataFrame:
    """Write one scan input file per data file, one row per message: id
    `row:turn`, text. Returns the rows' own `id` and `source`, where the
    dataset has them."""
    shutil.rmtree(out, ignore_errors=True)
    out.mkdir(parents=True)
    rows, row = [], 0
    for f in sorted((raw / "data").glob("*.parquet")):
        pf = pq.ParquetFile(f)
        ids, texts = [], []
        for batch in pf.iter_batches(batch_size=20_000):
            for record in batch.to_pylist():
                turns = _turns(record, d.kind)
                for turn, text in turns:
                    ids.append(f"{row}:{turn}")
                    texts.append(text)
                rows.append((row, record.get("id"), record.get("source"), len(turns)))
                row += 1
        pq.write_table(pa.table({"id": pa.array(ids, pa.string()),
                                 "text": pa.array(texts, pa.string())}), out / f.name)
    return pd.DataFrame(rows, columns=["row", "id", "source", "messages"]).assign(dataset=d.name)


def scan(in_dir: Path, index_path: Path, out_dir: Path, min_hits: int,
         min_coverage: float, workers: int | None = None, split: bool = False) -> list[dict]:
    """corpus-assay's record of every flagged message. With `split`, each
    text is split into documents on corpus-assay's default separator
    (`<|endoftext|>`), as packed pretraining records are."""
    from corpus_assay import do_scan
    from corpus_assay.config import ScanConfig
    from corpus_assay.scanner import OutputLayout, write_run_outputs
    shutil.rmtree(out_dir, ignore_errors=True)
    sep = {} if split else dict(packed_doc_sep=NO_SPLIT, packed_doc_sep_typo=None)
    cfg = ScanConfig(inputs=[str(in_dir)], text_key="text", id_key="id",
                     index_path=str(index_path), out_dir=str(out_dir), ngram=N,
                     min_hits=min_hits, min_coverage=min_coverage, gate_mode="union",
                     max_return_records=None, boilerplate_threshold=BOILERPLATE_ITEMS,
                     workers=workers or min(SCAN_WORKERS, os.cpu_count() or 1), **sep)
    run = do_scan(cfg, print_progress=False)
    write_run_outputs(layout=OutputLayout(Path(cfg.out_dir)), run=run, cfg=cfg)
    if run.failed_files:
        raise RuntimeError(f"corpus-assay could not scan {in_dir}: see {out_dir}")
    return [json.loads(line) for f in sorted((out_dir / "contaminated_docs").glob("*.ndjson"))
            for line in f.read_text().splitlines()]


def _message(record: dict) -> str:
    """A record's message id, without the sub-document number corpus-assay
    adds (always 0 here, since no message is split)."""
    return record["doc_id"].rsplit("-part-", 1)[0]


def overlap(d: Dataset, rule: list[dict], hits: list[dict], items: pd.DataFrame,
            rows: pd.DataFrame) -> pd.DataFrame:
    flagged = {_message(r) for r in rule}
    out = []
    for r in hits:
        row, turn = _message(r).split(":", 1)
        for it in r.get("item_hits", []):
            out.append((d.name, int(row), turn, it["item_id"], it["hits"], it["longest_run_tokens"],
                        it["boilerplate_hits"], _message(r) in flagged))
    df = pd.DataFrame(out, columns=["dataset", "row", "turn", "item_index", "hits", "longest_run",
                                    "boilerplate_hits", "flagged"])
    df = df.join(items, on="item_index").drop(columns="item_index")
    df["source"] = rows.set_index("row")["source"].reindex(df["row"]).to_numpy()
    return df[["dataset", "row", "turn", "source", "benchmark", "item", "hits", "item_grams",
               "longest_run", "boilerplate_hits", "flagged"]]


def main(logs: Path, out_dir: Path, datasets: list[str] | None = None,
         reasoning: bool = False) -> int:
    """Download what is missing, build the index, scan each dataset (both
    13-gram scans and the exact pass). With `reasoning`, scan the reasoning
    questions instead (`reasoning_questions`)."""
    from huggingface_hub import snapshot_download
    out_dir.mkdir(parents=True, exist_ok=True)
    items_path = out_dir / "benchmark_items.parquet"
    if items_path.exists():
        items = pd.read_parquet(items_path)
    else:
        items = benchmark_items(logs, out_dir / "benchmarks")
        items.to_parquet(items_path, index=False)
    print("  " + ", ".join(f"{b} {n}" for b, n in items.groupby("benchmark", sort=False).size().items()))
    index_path = out_dir / "index" / "benchmarks.native"
    if not index_path.exists():
        build_index(items, out_dir / "benchmarks" / "items", index_path)
    if reasoning:
        df = reasoning_questions(Path(out_dir).parent, out_dir, items, index_path)
        df.to_parquet(out_dir / "reasoning_questions.parquet", index=False)
        print(f"  reasoning questions: {df['row'].nunique():,} with a match")
        return 0
    table = item_table(index_path, items)
    all_rows, parts, exacts = [], [], []
    for d in DATASETS:
        if datasets and d.name not in datasets:
            continue
        raw = out_dir / "raw" / d.name
        snapshot_download(d.dataset, repo_type="dataset", revision=d.revision,
                          allow_patterns=["data/*.parquet"], local_dir=raw)
        in_dir = out_dir / "scan" / d.name / "in"
        rows = messages(d, raw, in_dir)
        all_rows.append(rows)
        rule = scan(in_dir, index_path, out_dir / "scan" / d.name / "rule", MIN_HITS, MIN_COVERAGE)
        hits = scan(in_dir, index_path, out_dir / "scan" / d.name / "items", 1, 0.0)
        parts.append(overlap(d, rule, hits, table, rows))
        ex = exact(items, _docs(in_dir))
        ex[["row", "turn"]] = ex["doc"].str.split(":", n=1, expand=True)
        exacts.append(ex.drop(columns="doc").astype({"row": int}).assign(dataset=d.name))
        print(f"  {d.name}: {len(rows):,} rows, {rows['messages'].sum():,} messages, "
              f"{len({_message(r) for r in rule}):,} messages flagged by the rule")
    pd.concat(all_rows, ignore_index=True).to_parquet(out_dir / "rows.parquet", index=False)
    pd.concat(parts, ignore_index=True).to_parquet(out_dir / "overlap.parquet", index=False)
    pd.concat(exacts, ignore_index=True).to_parquet(out_dir / "exact.parquet", index=False)
    return 0


# ---------------------------------------------------------------------------
# Exact matches

#: Left out of the exact pass: their items are long passages, which the
#: 13-gram scans cover (8 BoolQ items and 1 HellaSwag item have fewer than
#: MIN_HITS 13-grams, none has none).
EXACT_SKIP = ("boolq", "hellaswag", "lambada_openai")
#: Questions shorter than this after `_plain` are not matched.
EXACT_MIN_CHARS = 12
#: A choice counts as shown with the question when it is within this many
#: characters before or after it.
EXACT_WINDOW = 1000


def _plain(text: str) -> str:
    """Case, Unicode form and white space only: numbers, operators, LaTeX,
    punctuation and stop words stay."""
    return " ".join(unicodedata.normalize("NFKC", text).lower().split())


def _docs(in_dir: Path):
    """(id, text) of every document of a scan input folder."""
    for f in sorted(in_dir.glob("*.parquet")):
        for batch in pq.ParquetFile(f).iter_batches(batch_size=50_000):
            yield from zip(batch.column("id").to_pylist(), batch.column("text").to_pylist())


def exact(items: pd.DataFrame, docs) -> pd.DataFrame:
    """Every document that holds an item's question verbatim after `_plain`:
    doc, benchmark, item, question_chars, choices (how many the item has)
    and choices_found (the most of them within EXACT_WINDOW of one
    occurrence of the question)."""
    import ahocorasick
    by: dict[str, list[int]] = {}
    for i, (b, q) in enumerate(zip(items["benchmark"], items["question"])):
        p = _plain(q)
        if b not in EXACT_SKIP and len(p) >= EXACT_MIN_CHARS:
            by.setdefault(p, []).append(i)
    auto = ahocorasick.Automaton()
    for p, idx in by.items():
        auto.add_word(p, (p, idx))
    auto.make_automaton()
    choices = [[_plain(c) for c in cs] for cs in items["choices"]]
    out = []
    for doc, text in docs:
        t = _plain(text or "")
        best: dict[int, int] = {}
        for end, (p, idx) in auto.iter(t):
            near = t[max(0, end + 1 - len(p) - EXACT_WINDOW):end + 1 + EXACT_WINDOW]
            for i in idx:
                best[i] = max(best.get(i, 0), sum(c in near for c in choices[i]))
        out += [(doc, i, len(_plain(items["question"].iat[i])), len(choices[i]), n)
                for i, n in best.items()]
    df = pd.DataFrame(out, columns=["doc", "item_index", "question_chars", "choices",
                                    "choices_found"])
    return (df.join(items[["benchmark", "item"]], on="item_index").drop(columns="item_index")
            [["doc", "benchmark", "item", "question_chars", "choices", "choices_found"]])


def roles(d: Dataset, raw: Path, wanted: pd.DataFrame) -> pd.Series:
    """The role of each (row, turn) of `wanted`: the message's own role, or
    `chosen` and `rejected` for the answers of the preference data."""
    need = {int(r) for r in wanted["row"]}
    column = "messages" if d.kind == "chat" else "chosen"
    found, start = {}, 0
    for f in sorted((raw / "data").glob("*.parquet")):
        n = pq.ParquetFile(f).metadata.num_rows
        local = sorted(r - start for r in need if start <= r < start + n)
        if local:
            t = pq.read_table(f, columns=[column]).take(local)
            found.update(zip((start + i for i in local), t[column].to_pylist()))
        start += n
    def role(row, turn):
        if turn[0] in "cr":
            return {"c": "chosen", "r": "rejected"}[turn[0]]
        return found[int(row)][int(turn)]["role"]
    return pd.Series([role(r, t) for r, t in zip(wanted["row"], wanted["turn"])],
                     index=wanted.index)


# ---------------------------------------------------------------------------
# Reasoning questions

#: The questions of the reasoning data that question_overlap.py already holds:
#: every distinct question of OpenThoughts3-1.2M, the release MixtureVitae's
#: `open_thoughts` source comes from and the Phase 2 30K are drawn from
#: (`in_phase2`), and its samples of the public releases of AceReason,
#: Nemotron-Cascade and Superior-Reasoning (not the 30K samples Phase 2 trains
#: on). Questions only: the answers are not in those files.
OT3_QUESTIONS = "ot3_questions.parquet"
RELEASE_SAMPLES = "samples"


def reasoning_questions(overlap_dir: Path, out_dir: Path, items: pd.DataFrame,
                        index_path: Path) -> pd.DataFrame:
    """Scan the reasoning questions like a dataset: one row per question and
    benchmark item with a 13-gram (`hits`, `item_grams`, `flagged`) or an
    exact match (`exact`, `choices_found`), with the question's corpus,
    subset, domain and `in_phase2`."""
    ot3 = pd.read_parquet(overlap_dir / OT3_QUESTIONS)
    ot3 = pd.DataFrame(dict(corpus="OpenThoughts3", subset="1.2M", domain=ot3["domain"],
                            in_phase2=ot3["in_phase2"], question=ot3["question"]))
    samples = pd.concat([pd.read_parquet(f) for f in sorted((overlap_dir / RELEASE_SAMPLES)
                                                            .glob("*.parquet"))], ignore_index=True)
    samples = samples[["corpus", "subset", "domain", "question"]].assign(in_phase2=False)
    qs = pd.concat([ot3, samples], ignore_index=True)
    in_dir = out_dir / "scan" / "reasoning_questions" / "in"
    shutil.rmtree(in_dir, ignore_errors=True)
    in_dir.mkdir(parents=True)
    pq.write_table(pa.table({"id": pa.array([f"{i}:0" for i in range(len(qs))], pa.string()),
                             "text": pa.array(qs["question"].tolist(), pa.string())}),
                   in_dir / "questions.parquet")
    work = out_dir / "scan" / "reasoning_questions"
    rule = scan(in_dir, index_path, work / "rule", MIN_HITS, MIN_COVERAGE)
    hits = scan(in_dir, index_path, work / "items", 1, 0.0)
    table = item_table(index_path, items)
    d = Dataset("reasoning_questions", "", "", "chat", False)
    grams = overlap(d, rule, hits, table, pd.DataFrame(dict(row=[], source=[])))
    ex = exact(items, _docs(in_dir))
    ex["row"] = ex["doc"].str.split(":").str[0].astype(int)
    df = grams.merge(ex[["row", "benchmark", "item", "question_chars", "choices",
                         "choices_found"]].assign(exact=True),
                     on=["row", "benchmark", "item"], how="outer")
    df["exact"] = df["exact"].fillna(False).astype(bool)
    info = qs.drop(columns="question")
    return df[["row", "benchmark", "item", "hits", "item_grams", "flagged", "exact",
               "question_chars", "choices", "choices_found"]].join(info, on="row")


# ---------------------------------------------------------------------------
# Found items

#: The kinds of evidence `found` gives that the sensitivity analyses exclude.
EXCLUDED = ("13-gram", "exact", "exact with choices", "question without choices")
#: A multiple-choice question found without its choices names the item from
#: this many characters up (after `_plain`). Shorter ones are stock stems
#: ("GDP measures", "Which of the following is false?").
LONG_QUESTION = 60


#: What stats/benchmark_scan.py scans on a cluster: the Phase 2 30K samples
#: as trained and as written, and MixtureVitae's reasoning shards.
CLUSTER_SCAN = "benchmark_scan.parquet"


def _texts(in_dir: Path, ids: set[str]) -> dict[str, str]:
    return {i: t for i, t in _docs(in_dir) if i in ids}


def _kinds(e: pd.DataFrame) -> np.ndarray:
    """The evidence of each exact match."""
    return np.select([e["choices"] == 0, e["choices_found"] == e["choices"],
                      e["question_chars"] >= LONG_QUESTION],
                     ["exact", "exact with choices", "question without choices"], "question only")


def found(out_dir: Path) -> pd.DataFrame:
    """Every (training text, benchmark item) where the item is found, by
    rules fixed before any model's answers are looked at:

      13-gram             the text holds HALF of the item's 13-grams; for
                          BoolQ also its question (after `_plain`), since
                          BoolQ reuses passages across its splits
      exact               an item without choices whose question the text
                          holds verbatim (`exact`)
      exact with choices  the text holds the question and every choice
                          within EXACT_WINDOW of it
      question without choices
                          the text holds a multiple-choice question of
                          LONG_QUESTION characters or more, not all choices
      question only       the same for a shorter question: reported, not in
                          EXCLUDED

    One row per text, item and evidence: where (a DATASETS name, or the
    corpus of a reasoning question), row, turn, role, source, benchmark,
    item, evidence, in_phase2."""
    items = pd.read_parquet(out_dir / "benchmark_items.parquet")
    questions = items.set_index(["benchmark", "item"])["question"]
    grams = pd.read_parquet(out_dir / "overlap.parquet")
    grams = grams[(grams["hits"] >= HALF * grams["item_grams"]) & (grams["item_grams"] > 0)]
    ex = pd.read_parquet(out_dir / "exact.parquet")
    rows = pd.read_parquet(out_dir / "rows.parquet")
    parts = []
    for d in (d for d in DATASETS if d.trained):
        g = grams[grams["dataset"] == d.name]
        boolq = g[g["benchmark"] == "boolq"]
        texts = _texts(out_dir / "scan" / d.name / "in",
                       set(boolq["row"].astype(str) + ":" + boolq["turn"]))
        keep = [b != "boolq" or _plain(questions[(b, i)]) in _plain(texts[f"{r}:{t}"])
                for b, i, r, t in zip(g["benchmark"], g["item"], g["row"], g["turn"])]
        e = ex[ex["dataset"] == d.name]
        kinds = _kinds(e)
        both = pd.concat([g[keep][["row", "turn", "benchmark", "item"]].assign(evidence="13-gram"),
                          e[["row", "turn", "benchmark", "item"]].assign(evidence=kinds)],
                         ignore_index=True)
        own = rows[rows["dataset"] == d.name].set_index("row")
        both["source"] = own["source"].reindex(both["row"]).to_numpy()
        both["role"] = roles(d, out_dir / "raw" / d.name, both)
        parts.append(both.assign(where=d.name, in_phase2=False))
    rq = pd.read_parquet(out_dir / "reasoning_questions.parquet")
    hit = (rq["hits"] >= HALF * rq["item_grams"]) & (rq["item_grams"] > 0)
    kinds = _kinds(rq.assign(choices=rq["choices"].fillna(0),
                             question_chars=rq["question_chars"].fillna(0)))
    for mask, kind in ((hit, None), (rq["exact"], kinds)):
        r = rq[mask]
        parts.append(pd.DataFrame(dict(
            where=r["corpus"], row=r["row"], turn="0", benchmark=r["benchmark"], item=r["item"],
            evidence="13-gram" if kind is None else kind[mask.to_numpy()], source=r["subset"],
            role="question", in_phase2=r["in_phase2"])))
    if (out_dir / CLUSTER_SCAN).exists():
        parts.append(cluster_found(pd.read_parquet(out_dir / CLUSTER_SCAN)))
    df = pd.concat(parts, ignore_index=True)
    return df.drop_duplicates(ignore_index=True)[
        ["where", "row", "turn", "role", "source", "benchmark", "item", "evidence", "in_phase2"]]


def cluster_found(s: pd.DataFrame) -> pd.DataFrame:
    """`found`'s rows for benchmark_scan.parquet, by the same rules."""
    hit = ((s["hits"] >= HALF * s["item_grams"]) & (s["item_grams"] > 0)
           & ((s["benchmark"] != "boolq") | s["boolq_question"].fillna(False).astype(bool)))
    kinds = _kinds(s.assign(choices=s["choices"].fillna(0),
                            question_chars=s["question_chars"].fillna(0)))
    parts = []
    for mask, kind in ((hit.to_numpy(), None), (s["exact"].to_numpy(), kinds)):
        r = s[mask]
        parts.append(pd.DataFrame(dict(
            where=r["where"], row=r["row"], turn=r["turn"], role=r["role"], source=r["source"],
            benchmark=r["benchmark"], item=r["item"],
            evidence="13-gram" if kind is None else kind[mask],
            in_phase2=r["where"] != "mv_reasoning")))
    return pd.concat(parts, ignore_index=True)


#: Where the gold answer of a multiple-choice item is: benchmark -> (field of
#: the gold label, how the label names a choice).
GOLD = {"commonsense_qa": "answerKey", "arc_challenge": "answerKey", "arc_easy": "answerKey",
        "openbookqa": "answerKey", "mmlu": "answer", "copa": "label", "piqa": "label",
        "winogrande": "answer"}


def gold(items: pd.DataFrame, cache: Path) -> pd.Series:
    """The gold answer's text by (benchmark, item): the correct choice, or
    the answer of the items in ANSWERS and GSM8K."""
    out = items.set_index(["benchmark", "item"])["answer"].dropna().to_dict()
    for bench, key in GOLD.items():
        s = FROM_HUB[bench]
        own = items.loc[items["benchmark"] == bench]
        for item, choices, r in zip(own["item"], own["choices"],
                                    _rows(_hub_file(s.dataset, s.revision, s.file, cache))):
            label = r[key]
            if isinstance(r.get("choices"), dict):          # letters, as the choices' labels
                at = list(r["choices"]["label"]).index(label)
            elif bench == "winogrande":
                at = int(label) - 1
            else:
                at = int(label)
            out[(bench, item)] = list(choices)[at]
    return pd.Series(out)


def answer_shown(found_items: pd.DataFrame, out_dir: Path) -> pd.Series:
    """Whether a response turn of the found text's row (an assistant turn,
    or a chosen or rejected answer) holds the item's gold answer after
    `_plain`. A heuristic: a short answer ("7", "bank") can be there by
    chance. False for the reasoning questions, whose answers are not here."""
    items = pd.read_parquet(out_dir / "benchmark_items.parquet")
    answers = gold(items, out_dir / "benchmarks")
    out = pd.Series(False, index=found_items.index)
    for d in (d for d in DATASETS if d.trained):
        f = found_items[found_items["where"] == d.name]
        if f.empty:
            continue
        wanted = {str(r) for r in f["row"]}
        turns = {i: t for i, t in _docs(out_dir / "scan" / d.name / "in")
                 if i.split(":", 1)[0] in wanted}
        keys = pd.DataFrame([dict(row=int(i.split(":", 1)[0]), turn=i.split(":", 1)[1])
                             for i in turns])
        keys["role"] = roles(d, out_dir / "raw" / d.name, keys).to_numpy()
        replies: dict[int, str] = {}
        for r, t, role in zip(keys["row"], keys["turn"], keys["role"]):
            if role in ("assistant", "chosen", "rejected"):
                replies[r] = replies.get(r, "") + " " + _plain(turns[f"{r}:{t}"])
        for i, b, it, r in zip(f.index, f["benchmark"], f["item"], f["row"]):
            a = answers.get((b, it))
            out[i] = a is not None and _plain(a) in replies.get(int(r), "")
    return out


#: Epochs over each dataset, from the checkpoints' `trainer_state.json` on
#: Hugging Face (`num_train_epochs`; the MV 16K extension, SFT and DPO
#: checkpoints, 2026-10-31). Every model of a stage uses the same recipe.
EPOCHS = {"long_sft": 1, "tulu3_sft_decontaminated": 2, "tulu3_dpo_decontaminated": 1}


def exposure(found_items: pd.DataFrame) -> pd.DataFrame:
    """How often training showed each found item, nominally: one row per item
    (EXCLUDED evidence, the trained datasets), with the rows holding it in each
    dataset (`rows_<dataset>`) and those of them whose responses hold the gold
    answer's text (`answer_rows_<dataset>`, `answer_shown`'s heuristic: not
    that the row answers this question correctly); `exposures`, the rows times
    the epochs summed over the datasets, and `answer_exposures`, the same for
    the answer rows. Rows times epochs, not checked against the tokens a run
    kept. The preference data's prompts are context, not targets, of the DPO
    loss; the SFT loss covers the prompts too (Phase 1 does not mask them)."""
    f = found_items[found_items["evidence"].isin(EXCLUDED) & found_items["where"].isin(EPOCHS)]
    rows = (f.groupby(["benchmark", "item", "where", "row"])["answer_shown"].max()
            .reset_index())
    out = rows.pivot_table(index=["benchmark", "item"], columns="where", values="row",
                           aggfunc="nunique", fill_value=0)
    ans = rows[rows["answer_shown"]].pivot_table(index=["benchmark", "item"], columns="where",
                                                 values="row", aggfunc="nunique", fill_value=0)
    epochs = pd.Series(EPOCHS)
    df = pd.DataFrame(index=out.index)
    for d in EPOCHS:
        df[f"rows_{d}"] = out.get(d, 0)
    for d in EPOCHS:
        df[f"answer_rows_{d}"] = (ans.get(d, pd.Series(dtype=int)).reindex(out.index, fill_value=0)
                                  .astype(int))
    df["exposures"] = (out.reindex(columns=list(EPOCHS), fill_value=0) * epochs).sum(axis=1)
    df["answer_exposures"] = (ans.reindex(index=out.index, columns=list(EPOCHS), fill_value=0)
                              * epochs).sum(axis=1)
    return df.reset_index()


def excluded(found_items: pd.DataFrame, where=None) -> dict[str, set[str]]:
    """The items to leave out, by benchmark: every item `found` with an
    EXCLUDED kind of evidence in any of `where` (default: all)."""
    f = found_items[found_items["evidence"].isin(EXCLUDED)]
    if where is not None:
        f = f[f["where"].isin(where)]
    return {b: set(g["item"]) for b, g in f.groupby("benchmark")}


# ---------------------------------------------------------------------------
# Rates


def rates(df: pd.DataFrame, items: pd.DataFrame) -> pd.DataFrame:
    """Per dataset and benchmark: rows with a message the rule flags and that
    shares a 13-gram with the benchmark, and the benchmark's items found in a
    message (from HALF of their 13-grams), as a count and a percent."""
    n_items = items.groupby("benchmark").size()
    df = df.assign(share=df["hits"] / df["item_grams"].where(df["item_grams"] > 0))
    out = []
    for (d, b), g in df.groupby(["dataset", "benchmark"], sort=False):
        found = g.loc[g["share"] >= HALF, "item"].nunique()
        out.append(dict(dataset=d, benchmark=b,
                        rows_flagged=g.loc[g["flagged"], "row"].nunique(),
                        rows_with_item=g.loc[g["share"] >= HALF, "row"].nunique(),
                        items_found=found, items=int(n_items[b]),
                        items_found_pct=100 * found / n_items[b]))
    return pd.DataFrame(out)
