"""How many tokens each Phase 2 reasoning dataset trains on.

    mvpt measure-phase2-tokens --tokenized-dir <folder of the tokenized caches>

The four Phase 2 runs train on the same number of samples (30K) for the same
number of epochs (EPOCHS), but their answers differ in length, so they need
not train on the same amount of text. LLaMA-Factory tokenizes each dataset once
for a run and keeps the result (its `tokenized_path`, a `save_to_disk` folder):
`input_ids` after the chat template and the cut at CUTOFF tokens, and `labels`,
which are LABEL_IGNORE on the prompt. CACHES names the caches of the
MixtureVitae-1.7B 16K SFT-Tulu3 runs. The open-sci 1.7B models share one
tokenizer, so the counts hold for the other open-sci starting points too.

LLaMA-Factory cuts a sample longer than CUTOFF without adding END_OF_TURN
after the cut, so a sample at CUTOFF is trained on the start of its answer
and is never trained to stop (`ended`). `cap_think_end` says how many of
those still reach the end of their reasoning.

One row per dataset:

  dataset          CACHES entry
  samples          rows of the cache
  seq_mean, seq_median, seq_p95
                   tokens per sample (prompt and answer, after the cut)
  at_cutoff        percent of samples at CUTOFF tokens
  ended            percent of samples whose last token is END_OF_TURN, the
                   token the open-sci template puts after an answer, with a
                   label (so the model is trained to stop)
  cap_think_end    percent of the samples at CUTOFF whose supervised tokens
                   hold THINK_END, the close of the reasoning
  sup_mean, sup_median
                   supervised tokens per sample (labels other than LABEL_IGNORE)
  seq_epoch, sup_epoch
                   the totals for one epoch
  sup_run          supervised tokens over the run (EPOCHS epochs)

    mvpt measure-phase2-tokens --ot3-release-dir DIR --tokenizer FILE [--tokenized-dir DIR]

reads the OT3 30K as released (OT3_RELEASE) instead, with the open-sci
tokenizer and chat template, to say what the cut removed and how the
released answers end. One row per domain and side of CUTOFF
(`phase2_ot3_release.csv`):

  domain, over     the release's domain label; whether the templated sample
                   is longer than CUTOFF tokens
  samples          how many
  tokens_median    tokens per templated sample
  removed_median   tokens past CUTOFF (the part the cut removes)
  think_end        percent of the released answers that hold "</think>"
  boxed            percent that hold "boxed{" (a final boxed answer)

With --tokenized-dir it also checks, row by row, that the OT3 cache holds
min(templated length, CUTOFF) tokens.
"""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
import pyarrow as pa
import pyarrow.compute as pc

CUTOFF = 16384
EPOCHS = 5
LABEL_IGNORE = -100
#: <end_of_turn> and "</think>" in the open-sci tokenizer.
END_OF_TURN = 50277
THINK_END = (870, 18959, 31)

#: The OT3 30K the Phase 2 runs trained on, as released (dataset, revision).
OT3_RELEASE = ("marianna13/open-thoughts3-30k", "283c172385569ff111f4ba6b166fa35dda92d70d")

#: LLaMA-Factory's tokenized caches of the four Phase 2 runs from
#: 1.7b-MixtureVitae-300BT-v1-decontaminated-16k-SFT-Tulu3-decontaminated
#: (cutoff_len CUTOFF, template open-sci).
_MV = "1-7b-MixtureVitae-300BT-v1-decontaminated-16k-SFT-Tulu3-decontaminated_tokenized"
CACHES = {
    "OpenThoughts3": f"f679a5c592c8dffb_hf_hub_{_MV}",
    "AceReason": f"AceReason-1-1-SFT-30k-sharegpt-hf_hf_hub_{_MV}",
    "Nemotron-Cascade": f"Nemotron-Cascade-SFT-Stage-1-30k-sharegpt-hf_hf_hub_{_MV}",
    "Superior-Reasoning": f"Superior-Reasoning-SFT-gpt-oss-120b-30k-sharegpt-hf_models_{_MV}",
}


def _holds(seq: np.ndarray, pattern: tuple[int, ...]) -> bool:
    if len(seq) < len(pattern):
        return False
    win = np.lib.stride_tricks.sliding_window_view(seq, len(pattern))
    return bool((win == np.asarray(pattern)).all(axis=1).any())


def lengths(cache: Path) -> pd.DataFrame:
    """seq (tokens), sup (supervised tokens), ended (the last token is a
    supervised END_OF_TURN) and think_end (the supervised tokens hold
    THINK_END) of every sample of one cache."""
    rows = []
    for f in sorted((cache / "train").glob("*.arrow")):
        with pa.memory_map(str(f)) as src:
            t = pa.ipc.open_stream(src).read_all()
        for ids_chunk, lab_chunk in zip(t["input_ids"].chunks, t["labels"].chunks):
            ids_off, ids = ids_chunk.offsets.to_numpy(), ids_chunk.values.to_numpy()
            lab_off, lab = lab_chunk.offsets.to_numpy(), lab_chunk.values.to_numpy()
            for i in range(len(ids_off) - 1):
                x = ids[ids_off[i]:ids_off[i + 1]]
                y = lab[lab_off[i]:lab_off[i + 1]]
                sup = y[y != LABEL_IGNORE]
                rows.append((len(x), len(sup),
                             len(x) > 0 and x[-1] == END_OF_TURN and y[-1] != LABEL_IGNORE,
                             _holds(sup, THINK_END)))
    if not rows:
        raise FileNotFoundError(f"no train/*.arrow in {cache}")
    return pd.DataFrame(rows, columns=["seq", "sup", "ended", "think_end"])


def summary(name: str, df: pd.DataFrame) -> dict:
    return dict(dataset=name, samples=len(df), seq_mean=df.seq.mean(),
                seq_median=df.seq.median(), seq_p95=df.seq.quantile(0.95),
                at_cutoff=100 * (df.seq >= CUTOFF).mean(), ended=100 * df.ended.mean(),
                cap_think_end=100 * df.think_end[df.seq >= CUTOFF].mean(), sup_mean=df.sup.mean(),
                sup_median=df.sup.median(), seq_epoch=int(df.seq.sum()),
                sup_epoch=int(df.sup.sum()), sup_run=int(EPOCHS * df.sup.sum()))


def main(tokenized_dir: Path, out_dir: Path) -> int:
    out = pd.DataFrame([summary(name, lengths(tokenized_dir / cache))
                        for name, cache in CACHES.items()])
    out_dir.mkdir(parents=True, exist_ok=True)
    dest = out_dir / "phase2_tokens.csv"
    out.to_csv(dest, index=False)
    print(out.round(1).to_string(index=False))
    print(f"  wrote {dest}")
    return 0


def templated(question: str, answer: str) -> str:
    """One sample under the open-sci chat template, as LLaMA-Factory trains it."""
    return "<|user|>\n" + question + "\n<|assistant|>\n" + answer + "<end_of_turn>"


def release_lengths(release_dir: Path, tokenizer: Path) -> pd.DataFrame:
    """domain, tokens (templated) and the answer's endings of every row of the
    OT3 release, in its order."""
    import pyarrow.parquet as pq
    from tokenizers import Tokenizer
    tok = Tokenizer.from_file(str(tokenizer))
    files = sorted(release_dir.glob("*.parquet"))
    if not files:
        raise FileNotFoundError(f"no parquet files of {OT3_RELEASE[0]} in {release_dir}")
    rows, texts = [], []
    for r in pq.read_table(files).to_pylist():
        turns = r["conversations"]
        q = next(m["value"] for m in turns if m["from"] == "human")
        a = next(m["value"] for m in turns if m["from"] == "gpt")
        rows.append(dict(domain=r["domain"], think_end="</think>" in a, boxed="boxed{" in a))
        texts.append(templated(q, a))
    n = [len(e.ids) for i in range(0, len(texts), 256)
         for e in tok.encode_batch(texts[i:i + 256], add_special_tokens=False)]
    return pd.DataFrame(rows).assign(tokens=n)


def release_summary(df: pd.DataFrame) -> pd.DataFrame:
    df = df.assign(over=df.tokens > CUTOFF, removed=(df.tokens - CUTOFF).clip(lower=0))
    rows = []
    for key, g in [((d, o), g) for (d, o), g in df.groupby(["domain", "over"])] + \
            [(("all", o), g) for o, g in df.groupby("over")]:
        rows.append(dict(domain=key[0], over=key[1], samples=len(g),
                         tokens_median=g.tokens.median(), removed_median=g.removed.median(),
                         think_end=100 * g.think_end.mean(), boxed=100 * g.boxed.mean()))
    return pd.DataFrame(rows)


def main_release(release_dir: Path, tokenizer: Path, out_dir: Path,
                 tokenized_dir: Path | None = None) -> int:
    df = release_lengths(release_dir, tokenizer)
    if tokenized_dir is not None:
        cache = lengths(tokenized_dir / CACHES["OpenThoughts3"])
        same = np.minimum(df.tokens.to_numpy(), CUTOFF) == cache.seq.to_numpy()
        print(f"  the OT3 cache holds min(templated length, {CUTOFF}) tokens in "
              f"{same.sum():,} of {len(same):,} rows")
        if not same.all():
            raise SystemExit("the release does not match the cache row by row")
    out = release_summary(df)
    out_dir.mkdir(parents=True, exist_ok=True)
    dest = out_dir / "phase2_ot3_release.csv"
    out.to_csv(dest, index=False, float_format="%.2f")
    print(out.round(1).to_string(index=False))
    print(f"  wrote {dest}")
    return 0
