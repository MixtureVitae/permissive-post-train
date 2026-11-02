"""Score probe texts with one checkpoint: for each probe, the log-probability
of its target after its context, that of each of its choices, and a greedy
continuation of its context.

    python score_texts.py PROBES.parquet MODEL_DIR OUT.parquet

PROBES has the columns id, context, target (may be empty), choices (a list,
may be empty) and generate (new tokens to decode greedily, 0 for none).
Context and continuation are tokenized together, as in training; the tokens
past the context's own tokenization are the continuation's. OUT.json beside
the output records the checkpoint, the versions and two checks: texts scored
one at a time against their batches, and greedy continuations decoded one at
a time against theirs."""
from __future__ import annotations

import json
import sys
import time
from pathlib import Path

import pandas as pd
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

MAX_TOKENS = 2048        # a longer text loses the start of its context
BATCH_TOKENS = 16_384    # padded tokens per scoring batch
GEN_BATCH = 64
CHECKS = 32
TOLERANCE = 1.0          # nats per text, alone vs in a batch: bf16 rounding moves a few
                         # tenths; attending to padding would move many nats


def encode(tok, context: str, cont: str) -> tuple[list[int], int]:
    """Token ids of context + cont, and where cont's tokens start."""
    ids = tok(context + cont, add_special_tokens=False)["input_ids"]
    own = tok(context, add_special_tokens=False)["input_ids"]
    start = 0
    while start < min(len(ids), len(own)) and ids[start] == own[start]:
        start += 1
    cut = max(0, len(ids) - MAX_TOKENS)
    if start - cut < 1 or start == len(ids):
        raise ValueError(f"no room for the continuation of {context[-60:]!r}")
    return ids[cut:], start - cut


@torch.no_grad()
def score(model, texts: list[tuple[list[int], int]], pad: int) -> list[dict]:
    """Sum of the continuation's log-probabilities, its tokens, how many are
    the argmax, and how many from its start are (greedy decoding's prefix)."""
    order = sorted(range(len(texts)), key=lambda i: -len(texts[i][0]))
    out: list[dict] = [{}] * len(texts)
    i = 0
    while i < len(order):
        width = len(texts[order[i]][0])
        batch = order[i:i + max(1, BATCH_TOKENS // width)]
        i += len(batch)
        ids = torch.full((len(batch), width), pad, dtype=torch.long)
        mask = torch.zeros_like(ids)
        for row, j in enumerate(batch):
            t = texts[j][0]
            ids[row, :len(t)] = torch.tensor(t)
            mask[row, :len(t)] = 1
        logits = model(input_ids=ids.cuda(), attention_mask=mask.cuda()).logits
        for row, j in enumerate(batch):
            t, start = texts[j]
            target = torch.tensor(t[start:], device=logits.device)
            lp = torch.log_softmax(logits[row, start - 1:len(t) - 1].float(), dim=-1)
            hit = (lp.argmax(-1) == target).cpu()
            out[j] = dict(logprob=float(lp.gather(1, target[:, None]).sum()), tokens=len(target),
                          argmax=int(hit.sum()), prefix=int(hit.cumprod(0).sum()))
        del logits
    return out


@torch.no_grad()
def generate(model, tok, prompts: list[list[int]], new: list[int]) -> list[str]:
    """Greedy continuations, in batches of prompts of one length (no padding)."""
    out = [""] * len(prompts)
    by_len: dict[tuple[int, int], list[int]] = {}
    for i, (p, n) in enumerate(zip(prompts, new)):
        by_len.setdefault((len(p), n), []).append(i)
    stops = [tok.eos_token_id] + ([model.generation_config.eos_token_id]
                                  if isinstance(model.generation_config.eos_token_id, int) else
                                  list(model.generation_config.eos_token_id or []))
    for (_, n), idx in sorted(by_len.items()):
        for k in range(0, len(idx), GEN_BATCH):
            chunk = idx[k:k + GEN_BATCH]
            ids = torch.tensor([prompts[i] for i in chunk]).cuda()
            gen = model.generate(input_ids=ids, attention_mask=torch.ones_like(ids),
                                 max_new_tokens=n, do_sample=False, temperature=None, top_p=None,
                                 top_k=None, eos_token_id=sorted(set(stops)),
                                 pad_token_id=tok.pad_token_id or tok.eos_token_id)
            for i, g in zip(chunk, gen[:, ids.shape[1]:].tolist()):
                for s in stops:
                    if s in g:
                        g = g[:g.index(s)]
                out[i] = tok.decode(g, skip_special_tokens=False)
    return out


def model_class(model_dir: str):
    """The class to load. A config may name another repo's module for it
    ("<repo>--modeling_x.Class"); offline, the checkpoint's own file of that
    name is used instead, and the reference returned for the record."""
    ref = json.loads(Path(model_dir, "config.json").read_text()).get("auto_map", {}).get(
        "AutoModelForCausalLM", "")
    if "--" not in ref:
        return AutoModelForCausalLM, None
    from transformers.dynamic_module_utils import get_class_from_dynamic_module
    return get_class_from_dynamic_module(ref.split("--", 1)[1], model_dir), ref


def main(probes_path: str, model_dir: str, out_path: str) -> int:
    t0 = time.time()
    probes = pd.read_parquet(probes_path)
    tok = AutoTokenizer.from_pretrained(model_dir, trust_remote_code=True)
    cls, remote_ref = model_class(model_dir)
    model = cls.from_pretrained(model_dir, trust_remote_code=True,
                                dtype=torch.bfloat16).cuda().eval()
    pad = tok.pad_token_id if tok.pad_token_id is not None else tok.eos_token_id

    texts, keys = [], []
    for r in probes.itertuples():
        for c, cont in enumerate([r.target or None, *list(r.choices)]):
            if cont is not None:
                texts.append(encode(tok, r.context, cont))
                keys.append((r.Index, c - 1))       # -1: the target, else the choice
    scored = score(model, texts, pad)
    picks = range(0, len(texts), max(1, len(texts) // CHECKS))
    alone = [score(model, [texts[i]], pad)[0]["logprob"] for i in picks]
    diffs = [abs(a - scored[i]["logprob"]) for a, i in zip(alone, picks)]
    if max(diffs) > TOLERANCE:
        raise SystemExit(f"batched and single scores differ by {max(diffs):.3f} nats")

    gen_rows = probes.index[probes["generate"] > 0]
    prompts = [tok(probes.at[i, "context"], add_special_tokens=False)["input_ids"][-MAX_TOKENS:]
               for i in gen_rows]
    news = probes.loc[gen_rows, "generate"].astype(int).tolist()
    gens = generate(model, tok, prompts, news)
    step = max(1, len(prompts) // CHECKS)
    alone_gens = [(generate(model, tok, [prompts[i]], [news[i]])[0], gens[i])
                  for i in range(0, len(prompts), step)]
    same = [a == b for a, b in alone_gens]
    # Where a differing pair parts, in characters of the batched continuation.
    parts = [next((k for k, (x, y) in enumerate(zip(a, b)) if x != y), min(len(a), len(b)))
             for a, b in alone_gens if a != b]

    res = probes[["id"]].copy()
    for col in ("target_logprob", "target_tokens", "target_argmax", "target_prefix"):
        res[col] = None
    res["choice_logprob"] = [[] for _ in range(len(res))]
    res["choice_tokens"] = [[] for _ in range(len(res))]
    for (i, c), s in zip(keys, scored):
        if c < 0:
            for k in ("logprob", "tokens", "argmax", "prefix"):
                res.at[i, f"target_{k}"] = s[k]
        else:
            res.at[i, "choice_logprob"].append(s["logprob"])
            res.at[i, "choice_tokens"].append(s["tokens"])
    res["generation"] = None
    res.loc[gen_rows, "generation"] = gens
    res.to_parquet(out_path, index=False)

    source = Path(model_dir, "SOURCE")
    meta = dict(model_dir=model_dir, source=source.read_text().strip() if source.exists() else None,
                local_code_for=remote_ref,
                probes=probes_path, n_probes=len(probes), n_texts=len(texts),
                n_generations=len(prompts), torch=torch.__version__,
                transformers=__import__("transformers").__version__,
                gpu=torch.cuda.get_device_name(0), seconds=round(time.time() - t0, 1),
                check_logprob_max_abs_diff=max(diffs), check_logprob_diffs=diffs,
                check_logprob_n=len(alone),
                check_generation_same=sum(same), check_generation_n=len(same),
                check_generation_parts_at=parts)
    Path(out_path).with_suffix(".json").write_text(json.dumps(meta, indent=1) + "\n")
    print(json.dumps(meta))
    return 0


if __name__ == "__main__":
    sys.exit(main(*sys.argv[1:]))
