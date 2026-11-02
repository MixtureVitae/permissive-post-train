"""What the models show of the CommonsenseQA questions of their training data:
can they reproduce them, and do they answer them better?

    mvpt build-memorization-probes        the texts to score
    python mv_post_train/gpu/score_texts.py PROBES MODEL_DIR OUT   (per checkpoint, on a GPU)
    mvpt measure-memorization             the comparisons

The Tulu3 SFT data holds 650 of CommonsenseQA's 1,221 validation questions
(670 are in the training data audited), in FLAN's templates
("<question>\\nOptions:\\n- <choice>\\n..."), mostly as worked examples inside
the prompt, which the Phase 1 SFT loss covers. The probes:

  options     the question, then "Options:": the log-probability of its five
              choices in order, and a greedy continuation. The distractors are
              arbitrary, so only memory gives all five. Validation, test and a
              sample of the training split.
  answer      the question and its choices as a chat turn: the log-probability
              of each choice as the reply, and a greedy reply.
  own         each trained-on question once, in the training text that holds
              it, up to "Options:", with its choices as written there.
  transplant  the same training text with a clean question in place of the
              trained-on one: the control for `own` (`lift`).

Each SFT comparison is a difference in differences: the change from parent
to SFT checkpoint on the trained-on questions minus that on the clean ones
(found nowhere), with a 95% interval that resamples questions within each
group, as `overlap_sensitivity.dose_response` does. For `own`, the measure
is the training text's `lift`, each question against itself; the interval
resamples the pairs. The groups are not randomized, so these are descriptive
comparisons of the questions found and the others, and a probe that finds
nothing says that these targets, prompts and greedy decoding elicited
nothing, not that nothing was memorized. Each interval is unadjusted; `q`
corrects the bootstrap p-values (Benjamini-Hochberg) within each probe's
comparisons of the SFT steps.

The pretraining comparison, an exploratory diagnostic of familiarity, sets
the bases' scores on the validation questions against those on the test
questions. Its rationale assumes that MixtureVitae's filter removed text
matching the test questions and nothing matching the validation questions,
as the index in the tool's repository would (the other corpora removed
neither); which index filtered the pretraining data is not established.

`memorization.csv` has one row per comparison, metric and group, and
`memorization_levels.csv` each checkpoint's mean of each metric by group.
"""
from __future__ import annotations

import json
import re
from pathlib import Path

import numpy as np
import pandas as pd
import pyarrow.parquet as pq
from statsmodels.stats.multitest import multipletests

from . import overlap_sensitivity as osens
from . import training_data_overlap as tdo
from .factorial import _two_sided_p

CSQA = "commonsense_qa"
PROBE_SEED = 20261031
TRAIN_SAMPLE = 1221
OPTIONS_NEW = 64          # tokens: five choices and some
ANSWER_NEW = 96
USER, ASSISTANT = "<|user|>\n", "\n<|assistant|>\n"   # the SFT chat template
OPTIONS = re.compile(r"\s*Options:((?:\n- [^\n]*){5})")
SFT = "tulu3_sft_decontaminated"
#: The bases of the pretraining comparison: MixtureVitae and the others.
BASES = ("mv.base", "comma.base", "fineweb.base", "fineweb.base4k")
MODELS = tuple(dict.fromkeys(k for step in osens.SFT_STEPS for k in step))
#: The revision each score file must come from (its OUT.json `source`).
CHECKPOINTS = {
    "mv.base": "ontocord/1.7b-MixtureVitae-300BT-v1-decontaminated-16k"
               "@679c22f0f97a3c574b7be19f3ec78f22103946e5",
    "mv.sft": "ali-elganzory/1.7b-MixtureVitae-300BT-v1-decontaminated-16k-SFT-Tulu3-decontaminated"
              "@257668b3c4beb702d6784f915a65a0d2cf44b471",
    "comma.base": "ontocord/1.7b-Comma0.1-300BT-longsft_16k@05885554b97145b3a1fef9d8260c56b34f917929",
    "comma.sft": "ali-elganzory/1.7b-Comma0.1-300BT-longsft_16k-SFT-Tulu3-decontaminated"
                 "@12e498ed0caa662d30e5352a70d1e3b8fa9fbe46",
    "fineweb.base4k": "open-sci/open-sci-ref-v0.02-1.7b-fineweb-edu-1.4t-300B-4096"
                      "@3dab16eef1598a1c2ca0e9045eb4c95b4da489f7",
    "fineweb.sft4k": "ali-elganzory/open-sci-ref-v0.02-1.7b-fineweb-edu-1.4t-300B-4096-SFT-Tulu3-"
                     "decontaminated@cf968d0818df9c2e09bafff431ebd5aceafed166",
    "fineweb.base": "open-sci/open-sci-ref-v0.02-1.7b-fineweb-edu-1.4t-300B-4096-longsft_16k"
                    "@4856beae011618d9837b822f682b6f00dd9823d9",
    "fineweb.sft": "ali-elganzory/open-sci-ref-v0.02-1.7b-fineweb-edu-1.4t-300B-4096-longsft_16k-SFT-"
                   "Tulu3-decontaminated@f9efae1e14a4e5a9bb77ee05d21df4478f0a3e28",
}
METRICS = {"options": ("logprob", "exact", "recalled"), "own": ("logprob", "exact", "recalled"),
           "answer": ("correct", "gold_prob", "reply_correct")}


def _block(choices) -> str:
    return "".join(f"\n- {c}" for c in choices)


def _split(out_dir: Path, split: str) -> pd.DataFrame:
    s = tdo.FROM_HUB[CSQA]
    rows = tdo._rows(tdo._hub_file(s.dataset, s.revision, f"data/{split}-00000-of-00001.parquet",
                                   out_dir / "benchmarks"))
    return pd.DataFrame(dict(
        item=[r["id"] for r in rows], question=[r["question"] for r in rows],
        choices=[list(r["choices"]["text"]) for r in rows],
        gold=[list(r["choices"]["label"]).index(r["answerKey"]) if r["answerKey"] else -1
              for r in rows]))


def groups(out_dir: Path) -> pd.DataFrame:
    """The validation questions: `group` trained (in the Tulu3 SFT data),
    clean (found nowhere, by `tdo.EXCLUDED` evidence) or other, with the SFT
    rows holding each and whether one shows its answer."""
    items = pd.read_parquet(out_dir / "benchmark_items.parquet")
    ids = items.loc[items["benchmark"] == CSQA, "item"].reset_index(drop=True)
    f = pd.read_parquet(out_dir / "found.parquet")
    found = set(f.loc[(f["benchmark"] == CSQA) & f["evidence"].isin(tdo.EXCLUDED), "item"])
    e = pd.read_parquet(out_dir / "exposure.parquet")
    e = e[e["benchmark"] == CSQA].set_index("item")
    rows = e[f"rows_{SFT}"].reindex(ids, fill_value=0).to_numpy()
    shown = e[f"answer_rows_{SFT}"].reindex(ids, fill_value=0).to_numpy() > 0
    return pd.DataFrame(dict(item=ids, sft_rows=rows, answer_shown=shown,
                             group=np.select([rows > 0, ~ids.isin(found)], ["trained", "clean"],
                                             "other")))


def _messages(raw: Path, rows: set[int]) -> dict[int, list]:
    out, start = {}, 0
    for f in sorted((raw / "data").glob("*.parquet")):
        n = pq.ParquetFile(f).metadata.num_rows
        local = sorted(r - start for r in rows if start <= r < start + n)
        if local:
            t = pq.read_table(f, columns=["messages"]).take(local)
            out.update(zip((start + i for i in local), t["messages"].to_pylist()))
        start += n
    return out


def _named(answer: str, text: str) -> bool:
    pattern = r"(?<![a-z])" + re.escape(answer.lower()) + r"(?![a-z])"
    return re.search(pattern, text.lower()) is not None


def in_context(out_dir: Path, val: pd.DataFrame, g: pd.DataFrame) -> pd.DataFrame:
    """Each trained-on question once, in the first SFT prompt that holds it
    verbatim with its five choices (`own`), and a clean question, drawn
    without replacement while they last, in the same text (`transplant`).
    Many templates give a question's rationale and answer before it, so the
    text since the previous question's choices must not name the answer:
    copying it from the context is not memory."""
    f = pd.read_parquet(out_dir / "found.parquet")
    f = f[(f["benchmark"] == CSQA) & (f["where"] == SFT) & (f["turn"] == "0")
          & f["evidence"].isin(tdo.EXCLUDED)].sort_values("row")
    msgs = _messages(out_dir / "raw" / SFT, set(f["row"].astype(int)))
    q = val.set_index("item")
    out, seen = [], set()
    for row, item in zip(f["row"].astype(int), f["item"]):
        if item in seen:
            continue
        text = msgs[row][0]["content"]
        at = text.find(q.at[item, "question"])
        m = OPTIONS.match(text, at + len(q.at[item, "question"])) if at >= 0 else None
        gold = q.at[item, "choices"][q.at[item, "gold"]]
        if m and not _named(gold, text[max(0, text.rfind("Options:", 0, at)):at]):
            seen.add(item)
            out.append(dict(context_item=item, row=row, before=USER + text[:at],
                            own=USER + text[:m.start(1)], target=m.group(1)))
    df = pd.DataFrame(out)
    clean = g.loc[g["group"] == "clean", "item"].to_numpy()
    rng = np.random.default_rng(PROBE_SEED)
    draw = np.concatenate([rng.permutation(clean) for _ in range(-(-len(df) // len(clean)))])
    df["clean_item"] = draw[:len(df)]
    return df


def probes(out_dir: Path) -> pd.DataFrame:
    items = pd.read_parquet(out_dir / "benchmark_items.parquet")
    answers = tdo.gold(items, out_dir / "benchmarks")
    val = _split(out_dir, "validation")
    own = items.loc[items["benchmark"] == CSQA, "item"].tolist()
    if val["item"].tolist() != own:
        raise ValueError("the validation file's order is not benchmark_items'")
    if any(c[k] != answers[(CSQA, i)] for i, c, k in zip(val["item"], val["choices"], val["gold"])):
        raise ValueError("gold answers disagree with training_data_overlap.gold")
    test = _split(out_dir, "test")
    train = _split(out_dir, "train")
    train = train.iloc[np.sort(np.random.default_rng(PROBE_SEED).choice(len(train), TRAIN_SAMPLE,
                                                                        replace=False))]
    out = []
    for split, df in (("validation", val), ("test", test), ("train", train)):
        for r in df.itertuples():
            out.append(dict(id=f"options:{split}:{r.item}", kind="options", split=split,
                            item=r.item, context_item=r.item, context=f"{r.question}\nOptions:",
                            target=_block(r.choices), choices=[], gold=r.gold,
                            generate=OPTIONS_NEW))
    for r in val.itertuples():
        out.append(dict(id=f"answer:validation:{r.item}", kind="answer", split="validation",
                        item=r.item, context_item=r.item,
                        context=f"{USER}{r.question}\nOptions:{_block(r.choices)}{ASSISTANT}",
                        target="", choices=list(r.choices), gold=r.gold, generate=ANSWER_NEW))
    q = val.set_index("item")
    for r in in_context(out_dir, val, groups(out_dir)).itertuples():
        c = r.clean_item
        out.append(dict(id=f"own:{r.context_item}", kind="own", split="validation",
                        item=r.context_item, context_item=r.context_item, context=r.own,
                        target=r.target, choices=[], gold=-1, generate=OPTIONS_NEW))
        out.append(dict(id=f"transplant:{r.context_item}", kind="transplant", split="validation",
                        item=c, context_item=r.context_item,
                        context=f"{r.before}{q.at[c, 'question']}\nOptions:",
                        target=_block(q.at[c, "choices"]), choices=[], gold=-1,
                        generate=OPTIONS_NEW))
    return pd.DataFrame(out)


# ---------------------------------------------------------------------------
# Scores

def _norm(s: str) -> str:
    return " ".join(s.lower().split())


def recalled(text: str | None, choices) -> int:
    """How many of the choices a continuation lists among its first five
    "- " lines."""
    lines = [ln.strip()[1:] for ln in (text or "").split("\n") if ln.strip().startswith("-")][:5]
    return len({_norm(ln) for ln in lines} & {_norm(c) for c in choices})


def picked(text: str | None, choices) -> int:
    """The choice a reply names last (the longer on a tie), or -1."""
    t, best = (text or "").lower(), (-1, -1, -1)
    for k, c in enumerate(choices):
        for m in re.finditer(r"(?<![a-z])" + re.escape(c.lower()) + r"(?![a-z])", t):
            best = max(best, (m.end(), len(c), k))
    return best[2]


def metrics(p: pd.DataFrame, s: pd.DataFrame, choices: dict) -> pd.DataFrame:
    """One row per probe: `logprob` (nats per token of the target), `exact`
    (greedy decoding gives the whole target), `recalled` (choices the greedy
    continuation lists), and for `answer`, `correct` (the gold choice is the
    most likely), `gold_prob` (its share of the five choices' probability)
    and `reply_correct` (the greedy reply names it)."""
    df = p.drop(columns=["context", "target"]).merge(s, on="id", validate="1:1")
    t = df["target_tokens"].astype(float)
    df["logprob"] = df["target_logprob"].astype(float) / t
    df["exact"] = np.where(t.isna(), np.nan, (df["target_prefix"].astype(float) == t) * 100.0)
    df["recalled"] = [recalled(g, choices[i]) if k != "answer" else np.nan
                      for g, i, k in zip(df["generation"], df["item"], df["kind"])]
    lp = [np.asarray(x, dtype=float) for x in df["choice_logprob"]]
    df["correct"] = [float(x.argmax() == k) * 100 if x.size else np.nan
                     for x, k in zip(lp, df["gold"])]
    df["gold_prob"] = [float(np.exp(x[k] - np.logaddexp.reduce(x))) * 100 if x.size else np.nan
                       for x, k in zip(lp, df["gold"])]
    df["reply_correct"] = [float(picked(g, choices[i]) == k) * 100 if kind == "answer" else np.nan
                           for g, i, k, kind in zip(df["generation"], df["item"], df["gold"],
                                                    df["kind"])]
    return df


def _interval(x: np.ndarray, y: np.ndarray | None, rng, n: int) -> tuple[float, float, float]:
    """The 95% interval of mean(x) - mean(y), and the bootstrap p-value of zero."""
    a = rng.choice(x, (n, x.size)).mean(axis=1)
    b = 0.0 if y is None else rng.choice(y, (n, y.size)).mean(axis=1)
    lo, hi = np.percentile(a - b, [2.5, 97.5])
    return float(lo), float(hi), _two_sided_p(a - b)


def lift(d: pd.DataFrame, metric: str) -> pd.Series:
    """By `own` probe: how much its training text raises the trained-on
    question's score over the question alone (its `options` probe), minus
    how much the same text raises a clean question's (the `transplant`).
    Each question is compared with itself, so its difficulty cancels."""
    alone = d[(d["kind"] == "options") & (d["split"] == "validation")].set_index("item")[metric]
    own = d[d["kind"] == "own"].set_index("context_item")
    other = d[d["kind"] == "transplant"].set_index("context_item").reindex(own.index)
    return ((own[metric] - alone.reindex(own["item"]).to_numpy())
            - (other[metric] - alone.reindex(other["item"]).to_numpy()))


def sft_steps(m: dict[str, pd.DataFrame], g: pd.DataFrame) -> pd.DataFrame:
    """For each SFT step and metric: the trained-on questions' change minus
    the clean ones', overall, by SFT rows, and with the answer shown; and the
    training text's `lift` at the SFT checkpoint and its change from the
    parent, which saw none of the Tulu3 data."""
    rng = np.random.default_rng(osens.DOSE_SEED)
    gi = g.set_index("item")
    trained = gi.index[gi["group"] == "trained"]
    sets = {"in the SFT data": trained, "1 SFT row (2 exposures)": gi.index[gi["sft_rows"] == 1],
            "2 or more SFT rows": gi.index[gi["sft_rows"] >= 2],
            "answer shown in SFT": gi.index[gi["answer_shown"]]}
    clean = gi.index[gi["group"] == "clean"]
    out = []
    for child, parent in osens.SFT_STEPS:
        c, p = m[child], m[parent]
        for kind in ("options", "answer"):
            sel = lambda d: d[(d["kind"] == kind) & (d["split"] == "validation")].set_index("item")
            cc, pp = sel(c), sel(p)
            for metric in METRICS[kind]:
                change = (cc[metric] - pp[metric]).dropna()
                y = change.reindex(clean).dropna().to_numpy()
                for name, keys in sets.items():
                    x = change.reindex(keys).dropna().to_numpy()
                    lo, hi, pv = _interval(x, y, rng, osens.DOSE_RESAMPLES)
                    out.append(dict(comparison="SFT step", step=f"{child} from {parent}",
                                    probe=kind, metric=metric, group=name, n=x.size,
                                    n_control=y.size, parent=pp[metric].reindex(keys).mean(),
                                    child=cc[metric].reindex(keys).mean(),
                                    parent_control=pp[metric].reindex(clean).mean(),
                                    child_control=cc[metric].reindex(clean).mean(),
                                    change=x.mean(), change_control=y.mean(),
                                    difference=x.mean() - y.mean(), lo=lo, hi=hi, p=pv))
        for metric in METRICS["own"]:
            lc, lp = lift(c, metric), lift(p, metric)
            for name, x in (("training text's lift, SFT checkpoint", lc),
                            ("training text's lift, change from parent", lc - lp)):
                x = x.dropna().to_numpy()
                lo, hi, pv = _interval(x, None, rng, osens.DOSE_RESAMPLES)
                out.append(dict(comparison="SFT step", step=f"{child} from {parent}", probe="own",
                                metric=metric, group=name, n=x.size, n_control=x.size,
                                parent=lp.mean(), child=lc.mean(), change=(lc - lp).mean(),
                                difference=x.mean(), lo=lo, hi=hi, p=pv))
    df = pd.DataFrame(out)
    df["q"] = df.groupby("probe")["p"].transform(lambda p: multipletests(p, method="fdr_bh")[1])
    return df


def pretraining(m: dict[str, pd.DataFrame]) -> pd.DataFrame:
    """Each base's validation-minus-test difference on the `options` probes,
    and MixtureVitae's minus each other base's. The bases score the same
    questions, so the second resamples each split's per-question differences
    between the two bases, keeping the pairing. An exploratory diagnostic of
    familiarity: it does not establish which index filtered the pretraining
    data."""
    rng = np.random.default_rng(osens.DOSE_SEED)
    out, by = [], {}
    for key in BASES:
        d = m[key][m[key]["kind"] == "options"]
        for metric in METRICS["options"]:
            x = d.loc[d["split"] == "validation", metric].to_numpy()
            y = d.loc[d["split"] == "test", metric].to_numpy()
            by[key, metric] = {s: d[d["split"] == s].set_index("item")[metric]
                               for s in ("validation", "test")}
            lo, hi, pv = _interval(x, y, rng, osens.DOSE_RESAMPLES)
            out.append(dict(comparison="validation minus test", step=key, probe="options",
                            metric=metric, group="validation", n=x.size, n_control=y.size,
                            child=x.mean(), child_control=y.mean(),
                            train=d.loc[d["split"] == "train", metric].mean(),
                            difference=x.mean() - y.mean(), lo=lo, hi=hi, p=pv))
    for key in BASES[1:]:
        for metric in METRICS["options"]:
            mv, ref = by["mv.base", metric], by[key, metric]
            x, y = ((mv[s] - ref[s].reindex(mv[s].index)).dropna().to_numpy()
                    for s in ("validation", "test"))
            lo, hi, pv = _interval(x, y, rng, osens.DOSE_RESAMPLES)
            out.append(dict(comparison="MixtureVitae's gap minus another base's", step=key,
                            probe="options", metric=metric, group="validation", n=x.size,
                            n_control=y.size, difference=x.mean() - y.mean(), lo=lo, hi=hi,
                            p=pv))
    return pd.DataFrame(out)


def levels(m: dict[str, pd.DataFrame], g: pd.DataFrame) -> pd.DataFrame:
    grp = g.set_index("item")["group"]
    out = []
    for key, d in m.items():
        d = d.assign(group=np.where(d["split"] == "validation",
                                    d["item"].map(grp).fillna("other"), d["split"]))
        d.loc[d["kind"] == "transplant", "group"] = "clean"
        for (kind, group), x in d.groupby(["kind", "group"]):
            row = dict(model=key, probe=kind, group=group, n=len(x))
            row.update({k: x[k].mean() for k in ("logprob", "exact", "recalled", "correct",
                                                 "gold_prob", "reply_correct")})
            out.append(row)
    return pd.DataFrame(out)


def build(out_dir: Path) -> int:
    p = probes(out_dir)
    (out_dir / "memorization").mkdir(exist_ok=True)
    p.to_parquet(out_dir / "memorization" / "probes.parquet", index=False)
    print(p.groupby(["kind", "split"]).size().to_string())
    return 0


def main(out_dir: Path) -> int:
    d = out_dir / "memorization"
    p = pd.read_parquet(d / "probes.parquet")
    choices = {r.item: list(r.choices) for r in p[p["kind"] == "answer"].itertuples()}
    for split in ("test", "train"):
        choices.update({r.item: list(r.choices) for r in _split(out_dir, split).itertuples()})
    for k in MODELS:
        meta = json.loads((d / "scores" / f"{k}.json").read_text())
        if meta["source"] != CHECKPOINTS[k] or meta["n_probes"] != len(p):
            raise ValueError(f"{k}: scores of {meta['source']} on {meta['n_probes']} probes")
    m = {k: metrics(p, pd.read_parquet(d / "scores" / f"{k}.parquet"), choices) for k in MODELS}
    g = groups(out_dir)
    pd.concat([sft_steps(m, g), pretraining(m)]).to_csv(out_dir / "memorization.csv", index=False)
    levels(m, g).to_csv(out_dir / "memorization_levels.csv", index=False)
    print((out_dir / "memorization.csv").read_text())
    return 0
