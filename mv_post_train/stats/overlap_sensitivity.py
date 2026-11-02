"""How the main comparisons change without the benchmark items found in the training data.

    mvpt measure-overlap-sensitivity

`training_data_overlap.found` lists the benchmark items found in the training
data, by rules fixed before any model's answers are looked at. This leaves the
same items out of every checkpoint, and recomputes each comparison with its
interval beside the full-benchmark one. The set is one for every comparison:
each item found with evidence in `training_data_overlap.EXCLUDED` in any data
scanned (the 16K extension data, the Tulu3 SFT and DPO data, the
OpenThoughts3 questions and the samples of the other reasoning corpora, and,
once stats/benchmark_scan.py has run, the Phase 2 30K samples as trained and
as written and MixtureVitae's reasoning shards), so no model gets a subset of
its own. The comparisons:

  phase2      the suite-average and MATH500 change of every run of Table 2,
              and the OT3 exclusion's two gains and their difference, on the
              average and on MATH500 (render/phase2_analyses.py)
  table4      Table 4's twenty MATH500 changes, five bases from their Tulu3
              SFT checkpoints with each of the four reasoning corpora
              (render/phase2_tables.py)
  factorial   the front-loading contrasts on the reasoning suite
              (stats/factorial.py)
  language    the language suite's parent-child changes and front-loading
              contrasts (render/language_intervals.py)

Each comparison's intervals are those of its own module, with its seed, so a
full-benchmark row equals the one the paper prints. This is a sensitivity
analysis to leaving out the items identified, not an evaluation on
uncontaminated items: the scans miss paraphrases, and MixtureVitae's
pretraining data is checked only in its reasoning shards. Phase 2's code
items (MBPP, LiveCodeBench) cannot be left out: its records have no
per-question verdicts for the code tasks.

`overlap_sensitivity.csv` has one row per comparison: analysis, family,
label, then estimate, lo, hi and q for the full benchmarks and with the items
left out (`_cut`), and `excluded_items.csv` the items left out.
`exposure.parquet` counts how often training nominally showed each item
(`training_data_overlap.exposure`: rows times epochs), and
`dose_response.csv` compares the Tulu3 SFT changes on CommonsenseQA's
trained-on items with those on its clean ones (`dose_response`). The groups
are not randomized, so the comparison is descriptive.
"""
from __future__ import annotations

from collections import Counter
from pathlib import Path

import pandas as pd

from . import phase2_suite as ps
from . import training_data_overlap as tdo

#: training_data_overlap's benchmark names -> the Phase 2 records' and the
#: manifest's task names, for the tasks with per-question verdicts.
PHASE2_TASKS = {"AIME24": "AIME24", "AIME25": "AIME25", "AMC23": "AMC23", "MATH500": "MATH500",
                "GSM8K": "GSM8K", "GPQADiamond": "GPQA-Diamond", "JEEBench": "JEEBench"}
#: ... and the Protocol A records' benchmark names. These have per-question
#: verdicts for MBPP and LiveCodeBench too (task and problem ids, as here).
PROTOCOL_A = {"AIME24": "AIME24", "AIME25": "AIME25", "AMC23": "AMC23", "MATH500": "MATH500",
              "GSM8K": "gsm8k", "GPQADiamond": "GPQADiamond", "JEEBench": "JEEBench",
              "MBPP": "MBPP", "LiveCodeBenchv5": "LiveCodeBench"}
#: The benchmarks whose record ids are positions in the benchmark's own order.
BY_POSITION = ("JEEBench",)


def _positions(items: pd.DataFrame, benchmark: str) -> dict[str, int]:
    ids = items.loc[items["benchmark"] == benchmark, "item"].tolist()
    return {item: i for i, item in enumerate(ids)}


def phase2_exclude(drop: dict[str, set[str]], items: pd.DataFrame,
                   meta: pd.DataFrame) -> dict[str, frozenset]:
    """Manifest task -> the questions (`example_idx`) to leave out."""
    out = {}
    for bench, task in PHASE2_TASKS.items():
        if bench not in drop:
            continue
        rows = meta[meta["task"] == bench].drop_duplicates(ps.QUESTION)
        if bench in BY_POSITION:
            pos = _positions(items, bench)
            idx = {pos[i] for i in drop[bench]}
        else:
            idx = set(rows.loc[rows["example_id"].astype(str).isin(drop[bench]), ps.QUESTION])
        if len(idx) != len(drop[bench]):
            raise ValueError(f"{bench}: {len(idx)} of the {len(drop[bench])} items are in the records")
        out[task] = frozenset(idx)
    return out


def phase2(manifest_path: Path, phase2_dir: Path, drop: dict[str, set[str]],
           items: pd.DataFrame) -> pd.DataFrame:
    from ..manifest import phase2 as p2
    from ..manifest.reader import Manifest
    from ..render import phase2_analyses as pa
    m = Manifest.load(manifest_path)
    _, meta = p2.load(phase2_dir)
    ifeval = p2.ifeval_counts(phase2_dir)
    runs = pa.table2_runs(m)
    with_, without = pa.exclusion_runs(m)
    contrasts = (pa.table2_contrasts(m, runs, "table2.avg")
                 + [ps.change("table2.m500", r.stem, r.key, r.parent, ("MATH500",)) for r in runs]
                 + pa.exclusion_contrasts(m, with_, without, "exclusion"))
    full = ps.evaluate(ps.Scores(m, meta, ifeval), contrasts)
    cut = ps.evaluate(ps.Scores(m, meta, ifeval, exclude=phase2_exclude(drop, items, meta)),
                      contrasts)
    return pd.DataFrame([dict(analysis="phase2", family=a.contrast.family, label=a.contrast.label,
                              estimate=a.estimate, lo=a.lo, hi=a.hi, q=a.q,
                              estimate_cut=b.estimate, lo_cut=b.lo, hi_cut=b.hi, q_cut=b.q)
                         for a, b in zip(full, cut)])


def table4(manifest_path: Path, phase2_dir: Path, drop: dict[str, set[str]]) -> pd.DataFrame:
    """Table 4's MATH500 changes, every base of phase2_tables.HELDOUT_BASES
    from its Tulu3 SFT checkpoint with each reasoning corpus, as
    `heldout_changes` gives them, and with the items left out of both
    checkpoints of each pair (the same seed, order and correction)."""
    from ..manifest import phase2 as p2
    from ..manifest.reader import Manifest
    from ..render import phase2_tables as pt
    from .paired_items import item_scores, paired_changes
    m = Manifest.load(manifest_path)
    _, meta = p2.load(phase2_dir)
    cells = pt.heldout_changes(m, meta)
    out = drop.get(pt.HELDOUT_TASK, set())

    def scores(key: str):
        src = m.result(key, pt.HELDOUT_TASK)["source"]
        s = item_scores(meta, src["base_model"], src["stage"], src["task"])
        if len(out & set(s.index.astype(str))) != len(out):
            raise ValueError(f"{key}: not every item left out is in the records")
        return s[~s.index.astype(str).isin(out)]
    cut = paired_changes([(scores(c.key), scores(m.checkpoint(c.base_model, "SFT-Tulu3")))
                          for c in cells])
    return pd.DataFrame([dict(analysis="table4", family="table4.m500",
                              label=f"{c.base_model} | {c.corpus}", estimate=c.change.estimate,
                              lo=c.change.lo, hi=c.change.hi, q=c.change.q, estimate_cut=b.estimate,
                              lo_cut=b.lo, hi_cut=b.hi, q_cut=b.q) for c, b in zip(cells, cut)])


def factorial(manifest_path: Path, data_dir: Path, drop: dict[str, set[str]],
              items: pd.DataFrame) -> pd.DataFrame:
    from ..render import frontloading as fr
    from . import factorial as fs
    df, _, _ = fr.load(manifest_path, data_dir)
    keys = set()
    for bench, name in PROTOCOL_A.items():
        pos = _positions(items, bench)
        keys |= {(name, str(pos[i]) if bench in BY_POSITION else i) for i in drop.get(bench, ())}
    cut = df[[k not in keys for k in zip(df["benchmark"], df["sample_id"].astype(str))]]
    full_c = fs.factorial_contrasts(df, pools=fr.POOLS, permissive=fr.PERMISSIVE)
    cut_c = fs.factorial_contrasts(cut, pools=fr.POOLS, permissive=fr.PERMISSIVE)
    on = ["family", "label", "context_length", "stage"]
    both = full_c.merge(cut_c, on=on, suffixes=("", "_cut"))
    both["label"] = (both["label"] + " | " + both["context_length"].astype(str) + " | "
                     + both["stage"].astype(str))
    return both.assign(analysis="factorial")[
        ["analysis", "family", "label", "estimate", "lo", "hi", "q",
         "estimate_cut", "lo_cut", "hi_cut", "q_cut"]]


def language_keys(drop: dict[str, set[str]], items: pd.DataFrame, cache: Path) -> set:
    """(task, item key of the language records) of the items to leave out.
    The records key a question by its place in the split (`doc_id`), MMLU's
    by its subject and its place in it."""
    keys = set()
    for bench in drop:
        if bench not in tdo.FROM_HUB or bench in tdo.FROM_LOGS or bench in PHASE2_TASKS:
            continue
        pos = _positions(items, bench)
        if bench == "mmlu":
            s = tdo.FROM_HUB[bench]
            subjects = [r["subject"] for r in tdo._rows(tdo._hub_file(s.dataset, s.revision,
                                                                       s.file, cache))]
            seen, key = Counter(), []
            for sub in subjects:
                key.append(f"{sub}/{seen[sub]}")
                seen[sub] += 1
            keys |= {(bench, key[pos[i]]) for i in drop[bench]}
        else:
            keys |= {(bench, str(pos[i])) for i in drop[bench]}
    return keys


def language(manifest_path: Path, log_dir: Path, drop: dict[str, set[str]], items: pd.DataFrame,
             cache: Path) -> pd.DataFrame:
    """The parent-child changes and front-loading contrasts of the `default`
    prompt seed, on the average and on each task with items left out."""
    from ..data import language_samples as ls
    from ..manifest.reader import Manifest
    from ..render import language_intervals as li
    m = Manifest.load(manifest_path)
    recs = ls.load(log_dir, li.folders_by_zip())
    keys = language_keys(drop, items, cache)
    cut = recs[[k not in keys for k in zip(recs["task"], recs["item"])]]
    tasks = ["avg", *sorted({t for t, _ in keys})]
    stats = ["estimate", "lo", "hi", "q"]
    out = []
    for kind, fn, on in (("change", lambda r: li.compute(r, m)[1], ["model", "child", "parent", "task"]),
                         ("frontloading", li.frontloading, ["family", "context_length", "label", "task"])):
        a, b = (fn(r) for r in (recs, cut))
        a, b = (x[(x["mode"] == "default") & x["task"].isin(tasks)] for x in (a, b))
        both = a[on + stats].merge(b[on + stats], on=on, suffixes=("", "_cut"))
        both["label"] = both[on[:-1]].astype(str).agg(" | ".join, axis=1)
        out.append(both.assign(analysis="language", family=kind + "." + both["task"]))
    df = pd.concat(out, ignore_index=True)
    return df[["analysis", "family", "label", *stats, *(f"{c}_cut" for c in stats)]]


#: The Tulu3 SFT steps of the language records: (child, parent).
SFT = "tulu3_sft_decontaminated"
SFT_STEPS = (("mv.sft", "mv.base"), ("comma.sft", "comma.base"), ("fineweb.sft", "fineweb.base"),
             ("fineweb.sft4k", "fineweb.base4k"))
DOSE_SEED = 20261031
DOSE_RESAMPLES = 4000


def dose_response(log_dir: Path, f: pd.DataFrame, exp: pd.DataFrame, items: pd.DataFrame,
                  benchmark: str = "commonsense_qa") -> pd.DataFrame:
    """Did training on an item help on it? For each Tulu3 SFT step, the mean
    per-question change (in points, each question's score its mean over the
    three prompt seeds) of the items in the SFT data, by exposures and with
    the answer shown, minus that of the clean items (found nowhere), with a
    95% interval that resamples questions within each group."""
    import numpy as np
    from ..data import language_samples as ls
    from ..render import language_intervals as li
    pos = _positions(items, benchmark)
    found_any = {str(pos[i]) for i in f.loc[(f["benchmark"] == benchmark)
                                            & f["evidence"].isin(tdo.EXCLUDED), "item"]}
    e = exp[(exp["benchmark"] == benchmark) & (exp["rows_tulu3_sft_decontaminated"] > 0)]
    sft = e.assign(key=[str(pos[i]) for i in e["item"]]).set_index("key")
    groups = {"in the SFT data": sft.index,
              "2 exposures in SFT": sft.index[sft["rows_tulu3_sft_decontaminated"] == 1],
              "4 or more in SFT": sft.index[sft["rows_tulu3_sft_decontaminated"] >= 2],
              "answer shown in SFT": sft.index[sft[f"answer_rows_{SFT}"] > 0]}
    recs = ls.load(log_dir, li.folders_by_zip())
    recs = recs[recs["task"] == benchmark]
    rng = np.random.default_rng(DOSE_SEED)
    out = []
    for child, parent in SFT_STEPS:
        score = {k: recs[recs["folder"].isin(li.CHECKPOINTS[k].folders)].groupby("item")["score"]
                 .mean() * 100.0 for k in (child, parent)}
        change = (score[child] - score[parent]).dropna()
        clean = change[~change.index.isin(found_any)].to_numpy()
        for name, keys in groups.items():
            x = change[change.index.isin(keys)].to_numpy()
            boot = (rng.choice(x, (DOSE_RESAMPLES, x.size)).mean(axis=1)
                    - rng.choice(clean, (DOSE_RESAMPLES, clean.size)).mean(axis=1))
            lo, hi = np.percentile(boot, [2.5, 97.5])
            out.append(dict(benchmark=benchmark, step=f"{child} from {parent}", group=name,
                            n=x.size, n_clean=clean.size,
                            parent_score=float(score[parent].reindex(change.index)[
                                change.index.isin(keys)].mean()),
                            change=float(x.mean()), change_clean=float(clean.mean()),
                            difference=float(x.mean() - clean.mean()), lo=float(lo), hi=float(hi)))
    return pd.DataFrame(out)


def main(manifest_path: Path, out_dir: Path, phase2_dir: Path, data_dir: Path,
         log_dir: Path) -> int:
    items = pd.read_parquet(out_dir / "benchmark_items.parquet")
    f = tdo.found(out_dir)
    f["answer_shown"] = tdo.answer_shown(f, out_dir)
    f.to_parquet(out_dir / "found.parquet", index=False)
    drop = tdo.excluded(f)
    rows = [(b, i) for b, ids in sorted(drop.items()) for i in sorted(ids)]
    pd.DataFrame(rows, columns=["benchmark", "item"]).to_csv(out_dir / "excluded_items.csv",
                                                             index=False)
    print("  left out: " + ", ".join(f"{b} {len(ids)}" for b, ids in sorted(drop.items())))
    exp = tdo.exposure(f)
    exp.to_parquet(out_dir / "exposure.parquet", index=False)
    dose_response(log_dir, f, exp, items).round(4).to_csv(out_dir / "dose_response.csv",
                                                          index=False)
    parts = [phase2(manifest_path, phase2_dir, drop, items),
             table4(manifest_path, phase2_dir, drop),
             factorial(manifest_path, data_dir, drop, items),
             language(manifest_path, log_dir, drop, items, out_dir / "benchmarks")]
    df = pd.concat(parts, ignore_index=True).round(4)
    df.to_csv(out_dir / "overlap_sensitivity.csv", index=False)
    print(f"  wrote {out_dir / 'overlap_sensitivity.csv'}: {len(df)} comparisons")
    return 0
