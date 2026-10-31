"""Render the revision's front-loading study, with Comma0.1 as a fourth web
substrate and a second permissive one.

    mvpt render-frontloading

The workshop paper's study has three web substrates (`render-workshop`, which
keeps reproducing it). The revision adds Comma0.1: its front-loaded arm is the
`comma-mv_reasoning` records, and its arm without front-loading is the
Comma0.1 model of Phase 1 (`data/protocol_a.py`, FRONTLOADING_POOLS,
says why). This command renders the study's tables and figures for that
four-substrate design:

  <out-dir>/tables/factorial_summary.tex        Table 5
  <out-dir>/tables/contrasts.tex                Table 28
  <out-dir>/tables/domain_robustness.tex        Table 29
  <out-dir>/tables/factorial_perbenchmark.tex   Table 30
  <out-dir>/tables/factorial_language.tex       Table 31
  <out-dir>/tables/frontloading_numbers.tex     \\pfl{<key>} macros for the text
  <out-dir>/figures/factorial_main.pdf          Figure 3
  <out-dir>/figures/frontloading_trajectory.pdf Figure 4
  <out-dir>/figures/frontloading_domains.pdf    Figure 5
  <out-dir>/preview.pdf                         all of the above, compiled
  <art-dir>/*.csv                               the summaries behind them

The files keep the paper's basenames and labels. Every score comes from the
results manifest: the reasoning cells and the language cells its
`front_loading_study` design names. Comma0.1's language scores are not in the
artifacts repository, so the manifest reads them from Harsh Raj's results
files (`data/protocol_a.py`, COMMA_LANGUAGE_CELLS). The contrasts resample the
per-question records, which are read from the inputs the manifest names and
checked against its per-seed scores.
"""
from __future__ import annotations

from pathlib import Path

import pandas as pd

from ..data import protocol_a as pa
from ..manifest.reader import Manifest
from ..manifest.suites import LANGUAGE
from ..stats import factorial as fs
from . import preview
from .macros import macro_lines, number_word
from . import workshop_tables as wt
from .figures import workshop as wf

POOLS = pa.FRONTLOADING_POOLS
PERMISSIVE = pa.FRONTLOADING_PERMISSIVE
# The revision names the web-data factor "source pool" (the workshop paper:
# "web substrate"), so its tables and Figure 2 say so too.
POOL_WORDS = ("source pool", "source pool")
#: Table 29's cells are differences of seed means, printed without their spread.
ROBUSTNESS_NOTE = (r"Protocol~A (Appendix Table~\ref{tab:eval-settings-A}). Each cell is a point "
                   r"estimate, the difference of two scores that are each averaged over decoding "
                   r"seeds 42, 1234, and 2025.")
#: The checkpoints of Table 32, in the names of the paper (tracker C6).
LANGUAGE_CHECKPOINTS = "4K pretrained bases and their 16K extended checkpoints"
ROBUSTNESS_CLAIM = (r"under every view below each front-loading effect and each source-pool "
                    r"contrast stays positive.")

CONTROL_NOTE = (
    r"Comma0.1 is a second permissive source pool. Its model without front-loading is the "
    r"Comma0.1 model of Table~\ref{tab:phase1_zoo}, and the front-loaded MixtureVitae model "
    r"is the main MixtureVitae model. Both are reused checkpoints, trained earlier than the "
    r"other models here with the same recorded \texttt{open-sci-ref} configuration.")
LANGUAGE_NOTE = (
    r"Comma0.1 without front-loading reuses the Comma0.1 model of Table~\ref{tab:phase1_zoo}, "
    r"and it also sits at chance on MMLU and CommonsenseQA. The front-loaded MixtureVitae rows "
    r"reuse the main MixtureVitae model, as in Table~\ref{tab:lang_eval_table2}. "
    r"Table~\ref{tab:contrasts_language} gives 95\% intervals for its contrasts.")
#: The summary table (tab:factorial) also points to the language contrasts' intervals.
SUMMARY_NOTE = (CONTROL_NOTE + r" Table~\ref{tab:contrasts_language} gives 95\% intervals for "
                r"the language contrasts.")


def load(manifest_path: Path, data_dir: Path):
    """(per-question records, seed scores, language scores) of the study.

    The cells are those the results manifest's `front_loading_study` design
    names, and the language scores are its results. The per-question records,
    which the contrasts resample, are read from the inputs the manifest names
    (R13 with its replacement R20, and R15, with R21's strict IFEval verdicts),
    and their per-seed scores must equal the manifest's. They
    are used unrounded, since the manifest keeps six decimals."""
    keys = pa.FACTORIAL_KEYS
    m = Manifest.load(manifest_path)
    design = m.design("front_loading_study")
    cells = [{"model_group": c["substrate"], "front_loading": c["front_loaded"],
              "context_length": c["context_length"], "stage": c["stage"],
              "checkpoint": c["checkpoint"]} for c in design["cells"]]
    reasoning = pa.revised_reasoning(data_dir)
    df = pa.frontloading(reasoning, pa.revision_records("R15", data_dir))
    m.check_records(df, cells, keys)
    # The manifest keeps six decimals. The figures plot the unrounded scores.
    scores = fs.benchmark_scores(df, keys)
    language = pd.DataFrame([
        {"model_group": c["substrate"], "front_loading": c["front_loaded"],
         "context_length": c["context_length"], "task": t,
         "performance": m.value(c["checkpoint"], t, LANGUAGE)}
        for c in design["language_cells"] for t in pa.LANGUAGE_TASKS])
    if language.groupby(["model_group", "front_loading", "context_length"]).ngroups \
            != len(POOLS) * 2 * len(pa.CONTEXTS):
        raise ValueError("front-loading: the manifest's language cells are not fully crossed")
    return df, scores, language


# ---------------------------------------------------------------------------
# Macros for numbers quoted in the text

POOL_KEY = {"MV": "mv", "Comma": "comma", "Nemotron": "nemotron", "FineWeb-Edu": "fineweb"}
CTX_KEY = {4096: "4k", 16384: "16k"}
FAMILY_KEY = {"front-loading": "fl", "permissiveness": "perm", "interaction": "int"}


def _signed(v: float, digits: int = 1) -> str:
    v = round(v, digits) + 0.0                  # no negative zero
    return f"\\ensuremath{{{v:+.{digits}f}}}"


def _plain(v: float) -> str:
    """A range end: no sign when positive, a math minus when negative."""
    return f"{v:.1f}" if v >= 0 else f"\\ensuremath{{{v:.1f}}}"


def _q(q: float, relation: str = "=") -> str:
    """As Table 28 prints it."""
    return r"\ensuremath{q < 0.001}" if q < 0.001 else f"\\ensuremath{{q {relation} {q:.3f}}}"


def _label_key(family: str, label: str) -> str:
    parts = label.split(" - ")
    if family == "permissiveness":
        return "-".join(POOL_KEY[p] for p in parts)
    return POOL_KEY[parts[0]]


def _count(out: dict[str, str], key: str, n: int) -> None:
    out[key] = str(n)
    out[f"{key}.word"] = number_word(n)


def macros(summary: pd.DataFrame, language_avg: pd.DataFrame, contrasts: pd.DataFrame,
           domains: pd.DataFrame, pairs: pd.DataFrame | None = None) -> dict[str, str]:
    """Key -> printed text, for `\\pfl{<key>}`. Keys use the pool ids of POOL_KEY,
    `4k`/`16k` and `base`/`sft`/`dpo`:
      suite.<pool>.<yes|no>.<ctx>.<stage>, suite.<pool>.delta.<ctx>.<stage>
          Table 17's reasoning cells and its Delta row
      lang.<pool>.<yes|no>.<ctx>, lang.<pool>.delta.<ctx>
          Table 17's language cells and Delta row. Every delta and gap is
          computed from unrounded values
      <fl|perm|int>.<label>.<ctx>.<stage>[.ci|.q]
          Table 28's estimates, intervals and q-values; the label is a pool id,
          two pool ids for the permissive minus non-permissive contrasts
          (perm.mv-nemotron), and a permissive pool id for the interaction
      fl.min, fl.max, fl.q, fl.n            every front-loading effect
      perm.<pool>[.<ctx>][.post].<min|max|q>  a permissive pool against both
          non-permissive pools, at one context length or both, over all stages,
          or SFT and DPO only (.post)
      int.<pool>.<ctx>.<min|max|q>          the interaction over the three stages
      dom.<domain>.<ctx>.<stage>            front-loading effect on a domain, mean
          over the substrates
      ifeval.post.n, ifeval.post.positive   post-training cells, and those where
          front-loading raises instruction following
      lang.gap.<ctx>.<min|max>              non-permissive minus permissive language
          average, both front-loaded
      pair.<pool>-<pool>.<ctx>.<stage>[.ci|.q], pair.<pool>-<pool>.<ctx>.<min|max|q>
          one front-loaded permissive pool minus the other (pair.mv-comma), from
          permissive_pair_contrasts, one Benjamini-Hochberg family of its own
      n.substrates, n.runs, n.checkpoints, n.language, with .word variants
    A `.q` over several contrasts is the largest q-value, as `q \\le ...`."""
    out: dict[str, str] = {}
    suite = summary[summary["benchmark"] == "__suite__"].set_index(pa.FACTORIAL_KEYS)["mean"]
    lang = language_avg.set_index(["model_group", "front_loading", "context_length"])["language"]
    exact_lang = {}
    for pool in POOLS:
        p = POOL_KEY[pool]
        for ctx, c in CTX_KEY.items():
            for stage in pa.STAGES:
                no, yes = (float(suite.loc[(pool, fl, ctx, stage)]) for fl in (False, True))
                out[f"suite.{p}.no.{c}.{stage}"] = f"{no:.1f}"
                out[f"suite.{p}.yes.{c}.{stage}"] = f"{yes:.1f}"
                out[f"suite.{p}.delta.{c}.{stage}"] = _signed(yes - no)
            no, yes = (float(lang.loc[(pool, fl, ctx)]) for fl in (False, True))
            exact_lang[(pool, ctx)] = yes
            out[f"lang.{p}.no.{c}"] = f"{no:.1f}"
            out[f"lang.{p}.yes.{c}"] = f"{yes:.1f}"
            out[f"lang.{p}.delta.{c}"] = _signed(yes - no)

    for r in contrasts.itertuples():
        stem = (f"{FAMILY_KEY[r.family]}.{_label_key(r.family, r.label)}."
                f"{CTX_KEY[r.context_length]}.{r.stage}")
        out[stem] = _signed(r.estimate, 2)
        out[f"{stem}.ci"] = f"\\ensuremath{{[{r.lo:+.2f}, {r.hi:+.2f}]}}"
        out[f"{stem}.q"] = _q(r.q)

    def span(prefix: str, rows: pd.DataFrame) -> None:
        out[f"{prefix}.min"] = _plain(round(rows["estimate"].min(), 1))
        out[f"{prefix}.max"] = _plain(round(rows["estimate"].max(), 1))
        out[f"{prefix}.q"] = _q(rows["q"].max(), r"\le")

    fl = contrasts[contrasts["family"] == "front-loading"]
    span("fl", fl)
    _count(out, "fl.n", len(fl))
    for family, key in (("permissiveness", "perm"), ("interaction", "int")):
        fam = contrasts[contrasts["family"] == family]
        for pool in PERMISSIVE:
            mine = fam[fam["label"].str.split(" - ").str[0] == pool]
            if key == "perm":
                span(f"{key}.{POOL_KEY[pool]}.post", mine[mine["stage"] != "base"])
            for ctx, c in CTX_KEY.items():
                at = mine[mine["context_length"] == ctx]
                span(f"{key}.{POOL_KEY[pool]}.{c}", at)
                if key == "perm":
                    span(f"{key}.{POOL_KEY[pool]}.{c}.post", at[at["stage"] != "base"])

    dom = domains.pivot_table(index=["model_group", "context_length", "stage", "domain"],
                              columns="front_loading", values="score")
    effect = (dom[True] - dom[False]).reset_index(name="effect")
    for (ctx, stage, domain), rows in effect.groupby(["context_length", "stage", "domain"]):
        out[f"dom.{domain}.{CTX_KEY[ctx]}.{stage}"] = _signed(rows["effect"].mean())
    ifeval = effect[(effect["domain"] == "instruction") & (effect["stage"] != "base")]
    _count(out, "ifeval.post.n", len(ifeval))
    _count(out, "ifeval.post.positive", int((ifeval["effect"] > 0).sum()))

    for ctx, c in CTX_KEY.items():
        gaps = [exact_lang[(b, ctx)] - exact_lang[(a, ctx)]
                for a in PERMISSIVE for b in POOLS if b not in PERMISSIVE]
        out[f"lang.gap.{c}.min"] = _plain(round(min(gaps), 1))
        out[f"lang.gap.{c}.max"] = _plain(round(max(gaps), 1))

    if pairs is not None and len(pairs):
        for r in pairs.itertuples():
            stem = f"pair.{_label_key('permissiveness', r.label)}.{CTX_KEY[r.context_length]}.{r.stage}"
            out[stem] = _signed(r.estimate, 2)
            out[f"{stem}.ci"] = f"\\ensuremath{{[{r.lo:+.2f}, {r.hi:+.2f}]}}"
            out[f"{stem}.q"] = _q(r.q)
        for label, rows in pairs.groupby("label"):
            for ctx, c in CTX_KEY.items():
                span(f"pair.{_label_key('permissiveness', label)}.{c}", rows[rows["context_length"] == ctx])

    _count(out, "n.substrates", len(POOLS))
    _count(out, "n.runs", 2 * len(POOLS))
    _count(out, "n.language", 2 * len(POOLS) * len(CTX_KEY))
    _count(out, "n.checkpoints", 2 * len(POOLS) * len(CTX_KEY) * len(pa.STAGES))
    return dict(sorted(out.items()))


def write_macros(values: dict[str, str], out_dir: Path) -> Path:
    lines = [
        "% Generated by `mvpt render-frontloading` (mv_post_train). Do not edit by hand.",
        "% Numbers for the text, from the same values as Tables 5 and 28 to 31.",
        r"% Use \pfl{<key>}, for example \pfl{suite.mv.delta.16k.sft} or \pfl{fl.max}.",
        "% A key that does not exist prints in bold between question marks.",
    ]
    lines += macro_lines("pfl", values)
    path = out_dir / "frontloading_numbers.tex"
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(lines) + "\n")
    return path


def render_all(manifest_path: Path, data_dir: Path, out_dir: Path,
               art_dir: Path | None = None) -> list[Path]:
    df, scores, language = load(manifest_path, data_dir)
    keys = pa.FACTORIAL_KEYS
    summary = fs.benchmark_summary(scores, keys)
    language_avg = fs.language_average(language)
    contrasts = fs.factorial_contrasts(df, pools=POOLS, permissive=PERMISSIVE)
    pairs = fs.permissive_pair_contrasts(df, permissive=PERMISSIVE)

    tables_dir, figures_dir = out_dir / "tables", out_dir / "figures"
    tables = [
        wt.factorial_summary(summary, language_avg, tables_dir, pools=POOLS,
                             permissive=PERMISSIVE, note=SUMMARY_NOTE, pretraining="pretraining",
                             pool_words=POOL_WORDS, exact_deltas=True),
        wt.contrasts_table(contrasts, tables_dir, note=(
            r"Comma0.1 is a second permissive source pool, so the permissiveness and interaction "
            r"families include its contrasts, and each family's $q$-values are adjusted over "
            r"all of them. Table~\ref{tab:contrasts_language} gives the same contrasts on the "
            r"language suite."), pools=POOLS, pool_words=POOL_WORDS, rounding_note=False),
        wt.domain_robustness(scores, tables_dir, pools=POOLS, permissive=PERMISSIVE,
                             note=CONTROL_NOTE, claim=ROBUSTNESS_CLAIM,
                             protocol_note=ROBUSTNESS_NOTE, fit=False),
        wt.factorial_perbenchmark(summary, tables_dir, pools=POOLS, note=CONTROL_NOTE,
                                  pretraining="pretraining", pool_words=POOL_WORDS,
                                  panels=wt.DOMAIN_PANELS),
        wt.factorial_language(language, tables_dir, pools=POOLS, note=LANGUAGE_NOTE,
                              pool_words=POOL_WORDS, checkpoint_words=LANGUAGE_CHECKPOINTS,
                              fit=False),
    ]
    figures = [
        wf.factorial_main(fs.suite_average(scores, keys), language_avg, figures_dir, pools=POOLS,
                          pool_word=POOL_WORDS[1]),
        wf.frontloading_trajectory(contrasts, figures_dir, pools=POOLS, permissive=PERMISSIVE),
        wf.frontloading_domains(scores, figures_dir, pools=POOLS),
    ]
    numbers = write_macros(macros(summary, language_avg, contrasts,
                                  fs.domain_scores(scores, keys), pairs), tables_dir)
    written = tables + [numbers] + figures
    pdf = preview.build("Front-loading study with Comma0.1 (render-frontloading)", tables,
                        figures, out_dir / "preview.pdf")
    if pdf is not None:
        written.append(pdf)
    if art_dir is not None:
        art_dir.mkdir(parents=True, exist_ok=True)
        for name, frame in (("factorial_benchmark_summary", summary), ("contrasts", contrasts),
                            ("permissive_pairs", pairs),
                            ("equal_domain", fs.equal_domain_average(scores, keys)),
                            ("leave_one_domain_out", fs.leave_one_domain_out(scores, keys)),
                            ("domain_scores", fs.domain_scores(scores, keys))):
            path = art_dir / f"{name}.csv"
            frame.to_csv(path, index=False)
            written.append(path)
    return written


def main(manifest_path: Path, data_dir: Path, out_dir: Path, art_dir: Path) -> int:
    for path in render_all(manifest_path, data_dir, out_dir, art_dir):
        print(f"  wrote {path}")
    return 0
