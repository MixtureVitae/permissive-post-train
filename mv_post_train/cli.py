"""CLI for the MV post-training pipeline.

This is the sole CLI surface in the repo. Every operation — data parsing,
stat-sig analysis, paper-table / figure rendering — goes through
`mvpt <subcommand>`. No standalone scripts.

Subcommands:
  extract-logs             unzip logs.zip into a usable directory
  parse-phase1             logs/ -> items_long.parquet + aggregates.parquet (Phase 1 only)
  parse-phase1-dump        HF dump -> phase1_dump_items.parquet (gsm8k, the Phase 1 side)
  parse-phase2             HF eval-responses parquet(s) -> phase2_items.parquet (Phase 2 only)
  parse-phase2-logs        logs/ OT30K dirs -> phase2_logs_*.parquet (all 11 benchmarks)
  analyze                  compute pairwise + suite-average artifacts (CSVs)
  aime-decoding-variance   score the temp-0.7 AIME replicates -> decoding-variance CSVs
  seed-variance            3-seed temp-0.7 Phase 1 replicates -> seed-variance CSVs
  released-scores          compare recovered IFEval/HumanEval/MBPP to the released grid
  contest-math-sig         item-level significance for sampled AIME24/25 + AMC23
  phase2-reconcile         check recomputed Phase 2 cells against submission Table 1
  phase2-transition        within-model paired tests for the Phase 1 -> OT30K step
  phase2-transition-dump   the same step for contest math, both ends from the dump
  phase2-vs-baselines      MV vs the matched-compute references AT the OT30K endpoint
  math-composite           all five math benchmarks pooled into one test per reference
  corpus-variation         vary the reasoning corpus (OT3/AceReason/NC/SR) with the base fixed
  ifeval-rescore           re-score IFEval from the response dump (vendored checker)
  code-rescore             re-score HumanEval + MBPP by executing generated code
  render-workshop          render the workshop paper's 11 generated tables + 5 figures
  check-workshop           compare rendered workshop tables with the paper's .tex, number by number
  workshop-release-check   compare the workshop per-sample input with the released 3-seed grid
  build                    run every paper target (mv_post_train/targets.py), or check them
  check-paper              compare a paper source tree with the generated files
  export-paper             copy the generated files into a paper source tree
  fetch-inputs             download the pinned inputs (mv_post_train/inputs.json) that are missing
  check-inputs             hash every local input and compare it with its pin
  fetch-phase2-records     download the three-seed Phase 2 records at a fixed revision
  measure-phase2-answers   the length in tokens of every sampled Phase 2 answer (R14-lengths)
  measure-phase2-tokens    the training tokens of each Phase 2 reasoning dataset
  snapshot-checkpoints     save Hugging Face records for every checkpoint and its parents
  build-manifest           build results/manifest.json, the authoritative reasoning results
  render-phase2            render the Phase 2 tables, Figure 2 and text macros from the manifest
  render-phase2-analyses   render the Phase 2 analyses: intervals, domain changes, cut-off answers
  render-overlap           render Table 4 without the identified benchmark items, and the audit's tables
  render-frontloading      render the revision's front-loading study, with Comma0.1 added
  render-language-intervals  render language-suite changes with question-level intervals
  render-statistics        render the statistics appendix (Tables 26 and 27) and its text macros
  render-older-tables      render Tables 7, 8, 9 and 15, the older experiments, and their text macros
  render-zoo               render Tables 1 and 6 (Phase 1 for all models) and their text macros
  render-language-suite    render Tables 10 to 14 and 16 (the language suite) and their text macros
  render-setup             render Tables 17 to 25 (setup and evaluation settings) and their text macros
  transcribe-submission    transcribe the submitted paper's results tables from its PDF
  render-reconciliation    compare every submitted results-table cell with the revision

Default I/O layout (relative to current working dir):
  ./data/      <- items_long.parquet, aggregates.parquet, phase2_items.parquet
  ./artifacts/ <- pairwise_*.csv, suite_avg_diff_*.csv, per_task_accuracy.csv
  ./results/   <- every generated file of the paper (`mvpt build`, docs/paper_build.md)
"""
from __future__ import annotations

import argparse
import sys
import zipfile
from pathlib import Path

from . import paths
from .data import (
    parse_phase1,
    checkpoint_guard,
    parse_phase1_dump,
    parse_phase2,
    parse_phase2_logs,
)
from .stats import (
    aime_decoding_variance,
    analyze,
    code_score,
    ifeval_score,
    contest_math_sig,
    phase2_transition,
    corpus_variation,
    math_composite,
    phase2_transition_dump,
    phase2_vs_baselines,
    released_scores,
    seed_variance,
)


def _add_data_args(p: argparse.ArgumentParser) -> None:
    p.add_argument("--data-dir", type=Path, default=Path("data"),
                   help="Directory for items_long.parquet / aggregates.parquet "
                        "/ phase2_items.parquet (default: ./data).")


def _add_art_args(p: argparse.ArgumentParser) -> None:
    p.add_argument("--art-dir", type=Path, default=Path("artifacts"),
                   help="Directory for analysis CSV artifacts (default: ./artifacts).")


def _extract_zip(zip_path: Path, out_dir: Path) -> Path:
    """Extract logs.zip; return the directory that holds per-model subdirs."""
    out_dir.mkdir(exist_ok=True, parents=True)
    print(f"extracting {zip_path} -> {out_dir} ...")
    with zipfile.ZipFile(zip_path) as zf:
        zf.extractall(out_dir)
    inner = out_dir / "logs"
    if inner.is_dir():
        return inner
    return out_dir


LATEST_RUN_HELP = ("For a checkpoint evaluated more than once, keep HumanEval's latest run, "
                   "like every other task. By default HumanEval keeps its run with the highest "
                   "python_pass@1, the old paper convention.")


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        prog="mvpt",
        description="MV post-training pipeline: parse logs, run stat-sig "
                    "analysis, render paper tables.",
    )
    sub = p.add_subparsers(dest="cmd", required=True)

    # ---- data layer
    pl = sub.add_parser("extract-logs", help="Unzip logs.zip into a directory.")
    pl.add_argument("zip_path", type=Path)
    pl.add_argument("--out-dir", type=Path, default=Path("extracted"))

    pp = sub.add_parser("parse-phase1", help="Parse extracted logs -> parquets.")
    pp.add_argument("logs_dir", type=Path)
    pp.add_argument("--latest-run", action="store_true", help=LATEST_RUN_HELP)
    pp.add_argument("--mean-unbroken-runs", action="store_true",
                    help="For a checkpoint evaluated more than once, average each task's complete "
                         "runs, leaving out runs that score near zero where another run does not "
                         "(data/run_rule.py). The rule of Tables 7, 9 and 27.")
    pp.add_argument("--gsm8k-flexible", action="store_true",
                    help="Score GSM8K with the flexible-extract parser, as Tables 7, 9 and 27 do, "
                         "instead of strict matching.")
    _add_data_args(pp)
    _add_art_args(pp)

    cg = sub.add_parser(
        "checkpoint-audit",
        help="Recover each dump cell's checkpoint from source_dir and flag "
             "cells whose stage label disagrees with it. Run before trusting "
             "any (base_model, stage) label.",
    )
    cg.add_argument("dump", type=Path, help="eval_responses.parquet")
    _add_art_args(cg)

    p1d = sub.add_parser(
        "parse-phase1-dump",
        help="Recover Phase 1 item rows from the HF response dump for "
             "benchmarks logs.zip carries only as aggregates (gsm8k). Without "
             "these the Phase 1 -> OT30K transition cannot pair gsm8k.",
    )
    p1d.add_argument("hf_main", type=Path, help="eval_responses.parquet")
    p1d.add_argument("--no-validate", action="store_true",
                     help="Skip the per-cell cross-check against logs.zip's "
                          "gsm8k aggregates.")
    _add_data_args(p1d)

    pc = sub.add_parser("parse-phase2", help="HF parquet(s) -> phase2_items.parquet.")
    pc.add_argument("hf_main", type=Path)
    pc.add_argument("--mv-best", type=Path, default=None)
    _add_data_args(pc)

    pcl = sub.add_parser(
        "parse-phase2-logs",
        help="Parse the OT30K model dirs in extracted logs -> "
             "phase2_logs_items.parquet + phase2_logs_aggregates.parquet. "
             "Complement of parse-phase1; covers all 11 reasoning benchmarks "
             "(the HF dump carries only 7).",
    )
    pcl.add_argument("logs_dir", type=Path)
    _add_data_args(pcl)
    _add_art_args(pcl)

    # ---- stats layer
    an = sub.add_parser("analyze", help="Compute pairwise + suite-average artifacts.")
    an.add_argument(
        "--phase2-source", choices=["hf", "logs"], default="hf",
        help="Which Phase 2 parquet to load. 'hf' (default) reproduces the "
             "manuscript's Table 1 across 5 scorable tasks; 'logs' covers all "
             "11 benchmarks but does not reproduce Table 1 "
             "(see docs/phase2-aime-provenance.md).",
    )
    _add_data_args(an)
    _add_art_args(an)

    cse = sub.add_parser(
        "code-rescore",
        help="Re-score HumanEval and MBPP from the response dump by EXECUTING "
             "the generated code in the vendored evalchemy/human_eval sandbox. "
             "Runs model-generated code -- read the module docstring first.",
    )
    cse.add_argument("dump", type=Path, help="eval_responses.parquet")
    cse.add_argument("--timeout", type=float, default=10.0,
                     help="Per-candidate wall-clock limit, seconds.")
    cse.add_argument("--workers", type=int, default=8)
    cse.add_argument("--tasks", nargs="+", choices=["HumanEval", "MBPP"],
                     default=["HumanEval", "MBPP"],
                     help="Which benchmarks to score. Scoring a subset keeps "
                          "the other's existing rows in code_rescored.csv "
                          "rather than discarding them.")
    cse.add_argument("--resume", action="store_true",
                     help="Skip cells already present in code_rescored.csv. "
                          "Cells recorded with status=error are retried. Use "
                          "after an interrupted run rather than paying for "
                          "every cell's sandboxed execution again.")
    cse.add_argument("--emit-items", action="store_true",
                     help="Also persist the per-problem verdicts as item rows "
                          "in recovered_items.parquet, so the benchmark can "
                          "take part in the paired transition tests.")
    _add_data_args(cse)
    _add_art_args(cse)

    sv = sub.add_parser(
        "seed-variance",
        help="Seed-to-seed variance from the 3-seed temperature-0.7 Phase 1 "
             "replicate runs. The only genuine replicate-seed evidence in the "
             "project; every other estimate here is item-level or two-run.",
    )
    sv.add_argument("--seed-dir", type=Path, default=Path("data/sampling07"),
                    help="Directory holding final_scores.csv / metric_scores.csv "
                         "from phase1/sampling-0.7 (default: ./data/sampling07).")
    _add_art_args(sv)

    cms = sub.add_parser(
        "contest-math-sig",
        help="Item-level paired and unpaired tests for MV vs each reference on "
             "the sampled AIME24 / AIME25 / AMC23 runs, with BH correction "
             "reported both within-benchmark and across all tests.",
    )
    cms.add_argument("--seed-dir", type=Path, default=Path("data/sampling07"),
                     help="Directory holding per_sample_scores.csv "
                          "(default: ./data/sampling07).")
    _add_art_args(cms)

    rs = sub.add_parser(
        "released-scores",
        help="Compare our recovered IFEval / HumanEval / MBPP columns against "
             "eval_scores_wide.csv from the response-dump release. Needs "
             "ifeval-rescore and/or code-rescore to have run first.",
    )
    rs.add_argument("wide", type=Path,
                    help="eval_scores_wide.csv from the release.")
    _add_art_args(rs)

    ife = sub.add_parser(
        "ifeval-rescore",
        help="Re-score IFEval from the released response dump using the "
             "vendored official checker. Recovers a column the dump ships "
             "without verdicts.",
    )
    ife.add_argument("dump", type=Path, help="eval_responses.parquet")
    ife.add_argument("prompts", type=Path,
                     help="ifeval_input_data.jsonl from google/IFEval.")
    ife.add_argument("--emit-items", action="store_true",
                     help="Also persist the per-prompt verdicts as item rows "
                          "in recovered_items.parquet, so IFEval can take part "
                          "in the paired Phase 1 -> OT30K transition tests.")
    _add_data_args(ife)
    _add_art_args(ife)

    pt = sub.add_parser(
        "phase2-transition",
        help="Within-model paired tests for the Phase 1 -> OT30K transition "
             "(the comparison Figure 3 actually makes).",
    )
    _add_data_args(pt)
    _add_art_args(pt)

    ptd = sub.add_parser(
        "phase2-transition-dump",
        help="Same as phase2-transition for the four contest-math tasks, but "
             "with BOTH ends scored from the HF response dump. Use these "
             "numbers: the default table takes the Phase 1 side from logs.zip, "
             "which disagrees with the dump by up to 14 pp on MV/base MATH500.",
    )
    ptd.add_argument("responses", type=Path,
                     help="Path to the dump's eval_responses.parquet.")
    _add_art_args(ptd)

    pvb = sub.add_parser(
        "phase2-vs-baselines",
        help="Paired MV-minus-reference contrasts AT each OT30K stage -- the "
             "between-model claim, as opposed to phase2-transition's "
             "within-model one. Covers GSM8K, MATH500, contest math and "
             "IFEval; the other five columns of the suite have no reference "
             "rows in the dump at any OT30K stage.",
    )
    pvb.add_argument("responses", type=Path,
                     help="Path to the dump's eval_responses.parquet.")
    pvb.add_argument("--ifeval-prompts", type=Path,
                     default=Path("data/laion/ifeval_input_data.jsonl"),
                     help="IFEval prompt file, for the IFEval contrasts. "
                          "Omit the file and IFEval is skipped, not faked.")
    _add_art_args(pvb)

    pmc = sub.add_parser(
        "math-composite",
        help="Pool all five math benchmarks into ONE paired contrast per "
             "(stage, reference), macro- and micro-weighted, with a "
             "leave-one-benchmark-out check. Gives the contest-math columns a "
             "test they can actually pass, since AIME24/25 cannot clear a "
             "multiplicity correction on their own 30 items.",
    )
    pmc.add_argument("responses", type=Path,
                     help="Path to the dump's eval_responses.parquet.")
    _add_art_args(pmc)

    pcv = sub.add_parser(
        "corpus-variation",
        help="Paired transitions for all four reasoning corpora (OpenThoughts3, "
             "AceReason, Nemotron-Cascade, Superior-Reasoning) on both the SFT "
             "and DPO routes, for every base. Gives the corpus-variation table "
             "intervals instead of released-grid subtractions.",
    )
    pcv.add_argument("responses", type=Path,
                     help="Path to the dump's eval_responses.parquet.")
    _add_art_args(pcv)

    pr = sub.add_parser(
        "phase2-reconcile",
        help="Compare recomputed Phase 2 cells against submission Table 1, "
             "cell by cell. Run before quoting any Phase 2 interval.",
    )
    _add_data_args(pr)
    _add_art_args(pr)

    av = sub.add_parser(
        "aime-decoding-variance",
        help="Score the temperature-0.7 AIME24/AIME25 replicates in the "
             "best-responses parquet and report dispersion across the three "
             "independent samples. Measures decoding variance only — these are "
             "not the manuscript's greedy cells.",
    )
    av.add_argument("mv_best", type=Path,
                    help="mv_aime24_amc23_best_responses.parquet (2026-07-27 revision "
                         "or later — needs the `decoding` column).")
    _add_art_args(av)

    # ---- workshop paper (Protocol A per-sample records + transcribed Protocol B)
    rw = sub.add_parser(
        "render-workshop",
        help="Render the workshop paper's 11 generated tables and 5 figures from the "
             "Protocol A per-sample records (see docs/workshop.md).",
    )
    rw.add_argument("--data-dir", type=Path, default=Path("data/workshop"),
                    help="Directory holding canonical_reasoning_per_sample.csv and the "
                         "language CSV (default: ./data/workshop).")
    rw.add_argument("--out-dir", type=Path, default=Path(paths.WORKSHOP_TABLES_DIR),
                    help=f"Directory for the .tex tables (default: ./{paths.WORKSHOP_TABLES_DIR}).")
    rw.add_argument("--figures-dir", type=Path, default=Path(paths.WORKSHOP_FIGURES_DIR),
                    help=f"Directory for the PDF figures (default: ./{paths.WORKSHOP_FIGURES_DIR}).")
    rw.add_argument("--art-dir", type=Path, default=Path(paths.WORKSHOP_ARTIFACTS_DIR),
                    help="Directory for the summary CSVs behind the tables "
                         f"(default: ./{paths.WORKSHOP_ARTIFACTS_DIR}).")

    cw = sub.add_parser(
        "check-workshop",
        help="Compare rendered workshop tables with the paper's .tex files, number by "
             "number. Exits 1 on any difference.",
    )
    cw.add_argument("paper_dir", type=Path,
                    help="The paper's tables directory, e.g. a checkout of the artifacts "
                         "repo's latex/tables.")
    cw.add_argument("--ours", type=Path, default=Path(paths.WORKSHOP_TABLES_DIR),
                    help=f"Rendered tables (default: ./{paths.WORKSHOP_TABLES_DIR}).")

    wr = sub.add_parser(
        "workshop-release-check",
        help="Compare the workshop per-sample input with the released three-seed grid, "
             "item by item, wherever both have the item.",
    )
    wr.add_argument("--data-dir", type=Path, default=Path("data/workshop"))
    wr.add_argument("--release", type=Path,
                    default=Path("data/paper_decoding_3seeds/per_sample_scores.csv"),
                    help="per_sample_scores.csv of the released grid (README input 5).")
    wr.add_argument("--art-dir", type=Path, default=Path(paths.WORKSHOP_ARTIFACTS_DIR))

    # ---- results manifest
    rz = sub.add_parser("render-zoo",
                        help="Render Tables 1 and 6 (Phase 1 for all models, Protocol A) and the "
                             "\\pzoo macros for the text.")
    rz.add_argument("--manifest", type=Path, default=Path(paths.MANIFEST))
    rz.add_argument("--out-dir", type=Path, default=Path(paths.ZOO_DIR))

    rls = sub.add_parser("render-language-suite",
                         help="Render Tables 10 to 14 and 16 (the language suite, one run per "
                              "checkpoint) and the \\plang macros for the text.")
    rls.add_argument("--manifest", type=Path, default=Path(paths.MANIFEST))
    rls.add_argument("--out-dir", type=Path, default=Path(paths.LANGUAGE_SUITE_DIR))

    rsu = sub.add_parser("render-setup",
                         help="Render Tables 17 to 25 (setup and evaluation settings) from "
                              "config/paper_setup.toml and the records, and the \\psetup macros.")
    rsu.add_argument("--manifest", type=Path, default=Path(paths.MANIFEST))
    rsu.add_argument("--out-dir", type=Path, default=Path(paths.SETUP_DIR))

    ts = sub.add_parser("transcribe-submission",
                        help="Transcribe the eleven results tables of the submitted paper from its "
                             "PDF into the file of input T-submission. Needs PyMuPDF.")
    ts.add_argument("pdf", type=Path, help="The PDF of the paper as submitted for review.")
    ts.add_argument("--out", type=Path, default=Path("submission/submission_tables.csv"))

    rrc = sub.add_parser("render-reconciliation",
                         help="Compare every cell of the submitted results tables with the "
                              "revision's tables, with the reason for each change. Not part of "
                              "the paper. Reads the tables the other render commands wrote.")
    rrc.add_argument("--manifest", type=Path, default=Path(paths.MANIFEST))
    rrc.add_argument("--results-root", type=Path, default=Path("."),
                     help="The folder that holds results/ with the revision's generated tables.")
    rrc.add_argument("--out-dir", type=Path, default=Path(paths.RECONCILIATION_DIR))

    bd = sub.add_parser("build",
                        help="Run every target that makes a file the paper uses (or the ones "
                             "named), in dependency order. See docs/paper_build.md.")
    bd.add_argument("targets", nargs="*", help="Target names from mv_post_train/targets.py.")
    bd.add_argument("--check", action="store_true",
                    help="Build into a temporary folder and compare every file byte for byte "
                         "(PNG previews pixel for pixel) with the committed ones under results/. "
                         "Exits 1 on a difference.")
    cp = sub.add_parser("check-paper",
                        help="Compare a paper source tree with the generated files: byte "
                             "equality, macro keys, hand-made tables and figures, and numbers "
                             "typed into the prose.")
    cp.add_argument("paper_dir", type=Path, help="The paper's sources (with main_neurips_2026.tex).")
    cp.add_argument("--numbers", action="store_true",
                    help="List every number that differs from the generated files, with its "
                         "reason. Exits 1 if a changed number has no reason.")
    cp.add_argument("--markdown", action="store_true",
                    help="With --numbers, print the list as a Markdown table for a pull request.")
    cp.add_argument("--by-row", action="store_true",
                    help="With --markdown, one line per table row, listing its changed cells.")
    ep = sub.add_parser("export-paper",
                        help="Copy every generated file the paper uses into a paper source tree.")
    ep.add_argument("paper_dir", type=Path)

    fi = sub.add_parser("fetch-inputs",
                        help="Download the pinned inputs that are missing (all paper inputs, "
                             "or the IDs given), and check each against its SHA-256. "
                             "See docs/inputs.md.")
    fi.add_argument("ids", nargs="*", help="Input IDs from mv_post_train/inputs.json.")
    fi.add_argument("--analyses", action="store_true",
                    help="Also fetch the inputs that only the analyses in docs/ use.")
    ci = sub.add_parser("check-inputs",
                        help="Hash every local input (or the IDs given) and compare it with "
                             "its pin. Exits 1 if a paper input is missing or differs.")
    ci.add_argument("ids", nargs="*", help="Input IDs from mv_post_train/inputs.json.")

    fp = sub.add_parser("fetch-phase2-records",
                        help="Download the three-seed Phase 2 records (eval_scores and the "
                             "small columns of eval_responses) at a fixed dataset revision.")
    fp.add_argument("--out-dir", type=Path, default=Path(paths.PHASE2_3SEED_DIR))
    fp.add_argument("--revision", default=None,
                    help="Dataset revision (default: the one the manifest pins).")

    ma = sub.add_parser("measure-phase2-answers",
                        help="Write the length of every sampled Phase 2 answer, in bytes and in "
                             "its checkpoint's tokens, and whether the grader found an answer "
                             "(input R14-lengths). Reads the 10 GB answer file (R14-responses) "
                             "and downloads each checkpoint's tokenizer.json. Needs the `measure` "
                             "extra.")
    ma.add_argument("--responses", type=Path, default=None,
                    help="eval_responses.parquet (default: R14-responses' local place).")
    ma.add_argument("--manifest", type=Path, default=Path(paths.MANIFEST))
    ma.add_argument("--snapshot", type=Path, default=Path(paths.HF_CHECKPOINTS))
    ma.add_argument("--out-dir", type=Path, default=Path(paths.PHASE2_3SEED_DIR))

    pt = sub.add_parser("measure-phase2-tokens",
                        help="Count the tokens and supervised tokens of each Phase 2 "
                             "dataset in LLaMA-Factory's tokenized caches "
                             "(stats/phase2_tokens.py).")
    pt.add_argument("--tokenized-dir", type=Path, default=None,
                    help="The folder holding the caches of phase2_tokens.CACHES.")
    pt.add_argument("--ot3-release-dir", type=Path, default=None,
                    help="Read the OT3 30K as released (phase2_tokens.OT3_RELEASE, its "
                         "parquet files) instead: what the cut removed, how the answers end. "
                         "Needs --tokenizer and the `measure` extra.")
    pt.add_argument("--tokenizer", type=Path, default=None,
                    help="The open-sci tokenizer.json, for --ot3-release-dir.")
    pt.add_argument("--out-dir", type=Path, default=Path(paths.PHASE2_TOKENS_DIR))

    qo = sub.add_parser("measure-question-overlap",
                        help="Sample the questions of AceReason, Nemotron-Cascade and "
                             "Superior-Reasoning and measure how much of each is in an "
                             "OpenThoughts3 question with corpus-assay "
                             "(stats/question_overlap.py). Needs the OT3-1.2M shards in "
                             "--raw-dir (28 GB), network access and the `overlap` extra.")
    qo.add_argument("--raw-dir", type=Path, default=Path(paths.OVERLAP_OT3_RAW_DIR))
    qo.add_argument("--out-dir", type=Path, default=Path(paths.QUESTION_OVERLAP_DIR))
    qo.add_argument("--subsets", nargs="*", default=None,
                    help="Only these corpora or subsets (default: all of SUBSETS).")
    qo.add_argument("--trained-dir", type=Path, default=None,
                    help="Match every question of the 30K samples the Phase 2 runs trained "
                         "on (question_overlap.TRAINED, folders in this directory) instead "
                         "of sampling the releases. Needs the reference and index of a "
                         "run without it.")

    mo = sub.add_parser("measure-mv-question-overlap",
                        help="Match every question of the 30K reasoning samples trained on "
                             "against MixtureVitae's other reasoning shards with corpus-assay "
                             "(stats/mv_question_overlap.py): `prepare` the index, scan each "
                             "`shard` (183 GB in all; one cluster array task each), `merge`. "
                             "Needs the `overlap` extra.")
    mo.add_argument("step", choices=["prepare", "shard", "merge"])
    mo.add_argument("name", nargs="?", default=None,
                    help="shard: the shard's name (mv_question_overlap.SHARDS, e.g. math-0).")
    mo.add_argument("--trained-dir", type=Path, default=None,
                    help="prepare: the folders of question_overlap.TRAINED.")
    mo.add_argument("--ot3-release-dir", type=Path, default=None,
                    help="prepare: the OT3 30K as released (phase2_tokens.OT3_RELEASE), "
                         "instead of --trained-dir; use its own --out-dir.")
    mo.add_argument("--shard-file", type=Path, default=None,
                    help="shard: the downloaded .jsonl.gz, checked against its SHA-256.")
    mo.add_argument("--workers", type=int, default=None)
    mo.add_argument("--ot3-dir", type=Path, default=Path(paths.QUESTION_OVERLAP_DIR),
                    help="merge: where question_overlap_trained.parquet is, to add the OT3 "
                         "matches.")
    mo.add_argument("--out-dir", type=Path, default=Path(paths.MV_QUESTION_OVERLAP_DIR))

    to = sub.add_parser("measure-training-data-overlap",
                        help="Match the post-training data with a public release (long_sft, "
                             "Tulu3 SFT and DPO) against the paper's 22 benchmarks with "
                             "corpus-assay (stats/training_data_overlap.py). Needs the logs "
                             "with the GPQA Diamond items, network access (5 GB of data and "
                             "4.4 GB of LiveCodeBench streamed) and the `overlap` extra.")
    to.add_argument("--logs", type=Path, default=Path("artifacts/logs.zip"))
    to.add_argument("--out-dir", type=Path, default=Path(paths.TRAINING_DATA_OVERLAP_DIR))
    to.add_argument("--datasets", nargs="*", default=None,
                    help="Only these datasets (default: all of DATASETS).")
    to.add_argument("--reasoning-questions", action="store_true",
                    help="Scan the OpenThoughts3 questions and the reasoning-corpus samples of "
                         "measure-question-overlap instead.")

    sb = sub.add_parser("scan-benchmarks",
                        help="Match the paper's benchmarks against the training data only a "
                             "cluster holds, the Phase 2 30K samples as trained and as written "
                             "and MixtureVitae's reasoning shards (stats/benchmark_scan.py): "
                             "decode the Phase 2 caches (`texts`), `scan` each source (one "
                             "cluster task each), `merge`. Needs the `overlap` extra.")
    sb.add_argument("step", choices=["texts", "scan", "merge"])
    sb.add_argument("source", nargs="?", default=None,
                    help="scan: a benchmark_scan.SOURCES entry, e.g. phase2:AceReason, "
                         "sample:AceReason or mv:math-0.")
    sb.add_argument("--file", type=Path, default=None,
                    help="scan: the decoded cache, the folder of the 30K samples, or the shard.")
    sb.add_argument("--tokenized-dir", type=Path, default=None,
                    help="texts: the folder holding phase2_tokens.CACHES.")
    sb.add_argument("--tokenizer", type=Path, default=None,
                    help="texts: a folder with the open-sci tokenizer.")
    sb.add_argument("--work-dir", type=Path, default=Path(paths.BENCHMARK_SCAN_DIR))
    sb.add_argument("--out-dir", type=Path, default=Path(paths.TRAINING_DATA_OVERLAP_DIR),
                    help="merge: where benchmark_scan.parquet goes.")
    sb.add_argument("--workers", type=int, default=None)

    os_ = sub.add_parser("measure-overlap-sensitivity",
                         help="Recompute the Phase 2, factorial and language comparisons without "
                              "the benchmark items found in the training data "
                              "(stats/overlap_sensitivity.py). Needs the outputs of "
                              "measure-training-data-overlap, with and without "
                              "--reasoning-questions.")
    os_.add_argument("--manifest", type=Path, default=Path(paths.MANIFEST))
    os_.add_argument("--out-dir", type=Path, default=Path(paths.TRAINING_DATA_OVERLAP_DIR))
    os_.add_argument("--phase2-dir", type=Path, default=Path(paths.PHASE2_3SEED_DIR))
    os_.add_argument("--data-dir", type=Path, default=Path("data/workshop"))
    os_.add_argument("--log-dir", type=Path, default=Path(paths.LANGUAGE_LOGS_DIR))

    mp = sub.add_parser("build-memorization-probes",
                        help="Write the CommonsenseQA memorization probes that "
                             "mv_post_train/gpu/score_texts.py scores on a GPU "
                             "(stats/memorization.py). Needs the outputs of "
                             "measure-overlap-sensitivity.")
    mp.add_argument("--out-dir", type=Path, default=Path(paths.TRAINING_DATA_OVERLAP_DIR))

    mm = sub.add_parser("measure-memorization",
                        help="Compare the checkpoints' scores on the memorization probes: "
                             "trained-on against clean questions across each Tulu3 SFT step, "
                             "and the bases' validation against test questions "
                             "(stats/memorization.py). Needs <out-dir>/memorization/scores.")
    mm.add_argument("--out-dir", type=Path, default=Path(paths.TRAINING_DATA_OVERLAP_DIR))

    sc = sub.add_parser("snapshot-checkpoints",
                        help="Save Hugging Face records (repos, weight hashes, model cards, "
                             "training summaries) for every checkpoint and its parents. "
                             "Needs network access.")
    sc.add_argument("--workshop-dir", type=Path, default=Path("data/workshop"))
    sc.add_argument("--phase2-dir", type=Path, default=Path(paths.PHASE2_3SEED_DIR))
    sc.add_argument("--out", type=Path, default=Path(paths.HF_CHECKPOINTS))
    sc.add_argument("--update", action="store_true",
                    help="Keep the records already in --out and fetch only new checkpoints.")

    bm = sub.add_parser("build-manifest",
                        help="Build the results manifest from the pinned inputs and the "
                             "checkpoint snapshot (offline). Exits 1 if an input check fails.")
    bm.add_argument("--workshop-dir", type=Path, default=Path("data/workshop"))
    bm.add_argument("--phase2-dir", type=Path, default=Path(paths.PHASE2_3SEED_DIR))
    bm.add_argument("--snapshot", type=Path, default=Path(paths.HF_CHECKPOINTS))
    bm.add_argument("--out", type=Path, default=Path(paths.MANIFEST))

    rp = sub.add_parser("render-phase2",
                        help="Render the paper's Phase 2 tables, Figures 1 and 2 and the "
                             "macros for the numbers in the text from the results manifest. "
                             "Exits 1 if a table asks for a result or change the manifest "
                             "does not allow.")
    rp.add_argument("--manifest", type=Path, default=Path(paths.MANIFEST))
    rp.add_argument("--phase2-dir", type=Path, default=Path(paths.PHASE2_3SEED_DIR),
                    help="The per-question Phase 2 records, for the MATH500 intervals.")
    rp.add_argument("--tables-dir", type=Path, default=Path(paths.PHASE2_TABLES_DIR))
    rp.add_argument("--figures-dir", type=Path, default=Path(paths.PHASE2_FIGURES_DIR))
    rp.add_argument("--art-dir", type=Path, default=Path(paths.PHASE2_ARTIFACTS_DIR),
                    help="Where to write the comparison with the submitted numbers.")

    ro = sub.add_parser("render-overlap",
                        help="Render the benchmark-overlap audit in the paper: Table 4 without the "
                             "identified items, the tables of Appendices A.3 and E, and the \\povl "
                             "macros (render/overlap.py, inputs O1 to O5).")
    ro.add_argument("--manifest", type=Path, default=Path(paths.MANIFEST))
    ro.add_argument("--phase2-dir", type=Path, default=Path(paths.PHASE2_3SEED_DIR))
    ro.add_argument("--out-dir", type=Path, default=Path(paths.OVERLAP_DIR))

    ra = sub.add_parser("render-phase2-analyses",
                        help="Render the Phase 2 analyses for the camera-ready: intervals for "
                             "Table 2's suite changes, the OT3 exclusion's difference of gains, "
                             "changes by domain, and answers cut off or without an answer "
                             "(docs/phase2_analyses.md).")
    ra.add_argument("--manifest", type=Path, default=Path(paths.MANIFEST))
    ra.add_argument("--phase2-dir", type=Path, default=Path(paths.PHASE2_3SEED_DIR),
                    help="The per-question Phase 2 records and the answer lengths.")
    ra.add_argument("--out-dir", type=Path, default=Path(paths.PHASE2_ANALYSES_DIR))

    rf = sub.add_parser("render-frontloading",
                        help="Render the revision's front-loading study (Tables 5 and 28 to 31, "
                             "Figures 3 to 5) with Comma0.1 as a fourth web substrate, plus a "
                             "compiled preview.pdf.")
    rf.add_argument("--manifest", type=Path, default=Path(paths.MANIFEST))
    rf.add_argument("--data-dir", type=Path, default=Path("data/workshop"),
                    help="The Protocol A records (R13 and R15), which the contrasts resample. "
                         "Downloaded here if missing.")
    rf.add_argument("--out-dir", type=Path, default=Path(paths.FRONTLOADING_DIR))
    rf.add_argument("--art-dir", type=Path, default=Path(paths.FRONTLOADING_ARTIFACTS_DIR))

    li = sub.add_parser("render-language-intervals",
                        help="Render each training step's change on the language suite, with "
                             "95%% paired intervals over the questions, from the per-question "
                             "lm-eval logs, plus a compiled preview.pdf.")
    li.add_argument("--manifest", type=Path, default=Path(paths.MANIFEST))
    li.add_argument("--log-dir", type=Path, default=Path(paths.LANGUAGE_LOGS_DIR),
                    help="Where the two log zips are, or are downloaded to from the project "
                         "bucket, and where the parsed scores are cached.")
    li.add_argument("--out-dir", type=Path, default=Path(paths.LANGUAGE_INTERVALS_DIR))

    rs = sub.add_parser("render-statistics",
                        help="Render the statistics appendix: Tables 26 and 27 under Protocol A "
                             "from the per-question records, the power figures and the text "
                             "macros, plus a compiled preview.pdf.")
    rs.add_argument("--manifest", type=Path, default=Path(paths.MANIFEST))
    rs.add_argument("--data-dir", type=Path, default=Path("data/workshop"),
                    help="The Protocol A records (R13 and R18).")
    rs.add_argument("--out-dir", type=Path, default=Path(paths.STATISTICS_DIR))

    ro = sub.add_parser("render-older-tables",
                        help="Render Tables 7, 8 and 9 (100B ablation, decontamination, 0.4B) "
                             "under Protocol A and Table 15 (masking) on the older setup, from "
                             "the results manifest, plus text macros, Table 15's runs and a "
                             "compiled preview.pdf.")
    ro.add_argument("--manifest", type=Path, default=Path(paths.MANIFEST))
    ro.add_argument("--out-dir", type=Path, default=Path(paths.OLDER_TABLES_DIR))

    return p


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)

    if args.cmd == "extract-logs":
        _extract_zip(args.zip_path, args.out_dir)
        return 0

    if args.cmd == "parse-phase1":
        return parse_phase1.main(args.logs_dir, args.data_dir, args.art_dir, args.latest_run,
                                 args.mean_unbroken_runs, args.gsm8k_flexible)

    if args.cmd == "checkpoint-audit":
        return checkpoint_guard.main(args.dump, args.art_dir)

    if args.cmd == "parse-phase1-dump":
        return parse_phase1_dump.main(args.hf_main, args.data_dir,
                                      do_validate=not args.no_validate)

    if args.cmd == "parse-phase2":
        return parse_phase2.main(args.hf_main, args.mv_best, args.data_dir)

    if args.cmd == "parse-phase2-logs":
        return parse_phase2_logs.main(args.logs_dir, args.data_dir, args.art_dir)

    if args.cmd == "analyze":
        return analyze.main(args.data_dir, args.art_dir, args.phase2_source)

    if args.cmd == "code-rescore":
        return code_score.main(args.dump, args.art_dir, args.timeout,
                               args.workers, tuple(args.tasks),
                               args.data_dir if args.emit_items else None,
                               args.resume)

    if args.cmd == "contest-math-sig":
        return contest_math_sig.main(args.seed_dir, args.art_dir)

    if args.cmd == "released-scores":
        return released_scores.main(args.wide, args.art_dir)

    if args.cmd == "seed-variance":
        return seed_variance.main(args.seed_dir, args.art_dir)

    if args.cmd == "ifeval-rescore":
        return ifeval_score.main(args.dump, args.prompts, args.art_dir,
                                 args.data_dir if args.emit_items else None)

    if args.cmd == "phase2-transition":
        return phase2_transition.main(args.data_dir, args.art_dir)

    if args.cmd == "phase2-transition-dump":
        return phase2_transition_dump.main(args.responses, args.art_dir)

    if args.cmd == "phase2-vs-baselines":
        return phase2_vs_baselines.main(args.responses, args.ifeval_prompts,
                                        args.art_dir)

    if args.cmd == "math-composite":
        return math_composite.main(args.responses, args.art_dir)

    if args.cmd == "corpus-variation":
        return corpus_variation.main(args.responses, args.art_dir)

    if args.cmd == "phase2-reconcile":
        from .stats import phase2_reconciliation
        return phase2_reconciliation.main(args.data_dir, args.art_dir)

    if args.cmd == "aime-decoding-variance":
        return aime_decoding_variance.main(args.mv_best, args.art_dir)

    if args.cmd == "render-workshop":
        from .render import workshop
        return workshop.main(args.data_dir, args.out_dir, args.figures_dir, args.art_dir)

    if args.cmd == "check-workshop":
        from .render import workshop_check
        return workshop_check.main(args.ours, args.paper_dir)

    if args.cmd == "workshop-release-check":
        from .stats import protocol_a_release
        return protocol_a_release.main(args.data_dir, args.release, args.art_dir)

    if args.cmd == "render-zoo":
        from .render import zoo
        return zoo.main(args.manifest, args.out_dir)

    if args.cmd == "render-language-suite":
        from .render import language_suite
        return language_suite.main(args.manifest, args.out_dir)

    if args.cmd == "render-setup":
        from .render import setup
        return setup.main(args.manifest, args.out_dir)

    if args.cmd == "transcribe-submission":
        from .data import submission
        rows = submission.transcribe(args.pdf)
        print(f"  wrote {submission.write(rows, args.out)}: {len(rows)} rows")
        return 0

    if args.cmd == "render-reconciliation":
        from .render import reconciliation
        return reconciliation.main(args.manifest, args.results_root, args.out_dir)

    if args.cmd == "build":
        from . import paper_check
        return paper_check.build_main(args.targets, args.check)

    if args.cmd == "check-paper":
        from . import paper_check
        return paper_check.check_paper_main(args.paper_dir, args.numbers, args.markdown,
                                            args.by_row)

    if args.cmd == "export-paper":
        from . import paper_check
        return paper_check.export_paper_main(args.paper_dir)

    if args.cmd == "fetch-inputs":
        from . import inputs
        return inputs.fetch_main(args.ids, args.analyses)

    if args.cmd == "check-inputs":
        from . import inputs
        return inputs.check_main(args.ids)

    if args.cmd == "fetch-phase2-records":
        from .manifest import phase2 as manifest_phase2
        return manifest_phase2.main(args.out_dir, args.revision or manifest_phase2.REVISION)

    if args.cmd == "measure-phase2-answers":
        from . import inputs as _inputs
        from .manifest import answer_lengths
        responses = args.responses or Path(_inputs.get("R14-responses").local)
        return answer_lengths.main(responses, args.manifest, args.snapshot, args.out_dir)

    if args.cmd == "measure-phase2-tokens":
        from .stats import phase2_tokens
        if args.ot3_release_dir is not None:
            if args.tokenizer is None:
                raise SystemExit("--ot3-release-dir needs --tokenizer")
            return phase2_tokens.main_release(args.ot3_release_dir, args.tokenizer,
                                              args.out_dir, args.tokenized_dir)
        if args.tokenized_dir is None:
            raise SystemExit("measure-phase2-tokens needs --tokenized-dir or --ot3-release-dir")
        return phase2_tokens.main(args.tokenized_dir, args.out_dir)

    if args.cmd == "measure-question-overlap":
        from .stats import question_overlap
        if args.trained_dir:
            return question_overlap.main_trained(args.trained_dir, args.out_dir)
        return question_overlap.main(args.raw_dir, args.out_dir, args.subsets)

    if args.cmd == "measure-mv-question-overlap":
        from .stats import mv_question_overlap as mq
        if args.step == "prepare":
            if args.trained_dir is None and args.ot3_release_dir is None:
                raise SystemExit("prepare needs --trained-dir or --ot3-release-dir")
            return mq.prepare(args.trained_dir, args.out_dir, args.ot3_release_dir)
        if args.step == "shard":
            if args.name is None or args.shard_file is None:
                raise SystemExit("shard needs a name and --shard-file")
            return mq.shard(args.name, args.shard_file, args.out_dir, args.workers)
        return mq.merge(args.out_dir, args.ot3_dir)

    if args.cmd == "measure-training-data-overlap":
        from .stats import training_data_overlap
        return training_data_overlap.main(args.logs, args.out_dir, args.datasets,
                                          args.reasoning_questions)

    if args.cmd == "scan-benchmarks":
        from .stats import benchmark_scan as bs
        if args.step == "texts":
            if args.tokenized_dir is None or args.tokenizer is None:
                raise SystemExit("texts needs --tokenized-dir and --tokenizer")
            return bs.texts(args.tokenized_dir, args.tokenizer, args.work_dir)
        if args.step == "scan":
            if args.source is None or args.file is None:
                raise SystemExit("scan needs a source and --file")
            return bs.scan(args.source, args.work_dir, args.file, args.workers)
        return bs.merge(args.work_dir, args.out_dir)

    if args.cmd == "measure-overlap-sensitivity":
        from .stats import overlap_sensitivity
        return overlap_sensitivity.main(args.manifest, args.out_dir, args.phase2_dir,
                                        args.data_dir, args.log_dir)

    if args.cmd == "build-memorization-probes":
        from .stats import memorization
        return memorization.build(args.out_dir)

    if args.cmd == "measure-memorization":
        from .stats import memorization
        return memorization.main(args.out_dir)

    if args.cmd == "snapshot-checkpoints":
        from .manifest import hf_snapshot
        return hf_snapshot.main(args.workshop_dir, args.phase2_dir, args.out, args.update)

    if args.cmd == "render-phase2":
        from .render import phase2 as render_phase2
        return render_phase2.main(args.manifest, args.phase2_dir, args.tables_dir,
                                  args.figures_dir, args.art_dir)

    if args.cmd == "render-overlap":
        from .render import overlap
        return overlap.main(args.manifest, args.phase2_dir, args.out_dir)

    if args.cmd == "render-phase2-analyses":
        from .render import phase2_analyses
        return phase2_analyses.main(args.manifest, args.phase2_dir, args.out_dir)

    if args.cmd == "render-frontloading":
        from .render import frontloading
        return frontloading.main(args.manifest, args.data_dir, args.out_dir, args.art_dir)

    if args.cmd == "render-language-intervals":
        from .render import language_intervals
        return language_intervals.main(args.manifest, args.log_dir, args.out_dir)

    if args.cmd == "render-statistics":
        from .render import statistics
        return statistics.main(args.manifest, args.data_dir, args.out_dir)

    if args.cmd == "render-older-tables":
        from .render import older_tables
        return older_tables.main(args.manifest, args.out_dir)

    if args.cmd == "build-manifest":
        from .manifest import build as manifest_build
        return manifest_build.main(args.workshop_dir, args.phase2_dir, args.snapshot, args.out)

    raise SystemExit(f"unknown command: {args.cmd}")


if __name__ == "__main__":
    sys.exit(main())
