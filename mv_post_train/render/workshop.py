"""Regenerate every generated table and figure of the workshop paper.

    mvpt render-workshop --data-dir data/workshop

Reads the two pinned Protocol A inputs (`data/protocol_a.py`), validates them
against the design, computes all summaries once, and writes

  <tables-dir>/*.tex    the eleven tables, same basenames as the paper's
  <figures-dir>/*.pdf   the five figures
  <art-dir>/*.csv       the summaries behind them, for anyone re-checking a cell

`mvpt check-workshop` then compares the tables with the paper's files.
"""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import pandas as pd

from .. import inputs
from ..data import protocol_a as pa
from ..data import protocol_b as pb
from ..stats import factorial as fs
from . import workshop_tables as wt
from .figures import workshop as wf

TABLES_SUBDIR = "workshop"


@dataclass
class WorkshopData:
    factorial: pd.DataFrame
    zoo: pd.DataFrame
    language: pd.DataFrame
    fac_scores: pd.DataFrame
    zoo_scores: pd.DataFrame
    fac_summary: pd.DataFrame
    zoo_summary: pd.DataFrame
    language_avg: pd.DataFrame
    contrasts: pd.DataFrame


def load(data_dir: Path) -> WorkshopData:
    reasoning = pa.load_reasoning(inputs.path("R13", data_dir / pa.REASONING_CSV))
    factorial, zoo = pa.factorial(reasoning), pa.zoo(reasoning)
    language = pa.load_language(inputs.path("R13-language", data_dir / pa.LANGUAGE_CSV))
    fac_scores = fs.benchmark_scores(factorial, pa.FACTORIAL_KEYS)
    zoo_scores = fs.benchmark_scores(zoo, pa.ZOO_KEYS)
    return WorkshopData(
        factorial=factorial, zoo=zoo, language=language,
        fac_scores=fac_scores, zoo_scores=zoo_scores,
        fac_summary=fs.benchmark_summary(fac_scores, pa.FACTORIAL_KEYS),
        zoo_summary=fs.benchmark_summary(zoo_scores, pa.ZOO_KEYS),
        language_avg=fs.language_average(language),
        contrasts=fs.factorial_contrasts(factorial),
    )


def render_all(data_dir: Path, tables_dir: Path, figures_dir: Path,
               art_dir: Path | None = None) -> list[Path]:
    problems = pb.check_constants()
    if problems:
        raise ValueError("Protocol B constants are inconsistent:\n  " + "\n  ".join(problems))
    d = load(data_dir)
    written = [
        wt.factorial_summary(d.fac_summary, d.language_avg, tables_dir),
        wt.factorial_perbenchmark(d.fac_summary, tables_dir),
        wt.zoo_perbenchmark(d.zoo_summary, tables_dir, headline_only=True),
        wt.zoo_perbenchmark(d.zoo_summary, tables_dir, headline_only=False),
        wt.factorial_language(d.language, tables_dir),
        wt.contrasts_table(d.contrasts, tables_dir),
        wt.domain_robustness(d.fac_scores, tables_dir),
        wt.phase2_endpoints(tables_dir),
        wt.ot3_pretrain_ablation(tables_dir),
        wt.heldout_reasoning(tables_dir),
        wt.heldout_mv_grid(tables_dir),
        wf.factorial_main(fs.suite_average(d.fac_scores, pa.FACTORIAL_KEYS), d.language_avg,
                          figures_dir),
        wf.frontloading_trajectory(d.contrasts, figures_dir),
        wf.phase1_zoo(fs.suite_average(d.zoo_scores, pa.ZOO_KEYS), figures_dir),
        wf.phase2_receptivity(figures_dir),
        wf.frontloading_domains(d.fac_scores, figures_dir),
    ]
    if art_dir is not None:
        art_dir.mkdir(parents=True, exist_ok=True)
        keys = pa.FACTORIAL_KEYS
        for name, frame in (
            ("factorial_benchmark_summary", d.fac_summary),
            ("zoo_benchmark_summary", d.zoo_summary),
            ("contrasts", d.contrasts),
            ("language_average", d.language_avg),
            ("equal_domain", fs.equal_domain_average(d.fac_scores, keys)),
            ("leave_one_domain_out", fs.leave_one_domain_out(d.fac_scores, keys)),
            ("domain_scores", fs.domain_scores(d.fac_scores, keys)),
        ):
            path = art_dir / f"{name}.csv"
            frame.to_csv(path, index=False)
            written.append(path)
    return written


def main(data_dir: Path, tables_dir: Path, figures_dir: Path, art_dir: Path) -> int:
    for path in render_all(data_dir, tables_dir, figures_dir, art_dir):
        print(f"  wrote {path}")
    return 0
