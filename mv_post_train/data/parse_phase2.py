"""Phase 2 HF parquet parser: reads the LAION mv-posttrain-eval-responses
dump (and optional MV best-responses override) and emits item rows in the
canonical schema.

Source: https://huggingface.co/datasets/laion/mv-posttrain-eval-responses

**Phase 2 only.** This module is the canonical source for OpenThoughts3-30K
cells (ot30k_on_base / ot30k_on_sft / ot30k_on_dpo). It deliberately does
*not* re-emit Phase 1 (base / sft / dpo) — those come from logs.zip via
`data/parse_phase1.py`. Clean source separation; no merge step.

Produces `phase2_items.parquet`, schema-compatible with `items_long.parquet`.

We score the five tasks the dump supports reliably:
  AIME24, AIME25, AMC23, MATH500 (math-verify), GSM8K (`is_correct` column).

If the Phase 2 HF parquet's "best-responses" override parquet is provided
(`mv_aime24_amc23_best_responses.parquet`), MV's AIME24 and AMC23 cells are
sourced from there (single `source_dir` per cell, matching the manuscript's
"+ OT30K" headline numbers).

Stage mapping (Phase 2 only):
  OT3        -> ot30k_on_base
  OT3+SFT    -> ot30k_on_sft
  OT3+DPO    -> ot30k_on_dpo

`Phase2Parser` is an `EvalParser` subclass (`data/parser.py`); the
module-level `main` is a thin CLI wrapper that constructs the parser and
calls `write_to_data_dir`.
"""
from __future__ import annotations

from pathlib import Path

import pandas as pd
from tqdm import tqdm

from .. import paths
from .benchmarks import AIME24, AIME25, AMC23, GSM8K, MATH500, SPECS
from .io_logs import ItemRow
from .parser import EvalParser, ParseResult
from .scorers import CompareScorer

# Phase 2-side task names. "GSM8K" (uppercase) is the external upstream
# key; our canonical name is lowercase `gsm8k` (the GSM8K constant). The
# other four are already in canonical form.
_PHASE2_RAW_GSM8K = "GSM8K"
SCORED_TASKS = [MATH500, _PHASE2_RAW_GSM8K, AMC23, AIME24, AIME25]

# External-name -> canonical-name fixups.
TASK_RENAME = {_PHASE2_RAW_GSM8K: GSM8K}

# Everything `_score_row` and the ItemRow construction below touch. Anything
# absent here (notably `model_output` and `problem`, which hold full
# generations) is never read and is not worth the memory.
_SCORING_COLUMNS = [
    "base_model", "stage", "task", "example_idx", "run_idx",
    "gold_answer", "model_answer", "is_correct",
]

# base_model (Phase 2 source) -> (family, model_basename_root). Mirrors the
# basenames our logs.zip parse emits so downstream consumers key on `model`.
FAMILY = {
    "MixtureVitae-1.7B":  ("MV",          "ali-elganzory__1.7b-MixtureVitae-300BT-v1-decontaminated-16k"),
    "Comma0.1-1.7B":      ("Comma",       "ali-elganzory__1.7b-Comma0.1-300BT-longsft_16k"),
    "NemotronHQ-1.7B":    ("Nemotron",    "ali-elganzory__open-sci-ref-v0.02-1.7b-nemotron-hq-300B-16k"),
    "FineWebEdu-1.7B":    ("FineWeb-Edu", "ali-elganzory__open-sci-ref-v0.02-1.7b-fineweb-edu-1.4t-300B-4096-longsft_16k"),
    "DCLM-1.7B":          ("DCLM",        "ali-elganzory__open-sci-ref-v0.02-1.7b-dclm-300B-4096-longsft_16k"),
}

# Phase 2 stage -> (canonical phase, model_basename suffix).
# Phase 2 only — Phase 1 cells (base / sft / dpo) come from logs.zip via
# parse_phase1.py, not from this dump.
STAGE = {
    "OT3":     ("ot30k_on_base", ""),
    "OT3+SFT": ("ot30k_on_sft",  "-SFT-Tulu3-decontaminated"),
    "OT3+DPO": ("ot30k_on_dpo",  "-DPO-Tulu3-decontaminated"),
}


def _model_name(base_model: str, stage: str) -> str:
    """Build the OT30K filesystem id for a Phase 2 row. All stages here
    are Phase 2 (ot30k_*), so every result starts with `open-sci__sft__ot30k_`.
    """
    _, root = FAMILY[base_model]
    _, suffix = STAGE[stage]
    return f"open-sci__sft__ot30k_{root.removeprefix('ali-elganzory__')}{suffix}"


def _score_row(task: str, row: pd.Series) -> int | None:
    """Score one flat Phase 2 HF parquet row.

    `task` here is the Phase 2 HF parquet's column value (still in upstream
    form). GSM8K is special-cased — the Phase 2 HF parquet stores a
    pre-computed `is_correct` boolean rather than a gold/pred pair, so
    there's no comparison to delegate. The other four tasks delegate to
    `SPECS[canonical_task].scorer.compare(gold, pred)` — the same Scorer
    instance that scores their per-example records from logs.zip.
    """
    if task == _PHASE2_RAW_GSM8K:
        if row.is_correct is None or (isinstance(row.is_correct, float) and pd.isna(row.is_correct)):
            return None
        return int(bool(row.is_correct))
    canonical = TASK_RENAME.get(task, task)
    scorer = SPECS[canonical].scorer
    if not isinstance(scorer, CompareScorer):
        return None  # Phase 2 only scores tasks that compare gold vs pred
    try:
        return scorer.compare(row.gold_answer, row.model_answer)
    except Exception:
        return None


class Phase2Parser(EvalParser):
    """LAION HF parquet parser. Emits item rows only (the dump
    doesn't carry the aggregate-task scalars that HumanEval / MBPP / IFEval
    / gsm8k need; those stay in logs.zip)."""

    parser_name = "phase2"
    items_filename = paths.PHASE2_ITEMS
    emits_aggregates = False
    audit_filename = None

    def __init__(self, hf_main: Path, mv_best: Path | None = None) -> None:
        self.hf_main = Path(hf_main)
        self.mv_best = Path(mv_best) if mv_best is not None else None

    def parse(self) -> ParseResult:
        if not self.hf_main.exists():
            raise FileNotFoundError(f"missing Phase 2 source: {self.hf_main}")

        # Read only the columns scoring needs. The dump carries the full
        # generated text in `model_output` (and the prompt in `problem`);
        # since the 2026-07-27 backfill those dominate the file — loading
        # them costs many GB of RAM for columns we never read. Push the
        # row filter down to the reader too, so only Phase 2 rows for the
        # five scored tasks are materialized.
        df = pd.read_parquet(
            self.hf_main,
            columns=_SCORING_COLUMNS,
            filters=[
                ("task", "in", SCORED_TASKS),
                ("base_model", "in", list(FAMILY)),
                ("stage", "in", list(STAGE)),
            ],
        )
        # `filters` prunes at row-group granularity, so re-apply exactly.
        df = df[
            df.task.isin(SCORED_TASKS)
            & df.base_model.isin(FAMILY)
            & df.stage.isin(STAGE)
        ].copy()

        if self.mv_best is not None and self.mv_best.exists():
            best = pd.read_parquet(self.mv_best)
            # The override file gained temperature-0.7 replicate runs in the
            # 2026-07-27 revision, tagged by a `decoding` column that is null
            # on the original greedy rows. Those replicates measure decoding
            # variance (see stats/aime_decoding_variance.py); they are not
            # the manuscript's cells, so keep only the greedy rows here.
            if "decoding" in best.columns:
                best = best[best.decoding.isna()]
            best = best[
                (best.base_model == "MixtureVitae-1.7B")
                & best.task.isin([AIME24, AMC23])
                & best.stage.isin(STAGE)
            ].copy()
            common = [c for c in df.columns if c in best.columns]
            best = best[common]
            mask = (
                (df.base_model == "MixtureVitae-1.7B")
                & df.task.isin([AIME24, AMC23])
                & df.stage.isin(best.stage.unique())
            )
            n_dropped = int(mask.sum())
            df = pd.concat([df[~mask], best], ignore_index=True)
            print(f"replaced {n_dropped} MV AIME24/AMC23 rows "
                  f"with {len(best)} from {self.mv_best.name}")

        print(f"Phase 2 rows after filter: {len(df)}")

        item_rows: list[ItemRow] = []
        for (base_model, stage, task), g in tqdm(
            df.groupby(["base_model", "stage", "task"]), desc=self.parser_name
        ):
            family, _ = FAMILY[base_model]
            phase, _ = STAGE[stage]
            model = _model_name(base_model, stage)
            canonical_task = TASK_RENAME.get(task, task)
            for _, r in g.iterrows():
                s = _score_row(task, r)
                if s is None:
                    continue
                item_rows.append(ItemRow(
                    model=model,
                    family=family,
                    phase=phase,
                    context_tier="16k",
                    tokens_b=300.0,
                    size_b=1.7,
                    task=canonical_task,
                    item_index=int(r.example_idx),
                    attempt_index=int(r.run_idx),
                    score=float(s),
                ))

        # Coverage diagnostic — printed for parity with the previous main();
        # not persisted to the audit log (this parser has audit_filename=None).
        if item_rows:
            cov_df = pd.DataFrame([r.model_dump() for r in item_rows])
            cov = (cov_df.groupby(["family", "phase", "task"]).size()
                         .unstack("task", fill_value=0))
            print("coverage (rows per (family, phase, task)):")
            print(cov.to_string())

        return ParseResult(item_rows=tuple(item_rows))


def main(hf_main: Path, mv_best: Path | None, data_dir: Path) -> int:
    """CLI entry point. Thin wrapper around `Phase2Parser.write_to_data_dir`."""
    if not Path(hf_main).exists():
        print(f"missing source: {hf_main}")
        return 1
    return Phase2Parser(hf_main, mv_best).write_to_data_dir(data_dir)
