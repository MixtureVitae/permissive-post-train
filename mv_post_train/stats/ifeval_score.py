"""Re-score IFEval from the released response dump.

IFEval is one of the six Phase 2 columns the LAION dump carries as raw
generations with no verdicts — `gold_answer` is empty in 100% of its rows, so
`parse_phase2.py` cannot score it and the reasoning-suite analysis has to omit
it. Unlike the code benchmarks, though, IFEval needs no execution: its verdicts
come from a deterministic Python checker applied to the response text. That
makes it the one unscorable column recoverable from the dump alone.

The join is `example_id` -> the IFEval `key` field. Instruction lists and
their kwargs come from the upstream prompt file
(`google/IFEval :: ifeval_input_data.jsonl`), which must be fetched once and
passed in.

The manuscript's Appendix Table 18 specifies **prompt-level accuracy under the
loose criterion**, so that is the headline column here; strict and the two
instruction-level variants are reported alongside, because the gap between them
is the main thing that identifies a convention mismatch if one exists.

Writes `artifacts/ifeval_rescored.csv`.
"""
from __future__ import annotations

import json
from pathlib import Path

import pandas as pd
import pyarrow.parquet as pq

from ..data.ifeval_checker import instructions_registry

RESCORED_CSV = "ifeval_rescored.csv"

TASK = "IFEval"

# Columns the scorer needs. `model_output` holds the full generation and is
# large, so nothing else is read.
_COLUMNS = ["base_model", "stage", "task", "example_id", "run_idx", "model_output"]


def load_prompt_metadata(path: Path) -> dict[int, dict]:
    """key -> {prompt, instruction_id_list, kwargs} from ifeval_input_data.jsonl."""
    meta: dict[int, dict] = {}
    with open(path) as fh:
        for line in fh:
            line = line.strip()
            if not line:
                continue
            rec = json.loads(line)
            meta[int(rec["key"])] = rec
    if not meta:
        raise ValueError(f"no records parsed from {path}")
    return meta


def _loose_variants(response: str) -> list[str]:
    """The eight response variants the loose criterion accepts.

    Copied from `evaluation_lib.test_instruction_following_loose`: strip the
    first line, the last line, both, and each of those with markdown emphasis
    removed. An instruction counts as followed if *any* variant satisfies it.
    """
    r = response.split("\n")
    remove_first = "\n".join(r[1:]).strip()
    remove_last = "\n".join(r[:-1]).strip()
    remove_both = "\n".join(r[1:-1]).strip()
    return [
        response,
        response.replace("*", ""),
        remove_first,
        remove_last,
        remove_both,
        remove_first.replace("*", ""),
        remove_last.replace("*", ""),
        remove_both.replace("*", ""),
    ]


def check_response(rec: dict, response: str) -> tuple[list[bool], list[bool]]:
    """Per-instruction (strict, loose) verdicts for one response.

    Mirrors `evaluation_lib.test_instruction_following_{strict,loose}`,
    including the re-`build_description` call for instructions that take the
    prompt as an argument, and the empty-response guard.
    """
    strict: list[bool] = []
    loose: list[bool] = []
    for index, instruction_id in enumerate(rec["instruction_id_list"]):
        cls = instructions_registry.INSTRUCTION_DICT[instruction_id]
        instruction = cls(instruction_id)
        instruction.build_description(**rec["kwargs"][index])
        args = instruction.get_instruction_args()
        if args and "prompt" in args:
            instruction.build_description(prompt=rec["prompt"])

        try:
            strict.append(bool(response.strip()) and bool(instruction.check_following(response)))
        except Exception:
            strict.append(False)

        followed = False
        for variant in _loose_variants(response):
            if not variant.strip():
                continue
            try:
                if instruction.check_following(variant):
                    followed = True
                    break
            except Exception:
                continue
        loose.append(followed)
    return strict, loose


def score(dump: Path, meta: dict[int, dict]) -> pd.DataFrame:
    """Score every IFEval row in the dump; one output row per (cell, response)."""
    f = pq.ParquetFile(dump)
    rows = []
    n_unmatched = 0
    for batch in f.iter_batches(batch_size=2048, columns=_COLUMNS):
        tasks = batch.column("task").to_pylist()
        if TASK not in tasks:
            continue
        base = batch.column("base_model").to_pylist()
        stage = batch.column("stage").to_pylist()
        eid = batch.column("example_id").to_pylist()
        run = batch.column("run_idx").to_pylist()
        out = batch.column("model_output").to_pylist()
        for t, bm, st, e, r, o in zip(tasks, base, stage, eid, run, out):
            if t != TASK:
                continue
            try:
                key = int(e)
            except (TypeError, ValueError):
                n_unmatched += 1
                continue
            rec = meta.get(key)
            if rec is None:
                n_unmatched += 1
                continue
            s, l = check_response(rec, o or "")
            rows.append({
                "base_model": bm, "stage": st, "example_id": key, "run_idx": r,
                "n_instructions": len(s),
                "strict_all": all(s), "loose_all": all(l),
                "strict_followed": sum(s), "loose_followed": sum(l),
            })
    if n_unmatched:
        print(f"note: {n_unmatched} IFEval rows had no matching prompt key, skipped")
    return pd.DataFrame(rows)


def summarize(scored: pd.DataFrame) -> pd.DataFrame:
    """Per-(model, stage) accuracies in all four IFEval conventions."""
    g = scored.groupby(["base_model", "stage"])
    out = g.agg(
        n_prompts=("example_id", "nunique"),
        n_responses=("example_id", "size"),
        prompt_strict=("strict_all", "mean"),
        prompt_loose=("loose_all", "mean"),
        instr_strict=("strict_followed", "sum"),
        instr_loose=("loose_followed", "sum"),
        n_instr=("n_instructions", "sum"),
    ).reset_index()
    out["prompt_strict"] = (out.prompt_strict * 100).round(2)
    out["prompt_loose"] = (out.prompt_loose * 100).round(2)
    out["instr_strict"] = (out.instr_strict / out.n_instr * 100).round(2)
    out["instr_loose"] = (out.instr_loose / out.n_instr * 100).round(2)
    return out.drop(columns="n_instr").sort_values(["base_model", "stage"])


# Submission Table 1, IFE column, transcribed from page 6. Keyed by the
# dump's own (base_model, stage) labels. Used only to identify which of the
# four IFEval conventions the manuscript reports — a re-scoring that matches
# none of them is a finding, not a number to quote.
TABLE1_IFE: dict[tuple[str, str], float] = {
    ("MixtureVitae-1.7B", "base"):     42.6,
    ("MixtureVitae-1.7B", "OT3+SFT"):  25.6,
    ("MixtureVitae-1.7B", "OT3+DPO"):  43.0,
    ("MixtureVitae-1.7B", "OT3"):      26.6,
    ("FineWebEdu-1.7B",   "SFT-Tulu3"): 27.6,
    ("FineWebEdu-1.7B",   "OT3+SFT"):  17.8,
    ("NemotronHQ-1.7B",   "SFT-Tulu3"): 36.5,
    ("NemotronHQ-1.7B",   "OT3+SFT"):  20.0,
    ("Comma0.1-1.7B",     "SFT-Tulu3"): 35.4,
    ("Comma0.1-1.7B",     "OT3+SFT"):  20.0,
}


def compare_to_table1(summary: pd.DataFrame) -> pd.DataFrame:
    """Which convention, if any, reproduces the manuscript's IFE column."""
    rows = []
    for _, r in summary.iterrows():
        reported = TABLE1_IFE.get((r.base_model, r.stage))
        if reported is None:
            continue
        rows.append({
            "base_model": r.base_model, "stage": r.stage, "table1": reported,
            "prompt_loose": r.prompt_loose, "prompt_strict": r.prompt_strict,
            "instr_loose": r.instr_loose, "instr_strict": r.instr_strict,
            "d_prompt_loose": round(r.prompt_loose - reported, 2),
            "d_prompt_strict": round(r.prompt_strict - reported, 2),
            "d_instr_loose": round(r.instr_loose - reported, 2),
            "d_instr_strict": round(r.instr_strict - reported, 2),
        })
    return pd.DataFrame(rows)


def emit_items(scored: pd.DataFrame, dump: Path, data_dir: Path) -> None:
    """Persist the per-prompt verdicts as canonical item rows.

    `summarize` throws this away, which is the only reason IFEval cannot take
    part in a paired Phase 1 -> OT30K test. The paired unit is the prompt, and
    the manuscript's convention is prompt-level loose, so `loose_all` is the
    per-item score.
    """
    from ..data import recover_items

    verdicts = scored.rename(columns={"example_id": "item_index",
                                      "run_idx": "attempt_index"})
    verdicts["score"] = verdicts.loose_all.astype(float)
    excluded = recover_items.excluded_cells(dump)
    rows = recover_items.to_item_rows(verdicts, TASK, excluded)
    recover_items.report(rows, TASK, excluded)
    if rows:
        recover_items.merge(data_dir, TASK, rows)
        print(f"wrote {len(rows)} {TASK} item rows to "
              f"{Path(data_dir) / 'recovered_items.parquet'}")


def main(dump: Path, prompts: Path, art_dir: Path,
         data_dir: Path | None = None) -> int:
    dump, prompts, art_dir = Path(dump), Path(prompts), Path(art_dir)
    for p in (dump, prompts):
        if not p.exists():
            print(f"missing {p}")
            return 1
    art_dir.mkdir(parents=True, exist_ok=True)

    meta = load_prompt_metadata(prompts)
    print(f"loaded {len(meta)} IFEval prompts")
    scored = score(dump, meta)
    if scored.empty:
        print("no IFEval rows scored")
        return 1
    if data_dir is not None:
        emit_items(scored, dump, Path(data_dir))
    summary = summarize(scored)
    summary.to_csv(art_dir / RESCORED_CSV, index=False)
    print(summary.to_string(index=False))

    cmp = compare_to_table1(summary)
    if not cmp.empty:
        print("\n=== vs submission Table 1 (IFE column) ===")
        print(cmp.to_string(index=False))
        best = min(
            ("prompt_loose", "prompt_strict", "instr_loose", "instr_strict"),
            key=lambda c: cmp[f"d_{c}"].abs().mean(),
        )
        print(f"\nclosest convention: {best} "
              f"(mean |Δ| = {cmp[f'd_{best}'].abs().mean():.2f} pp across "
              f"{len(cmp)} cells)")
        cmp.to_csv(art_dir / "ifeval_vs_table1.csv", index=False)
        print(f"wrote {art_dir / 'ifeval_vs_table1.csv'}")

    print(f"\nwrote {art_dir / RESCORED_CSV}")
    return 0
