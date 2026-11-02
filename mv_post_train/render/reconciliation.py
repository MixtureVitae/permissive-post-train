"""Every number of the submitted paper's results tables against the
revision's, with the reason for each change. It is not part of the paper: it
is for the authors, and for the reviewers to whom it was promised.

`mvpt render-reconciliation` reads the submission's eleven results tables
(input T-submission, `data/submission.py`) and the tables the revision prints
(the generated files of the zoo, phase2, older and language-suite targets, read
with `paper_check.table_rows`), pairs every submitted row with the revised row
of the same checkpoint, and writes:

  <out-dir>/submission_vs_revision.csv    one row per submitted cell
  <out-dir>/summary.md                    a summary by submitted table, with the reasons

Pairing. The revision prints the rows of submitted Tables 2 and 4 to 11 in the
same order, so they are paired by position, and the build stops unless each
pair has the same label once the revision's names are applied (MV is
"MixtureVitae" in Table 6, for example). A row the revision adds, such as MV's
→ SFT → OT30K row of the Phase 2 language table, is named in its pairing and
left out. The Phase 2 tables (submitted Tables
1 and 3) are paired by the records' model and stage. The submission prints
some Phase 2 rows that the revision does not: the reference models' DPO →
OT30K runs and the released Instruct models. Their revised values are the
results manifest's, rounded as the revision rounds, and they are marked as not
printed.

Reasons. Every cell gets exactly one, and the build stops if a changed cell
fits none, so that no number changes without a stated reason. Where the
records can confirm a reason, it is checked. The submission's older tables
(its Tables 4 to 6) printed the older single-run evaluations, which took each
score field at its highest over a checkpoint's runs. The revision prints
those checkpoints under Protocol A (Tables 7 to 9), so their cells change for
the same reason as Phase 1's.
"""
from __future__ import annotations

import csv
import re
from collections import Counter
from dataclasses import dataclass
from pathlib import Path

from .. import inputs
from ..data import submission as sub
from ..manifest.build import PHASE2
from ..manifest.lineage import bare
from ..manifest.reader import Manifest
from ..paper_check import table_rows
from . import phase2_tables as p2
from .macros import number_word

ARROW = sub.ARROW
#: Tolerances, in points, for a difference that is only rounding: one decimal
#: on both sides in the reasoning tables. The submission's language tables
#: print whole numbers, and their Avg is the mean of rounded cells.
ROUNDING = 0.1 + 1e-9
WHOLE = 0.5 + 1e-9
WHOLE_AVG = 1.0 + 1e-9

#: Reason -> (letter in the summary, what it means). "same", "rounding" and
#: "one_decimal" are counted in their own columns.
REASONS = {
    "same": ("", "printed the same"),
    "rounding": ("", "rounding: the revision rounds each value once, the submission rounded "
                     "differently or averaged rounded values"),
    "one_decimal": ("", "the revision prints one decimal where the submission printed whole "
                        "numbers"),
    "protocol_a": ("a", "Protocol A: the mean of three greedy runs, with HumanEval "
                        "scored on its Python problems and IFEval at instruction level with "
                        "the strict checker"),
    "three_runs": ("b", "Phase 2: the mean of three sampled runs from the released records, "
                        "where the submission printed one earlier evaluation"),
    "gsm8k_rerun": ("c", "Phase 2 GSM8K: the mean of the three runs of the GSM8K re-run, "
                         "with flexible answer extraction"),
    "other_run": ("d", "another language run: the submitted row differs from the run the "
                       "revision prints by more than rounding"),
    "unsettled": ("e", "the run the records label as OT30K on the Instruct model is named "
                       "after the base model, so the model it was trained from is not settled"),
    "ifeval_definition": ("f", "Phase 2 IFEval: the mean of three sampled runs, scored at "
                               "instruction level with the strict checker, the definition of "
                               "Protocol A, where the submission printed one earlier evaluation "
                               "scored by another definition"),
}
ROUNDING_REASONS = ("rounding", "one_decimal")


@dataclass(frozen=True)
class Pairing:
    submitted: str              # "Table 2"
    shows: str                  # what the submitted table shows
    generated: str              # the revision's table, under the results root
    kind: str                   # "phase2", "zoo", "older" or "language"
    renames: tuple[tuple[str, str], ...] = ()   # submitted model, stage or section -> the revision's
    added: tuple[str, ...] = ()  # labels of revised rows the submission does not have


ZOO_NAMES = ((r"^MV", "MixtureVitae"), (r"^Nemotron", "Nemotron-CC-HQ"), (r"-16k$", "-16K"))
#: The revision names the two arms of the decontamination tables by the historical filtering.
FILTERING = ((r"^Not Decontaminated$", "Without filtering"),
             (r"^Decontaminated$", "With filtering"))
PAIRINGS = (
    Pairing("Table 1", "Phase 2", "results/phase2/tables/phase2_endpoints_full.tex", "phase2"),
    Pairing("Table 2", "Phase 1", "results/zoo/tables/phase1_zoo_full.tex", "zoo",
            renames=ZOO_NAMES, added=("DCLM-16K", f"DCLM-16K / {ARROW} SFT",
                                      f"DCLM-16K / {ARROW} SFT {ARROW} DPO")),
    Pairing("Table 3", "Phase 2, all rows", "results/phase2/tables/phase2_endpoints_full.tex", "phase2"),
    Pairing("Table 4", "100B ablation", "results/older/tables/ablation_100b_reasoning.tex",
            "older"),
    Pairing("Table 5", "Decontamination", "results/older/tables/decontamination_reasoning.tex",
            "older", renames=FILTERING),
    Pairing("Table 6", "0.4B scale", "results/older/tables/smallscale_reasoning.tex", "older"),
    Pairing("Table 7", "100B ablation",
            "results/language_suite/tables/language_ablation_100b.tex", "language"),
    Pairing("Table 8", "Main comparison",
            "results/language_suite/tables/language_main.tex", "language"),
    Pairing("Table 9", "Decontamination",
            "results/language_suite/tables/language_decontamination.tex", "language",
            renames=FILTERING),
    Pairing("Table 10", "0.4B scale",
            "results/language_suite/tables/language_smallscale.tex", "language"),
    Pairing("Table 11", "Phase 2",
            "results/language_suite/tables/language_phase2.tex", "language",
            added=(f"MV-16k / {ARROW} SFT {ARROW} OT30K",),
            renames=((r"\(no Phase 1\)$", "(without Tulu3)"),)),
)

#: The submission's Phase 2 model names -> the records' base_model.
PHASE2_MODEL = {
    "MV-16k (base)": p2.MV, "FineWeb-Edu-16k": "FineWebEdu-1.7B", "Nemotron-16k": "NemotronHQ-1.7B",
    "Comma0.1-16k": "Comma0.1-1.7B", "SmolLM2-16k": "SmolLM2-1.7B-16k-Base",
    "SmolLM2-16k-Base": "SmolLM2-1.7B-16k-Base", "Qwen2.5": "Qwen2.5-1.5B-Base",
    "Qwen2.5-Base": "Qwen2.5-1.5B-Base", "Qwen3-Base": "Qwen3-1.7B-Base",
    "Qwen2.5-Instruct": "Qwen2.5-1.5B-Instruct", "Qwen3-Instruct": "Qwen3-1.7B-Instruct",
    "SmolLM2-Instruct-16k": "SmolLM2-1.7B-Instruct-16k",
}
#: The submission's Phase 2 stages -> the records' stages. A reference model's
#: first row is its Tulu3 SFT checkpoint, and "→ OT30K" under it is OT30K
#: trained from that checkpoint. An Instruct model's "→ OT30K" is trained from
#: the released model itself.
PHASE2_STAGE = {"": "base", f"{ARROW} SFT": "SFT-Tulu3", f"{ARROW} OT30K": "OT3+SFT",
                f"{ARROW} SFT {ARROW} OT30K": "OT3+SFT", f"{ARROW} DPO {ARROW} OT30K": "OT3+DPO",
                f"{ARROW} OT30K (no Phase 1)": "OT3"}
INSTRUCT_STAGE = {"": "base", f"{ARROW} OT30K": "OT3"}
TASK_OF = {v: k for k, v in p2.ABBR.items()}


@dataclass
class Cell:
    pairing: Pairing
    row: sub.Row
    column: str
    submitted: float
    revised_table: str          # "Table 6", or "" when the revision does not print the row
    revised_row: str            # the revised row's label, or the checkpoint if not printed
    revised: float | None       # None: the revision prints a dash
    reason: str = ""
    in_release: str = ""        # Phase 2: "yes" if the submitted value is one of the released runs
    unsettled: bool = False     # Phase 2: the manifest rejects the checkpoint's pairing

    @property
    def difference(self) -> float | None:
        return None if self.revised is None else round(self.revised - self.submitted, 1) + 0.0


# ---------------------------------------------------------------------------
# The revised rows


def _norm(label: str) -> str:
    s = label.replace(r"\to", ARROW).replace("$", "").replace("~", " ").replace(r"\quad", " ")
    s = re.sub(r"\\[A-Za-z]+\{([^}]*)\}", r"\1", s)
    s = re.sub(r"\s*\(\s*", " (", re.sub(r"\s*\)", ")", s))
    return re.sub(r"\s+", " ", s).strip()


def _submitted_label(row: sub.Row, renames) -> str:
    """The submitted row's label, with each rename applied to its section, model and stage."""
    section, model, stage = row.section or "", row.model, row.stage or ""
    for pattern, repl in renames:
        section, model, stage = (re.sub(pattern, repl, x) for x in (section, model, stage))
    return _norm(" / ".join(x for x in (section, model, stage) if x))


def _values(rev_row) -> list[float | None]:
    return [float(c[0]) if c else None for c in rev_row.cells]


def _item(generated: str) -> str:
    from ..targets import PAPER_FILES
    return next(f.item for f in PAPER_FILES if f.generated == generated)


def _positional(pairing: Pairing, rows: list[sub.Row], root: Path) -> list[Cell]:
    revised = [r for r in table_rows((root / pairing.generated).read_text())
               if _norm(r.label) not in pairing.added]
    if len(revised) != len(rows):
        raise ValueError(f"Submitted {pairing.submitted} has {len(rows)} rows and "
                         f"{pairing.generated} {len(revised)}.")
    item, cells = _item(pairing.generated), []
    for r, rev in zip(rows, revised):
        if _submitted_label(r, pairing.renames) != _norm(rev.label):
            raise ValueError(f"Submitted {pairing.submitted} row {r.label!r} is paired with "
                             f"{rev.label!r} of {pairing.generated}.")
        values = _values(rev)
        if len(values) != len(r.columns):
            raise ValueError(f"{pairing.generated}, {rev.label}: {len(values)} values.")
        cells += [Cell(pairing, r, c, s, item, _norm(rev.label), v)
                  for c, s, v in zip(r.columns, r.values, values)]
    return cells


def _phase2(pairing: Pairing, rows: list[sub.Row], root: Path, m: Manifest) -> list[Cell]:
    shown = p2.full_rows()
    revised = table_rows((root / pairing.generated).read_text())
    if len(revised) != len(shown):
        raise ValueError(f"{pairing.generated} has {len(revised)} rows, Table 2's design "
                         f"{len(shown)}.")
    printed = {k: (_norm(rev.label), _values(rev)) for k, rev in zip(shown, revised)}
    rejected = {c["checkpoint"] for c in m.data["rejected_changes"] if c["protocol"] == PHASE2}
    item, cells = _item(pairing.generated), []
    for r in rows:
        bm = PHASE2_MODEL[r.model]
        stage = (INSTRUCT_STAGE if bm.endswith(("Instruct", "Instruct-16k")) else PHASE2_STAGE)[r.stage]
        key = m.checkpoint(bm, stage)
        if (bm, stage) in printed:
            label, values = printed[(bm, stage)]
            where = item
        else:
            label, where = key, ""
            values = [p2.printed(m.result(key, TASK_OF[c])["mean"]) for c in r.columns[:-1]]
            values.append(p2.printed(m.suite(key)["mean"]))
        for c, s, v in zip(r.columns, r.values, values):
            cell = Cell(pairing, r, c, s, where, label, v, unsettled=key in rejected)
            if c not in ("Avg", "GSM8K"):
                runs = m.result(key, TASK_OF[c])["per_seed"].values()
                cell.in_release = "yes" if any(abs(s - x) <= 0.05 + 1e-6 for x in runs) else "no"
            cells.append(cell)
    return cells


# ---------------------------------------------------------------------------
# Reasons


def _row_cells(cells: list[Cell]) -> list[list[Cell]]:
    out: list[list[Cell]] = []
    for c in cells:
        if out and out[-1][0].row is c.row:
            out[-1].append(c)
        else:
            out.append([c])
    return out


def assign_reasons(cells: list[Cell]) -> None:
    """Give every cell its reason. Stops on a changed cell that fits no reason."""
    for pairing in PAIRINGS:
        rows = _row_cells([c for c in cells if c.pairing is pairing])
        for row in rows:
            *tasks, avg = row
            if pairing.kind == "language":
                whole = (all(abs(c.revised - c.submitted) <= WHOLE for c in tasks)
                         and abs(avg.revised - avg.submitted) <= WHOLE_AVG)
                for c in row:
                    c.reason = ("same" if c.revised == c.submitted else
                                "one_decimal" if whole else "other_run")
            else:
                for c in row:
                    c.reason = ("same" if c.revised == c.submitted else
                                "protocol_a" if pairing.kind in ("zoo", "older") else
                                "unsettled" if c.unsettled else
                                "gsm8k_rerun" if c.column == "GSM8K" else
                                "ifeval_definition" if c.column == "IFE" else "three_runs")
    unexplained = [c for c in cells if not c.reason]
    if unexplained:
        listed = "; ".join(f"{c.pairing.submitted} {c.row.label} {c.column}: {c.submitted} -> "
                           f"{c.revised}" for c in unexplained[:10])
        raise ValueError(f"{len(unexplained)} changed cells have no reason: {listed}")


# ---------------------------------------------------------------------------
# All cells


def reconcile(manifest_path: Path, results_root: Path) -> list[Cell]:
    m = Manifest.load(manifest_path)
    rows = sub.load(inputs.path("T-submission"))
    cells: list[Cell] = []
    for pairing in PAIRINGS:
        mine = [r for r in rows if r.table == pairing.submitted]
        if pairing.kind == "phase2":
            cells += _phase2(pairing, mine, results_root, m)
        else:
            cells += _positional(pairing, mine, results_root)
    assign_reasons(cells)
    return cells


def _num(x: float | None) -> str:
    return "" if x is None else f"{x:.1f}"


def write_cells(cells: list[Cell], path: Path) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", newline="") as f:
        w = csv.writer(f, lineterminator="\n")
        w.writerow(("submitted_table", "page", "section", "model", "stage", "column", "submitted",
                    "revised_table", "revised_row", "printed", "revised", "difference", "reason",
                    "submitted_in_release"))
        for c in cells:
            d = c.difference
            w.writerow((c.pairing.submitted, c.row.page, c.row.section, c.row.model, c.row.stage,
                        c.column, _num(c.submitted), c.revised_table, c.revised_row,
                        "yes" if c.revised_table else "no", _num(c.revised),
                        "" if d is None else f"{d:+.1f}", c.reason, c.in_release))
    return path


# ---------------------------------------------------------------------------
# The summary


@dataclass
class Summary:
    pairing: Pairing
    revised: str
    rows: int
    not_printed: int
    cells: int
    same: int
    rounding: int
    changed: int
    avg_low: float
    avg_high: float
    letters: str


def summarize(cells: list[Cell]) -> list[Summary]:
    out = []
    for pairing in PAIRINGS:
        mine = [c for c in cells if c.pairing is pairing]
        rows = _row_cells(mine)
        reasons = Counter(c.reason for c in mine)
        avgs = [r[-1].difference for r in rows if r[-1].difference is not None]
        letters = sorted({REASONS[c.reason][0] for c in mine} - {""})
        out.append(Summary(
            pairing, _item(pairing.generated), len(rows),
            sum(1 for r in rows if not r[0].revised_table), len(mine), reasons["same"],
            sum(reasons[k] for k in ROUNDING_REASONS),
            len(mine) - reasons["same"] - sum(reasons[k] for k in ROUNDING_REASONS),
            min(avgs), max(avgs), ", ".join(letters)))
    return out


def _signed(x: float) -> str:
    return "0.0" if x == 0 else f"{x:+.1f}"


def _range(lo: float, hi: float) -> str:
    return _signed(lo) if lo == hi else f"{_signed(lo)} to {_signed(hi)}"


def summary_md(summaries: list[Summary], cells: list[Cell]) -> str:
    reasons = Counter(c.reason for c in cells)
    rounding = sum(reasons[k] for k in ROUNDING_REASONS)
    changed = len(cells) - reasons["same"] - rounding
    # Submitted Table 3 holds every Phase 2 row. Its task cells other than GSM8K
    # and IFEval that changed for reason b, and how many of them are one of the
    # released runs.
    p2cells = [c for c in cells if c.pairing.submitted == "Table 3" and c.reason == "three_runs"
               and c.column != "Avg"]
    none = sum(c.in_release == "no" for c in p2cells)
    not_printed = " and ".join(f"{number_word(s.not_printed)} rows of submitted {s.pairing.submitted}"
                               for s in summaries if s.not_printed)
    lines = [
        "# The submitted results tables against the revision", "",
        "Generated by `mvpt render-reconciliation`. Do not edit by hand. Every cell is in",
        "`submission_vs_revision.csv`, and `docs/reconciliation.md` explains the method.", "",
        f"The submission's eleven results tables hold {len(cells):,} cells. Against the table "
        f"of the revision that holds the same checkpoints, {reasons['same']:,} are the same, "
        f"{rounding:,} differ only by rounding and {changed:,} changed. \"Avg change\" is the "
        "range of the change in the rows' averages, in points.", "",
        "| Submitted | Revised | Rows | Cells | Same | Rounding | Changed | Avg change | Reasons |",
        "|---|---|---|---|---|---|---|---|---|"]
    for s in summaries:
        star = "\\*" if s.not_printed else ""
        suite = "reasoning" if s.pairing.kind != "language" else "language"
        lines.append(f"| {s.pairing.submitted} ({s.pairing.shows}, {suite}) | {s.revised}{star} | "
                     f"{s.rows} | {s.cells} | {s.same} | {s.rounding} | {s.changed} | "
                     f"{_range(s.avg_low, s.avg_high)} | {s.letters or '-'} |")
    lines += ["", "Reasons:", ""]
    lines += [f"- ({letter}) {text[0].upper()}{text[1:]}." for letter, text in REASONS.values()
              if letter]
    lines += ["", f"\\* The revision does not print {not_printed}: the reference models' "
              "DPO → OT30K runs and the released Instruct models. They are compared with their "
              "results in the manifest.", "",
              f"In submitted Table 3, {none} of the {len(p2cells)} Phase 2 task cells that "
              "changed for reason b (GSM8K and IFEval aside) are none of the three released runs "
              "(`submitted_in_release`).", ""]
    return "\n".join(lines)


def render_all(manifest_path: Path, results_root: Path, out_dir: Path) -> list[Path]:
    cells = reconcile(manifest_path, results_root)
    summary = out_dir / "summary.md"
    summary.parent.mkdir(parents=True, exist_ok=True)
    summary.write_text(summary_md(summarize(cells), cells))
    return [write_cells(cells, out_dir / "submission_vs_revision.csv"), summary]


def main(manifest_path: Path, results_root: Path, out_dir: Path) -> int:
    for path in render_all(manifest_path, results_root, out_dir):
        print(f"  wrote {path}")
    return 0
