"""The results tables of the paper as submitted for review,
transcribed cell by cell: the "before" side of the reconciliation with the
revision (`render/reconciliation.py`).

The submission has eleven results tables: the Phase 2 endpoints (Table 1, in
the main text, and Table 3), Phase 1 for all models (Table 2), the older
reasoning evaluations (Tables 4 to 6) and the language suite (Tables 7 to
11). Every row has eleven task scores and their average. The transcription is
committed as `submission/submission_tables.csv` (input T-submission), one row
per cell, in the order the tables print them:

  table    "Table 1" to "Table 11"
  page     the PDF page
  section  a heading inside the table ("Decontaminated"), or empty
  model    the model the row belongs to, as printed
  stage    the stage as printed ("→ SFT → DPO"), empty on a model's first row
  column   the column heading ("A24", ..., "Avg")
  value    the value as printed

`mvpt transcribe-submission <pdf>` writes the file from the PDF, so anyone with
the submission can check it. `load` reads it back and stops unless every row
has the twelve columns of its table and each Avg is within rounding of the mean
of its eleven cells, so a dropped or misread number cannot pass.
"""
from __future__ import annotations

import csv
import re
from dataclasses import dataclass
from pathlib import Path

#: Table number -> the page that prints it.
PAGES = {1: 6, 2: 22, 3: 23, 4: 24, 5: 24, 6: 25, 7: 25, 8: 26, 9: 26, 10: 26, 11: 27}
REASONING = ("A24", "A25", "AMC", "M500", "GSM8K", "HE", "LCB", "MBPP", "GPQA", "JEE", "IFE", "Avg")
LANGUAGE = ("ARC-C", "ARC-E", "BoolQ", "CSQA", "COPA", "HellaS", "LAMBADA", "MMLU", "OBQA",
            "PIQA", "WinoG", "Avg")
#: How far a printed Avg may be from the mean of its printed cells. The reasoning
#: tables print one decimal, so both sides are off by at most 0.05. The language
#: tables print whole numbers, so by at most 0.5.
AVG_TOLERANCE = {REASONING: 0.1, LANGUAGE: 1.0}
FIELDS = ("table", "page", "section", "model", "stage", "column", "value")
ARROW = "→"
_VALUE = re.compile(r"^\d+\.\d$")


@dataclass(frozen=True)
class Row:
    table: str
    page: int
    section: str
    model: str
    stage: str
    columns: tuple[str, ...]
    values: tuple[float, ...]

    @property
    def label(self) -> str:
        return " / ".join(x for x in (self.section, self.model, self.stage) if x)

    def value(self, column: str) -> float:
        return self.values[self.columns.index(column)]


def columns_of(table: str) -> tuple[str, ...]:
    return REASONING if int(table.split()[1]) <= 6 else LANGUAGE


# ---------------------------------------------------------------------------
# From the PDF


def _lines(page) -> list[str]:
    return [line.strip() for line in page.get_text("text").splitlines() if line.strip()]


def _body(lines: list[str], n: int) -> list[str]:
    """The lines of Table n after its column headings."""
    start = next(i for i, line in enumerate(lines) if line.startswith(f"Table {n}:"))
    head = next(i for i in range(start, len(lines)) if lines[i] in ("Model", "Base Model"))
    cols = tuple(lines[head + 1:head + 13])
    if cols != columns_of(f"Table {n}"):
        raise ValueError(f"Table {n} has the columns {cols}.")
    return lines[head + 13:]


def _labels(lines: list[str]) -> tuple[list[str], str]:
    """(the headings and model of a row, its stage). The PDF prints the arrow as "!"."""
    names, stage = [], ""
    for line in (x.replace("!", ARROW) for x in lines):
        if line.startswith(ARROW):
            stage = line
        elif ARROW in line:
            name, rest = line.split(ARROW, 1)
            names.append(name.strip())
            stage = ARROW + rest
        else:
            names.append(line)
    return names, re.sub(rf"\s*{ARROW}\s*", f" {ARROW} ", stage).strip()


def _rows(n: int, page: int, body: list[str]) -> list[Row]:
    rows, labels, values, section, model = [], [], [], "", ""
    cols = columns_of(f"Table {n}")
    for line in body:
        if _VALUE.match(line):
            values.append(line)
            if len(values) == len(cols):
                names, stage = _labels(labels)
                if len(names) > 1:
                    section = " ".join(names[:-1])
                if names:
                    model = names[-1]
                rows.append(Row(f"Table {n}", page, section, model, stage, cols,
                                tuple(float(v) for v in values)))
                labels, values = [], []
            continue
        if values:
            raise ValueError(f"Table {n}: {line!r} inside a row of values {values}.")
        if line.startswith("Table ") or len(line.split()) > 7 or line.isdigit():
            break                                 # the caption or text after the table
        labels.append(line)
    return rows


def transcribe(pdf: Path) -> list[Row]:
    """Every row of the eleven results tables, read from the submission's PDF."""
    import fitz
    fitz.TOOLS.mupdf_display_errors(False)
    with fitz.open(pdf) as doc:
        rows = []
        for n, page in PAGES.items():
            rows += _rows(n, page, _body(_lines(doc[page - 1]), n))
    check(rows)
    return rows


# ---------------------------------------------------------------------------
# The committed file


def write(rows: list[Row], path: Path) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", newline="") as f:
        w = csv.writer(f, lineterminator="\n")
        w.writerow(FIELDS)
        for r in rows:
            for col, v in zip(r.columns, r.values):
                w.writerow((r.table, r.page, r.section, r.model, r.stage, col, f"{v:.1f}"))
    return path


def load(path: Path) -> list[Row]:
    """The rows of the transcription, in the order the submission prints them."""
    with open(path, newline="") as f:
        cells = list(csv.DictReader(f))
    rows, i = [], 0
    while i < len(cells):
        first = cells[i]
        cols = columns_of(first["table"])
        group = cells[i:i + len(cols)]
        key = [(c["table"], c["section"], c["model"], c["stage"]) for c in group]
        if [c["column"] for c in group] != list(cols) or len(set(key)) != 1:
            raise ValueError(f"{path}: the row starting at line {i + 2} does not have the "
                             f"columns {', '.join(cols)}.")
        rows.append(Row(first["table"], int(first["page"]), first["section"], first["model"],
                        first["stage"], cols, tuple(float(c["value"]) for c in group)))
        i += len(cols)
    check(rows)
    return rows


def check(rows: list[Row]) -> None:
    """Stop unless every table is there and each Avg is the mean of its row."""
    tables = {r.table for r in rows}
    missing = [f"Table {n}" for n in PAGES if f"Table {n}" not in tables]
    if missing:
        raise ValueError(f"No rows for {', '.join(missing)}.")
    for r in rows:
        mean = sum(r.values[:-1]) / (len(r.values) - 1)
        if abs(mean - r.values[-1]) > AVG_TOLERANCE[r.columns] + 1e-9:
            raise ValueError(f"{r.table}, {r.label}: Avg {r.values[-1]} but the cells average "
                             f"{mean:.2f}.")
