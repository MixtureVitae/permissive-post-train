"""Compare rendered workshop tables with the paper's `.tex` files, number by number.

Only table rows (lines with `&`) are compared. Every number in a row is
extracted after dropping formatting (bold, math mode, \\pm, brackets), and
rows are aligned in order by their label. A table passes when every row has
the same numbers in the same order. Caption or wording changes are reported
separately and do not fail the check, since they are prose, not results.
"""
from __future__ import annotations

import difflib
import re
from dataclasses import dataclass, field
from pathlib import Path

# A number not glued to a letter or digit on either side, so "A24", "16K",
# "Comma0.1" and "OT30K" in labels and headers are not read as values.
_NUMBER = re.compile(r"(?<![A-Za-z\d.])[-+]?\d+(?:\.\d+)?(?![\dA-Za-z])")
_LABEL_NOISE = re.compile(r"\\(?:multirow|textbf|emph|textit)\b|\{\d\}|\{\*\}|[{}$]")
# LaTeX commands are dropped before numbers are read, so "\pm0.29" gives 0.29.
_COMMAND = re.compile(r"\\[A-Za-z]+")


@dataclass
class Row:
    label: str
    cells: list[list[str]]      # numbers in each cell after the label
    header: list[str]           # column names of the most recent header row


@dataclass
class TableReport:
    name: str
    rows_compared: int = 0
    mismatches: list[str] = field(default_factory=list)
    wording_lines: int = 0
    missing: bool = False

    @property
    def ok(self) -> bool:
        return not self.missing and not self.mismatches


def _label(cell: str) -> str:
    return " ".join(_LABEL_NOISE.sub(" ", cell).replace("\\quad", "").split())


def parse_rows(tex: str) -> list[Row]:
    rows, group, header = [], "", []
    for line in tex.splitlines():
        if "&" not in line or line.lstrip().startswith("%"):
            continue
        cells = line.replace("\\\\", "").split("&")
        numbers = [[n.lstrip("+") for n in _NUMBER.findall(_COMMAND.sub(" ", c))]
                   for c in cells[1:]]
        if not any(numbers):
            header = [_label(c) for c in cells[1:]]
        label = _label(cells[0])
        if label and "\\quad" not in cells[0]:
            group = label
        shown = label if label and "\\quad" not in cells[0] else f"{group} / {label or '(row)'}"
        rows.append(Row(shown, numbers, header))
    return rows


def _diff_cells(ours: Row, paper: Row) -> list[str]:
    out = []
    for k in range(max(len(ours.cells), len(paper.cells))):
        x = ours.cells[k] if k < len(ours.cells) else None
        y = paper.cells[k] if k < len(paper.cells) else None
        if x != y:
            name = paper.header[k] if k < len(paper.header) and paper.header[k] else f"cell {k + 1}"
            show = lambda v: "missing" if v is None else "/".join(v) or "empty"  # noqa: E731
            out.append(f"{name}: ours {show(x)} vs paper {show(y)}")
    return out


def compare_tables(ours: str, paper: str, name: str) -> TableReport:
    report = TableReport(name)
    a, b = parse_rows(ours), parse_rows(paper)
    matcher = difflib.SequenceMatcher(a=[r.label for r in a], b=[r.label for r in b],
                                      autojunk=False)
    for tag, i1, i2, j1, j2 in matcher.get_opcodes():
        if tag in ("equal", "replace") and i2 - i1 == j2 - j1:
            for ra, rb in zip(a[i1:i2], b[j1:j2]):
                report.rows_compared += 1
                if ra.cells != rb.cells:
                    report.mismatches.append(f"{rb.label}: " + "; ".join(_diff_cells(ra, rb)))
            continue
        report.mismatches += [f"only in ours: {r.label}" for r in a[i1:i2]]
        report.mismatches += [f"only in paper: {r.label}" for r in b[j1:j2]]
    body = lambda s: [l for l in s.splitlines() if not l.startswith("%")]  # noqa: E731
    report.wording_lines = sum(1 for l in difflib.unified_diff(body(paper), body(ours), n=0)
                               if l[:1] in "+-" and not l.startswith(("+++", "---")))
    return report


def check_dir(ours_dir: Path, paper_dir: Path, names: list[str]) -> list[TableReport]:
    reports = []
    for name in names:
        ours, paper = ours_dir / name, paper_dir / name
        if not ours.exists() or not paper.exists():
            reports.append(TableReport(name, missing=True))
            continue
        reports.append(compare_tables(ours.read_text(), paper.read_text(), name))
    return reports


def main(ours_dir: Path, paper_dir: Path) -> int:
    """Exit 1 when any table is missing or differs in a number."""
    names = sorted(p.name for p in ours_dir.glob("*.tex"))
    if not names:
        print(f"no .tex files in {ours_dir}; run `mvpt render-workshop` first")
        return 1
    reports = check_dir(ours_dir, paper_dir, names)
    for r in reports:
        if r.missing:
            print(f"MISSING  {r.name}: not in {paper_dir}")
            continue
        wording = f", {r.wording_lines} lines of wording differ" if r.wording_lines else ""
        print(f"{'OK     ' if r.ok else 'DIFFERS'}  {r.name}: {r.rows_compared} rows{wording}")
        for m in r.mismatches:
            print(f"           {m}")
    return 0 if all(r.ok for r in reports) else 1
