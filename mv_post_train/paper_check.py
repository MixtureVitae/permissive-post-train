"""`mvpt build`, `mvpt check-paper` and `mvpt export-paper`.

build         runs every target of targets.py (or the ones named), in order.
build --check runs them into a temporary folder and compares every file
              they write, byte for byte, with the committed one under results/.
              preview.pdf is left out, because pdflatex versions differ, and
              a PNG preview counts as the same if its pixels are, because its
              bytes depend on the zlib that Pillow was built with.
check-paper   compares a paper source tree with the generated files:
              - every file of targets.PAPER_FILES must be byte-identical
              - every macro key the text uses must exist
              - tables and figures the paper includes that no target makes
                are listed as hand-made
              - files under tables/ and figures/ that the paper does not
                include are listed as unused
              - numbers typed into the prose are listed (report only)
              With --numbers it lists every number that differs between the
              paper's copy and the generated file, with the reason from
              targets.REASONS, and fails if a changed number has no reason.
export-paper  copies the generated files into a paper source tree.
"""
from __future__ import annotations

import difflib
import re
import shutil
import subprocess
import sys
import tempfile
from dataclasses import dataclass, field
from pathlib import Path

from . import targets
from .render.macros import parse_macros
from .render.workshop_check import _COMMAND, _NUMBER, _label

# ---------------------------------------------------------------------------
# build


def _files(path: Path) -> list[Path]:
    if path.is_file():
        return [path]
    return sorted(p for p in path.rglob("*") if p.is_file()) if path.is_dir() else []


def _same(committed: Path, generated: Path) -> bool:
    """Byte for byte, or for two PNGs, pixel for pixel."""
    if committed.read_bytes() == generated.read_bytes():
        return True
    if committed.suffix != ".png":
        return False
    from PIL import Image
    with Image.open(committed) as a, Image.open(generated) as b:
        return a.size == b.size and a.mode == b.mode and a.tobytes() == b.tobytes()


def _run(target: targets.Target, root: Path) -> int:
    """Run one target in its own Python process, so the memory a target uses
    (up to about 3 GB for the older tables) is freed before the next starts."""
    print(f"== {target.name}: {target.command}" + ("" if root == Path(".") else f" (into {root})"),
          flush=True)
    code = subprocess.run([sys.executable, "-m", "mv_post_train.paper_check", "run-target",
                           target.name, str(root)]).returncode
    if code:
        print(f"error: target {target.name} failed")
    return code


def build_main(only: list[str], check: bool) -> int:
    chosen = targets.order(only or None)
    if not check:
        return next((code for target in chosen if (code := _run(target, Path(".")))), 0)
    with tempfile.TemporaryDirectory(prefix="mvpt-build-") as tmp:
        root = Path(tmp)
        for target in chosen:
            if _run(target, root):
                return 1
        produced = [p for p in _files(root / "results") if p.name != "preview.pdf"]
        differ, new = [], []
        for path in produced:
            committed = Path(path.relative_to(root))
            if not committed.exists():
                new.append(committed)
            elif not _same(committed, path):
                differ.append(committed)
        made = {Path(p.relative_to(root)) for p in _files(root / "results")}
        stale = [p for t in chosen for p in _files(Path(t.out))
                 if p not in made and p.name not in ("README.md", "preview.pdf")]
    print(f"\n{len(produced)} generated files compared with results/: {len(differ)} differ, "
          f"{len(new)} not committed, {len(stale)} committed but not generated")
    for label, group in (("differs", differ), ("not committed", new),
                         ("committed but not generated", stale)):
        for p in group:
            print(f"  {label}: {p}")
    return 1 if differ or new or stale else 0


# ---------------------------------------------------------------------------
# the paper's include graph

_COMMENT = re.compile(r"(?<!\\)%.*")
_INPUT = re.compile(r"\\(?:input|include)\{([^}]+)\}")
_GRAPHIC = re.compile(r"\\includegraphics(?:\[[^\]]*\])?\{([^}]+)\}")
_TABLE = re.compile(r"\\begin\{table\*?\}(.*?)\\end\{table\*?\}", re.S)
_LABEL = re.compile(r"\\label\{([^}]+)\}")
MAIN = "main_neurips_2026.tex"


def strip_comments(tex: str) -> str:
    return "\n".join(_COMMENT.sub("", line) for line in tex.splitlines())


def _resolve(paper: Path, name: str, suffixes: tuple[str, ...]) -> Path | None:
    for cand in [name] + [name + s for s in suffixes]:
        if (paper / cand).is_file():
            return paper / cand
    return None


def include_graph(paper: Path, main: str = MAIN) -> tuple[list[Path], list[Path]]:
    """(.tex files, graphics files) the paper includes, starting from `main`."""
    texs, graphics, todo = [], [], [paper / main]
    while todo:
        tex = todo.pop(0)
        if tex in texs:
            continue
        texs.append(tex)
        body = strip_comments(tex.read_text(errors="replace"))
        for name in _INPUT.findall(body):
            found = _resolve(paper, name.strip(), (".tex",))
            if found is not None:
                todo.append(found)
        for name in _GRAPHIC.findall(body):
            found = _resolve(paper, name.strip(), (".pdf", ".png", ".jpg"))
            if found is not None and found not in graphics:
                graphics.append(found)
    return texs, graphics


def aux_numbers(paper: Path, main: str = MAIN) -> dict[str, str]:
    """label -> printed number, from the paper's .aux file if it was compiled."""
    aux = paper / main.replace(".tex", ".aux")
    if not aux.exists():
        return {}
    return dict(re.findall(r"\\newlabel\{([^}]+)\}\{\{([^}]*)\}", aux.read_text(errors="replace")))


# ---------------------------------------------------------------------------
# prose numbers

_MACRO_CALL = re.compile(r"\\(?:ptwo|pstat|pfl|pold|langci|pzoo|plang|psetup)\{[^}]*\}")
_DROP = re.compile(r"\\(?:cite[a-z]*|[Cc]ref|ref|eqref|label|input|include|url|href|hspace|"
                   r"vspace|includegraphics|setlength|resizebox|acld|todo)\*?(?:\[[^\]]*\])?\{[^}]*\}")
_ENVS = re.compile(r"\\begin\{(table\*?|tabular\*?|tabularx|figure\*?|algorithm|equation\*?)\}"
                   r".*?\\end\{\1\}", re.S)
_PROSE_NUMBER = re.compile(r"(?<![A-Za-z\d.\\@_-])[-+\u2212]?\d+(?:\.\d+)?(?![\dA-Za-z])")


def prose_numbers(tex: str) -> list[str]:
    """Numbers typed into prose: outside tables, figures, macros, references
    and lengths. Years (1900 to 2099) are left out."""
    body = _DROP.sub(" ", _MACRO_CALL.sub(" ", _ENVS.sub(" ", strip_comments(tex))))
    body = re.sub(r"\d*\.?\d+\\(?:linewidth|textwidth|columnwidth)", " ", body)
    return [n for n in _PROSE_NUMBER.findall(body) if not re.fullmatch(r"(19|20)\d\d", n)]


# ---------------------------------------------------------------------------
# number changes

@dataclass
class Change:
    paper: str
    item: str
    where: str     # table row and column, or macro key
    old: str
    new: str
    reason: str = ""


@dataclass
class _Row:
    label: str
    cells: list[list[str]]
    header: list[str]
    first: int = 1      # index of the first data cell in the full row


_LAYOUT = re.compile(r"\\(?:multirow\{\d+\}\{[^}]*\}|rotatebox\{\d+\}|cmidrule(?:\([^)]*\))?\{[^}]*\}"
                     r"|addlinespace\[[^\]]*\]|(?:shortstack|makecell)(?:\[[a-z]\])?)|\^\\dagger")
#: A number in a table cell. Unlike _NUMBER, the "1" of "1.5B" in a label such
#: as "Qwen2.5-1.5B" is not one.
_CELL_NUMBER = re.compile(r"(?<![A-Za-z\d.])[-+]?\d+(?:\.\d+)?(?![\dA-Za-z]|\.\d)")


def _numbers(cell: str) -> list[str]:
    cell = _LAYOUT.sub(" ", cell)
    return [n.lstrip("+") for n in _CELL_NUMBER.findall(_COMMAND.sub(" ", cell))]


def _is_value(cell: str) -> bool:
    """A cell of numbers only, such as "54.0" or "45.2 $\\pm$ 0.3". A label
    such as "(no Phase 1)" holds a number but also words."""
    text = _COMMAND.sub(" ", _LAYOUT.sub(" ", cell))
    return bool(_CELL_NUMBER.search(text)) and not re.search(r"[A-Za-z]", text)


_TABULAR = re.compile(r"\\begin\{tabular\}.*?\\end\{tabular\}", re.S)


def table_rows(tex: str) -> list[_Row]:
    """The data rows of a LaTeX table file, as `_block_rows` reads them.

    A file can show one table in several panels, each its own `tabular`. A
    panel with the same column names as the first holds more rows (a row
    panel). A panel with other column names holds more columns of the rows
    just above it (a column panel): its rows must have the same labels, in the
    same order, and their cells are appended to them. So a table and the same
    table cut into panels give the same rows."""
    blocks = _TABULAR.findall(tex)
    if len(blocks) <= 1:
        return _block_rows(tex)
    groups: list[list[_Row]] = []
    for block in blocks:
        rows = _block_rows(block)
        if not rows:
            continue
        if not groups or rows[0].header == groups[0][0].header:
            groups.append(rows)
            continue
        base = groups[-1]
        if [r.label for r in rows] != [r.label for r in base]:
            raise ValueError("a column panel's rows do not match the rows of the panel above: "
                             f"{[r.label for r in rows][:3]} against {[r.label for r in base][:3]}")
        groups[-1] = [_Row(a.label, a.cells + b.cells, a.header + b.header[b.first:], a.first)
                      for a, b in zip(base, rows)]
    return [r for g in groups for r in g]


def _block_rows(tex: str) -> list[_Row]:
    """The data rows of a LaTeX table (lines with `&`), each cell's numbers,
    and the column names, which are the row just above the first midrule.
    The leading cells without numbers are the row's label: a section, a
    model and an indented stage, each kept from the rows above when empty."""
    rows, header, path, stage, last = [], [], [], "", None
    for line in tex.splitlines():
        s = line.strip()
        if s.startswith("%"):
            continue
        if r"\midrule" in s:
            if not header and last is not None:
                header = [_label(_LAYOUT.sub(" ", c)) for c in last]
                rows = [_Row(r.label, r.cells, header, r.first) for r in rows
                        if r.cells != [_numbers(c) for c in last[r.first:]]]
                path, stage = [], ""              # the column names are no row's label
            continue
        if "&" not in s:
            if r"\multirow" in s:                 # a section label on a line of its own
                path, stage = [_label(_LAYOUT.sub(" ", s.replace("\\\\", "")))], ""
            continue
        cells = s.replace("\\\\", "").split("&")
        last = cells
        nums = [_numbers(c) for c in cells]
        first = next((i for i, c in enumerate(cells) if _is_value(c) or c_is_dash(c)), len(cells))
        for i, c in enumerate(cells[:first]):
            lab = _label(_LAYOUT.sub(" ", c))
            if not lab:
                continue
            if r"\quad" in c:
                stage = lab
            else:
                path, stage = path[:i] + [""] * max(0, i - len(path)) + [lab], ""
        if first == len(cells):
            continue                              # a row of labels only
        shown = " / ".join(x for x in path + [stage] if x)
        rows.append(_Row(shown or "(row)", nums[first:], header, first))
    return rows


def c_is_dash(cell: str) -> bool:
    return cell.strip() in ("--", "---")


def table_changes(old: str, new: str) -> list[tuple[str, str, str]]:
    """(row and column, old numbers, new numbers) for every cell that differs."""
    a, b = table_rows(old), table_rows(new)
    out = []
    matcher = difflib.SequenceMatcher(a=[r.label for r in a], b=[r.label for r in b],
                                      autojunk=False)
    show = lambda cell: "/".join(cell) or "empty"  # noqa: E731
    for tag, i1, i2, j1, j2 in matcher.get_opcodes():
        if tag in ("equal", "replace") and i2 - i1 == j2 - j1:
            for ra, rb in zip(a[i1:i2], b[j1:j2]):
                for k in range(max(len(ra.cells), len(rb.cells))):
                    x = ra.cells[k] if k < len(ra.cells) else None
                    y = rb.cells[k] if k < len(rb.cells) else None
                    if x != y:
                        j = k + rb.first
                        name = rb.header[j] if j < len(rb.header) else ""
                        repeated = name and rb.header.count(name) > 1
                        col = (f"{name} (column {j + 1})" if repeated else name) or f"column {j + 1}"
                        out.append((f"{rb.label} / {col}", "missing" if x is None else show(x),
                                    "missing" if y is None else show(y)))
            continue
        out += [(r.label, "row", "removed") for r in a[i1:i2]]
        out += [(r.label, "added", "row") for r in b[j1:j2]]
    return out


def figure_changes(old: Path, new: Path) -> list[tuple[str, str, str]]:
    """The numbers printed in a figure that differ, read with PyMuPDF if it is
    installed. Otherwise the figure is reported as a whole."""
    try:
        import fitz
    except ImportError:
        return [("the figure", "the paper's file", "a different file")]

    def numbers(p: Path) -> list[str]:
        with fitz.open(p) as doc:
            text = " ".join(page.get_text() for page in doc).replace("\u2212", "-")
        return [n.lstrip("+") for n in _NUMBER.findall(text)]

    a, b = numbers(old), numbers(new)
    out = []
    for tag, i1, i2, j1, j2 in difflib.SequenceMatcher(a=a, b=b, autojunk=False).get_opcodes():
        if tag != "equal":
            out.append(("printed numbers", " ".join(a[i1:i2]) or "none", " ".join(b[j1:j2]) or "none"))
    return out or [("the figure", "same printed numbers", "other drawing")]


def macro_changes(old: Path, new: Path) -> list[tuple[str, str, str]]:
    """Keys whose value changes or that are removed. New keys are not changes,
    since the paper does not use them yet (`new_macro_keys` counts them)."""
    _, a = parse_macros(old)
    _, b = parse_macros(new)
    return [(k, a[k], b.get(k, "removed")) for k in sorted(a) if a[k] != b.get(k)]


def new_macro_keys(paper: Path, results_root: Path = Path(".")) -> int:
    n = 0
    for pf in targets.PAPER_FILES:
        old, gen = paper / pf.paper, results_root / pf.generated
        if pf.item.endswith("macros") and gen.exists():
            _, b = parse_macros(gen)
            n += len(set(b) - (set(parse_macros(old)[1]) if old.exists() else set()))
    return n


def _reason(paper: str, where: str) -> str:
    import fnmatch
    hits = [r for (f, pattern), r in targets.REASONS.items()
            if f == paper and fnmatch.fnmatchcase(where, pattern)]
    return hits[0] if hits else ""


def table_by_label(paper: Path, label: str) -> str | None:
    """The table environment labelled `label` anywhere the paper includes."""
    texs, _ = include_graph(paper)
    for tex in texs:
        for block in _TABLE.findall(strip_comments(tex.read_text(errors="replace"))):
            if f"\\label{{{label}}}" in block:
                return block
    return None


def number_changes(paper: Path, results_root: Path = Path(".")) -> list[Change]:
    """Every number that differs between the paper and the generated files. A
    generated table the paper has no file for yet is compared with the table
    of the same label that the paper has, typed in by hand."""
    out = []
    for pf in targets.PAPER_FILES:
        gen, old = results_root / pf.generated, paper / pf.paper
        if not gen.exists() or (old.exists() and gen.read_bytes() == old.read_bytes()):
            continue
        if gen.suffix == ".tex" and pf.item.endswith("macros"):
            if not old.exists():
                continue
            rows = macro_changes(old, gen)
        elif gen.suffix == ".tex":
            if old.exists():
                before = old.read_text()
            else:
                labels = _LABEL.findall(gen.read_text())
                before = table_by_label(paper, labels[0]) if labels else None
                if before is None:
                    continue
            rows = table_changes(before, gen.read_text())
        else:
            if not old.exists():
                continue
            rows = figure_changes(old, gen)
        out += [Change(pf.paper, pf.item, w, o, n, _reason(pf.paper, w)) for w, o, n in rows]
    return out


# ---------------------------------------------------------------------------
# check-paper

@dataclass
class PaperReport:
    identical: list = field(default_factory=list)
    differ: list = field(default_factory=list)
    missing: list = field(default_factory=list)
    allowed: list = field(default_factory=list)
    hand_tables: list = field(default_factory=list)      # (file, label)
    hand_graphics: list = field(default_factory=list)
    missing_keys: list = field(default_factory=list)     # (file, command, key)
    unused: list = field(default_factory=list)
    prose: dict = field(default_factory=dict)            # file -> numbers


def check_paper(paper: Path, results_root: Path = Path(".")) -> PaperReport:
    rep = PaperReport()
    mapped = {pf.paper: pf for pf in targets.PAPER_FILES}
    for pf in targets.PAPER_FILES:
        gen, cp = results_root / pf.generated, paper / pf.paper
        if not cp.exists():
            rep.missing.append(pf)
        elif gen.read_bytes() == cp.read_bytes():
            rep.identical.append(pf)
        elif pf.paper in targets.PAPER_ALLOWED:
            rep.allowed.append(pf)
        else:
            rep.differ.append(pf)
    texs, graphics = include_graph(paper)
    rel = lambda p: str(p.relative_to(paper))  # noqa: E731
    macro_keys = {}
    for pf in targets.PAPER_FILES:
        if pf.item.endswith("macros") and (results_root / pf.generated).exists():
            cmd, values = parse_macros(results_root / pf.generated)
            macro_keys[cmd] = set(values)
    for tex in texs:
        name = rel(tex)
        if name in mapped:
            continue
        body = strip_comments(tex.read_text(errors="replace"))
        for block in _TABLE.findall(body):
            if not any(f"\\input{{{m.removesuffix('.tex')}}}" in block or f"\\input{{{m}}}" in block
                       for m in mapped):
                labels = _LABEL.findall(block)
                rep.hand_tables.append((name, labels[0] if labels else "(no label)"))
        for cmd, keys in macro_keys.items():
            for key in re.findall(rf"\\{cmd}\{{([^}}]*)\}}", body):
                if key not in keys:
                    rep.missing_keys.append((name, cmd, key))
        numbers = prose_numbers(body)
        if numbers:
            rep.prose[name] = numbers
    rep.hand_graphics = [rel(g) for g in graphics if rel(g) not in mapped]
    used = {rel(t) for t in texs} | {rel(g) for g in graphics}
    for sub in ("tables", "figures"):
        for p in sorted((paper / sub).glob("*")) if (paper / sub).is_dir() else []:
            if p.is_file() and p.suffix in (".tex", ".pdf", ".png") and rel(p) not in used:
                rep.unused.append(rel(p))
    return rep


def _print_report(rep: PaperReport, labels: dict[str, str]) -> None:
    n = len(targets.PAPER_FILES)
    print(f"Generated files: {len(rep.identical)} of {n} identical, {len(rep.differ)} differ, "
          f"{len(rep.missing)} missing in the paper, {len(rep.allowed)} allowed to differ")
    for group, label in ((rep.differ, "differs"), (rep.missing, "missing in the paper"),
                         (rep.allowed, "allowed to differ")):
        for pf in group:
            print(f"  {label}: {pf.paper} ({pf.item}, from {pf.generated})")
    print(f"\nMacro keys the text uses that no generated file defines: {len(rep.missing_keys)}")
    for name, cmd, key in rep.missing_keys:
        print(f"  {name}: \\{cmd}{{{key}}}")
    print(f"\nHand-made tables (no target makes them): {len(rep.hand_tables)}")
    for name, label in rep.hand_tables:
        num = labels.get(label)
        print(f"  {name}: {label}" + (f" (Table {num})" if num else ""))
    print(f"\nHand-made figures: {len(rep.hand_graphics)}")
    for g in rep.hand_graphics:
        print(f"  {g}")
    total = sum(len(v) for v in rep.prose.values())
    print(f"\nNumbers typed into the prose (report only): {total}")
    for name, numbers in rep.prose.items():
        print(f"  {name}: {len(numbers)}")
    print(f"\nFiles under tables/ and figures/ the paper does not include: {len(rep.unused)}")
    for u in rep.unused:
        print(f"  {u}")


def _md(where: str) -> str:
    """A row label for Markdown: "\\to SFT" as "→SFT"."""
    return where.replace("\\to ", "→")


def changes_markdown(changes: list[Change], new_keys: int = 0) -> str:
    extra = f" The generated macro files add {new_keys} new keys, which the paper does not use yet." \
        if new_keys else ""
    if not changes:
        return ("**No number changes.** Every number the paper prints from a generated file "
                "is the same." + extra)
    lines = [f"{len(changes)} numbers change." + extra, "",
             "| Item | Where | Old | New | Reason |", "|---|---|---|---|---|"]
    lines += [f"| {c.item} | {_md(c.where)} | {c.old} | {c.new} | {c.reason or '**no reason given**'} |"
              for c in changes]
    return "\n".join(lines)


def changes_by_row(changes: list[Change], new_keys: int = 0) -> str:
    """The same list with one line per table row and reason: each changed
    cell as `column old→new`. For tables where most cells change."""
    if not changes:
        return changes_markdown(changes, new_keys)
    groups: dict[tuple[str, str, str], list[str]] = {}
    for c in changes:
        row, _, col = c.where.rpartition(" / ")
        row = row or c.where
        groups.setdefault((c.item, row, c.reason), []).append(
            f"{col} {c.old}→{c.new}" if row != c.where else f"{c.old}→{c.new}")
    extra = f" The generated macro files add {new_keys} new keys, which the paper does not use yet." \
        if new_keys else ""
    reasons = sorted({c.reason for c in changes if c.reason})
    note = {r: i + 1 for i, r in enumerate(reasons)}
    lines = [f"{len(changes)} numbers change, in {len(groups)} rows." + extra, "",
             "| Item | Row | Changes | Reason |", "|---|---|---|---|"]
    lines += [f"| {item} | {_md(row)} | {', '.join(cells)} | "
              f"{f'({note[reason]})' if reason else '**no reason given**'} |"
              for (item, row, reason), cells in groups.items()]
    lines += [""] + [f"({note[r]}) {r}" for r in reasons]
    return "\n".join(lines)


def check_paper_main(paper: Path, numbers: bool, markdown: bool, by_row: bool = False) -> int:
    if not (paper / MAIN).exists():
        print(f"error: {paper} has no {MAIN}")
        return 1
    if numbers:
        changes = number_changes(paper)
        if markdown:
            show = changes_by_row if by_row else changes_markdown
            print(show(changes, new_macro_keys(paper)))
        else:
            print(f"{len(changes)} numbers differ between {paper} and the generated files")
            for c in changes:
                print(f"  {c.item} | {c.where}: {c.old} -> {c.new}"
                      f"  [{c.reason or 'NO REASON GIVEN'}]")
        return 1 if any(not c.reason for c in changes) else 0
    rep = check_paper(paper)
    _print_report(rep, aux_numbers(paper))
    return 1 if rep.differ or rep.missing or rep.missing_keys else 0


def export_paper_main(paper: Path) -> int:
    """Copy every generated file into a paper source tree; list the ones that changed."""
    changed = []
    for pf in targets.PAPER_FILES:
        src, dest = Path(pf.generated), paper / pf.paper
        if dest.exists() and dest.read_bytes() == src.read_bytes():
            continue
        dest.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(src, dest)
        changed.append(pf.paper)
    print(f"{len(changed)} files copied into {paper}")
    for c in changed:
        print(f"  {c}")
    return 0


if __name__ == "__main__":
    # `python -m mv_post_train.paper_check run-target NAME ROOT`, used by build_main.
    if len(sys.argv) == 4 and sys.argv[1] == "run-target":
        sys.exit(targets.BY_NAME[sys.argv[2]].run(Path(sys.argv[3])))
    sys.exit("usage: python -m mv_post_train.paper_check run-target NAME ROOT")
