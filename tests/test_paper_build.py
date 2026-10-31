"""Tests for the paper build (mv_post_train/targets.py) and its checks
(mv_post_train/paper_check.py, render/macros.py).

They pin the target order, that every generated paper file is committed and
mapped once, that the macro files keep one form and read back, that number
words are spelled out, and what `check-paper` reports on a small paper tree:
a generated file that differs, a hand-made table, a missing macro key, an
unused file, and a changed number with and without a reason.
"""
from __future__ import annotations

from pathlib import Path

import pytest

from mv_post_train import paper_check as pc
from mv_post_train import targets
from mv_post_train.render.macros import macro_lines, number_word, parse_macros

ROOT = Path(__file__).resolve().parents[1]


def test_targets_run_after_the_ones_they_read():
    names = [t.name for t in targets.order()]
    assert names.index("manifest") < names.index("phase2")
    assert [t.name for t in targets.order(["statistics"])] == ["manifest", "statistics"]
    # Every renderer reads its results through the manifest.
    assert all("manifest" in t.needs for t in targets.TARGETS if t.name != "manifest")
    with pytest.raises(KeyError):
        targets.order(["no-such-target"])


def test_a_png_is_the_same_if_its_pixels_are(tmp_path):
    from PIL import Image
    img = Image.new("RGBA", (40, 20), (200, 30, 30, 255))
    a, b, c = tmp_path / "a.png", tmp_path / "b.png", tmp_path / "c.png"
    img.save(a, compress_level=1)
    img.save(b, compress_level=9)
    img.copy().rotate(90, expand=True).save(c)
    assert a.read_bytes() != b.read_bytes()
    assert pc._same(a, b)
    assert not pc._same(a, c)
    t1, t2 = tmp_path / "t1.tex", tmp_path / "t2.tex"
    t1.write_text("1.0\n"); t2.write_text("1.00\n")
    assert pc._same(t1, t1) and not pc._same(t1, t2)


def test_every_paper_file_is_committed_and_mapped_once():
    pfs = targets.PAPER_FILES
    assert len({p.generated for p in pfs}) == len({p.paper for p in pfs}) == len(pfs)
    for pf in pfs:
        assert (ROOT / pf.generated).is_file(), pf.generated
        assert pf.target in targets.BY_NAME
        assert pf.generated.startswith(targets.BY_NAME[pf.target].out)
    labels = [line for pf in pfs if pf.generated.endswith(".tex")
              for line in (ROOT / pf.generated).read_text().splitlines() if "\\label{" in line]
    assert len(labels) == len(set(labels))


def test_number_words():
    assert [number_word(n) for n in (4, 11, 18, 24, 36, 48, 100)] == [
        "four", "eleven", "eighteen", "twenty-four", "thirty-six", "forty-eight", "100"]


def test_every_macro_file_reads_back():
    for pf in targets.PAPER_FILES:
        if pf.item.endswith("macros"):
            cmd, values = parse_macros(ROOT / pf.generated)
            assert values and pf.item.startswith(f"\\{cmd} ")
    lines = macro_lines("pxy", {"b.c": r"\ensuremath{+1.0}", "a": "4"}, sort=True)
    assert lines[0] == (r"\providecommand{\pxy}[1]{\ifcsname pxy@#1\endcsname\csname pxy@#1"
                        r"\endcsname\else\textbf{??#1??}\fi}")
    assert lines[1].startswith(r"\expandafter\def\csname pxy@a\endcsname")


TABLE = r"""\begin{table}[t]
\caption{A table.}\label{tab:x}
\begin{tabular}{lrr}
\toprule
Model & A24 & Avg \\
\midrule
MV & 1.0 & 10.5 \\
\quad $\to$ SFT & 2.0 & 12.0 \\
\bottomrule
\end{tabular}
\end{table}
"""


def _paper(tmp_path: Path, table: str, macro_use: str) -> Path:
    paper = tmp_path / "paper"
    (paper / "tables").mkdir(parents=True)
    (paper / "figures").mkdir()
    (paper / "main_neurips_2026.tex").write_text(
        "\\input{tables/x_numbers}\n\\input{tables/x}\n"
        "The model scores " + macro_use + " and 31.7 points. % 99 in a comment\n"
        "\\begin{table}\\caption{Typed in.}\\label{tab:hand}\\begin{tabular}{lr}"
        "a & 1 \\\\\\end{tabular}\\end{table}\n")
    (paper / "tables" / "x.tex").write_text(table)
    (paper / "tables" / "x_numbers.tex").write_text(
        "\n".join(macro_lines("pxy", {"mv.avg": "10.5"})) + "\n")
    (paper / "tables" / "old.tex").write_text("unused\n")
    return paper


@pytest.fixture
def generated(tmp_path, monkeypatch):
    res = tmp_path / "repo"
    (res / "results" / "x" / "tables").mkdir(parents=True)
    (res / "results" / "x" / "tables" / "x.tex").write_text(TABLE.replace("12.0", "12.5"))
    (res / "results" / "x" / "tables" / "x_numbers.tex").write_text(
        "\n".join(macro_lines("pxy", {"mv.avg": "10.5"})) + "\n")
    monkeypatch.setattr(targets, "PAPER_FILES", (
        targets.PaperFile("results/x/tables/x.tex", "tables/x.tex", "Table 1", "x"),
        targets.PaperFile("results/x/tables/x_numbers.tex", "tables/x_numbers.tex",
                          r"\pxy macros", "x")))
    monkeypatch.setattr(targets, "REASONS", {})
    return res


def test_check_paper_reports_what_is_not_generated(tmp_path, generated):
    paper = _paper(tmp_path, TABLE, r"\pxy{mv.avg} and \pxy{mv.sd}")
    rep = pc.check_paper(paper, generated)
    assert [p.paper for p in rep.differ] == ["tables/x.tex"]
    assert [p.paper for p in rep.identical] == ["tables/x_numbers.tex"]
    assert rep.hand_tables == [("main_neurips_2026.tex", "tab:hand")]
    assert rep.missing_keys == [("main_neurips_2026.tex", "pxy", "mv.sd")]
    assert rep.unused == ["tables/old.tex"]
    assert rep.prose["main_neurips_2026.tex"] == ["31.7"]


def test_a_changed_number_needs_a_reason(tmp_path, generated, monkeypatch):
    paper = _paper(tmp_path, TABLE, r"\pxy{mv.avg}")
    changes = pc.number_changes(paper, generated)
    assert [(c.item, c.where, c.old, c.new, c.reason) for c in changes] == [
        ("Table 1", "MV / \\to SFT / Avg", "12.0", "12.5", "")]
    assert "**no reason given**" in pc.changes_markdown(changes)
    monkeypatch.setattr(targets, "REASONS", {("tables/x.tex", "MV / *"): "One decimal."})
    assert pc.number_changes(paper, generated)[0].reason == "One decimal."


def test_a_hand_made_table_is_compared_by_its_label(tmp_path, generated):
    paper = _paper(tmp_path, TABLE, r"\pxy{mv.avg}")
    (paper / "tables" / "x.tex").unlink()
    main = paper / "main_neurips_2026.tex"
    main.write_text(main.read_text().replace("\\input{tables/x}\n", TABLE.replace("& 2.0 &", "& 3.0 &")))
    changes = pc.number_changes(paper, generated)
    assert [(c.where, c.old, c.new) for c in changes] == [
        ("MV / \\to SFT / A24", "3.0", "2.0"), ("MV / \\to SFT / Avg", "12.0", "12.5")]


def test_docs_list_every_paper_file():
    doc = (ROOT / "docs" / "paper_build.md").read_text()
    missing = [pf.generated for pf in targets.PAPER_FILES if f"`{pf.generated}`" not in doc]
    missing += [t.name for t in targets.TARGETS if f"`{t.name}`" not in doc]
    assert not missing, f"docs/paper_build.md does not list {missing}"


def test_digits_in_a_row_label_are_not_values():
    tex = TABLE.replace("MV & 1.0 & 10.5", "Qwen2.5-1.5B & & \\\\\n\\quad $\\to$OT30K (\\textbf{no Phase 1}) & 1.0 & 10.5"
                        ).replace("\\quad $\\to$ SFT & 2.0", "MV (0.4B)$^\\dagger$ & 2.0")
    assert [(r.label, r.cells) for r in pc.table_rows(tex)] == [
        ("Qwen2.5-1.5B / \\to OT30K ( no Phase 1 )", [["1.0"], ["10.5"]]),
        ("MV (0.4B)", [["2.0"], ["12.0"]])]


def test_changes_by_row_lists_every_cell_once():
    changes = [pc.Change("tables/x.tex", "Table 1", f"MV / \\to SFT / {col}", "2.0", new, "One decimal.")
               for col, new in (("A24", "2.4"), ("Avg", "12.3"))]
    text = pc.changes_by_row(changes)
    assert "| Table 1 | MV / →SFT | A24 2.0→2.4, Avg 2.0→12.3 | (1) |" in text
    assert text.splitlines()[0] == "2 numbers change, in 1 rows."
    assert text.endswith("(1) One decimal.")


def _tabular(cols: list[str], rows: list[tuple[str, list[str]]]) -> str:
    lines = [r"\begin{tabular}{l" + "r" * len(cols) + "}", r"\toprule",
             r"\textbf{Model} & " + " & ".join(rf"\textbf{{{c}}}" for c in cols) + r" \\", r"\midrule"]
    lines += [f"{label} & " + " & ".join(cells) + r" \\" for label, cells in rows]
    return "\n".join(lines + [r"\bottomrule", r"\end{tabular}"])


def test_column_panels_parse_like_one_table():
    from mv_post_train.paper_check import table_rows, table_changes
    rows = [("MV", ["1.0", "2.0", "3.0"]), (r"\quad $\to$ SFT", ["4.0", "5.0", "6.0"])]
    one = _tabular(["A", "B", "C"], rows)
    two = "\n".join([_tabular(["A"], [(l, c[:1]) for l, c in rows]), r"\par\medskip",
                     _tabular(["B", "C"], [(l, c[1:]) for l, c in rows])])
    assert [(r.label, r.cells) for r in table_rows(two)] == [(r.label, r.cells) for r in table_rows(one)]
    assert table_rows(two)[0].header == ["Model", "A", "B", "C"]
    assert table_changes(one, two) == []


def test_row_panels_append_rows_and_mismatched_column_panels_raise():
    from mv_post_train.paper_check import table_rows
    a = _tabular(["A"], [("MV", ["1.0"])])
    b = _tabular(["A"], [("Comma", ["2.0"])])
    assert [r.label for r in table_rows(a + "\n" + b)] == ["MV", "Comma"]
    with pytest.raises(ValueError):
        table_rows(a + "\n" + _tabular(["B"], [("Comma", ["2.0"])]))
