"""One PDF holding a set of rendered tables and figures, for reading them
without the paper.

The tables are `\\input` as written, so the PDF also shows that they compile.
References to tables of the paper that are not in the set print as "??".
Needs `pdflatex` with booktabs, multirow, graphicx, bm and amsmath. Without
it, a warning is printed and no PDF is written.
"""
from __future__ import annotations

import shutil
import subprocess
import tempfile
from pathlib import Path

PREAMBLE = r"""\documentclass{article}
\usepackage[margin=0.7in]{geometry}
\usepackage{booktabs,multirow,graphicx,bm,amsmath,makecell,caption}
\pdfinfoomitdate=1 \pdftrailerid{} \pdfsuppressptexinfo=-1
\begin{document}
"""


def build(title: str, tables: list[Path], figures: list[Path], out: Path,
          preamble: str = "") -> Path | None:
    """`preamble` adds packages or definitions that the tables need."""
    body = [f"\\pdfinfo{{/Title ({title})}}"]
    body += [f"\\input{{{t.resolve()}}}" for t in tables]
    body += [f"\\begin{{figure}}[p]\\centering\\includegraphics[width=\\linewidth]{{{f.resolve()}}}"
             f"\\par\\smallskip\\texttt{{{f.name.replace('_', chr(92) + '_')}}}\\end{{figure}}"
             for f in figures]
    doc = PREAMBLE.replace("\\begin{document}", preamble + "\\begin{document}") \
        + "\n".join(body) + "\n\\end{document}\n"
    with tempfile.TemporaryDirectory() as tmp:
        tex = Path(tmp) / "preview.tex"
        tex.write_text(doc)
        try:
            for _ in range(2):      # the second pass resolves references within the set
                subprocess.run(["pdflatex", "-interaction=nonstopmode", "-halt-on-error",
                                tex.name], cwd=tmp, check=True, capture_output=True, text=True)
        except FileNotFoundError:
            print("Warning: pdflatex is not on PATH, so no preview PDF was written.")
            return None
        except subprocess.CalledProcessError as exc:
            print(f"Warning: the preview did not compile:\n{exc.stdout[-2000:]}")
            return None
        out.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy(Path(tmp) / "preview.pdf", out)
    return out
