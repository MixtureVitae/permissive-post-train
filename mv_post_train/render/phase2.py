"""Regenerate the paper's Phase 2 tables, Figure 2 and Figure 1 from the results manifest.

    mvpt render-phase2

Reads `results/manifest.json`, and the per-question Phase 2 records for the
MATH500 intervals (`mvpt fetch-phase2-records`), and writes

  <tables-dir>/phase2_endpoints_full.tex   Table 23, every checkpoint of Table 2
  <tables-dir>/ot3_pretrain_ablation.tex   Table 3
  <tables-dir>/heldout_reasoning.tex       Table 4
  <tables-dir>/heldout_mv_grid.tex         Table 32
  <tables-dir>/eval_settings_B.tex         Table 24, the protocol
  <tables-dir>/phase2_numbers.tex          macros for the numbers in the text
  <figures-dir>/phase2_receptivity.pdf     Figure 2
  <figures-dir>/figure_pipeline.pdf        Figure 1, the pipeline (and a .png preview)
  <tables-dir>/../preview.pdf              all of the above, compiled with pdflatex
  <art-dir>/phase2_vs_submission.csv       every submitted cell next to its new value

The table files keep the paper's basenames and labels, so they replace the
hand-copied ones. See `render/phase2_tables.py` for the rules every number
follows, and docs/phase2_tables.md for what changes against the submission.
"""
from __future__ import annotations

from pathlib import Path

from ..manifest import phase2 as p2
from ..manifest.reader import Manifest, ManifestError
from . import phase2_tables as pt
from . import preview
from .figures import phase2 as pf
from .figures import pipeline as pl


def check_pipeline(m: Manifest) -> None:
    """Stop if Figure 1 draws a Phase 2 branch that has no results behind it.

    A branch has results when some base model's OT30K run from that starting
    checkpoint has a suite-average change in the manifest.
    """
    for _, start, role, _ in pl.PHASE2:
        stage = pt.corpus_stage("OT3", start)
        for bm in pt.MODELS:
            if not m.labelled(bm, stage):
                continue
            try:
                m.change(m.checkpoint(bm, stage))
            except ManifestError:
                continue
            break
        else:
            raise ManifestError(f"Figure 1 draws the '{role}' branch (OT30K from {start}), "
                                f"but no base model has that run with a change in the manifest.")


def render_all(manifest_path: Path, phase2_dir: Path, tables_dir: Path, figures_dir: Path,
               art_dir: Path | None = None) -> list[Path]:
    m = Manifest.load(manifest_path)
    check_pipeline(m)
    _, meta = p2.load(phase2_dir)
    endpoints = pt.resolve(m, pt.endpoint_blocks())
    ablation = pt.resolve(m, pt.ot3_ablation_blocks())
    grid = pt.resolve(m, pt.mv_grid_blocks())
    cells = pt.heldout_changes(m, meta)

    sft_ot3 = {r.row.base_model: r for _, rows in endpoints for r in rows
               if r.row.stage == "OT3+SFT"}
    # the change from unrounded scores, as Table 2 and the interval table print it
    bars = [(pt.MODELS[bm][0], pt.printed(m.change(r.key)["change"])) for bm, r in sft_ot3.items()]
    heldout = {pt.MODELS[bm][0]: [(pt.printed(c.change.estimate), c.change.lo, c.change.hi)
                                  for c in cells if c.base_model == bm]
               for bm in pt.HELDOUT_BASES}

    written = [
        pt.phase2_endpoints_full(m, endpoints, tables_dir),
        pt.ot3_pretrain_ablation(m, ablation, tables_dir),
        pt.heldout_reasoning(cells, tables_dir),
        pt.heldout_mv_grid(m, grid, tables_dir),
        pt.eval_settings(m, tables_dir),
        pt.write_macros(pt.macros([endpoints, ablation, grid], cells), tables_dir),
        pf.receptivity(bars, heldout, [c for c, _ in pt.CORPORA], figures_dir),
        pl.pipeline(figures_dir),
    ]
    if art_dir is not None:
        art_dir.mkdir(parents=True, exist_ok=True)
        path = art_dir / "phase2_vs_submission.csv"
        pt.compare_with_submission(m, cells).to_csv(path, index=False)
        written.append(path)
    return written


def main(manifest_path: Path, phase2_dir: Path, tables_dir: Path, figures_dir: Path,
         art_dir: Path) -> int:
    try:
        written = render_all(manifest_path, phase2_dir, tables_dir, figures_dir, art_dir)
    except ManifestError as exc:
        print(f"error: {exc}")
        return 1
    tables = [p for p in written if p.suffix == ".tex" and p.name != "phase2_numbers.tex"]
    figures = [p for p in written if p.suffix == ".pdf"]
    pdf = preview.build("Phase 2 tables and figures (render-phase2)", tables, figures,
                        tables_dir.parent / "preview.pdf")
    for path in written + ([pdf] if pdf else []):
        print(f"  wrote {path}")
    return 0
