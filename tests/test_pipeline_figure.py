"""Tests for Figure 1, the post-training pipeline (mvpt render-phase2).

They pin what the reviewer asked for: Phase 2 starts from the Tulu3 SFT
checkpoint, DPO is a separate branch, and the two controls start from the 16K
base and DPO checkpoints. Phase 1 runs at 4K and 16K, and only 16K checkpoints
enter Phase 2. They also check that every label fits its box at print size,
that the PDF is exactly the NeurIPS text width, and that every drawn branch
has results in the manifest.
"""
from __future__ import annotations

from pathlib import Path

import pytest

pytest.importorskip("matplotlib")

from mv_post_train.manifest.reader import Manifest, ManifestError  # noqa: E402
from mv_post_train.render import phase2_tables as pt  # noqa: E402
from mv_post_train.render.figures import pipeline as pl  # noqa: E402
from mv_post_train.render.phase2 import check_pipeline  # noqa: E402

P2 = "phase2_three_seed"
ROOT = Path(__file__).resolve().parents[1]


def _manifest(runs: dict[str, list[str]]) -> Manifest:
    """For each base model, its parent checkpoints and an OT3 run from each listed start."""
    checkpoints, changes = {}, []
    for bm, starts in runs.items():
        for start in ("base", "SFT-Tulu3", "DPO-Tulu3"):
            checkpoints[f"{bm}/{start}"] = {
                "parent": None, "parent_status": "none recorded",
                "labels": {P2: {"base_model": bm, "stage": start}}}
        for start in starts:
            key = f"{bm}/{pt.corpus_stage('OT3', start)}"
            checkpoints[key] = {"parent": f"{bm}/{start}", "parent_status": "verified",
                                "labels": {P2: {"base_model": bm,
                                                "stage": pt.corpus_stage("OT3", start)}}}
            changes.append({"checkpoint": key, "protocol": P2, "parent": f"{bm}/{start}",
                            "change": 1.0})
    return Manifest({"protocols": {P2: {"tasks": {}}}, "checkpoints": checkpoints,
                     "results": [], "suite_averages": [], "changes": changes,
                     "rejected_changes": []})


def test_phase2_branches_from_three_16k_checkpoints_and_dpo_from_sft():
    stages, dots, edges = pl.layout()
    into_phase2 = {e.src: e.kind for e in edges if e.dst == "reasoning"}
    assert into_phase2 == {"base_16K": "control", "sft_16K": "main", "dpo_16K": "control"}
    # Phase 1 runs at both context lengths: SFT from the base, DPO from SFT.
    assert sorted(e.src for e in edges if e.dst == "sft") == ["base_16K", "base_4K"]
    assert sorted(e.src for e in edges if e.dst == "dpo") == ["sft_16K", "sft_4K"]
    assert [e.src for e in edges if e.dst == "extension"] == ["pretrain"]
    # The 4K checkpoints end at Phase 1.
    assert not [e for e in edges if e.src.endswith("_4K") and e.dst == "reasoning"]
    # Every edge joins things the layout defines.
    for e in edges:
        assert e.src in stages or e.src in dots
        assert e.dst in stages or e.dst in dots


def test_every_label_fits_its_box_and_the_figure():
    fig = pl.draw()
    fig.canvas.draw()
    renderer, ax = fig.canvas.get_renderer(), fig.axes[0]
    boxes = {p.get_gid(): p.get_window_extent(renderer) for p in ax.patches if p.get_gid()}
    pad = 0.02 * fig.dpi
    for text in ax.texts:
        ext = text.get_window_extent(renderer)
        assert fig.bbox.x0 <= ext.x0 and ext.x1 <= fig.bbox.x1, text.get_text()
        assert fig.bbox.y0 <= ext.y0 and ext.y1 <= fig.bbox.y1, text.get_text()
        if text.get_gid() in boxes:
            box = boxes[text.get_gid()]
            assert box.x0 + pad <= ext.x0 and ext.x1 <= box.x1 - pad, text.get_text()
            assert box.y0 + pad <= ext.y0 and ext.y1 <= box.y1 - pad, text.get_text()


def test_written_at_print_width_with_embedded_truetype(tmp_path):
    pdf = pl.pipeline(tmp_path)
    import matplotlib.image as mpimg
    png = mpimg.imread(pdf.with_suffix(".png"))
    assert png.shape[1] == round(pl.WIDTH * 300)
    assert b"/FontFile2" in pdf.read_bytes()


def test_a_branch_without_results_stops_the_render():
    m = _manifest({pt.MV: ["base", "SFT-Tulu3"], "Qwen3-1.7B-Base": ["SFT-Tulu3"]})
    with pytest.raises(ManifestError, match="after DPO"):
        check_pipeline(m)
    check_pipeline(_manifest({pt.MV: ["base", "SFT-Tulu3", "DPO-Tulu3"],
                              "Qwen3-1.7B-Base": ["SFT-Tulu3"]}))


def test_every_branch_has_results_in_the_committed_manifest():
    check_pipeline(Manifest.load(ROOT / "results" / "manifest.json"))
