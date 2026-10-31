"""Figure 1 of the revision: the post-training pipeline as a branching experiment.

Boxes are training stages and dots are checkpoints, so every run visibly
starts from a checkpoint. The checkpoints form a grid: a 4K row and a 16K row,
each with a base, an SFT and a DPO checkpoint. These six are what Phase 1
reports, for all models and the factorial alike. Phase 2 is one wide box
below the 16K row, because it only starts from 16K checkpoints. The main
comparison starts from the SFT checkpoint (solid), and two controls start from
the base and DPO checkpoints (dashed). The 4K row ends at Phase 1.

The figure is drawn at its printed size (the NeurIPS text width), with axes
limits in inches, so `\\includegraphics[width=\\linewidth]` prints it at scale
1.0. `render/phase2.py` checks that every drawn Phase 2 branch has results in
the manifest before drawing it.
"""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
from matplotlib.patches import FancyBboxPatch, PathPatch, Polygon  # noqa: E402
from matplotlib.path import Path as MPath  # noqa: E402

from .workshop import STYLE  # noqa: E402

WIDTH, HEIGHT = 5.5, 2.34
FILE = "figure_pipeline.pdf"

# The current TikZ Figure 1's palette: shared steps grey, Phase 1 blue,
# Phase 2 orange (OpenThoughts3's colour in Figure 2).
FAMILY = {
    "shared": ("#ECECEC", "#6E6E6E"),
    "phase1": ("#DCE5EE", "#3D5A80"),
    "phase2": ("#F7DFCE", "#C0552B"),
}
INK, MUTED = "#2b2b2b", "#6E6E6E"
BAND_ALPHA = 0.32

ROW = {"4K": 1.83, "16K": 1.23}
BOX_H = 0.42
DOT_R = 0.038
#: Column centres of the base, SFT and DPO checkpoints.
COL = {"base": 1.62, "sft": 3.345, "dpo": 5.07}
#: Line width (pt) and dash pattern of each edge kind.
EDGE_STYLE = {
    "trunk": (1.3, "solid"),
    "main": (2.0, "solid"),
    "control": (1.1, (0, (3.2, 1.8))),
}
# No tight bbox: the PDF must stay exactly WIDTH inches wide.
PRINT_STYLE = {**STYLE, "axes.grid": False, "savefig.bbox": None, "savefig.pad_inches": 0}
TITLE_PT, SUB_PT, HEADER_PT, LABEL_PT = 7.2, 6.3, 7.0, 6.3

#: Phase 2 branches: (starting checkpoint dot, its record stage, role, edge kind).
#: The record stage of the 16K extended checkpoint is "base".
PHASE2 = (
    ("base_16K", "base", "without Tulu3", "control"),
    ("sft_16K", "SFT-Tulu3", "main", "main"),
    ("dpo_16K", "DPO-Tulu3", "after DPO", "control"),
)


@dataclass(frozen=True)
class Stage:
    id: str
    x0: float
    w: float
    y0: float
    h: float
    title: str
    sub: str
    family: str

    @property
    def x1(self) -> float:
        return self.x0 + self.w

    @property
    def y1(self) -> float:
        return self.y0 + self.h

    @property
    def xc(self) -> float:
        return self.x0 + self.w / 2

    @property
    def yc(self) -> float:
        return self.y0 + self.h / 2


@dataclass(frozen=True)
class Dot:
    id: str
    x: float
    y: float
    family: str
    filled: bool = False


@dataclass(frozen=True)
class Edge:
    src: str
    dst: str
    kind: str
    family: str
    points: tuple[tuple[float, float], ...]
    head: bool = True


def layout() -> tuple[dict[str, Stage], dict[str, Dot], list[Edge]]:
    """Every box, checkpoint and edge, in inches from the lower left corner."""
    r, gap = DOT_R, 0.22  # gap between a dot and the next box
    y4, y16 = ROW["4K"], ROW["16K"]
    left_w = 1.18
    pre = Stage("pretrain", COL["base"] - gap - left_w, left_w, y4 - BOX_H / 2, BOX_H,
                "Pre-training", "300B tokens, 4K", "shared")
    ext = Stage("extension", pre.x0, left_w, y16 - BOX_H / 2, BOX_H,
                "Long context", "YaRN + long-context SFT", "shared")
    tall = (y16 - BOX_H / 2, y4 - y16 + BOX_H)
    sft = Stage("sft", COL["base"] + gap, COL["sft"] - COL["base"] - 2 * gap, *tall,
                "Tulu3 SFT", "Instructions", "phase1")
    dpo = Stage("dpo", COL["sft"] + gap, COL["dpo"] - COL["sft"] - 2 * gap, *tall,
                "Tulu3 DPO", "Preferences", "phase1")
    reason = Stage("reasoning", COL["base"] - 0.25, COL["dpo"] - COL["base"] + 0.50, 0.42,
                   0.36, "Reasoning SFT", "OpenThoughts3, 30K samples, 5 epochs", "phase2")
    stages = {s.id: s for s in (pre, ext, sft, dpo, reason)}

    dots = {}
    for col, family in (("base", "shared"), ("sft", "phase1"), ("dpo", "phase1")):
        for row, y in ROW.items():
            dots[f"{col}_{row}"] = Dot(f"{col}_{row}", COL[col], y, family)
    for i, (start, *_) in enumerate(PHASE2):
        dots[f"end_{i}"] = Dot(f"end_{i}", COL[start.split("_")[0]], reason.y0 - 0.12,
                               "phase2", filled=True)

    # A stage's output runs into its checkpoint as a plain line. Arrowheads
    # only point into stages.
    def edge(src, dst, kind, family, *points, head=True):
        return Edge(src, dst, kind, family, tuple(points), head)

    edges = [
        edge("pretrain", "extension", "trunk", "shared", (pre.xc, pre.y0), (ext.xc, ext.y1)),
        edge("pretrain", "base_4K", "trunk", "shared", (pre.x1, y4), (COL["base"] - r, y4),
             head=False),
        edge("extension", "base_16K", "trunk", "shared", (ext.x1, y16), (COL["base"] - r, y16),
             head=False),
    ]
    # Phase 1: the same two stages at both context lengths.
    for src, stage, dst in (("base", sft, "sft"), ("sft", dpo, "dpo")):
        for row, y in ROW.items():
            edges.append(edge(f"{src}_{row}", stage.id, "trunk", "phase1",
                              (COL[src] + r, y), (stage.x0, y)))
            edges.append(edge(stage.id, f"{dst}_{row}", "trunk", "phase1",
                              (stage.x1, y), (COL[dst] - r, y), head=False))
    # Phase 2: three 16K starting checkpoints, one update.
    for i, (start, _, _, kind) in enumerate(PHASE2):
        x = dots[start].x
        edges.append(edge(start, "reasoning", kind, "phase2", (x, y16 - r), (x, reason.y1)))
        edges.append(edge("reasoning", f"end_{i}", kind, "phase2", (x, reason.y0),
                          (x, dots[f"end_{i}"].y + r), head=False))
    return stages, dots, edges


def _rounded(points: tuple[tuple[float, float], ...], radius: float) -> MPath:
    """An orthogonal polyline with its corners rounded by quadratic curves."""
    verts, codes = [points[0]], [MPath.MOVETO]
    for prev, corner, nxt in zip(points, points[1:], points[2:]):
        din = _unit(prev, corner)
        dout = _unit(corner, nxt)
        rr = min(radius, _len(prev, corner) / 2, _len(corner, nxt) / 2)
        verts += [(corner[0] - din[0] * rr, corner[1] - din[1] * rr), corner,
                  (corner[0] + dout[0] * rr, corner[1] + dout[1] * rr)]
        codes += [MPath.LINETO, MPath.CURVE3, MPath.CURVE3]
    verts.append(points[-1])
    codes.append(MPath.LINETO)
    return MPath(verts, codes)


def _len(a, b) -> float:
    return ((b[0] - a[0]) ** 2 + (b[1] - a[1]) ** 2) ** 0.5


def _unit(a, b) -> tuple[float, float]:
    n = _len(a, b)
    return ((b[0] - a[0]) / n, (b[1] - a[1]) / n)


def _draw_edge(ax, e: Edge) -> None:
    lw, ls = EDGE_STYLE[e.kind]
    color = FAMILY[e.family][1]
    if not e.head:
        ax.add_patch(PathPatch(_rounded(e.points, 0.07), fill=False, lw=lw, ls="solid",
                               ec=color, capstyle="butt", zorder=2))
        return
    head = 0.028 + 0.012 * lw  # inches, grows a little with the line
    *body, tip = e.points
    d = _unit(body[-1], tip)
    base = (tip[0] - d[0] * head, tip[1] - d[1] * head)
    ax.add_patch(PathPatch(_rounded((*body, base), 0.07), fill=False, lw=lw, ls=ls,
                           ec=color, capstyle="butt", joinstyle="round", zorder=2))
    # The head is drawn solid, so a dashed line still ends in a clean arrow.
    half = head * 0.55
    ax.add_patch(Polygon([tip, (base[0] - d[1] * half, base[1] + d[0] * half),
                          (base[0] + d[1] * half, base[1] - d[0] * half)],
                         closed=True, fc=color, ec="none", zorder=3))


def _draw_stage(ax, s: Stage) -> None:
    light, dark = FAMILY[s.family]
    if s.id == "pretrain":  # a card stack: the same pipeline for every base
        for k in (2, 1):
            ax.add_patch(FancyBboxPatch((s.x0 - 0.045 * k, s.y0 + 0.045 * k), s.w, s.h,
                                        boxstyle="round,pad=0,rounding_size=0.05",
                                        fc="white", ec=dark, lw=0.6, zorder=3))
    ax.add_patch(FancyBboxPatch((s.x0, s.y0), s.w, s.h, gid=s.id,
                                boxstyle="round,pad=0,rounding_size=0.05",
                                fc=light, ec=dark, lw=0.9, zorder=4))
    n_title, n_sub = s.title.count("\n") + 1, s.sub.count("\n") + 1
    line = 1.22 / 72  # inches per point of line height
    block = n_title * TITLE_PT * line + 0.03 + n_sub * SUB_PT * line
    top = s.yc + block / 2
    ax.text(s.xc, top, s.title, ha="center", va="top", fontsize=TITLE_PT,
            fontweight="bold", color=INK, linespacing=1.1, zorder=5, gid=s.id)
    ax.text(s.xc, top - n_title * TITLE_PT * line - 0.03, s.sub, ha="center",
            va="top", fontsize=SUB_PT, color=MUTED, linespacing=1.15, zorder=5, gid=s.id)


def _draw_dot(ax, d: Dot) -> None:
    dark = FAMILY[d.family][1]
    ax.add_patch(plt.Circle((d.x, d.y), DOT_R, fc=dark if d.filled else "white", ec=dark,
                            lw=1.0, zorder=6, gid=d.id))


def draw() -> plt.Figure:
    stages, dots, edges = layout()
    with plt.rc_context(PRINT_STYLE):
        fig = plt.figure(figsize=(WIDTH, HEIGHT))
        ax = fig.add_axes((0, 0, 1, 1))
        ax.set_xlim(0, WIDTH)
        ax.set_ylim(0, HEIGHT)
        ax.set_aspect("equal")
        ax.axis("off")

        # Phase bands. Phase 1 holds the Tulu3 stages and the four checkpoints
        # they produce. The base checkpoints sit outside it. Each band has its
        # header on its outer edge.
        sft, reason = stages["sft"], stages["reasoning"]
        x1 = COL["dpo"] + 0.30
        y1 = reason.y0 - 0.12 - DOT_R - 0.05  # below the Phase 2 endpoints
        bands = (
            (sft.x0 - 0.09, sft.y0 - 0.09, sft.y1 + 0.09, "phase1",
             "Phase 1: Instruction alignment", "bottom"),
            (reason.x0 - 0.09, y1, reason.y1 + 0.09, "phase2",
             "Phase 2: Reasoning post-training", "top"),
        )
        for bx0, by0, by1, fam, header, va in bands:
            ax.add_patch(FancyBboxPatch((bx0, by0), x1 - bx0, by1 - by0,
                                        boxstyle="round,pad=0,rounding_size=0.06",
                                        fc=FAMILY[fam][0], ec="none", alpha=BAND_ALPHA,
                                        zorder=0))
            y = by1 + 0.04 if va == "bottom" else by0 - 0.04
            ax.text(bx0 + 0.03, y, header, ha="left", va=va, fontsize=HEADER_PT,
                    style="italic", color=FAMILY[fam][1], zorder=5)

        for e in edges:
            _draw_edge(ax, e)
        for s in stages.values():
            _draw_stage(ax, s)
        for d in dots.values():
            _draw_dot(ax, d)

        # Row labels: the context length of each row of checkpoints.
        for row, y in ROW.items():
            ax.text(COL["dpo"] + 0.075, y, row, ha="left", va="center", fontsize=LABEL_PT,
                    color=MUTED, zorder=5)
    return fig


def pipeline(out_dir: Path) -> Path:
    """Write Figure 1 as PDF, with a PNG preview next to it."""
    fig = draw()
    out_dir.mkdir(parents=True, exist_ok=True)
    path = out_dir / FILE
    # No creation date, so the committed files only change when the figure does.
    with plt.rc_context(PRINT_STYLE):
        fig.savefig(path.with_suffix(".png"), dpi=300, metadata={"Software": None})
        fig.savefig(path, metadata={"CreationDate": None})
    plt.close(fig)
    return path
