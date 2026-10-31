"""Figure 2 of the revision (receptivity), rendered from the results manifest.

Left: the suite-average change under OpenThoughts3-30K, for every model from
its own Tulu3 SFT checkpoint, so each bar is the same treatment. The bars are
the printed changes of the endpoints table. Right: the MATH500 change under
four reasoning corpora, with the intervals of the held-out table. Styling is
that of `figures/workshop.py`.
"""
from __future__ import annotations

from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402

from .workshop import GREY, MV_DARK, REF_DARK, STYLE, _save  # noqa: E402

CORPUS_COLORS = (MV_DARK, "#5c6bc0", "#2a9d8f", "#b0a08c")
TICK = {"FineWeb-Edu": "FineWeb-\nEdu", "Nemotron-CC-HQ": "Nemotron-\nCC-HQ"}


def receptivity(bars: list[tuple[str, float]], heldout: dict[str, list[tuple]],
                corpora: list[str], out_dir: Path) -> Path:
    """`bars`: (model, printed change).
    `heldout`: model -> one (estimate, lo, hi) per corpus."""
    with plt.rc_context(STYLE):
        fig, axes = plt.subplots(1, 2, figsize=(7.2, 2.45),
                                 gridspec_kw={"width_ratios": [0.95, 1.3]})
        ax = axes[0]
        ordered = sorted(bars, key=lambda b: b[1])
        names = [b[0] for b in ordered]
        deltas = [b[1] for b in ordered]
        ys = np.arange(len(names))
        ax.barh(ys, deltas, height=0.66,
                color=[MV_DARK if n == "MixtureVitae" else REF_DARK for n in names])
        ax.axvline(0, color="black", lw=0.8)
        ax.set_yticks(ys)
        ax.set_yticklabels(names, fontsize=7.5)
        span = max(deltas) - min(deltas)
        for y, (_, d) in zip(ys, ordered):
            ax.text(d + (0.02 if d >= 0 else -0.02) * span, y, f"{d:+.1f}", va="center",
                    ha="left" if d >= 0 else "right", fontsize=7)
        ax.set_xlim(min(0, min(deltas)) - 0.3 * span, max(0, max(deltas)) + 0.3 * span)
        ax.set_xlabel(r"$\Delta$ reasoning suite avg. under OT3-30K")
        ax.set_title("Each model from its own Tulu3 SFT checkpoint", style="italic", color=GREY)
        ax.grid(axis="y", visible=False)

        ax = axes[1]
        bases = list(heldout)
        xs, width = np.arange(len(bases)), 0.19
        for j, corpus in enumerate(corpora):
            cells = [heldout[b][j] for b in bases]
            est = [c[0] for c in cells]
            err = [[c[0] - c[1] for c in cells], [c[2] - c[0] for c in cells]]
            ax.bar(xs + (j - 1.5) * width, est, width, color=CORPUS_COLORS[j], label=corpus,
                   yerr=err, error_kw=dict(ecolor=GREY, lw=0.7, capsize=1.4))
        ax.axhline(0, color="black", lw=0.8)
        ax.set_xticks(xs)
        ax.set_xticklabels([TICK.get(b, b) for b in bases], fontsize=7)
        ax.set_ylabel(r"$\Delta$ MATH500 (pp)")
        ax.set_title("Four reasoning corpora, from the Tulu3 SFT checkpoint",
                     style="italic", color=GREY)
        top = max(c[2] for cells in heldout.values() for c in cells)
        ax.set_ylim(min(-6, ax.get_ylim()[0]), top * 1.25)
        ax.legend(frameon=False, fontsize=6.8, ncol=2, loc="upper right",
                  bbox_to_anchor=(1.02, 1.03), handlelength=1.2, columnspacing=1.0)
        fig.tight_layout()
        return _save(fig, out_dir, "phase2_receptivity.pdf")
