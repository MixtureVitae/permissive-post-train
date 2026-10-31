"""The workshop paper's five generated figures.

Four read Protocol A summaries from `stats/factorial.py`; `phase2_receptivity`
plots the transcribed Protocol B values in `data/protocol_b.py`. The two
protocols never share an axis. Styling follows the paper's generator, scoped
to each call with `rc_context` so it does not leak into other figures.
"""
from __future__ import annotations

from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
from matplotlib.patches import Patch  # noqa: E402

from ...data import protocol_b as pb  # noqa: E402
from ...data.protocol_a import (  # noqa: E402
    DOMAINS,
    FACTORIAL_KEYS,
    PERMISSIVE,
    POOL_LABEL,
    POOLS,
)
from ...stats.factorial import domain_scores  # noqa: E402

# MixtureVitae, the permissive substrate, is warm; the others are cool.
MV_LIGHT, MV_DARK = "#f0b27a", "#c0552b"
REF_LIGHT, REF_DARK = "#c7d3e0", "#3d5a80"
GREY = "#5a5a5a"
# Comma0.1, the second permissive substrate of the revision, is purple.
COMMA_LIGHT, COMMA_DARK = "#d9c8e8", "#6d4c9f"
POOL_COLORS = {"MV": (MV_LIGHT, MV_DARK), "Nemotron": (REF_LIGHT, REF_DARK),
               "FineWeb-Edu": (REF_LIGHT, REF_DARK), "Comma": (COMMA_LIGHT, COMMA_DARK)}
LINE_COLORS = {"MV": MV_DARK, "Nemotron": "#2a9d8f", "FineWeb-Edu": "#5c6bc0",
               "Comma": COMMA_DARK}
POOL_TICK = {"MV": "MixtureVitae\n(permissive)", "Nemotron": "Nemotron-\nCC-HQ",
             "FineWeb-Edu": "FineWeb-\nEdu", "Comma": "Comma0.1\n(permissive)"}
STAGE_LABEL = {"base": "Base", "sft": "SFT", "dpo": "DPO"}
STAGES = ("base", "sft", "dpo")

STYLE = {
    "font.family": "sans-serif", "font.size": 8, "axes.titlesize": 8.5,
    "axes.labelsize": 8, "xtick.labelsize": 7.5, "ytick.labelsize": 7.5,
    "legend.fontsize": 7.5, "axes.spines.top": False, "axes.spines.right": False,
    "axes.grid": True, "axes.axisbelow": True, "grid.color": "#e4e4e4",
    "grid.linewidth": 0.6, "figure.dpi": 200, "savefig.bbox": "tight",
    "savefig.pad_inches": 0.02, "pdf.fonttype": 42,
}

FIGURE_FILES = ("factorial_main.pdf", "frontloading_trajectory.pdf", "phase1_zoo.pdf",
                "phase2_receptivity.pdf", "frontloading_domains.pdf")


def _save(fig: plt.Figure, out_dir: Path, name: str) -> Path:
    out_dir.mkdir(parents=True, exist_ok=True)
    path = out_dir / name
    fig.savefig(path, metadata={"CreationDate": None})     # same file for the same figure
    plt.close(fig)
    return path


def _row(frame: pd.DataFrame, pool: str, fl: bool, ctx: int, stage: str) -> pd.Series:
    sel = frame[(frame["model_group"] == pool) & (frame["front_loading"] == fl)
                & (frame["context_length"] == ctx) & (frame["stage"] == stage)]
    return sel.iloc[0]


def _paired_panel(ax, suite: pd.DataFrame, ctx: int, stage: str, *,
                  ylabel: str | None = None, title: str | None = None,
                  pools: tuple[str, ...] = POOLS, printed_deltas: bool = False) -> None:
    """Without / with front-loading bars for each substrate, with value labels
    and the front-loading delta above each pair. A substrate missing from
    `suite` is marked as pending. With `printed_deltas`, each delta is the
    difference of the two printed values, as Table 5's Delta row is. Without it,
    the delta subtracts the unrounded means, as the workshop paper's figure does."""
    width, offset = 0.32, 0.19
    xs = np.arange(len(pools))
    tops: list[float] = []
    for i, pool in enumerate(pools):
        if not (suite["model_group"] == pool).any():
            ax.text(xs[i], 1.0, "pending", ha="center", va="bottom", fontsize=6.5,
                    style="italic", color=GREY)
            continue
        light, dark = POOL_COLORS[pool]
        no, yes = _row(suite, pool, False, ctx, stage), _row(suite, pool, True, ctx, stage)
        v0, v1, e0, e1 = float(no["mean"]), float(yes["mean"]), float(no["sd"]), float(yes["sd"])
        err = dict(ecolor=GREY, lw=0.8, capsize=1.8)
        ax.bar(xs[i] - offset, v0, width, color=light, edgecolor="white", yerr=e0, error_kw=err)
        ax.bar(xs[i] + offset, v1, width, color=dark, edgecolor="white", yerr=e1, error_kw=err)
        ax.text(xs[i] - offset, v0 + e0, f"{v0:.1f}", ha="center", va="bottom",
                fontsize=6.5, color=GREY)
        ax.text(xs[i] + offset, v1 + e1, f"{v1:.1f}", ha="center", va="bottom",
                fontsize=6.5, fontweight="bold", color=dark)
        tops.append(max(v0 + e0, v1 + e1))
        # The paper's generator places each delta at the running maximum of the
        # pairs drawn so far, so later labels never sit below earlier ones.
        delta = round(v1, 1) - round(v0, 1) if printed_deltas else v1 - v0
        ax.text(xs[i], max(tops) * 1.16, f"$\\Delta$ {delta:+.1f}", ha="center",
                va="bottom", fontsize=7.5, style="italic",
                color=MV_DARK if pool == "MV" else GREY)
    ax.set_xticks(xs)
    ax.set_xticklabels([POOL_TICK[p] for p in pools], fontsize=7 if len(pools) == 3 else 6)
    ax.set_ylim(0, max(tops) * 1.35)
    if ylabel:
        ax.set_ylabel(ylabel)
    if title:
        ax.set_title(title, style="italic", color=GREY)


def factorial_main(suite: pd.DataFrame, language_avg: pd.DataFrame, out_dir: Path, *,
                   pools: tuple[str, ...] = POOLS, printed_deltas: bool = False,
                   pool_word: str = "substrate") -> Path:
    """Headline factorial: reasoning at 4K and 16K after SFT, language at the 16K base.
    `printed_deltas` as in _paired_panel. `pool_word` names the web-data factor in
    the legend: "substrate" in the workshop paper, "source pool" in the revision."""
    lang = language_avg.rename(columns={"language": "mean"}).assign(stage="base", sd=0.0)
    with plt.rc_context(STYLE):
        fig, axes = plt.subplots(1, 3, figsize=(7.2 if len(pools) == 3 else 8.8, 2.25))
        panels = ((axes[0], suite, 4096, "sft"), (axes[1], suite, 16384, "sft"),
                  (axes[2], lang, 16384, "base"))
        _paired_panel(axes[0], suite, 4096, "sft", ylabel="Reasoning suite avg.",
                      title="Reasoning, 4K, after Tulu3 SFT", pools=pools,
                      printed_deltas=printed_deltas)
        _paired_panel(axes[1], suite, 16384, "sft", title="Reasoning, 16K, after Tulu3 SFT",
                      pools=pools, printed_deltas=printed_deltas)
        _paired_panel(axes[2], lang, 16384, "base", ylabel="Language understanding avg.",
                      title="Language understanding, 16K base", pools=pools,
                      printed_deltas=printed_deltas)
        # The permissive substrate's front-loaded level, so the substrate
        # contrast at matched front-loading reads straight off the axis.
        for ax, frame, ctx, stage in panels:
            ax.axhline(float(_row(frame, "MV", True, ctx, stage)["mean"]), color=MV_DARK,
                       lw=0.8, ls=(0, (4, 3)), alpha=0.75, zorder=0)
        handles = [Patch(facecolor=REF_LIGHT, label="no reasoning front-loading"),
                   Patch(facecolor=REF_DARK, label="reasoning front-loaded"),
                   Patch(facecolor=MV_DARK, label=f"permissive {pool_word} (highlighted)")]
        if "Comma" in pools:
            handles.append(Patch(facecolor=COMMA_DARK,
                                 label=f"Comma0.1, a second permissive {pool_word}"))
        fig.legend(handles=handles, loc="upper center", ncol=len(handles), frameon=False,
                   bbox_to_anchor=(0.5, 1.10))
        fig.tight_layout()
        return _save(fig, out_dir, "factorial_main.pdf")


def frontloading_trajectory(contrasts: pd.DataFrame, out_dir: Path, *,
                            pools: tuple[str, ...] = POOLS,
                            permissive: tuple[str, ...] = PERMISSIVE) -> Path:
    """Front-loading effect with its bootstrap interval across stages."""
    fl = contrasts[contrasts["family"] == "front-loading"]
    xs = np.arange(len(STAGES))
    with plt.rc_context(STYLE):
        fig, axes = plt.subplots(1, 2, figsize=(5.4, 2.2), sharey=True)
        for ax, ctx, title in ((axes[0], 4096, "4K context"), (axes[1], 16384, "16K context")):
            for pool in pools:
                sub = fl[(fl["label"] == pool) & (fl["context_length"] == ctx)].set_index("stage")
                ax.plot(xs, [sub.loc[s, "estimate"] for s in STAGES], "-o", ms=3.5, lw=1.6,
                        color=LINE_COLORS[pool],
                        label=POOL_LABEL[pool] + (" (permissive)" if pool in permissive else ""))
                ax.fill_between(xs, [sub.loc[s, "lo"] for s in STAGES],
                                [sub.loc[s, "hi"] for s in STAGES],
                                color=LINE_COLORS[pool], alpha=0.14, lw=0)
            ax.set_xticks(xs)
            ax.set_xticklabels([STAGE_LABEL[s] for s in STAGES])
            ax.set_title(title, style="italic", color=GREY)
            ax.set_ylim(0, None)
        axes[0].set_ylabel("Front-loading effect (pp)")
        axes[0].legend(frameon=False, loc="lower left", fontsize=7)
        fig.tight_layout()
        return _save(fig, out_dir, "frontloading_trajectory.pdf")


ZOO_PANELS = (
    ("Matched compute, 300B tokens",
     (("MV", 16384, MV_DARK), ("Comma0.1", 16384, "#7f8c8d"),
      ("Nemotron", 16384, "#2a9d8f"), ("FineWeb-Edu", 16384, "#5c6bc0"))),
    ("Open-weights context, 11–36T tokens",
     (("MV", 16384, MV_DARK), ("SmolLM2", 16384, "#6ab04c"),
      ("Qwen2.5", 4096, "#e1b12c"), ("Qwen3", 4096, "#8e6a3f"))),
)
ZOO_NAME = {"MV": "MixtureVitae 16K", "Comma0.1": "Comma0.1 16K",
            "Nemotron": "Nemotron-CC-HQ 16K", "FineWeb-Edu": "FineWeb-Edu 16K",
            "SmolLM2": "SmolLM2 16K", "Qwen2.5": "Qwen2.5", "Qwen3": "Qwen3"}


def phase1_zoo(zoo_suite: pd.DataFrame, out_dir: Path) -> Path:
    """Phase 1 suite-average trajectories for all models."""
    xs = np.arange(len(STAGES))
    with plt.rc_context(STYLE):
        fig, axes = plt.subplots(1, 2, figsize=(6.6, 2.1), sharey=True)
        for ax, (title, spec) in zip(axes, ZOO_PANELS):
            for group, ctx, color in spec:
                sub = zoo_suite[(zoo_suite["model_group"] == group)
                                & (zoo_suite["context_length"] == ctx)].set_index("stage")
                ax.errorbar(xs, [sub.loc[s, "mean"] for s in STAGES],
                            yerr=[sub.loc[s, "sd"] for s in STAGES], fmt="-o", ms=3.5,
                            lw=1.6, capsize=2, color=color, label=ZOO_NAME[group],
                            zorder=3 if group == "MV" else 2)
            ax.set_xticks(xs)
            ax.set_xticklabels([STAGE_LABEL[s] for s in STAGES])
            ax.set_title(title, style="italic", color=GREY)
            ax.legend(fontsize=7, loc="best", frameon=True, framealpha=0.9, edgecolor="none",
                      facecolor="white", borderpad=0.3, labelspacing=0.35, handlelength=1.6)
        axes[0].set_ylabel("Reasoning suite avg.")
        axes[0].margins(y=0.14)
        fig.tight_layout()
        return _save(fig, out_dir, "phase1_zoo.pdf")


def phase2_receptivity(out_dir: Path) -> Path:
    """Protocol B, single seed: suite-average change under OT30K per model, and
    the MATH500 change under four reasoning corpora."""
    with plt.rc_context(STYLE):
        fig, axes = plt.subplots(1, 2, figsize=(7.2, 2.45),
                                 gridspec_kw={"width_ratios": [0.95, 1.3]})
        ax = axes[0]
        ordered = sorted(pb.PHASE2_PAIRED, key=lambda r: r[3] - r[2])
        names = [r[0] for r in ordered]
        deltas = [end - start for _, _, start, end in ordered]
        ys = np.arange(len(names))
        ax.barh(ys, deltas, height=0.66,
                color=[MV_DARK if n == "MixtureVitae" else REF_DARK for n in names])
        ax.axvline(0, color="black", lw=0.8)
        ax.set_yticks(ys)
        ax.set_yticklabels(names, fontsize=7.5)
        for y, d in zip(ys, deltas):
            ax.text(d + (0.5 if d >= 0 else -0.5), y, f"{d:+.1f}", va="center",
                    ha="left" if d >= 0 else "right", fontsize=7)
        ax.set_xlim(min(deltas) * 1.25, max(deltas) * 3.2)
        ax.set_xlabel(r"$\Delta$ reasoning suite avg. under OT3-30K")
        ax.set_title("Each model from its own starting checkpoint", style="italic", color=GREY)
        ax.grid(axis="y", visible=False)

        ax = axes[1]
        bases = ["MixtureVitae", "SmolLM2", "Comma0.1", "FineWeb-Edu", "Nemotron-CC-HQ"]
        xs, width = np.arange(len(bases)), 0.19
        colors = [MV_DARK, "#5c6bc0", "#2a9d8f", "#b0a08c"]
        for j, corpus in enumerate(pb.HELDOUT_CORPORA):
            cells = [pb.HELDOUT_MATH500[b][j] for b in bases]
            est = [c[0] for c in cells]
            err = [[c[0] - c[1] for c in cells], [c[2] - c[0] for c in cells]]
            ax.bar(xs + (j - 1.5) * width, est, width, color=colors[j], label=corpus,
                   yerr=err, error_kw=dict(ecolor=GREY, lw=0.7, capsize=1.4))
        ax.axhline(0, color="black", lw=0.8)
        ax.set_xticks(xs)
        ax.set_xticklabels(["MixtureVitae", "SmolLM2", "Comma0.1", "FineWeb-\nEdu",
                            "Nemotron-\nCC-HQ"], fontsize=7)
        ax.set_ylabel(r"$\Delta$ MATH500 (pp)")
        ax.set_title("Four reasoning corpora, from the Tulu3 SFT checkpoint",
                     style="italic", color=GREY)
        ax.set_ylim(min(-6, ax.get_ylim()[0]), 52)
        ax.legend(frameon=False, fontsize=6.8, ncol=2, loc="upper right",
                  bbox_to_anchor=(1.02, 1.03), handlelength=1.2, columnspacing=1.0)
        fig.tight_layout()
        return _save(fig, out_dir, "phase2_receptivity.pdf")


def frontloading_domains(scores: pd.DataFrame, out_dir: Path, *,
                         pools: tuple[str, ...] = POOLS) -> Path:
    """Front-loading effect after Tulu3 SFT by capability domain."""
    dom = domain_scores(scores, FACTORIAL_KEYS)
    xs, width = np.arange(len(DOMAINS)), (0.26 if len(pools) == 3 else 0.2)
    with plt.rc_context(STYLE):
        fig, axes = plt.subplots(1, 2, figsize=(6.4 if len(pools) == 3 else 7.6, 2.4),
                                 sharey=True)
        for ax, ctx, title in ((axes[0], 4096, "4K context"), (axes[1], 16384, "16K context")):
            for i, pool in enumerate(pools):
                sel = dom[(dom["model_group"] == pool) & (dom["context_length"] == ctx)
                          & (dom["stage"] == "sft")].set_index(["domain", "front_loading"])["score"]
                eff = [float(sel[(d, True)] - sel[(d, False)]) for d in DOMAINS]
                pos = xs + (i - (len(pools) - 1) / 2) * width
                ax.bar(pos, eff, width, color=LINE_COLORS[pool])
                for x, v in zip(pos, eff):
                    ax.text(x, v + (0.5 if v >= 0 else -0.5), f"{v:+.0f}", ha="center",
                            va="bottom" if v >= 0 else "top",
                            fontsize=6.2 if len(pools) == 3 else 5.2, color=GREY)
            ax.axhline(0, color="black", lw=0.8)
            ax.set_xticks(xs)
            ax.set_xticklabels([d.capitalize() for d in DOMAINS], fontsize=7)
            ax.set_title(title, style="italic", color=GREY)
            ax.set_ylim(-6, 38)
        axes[0].set_ylabel("Front-loading effect (pp)")
        fig.legend(handles=[Patch(facecolor=LINE_COLORS[p], label=POOL_LABEL[p]) for p in pools],
                   loc="upper center", ncol=len(pools), frameon=False, bbox_to_anchor=(0.5, 1.09))
        fig.tight_layout()
        return _save(fig, out_dir, "frontloading_domains.pdf")
