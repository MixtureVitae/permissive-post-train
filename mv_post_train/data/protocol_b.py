"""Protocol B values for the workshop paper's OT30K tables, transcribed.

Protocol B is the reasoning post-training protocol: one seed, sampled decoding
on several benchmarks, longer generation budgets, LiveCodeBench v5 and
prompt-level IFEval. Every OpenThoughts3-30K and held-out-corpus experiment
uses it.

**These numbers are not recomputed here.** They are copied from
`code/src/protocol_b.py` of the artifacts repository at
`protocol_a.SOURCE_COMMIT`, which in turn transcribed them from the
manuscript. That repository ships no per-item records for them, so this
module only makes the tables regenerable and keeps the values in one place.
`PHASE2_ENDPOINTS` is the submission's Phase 2 table. Its MV OT30K rows can be
compared against the released LAION records with `mvpt phase2-reconcile`, and
several cells do not match any released record.

Nothing in this module may be compared against a Protocol A number.

These values are what the submission printed, and `render-workshop` keeps
rendering them so the submission stays reproducible. The revision's Phase 2
tables are generated from the results manifest instead (`render-phase2`,
`render/phase2_tables.py`), which also lists every submitted cell next to its
new value.
"""
from __future__ import annotations

#: Column order of every row below; the twelfth value is the suite average.
BENCH_ORDER = ("A24", "A25", "AMC", "M500", "GSM8K", "HE", "LCB", "MBPP", "GPQA", "JEE", "IFE")

#: (group, row label, is a starting checkpoint, 11 scores + Avg)
PHASE2_ENDPOINTS = [
    ("MV-16k", "base", True, [0.0, 10.0, 30.0, 33.6, 58.4, 48.2, 11.7, 41.0, 26.3, 8.3, 42.6, 28.2]),
    ("MV-16k", r"$\to$ SFT $\to$ OT30K", False, [20.0, 16.7, 29.8, 72.8, 52.5, 58.5, 10.0, 29.2, 36.4, 4.6, 25.6, 32.4]),
    ("MV-16k", r"$\to$ DPO $\to$ OT30K", False, [10.0, 6.7, 20.0, 36.4, 51.4, 46.3, 11.9, 40.8, 26.8, 11.5, 43.0, 27.7]),
    ("MV-16k", r"$\to$ OT30K \textbf{(no Phase 1)}", False, [20.0, 16.7, 25.0, 71.4, 58.9, 57.3, 10.0, 31.6, 26.3, 6.0, 26.6, 31.8]),
    ("FineWeb-Edu-16k", r"$\to$ SFT", True, [0.0, 0.0, 2.5, 1.4, 8.5, 2.4, 0.3, 2.2, 30.3, 3.0, 27.6, 7.1]),
    ("FineWeb-Edu-16k", r"\quad $\to$ OT30K", False, [0.0, 0.0, 0.0, 0.8, 9.2, 0.0, 0.0, 0.0, 23.7, 0.0, 17.8, 4.7]),
    ("Nemotron-16k", r"$\to$ SFT", True, [0.0, 0.0, 2.5, 1.4, 6.4, 1.5, 0.2, 0.2, 25.9, 3.3, 36.5, 7.1]),
    ("Nemotron-16k", r"\quad $\to$ OT30K", False, [0.0, 0.0, 0.0, 0.4, 5.3, 0.0, 0.0, 0.2, 21.2, 0.2, 20.0, 4.3]),
    ("Comma0.1-16k", r"$\to$ SFT", True, [0.0, 0.0, 5.0, 2.2, 5.8, 12.8, 0.8, 23.6, 30.3, 5.0, 35.4, 11.0]),
    ("Comma0.1-16k", r"\quad $\to$ OT30K", False, [0.0, 0.0, 0.0, 1.6, 14.0, 1.8, 0.0, 0.0, 28.8, 0.0, 20.0, 6.0]),
    ("SmolLM2-16k", r"$\to$ SFT", True, [0.0, 3.3, 2.5, 6.6, 31.9, 28.7, 3.3, 37.4, 25.8, 6.5, 45.0, 17.4]),
    ("SmolLM2-16k", r"\quad $\to$ OT30K", False, [0.0, 0.0, 0.0, 14.0, 26.3, 23.2, 0.5, 15.8, 27.3, 3.3, 34.0, 13.1]),
    ("Qwen2.5", r"$\to$ SFT", True, [0.0, 0.0, 7.5, 19.6, 39.7, 48.2, 3.8, 44.6, 22.7, 7.5, 44.4, 21.6]),
    ("Qwen2.5", r"\quad $\to$ OT30K", False, [10.0, 6.7, 15.0, 49.2, 56.9, 20.7, 1.4, 8.4, 25.8, 2.1, 27.6, 20.3]),
    ("Qwen3", r"$\to$ SFT", True, [10.0, 0.0, 12.5, 23.0, 69.2, 64.6, 3.8, 55.8, 25.8, 11.7, 53.2, 30.0]),
    ("Qwen3", r"\quad $\to$ OT30K", False, [0.0, 0.0, 0.0, 10.6, 29.0, 7.3, 1.1, 10.6, 28.3, 0.5, 30.4, 10.7]),
]

#: Suite-average change under OT30K from each model's own starting checkpoint,
#: as plotted in the receptivity figure: (label, starting point, start, end).
#: MV's row starts from the post-YaRN base; every other row from Tulu3 SFT.
PHASE2_PAIRED = [
    ("MixtureVitae", "post-YaRN base", 28.2, 31.8),
    ("FineWeb-Edu", "Tulu3 SFT", 7.1, 4.7),
    ("Nemotron-CC-HQ", "Tulu3 SFT", 7.1, 4.3),
    ("Comma0.1", "Tulu3 SFT", 11.0, 6.0),
    ("SmolLM2", "Tulu3 SFT", 17.4, 13.1),
    ("Qwen2.5", "Tulu3 SFT", 21.6, 20.3),
    ("Qwen3", "Tulu3 SFT", 30.0, 10.7),
]

#: OpenThoughts3 removed from the 300B pre-training mixture, then the same
#: OT30K post-training on both bases. The base without OT3 is not shown.
OT3_PRETRAIN_ABLATION = [
    ("Base with OT3, before reasoning post-training",
     [3.3, 8.3, 25.6, 34.6, 54.1, 42.1, 8.4, 29.2, 27.3, 10.2, 45.2, 26.2]),
    (r"OT3 in pre-training, $\to$ OT30K",
     [11.7, 10.0, 47.5, 69.8, 56.4, 57.9, 12.7, 26.6, 28.3, 20.6, 27.6, 33.6]),
    (r"OT3 removed from pre-training, $\to$ OT30K",
     [0.0, 0.0, 17.5, 38.0, 53.3, 38.4, 6.2, 33.8, 28.8, 11.0, 36.4, 23.9]),
]

#: MATH500 change when each base's Tulu3 SFT checkpoint gets 30K samples of one
#: reasoning corpus: (estimate, lo, hi, sig) with a 95% paired bootstrap
#: interval over the 500 problems. `sig` is "+" / "-" for a significant
#: increase / decrease after Benjamini-Hochberg, "" otherwise. Supplied
#: constants: the intervals are not recomputed anywhere.
HELDOUT_CORPORA = ("OpenThoughts3", "AceReason", "Nemotron-Cascade", "Superior-Reasoning")
HELDOUT_MATH500 = {
    "MixtureVitae": [(35.4, 30.4, 40.4, "+"), (34.2, 29.4, 38.8, "+"),
                     (31.6, 27.0, 36.4, "+"), (16.4, 11.2, 21.4, "+")],
    "Comma0.1": [(-0.4, -2.2, 1.4, ""), (0.4, -1.4, 2.2, ""),
                 (0.8, -1.2, 2.8, ""), (0.0, -1.6, 1.6, "")],
    "FineWeb-Edu": [(-0.2, -1.6, 1.2, ""), (1.0, -0.4, 2.6, ""),
                    (0.6, -1.0, 2.2, ""), (-0.8, -2.0, 0.4, "")],
    "Nemotron-CC-HQ": [(-2.2, -3.8, -0.8, "-"), (0.0, -2.0, 2.0, ""),
                       (-0.8, -2.6, 1.0, ""), (-1.0, -2.8, 0.8, "")],
    "SmolLM2": [(-0.6, -3.4, 2.2, ""), (8.4, 5.0, 11.8, "+"),
                (6.6, 3.2, 10.2, "+"), (5.6, 2.2, 9.0, "+")],
}

#: The MV grid: four corpora from three starting checkpoints. None = not
#: evaluated; that row's average is over the remaining ten.
HELDOUT_MV_GRID = {
    "base": [
        ("starting checkpoint", [0.0, 10.0, 30.0, 33.6, 58.6, 48.2, 11.7, 41.0, 26.3, 8.3, 42.6, 28.2]),
        ("OpenThoughts3", [20.0, 16.7, 25.0, 70.2, 59.1, 57.3, 10.0, 31.6, 26.3, 6.0, 26.6, 31.7]),
        ("AceReason", [3.3, 13.3, 30.0, 69.2, 61.8, 52.4, 9.8, 35.6, 26.3, 10.4, 24.6, 30.6]),
        ("Nemotron-Cascade", [3.3, 6.7, 27.5, 70.0, 60.0, 49.4, 11.4, 35.8, 26.8, 10.2, 29.0, 30.0]),
        ("Superior-Reasoning", [3.3, 3.3, 20.0, 51.0, 62.5, 35.4, 6.5, 12.2, 28.3, 4.2, 26.4, 23.0]),
    ],
    "sft": [
        ("starting checkpoint", [0.0, 0.0, 5.0, 35.2, 51.0, 45.1, 8.1, 40.2, 28.3, 6.5, 41.8, 23.8]),
        ("OpenThoughts3", [6.7, 10.0, 20.0, 70.6, 50.7, 58.5, 10.0, 29.2, 36.4, 4.6, 25.6, 29.3]),
        ("AceReason", [3.3, 20.0, 25.0, 69.4, 54.4, 48.2, 10.8, 33.6, 27.3, 9.9, 25.6, 29.8]),
        ("Nemotron-Cascade", [3.3, 13.3, 27.5, 66.8, 52.6, 50.6, None, 32.8, 34.8, 7.3, 30.6, 32.0]),
        ("Superior-Reasoning", [6.7, 3.3, 20.0, 51.6, 55.6, 22.0, 6.2, 15.6, 25.8, 4.9, 27.0, 21.7]),
    ],
    "dpo": [
        ("starting checkpoint", [0.0, 0.0, 2.5, 33.2, 51.9, 47.6, 8.7, 40.4, 27.8, 8.9, 45.0, 24.2]),
        ("OpenThoughts3", [10.0, 6.7, 20.0, 36.4, 50.7, 44.5, 11.9, 40.8, 26.8, 11.5, 43.0, 27.5]),
        ("AceReason", [6.7, 16.7, 27.5, 70.8, 55.4, 48.2, 11.4, 32.6, 23.7, 9.8, 25.0, 29.8]),
        ("Nemotron-Cascade", [6.7, 10.0, 25.0, 68.2, 53.5, 50.0, 13.3, 34.8, 28.8, 9.7, 30.4, 30.0]),
        ("Superior-Reasoning", [3.3, 6.7, 20.0, 54.2, 55.1, 28.7, 5.7, 13.2, 24.7, 5.3, 28.0, 22.3]),
    ],
}

STARTING_POINT_LABEL = {"base": "post-YaRN base", "sft": "Tulu3 SFT",
                        "dpo": "Tulu3 SFT + DPO"}


def check_constants() -> list[str]:
    """Internal consistency of the transcribed rows: lengths, interval order,
    and each printed average against the mean of its own printed cells (within
    0.15, since the cells are themselves rounded). Returns the problems found."""
    problems: list[str] = []

    def row_ok(where: str, row: list) -> None:
        if len(row) != 12:
            problems.append(f"{where}: {len(row)} values, expected 12")
            return
        present = [v for v in row[:11] if v is not None]
        mean = sum(present) / len(present)
        if abs(mean - row[11]) >= 0.15:
            problems.append(f"{where}: printed Avg {row[11]} vs mean of cells {mean:.2f}")

    for group, label, _, row in PHASE2_ENDPOINTS:
        row_ok(f"PHASE2_ENDPOINTS {group} {label}", row)
    for label, row in OT3_PRETRAIN_ABLATION:
        row_ok(f"OT3_PRETRAIN_ABLATION {label}", row)
    for stage, rows in HELDOUT_MV_GRID.items():
        for label, row in rows:
            row_ok(f"HELDOUT_MV_GRID {stage} {label}", row)
    for base, cells in HELDOUT_MATH500.items():
        if len(cells) != len(HELDOUT_CORPORA):
            problems.append(f"HELDOUT_MATH500 {base}: {len(cells)} corpora")
        for est, lo, hi, _ in cells:
            if not lo <= est <= hi:
                problems.append(f"HELDOUT_MATH500 {base}: {est} outside [{lo}, {hi}]")
    return problems
