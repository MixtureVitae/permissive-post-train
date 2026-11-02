"""Render Tables 7, 8, 9 and 15, the older experiments.

    mvpt render-older-tables

Tables 7 (the 100B mixture ablation), 8 (decontamination) and 9 (the 0.4B
comparison) are under Protocol A, like Tables 1 and 6: each cell is the mean
over the three decoding seeds with its SD, and Avg is the 11-task suite
average of each seed, averaged over the seeds. The values are the results
manifest's. Its Protocol A results for these checkpoints come from R18, and
for Table 8's decontaminated rows, the main MV models, from R13. No cell is
bold: a column maximum across contexts and stages is no comparison the paper makes.

Table 15 (masking) is under Protocol A for both arms too. Its rows with
masking are the four masked MV checkpoints of R19, which the manifest keys by
the `laion` repos whose weights they copy, the checkpoints of Table 16. Its rows
without masking are Table 8's decontaminated SFT and DPO rows, the main MV
models of R13.

The functions that read the older evaluations of `logs.zip` (R6) and score them
by the older setup's rules stay here for the manifest's `older_evalchemy`
results, which no table prints.

  <out-dir>/tables/ablation_100b_reasoning.tex   Table 7
  <out-dir>/tables/decontamination_reasoning.tex Table 8
  <out-dir>/tables/smallscale_reasoning.tex      Table 9
  <out-dir>/tables/masking_reasoning.tex         Table 15
  <out-dir>/tables/older_numbers.tex             \\pold{<key>} macros for the text
  <out-dir>/preview.pdf                          the four tables, compiled
"""
from __future__ import annotations

import hashlib
import json
import zipfile
from dataclasses import dataclass
from pathlib import Path

from ..data.run_rule import broken, run_value
from .. import inputs
from . import preview
from . import workshop_tables as wt
from .macros import macro_lines, rotated

#: logs.zip, input R6 of mv_post_train/inputs.json.
LOGS_ZIP_SHA256 = inputs.get("R6").sha256
LOGS_ZIP_URI = inputs.get("R6").gcs
#: Hashing the zip takes about a minute, so every run checks a digest of the
#: results files it reads (members_digest) instead.
LOGS_MEMBERS_SHA256 = inputs.member_digest("R6", "older_tables")

#: (abbreviation, task) in the paper's column order.
BENCHMARKS = (("A24", "AIME24"), ("A25", "AIME25"), ("AMC", "AMC23"), ("M500", "MATH500"),
              ("GSM8K", "gsm8k"), ("HE", "HumanEval"), ("LCB", "LiveCodeBench"),
              ("MBPP", "MBPP"), ("GPQA", "GPQADiamond"), ("JEE", "JEEBench"), ("IFE", "IFEval"))
TASKS = tuple(t for _, t in BENCHMARKS)

STAGE_SUFFIX = (("", ""), (r"\quad $\to$ SFT", "-SFT-Tulu3-decontaminated"),
                (r"\quad $\to$ SFT $\to$ DPO", "-DPO-Tulu3-decontaminated"))


@dataclass(frozen=True)
class Group:
    label: str          # the paper's row label of the base checkpoint
    key: str            # macro key
    base: str           # Hugging Face id of the base checkpoint

    def rows(self):
        """(row label, macro key, checkpoint) for base, SFT and DPO."""
        for (label, suffix), stage in zip(STAGE_SUFFIX, ("base", "sft", "dpo")):
            yield label or self.label, f"{self.key}.{stage}", self.base + suffix


def _pair(label: str, key: str, base: str) -> tuple[Group, Group]:
    return (Group(label, f"{key}.4k", base),
            Group(f"{label}-16k" if "(" not in label else label.replace(" (", "-16k ("),
                  f"{key}.16k", base + "-longsft_16k"))


A = "ali-elganzory/"
NOTE = " " + wt.PROTOCOL_A_NOTE
TABLES = (
    dict(name="ablation_100b_reasoning.tex", label="tab:evalchemy_table1", key="abl",
         caption=(r"Per-benchmark results for the 100B-token MixtureVitae mixture composition "
                  r"ablation on the primary reasoning suite. Removing the instruction-and-reasoning "
                  r"subset causes a broad collapse in math, code, and reasoning that Phase 1 Tulu3 "
                  r"SFT/DPO does not recover, whereas removing the curated web subset leaves "
                  r"performance comparable to, and on several benchmarks stronger than, the full "
                  r"mixture at both 4K and 16K context."),
         groups=(*_pair("MV-noinstruct", "noinst", A + "1.7b-MixtureVitae-web_curated-100BT"),
                 *_pair("MV-noweb", "noweb", A + "1.7b-MixtureVitae-curated_instruct-100BT"),
                 *_pair("MV", "mv100", A + "1.7b-MixtureVitae-100BT"))),
    dict(name="smallscale_reasoning.tex", label="tab:evalchemy_table4", key="small",
         caption=(r"Reasoning suite results for the smaller-scale comparison between MV (0.4B) "
                  r"and Baguettotron. Across 4K and 16K context, MV is substantially more receptive "
                  r"to Phase 1 post-training, especially on code and broader reasoning, while "
                  r"Baguettotron remains weak throughout."),
         groups=(Group("Baguettotron", "bag.4k", A + "Baguettotron"),
                 Group("Baguettotron 16k", "bag.16k", A + "Baguettotron-longsft_16k"),
                 Group("MV (0.4B)", "mv04.4k",
                       A + "0.4b-mixturevitae-v1-decontaminated-300B-4096"),
                 Group("MV-16k (0.4B)", "mv04.16k",
                       A + "0.4b-mixturevitae-v1-decontaminated-300B-4096-longsft_16k"))),
)


#: Table 8 (decontamination): (section label, macro key, blocks), each block
#: (base row label, context key, base, SFT and DPO checkpoints).
O = A + "1.7b-MixtureVitae-300BT-v1-"
DECON = (
    ("Without filtering", "orig", (
        ("MV", "4k", O + "WithChatTemplate", O + "SFT-Tulu3", O + "DPO-Tulu3"),
        ("MV-16k", "16k", O + "16k-WithChatTemplate", O + "16k-SFT-Tulu3", O + "16k-DPO-Tulu3"))),
    ("With filtering", "clean", (
        ("MV", "4k", O + "decontaminated", O + "decontaminated-SFT-Tulu3-decontaminated",
         O + "decontaminated-DPO-Tulu3-decontaminated"),
        ("MV-16k", "16k", O + "decontaminated-16k",
         O + "decontaminated-16k-SFT-Tulu3-decontaminated",
         O + "decontaminated-16k-DPO-Tulu3-decontaminated"))),
)
STAGE_ROWS = (("base", ""), ("sft", r"\quad $\to$ SFT"), ("dpo", r"\quad $\to$ SFT $\to$ DPO"))
#: Table 15's rows with masking: (context, stage) -> checkpoint. R19 names
#: copies of these repos (data/protocol_a.py, RECORD_IDENTITY).
_M = "laion/1.7b-MixtureVitae-300BT-v1-decontaminated-{}-Tulu3-decontaminated-masked"
MASKED = {("4k", "sft"): _M.format("SFT"), ("4k", "dpo"): _M.format("DPO"),
          ("16k", "sft"): _M.format("16k-SFT"), ("16k", "dpo"): _M.format("16k-DPO")}
DECON_CAPTION = (
    r"Reasoning suite of 300B MixtureVitae checkpoints trained without and with the historical "
    r"13-gram filtering of their data, at 4K and 16K context, before and after Phase 1 Tulu3 "
    r"SFT/DPO, a check of that procedure, not of benchmark-disjoint data "
    r"(Appendix~\ref{app:decontam})." + NOTE)
MASKING_CAPTION = (
    r"Reasoning suite with vs.\ without prompt-token loss masking in Phase-1 post-training "
    r"(1.7B MixtureVitae 300BT, decontaminated). The rows with "
    r"masking are the \texttt{laion} masked checkpoints of Table~\ref{tab:lang_eval_masking}, "
    r"and the rows without masking are the main MV checkpoints of "
    r"Table~\ref{tab:evalchemy_table3}. Masking {moves} the suite average by {spread} points "
    r"at every stage, and these differences do not affect our comparison." + NOTE)

def unmasked() -> dict[tuple[str, str], str]:
    """Table 15's rows without masking: Table 8's decontaminated SFT and DPO checkpoints."""
    clean = {ctx: cks for _, skey, blocks in DECON if skey == "clean" for _, ctx, *cks in blocks}
    return {(ctx, stage): clean[ctx][i] for ctx in ("4k", "16k")
            for i, stage in ((1, "sft"), (2, "dpo"))}


def checkpoints() -> list[str]:
    return ([ck for t in TABLES for g in t["groups"] for _, _, ck in g.rows()]
            + [ck for _, _, blocks in DECON for b in blocks for ck in b[2:]])


# ---------------------------------------------------------------------------
# Reading the logs


def _scores_only(raw: bytes) -> bytes:
    """A results file without its saved answers (`examples`), which the tables
    do not read and which can be megabytes per task."""
    text = raw.decode().strip()
    if not text:
        return b""
    data = json.loads(text)
    results = {task: {k: v for k, v in payload.items() if k != "examples"}
               if isinstance(payload, dict) else payload
               for task, payload in data.get("results", {}).items()}
    return json.dumps({"results": results}).encode()


def read_folders(logs_zip: Path, wanted: list[str],
                 digests: list | None = None) -> dict[str, list[tuple[str, bytes]]]:
    """Checkpoint -> (member, scores-only bytes) of its results files in logs.zip.
    If `digests` is a list, (member, SHA-256 of the file's bytes) is appended to it
    for every file read, for `members_digest`."""
    folders = {ck.replace("/", "__"): ck for ck in wanted}
    out: dict[str, list[tuple[str, bytes]]] = {ck: [] for ck in wanted}
    with zipfile.ZipFile(logs_zip) as z:
        for member in sorted(z.namelist()):
            parts = member.split("/")
            if len(parts) == 3 and parts[1] in folders and parts[2].startswith("results_") \
                    and parts[2].endswith(".json"):
                raw = z.read(member)
                if digests is not None:
                    digests.append((member, hashlib.sha256(raw).digest()))
                out[folders[parts[1]]].append((member, _scores_only(raw)))
    missing = [ck for ck, files in out.items() if not files]
    if missing:
        raise ValueError(f"{logs_zip} has no results files for {missing}")
    return out


def members_digest(digests: list[tuple[str, bytes]]) -> str:
    """SHA-256 over each member's name and the SHA-256 of its bytes, in name order."""
    h = hashlib.sha256()
    for member, d in sorted(digests):
        h.update(member.encode() + b"\0" + d)
    return h.hexdigest()


def runs(files: list[tuple[str, bytes]]) -> dict[str, list[tuple[str, float]]]:
    """Task -> (results file, value) for every run, as the paper's script reads
    them: files that are empty or hold an empty first result are skipped."""
    out: dict[str, list[tuple[str, float]]] = {}
    for member, raw in files:
        text = raw.decode().strip()
        if not text:
            continue
        results = json.loads(text).get("results", {})
        if not results or list(results.values())[0] == {}:
            continue
        for task, payload in results.items():
            if task in TASKS and payload:
                out.setdefault(task, []).append((member, run_value(payload)))
    return out


def scores_unbroken(files: list[tuple[str, bytes]]) -> dict[str, float]:
    """Task -> mean value over the runs that are not broken."""
    out = {}
    for task, rs in runs(files).items():
        values = [v for _, v in rs]
        kept = [v for v, bad in zip(values, broken(values)) if not bad]
        out[task] = sum(kept) / len(kept)
    return out


# ---------------------------------------------------------------------------
# Tables and macros


@dataclass(frozen=True)
class ProtocolA:
    """Protocol A means and SDs of the tables' checkpoints, from the manifest:
    checkpoint -> task -> value, and checkpoint -> (suite mean, suite SD)."""
    mean: dict[str, dict[str, float]]
    sd: dict[str, dict[str, float]]
    avg: dict[str, tuple[float, float]]


def _cell(mean: float, sd: float, bold: bool) -> str:
    text = f"\\textbf{{{mean:.1f}}}" if bold else f"{mean:.1f}"
    return f"{text}$_{{\\pm{sd:.2f}}}$"


def _pa_cells(pa: ProtocolA, ck: str) -> list[tuple[float, float]]:
    """(mean, SD) of the eleven tasks and Avg, in column order."""
    return [(pa.mean[ck][t], pa.sd[ck][t]) for t in TASKS] + [pa.avg[ck]]


def _rows(table: dict, pa: ProtocolA):
    """(label, key, cells, group index) for every row, cells as _pa_cells."""
    return [(label, key, _pa_cells(pa, ck), gi)
            for gi, group in enumerate(table["groups"]) for label, key, ck in group.rows()]


#: The two column panels of the wide reasoning tables: the five mathematics tasks,
#: then the other six tasks and Avg, as indexes into `_pa_cells`. One float holds
#: both, at a readable size, where the tables used to be scaled to the page width.
PANELS = (("(a) Mathematics", range(0, 5)),
          ("(b) Code, science and instruction following, and the average", range(5, 12)))
HEADS = [a for a, _ in BENCHMARKS] + ["Avg"]


def _paneled(caption: str, label: str, lead_head: list[str], body: list[tuple],
             placement: str = "ht") -> str:
    """One float with the column panels of PANELS. `body` holds ("row", lead cells,
    cells as `_pa_cells` or None for a row without values), ("mid",) for a rule
    with space between blocks, or ("cmid", first column) for a rule over the
    columns from `first` on."""
    lines = [f"\\begin{{table}}[{placement}]", r"\centering", f"\\caption{{{caption}}}",
             f"\\label{{{label}}}", r"\footnotesize", r"\setlength{\tabcolsep}{3pt}"]
    lead = len(lead_head)
    for k, (title, cols) in enumerate(PANELS):
        width = lead + len(cols)
        lines += ([r"\par\medskip"] if k else []) + [
            r"\begin{tabular}{" + "l" * lead + "r" * len(cols) + "}", r"\toprule",
            f"\\multicolumn{{{width}}}{{l}}{{\\textbf{{{title}}}}} \\\\", r"\midrule",
            " & ".join(lead_head + [f"\\textbf{{{HEADS[c]}}}" for c in cols]) + r" \\",
            r"\midrule"]
        for item in body:
            if item[0] == "mid":
                lines += [r"\midrule", r"\addlinespace[0.3em]"]
            elif item[0] == "cmid":
                lines.append(f"\\cmidrule{{{item[1]}-{width}}}")
            else:
                _, cells_lead, cells = item
                values = ([""] * len(cols) if cells is None
                          else [_cell(*cells[c], False) for c in cols])
                lines.append(" & ".join(cells_lead + values) + r" \\")
        lines += [r"\bottomrule", r"\end{tabular}"]
    return "\n".join(lines + [r"\end{table}", ""])


def table_tex(table: dict, pa: ProtocolA) -> str:
    body, last = [], 0
    for label, _, cells, gi in _rows(table, pa):
        if gi != last:
            body.append(("mid",))
            last = gi
        body.append(("row", [label], cells))
    return _paneled(table["caption"] + NOTE, table["label"], [r"\textbf{Model}"], body)


def _decon_rows(pa: ProtocolA):
    """(section, key, block label, stage, row label, cells) for Table 8's twelve rows."""
    for section, skey, blocks in DECON:
        for label, ctx, *cks in blocks:
            for (stage, row_label), ck in zip(STAGE_ROWS, cks):
                yield (section, f"{skey}.{ctx}.{stage}", label, stage, row_label or label,
                       _pa_cells(pa, ck))


def decontamination_tex(pa: ProtocolA) -> str:
    """Table 8, laid out as the paper's: two sections of two blocks each."""
    body = []
    for k, (section, _, label, stage, row_label, cells) in enumerate(_decon_rows(pa)):
        first = k % 6 == 0
        if first and k:
            body.append(("mid",))
        elif stage == "base" and k:
            body.append(("cmid", 2))
        lead = f"\\multirow{{6}}{{*}}{{{rotated(section)}}}" if first else ""
        body.append(("row", [lead, row_label], cells))
    return _paneled(DECON_CAPTION, "tab:evalchemy_table3", ["", r"\textbf{Model}"], body)


def _suite_changes(pa: ProtocolA) -> list[float]:
    """Masked minus unmasked suite average at each stage, from the unrounded means."""
    um = unmasked()
    return [round(pa.avg[MASKED[k]][0] - pa.avg[um[k]][0], 1) + 0.0 for k in MASKED]


def masking_caption(pa: ProtocolA) -> str:
    d = _suite_changes(pa)
    moves = "raises" if min(d) > 0 else "lowers" if max(d) < 0 else "changes"
    lo, hi = sorted((abs(min(d)), abs(max(d)))) if moves != "changes" else (min(d), max(d))
    spread = f"{lo:.1f} to {hi:.1f}" if moves != "changes" else f"{lo:+.1f} to {hi:+.1f}"
    return MASKING_CAPTION.replace("{moves}", moves).replace("{spread}", spread)


def masking_tex(pa: ProtocolA) -> str:
    """Table 15: the masked checkpoints of R19 against Table 8's decontaminated
    ones, both under Protocol A."""
    body = []
    for name, cks in (("With masking", MASKED), ("Without masking", unmasked())):
        if cks is not MASKED:
            body.append(("mid",))
        lead = f"\\multirow{{6}}{{*}}{{{rotated(name)}}}"
        for ctx, label in (("4k", "MV"), ("16k", "MV-16k")):
            if ctx == "16k":
                body.append(("cmid", 2))
            body.append(("row", [lead, label], None))
            lead = ""
            for stage, row_label in STAGE_ROWS[1:]:
                body.append(("row", ["", row_label], _pa_cells(pa, cks[(ctx, stage)])))
    return _paneled(masking_caption(pa), "tab:evalchemy_masking", ["", r"\textbf{Model}"], body,
                    placement="htbp")


def _signed(v: float) -> str:
    return f"\\ensuremath{{{v:+.1f}}}"


def macros(pa: ProtocolA) -> dict[str, str]:
    """Key -> printed text, for `\\pold{<key>}`. Tables are `abl` (Table 7) and
    `small` (Table 9). Groups are `noinst`, `noweb` and `mv100` in Table 7, and
    `bag` and `mv04` in Table 9. Tasks are the column abbreviations in lower case
    (`a24`, `m500`, `he`, `mbpp`, ...), and `avg`.
      <table>.<group>.<ctx>.<stage>.<task>[.sd]   one cell, for example small.mv04.4k.dpo.he
      <table>.<group>.<ctx>.<task>.min/.max   the range of a task over a block's three rows
      <table>.<group>.<ctx>.post.<task>.min/.max   the same over SFT and DPO only
      <table>.<group>.<task>.min/.max         the range over both context lengths
      <table>.<group>.post.<task>.min/.max    the same over SFT and DPO only
      decon.<orig|clean>.<ctx>.<stage>.<task>[.sd]    one cell of Table 8
      decon.change.<ctx>.<stage>.<task>[.abs]   decontaminated minus not, from the
          unrounded means, signed, or its size
      decon.change.<ctx>.post.<task>.absmax    the largest size over SFT and DPO
      mask.<masked|unmasked>.<ctx>.<stage>.<task>[.sd]   one cell of Table 15 (stage
          sft or dpo)
      mask.change.<ctx>.<stage>.<task>   masked minus unmasked, from the unrounded means
      mask.change.<task>.min/.max        its range over the four stages
    Every table is Protocol A means and SDs. Ranges are over printed means.
    """
    out: dict[str, str] = {}
    names = [a.lower() for a, _ in BENCHMARKS] + ["avg"]
    for table in TABLES:
        spans: dict[str, list[float]] = {}
        for _, key, cells, _ in _rows(table, pa):
            group, ctx, stage = key.split(".")
            stem = f"{table['key']}.{key}"
            for name, (mean, sd) in zip(names, cells):
                printed = round(mean, 1)
                out[f"{stem}.{name}"] = f"{printed:.1f}"
                out[f"{stem}.{name}.sd"] = f"{sd:.2f}"
                block = [f"{table['key']}.{group}.{ctx}", f"{table['key']}.{group}"]
                if stage != "base":
                    block += [f"{table['key']}.{group}.{ctx}.post", f"{table['key']}.{group}.post"]
                for span in block:
                    spans.setdefault(f"{span}.{name}", []).append(printed)
        for span, v in spans.items():
            out[f"{span}.min"], out[f"{span}.max"] = f"{min(v):.1f}", f"{max(v):.1f}"
    raw: dict[str, float] = {}
    for _, key, *_rest, cells in _decon_rows(pa):
        for name, (mean, sd) in zip(names, cells):
            out[f"decon.{key}.{name}"] = f"{round(mean, 1):.1f}"
            out[f"decon.{key}.{name}.sd"] = f"{sd:.2f}"
            raw[f"decon.{key}.{name}"] = mean
    for ctx in ("4k", "16k"):
        for name in names:
            post = []
            for stage in ("base", "sft", "dpo"):
                a, b = raw[f"decon.orig.{ctx}.{stage}.{name}"], raw[f"decon.clean.{ctx}.{stage}.{name}"]
                d = round(b - a, 1) + 0.0
                out[f"decon.change.{ctx}.{stage}.{name}"] = _signed(d)
                out[f"decon.change.{ctx}.{stage}.{name}.abs"] = f"{abs(d):.1f}"
                if stage != "base":
                    post.append(abs(d))
            out[f"decon.change.{ctx}.post.{name}.absmax"] = f"{max(post):.1f}"
    for kind, cks in (("masked", MASKED), ("unmasked", unmasked())):
        for (ctx, stage), ck in cks.items():
            for name, (mean, sd) in zip(names, _pa_cells(pa, ck)):
                out[f"mask.{kind}.{ctx}.{stage}.{name}"] = f"{round(mean, 1):.1f}"
                out[f"mask.{kind}.{ctx}.{stage}.{name}.sd"] = f"{sd:.2f}"
                raw[f"mask.{kind}.{ctx}.{stage}.{name}"] = mean
    for name in names:
        changes = []
        for ctx, stage in MASKED:
            a = raw[f"mask.unmasked.{ctx}.{stage}.{name}"]
            b = raw[f"mask.masked.{ctx}.{stage}.{name}"]
            changes.append(round(b - a, 1) + 0.0)
            out[f"mask.change.{ctx}.{stage}.{name}"] = _signed(changes[-1])
        out[f"mask.change.{name}.min"] = _signed(min(changes))
        out[f"mask.change.{name}.max"] = _signed(max(changes))
    return dict(sorted(out.items()))


def write_macros(values: dict[str, str], path: Path) -> Path:
    lines = [
        "% Generated by `mvpt render-older-tables` (mv_post_train). Do not edit by hand.",
        "% Numbers for the text, from the same values as Tables 7, 8, 9 and 15.",
        r"% Use \pold{<key>}, for example \pold{small.mv04.4k.dpo.avg}.",
        "% A key that does not exist prints in bold between question marks.",
    ]
    lines += macro_lines("pold", values)
    path.write_text("\n".join(lines) + "\n")
    return path


def from_manifest(manifest_path: Path) -> ProtocolA:
    """The Protocol A values of every checkpoint of the four tables, from the
    results manifest."""
    from ..manifest.build import PROTOCOL_A
    from ..manifest.lineage import bare
    from ..manifest.reader import Manifest
    from ..manifest.suites import OLDER_TASK
    m = Manifest.load(manifest_path)
    raw = {v: k for k, v in OLDER_TASK.items()}
    mean, sd, avg = {}, {}, {}
    pa_cks = ([ck for t in TABLES for g in t["groups"] for _, _, ck in g.rows()]
              + [ck for _, _, blocks in DECON for b in blocks for ck in b[2:]]
              + list(MASKED.values()))
    for ck in dict.fromkeys(pa_cks):
        results = m.results_for(bare(ck), PROTOCOL_A)
        if len(results) != len(TASKS):
            raise ValueError(f"The manifest has {len(results)} Protocol A results for {ck}.")
        mean[ck] = {raw.get(r["task"], r["task"]): r["mean"] for r in results}
        sd[ck] = {raw.get(r["task"], r["task"]): r["sd"] for r in results}
        suite = m.suite(bare(ck), PROTOCOL_A)
        avg[ck] = (suite["mean"], suite["sd"])
    return ProtocolA(mean, sd, avg)


def render_all(manifest_path: Path, out_dir: Path) -> list[Path]:
    pa = from_manifest(manifest_path)
    tables_dir = out_dir / "tables"
    tables_dir.mkdir(parents=True, exist_ok=True)
    header = "% Generated by `mvpt render-older-tables` (mv_post_train). Do not edit by hand.\n"
    tables = []
    for table in TABLES:
        path = tables_dir / table["name"]
        path.write_text(header + table_tex(table, pa))
        tables.append(path)
    for name, body in (("decontamination_reasoning.tex", decontamination_tex(pa)),
                       ("masking_reasoning.tex", masking_tex(pa))):
        (tables_dir / name).write_text(header + body)
        tables.append(tables_dir / name)
    written = tables + [write_macros(macros(pa), tables_dir / "older_numbers.tex")]
    pdf = preview.build("Tables 7, 8, 9 and 15, the older experiments (render-older-tables)",
                        tables, [], out_dir / "preview.pdf")
    if pdf is not None:
        written.append(pdf)
    return written


def main(manifest_path: Path, out_dir: Path) -> int:
    for path in render_all(manifest_path, out_dir):
        print(f"  wrote {path}")
    return 0
