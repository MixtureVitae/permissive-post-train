# Phase 2 tables from the results manifest

The paper's Phase 2 tables used to hold numbers copied by hand
(`mv_post_train/data/protocol_b.py`). This page explains how they are now
generated from `results/manifest.json` (see [manifest.md](manifest.md)), and
the rules every number follows.

```bash
mvpt fetch-phase2-records      # once, for the MATH500 intervals
mvpt render-phase2
```

## What it writes

| File | Paper | Contents |
|---|---|---|
| `results/phase2/tables/phase2_endpoints_full.tex` | Table 23 | Every checkpoint of Table 2, before and after its OpenThoughts3-30K (OT30K) update, on every task with its SD, in two column panels. Table 2 itself, one row per update with the change and its interval, is written by `mvpt render-phase2-analyses` (`docs/phase2_analyses.md`) |
| `results/phase2/tables/ot3_pretrain_ablation.tex` | Table 3 | The 300B models with and without OpenThoughts3 in pretraining, before and after OT30K |
| `results/phase2/tables/heldout_reasoning.tex` | Table 4 | MATH500 change under four reasoning corpora, from each base's Tulu3 SFT checkpoint, with 95% intervals |
| `results/phase2/tables/heldout_mv_grid.tex` | Table 32 | MixtureVitae under the four corpora, from each of its three starting checkpoints |
| `results/phase2/tables/eval_settings_B.tex` | Table 24 | The evaluation protocol of these tables |
| `results/phase2/tables/phase2_numbers.tex` | text | LaTeX macros for every number the tables print |
| `results/phase2/figures/phase2_receptivity.pdf` | Figure 2 | Suite-average change under OT30K, and the MATH500 changes of Table 4 |
| `results/phase2/figures/figure_pipeline.pdf` | Figure 1 | The post-training pipeline. Phase 1 runs at 4K and 16K. Phase 2 starts from three 16K checkpoints: SFT for the main comparison, and the base and DPO checkpoints for two controls. The command stops if a drawn Phase 2 branch has no run in the manifest. A `.png` copy sits next to it |
| `results/phase2/preview.pdf` | none | All of the tables and figures above, compiled with `pdflatex` |
| `artifacts/phase2/phase2_vs_submission.csv` | none | Every submitted cell next to its new value |

The tables and figures are committed, like the manifest, so a change to any
number shows up in review, and two renders give identical files. The table
files keep the paper's file names and labels, so they replace the old ones
without other edits. They need the LaTeX packages `booktabs`,
`graphicx`, `bm` and `amsmath`, and build with `pdflatex` (TeX Live 2023)
without warnings. On Ubuntu, `texlive-latex-base`, `texlive-latex-recommended`
and `texlive-fonts-recommended` provide them, and `texlive-latex-extra` adds
the `multirow` package that the repo's other tables use. `mvpt render-workshop` still renders the
submitted Phase 2 tables from the copied numbers, so the submission stays
reproducible.

## Rules

- **Every cell is a manifest result.** It is the mean over the three sampled
  runs, and the subscript is the standard deviation over those runs.
- **Every Avg is a manifest suite average**, over the tasks shown.
- **Every change is against the actual parent.** The change printed in
  parentheses is against the checkpoint the run was trained from, as the
  manifest records it. That checkpoint must be printed above it in the same
  block, and the manifest must have computed the change, so the two sides
  share their tasks and protocol. If any of this fails, the command stops
  with the reason instead of printing the table.
- **Parents known only by name are marked.** A change marked † is against a
  parent that only the checkpoint names give, because no training summary or
  model card names it.
- **One rounding.** Each value is rounded once. A printed change is the
  difference of the two printed averages, so a reader can check it by eye. It
  is always within 0.1 of the manifest's exact change.
- **Intervals come from the per-question records.** Table 4's intervals are
  95% paired bootstrap intervals over the 500 MATH500 problems. Each problem
  is scored as its mean over the three runs. Bold and italic mark changes
  that stay significant after a Benjamini-Hochberg correction over the 20
  changes. The records must first reproduce the manifest's score for every
  run, or the command stops.
- **Excluded tasks have no column.** A task the manifest leaves out gets no
  column, and the caption says why. Phase 2 GSM8K was left out until its
  re-run of 2026-10-15. No task is left out now.

## Text macros

`phase2_numbers.tex` defines `\ptwo{<key>}`. Load it once in the preamble and
write, for example, `\ptwo{mv.sft-ot3.avg}` instead of typing 31.2. Keys are:

- `<model>.<checkpoint>.<column>` for a value, such as `mv.base.m500`,
- `<model>.<checkpoint>.change.<column>` for a printed change, such as
  `mv.sft-ot3.change.avg`,
- `<model>.sft-<corpus>.change.m500`, `.lo` and `.hi` for Table 4.

Models are `mv`, `mvnoot3`, `fineweb`, `nemotron`, `comma`, `smollm`, `qwen25`
and `qwen3`. Checkpoints are `base`, `sft`, `dpo`, a corpus alone (`ot3`,
`ar`, `nc`, `sr`) when it was trained from the base, and `sft-<corpus>` or
`dpo-<corpus>` when it was trained from the SFT or DPO checkpoint. A key that
does not exist prints as bold question marks, so it cannot go unnoticed.

## IFEval at instruction level, strict (2026-10-19)

marianna13's revision `e820fb34` of the records scores IFEval by all four
definitions, prompt level and instruction level, each with the strict and the
loose checker, and adds `ifeval_allmetrics_persample.parquet`, the counts of
instructions followed per prompt and seed. Nothing else changed: every other
task has the same scores and the same answers. The authors chose
instruction-level strict on 2026-10-20, the definition her revision names as
Protocol A's, so the two protocols now agree. Against the loose prompt-level
score of before, Phase 2 IFEval rises by 10.9 points on average (5.3 to 14.5),
an IFEval change moves by up to 8.8 points, an 11-task average by up to 1.3 and
an 11-task change by up to 0.8. The manifest takes the item count of IFEval from the counts, 769
instructions over 500 prompts, and checks that the released score reproduces
from them.

## The GSM8K re-run (2026-10-15)

marianna13 re-ran every Phase 2 GSM8K cell with a limit of 16,000 new tokens
(revision `b219904b` of the records). Before, 83 of the 94 cells had no
response longer than 2,000 characters, so the answers were cut off at the
evaluation tool's default of 256 new tokens. Now no cell is cut off, and 0.4%
of responses have no extracted answer. Nothing else in the records changed:
every other task has the same scores and the same answers. The tables have
eleven task columns again, and every average is over all eleven tasks.

- **Figure 2:** OT30K adds 10.3 points to MixtureVitae's SFT checkpoint (22.5
  to 32.8), instead of 11.1. The other models still decline, by 1.0
  (Qwen2.5) to 17.7 (Qwen3).
- **Table 3:** the post-YaRN base averages 27.5 instead of 24.8. OT30K adds 7.0
  points to it (to 34.5) and 1.7 to the base without OpenThoughts3 (22.9 to
  24.6). The two bases differ by 4.6 points instead of 5.2.
- **Table 4** is MATH500 only, so it does not change.
- **DCLM's OT30K run** has its GSM8K cell from another checkpoint folder
  (`…-ot30k-rope8-32k`) than its other tasks (`…-ot30k`). The manifest keeps
  the two apart. No table prints DCLM's Phase 2 results.

## Limitations

- **Some runs disagree more than sampling explains.** In 4 of the 93 MATH500
  cells, the spread between the three runs is 2.8 to 6.3 times what
  independent sampling of the same questions would give, although each cell's
  runs come from one results file with one token limit. Two of them are starting points used here: the MixtureVitae
  post-YaRN base (33.6, 51.6 and 54.4) and its SFT checkpoint (35.4, 22.6 and
  29.6). Table 4's intervals treat the questions as the only source of
  variation, so they understate the uncertainty for these two. The evaluator
  should check these runs.
- **Several parents rest on names.** The AceReason, Nemotron-Cascade and
  Superior-Reasoning runs, and the run without OpenThoughts3, have no training
  summary or model card naming their parent. Their changes are marked †.
- **Table 3 pairs the post-YaRN base of Table 2 with the run without
  OpenThoughts3.** The records do not say which run the submitted Table 3
  used.
- **Nemotron-CC-HQ in Phase 2 descends from a separate 16K pretraining run**,
  not from the 4K base that Phase 1 extends.
- **IFEval is the instruction-level score with the strict checker** since
  2026-10-20, the definition of Protocol A, so both protocols use one. The
  records give all four definitions since marianna13's revision `e820fb34` of
  2026-10-19 (see below).
- **The protocol table lists what the manifest records:** task, metric,
  number of questions, token limit, and sampling at temperature 0.7 for every
  task. The number of shots and the citations are not in the manifest.
- **Some caption wording is carried over from the paper**, such as "30K
  samples, 5 epochs" and "differ only in whether the OpenThoughts3 subset is
  present". The manifest does not record training settings.
- **Table 3 no longer bolds the best value in each column.** The submitted
  caption called those values significant, and no test backs that for every
  task.
- **The macros cover values and parent-to-child changes only.** Differences
  between two models, such as the gap between the two bases in Table 3, are
  not generated.
