# Language changes with intervals

`mvpt render-language-intervals` measures how each training step changes the
language suite, with a 95% interval for every change. It answers the question
the paper's language tables leave open: is a change of a point or two on the
language average real, or within what the sample of questions allows? It
does the same for the front-loading study's language contrasts (Table 28),
as Table 27 does for reasoning.

## What it reads

The language evaluation logs (inputs R8, R8-fineweb and R23):

| File | What it holds |
|---|---|
| `language_eval-logs.zip` | 19 checkpoints: MixtureVitae, Comma0.1 and Nemotron-CC-HQ, from the base to the reasoning runs |
| `logs-fineweb.zip` | 7 FineWeb-Edu checkpoints, uploaded by Harsh Raj on 2026-10-07 |
| `language_eval_all_models_3seed_2026-10-26.zip` (R23) | Harsh Raj's runs of 2026-10-26, received on 2026-10-27. The command reads 9 of its 24 checkpoints with per-question files: the front-loading study's language cells that the other two zips do not hold |

All three are in the project bucket and pinned by SHA-256
(`mv_post_train/inputs.json`). If they are not in `data/language_logs/`, the
command downloads them. The parsed scores are cached in the same folder.

Each checkpoint was run three times, once per prompt-example seed. Every run
saved one line per question, with the question's score. So each checkpoint has
a score for every question of every task, three times over. The seeds of R8
are `default` (1234), `1765181304` and `1819856179`, those of R23 1234, 42 and
2025. R23 keeps each task and seed of a checkpoint in a folder of its own,
which the command reads as one folder per checkpoint and seed
(`<checkpoint>-seed<N>`).

## How the scores are read

- **The run.** Each folder holds one run per task, and the command picks it
  with the same rule as `LanguageParser`, the language tables' parser.
  `logs-fineweb.zip` stores its whole log tree twice, under two root folders,
  with identical files. Each file is read once, and the command stops if two
  copies ever differ.
- **The metric.** The one the paper's language tables print: normalized
  accuracy for ARC-C, ARC-E, HellaSwag, OBQA and PIQA, plain accuracy for the
  other six tasks. MMLU is the mean over all 14,042 questions of its 57
  subjects, which is how the evaluation tool computes it.
- **Three checks.** The command stops unless all of these hold, and all 78
  folders pass them:
  - For every checkpoint, seed and task, the mean of the question scores equals
    the score in the results file.
  - Every question key (`doc_id`, with the subject for MMLU) names the same
    question, by its `doc_hash`, in every folder, so the pairing of questions
    is right.
  - The three seed folders of a checkpoint give the same answer on every
    question of COPA, LAMBADA, MMLU, OBQA and WinoGrande. These run with no
    examples in the prompt, so the seed cannot change them, and a difference
    would mean the folders hold different models.
- **The seed.** The tables use seed 1234, R8's `default`, like the paper and
  the language results of 2026-10-14 (R16). A second set of numbers averages
  each question over the three seeds. A contrast between an R8 checkpoint and
  an R23 checkpoint then averages different seeds on its two sides, apart
  from 1234.

The seeds change only which solved examples are put in the prompt, and the
third check confirms that the five tasks without examples give the same answers
under all three. So the three seeds measure how much the choice of examples matters. They are not
three training runs, and no interval here describes training variation.

## How the intervals are computed

The same way as the Protocol A intervals of the paper (`appendix/statistics.tex`,
`stats/factorial.py`):

1. Each question gets one score per checkpoint.
2. A replicate draws the questions of each task again, with replacement, and
   uses the same draw for every checkpoint of the model. So a change is always
   measured on the same questions on both sides.
3. The 11-task average is the plain mean of the 11 task scores, recomputed in
   every replicate.
4. 2,000 replicates give the 95% interval and a two-sided p-value.
5. The p-values are corrected with the Benjamini-Hochberg method, once over a
   model's changes in the average and once over its changes in single tasks.

In the table of averages, which prints the two scores next to each change, a
printed change is the difference of the two printed scores, like everywhere
else in this repo. The per-task table prints no scores, so there each task's
change is its own estimate, rounded, and it matches the CSV and sits inside its
interval. Its Avg column prints the change as the table of averages does, so
the two tables never show different numbers for the same change. For MV's
Phase 1 change, for example, both print +0.55, the difference of 55.51 and
54.96, while the estimate rounds to +0.56. The macros follow the same rules.

## Which changes

Each change is a checkpoint minus the checkpoint it was trained from. The
command checks every change against the results manifest's lineage, whose
parents come from the model cards and training summaries in the Hugging Face
snapshot (`data/manifest/hf_checkpoints.json`), and from the checkpoint names
where those say nothing. Two names with the same weights are one model. R8's
FineWeb-Edu 16K run is the April 16K model, the parent of the 16K SFT model,
because the repo it loaded held the April weights in May 2026.

| Model | Changes |
|---|---|
| MixtureVitae | 16K extension, SFT, DPO, OT30K from the base, from SFT and from DPO. From the 16K base: through SFT and DPO, through SFT and OT30K, through SFT, DPO and OT30K |
| Comma0.1 | 16K extension, SFT, DPO, OT30K from SFT and from DPO. From the 16K base: through SFT and DPO, through SFT and OT30K, through SFT, DPO and OT30K |
| Nemotron-CC-HQ | 16K extension, DPO, OT30K from SFT and from DPO |
| FineWeb-Edu | SFT and DPO at 4K, 16K extension, SFT and DPO at 16K, OT30K from SFT. From the 16K base: through SFT and DPO, through SFT and OT30K |

A change over several steps, from the 16K base, is marked with a double arrow.
These answer the paper's questions directly: where does the language average
end up after Phase 1, and after the whole pipeline, compared with the base?

Two caveats about the models:

- **Nemotron-CC-HQ.** The logs' 16K base (`…4096-long_sft_16k`) is not the
  model the 16K SFT checkpoint was trained from (`…16384-rope_theta-1M-long_sft_16k`). So no change starts from that base, and the
  command refuses one if it is asked for.
- **FineWeb-Edu.** Its 16K logs were run in May, on the April weights of the
  16K model. That is the model its 16K SFT checkpoint was trained from, so
  its changes are on one line of models. They are not the August weights that
  Tables 1, 5 and 31 use.

## The front-loading study

Table 31 has sixteen language cells: four web datasets, with and without
front-loading, each as the 4K base and its post-YaRN 16K extension. Seven of
them are R8 checkpoints of the changes above. The other nine read R23, the
runs the results manifest prints for them since 2026-10-27
(`manifest/suites.py`, R23_CELLS). The command stops unless every cell is the
checkpoint the manifest's `front_loading_study` design names.

The contrasts are those of Table 27, on the 11-task language average, at 4K
and at 16K:

| Family | Contrasts | What it asks |
|---|---|---|
| Front-loading | 4 per context | Does front-loading change each dataset's language average? |
| Permissive minus non-permissive | 4 per context: MV and Comma0.1, each minus Nemotron-CC-HQ and minus FineWeb-Edu, all front-loaded | How large is the gap that front-loading leaves? |
| Interaction | 2 per context | Does front-loading help a permissive dataset more than the mean of the two others? |

The method is that of the changes above: one shared draw of the questions per
replicate, 2,000 replicates, and one Benjamini-Hochberg correction per family,
over both contexts, as Table 27 corrects its families.

## What is not covered

- **The larger open-weights models** (SmolLM2, Qwen2.5 and Qwen3) and the
  100B, decontamination, 0.4B and masking checkpoints. R16 has their results
  files but no per-question files.

## Outputs

All under `results/language/`, committed:

| File | What it is |
|---|---|
| `tables/language_changes.tex` | Every change in the 11-task average, with its interval, q-value and the three-seed version |
| `tables/language_changes_mv.tex` | MixtureVitae's changes on each task |
| `tables/language_frontloading.tex` | The front-loading study's language contrasts, Table 28 |
| `tables/language_numbers.tex` | A macro for every number, `\langci{<key>}` |
| `language_changes.csv` | Every change on every task, both seed settings, with intervals, p and q |
| `language_frontloading.csv` | The same for every front-loading contrast |
| `preview.pdf` | The two tables, compiled |

Macro keys are `<model>.<checkpoint>.<task>` for a score,
`<model>.<checkpoint>.change.<task>` for a single-step change and
`<model>.<checkpoint>.from-base.<task>` for a change from the 16K base (for
example `mv.dpo.from-base.avg` for Phase 1 and `mv.dpo-ot3.from-base.avg` for
the whole pipeline), each
change with `.lo`, `.hi` and `.q`. The front-loading contrasts are
`fl.<dataset>.<ctx>.<task>` for a front-loading effect (`fl.mv.16k.avg`),
`fl.<permissive>-<non-permissive>.<ctx>.<task>` for a gap
(`fl.comma-fineweb.4k.avg`) and `fl.<permissive>-nonperm.<ctx>.<task>` for an
interaction. `fl.<family>.<ctx>.min`, `.max` and `.qmax` are the smallest
and largest estimate and the largest q of a family's average contrasts at one
context length. The front-loading q macros print with their q, as the
front-loading study's `\pfl` macros do: "q = 0.258", "q < 0.001", or "q ≤
0.028" for a largest q. Tasks are the paper's abbreviations in lower
case (`arc-c`, `hellas`, `winog`) and `avg`.

## Results

Every number below is as the outputs print it (`results/language/tables/`,
the macros and the CSV), for the `default` seed. Intervals are 95%. Changes in
the average have two decimals, changes in one task have one, and where a
second decimal matters, the CSV's value is given too.

### MixtureVitae keeps its language average

| Change on the 11-task average | From | To | Change [interval] | q |
|---|---|---|---|---|
| 16K base, then SFT, DPO and OT30K | 54.96 | 55.74 | +0.78 [+0.19, +1.41] | 0.042 |
| 16K base, then SFT and OT30K | 54.96 | 55.68 | +0.72 [+0.12, +1.35] | 0.042 |
| 16K base, then OT30K (no Phase 1) | 54.96 | 55.64 | +0.68 [+0.30, +1.10] | 0.005 |
| 16K base, then SFT and DPO (Phase 1) | 54.96 | 55.51 | +0.55 [−0.07, +1.18] | 0.152 |
| SFT | 54.96 | 55.45 | +0.49 [−0.12, +1.13] | 0.167 |
| DPO | 55.45 | 55.51 | +0.06 [−0.01, +0.13] | 0.152 |
| OT30K after SFT | 55.45 | 55.68 | +0.23 [−0.23, +0.66] | 0.344 |
| OT30K after DPO | 55.51 | 55.74 | +0.23 [−0.24, +0.67] | 0.344 |
| 16K extension, from the 4K base | 55.82 | 54.96 | −0.86 [−1.56, −0.15] | 0.042 |

- **No route from the 16K base lowers the average.** The three routes that end
  with OT30K end slightly higher, and their intervals exclude a loss.
- **Phase 1 alone holds the average but cannot rule out a small loss.** Through
  SFT and DPO the average moves by +0.55, with an interval from −0.07 to +1.18.
- **The step that costs language is the 16K extension**, by close to a point.
  After the full pipeline, MV is back at its 4K level (55.74 against 55.82).
- **Single tasks move both ways, and ARC-C drops on every route** (one decimal,
  as `language_changes_mv.tex` prints):

| Route from the 16K base | ARC-C | ARC-E | HellaSwag | OBQA |
|---|---|---|---|---|
| SFT | −2.5 [−4.1, −0.9], q 0.042 | −1.2 [−2.2, −0.1], q 0.087 | +1.8 [+1.3, +2.3], q 0.007 | +2.4 [+0.6, +4.2], q 0.042 |
| SFT, DPO | −2.5 [−4.1, −0.9], q 0.032 | −0.9 [−1.9, +0.2], q 0.267 | +1.9 [+1.4, +2.4], q 0.007 | +2.6 [+0.6, +4.4], q 0.042 |
| SFT, OT30K | −2.3 [−3.9, −0.7], q 0.042 | −1.1 [−2.2, −0.1], q 0.115 | +2.1 [+1.6, +2.6], q 0.007 | +2.6 [+0.8, +4.4], q 0.042 |
| SFT, DPO, OT30K | −2.0 [−3.6, −0.3], q 0.056 | −1.4 [−2.4, −0.4], q 0.042 | +2.3 [+1.8, +2.9], q 0.007 | +2.0 [+0.2, +3.8], q 0.129 |

  The q-values are corrected over MV's 99 per-task changes (9 changes, 11
  tasks). In the CSV, ARC-C's drop is 2.47 points through SFT, 2.30 through SFT
  and OT30K and 1.96 through SFT, DPO and OT30K.

### What this means for the paper's text

- **§3.2** says MV's language scores "remain at the base level, 56.0 after the
  reasoning update from either starting point against 55.0 through Phase 1".
  The logs give 55.68 and 55.74 after OT30K, against 54.96 for the 16K base and
  55.45 and 55.51 after SFT and DPO. So the claim holds for the reasoning
  update, and the one-point gap comes from rounding to whole numbers. For
  Phase 1 itself, the honest statement is "no detectable change" (+0.55,
  interval −0.07 to +1.18).
- **The appendix paragraph on Table 14** says "no benchmark drops by more than
  2 points (ARC-C, 0.40 against 0.42)". That holds only on the route through
  DPO, where ARC-C drops 1.96 points (41.64 to 39.68). On the routes without
  DPO it is false as an estimate, and the drop survives the correction: ARC-C
  drops 2.47 points through SFT alone (q 0.042) and 2.30 through SFT and OT30K
  (q 0.042), which prints as 0.39 against 0.42. ARC-E also drops on the DPO
  route, by 1.4 points, with an interval that excludes zero. A correct sentence
  names the route and both tasks, for example: "after SFT, DPO and OT30K, only
  ARC-C and ARC-E fall by more than half a point, by 2.0 and 1.4, and without
  DPO, ARC-C falls by 2.3 and ARC-E by 1.1".
- **Comma0.1.** The same paragraph says Comma0.1 after OT30K stays "close to
  the Comma0.1 base level of 0.50". That 0.50 is the 4K base (50.05). The 16K
  base, which the SFT model was trained from, scores 52.10. Against it,
  Comma0.1 loses 0.76 through Phase 1 [−1.37, −0.13], 0.80 through SFT and
  OT30K [−1.45, −0.18] and 1.89 through SFT, DPO and OT30K [−2.55, −1.28]. Most
  of the last loss is on LAMBADA (−7.9 [−8.9, −6.8]).
- **Nemotron-CC-HQ.** OT30K lowers its average by 0.53 after SFT
  [−1.00, −0.10] and by 0.31 after DPO [−0.78, +0.10].
- **FineWeb-Edu.** From its 16K base, the average moves by −0.46 through Phase
  1 [−1.05, +0.09] and by −0.18 through SFT and OT30K [−0.80, +0.46].
- **Table 14's MV-16K →OT30K row** differs from these logs in three cells
  (ARC-C 43 against 42.5, BoolQ 74 against 74.5, MMLU 42 against 41.3). The changes here use the logs.

### The three seeds agree

Averaging each question over the three seeds changes no average change by more
than 0.18 points. It moves five changes across q = 0.05, all of them changes of
0.62 points or less (MV's DPO step, three FineWeb-Edu steps and Nemotron-CC-HQ's
OT30K after SFT).

### The front-loading study (Table 28)

| Contrast on the 11-task average | 4K | 16K |
|---|---|---|
| Front-loading, MixtureVitae | +4.78 [+3.94, +5.60], q < 0.001 | +4.12 [+3.26, +4.99], q < 0.001 |
| Front-loading, Comma0.1 | +6.57 [+5.76, +7.39], q < 0.001 | +4.93 [+4.09, +5.75], q < 0.001 |
| Front-loading, Nemotron-CC-HQ | +0.38 [−0.42, +1.21], q 0.377 | +0.51 [−0.38, +1.34], q 0.258 |
| Front-loading, FineWeb-Edu | +6.10 [+5.31, +6.84], q < 0.001 | +5.27 [+4.51, +6.01], q < 0.001 |
| MV minus Nemotron-CC-HQ, both front-loaded | −8.73 [−9.65, −7.83] | −9.62 [−10.56, −8.65] |
| MV minus FineWeb-Edu | −7.02 [−7.99, −6.05] | −7.62 [−8.53, −6.73] |
| Comma0.1 minus Nemotron-CC-HQ | −7.94 [−8.97, −6.93] | −7.55 [−8.50, −6.63] |
| Comma0.1 minus FineWeb-Edu | −6.23 [−7.18, −5.26] | −5.54 [−6.42, −4.67] |
| Interaction, MV | +1.53 [+0.47, +2.50], q 0.010 | +1.22 [+0.15, +2.33], q 0.028 |
| Interaction, Comma0.1 | +3.33 [+2.33, +4.35], q 0.001 | +2.04 [+1.06, +3.09], q 0.001 |

Every difference of two datasets has q < 0.001.

- **Front-loading raises the language average of three datasets, not of Nemotron-CC-HQ.** Its change is half a point, and the interval includes zero at both lengths. So §3.2 can no longer say that front-loading "helps every dataset".
- **The permissive gap holds.** At matched front-loading, both permissive datasets trail both non-permissive ones by 5.5 to 9.6 points.
- **The interactions are positive** because Nemotron-CC-HQ gains so little. FineWeb-Edu gains more than MixtureVitae at both lengths, and more than Comma0.1 at 16K.
- **The three seeds agree.** Averaging each question over the three seeds moves no estimate by more than 0.38 points, and no contrast crosses q = 0.05.
