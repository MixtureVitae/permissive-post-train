# Phase 2 analyses

Four analyses of the Phase 2 results (N1 to N4), and the comparisons of
MixtureVitae with each reference that the authors' reply to the reviewers
promised for the camera-ready, recomputed on the current records.
Each one measures something a headline claim of the paper needs before it can
be kept or reworded. This page says what each analysis computes, how, and what
the first render shows.

```bash
mvpt measure-phase2-answers --responses <eval_responses.parquet>   # once, writes input R14-lengths
mvpt render-phase2-analyses                                        # or: mvpt build phase2-analyses
```

`render-phase2-analyses` writes everything under `results/phase2_analyses/`:

| File | Item | What it holds |
|---|---|---|
| `tables/phase2_endpoints.tex` | Table 2 | One row per OT30K update: each task's endpoint, and the suite average before and after with the change and its 95% interval, the estimate of `phase2_change_intervals.tex` |
| `tables/phase2_change_intervals.tex` | N3 | Table 2's changes on the suite average, with 95% intervals and q values |
| `tables/ot3_exclusion_gains.tex` | N1 | The OT3 exclusion's two gains and their difference, on the suite average and on MATH500 |
| `tables/phase2_domain_changes.tex` | N2 | Every Phase 2 run's change in each domain, with intervals |
| `tables/phase2_endpoint_gaps.tex` | reply | MV minus each reference on pooled mathematics and contest mathematics, before and after OT30K, the widening between them, and IFEval after OT30K |
| `tables/phase2_leave_one_out.tex` | reply | MV minus each reference after OT30K on pooled mathematics, with each task left out |
| `tables/phase2_cutoff.tex` | N4 | The share of answers cut off at the token limit, per checkpoint and task |
| `tables/phase2_no_answer.tex` | N4 | The share of answers in which the grader finds no answer |
| `tables/phase2_analyses_numbers.tex` | all | `\ptwoan` macros for the text |
| `phase2_contrasts.csv` | N1 to N3, reply | Every contrast, unrounded, with its interval, p and q values, and the code bound's share |
| `answer_endings.csv` | N4 | The shares for every checkpoint and task of the records |
| `length_shares.csv` | N4 | How many re-encoded answers end in each 1% of the token limit, from 80% up |
| `preview.pdf` | all | The seven tables, compiled |

The statistics appendix includes the first five tables and the macro file
from `paper_stats_appendix_update/` (-17) on, and the two tables of N4 from
`paper_claims_update/` on, so all seven are in `targets.PAPER_FILES`.
That update also quotes the macros in the main text.

## Which changes

Every change is against the checkpoint its run was trained from, the parent
the results manifest records, as in the Phase 2 tables
([phase2_tables.md](phase2_tables.md)). A change marked † has a parent known
only from the checkpoint names.

- **N3** takes the eleven changes Table 2 prints: five for MixtureVitae and the OT30K update of each of the six references.
- **N1** takes Table 3's two OT30K runs, MixtureVitae with and without OpenThoughts3 in pretraining, each from its own 16K checkpoint. It gives each gain and the difference of the two gains, (with OT3, after minus before) minus (without OT3, after minus before), on the suite average and on MATH500. The two 16K checkpoints differ before the update, so the difference of the gains is not the difference of the two endpoints that the abstract quotes.
- **N2** takes every run of Tables 2, 3, 4 and 32: 33 runs. For each one it gives the change in the mean of each domain's tasks: mathematics (AIME24, AIME25, AMC23, MATH500, GSM8K), code (HumanEval, LiveCodeBench, MBPP), science and question answering (GPQA-Diamond, JEEBench), instruction following (IFEval), and every task but mathematics. SmolLM2's AceReason run has no AIME24, AIME25 and AMC23 results, so its mathematics cell is empty.
- **N4** takes the checkpoints of Tables 2 and 3.
- **MV against each reference** pairs MixtureVitae with each of the six references at the same two points: their Tulu3 SFT checkpoints, and after OT30K from them. It gives MV minus the reference on the macro average of the five mathematics tasks and on the macro average of the three contest tasks (AIME24, AIME25, AMC23, 100 problems), at both points, and the widening, (MV after minus MV before) minus (reference after minus reference before), which is the difference-of-differences interval the reply promised. It also gives the IFEval gap after OT30K, and the pooled mathematics gap after OT30K with each task left out in turn. The leave-one-out intervals are diagnostics and are not corrected.

## How the intervals are computed

`stats/phase2_suite.py` computes them, from the per-question records of R14
(inputs R14-meta and R14-ifeval).

- **Each question's score** is its mean over the three runs. A task's score is the mean over its questions. IFEval's score is the share of instructions followed, so a prompt counts with its number of instructions.
- **A replicate** resamples the questions of each task with replacement, with the same draw for every checkpoint of the change, and recomputes the change on the mean of the tasks. There are 2,000 replicates, with the seed of the Protocol A contrasts. The interval is the 2.5th to 97.5th percentile of the replicates.
- **Each family of changes** gets one Benjamini-Hochberg correction: Table 2's eleven changes, N1's six results, the changes of each domain, and MV against the references in each of mathematics, contest mathematics and IFEval.
- **The records must give the manifest's scores.** For every checkpoint and task, the per-run scores of the per-question records must be the manifest's, or the command stops. With all three runs, every estimate equals the manifest's change.
- **JEEBench questions are told apart by `example_idx`.** JEEBench's `example_id` names a subject and is shared by many questions. For the other tasks the two columns name the same question.

### The code tasks

The records have no per-question verdicts for HumanEval, LiveCodeBench and
MBPP. Their released scores are there, but not which questions each run
solved. So a code task's part of each replicate is its observed change plus a
normal draw with standard deviation

  sum over the change's checkpoints of sqrt(p (1 - p) / n),

with p the checkpoint's score as a share and n the number of questions (164,
369 and 500). This is an upper bound on what resampling the questions would
give: a question's score lies between 0 and 1, so its variance is at most
p (1 - p), and the standard deviation of a sum is at most the sum of the
standard deviations, whatever the correlation between the checkpoints. So the
code tasks make each interval at least as wide as per-question verdicts
would, never narrower. This was decided on 2026-10-24.

The bound matters most where the code scores are near the middle, as for
MixtureVitae, SmolLM2 and the Qwen models. `phase2_contrasts.csv` gives, for
every change, the bound's share of the replicates' standard deviation
(`code_sd_share`). Per-question code verdicts from the evaluation runs would
replace the bound and narrow those intervals.

### What the intervals leave out

Like Table 4's, the intervals cover the choice of questions and the sampling
of answers. They do not cover differences between training runs, which these
results cannot estimate.

They also take the three runs as three samples from the same model. But
run 0 repeats an earlier generation in some cells, among them the MATH500 cells of MixtureVitae's 16K
checkpoint and SFT checkpoint and of the 16K checkpoint without OT3, which are
starting points of N1 and N3. So N1 and N3 are also given from runs 1 and 2
only, for every checkpoint and task. The code tasks then use the mean of
their released per-run scores 1 and 2.

## Cut-off answers and answers without one (N4)

**Measuring the answers.** `mvpt measure-phase2-answers` reads the whole
answer file of the pinned revision (input R14-responses, 10 GB) one batch at a
time and writes one row per sampled answer, without its text (input
R14-lengths). For each answer it keeps the length in bytes, and whether the
grader's extracted answer and the extracted code are empty. Answers that are
at least 90% of the token limit long in bytes are re-encoded with their
checkpoint's own tokenizer: the `tokenizer.json` of its Hugging Face repo, at
the revision the checkpoint snapshot (input S1) recorded, checked against the
hash Hugging Face lists for it. Every tokenizer here is a byte-level BPE, so
an answer has at most as many tokens as bytes, and a shorter answer cannot
have reached the limit. One checkpoint has no tokenizer in the manifest,
SmolLM2-1.7B-Instruct-16k, which no table uses. Its answers have no token
counts. The command takes about two hours on two CPUs, most of it to
re-encode the long answers. It keeps each row group's rows in
`answer_lengths.parts/` until the file is written, so a run that is stopped
starts again where it stopped.

**Cut off.** An answer counts as cut off when its re-encoded length is at
least 95% of the limit (`LIMIT_SHARE` in `stats/answer_endings.py`), because
answers that degenerate into repetition re-encode up to a few percent short
of the limit they reached. The last paragraph of this document gives the
evidence. The limit is the record's `max_tokens`: 32,000 tokens for AIME24, AIME25, AMC23,
MATH500, GPQA-Diamond and JEEBench, and 16,000 for GSM8K, the code tasks and
IFEval.

**No answer found.** For AIME24, AIME25, AMC23, MATH500 and JEEBench, the
share of answers whose extracted answer (`model_answer`) is empty. Such an
answer is scored wrong. GSM8K and GPQA-Diamond are left out, because their
graders take an answer from every response. For the code tasks, the share with
no code block (`model_answer_extracted` empty), for which the checker runs the
whole answer instead. IFEval has no answer to extract. A cell is left empty
when its records do not hold the column: on an answer task, when an answer
with an empty extracted answer is scored correct or every answer's is empty,
and on a code task, when no answer has its extracted code. The second happens
in two cells, MixtureVitae's Tulu3 SFT checkpoint on HumanEval and MBPP, whose
DPO checkpoint has the column for every answer.

**Ranges for the text.** The macro file also gives the `.min` and `.max` of
the shares and changes that the text quotes as a range, for example
`cutoff.matched.sft-ot3.but-gsm8k` (the three matched-compute references after
OT30K, every task but GSM8K) and `range.mv.from-sft.math` (MixtureVitae's four
reasoning runs from SFT). `macros()` in `render/phase2_analyses.py` lists them.

## What the first render shows

From the records of revision `e820fb34`, with the code tasks' bound. Every
number is in `phase2_contrasts.csv` and the macro file.

**N3, Table 2's changes.** The three matched-compute references decline, and
each decline's interval excludes zero: FineWeb-Edu −1.3 [−2.3, −0.5],
Nemotron-CC-HQ −2.1 [−3.1, −1.2] and Comma0.1 −4.0 [−5.1, −3.0]. So "every
reference declines" holds for the matched-compute references. Of the external
references, SmolLM2 (−3.8) and Qwen3 (−17.2) decline, and Qwen2.5's −0.7
[−2.6, +1.3] cannot be told from zero. MixtureVitae's gain from SFT is +10.7
[+8.2, +13.2], from its 16K checkpoint +7.6 [+5.2, +10.0], and from DPO +0.5
[−1.5, +2.7], which cannot be told from zero. With runs 1 and 2 only, every
conclusion stays the same. MixtureVitae's gain from its 16K checkpoint falls to
+5.7 [+3.3, +8.0].

**N1, the OT3 exclusion.** With OpenThoughts3 in pretraining the model gains
+7.6 [+5.2, +10.0] on the suite average, without it +1.7 [−0.2, +3.6], which
cannot be told from zero. The difference of the two gains is +5.9 [+2.2, +9.9]
points. On MATH500 the gains are +23.3 [+20.3, +26.0] and +8.7 [+6.0, +11.6],
and their difference is +14.5 [+10.9, +18.4]. The difference depends on run
0: with runs 1 and 2 only, it is +4.0 [+0.1, +8.0] on the suite average
and +5.6 [+1.4, +10.3] on MATH500, because MixtureVitae's 16K checkpoint scores
53.0 on MATH500 in runs 1 and 2, against 46.5 over all three. The suite
difference is the code bound's most affected result: the bound makes up most
of its spread (`code_sd_share` 0.87).

**N2, the domains.** Every MixtureVitae run from SFT gains in mathematics, by
13.5 to 25.4 points, and loses in IFEval, by 9.9 to 12.8 points. From DPO, the
three other datasets do the same, but OpenThoughts3 gains only +1.4 [−0.8,
+3.8] in mathematics. In
code the four datasets differ: AceReason and Nemotron-Cascade gain about 7
points from SFT, OpenThoughts3 +2.4 [−3.8, +8.6], and Superior-Reasoning loses
8.2. IFEval falls under 29 of the 31 reasoning runs, with an interval that
excludes zero. The two exceptions are Superior-Reasoning on FineWeb-Edu and
Nemotron-CC-HQ. From DPO, OpenThoughts3 lowers IFEval by 5.3, and the other
three datasets by 10.7 to 13.4. The references' OpenThoughts3 runs lose in
almost every domain, and Qwen2.5 gains +10.4 in mathematics while it loses 17.6
in code and 14.3 in IFEval.

**MV against each reference.** At the Tulu3 SFT checkpoints MixtureVitae leads
the three matched-compute references on pooled mathematics by 13.7 to 15.1
points, and after OT30K by 35.9 to 38.2. The widening is 22.0 to 23.7 points,
with every interval above zero. On contest mathematics the gaps are 1.5 to 2.8
points before (FineWeb-Edu's and SmolLM2's are significant) and 24.3 to 24.8
after. Against Qwen2.5 the mathematics gap grows from +0.4 [−1.1, +2.0] to
+11.9, against Qwen3 from −3.6 to +31.2. All thirty leave-one-out gaps lie
above zero, and the smallest is Qwen2.5 without MATH500, +8.4 [+4.2, +12.9].
On IFEval after OT30K, MixtureVitae leads the matched-compute references by
5.3 to 8.7 points and is level with SmolLM2 (+0.4 [−2.7, +3.6]).

These differ from the reply of 2026-08-01, which was computed on the answer
file released then, mostly one greedy answer per question. Two of its
statements no longer hold on the current records: the contest-mathematics gap
before the update is not zero for every reference, and MixtureVitae does not
trail SmolLM2 on IFEval after OT30K.

**N4, how the answers end.** After OT30K, most answers of the matched-compute
references are cut off at the token limit: 80 to 100% on every task but GSM8K
for FineWeb-Edu and Nemotron-CC-HQ (FineWeb-Edu's IFEval, 79.9%, is the
lowest), and 68 to 100% for Comma0.1. Their Phase 2
declines are therefore mostly answers that never finish, not wrong answers
that finish. MixtureVitae's answers after OT30K are cut off less often but
still often on the hardest tasks: 70% on AIME24, 81% on LiveCodeBench and 16%
on MATH500. Qwen3 is the exception: almost none of its answers after OT30K are
cut off, but the grader finds no final answer in 76 to 96% of its mathematics
answers, which fits a mismatch of format rather than of length. Between 80%
and 92% of the limit, about 390 (16,000-token tasks) and 180 (32,000-token
tasks) answers end in each 1% of it, and from 95% the counts rise to tens of
thousands, so the 95% threshold counts about 1,900 and 900 finished answers as
cut off, out of about 168,000 and 43,000, and misses a few hundred cut-off
ones (`length_shares.csv`). Two checkpoints hold no extracted answer on the
contest tasks, the 16K checkpoint without OT3 and Qwen2.5's Tulu3 SFT
checkpoint: every answer's is empty, and some of them are scored correct. The
table leaves those cells empty.
