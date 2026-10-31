# The statistics appendix (Tables 26 and 27)

`mvpt render-statistics` writes two tables of the paper's statistics appendix,
Appendix F "Statistical methodology and uncertainty" (`appendix/statistics.tex`),
and the `\pstat` numbers its text quotes, into `results/statistics/`. Since
-17, Table 26 is in F.1, Table 27 in F.2 and the power figures in F.5. The
Phase 2 tables of F.4 come from `mvpt render-phase2-analyses`
([phase2_analyses.md](phase2_analyses.md)).

| File | What it is |
|---|---|
| `tables/stat_matched_compute.tex` | Table 26, MV against the matched-compute references, under Protocol A |
| `tables/stat_mixture_ablation.tex` | Table 27, the 100B mixture ablation, under Protocol A (R18) |
| `tables/statistics_numbers.tex` | `\pstat{<key>}` macros for every number the appendix text quotes from these tables and the power analysis |
| `power.csv` | The minimum detectable effect of each benchmark, with the simulation's settings |
| `preview.pdf` | Both tables, compiled |

## Why it changed

The paper had two statistics appendices that described different procedures.
`appendix/statistics.tex` describes the
Protocol A method behind the front-loading study. `appendix/stat_sig.tex`
described an older one: single runs, a 12-task suite with a pipeline bug that
was fixed in May 2026, 10,000 resamples and Newcombe intervals for the tasks
without per-item records. Its two tables came from that older setup,
although Table 26 supports the Phase 1 claims, which use Protocol A.

Now Table 26 uses the Protocol A records and the method of
`appendix/statistics.tex`. Table 27 followed on 2026-10-18, when the Protocol A
records of its checkpoints arrived (R18). -17 merges the two appendices into
one, `appendix/statistics.tex` (pull request 38), so `stat_sig.tex` is gone.

## Table 26: matched-compute contrasts

- **Records:** the Protocol A per-question records (R13,
  `data/workshop/canonical_reasoning_per_sample.csv`), the same records as
  Tables 1, 6 and 28.
- **Checkpoints:** MV-16K, Comma0.1-16K and Nemotron-CC-HQ-16K after Tulu3 SFT,
  and after DPO for the suite-average rows. These are the rows of Table 1.
- **Method** (`mv_post_train/stats/pairwise_a.py`):
  - each item is scored as its mean over the three decoding seeds.
  - each replicate resamples the items of each benchmark, with the same draw for every checkpoint, so the comparison stays paired.
  - the interval is the 2.5th to 97.5th percentile of the replicates, and the p-value is the two-sided bootstrap p-value.
  - the seed is that of Table 28 (`stats/factorial.py`).
- **Replicates:** 20,000, where Table 28 uses 2,000. With 2,000 the smallest
  p-value is 1/2,000, and a column's q-values then depend on how many of its
  tests sit at that floor. In the first version, MATH500 printed q = 0.001
  against Comma0.1 and q < 0.001 against Nemotron-CC-HQ, although no replicate
  crossed zero in either. With 20,000, every such test prints q < 0.001. The
  caption and `appendix/statistics.tex` say so.
- **Reweighted suite averages:** the last block gives MV's lead after SFT with
  the suite average reweighted by domain, the views of Table 29: the four
  domains of Table 23 weighted equally, and the average without each domain in
  turn (`pairwise_a.domain_views`). Each view weights the same replicates
  differently, so the other rows do not change. The authors' reply to the
  reviewers promised this comparison against the strongest reference.
  Comma0.1 is the strongest matched-compute reference after SFT at 16K.
- **Corrections:** one Benjamini-Hochberg family per column for the per-task
  rows, one for the four suite-average rows, and one for the ten reweighted
  rows. A benchmark on which both models score zero on every item is not
  tested, prints a dash, and is left out of its family.
- **q-values** are printed as Table 28 prints them.

## Table 27: the 100B mixture ablation

- **Records:** the Protocol A records of the older experiments (R18), for the
  18 checkpoints of Table 7. Their labels call every 100B variant "MV", so
  `render/statistics.py` tells them apart by name (`ABLATION_MODELS`), and it
  checks them against the results manifest's per-seed scores.
- **Method:** that of Table 26. Each item is scored as its mean over the three
  decoding seeds, and the paired item-level bootstrap resamples items within
  each benchmark, with 20,000 replicates.
- **Rows:** the 11-task suite average of each ablation minus the full 100B
  mixture (MV-100) at the six settings (base, SFT and DPO at 4K and 16K), and
  each task after SFT at 16K.
- **Corrections:** one Benjamini-Hochberg family per column for the per-task
  rows, and one for the twelve suite-average rows. A benchmark on which both
  models score zero on every item is not tested, prints a dash, and is left
  out of its family.
- **Before 2026-10-18** the table used the older single-run evaluations of
  `logs.zip` through the older pipeline, with other tests for the tasks that
  had no per-item records. The Protocol A records of R18 replaced them.

## Power

`stats.core.mde_paired_binomial` finds the smallest difference that a paired
sign-flip test detects with 80% power at a two-sided 5% level, by simulation,
with a correlation of 0.3 between the two models' 0/1 outcomes on an item. It
is computed for each benchmark at its Protocol A size and at the mean score of
the two references.
- **Baseline:** the figure depends on it. On AIME it is 26 points at a baseline of zero and 28 points at 6.7%.
- **A different test from Table 26's:** the simulation has one outcome per item and a sign-flip test, while Table 26 averages each item over three seeds and uses the bootstrap. So the figures describe a single run of each model. The appendix text says so, and that a smaller difference can still be significant, as AMC23 is (+12.5 against an MDE of 20).
- **Run time and cache:** the simulation takes about six minutes, so `power.csv` records every setting (correlation, level, power, trials, permutations, seed). A row is reused only when the benchmark, size, baseline and every setting match.

## Text macros

`statistics_numbers.tex` defines `\pstat{<key>}`, in the same way as
`phase2_numbers.tex` defines `\ptwo`. A key that does not exist prints in bold
between question marks. Task keys are the lower-case abbreviations (`a24`,
`amc`, `m500`, `gsm8k`, `he`, `lcb`, `mbpp`, `gpqa`, `jee`, `ife`).

| Key | Value |
|---|---|
| `mc.<ref>.<task>` and `.lo`, `.hi`, `.q` | Table 26, MV minus `comma` or `nemotron` after SFT |
| `mc.<ref>.<task>.level`, `mc.mv.<task>.level` | The two scores, in percent |
| `mc.<stage>.<ref>.suite` and `.lo`, `.hi`, `.q`, `.level` | The suite-average rows, `sft` or `dpo` |
| `abl.<ablation>.<stage><ctx>.suite` and `.lo`, `.hi` | Table 27's suite rows, `noweb` or `noinst`, for example `abl.noweb.base4k.suite` |
| `abl.<ablation>.suite.min`, `.max` | The range of those six rows |
| `abl.<ablation>.<task>` and `.lo`, `.hi`, `.q` | Table 27's per-task rows |
| `mde.<task>`, `mde.<task>.baseline` | The minimum detectable effect and the baseline it assumes |

## What changes with Table 7's rules (2026-10-17)

Table 27 used the latest complete run of every task and strict GSM8K. With
Table 7's rules:

- **HumanEval after SFT at 16K:** MV-noinstruct minus MV-100 is −7.5
  [−11.5, −3.6] (was −5.6), and MV-noweb minus MV-100 is −2.5 (was −0.3).
- **The 16K DPO suite row of MV-noinstruct** is −3.6 (was −4.1), because its
  broken HumanEval run, which scores 0.0, is left out.
- **Every other suite row** moves by at most 0.2 points.
- **Every claim in the appendix text still holds.**
  - MV-noweb is never significantly below MV-100, and its interval excludes
    zero at the same three stages.
  - After SFT at 16K, MV-noweb is ahead only on MATH500 and LiveCodeBench.
  - MV-noinstruct's loss is concentrated on GSM8K, HumanEval and MBPP, and
    its IFEval difference does not survive the correction (q = 0.078).

## What changes with IFEval's strict checker (2026-10-24)

Protocol A IFEval now comes from R21, scored with the strict checker
(`docs/inputs.md`). It is 2.3 points lower on average, and no other task
changes.

- **Table 26's IFEval rows:** MV minus Comma0.1 after SFT is +4.7
  [+0.7, +8.7], q = 0.031 (was +6.2, q = 0.003), and MV minus
  Nemotron-CC-HQ +9.0 (was +9.4). Both still favour MV.
- **Table 27's IFEval row:** MV-noinstruct minus MV-100 after SFT at 16K is
  −3.4 [−7.1, +0.3], q = 0.100 (was −4.1, q = 0.039). It no longer survives
  the correction, so the appendix sentence that listed it with MATH500 and
  JEEBench now says that it does not. MATH500 and JEEBench still do
  (q < 0.001 and 0.028).
- **The suite rows** move by at most 0.4 points, and every other claim of the
  appendix text still holds.

## What changes against the paper's -6 revision

- **Table 26** moves a lot, because both the records and the suite change. MV
  minus Comma0.1 after SFT:
  - MATH500 is +32.5 instead of +29.6.
  - HumanEval is +37.6 instead of +28.4 (Python problems only, 164 of them).
  - MBPP is +15.2 instead of +37.2.
  AMC23 is now significant against both references (q of 0.015 and 0.003). On
  the suite average after SFT, MV scores 27.3 against 12.0 and 8.4, where the
  old text gave 27.7 against 10.9 and 9.5 from the 12-task suite. The table
  now gives a paired interval for each difference, instead of a separate
  interval for each model.
- **Table 27's suite rows** use the 11-task suite. For example, MV-noweb at the
  4K base is +9.0 instead of +10.4, and MV-noinstruct's losses range from 3.8
  to 9.5 points instead of 3.2 to 9.2. MV-noweb ranges from −0.2 to +9.0, so
  it is no longer "at least as strong as MV-100 at every stage": at the 16K
  base it is 0.2 points lower, well inside its interval. The appendix text now
  says it is never significantly below MV-100.
- **Table 27's HumanEval row**, now from the latest runs, is −0.3 and −5.6
  instead of −1.2 and −11.0. IFEval is one row instead of two. Both are as
  Table 7 defines them.
- **Power:** 26 points on AIME and 20 on AMC23 at the references' scores,
  where the old text gave 27 and 25. `appendix/statistics.tex` gave 28 and 30
  for the same benchmarks. It now quotes the same macros.

## Limitations

- **The minimum detectable effect rests on assumptions:** the correlation
  between the two models' outcomes, the baseline score, and a single-run
  design.
