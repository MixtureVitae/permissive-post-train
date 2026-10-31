# Tables 7, 8, 9 and 15, the older experiments

`mvpt render-older-tables` renders the four tables of the older experiments:
Table 7 (the 100B mixture ablation), Table 8 (decontamination), Table 9 (the
0.4B comparison) and Table 15 (masking). It reads every value from the results
manifest ([manifest.md](manifest.md)).

```bash
mvpt render-older-tables
```

## Tables 7, 8 and 9 under Protocol A

Since 2026-10-18 these three tables use Protocol A, like Tables 1 and 6, and
Table 15 follows since 2026-10-19:

- **Records:** the per-question records of R18, received on 2026-10-18, for
  the 36 checkpoints of Tables 7 and 9 and of Table 8's rows without
  decontamination. Table 8's decontaminated rows are the main MV models, whose
  records are in R13. R18 has the same format and the same questions as R13.
- **Cells:** the mean over the three decoding seeds (42, 1234 and 2025), with
  the SD as a subscript.
- **Avg:** the 11-task suite average of each seed, then the mean and SD over
  the seeds.
- **Bold** marks the printed column maximum, ties included.

`tables/older_numbers.tex` defines `\pold{<key>}` for every number the text
quotes from the tables. A cell's SD has the key of its mean followed by `.sd`.
The keys are listed in `macros()` in `render/older_tables.py`.

## How Protocol A differs from the older evaluations

Earlier drafts of the revision printed the older single-run evaluations of
R6 in these tables. Seven tasks move little: on the 42 checkpoints of the three
tables, AIME24, AIME25, GSM8K, LiveCodeBench, JEEBench and GPQA-Diamond move
by less than 0.3 points on average, and AMC23 by 0.4. These tasks have few
questions, so single cells move by up to 10 points (AMC23) when the seeds
disagree. MATH500 rises by 1.8 points on average. Three tasks move a lot:

- **HumanEval** rises by 16.6 points on average. Protocol A scores the Python
  tests alone, where the older value was the mean of the Python and shell
  tests. Against the older Python value alone, the rise is 7.3 points, all of
  it in Table 7's 16K rows and Table 9's MV (0.4B) rows. Their older Python
  runs scored 0.2 to 19.5, where Protocol A gives 10.4 to 41.3. Elsewhere the
  older Python value is within 2 points of Protocol A.
- **MBPP** rises by 11.3 points on average at 16K and by 0.5 at 4K. The older
  MBPP runs of Table 7's 16K checkpoints scored 1.5 to 8.2, where Protocol A
  gives 12.3 to 36.4. The 0.4B MV rows rise by up to 10.9. Table 8's
  checkpoints move by at most 1.3.
- **IFEval** rises by 5.5 points on average. Protocol A scores it at the
  instruction level with the strict checker (R21, since 2026-10-24), where
  the older value was the mean of the prompt level and the instruction level.
  Against the older instruction level alone, it is 0.7 points lower. With the
  loose checker, which these tables used until 2026-10-24, the rise was 7.7
  points, and 1.5 against the older instruction level.

So the row averages rise by up to 7 points. Table 9's Baguettotron-16k rows now
have LiveCodeBench values, which the older evaluations lacked, so they average
all eleven tasks.

Table 8's decontaminated 16K rows now equal Table 1's MV rows at 16K (30.0,
27.1 and 27.4), since both are R13.

## Table 15 under Protocol A

Table 15 compares Phase 1 with and without masking the prompt tokens in the
loss. Since 2026-10-19 both arms use Protocol A:

- **Rows with masking:** the records of R19, received on 2026-10-19, for the
  four masked MV checkpoints at 4K and 16K after SFT and after DPO. They have
  the same format and the same questions as R13. The records name
  `ali-elganzory/*-masked-laion-eval`, copies of the `laion/*-masked` models
  uploaded on 2026-10-18, whose `model.safetensors` has the same SHA-256 on
  Hugging Face. The manifest keys them by the `laion` repos
  (`RECORD_IDENTITY` in `data/protocol_a.py`), which are the masked rows of
  Table 16, so both tables print the same models. The other masked set,
  `ali-elganzory/*-masked` from July 2026, has different weights and is not used.
- **Rows without masking:** Table 8's decontaminated SFT and DPO rows, the main
  MV models of R13.
- **Cells and Avg** as in Tables 7 to 9, without bold.

`\pold{mask.<masked|unmasked>.<ctx>.<stage>.<task>}` gives each cell, with
`.sd` for its SD. `\pold{mask.change.<ctx>.<stage>.<task>}` is the masked row
minus the row without masking, from the printed means, and
`\pold{mask.change.<task>.min}` and `.max` its range over the four stages.

**What changes against earlier drafts.** Earlier drafts printed the masked
rows from an earlier version of the table, because no record held them, and the rows without masking from the older evaluations. Both arms
now come from records and use Protocol A's scoring. Masking raises the suite
average by 0.5 to 1.5 points at every stage. GSM8K, HumanEval, MBPP and IFEval
are higher with masking at every stage, by up to 5.9, 9.4, 2.7 and 4.8
points, while AMC23 and MATH500 are lower at three of the four stages. These
are the values with IFEval's strict checker (2026-10-24). With the loose one,
IFEval's largest gain was 4.7 points.

## The older evaluations

Until 2026-10-18, Tables 7, 8 and 9 printed the older evaluations of
`logs.zip`, and so did Table 15's rows without masking until 2026-10-19. The
manifest still holds these evaluations (protocol `older_evalchemy`), scored by
the older setup's rules: `accuracy_avg` (`accuracy` for MATH500), GSM8K with
the flexible-extract answer parser, HumanEval as the mean of its Python and
shell scores, and IFEval as the mean of its prompt-level and instruction-level
scores. The paper's earlier script (`tables.py` in Ali Elganzory's evalchemy
fork) took each score field at its highest over a checkpoint's runs and
rounded twice. This repo replaced that with the mean of
the unbroken runs, rounded once, which left out six HumanEval runs of Tables 7
and 9 written late in one batch on 2026-04-15. Those six runs no longer enter
any table. Table 27 used the same evaluations through the older pipeline of
this repo, and now resamples R18 ([statistics_appendix.md](statistics_appendix.md)).
