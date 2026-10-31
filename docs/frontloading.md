# Front-loading study with Comma0.1

The revision adds Comma0.1 to the 300B front-loading study as a fourth web
substrate, and as a second permissive one next to MixtureVitae. This page
explains where Comma0.1's two arms come from, what the command writes, and
what the first render shows.

```bash
mvpt render-frontloading
```

## The FineWeb-Edu 16K base without front-loading

R13 holds the August 16K model of this arm, but its SFT and DPO models were
trained from the April 16K model, a second run of the same extension. Since 2026-10-23 the study uses the April
model: its Protocol A records are input R20, which replaces the August model
in its cell (`protocol_a.REPLACEMENTS`), and its language scores are R8's run
of the April model, of May 2026 (`suites.LANGUAGE_REPLACEMENTS`). The 16K base
effect of front-loading on FineWeb-Edu is now +18.6 points, where it was +19.0
against the August model.

## Where the Comma0.1 arms come from

**Front-loaded.** The `comma-mv_reasoning` models: the 4K base, its 16K
extension, and SFT and DPO at both lengths. They are open-sci-ref v0.02 runs,
like the Nemotron-CC-HQ and FineWeb-Edu arms. Their Protocol A records are
input R15 ([inputs.md](inputs.md)).

**Without front-loading.** No open-sci-ref v0.02 run of Comma0.1 without
front-loading exists. The arm is the Comma0.1 model of Phase 1:
`ontocord/1.7b-Comma0.1-300BT` (uploaded 2025-09-17) and the 16K, SFT and DPO
models trained from it. This was decided on 2026-10-13, for these reasons:

- Its model files record the open-sci-ref configuration it was trained with:
  1.7B parameters, GPT-NeoX tokenizer, 300B tokens, batch size 1024, 4K
  context, rope base 100000, WSD schedule, learning rate 4e-3 and 25,000
  warmup steps. Its architecture settings are identical to the v0.02 models.
- Its 16K extension uses the same YaRN settings (factor 4) as the other arms,
  and its 16K, SFT and DPO models name their parents in their model cards.
- It was trained on JSC, so the data corruption on Leonardo that led to the
  v0.02 re-runs did not affect it. The authors kept it for the v0.02
  comparisons for that reason.

What this does not settle: it is a separate run, trained about a year before
the front-loaded one, and the v0.02 run does not record which version of the
Comma data it used.

**MixtureVitae has the same caveat.** Its front-loaded arm is the main
MixtureVitae model (`1.7b-MixtureVitae-300BT-v1-decontaminated`, January
2026), not the v0.02 MixtureVitae run, whose weights differ. Both caveats are
in the table captions and in the manifest, under
`designs.front_loading_study`, which also lists the checkpoint behind every
cell.

**Language scores.** The command reads every score of the study from the
results manifest ([manifest.md](manifest.md)), which reads them as below. The
language file of the artifacts repository covers only the three original
substrates. Comma0.1's language scores come from Harsh Raj's lm-eval results
files, in two zips in the project bucket:

- **Front-loaded:** input R23, `language_eval_all_models_3seed_2026-10-26.zip`
  (3.5 GB), the seed-1234 runs of 2026-10-26, which have per-question files.
  Their scores equal those of R16, `language_eval_all_models_2026-10-16.zip`,
  which the study read until 2026-10-27. The other front-loaded cells and the
  cells without front-loading that R8 does not hold read R23 too, so that the
  intervals of Table 28 come from the runs the tables print
  ([language_intervals.md](language_intervals.md)). Their scores equal
  R13-language's, apart from one OpenBookQA question of the front-loaded
  FineWeb-Edu 16K model (`suites.R13_LANGUAGE_REPLACED`).
- **Without front-loading:** input R8, `language_eval-logs.zip` (2.8 GB), the May 2026
  rerun, `default` folder, in `data/language_logs/`. Hashing the whole zip
  takes minutes, so the manifest build checks a digest of the results files
  it reads from it instead.

Every score must use the shot count and metric that the language file uses for
its task, and the prompt-example seed 1234 (the `default` folder). The command
stops otherwise. This matters, because 23 other runs in the October zip used 0
shots on every task. Some language checkpoints have different repository
names from the reasoning checkpoints of the same cell, but each pair has
identical weights on Hugging Face, and the command checks that each cell's
reasoning records use the checkpoint listed here:

| Cell | Language checkpoint | Reasoning checkpoint |
|---|---|---|
| Front-loaded, 4K | `open-sci/open-sci-ref-v0.02-1.7b-comma-mv_reasoning-300B-4096` | `ali-elganzory/open-sci-ref-v0.02-1.7b-comma-mv_reasoning-300B-4096` |
| Front-loaded, 16K | the same, `-longsft_16k` | the same, `-longsft_16k` |
| Without, 4K | `ontocord/1.7b-Comma0.1-300BT` | `ali-elganzory/1.7b-Comma0.1-300BT-WithChatTemplate` |
| Without, 16K | `ontocord/1.7b-Comma0.1-300BT-longsft_16k` | `ali-elganzory/1.7b-Comma0.1-300BT-longsft_16k` |

## What it writes

| File | Paper | Contents |
|---|---|---|
| `results/frontloading/tables/factorial_summary.tex` | Table 5 | Suite averages and the front-loading effect, plus language |
| `results/frontloading/tables/contrasts.tex` | Table 28 | Every contrast with its 95% interval and $q$-value |
| `results/frontloading/tables/domain_robustness.tex` | Table 29 | The 16K SFT effects under other ways of averaging |
| `results/frontloading/tables/factorial_perbenchmark.tex` | Table 30 | Every task, every checkpoint |
| `results/frontloading/tables/factorial_language.tex` | Table 31 | The language suite |
| `results/frontloading/tables/frontloading_numbers.tex` | the text | `\pfl{<key>}` macros for every number the text quotes, printed as Tables 5 and 28 print them. The keys are listed in `macros()` in `render/frontloading.py` |
| `results/frontloading/figures/*.pdf` | Figures 3, 4 and 5 | |
| `results/frontloading/preview.pdf` | none | All of the above, compiled with `pdflatex` |
| `artifacts/frontloading/*.csv` | none | The summaries behind the tables |

The files keep the paper's names and labels. They are committed, like the
manifest, so any change to a number shows up in review. `mvpt render-workshop`
still renders the workshop paper's three-substrate study, and its output is
byte-for-byte the same as before.

Figure 3's front-loading labels are the difference of the two printed values,
as Table 5's $\Delta$ row is, so every label matches the table. The workshop
paper's figure subtracts the unrounded means instead, which puts two of its
labels 0.1 point away from its table (MixtureVitae and FineWeb-Edu at 4K).
`render-workshop` keeps that behaviour, so it still reproduces the workshop
figure.

## Contrasts

- **Front-loading:** with minus without front-loading, for all four substrates.
- **Permissiveness:** each permissive substrate (MixtureVitae, Comma0.1)
  minus each non-permissive one (Nemotron-CC-HQ, FineWeb-Edu), all
  front-loaded.
- **Interaction:** each permissive substrate's front-loading effect minus the
  mean effect of the non-permissive ones.

The bootstrap resamples questions only, so the 36 contrasts of the workshop
version keep exactly the same estimates and intervals. Their $q$-values are
adjusted over larger families now. 15 of them move, and none crosses 0.05.

## What the first render shows (2026-10-13)

- **Comma0.1's front-loading effect** is +20.9, +14.6 and +15.6 points at 4K
  (base, SFT, DPO) and +16.7, +13.9 and +13.8 at 16K. That is in the range of
  Nemotron-CC-HQ and FineWeb-Edu. Its interaction with the non-permissive
  mean is not significant, except at the 16K base, where its effect is 2.4
  points smaller (q = 0.026).
- **Front-loaded Comma0.1 is the strongest substrate at 4K.** It scores 28.3,
  26.0 and 27.4, against 17.7, 23.0 and 23.9 for MixtureVitae, and it is 3.1
  to 5.2 points ahead of both non-permissive substrates at every 4K stage
  (q < 0.001).
- **At 16K, MixtureVitae leads.** Front-loaded MixtureVitae is 3.4 to 5.7
  points ahead of both non-permissive substrates. Comma0.1 is level with them
  at the base and 3.3 to 4.0 points ahead after SFT and DPO.

## The two permissive substrates against each other (added 2026-10-26)

`stats/factorial.py` (`permissive_pair_contrasts`) gives front-loaded
MixtureVitae minus front-loaded Comma0.1 on the suite average, per context
and stage. It draws the replicates as `factorial_contrasts` does, in the same
order from the same seed, so the intervals rest on the same resampled
questions as Table 28's. Its six contrasts are one Benjamini-Hochberg family
of their own, so Table 28's q-values do not move. They are macros only
(`pair.mv-comma.<ctx>.<stage>`, with `.ci` and `.q`, and
`pair.mv-comma.<ctx>.<min|max|q>`), and `artifacts/frontloading/permissive_pairs.csv`.

| MV minus Comma0.1 | Base | SFT | DPO |
|---|---|---|---|
| 4K | −10.44 [−12.33, −8.62], q = 0.002 | −2.87 [−4.39, −1.29], q = 0.003 | −3.38 [−4.94, −1.71], q = 0.002 |
| 16K | +2.35 [+0.54, +4.10], q = 0.011 | +1.38 [−0.34, +3.08], q = 0.134 | +1.58 [+0.03, +3.35], q = 0.053 |

So Comma0.1 with the same subset is ahead of MixtureVitae at every 4K stage.
At 16K MixtureVitae is ahead at the base, and after SFT and DPO the two cannot
be told apart once corrected. The paper says so since `paper_frontloaded_update/`.

## Language (added 2026-10-14)

- **Front-loading raises Comma0.1's language average** by +6.6 points at 4K
  (50.0 to 56.6) and +4.9 at 16K (52.1 to 57.0). That is close to
  MixtureVitae (+5.0 and +5.0) and FineWeb-Edu (+6.1 and +5.2), and unlike
  Nemotron-CC-HQ (+0.4 and +0.5).
- **The language gap holds for Comma0.1 too.** Front-loaded Comma0.1 is 0.6
  and 1.2 points above front-loaded MixtureVitae, and 5.6 to 7.9 points below
  front-loaded Nemotron-CC-HQ and FineWeb-Edu.
- **Without front-loading, Comma0.1 is at chance** on MMLU and CommonsenseQA,
  like MixtureVitae and FineWeb-Edu.

## IFEval with the strict checker (2026-10-24)

Every Protocol A IFEval score now comes from R21, scored with the strict
checker. Front-loading effects move by at most 0.4 points (MixtureVitae's 16K
base, +19.1 to +18.7), and every effect still excludes zero. Every claim of
§3.4 still holds: the dataset contrasts keep their signs and significance,
and the interaction stays significant for MixtureVitae at 4K only, and for
Comma0.1 at the 16K base only. Instruction following still improves in all
16 post-training cells.

## Limitations

- **Both Comma0.1 arms come from different runs**, as described above. Their
  language scores also come from different runs: May 2026 without
  front-loading, October 2026 with it, with the same settings and seed.
- **MixtureVitae's front-loaded language rows** are R8's runs of the main
  MixtureVitae model, the ones Table 11 prints, since 2026-10-26
  (`suites.LANGUAGE_REPLACEMENTS`). The language file of the artifacts
  repository scored the v0.02 MixtureVitae run
  (`open-sci/open-sci-ref-v0.02-1.7b-mixturevitae-300B-4096`) instead, a
  different pretraining run from the main model that the reasoning rows use.
  The authors chose the main model for both.
  MixtureVitae's language gain from front-loading is +4.8 points at 4K and +4.2
  at 16K, where the v0.02 run gave +5.0 at both.
- **The 4K base row of Comma0.1** uses the copy with a chat template
  (`ali-elganzory/1.7b-Comma0.1-300BT-WithChatTemplate`), as Phase 1 does. Its
  weights are identical to `ontocord/1.7b-Comma0.1-300BT`.
- **Decontamination of the Comma0.1 pretraining data** is stated in the paper
  but not recorded anywhere this repo can check.
