# Building and checking the paper's generated files

Every table, figure and text-number file that the paper uses is written by a
command of this repo, from pinned inputs ([inputs.md](inputs.md)). This page
says which command makes each file, how to rebuild them all, and how to check
a paper source tree against them.

```bash
mvpt fetch-inputs                       # the pinned inputs, if they are missing
mvpt build                              # every target, in order, into results/
mvpt build statistics                   # one target and the ones it reads
mvpt build --check                      # rebuild into a temporary folder and compare with results/
mvpt check-paper <paper sources>        # compare a paper tree with results/
mvpt check-paper <paper sources> --numbers --markdown   # every number that would change
mvpt export-paper <paper sources>       # copy the generated files into a paper tree
```

Commands run from the repository root. `mvpt build` takes about 10 minutes,
most of it for the paired bootstraps of Tables 26 and 27 and the language
changes.

## Targets

`mv_post_train/targets.py` lists them. Each one reads only pinned inputs, and
writes only under the folder it is given, so `build --check` leaves the
working tree as it was.

Every renderer reads its results through the manifest
([manifest.md](manifest.md), "Who reads it"). The ones that resample
per-question records (front-loading, statistics and the language changes)
read them from the inputs the manifest names, and stop unless they give the
manifest's scores. The setup tables also read `config/paper_setup.toml`.

| Target | Command | Inputs | Writes |
|---|---|---|---|
| `manifest` | `mvpt build-manifest` | R13, R15, R18, R19, R20, R21, R14-scores, R14-meta, R14-ifeval, S1, R6, R8, R8-fineweb, R16, R17, R23, R13-language | `results/manifest.json` |
| `zoo` | `mvpt render-zoo` | the manifest | `results/zoo/` |
| `phase2` | `mvpt render-phase2` | the manifest, R14-meta | `results/phase2/` |
| `phase2-analyses` | `mvpt render-phase2-analyses` | the manifest, R14-scores, R14-meta, R14-ifeval, R14-lengths | `results/phase2_analyses/` |
| `frontloading` | `mvpt render-frontloading` | the manifest, R13, R15, R20, R21 | `results/frontloading/` |
| `older` | `mvpt render-older-tables` | the manifest | `results/older/` |
| `overlap` | `mvpt render-overlap` | the manifest, O1 to O7, R14-meta | `results/overlap/` |
| `statistics` | `mvpt render-statistics` | the manifest, R13, R18, R20, R21 | `results/statistics/` |
| `language-suite` | `mvpt render-language-suite` | the manifest | `results/language_suite/` |
| `setup` | `mvpt render-setup` | the manifest, C-setup, S2-0.4b, S2-1.7b, R14-meta, R13-language | `results/setup/` |
| `language` | `mvpt render-language-intervals` | the manifest, R8, R8-fineweb, R23 | `results/language/` |
| `reconciliation` | `mvpt render-reconciliation` | the manifest, the tables of `zoo`, `phase2`, `older` and `language-suite`, T-submission | `results/reconciliation/` |

`setup` prints the values of `config/paper_setup.toml` that no record holds,
computes every column that follows from others, and writes
`results/setup/checks.csv`, which checks the stated values against each other
and the records. A check that fails is a warning, not an error: in a draft of the
revision, Table 17's 100B row stated 4.12M tokens per step, but its
iterations give 4.19M.

`phase2-analyses` writes the Phase 2 analyses of the camera-ready plan
([phase2_analyses.md](phase2_analyses.md)): intervals for Table 3's changes,
the OT3 exclusion's difference of gains, MV against each reference before and
after OT30K, changes by domain, and the answers cut off at the token limit.
The statistics appendix includes the first five tables and their macros from
`paper_stats_appendix_update/` on. The two tables of cut-off answers are not
in the paper.

`reconciliation` compares the submitted paper's results tables with the
tables the other targets wrote, so it runs after them. The paper does not
print it, and the public copy of this repository leaves the submitted tables
out, so there the target is skipped.

## The paper's copies

`targets.PAPER_FILES` maps each generated file to its place in the paper's
sources. The table and figure numbers are those of a draft of the revision,
and can differ from the published paper's; the paper file names do not.

| Item | Generated file | Paper file |
|---|---|---|
| Figure 1 | `results/phase2/figures/figure_pipeline.pdf` | `figures/figure_pipeline.pdf` |
| Table 1 | `results/zoo/tables/phase1_zoo.tex` | `tables/phase1_zoo.tex` |
| Table 6 | `results/zoo/tables/phase1_zoo_full.tex` | `tables/phase1_zoo_full.tex` |
| `\pzoo` macros | `results/zoo/tables/zoo_numbers.tex` | `tables/zoo_numbers.tex` |
| Table 3 | `results/phase2/tables/phase2_endpoints_full.tex` | `tables/phase2_endpoints_full.tex` |
| Phase 2 endpoints and changes (main text, since the Table 2 update) | `results/phase2_analyses/tables/phase2_endpoints.tex` | `tables/phase2_endpoints.tex` |
| Figure 3 | `results/phase2/figures/phase2_receptivity.pdf` | `figures/phase2_receptivity.pdf` |
| Table 4 | `results/phase2/tables/ot3_pretrain_ablation.tex` | `tables/ot3_pretrain_ablation.tex` |
| Table 4 on all 500 problems (Appendix E) | `results/phase2/tables/heldout_reasoning.tex` | `tables/heldout_reasoning.tex` |
| Table 4 | `results/overlap/tables/heldout_reasoning_retained.tex` | `tables/heldout_reasoning_retained.tex` |
| Overlap filtering and audit coverage (Appendix A.3) | `results/overlap/tables/overlap_coverage.tex` | `tables/overlap_coverage.tex` |
| Benchmark items identified (Appendix A.3) | `results/overlap/tables/overlap_items.tex` | `tables/overlap_items.tex` |
| Principal comparisons with and without the items (Appendix A.3) | `results/overlap/tables/overlap_results.tex` | `tables/overlap_results.tex` |
| Every comparison without the items (Appendix A.3) | `results/overlap/tables/overlap_summary.tex` | `tables/overlap_summary.tex` |
| Calls that change without the items (Appendix A.3) | `results/overlap/tables/overlap_calls.tex` | `tables/overlap_calls.tex` |
| Phase 2 samples in MixtureVitae's pool (Appendix E) | `results/overlap/tables/corpus_overlap.tex` | `tables/corpus_overlap.tex` |
| Training tokens of each reasoning dataset (Appendix A.2) | `results/overlap/tables/phase2_tokens.tex` | `tables/phase2_tokens.tex` |
| Table 32 | `results/phase2/tables/heldout_mv_grid.tex` | `tables/heldout_mv_grid.tex` |
| `\ptwo` macros | `results/phase2/tables/phase2_numbers.tex` | `tables/phase2_numbers.tex` |
| Phase 2 changes with intervals (statistics appendix) | `results/phase2_analyses/tables/phase2_change_intervals.tex` | `tables/phase2_change_intervals.tex` |
| OT3 exclusion gains (statistics appendix) | `results/phase2_analyses/tables/ot3_exclusion_gains.tex` | `tables/ot3_exclusion_gains.tex` |
| MV against the references (statistics appendix) | `results/phase2_analyses/tables/phase2_endpoint_gaps.tex` | `tables/phase2_endpoint_gaps.tex` |
| Leave-one-out (statistics appendix) | `results/phase2_analyses/tables/phase2_leave_one_out.tex` | `tables/phase2_leave_one_out.tex` |
| Phase 2 changes by domain (statistics appendix) | `results/phase2_analyses/tables/phase2_domain_changes.tex` | `tables/phase2_domain_changes.tex` |
| Answers cut off at the token limit (statistics appendix) | `results/phase2_analyses/tables/phase2_cutoff.tex` | `tables/phase2_cutoff.tex` |
| Answers with no final answer (statistics appendix) | `results/phase2_analyses/tables/phase2_no_answer.tex` | `tables/phase2_no_answer.tex` |
| `\ptwoan` macros | `results/phase2_analyses/tables/phase2_analyses_numbers.tex` | `tables/phase2_analyses_numbers.tex` |
| `\povl` macros | `results/overlap/tables/overlap_numbers.tex` | `tables/overlap_numbers.tex` |
| Table 2 | `results/frontloading/tables/factorial_summary.tex` | `tables/factorial_summary.tex` |
| Figure 2 | `results/frontloading/figures/factorial_main.pdf` | `figures/factorial_main.pdf` |
| Table 28 | `results/frontloading/tables/contrasts.tex` | `tables/contrasts.tex` |
| Table 29 | `results/frontloading/tables/domain_robustness.tex` | `tables/domain_robustness.tex` |
| Figure 4 | `results/frontloading/figures/frontloading_trajectory.pdf` | `figures/frontloading_trajectory.pdf` |
| Table 30 | `results/frontloading/tables/factorial_perbenchmark.tex` | `tables/factorial_perbenchmark.tex` |
| Table 31 | `results/frontloading/tables/factorial_language.tex` | `tables/factorial_language.tex` |
| Figure 5 | `results/frontloading/figures/frontloading_domains.pdf` | `figures/frontloading_domains.pdf` |
| `\pfl` macros | `results/frontloading/tables/frontloading_numbers.tex` | `tables/frontloading_numbers.tex` |
| Table 7 | `results/older/tables/ablation_100b_reasoning.tex` | `tables/ablation_100b_reasoning.tex` |
| Table 8 | `results/older/tables/decontamination_reasoning.tex` | `tables/decontamination_reasoning.tex` |
| Table 9 | `results/older/tables/smallscale_reasoning.tex` | `tables/smallscale_reasoning.tex` |
| Table 10 | `results/language_suite/tables/language_ablation_100b.tex` | `tables/language_ablation_100b.tex` |
| Table 11 | `results/language_suite/tables/language_main.tex` | `tables/language_main.tex` |
| Table 12 | `results/language_suite/tables/language_decontamination.tex` | `tables/language_decontamination.tex` |
| Table 13 | `results/language_suite/tables/language_smallscale.tex` | `tables/language_smallscale.tex` |
| Table 14 | `results/language_suite/tables/language_phase2.tex` | `tables/language_phase2.tex` |
| Table 15 | `results/older/tables/masking_reasoning.tex` | `tables/masking_reasoning.tex` |
| Table 16 | `results/language_suite/tables/masking_language.tex` | `tables/masking_language.tex` |
| `\plang` macros | `results/language_suite/tables/language_suite_numbers.tex` | `tables/language_suite_numbers.tex` |
| `\pold` macros | `results/older/tables/older_numbers.tex` | `tables/older_numbers.tex` |
| Table 17 | `results/setup/tables/training_schedules.tex` | `tables/training_schedules.tex` |
| Table 18 | `results/setup/tables/opensciref_scales.tex` | `tables/opensciref_scales.tex` |
| Table 19 | `results/setup/tables/ot30k_sft_hyperparams.tex` | `tables/ot30k_sft_hyperparams.tex` |
| Table 20 | `results/setup/tables/ot30k_sft_gpu_hours.tex` | `tables/ot30k_sft_gpu_hours.tex` |
| Table 21 | `results/setup/tables/long_ctx_gpu_hours.tex` | `tables/long_ctx_gpu_hours.tex` |
| Table 22 | `results/setup/tables/budget.tex` | `tables/budget.tex` |
| Table 23 | `results/setup/tables/eval_settings_a.tex` | `tables/eval_settings_a.tex` |
| Table 24 | `results/setup/tables/eval_settings_b.tex` | `tables/eval_settings_b.tex` |
| Table 25 | `results/setup/tables/eval_settings_general.tex` | `tables/eval_settings_general.tex` |
| `\psetup` macros | `results/setup/tables/setup_numbers.tex` | `tables/setup_numbers.tex` |
| Table 26 | `results/statistics/tables/stat_matched_compute.tex` | `tables/stat_matched_compute.tex` |
| Table 27 | `results/statistics/tables/stat_mixture_ablation.tex` | `tables/stat_mixture_ablation.tex` |
| `\pstat` macros | `results/statistics/tables/statistics_numbers.tex` | `tables/statistics_numbers.tex` |
| Table 28 | `results/language/tables/language_frontloading.tex` | `tables/language_frontloading.tex` |
| Table 33 | `results/language/tables/language_changes.tex` | `tables/language_changes.tex` |
| Table 34 | `results/language/tables/language_changes_mv.tex` | `tables/language_changes_mv.tex` |
| `\langci` macros | `results/language/tables/language_numbers.tex` | `tables/language_numbers.tex` |

The rule is to change the generator, never the paper's copy. `check-paper`
reports a copy that was edited by hand, because it is no longer
byte-identical. `targets.PAPER_ALLOWED` can list a copy that may differ, with
the reason, and it is meant to stay empty.

## What `check-paper` reports

- **Generated files:** each mapped copy is identical, differs, or is missing.
- **Macro keys:** every `\ptwo{...}`, `\pfl{...}` and so on that the text uses must exist in the generated macro file. A missing key prints in bold between question marks in the PDF.
- **Hand-made tables and figures:** every table environment and graphic the paper includes (from `main_neurips_2026.tex`, following `\input`) that no target makes. In a draft of the revision these were 15 tables: 10 to 14 and 16 to 25.
- **Numbers typed into the prose:** counted per file, outside tables, figures, macros, references and lengths. This is a report for now. It becomes a failing check once every result number in the text is a macro.
- **Unused files** under `tables/` and `figures/`.

It exits 1 if a mapped copy differs or is missing, or if a macro key is
missing.

## Every changed number goes into the pull request

`mvpt check-paper <sources> --numbers --markdown` lists every number that
differs between the paper (the latest uploaded revision) and the generated
files. With `--by-row` it prints one line per table row instead, which keeps
the list short when most cells of a table change:
- table cells, by row and column
- macro values, by key
- the numbers printed in figures, read with PyMuPDF when it is installed

A generated table that the paper does not have as a file yet is compared with
the hand-made table of the same `\label`. Every changed number needs a reason
in `targets.REASONS`, and the command exits 1 for a change without one. A
macro key that is new is not a change, since the paper does not use it yet.
The list counts the new keys. The
pull request that changes numbers pastes this list into its description, and
the paper delivery uses the same list in its CHANGES.md.

## Continuous integration

`.github/workflows/ci.yml` runs the tests on every pull request, with the
pinned package versions, and nothing else. The inputs are public, and
`.github/workflows/reproduce.yml` fetches them and runs `mvpt build --check` on
a clean runner when it is started by hand. It needs no credentials. `mvpt build
--check` also runs locally, before a pull request that touches `mv_post_train/`, `results/` or
`inputs.json` is merged, and before every paper delivery. Its summary line goes
into the pull request's description.
