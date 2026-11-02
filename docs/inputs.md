# Inputs

Every table, figure and text number of the paper is built from a small set of
record files. Each one is pinned by its SHA-256 in
[`mv_post_train/inputs.json`](../mv_post_train/inputs.json), with the places
it can be downloaded from. Every command reads them through
`mv_post_train/inputs.py`, so a file that is not the pinned version is never
used.

```bash
mvpt fetch-inputs                # download every paper input that is missing
mvpt fetch-inputs --analyses     # also the inputs of the analyses in docs/
mvpt fetch-inputs R13 R16        # only these
mvpt check-inputs                # hash every local copy and compare it with its pin
```

The commands that need an input also download it on first use. A download is
kept only if its SHA-256 is the pinned one. A copy smaller than 300 MB is
checked again every time it is read. The three large zips (R6, R8 and
R8-fineweb) are checked when they are downloaded and by `mvpt check-inputs`.
Hashing them takes minutes, so their readers check a digest of the members
they read instead, which `inputs.json` pins as well (`member_digests`).

## Where they come from

The sources of each input are tried in order:
- **A URL at a fixed revision**, for files on Hugging Face and GitHub. A URL
  always names a commit or dataset revision, never a branch, because those
  repositories keep changing.
- **The records dataset**
  [`permissive-post-train/evaluation-records`](https://huggingface.co/datasets/permissive-post-train/evaluation-records)
  holds every input of the paper that is not in this repository, at revision
  `bca5a6e0`: those that were only in the bucket (R6 as a copy without the GPQA
  Diamond items), and copies of those on GitHub and in other Hugging Face
  repositories, whose original sources stay as fallbacks. Its card is
  `release/records/README.md`, and `release/records/upload.py` uploads the
  files and pins them. It is public since 2026-10-26.
- **The project bucket** (`gs://mayvic-cc-gating/projects/mv-post-train/`),
  through the `gcloud` command. It is private, so it needs access.
- **An mvpt command** that writes the file. It is named in the error message,
  not run.

## The inputs of the paper

The IDs are the names that the code and the results manifest give each input.

| ID | File | Size | What it holds, and what uses it | Fetched from |
|---|---|---|---|---|
| `R6` | `logs_without_gpqa_items.zip` | 416 MB | The results files of the older evaluations, January to May 2026, in a copy of the bucket's `logs.zip` without the GPQA Diamond items, which GPQA asks not to publish in plain text. `release/records/strip_gpqa.py` writes it: each item keeps only its record ID, the letter of the correct choice and the model's letter answers, so every score is the same. The results manifest keeps them, and no table of the paper prints them since 2026-10-19. | [records dataset](https://huggingface.co/datasets/permissive-post-train/evaluation-records), then the bucket |
| `R8` | `language_eval-logs.zip` | 2.9 GB | Harsh Raj's language results with per-question files and three prompt-example seeds, run in May 2026: MV, Comma0.1 and Nemotron-CC-HQ. Tables 15, 17, 19, 22, 30 to 32, 39 and 43, and Figure 2's language panel. | [records dataset](https://huggingface.co/datasets/permissive-post-train/evaluation-records), then the bucket |
| `R8-fineweb` | `logs-fineweb.zip` | 2.0 GB | The same for the 7 FineWeb-Edu checkpoints. Tables 15, 17, 19, 22, 30 and 31, and Figure 2's language panel. | [records dataset](https://huggingface.co/datasets/permissive-post-train/evaluation-records), then the bucket |
| `R13` | `canonical_reasoning_per_sample.csv` | 105 MB | The Protocol A records: one score per checkpoint, seed, task and question, for 57 checkpoints. Tables 1, 14, 16 to 18, 20, 21, 38 (its decontaminated rows) and 42 (its rows without masking), Figures 2 to 4, and the results manifest. | [records dataset](https://huggingface.co/datasets/permissive-post-train/evaluation-records), then GitHub, `Ali-Elganzory/permissive-llms-artifacts` at commit `8cc2e51`, then the bucket |
| `R13-language` | `ablation_frontloading_x_permissiveness_language_evaluations.csv` | 16 kB | One language score per task for the front-loading study's checkpoints without Comma0.1, and each task's shot count and metric. It gives the shot counts and metrics of Table 13, and it is a second source for most language cells of Tables 17 and 22, which print the runs of R8, R8-fineweb and R23. | [records dataset](https://huggingface.co/datasets/permissive-post-train/evaluation-records), then the same commit, then the bucket |
| `R14-scores` | `eval_scores.parquet` | 61 kB | marianna13's three-seed Phase 2 scores, `laion/mv-posttrain-eval-responses` at revision `e820fb34` (2026-10-19), with IFEval under all four definitions. The manifest's Phase 2 results, with IFEval at instruction level, strict. | [records dataset](https://huggingface.co/datasets/permissive-post-train/evaluation-records), then Hugging Face at that revision |
| `R14-meta` | `eval_responses_meta.parquet` | 2.4 MB | The small columns of the same revision's answer file (model, stage, task, question, seed, correctness, checkpoint folder, token limit), written by `mvpt fetch-phase2-records`. Its SHA-256 holds only with the pinned pyarrow (`requirements-lock.txt`). | [records dataset](https://huggingface.co/datasets/permissive-post-train/evaluation-records), then `mvpt fetch-phase2-records` |
| `R14-ifeval` | `ifeval_allmetrics_persample.parquet` | 0.2 MB | The same revision's IFEval counts per prompt and seed: instructions, and those followed with the strict and the loose checker. The manifest checks that the released instruction-level IFEval reproduces from them. | [records dataset](https://huggingface.co/datasets/permissive-post-train/evaluation-records), then Hugging Face at that revision |
| `R14-lengths` | `answer_lengths.parquet` | 5.5 MB | One row per sampled answer of the same revision, without its text: its length in bytes and, for answers long enough to reach the token limit, in its checkpoint's tokens, and whether the grader found an answer and a code block in it. Written by `mvpt measure-phase2-answers` from R14-responses, with tokenizers 0.22.1 (the `measure` extra) and the pinned pyarrow, which its SHA-256 depends on. The N4 tables of the Phase 2 analyses ([phase2_analyses.md](phase2_analyses.md)), in the statistics appendix from `paper_claims_update/` on. | [records dataset](https://huggingface.co/datasets/permissive-post-train/evaluation-records), then the bucket, or `mvpt measure-phase2-answers` (about two hours) |
| `R15` | `comma_reasoning_unified_per_sample.csv` | 12 MB | Protocol A records of the front-loaded Comma0.1 arm, received 2026-10-12, in R13's format. Tables 17, 18, 20 and 21, Figures 2 to 4, and the manifest. | [records dataset](https://huggingface.co/datasets/permissive-post-train/evaluation-records), then the bucket |
| `R18` | `ablation_decontamination_0_4B_reasoning_per_sample.csv` | 61 MB | Protocol A records, in R13's format, of the 36 checkpoints of the older experiments that R13 lacks: the 100B mixture ablation, the non-decontaminated MV and the 0.4B comparison. Received on 2026-10-18. Tables 35, 37, 38 (its rows without decontamination) and 40, and the manifest. | [records dataset](https://huggingface.co/datasets/permissive-post-train/evaluation-records), then the bucket |
| `R19` | `masked_reasoning_per_sample.csv` | 9 MB | Protocol A records, in R13's format, of the four masked MV checkpoints, received on 2026-10-19. The records name `ali-elganzory/*-masked-laion-eval`, copies of the `laion/*-masked` models with the same weights, and the manifest keys them by the `laion` repos, which Table 43 also uses. Table 42's rows with masking, and the manifest. | [records dataset](https://huggingface.co/datasets/permissive-post-train/evaluation-records), then the bucket |
| `R20` | `fineweb_edu_16k_april_base_unified_per_sample.csv` | 2 MB | Protocol A records, in R13's format, of the April FineWeb-Edu 16K base (`…-300B-4096-4096-longsft_16k`), the model its SFT and DPO rows were trained from. Received on 2026-10-23. It replaces R13's August 16K model in Phase 1 and the front-loading study (`protocol_a.REPLACEMENTS`): Tables 1, 14, 17, 18, 20 and 21, Figures 2 to 4, and the manifest. | [records dataset](https://huggingface.co/datasets/permissive-post-train/evaluation-records), then the bucket |
| `R21` | `ifeval_strict_per_sample_protocol_a.csv` | 34 MB | Protocol A IFEval verdicts with the strict checker, in R13's format: one row per checkpoint, seed and instruction, for 112 checkpoints, received on 2026-10-24. R13, R15, R18, R19 and R20 score IFEval with the loose checker. The revision takes every Protocol A IFEval score from R21 (`protocol_a.with_strict_ifeval`), for every checkpoint of those files but R13's August FineWeb-Edu 16K model, which no table uses and which then has no IFEval result. Tables 1, 14, 16 to 18, 20, 21, 35, 37, 38, 40 and 42, Figures 2 to 4, and the manifest. The workshop paper keeps R13's loose verdicts. | [records dataset](https://huggingface.co/datasets/permissive-post-train/evaluation-records), then the bucket |
| `R22` | `phase1_livecodebenchv5_per_sample.csv` | 17 MB | Protocol A LiveCodeBench v5 verdicts from Ali Elganzory, in R13's format: one row per checkpoint, seed and problem, for 112 checkpoints, on the 369 problems Protocol B runs (R14), received on 2026-10-26. R13, R15, R18, R19 and R20 run LiveCodeBench v2 (511 problems). The revision takes every Protocol A LiveCodeBench score from R22 (`protocol_a.with_lcb_v5`), for every checkpoint of those files but R13's August FineWeb-Edu 16K model, which no table uses and which then has no LiveCodeBench result. It holds no generation settings. Every Protocol A table and figure, and the manifest. The workshop paper keeps R13's v2 verdicts. | [records dataset](https://huggingface.co/datasets/permissive-post-train/evaluation-records), then the bucket |
| `R23` | `language_eval_all_models_3seed_2026-10-26.zip` | 3.5 GB | Harsh Raj's language runs of 2026-10-26, received on 2026-10-27: 24 checkpoints with prompt-example seeds 1234, 42 and 2025 and per-question files, and 41 checkpoints with one seed whose results files are R16's. The front-loading study reads its nine language cells without R8 logs from here: seed 1234 for Tables 17 and 22 and Figure 2, all three seeds for the intervals of Table 19 (`docs/language_intervals.md`). | [records dataset](https://huggingface.co/datasets/permissive-post-train/evaluation-records), then the bucket |
| `R24` | `dclm_16k_reasoning_evals_per_sample.csv` | 6 MB | Protocol A records, in R13's format, of the DCLM 16K checkpoints: the 16K extension of the DCLM base and its Tulu3 SFT and DPO models, three seeds each. Received on 2026-10-28. R13 has DCLM at 4K only, so the three checkpoints join the revision's Phase 1 in cells of their own (`protocol_a.ADDITIONS`): Tables 1 and 14. R21 and R22 have no verdicts for them, so the file also holds their IFEval verdicts, instruction-level with the strict checker (confirmed by Ali Elganzory on 2026-10-29), and their LiveCodeBench v5 verdicts, as task `LiveCodeBenchv5` next to v2 (`protocol_a.OWN_VERDICTS`). The workshop paper does not use it. | [records dataset](https://huggingface.co/datasets/permissive-post-train/evaluation-records), then the bucket |
| `R16` | `language_eval_all_models_2026-10-16.zip` | 3.1 MB | Harsh Raj's language results files for 82 checkpoints, one prompt-example seed, no per-question files. Tables 15, 30, 36, 39, 41 and 43. | [records dataset](https://huggingface.co/datasets/permissive-post-train/evaluation-records), then the bucket |
| `R17` | `masked_models_lang_evals.zip` | 0.5 MB | Harsh Raj's language results files for five masked checkpoints, run on 2026-10-12 with the models loaded from Hugging Face. Table 43's rows with masking. | [records dataset](https://huggingface.co/datasets/permissive-post-train/evaluation-records), then the bucket |
| `S1` | `hf_checkpoints.json` | 173 kB | The snapshot of Hugging Face model records that gives every checkpoint's parent in the manifest, taken by `mvpt snapshot-checkpoints` on 2026-10-12, updated on 2026-10-13, and updated on 2026-10-31 with the Phase 2 checkpoints uploaded on 2026-10-30 and HuggingFaceTB's SmolLM2-1.7B-Instruct-16k. | [records dataset](https://huggingface.co/datasets/permissive-post-train/evaluation-records), then the bucket |
| `S2-0.4b`, `S2-1.7b` | `config.json` of the 0.4B and 1.7B MV models | 1 kB each | The architectures of Table 5, from which `mvpt render-setup` computes the parameters and FLOPs. | [records dataset](https://huggingface.co/datasets/permissive-post-train/evaluation-records), then Hugging Face at a fixed revision |
| `C-setup` | `config/paper_setup.toml` | 8 kB | The setup values of Tables 5 to 13 that no evaluation record holds, each table with its source. | this repository |
| `O1` | `excluded_items.csv` | 57 kB | The 1,720 benchmark items of 17 benchmarks that the audit of the training data identified (pull request 88, `docs/investigations/training_data_overlap.md`), one row per benchmark and item, as frozen on 2026-11-01. The comparisons without the identified items leave them out of every checkpoint they compare: Table 4, Appendix A.3. | [records dataset](https://huggingface.co/datasets/permissive-post-train/evaluation-records), then the bucket |
| `O2` | `overlap_sensitivity.csv` | 75 kB | Every comparison of the Phase 2 tables, Table 4, the front-loading study and the language suite, on all items and without O1's, each with its own analysis's intervals and seed: 647 rows, written by `mvpt measure-overlap-sensitivity`. Its rows on all items equal the published values to 0.0001. Appendix A.3 and the `\povl` macros. `mvpt render-overlap` recomputes Table 4's rows from R14-meta and O1 and stops if they differ. | [records dataset](https://huggingface.co/datasets/permissive-post-train/evaluation-records), then the bucket |
| `O3` | `found.parquet` | 1.5 MB | Where the audit found each item: the training dataset, row, turn and evidence (13-gram or exact), with no benchmark text. Appendix A.3's table of items found. `mvpt render-overlap` checks that O1 is exactly the items O3 found with the evidence that is left out. | [records dataset](https://huggingface.co/datasets/permissive-post-train/evaluation-records), then the bucket |
| `O4` | `mv_question_overlap_rates.csv` | 1 kB | The share of the questions of the AceReason, Nemotron-Cascade and Superior-Reasoning 30K samples trained on that match MixtureVitae's reasoning data other than OpenThoughts3, by domain (`mvpt measure-mv-question-overlap`). Appendix E's corpus-overlap table. | [records dataset](https://huggingface.co/datasets/permissive-post-train/evaluation-records), then the bucket |
| `O5` | `mv_question_overlap_rates.csv` (OT3) | 0.3 kB | The same for the Phase 2 OpenThoughts3 30K sample, against the same data, which the model pretrained without the OpenThoughts3 source keeps. | [records dataset](https://huggingface.co/datasets/permissive-post-train/evaluation-records), then the bucket |
| `O6` | `phase2_tokens.csv` | 1 kB | The training tokens of the four Phase 2 30K samples of the MixtureVitae runs, counted in LLaMA-Factory's tokenized caches with the open-sci tokenizer (`mvpt measure-phase2-tokens`, `docs/investigations/phase2_training_tokens.md`): mean and median length, the share cut at 16,384 tokens, the share trained to stop, and the supervised tokens. Appendix A.2's token table and the token numbers of §3.3 and §3.4. | [records dataset](https://huggingface.co/datasets/permissive-post-train/evaluation-records), then the bucket |
| `O7` | `phase2_ot3_release.csv` | 0.4 kB | The OpenThoughts3 30K sample as released, by domain and by whether it exceeds 16,384 tokens: how many answers close their reasoning and hold a boxed answer. Appendix A.2's share of answers that never close their reasoning. | [records dataset](https://huggingface.co/datasets/permissive-post-train/evaluation-records), then the bucket |

**R16 has two versions.** The zip of 2026-10-16 replaces the one of
2026-10-14 (both are in the bucket's `data/workshop/`). It runs the 23 checkpoints that
had used 0 shots on every task again, with Table 13's shot counts, and keeps
every other results file, among them those of the front-loaded Comma0.1
checkpoints.

## The inputs of the analyses

These are used by older analyses, not by the paper's tables or the results
manifest, except R14-responses, from which R14-lengths is made.

| ID | File | What it is | Fetched from |
|---|---|---|---|
| `R14-responses` | `eval_responses.parquet` | Every saved Phase 2 answer at revision `e820fb34` (10 GB), which `mvpt measure-phase2-answers` reads to write R14-lengths | `laion/mv-posttrain-eval-responses` at revision `e820fb34` |
| `R3` | `mv_aime24_amc23_best_responses.parquet` | A replacement file for MV's AIME and AMC answers | `laion/mv-posttrain-eval-responses` at revision `481cf954` |
| `R4` | `eval_responses.parquet` | Every saved Phase 2 answer at that revision (1.5 GB) | the same |
| `R1-wide` | `eval_scores_wide.csv` | The released Phase 2 score grid that `mvpt released-scores` compares against | the same |
| `ifeval-prompts` | `ifeval_input_data.jsonl` | The IFEval prompts and instruction arguments | `google/IFEval` at revision `966cd895` |
| `T-submission` | `submission/submission_tables.csv` | The eleven results tables of the paper as submitted for review, 2,436 cells, transcribed from its PDF by `mvpt transcribe-submission`, which `mvpt render-reconciliation` compares with the revision. The public copy of this repository leaves it out, and the `reconciliation` target is then skipped | this repository |
| `R7-greedy-final`, `R7-greedy-metric`, `R7-greedy-items` | `final_scores.csv`, `metric_scores.csv`, `per_sample_scores.csv` | The three-seed Phase 1 grid at the submitted decoding (July settings), under `data/paper_decoding_3seeds/` | `ali-elganzory/mv-posttraining-scores` at revision `24d1f140` |
| `R7-sampling-final`, `R7-sampling-metric`, `R7-sampling-items` | the same three files | The same grid sampled at temperature 0.7, under `data/sampling07/` | the same revision |

The local copies of all ten were checked against the files Hugging Face
serves at those revisions on 2026-10-17.

## Changing an input

When a record file gets a new version, for example new runs from an evaluator:
1. Put the new file in the bucket under a new name, or note its new Hugging Face revision. Keep the old one, because older versions of the results were built from it.
2. Change its entry in `inputs.json`: file name, `sha256`, `bytes`, sources, and `member_digests` if it has them. A test checks that every URL names a fixed revision.
3. Run the commands that read it, and commit what changes under `results/`. Every changed paper number goes into the pull request's description, with the reason.

After `mvpt snapshot-checkpoints --update`, the snapshot changes, so
`mvpt build-manifest` records a failed check until the new file is uploaded to
the bucket and pinned as S1.
