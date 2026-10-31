---
license: cc-by-4.0
pretty_name: MixtureVitae post-training evaluation records
tags:
- evaluation
- reproducibility
---

# MixtureVitae post-training evaluation records

The evaluation records behind the tables and figures of *Strong
Post-Training from Permissive, Reasoning-Dominant Pretraining*
(NeurIPS 2026), in one place. The pipeline
that turns them into the paper's tables, figures and quoted numbers is
[`mv-post-train`](https://github.com/MixtureVitae/mv-post-train). Its
`mv_post_train/inputs.json` pins every file here by SHA-256 and by a
revision of this repository, and `mvpt fetch-inputs` downloads them:

```bash
pip install -e . -r requirements-lock.txt
mvpt fetch-inputs        # every input of the paper, from here and its other sources
mvpt build --check       # rebuild every generated file and compare with results/
```

Some files are copies of public ones, and `inputs.json` keeps their original
sources as fallbacks: R13 and R13-language of
[`Ali-Elganzory/permissive-llms-artifacts`](https://github.com/Ali-Elganzory/permissive-llms-artifacts)
at commit `8cc2e51`, R14-scores and R14-ifeval of
[`laion/mv-posttrain-eval-responses`](https://huggingface.co/datasets/laion/mv-posttrain-eval-responses)
at revision `e820fb34` (R14), and S2-0.4b and S2-1.7b of the two checkpoints'
model repositories. The only other input of the paper is a small file in the
pipeline repository itself, its setup values. `docs/inputs.md` of the pipeline describes every
input.

## Files

The IDs are the pipeline's input IDs. Table and figure numbers are those of the
paper's revision of 2026-10-28.

| ID | File | Size | SHA-256 | What it holds |
|---|---|---:|---|---|
| R13 | `protocol_a/canonical_reasoning_per_sample.csv` | 105 MB | `867ff6d6782b…` | The main Protocol A records: one score per checkpoint, seed, task and question, for 57 checkpoints. Tables 1, 14, 16 to 18, 20, 21, 38 (its decontaminated rows) and 42 (its rows without masking), Figures 2 to 4. |
| R15 | `protocol_a/comma_reasoning_unified_per_sample.csv` | 12 MB | `a8e33f163ce4…` | Protocol A records of the front-loaded Comma0.1 arm (six checkpoints). Tables 17, 18, 20 and 21, Figures 2 to 4. |
| R18 | `protocol_a/ablation_decontamination_0_4B_reasoning_per_sample.csv` | 61 MB | `021b81d039ec…` | Protocol A records of the 36 checkpoints of the 100B mixture ablation, the MV model without decontamination and the 0.4B comparison. Tables 35, 37, 38 (its rows without decontamination) and 40. |
| R19 | `protocol_a/masked_reasoning_per_sample.csv` | 9 MB | `8084115c57e8…` | Protocol A records of the four masked MV checkpoints (copies of the `laion/*-masked` models with the same weights). Table 42's rows with masking. |
| R20 | `protocol_a/fineweb_edu_16k_april_base_unified_per_sample.csv` | 2 MB | `153341e3f95a…` | Protocol A records of the FineWeb-Edu 16K base that its SFT and DPO models were trained from. Tables 1, 14, 17, 18, 20 and 21, Figures 2 to 4. |
| R21 | `protocol_a/ifeval_strict_per_sample_protocol_a.csv` | 34 MB | `eac0a09812bc…` | Protocol A IFEval verdicts with the strict checker, one row per checkpoint, seed and instruction, for 112 checkpoints. Every Protocol A IFEval score of the paper except DCLM-16K's, which R24 holds. |
| R22 | `protocol_a/phase1_livecodebenchv5_per_sample.csv` | 17 MB | `80b2d27da77a…` | Protocol A LiveCodeBench v5 verdicts, one row per checkpoint, seed and problem, for 112 checkpoints, on the 369 problems of LiveCodeBench v5 that Phase 2 runs. Every Protocol A LiveCodeBench score of the paper except DCLM-16K's, which R24 holds. |
| R24 | `protocol_a/dclm_16k_reasoning_evals_per_sample.csv` | 6 MB | `fcb6bd7b79f3…` | Protocol A records of the DCLM 16K checkpoints (base, Tulu3 SFT and DPO), with their IFEval verdicts and their LiveCodeBench v5 verdicts (task `LiveCodeBenchv5`) next to v2, because R21 and R22 do not cover them. DCLM's 16K rows of Tables 1 and 14. |
| R13-language | `language/ablation_frontloading_x_permissiveness_language_evaluations.csv` | 16 kB | `e108272f2227…` | One language score per task for the front-loading study's checkpoints without Comma0.1, and each task's shot count and metric. The shot counts and metrics of Table 13. The scores are a second source for most language cells of Tables 17 and 22, which print the runs of R8, R8-fineweb and R23. |
| R8 | `language/language_eval-logs.zip` | 2.9 GB | `0492e8a69dd6…` | LM Evaluation Harness results with per-question files and three prompt-example seeds, for 19 checkpoints of MV, Comma0.1 and Nemotron-CC-HQ. Tables 15, 17, 19, 22, 30 to 32, 39 and 43, and Figure 2's language panel. |
| R8-fineweb | `language/logs-fineweb.zip` | 2.0 GB | `d940f546b4dc…` | The same for 7 FineWeb-Edu checkpoints. Tables 15, 17, 19, 22, 30 and 31, and Figure 2's language panel. |
| R16 | `language/language_eval_all_models_2026-10-16.zip` | 3 MB | `bb0b7b1a2fc7…` | LM Evaluation Harness results for 82 checkpoints, one prompt-example seed, without per-question files. Tables 15, 30, 36, 39, 41 and 43. |
| R17 | `language/masked_models_lang_evals.zip` | 0.5 MB | `c53217be762c…` | LM Evaluation Harness results for five masked checkpoints. Table 43's rows with masking. |
| R23 | `language/language_eval_all_models_3seed_2026-10-26.zip` | 3.5 GB | `1a0422576760…` | LM Evaluation Harness results with per-question files and three prompt-example seeds (1234, 42 and 2025) for 24 checkpoints, run on 2026-10-26, and one-seed results files for 41 more, which are R16's. The front-loading study's language cells that R8 does not hold, and the intervals of its language contrasts: Tables 17, 19 and 22, and Figure 2's language panel. |
| R14-scores | `phase2/eval_scores.parquet` | 70 kB | `666002c3c4e9…` | The three-seed Phase 2 scores of R14, with IFEval under all four definitions. Every Phase 2 result of the paper. |
| R14-ifeval | `phase2/ifeval_allmetrics_persample.parquet` | 0.2 MB | `1dd7734ef79a…` | R14's IFEval counts per prompt and seed: instructions, and those followed with the strict and the loose checker. |
| R14-meta | `phase2/eval_responses_meta.parquet` | 2 MB | `59872f7e7a31…` | The small columns (model, stage, task, question, seed, correctness, checkpoint folder, token limit) of R14's answer file, `eval_responses.parquet` (10 GB), which is not copied here. `mvpt fetch-phase2-records` writes the same file from R14. |
| R14-lengths | `phase2/answer_lengths.parquet` | 6 MB | `9863758e52d1…` | One row per Phase 2 answer of R14, without its text: its length, and whether the grader found an answer and a code block. `mvpt measure-phase2-answers` writes the same file from R14 in about two hours. |
| S1 | `snapshots/hf_checkpoints.json` | 173 kB | `4d4723136230…` | A snapshot of the Hugging Face records of every checkpoint the paper evaluates (repos, weight hashes, model cards, training summaries, context settings), taken on 2026-10-12 and 2026-10-13 and updated on 2026-10-31 with the Phase 2 checkpoints uploaded on 2026-10-30 and HuggingFaceTB's SmolLM2-1.7B-Instruct-16k. It gives each checkpoint's parent in the results manifest. |
| S2-0.4b, S2-1.7b | `setup/*.config.json` | 1 kB each | `24a64d86ef89…`, `b2d8faeee3c6…` | The `config.json` of the 0.4B and 1.7B MV base models, the architectures of Table 5. |
| R6 | `older/logs_without_gpqa_items.zip` | 416 MB | `2f9db28c0e7a…` | LM Evaluation Harness and Evalchemy results files of the older evaluations, January to May 2026, without the GPQA Diamond items (see below). No table of the paper prints them, but the results manifest keeps them. |

The full SHA-256 of each file is in `inputs.json`, which checks every download.

## Formats

**Protocol A records** (R13, R15, R18 to R22, R24) have one row per checkpoint, decoding
seed, benchmark and question: `model_id, stage, model_group, front_loading,
context_length, seed, benchmark, sample_id, correct`, and in some files the
experiments the checkpoint belongs to. They hold no question or answer text.
Protocol A is greedy decoding with seeds 42, 1234 and 2025 (the paper's
Table 11).

**Language results** (R8, R8-fineweb, R16, R17, R23) are the results files
that the LM Evaluation Harness writes, zipped as received. R8, R8-fineweb and
R23 also hold its per-question files, which contain the benchmark questions
and the models' answers. R23 keeps each checkpoint, task and seed in a folder
of its own (`<checkpoint>/<task>/seed<N>/<checkpoint>/`).

**Older evaluations** (R6) are the results files as received, except for
GPQA Diamond. GPQA asks that its items not be published in plain text, and
Evalchemy saves every item it scores with the results, so here each GPQA item
keeps only its GPQA record ID, the letter of the correct choice, and the model's
answer in each repetition if it is a letter (otherwise an empty string). The
scores and every other file are unchanged. `release/records/strip_gpqa.py` of
the pipeline writes this copy from the original.

The paths with `/e/scratch/jureap59/` in the results files are the working
directory of the cluster the evaluations ran on.

## License

The records are under [CC BY 4.0](https://creativecommons.org/licenses/by/4.0/).
The benchmark questions in R6, R8, R8-fineweb and R23 keep the licenses of
their benchmarks, and the model answers in them those of the models that wrote
them.

## Versions

Records are added here as new runs and checkpoints are published. Each version
of the pipeline pins the revision it read, so a release of the paper's tables
always names the exact files behind it.
