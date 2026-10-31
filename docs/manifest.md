# Results manifest

`results/manifest.json` is the authoritative set of results for the paper. It
holds every result the paper prints, and ties each one to a named checkpoint,
the checkpoint that checkpoint was trained from, the evaluation protocol and
the source record. Every renderer reads its results through it (see "Who reads
it" below).

It is built by a script from pinned inputs, so anyone can rebuild it and get
the same file:

```bash
mvpt build-manifest                 # -> results/manifest.json
```

## Scope

Included:

- **Protocol A results** for Phase 1, the front-loading
  study, the older experiments and masking: 104 checkpoints, all 11 reasoning
  tasks, three seeds each, computed from the per-question records. 57 come
  from the main record file, 6, the front-loaded Comma arm, from a second, 36,
  the 100B mixture ablation, the non-decontaminated MV and the 0.4B
  comparison, from a third (R18, received on 2026-10-18), and 4, the masked MV
  models of Table 15, from a fourth (R19, received on 2026-10-19). R19's
  records name copies of the `laion` masked models with the same weights, so
  their results are keyed by the `laion` repos, with the records' name in each
  source (`records_name`). A fifth file (R20, received on 2026-10-23) holds the
  April FineWeb-Edu 16K base, which replaces the main file's August 16K model
  in Phase 1 and the front-loading study (`protocol_a.REPLACEMENTS`). The
  August model keeps its results but belongs to no experiment, and the
  FineWeb-Edu 16K SFT model's change is now measured from the base it was
  trained from.
- **Phase 2 results** from the three-seed records released on 2026-10-07, with
  GSM8K from the re-run of 2026-10-15: the OT30K runs, the runs on the other
  reasoning datasets (AceReason, Nemotron-Cascade and Superior-Reasoning), the
  removal of OpenThoughts3 from pretraining, and the 100B runs. All 11 tasks.

- **The older evaluations** behind Tables 7, 8, 9 and 15 (protocol
  `older_evalchemy`): the 42 checkpoints of the 100B mixture ablation, the
  decontamination comparison and the 0.4B comparison, from `logs.zip`. Each
  result is the mean of the runs that are not broken, and lists every run with
  its value and whether the rule uses it.
- **The language suite** (protocol `language_suite`): one lm-eval run per
  checkpoint for Tables 10 to 14, 16, 31, 33 and 34, from Harsh Raj's results
  files and R13-language, with each run's results files and whether it loaded
  a `patched_models` copy.

Left out for now:

- **Protocol B values that exist only as numbers copied by hand**
  (`mv_post_train/data/protocol_b.py`), which only the workshop paper and the
  comparison with the submission use. Phase 2 now comes from records instead.
- **The setup values** of Tables 17 to 25, which are not results. They are in
  `config/paper_setup.toml` (input C-setup).

Suite averages and changes are computed for the two reasoning protocols only.
The older and language tables compute their own averages from the manifest's
task results, and the language changes of Tables 33 and 34 check their pairs
against its lineage.

## Inputs

Every input is pinned in `mv_post_train/inputs.json` and described in
[inputs.md](inputs.md). `mvpt fetch-inputs` downloads them.

| Input | What it is | ID |
|---|---|---|
| `data/workshop/canonical_reasoning_per_sample.csv` | Protocol A records, one score per checkpoint, seed, task and question. From `Ali-Elganzory/permissive-llms-artifacts` at commit `8cc2e51` | R13 |
| `data/workshop/ablation_decontamination_0_4B_reasoning_per_sample.csv` | Protocol A records of the older experiments' 36 checkpoints, in the same format and on the same questions. Its labels put several checkpoints in one cell, so they are told apart by name | R18 |
| `data/workshop/masked_reasoning_per_sample.csv` | Protocol A records of the four masked MV checkpoints, in the same format and on the same questions. They name copies of the `laion` masked models, and are keyed by the `laion` repos | R19 |
| `data/workshop/fineweb_edu_16k_april_base_unified_per_sample.csv` | Protocol A records of the April FineWeb-Edu 16K base, in the same format and on the same questions. It replaces the August 16K model in Phase 1 and the front-loading study | R20 |
| `data/workshop/ifeval_strict_per_sample_protocol_a.csv` | Protocol A IFEval verdicts with the strict checker, in the same format and on the same instructions, for every checkpoint of the files above but the August FineWeb-Edu 16K model. The record files score IFEval with the loose checker. Every Protocol A IFEval result is computed from this file instead (`protocol_a.with_strict_ifeval`), so the August model has no IFEval result | R21 |
| `data/workshop/comma_reasoning_unified_per_sample.csv` | Protocol A records for the front-loaded Comma arm (`comma-mv_reasoning`), six checkpoints in the same format. Received on 2026-10-12 and not yet in the artifacts repository. Its home is the project bucket | R15 |
| `data/phase2_3seed/eval_scores.parquet`, `eval_responses_meta.parquet` and `ifeval_allmetrics_persample.parquet` | Phase 2 records from `laion/mv-posttrain-eval-responses` at revision `e820fb34`: the released scores, the small columns of the answer file (model, stage, task, question, seed, correctness, checkpoint folder, token limit), written by `mvpt fetch-phase2-records`, and the IFEval counts per prompt | R14-scores, R14-meta, R14-ifeval |
| `data/manifest/hf_checkpoints.json` | A snapshot of the Hugging Face records of every checkpoint and every parent they name: repos, weight hashes, model cards, training summaries and context settings, taken by `mvpt snapshot-checkpoints` | S1 |
| `artifacts/logs.zip` | The older evaluations' results files | R6 |
| `data/language_logs/language_eval-logs.zip`, `logs-fineweb.zip` | Harsh Raj's language runs of May 2026, of which the `default` folder is read | R8, R8-fineweb |
| `data/workshop/language_eval_all_models_2026-10-16.zip`, `masked_models_lang_evals.zip` | His language results files of October 2026 | R16, R17 |
| `data/language_logs/language_eval_all_models_3seed_2026-10-26.zip` | His runs of 2026-10-26: the seed-1234 results files of the front-loading study's nine cells without R8 logs | R23 |
| `data/workshop/ablation_frontloading_x_permissiveness_language_evaluations.csv` | The shot count and metric of each language task, and Table 31's language scores without Comma0.1 | R13-language |

An input that is missing is downloaded into the local data folder the first
time a command needs it, and the local copy is used from then on. The build
records the SHA-256 of every input as a check, and fails if one is not the
pinned value. It also stops if the two Protocol A
files do not cover exactly the same questions on every task, or if a
checkpoint appears in both. The snapshot is what makes the build reproducible
offline: Hugging Face repos can be overwritten, and the snapshot records what
they held when it was taken. Its date is in the manifest.

When a new record file adds checkpoints, `mvpt snapshot-checkpoints --update`
fetches only those checkpoints and the parents they name. The records already
in the snapshot are kept as they are, so the rest of the manifest does not
change. Each update is logged in the snapshot with its date and the repos it
added.

## What the file contains

| Section | Contents |
|---|---|
| `about` | What the file is, the rules it follows and what is out of scope |
| `sources` | Each input with its version and SHA-256 |
| `protocols` | Protocol A and the Phase 2 three-seed protocol: decoding, seeds, answers per question, and for each task the number of questions and the metric. Excluded tasks are listed with the reason |
| `checkpoints` | One entry per checkpoint, keyed by its folder or repo name: its Hugging Face repos, weight hashes, upload date, context settings, training data where the training summary names it, its parent, how the parent was established, and the labels the records give it |
| `results` | One entry per checkpoint, protocol and task: mean, SD, the three per-seed scores, the number of questions and the source record |
| `suite_averages` | One entry per checkpoint and protocol: the mean over the protocol's tasks for each seed, then the mean and SD over seeds. A checkpoint missing a task gets no average, and the entry lists the missing tasks |
| `changes` | Single-step changes: a checkpoint's suite average minus its parent's, under one protocol, over the same tasks, with the change on each task |
| `rejected_changes` | Pairings the records' labels imply but that cannot be computed, with the reason, such as a checkpoint that was trained from a different model than the row it is paired with |
| `designs` | Which checkpoint fills each cell or row. `front_loading_study` covers the four web substrates of the revision's front-loading study (see [frontloading.md](frontloading.md)), with notes on the two arms that come from earlier runs, and its `language_cells`, which are not always the cell's reasoning checkpoint. `phase1_zoo` covers Tables 1 and 6, `older_tables` Tables 7, 8, 9 and 15, and `language_tables` Tables 10 to 14 and 16 |
| `checks` | The checks run during the build and their outcome |

## How a parent is established

Three kinds of evidence are used, strongest first:

1. **Training summary.** `run_summary.json` in the model repo names the model
   the training run loaded. The OT30K runs have one.
2. **Model card.** The `base_model` field of the repo's card.
3. **Name rule.** The parent a checkpoint's own name implies, for example
   `sft__ot30k_<parent>`, `<parent>-ar30k` or `<parent>-SFT-Tulu3-decontaminated`.

A parent is **verified** when a training summary or a model card names it and
the two agree. It is **name only** when only the name gives it, and
**conflict** when the evidence disagrees. Two names are the same model when
their weight files have the same SHA-256, so a copy such as
`SmolLM2-1.7B-WithChatTemplate` is matched to the `HuggingFaceTB/SmolLM2-1.7B`
that a model card names.
The paper treats a name-only parent as the parent and marks nothing, at the
authors' request of 2026-10-31. Until then a change against such a parent
carried a dagger. A conflict or a rejected parent still stops the renderers.

## Rules for changes

A change is computed only when:

- the parent is the checkpoint the run was actually trained from,
- both checkpoints have results under the same protocol, and
- both have every task of that protocol.

The records also label each checkpoint with a stage (for example `OT3+SFT` or
`sft`), and the tables pair each row with the row above it. When that pairing
names a different model from the checkpoint's actual parent, the change is
listed in `rejected_changes` instead.

## What the first build found (2026-10-12)

- **1,557 results.** 627 under Protocol A (57 checkpoints, 11 tasks each) and 930
  under the Phase 2 protocol (93 cells, 10 tasks each, before GSM8K was
  re-run). Every Protocol A mean
  and SD equals the values behind the paper's generated tables.
- **Phase 2 records are internally consistent.** On the 7 tasks with
  per-question correctness (AIME24, AIME25, AMC23, MATH500, GPQA-Diamond,
  JEEBench and IFEval), every released per-seed score equals the mean of the
  answers. In 69 cells (68 IFEval, 1 AIME24) the two files list the same three
  values in a different seed order. Means and SDs are unaffected.
- **Parents.** 65 are verified, 69 come from the name only, and 9 checkpoints
  have none recorded: the pretrained bases and external releases. 58 of the
  143 checkpoints are not on Hugging Face, mostly the runs on AceReason,
  Nemotron-Cascade and Superior-Reasoning, the 100B OT30K runs and the two
  Qwen-Instruct OT30K runs, so their parents come from their names.
- **97 single-step changes** were computed.
- **3 pairings were rejected:**
  - FineWeb-Edu-16K without front-loading, base to SFT, under Protocol A. The
    SFT model was trained from the April 16K base, not the August base that
    the row shows.
  - The two Qwen-Instruct OT30K runs. The records label them as OT30K on the
    Instruct models, but their names (`sft__ot30k_Qwen2.5-1.5B` and
    `sft__ot30k_Qwen3-1.7B-Base`) point to the base models, and neither is on
    Hugging Face with a training summary to settle it.
- **Phase 2's Nemotron-CC-HQ** traces back to a separate pretraining run with a
  16K context (`…nemotron-hq-300B-16384-rope_theta-1M`), not to the 4K base
  that Phase 1 uses. Its checkpoint entries show this.

## What adding the Comma arm found (2026-10-12)

- **66 new results** for 6 checkpoints, 11 tasks each: the front-loaded Comma
  base at 4K and 16K, and its SFT and DPO models at each length. The records
  cover the same questions as the main file on every task. Nothing else in
  the manifest changed.
- **5 new single-step changes**, and no rejected pairing. Three parents are
  verified by the model card: 4K SFT to DPO, 16K base to SFT and 16K SFT to
  DPO. Two come from the name only: 4K base to SFT and 4K base to 16K base.
  The 4K base repo was created on Hugging Face two days after its SFT model,
  so the upload dates do not settle the first of these either.
- **Suite averages** (mean over seeds of the 11-task average):

  | Stage | 4K | 16K |
  |---|---|---|
  | Base | 28.3 | 27.9 |
  | SFT | 26.0 | 25.9 |
  | DPO | 27.4 | 26.1 |

- **No matching arm without front-loading.** The other three pretraining
  sources each have a run without front-loading in the same setup. For Comma,
  the only checkpoints without front-loading are Phase 1's `1.7b-Comma0.1-300BT`
  models, which come from a different pretraining run. The manifest records
  both arms but pairs neither with the other.

## Who reads it

| Renderer | What it takes from the manifest |
|---|---|
| `render-zoo` (Tables 1 and 6) | The `phase1_zoo` cells and their per-seed results |
| `render-phase2` (Tables 2, 3, 4 and 32, Figures 1 and 2) | Results, suite averages and changes of the Phase 2 protocol |
| `render-frontloading` (Tables 5 and 28 to 31, Figures 3 to 5) | The `front_loading_study` cells and language cells, and the language results. The contrasts resample the per-question records (R13 and R15), which must give the manifest's per-seed scores |
| `render-older-tables` (Tables 7, 8, 9 and 15) | The Protocol A results and suite averages of their checkpoints: R18 for Tables 7 and 9 and Table 8's rows without decontamination, R13 for Table 8's decontaminated rows and Table 15's rows without masking, and R19 for Table 15's rows with masking. The `older_tables` design names the checkpoint of every row |
| `render-language-suite` (Tables 10 to 14 and 16) | The `language_suite` results of each row's run and the `language_tables` design |
| `render-statistics` (Tables 26 and 27) | The `phase1_zoo` cells, and the per-seed scores of Table 27's checkpoints. The contrasts resample R13 and R18, checked like the front-loading study's |
| `render-language-intervals` (Tables 33 and 34) | Each change must follow the manifest's lineage, and the per-question scores must give its language results |
| `render-setup` (Tables 17 to 25) | The protocols' seeds and temperature |
| `render-reconciliation` (not in the paper) | The Phase 2 results of the rows the revision does not print, their per-seed scores, and the rejected pairings. Everything else it reads from the tables the other renderers wrote |

One part of the paper is built from values the manifest does not hold as
results: the setup tables read `config/paper_setup.toml`.

## What adding the older evaluations and the language suite found (2026-10-17)

- **460 older results** for 42 checkpoints and **1,067 language results** for
  97 runs. Every reasoning result, suite average, change, rejected pairing and
  checkpoint entry of the earlier manifest is unchanged. The new checkpoints'
  entries are built after the others, so that the reasoning checkpoints'
  parents do not depend on them.
- **One language run is keyed by the weights it used, not by its name.** R8's
  FineWeb-Edu 16K run of May 2026 loaded `open-sci/…-300B-4096-longsft_16k`,
  which then held the April 16K weights. The repo switched to the August model
  on 2026-08-18. The run is keyed as the April model
  (`…-300B-4096-4096-longsft_16k`), the parent of the 16K SFT model, so Table
  33's FineWeb-Edu changes follow the lineage.
- **Two names of the Comma0.1 base are one model.** `ontocord/1.7b-Comma0.1-300BT`
  and `ali-elganzory/1.7b-Comma0.1-300BT-WithChatTemplate` have the same
  weights, so Table 33's Comma0.1 context extension starts from either.
- **Table 31's language scores agree across sources.** The 22 cells where
  R13-language and one of Harsh Raj's runs cover the same checkpoint are
  identical.
- **14 language runs load a `patched_models` copy** of the model, and what
  was changed in those copies is not recorded (a `note` check).


## What the Phase 2 checkpoint uploads changed (2026-10-31)

On 2026-10-30 open-sci uploaded the Phase 2 checkpoints that were not on
Hugging Face, and Marianna Nezhurina listed them in the collection
`open-sci/mv-post-training-17b-checkpoints`. `mvpt snapshot-checkpoints
--update` now looks up again every name it found no repository for, and it
found 58 of the 59. The 59th, `SmolLM2-1.7B-Instruct-16k`, is the SmolLM2
authors' own release, `HuggingFaceTB/SmolLM2-1.7B-Instruct-16k`. The lookup has
found it since a name that no team repository has is also tried, by its exact
name, under the organizations of the external releases (`EXTERNAL_ORGS`). Its
OT30K run's training summary starts from a folder of that name, and the run
keeps that release's `rope_theta` of 500,000. All 190 checkpoints of the
manifest are on Hugging Face.
The rebuilt manifest changes in these ways, and in no result, suite average or
change value:

- **82 parents are verified** that came from the name only. The new repositories have training summaries that name them, and these agree with the name rule. 22 parents still come from the name only, and 20 checkpoints have none recorded.
- **One change is new.** The 16K SFT checkpoint with prompt tokens masked has its parent from its model card, so its Protocol A change from the 16K extended checkpoint is computed. No table prints it.
- **The two Qwen OT30K runs** were trained from the base models, `Qwen/Qwen2.5-1.5B` and `Qwen/Qwen3-1.7B-Base`, as their training summaries say. The records pair them with the Instruct models, so both pairings stay rejected, now for a recorded reason.
- **A copy is not its own parent.** The card of `ali-elganzory/1.7b-MixtureVitae-300BT-v1-16k-WithChatTemplate` names `ontocord/1.7b-MixtureVitae-300BT-v1-long_instruct`, which has the same weights, so the lineage now ignores evidence that names the checkpoint's own weights and keeps the name rule's parent.
