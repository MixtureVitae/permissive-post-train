"""Canonical filenames for every artifact the pipeline reads or writes.

One source of truth. If you change a filename here, every reader/writer
picks it up. No hardcoded basenames elsewhere in the package.

Parent directories (e.g., `./data/`, `./artifacts/`) come from the CLI's
`--data-dir` / `--art-dir` flags; this module owns only the basenames.

Grouped by layer:

  Data-layer parquets/CSVs        ← produced by data/parse_phase1.py,
                                    data/parse_phase2.py
  Stats-layer CSVs                ← produced by stats/analyze.py
  Render-layer outputs (defaults) ← produced by render/* and the cli.py
                                    `--out` defaults

Constants are plain `str` (basenames). Use them as `data_dir / paths.NAME`
or `Path(paths.NAME)` at call sites.
"""
from __future__ import annotations


# ---------------------------------------------------------------------------
# Data-layer artifacts (under --data-dir; default ./data/)

ITEMS_LONG:           str = "items_long.parquet"
AGGREGATES:           str = "aggregates.parquet"
PHASE2_ITEMS:         str = "phase2_items.parquet"

# Phase 2 sourced from logs.zip rather than the HF dump. Separate files so
# the HF-sourced Phase 2 pipeline (which feeds the paper tables) is
# untouched; these carry all eleven reasoning benchmarks, the HF dump only
# carries seven. See data/parse_phase2_logs.py.
PHASE2_LOGS_ITEMS:       str = "phase2_logs_items.parquet"
PHASE2_LOGS_AGGREGATES:  str = "phase2_logs_aggregates.parquet"

# Phase 1 item rows recovered from the HF dump for benchmarks logs.zip carries
# only as aggregates. Without these the Phase 1 -> OT30K transition has nothing
# to pair a Phase 2 gsm8k cell against, so the benchmark drops out of the
# paired analysis despite both ends existing. See data/parse_phase1_dump.py.
PHASE1_DUMP_ITEMS:       str = "phase1_dump_items.parquet"

# Item rows for benchmarks the dump ships without verdicts, recovered by
# re-scoring its raw generations (IFEval via the official checker, MBPP via
# execution). Covers Phase 1 and Phase 2 stages together, because neither end
# exists at item level anywhere else. See data/recover_items.py.
RECOVERED_ITEMS:         str = "recovered_items.parquet"


# ---------------------------------------------------------------------------
# Stats-layer artifacts (under --art-dir; default ./artifacts/).
#
# Organized into per-section subdirectories that mirror the paper:
#   matched_compute/  -- Phase 1 matched-compute pairwise + suite (§3.1, feeds
#                        statsig_matched_compute + statsig_suite_average tables)
#   ablation/         -- §3.3 mixture-composition ablation (feeds
#                        statsig_mixture_ablation table)
#   audit/            -- file-selection + pairing diagnostics (cross-section)

# Matched compute (Phase 1).
PER_TASK_ACCURACY:              str = "matched_compute/per_task_accuracy.csv"
PAIRWISE_FULL_MATRIX:           str = "matched_compute/pairwise_full_matrix.csv"
PAIRWISE_MV_VS_COMMA:           str = "matched_compute/pairwise_mv_vs_comma.csv"
PAIRWISE_MV_VS_NEMOTRON:        str = "matched_compute/pairwise_mv_vs_nemotron.csv"
PAIRWISE_MV_VS_FINEWEB:         str = "matched_compute/pairwise_mv_vs_fineweb.csv"
PAIRWISE_COMMA_VS_NEMOTRON:     str = "matched_compute/pairwise_comma_vs_nemotron.csv"
SUITE_AVG_DIFF_MATCHED:         str = "matched_compute/suite_avg_diff_matched.csv"
SUITE_AVERAGE_CI:               str = "matched_compute/suite_average_ci.csv"

# Mixture-composition ablation (§3.3).
PAIRWISE_ABLATION_FULL_MATRIX:  str = "ablation/pairwise_ablation_full_matrix.csv"
PAIRWISE_MVNOWEB_VS_MVNOINST:   str = "ablation/pairwise_mvnoweb_vs_mvnoinst.csv"
PAIRWISE_MVNOWEB_VS_MV100:      str = "ablation/pairwise_mvnoweb_vs_mv100.csv"
PAIRWISE_MVNOINST_VS_MV100:     str = "ablation/pairwise_mvnoinst_vs_mv100.csv"
SUITE_AVG_DIFF_ABLATION:        str = "ablation/suite_avg_diff_ablation.csv"

# Audit / diagnostic logs (cross-section).
PAIRING_AUDIT:               str = "audit/pairing_audit.csv"
FILE_SELECTION_LOG:          str = "audit/file_selection_log.csv"
PHASE2_LOGS_FILE_SELECTION_LOG: str = "audit/phase2_logs_file_selection_log.csv"


# ---------------------------------------------------------------------------
# Render-layer outputs (CLI defaults, relative to the repository root)

# Workshop paper (render-workshop): the eleven generated tables and five
# figures keep the paper's own basenames inside these directories, so
# check-workshop can pair them with the paper's files by name.
WORKSHOP_TABLES_DIR:         str = "tables/workshop"
WORKSHOP_FIGURES_DIR:        str = "figures/workshop"
WORKSHOP_ARTIFACTS_DIR:      str = "artifacts/workshop"

# Results manifest (build-manifest). The manifest itself is committed: it is
# the authoritative set of reasoning results. Its inputs live under data/.
PHASE2_3SEED_DIR:            str = "data/phase2_3seed"
# The other reasoning corpora's questions against OpenThoughts3
# (stats/question_overlap.py). OVERLAP_OT3_RAW_DIR holds the downloaded OT3
# shards, which the reference questions are read from once.
QUESTION_OVERLAP_DIR:        str = "data/overlap"
OVERLAP_OT3_RAW_DIR:         str = "data/overlap/ot3_raw"
HF_CHECKPOINTS:              str = "data/manifest/hf_checkpoints.json"
#: logs.zip without the GPQA Diamond items, input R6 (mv_post_train/inputs.json):
#: the older evaluations.
LOGS_ZIP:                    str = "artifacts/logs_without_gpqa_items.zip"
MANIFEST:                    str = "results/manifest.json"

# Phase 2 tables and Figure 2 rendered from the manifest (render-phase2). The
# files keep the paper's basenames, so they replace the hand-copied ones. They
# are committed, like the manifest, so a change to any number shows up in
# review, with a compiled preview.pdf next to the two folders.
PHASE2_TABLES_DIR:           str = "results/phase2/tables"
PHASE2_FIGURES_DIR:          str = "results/phase2/figures"
PHASE2_ARTIFACTS_DIR:        str = "artifacts/phase2"
# The Phase 2 analyses N1 to N4 of docs/phase2_analyses.md (render/phase2_analyses.py).
PHASE2_ANALYSES_DIR:         str = "results/phase2_analyses"

# The revision's front-loading study with Comma0.1 as a fourth web substrate
# (render-frontloading). The rendered tables and figures are committed, like
# the manifest, so a change to any number shows up in review.
FRONTLOADING_DIR:            str = "results/frontloading"
FRONTLOADING_ARTIFACTS_DIR:  str = "artifacts/frontloading"

# Language-suite changes with question-level intervals
# (render-language-intervals). The two log zips are downloaded into
# LANGUAGE_LOGS_DIR from the project bucket if missing, and the parsed
# per-question scores are cached there. The rendered files are committed.
LANGUAGE_LOGS_DIR:           str = "data/language_logs"
LANGUAGE_INTERVALS_DIR:      str = "results/language"

# The statistics appendix, Tables 26 and 27 and their text macros
# (render-statistics), committed like the other rendered results.
STATISTICS_DIR:              str = "results/statistics"

# Tables 7, 8, 9 and 15 from the older evaluations, and their text macros
# (render-older-tables), committed like the other rendered results.
OLDER_TABLES_DIR:            str = "results/older"
# Tables 1 and 6 (Phase 1 for all models) and their text macros, `mvpt render-zoo`.
ZOO_DIR:                     str = "results/zoo"
# Tables 10 to 14 and 16 (the language suite) and their text macros,
# `mvpt render-language-suite`.
LANGUAGE_SUITE_DIR:          str = "results/language_suite"
# Tables 17 to 25 (setup and evaluation settings) and their text macros,
# `mvpt render-setup`.
SETUP_DIR:                   str = "results/setup"
# Every cell of the submitted paper's results tables against the revision,
# `mvpt render-reconciliation`. Not part of the paper.
RECONCILIATION_DIR:          str = "results/reconciliation"
