"""Every generated file of the paper: which command makes it, from which inputs,
and where the paper's sources keep their copy.

`mvpt build` runs the TARGETS in dependency order and writes everything under
`results/`. `mvpt build --check` writes into a temporary folder instead and
compares the files byte for byte with the committed ones. `mvpt check-paper`
compares a paper source tree with them, using PAPER_FILES.

Every target reads its inputs through `mv_post_train/inputs.py` (IDs from
inputs.json) and writes only under the build root it is given, so a check
leaves the working tree as it was. Commands run from the repository root,
where the inputs' default places (`data/`, `artifacts/`) are.
"""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Callable

from . import paths


@dataclass(frozen=True)
class Target:
    name: str
    command: str                 # the mvpt command that makes the same files
    inputs: tuple[str, ...]      # input IDs (mv_post_train/inputs.json)
    needs: tuple[str, ...]       # targets whose files this one reads
    out: str                     # the folder or file it writes, under the build root
    run: Callable[[Path], int]   # build root -> exit code


def _manifest(root: Path) -> int:
    from .manifest import build
    return build.main(Path("data/workshop"), Path(paths.PHASE2_3SEED_DIR),
                      Path(paths.HF_CHECKPOINTS), root / paths.MANIFEST)


def _zoo(root: Path) -> int:
    from .render import zoo
    return zoo.main(root / paths.MANIFEST, root / paths.ZOO_DIR)


def _phase2(root: Path) -> int:
    from .render import phase2
    return phase2.main(root / paths.MANIFEST, Path(paths.PHASE2_3SEED_DIR),
                       root / paths.PHASE2_TABLES_DIR, root / paths.PHASE2_FIGURES_DIR,
                       root / paths.PHASE2_ARTIFACTS_DIR)


def _phase2_analyses(root: Path) -> int:
    from .render import phase2_analyses
    return phase2_analyses.main(root / paths.MANIFEST, Path(paths.PHASE2_3SEED_DIR),
                                root / paths.PHASE2_ANALYSES_DIR)


def _frontloading(root: Path) -> int:
    from .render import frontloading
    return frontloading.main(root / paths.MANIFEST, Path("data/workshop"),
                             root / paths.FRONTLOADING_DIR, root / paths.FRONTLOADING_ARTIFACTS_DIR)


def _older(root: Path) -> int:
    from .render import older_tables
    return older_tables.main(root / paths.MANIFEST, root / paths.OLDER_TABLES_DIR)


def _statistics(root: Path) -> int:
    from .render import statistics
    return statistics.main(root / paths.MANIFEST, Path("data/workshop"), root / paths.STATISTICS_DIR)


def _language_suite(root: Path) -> int:
    from .render import language_suite
    return language_suite.main(root / paths.MANIFEST, root / paths.LANGUAGE_SUITE_DIR)


def _setup(root: Path) -> int:
    from .render import setup
    return setup.main(root / paths.MANIFEST, root / paths.SETUP_DIR)


def _language(root: Path) -> int:
    from .render import language_intervals
    return language_intervals.main(root / paths.MANIFEST, Path(paths.LANGUAGE_LOGS_DIR),
                                   root / paths.LANGUAGE_INTERVALS_DIR)


def _reconciliation(root: Path) -> int:
    from . import inputs
    from .render import reconciliation
    if not Path(inputs.get("T-submission").local).exists():
        # The public copy leaves the submitted tables out (release/code/export.py).
        print("  skipped: the submitted tables (input T-submission) are not in this copy")
        return 0
    return reconciliation.main(root / paths.MANIFEST, root, root / paths.RECONCILIATION_DIR)


TARGETS: tuple[Target, ...] = (
    Target("manifest", "mvpt build-manifest",
           ("R13", "R15", "R18", "R19", "R20", "R24", "R21", "R22", "R14-scores", "R14-meta",
            "R14-ifeval", "S1", "R6", "R8", "R8-fineweb", "R16", "R17", "R23", "R13-language"),
           (), paths.MANIFEST, _manifest),
    Target("zoo", "mvpt render-zoo", (), ("manifest",), paths.ZOO_DIR, _zoo),
    Target("phase2", "mvpt render-phase2", ("R14-meta",), ("manifest",),
           str(Path(paths.PHASE2_TABLES_DIR).parent), _phase2),
    Target("phase2-analyses", "mvpt render-phase2-analyses",
           ("R14-scores", "R14-meta", "R14-ifeval", "R14-lengths"), ("manifest",),
           paths.PHASE2_ANALYSES_DIR, _phase2_analyses),
    Target("frontloading", "mvpt render-frontloading", ("R13", "R15", "R20", "R24", "R21", "R22"), ("manifest",),
           paths.FRONTLOADING_DIR, _frontloading),
    Target("older", "mvpt render-older-tables", (), ("manifest",), paths.OLDER_TABLES_DIR, _older),
    Target("statistics", "mvpt render-statistics", ("R13", "R18", "R20", "R24", "R21", "R22"), ("manifest",),
           paths.STATISTICS_DIR, _statistics),
    Target("language-suite", "mvpt render-language-suite", (), ("manifest",),
           paths.LANGUAGE_SUITE_DIR, _language_suite),
    Target("setup", "mvpt render-setup", ("C-setup", "S2-0.4b", "S2-1.7b", "R14-meta", "R13-language"),
           ("manifest",), paths.SETUP_DIR, _setup),
    Target("language", "mvpt render-language-intervals", ("R8", "R8-fineweb", "R23"), ("manifest",),
           paths.LANGUAGE_INTERVALS_DIR, _language),
    Target("reconciliation", "mvpt render-reconciliation", ("T-submission",),
           ("manifest", "zoo", "phase2", "older", "language-suite"),
           paths.RECONCILIATION_DIR, _reconciliation),
)
BY_NAME = {t.name: t for t in TARGETS}


def order(names: list[str] | None = None) -> list[Target]:
    """The targets to run for `names` (all if None), upstream ones first."""
    wanted, out = list(names or BY_NAME), []
    unknown = [n for n in wanted if n not in BY_NAME]
    if unknown:
        raise KeyError(f"No target {unknown[0]!r}. Targets: {', '.join(BY_NAME)}.")

    def visit(name: str) -> None:
        target = BY_NAME[name]
        for up in target.needs:
            visit(up)
        if target not in out:
            out.append(target)

    for name in wanted:
        visit(name)
    return out


@dataclass(frozen=True)
class PaperFile:
    generated: str   # path in this repo
    paper: str       # path in the paper's sources
    item: str        # what it is in the -10 revision
    target: str


def _pf(target: str, generated: str, paper: str, item: str) -> PaperFile:
    return PaperFile(generated, paper, item, target)


#: Every generated file the paper uses, with its place in the paper's sources.
PAPER_FILES: tuple[PaperFile, ...] = (
    _pf("phase2", "results/phase2/figures/figure_pipeline.pdf", "figures/figure_pipeline.pdf", "Figure 1"),
    _pf("zoo", "results/zoo/tables/phase1_zoo.tex", "tables/phase1_zoo.tex", "Table 1"),
    _pf("zoo", "results/zoo/tables/phase1_zoo_full.tex", "tables/phase1_zoo_full.tex", "Table 6"),
    _pf("zoo", "results/zoo/tables/zoo_numbers.tex", "tables/zoo_numbers.tex", r"\pzoo macros"),
    _pf("phase2", "results/phase2/tables/phase2_endpoints_full.tex", "tables/phase2_endpoints_full.tex", "Table 3"),
    _pf("phase2-analyses", "results/phase2_analyses/tables/phase2_endpoints.tex", "tables/phase2_endpoints.tex", "Phase 2 endpoints and changes (main text, since the Table 2 update)"),
    _pf("phase2", "results/phase2/figures/phase2_receptivity.pdf", "figures/phase2_receptivity.pdf", "Figure 3"),
    _pf("phase2", "results/phase2/tables/ot3_pretrain_ablation.tex", "tables/ot3_pretrain_ablation.tex", "Table 4"),
    _pf("phase2", "results/phase2/tables/heldout_reasoning.tex", "tables/heldout_reasoning.tex", "Table 5"),
    _pf("phase2", "results/phase2/tables/heldout_mv_grid.tex", "tables/heldout_mv_grid.tex", "Table 32"),
    _pf("phase2", "results/phase2/tables/phase2_numbers.tex", "tables/phase2_numbers.tex", r"\ptwo macros"),
    _pf("phase2-analyses", "results/phase2_analyses/tables/phase2_change_intervals.tex", "tables/phase2_change_intervals.tex", "Phase 2 changes with intervals (statistics appendix)"),
    _pf("phase2-analyses", "results/phase2_analyses/tables/ot3_exclusion_gains.tex", "tables/ot3_exclusion_gains.tex", "OT3 exclusion gains (statistics appendix)"),
    _pf("phase2-analyses", "results/phase2_analyses/tables/phase2_endpoint_gaps.tex", "tables/phase2_endpoint_gaps.tex", "MV against the references (statistics appendix)"),
    _pf("phase2-analyses", "results/phase2_analyses/tables/phase2_leave_one_out.tex", "tables/phase2_leave_one_out.tex", "Leave-one-out (statistics appendix)"),
    _pf("phase2-analyses", "results/phase2_analyses/tables/phase2_domain_changes.tex", "tables/phase2_domain_changes.tex", "Phase 2 changes by domain (statistics appendix)"),
    _pf("phase2-analyses", "results/phase2_analyses/tables/phase2_cutoff.tex", "tables/phase2_cutoff.tex", "Answers cut off at the token limit (statistics appendix)"),
    _pf("phase2-analyses", "results/phase2_analyses/tables/phase2_no_answer.tex", "tables/phase2_no_answer.tex", "Answers with no final answer (statistics appendix)"),
    _pf("phase2-analyses", "results/phase2_analyses/tables/phase2_analyses_numbers.tex", "tables/phase2_analyses_numbers.tex", r"\ptwoan macros"),
    _pf("frontloading", "results/frontloading/tables/factorial_summary.tex", "tables/factorial_summary.tex", "Table 2"),
    _pf("frontloading", "results/frontloading/figures/factorial_main.pdf", "figures/factorial_main.pdf", "Figure 2"),
    _pf("frontloading", "results/frontloading/tables/contrasts.tex", "tables/contrasts.tex", "Table 28"),
    _pf("frontloading", "results/frontloading/tables/domain_robustness.tex", "tables/domain_robustness.tex", "Table 29"),
    _pf("frontloading", "results/frontloading/figures/frontloading_trajectory.pdf", "figures/frontloading_trajectory.pdf", "Figure 4"),
    _pf("frontloading", "results/frontloading/tables/factorial_perbenchmark.tex", "tables/factorial_perbenchmark.tex", "Table 30"),
    _pf("frontloading", "results/frontloading/tables/factorial_language.tex", "tables/factorial_language.tex", "Table 31"),
    _pf("frontloading", "results/frontloading/figures/frontloading_domains.pdf", "figures/frontloading_domains.pdf", "Figure 5"),
    _pf("frontloading", "results/frontloading/tables/frontloading_numbers.tex", "tables/frontloading_numbers.tex", r"\pfl macros"),
    _pf("older", "results/older/tables/ablation_100b_reasoning.tex", "tables/ablation_100b_reasoning.tex", "Table 7"),
    _pf("older", "results/older/tables/decontamination_reasoning.tex", "tables/decontamination_reasoning.tex", "Table 8"),
    _pf("older", "results/older/tables/smallscale_reasoning.tex", "tables/smallscale_reasoning.tex", "Table 9"),
    _pf("language-suite", "results/language_suite/tables/language_ablation_100b.tex", "tables/language_ablation_100b.tex", "Table 10"),
    _pf("language-suite", "results/language_suite/tables/language_main.tex", "tables/language_main.tex", "Table 11"),
    _pf("language-suite", "results/language_suite/tables/language_decontamination.tex", "tables/language_decontamination.tex", "Table 12"),
    _pf("language-suite", "results/language_suite/tables/language_smallscale.tex", "tables/language_smallscale.tex", "Table 13"),
    _pf("language-suite", "results/language_suite/tables/language_phase2.tex", "tables/language_phase2.tex", "Table 14"),
    _pf("older", "results/older/tables/masking_reasoning.tex", "tables/masking_reasoning.tex", "Table 15"),
    _pf("language-suite", "results/language_suite/tables/masking_language.tex", "tables/masking_language.tex", "Table 16"),
    _pf("language-suite", "results/language_suite/tables/language_suite_numbers.tex", "tables/language_suite_numbers.tex", r"\plang macros"),
    _pf("older", "results/older/tables/older_numbers.tex", "tables/older_numbers.tex", r"\pold macros"),
    _pf("setup", "results/setup/tables/training_schedules.tex", "tables/training_schedules.tex", "Table 17"),
    _pf("setup", "results/setup/tables/opensciref_scales.tex", "tables/opensciref_scales.tex", "Table 18"),
    _pf("setup", "results/setup/tables/ot30k_sft_hyperparams.tex", "tables/ot30k_sft_hyperparams.tex", "Table 19"),
    _pf("setup", "results/setup/tables/ot30k_sft_gpu_hours.tex", "tables/ot30k_sft_gpu_hours.tex", "Table 20"),
    _pf("setup", "results/setup/tables/long_ctx_gpu_hours.tex", "tables/long_ctx_gpu_hours.tex", "Table 21"),
    _pf("setup", "results/setup/tables/budget.tex", "tables/budget.tex", "Table 22"),
    _pf("setup", "results/setup/tables/eval_settings_a.tex", "tables/eval_settings_a.tex", "Table 23"),
    _pf("setup", "results/setup/tables/eval_settings_b.tex", "tables/eval_settings_b.tex", "Table 24"),
    _pf("setup", "results/setup/tables/eval_settings_general.tex", "tables/eval_settings_general.tex", "Table 25"),
    _pf("setup", "results/setup/tables/setup_numbers.tex", "tables/setup_numbers.tex", r"\psetup macros"),
    _pf("statistics", "results/statistics/tables/stat_matched_compute.tex", "tables/stat_matched_compute.tex", "Table 26"),
    _pf("statistics", "results/statistics/tables/stat_mixture_ablation.tex", "tables/stat_mixture_ablation.tex", "Table 27"),
    _pf("statistics", "results/statistics/tables/statistics_numbers.tex", "tables/statistics_numbers.tex", r"\pstat macros"),
    _pf("language", "results/language/tables/language_frontloading.tex", "tables/language_frontloading.tex", "Table 28"),
    _pf("language", "results/language/tables/language_changes.tex", "tables/language_changes.tex", "Table 33"),
    _pf("language", "results/language/tables/language_changes_mv.tex", "tables/language_changes_mv.tex", "Table 34"),
    _pf("language", "results/language/tables/language_numbers.tex", "tables/language_numbers.tex", r"\langci macros"),
)

#: Paper files allowed to differ from the generated ones, with the reason.
#: It is meant to stay empty.
PAPER_ALLOWED: dict[str, str] = {}

#: Why a paper number changes, for `mvpt check-paper --numbers`: (paper file,
#: pattern) -> reason. The pattern is matched, shell-style, against "row /
#: column" for a table cell or the key for a macro, and the first that matches
#: gives the reason. A pull request that changes numbers adds its reasons here,
#: and the check fails for a changed number without one. Entries are removed
#: once the paper uses the new numbers.
REASONS: dict[tuple[str, str], str] = {}
