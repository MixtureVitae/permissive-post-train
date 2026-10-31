"""Protocol A per-sample records: the input behind the workshop paper's Phase 1
tables and the 300B front-loading factorial.

Protocol A is the greedy protocol with three decoding seeds (42, 1234, 2025).
Every Phase 1 number in the workshop paper, and in the camera-ready built from
it, is a mean over these seeds. The records come from Ali Elganzory's artifacts
repository, pinned to one commit (see README "Inputs"):

  canonical_reasoning_per_sample.csv
      one row per (checkpoint, seed, benchmark, item), `correct` in [0, 1]
      (JEEBench carries partial credit in steps of 0.25)
  ablation_frontloading_x_permissiveness_language_evaluations.csv
      one row per (checkpoint, task) for the language suite, single seed

The loaders below validate both files against the experimental design before
anything is computed, so a missing seed, a missing item or a duplicated
checkpoint stops the pipeline instead of shifting a mean.

Protocol B (the OT30K reasoning post-training protocol) has no per-sample
records in that repository; its values live in `protocol_b.py` and must never
be mixed with these.
"""
from __future__ import annotations

import functools
import hashlib
import json
import zipfile
from pathlib import Path

import numpy as np
import pandas as pd

from .. import inputs
from .language_samples import LOG_ZIPS

#: Commit of github.com/Ali-Elganzory/permissive-llms-artifacts the inputs and
#: the paper's generated .tex tables are taken from.
SOURCE_REPO = "https://github.com/Ali-Elganzory/permissive-llms-artifacts"
SOURCE_COMMIT = "8cc2e51ba5ac16ea3110f74252d6992a4853b4a0"

#: The two files, pinned in mv_post_train/inputs.json as R13 and R13-language.
REASONING_CSV = inputs.get("R13").file
LANGUAGE_CSV = inputs.get("R13-language").file
REASONING_CSV_SHA256 = inputs.get("R13").sha256
LANGUAGE_CSV_SHA256 = inputs.get("R13-language").sha256

#: The front-loaded Comma arm (`comma-mv_reasoning`): six checkpoints, 4K and
#: 16K, base, SFT and DPO, in the same format and on the same questions as
#: REASONING_CSV. Received on 2026-10-12 as a separate file. It is not in the
#: artifacts repository at SOURCE_COMMIT, so its home is the project bucket,
#: and the manifest commands download it from there on first use.
COMMA_CSV = inputs.get("R15").file
COMMA_CSV_SHA256 = inputs.get("R15").sha256
COMMA_CSV_URI = inputs.get("R15").gcs

#: Every Protocol A record file the results manifest reads, as
#: (source name, input ID in mv_post_train/inputs.json). The first is the
#: reference for the question set: every other file must cover exactly the
#: same questions.
RECORD_FILES = (
    ("protocol_a_records", "R13"),
    ("protocol_a_comma_records", "R15"),
    ("protocol_a_older_records", "R18"),
    ("protocol_a_masked_records", "R19"),
    ("protocol_a_fineweb_april_records", "R20"),
    ("protocol_a_dclm_16k_records", "R24"),
)
#: The older experiments' Protocol A records (input R18): the 100B mixture
#: ablation, the non-decontaminated MV and the 0.4B comparison. Their labels
#: put several checkpoints in one cell (all 100B variants are "MV"), so each is
#: known by its model_id, and they belong to no experiment of R13.
OLDER_CSV = inputs.get("R18").file
#: The masked MV checkpoints' Protocol A records (input R19), Table 15's rows
#: with masking. Their cells repeat those of the main MV models in R13, so the
#: manifest keys them by name.
MASKED_CSV = inputs.get("R19").file
#: How load_reasoning checks each record file. R18 puts several checkpoints in
#: one cell, and R19 holds post-trained checkpoints only, which belong to no
#: experiment of R13 and repeat the cells of its main MV models.
LOAD_OPTIONS = {"R13": {}, "R15": {}, "R18": {"one_checkpoint_per_cell": False},
                "R19": {"one_checkpoint_per_cell": False, "stages": ("sft", "dpo")},
                "R20": {"stages": ("base",)}, "R24": {}}
#: Record files whose checkpoint takes the place of one of R13's in Phase 1 and
#: the factorial: input -> the R13 checkpoint it replaces. R20 holds the April
#: FineWeb-Edu 16K base, the model that row's SFT and DPO checkpoints were
#: trained from. R13 holds the August 16K model, a second run of the same
#: extension, which is kept in the manifest but in no experiment.
FINEWEB_APRIL_CSV = inputs.get("R20").file
REPLACEMENTS = {"R20": "ali-elganzory/open-sci-ref-v0.02-1.7b-fineweb-edu-1.4t-300B-4096-longsft_16k"}
#: Record files whose checkpoints join the revision's Phase 1 in cells R13 does not
#: have. R24, received on 2026-10-28, holds the DCLM 16K checkpoints (base,
#: Tulu3 SFT and DPO). R13 has DCLM at 4K only. The workshop paper does not use
#: them.
DCLM_16K_CSV = inputs.get("R24").file
ADDITIONS = ("R24",)
#: Record files that hold their checkpoints' strict IFEval verdicts and
#: LiveCodeBench v5 verdicts (task LCB_V5_TASK) themselves, because R21 and R22
#: have none for them. Their IFEval rows are strict (R24's author confirmed it on
#: 2026-10-29, and that its GSM8K uses flexible extract), and their v5 rows
#: take the place of their v2 rows, as R21's and R22's do for the other files.
OWN_VERDICTS = ("R24",)
#: Protocol A IFEval verdicts with the strict checker (input R21), received
#: 2026-10-24: one row per checkpoint, seed and instruction. The record files
#: above score IFEval with the loose checker. The revision scores it strict,
#: as Protocol B does since 2026-10-20 (Table 24), so every reader of the
#: revision's records takes its IFEval verdicts from R21 (`with_strict_ifeval`).
#: The workshop paper keeps R13's loose verdicts.
IFEVAL_STRICT_CSV = inputs.get("R21").file
#: Checkpoints of the record files that R21 has no verdicts for, with the
#: reason. The revision's records leave out their IFEval rows, so they have no
#: IFEval result and no suite average.
IFEVAL_STRICT_MISSING = {
    REPLACEMENTS["R20"]: "R20 replaces it in every experiment, so no table uses it",
}
#: Protocol A LiveCodeBench v5 verdicts (input R22), received 2026-10-26 from
#: Ali Elganzory: one row per checkpoint, seed and problem. The record files
#: above run LiveCodeBench v2 (511 problems). The revision runs v5, on the 369
#: problems that Protocol B runs, so every reader of the revision's records
#: takes its LiveCodeBench verdicts from R22 (`with_lcb_v5`). The workshop
#: paper keeps R13's v2 verdicts.
LCB_V5_CSV = inputs.get("R22").file
LCB_V5_TASK = "LiveCodeBenchv5"
#: Checkpoints of the record files that R22 has no verdicts for, with the
#: reason. The revision's records leave out their LiveCodeBench rows, so they
#: have no LiveCodeBench result and no suite average.
LCB_V5_MISSING = dict(IFEVAL_STRICT_MISSING)
#: Checkpoints that a record file names by a copy of another repo: the name in
#: the records -> the repo the manifest keys them under. R19 evaluated copies
#: of the `laion` masked models, uploaded on 2026-10-18, whose model.safetensors
#: has the same SHA-256 on Hugging Face. Table 16's language runs (R17) name
#: the `laion` repos, so both tables print one checkpoint.
_MASKED = "1.7b-MixtureVitae-300BT-v1-decontaminated-{}-Tulu3-decontaminated-masked"
RECORD_IDENTITY = {f"ali-elganzory/{_MASKED.format(v)}-laion-eval": f"laion/{_MASKED.format(v)}"
                   for v in ("SFT", "DPO", "16k-SFT", "16k-DPO")}

SEEDS = (42, 1234, 2025)
STAGES = ("base", "sft", "dpo")
CONTEXTS = (4096, 16384)

#: Suite order, abbreviation, scored items and domain for the eleven reasoning
#: benchmarks. IFEval is instruction-level over the 500-prompt subset (769
#: instructions), loose in the record files and strict in R21, HumanEval is the
#: 164 Python problems, and LiveCodeBench is v5's 369 problems (R22). The record
#: files run LiveCodeBench v2, with 511 problems (RECORD_ITEMS).
BENCHMARKS = (
    "AIME24", "AIME25", "AMC23", "MATH500", "gsm8k",
    "HumanEval", "LiveCodeBench", "MBPP",
    "GPQADiamond", "JEEBench",
    "IFEval",
)
BENCH_ABBR = {
    "AIME24": "A24", "AIME25": "A25", "AMC23": "AMC", "MATH500": "M500",
    "gsm8k": "GSM8K", "HumanEval": "HE", "LiveCodeBench": "LCB", "MBPP": "MBPP",
    "GPQADiamond": "GPQA", "JEEBench": "JEE", "IFEval": "IFE",
}
BENCH_ITEMS = {
    "AIME24": 30, "AIME25": 30, "AMC23": 40, "MATH500": 500, "gsm8k": 1319,
    "HumanEval": 164, "LiveCodeBench": 369, "MBPP": 500,
    "GPQADiamond": 198, "JEEBench": 515, "IFEval": 769,
}
RECORD_ITEMS = {**BENCH_ITEMS, "LiveCodeBench": 511}
DOMAIN = {
    "AIME24": "math", "AIME25": "math", "AMC23": "math", "MATH500": "math",
    "gsm8k": "math",
    "HumanEval": "code", "LiveCodeBench": "code", "MBPP": "code",
    "GPQADiamond": "science", "JEEBench": "science",
    "IFEval": "instruction",
}
DOMAINS = ("math", "code", "science", "instruction")

#: The eleven-task language suite. The raw file also carries social_iqa, which
#: the paper does not report.
LANGUAGE_TASKS = (
    "mmlu", "hellaswag", "commonsense_qa", "arc_challenge", "arc_easy", "piqa",
    "boolq", "winogrande", "openbookqa", "copa", "lambada_openai",
)
LANGUAGE_ABBR = {
    "mmlu": "MMLU", "hellaswag": "HellaS", "commonsense_qa": "CSQA",
    "arc_challenge": "ARC-C", "arc_easy": "ARC-E", "piqa": "PIQA",
    "boolq": "BoolQ", "winogrande": "WinoG", "openbookqa": "OBQA",
    "copa": "COPA", "lambada_openai": "LAMBADA",
}

#: Web substrates of the 3 x 2 factorial. MV is the permissive one.
POOLS = ("MV", "Nemotron", "FineWeb-Edu")
POOL_LABEL = {"MV": "MixtureVitae", "Nemotron": "Nemotron-CC-HQ",
              "FineWeb-Edu": "FineWeb-Edu", "Comma": "Comma0.1"}
PERMISSIVE = ("MV",)

#: The front-loading study of the revision: Comma0.1 joins as a fourth web
#: substrate, and a second permissive one. Its front-loaded arm is COMMA_CSV
#: (`comma-mv_reasoning`). No open-sci-ref v0.02 run without front-loading
#: exists for Comma, so its other arm is the Comma0.1 model of Phase 1
#: (`ontocord/1.7b-Comma0.1-300BT` and the SFT, DPO and 16K models trained
#: from it). That model was trained in September 2025 with the same recorded
#: open-sci-ref configuration (1.7B, GPT-NeoX tokenizer, 300B tokens, batch
#: 1024, 4K context, rope 100000, WSD, lr 4e-3, 25,000 warmup steps), and the
#: authors kept it for the v0.02 comparisons because its data was not affected
#: by the corruption that led to the v0.02 re-runs. It is still a separate,
#: earlier run, as MV's front-loaded arm is. Decided on 2026-10-13.
FRONTLOADING_POOLS = ("MV", "Comma", "Nemotron", "FineWeb-Edu")
FRONTLOADING_PERMISSIVE = ("MV", "Comma")
COMMA_CONTROL_GROUP = "Comma0.1"

#: Language scores for the Comma0.1 arms. LANGUAGE_CSV has none, so they are
#: read from Harsh Raj's lm-eval results files, in two zips that live in the
#: project bucket:
#:   LANGUAGE_RESULTS_ZIP (input R16), his results of 2026-10-16 (2.9 MB),
#:     for the front-loaded arm. Downloaded into the data folder on first
#:     use, and its SHA-256 is checked on every read. It replaces the zip of
#:     2026-10-14, whose front-loaded Comma0.1 results files are the same.
#:   LANGUAGE_LOGS_ZIP (input R8), his rerun of May 2026 (2.8 GB), for the
#:     arm without front-loading. Its SHA-256 is checked when it is
#:     downloaded. Hashing it takes minutes, so every read checks
#:     LANGUAGE_LOGS_MEMBERS_SHA256 instead, a digest of the results files
#:     read from it (logs_members_digest).
#: LANGUAGE_CSV's runs fix the settings: one shot count and one metric per
#: task, and the prompt-example seed 1234 (the `default` folder of the May
#: rerun). load_comma_language checks both, because the zip of 2026-10-14 had
#: 23 runs with 0 shots on every task.
LANGUAGE_RESULTS_ZIP = inputs.get("R16").file
LANGUAGE_RESULTS_ZIP_SHA256 = inputs.get("R16").sha256
LANGUAGE_RESULTS_ZIP_URI = inputs.get("R16").gcs
LANGUAGE_LOGS_ZIP, LANGUAGE_LOGS_ZIP_SHA256, LANGUAGE_LOGS_ZIP_URI = next(
    z for z in LOG_ZIPS if z[0] == "language_eval-logs.zip")
LANGUAGE_LOGS_MEMBERS_SHA256 = inputs.member_digest("R8", "comma_language")
FEWSHOT_SEED = 1234

#: (front_loading, context_length) -> (zip, folder in the zip, the checkpoint
#: its results files name, the checkpoint the cell's reasoning records use).
#: Each pair has identical weights on Hugging Face (model.safetensors SHA-256
#: 0f610737..., f7408060..., 84e775e2... and 33c576b9..., checked 2026-10-14).
_MAY_LOGS = "e/data1/datasets/playground/mmlaion/shared/oellm_shared_evals/raj3/logs/"
COMMA_LANGUAGE_CELLS = {
    (True, 4096): (LANGUAGE_RESULTS_ZIP, "raw_json/",
                   "open-sci/open-sci-ref-v0.02-1.7b-comma-mv_reasoning-300B-4096",
                   "ali-elganzory/open-sci-ref-v0.02-1.7b-comma-mv_reasoning-300B-4096"),
    (True, 16384): (LANGUAGE_RESULTS_ZIP, "raw_json/",
                    "open-sci/open-sci-ref-v0.02-1.7b-comma-mv_reasoning-300B-4096-longsft_16k",
                    "ali-elganzory/open-sci-ref-v0.02-1.7b-comma-mv_reasoning-300B-4096-longsft_16k"),
    (False, 4096): (LANGUAGE_LOGS_ZIP, _MAY_LOGS + "ontocord__1.7b-Comma0.1-300BT-default/",
                    "ontocord/1.7b-Comma0.1-300BT",
                    "ali-elganzory/1.7b-Comma0.1-300BT-WithChatTemplate"),
    (False, 16384): (LANGUAGE_LOGS_ZIP,
                     _MAY_LOGS + "ontocord__1.7b-Comma0.1-300BT-longsft_16k-default/",
                     "ontocord/1.7b-Comma0.1-300BT-longsft_16k",
                     "ali-elganzory/1.7b-Comma0.1-300BT-longsft_16k"),
}

#: Model families of Phase 1, in table order. The first five share the
#: 300B-token open-sci-ref setup; the last three are external references.
ZOO_ORDER = ("MV", "Nemotron", "FineWeb-Edu", "DCLM", "Comma0.1",
             "SmolLM2", "Qwen2.5", "Qwen3")
ZOO_MATCHED = ("MV", "Nemotron", "FineWeb-Edu", "DCLM", "Comma0.1")
ZOO_LABEL = {**POOL_LABEL, "DCLM": "DCLM", "Comma0.1": "Comma0.1",
             "SmolLM2": "SmolLM2", "Qwen2.5": "Qwen2.5", "Qwen3": "Qwen3"}

FACTORIAL_KEYS = ["model_group", "front_loading", "context_length", "stage"]
ZOO_KEYS = ["model_group", "context_length", "stage"]


class DesignError(ValueError):
    """The records do not match the experimental design."""


def _require(ok: bool, message: str) -> None:
    if not ok:
        raise DesignError(message)


def _validate_cells(df: pd.DataFrame, name: str, keys: list[str],
                    stages: tuple[str, ...] = STAGES) -> None:
    """Every (cell, seed) scores every item of every benchmark exactly once."""
    _require(not df.isna().any().any(), f"{name}: null values")
    _require(df["correct"].between(0.0, 1.0).all(), f"{name}: `correct` outside [0, 1]")
    _require(sorted(df["seed"].unique()) == sorted(SEEDS),
             f"{name}: seeds {sorted(df['seed'].unique())}, expected {SEEDS}")
    _require(set(df["benchmark"]) == set(BENCHMARKS), f"{name}: benchmark set differs")
    _require(set(df["stage"]) == set(stages), f"{name}: stage set differs")
    items = df.groupby("benchmark")["sample_id"].nunique()
    # The record files run LiveCodeBench v2, and the revision's records v5
    # (with_lcb_v5), so the expected counts follow the version the frame holds.
    expected = BENCH_ITEMS if items["LiveCodeBench"] == BENCH_ITEMS["LiveCodeBench"] \
        else RECORD_ITEMS
    for bench, n in expected.items():
        _require(items[bench] == n, f"{name}: {bench} has {items[bench]} items, expected {n}")
    dup = df.duplicated(subset=keys + ["seed", "benchmark", "sample_id"]).sum()
    _require(dup == 0, f"{name}: {dup} duplicated (cell, seed, item) rows")
    rows = df.groupby(keys + ["seed"], observed=True).size()
    bad = rows[rows != sum(expected.values())]
    _require(bad.empty, f"{name}: {len(bad)} (cell, seed) groups are incomplete\n{bad.head()}")


def load_reasoning(path: str | Path, *, one_checkpoint_per_cell: bool = True,
                   stages: tuple[str, ...] = STAGES) -> pd.DataFrame:
    """Read and validate the per-sample CSV. Keeps `model_id` for tracing.
    With `one_checkpoint_per_cell` False (R18 and R19, see LOAD_OPTIONS), a
    cell may hold several checkpoints, which are then told apart by model_id,
    and a checkpoint need not belong to an experiment. `stages` is the set of
    stages the file must hold. `path` may also be a frame of a record file's
    rows (split_own_verdicts)."""
    df = _read(path)
    _validate_cells(df, "reasoning", FACTORIAL_KEYS if one_checkpoint_per_cell else ["model_id"],
                    stages)
    for col in ("front_loading", "in_factorial", "in_zoo"):
        _require(df[col].dtype == bool, f"reasoning: `{col}` is not boolean")
    _require(df.groupby("model_id")[FACTORIAL_KEYS].nunique().eq(1).all().all(),
             "reasoning: one checkpoint maps to several cells")
    if one_checkpoint_per_cell:
        _require(df.groupby(FACTORIAL_KEYS)["model_id"].nunique().eq(1).all(),
                 "reasoning: one cell maps to several checkpoints")
        _require((df["in_factorial"] | df["in_zoo"]).all(),
                 "reasoning: a checkpoint belongs to no experiment")
    return df


def with_replacements(reasoning: pd.DataFrame, extra: dict[str, pd.DataFrame]) -> pd.DataFrame:
    """R13's records (`reasoning`) with each checkpoint of REPLACEMENTS taken out
    of Phase 1 and the factorial, and the records that replace it (`extra`,
    input -> records) added in its cell."""
    out = reasoning.copy()
    for input_id, model_id in REPLACEMENTS.items():
        old = out["model_id"] == model_id
        new = extra[input_id]
        _require(old.any(), f"replacements: R13 has no records of {model_id}")
        cell = lambda d: d[FACTORIAL_KEYS].drop_duplicates().to_dict("records")
        _require(cell(out[old]) == cell(new),
                 f"replacements: {input_id} is not in the cell of {model_id}")
        out.loc[old, ["in_zoo", "in_factorial"]] = False
        out = pd.concat([out, new], ignore_index=True)
    return out


def with_additions(reasoning: pd.DataFrame, extra: dict[str, pd.DataFrame]) -> pd.DataFrame:
    """`reasoning` with the records of each file of ADDITIONS (`extra`, input ->
    records) added, in cells where `reasoning` has no checkpoint of Phase 1 or
    the factorial."""
    out = reasoning
    for input_id in ADDITIONS:
        new = extra[input_id]
        taken = out.loc[out["in_zoo"] | out["in_factorial"], FACTORIAL_KEYS].drop_duplicates()
        clash = new[FACTORIAL_KEYS].drop_duplicates().merge(taken)
        _require(clash.empty, f"additions: {input_id} has cells that are already filled\n{clash}")
        out = pd.concat([out, new], ignore_index=True)
    return out


def _read(source: str | Path | pd.DataFrame, usecols: list[str] | None = None,
          **kwargs) -> pd.DataFrame:
    """A record file, or a frame of a record file's rows (split_own_verdicts)."""
    if isinstance(source, pd.DataFrame):
        return (source[usecols] if usecols else source).reset_index(drop=True)
    return pd.read_csv(source, usecols=usecols, **kwargs)


def split_own_verdicts(path: str | Path) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """A record file of OWN_VERDICTS as three frames: its records with
    LiveCodeBench v2, for load_reasoning, and its IFEval and LiveCodeBench v5
    rows, for load_ifeval_strict and load_lcb_v5, which check them as they check
    R21 and R22."""
    df = pd.read_csv(path, dtype={"sample_id": str})
    v5 = df["benchmark"] == LCB_V5_TASK
    return df[~v5], df[df["benchmark"] == "IFEval"], df[v5]


def load_ifeval_strict(path: str | Path | pd.DataFrame) -> pd.DataFrame:
    """Read and validate R21: IFEval only, the three seeds, and every
    instruction once per checkpoint and seed, scored 0 or 1."""
    df = _read(path, usecols=["model_id", "stage", "seed", "benchmark", "sample_id", "correct"])
    _require(not df.isna().any().any(), "ifeval strict: null values")
    _require(set(df["benchmark"]) == {"IFEval"}, "ifeval strict: tasks other than IFEval")
    _require(df["correct"].isin([0.0, 1.0]).all(), "ifeval strict: `correct` is not 0 or 1")
    _require(sorted(df["seed"].unique()) == sorted(SEEDS),
             f"ifeval strict: seeds {sorted(df['seed'].unique())}, expected {SEEDS}")
    _require(df.groupby("model_id")["stage"].nunique().eq(1).all(),
             "ifeval strict: one checkpoint has several stages")
    keys = ["model_id", "seed", "sample_id"]
    _require(not df.duplicated(subset=keys).any(), "ifeval strict: duplicated rows")
    counts = df.groupby(["model_id", "seed"])["sample_id"].nunique()
    _require(counts.eq(BENCH_ITEMS["IFEval"]).all(),
             f"ifeval strict: a checkpoint and seed do not have {BENCH_ITEMS['IFEval']} "
             "instructions")
    return df


def with_strict_ifeval(records: pd.DataFrame, strict: pd.DataFrame) -> pd.DataFrame:
    """`records` (load_reasoning) with every IFEval verdict replaced by its
    strict one from `strict` (load_ifeval_strict), on the same instructions.
    The IFEval rows of the checkpoints in IFEVAL_STRICT_MISSING are left out.
    Every other checkpoint must have a strict verdict for every seed and
    instruction."""
    keys = ["model_id", "seed", "sample_id"]
    drop = (records["benchmark"] == "IFEval") & records["model_id"].isin(IFEVAL_STRICT_MISSING)
    out = records[~drop].reset_index(drop=True)
    ife = out["benchmark"] == "IFEval"
    lookup = strict.set_index(keys)
    items = pd.MultiIndex.from_frame(out.loc[ife, keys])
    found = items.isin(lookup.index)
    no_verdict = sorted(out.loc[ife, "model_id"][~found].unique())
    _require(not no_verdict, f"ifeval strict: no strict verdicts for {no_verdict}")
    _require((lookup["stage"].reindex(items).to_numpy() == out.loc[ife, "stage"].to_numpy()).all(),
             "ifeval strict: a checkpoint has another stage in R21")
    out.loc[ife, "correct"] = lookup["correct"].reindex(items).to_numpy()
    return out


def load_lcb_v5(path: str | Path | pd.DataFrame) -> pd.DataFrame:
    """Read and validate R22: LiveCodeBench v5 only, the three seeds, and every
    problem once per checkpoint and seed, scored 0 or 1. Problem ids are
    LiveCodeBench's question ids, read as text."""
    df = _read(path, usecols=["model_id", "stage", "seed", "benchmark", "sample_id", "correct"],
               dtype={"sample_id": str})
    _require(not df.isna().any().any(), "lcb v5: null values")
    _require(set(df["benchmark"]) == {LCB_V5_TASK}, f"lcb v5: tasks other than {LCB_V5_TASK}")
    _require(df["correct"].isin([0.0, 1.0]).all(), "lcb v5: `correct` is not 0 or 1")
    _require(sorted(df["seed"].unique()) == sorted(SEEDS),
             f"lcb v5: seeds {sorted(df['seed'].unique())}, expected {SEEDS}")
    _require(df.groupby("model_id")["stage"].nunique().eq(1).all(),
             "lcb v5: one checkpoint has several stages")
    keys = ["model_id", "seed", "sample_id"]
    _require(not df.duplicated(subset=keys).any(), "lcb v5: duplicated rows")
    counts = df.groupby(["model_id", "seed"])["sample_id"].nunique()
    _require(counts.eq(BENCH_ITEMS["LiveCodeBench"]).all(),
             f"lcb v5: a checkpoint and seed do not have {BENCH_ITEMS['LiveCodeBench']} problems")
    _require(df.groupby("model_id")["sample_id"].nunique().eq(BENCH_ITEMS["LiveCodeBench"]).all()
             and df["sample_id"].nunique() == BENCH_ITEMS["LiveCodeBench"],
             "lcb v5: the checkpoints do not share one problem set")
    return df


def with_lcb_v5(records: pd.DataFrame, v5: pd.DataFrame) -> pd.DataFrame:
    """`records` (load_reasoning) with their LiveCodeBench v2 rows replaced by
    the v5 verdicts of `v5` (load_lcb_v5): for each checkpoint, R22's rows for
    every seed and problem, in the checkpoint's own cell. The LiveCodeBench rows
    of the checkpoints in LCB_V5_MISSING are left out. Every other checkpoint
    must have v5 verdicts, at the same stage."""
    lcb = records["benchmark"] == "LiveCodeBench"
    cells = (records.loc[lcb].drop(columns=["seed", "benchmark", "sample_id", "correct"])
             .drop_duplicates())
    cells = cells[~cells["model_id"].isin(LCB_V5_MISSING)]
    _require(cells["model_id"].is_unique, "lcb v5: a checkpoint has several cells")
    no_verdict = sorted(set(cells["model_id"]) - set(v5["model_id"]))
    _require(not no_verdict, f"lcb v5: no v5 verdicts for {no_verdict}")
    new = v5[v5["model_id"].isin(cells["model_id"])].merge(
        cells, on="model_id", suffixes=("_v5", ""), validate="many_to_one")
    _require((new["stage_v5"] == new["stage"]).all(),
             "lcb v5: a checkpoint has another stage in R22")
    new = new.assign(benchmark="LiveCodeBench")[records.columns]
    return pd.concat([records[~lcb], new], ignore_index=True)


def revised_frame(input_id: str, path: str | Path, strict: pd.DataFrame,
                  v5: pd.DataFrame) -> pd.DataFrame:
    """One record file of RECORD_FILES as the revision reads it: loaded with
    its LOAD_OPTIONS, with the strict IFEval verdicts of `strict` (R21) and the
    LiveCodeBench v5 verdicts of `v5` (R22), or, for a file of OWN_VERDICTS,
    with its own."""
    records = path
    if input_id in OWN_VERDICTS:
        records, own_strict, own_v5 = split_own_verdicts(path)
        strict, v5 = load_ifeval_strict(own_strict), load_lcb_v5(own_v5)
    records = load_reasoning(records, **LOAD_OPTIONS[input_id])
    return with_lcb_v5(with_strict_ifeval(records, strict), v5)


def revision_records(input_id: str, data_dir: str | Path) -> pd.DataFrame:
    """One record file of RECORD_FILES as the revision reads it (revised_frame)."""
    data_dir = Path(data_dir)
    return revised_frame(input_id, inputs.path(input_id, data_dir / inputs.get(input_id).file),
                         _ifeval_strict(str(data_dir / IFEVAL_STRICT_CSV)),
                         _lcb_v5(str(data_dir / LCB_V5_CSV)))


@functools.lru_cache(maxsize=2)
def _ifeval_strict(dest: str) -> pd.DataFrame:
    """R21, read once per process. It is never modified."""
    return load_ifeval_strict(inputs.path("R21", dest))


@functools.lru_cache(maxsize=2)
def _lcb_v5(dest: str) -> pd.DataFrame:
    """R22, read once per process. It is never modified."""
    return load_lcb_v5(inputs.path("R22", dest))


def revised_reasoning(data_dir: str | Path) -> pd.DataFrame:
    """R13 with its replacements (REPLACEMENTS) and additions (ADDITIONS), the
    strict IFEval verdicts and the LiveCodeBench v5 verdicts, the records of the
    revision's Phase 1 and factorial. The workshop paper uses R13 as it is."""
    revised = with_replacements(revision_records("R13", data_dir),
                                {i: revision_records(i, data_dir) for i in REPLACEMENTS})
    return with_additions(revised, {i: revision_records(i, data_dir) for i in ADDITIONS})


def factorial(reasoning: pd.DataFrame, pools: tuple[str, ...] = POOLS) -> pd.DataFrame:
    """The substrates x 2 front-loading x 2 contexts x 3 stages design."""
    df = reasoning[reasoning["in_factorial"]]
    _validate_cells(df, "factorial", FACTORIAL_KEYS)
    design = pd.MultiIndex.from_product([pools, (False, True), CONTEXTS, STAGES],
                                        names=FACTORIAL_KEYS)
    found = pd.MultiIndex.from_frame(df[FACTORIAL_KEYS].drop_duplicates())
    _require(found.sort_values().equals(design.sort_values()),
             "factorial: the design is not fully crossed")
    return df


def frontloading(reasoning: pd.DataFrame, comma: pd.DataFrame) -> pd.DataFrame:
    """The revision's 4 x 2 front-loading study: the factorial of `reasoning`,
    the front-loaded Comma arm of `comma`, and Phase 1's Comma0.1 as that arm's
    control (see FRONTLOADING_POOLS)."""
    _require(set(comma["model_group"]) == {"Comma"} and comma["front_loading"].all(),
             "frontloading: the Comma file should hold only the front-loaded Comma arm")
    control = reasoning[reasoning["in_zoo"]
                        & (reasoning["model_group"] == COMMA_CONTROL_GROUP)]
    _require(not control["front_loading"].any(),
             "frontloading: the Comma0.1 control is labelled front-loaded")
    df = pd.concat([reasoning[reasoning["in_factorial"]],
                    control.assign(model_group="Comma", in_factorial=True),
                    comma.assign(in_factorial=True)], ignore_index=True)
    _require(df.groupby(FACTORIAL_KEYS)["model_id"].nunique().eq(1).all(),
             "frontloading: one cell maps to several checkpoints")
    return factorial(df, FRONTLOADING_POOLS)


def zoo(reasoning: pd.DataFrame) -> pd.DataFrame:
    """Phase 1 (base, Tulu3 SFT, DPO) for all models."""
    df = reasoning[reasoning["in_zoo"]]
    _validate_cells(df, "zoo", ZOO_KEYS)
    _require(set(df["model_group"]) == set(ZOO_ORDER), "zoo: model families differ")
    return df


def load_language(path: str | Path) -> pd.DataFrame:
    """Language-suite scores (percent) for the twelve factorial base checkpoints.

    Only the base and post-YaRN 16K checkpoints were evaluated, single seed.
    Factor levels are parsed from the open-sci-ref checkpoint name: a
    `mixturevitae` stem is front-loaded unless it says `noinstruct`, and the
    other two substrates are front-loaded when they say `mv_reasoning`.
    """
    df = pd.read_csv(path)
    df = df[df["task"].isin(LANGUAGE_TASKS)].copy()
    _require(len(df) == 12 * len(LANGUAGE_TASKS), f"language: {len(df)} rows, expected 132")
    name = df["model_name"].str.removeprefix("open-sci/open-sci-ref-v0.02-1.7b-")
    df["context_length"] = np.where(name.str.contains("longsft_16k"), 16384, 4096)
    stem = name.str.replace("-longsft_16k", "", regex=False)
    df["model_group"] = np.select(
        [stem.str.startswith("mixturevitae"), stem.str.startswith("nemotron-hq"),
         stem.str.startswith("fineweb-edu")],
        ["MV", "Nemotron", "FineWeb-Edu"], default="?")
    df["front_loading"] = ~stem.str.contains("noinstruct") & (
        stem.str.contains("mv_reasoning") | stem.str.startswith("mixturevitae"))
    _require((df["model_group"] != "?").all(), "language: unrecognised checkpoint name")
    _require(df.groupby(["model_group", "front_loading", "context_length"]).ngroups == 12,
             "language: the design is not fully crossed")
    df["performance"] = df["performance"] * 100.0
    return df


def language_settings(path: str | Path) -> dict[str, tuple[int, str]]:
    """The shot count and metric of each language task, as LANGUAGE_CSV's runs used them."""
    raw = pd.read_csv(path)
    raw = raw[raw["task"].isin(LANGUAGE_TASKS)]
    _require(raw.groupby("task")[["n_shot", "metric_name"]].nunique().eq(1).all().all(),
             "language: a task was run with several shot counts or metrics")
    first = raw.drop_duplicates("task").set_index("task")
    return {t: (int(first.at[t, "n_shot"]), str(first.at[t, "metric_name"]))
            for t in LANGUAGE_TASKS}


def _shots(result: dict, task: str) -> set[int]:
    """Shot count of `task` in an lm-eval results file. A group such as mmlu
    records none of its own, so it takes its subtasks' count."""
    n = result.get("n-shot", {})
    if task in n:
        return {int(n[task])}
    return {int(v) for k, v in n.items() if k.startswith(task + "_")}


def _results_files(zf: zipfile.ZipFile, folder: str, checkpoint: str):
    """(member, raw bytes, parsed) for every results file of `checkpoint` in `folder`."""
    out = []
    for member in sorted(zf.namelist()):
        if member.startswith(folder) and member.endswith(".json"):
            raw = zf.read(member)
            result = json.loads(raw)
            if result.get("model_name") == checkpoint:
                out.append((member, raw, result))
    return out


def logs_members_digest(files) -> str:
    """Digest of results files read from a zip: SHA-256 over each member's
    name and the SHA-256 of its bytes, in name order."""
    h = hashlib.sha256()
    for member, raw, _ in sorted(files, key=lambda f: f[0]):
        h.update(member.encode() + b"\0" + hashlib.sha256(raw).digest())
    return h.hexdigest()


def load_comma_language(results_zip: str | Path, logs_zip: str | Path,
                        settings: dict[str, tuple[int, str]]) -> pd.DataFrame:
    """Language-suite scores (percent) of the four Comma0.1 cells, in the
    columns of load_language. Each cell is read from COMMA_LANGUAGE_CELLS, and
    every score must use the shot count, metric and prompt-example seed of
    LANGUAGE_CSV (`settings`)."""
    paths = {LANGUAGE_RESULTS_ZIP: Path(results_zip), LANGUAGE_LOGS_ZIP: Path(logs_zip)}
    rows, from_logs = [], []
    for (fl, ctx), (source, folder, checkpoint, _) in COMMA_LANGUAGE_CELLS.items():
        with zipfile.ZipFile(paths[source]) as zf:
            files = _results_files(zf, folder, checkpoint)
        if source == LANGUAGE_LOGS_ZIP:
            from_logs += files
        for task in LANGUAGE_TASKS:
            n_shot, metric = settings[task]
            hits = [r for _, _, r in files if task in r.get("results", {})]
            _require(len(hits) == 1, f"language: {checkpoint} has {len(hits)} results for {task}")
            result = hits[0]
            _require(result["config"].get("fewshot_seed") == FEWSHOT_SEED,
                     f"language: {checkpoint} {task} was run with prompt-example seed "
                     f"{result['config'].get('fewshot_seed')}, not {FEWSHOT_SEED}")
            _require(_shots(result, task) == {n_shot},
                     f"language: {checkpoint} {task} was run with {sorted(_shots(result, task))} "
                     f"shots, not {n_shot}")
            rows.append(dict(model_name=checkpoint, task=task, n_shot=n_shot,
                             performance=float(result["results"][task][metric]) * 100.0,
                             metric_name=metric, context_length=ctx, model_group="Comma",
                             front_loading=fl))
    _require(logs_members_digest(from_logs) == LANGUAGE_LOGS_MEMBERS_SHA256,
             f"language: the results files read from {LANGUAGE_LOGS_ZIP} are not the pinned ones")
    return pd.DataFrame(rows)


def frontloading_language(language: pd.DataFrame, comma: pd.DataFrame,
                          reasoning: pd.DataFrame) -> pd.DataFrame:
    """Language scores of the revision's front-loading study: the twelve
    checkpoints of LANGUAGE_CSV and the four Comma0.1 cells. Each Comma0.1 cell
    must stand for the base checkpoint its reasoning records use
    (`reasoning`, as returned by frontloading)."""
    base = reasoning[(reasoning["model_group"] == "Comma") & (reasoning["stage"] == "base")]
    used = base.groupby(["front_loading", "context_length"])["model_id"].unique()
    for cell, (_, _, checkpoint, model_id) in COMMA_LANGUAGE_CELLS.items():
        _require(list(used.get(cell, [])) == [model_id],
                 f"language: the Comma0.1 cell {cell} reads {checkpoint}, which stands for "
                 f"{model_id}, but its reasoning records use {list(used.get(cell, []))}")
    df = pd.concat([language, comma], ignore_index=True)
    cells = df.groupby(["model_group", "front_loading", "context_length"])["task"]
    _require(cells.ngroups == len(FRONTLOADING_POOLS) * 2 * len(CONTEXTS),
             "language: the front-loading design is not fully crossed")
    _require(cells.apply(lambda t: sorted(t) == sorted(LANGUAGE_TASKS)).all(),
             "language: a checkpoint lacks a task or has one twice")
    return df
