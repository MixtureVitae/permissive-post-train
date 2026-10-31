"""Re-score HumanEval and MBPP from the released response dump.

Both benchmarks ship in the LAION dump as raw generations with an empty
`gold_answer`, so `parse_phase2.py` cannot score them. Unlike IFEval, their
verdicts require *executing* the generated code against the benchmark's test
suite. That is what this module does.

**It runs model-generated code.** The execution path is evalchemy's, which is
OpenAI's `human_eval` harness: each candidate runs in its own forked process,
inside a throwaway temp directory, under `reliability_guard()` (which disables
`os.system`, `subprocess`, `shutil.rmtree`, and the rest of the destructive
surface), with `swallow_io()` and a wall-clock `time_limit`. That is the
standard sandbox for this benchmark, and it is not a substitute for running the
whole thing in a container if you do not trust the generations.

Extraction and scoring are deliberately *not* reimplemented — both are vendored
from evalchemy under `data/code_exec/`, because the conventions are where the
numbers actually come from:

  * HumanEval uses `extract_generation_code`, which takes the **first**
    ```python block in the response, re-prepends the function prefix, and falls
    back to `prompt + output` when extraction fails. For a reasoning model that
    drafts code inside `<think>`, "first block" can be a discarded draft — that
    is evalchemy's behaviour and so it is reproduced here rather than improved.
  * MBPP uses a bare `re.findall(r"```python\\n(.*?)```", ...)[0]` with the full
    completion as fallback.

Writes `artifacts/code_rescored.csv`.
"""
from __future__ import annotations

import json
import re
import tempfile
from pathlib import Path

import pandas as pd
import pyarrow.parquet as pq

from ..data.code_exec.human_eval import execution as _execution
from ..data.code_exec.human_eval.evaluation import evaluate_functional_correctness
from ..data.code_exec.utils.utils import extract_generation_code

# `reliability_guard` accepts a memory cap but evalchemy's `check_correctness`
# calls it with none, so a generated program that allocates without bound is
# limited only by system RAM. In practice that means the OOM killer takes out
# the *parent* and the whole run dies silently mid-cell, which is exactly what
# happened on the first MBPP attempt. Bind a default here instead of editing
# the vendored file: the guard runs inside each forked child, so the RLIMIT_AS
# / RLIMIT_DATA caps land on the untrusted code and nothing else.
MEMORY_LIMIT_BYTES = 2 * 1024**3

_unbounded_reliability_guard = _execution.reliability_guard


def _bounded_reliability_guard(maximum_memory_bytes: int | None = None) -> None:
    _unbounded_reliability_guard(maximum_memory_bytes or MEMORY_LIMIT_BYTES)


_execution.reliability_guard = _bounded_reliability_guard

RESCORED_CSV = "code_rescored.csv"

_RESULT_COLUMNS = ["base_model", "stage", "task", "n_rows",
                   "n_problems_matched", "n_problems_expected", "pass@1",
                   "table1", "delta", "status"]

DATA_DIR = Path(__file__).resolve().parent.parent / "data" / "code_exec" / "data"
HUMANEVAL_PROBLEMS = DATA_DIR / "humaneval-python.jsonl"
MBPP_PROBLEMS = DATA_DIR / "mbpp_test.jsonl"

_COLUMNS = ["base_model", "stage", "task", "example_id", "model_output"]

# Submission Table 1, transcribed from page 6. HE and MBPP columns.
TABLE1: dict[tuple[str, str], dict[str, float]] = {
    ("MixtureVitae-1.7B", "base"):      {"HumanEval": 48.2, "MBPP": 41.0},
    ("MixtureVitae-1.7B", "OT3+SFT"):   {"HumanEval": 58.5, "MBPP": 29.2},
    ("MixtureVitae-1.7B", "OT3+DPO"):   {"HumanEval": 46.3, "MBPP": 40.8},
    ("MixtureVitae-1.7B", "OT3"):       {"HumanEval": 57.3, "MBPP": 31.6},
    ("FineWebEdu-1.7B", "SFT-Tulu3"):   {"HumanEval": 2.4,  "MBPP": 2.2},
    ("FineWebEdu-1.7B", "OT3+SFT"):     {"HumanEval": 0.0,  "MBPP": 0.0},
    ("NemotronHQ-1.7B", "SFT-Tulu3"):   {"HumanEval": 1.5,  "MBPP": 0.2},
    ("NemotronHQ-1.7B", "OT3+SFT"):     {"HumanEval": 0.0,  "MBPP": 0.2},
    ("Comma0.1-1.7B", "SFT-Tulu3"):     {"HumanEval": 12.8, "MBPP": 23.6},
    ("Comma0.1-1.7B", "OT3+SFT"):       {"HumanEval": 1.8,  "MBPP": 0.0},
    ("SmolLM2-1.7B-16k-Base", "SFT-Tulu3"): {"HumanEval": 28.7, "MBPP": 37.4},
    ("Qwen2.5-1.5B-Base", "SFT-Tulu3"): {"HumanEval": 48.2, "MBPP": 44.6},
    ("Qwen2.5-1.5B-Base", "OT3+SFT"):   {"HumanEval": 20.7, "MBPP": 8.4},
    ("Qwen3-1.7B-Base", "SFT-Tulu3"):   {"HumanEval": 64.6, "MBPP": 55.8},
    ("Qwen3-1.7B-Base", "OT3+SFT"):     {"HumanEval": 7.3,  "MBPP": 10.6},
}


def _load_problems(path: Path) -> dict:
    with open(path) as fh:
        return {json.loads(l)["task_id"]: json.loads(l) for l in fh if l.strip()}


# The dump identifies HumanEval problems as `HumanEval_<n>_<function_name>`
# while the vendored problem file keys them `Python/<n>`. Same 164 problems,
# two spellings; without this the join matches almost nothing and every cell
# fails the coverage gate as if the data were missing. (It is worth being
# explicit that this was the actual cause of the "6 of 164" reading, not a
# short release.)
_HUMANEVAL_ID = re.compile(r"^HumanEval[_/](\d+)(?:_(.+))?$")


def _function_name(prompt: str) -> str | None:
    """Name of the last function defined in a HumanEval prompt."""
    names = re.findall(r"^\s*def\s+([A-Za-z_]\w*)", prompt, re.MULTILINE)
    return names[-1] if names else None


def normalize_humaneval_id(raw: object, problems: dict) -> object | None:
    """Map a dump `example_id` onto the problem file's key, or None.

    The trailing function name in the dump's id is checked against the problem
    text rather than trusted: if the release ever renumbers, a silent
    off-by-one would score every completion against the wrong tests and still
    produce a plausible number.
    """
    if raw in problems:
        return raw
    m = _HUMANEVAL_ID.match(str(raw))
    if not m:
        return None
    key = f"Python/{int(m.group(1))}"
    if key not in problems:
        return None
    expected = m.group(2)
    if expected:
        actual = _function_name(problems[key].get("prompt", ""))
        if actual and actual != expected:
            raise ValueError(
                f"HumanEval id mismatch: dump {raw!r} maps to {key} whose "
                f"function is {actual!r}, not {expected!r}. The release and the "
                f"vendored problem file disagree; refusing to score.")
    return key


def _extract_mbpp(completion: str) -> str:
    """evalchemy's MBPP extraction, verbatim in behaviour."""
    try:
        return re.findall(r"```python\n(.*?)```", completion, re.DOTALL | re.IGNORECASE)[0]
    except Exception:
        return completion


def collect(dump: Path, task: str) -> dict[tuple[str, str], list[dict]]:
    """(base_model, stage) -> [{task_id, prompt, output}] for one task."""
    f = pq.ParquetFile(dump)
    cells: dict[tuple[str, str], list[dict]] = {}
    for batch in f.iter_batches(batch_size=2048, columns=_COLUMNS):
        tasks = batch.column("task").to_pylist()
        if task not in tasks:
            continue
        bm = batch.column("base_model").to_pylist()
        st = batch.column("stage").to_pylist()
        eid = batch.column("example_id").to_pylist()
        out = batch.column("model_output").to_pylist()
        for t, m, s, e, o in zip(tasks, bm, st, eid, out):
            if t != task:
                continue
            cells.setdefault((m, s), []).append({"example_id": e, "output": o or ""})
    return cells


# A cell is only scored when nearly all of the benchmark's problems are
# present. Without this guard an incomplete cell still yields a plausible-
# looking pass@1 over whatever survived: the dump's HumanEval carries just 6
# of the 164 Python problems, which scored as clean multiples of 1/6 and would
# have been quoted as if they were the real thing.
MIN_COVERAGE = 0.95

# Coverage is necessary but not sufficient. The released HumanEval generations
# are reasoning traces, not programs: in the two cells checked by hand, only
# 7/164 and 6/164 responses contain a `def` at all -- one cell runs to a median
# of 60,000 characters and truncates mid-thought, the other terminates cleanly
# having written prose. Both still clear MIN_COVERAGE, because the *problems*
# are all present; what is missing is code. Without this second gate such a
# cell is reported as `scored` at 0.6%, which reads as a real measurement of a
# model that Table 1 puts at 48.2.
#
# A cell that genuinely answers the benchmark has a program in nearly every
# response, so the threshold is deliberately far below any plausible real
# value and only fires on wholesale absence.
MIN_EXTRACTABLE = 0.5


def _looks_like_a_program(text: str) -> bool:
    return "def " in text


def score_cell(task: str, rows: list[dict], problems: dict, tmp: Path,
               timeout: float, n_workers: int, detail_out: dict | None = None
               ) -> tuple[float | None, int, int, str]:
    """(pass@1, n_matched, n_expected, status) for one (model, stage) cell.

    pass@1 is None whenever the cell cannot be scored honestly — either
    coverage falls below `MIN_COVERAGE`, or fewer than `MIN_EXTRACTABLE` of the
    responses contain a program at all. An unscorable cell is reported as
    unscorable, never as a number over a subset or over empty text.
    """
    samples = []
    for r in rows:
        raw = r["example_id"]
        if task == "HumanEval":
            task_id = normalize_humaneval_id(raw, problems)
            if task_id is None:
                continue
        else:
            task_id = raw if raw in problems else None
            if task_id is None:
                # MBPP task_ids are ints in the problem file, strings in the dump.
                try:
                    task_id = int(raw)
                except (TypeError, ValueError):
                    continue
                if task_id not in problems:
                    continue
        prob = problems[task_id]
        if task == "HumanEval":
            ex = extract_generation_code(
                {"task_id": task_id, "prompt": prob["prompt"], "output": r["output"]},
                lang_code="python",
            )
            generation = ex["generation"]
            # `process_humaneval_test` reads `prompt` off the sample, not off
            # the problem — evalchemy carries the whole example through, so
            # the field has to be present here too.
            # Keep the model's own text: the extractability gate below cannot
            # look at `generation`, because evalchemy falls back to
            # `prompt + output` when extraction fails, and the prompt always
            # contains the problem's own `def`. Checking the generation would
            # therefore pass every cell, which is exactly what it did.
            samples.append({"task_id": task_id, "prompt": prob["prompt"],
                            "generation": generation, "_raw": r["output"]})
            continue
        generation = _extract_mbpp(r["output"])
        samples.append({"task_id": task_id, "generation": generation,
                        "_raw": r["output"]})

    n_matched = len({s["task_id"] for s in samples})
    n_expected = len(problems)
    if not samples or n_matched < MIN_COVERAGE * n_expected:
        return None, n_matched, n_expected, "insufficient_coverage"

    n_program = sum(1 for s in samples if _looks_like_a_program(s["_raw"]))
    if n_program < MIN_EXTRACTABLE * len(samples):
        print(f"      only {n_program}/{len(samples)} responses contain a "
              f"program — not scoring")
        return None, n_matched, n_expected, "no_extractable_code"

    infile = tmp / "samples.jsonl"
    with open(infile, "w") as fh:
        for s in samples:
            # `_raw` is ours, for the gate above; the harness must not see it,
            # and these traces run to tens of kB apiece.
            fh.write(json.dumps({k: v for k, v in s.items()
                                 if k != "_raw"}) + "\n")

    result = evaluate_functional_correctness(
        input_file=str(infile),
        tmp_dir=str(tmp),
        n_workers=n_workers,
        timeout=timeout,
        problem_file=str(HUMANEVAL_PROBLEMS if task == "HumanEval" else MBPP_PROBLEMS),
        k=[1],
        is_mbpp=(task == "MBPP"),
        language="python",
        detail_out=detail_out,
    )
    for key in ("pass@1", "python_pass@1"):
        if key in result:
            return float(result[key]) * 100, n_matched, n_expected, "scored"
    return None, n_matched, n_expected, "no_metric_returned"


def _problem_number(task_id: str) -> int | None:
    """The numeric index inside a harness task_id, for use as `item_index`.

    HumanEval ids are `Python/<n>`; MBPP's are the problem number itself.
    Items must key on something stable across cells, or the pairing that a
    within-model transition test depends on has nothing to join on.
    """
    s = str(task_id)
    tail = s.rsplit("/", 1)[-1]
    return int(tail) if tail.lstrip("-").isdigit() else None


_EXCLUDED_CACHE: dict[str, set] = {}


def _flush_items(data_dir: Path, task: str, verdicts: list[dict],
                 dump: Path, keep_models: set[str]) -> None:
    """Write everything scored for `task` so far.

    Called after every cell, so it must not re-scan the dump each time: the
    checkpoint audit is a full pass over a 1.2 GB file and would dwarf the
    scoring it is protecting.
    """
    from ..data import recover_items

    key = str(dump)
    if key not in _EXCLUDED_CACHE:
        _EXCLUDED_CACHE[key] = recover_items.excluded_cells(dump)
    item_rows = recover_items.to_item_rows(
        pd.DataFrame(verdicts), task, _EXCLUDED_CACHE[key])
    if item_rows:
        recover_items.merge(data_dir, task, item_rows, keep_models)


def main(dump: Path, art_dir: Path, timeout: float = 10.0,
         n_workers: int = 8, tasks: tuple[str, ...] = ("HumanEval", "MBPP"),
         data_dir: Path | None = None, resume: bool = False) -> int:
    dump, art_dir = Path(dump), Path(art_dir)
    if not dump.exists():
        print(f"missing {dump}")
        return 1
    art_dir.mkdir(parents=True, exist_ok=True)

    problems = {
        "HumanEval": _load_problems(HUMANEVAL_PROBLEMS),
        "MBPP": _load_problems(MBPP_PROBLEMS),
    }

    # Scoring one task must not discard the other's rows: each cell costs
    # minutes of sandboxed execution, so a re-run of HumanEval alone keeps the
    # MBPP rows already on disk rather than blanking them.
    rows = []
    done: set[tuple[str, str, str]] = set()
    existing = art_dir / RESCORED_CSV
    if existing.exists() and (resume or set(tasks) != {"HumanEval", "MBPP"}):
        prior = pd.read_csv(existing)
        kept = prior[~prior.task.isin(tasks)]
        if not kept.empty:
            print(f"keeping {len(kept)} prior rows for "
                  f"{sorted(set(kept.task))} from {existing}")
            rows.extend(kept.to_dict("records"))
        if resume:
            # Cells this run would otherwise recompute. A cell that errored is
            # deliberately NOT treated as done: the whole point of resuming is
            # to get past whatever killed the previous attempt.
            redo = prior[prior.task.isin(tasks)]
            redo = redo[redo.status != "error"]
            rows.extend(redo.to_dict("records"))
            done = set(zip(redo.base_model, redo.stage, redo.task))
            if done:
                print(f"resuming: {len(done)} cells already scored, "
                      f"{len(prior) - len(redo) - len(kept)} errored and will retry")

    from ..data.recover_items import FAMILY as _FAMILY, STAGE as _STAGE
    from ..data.recover_items import _model_name as _canonical_model

    verdicts: dict[str, list[dict]] = {t: [] for t in tasks}
    for task in tasks:
        cells = collect(dump, task)
        print(f"{task}: {len(cells)} cells")
        # Item rows this run must not overwrite: the ones a resumed run is
        # skipping because they are already on disk.
        skipped = {_canonical_model(bm, st) for bm, st, t in done
                   if t == task and bm in _FAMILY and st in _STAGE}
        for (bm, st), items in sorted(cells.items()):
            if (bm, st, task) in done:
                print(f"  {bm:24s} {st:10s} already scored, skipping")
                continue
            # A cell that blows up must not take the run with it — record the
            # failure and continue, so 37 good cells survive one bad one.
            detail: dict = {}
            try:
                with tempfile.TemporaryDirectory() as td:
                    acc, matched, expected, status = score_cell(
                        task, items, problems[task], Path(td), timeout,
                        n_workers, detail_out=detail)
            except Exception as exc:
                print(f"  {bm:24s} {st:10s} FAILED: {type(exc).__name__}: {exc}")
                acc, matched, expected, status = None, -1, len(problems[task]), "error"
            # Only a cell the scorer stood behind contributes items. An
            # unscorable cell's per-problem verdicts are the same fiction as
            # its pass@1, and pairing against them would launder that.
            if data_dir is not None and status == "scored":
                for task_id, frac in detail.items():
                    idx = _problem_number(task_id)
                    if idx is None:
                        continue
                    verdicts[task].append({
                        "base_model": bm, "stage": st, "item_index": idx,
                        "attempt_index": 0, "score": float(frac)})
            reported = TABLE1.get((bm, st), {}).get(task)
            rows.append({
                "base_model": bm, "stage": st, "task": task,
                "n_rows": len(items),
                "n_problems_matched": matched,
                "n_problems_expected": expected,
                "pass@1": None if acc is None else round(acc, 2),
                "table1": reported,
                "delta": (None if acc is None or reported is None
                          else round(acc - reported, 2)),
                "status": status,
            })
            shown = "unscorable" if acc is None else round(acc, 2)
            print(f"  {bm:24s} {st:10s} pass@1={shown}"
                  f"  ({matched}/{expected} problems)  table1={reported}")

            # Checkpoint after every cell. A full MBPP pass is 39 cells at
            # minutes apiece, and holding all of it in memory until the loop
            # ends means a crash at cell 38 throws away everything. Both
            # writes are small enough to be free next to the execution itself.
            pd.DataFrame(rows, columns=_RESULT_COLUMNS).to_csv(
                art_dir / RESCORED_CSV, index=False)
            if data_dir is not None and verdicts[task]:
                _flush_items(data_dir, task, verdicts[task], dump, skipped)

    out = pd.DataFrame(rows, columns=_RESULT_COLUMNS)
    if out.empty:
        # No cells matched -- usually a wrong dump path or a task name the
        # release does not carry. Write a header-only file so downstream
        # readers still parse it, rather than a zero-byte file that raises.
        print("no cells scored")
        out.to_csv(art_dir / RESCORED_CSV, index=False)
        return 0
    out = out.sort_values(["task", "base_model", "stage"])
    out.to_csv(art_dir / RESCORED_CSV, index=False)
    print("\n" + out.to_string(index=False))

    if data_dir is not None:
        # The per-cell checkpoint already wrote these; this pass only reports
        # what landed, reusing the cached audit rather than re-scanning 1.2 GB.
        from ..data import recover_items
        excluded = _EXCLUDED_CACHE.get(str(dump), set())
        for task in tasks:
            if not verdicts[task]:
                print(f"\nno per-problem verdicts to emit for {task}")
                continue
            item_rows = recover_items.to_item_rows(
                pd.DataFrame(verdicts[task]), task, excluded)
            recover_items.report(item_rows, task, excluded)
            print(f"wrote {len(item_rows)} {task} item rows to "
                  f"{Path(data_dir) / 'recovered_items.parquet'}")

    for task in sorted(out.task.unique()):
        t = out[out.task == task]
        bad = t[t.status == "insufficient_coverage"]
        if not bad.empty:
            # Deliberately phrased as "matched", not "the dump carries": a low
            # match count means the join failed, and the join failing because
            # the two sides spell task_ids differently looks identical to the
            # release being short. Reading this message as the latter is what
            # produced the false "HumanEval ships 6 of 164 problems" claim.
            print(f"\n{task}: {len(bad)}/{len(t)} cells unscorable — only "
                  f"{bad.n_problems_matched.max()} of "
                  f"{bad.n_problems_expected.max()} problems matched the "
                  f"problem file. Check the id join before concluding the "
                  f"release is incomplete.")

        nocode = t[t.status == "no_extractable_code"]
        if not nocode.empty:
            print(f"\n{task}: {len(nocode)}/{len(t)} cells carry the problems "
                  f"but not the answers — fewer than "
                  f"{MIN_EXTRACTABLE:.0%} of responses contain a program. "
                  f"These generations cannot reproduce the paper's numbers "
                  f"under any extraction convention; the released responses "
                  f"are not the ones the table was computed from.")

    cmp = out.dropna(subset=["delta"])
    if not cmp.empty:
        print(f"\nvs Table 1: mean |Δ| = {cmp.delta.abs().mean():.2f} pp, "
              f"median |Δ| = {cmp.delta.abs().median():.2f} pp, "
              f"within 1 pp: {(cmp.delta.abs() <= 1.0).sum()}/{len(cmp)}")
    print(f"\nwrote {art_dir / RESCORED_CSV}")
    return 0
