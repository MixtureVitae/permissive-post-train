"""Derive checkpoint identity from `source_dir` rather than trusting `stage`.

The dump's `stage` label is not reliable per-task. For
(MixtureVitae-1.7B, base) the math and gsm8k rows carry

    ...__checkpoints__1.7b-MixtureVitae-300BT-v1-decontaminated-16k

-- the genuine base checkpoint, which reproduces logs.zip to 0.15 pp -- while
HumanEval, IFEval, LiveCodeBenchv5 and MBPP under the *same* label carry

    ...__checkpoints__sft__Superior-Reasoning-SFT-gpt-oss-120b-30k-sharegpt-hf__...

a Superior-Reasoning SFT checkpoint. That is the whole explanation for a
discrepancy previously recorded as an unexplained 16 pp IFEval gap and pinned
by hand in `stats/released_scores.KNOWN_MISLABELLED`.

A hardcoded exclusion list cannot be trusted to stay correct across dump
revisions, and the failure is silent: pair the wrong checkpoint and you get a
plausible number rather than an error. So instead of naming the bad cells, this
module recovers what each cell actually is, and callers drop the ones whose
checkpoint disagrees with the rest of their own (base_model, stage) group.

The rule: within one (base_model, stage), every task should have been evaluated
on one checkpoint. Tasks disagreeing with their group's modal checkpoint are
`consistent=False`. Evaluation *host* differences are ignored -- IFEval was run
from a different filesystem root than the math tasks for every cell, which is
irrelevant -- because only the checkpoint token is compared.
"""
from __future__ import annotations

from collections import Counter, defaultdict
from pathlib import Path

import pandas as pd
import pyarrow.parquet as pq

AUDIT_CSV = "audit/checkpoint_identity.csv"

_COLUMNS = ["base_model", "stage", "task", "source_dir"]

# The dump uses three path layouts. Everything up to and including the marker
# is the evaluation host and scratch path; what follows names the checkpoint.
#
#   ...__checkpoints__<name>                      trained checkpoints
#   ...__models__<name>                           the transfernetx mirror of the same
#   ...__hub__models--Qwen--X__snapshots__<hash>  an unmodified HF release
#
# `__snapshots__` is tried first because a hub path contains neither of the
# other markers, and its hash is what the `checkpoints__` layout records for
# the same model -- so the two spellings normalize to the same token instead of
# every hub-evaluated cell looking like a checkpoint disagreement.
_MARKERS = ("__snapshots__", "checkpoints__", "__models__")


def checkpoint_id(source_dir: str | None) -> str | None:
    """The checkpoint token from a `source_dir`, or None if unrecognizable.

    Two cells evaluated on the same checkpoint from different filesystem roots
    must return the same value here, or every IFEval cell would look wrong:
    IFEval ran from `__p__scratch__laionize__` for every model while the math
    tasks ran from `__e__scratch__jureap59__`.
    """
    if not source_dir:
        return None
    for marker in _MARKERS:
        idx = source_dir.rfind(marker)
        if idx < 0:
            continue
        tail = source_dir[idx + len(marker):].strip("_")
        if tail:
            return tail
    return None


def collect(dump: Path) -> pd.DataFrame:
    """One row per (base_model, stage, task) with its checkpoint token."""
    f = pq.ParquetFile(Path(dump))
    seen: dict[tuple[str, str, str], Counter] = defaultdict(Counter)
    for batch in f.iter_batches(batch_size=4096, columns=_COLUMNS):
        d = batch.to_pydict()
        for bm, st, task, sd in zip(d["base_model"], d["stage"], d["task"],
                                    d["source_dir"]):
            seen[(bm, st, task)][checkpoint_id(sd)] += 1
    rows = []
    for (bm, st, task), c in seen.items():
        ckpt, _ = c.most_common(1)[0]
        rows.append({"base_model": bm, "stage": st, "task": task,
                     "checkpoint": ckpt, "n_distinct_in_cell": len(c)})
    return pd.DataFrame(rows).sort_values(["base_model", "stage", "task"])


def _same_checkpoint(a: str | None, b: str | None) -> bool:
    """Whether two tokens name the same checkpoint.

    Some paths are recorded truncated -- `sft_Qwen3-1-7B_open-thoughts` is the
    same run as `sft_Qwen3-1-7B_open-thoughts_OpenThoughts3-1-2M_30000_...`,
    just cut short. Treating a strict prefix as a disagreement would discard
    good cells. An abbreviated *rename* is not a prefix and stays flagged.
    """
    if a is None or b is None:
        return a is b
    return a == b or a.startswith(b) or b.startswith(a)


def _reference(values: list[str]) -> str:
    """The checkpoint a group should be judged against.

    Values that are prefixes of one another are the same run recorded at
    different truncations, so they are pooled into one family. The winning
    family is the one covering the most cells, and its *longest* spelling
    becomes the reference: judging against a truncated one would let a
    genuinely different run that happens to share the prefix pass unnoticed.
    """
    families: list[list[str]] = []
    for v, _ in Counter(values).most_common():
        for fam in families:
            if _same_checkpoint(v, fam[0]):
                fam.append(v)
                break
        else:
            families.append([v])
    counts = Counter(values)
    best = max(families, key=lambda fam: sum(counts[v] for v in fam))
    return max(best, key=len)


def flag(identity: pd.DataFrame) -> pd.DataFrame:
    """Mark tasks whose checkpoint disagrees with their group's modal one."""
    out = identity.copy()
    modal = {}
    for (bm, st), g in out.groupby(["base_model", "stage"]):
        vals = [c for c in g.checkpoint if c]
        modal[(bm, st)] = _reference(vals) if vals else None
    out["group_checkpoint"] = [modal[(b, s)] for b, s
                               in zip(out.base_model, out.stage)]
    out["consistent"] = [_same_checkpoint(c, g) for c, g
                         in zip(out.checkpoint, out.group_checkpoint)]
    return out


def inconsistent_cells(identity: pd.DataFrame) -> set[tuple[str, str, str]]:
    """(base_model, stage, task) triples that must not be scored as labelled."""
    bad = identity[~identity.consistent]
    return set(zip(bad.base_model, bad.stage, bad.task))


def main(dump: Path, art_dir: Path) -> int:
    dump, art_dir = Path(dump), Path(art_dir)
    if not dump.exists():
        print(f"missing {dump}")
        return 1
    out = flag(collect(dump))
    (art_dir / "audit").mkdir(parents=True, exist_ok=True)
    out.to_csv(art_dir / AUDIT_CSV, index=False)

    bad = out[~out.consistent]
    print(f"{len(out)} (base_model, stage, task) cells; "
          f"{len(bad)} disagree with their group's checkpoint")
    if not bad.empty:
        print("\n=== cells whose stage label does not match their checkpoint ===")
        print(bad[["base_model", "stage", "task", "checkpoint",
                   "group_checkpoint"]].to_string(index=False))
    split = out[out.n_distinct_in_cell > 1]
    if not split.empty:
        print("\n=== cells mixing several checkpoints internally ===")
        print(split[["base_model", "stage", "task",
                     "n_distinct_in_cell"]].to_string(index=False))
    print(f"\nwrote {art_dir / AUDIT_CSV}")
    return 0
