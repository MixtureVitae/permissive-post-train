"""How Phase 2 answers end: cut off at the token limit, or without an answer the grader finds.

Reads R14-lengths (`manifest/answer_lengths.py`), one row per sampled answer.

An answer counts as cut off when its length, re-encoded with its checkpoint's
tokenizer, is at least LIMIT_SHARE of the limit. Re-encoding gives back the
generated length for most answers of a well-trained model, but sampled
answers that degenerate into repetition often use token splits the
tokenizer would not choose, and re-encode up to a few percent shorter. Below
the limit, the answers that ended on their own spread evenly, and the cut-off
answers form a peak in the last few percent (`length_shares`, and
docs/phase2_analyses.md). Answers too short in bytes to reach the threshold
were not re-encoded, and count as not cut off.

"No answer found" is the share of answers whose extracted answer
(`model_answer`) is empty, for the tasks whose grader compares an extracted
answer with the reference and can find none. The records score every such
answer wrong, except in cells where the column was not filled in: there every
answer's is empty, or an empty one is scored correct, and the cell has no
share. GSM8K and
GPQA-Diamond are left out, because their graders take an answer from every
response. For the code tasks it is the share with no code block
(`model_answer_extracted` empty), which the checker then replaces with the
whole answer. IFEval has neither.
"""
from __future__ import annotations

import pandas as pd

#: The share of the token limit from which a re-encoded answer counts as cut off.
LIMIT_SHARE = 0.95
#: Record task names whose grader extracts an answer.
ANSWER_TASKS = ("AIME24", "AIME25", "AMC23", "MATH500", "JEEBench")
CODE_TASKS = ("HumanEval", "LiveCodeBenchv5", "MBPP")
CELL = ["base_model", "stage", "task"]
ANSWER = ["base_model", "stage", "task", "example_idx", "run_idx"]


def cut_off(lengths: pd.DataFrame) -> pd.Series:
    """Whether each answer reached its token limit."""
    return lengths["n_tokens"].fillna(0) >= LIMIT_SHARE * lengths["max_tokens"]


def unrecorded(lengths: pd.DataFrame, verdicts: pd.DataFrame) -> pd.Index:
    """The (base_model, stage, task) cells of ANSWER_TASKS whose `model_answer`
    column does not hold the grader's answer: an answer with an empty one is
    scored correct, or every answer's is empty. `verdicts` has the ANSWER
    columns and `is_correct` (R14-meta)."""
    answers = lengths[lengths["task"].isin(ANSWER_TASKS)]
    graded = answers[answers["no_answer"]]
    hit = graded.merge(verdicts[ANSWER + ["is_correct"]].drop_duplicates(ANSWER), on=ANSWER)
    correct = hit.loc[hit["is_correct"] > 0, CELL].drop_duplicates()
    empty = answers.groupby(CELL)["no_answer"].all()
    every = empty[empty].index.to_frame(index=False)
    return pd.MultiIndex.from_frame(pd.concat([correct, every]).drop_duplicates())


def shares(lengths: pd.DataFrame, verdicts: pd.DataFrame | None = None) -> pd.DataFrame:
    """One row per (base_model, stage, task): `answers`, and in percent
    `cut_off`, `no_answer` (ANSWER_TASKS, else NaN) and `no_code` (CODE_TASKS,
    else NaN), each over the three runs' answers together. With `verdicts`,
    `no_answer` is NaN for the cells of `unrecorded`. `no_code` is NaN for a
    cell in which no answer has its extracted code, which means the column was
    not recorded: MixtureVitae's Tulu3 SFT checkpoint on HumanEval and MBPP,
    where its DPO checkpoint has it for every answer."""
    frame = lengths.assign(cut_off=cut_off(lengths))
    out = (frame.groupby(CELL, observed=True)
           .agg(answers=("cut_off", "size"), cut_off=("cut_off", "mean"),
                no_answer=("no_answer", "mean"), no_code=("no_extracted", "mean"))
           .reset_index())
    out[["cut_off", "no_answer", "no_code"]] *= 100.0
    out.loc[~out["task"].isin(ANSWER_TASKS), "no_answer"] = float("nan")
    out.loc[~out["task"].isin(CODE_TASKS), "no_code"] = float("nan")
    out.loc[out["no_code"] == 100.0, "no_code"] = float("nan")
    if verdicts is not None:
        bad = out.set_index(CELL).index.isin(unrecorded(lengths, verdicts))
        out.loc[bad, "no_answer"] = float("nan")
    return out


def length_shares(lengths: pd.DataFrame, start: float = 0.80) -> pd.DataFrame:
    """How many re-encoded answers end in each 1% of the token limit from
    `start` up, for each limit: the evidence for LIMIT_SHARE."""
    frame = lengths.dropna(subset=["n_tokens"])
    share = frame["n_tokens"] / frame["max_tokens"]
    edges = [round(start + i / 100, 2) for i in range(round((1 - start) * 100) + 1)] + [float("inf")]
    binned = pd.cut(share, edges, right=False, labels=[f"{e:.2f}" for e in edges[:-1]])
    out = (frame.assign(share_from=binned).dropna(subset=["share_from"])
           .groupby(["max_tokens", "share_from"], observed=False).size().reset_index(name="answers"))
    out["max_tokens"] = out["max_tokens"].astype(int)
    return out
