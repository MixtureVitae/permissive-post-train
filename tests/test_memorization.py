"""The memorization probes' scoring and comparisons."""
from __future__ import annotations

import numpy as np
import pandas as pd

from mv_post_train.stats import memorization as mem
from mv_post_train.stats import overlap_sensitivity as osens

CHOICES = ["mouth", "grocery cart", "cart", "shop", "bag"]


def test_a_continuation_recalls_the_choices_it_lists():
    assert mem.recalled("\n- Mouth\n-  grocery  cart\n- trolley\n- bag\n- cart\n- shop", CHOICES) == 4
    assert mem.recalled(None, CHOICES) == 0


def test_a_reply_picks_the_choice_it_names_last():
    assert mem.picked("A cart? No: the answer is grocery cart.", CHOICES) == 1
    assert mem.picked("Bags go in the cart.", CHOICES) == 2
    assert mem.picked("No idea.", CHOICES) == -1
    assert mem._named("bag", "a bag.") and not mem._named("bag", "baggage")


def test_metrics_per_probe():
    p = pd.DataFrame(dict(id=["o", "a"], kind=["options", "answer"], split="validation",
                          item=["q", "q"], context_item=["q", "q"], context="", target="",
                          choices=[[], CHOICES], gold=[-1, 1], generate=[64, 96]))
    s = pd.DataFrame(dict(id=["o", "a"], target_logprob=[-6.0, None], target_tokens=[12, None],
                          target_argmax=[12, None], target_prefix=[12, None],
                          choice_logprob=[[], [-3.0, -1.0, -2.0, -5.0, -5.0]],
                          choice_tokens=[[], [1, 2, 1, 1, 1]],
                          generation=["\n- mouth\n- shop", "It is the cart."]))
    m = mem.metrics(p, s, {"q": CHOICES}).set_index("id")
    assert m.at["o", "logprob"] == -0.5 and m.at["o", "exact"] == 100 and m.at["o", "recalled"] == 2
    assert np.isnan(m.at["a", "exact"]) and m.at["a", "correct"] == 100
    assert m.at["a", "reply_correct"] == 0
    assert abs(m.at["a", "gold_prob"] - 100 / (1 + np.exp(-2) + np.exp(-1) + 2 * np.exp(-4))) < 1e-9


def test_the_sft_comparison_is_trained_minus_clean(monkeypatch):
    monkeypatch.setattr(osens, "SFT_STEPS", (("c", "p"),))
    monkeypatch.setattr(osens, "DOSE_RESAMPLES", 200)
    g = pd.DataFrame(dict(item=list("abcd"), sft_rows=[1, 2, 0, 0], answer_shown=[True, False] * 2,
                          group=["trained", "trained", "clean", "clean"]))

    def model(lift):
        rows = [dict(kind="options", split="validation", item=i, context_item=i,
                     logprob=-1.0 + (lift if i in "ab" else 0.0), exact=0.0, recalled=1.0)
                for i in "abcd"]
        rows += [dict(kind="answer", split="validation", item=i, context_item=i, correct=50.0,
                      gold_prob=20.0, reply_correct=0.0) for i in "abcd"]
        # The training text adds 0.2 more to its own question, after SFT.
        rows += [dict(kind=k, split="validation", item=i, context_item="a",
                      logprob=-0.5 + (2 * lift if k == "own" else 0.0), exact=0.0, recalled=1.0)
                 for k, i in (("own", "a"), ("transplant", "c"))]
        return pd.DataFrame(rows)
    out = mem.sft_steps({"c": model(0.2), "p": model(0.0)}, g)
    lp = out[(out["metric"] == "logprob")].set_index(["probe", "group"])["difference"]
    assert abs(lp["options", "in the SFT data"] - 0.2) < 1e-12
    assert abs(lp["own", "training text's lift, SFT checkpoint"] - 0.2) < 1e-12
    assert abs(lp["own", "training text's lift, change from parent"] - 0.2) < 1e-12
    assert out.loc[out["metric"] == "correct", "difference"].eq(0).all()


def test_the_bases_gaps_are_compared_question_by_question(monkeypatch):
    monkeypatch.setattr(osens, "DOSE_RESAMPLES", 200)

    def base(shift):
        # Each question's difficulty is its own and the same for every base.
        return pd.DataFrame([dict(kind="options", split=s, item=f"{s}{i}", exact=0.0, recalled=0.0,
                                  logprob=-(i % 20) / 2 + (shift if s == "validation" else 0.0))
                             for s in ("validation", "test", "train") for i in range(100)])
    m = {k: base(0.0) for k in mem.BASES} | {"mv.base": base(0.1)}
    out = mem.pretraining(m)
    gap = out[(out["comparison"] == "MixtureVitae's gap minus another base's")
              & (out["metric"] == "logprob")]
    assert len(gap) == len(mem.BASES) - 1
    assert np.allclose(gap[["difference", "lo", "hi"]].to_numpy(), 0.1)
