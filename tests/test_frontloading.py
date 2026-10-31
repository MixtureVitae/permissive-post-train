"""Tests for the revision's front-loading study with Comma0.1 (render-frontloading).

Like test_workshop.py, they shrink every benchmark to two items. They pin:
Phase 1's Comma0.1 fills the Comma arm without front-loading; a malformed
Comma file is refused; adding substrates leaves every other contrast's
estimate and interval unchanged; a substrate with no language scores prints
dashes instead of stopping the table; and Comma0.1's language scores are read
with the paper's shot counts and metrics, from the pinned May files, for the
checkpoints its reasoning records use.
"""
from __future__ import annotations

import itertools
import json
import zipfile

import numpy as np
import pandas as pd
import pytest

from mv_post_train.data import protocol_a as pa
from mv_post_train.render import workshop_tables
from mv_post_train.stats import factorial as fs

ITEMS = 2


@pytest.fixture(autouse=True)
def _two_items_per_benchmark(monkeypatch):
    monkeypatch.setattr(pa, "BENCH_ITEMS", {b: ITEMS for b in pa.BENCHMARKS})


def _rows(group, fl, *, factorial, zoo, p, seed=0):
    rng = np.random.default_rng(seed)
    rows = []
    for ctx, stage in itertools.product(pa.CONTEXTS, pa.STAGES):
        mid = f"x/{group}-{fl}-{ctx}-{stage}"
        for s, bench, item in itertools.product(pa.SEEDS, pa.BENCHMARKS, range(ITEMS)):
            rows.append(dict(model_id=mid, stage=stage, model_group=group, front_loading=fl,
                             context_length=ctx, seed=s, benchmark=bench,
                             sample_id=f"{bench}-{item}", correct=float(rng.random() < p),
                             in_factorial=factorial, in_zoo=zoo))
    return rows


def _records():
    main = []
    for i, (pool, fl) in enumerate(itertools.product(pa.POOLS, (False, True))):
        main += _rows(pool, fl, factorial=True, zoo=False, p=0.7 if fl else 0.3, seed=i)
    main += _rows("Comma0.1", False, factorial=False, zoo=True, p=0.3, seed=10)
    comma = _rows("Comma", True, factorial=True, zoo=False, p=0.7, seed=11)
    return pd.DataFrame(main), pd.DataFrame(comma)


def test_the_zoo_comma_is_the_control_of_the_comma_arm():
    main, comma = _records()
    df = pa.frontloading(main, comma)
    assert df.groupby(pa.FACTORIAL_KEYS).ngroups == 4 * 2 * 2 * 3
    control = df[(df["model_group"] == "Comma") & ~df["front_loading"]]
    assert set(control["model_id"]) == set(main.loc[main["model_group"] == "Comma0.1", "model_id"])


def test_a_comma_file_with_other_arms_is_refused():
    main, comma = _records()
    with pytest.raises(pa.DesignError, match="only the front-loaded Comma arm"):
        pa.frontloading(main, comma.assign(front_loading=False))


def test_adding_substrates_leaves_the_other_contrasts_unchanged():
    main, comma = _records()
    three = fs.factorial_contrasts(pa.factorial(main), n_boot=50)
    four = fs.factorial_contrasts(pa.frontloading(main, comma), n_boot=50,
                                  pools=pa.FRONTLOADING_POOLS,
                                  permissive=pa.FRONTLOADING_PERMISSIVE)
    keys = ["family", "label", "context_length", "stage"]
    both = three.merge(four, on=keys, suffixes=("_3", "_4"))
    assert len(both) == len(three) and len(four) == 2 * 3 * (4 + 4 + 2)
    for col in ("estimate", "lo", "hi"):
        np.testing.assert_array_equal(both[f"{col}_3"], both[f"{col}_4"])
    assert {"Comma - Nemotron", "Comma - non-permissive"} <= set(four["label"])


def test_a_substrate_without_language_scores_prints_dashes(tmp_path):
    keys = list(itertools.product(pa.FRONTLOADING_POOLS, (False, True), pa.CONTEXTS, pa.STAGES))
    summary = pd.DataFrame([dict(model_group=p, front_loading=fl, context_length=c, stage=s,
                                 benchmark="__suite__", mean=20.0 if fl else 5.0, sd=0.1)
                            for p, fl, c, s in keys])
    lang = pd.DataFrame([dict(model_group=p, front_loading=fl, context_length=c, language=50.0)
                         for p, fl, c in itertools.product(pa.POOLS, (False, True), pa.CONTEXTS)])
    tex = workshop_tables.factorial_summary(
        summary, lang, tmp_path, pools=pa.FRONTLOADING_POOLS,
        permissive=pa.FRONTLOADING_PERMISSIVE, note="Pending.").read_text()
    comma = tex[tex.index("Comma0.1"):tex.index("Nemotron-CC-HQ")]
    assert comma.count("--") == 6 and "$+15.0$" in comma
    assert r"Comma0.1 \emph{(permissive)}" in tex and "Pending.}" in tex


def test_captions_spell_pretraining_as_each_paper_does(tmp_path):
    """The workshop paper writes "pre-training", the main revision "pretraining"."""
    keys = list(itertools.product(pa.POOLS, (False, True), pa.CONTEXTS, pa.STAGES))
    summary = pd.DataFrame([dict(model_group=p, front_loading=fl, context_length=c, stage=s,
                                 benchmark=b, mean=10.0, sd=0.1)
                            for p, fl, c, s in keys for b in [*workshop_tables.BENCHMARKS, "__suite__"]])
    lang = pd.DataFrame([dict(model_group=p, front_loading=fl, context_length=c, language=50.0)
                         for p, fl, c in itertools.product(pa.POOLS, (False, True), pa.CONTEXTS)])
    for spelling, sub in (("pre-training", "workshop"), ("pretraining", "revision")):
        kw = {} if spelling == "pre-training" else {"pretraining": spelling}
        out = tmp_path / sub
        table5 = workshop_tables.factorial_summary(summary, lang, out, **kw).read_text()
        table30 = workshop_tables.factorial_perbenchmark(summary, out, **kw).read_text()
        assert f"300B {spelling} tokens" in table5 and f"whose {spelling} mixture" in table30


SETTINGS = {t: (5, "acc,none") if t == "mmlu" else (10, "acc_norm,none") for t in pa.LANGUAGE_TASKS}


def _language_zips(tmp_path, shots=None):
    """One results file per task for every Comma0.1 cell, in two zips laid out
    like COMMA_LANGUAGE_CELLS. `shots` overrides the shot count of every task."""
    paths = {pa.LANGUAGE_RESULTS_ZIP: tmp_path / "results.zip",
             pa.LANGUAGE_LOGS_ZIP: tmp_path / "logs.zip"}
    members = {name: [] for name in paths}
    for i, (source, folder, checkpoint, _) in enumerate(pa.COMMA_LANGUAGE_CELLS.values()):
        for j, task in enumerate(pa.LANGUAGE_TASKS):
            n, metric = SETTINGS[task]
            n = n if shots is None else shots
            result = {"model_name": checkpoint, "config": {"fewshot_seed": pa.FEWSHOT_SEED},
                      "results": {task: {metric: (i * 11 + j) / 100, "acc,none": 0.0}},
                      "n-shot": {f"{task}_anatomy" if task == "mmlu" else task: n}}
            members[source].append((f"{folder}{i}_{task}.json", json.dumps(result).encode(), result))
    for source, path in paths.items():
        with zipfile.ZipFile(path, "w") as zf:
            for member, raw, _ in members[source]:
                zf.writestr(member, raw)
    return paths, pa.logs_members_digest(members[pa.LANGUAGE_LOGS_ZIP])


def test_comma_language_is_read_with_the_paper_settings(tmp_path, monkeypatch):
    paths, digest = _language_zips(tmp_path)
    monkeypatch.setattr(pa, "LANGUAGE_LOGS_MEMBERS_SHA256", digest)
    lang = pa.load_comma_language(paths[pa.LANGUAGE_RESULTS_ZIP],
                                  paths[pa.LANGUAGE_LOGS_ZIP], SETTINGS)
    assert len(lang) == 4 * len(pa.LANGUAGE_TASKS) and set(lang["model_group"]) == {"Comma"}
    row = lang[(lang["task"] == "arc_challenge") & ~lang["front_loading"]
               & (lang["context_length"] == 16384)].iloc[0]
    assert row["performance"] == pytest.approx(3 * 11 + 3)   # acc_norm, not acc
    assert row["model_name"] == "ontocord/1.7b-Comma0.1-300BT-longsft_16k"


def test_a_comma_run_with_other_shot_counts_is_refused(tmp_path, monkeypatch):
    paths, digest = _language_zips(tmp_path, shots=0)
    monkeypatch.setattr(pa, "LANGUAGE_LOGS_MEMBERS_SHA256", digest)
    with pytest.raises(pa.DesignError, match=r"was run with \[0\] shots"):
        pa.load_comma_language(paths[pa.LANGUAGE_RESULTS_ZIP],
                               paths[pa.LANGUAGE_LOGS_ZIP], SETTINGS)


def test_changed_may_results_are_refused(tmp_path):
    paths, _ = _language_zips(tmp_path)
    with pytest.raises(pa.DesignError, match="not the pinned ones"):
        pa.load_comma_language(paths[pa.LANGUAGE_RESULTS_ZIP],
                               paths[pa.LANGUAGE_LOGS_ZIP], SETTINGS)


def test_comma_language_must_stand_for_the_reasoning_checkpoints(tmp_path, monkeypatch):
    paths, digest = _language_zips(tmp_path)
    monkeypatch.setattr(pa, "LANGUAGE_LOGS_MEMBERS_SHA256", digest)
    comma = pa.load_comma_language(paths[pa.LANGUAGE_RESULTS_ZIP],
                                   paths[pa.LANGUAGE_LOGS_ZIP], SETTINGS)
    others = pd.DataFrame([dict(model_group=p, front_loading=fl, context_length=c, task=t,
                                performance=50.0)
                           for p, fl, c, t in itertools.product(pa.POOLS, (False, True),
                                                                pa.CONTEXTS, pa.LANGUAGE_TASKS)])
    df = pa.frontloading(*_records())
    with pytest.raises(pa.DesignError, match="reasoning records use"):
        pa.frontloading_language(others, comma, df)
    for (fl, ctx), (*_, model_id) in pa.COMMA_LANGUAGE_CELLS.items():
        cell = (df["model_group"] == "Comma") & (df["front_loading"] == fl) & (
            df["context_length"] == ctx) & (df["stage"] == "base")
        df.loc[cell, "model_id"] = model_id
    lang = pa.frontloading_language(others, comma, df)
    assert lang.groupby(["model_group", "front_loading", "context_length"]).ngroups == 16


def test_text_macros_print_what_the_tables_print(tmp_path):
    from mv_post_train.render import frontloading
    df = pa.frontloading(*_records())
    keys = pa.FACTORIAL_KEYS
    scores = fs.benchmark_scores(df, keys)
    summary = fs.benchmark_summary(scores, keys)
    contrasts = fs.factorial_contrasts(df, n_boot=50, pools=pa.FRONTLOADING_POOLS,
                                       permissive=pa.FRONTLOADING_PERMISSIVE)
    lang = pd.DataFrame([dict(model_group=p, front_loading=fl, context_length=c,
                              language=60.0 + 5 * fl + i)
                         for i, (p, fl, c) in enumerate(itertools.product(
                             pa.FRONTLOADING_POOLS, (False, True), pa.CONTEXTS))])
    m = frontloading.macros(summary, lang, contrasts, fs.domain_scores(scores, keys))
    table5 = workshop_tables.factorial_summary(
        summary, lang, tmp_path, pools=pa.FRONTLOADING_POOLS,
        permissive=pa.FRONTLOADING_PERMISSIVE).read_text()
    table28 = workshop_tables.contrasts_table(contrasts, tmp_path,
                                              pools=pa.FRONTLOADING_POOLS).read_text()
    unwrap = lambda v: v.removeprefix(r"\ensuremath{").removesuffix("}")
    comma = table5[table5.index("Comma0.1"):table5.index("Nemotron-CC-HQ")]
    assert f"${unwrap(m['suite.comma.delta.16k.sft'])}$" in comma
    assert f"${unwrap(m['lang.comma.delta.4k'])}$" in comma
    row = contrasts[(contrasts["label"] == "Comma - FineWeb-Edu")
                    & (contrasts["context_length"] == 4096) & (contrasts["stage"] == "dpo")].iloc[0]
    assert unwrap(m["perm.comma-fineweb.4k.dpo"]) == f"{row.estimate:+.2f}"
    assert f"${unwrap(m['perm.comma-fineweb.4k.dpo'])}$ & ${unwrap(m['perm.comma-fineweb.4k.dpo.ci'])}$" in table28
    assert m["fl.n.word"] == "twenty-four" and m["n.checkpoints"] == "48"
    assert m["ifeval.post.n"] == "16"
    path = frontloading.write_macros(m, tmp_path)
    assert r"\providecommand{\pfl}" in path.read_text()


def test_figure3_labels_can_follow_table5():
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from mv_post_train.render.figures import workshop as wf
    # 11.36 and 23.23 print as 11.4 and 23.2: Table 5 gives +11.8, the unrounded difference +11.9
    suite = pd.DataFrame([dict(model_group=p, front_loading=fl, context_length=c, stage="sft",
                               mean=23.23 if fl else 11.36, sd=0.1)
                          for p, fl, c in itertools.product(pa.FRONTLOADING_POOLS, (False, True),
                                                            pa.CONTEXTS)])
    labels = {}
    for printed in (False, True):
        fig, ax = plt.subplots()
        wf._paired_panel(ax, suite, 4096, "sft", pools=pa.FRONTLOADING_POOLS,
                         printed_deltas=printed)
        labels[printed] = {t.get_text() for t in ax.texts if "Delta" in t.get_text()}
        plt.close(fig)
    assert labels == {False: {r"$\Delta$ +11.9"}, True: {r"$\Delta$ +11.8"}}


def test_the_permissive_pair_is_mv_minus_comma_and_its_own_family():
    df = pa.frontloading(*_records())
    kw = dict(n_boot=50, permissive=pa.FRONTLOADING_PERMISSIVE)
    contrasts = fs.factorial_contrasts(df, pools=pa.FRONTLOADING_POOLS, **kw)
    pairs = fs.permissive_pair_contrasts(df, **kw)
    assert set(pairs["label"]) == {"MV - Comma"} and len(pairs) == 6
    perm = contrasts[contrasts["family"] == "permissiveness"].set_index(
        ["label", "context_length", "stage"])
    for r in pairs.itertuples():
        mv, comma = (perm.loc[(f"{p} - FineWeb-Edu", r.context_length, r.stage), "estimate"]
                     for p in ("MV", "Comma"))
        assert abs(r.estimate - (mv - comma)) < 1e-9
    # its own Benjamini-Hochberg family: Table 28's q-values do not see it
    again = fs.factorial_contrasts(df, pools=pa.FRONTLOADING_POOLS, **kw)
    pd.testing.assert_frame_equal(contrasts, again)
    assert fs.permissive_pair_contrasts(df, n_boot=50, permissive=("MV",)).empty
