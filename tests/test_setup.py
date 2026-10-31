"""Tests for Tables 17 to 25, the setup and evaluation settings
(mv_post_train/render/setup.py).

They pin that the parameter count from a model's configuration equals the
count Hugging Face reports for its weights, that FLOPs print as open-sci-ref
prints them, that a batch size that disagrees with the iterations is flagged,
that the computed columns (cooldown, ratios, totals) follow from the
stated values in config/paper_setup.toml, and that Table 23 names each task's
domain and the weight the suite average gives it.
"""
from __future__ import annotations

import tomllib
from pathlib import Path

from mv_post_train.render import setup

ROOT = Path(__file__).resolve().parents[1]
#: The two architectures, from their config.json at the pinned revisions.
BASE = dict(num_key_value_heads=None, tie_word_embeddings=True, head_dim=64, qk_layernorm=True,
            vocab_size=50304)
CONFIG_04 = BASE | dict(hidden_size=1024, intermediate_size=3840, num_hidden_layers=22,
                        num_attention_heads=16, num_key_value_heads=16)
CONFIG_17 = BASE | dict(hidden_size=2048, intermediate_size=8192, num_hidden_layers=24,
                        num_attention_heads=32, num_key_value_heads=32)


def test_parameters_match_the_released_weights():
    # Hugging Face reports 403,639,040 and 1,714,377,728 BF16 parameters for the
    # model.safetensors of the two models (checked 2026-10-17).
    assert sum(setup.parameters(CONFIG_04)) == 403_639_040
    assert sum(setup.parameters(CONFIG_17)) == 1_714_377_728
    assert setup.parameters(CONFIG_17)[1] == 50304 * 2048


def test_flops_print_with_one_decimal_and_a_power_of_ten():
    assert setup._sci(6 * 403_639_040) == "2.4\\times10^{9}"
    assert setup._sci(6 * 1_714_377_728) == "1.0\\times10^{10}"
    assert setup._sci(9.96e9) == "1.0\\times10^{10}"


def test_the_schedules_compute_cooldown_and_flag_the_100b_batch():
    conf = tomllib.loads((ROOT / "config" / "paper_setup.toml").read_text())
    checks, out = setup.Checks(), {}
    tex = setup.training_schedules(conf["training_schedules"], checks, out)
    assert out["schedule.300b.cooldown"] == "14,532" and out["schedule.100b.cooldown"] == "4,768"
    assert [r["result"] for r in checks.rows] == ["differs", "ok"]
    assert "1024.0 sequences" in checks.rows[0]["computed"]
    assert "Cooldown \\\\ (20\\%)" in tex


def test_totals_and_ratios_follow_from_the_stated_values():
    conf = tomllib.loads((ROOT / "config" / "paper_setup.toml").read_text())
    out = {}
    setup.sft_gpu_hours(conf["ot30k_sft_gpu_hours"], out)
    setup.budget(conf["budget"], out)
    setup.sft_hyperparams(conf["ot30k_sft_hyperparams"], out)
    assert out["gpu.ot30k.total"] == "1,417" and out["gpu.ot30k.runs"] == "18"
    assert (out["budget.smollm2.ratio"], out["budget.qwen3.ratio"]) == ("36.7", "120.0")
    assert out["sft.effective-batch"] == "128" and out["sft.world-size"] == "32"


def test_table23_gives_each_task_its_domain_and_the_domain_weights():
    conf = tomllib.loads((ROOT / "config" / "paper_setup.toml").read_text())
    out = {}
    manifest = {"protocols": {"protocol_a": {"seeds": [42, 1234, 2025]}}}
    tex = setup.protocol_a(conf["protocol_a"], manifest, out)
    assert r"\textbf{Domain}" in tex and "GSM8K & GSM8K              & Mathematics &" in tex
    assert "IFE   & IFEval             & Instruction following &" in tex
    assert [out[f"domain.{d}.weight"] for d in ("math", "code", "science", "instruction")] == [
        "45", "27", "18", "9"]
    assert out["domain.math.n.word"] == "five" and "mathematics at 45\\%" in tex
    # Table 24 keeps its columns
    assert r"\textbf{Domain}" not in "\n".join(setup._protocol_head(False))


def test_gsm8k_is_scored_the_same_way_under_both_protocols():
    conf = tomllib.loads((ROOT / "config" / "paper_setup.toml").read_text())
    metric = {p: {a: m for a, _, _, m, *_ in conf[p]["tasks"]} for p in ("protocol_a", "protocol_b")}
    assert metric["protocol_a"]["GSM8K"] == metric["protocol_b"]["GSM8K"] == "Exact match (flexible extract)"
    out = {}
    tex = setup.protocol_a(conf["protocol_a"], {"protocols": {"protocol_a": {"seeds": [42, 1234, 2025]}}}, out)
    assert r"GSM8K is scored as under Protocol~B (Table~\ref{tab:eval-settings-B})" in tex
