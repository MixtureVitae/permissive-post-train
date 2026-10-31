"""Make the Hugging Face collections of the paper's checkpoints in the
`permissive-post-train` organisation, one per part of the paper, from the
results manifest.

    python release/checkpoints/collections.py            # print the collections
    python release/checkpoints/collections.py apply      # create or update them
    python release/checkpoints/collections.py public     # make them public

Each collection lists the checkpoints of its tables in the order of the
tables' rows, with a note that names the row, then the records dataset. A
collection can list repositories of any account, so every checkpoint stays
where it is. A checkpoint with several repositories of the same weights is
listed by the one its evaluation records name. Checkpoints that are not on
Hugging Face are left out, and `apply` adds them once the manifest has their
repositories. It adds missing items, updates notes, and removes items that are
no longer listed. New collections are private until `public`.
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

from huggingface_hub import HfApi

ORG = "permissive-post-train"
RECORDS = "permissive-post-train/evaluation-records"
ROOT = Path(__file__).resolve().parents[2]

GROUP = {"MV": "MixtureVitae", "Nemotron": "Nemotron-CC-HQ", "Comma0.1": "Comma0.1",
         "FineWeb-Edu": "FineWeb-Edu", "DCLM": "DCLM", "SmolLM2": "SmolLM2-1.7B",
         "Qwen2.5": "Qwen2.5-1.5B", "Qwen3": "Qwen3-1.7B"}
SUBSTRATE = {"MV": "MixtureVitae web part", "Comma": "Comma0.1",
             "Nemotron": "Nemotron-CC-HQ", "FineWeb-Edu": "FineWeb-Edu"}
# Phase 2 records: base model -> name
PHASE2_MODEL = {
    "MixtureVitae-1.7B": "MixtureVitae",
    "MixtureVitae-1.7B-100B": "MixtureVitae, 100B tokens",
    "MixtureVitae-1.7B-100B-woOT3": "MixtureVitae without OT3, 100B tokens",
    "MixtureVitae-1.7B-300B-woOT3": "MixtureVitae without OT3",
    "NemotronHQ-1.7B": "Nemotron-CC-HQ, 16K pretraining run",
    "NemotronHQ-1.7B-100B": "Nemotron-CC-HQ, 100B tokens",
    "NemotronHQ-1.7B-100B-mvR": "Nemotron-CC-HQ with MixtureVitae's reasoning subset, 100B tokens",
    "Comma0.1-1.7B": "Comma0.1", "FineWebEdu-1.7B": "FineWeb-Edu", "DCLM-1.7B": "DCLM",
    "SmolLM2-1.7B-16k-Base": "SmolLM2-1.7B", "SmolLM2-1.7B-Instruct-16k": "SmolLM2-1.7B-Instruct",
    "Qwen2.5-1.5B-Base": "Qwen2.5-1.5B", "Qwen2.5-1.5B-Instruct": "Qwen2.5-1.5B-Instruct",
    "Qwen3-1.7B-Base": "Qwen3-1.7B-Base", "Qwen3-1.7B-Instruct": "Qwen3-1.7B",
}
REASONING = {"OT3": "OT30K", "AR": "AceReason 30K", "NC": "Nemotron-Cascade 30K",
             "SR": "Superior-Reasoning 30K"}
PHASE2_STAGE = {"base": "starting point", "SFT-Tulu3": "Tulu3 SFT, starting point",
                "DPO-Tulu3": "Tulu3 DPO, starting point",
                **{k: v for code, data in REASONING.items() for k, v in (
                    (code, data), (f"{code}+SFT", f"Tulu3 SFT, then {data}"),
                    (f"{code}+DPO", f"Tulu3 DPO, then {data}"))}}
# older tables: row group -> name
OLDER = {
    "abl.noinst": "MixtureVitae without instruction data, 100B tokens",
    "abl.noweb": "MixtureVitae without web data, 100B tokens",
    "abl.mv100": "MixtureVitae, 100B tokens",
    "decon.orig": "MixtureVitae before decontamination",
    "decon.clean": "MixtureVitae, decontaminated",
    "small.bag": "Baguettotron",
    "small.mv04": "MixtureVitae 0.4B",
    "mask.masked": "MixtureVitae, prompt tokens masked",
    "mask.unmasked": "MixtureVitae, prompt tokens not masked",
}


def stage(context: int | str, name: str) -> str:
    k16 = context in (16384, "16k")
    if name == "base":
        return "extended to 16K" if k16 else "base"
    return ("16K, " if k16 else "") + {"sft": "Tulu3 SFT", "dpo": "Tulu3 DPO"}[name]


def phase1(m: dict) -> list[tuple[str, str]]:
    rows = [(c["checkpoint"], f"{GROUP[c['model_group']]} · {stage(c['context_length'], c['stage'])}")
            for c in m["designs"]["phase1_zoo"]["cells"]]
    # the language table's Comma0.1 and Nemotron-CC-HQ rows use other checkpoints
    lang = m["designs"]["language_tables"]["tables"]["tab:lang_eval_table2"]
    names = {"main.comma.base": "Comma0.1 · base (language suite)",
             "main.nemotron.sft": "Nemotron-CC-HQ, 16K pretraining run · Tulu3 SFT (language suite)",
             "main.nemotron.dpo": "Nemotron-CC-HQ, 16K pretraining run · Tulu3 DPO (language suite)"}
    return rows + [(r["checkpoint"], names[r["row"]]) for r in lang if r["row"] in names]


def front_loading(m: dict) -> list[tuple[str, str]]:
    d = m["designs"]["front_loading_study"]
    def note(c: dict) -> str:
        return (f"{SUBSTRATE[c['substrate']]}, {'with' if c['front_loaded'] else 'without'} "
                f"front-loading · {stage(c['context_length'], c.get('stage', 'base'))}")
    return ([(c["checkpoint"], note(c)) for c in d["cells"]]
            + [(c["checkpoint"], note(c) + " (language suite)") for c in d["language_cells"]])


def phase2(m: dict) -> list[tuple[str, str]]:
    rows = []
    for key, c in m["checkpoints"].items():
        if (l := c["labels"].get("phase2_three_seed")):
            rows.append((list(PHASE2_MODEL).index(l["base_model"]), list(PHASE2_STAGE).index(l["stage"]),
                         key, f"{PHASE2_MODEL[l['base_model']]} · {PHASE2_STAGE[l['stage']]}"))
    return [(key, note) for *_, key, note in sorted(rows)]


def older(m: dict) -> list[tuple[str, str]]:
    rows = []
    for table in m["designs"]["older_tables"]["tables"].values():
        for r in table:
            kind, group, context, name = r["row"].split(".")
            rows.append((r["checkpoint"], f"{OLDER[f'{kind}.{group}']} · {stage(context, name)}"))
    return rows


COLLECTIONS = [
    ("Phase 1: Tulu3 SFT and DPO",
     "Eight 1.7B-scale bases, each also extended to 16K, then given Tulu3 SFT and DPO.",
     phase1),
    ("Front-loading study",
     "Four web datasets pretrained with and without MixtureVitae's instruction and reasoning "
     "subset, then extended to 16K and given Tulu3 SFT and DPO.",
     front_loading),
    ("Phase 2: reasoning SFT",
     "Reasoning SFT on 30K samples of OpenThoughts3 (OT30K) or another reasoning dataset, "
     "and the checkpoints each run starts from.",
     phase2),
    ("Mixture ablation, decontamination, 0.4B and masking",
     "The 100B mixture ablation, MixtureVitae before and after decontamination, "
     "the 0.4B comparison, and SFT with prompt tokens masked.",
     older),
]


def repo(c: dict) -> str | None:
    named = [l.get("model_id") for l in c["labels"].values()]
    return next((r for r in named if r in c["hf_repos"]), c["hf_repos"][0] if c["hf_repos"] else None)


def plan() -> list[tuple[str, str, list[tuple[str, str]], int]]:
    m = json.loads((ROOT / "results/manifest.json").read_text())
    out = []
    for title, description, rows in COLLECTIONS:
        items, seen, missing = [], set(), 0
        for key, note in rows(m):
            if key in seen:
                continue
            seen.add(key)
            if (r := repo(m["checkpoints"][key])):
                items.append((r, note))
            else:
                missing += 1
        out.append((title, description, items, missing))
    return out


def apply(public: bool = False) -> None:
    api = HfApi()
    existing = {c.title: c.slug for c in api.list_collections(owner=ORG, limit=100)}
    for title, description, items, _ in plan():
        if public:
            api.update_collection_metadata(existing[title], private=False)
            print(f"{title}: public")
            continue
        if title in existing:
            slug = existing[title]
            api.update_collection_metadata(slug, description=description)
        else:
            slug = api.create_collection(title, namespace=ORG, description=description, private=True).slug
        want = [(r, "model", note) for r, note in items]
        want.append((RECORDS, "dataset", "The evaluation records of every checkpoint here"))
        have = {i.item_id: i for i in api.get_collection(slug).items}
        for r, kind, note in want:
            if r not in have:
                api.add_collection_item(slug, r, kind, note=note)
            elif have[r].note != note:
                api.update_collection_item(slug, have[r].item_object_id, note=note)
        for r, i in have.items():
            if r not in {w[0] for w in want}:
                api.delete_collection_item(slug, i.item_object_id)
        have = {i.item_id: i for i in api.get_collection(slug).items}
        for position, (r, *_) in enumerate(want):
            if have[r].position != position:
                api.update_collection_item(slug, have[r].item_object_id, position=position)
        print(f"{title}: {len(want)} items, https://huggingface.co/collections/{slug}")


def main(args: list[str]) -> None:
    if args == ["apply"]:
        apply()
    elif args == ["public"]:
        apply(public=True)
    elif not args:
        for title, description, items, missing in plan():
            print(f"## {title} ({len(items)} on Hugging Face, {missing} not)\n{description}")
            for r, note in items:
                print(f"  {r:95s} {note}")
            print()
    else:
        raise SystemExit(__doc__)


if __name__ == "__main__":
    main(sys.argv[1:])
