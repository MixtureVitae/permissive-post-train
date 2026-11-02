"""Upload the pipeline inputs of the paper to the records dataset on Hugging
Face, one commit per call: those that were only in the project bucket, and
copies of those on GitHub and other Hugging Face repositories.

    python release/records/upload.py small     # the card and the files under 100 MB
    python release/records/upload.py large     # the four zips over 100 MB
    python release/records/upload.py R6 R13    # the card and the inputs named
    python release/records/upload.py card      # the card only
    python release/records/upload.py pin       # make the dataset's head the first source

Every file is checked against its pin in `mv_post_train/inputs.json` first.
`pin` checks that every file at the dataset's head has its pinned SHA-256.
R6 is the copy of logs.zip that `strip_gpqa.py` writes, without the GPQA items.
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

from huggingface_hub import CommitOperationAdd, HfApi

from mv_post_train import inputs

REPO = "permissive-post-train/evaluation-records"
ROOT = Path(__file__).resolve().parents[2]

# input ID -> path in the dataset
FILES = {
    "R15": "protocol_a/comma_reasoning_unified_per_sample.csv",
    "R18": "protocol_a/ablation_decontamination_0_4B_reasoning_per_sample.csv",
    "R19": "protocol_a/masked_reasoning_per_sample.csv",
    "R20": "protocol_a/fineweb_edu_16k_april_base_unified_per_sample.csv",
    "R21": "protocol_a/ifeval_strict_per_sample_protocol_a.csv",
    "R22": "protocol_a/phase1_livecodebenchv5_per_sample.csv",
    "R24": "protocol_a/dclm_16k_reasoning_evals_per_sample.csv",
    "R16": "language/language_eval_all_models_2026-10-16.zip",
    "R17": "language/masked_models_lang_evals.zip",
    "R23": "language/language_eval_all_models_3seed_2026-10-26.zip",
    "R14-meta": "phase2/eval_responses_meta.parquet",
    "R14-lengths": "phase2/answer_lengths.parquet",
    "S1": "snapshots/hf_checkpoints.json",
    "R8": "language/language_eval-logs.zip",
    "R8-fineweb": "language/logs-fineweb.zip",
    "R6": "older/logs_without_gpqa_items.zip",
    # the benchmark-overlap audit of the training data, as frozen on 2026-11-01
    "O1": "overlap/excluded_items.csv",
    "O2": "overlap/overlap_sensitivity.csv",
    "O3": "overlap/found.parquet",
    "O4": "overlap/mv_question_overlap_rates.csv",
    "O5": "overlap/mv_question_overlap_rates_ot3.csv",
    "O6": "overlap/phase2_tokens.csv",
    "O7": "overlap/phase2_ot3_release.csv",
    # copies; their original sources stay in inputs.json after this one
    "R13": "protocol_a/canonical_reasoning_per_sample.csv",
    "R13-language": "language/ablation_frontloading_x_permissiveness_language_evaluations.csv",
    "R14-scores": "phase2/eval_scores.parquet",
    "R14-ifeval": "phase2/ifeval_allmetrics_persample.parquet",
    "S2-0.4b": "setup/ali-elganzory__0.4b-mixturevitae-v1-decontaminated-300B-4096.config.json",
    "S2-1.7b": "setup/open-sci__open-sci-ref-v0.02-1.7b-mixturevitae-300B-4096.config.json",
}
LARGE = {"R8", "R8-fineweb", "R6", "R23"}


def pin() -> None:
    api = HfApi()
    info = api.dataset_info(REPO, files_metadata=True)
    remote = {s.rfilename: s.lfs.sha256 if s.lfs else None for s in info.siblings}
    data = json.loads(inputs.INPUTS_JSON.read_text())
    for i, path in FILES.items():
        entry = data["inputs"][i]
        digest = remote.get(path)
        if digest is None and path in remote:
            # stored in git, not LFS: hash the file at that revision
            local = api.hf_hub_download(REPO, path, repo_type="dataset", revision=info.sha)
            digest = inputs.sha256(Path(local))
        if digest != entry["sha256"]:
            raise SystemExit(f"{i}: {path} at {info.sha} is not the pinned file")
        url = f"https://huggingface.co/datasets/{REPO}/resolve/{info.sha}/{path}"
        entry["sources"] = [{"url": url}] + [s for s in entry["sources"]
                                             if REPO not in s.get("url", "")]
        print(f"{i:12} {url}")
    inputs.INPUTS_JSON.write_text(json.dumps(data, indent=2, ensure_ascii=False) + "\n")


def main(parts: list[str]) -> None:
    if parts == ["pin"]:
        return pin()
    part = " ".join(parts)
    ids = [] if part == "card" else parts if set(parts) <= FILES.keys() else \
        [i for i in FILES if (i in LARGE) == (part == "large")]
    ops = [CommitOperationAdd("README.md", str(ROOT / "release/records/README.md"))] \
        if part != "large" else []
    for i in ids:
        local = ROOT / inputs.get(i).local
        status = inputs.check(i, local)
        print(f"{i:12} {status}  {local.relative_to(ROOT)} -> {FILES[i]}")
        if status != "ok":
            raise SystemExit(f"{i} is {status}: run `mvpt fetch-inputs {i}`")
        ops.append(CommitOperationAdd(FILES[i], str(local)))
    api = HfApi()
    api.create_repo(REPO, repo_type="dataset", exist_ok=True)
    info = api.create_commit(REPO, ops, repo_type="dataset",
                             commit_message="Update the card" if part == "card"
                             else f"Add the {part} pipeline inputs")
    print("commit", info.oid)


if __name__ == "__main__":
    main(sys.argv[1:])
