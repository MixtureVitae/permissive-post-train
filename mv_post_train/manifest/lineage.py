"""Which checkpoint each checkpoint was trained from.

Three kinds of evidence, strongest first:

  training summary   `run_summary.json` in the model repo: the model the run
                     actually loaded (OT30K runs record it)
  model card         the `base_model` field of the repo's card
  name rule          the parent the checkpoint's own name implies, such as
                     `sft__ot30k_<parent>` or `<parent>-SFT-Tulu3-decontaminated`

Two names are the same model when their weight files have the same SHA-256.
That is how a copy such as `SmolLM2-1.7B-WithChatTemplate` is matched to the
`HuggingFaceTB/SmolLM2-1.7B` a model card names.

A parent is "verified" when a training summary or model card names it (and
they agree), "name only" when only the name rule gives it, "conflict" when the
evidence disagrees, and "none recorded" when nothing names one. The last is
what pretrained bases and external releases get.
"""
from __future__ import annotations

import re
from dataclasses import dataclass, field

_RULES = [
    # (pattern, how to build the parent name, relation)
    (re.compile(r"^sft__ot30k_(?P<p>.+)$"), "{p}", "OT30K reasoning SFT"),
    (re.compile(r"^(?P<p>.+)-ot30k(-rope8-32k)?$"), "{p}", "OT30K reasoning SFT"),
    (re.compile(r"^(?P<p>.+)-ar30k$"), "{p}", "AceReason 30K reasoning SFT"),
    (re.compile(r"^(?P<p>.+)-nc30k$"), "{p}", "Nemotron-Cascade 30K reasoning SFT"),
    (re.compile(r"^(?P<p>.+)-sr30k$"), "{p}", "Superior-Reasoning 30K reasoning SFT"),
    (re.compile(r"^(?P<p>.+)-DPO-Tulu3(?P<d>-decontam\w*)?$"), "{p}-SFT-Tulu3{d}", "Tulu3 DPO"),
    (re.compile(r"^(?P<p>.+)-SFT-Tulu3(-decontam\w*)?$"), "{p}", "Tulu3 SFT"),
    (re.compile(r"^(?P<p>.+?)-(longsft_16k|long_sft_16k|16k)$"), "{p}", "16K context extension"),
    (re.compile(r"^(?P<p>.+)-WithChatTemplate$"), "{p}", "chat template added"),
]


def name_rule(name: str) -> tuple[str, str] | None:
    """(parent name, relation) implied by a checkpoint's name, or None."""
    for pattern, template, relation in _RULES:
        m = pattern.match(name)
        if m:
            groups = {k: (v or "") for k, v in m.groupdict().items()}
            parent = template.format(**groups)
            # `...-4096-4096-longsft_16k` extends `...-4096`.
            parent = re.sub(r"(-4096)+$", "-4096", parent)
            return parent, relation
    return None


def bare(repo_or_name: str) -> str:
    return repo_or_name.split("/", 1)[1] if "/" in repo_or_name else repo_or_name


@dataclass
class Snapshot:
    """The Hugging Face snapshot written by `hf_snapshot.snapshot`."""
    names: dict[str, list[str]]
    repos: dict[str, dict]
    weight_class: dict[str, tuple] = field(default_factory=dict)

    def __post_init__(self):
        for repo, rec in self.repos.items():
            if rec.get("weights"):
                self.weight_class[repo] = tuple(rec["weights"])

    def register(self, key: str, repos: list[str]) -> None:
        """Let a checkpoint key (which may differ from its folder name) be
        looked up like a name."""
        self.names.setdefault(key, [r for r in repos if r in self.repos])

    def repos_for(self, name: str) -> list[str]:
        """Repos carrying a checkpoint name (a bare folder name or a repo id)."""
        if "/" in name:
            return [name] if name in self.repos else []
        return [r for r in self.names.get(name, []) if r in self.repos]

    def weights(self, name: str) -> set[tuple]:
        return {self.weight_class[r] for r in self.repos_for(name) if r in self.weight_class}

    def same_model(self, a: str, b: str) -> bool:
        if bare(a) == bare(b):
            return True
        wa, wb = self.weights(a), self.weights(b)
        return bool(wa and wb and wa & wb)


@dataclass
class Parent:
    parent: str | None
    status: str                 # verified | name only | conflict | none recorded
    relation: str | None
    evidence: dict


def _reasoning_run(name: str) -> bool:
    rule = name_rule(name)
    return rule is not None and "reasoning SFT" in rule[1]


def resolve_prefix(name: str, known: set[str]) -> str:
    """Folder names are sometimes cut short (`...-decontaminate`). Map a cut
    name to the one known checkpoint name it is the start of. Reasoning runs
    are left out: one trained from the cut name is not its full form."""
    if name in known:
        return name
    matches = sorted((k for k in known if k.startswith(name) and not _reasoning_run(k)), key=len)
    if matches and all(m.startswith(matches[0]) for m in matches):
        return matches[0]
    return name


def find_parent(name: str, repos: list[str], snap: Snapshot, known: set[str]) -> Parent:
    """`name`: the checkpoint's folder or repo name, for the name rule.
    `repos`: the Hugging Face repos that carry it. `known`: every checkpoint key."""
    hard: dict[str, str] = {}

    def other(model: str) -> bool:
        """A repo that a card names but that holds these same weights is a copy,
        not a parent (a `-WithChatTemplate` copy can name the repo it copies)."""
        return not any(snap.same_model(model, r) for r in repos)

    for repo in repos:
        rec = snap.repos.get(repo, {})
        if rec.get("run_summary_base_model") and other(rec["run_summary_base_model"]):
            hard.setdefault("training_summary", rec["run_summary_base_model"])
        if rec.get("card_base_model") and other(rec["card_base_model"]):
            hard.setdefault("model_card", rec["card_base_model"])
    rule = name_rule(name)
    evidence = dict(hard)
    if rule:
        # Only a parent name embedded in a reasoning run's folder name can be
        # cut short. A name made by stripping a suffix is complete as it is.
        embedded = "reasoning SFT" in rule[1]
        evidence["name_rule"] = (resolve_prefix(rule[0], known - {name}) if embedded
                                 else rule[0])
    relation = rule[1] if rule else None
    if hard:
        values = list(hard.values())
        agree = all(snap.same_model(values[0], v) for v in values[1:])
        chosen = hard.get("training_summary") or hard["model_card"]
        if not agree:
            return Parent(chosen, "conflict", relation, evidence)
        return Parent(chosen, "verified", relation, evidence)
    if rule:
        return Parent(evidence["name_rule"], "name only", relation, evidence)
    return Parent(None, "none recorded", None, evidence)


def to_key(parent: str | None, snap: Snapshot, keys: set[str]) -> str | None:
    """Map a parent (repo id or name) to the checkpoint key that is the same
    model, or to its bare name when no checkpoint with results is that model."""
    if parent is None:
        return None
    if bare(parent) in keys:
        return bare(parent)
    for key in sorted(keys):
        if snap.same_model(parent, key):
            return key
    return bare(parent)
