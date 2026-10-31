"""The pinned inputs of every table, figure and text number, read from inputs.json.

Every command that reads a record file asks for it here, by its ID (R6, R13,
...). `path(id)` returns the local copy, after downloading it from the first
source that works if it is not there yet. A download is kept only if its
SHA-256 is the pinned one. A local copy smaller than EVERY_READ_LIMIT is also
checked every time it is read. The larger zips are checked when they are
downloaded and by `mvpt check-inputs`, and their readers check a digest of
the members they read instead (`member_digest`), because hashing them takes
minutes.

Sources are tried in the order inputs.json lists them:
  {"url": ...}      an HTTPS download, for Hugging Face and GitHub at a fixed revision
  {"gcs": ...}      the project bucket, through the `gcloud` command
  {"command": ...}  an mvpt command that writes the file; it is named, not run
  {"repo": ...}     a file committed in this repository (config/)
"""
from __future__ import annotations

import hashlib
import json
import subprocess
import urllib.request
from dataclasses import dataclass, field
from functools import lru_cache
from pathlib import Path

INPUTS_JSON = Path(__file__).with_name("inputs.json")
#: Local copies smaller than this are hashed on every read.
EVERY_READ_LIMIT = 300 * 2**20


class InputError(Exception):
    """An input is missing and cannot be fetched, or it is not the pinned version."""


@dataclass(frozen=True)
class Input:
    id: str
    file: str
    local: str
    sha256: str
    bytes: int
    sources: tuple[dict, ...]
    paper: bool
    what: str
    member_digests: dict = field(default_factory=dict)

    @property
    def gcs(self) -> str | None:
        """The bucket URI, if the input has one."""
        return next((s["gcs"] for s in self.sources if "gcs" in s), None)

    @property
    def checked_on_read(self) -> bool:
        return self.bytes < EVERY_READ_LIMIT


@lru_cache(maxsize=None)
def registry() -> dict[str, Input]:
    data = json.loads(INPUTS_JSON.read_text())["inputs"]
    return {k: Input(id=k, file=v["file"], local=v["local"], sha256=v["sha256"],
                     bytes=v["bytes"], sources=tuple(v["sources"]), paper=v["paper"],
                     what=v["what"], member_digests=dict(v.get("member_digests", {})))
            for k, v in data.items()}


def get(input_id: str) -> Input:
    try:
        return registry()[input_id]
    except KeyError:
        raise InputError(f"No input {input_id!r} in {INPUTS_JSON.name}.") from None


def member_digest(input_id: str, name: str) -> str:
    """The pinned digest of the members `name` reads from a large zip."""
    return get(input_id).member_digests[name]


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(2**22), b""):
            h.update(chunk)
    return h.hexdigest()


def verify(inp: Input, path: Path) -> None:
    """Raise InputError unless `path` has the pinned SHA-256."""
    if sha256(path) != inp.sha256:
        raise InputError(f"{path} is not the pinned version of input {inp.id} "
                         f"({inp.file}, SHA-256 {inp.sha256[:12]}...). Delete it and run "
                         f"`mvpt fetch-inputs {inp.id}`.")


def _download_url(url: str, part: Path) -> None:
    req = urllib.request.Request(url, headers={"User-Agent": "mv_post_train-inputs"})
    with urllib.request.urlopen(req, timeout=300) as resp, open(part, "wb") as out:
        for chunk in iter(lambda: resp.read(2**22), b""):
            out.write(chunk)


def _download_gcs(uri: str, part: Path) -> None:
    try:
        subprocess.run(["gcloud", "storage", "cp", uri, str(part)], check=True,
                       capture_output=True, text=True)
    except FileNotFoundError:
        raise InputError(f"{uri} needs the gcloud command, logged in with access to "
                         "the bucket.") from None
    except subprocess.CalledProcessError as exc:
        raise InputError(f"Could not download {uri}: {exc.stderr.strip()}") from None


def fetch(inp: Input, dest: Path) -> Path:
    """Download `inp` to `dest` from the first source that works."""
    dest.parent.mkdir(parents=True, exist_ok=True)
    part = dest.with_name(dest.name + ".part")
    errors = []
    for source in inp.sources:
        try:
            if "url" in source:
                _download_url(source["url"], part)
            elif "gcs" in source:
                _download_gcs(source["gcs"], part)
            elif "command" in source:
                errors.append(f"run `{source['command']}`")
                continue
            elif "repo" in source:
                errors.append(f"it is committed in this repository as {source['repo']}; "
                              "restore it with git")
                continue
            got = sha256(part)
            if got != inp.sha256:
                errors.append(f"{next(iter(source.values()))} has SHA-256 {got}, not the "
                              f"pinned {inp.sha256}")
                part.unlink()
                continue
            part.rename(dest)
            print(f"  downloaded {inp.id} ({inp.file}) -> {dest}")
            return dest
        except (InputError, OSError) as exc:
            errors.append(str(exc))
            part.unlink(missing_ok=True)
    raise InputError(f"Input {inp.id} ({inp.file}) is not at {dest} and could not be "
                     "fetched: " + "; ".join(errors))


def path(input_id: str, dest: str | Path | None = None) -> Path:
    """The local copy of `input_id` (at `dest`, or its default place), fetched
    first if it is missing, and checked against its pinned SHA-256."""
    inp = get(input_id)
    dest = Path(dest) if dest is not None else Path(inp.local)
    if not dest.exists():
        return fetch(inp, dest)
    if inp.checked_on_read:
        verify(inp, dest)
    return dest


def check(input_id: str, dest: str | Path | None = None) -> str:
    """'ok', 'missing' or 'differs', with a full hash even for the large zips."""
    inp = get(input_id)
    dest = Path(dest) if dest is not None else Path(inp.local)
    if not dest.exists():
        return "missing"
    return "ok" if sha256(dest) == inp.sha256 else "differs"


def fetch_main(ids: list[str], include_analysis: bool) -> int:
    """`mvpt fetch-inputs`: fetch the named inputs, or every paper input."""
    chosen = ids or [i for i, inp in registry().items() if inp.paper or include_analysis]
    failed = 0
    for input_id in chosen:
        try:
            print(f"{input_id:20s} {path(input_id)}")
        except InputError as exc:
            print(f"{input_id:20s} error: {exc}")
            failed += 1
    return 1 if failed else 0


def check_main(ids: list[str]) -> int:
    """`mvpt check-inputs`: hash every local copy and compare it with its pin."""
    chosen = ids or list(registry())
    bad = 0
    for input_id in chosen:
        inp = get(input_id)
        status = check(input_id)
        bad += status == "differs" or (status == "missing" and inp.paper)
        print(f"{input_id:20s} {status:8s} {inp.local}{'' if inp.paper else '  (analyses only)'}")
    return 1 if bad else 0
