"""Inputs kept in the project's private Google Cloud Storage bucket.

Some inputs are not published anywhere else yet. Their home is the bucket,
and the commands that need one download it on first use into the local data
folder, check its SHA-256, and read the local copy from then on. Reading the
bucket needs the `gcloud` command, logged in with access to it.
"""
from __future__ import annotations

import hashlib
import subprocess
from pathlib import Path


class FetchError(Exception):
    """The file could not be downloaded, or it is not the pinned version."""


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(2**20), b""):
            h.update(chunk)
    return h.hexdigest()


def local_copy(uri: str | None, dest: Path, pinned: str) -> Path:
    """`dest`, downloaded from `uri` first if it is not there yet. A download
    is kept only if its SHA-256 is `pinned`. With no `uri`, `dest` is returned
    as it is: the file comes from somewhere else (see the README)."""
    if uri is None or dest.exists():
        return dest
    dest.parent.mkdir(parents=True, exist_ok=True)
    part = dest.with_name(dest.name + ".part")
    try:
        subprocess.run(["gcloud", "storage", "cp", uri, str(part)], check=True,
                       capture_output=True, text=True)
    except FileNotFoundError:
        raise FetchError(f"{dest.name} lives at {uri}, and reading it needs the gcloud "
                         "command.") from None
    except subprocess.CalledProcessError as exc:
        raise FetchError(f"Could not download {uri}: {exc.stderr.strip()}") from None
    got = sha256(part)
    if got != pinned:
        part.unlink()
        raise FetchError(f"{uri} has SHA-256 {got}, not the pinned {pinned}.")
    part.rename(dest)
    print(f"  downloaded {uri} -> {dest}")
    return dest
