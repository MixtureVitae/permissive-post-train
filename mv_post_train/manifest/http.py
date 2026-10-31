"""Small HTTP helpers for the manifest inputs, using only the standard library.

`get` retries on rate limits (HTTP 429) and server errors, honouring
`Retry-After`. `RangeFile` is a seekable read-only file over an HTTP URL that
fetches only the byte ranges asked for, so pyarrow can read a few columns of
a multi-gigabyte Parquet file without downloading all of it.
"""
from __future__ import annotations

import io
import time
import urllib.error
import urllib.request

USER_AGENT = "mv_post_train-manifest"


class NotFound(Exception):
    """The URL returned 404."""


def get(url: str, *, headers: dict | None = None, tries: int = 8,
        timeout: float = 120.0) -> tuple[bytes, dict]:
    """GET `url`, returning (body, response headers). Raises NotFound on 404."""
    hdrs = {"User-Agent": USER_AGENT, **(headers or {})}
    delay = 5.0
    for attempt in range(tries):
        try:
            with urllib.request.urlopen(urllib.request.Request(url, headers=hdrs),
                                        timeout=timeout) as resp:
                return resp.read(), resp.headers
        except urllib.error.HTTPError as e:
            if e.code == 404:
                raise NotFound(url) from e
            if e.code not in (429, 500, 502, 503, 504) or attempt == tries - 1:
                raise
            wait = float(e.headers.get("Retry-After") or delay)
        except (urllib.error.URLError, TimeoutError):
            if attempt == tries - 1:
                raise
            wait = delay
        time.sleep(min(wait, 300.0))
        delay *= 2
    raise RuntimeError("unreachable")


class RangeFile(io.RawIOBase):
    """Seekable, read-only file over HTTP range requests."""

    def __init__(self, url: str):
        self.url = url
        self.pos = 0
        _, headers = get(url, headers={"Range": "bytes=0-0"})
        total = headers.get("Content-Range", "").rpartition("/")[2]
        if not total.isdigit():
            raise ValueError(f"server does not support range requests for {url}")
        self.size = int(total)

    def readable(self) -> bool:
        return True

    def seekable(self) -> bool:
        return True

    def tell(self) -> int:
        return self.pos

    def seek(self, offset: int, whence: int = io.SEEK_SET) -> int:
        base = {io.SEEK_SET: 0, io.SEEK_CUR: self.pos, io.SEEK_END: self.size}[whence]
        self.pos = max(0, base + offset)
        return self.pos

    def readinto(self, buffer) -> int:
        n = min(len(buffer), self.size - self.pos)
        if n <= 0:
            return 0
        body, _ = get(self.url, headers={"Range": f"bytes={self.pos}-{self.pos + n - 1}"})
        buffer[:len(body)] = body
        self.pos += len(body)
        return len(body)


def open_url(url: str, buffer_size: int = 8 * 2**20) -> io.BufferedReader:
    return io.BufferedReader(RangeFile(url), buffer_size=buffer_size)
