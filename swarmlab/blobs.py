"""Content-addressed blob store (docs/INTERFACE.md §5).

Layout: `<dir>/<sha[:2]>/<sha>` where `sha` is the lowercase SHA-256 hex digest of the content.
The two-character shard keeps directories small at hundreds of agents; INTERFACE §15's
`blobs/<sha>` is satisfied in spirit (one file per blob, named by its hash).

Writes are atomic (temp file in the shard directory, then rename) and idempotent: putting
content that already exists is a no-op. `put(..., sync=True)` additionally fsyncs the file and
its directory; snapshot blobs use it so a manifest never points at a blob lost in a crash.
Delivered message content may use the default `sync=False` because it is re-created when an
uncommitted round is re-executed.
"""
from __future__ import annotations

import hashlib
from pathlib import Path

from ._io import atomic_write_bytes


def sha256_hex(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


class BlobStore:
    def __init__(self, dir: Path | str) -> None:
        self.dir = Path(dir)
        self.dir.mkdir(parents=True, exist_ok=True)

    def path(self, sha: str) -> Path:
        if len(sha) != 64 or any(c not in "0123456789abcdef" for c in sha):
            raise ValueError(f"not a sha256 hex digest: {sha!r}")
        return self.dir / sha[:2] / sha

    def put(self, data: bytes, *, sync: bool = False) -> str:
        if not isinstance(data, (bytes, bytearray, memoryview)):
            raise TypeError(f"BlobStore.put expects bytes, got {type(data).__name__}")
        data = bytes(data)
        sha = sha256_hex(data)
        p = self.path(sha)
        if not p.exists():
            atomic_write_bytes(p, data, sync=sync)
        return sha

    def get(self, sha: str) -> bytes:
        p = self.path(sha)
        try:
            return p.read_bytes()
        except FileNotFoundError:
            raise KeyError(sha) from None

    def has(self, sha: str) -> bool:
        return self.path(sha).exists()

    __contains__ = has

    def put_text(self, text: str, *, sync: bool = False) -> str:
        """UTF-8 convenience wrapper; the hash is of the encoded bytes."""
        return self.put(text.encode("utf-8"), sync=sync)

    def get_text(self, sha: str) -> str:
        return self.get(sha).decode("utf-8")
