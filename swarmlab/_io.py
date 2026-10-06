"""Small filesystem helpers shared by the log, blob store, and snapshot store (WP1).

`atomic_write_bytes` writes to a temporary file in the destination directory, optionally fsyncs it,
then `os.replace`s it into place and (when syncing) fsyncs the directory, so readers see either the
old file or the complete new one, never a partial write.
"""
from __future__ import annotations

import contextlib
import os
import tempfile
from pathlib import Path


def fsync_dir(path: Path) -> None:
    """fsync a directory so a rename inside it is durable. No-op where unsupported."""
    try:
        fd = os.open(path, os.O_RDONLY)
    except OSError:
        return
    try:
        with contextlib.suppress(OSError):
            os.fsync(fd)
    finally:
        os.close(fd)


def atomic_write_bytes(path: Path | str, data: bytes, *, sync: bool = True) -> Path:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, tmp = tempfile.mkstemp(prefix=f".{path.name}.", suffix=".tmp", dir=path.parent)
    try:
        with os.fdopen(fd, "wb") as f:
            f.write(data)
            if sync:
                f.flush()
                os.fsync(f.fileno())
        os.replace(tmp, path)
    except BaseException:
        with contextlib.suppress(OSError):
            os.unlink(tmp)
        raise
    if sync:
        fsync_dir(path.parent)
    return path
