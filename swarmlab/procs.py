"""Process sweeps over `/proc` (Linux), shared by the ProgramBench world and its grader.

- `kill_matching(match)`: SIGKILL every live process a predicate selects (by uid: `uid_of`; by an
  entry of its environment: `environ_has`), never PID 1, this process or its ancestors; repeated
  until a pass finds none, since a process may fork meanwhile.
- `group_alive(pgid)` / `kill_group(pgid)`: a process group as a unit, independent of its leader
  (a command started with `start_new_session=True` has pgid = its pid): alive while it has a
  member that is not a zombie; killed with `killpg(SIGKILL)` and polled until empty (bounded).

Where `/proc` is missing (not Linux) the scans find nothing and `group_alive` falls back to
`killpg(pgid, 0)` alone.
"""
from __future__ import annotations

import contextlib
import os
import signal
import time
from collections.abc import Callable
from pathlib import Path


def ancestors() -> set[int]:
    """This process and its ancestors (`/proc/<pid>/stat` ppid chain)."""
    pids, pid = {os.getpid()}, os.getpid()
    while pid > 1:
        try:
            stat = Path(f"/proc/{pid}/stat").read_text()
            pid = int(stat.rsplit(")", 1)[1].split()[1])
        except (OSError, ValueError, IndexError):
            break
        pids.add(pid)
    return pids


def pids() -> set[int]:
    try:
        return {int(e) for e in os.listdir("/proc") if e.isdigit()}
    except OSError:
        return set()


def stat_fields(pid: int) -> list[str]:
    """`/proc/<pid>/stat` after the command name: [state, ppid, pgrp, session, ...] ([] if gone)."""
    try:
        return Path(f"/proc/{pid}/stat").read_text().rsplit(")", 1)[1].split()
    except (OSError, IndexError):
        return []


def uid_of(pid: int) -> int | None:
    try:
        return os.stat(f"/proc/{pid}").st_uid
    except OSError:
        return None


def environ_has(pid: int, entry: bytes) -> bool:
    """`entry` (`b"NAME=value"`) is one of the process's environment entries (as it was at exec;
    unreadable: False)."""
    try:
        return entry in Path(f"/proc/{pid}/environ").read_bytes().split(b"\0")
    except OSError:
        return False


def kill_matching(match: Callable[[int], bool], passes: int = 20) -> list[int]:
    """SIGKILL every live (not zombie) process `match` selects, except PID 1 and this process
    and its ancestors; repeated until a pass finds none (a process may fork meanwhile), at most
    `passes` times. Returns the pids killed."""
    keep = ancestors() | {1}
    killed: list[int] = []
    for _ in range(passes):
        found = [pid for pid in pids() if pid not in keep and match(pid)
                 and stat_fields(pid)[:1] not in ([], ["Z"])]
        if not found:
            break
        for pid in found:
            with contextlib.suppress(OSError):
                os.kill(pid, signal.SIGKILL)
                killed.append(pid)
        time.sleep(0.05)
    return killed


def group_alive(pgid: int) -> bool:
    """The process group has a member that is not a zombie. `killpg(pgid, 0)` first (cheap; a
    group we may not signal counts as alive); when it succeeds, a `/proc` scan confirms a live
    member, so a group of unreaped zombies (orphans whose new parent does not reap) is empty."""
    try:
        os.killpg(pgid, 0)
    except ProcessLookupError:
        return False
    except PermissionError:
        return True
    if not os.path.isdir("/proc"):
        return True
    for pid in pids():
        f = stat_fields(pid)
        if len(f) > 2 and f[2] == str(pgid) and f[0] != "Z":
            return True
    return False


def kill_group(pgid: int, settle_s: float = 2.0) -> bool:
    """SIGKILL the whole process group, whatever its leader's state, then poll until it is empty
    (at most `settle_s`). Returns whether it is empty."""
    deadline = time.monotonic() + settle_s
    while True:
        with contextlib.suppress(ProcessLookupError, PermissionError):
            os.killpg(pgid, signal.SIGKILL)
        if not group_alive(pgid):
            return True
        if time.monotonic() >= deadline:
            return False
        time.sleep(0.01)
