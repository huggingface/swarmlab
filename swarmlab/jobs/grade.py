"""Grade one ProgramBench submission inside an HF Job with the official evaluator, without Docker.

Run inside the grade job that `swarmlab programbench grade` launches (bootstrap.sh, mode
`grade`), as root, in the task's clean image `programbench/<task>:task_cleanroom_v6`:

    python -m swarmlab.jobs.grade --task T --submission submission.tar.gz \\
        --tests /root/programbench-tests --out grade.json [--image IMG] [--sweep-on-timeout]

What runs is `programbench.eval.eval.Evaluator` from the pinned `programbench==1.2.5` (the
`swarmlab[programbench]` extra), unchanged, filled exactly as `eval_batch._evaluate_instance`
fills it: `image_name` and `eval_clean_hashes` from `load_all_instances()`, the active branches
(`get_active_branches`), their expected test lists, the ignored tests and branches, `blob_dir =
get_blob_dir(task)` with `PROGRAMBENCH_BLOB_DIR` pointing at the mounted
`programbench/ProgramBench-Tests` dataset, `solution_branch="submission"`, `image_tag=
"task_cleanroom_v6"`, `branch_workers=1`, `branch_retries=1` (the CLI defaults) and
`docker_cpus` = the job's real CPU count (`detect_resources`, from the cgroup's `cpu.max`; `nproc`
lies in a job). Every step the evaluator performs therefore happens as in the official run:
workspace wipe, submission extraction, removal of gold-hash files and `./executable`, the seeded
git repo, `compile.sh` with DNS blackholed (the evaluator's own `block_build_internet_dns`, which
works as root here) under its 900 s timeout, the stashed and hashed executable, the
`pytest-rerunfailures==16.4` install; then per branch the test archive, the restored and
re-hashed executable, the `--timeout-method=signal` patch, the timeout-retry plugin with
`--reruns=2 --reruns-delay=1`, `eval/run.sh` under 3600 s with `--max-worker-restart=4`, the
serial retry after xdist worker crashes, and the JUnit XML parsing.

The one substitution: `programbench.eval.eval.ContainerEnvironment` (and `remove_image`) are
swapped, for the duration of the run, for `LocalEvalEnvironment` bound to one `LocalMachine`: the
job's own filesystem is the container.

- Constructor: `LocalEvalEnvironment(*, image, cwd, executable, timeout, cpus, env, run_args)`,
  the `ContainerEnvironment` signature (`executable` and `run_args` are ignored). The command
  environment is the job's minus secrets (`is_secret`, and every name `SWARMLAB_SECRET_NAMES`
  lists: the launcher's exact `--secrets` names), swarmlab's own variables (`SWARMLAB_*`,
  `UV_*`, `PROGRAMBENCH_*`, `HF_HOME`, `PYTHONUNBUFFERED`, `PYTHONDONTWRITEBYTECODE`) and its
  virtualenv (`VIRTUAL_ENV`, its `bin` on PATH), plus `PYTEST_XDIST_AUTO_NUM_WORKERS=<cpus>`
  then `env`, as `docker run -e` gives the container, then `SWARMLAB_EVAL_ENV=<container_id>`
  (the teardown marker, below). The image must be the job's own image
  (`LocalMachine.base_image`; none given: any) or a reference `commit` returned; else
  RuntimeError. Building one from a committed reference is "a fresh container from the
  committed image": the machine resets the workspace and the kept paths from that commit first.
- `execute(command, *, timeout=None)`: `bash -lc <command>` with that cwd and environment
  (a command that runs submitted code: as `run_as`, below), in its own session, stdout and stderr to temp files; returns `{"output": stdout + stderr (text as
  subprocess's text mode decodes it), "returncode", "exception_info": ""}`. On timeout (default
  the constructor's): the command's process group is killed (with `--sweep-on-timeout`, every
  other process too except PID 1, this process's ancestors and the processes alive before
  grading began, as the official `kill -KILL -1` sweep inside the container) and it returns `{"output": "", "returncode": -1,
  "exception_info": "Command timed out after <t>s"}`, the official values.
- `copy_in(path, dest)`: a directory is streamed through `tar -cf - . | tar -C dest -xf -`, a file
  copied (`docker cp` semantics). `copy_in_tar(tar, dest)`: `mkdir -p dest` then `tar -C dest
  -xzf` (or `-xf`), the system tar, as the container's tar would.
- `commit(ref)`: a tar of the workspace and of `LocalMachine.keep` (the evaluator's stashed
  executable, which lives outside the workspace) on local disk, under `ref`.
- `local_evaluator(machine)` also sets the evaluator's `WORKSPACE_DIR` and
  `Evaluator._stashed_executable` to the machine's (`/workspace` and the official path in a job,
  i.e. unchanged; temp paths in tests).
  `remove_image(ref)` deletes it.
- `cleanup()` (every environment teardown: the evaluator calls it after compile and after each
  branch): kills everything this environment's commands left running, as removing the
  container does: the members of their sessions; every process whose `/proc/<pid>/environ`
  holds this environment's `SWARMLAB_EVAL_ENV=<container_id>` (a `setsid` daemon leaves the
  session but keeps the environment); and, when isolating, every process of `run_as` (`kill
  -KILL -1` as that user, then a `/proc` sweep by uid until none is left): in a grade job
  nothing else runs as that user, so that is exactly the submission's process set, including
  one that cleared its environment. SIGKILL, repeated while a pass still finds one (forks);
  never PID 1, this process or its ancestors; zombies are skipped (their parent reaps them).
  Without isolation (non-root, tests) the marker is the only catch-all: a process that execs
  with a cleared environment escapes it.
- `executable` / `container_id`: the evaluator reads `eval/results.xml` with
  `subprocess.run([env.executable, "cp", f"{env.container_id}:{path}", host_tmp])`, bypassing the
  class; `executable` is a two-line shim (`LocalMachine.docker_shim`) that does exactly that
  `cp`, so the evaluator's own read (with its timeout and log entry) runs unchanged.

Isolation of the submission's code. The official evaluator runs everything as root, but in a
compile container with no dataset mount, and copies only one branch's test archive into each
later container. Here the hidden tests (`/root/programbench-tests`, which hold every branch and
the original source) and the grader (`/opt/swarmlab`, the store, `grade.json`) share the
machine with the submission, so its code runs as an unprivileged user instead:

- Which commands: `runs_submitted_code` matches exactly three of the pinned
  `programbench==1.2.5` evaluator's command strings (`SEED_GIT_COMMAND`, `COMPILE_COMMAND`,
  `RUN_TESTS_COMMAND`, the last bare or after the evaluator's `export ... && ` prefix for the
  timeout-retry plugin, which is how `run_cmd` is finalized when `pytest-rerunfailures`
  installed). The git seed is one of them although it is the evaluator's own command: a
  submitted `.git` *file* (`gitdir: ./elsewhere`) passes its `[ ! -d .git ]` test and points
  `git add` at a submitted config, whose `core.fsmonitor` (or a filter or hook) git executes.
  A test drives the installed Evaluator, records every command and fails when the matched set
  is not exactly {seed_git, compile, run_tests}, so a version bump breaks loudly. Every other
  command (the wipe, hash removal, the DNS blackhole, `mv ./executable <stash>`, the hashes,
  the `pytest-rerunfailures` install, the `sed` patch, `rm`) runs as root, as before.
- As whom: `LocalMachine(run_as=...)`, default `agent` (the task image's user, the world's
  `user`) when this process is root and the user exists, else None (no drop: tests, a non-root
  run). The prefix is the world's, `setpriv --reuid=<uid> --regid=<gid> --init-groups`, plus
  `--no-new-privs`: the task image grants `agent` sudo, which the agent job's lockdown removes
  but nothing removes here; with no-new-privs no setuid binary (sudo) elevates. The command's
  environment gets the user's `HOME`, `USER`, `LOGNAME` and `SHELL` (the environment the agent
  built in; root's `HOME` is unreadable to it).
- Ownership: right before each such command every leftover process of the user is killed and
  the workspace tree is chowned to it (`chown_tree`: `lchown`, never following a symlink; a
  non-directory with several hard links is left alone, so a hard link to a file elsewhere is
  never handed over). Doing it right before the command, not after each copy, covers every way
  a file reached the workspace (the extraction, the per-branch reset, root's steps); the reset's
  `tar -xpf` as root restores the owners recorded at commit anyway (GNU tar's `--same-owner`
  default for root). Root reads the results afterwards as it reads anything. A file `copy_in`
  puts outside the workspace (the timeout-retry plugin under `/tmp`) is made world-readable,
  since the tests that import it run as the user.
- What it cannot reach: `/root` is made 0700 before grading and at every environment start
  (`private`; the Ubuntu default), the store (commit tars, the docker shim) is 0700, `grade()`
  makes the output dir 0700 and then checks as the user (`test -r || test -x` must fail) that
  the tests dir, the store and the output dir are out of reach, else RuntimeError. The grader's
  install is `/opt/swarmlab`, 0700 (bootstrap.sh). The grader's own helpers that run as the
  user (that check, the `kill -1`) get a bare `PATH` environment, never this process's (which
  holds the job's secrets and which a same-uid process could read from `/proc`). The stashed executable stays at the evaluator's path,
  `/opt/programbench-stashed-executable-do-not-modify`, outside the workspace and not under
  `/root`: it is the submission's own binary, moved there by root after `compile.sh`, and its
  hash is checked when each branch restores it. The DNS blackhole is root's `/etc/resolv.conf`,
  which the user cannot undo (the official one can be undone by a root `compile.sh`).
- Fidelity cost (in `DEVIATION`): a `compile.sh` that needs root (writes `/usr/local`, runs
  `apt`) fails here, and a test whose expected output depends on running as root (owner names,
  a permission-denied path) can fail here while it passes officially.

Deviations (both in `DEVIATION`, recorded in grade.json): the user above; and the official
evaluator starts one fresh container per branch (and per retry) from the post-compile image,
while here every branch runs on the same machine after a reset of `/workspace` and the stashed
executable from the post-compile tar. Anything else a test writes outside the workspace (`/tmp`,
the user's `$HOME`) persists into later branches, and the `pytest-rerunfailures` install and any
system-wide change stay, as they would in the committed image. Scores are not cheat-filtered.

Output (`--out`, `grade.json`): `{"task", "image", "programbench_version", "cpus",
"pytest_workers", "memory_gb", "started_at", "ended_at", "wall_s", "deviation", "score":
{"score", "passed", "total", "resolved", "near_resolved", "branches", "compile_ok",
"error_code"}, "evaluation": <EvaluationResult as JSON>}`. The score is the official one:
`programbench.submission.test_results_map` (active branches, ignored tests removed) and
`score_from_tests`, `resolved` = score >= `RESOLVED_THRESHOLD` (1.0), `near_resolved` = score >=
`NEAR_RESOLVED_THRESHOLD` (0.95); `compile_ok` = the evaluation has no `error_code`. Exit code 0
once grade.json is written (whatever the score), 1 otherwise.
"""
from __future__ import annotations

import argparse
import contextlib
import json
import locale
import os
import pwd
import shutil
import signal
import stat
import subprocess
import sys
import tempfile
import time
import uuid
from collections.abc import Iterator, Mapping
from pathlib import Path
from typing import Any

from ..procs import ancestors as _ancestors
from ..procs import environ_has as _environ_has
from ..procs import kill_matching
from ..procs import pids as _pids
from ..procs import uid_of as _uid_of
from ..world.programbench import (
    IMAGE_TAG,
    _venv_bin,
    declared_secrets,
    detect_resources,
    is_secret,
)

STASHED_EXECUTABLE = "/opt/programbench-stashed-executable-do-not-modify"  # Evaluator's
DEFAULT_RUN_AS = "agent"         # the task image's unprivileged user (the world's `user`)
PRIVATE_DIRS = ("/root",)        # made 0700 when isolating: the hidden tests are mounted under it
EVAL_ENV_MARKER = "SWARMLAB_EVAL_ENV"   # =<container_id> in every command's environment
DEVIATION = ("branches share one machine: /workspace and the stashed executable are reset from "
             "the post-compile tar before each branch, instead of one fresh container per branch; "
             "the submission's code (git seed, compile.sh, eval/run.sh) runs as the unprivileged "
             "user agent (no new privileges) instead of root")
# The three evaluator commands that execute the submission's code, verbatim from the pinned
# programbench==1.2.5 eval.py (`Evaluator._compile_executable`, `_run_test_branch`); a test drives
# the installed Evaluator and fails when they change, so a version bump breaks loudly.
SEED_GIT_COMMAND = (
    "if [ ! -d .git ]; then "
    "GIT_AUTHOR_DATE='2000-01-01T00:00:00Z' "
    "GIT_COMMITTER_DATE='2000-01-01T00:00:00Z' "
    "git -c init.defaultBranch=gold init -q && "
    "git -c user.email=gold@local -c user.name=gold "
    "-c commit.gpgsign=false add -A && "
    "GIT_AUTHOR_DATE='2000-01-01T00:00:00Z' "
    "GIT_COMMITTER_DATE='2000-01-01T00:00:00Z' "
    "git -c user.email=gold@local -c user.name=gold "
    "-c commit.gpgsign=false commit -q --allow-empty -m gold; "
    "fi")
COMPILE_COMMAND = "chmod +x ./compile.sh && ./compile.sh"
RUN_TESTS_COMMAND = "chmod +x ./eval/run.sh && ./eval/run.sh"
_DROP_PREFIXES = ("SWARMLAB_", "UV_", "PROGRAMBENCH_")
_DROP = {"VIRTUAL_ENV", "HF_HOME", "PYTHONUNBUFFERED", "PYTHONDONTWRITEBYTECODE"}
# the grader's own helpers that run as the user (kill, test): never this process's environment,
# which holds the job's secrets and is readable (/proc/<pid>/environ) by the user's processes
_BARE_ENV = {"PATH": "/usr/local/sbin:/usr/local/bin:/usr/sbin:/usr/bin:/sbin:/bin"}
DOCKER_SHIM = """#!/bin/sh
# swarmlab grade: stands in for `docker cp <container>:<path> <dest>`, the evaluator's read of
# eval/results.xml; the container is this machine
if [ "$1" = cp ] && [ "$#" -eq 3 ]; then exec cp -- "${2#*:}" "$3"; fi
echo "swarmlab local docker shim: only 'cp <container>:<path> <dest>' is supported" >&2
exit 2
"""


def runs_submitted_code(command: str) -> bool:
    """True for the evaluator commands that execute the submission's code: the git seed (a
    submitted `.git` file can point git at a submitted config), `compile.sh` and `eval/run.sh`
    (bare, or after the evaluator's `export ... && ` prefix for the timeout-retry plugin)."""
    return (command in (SEED_GIT_COMMAND, COMPILE_COMMAND, RUN_TESTS_COMMAND)
            or command.endswith(" && " + RUN_TESTS_COMMAND))


def grade_env(base: Mapping[str, str], venv_bin: str | None = None) -> dict[str, str]:
    """The job's environment as a container would have it: no secrets (`is_secret`, and every
    name `SWARMLAB_SECRET_NAMES` declares), nothing of swarmlab's."""
    declared = declared_secrets(base)
    env = {k: v for k, v in base.items()
           if not is_secret(k) and k not in declared and k not in _DROP
           and not k.startswith(_DROP_PREFIXES)}
    if venv_bin and "PATH" in env:
        env["PATH"] = os.pathsep.join(p for p in env["PATH"].split(os.pathsep)
                                      if p.rstrip("/") != venv_bin.rstrip("/"))
    return env


def _text(data: bytes) -> str:
    """Bytes as `subprocess.run(text=True)` returns them (locale encoding, universal newlines)."""
    return data.decode(locale.getpreferredencoding(False)).replace("\r\n", "\n").replace("\r", "\n")


def _sessions() -> set[int]:
    """Session ids of the processes alive now (`/proc/<pid>/stat` field 6)."""
    out = set()
    for entry in os.listdir("/proc"):
        if entry.isdigit():
            try:
                out.add(int(Path(f"/proc/{entry}/stat").read_text().rsplit(")", 1)[1].split()[3]))
            except (OSError, ValueError, IndexError):
                pass
    return out


def chown_tree(root: str, uid: int, gid: int) -> None:
    """`root` and everything under it to uid:gid, never following a symlink (`lchown`); a
    non-directory with more than one hard link is left as it is (a hard link to a file
    elsewhere must never hand that file over)."""
    def one(path: str) -> None:
        with contextlib.suppress(FileNotFoundError):
            st = os.lstat(path)
            if (st.st_uid, st.st_gid) == (uid, gid):
                return
            if not stat.S_ISDIR(st.st_mode) and st.st_nlink > 1:
                return
            os.chown(path, uid, gid, follow_symlinks=False)

    one(root)
    for dirpath, dirnames, filenames in os.walk(root, followlinks=False):
        for name in (*dirnames, *filenames):
            one(os.path.join(dirpath, name))


def sweep_processes(protected: set[int] | frozenset[int] = frozenset()) -> list[int]:
    """SIGKILL every process but PID 1, this process's ancestors and `protected` (the processes
    alive before grading began: the job's own, e.g. the bootstrap's log `tee`). The official
    teardown kills every non-PID-1 process in the container. Only for a dedicated job container."""
    keep = _ancestors() | {1} | set(protected)
    killed = []
    for entry in os.listdir("/proc"):
        if entry.isdigit() and int(entry) not in keep:
            with contextlib.suppress(OSError):
                os.kill(int(entry), signal.SIGKILL)
                killed.append(int(entry))
    return killed


class _Auto:
    def __repr__(self) -> str:
        return "AUTO"


AUTO = _Auto()


def default_run_as() -> str | None:
    """`DEFAULT_RUN_AS` when this process is root and that user exists, else None."""
    if os.geteuid() != 0:
        return None
    try:
        pwd.getpwnam(DEFAULT_RUN_AS)
    except KeyError:
        return None
    return DEFAULT_RUN_AS


class LocalMachine:
    """The one filesystem every `LocalEvalEnvironment` of a grade shares (module docstring)."""

    def __init__(self, store: Path | str, *, workspace: str = "/workspace",
                 keep: tuple[str, ...] = (STASHED_EXECUTABLE,), base_image: str | None = None,
                 sweep: bool = False, environ: Mapping[str, str] | None = None,
                 run_as: str | None | _Auto = AUTO,
                 private: tuple[str, ...] = PRIVATE_DIRS) -> None:
        self.store = Path(store)
        (self.store / "images").mkdir(parents=True, exist_ok=True)
        self.store.chmod(0o700)
        self.workspace = workspace
        self.keep = tuple(keep)
        self.base_image = base_image
        self.sweep = sweep
        self.protected = frozenset(_pids()) if sweep else frozenset()
        self.images: dict[str, Path] = {}
        self.resets: list[str] = []
        self.env = grade_env(os.environ if environ is None else environ, _venv_bin())
        self.docker_shim = self.store / "bin" / "docker"
        self.docker_shim.parent.mkdir(parents=True, exist_ok=True)
        self.docker_shim.write_text(DOCKER_SHIM)
        self.docker_shim.chmod(0o755)
        self.run_as = default_run_as() if isinstance(run_as, _Auto) else run_as
        self.private = tuple(private)
        self.uid: int | None = None
        self.gid: int | None = None
        self.prefix: list[str] = []          # the drop to `run_as` (empty: no drop)
        self.identity: dict[str, str] = {}   # its HOME, USER, LOGNAME, SHELL
        if self.run_as:
            try:
                pw = pwd.getpwnam(self.run_as)
            except KeyError as e:
                raise ValueError(f"run_as user {self.run_as!r} does not exist") from e
            self.uid, self.gid = pw.pw_uid, pw.pw_gid
            self.prefix = ["setpriv", f"--reuid={pw.pw_uid}", f"--regid={pw.pw_gid}",
                           "--init-groups", "--no-new-privs", "--"]
            self.identity = {"HOME": pw.pw_dir, "USER": pw.pw_name, "LOGNAME": pw.pw_name,
                             "SHELL": pw.pw_shell or "/bin/bash"}

    @property
    def isolating(self) -> bool:
        """Submitted code runs as `run_as` with the side effects that need root (chown, killing
        the user's processes, private dirs): only when this process is root."""
        return self.uid is not None and os.geteuid() == 0

    def _paths(self) -> list[str]:
        return [self.workspace, *self.keep]

    def make_private(self) -> None:
        """`private` dirs (and the store) to 0700 when isolating."""
        if not self.isolating:
            return
        for d in (*self.private, str(self.store)):
            with contextlib.suppress(FileNotFoundError):
                if stat.S_IMODE(os.stat(d).st_mode) & 0o077:
                    os.chmod(d, 0o700)

    def kill_user(self) -> list[int]:
        """Kill every process of `run_as` (the submission's whole process set: nothing else runs
        as that user in a grade job): `kill -KILL -1` as the user, then a /proc sweep by uid."""
        if not self.isolating:
            return []
        with contextlib.suppress(OSError, subprocess.TimeoutExpired):
            subprocess.run([*self.prefix, "bash", "-c", "kill -KILL -1 2>/dev/null; exit 0"],
                           env=_BARE_ENV, cwd="/", stdin=subprocess.DEVNULL,
                           capture_output=True, timeout=30, check=False)
        return kill_matching(lambda pid: _uid_of(pid) == self.uid)

    def prepare_unprivileged(self) -> None:
        """Right before a command that runs submitted code: no process of the user left over,
        the workspace tree owned by it (whatever put the files there)."""
        if not self.isolating:
            return
        assert self.uid is not None and self.gid is not None
        self.kill_user()
        chown_tree(self.workspace, self.uid, self.gid)

    def hidden_from_user(self, path: str | Path) -> bool:
        """True when `run_as` can neither read nor search `path` (`test -r || test -x` as the
        user fails: a 0700 root directory, or anything under one); True when not isolating
        (nothing runs as another user)."""
        if not self.isolating:
            return True
        r = subprocess.run([*self.prefix, "sh", "-c", 'test -r "$1" || test -x "$1"', "sh",
                            str(path)], env=_BARE_ENV, cwd="/", stdin=subprocess.DEVNULL,
                           capture_output=True, timeout=30, check=False)
        return r.returncode != 0

    def start(self, image: str) -> None:
        """What `docker run <image>` would give: the job's own state for the base image, the
        committed state (reset) for a reference `commit` returned."""
        self.make_private()
        if image in self.images:
            self.reset(image)
        elif self.base_image is not None and image != self.base_image:
            raise RuntimeError(f"local grading runs in {self.base_image}; the evaluator asked "
                               f"for {image}")

    def commit(self, ref: str) -> str:
        path = self.store / "images" / f"{uuid.uuid4().hex}.tar"
        rel = [p.lstrip("/") for p in self._paths() if os.path.lexists(p)]
        r = subprocess.run(["tar", "-C", "/", "-cf", str(path), *rel], capture_output=True,
                           text=True, check=False)
        if r.returncode != 0:
            raise RuntimeError(f"commit (tar of {rel}) failed: {r.stderr.strip()}")
        self.images[ref] = path
        return ref

    def reset(self, ref: str) -> None:
        if os.path.realpath(self.workspace) == "/":
            raise ValueError("refusing to reset / as a workspace")
        os.makedirs(self.workspace, exist_ok=True)
        for entry in os.scandir(self.workspace):
            if entry.is_dir(follow_symlinks=False):
                shutil.rmtree(entry.path)
            else:
                os.unlink(entry.path)
        for p in self.keep:
            if os.path.isdir(p) and not os.path.islink(p):
                shutil.rmtree(p)
            elif os.path.lexists(p):
                os.unlink(p)
        r = subprocess.run(["tar", "-C", "/", "-xpf", str(self.images[ref])], capture_output=True,
                           text=True, check=False)
        if r.returncode != 0:
            raise RuntimeError(f"reset from {ref} failed: {r.stderr.strip()}")
        self.resets.append(ref)

    def remove_image(self, ref: str, *, executable: str = "docker") -> None:
        path = self.images.pop(ref, None)
        if path is not None:
            path.unlink(missing_ok=True)


class LocalEvalEnvironment:
    """`programbench.container.ContainerEnvironment`'s interface on this machine (module doc)."""

    def __init__(self, *, image: str, cwd: str = "/", executable: str = "docker",
                 timeout: int = 30, cpus: int = 10, env: dict[str, str] | None = None,
                 run_args: list[str] | None = None, machine: LocalMachine) -> None:
        self.machine = machine
        self.image = image
        self.cwd = cwd
        self.default_timeout = timeout
        self.cpus = cpus
        self.executable = str(machine.docker_shim)
        self.container_id = f"local-{uuid.uuid4().hex[:12]}"
        self.env = {**machine.env, "PYTEST_XDIST_AUTO_NUM_WORKERS": str(cpus), **(env or {}),
                    EVAL_ENV_MARKER: self.container_id}
        self._marker = f"{EVAL_ENV_MARKER}={self.container_id}".encode()
        self._groups: list[int] = []
        self._ran = False
        machine.start(image)

    def execute(self, command: str, *, timeout: int | None = None) -> dict[str, Any]:
        timeout = timeout or self.default_timeout
        argv, env = ["bash", "-lc", command], self.env
        if self.machine.prefix and runs_submitted_code(command):
            self.machine.prepare_unprivileged()
            argv, env = [*self.machine.prefix, *argv], {**self.env, **self.machine.identity}
        with tempfile.TemporaryFile() as out, tempfile.TemporaryFile() as err:
            self._ran = True
            proc = subprocess.Popen(argv, cwd=self.cwd, env=env,
                                    stdin=subprocess.DEVNULL, stdout=out, stderr=err,
                                    start_new_session=True)
            self._groups.append(proc.pid)
            try:
                rc = proc.wait(timeout=timeout)
            except subprocess.TimeoutExpired:
                self._kill(proc.pid)
                if self.machine.sweep:
                    sweep_processes(self.machine.protected)
                with contextlib.suppress(subprocess.TimeoutExpired):
                    proc.wait(timeout=30)
                return {"output": "", "returncode": -1,
                        "exception_info": f"Command timed out after {timeout}s"}
            out.seek(0)
            err.seek(0)
            return {"output": _text(out.read()) + _text(err.read()), "returncode": rc,
                    "exception_info": ""}

    @staticmethod
    def _kill(pgid: int) -> None:
        with contextlib.suppress(OSError):
            os.killpg(pgid, signal.SIGKILL)

    def copy_in(self, local_path: Path, container_path: str) -> None:
        local_path = Path(local_path)
        if local_path.is_dir():
            os.makedirs(container_path, exist_ok=True)
            producer = subprocess.Popen(["tar", "-C", str(local_path), "-cf", "-", "."],
                                        stdout=subprocess.PIPE, stderr=subprocess.PIPE)
            cp = subprocess.run(["tar", "-C", container_path, "-xf", "-"], stdin=producer.stdout,
                                capture_output=True, text=True, check=False)
            assert producer.stdout is not None
            producer.stdout.close()
            prc = producer.wait()
            if prc != 0 or cp.returncode != 0:
                raise RuntimeError(f"tar stream into container failed (producer rc={prc}, exec "
                                   f"rc={cp.returncode}): {cp.stderr.strip()!r}")
            return
        try:
            dest = shutil.copy2(local_path, container_path)
            if self.machine.isolating:  # e.g. the retry plugin, read by eval/run.sh as the user
                os.chmod(dest, stat.S_IMODE(os.stat(dest).st_mode) | 0o444)
        except OSError as e:
            raise RuntimeError(f"docker cp failed: {e}") from e

    def copy_in_tar(self, tar_path: Path, container_path: str) -> None:
        tar_path = Path(tar_path)
        is_gz = tar_path.name.endswith((".tar.gz", ".tgz")) or tar_path.suffix == ".gz"
        try:
            os.makedirs(container_path, exist_ok=True)
        except OSError as e:
            raise RuntimeError(f"mkdir -p {container_path} in container failed: {e}") from e
        r = subprocess.run(["tar", "-C", container_path, "-xzf" if is_gz else "-xf",
                            str(tar_path)], capture_output=True, text=True, check=False)
        if r.returncode != 0:
            raise RuntimeError(f"tar stream into container failed: {r.stderr.strip()}")

    def commit(self, image_ref: str) -> str:
        return self.machine.commit(image_ref)

    def cleanup(self) -> None:
        """Kill everything this environment's commands left running, as removing the container
        would: the members of their sessions (each command is a session leader, so a pid reused
        since is not a match), every process whose environment carries this environment's
        `SWARMLAB_EVAL_ENV` marker (a `setsid` daemon leaves the session, not the environment),
        and when isolating every process of `run_as` (one that cleared its environment too).
        Never this process or its ancestors."""
        if not self._ran:
            return
        live = _sessions()
        for sid in self._groups:
            if sid in live:
                self._kill(sid)
        self._groups = []
        kill_matching(lambda pid: _environ_has(pid, self._marker))
        self.machine.kill_user()
        self._ran = False

    def __del__(self) -> None:
        try:
            self.cleanup()
        except Exception:  # noqa: BLE001, S110 - interpreter shutdown: modules may be gone
            pass


@contextlib.contextmanager
def local_evaluator(machine: LocalMachine) -> Iterator[None]:
    """Swap the evaluator's container class and image removal for the local ones, then restore."""
    import functools

    from programbench.eval import eval as pb_eval  # type: ignore[import-not-found]

    saved = (pb_eval.ContainerEnvironment, pb_eval.remove_image, pb_eval.WORKSPACE_DIR,
             pb_eval.Evaluator._stashed_executable)
    pb_eval.ContainerEnvironment = functools.partial(LocalEvalEnvironment, machine=machine)
    pb_eval.remove_image = machine.remove_image
    # the official values in a job (/workspace, the official stash path); a temp dir in tests
    pb_eval.WORKSPACE_DIR = machine.workspace
    pb_eval.Evaluator._stashed_executable = machine.keep[0]
    try:
        yield
    finally:
        (pb_eval.ContainerEnvironment, pb_eval.remove_image, pb_eval.WORKSPACE_DIR,
         pb_eval.Evaluator._stashed_executable) = saved


def official_score(eval_json: Path, instance: dict) -> dict[str, Any]:
    """The per-instance score exactly as `programbench.submission` computes it."""
    from programbench.eval.eval import EvaluationResult  # type: ignore[import-not-found]
    from programbench.submission import (  # type: ignore[import-not-found]
        NEAR_RESOLVED_THRESHOLD,
        RESOLVED_THRESHOLD,
        score_from_tests,
        test_results_map,
    )
    from programbench.utils.load_data import get_active_branches  # type: ignore[import-not-found]

    tests = test_results_map(eval_json, instance)
    score = score_from_tests(tests)
    result = EvaluationResult.model_validate_json(eval_json.read_text())
    return {"score": score, "passed": sum(tests.values()), "total": len(tests),
            "resolved": score >= RESOLVED_THRESHOLD, "near_resolved": score >= NEAR_RESOLVED_THRESHOLD,
            "branches": len(get_active_branches(instance)), "compile_ok": result.error_code is None,
            "error_code": result.error_code}


def grade(task: str, submission: Path, tests_dir: Path, out: Path, *, image: str | None = None,
          sweep: bool = False, workspace: str = "/workspace",
          stash: str = STASHED_EXECUTABLE,
          environ: Mapping[str, str] | None = None,
          run_as: str | None | _Auto = AUTO) -> dict[str, Any]:
    """Evaluate `submission` for `task` with the official evaluator on this machine; write and
    return the grade.json document."""
    os.environ["PROGRAMBENCH_BLOB_DIR"] = str(tests_dir)  # read when blob_store is imported
    from importlib.metadata import version

    from programbench.eval.eval import Evaluator  # type: ignore[import-not-found]
    from programbench.utils import blob_store  # type: ignore[import-not-found]
    from programbench.utils.load_data import (  # type: ignore[import-not-found]
        get_active_branches,
        get_ignored_branches,
        get_ignored_tests,
        load_all_instances,
    )

    blob_store.BLOB_LOCAL_DIR = str(tests_dir)  # in case it was imported before
    instance = next((i for i in load_all_instances() if i["instance_id"] == task), None)
    if instance is None:
        raise ValueError(f"unknown ProgramBench task {task!r}")
    branches = get_active_branches(instance)
    if not branches:
        raise ValueError(f"task {task!r} has no active test branches")
    base = f"{instance['image_name']}:{IMAGE_TAG}"
    if image and image != base:
        raise ValueError(f"this job runs {image}, but {task} is graded in {base}")
    blob_dir = blob_store.get_blob_dir(task)
    if blob_dir is None:
        raise FileNotFoundError(f"no tests for {task} under {tests_dir}")
    data = instance.get("branches", {})
    cpus, memory_gb = detect_resources()
    workers = max(1, int(cpus or 1))
    out.parent.mkdir(parents=True, exist_ok=True)
    started, t0 = time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()), time.monotonic()
    with tempfile.TemporaryDirectory(prefix="swarmlab-grade-") as store:
        machine = LocalMachine(store, workspace=workspace, keep=(stash,), base_image=base,
                               sweep=sweep, environ=environ, run_as=run_as)
        if machine.isolating:
            out.parent.chmod(0o700)
            machine.make_private()
            seen = [str(p) for p in (tests_dir, machine.store, out.parent)
                    if not machine.hidden_from_user(p)]
            if seen:
                raise RuntimeError(f"user {machine.run_as} can reach {', '.join(seen)}; the "
                                   "hidden tests, the grader's store and its output must be "
                                   "out of its reach (a 0700 directory above each)")
        with local_evaluator(machine):
            evaluator = Evaluator(
                image_name=instance["image_name"], solution_branch="submission",
                submission_archive=Path(submission), blob_dir=blob_dir, tests_branches=branches,
                remove_hashes=instance.get("eval_clean_hashes", []), image_tag=IMAGE_TAG,
                tests_by_branch={b: data[b]["tests"] for b in branches if b in data},
                ignored_tests=get_ignored_tests(instance),
                ignored_branches=get_ignored_branches(instance), instance_id=task,
                docker_cpus=workers, branch_workers=1, branch_retries=1)
            result = evaluator.run()
    eval_json = out.parent / f"{task}.eval.json"
    eval_json.write_text(result.model_dump_json(indent=2))
    doc = {"task": task, "image": image or base, "programbench_version": version("programbench"),
           "cpus": cpus, "pytest_workers": workers, "memory_gb": memory_gb,
           "started_at": started, "ended_at": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
           "wall_s": round(time.monotonic() - t0, 1), "deviation": DEVIATION,
           "score": official_score(eval_json, instance),
           "evaluation": json.loads(result.model_dump_json())}
    out.write_text(json.dumps(doc, indent=1))
    return doc


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(prog="python -m swarmlab.jobs.grade", description=__doc__.split(
        "\n", 1)[0])
    ap.add_argument("--task", required=True)
    ap.add_argument("--submission", required=True, type=Path)
    ap.add_argument("--tests", required=True, type=Path, help="the mounted ProgramBench-Tests")
    ap.add_argument("--out", required=True, type=Path)
    ap.add_argument("--image", default=None)
    ap.add_argument("--sweep-on-timeout", action="store_true",
                    help="kill every other process on a command timeout (job containers only)")
    args = ap.parse_args(argv)
    import logging

    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(name)s %(levelname)s %(message)s")
    try:
        doc = grade(args.task, args.submission, args.tests, args.out, image=args.image or None,
                    sweep=args.sweep_on_timeout)
    except Exception as e:  # noqa: BLE001 - the job log must say why there is no grade.json
        print(f"GRADE ERROR: {type(e).__name__}: {e}", file=sys.stderr)
        return 1
    print("GRADE " + json.dumps(doc["score"], sort_keys=True))
    return 0


if __name__ == "__main__":  # pragma: no cover
    sys.exit(main())
