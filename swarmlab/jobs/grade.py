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
  environment is the job's minus secrets (`is_secret`), swarmlab's own variables (`SWARMLAB_*`,
  `UV_*`, `PROGRAMBENCH_*`, `HF_HOME`, `PYTHONUNBUFFERED`, `PYTHONDONTWRITEBYTECODE`) and its
  virtualenv (`VIRTUAL_ENV`, its `bin` on PATH), plus `PYTEST_XDIST_AUTO_NUM_WORKERS=<cpus>`
  then `env`, as `docker run -e` gives the container. The image must be the job's own image
  (`LocalMachine.base_image`; none given: any) or a reference `commit` returned; else
  RuntimeError. Building one from a committed reference is "a fresh container from the
  committed image": the machine resets the workspace and the kept paths from that commit first.
- `execute(command, *, timeout=None)`: `bash -lc <command>` with that cwd and environment, in its
  own session, stdout and stderr to temp files; returns `{"output": stdout + stderr (text as
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
- `cleanup()`: kills what this environment's commands left running (their process groups);
  nothing else (no container to stop).
- `executable` / `container_id`: the evaluator reads `eval/results.xml` with
  `subprocess.run([env.executable, "cp", f"{env.container_id}:{path}", host_tmp])`, bypassing the
  class; `executable` is a two-line shim (`LocalMachine.docker_shim`) that does exactly that
  `cp`, so the evaluator's own read (with its timeout and log entry) runs unchanged.

Deviation (the one documented one): the official evaluator starts one fresh container per branch
(and per retry) from the post-compile image; here every branch runs on the same machine after a
reset of `/workspace` and the stashed executable from the post-compile tar. Anything else a test
writes outside the workspace (`/tmp`, `$HOME`) persists into later branches, and the
`pytest-rerunfailures` install and any system-wide change `compile.sh` made stay, as they would
in the committed image. Also: the submission's `compile.sh` and tests run as root in the same
job as this process (as in the official compile container, but here next to the grader), so a
hostile submission could tamper with the result; scores are not cheat-filtered either way.

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
import shutil
import signal
import subprocess
import sys
import tempfile
import time
import uuid
from collections.abc import Iterator, Mapping
from pathlib import Path
from typing import Any

from ..world.programbench import IMAGE_TAG, _venv_bin, detect_resources, is_secret

STASHED_EXECUTABLE = "/opt/programbench-stashed-executable-do-not-modify"  # Evaluator's
DEVIATION = ("branches share one machine: /workspace and the stashed executable are reset from "
             "the post-compile tar before each branch, instead of one fresh container per branch")
_DROP_PREFIXES = ("SWARMLAB_", "UV_", "PROGRAMBENCH_")
_DROP = {"VIRTUAL_ENV", "HF_HOME", "PYTHONUNBUFFERED", "PYTHONDONTWRITEBYTECODE"}
DOCKER_SHIM = """#!/bin/sh
# swarmlab grade: stands in for `docker cp <container>:<path> <dest>`, the evaluator's read of
# eval/results.xml; the container is this machine
if [ "$1" = cp ] && [ "$#" -eq 3 ]; then exec cp -- "${2#*:}" "$3"; fi
echo "swarmlab local docker shim: only 'cp <container>:<path> <dest>' is supported" >&2
exit 2
"""


def grade_env(base: Mapping[str, str], venv_bin: str | None = None) -> dict[str, str]:
    """The job's environment as a container would have it: no secrets, nothing of swarmlab's."""
    env = {k: v for k, v in base.items()
           if not is_secret(k) and k not in _DROP and not k.startswith(_DROP_PREFIXES)}
    if venv_bin and "PATH" in env:
        env["PATH"] = os.pathsep.join(p for p in env["PATH"].split(os.pathsep)
                                      if p.rstrip("/") != venv_bin.rstrip("/"))
    return env


def _text(data: bytes) -> str:
    """Bytes as `subprocess.run(text=True)` returns them (locale encoding, universal newlines)."""
    return data.decode(locale.getpreferredencoding(False)).replace("\r\n", "\n").replace("\r", "\n")


def _ancestors() -> set[int]:
    pids, pid = {os.getpid()}, os.getpid()
    while pid > 1:
        try:
            stat = Path(f"/proc/{pid}/stat").read_text()
            pid = int(stat.rsplit(")", 1)[1].split()[1])
        except (OSError, ValueError, IndexError):
            break
        pids.add(pid)
    return pids


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


def _pids() -> set[int]:
    return {int(e) for e in os.listdir("/proc") if e.isdigit()}


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


class LocalMachine:
    """The one filesystem every `LocalEvalEnvironment` of a grade shares (module docstring)."""

    def __init__(self, store: Path | str, *, workspace: str = "/workspace",
                 keep: tuple[str, ...] = (STASHED_EXECUTABLE,), base_image: str | None = None,
                 sweep: bool = False, environ: Mapping[str, str] | None = None) -> None:
        self.store = Path(store)
        (self.store / "images").mkdir(parents=True, exist_ok=True)
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

    def _paths(self) -> list[str]:
        return [self.workspace, *self.keep]

    def start(self, image: str) -> None:
        """What `docker run <image>` would give: the job's own state for the base image, the
        committed state (reset) for a reference `commit` returned."""
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
        self.env = {**machine.env, "PYTEST_XDIST_AUTO_NUM_WORKERS": str(cpus), **(env or {})}
        self._groups: list[int] = []
        machine.start(image)

    def execute(self, command: str, *, timeout: int | None = None) -> dict[str, Any]:
        timeout = timeout or self.default_timeout
        with tempfile.TemporaryFile() as out, tempfile.TemporaryFile() as err:
            proc = subprocess.Popen(["bash", "-lc", command], cwd=self.cwd, env=self.env,
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
            shutil.copy2(local_path, container_path)
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
        """Kill what this environment's commands left running: the members of their sessions
        (each command is a session leader, so a pid reused since is not a match)."""
        if not self._groups:
            return
        live = _sessions()
        for sid in self._groups:
            if sid in live:
                self._kill(sid)
        self._groups = []

    def __del__(self) -> None:
        self.cleanup()


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
          environ: Mapping[str, str] | None = None) -> dict[str, Any]:
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
                               sweep=sweep, environ=environ)
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
