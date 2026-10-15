"""ProgramBench: rebuild a program from its reference executable and docs, through one bash tool.

ProgramBench (Meta; arXiv 2605.03546; package `programbench` 1.2.5) gives an agent a workspace
holding a compiled reference program (`./executable`, execute-only), its bundled documentation and
test assets, but no source code and no internet. The agent writes an independent implementation
plus a `compile.sh` that builds a new `./executable`; the deliverable is the whole workspace. Hidden
pytest suites later run against the rebuilt binary in a fresh container. This world runs the
episode and keeps the workspace; it never grades (see "Grade seam" below).

It replicates the official harness's environment (mini-swe-agent 2.4.6 with
`config/benchmarks/programbench.yaml`), so a solo run is a valid leaderboard-style submission:
the task image, the workspace, the per-command semantics, the submission rule and the limits are
the benchmark's; only the agent harness differs.

Constructor
-----------
`ProgramBench(task, backend="local", workspace="/workspace", user="agent", step_limit=1000,
wall_limit_s=21600, command_timeout_s=180, lockdown=True,
pin_hosts=("api.anthropic.com", "huggingface.co", "hf.co"), max_output_bytes=1_000_000,
cpus=None, memory_gb=None, snapshot_exclude=(), check_task=True)`

- `task`: a ProgramBench instance id, `<owner>__<repo>.<short sha>` (`abishekvashok__cmatrix.5c082c6`).
  The image is `programbench/<task with "__" -> "_1776_">:task_cleanroom_v6` (`image_name`, the
  rule of `programbench.constants.image_name_from_instance_id` with the default org; the
  `PROGRAMBENCH_DOCKER_ORG` override is not read, so the image is a function of the spec).
- `backend`: `"local"` runs each command on this machine (swarmlab inside the task image, the HF
  Jobs shape) or `"docker"` starts the task image and runs each command with `docker exec`
  (mini-swe-agent's `DockerEnvironment`). Both behind `Backend`: `start()`, `execute(command,
  timeout) -> dict`, `tar(exclude)`, `restore_workspace(tar)`, `stop()`.
- `workspace`: the task directory, every command's cwd (`/workspace` in the image).
- `user`: who runs the commands. Docker: `--user <user>` (`None`: the image's default). Local:
  when swarmlab runs as root and the user exists, commands run as that user through
  `setpriv --reuid=<uid> --regid=<gid> --init-groups --` (util-linux, in the Ubuntu task images;
  chosen over `runuser` because it opens no PAM session and leaves the environment exactly as
  the world builds it, with HOME, USER, LOGNAME and SHELL from the user's passwd entry);
  otherwise as the current user (tests use `user=None`).
- `step_limit`: the benchmark's model-call limit (1000). The world does not enforce it (the
  official limit counts model calls, and one call may run several commands): the participant
  does, and in the one-round solo shape `check_experiment` requires
  `options.max_calls_per_turn >= step_limit` so the executor's cap cannot end the episode first.
- `wall_limit_s`: the 6 h wall-clock limit. Once the world's clock passes it, `terminal()` is
  true and `validate` refuses further commands (`wall_limit`).
- `command_timeout_s`: the per-command timeout (180 s).
- `lockdown`, `pin_hosts`: job-mode network and privilege lockdown (below); acts only with the
  local backend when swarmlab runs as root.
- `max_output_bytes`: a safety cap on one command's output (below).
- `cpus`, `memory_gb`: the resources the task text states and the docker flags use. `None`:
  docker uses the official 20 CPUs / 60 GB; the local backend detects them from the cgroup when
  lockdown applies (else unknown).
- `snapshot_exclude`: fnmatch patterns left out of the workspace snapshot (default none).
- `check_task`: at run initialisation (`check_experiment`), load the task's `task.yaml` from the
  `programbench` package (the `swarmlab[programbench]` extra) and refuse an unknown task. `False`
  skips it (tests, or an image built outside the package). Nothing else imports `programbench`,
  so building the world, the viewer, replay and reports work without the extra.

Every argument is constructor config recorded in `params`, so it is part of the spec and its
hash; `backend` is deliberately so (a run inside the task image and a Docker run are different
experiments). Snapshots carry no config: a fork or resume runs under its own spec's settings.

The bash tool and its feedback (exact)
--------------------------------------
One world tool, `bash(command: str)`, with mini-swe-agent's schema ("Execute a bash command";
`command`: "The bash command to execute"). `Outcome.feedback` is exactly the baseline
environment's result dict:

    {"output": str, "returncode": int, "exception_info": str}

`output` is stdout and stderr merged, decoded as UTF-8 with `errors="replace"`; a command that
finished has its line endings normalised as Python's text mode does (`\\r\\n` and `\\r` -> `\\n`),
the partial output of a timed-out one does not (both as in `docker.py`). `exception_info` is `""`
unless the command could not complete. There is no truncation for the agent's eyes: the
baseline's 5000/5000-character head/tail rendering past 10000 characters is the participant's
job (its observation template), not the world's. The one exception is `max_output_bytes`: output
larger than it keeps its first and last `max_output_bytes // 2` bytes around one line
`[swarmlab: N bytes of output elided (max_output_bytes=M)]`, so a runaway command cannot fill
memory, the event log and the model context; the head and tail a participant renders are then
unchanged (only its elided-character count differs). `None` disables the cap.

Official semantics replicated
-----------------------------
- Every call is a fresh `bash -lc <command>` with cwd = `workspace`: no cwd or environment
  persists between calls.
- The command environment: the baseline's `PAGER=cat MANPAGER=cat LESS=-R PIP_PROGRESS_BAR=off
  TQDM_DISABLE=1`. Docker passes only these (`docker exec -e`), as mini-swe-agent does. The local
  backend adds the inherited environment minus anything secret-looking (`ANTHROPIC_*`,
  `OPENAI_*`, `HF_TOKEN`, `*_KEY`, `*_TOKEN`, `*_SECRET`, `*_PASSWORD`, `*_CREDENTIALS`;
  `is_secret`) and minus swarmlab's own virtualenv (`VIRTUAL_ENV` and its `bin` on `PATH`), so the
  agent's `python3` is the image's; plus the thread-count variables below when the CPU count is
  known.
- Timeout (`command_timeout_s`): the call returns the output captured so far, `returncode: -1`
  and `exception_info` in the baseline's wording, `An error occurred while executing the command:
  Command '<command>' timed out after 180 seconds` (docker: the `docker exec` argument list, as
  `docker.py` prints it). The process keeps running, as under Docker, where only the waiting
  client is killed. The local backend starts the command in its own session writing to an
  unlinked temp file and stops waiting at the deadline.
- Submission: a command whose output's first non-blank line is exactly
  `COMPLETE_TASK_AND_SUBMIT_FINAL_OUTPUT` with return code 0 submits (`submission_of`). The world
  records `submitted`, the rest of the output as `submission`, who and when; `terminal()` is true
  from then on and `validate` refuses every later `bash` (`Ack(ok=False, error="submitted")`).
  There is no `submit` tool.
- Docker: `docker run -d --name <n> -w <workspace> --rm --network none --cpus 20 --memory 60g
  --memory-swap 60g --user agent --cap-drop SYS_PTRACE <image> sleep 7h`, then per call
  `docker exec -w <workspace> -e K=V ... <id> bash -lc <command>`, as mini-swe-agent's
  `DockerEnvironment` with the benchmark's `run_args`.
- Before the first command, the world runs the baseline's setup command
  `git config user.name "mini-swe-agent" && git config user.email "mini-swe-agent@proton.me"`
  (only when the workspace is a git repo root, so a local workspace inside another repo is never
  touched); it is not a step and is not shown to the agent.
- The task text (`description()`) is the baseline's `system_template` followed by its
  `instance_template`, rendered as mini-swe-agent renders them (`prompt_parts`).

Documented deviations (world-relevant items, from the settled plan)
-------------------------------------------------------------------
- Network: DNS blackhole + pinned API/Hub hosts + sudo removed, instead of `--network none`.
  Raw-IP connections remain possible. Unauthorized access is found from the event log, not a
  report.
- Resources: default `cpu-upgrade` (8 vCPU, 32 GB) vs official 20 CPU / 60 GB; flavor is a launch
  option. One added prompt sentence states the real CPU count and memory; MAKEFLAGS,
  CARGO_BUILD_JOBS and similar are set to the real count (the job's `nproc` lies).
- API calls go through swarmlab's providers (retries, timeouts, caching differ from litellm; the
  words sent are identical). Tool results reach a swarmlab `llm` agent in a JSON wrapper (only
  relevant to the optional non-faithful participant).
- Solo shape: one round, one turn (no mid-episode snapshot, so no resume/fork of a half episode).

Decisions the contract leaves open
----------------------------------
- Lazy backend start. `reset` only records state: `swarmlab prompts`, `replay` and `report`
  call it, and must never start a container or lock a machine down. The backend starts (docker
  run, or the local lockdown) at the first `bash` call, or in `begin_round` when a restored
  workspace is waiting to be materialised (resume, fork). The time a backend start takes (an
  image pull) is not on the world's clock, as the baseline starts its clock after the container.
- Clock. `elapsed_s` runs from the first `begin_round` of a process and accumulates across
  resumes (time between processes is not counted). It is updated after every command and in
  `end_round`, and `score()` and `terminal()` read the stored value, so replay reproduces them.
- Docker cleanup. The World contract has no end-of-run hook, so `end_round` captures the
  workspace tar (the snapshot that follows uses it) and removes the container (`docker rm -f`)
  when `terminal()` or when the round equals the `max_rounds` recorded by `check_experiment`. A
  run that ends otherwise (budget, stop condition, crash) relies on the safety net: a
  `weakref.finalize` per container, which runs when the backend is collected and at interpreter
  exit. Deep copies and pickles of the world never carry the backend handle (`__getstate__`), so
  a copy cannot remove a container it does not own.
- Lockdown seam. When `lockdown`, `backend == "local"` and `os.geteuid() == 0`, the backend's
  start resolves every `pin_hosts` host (`socket.getaddrinfo`) and appends it to `/etc/hosts`,
  writes `nameserver 0.0.0.0` to `/etc/resolv.conf` (both in place: they are often bind mounts),
  removes the agent user's sudo (its rules in `/etc/sudoers.d/*` and `/etc/sudoers`, then
  `chmod 0000 /usr/bin/sudo`). `reset` reads the real CPU quota (`/sys/fs/cgroup/cpu.max`,
  `<quota> <period>` or `max`) and memory (`/sys/fs/cgroup/memory.max`, or `/proc/meminfo` when
  unlimited) under the same condition. The helpers take their paths as arguments
  (`LockdownPaths`, `detect_resources`, `pin_hosts_file`, `blackhole_dns`, `remove_sudo`) so they
  are tested on temp files. With a known CPU count n the local backend sets `MAKEFLAGS=-j<n>`,
  `CARGO_BUILD_JOBS`, `OMP_NUM_THREADS` and `GOMAXPROCS` to n (the floor, at least 1).
- Resource sentence. `description()` ends with one sentence, `This machine has 8 CPUs and 32 GB
  of memory.`, from `cpus`/`memory_gb` (explicit, detected, or docker's 20/60); a half is kept
  only when it is not a whole number. It is omitted when both are unknown.
- Template variables. The instance template's `{{system}} {{release}} {{version}} {{machine}}`
  are `platform.uname()` of the machine running swarmlab, recorded at `reset`: what
  mini-swe-agent's environments substitute (the Docker host's kernel is the container's).
- A local command returns when its `bash` exits; a background process that keeps the output
  open does not hold the call (the temp file is read as it is).

Snapshots, restore and the grade seam
-------------------------------------
`snapshot()` is a pickle of `{"meta": <plain state dict>, "workspace_tar": <bytes | None>}`.
The tar is the WHOLE workspace minus `snapshot_exclude`, gzip-compressed deterministically
(sorted names, integer mtimes, gzip header mtime 0), so an unchanged workspace gives identical
bytes and the blob store keeps one copy. The local backend builds it in Python
(`workspace_tar`); docker runs `tar --sort=name --format=gnu ... | gzip -n` in the container as
root (the reference binary is execute-only). Unreadable files (a non-root local run) are skipped
and listed in `snapshot_skipped`. `None` means the docker backend never started: the workspace
is the image's, untouched.

`restore(blob)` touches no container and no file: the meta goes into ordinary attributes and the
tar waits in `_pending_workspace`, so the viewer can build a world from any round's snapshot and
call `render_state()` without a backend. `begin_round` materialises it (resume, fork): it starts
the backend, empties the workspace and extracts the tar.

Grade seam: grading is a later, separate step. It reads the run's last snapshot
(`workspace_tar_of(blob)`) and hands that tar to the official evaluator as
`submission.tar.gz`; `score()["graded"]` stays False here.

Evaluator side
--------------
- `score()` (cheap, every round): `{"submitted", "steps", "elapsed_s", "graded": False}`.
- `end_round(r)` -> `{"round", "steps", "round_steps", "submitted", "elapsed_s"}` (logged as
  `round_resolved`), plus the docker cleanup above.
- `render_state()`: task, backend, steps, submitted, elapsed, the workspace's top-level entries
  with sizes (from the snapshot's meta, never the filesystem) and the last 10 commands (first 200
  characters each) with their return codes.
- `observe(agent)`: a short text (workspace, the agent's own command count, submitted or not);
  `private` is empty. Nothing agent-visible reveals correctness (there is none until grading).
- `requires = Requirements(commit="immediate")`: a ReAct coding loop needs each command's output
  before its next call.

Fake script
-----------
`fake:programbench_echo` runs `ls`, writes a trivial `compile.sh` that creates `./executable`,
runs it, then submits with `echo COMPLETE_TASK_AND_SUBMIT_FINAL_OUTPUT` and ends its turn: a free
plumbing check for the bash tool, submission, snapshot and replay.

Source of the task text: mini-swe-agent (MIT License, Copyright (c) 2025 Kilian A. Lieret and
Carlos E. Jimenez), `src/minisweagent/config/benchmarks/programbench.yaml` at commit
04d809ceab9df28f9adaed044884180159172930, copied verbatim into `SYSTEM_TEMPLATE` and
`INSTANCE_TEMPLATE`.
"""
from __future__ import annotations

import fnmatch
import gzip
import io
import logging
import os
import pickle
import platform
import pwd
import random
import shlex
import socket
import subprocess
import sys
import tarfile
import tempfile
import time
import uuid
import weakref
from collections.abc import Callable, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any, ClassVar, Protocol

from ..ids import AgentId
from ..view import Observation, text_observation
from .base import Ack, Action, Outcome, Requirements, World, tool

SUBMIT_MARKER = "COMPLETE_TASK_AND_SUBMIT_FINAL_OUTPUT"
IMAGE_ORG = "programbench"
IMAGE_TAG = "task_cleanroom_v6"
EXTRA = "swarmlab[programbench]"
BASELINE_ENV: dict[str, str] = {"PAGER": "cat", "MANPAGER": "cat", "LESS": "-R",
                                "PIP_PROGRESS_BAR": "off", "TQDM_DISABLE": "1"}
THREAD_VARS = ("MAKEFLAGS", "CARGO_BUILD_JOBS", "OMP_NUM_THREADS", "GOMAXPROCS")
GIT_SETUP = ('if [ -d .git ]; then git config user.name "mini-swe-agent" && '
             'git config user.email "mini-swe-agent@proton.me"; fi')
DOCKER_CPUS, DOCKER_MEMORY_GB = 20, 60          # programbench.yaml run_args
DOCKER_CONTAINER_TIMEOUT = "7h"                 # programbench.yaml container_timeout
DOCKER_START_TIMEOUT_S = 1800                   # a pull of a 1-2 GB image; not task semantics
HISTORY_KEEP = 10
_ERROR_PREFIX = "An error occurred while executing the command: "
log = logging.getLogger(__name__)

# ---- the official task text (mini-swe-agent, MIT; see the module docstring) -----------------

SYSTEM_TEMPLATE = """You are a helpful assistant that can interact with a computer.

Your task is to recreate a program from scratch based only on its observable behavior and bundled documentation.
You are given a reference executable and its documentation.
Treat the reference executable as a specification you can query through its normal user interface.

Your submission must be an independent implementation.
Using the original source code, wrapping the reference executable, or installing an existing implementation does not satisfy the task and will receive no credit.
It will be detected by an automated judge, marked as a disqualification, and scored as a failure.

<RULES>
## What is NOT allowed

### 1. Using an existing implementation

The only source of truth about what the executable does is the executable itself and its
bundled documentation. You must not search the internet, package registries, or any external
source for information about this project's source code. Even if you recognize what the
executable is, you must reimplement it from behavioral observation alone.

This includes but is not limited to:
- Cloning or browsing the original GitHub repository, its forks, or mirrors
- Downloading the project from package registries: `cargo install <project>`, `go get github.com/<org>/<project>`, `pip install <project>`, `apt-get source <project>`, `npm install <project>`, etc.
- Fetching source tarballs from project websites (e.g., `curl https://lua.org/ftp/lua-5.5.0.tar.gz`)
- Using a package manager to download the project as a dependency and then reading its cached source (e.g., navigating into `~/.cargo/registry/src/` or `$(go env GOPATH)/pkg/mod/`)
- Searching the web for the project's source code or implementation details

### 2. Wrapping or reusing the reference executable

Your submission must be a genuine reimplementation. The reference `./executable` is for
observation only — your final solution must not depend on it or any other pre-built version
of the same tool at runtime.

This includes but is not limited to:
- Writing a wrapper script that delegates to the reference executable (e.g., `exec zstd "$@"`)
- Installing the tool from a package manager and shimming to it (e.g., `apt-get install nnn && cp $(which nnn) ./executable`)
- Writing a `compile.sh` that only runs `chmod +x ./executable` or copies the reference executable (`cp ./executable ./executable`)
- Building a binary whose main function shells out to an external tool (e.g., `Command::new("miniserve").args(args).exec()`)
- Re-linking prebuilt `.o` object files found in the workspace without writing new source code

### 3. How the reference executable may be observed

All information about the reference `./executable` must be obtained by interacting with it
through its normal user interface (CLI flags, stdin/stdout, etc.).
- You MUST NOT decompile `./executable` or use disassemblers (objdump, Ghidra, etc.) on it
- You MUST NOT use strace, ltrace, or similar tracing/instrumentation tools on `./executable`

Note: this restriction applies ONLY to the reference `./executable`. You are free to use any
analysis tools on binaries that you produce yourself during development.

## What IS allowed

- Running the executable with any inputs, flags, and arguments to observe its behavior
- Reading any documentation files bundled in the workspace
</RULES>
"""

INSTANCE_TEMPLATE = """## Task context

Build a new implementation from scratch that matches the behavior of the reference executable.
The executable is located at `./executable` in the workspace root.

You also have access to the existing documentation.

## Your task

Implement a program whose externally observable behavior matches the reference executable.

No project-specific dependencies are pre-installed.
You do NOT have access to the internet.
**IMPORTANT**: Make sure that the executable(s) and everything else that is an artifact is not committed, i.e., is in your `.gitignore` file.
Finally, commit your changes.

Make sure that you have a `./compile.sh` file that produces an executable `./executable` in the workspace root.
`compile.sh` should be executable and should install any dependencies needed to compile the executable.
If your compile.sh fails to compile on a fresh checkout, your task has failed.

## Important: Build an independent implementation

Your goal is to create an independently authored implementation that reproduces the reference executable's externally observable behavior.
Use the bundled documentation and experiments with the reference executable to determine the required behavior.

Attempting to obtain source code — whether successful or not — or wrapping/reusing the
reference executable does not satisfy the task and will receive no credit.
See the full rules in the system prompt above. Key points:

- Do NOT search the internet, clone repos, or download the project from any package registry
- Do NOT wrap, shim, or delegate to the reference `./executable` or any installed version of the same tool
- Do NOT decompile the reference `./executable` or use strace/ltrace on it (analyzing your own binaries is fine)
- You SHOULD extensively test the executable to understand its behavior before writing code.
  If you are dealing with a TUI, tmux/libtmux has been installed to help you test/inspect/it.

## Recommended Workflow

1. Explore all documentation files
2. Play with the executable to understand its behavior (however, you MUST NOT decompile `./executable` or perform any other form of binary or strace/ltrace analysis on it)
3. Write the source code to implement the behavior

## Command Execution Rules

You can execute bash commands and edit files to implement the necessary changes.

You are operating in an environment where

1. You issue at least one command
2. The system executes the command(s) in a subshell
3. You see the result(s)
4. You write your next command(s)

Each response should include:

1. **Reasoning text** where you explain your analysis and plan
2. At least one tool call with your command

**CRITICAL REQUIREMENTS:**

- Your response SHOULD include reasoning text explaining what you're doing
- Your response MUST include AT LEAST ONE bash tool call
- Directory or environment variable changes are not persistent. Every action is executed in a new subshell.
- However, you can prefix any action with `MY_ENV_VAR=MY_VALUE cd /path/to/working/dir && ...` or write/load environment variables from files
- Submit your changes and finish your work by issuing the following command: `echo COMPLETE_TASK_AND_SUBMIT_FINAL_OUTPUT`.
  Do not combine it with any other command. <important>After this command, you cannot continue working on this task.</important>

Example of a CORRECT response:
<example_response>
I need to understand the structure of the repository first. Let me check what files are in the current directory to get a better understanding of the codebase.

[Makes bash tool call with {"command": "ls -la"} as arguments]
</example_response>

<system_information>
{{system}} {{release}} {{version}} {{machine}}
</system_information>

## Useful command examples

python is available as python3

### Create a new file:

```bash
cat <<'EOF' > newfile.py
import numpy as np
hello = "world"
print(hello)
EOF
```

### Edit files with sed:

```bash
# Replace all occurrences
sed -i 's/old_string/new_string/g' filename.py

# Replace only first occurrence
sed -i 's/old_string/new_string/' filename.py

# Replace first occurrence on line 1
sed -i '1s/old_string/new_string/' filename.py

# Replace all occurrences in lines 1-10
sed -i '1,10s/old_string/new_string/g' filename.py
```

### View file content:

```bash
# View specific lines with numbers
nl -ba filename.py | sed -n '10,20p'
```

### Any other command you want to run

```bash
anything
```
"""


# ---- pure helpers ----------------------------------------------------------------------------

def image_name(task: str, tag: str = IMAGE_TAG) -> str:
    """The task image: `programbench/<task with "__" -> "_1776_">:<tag>`."""
    return f"{IMAGE_ORG}/{task.replace('__', '_1776_')}:{tag}"


def check_task_id(task: Any) -> str:
    """`task` if it looks like an instance id (`<owner>__<repo>.<sha>`), else ValueError."""
    if not isinstance(task, str) or "__" not in task or "." not in task.split("__", 1)[1] \
            or any(c.isspace() or c in "/:" for c in task):
        raise ValueError(f"task must be a ProgramBench instance id such as "
                         f"'abishekvashok__cmatrix.5c082c6', got {task!r}")
    return task


def task_metadata(task: str) -> dict:
    """The task's `task.yaml` from the `programbench` package (repository, commit, language,
    difficulty, ...). ImportError naming the extra when the package is missing; ValueError for a
    task the package does not know."""
    try:
        from programbench.constants import TASK_YAML, TASKS_DIR  # type: ignore[import-not-found]
    except ImportError as e:
        raise ImportError(f"the programbench world's task check needs the `programbench` package: "
                          f'pip install "{EXTRA}" (or set check_task: false)') from e
    import yaml

    path = Path(TASKS_DIR) / task / TASK_YAML
    if not path.is_file():
        raise ValueError(f"unknown ProgramBench task {task!r} (no {path})")
    return dict(yaml.safe_load(path.read_text()) or {})


def prompt_parts(uname: dict | None = None) -> tuple[str, str]:
    """(system text, instance text): the baseline's two templates rendered as mini-swe-agent
    renders them (jinja2, StrictUndefined), with `system`, `release`, `version` and `machine`
    from `uname` (default: this machine's `platform.uname()`)."""
    from jinja2 import StrictUndefined, Template

    names = ("system", "release", "version", "machine")
    if uname is None:
        uname = platform.uname()._asdict()
    values = {k: uname.get(k, "") for k in names}
    return (Template(SYSTEM_TEMPLATE, undefined=StrictUndefined).render(**values),
            Template(INSTANCE_TEMPLATE, undefined=StrictUndefined).render(**values))


def _number(x: float) -> str:
    return str(round(x)) if abs(x - round(x)) < 0.05 else f"{x:.1f}"


def resource_sentence(cpus: float | None, memory_gb: float | None) -> str:
    """`This machine has 8 CPUs and 32 GB of memory.` (either half alone; "" when both unknown)."""
    parts = []
    if cpus:
        parts.append(f"{_number(cpus)} CPU{'' if _number(cpus) == '1' else 's'}")
    if memory_gb:
        parts.append(f"{_number(memory_gb)} GB of memory")
    return f"This machine has {' and '.join(parts)}." if parts else ""


def thread_env(cpus: float | None) -> dict[str, str]:
    """MAKEFLAGS, CARGO_BUILD_JOBS, OMP_NUM_THREADS, GOMAXPROCS for a CPU count ({} if unknown)."""
    if not cpus:
        return {}
    n = str(max(1, int(cpus)))
    return {"MAKEFLAGS": f"-j{n}", "CARGO_BUILD_JOBS": n, "OMP_NUM_THREADS": n, "GOMAXPROCS": n}


_SECRET_PREFIXES = ("ANTHROPIC_", "OPENAI_")
_SECRET_SUFFIXES = ("_KEY", "_TOKEN", "_SECRET", "_PASSWORD", "_CREDENTIALS")


def is_secret(name: str) -> bool:
    """True for an environment variable that must never reach the agent's commands."""
    up = name.upper()
    return (up.startswith(_SECRET_PREFIXES) or up.endswith(_SECRET_SUFFIXES)
            or up in ("HF_TOKEN", "TOKEN", "SECRET", "PASSWORD"))


def command_env(base: dict[str, str], extra: dict[str, str] | None = None,
                venv_bin: str | None = None) -> dict[str, str]:
    """The local backend's command environment: `base` minus secrets and swarmlab's virtualenv
    (`VIRTUAL_ENV`, `venv_bin` on PATH), then the baseline variables, then `extra`."""
    env = {k: v for k, v in base.items() if not is_secret(k) and k != "VIRTUAL_ENV"}
    if venv_bin and "PATH" in env:
        env["PATH"] = os.pathsep.join(p for p in env["PATH"].split(os.pathsep)
                                      if p.rstrip("/") != venv_bin.rstrip("/"))
    return {**env, **BASELINE_ENV, **(extra or {})}


def _venv_bin() -> str | None:
    return os.path.join(sys.prefix, "bin") if sys.prefix != sys.base_prefix else None


def submission_of(output: str, returncode: int) -> str | None:
    """The submission text when `output` submits (first non-blank line is the marker and the
    return code is 0): the rest of the output, as the baseline's `_check_finished`; else None."""
    lines = output.lstrip().splitlines(keepends=True)
    if lines and lines[0].strip() == SUBMIT_MARKER and returncode == 0:
        return "".join(lines[1:])
    return None


def _decode(data: bytes | str | None, translate: bool) -> str:
    if data is None:
        return ""
    text = data if isinstance(data, str) else data.decode("utf-8", errors="replace")
    return text.replace("\r\n", "\n").replace("\r", "\n") if translate else text


def _elision(total: int, cap: int) -> bytes:
    return f"\n[swarmlab: {total - cap} bytes of output elided (max_output_bytes={cap})]\n".encode()


def cap_bytes(data: bytes, cap: int | None) -> bytes:
    """`data`, or its first and last `cap // 2` bytes around an elision line when longer."""
    if cap is None or len(data) <= cap:
        return data
    head = cap // 2
    return data[:head] + _elision(len(data), cap) + data[len(data) - (cap - head):]


def _read_capped(f: Any, cap: int | None) -> bytes:
    size = f.seek(0, os.SEEK_END)
    f.seek(0)
    if cap is None or size <= cap:
        return f.read(size)
    head = cap // 2
    first = f.read(head)
    f.seek(size - (cap - head))
    return first + _elision(size, cap) + f.read(cap - head)


def _result(output: str, returncode: int, exception_info: str = "") -> dict:
    return {"output": output, "returncode": returncode, "exception_info": exception_info}


# ---- lockdown and resources (the seam: every path is an argument) ------------------------------

@dataclass(frozen=True)
class LockdownPaths:
    hosts: str = "/etc/hosts"
    resolv_conf: str = "/etc/resolv.conf"
    sudoers: str = "/etc/sudoers"
    sudoers_d: str = "/etc/sudoers.d"
    sudo_bin: str = "/usr/bin/sudo"
    cpu_max: str = "/sys/fs/cgroup/cpu.max"
    memory_max: str = "/sys/fs/cgroup/memory.max"
    meminfo: str = "/proc/meminfo"


def _read(path: str) -> str | None:
    try:
        return Path(path).read_text()
    except OSError:
        return None


def detect_resources(cpu_max: str = LockdownPaths.cpu_max,
                     memory_max: str = LockdownPaths.memory_max,
                     meminfo: str = LockdownPaths.meminfo) -> tuple[float | None, float | None]:
    """(CPUs, memory in GB) from cgroup v2 files: `cpu.max` is `<quota> <period>` or `max ...`
    (no quota: the CPUs this process may use); `memory.max` is bytes or `max` (no limit:
    `MemTotal` of `meminfo`). A missing file falls back the same way; GB = 2**30 bytes."""
    cpus: float | None = None
    text = _read(cpu_max)
    if text and text.split() and text.split()[0] != "max":
        quota, period = text.split()[:2]
        cpus = int(quota) / int(period)
    if cpus is None:
        cpus = float(len(os.sched_getaffinity(0)) if hasattr(os, "sched_getaffinity")
                     else (os.cpu_count() or 0)) or None
    mem: float | None = None
    text = _read(memory_max)
    if text and text.strip() and text.strip() != "max":
        mem = int(text.strip()) / 2**30
    if mem is None:
        for line in (_read(meminfo) or "").splitlines():
            if line.startswith("MemTotal:"):
                mem = int(line.split()[1]) * 1024 / 2**30
                break
    return cpus, mem


Resolver = Callable[..., list]


def pin_hosts_file(hosts: Sequence[str], hosts_path: str,
                   resolve: Resolver = socket.getaddrinfo) -> dict[str, str]:
    """Resolve every host (IPv4 preferred) and append `<ip> <host>` lines to `hosts_path`, in
    place. Returns {host: ip}; RuntimeError when a host does not resolve."""
    pinned: dict[str, str] = {}
    for host in hosts:
        try:
            infos = resolve(host, 443, type=socket.SOCK_STREAM)
        except OSError as e:
            raise RuntimeError(f"lockdown: cannot resolve {host!r} to pin it: {e}") from e
        infos = sorted(infos, key=lambda i: i[0] != socket.AF_INET)
        if not infos:
            raise RuntimeError(f"lockdown: {host!r} resolved to no address")
        pinned[host] = infos[0][4][0]
    with open(hosts_path, "a") as f:
        f.write("\n# swarmlab programbench lockdown: pinned hosts\n")
        f.writelines(f"{ip}\t{host}\n" for host, ip in pinned.items())
    return pinned


def blackhole_dns(resolv_path: str) -> None:
    """Point the resolver at 0.0.0.0 (written in place: often a bind mount)."""
    with open(resolv_path, "w") as f:
        f.write("nameserver 0.0.0.0\n")


def _rule_for(line: str, user: str) -> bool:
    parts = line.split()
    return bool(parts) and parts[0] == user


def remove_sudo(user: str, sudoers: str = LockdownPaths.sudoers,
                sudoers_d: str = LockdownPaths.sudoers_d,
                sudo_bin: str = LockdownPaths.sudo_bin) -> dict:
    """Delete the `sudoers_d` files holding a rule for `user`, drop its rule lines from
    `sudoers`, and `chmod 0000` the sudo binary. Returns what was done."""
    done: dict[str, Any] = {"removed_files": [], "sudoers_lines": 0, "sudo_disabled": False}
    d = Path(sudoers_d)
    if d.is_dir():
        for f in sorted(d.iterdir()):
            text = _read(str(f))
            if f.is_file() and text and any(_rule_for(line, user) for line in text.splitlines()):
                f.unlink()
                done["removed_files"].append(f.name)
    text = _read(sudoers)
    if text is not None:
        lines = text.splitlines(keepends=True)
        kept = [line for line in lines if not _rule_for(line, user)]
        if len(kept) != len(lines):
            with open(sudoers, "w") as f:
                f.writelines(kept)
            done["sudoers_lines"] = len(lines) - len(kept)
    if os.path.exists(sudo_bin):
        os.chmod(sudo_bin, 0)
        done["sudo_disabled"] = True
    return done


def apply_lockdown(user: str | None, hosts: Sequence[str], paths: LockdownPaths | None = None,
                   resolve: Resolver = socket.getaddrinfo) -> dict:
    """Pin `hosts`, blackhole DNS, remove `user`'s sudo, in that order (resolution needs DNS)."""
    paths = paths or LockdownPaths()
    pinned = pin_hosts_file(hosts, paths.hosts, resolve)
    blackhole_dns(paths.resolv_conf)
    sudo = remove_sudo(user, paths.sudoers, paths.sudoers_d, paths.sudo_bin) if user else {}
    return {"applied": True, "pinned": pinned, "sudo": sudo}


# ---- workspace tars ----------------------------------------------------------------------------

def _excluded(rel: str, patterns: Sequence[str]) -> bool:
    name = rel.rsplit("/", 1)[-1]
    return any(fnmatch.fnmatchcase(rel, p) or fnmatch.fnmatchcase(name, p) for p in patterns)


def workspace_tar(root: str | Path, exclude: Sequence[str] = ()) -> tuple[bytes, list[str]]:
    """(deterministic .tar.gz of everything under `root`, relative paths skipped as unreadable).

    Names are relative (no `./`), walked in sorted order; mtimes are whole seconds; the gzip
    header has mtime 0 and no file name. A path matching an `exclude` pattern (fnmatch against
    the relative path or the base name) is left out, a directory with its contents. Symlinks are
    stored as links, sockets are skipped."""
    root = str(root)
    skipped: list[str] = []
    buf = io.BytesIO()

    def tidy(ti: tarfile.TarInfo) -> tarfile.TarInfo:
        ti.mtime = int(ti.mtime)
        return ti

    with gzip.GzipFile(filename="", mode="wb", fileobj=buf, mtime=0, compresslevel=6) as gz, \
            tarfile.open(fileobj=gz, mode="w|", format=tarfile.PAX_FORMAT) as tar:
        for dirpath, dirnames, filenames in os.walk(root):
            rel_dir = os.path.relpath(dirpath, root)
            rel_dir = "" if rel_dir == "." else rel_dir
            keep = []
            for name in sorted(dirnames + filenames):
                rel = f"{rel_dir}/{name}" if rel_dir else name
                if _excluded(rel, exclude):
                    continue
                full = os.path.join(dirpath, name)
                try:
                    tar.add(full, arcname=rel, recursive=False, filter=tidy)
                except (PermissionError, FileNotFoundError):
                    skipped.append(rel)
                    continue
                if name in dirnames and not os.path.islink(full):
                    keep.append(name)
            dirnames[:] = keep
    return buf.getvalue(), skipped


def workspace_entries(tar_bytes: bytes | None) -> list[dict]:
    """Top-level entries of a workspace tar: `{"name", "type", "bytes"}` (bytes summed over
    regular files below the entry), sorted by name."""
    if not tar_bytes:
        return []
    out: dict[str, dict] = {}
    with tarfile.open(fileobj=io.BytesIO(tar_bytes), mode="r:gz") as tar:
        for m in tar:
            name = m.name.removeprefix("./").strip("/")
            if not name or name == ".":
                continue
            top = name.split("/", 1)[0]
            entry = out.setdefault(top, {"name": top, "type": "dir", "bytes": 0})
            if name == top:
                entry["type"] = ("dir" if m.isdir() else "symlink" if m.issym() else
                                 "file" if m.isfile() else "other")
            if m.isfile():
                entry["bytes"] += m.size
    return [out[k] for k in sorted(out)]


def _clear_dir(root: str) -> None:
    if os.path.realpath(root) == "/":
        raise ValueError("refusing to empty / as a workspace")
    for entry in os.scandir(root):
        if entry.is_dir(follow_symlinks=False):
            import shutil

            shutil.rmtree(entry.path)
        else:
            os.unlink(entry.path)


def extract_workspace(tar_bytes: bytes, root: str | Path) -> None:
    """Empty `root` and extract a workspace tar into it (as root, ownership comes back too)."""
    root = str(root)
    os.makedirs(root, exist_ok=True)
    _clear_dir(root)
    with tarfile.open(fileobj=io.BytesIO(tar_bytes), mode="r:gz") as tar:
        tar.extractall(root, filter="tar")


def workspace_tar_of(snapshot_blob: bytes) -> bytes | None:
    """The workspace tar inside a ProgramBench world snapshot (the grade step's input; None: the
    image's untouched workspace)."""
    return pickle.loads(snapshot_blob).get("workspace_tar")


# ---- backends -------------------------------------------------------------------------------

class Backend(Protocol):
    started: bool

    def start(self) -> dict: ...
    def execute(self, command: str, timeout: float) -> dict: ...
    def tar(self, exclude: Sequence[str]) -> tuple[bytes | None, list[str]]: ...
    def restore_workspace(self, tar_bytes: bytes) -> None: ...
    def stop(self) -> None: ...


class LocalBackend:
    """Commands on this machine: `bash -lc` in its own session, output to an unlinked temp file."""

    def __init__(self, workspace: str, user: str | None, extra_env: dict[str, str],
                 max_output_bytes: int | None, lockdown: Callable[[], dict] | None = None) -> None:
        self.workspace = workspace
        self.user = user
        self.extra_env = dict(extra_env)
        self.max_output_bytes = max_output_bytes
        self.lockdown = lockdown
        self.started = False
        self.prefix: list[str] = []
        self.env: dict[str, str] = {}

    def start(self) -> dict:
        record: dict[str, Any] = {"applied": False}
        if self.lockdown is not None:
            record = self.lockdown()
        os.makedirs(self.workspace, exist_ok=True)
        identity: dict[str, str] = {}
        run_as = None
        if self.user and os.geteuid() == 0:
            try:
                pw = pwd.getpwnam(self.user)
            except KeyError:
                pw = None
            if pw is not None:
                run_as = self.user
                self.prefix = ["setpriv", f"--reuid={pw.pw_uid}", f"--regid={pw.pw_gid}",
                               "--init-groups", "--"]
                identity = {"HOME": pw.pw_dir, "USER": pw.pw_name, "LOGNAME": pw.pw_name,
                            "SHELL": pw.pw_shell or "/bin/bash"}
        self.env = command_env(dict(os.environ), {**identity, **self.extra_env}, _venv_bin())
        self.started = True
        record["run_as"] = run_as or pwd.getpwuid(os.geteuid()).pw_name
        return record

    def execute(self, command: str, timeout: float) -> dict:
        with tempfile.TemporaryFile() as out:
            try:
                proc = subprocess.Popen([*self.prefix, "bash", "-lc", command], cwd=self.workspace,
                                        env=self.env, stdin=subprocess.DEVNULL, stdout=out,
                                        stderr=subprocess.STDOUT, start_new_session=True)
            except OSError as e:
                return _result("", -1, f"{_ERROR_PREFIX}{e}")
            try:
                rc = proc.wait(timeout=timeout)
            except subprocess.TimeoutExpired:
                # the process keeps running (Docker semantics); Popen reaps it once it exits
                data = _read_capped(out, self.max_output_bytes)
                err = subprocess.TimeoutExpired(command, timeout)
                return _result(_decode(data, translate=False), -1, f"{_ERROR_PREFIX}{err}")
            return _result(_decode(_read_capped(out, self.max_output_bytes), translate=True), rc)

    def tar(self, exclude: Sequence[str]) -> tuple[bytes | None, list[str]]:
        if not os.path.isdir(self.workspace):
            return None, []
        return workspace_tar(self.workspace, exclude)

    def restore_workspace(self, tar_bytes: bytes) -> None:
        extract_workspace(tar_bytes, self.workspace)

    def stop(self) -> None:
        self.started = False


def _docker_rm(executable: str, container_id: str) -> None:
    try:
        subprocess.run([executable, "rm", "-f", container_id], capture_output=True, timeout=60,
                       check=False)
    except Exception as e:  # noqa: BLE001 - best-effort cleanup (interpreter exit, daemon gone)
        log.warning("programbench: could not remove container %s: %s", container_id, e)


class DockerBackend:
    """The task image under Docker, driven as mini-swe-agent's `DockerEnvironment` drives it."""

    def __init__(self, image: str, workspace: str, user: str | None, cpus: float,
                 memory_gb: float, max_output_bytes: int | None,
                 executable: str = "docker") -> None:
        self.image = image
        self.workspace = workspace
        self.user = user
        self.cpus = cpus
        self.memory_gb = memory_gb
        self.max_output_bytes = max_output_bytes
        self.executable = executable
        self.container_id: str | None = None
        self.started = False
        self._finalizer: weakref.finalize | None = None

    def run_command(self, name: str) -> list[str]:
        mem = (f"{int(self.memory_gb)}g" if float(self.memory_gb).is_integer()
               else f"{int(self.memory_gb * 1024)}m")
        run_args = ["--rm", "--network", "none", "--cpus", _number(self.cpus), "--memory", mem,
                    "--memory-swap", mem, *(["--user", self.user] if self.user else []),
                    "--cap-drop", "SYS_PTRACE"]
        return [self.executable, "run", "-d", "--name", name, "-w", self.workspace, *run_args,
                self.image, "sleep", DOCKER_CONTAINER_TIMEOUT]

    def start(self) -> dict:
        name = f"swarmlab-programbench-{uuid.uuid4().hex[:8]}"
        result = subprocess.run(self.run_command(name), capture_output=True, text=True,
                                timeout=DOCKER_START_TIMEOUT_S, check=True)
        self.container_id = result.stdout.strip()
        self._finalizer = weakref.finalize(self, _docker_rm, self.executable, self.container_id)
        self.started = True
        return {"applied": False, "container": self.container_id, "image": self.image}

    def exec_command(self, command: str) -> list[str]:
        env = [a for k, v in BASELINE_ENV.items() for a in ("-e", f"{k}={v}")]
        return [self.executable, "exec", "-w", self.workspace, *env, str(self.container_id),
                "bash", "-lc", command]

    def execute(self, command: str, timeout: float) -> dict:
        try:
            result = subprocess.run(self.exec_command(command), timeout=timeout, check=False,
                                    stdout=subprocess.PIPE, stderr=subprocess.STDOUT)
            data = cap_bytes(result.stdout or b"", self.max_output_bytes)
            return _result(_decode(data, translate=True), result.returncode)
        except Exception as e:  # noqa: BLE001 - the baseline reports every failure the same way
            raw = getattr(e, "output", None)
            raw = cap_bytes(raw, self.max_output_bytes) if isinstance(raw, bytes) else raw
            return _result(_decode(raw, translate=False), -1, f"{_ERROR_PREFIX}{e}")

    def tar(self, exclude: Sequence[str]) -> tuple[bytes | None, list[str]]:
        if not self.started:
            return None, []
        excl = " ".join(shlex.quote(f"--exclude={p}") for p in exclude)
        script = (f"set -o pipefail; cd {shlex.quote(self.workspace)} && "
                  f"tar --sort=name --format=gnu {excl} -cf - . | gzip -n")
        result = subprocess.run([self.executable, "exec", "-u", "root", str(self.container_id),
                                 "bash", "-c", script], capture_output=True, timeout=3600,
                                check=False)
        if result.returncode not in (0, 1):  # 1: a file changed while it was read
            raise RuntimeError(f"programbench: workspace tar failed: "
                               f"{result.stderr.decode(errors='replace')[-2000:]}")
        return result.stdout, []

    def restore_workspace(self, tar_bytes: bytes) -> None:
        ws = shlex.quote(self.workspace)
        subprocess.run([self.executable, "exec", "-i", "-u", "root", str(self.container_id),
                        "bash", "-c", f"find {ws} -mindepth 1 -delete && tar -xzf - -C {ws}"],
                       input=tar_bytes, capture_output=True, timeout=3600, check=True)

    def stop(self) -> None:
        if self._finalizer is not None:
            self._finalizer()
        self.started = False


# ---- the world ---------------------------------------------------------------------------------

_CONFIG = ("task", "backend", "workspace", "user", "step_limit", "wall_limit_s",
           "command_timeout_s", "lockdown", "pin_hosts", "max_output_bytes", "cpus", "memory_gb",
           "snapshot_exclude", "check_task")


def _positive(name: str, value: Any, allow_none: bool = False) -> None:
    if value is None and allow_none:
        return
    if isinstance(value, bool) or not isinstance(value, (int, float)) or value <= 0:
        raise ValueError(f"{name} must be a positive number{' or None' if allow_none else ''}, "
                         f"got {value!r}")


class ProgramBench(World):
    """ProgramBench: rebuild a program from its reference executable and docs with one bash tool, official semantics."""

    entry_point: ClassVar[str | None] = "programbench"
    name = "programbench"
    requires: ClassVar[Requirements] = Requirements(commit="immediate")
    _skip_in_snapshot: ClassVar[tuple[str, ...]] = ("params", *_CONFIG)

    def __init__(self, task: str, backend: str = "local", workspace: str = "/workspace",
                 user: str | None = "agent", step_limit: int = 1000, wall_limit_s: float = 21600,
                 command_timeout_s: float = 180, lockdown: bool = True,
                 pin_hosts: Sequence[str] = ("api.anthropic.com", "huggingface.co", "hf.co"),
                 max_output_bytes: int | None = 1_000_000, cpus: float | None = None,
                 memory_gb: float | None = None, snapshot_exclude: Sequence[str] = (),
                 check_task: bool = True) -> None:
        self.task = check_task_id(task)
        if backend not in ("local", "docker"):
            raise ValueError(f"backend must be 'local' or 'docker', got {backend!r}")
        if not isinstance(workspace, str) or not workspace:
            raise ValueError(f"workspace must be a path string, got {workspace!r}")
        if user is not None and not isinstance(user, str):
            raise ValueError(f"user must be a user name or None, got {user!r}")
        if isinstance(step_limit, bool) or not isinstance(step_limit, int) or step_limit < 1:
            raise ValueError(f"step_limit must be a positive integer, got {step_limit!r}")
        _positive("wall_limit_s", wall_limit_s)
        _positive("command_timeout_s", command_timeout_s)
        _positive("cpus", cpus, allow_none=True)
        _positive("memory_gb", memory_gb, allow_none=True)
        if max_output_bytes is not None and (isinstance(max_output_bytes, bool) or not isinstance(
                max_output_bytes, int) or max_output_bytes < 2):
            raise ValueError(f"max_output_bytes must be an integer >= 2 or None, got {max_output_bytes!r}")
        for key, seq in (("pin_hosts", pin_hosts), ("snapshot_exclude", snapshot_exclude)):
            if isinstance(seq, str) or not all(isinstance(s, str) for s in seq):
                raise ValueError(f"{key} must be a list of strings, got {seq!r}")
        self.backend = backend
        self.workspace = workspace
        self.user = user
        self.step_limit = step_limit
        self.wall_limit_s = wall_limit_s
        self.command_timeout_s = command_timeout_s
        self.lockdown = bool(lockdown)
        self.pin_hosts = list(pin_hosts)
        self.max_output_bytes = max_output_bytes
        self.cpus = cpus
        self.memory_gb = memory_gb
        self.snapshot_exclude = list(snapshot_exclude)
        self.check_task = bool(check_task)
        # ---- state (snapshotted as the meta dict) ----
        self.task_info: dict = {}               # from task.yaml at check_experiment (check_task)
        self._zero_state()
        # ---- transient (never snapshotted, never copied: see __getstate__) ----
        self._runtime: Any = None
        self._pending_workspace: bytes | None = None
        self._max_rounds: int | None = None
        self._clock_start: float | None = None  # monotonic time the clock (re)started
        self._clock_base = 0.0                  # elapsed_s when it (re)started

    def _zero_state(self) -> None:
        self.agents: list[str] = []
        self.round = 0
        self.steps: dict[str, int] = {}
        self.steps_total = 0
        self.round_steps = 0
        self.timeouts = 0
        self.submitted = False
        self.submission = ""
        self.submitted_by: str | None = None
        self.submitted_round: int | None = None
        self.elapsed_s = 0.0
        self.history: list[dict] = []           # the last HISTORY_KEEP commands
        self.workspace_entries: list[dict] = []
        self.snapshot_skipped: list[str] = []
        self.resources: dict = {"cpus": None, "memory_gb": None, "source": None}
        self.uname: dict = {}
        self.lockdown_record: dict = {}

    @property
    def image(self) -> str:
        return image_name(self.task)

    def __getstate__(self) -> dict:
        # deep copies and pickles never carry the live backend: a copy must not own a container
        state = dict(self.__dict__)
        state["_runtime"] = None
        state["_clock_start"] = None
        return state

    # ---- lifecycle -----------------------------------------------------------------------------
    def _lockdown_applies(self) -> bool:
        return self.lockdown and self.backend == "local" and os.geteuid() == 0

    def _detect_resources(self) -> dict:
        cpus, mem, source = self.cpus, self.memory_gb, "params"
        if self.backend == "docker":
            cpus, mem = cpus or DOCKER_CPUS, mem or DOCKER_MEMORY_GB
            source = "docker"
        elif self._lockdown_applies() and (cpus is None or mem is None):
            d_cpus, d_mem = detect_resources()
            cpus, mem, source = cpus or d_cpus, mem or d_mem, "cgroup"
        return {"cpus": cpus, "memory_gb": mem, "source": source if cpus or mem else None}

    def reset(self, rng: random.Random, agents: list[AgentId]) -> None:
        self._zero_state()
        self.agents = [str(a) for a in agents]
        self.steps = dict.fromkeys(self.agents, 0)
        u = platform.uname()
        self.uname = {"system": u.system, "release": u.release, "version": u.version,
                      "machine": u.machine}
        self.resources = self._detect_resources()

    def check_experiment(self, experiment: Any) -> None:
        """Requirements, then: in the one-round solo shape `max_calls_per_turn >= step_limit`;
        records `max_rounds` (docker cleanup); with `check_task`, the task must be known."""
        super().check_experiment(experiment)
        options = dict(getattr(experiment, "options", None) or {})
        self._max_rounds = options.get("max_rounds")
        cap = options.get("max_calls_per_turn", 20)
        if self._max_rounds == 1 and cap < self.step_limit:
            raise ValueError(
                f"programbench: with max_rounds 1 (the official solo shape, where one turn is the "
                f"whole episode) options.max_calls_per_turn must be at least step_limit "
                f"({self.step_limit}), got {cap}: a {cap}-call turn would end the episode before "
                f"the benchmark's own {self.step_limit}-step limit. Set max_calls_per_turn: "
                f"{self.step_limit} (or lower step_limit for a dry run).")
        if self.check_task:
            meta = task_metadata(self.task)
            self.task_info = {k: meta.get(k) for k in ("language", "difficulty") if k in meta}

    def _effective_cpus(self) -> float | None:
        return self.resources.get("cpus") if self.resources else self.cpus

    def _make_runtime(self) -> Backend:
        if self.backend == "docker":
            res = self.resources or {}
            return DockerBackend(self.image, self.workspace, self.user,
                                 res.get("cpus") or self.cpus or DOCKER_CPUS,
                                 res.get("memory_gb") or self.memory_gb or DOCKER_MEMORY_GB,
                                 self.max_output_bytes)
        lockdown = None
        if self._lockdown_applies():
            user, hosts = self.user, list(self.pin_hosts)

            def lockdown() -> dict:
                return apply_lockdown(user, hosts)
        return LocalBackend(self.workspace, self.user, thread_env(self._effective_cpus()),
                            self.max_output_bytes, lockdown)

    def _ensure_started(self) -> Backend:
        """The running backend, started now if needed (lockdown, docker run, git identity)."""
        if self._runtime is None:
            self._runtime = self._make_runtime()
        if not self._runtime.started:
            t0 = time.monotonic()
            self.lockdown_record = self._runtime.start()
            self._runtime.execute(GIT_SETUP, self.command_timeout_s)
            if self._clock_start is not None:  # backend start time is not on the clock
                self._clock_start += time.monotonic() - t0
        return self._runtime

    def _tick(self) -> None:
        if self._clock_start is not None:
            self.elapsed_s = round(self._clock_base + time.monotonic() - self._clock_start, 3)

    def _live_elapsed(self) -> float:
        if self._clock_start is None:
            return self.elapsed_s
        return self._clock_base + time.monotonic() - self._clock_start

    def begin_round(self, round: int) -> None:
        self.round = round
        self.round_steps = 0
        if self._pending_workspace is not None:  # resume / fork: materialise the restored tar
            self._ensure_started().restore_workspace(self._pending_workspace)
            self._pending_workspace = None
        if self._clock_start is None:
            self._clock_base, self._clock_start = self.elapsed_s, time.monotonic()

    def end_round(self, round: int) -> dict:
        """Update the clock; on the last round (terminal, or `max_rounds`), capture the docker
        workspace for the snapshot and remove the container."""
        self._tick()
        last = self.terminal() or (self._max_rounds is not None and round >= self._max_rounds)
        if last and self.backend == "docker" and self._runtime is not None and self._runtime.started:
            try:
                tar, self.snapshot_skipped = self._runtime.tar(self.snapshot_exclude)
                self._pending_workspace = tar
            finally:
                self._runtime.stop()
                self._runtime = None
        return {"round": round, "steps": self.steps_total, "round_steps": self.round_steps,
                "submitted": self.submitted, "elapsed_s": self.elapsed_s}

    def terminal(self) -> bool:
        return self.submitted or self.elapsed_s >= self.wall_limit_s

    # ---- snapshot / restore ------------------------------------------------------------------------
    def _meta(self) -> dict:
        skip = set(self._skip_in_snapshot)
        return {k: v for k, v in self.__dict__.items()
                if not k.startswith("_") and k not in skip}

    def snapshot(self) -> bytes:
        """`{"meta": <state>, "workspace_tar": <deterministic .tar.gz | None>}` pickled."""
        if self._pending_workspace is not None:
            tar = self._pending_workspace
        else:
            if self._runtime is None and self.backend == "local":
                self._runtime = self._make_runtime()
            tar, self.snapshot_skipped = (self._runtime.tar(self.snapshot_exclude)
                                          if self._runtime is not None else (None, []))
        self.workspace_entries = workspace_entries(tar)
        return pickle.dumps({"meta": self._meta(), "workspace_tar": tar},
                            protocol=pickle.HIGHEST_PROTOCOL)

    def restore(self, blob: bytes) -> None:
        """Meta into attributes, the tar into `_pending_workspace`; no container, no file."""
        data = pickle.loads(blob)
        self.__dict__.update(data["meta"])
        self._pending_workspace = data.get("workspace_tar")

    # ---- agent-facing ----------------------------------------------------------------------------
    def description(self) -> str:
        system, instance = prompt_parts(self.uname or None)
        res = self.resources or {}
        sentence = resource_sentence(res.get("cpus"), res.get("memory_gb"))
        return f"{system}\n\n{instance}" + (f"\n\n{sentence}" if sentence else "")

    def observe(self, agent: AgentId) -> Observation:
        n = self.steps.get(str(agent), 0)
        return text_observation(
            f"Workspace: {self.workspace} (each bash command runs in a fresh shell there).\n"
            f"Bash commands you have run: {n}.\n"
            f"Submitted: {'yes' if self.submitted else 'no'}.")

    def validate(self, agent: AgentId, action: Action, pending: Sequence[Action] = ()) -> Ack:
        ack = super().validate(agent, action)
        if not ack.ok:
            return ack
        if not isinstance(action.args.get("command"), str):
            return Ack(ok=False, error="bad args: command must be a string")
        if self.submitted:
            return Ack(ok=False, error="submitted")
        if self._live_elapsed() >= self.wall_limit_s:
            return Ack(ok=False, error="wall_limit")
        return ack

    @tool("bash", "Execute a bash command",
          {"command": {"type": "string", "description": "The bash command to execute"}})
    def bash(self, agent: AgentId, command: str) -> Outcome:
        if not isinstance(command, str):
            return Outcome(accepted=False, feedback={"error": "bad args: command must be a string"})
        if self.submitted:
            return Outcome(accepted=False, feedback={"error": "submitted"})
        out = self._ensure_started().execute(command, self.command_timeout_s)
        self._tick()
        a = str(agent)
        self.steps[a] = self.steps.get(a, 0) + 1
        self.steps_total += 1
        self.round_steps += 1
        if out["returncode"] == -1 and out["exception_info"]:
            self.timeouts += 1
        self.history = [*self.history, {"round": self.round, "agent": a, "command": command[:200],
                                        "returncode": out["returncode"]}][-HISTORY_KEEP:]
        sub = submission_of(out["output"], out["returncode"])
        if sub is not None:
            self.submitted, self.submission = True, sub
            self.submitted_by, self.submitted_round = a, self.round
        return Outcome(accepted=True, feedback=out)

    # ---- evaluator-only --------------------------------------------------------------------------
    def score(self) -> dict:
        """Cheap, every round. Grading reads the last snapshot's `workspace_tar` later."""
        return {"submitted": self.submitted, "steps": self.steps_total,
                "elapsed_s": self.elapsed_s, "graded": False}

    def verify(self) -> dict:
        return {"task": self.task, "image": self.image, **self.task_info}

    def render_state(self) -> dict | None:
        """Task, steps, submission, top-level workspace entries and the last commands."""
        return {"task": self.task, "backend": self.backend, "steps": self.steps_total,
                "submitted": self.submitted, "submitted_by": self.submitted_by,
                "elapsed_s": self.elapsed_s, "timeouts": self.timeouts,
                "workspace": [dict(e) for e in self.workspace_entries[:200]],
                "last commands": [dict(h) for h in self.history]}


# ---- fake LLM script (`fake:programbench_echo`) ------------------------------------------------

FAKE_COMPILE = """cat > compile.sh <<'EOF'
#!/bin/bash
printf '#!/bin/sh\\necho hello\\n' > executable
chmod +x executable
EOF
chmod +x compile.sh && ./compile.sh && ls"""
FAKE_COMMANDS = ("ls", FAKE_COMPILE, f"echo {SUBMIT_MARKER}")


def fake_echo(request: Any, rng: random.Random) -> Any:
    """Fake script: run `ls`, write a trivial compile.sh that builds ./executable, then submit."""
    import json

    from ..providers.fake import _assistant_calls, _response

    tools = {t.name for t in request.tools}
    if not tools:
        return _response(request, rng, [], json.dumps({"answer": "none"}))
    done = sum(1 for m in request.messages for name, _ in _assistant_calls(m) if name == "bash")
    if done < len(FAKE_COMMANDS) and "bash" in tools:
        return _response(request, rng, [("bash", {"command": FAKE_COMMANDS[done]})])
    return _response(request, rng, [("end_turn", {})] if "end_turn" in tools else [])
