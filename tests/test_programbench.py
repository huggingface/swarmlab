"""ProgramBench world (temp-dir local backend, mocked docker), its lockdown seam and the fake
script `programbench_echo`. Nothing here needs Docker, the task images, the network or the
`programbench` package (the tests that read the package skip without it)."""
from __future__ import annotations

import asyncio
import copy
import io
import json
import os
import random
import socket
import stat
import subprocess
import tarfile
import time
from pathlib import Path
from types import SimpleNamespace

import pytest

from swarmlab import Experiment, Run
from swarmlab.participants import LLMAgent
from swarmlab.participants.base import Participant, TurnUsage
from swarmlab.providers.fake import resolve_script
from swarmlab.registry import build, resolve
from swarmlab.snapshot import SnapshotStore
from swarmlab.viewer.build import build as build_view
from swarmlab.viewer.build import collect
from swarmlab.world import programbench as pb
from swarmlab.world.base import Action
from swarmlab.world.programbench import (
    BASELINE_ENV,
    SUBMIT_MARKER,
    LockdownPaths,
    ProgramBench,
    apply_lockdown,
    blackhole_dns,
    command_env,
    detect_resources,
    fake_echo,
    image_name,
    is_secret,
    pin_hosts_file,
    prompt_parts,
    remove_sudo,
    resource_sentence,
    submission_of,
    workspace_entries,
    workspace_tar,
    workspace_tar_of,
)

TASK = "abishekvashok__cmatrix.5c082c6"
UNAME = {"system": "Linux", "release": "6.1.0", "version": "#1 SMP", "machine": "x86_64"}


def make(tmp_path: Path, name: str = "ws", agents=("a000",), **kw) -> ProgramBench:
    """A local-backend world on a temp workspace, reset and in round 1."""
    params = {"workspace": str(tmp_path / name), "user": None, "lockdown": False,
              "check_task": False, **kw}
    w = ProgramBench(TASK, **params)
    w.reset(random.Random(0), list(agents))
    w.begin_round(1)
    return w


def bash(w: ProgramBench, command: str, agent: str = "a000"):
    """The world's `bash` tool (an `async def`) run to completion."""
    return asyncio.run(w.bash(agent, command))


def sh(w: ProgramBench, command: str, agent: str = "a000") -> dict:
    out = bash(w, command, agent)
    assert out.accepted
    return out.feedback


def options(**kw) -> SimpleNamespace:
    """A stand-in for the runtime experiment `check_experiment` receives."""
    return SimpleNamespace(options={"commit": "immediate", "max_rounds": 1,
                                    "max_calls_per_turn": 20, **kw},
                           participants=[object()], medium=None)


# ---- the bash tool ---------------------------------------------------------------------------

def test_bash_feedback_is_the_baseline_dict_with_merged_output(tmp_path):
    w = make(tmp_path)
    fb = sh(w, "echo out; echo err >&2; exit 3")
    assert fb == {"output": "out\nerr\n", "returncode": 3, "exception_info": ""}
    assert w.tool_schemas()[0].model_dump()["parameters"] == {
        "type": "object", "required": ["command"], "additionalProperties": False,
        "properties": {"command": {"type": "string", "description": "The bash command to execute"}}}
    assert w.tool_schemas()[0].description == "Execute a bash command"


def test_every_call_is_a_fresh_login_shell_in_the_workspace(tmp_path):
    w = make(tmp_path)
    assert sh(w, "pwd")["output"] == f"{tmp_path / 'ws'}\n"
    sh(w, "cd /; export SWARMLAB_PB_PROBE=1")
    assert sh(w, "pwd; echo ${SWARMLAB_PB_PROBE:-unset}")["output"] == f"{tmp_path / 'ws'}\nunset\n"
    assert sh(w, "shopt -q login_shell && echo login")["output"] == "login\n"


def test_baseline_env_present_and_secrets_filtered(tmp_path, monkeypatch):
    for k in ("ANTHROPIC_API_KEY", "HF_TOKEN", "MY_SERVICE_TOKEN", "DB_PASSWORD", "FOO_SECRET",
              "OPENAI_BASE_URL"):
        monkeypatch.setenv(k, "leak")
    monkeypatch.setenv("SWARMLAB_PB_PLAIN", "kept")
    for k in pb.THREAD_VARS:
        monkeypatch.delenv(k, raising=False)
    w = make(tmp_path)
    env = dict(line.split("=", 1) for line in sh(w, "env")["output"].splitlines() if "=" in line)
    assert {k: env.get(k) for k in BASELINE_ENV} == BASELINE_ENV
    assert env["SWARMLAB_PB_PLAIN"] == "kept"
    assert "leak" not in env.values()
    assert not any(k in env for k in pb.THREAD_VARS)   # CPU count unknown: no thread variables


def test_command_env_unit():
    base = {"PATH": "/venv/bin:/usr/bin", "VIRTUAL_ENV": "/venv", "ANTHROPIC_API_KEY": "x",
            "AWS_SECRET": "x", "HOME": "/root", "LESS": "-X"}
    env = command_env(base, {"HOME": "/home/agent", "MAKEFLAGS": "-j8"}, venv_bin="/venv/bin")
    assert env == {"PATH": "/usr/bin", "HOME": "/home/agent", "MAKEFLAGS": "-j8", **BASELINE_ENV}
    # a secret whose name looks harmless, declared by the job launcher, is dropped too
    declared = command_env({"MODEL_AUTH": "dummy-value", "SWARMLAB_SECRET_NAMES": "HF_TOKEN MODEL_AUTH",
                            "LANG": "C"})
    assert "MODEL_AUTH" not in declared and declared["LANG"] == "C"
    assert all(is_secret(k) for k in ("HF_TOKEN", "ANTHROPIC_BASE_URL", "GH_TOKEN", "X_KEY"))
    assert not any(is_secret(k) for k in ("PATH", "HOME", "KEYBOARD", "TOKENIZERS_PARALLELISM"))


def test_thread_vars_and_resource_sentence_follow_the_cpu_count(tmp_path):
    w = make(tmp_path, cpus=8, memory_gb=32)
    out = sh(w, "echo $MAKEFLAGS $CARGO_BUILD_JOBS $OMP_NUM_THREADS $GOMAXPROCS")["output"]
    assert out == "-j8 8 8 8\n"
    assert w.description().endswith("\n\nThis machine has 8 CPUs and 32 GB of memory.")
    assert resource_sentence(1, None) == "This machine has 1 CPU."
    assert resource_sentence(None, 15.5) == "This machine has 15.5 GB of memory."
    assert resource_sentence(None, None) == ""
    assert not make(tmp_path, "ws2").description().rstrip().endswith("memory.")


def test_timeout_returns_partial_output_and_leaves_the_process_running(tmp_path):
    w = make(tmp_path, command_timeout_s=0.5)
    cmd = "echo partial; sleep 1.5; touch survived"
    t0 = time.monotonic()
    fb = sh(w, cmd)
    assert time.monotonic() - t0 < 1.4
    assert fb == {"output": "partial\n", "returncode": -1,
                  "exception_info": "An error occurred while executing the command: "
                                    f"Command '{cmd}' timed out after 0.5 seconds"}
    assert not (tmp_path / "ws" / "survived").exists()
    deadline = time.monotonic() + 10
    while not (tmp_path / "ws" / "survived").exists() and time.monotonic() < deadline:
        time.sleep(0.1)
    assert (tmp_path / "ws" / "survived").exists()   # only the wait stopped, as under Docker
    assert w.timeouts == 1 and w.steps_total == 1


def test_output_cap_keeps_head_and_tail(tmp_path):
    w = make(tmp_path, max_output_bytes=100)
    out = sh(w, "printf 'H%.0s' $(seq 60); printf 'x%.0s' $(seq 1000); printf 'T%.0s' $(seq 60)")
    head, rest = out["output"].split("\n[swarmlab: ", 1)
    assert head == "H" * 50 and rest.endswith("]\n" + "T" * 50)
    assert "1020 bytes of output elided (max_output_bytes=100)" in rest
    w = make(tmp_path, "ws2", max_output_bytes=None)
    assert len(sh(w, "printf 'x%.0s' $(seq 3000)")["output"]) == 3000


def test_carriage_returns_are_normalised_like_text_mode(tmp_path):
    assert sh(make(tmp_path), r"printf 'a\r\nb\rc\n'")["output"] == "a\nb\nc\n"


# ---- submission ------------------------------------------------------------------------------

def test_submission_rule():
    assert submission_of(f"\n  \n{SUBMIT_MARKER}\nnotes\nmore\n", 0) == "notes\nmore\n"
    assert submission_of(f"  {SUBMIT_MARKER}  \n", 0) == ""
    assert submission_of(f"{SUBMIT_MARKER}\n", 1) is None
    assert submission_of(f"ok\n{SUBMIT_MARKER}\n", 0) is None
    assert submission_of("", 0) is None


def test_submission_ends_the_episode_and_refuses_later_commands(tmp_path):
    w = make(tmp_path, agents=("a000", "a001"))
    sh(w, f"echo {SUBMIT_MARKER}; exit 1")                     # non-zero: not a submission
    assert not w.submitted and not w.terminal()
    sh(w, f"echo; echo {SUBMIT_MARKER}; echo handoff", agent="a001")
    assert w.submitted and w.terminal()
    assert (w.submission, w.submitted_by, w.submitted_round) == ("handoff\n", "a001", 1)
    action = Action(name="bash", args={"command": "ls"})
    assert w.validate("a000", action).model_dump() == {"ok": False, "error": "submitted"}
    assert bash(w, "ls").accepted is False
    assert w.score() == {"submitted": True, "steps": 2, "elapsed_s": w.elapsed_s, "graded": False}
    assert w.end_round(1) == {"round": 1, "steps": 2, "round_steps": 2, "submitted": True,
                              "elapsed_s": w.elapsed_s}


def test_validate_checks_args_and_the_wall_limit(tmp_path):
    w = make(tmp_path, wall_limit_s=0.05)
    assert w.validate("a000", Action(name="bash", args={"command": 3})).error == \
        "bad args: command must be a string"
    assert w.validate("a000", Action(name="bash", args={})).ok is False
    assert w.validate("a000", Action(name="bash", args={"command": "ls"})).ok
    time.sleep(0.1)
    assert w.validate("a000", Action(name="bash", args={"command": "ls"})).error == "wall_limit"
    assert not w.terminal()          # terminal() reads the stored clock, updated by end_round
    w.end_round(1)
    assert w.terminal() and w.elapsed_s >= 0.05


def test_observe_is_short_and_private_is_empty(tmp_path):
    w = make(tmp_path, agents=("a000", "a001"))
    sh(w, "true")
    obs = w.observe("a001")
    assert obs.private == {}
    assert obs.parts[0].text == (f"Workspace: {tmp_path / 'ws'} (each bash command runs in a fresh "
                                 "shell there).\nBash commands you have run: 0.\nSubmitted: no.")
    assert "Bash commands you have run: 1." in w.observe("a000").parts[0].text


def test_git_identity_is_set_only_in_a_repo_root(tmp_path):
    ws = tmp_path / "ws"
    ws.mkdir()
    subprocess.run(["git", "init", "-q", str(ws)], check=True)
    w = make(tmp_path)
    assert sh(w, "git config user.name; git config user.email")["output"] == \
        "mini-swe-agent\nmini-swe-agent@proton.me\n"
    assert w.steps_total == 1 and len(w.history) == 1   # the setup is not a step
    assert not (tmp_path / "plain" / ".git").exists()
    sh(make(tmp_path, "plain"), "true")
    assert not (tmp_path / "plain" / ".git").exists()


# ---- task text -------------------------------------------------------------------------------

def test_description_is_the_baseline_text_with_uname(tmp_path):
    system, instance = prompt_parts(UNAME)
    assert system.startswith("You are a helpful assistant that can interact with a computer.\n")
    assert system.endswith("</RULES>")
    assert instance.startswith("## Task context\n")
    assert "<system_information>\nLinux 6.1.0 #1 SMP x86_64\n</system_information>" in instance
    assert instance.endswith("```bash\nanything\n```")
    assert "{{" not in system + instance
    w = make(tmp_path)
    s, i = prompt_parts(w.uname)
    assert w.description() == f"{s}\n\n{i}"


def test_templates_are_the_installed_mini_swe_agent_config(monkeypatch):
    # A cross-check against the real package, which swarmlab does not install (the participant
    # uses a vendored copy): it runs when a developer has mini-swe-agent 2.4.6 installed.
    monkeypatch.setenv("MSWEA_SILENT_STARTUP", "1")
    minisweagent = pytest.importorskip("minisweagent")
    import yaml

    path = Path(minisweagent.__file__).parent / "config" / "benchmarks" / "programbench.yaml"
    cfg = yaml.safe_load(path.read_text())["agent"]
    assert (cfg["system_template"], cfg["instance_template"]) == \
        (pb.SYSTEM_TEMPLATE, pb.INSTANCE_TEMPLATE)


# ---- identity, spec, requirements ------------------------------------------------------------

def test_image_name_and_task_id():
    assert image_name(TASK) == "programbench/abishekvashok_1776_cmatrix.5c082c6:task_cleanroom_v6"
    assert ProgramBench(TASK).image == image_name(TASK)
    for bad in ("cmatrix", "a/b__c.1", "owner__repo", 7, "a__b.c d"):
        with pytest.raises(ValueError, match="instance id"):
            ProgramBench(bad)


def test_image_name_matches_the_package_rule():
    constants = pytest.importorskip("programbench.constants")
    assert image_name(TASK).split(":")[0] == constants.image_name_from_instance_id(TASK)


@pytest.mark.parametrize("kw", [{"backend": "podman"}, {"step_limit": 0}, {"wall_limit_s": 0},
                                {"command_timeout_s": -1}, {"cpus": 0}, {"memory_gb": True},
                                {"max_output_bytes": 1}, {"pin_hosts": "hf.co"},
                                {"snapshot_exclude": [3]}, {"workspace": ""}, {"user": 5}])
def test_constructor_rejects_bad_params(kw):
    with pytest.raises(ValueError):
        ProgramBench(TASK, **kw)


def test_spec_round_trip_includes_backend():
    w = ProgramBench(TASK, backend="docker", pin_hosts=("api.anthropic.com",),
                     snapshot_exclude=("target",))
    spec = w.spec()
    assert spec["type"] == "programbench"
    assert spec["params"] == {
        "task": TASK, "backend": "docker", "workspace": "/workspace", "user": "agent",
        "step_limit": 1000, "wall_limit_s": 21600, "command_timeout_s": 180, "lockdown": True,
        "pin_hosts": ["api.anthropic.com"], "max_output_bytes": 1_000_000, "cpus": None,
        "memory_gb": None, "snapshot_exclude": ["target"], "check_task": True}
    rebuilt = build(json.loads(json.dumps(spec)), "swarmlab.worlds")
    assert isinstance(rebuilt, ProgramBench) and rebuilt.spec() == spec
    assert ProgramBench(TASK).spec()["params"]["backend"] == "local"


def test_entry_points_resolve():
    assert resolve("programbench", "swarmlab.worlds") is ProgramBench
    assert resolve_script("programbench_echo") is fake_echo


def test_requires_immediate_or_async_commit():
    with pytest.raises(ValueError, match=r"requires commit in \[immediate, async\], got round_end"):
        Experiment(name="pb", world=ProgramBench(TASK, check_task=False),
                   participants=[LLMAgent(model="fake:programbench_echo")])


def test_check_experiment_guards_the_solo_shape():
    w = ProgramBench(TASK, check_task=False)
    with pytest.raises(ValueError, match="max_calls_per_turn must be at least step_limit"):
        w.check_experiment(options())
    w.check_experiment(options(max_calls_per_turn=1000))
    w.check_experiment(options(max_rounds=50))           # rounds of k calls: the participant's limit
    assert w._max_rounds == 50
    ProgramBench(TASK, step_limit=5, check_task=False).check_experiment(options(max_calls_per_turn=5))
    with pytest.raises(ValueError, match=r"requires commit in \[immediate, async\]"):
        w.check_experiment(options(commit="round_end", max_calls_per_turn=1000))


def test_check_task_names_the_extra_when_programbench_is_missing(monkeypatch):
    monkeypatch.setitem(__import__("sys").modules, "programbench", None)
    monkeypatch.setitem(__import__("sys").modules, "programbench.constants", None)
    with pytest.raises(ImportError, match=r"swarmlab\[programbench\]"):
        ProgramBench(TASK).check_experiment(options(max_calls_per_turn=1000))
    ProgramBench(TASK, check_task=False).check_experiment(options(max_calls_per_turn=1000))


def test_check_task_reads_the_package_metadata():
    pytest.importorskip("programbench")
    w = ProgramBench(TASK)
    w.check_experiment(options(max_calls_per_turn=1000))
    assert w.task_info == {"language": "c", "difficulty": "easy"}
    with pytest.raises(ValueError, match="unknown ProgramBench task"):
        ProgramBench("nobody__nothing.0000000").check_experiment(options(max_calls_per_turn=1000))


# ---- snapshots ---------------------------------------------------------------------------------

def test_snapshot_restore_round_trip(tmp_path):
    w = make(tmp_path)
    sh(w, "mkdir -p src/deep && echo 'fn main() {}' > src/deep/main.rs && ln -s src link")
    sh(w, "printf '#!/bin/sh\\necho hi\\n' > compile.sh && chmod 755 compile.sh")
    (tmp_path / "ws" / "direct.txt").write_text("written directly\n")
    (tmp_path / "ws" / "direct.txt").chmod(0o664)                 # group write survives
    os.utime(tmp_path / "ws" / "link", (1_000_000, 1_000_000), follow_symlinks=False)
    w.end_round(1)
    blob = w.snapshot()
    fresh = ProgramBench(TASK, workspace=str(tmp_path / "ws2"), user=None, lockdown=False,
                         check_task=False)
    fresh.reset(random.Random(0), ["a000"])
    fresh.restore(blob)
    assert not (tmp_path / "ws2").exists()                     # restore touches no file
    assert fresh.score() == w.score() and fresh.history == w.history
    fresh.begin_round(2)
    ws2 = tmp_path / "ws2"
    assert (ws2 / "src" / "deep" / "main.rs").read_text() == "fn main() {}\n"
    assert (ws2 / "direct.txt").read_text() == "written directly\n"
    assert os.readlink(ws2 / "link") == "src"
    assert stat.S_IMODE((ws2 / "compile.sh").stat().st_mode) == 0o755
    assert stat.S_IMODE((ws2 / "direct.txt").stat().st_mode) == 0o664
    assert sh(fresh, "./compile.sh")["output"] == "hi\n"
    assert fresh.steps_total == 3
    assert workspace_tar_of(fresh.snapshot()) == workspace_tar_of(blob)   # same tree, same bytes


def test_restore_replaces_the_workspace_contents(tmp_path):
    w = make(tmp_path)
    sh(w, "echo keep > keep.txt")
    blob = w.snapshot()
    sh(w, "echo junk > junk.txt && rm keep.txt")
    w.restore(blob)
    w.begin_round(2)
    assert sorted(os.listdir(tmp_path / "ws")) == ["keep.txt"]


def test_tar_is_deterministic_and_honours_excludes(tmp_path):
    root = tmp_path / "ws"
    (root / "target" / "debug").mkdir(parents=True)
    (root / "target" / "debug" / "big.o").write_bytes(b"x" * 1000)
    (root / "src").mkdir()
    (root / "src" / "a.o").write_bytes(b"o")
    (root / "src" / "a.c").write_text("int main;\n")
    first, skipped = workspace_tar(root)
    assert workspace_tar(root)[0] == first and skipped == []
    with tarfile.open(fileobj=io.BytesIO(first), mode="r:gz") as tar:
        assert tar.getnames() == ["src", "src/a.c", "src/a.o", "target", "target/debug",
                                  "target/debug/big.o"]
    assert first[4:8] == b"\x00\x00\x00\x00"                     # gzip header mtime 0
    excl, _ = workspace_tar(root, ["target", "*.o"])
    with tarfile.open(fileobj=io.BytesIO(excl), mode="r:gz") as tar:
        assert tar.getnames() == ["src", "src/a.c"]
    assert workspace_entries(first) == [{"name": "src", "type": "dir", "bytes": 11},
                                        {"name": "target", "type": "dir", "bytes": 1000}]
    w = make(tmp_path, "ws", snapshot_exclude=["target"])
    assert [e["name"] for e in workspace_entries(workspace_tar_of(w.snapshot()))] == ["src"]


@pytest.mark.skipif(os.geteuid() == 0, reason="root reads every file")
def test_unreadable_files_are_skipped_and_listed(tmp_path):
    w = make(tmp_path)
    sh(w, "echo secret > locked && chmod 000 locked && echo ok > open")
    w.snapshot()
    assert w.snapshot_skipped == ["locked"]
    assert [e["name"] for e in w.workspace_entries] == ["open"]


def test_render_state_from_a_restored_world_needs_no_backend(tmp_path):
    w = make(tmp_path)
    sh(w, "mkdir src && head -c 300 /dev/zero > src/blob && echo hi > README")
    sh(w, "false")
    blob = w.snapshot()
    viewer = ProgramBench(TASK, backend="docker", workspace=str(tmp_path / "nowhere"))
    viewer.restore(blob)
    st = viewer.render_state()
    assert not (tmp_path / "nowhere").exists() and viewer._runtime is None
    assert st["workspace"] == [{"name": "README", "type": "file", "bytes": 3},
                               {"name": "src", "type": "dir", "bytes": 300}]
    assert st["last commands"][-1] == {"round": 1, "agent": "a000", "command": "false",
                                       "returncode": 1}
    assert (st["steps"], st["submitted"], st["task"]) == (2, False, TASK)
    json.dumps(st)


def test_copies_never_carry_the_backend(tmp_path):
    w = make(tmp_path)
    sh(w, "true")
    assert w._runtime is not None
    assert copy.deepcopy(w)._runtime is None
    assert copy.deepcopy(w).steps_total == 1


# ---- lockdown and resources (the seam, on temp files) -------------------------------------------

def _resolver(table):
    def resolve(host, port, type=0):
        if host not in table:
            raise socket.gaierror(f"no {host}")
        return [(fam, socket.SOCK_STREAM, 6, "", (ip, port)) for fam, ip in table[host]]
    return resolve


def test_pin_hosts_prefers_ipv4_and_appends_in_place(tmp_path):
    hosts = tmp_path / "hosts"
    hosts.write_text("127.0.0.1\tlocalhost\n")
    resolve = _resolver({"api.anthropic.com": [(socket.AF_INET6, "2600::1"),
                                               (socket.AF_INET, "160.79.104.10")],
                         "hf.co": [(socket.AF_INET6, "2600::2")]})
    inode = hosts.stat().st_ino
    assert pin_hosts_file(["api.anthropic.com", "hf.co"], str(hosts), resolve) == \
        {"api.anthropic.com": "160.79.104.10", "hf.co": "2600::2"}
    text = hosts.read_text()
    assert text.startswith("127.0.0.1\tlocalhost\n") and hosts.stat().st_ino == inode
    assert "160.79.104.10\tapi.anthropic.com\n" in text and "2600::2\thf.co\n" in text
    with pytest.raises(RuntimeError, match="cannot resolve 'nope'"):
        pin_hosts_file(["nope"], str(hosts), resolve)


def test_dns_blackhole_and_sudo_removal(tmp_path):
    resolv = tmp_path / "resolv.conf"
    resolv.write_text("nameserver 10.0.0.2\nsearch x\n")
    blackhole_dns(str(resolv))
    assert resolv.read_text() == "nameserver 0.0.0.0\n"
    sudoers = tmp_path / "sudoers"
    sudoers.write_text("root ALL=(ALL) ALL\nagent ALL=(ALL) NOPASSWD: ALL\n%sudo ALL=(ALL) ALL\n")
    d = tmp_path / "sudoers.d"
    d.mkdir()
    (d / "agent").write_text("agent ALL=(ALL) NOPASSWD: /usr/bin/apt-get\n")
    (d / "other").write_text("bob ALL=(ALL) ALL\n# agent is mentioned in a comment only\n")
    sudo = tmp_path / "sudo"
    sudo.write_text("#!/bin/sh\n")
    sudo.chmod(0o4755)
    done = remove_sudo("agent", str(sudoers), str(d), str(sudo))
    assert done == {"removed_files": ["agent"], "sudoers_lines": 1, "sudo_disabled": True}
    assert sudoers.read_text() == "root ALL=(ALL) ALL\n%sudo ALL=(ALL) ALL\n"
    assert sorted(os.listdir(d)) == ["other"]
    assert stat.S_IMODE(sudo.stat().st_mode) == 0


def test_apply_lockdown_runs_every_step(tmp_path):
    for name in ("hosts", "resolv.conf", "sudoers"):
        (tmp_path / name).write_text("")
    (tmp_path / "sudoers.d").mkdir()
    paths = LockdownPaths(hosts=str(tmp_path / "hosts"), resolv_conf=str(tmp_path / "resolv.conf"),
                          sudoers=str(tmp_path / "sudoers"), sudoers_d=str(tmp_path / "sudoers.d"),
                          sudo_bin=str(tmp_path / "missing-sudo"))
    rec = apply_lockdown("agent", ["hf.co"], paths, _resolver({"hf.co": [(socket.AF_INET, "1.2.3.4")]}))
    assert rec == {"applied": True, "pinned": {"hf.co": "1.2.3.4"},
                   "sudo": {"removed_files": [], "sudoers_lines": 0, "sudo_disabled": False}}
    assert (tmp_path / "resolv.conf").read_text() == "nameserver 0.0.0.0\n"


def test_detect_resources_from_cgroup_files(tmp_path):
    cpu, mem, meminfo = (tmp_path / n for n in ("cpu.max", "memory.max", "meminfo"))
    cpu.write_text("800000 100000\n")
    mem.write_text(f"{32 * 2**30}\n")
    assert detect_resources(str(cpu), str(mem), str(meminfo)) == (8.0, 32.0)
    cpu.write_text("max 100000\n")
    mem.write_text("max\n")
    meminfo.write_text("MemTotal:       16777216 kB\nMemFree: 1 kB\n")
    cpus, gb = detect_resources(str(cpu), str(mem), str(meminfo))
    assert cpus == float(len(os.sched_getaffinity(0))) and gb == 16.0
    cpus, gb = detect_resources(str(tmp_path / "none"), str(tmp_path / "none"), str(tmp_path / "none"))
    assert cpus and gb is None


def test_lockdown_and_detection_are_skipped_when_not_root(tmp_path, monkeypatch):
    calls = []
    monkeypatch.setattr(pb, "apply_lockdown", lambda *a, **k: calls.append(a) or {"applied": True})
    monkeypatch.setattr(pb, "detect_resources", lambda *a: (4.0, 16.0))
    monkeypatch.setattr(pb.os, "geteuid", lambda: 1000)
    w = make(tmp_path, lockdown=True)
    sh(w, "true")
    assert calls == [] and w.resources["cpus"] is None and w.backend_info["applied"] is False


def test_lockdown_runs_at_the_first_command_when_root(tmp_path, monkeypatch):
    calls = []
    monkeypatch.setattr(pb, "apply_lockdown", lambda user, hosts: calls.append((user, hosts))
                        or {"applied": True, "pinned": {}})
    monkeypatch.setattr(pb, "detect_resources", lambda *a: (4.0, 16.0))
    monkeypatch.setattr(pb.os, "geteuid", lambda: 0)
    w = make(tmp_path, lockdown=True, memory_gb=12, pin_hosts=["hf.co"])
    assert w.resources == {"cpus": 4.0, "memory_gb": 12, "source": "cgroup"}   # params win
    assert w.description().endswith("This machine has 4 CPUs and 12 GB of memory.")
    assert calls == []                        # reset and begin_round never lock a machine down
    assert sh(w, "echo $MAKEFLAGS")["output"] == "-j4\n"
    assert calls == [(None, ["hf.co"])] and w.backend_info["applied"] is True
    sh(w, "true")
    assert len(calls) == 1
    docker = ProgramBench(TASK, backend="docker", check_task=False)
    docker.reset(random.Random(0), ["a000"])
    assert docker._lockdown_applies() is False


# ---- docker backend (subprocess.run mocked) ------------------------------------------------------

class FakeDocker:
    def __init__(self, tar_bytes: bytes) -> None:
        self.calls: list[list[str]] = []
        self.tar_bytes = tar_bytes

    def __call__(self, cmd, **kw):
        cmd = list(cmd)
        self.calls.append(cmd)
        if cmd[1] == "run":
            return subprocess.CompletedProcess(cmd, 0, stdout="cid123\n", stderr="")
        if cmd[1] == "rm":
            return subprocess.CompletedProcess(cmd, 0, stdout=b"", stderr=b"")
        if "tar --sort=name" in cmd[-1]:
            return subprocess.CompletedProcess(cmd, 0, stdout=self.tar_bytes, stderr=b"")
        if cmd[-1] == "sleep 999":
            raise subprocess.TimeoutExpired(cmd, kw["timeout"], output=b"part")
        return subprocess.CompletedProcess(cmd, 0, stdout=b"out\r\n", stderr=None)

    def popen(self, cmd, **kw):
        """`subprocess.Popen` for the agents' `docker exec` (the backend's `aexecute`)."""
        return FakeExec(self, list(cmd))

    def install(self, monkeypatch, run=None) -> None:
        monkeypatch.setattr(subprocess, "run", run or self)
        monkeypatch.setattr(subprocess, "Popen", self.popen)


class FakeExec:
    """A `docker exec` client: answers as FakeDocker's `run` does, through `communicate`."""

    def __init__(self, fake: FakeDocker, cmd: list[str]) -> None:
        fake.calls.append(cmd)
        self.args, self.pid, self.returncode = cmd, 4242 + len(fake.calls), None

    def communicate(self, timeout=None):
        if self.args[-1] == "sleep 999":
            raise subprocess.TimeoutExpired(self.args, timeout, output=b"part")
        self.returncode = 0
        return b"out\r\n", None

    def poll(self):
        return self.returncode

    def kill(self):
        self.returncode = -9

    def wait(self, timeout=None):
        self.returncode = -9 if self.returncode is None else self.returncode
        return self.returncode


def test_docker_backend_commands(tmp_path, monkeypatch):
    (tmp_path / "src").mkdir()
    (tmp_path / "src" / "main.c").write_text("int main;\n")
    tar_bytes, _ = workspace_tar(tmp_path / "src")
    fake = FakeDocker(tar_bytes)
    fake.install(monkeypatch)
    w = ProgramBench(TASK, backend="docker", check_task=False)
    w.check_experiment(options(max_calls_per_turn=1000))
    w.reset(random.Random(0), ["a000"])
    assert w.description().endswith("This machine has 20 CPUs and 60 GB of memory.")
    w.begin_round(1)
    assert fake.calls == []                                     # lazy: nothing before a command
    assert sh(w, "ls") == {"output": "out\n", "returncode": 0, "exception_info": ""}
    run = fake.calls[0]
    assert run[:4] == ["docker", "run", "-d", "--name"] and run[4].startswith("swarmlab-programbench-")
    assert run[5:] == ["-w", "/workspace", "--rm", "--network", "none", "--cpus", "20",
                       "--memory", "60g", "--memory-swap", "60g", "--user", "agent",
                       "--cap-drop", "SYS_PTRACE",
                       "programbench/abishekvashok_1776_cmatrix.5c082c6:task_cleanroom_v6",
                       "sleep", "7h"]
    env = ["-e", "PAGER=cat", "-e", "MANPAGER=cat", "-e", "LESS=-R", "-e", "PIP_PROGRESS_BAR=off",
           "-e", "TQDM_DISABLE=1"]
    assert fake.calls[1] == ["docker", "exec", "-w", "/workspace", *env, "cid123", "bash", "-lc",
                             pb.GIT_SETUP]
    assert fake.calls[2] == ["docker", "exec", "-w", "/workspace", *env, "cid123", "bash", "-lc",
                             "ls"]
    fb = sh(w, "sleep 999")
    assert (fb["output"], fb["returncode"]) == ("part", -1)
    assert fb["exception_info"].startswith("An error occurred while executing the command: Command '[")
    assert fb["exception_info"].endswith("timed out after 180 seconds")
    assert len([c for c in fake.calls if c[1] == "run"]) == 1   # one container for the episode
    w.end_round(1)
    assert fake.calls[-1][-1] == "sleep 999"                     # end_round only ticks
    w.end_run("max_rounds")                                      # the run's end: capture, remove
    # the run's end kills what is still running in the container before the tar
    assert fake.calls[-3] == ["docker", "exec", "-u", "root", "cid123", "bash", "-c", "kill -9 -1"]
    assert fake.calls[-2][:5] == ["docker", "exec", "-u", "root", "cid123"]
    assert "tar --sort=name --format=gnu" in fake.calls[-2][-1] and "gzip -n" in fake.calls[-2][-1]
    assert fake.calls[-1] == ["docker", "rm", "-f", "cid123"] and w._runtime is None
    blob = w.snapshot()
    assert workspace_tar_of(blob) == tar_bytes
    assert w.render_state()["workspace"] == [{"name": "main.c", "type": "file", "bytes": 10}]
    n = len(fake.calls)
    w.snapshot()
    assert len(fake.calls) == n                                  # no container after cleanup


def test_docker_restore_materialises_in_a_new_container(tmp_path, monkeypatch):
    fake = FakeDocker(b"")
    fake.install(monkeypatch)
    w = ProgramBench(TASK, backend="docker", user=None, cpus=8, memory_gb=31.5, check_task=False)
    w.reset(random.Random(0), ["a000"])
    assert w.snapshot() and workspace_tar_of(w.snapshot()) is None   # never started: image's own
    assert fake.calls == []
    w.restore(pickle_blob({"steps_total": 4}, b"TAR"))
    w.begin_round(2)
    run, _git, restore = fake.calls
    assert "--user" not in run and run[run.index("--cpus") + 1] == "8"
    assert run[run.index("--memory") + 1] == "32256m"
    assert restore[:6] == ["docker", "exec", "-i", "-u", "root", "cid123"]
    assert "find /workspace -mindepth 1 -delete && tar -xzf - -C /workspace" in restore[-1]
    w._runtime.stop()


def pickle_blob(meta: dict, tar: bytes | None) -> bytes:
    import pickle

    return pickle.dumps({"meta": meta, "workspace_tar": tar})


# ---- a scripted dry run through Experiment.run ---------------------------------------------------

def test_fake_script_dry_run_submits_and_replays(tmp_path):
    ws = tmp_path / "ws"
    world = ProgramBench(TASK, workspace=str(ws), user=None, lockdown=False, step_limit=5,
                         check_task=False)
    exp = Experiment(name="pb-dry", world=world,
                     participants=[LLMAgent(model="fake:programbench_echo", max_tokens=256)],
                     options={"commit": "immediate", "max_calls_per_turn": 5})
    run = exp.run(seed=0, max_rounds=1, out=tmp_path / "runs")
    assert run.meta["status"] == "ended" and run.meta["end_reason"] == "terminal"
    sc = run.score
    assert (sc["submitted"], sc["steps"], sc["graded"]) == (True, 3, False)
    acts = [e for e in run.events if e["type"] == "action_committed"]
    assert [a["action"]["args"]["command"] for a in acts] == list(pb.FAKE_COMMANDS)
    assert all(set(a["feedback"]) == {"output", "returncode", "exception_info"} for a in acts)
    assert acts[-1]["feedback"]["output"] == f"{SUBMIT_MARKER}\n"
    assert (ws / "executable").read_text() == "#!/bin/sh\necho hello\n"
    resolved = next(e for e in run.events if e["type"] == "round_resolved")["data"]
    assert (resolved["steps"], resolved["submitted"]) == (3, True)
    store = SnapshotStore(run.dir)
    tar_bytes = workspace_tar_of(store.load(store.latest())["world"])
    with tarfile.open(fileobj=io.BytesIO(tar_bytes), mode="r:gz") as tar:
        assert sorted(tar.getnames()) == ["compile.sh", "executable"]
    assert Run.load(run.dir).replay()["score"] == sc
    states = collect(run.dir)["world_states"]
    assert [e["name"] for e in states["1"]["workspace"]] == ["compile.sh", "executable"]
    assert build_view(run.dir).exists()


def test_fake_script_needs_a_large_enough_cap(tmp_path):
    world = ProgramBench(TASK, workspace=str(tmp_path / "ws"), user=None, lockdown=False,
                         check_task=False)
    exp = Experiment(name="pb-cap", world=world,
                     participants=[LLMAgent(model="fake:programbench_echo", max_tokens=256)],
                     options={"commit": "immediate"})
    with pytest.raises(ValueError, match="max_calls_per_turn must be at least step_limit"):
        exp.run(seed=0, max_rounds=1, out=tmp_path / "runs")


def test_failed_docker_tar_keeps_the_container(tmp_path, monkeypatch):
    fake = FakeDocker(b"")

    def failing(cmd, **kw):
        if "tar --sort=name" in list(cmd)[-1]:
            fake.calls.append(list(cmd))
            return subprocess.CompletedProcess(cmd, 2, stdout=b"", stderr=b"tar: boom")
        return fake(cmd, **kw)

    fake.install(monkeypatch, failing)
    w = ProgramBench(TASK, backend="docker", check_task=False)
    w.check_experiment(options(max_calls_per_turn=1000))
    w.reset(random.Random(0), ["a000"])
    w.begin_round(1)
    sh(w, "ls")
    w.end_round(1)
    with pytest.raises(RuntimeError, match="workspace tar failed: tar: boom"):
        w.end_run("terminal")
    assert ["docker", "rm", "-f", "cid123"] not in fake.calls and w._runtime.started
    w._runtime.stop()
    assert fake.calls[-1] == ["docker", "rm", "-f", "cid123"]


def test_machine_facts_come_from_reset_not_the_snapshot(tmp_path):
    w = make(tmp_path, cpus=8)
    w.uname = {**UNAME, "machine": "elsewhere"}
    blob = w.snapshot()
    other = make(tmp_path, "ws2", cpus=2)
    other.restore(blob)
    assert other.resources["cpus"] == 2 and other.uname["machine"] != "elsewhere"
    assert "This machine has 2 CPUs." in other.description()


# ---- commit: async: concurrent commands, the submission freeze, run end, cancellation ----------

def _alive(pid: int) -> bool:
    try:
        os.kill(pid, 0)
    except ProcessLookupError:
        return False
    return True


def _pid(path: Path, deadline_s: float = 10) -> int:
    deadline = time.monotonic() + deadline_s
    while not (path.exists() and path.read_text().strip()) and time.monotonic() < deadline:
        time.sleep(0.02)
    return int(path.read_text())


def test_bash_is_a_coroutine_and_state_changes_only_after_the_command(tmp_path):
    import inspect

    w = make(tmp_path)
    assert inspect.iscoroutinefunction(ProgramBench.bash)

    async def go():
        task = asyncio.create_task(w.bash("a000", "sleep 0.3; echo done"))
        await asyncio.sleep(0.1)
        assert (w.steps_total, w.history) == (0, [])      # nothing recorded mid-command
        assert list(w.running_commands()) == ["a000"]
        return await task

    assert asyncio.run(go()).feedback == {"output": "done\n", "returncode": 0, "exception_info": ""}
    assert w.steps_total == 1 and w.running_commands() == {}


def test_two_agents_first_commands_start_the_backend_once(tmp_path, monkeypatch):
    calls = []
    monkeypatch.setattr(pb, "apply_lockdown", lambda user, hosts: calls.append(user)
                        or {"applied": True, "pinned": {}})
    monkeypatch.setattr(pb, "detect_resources", lambda *a: (4.0, 16.0))
    monkeypatch.setattr(pb.os, "geteuid", lambda: 0)
    w = make(tmp_path, agents=("a000", "a001"), lockdown=True)

    async def go():
        return await asyncio.gather(w.bash("a000", "echo a"), w.bash("a001", "echo b"))

    a, b = asyncio.run(go())
    assert (a.feedback["output"], b.feedback["output"]) == ("a\n", "b\n")
    assert calls == [None]


def test_submission_kills_the_other_agents_running_command(tmp_path):
    w = make(tmp_path, agents=("a000", "a001"))
    ws = tmp_path / "ws"

    async def go():
        other = asyncio.create_task(w.bash("a001", "echo $$ > b.pid; echo started; sleep 5; "
                                                   "touch late"))
        pid = await asyncio.to_thread(_pid, ws / "b.pid")
        await asyncio.sleep(0.1)
        sub = await w.bash("a000", f"echo {SUBMIT_MARKER}")
        assert w.submitted and not _alive(pid)               # killed at once, not at its end
        return sub, await other

    sub, other = asyncio.run(go())
    assert sub.feedback["returncode"] == 0
    assert other.accepted and other.feedback == {
        "output": "started\n", "returncode": -9,
        "exception_info": "An error occurred while executing the command: Command 'echo $$ > "
                          "b.pid; echo started; sleep 5; touch late' was killed: a000 submitted "
                          "the final output (the episode is over)"}
    assert w.running_commands() == {} and w.steps == {"a000": 1, "a001": 1}
    assert [h["agent"] for h in w.history] == ["a000", "a001"]
    assert w.timeouts == 0
    time.sleep(0.2)
    assert not (ws / "late").exists()
    assert bash(w, "ls", "a001").accepted is False           # and nothing new starts


def test_submission_kills_timed_out_commands_too(tmp_path):
    w = make(tmp_path, command_timeout_s=0.3)
    fb = sh(w, "echo $$ > o.pid; sleep 30")
    assert fb["returncode"] == -1 and "timed out" in fb["exception_info"]
    pid = _pid(tmp_path / "ws" / "o.pid")
    assert w.running_commands() == {"a000": [pid]}           # the orphan is tracked
    sh(w, f"echo {SUBMIT_MARKER}")
    assert not _alive(pid) and w.running_commands() == {}


def test_run_end_kills_orphans(tmp_path):
    w = make(tmp_path, command_timeout_s=0.3)
    w.check_experiment(options(max_rounds=3))
    sh(w, "echo $$ > o.pid; sleep 30")
    pid = _pid(tmp_path / "ws" / "o.pid")
    w.end_round(1)
    w.begin_round(2)
    assert _alive(pid)                                       # not the run's end: still running
    w.end_round(3)                                           # end_round alone kills nothing
    assert _alive(pid)
    w.end_run("max_rounds")                                  # the run's end
    assert not _alive(pid) and w.running_commands() == {}


def test_cancelled_command_is_killed_and_records_nothing(tmp_path):
    w = make(tmp_path)

    async def go():
        task = asyncio.create_task(w.bash("a000", "echo $$ > c.pid; sleep 30"))
        pid = await asyncio.to_thread(_pid, tmp_path / "ws" / "c.pid")
        task.cancel()
        with pytest.raises(asyncio.CancelledError):
            await task
        return pid

    pid = asyncio.run(go())
    time.sleep(0.1)
    assert not _alive(pid) and w.running_commands() == {}
    assert (w.steps_total, w.steps, w.history, w.elapsed_s) == (0, {"a000": 0}, [], 0.0)


BACKGROUND = "(echo $BASHPID > bg.pid; sleep 1; echo changed > late) &"


def test_a_finished_shells_background_job_stays_tracked_and_the_freeze_kills_it(tmp_path):
    w = make(tmp_path)
    ws = tmp_path / "ws"
    fb = sh(w, BACKGROUND)
    assert fb == {"output": "", "returncode": 0, "exception_info": ""}
    pid = _pid(ws / "bg.pid")
    shell = next(iter(w.running_commands()["a000"]))
    assert w.running_commands() == {"a000": [shell]} and _alive(pid)   # its group is alive
    sh(w, f"echo {SUBMIT_MARKER}")                       # the submission freezes the workspace
    assert not _alive(pid) and w.running_commands() == {}
    blob = w.snapshot()
    time.sleep(1.2)
    assert not (ws / "late").exists()
    assert workspace_tar_of(blob) == workspace_tar(ws)[0]


def test_a_finished_shells_background_job_is_killed_at_run_end(tmp_path):
    w = make(tmp_path)
    w.check_experiment(options(max_rounds=1, max_calls_per_turn=1000))
    ws = tmp_path / "ws"
    sh(w, BACKGROUND)
    pid = _pid(ws / "bg.pid")
    w.end_round(1)
    w.end_run("max_rounds")                              # the run's end
    assert not _alive(pid) and w.running_commands() == {}
    time.sleep(1.2)
    assert not (ws / "late").exists()


def test_a_job_that_left_its_group_is_killed_by_the_marker_sweep(tmp_path):
    w = make(tmp_path)
    ws = tmp_path / "ws"
    fb = sh(w, f"echo ${pb.RUN_MARKER}; setsid sh -c 'echo $$ > s.pid; sleep 1; "
               "echo changed > late' > /dev/null 2>&1 < /dev/null &")
    assert fb["returncode"] == 0 and fb["output"].strip() == w._runtime.marker.split("=", 1)[1]
    pid = _pid(ws / "s.pid")
    assert _alive(pid)
    assert pb.procs.stat_fields(pid)[2] != str(next(iter(w._runtime.commands), -1))
    sh(w, f"echo {SUBMIT_MARKER}")
    assert not _alive(pid)
    time.sleep(1.2)
    assert not (ws / "late").exists()


def test_the_dedicated_users_uid_is_swept_only_under_lockdown(tmp_path, monkeypatch):
    user = SimpleNamespace(pw_uid=4242, pw_gid=4242, pw_dir="/home/agent", pw_name="agent",
                           pw_shell="/bin/bash")
    monkeypatch.setattr(pb.pwd, "getpwnam", lambda name: user)
    monkeypatch.setattr(pb.os, "geteuid", lambda: 0)
    locked = pb.LocalBackend(str(tmp_path / "ws"), "agent", {}, None,
                             lockdown=lambda: {"applied": True})
    locked.start()
    assert locked._sweep["uid"] == 4242 and locked.prefix[0] == "setpriv"
    assert locked.env[pb.RUN_MARKER] == locked.marker.split("=", 1)[1]
    plain = pb.LocalBackend(str(tmp_path / "ws"), "agent", {}, None)
    plain.start()
    assert plain._sweep["uid"] is None                   # a dry run: the marker sweep only
    user.pw_uid = pb.procs.uid_of(os.getpid())           # never the uid swarmlab runs as
    mine = pb.LocalBackend(str(tmp_path / "ws"), "agent", {}, None,
                           lockdown=lambda: {"applied": True})
    mine.start()
    assert mine._sweep["uid"] is None
    seen = []
    monkeypatch.setattr(pb.procs, "kill_matching", lambda match: seen.append(match) or [])
    monkeypatch.setattr(pb.procs, "uid_of", lambda pid: {1: 4242, 2: 7}[pid])
    pb._sweep_local(locked._sweep)
    _, uid_match = seen
    assert (uid_match(1), uid_match(2)) == (True, False)


def test_snapshot_tar_survives_a_file_shrinking_while_read(tmp_path):
    root = tmp_path / "ws"
    root.mkdir()
    (root / "big").write_bytes(b"x" * 100_000)
    real_open = open

    def shrinking_open(path, mode="r", *a, **kw):
        f = real_open(path, mode, *a, **kw)
        if str(path).endswith("big") and "b" in mode:
            os.truncate(path, 10)                             # another agent rewrites it now
        return f

    import builtins

    builtins.open, saved = shrinking_open, builtins.open
    try:
        data, _ = workspace_tar(root)
    finally:
        builtins.open = saved
    with tarfile.open(fileobj=io.BytesIO(data), mode="r:gz") as tar:
        member = tar.getmember("big")
        assert member.size == 100_000 and tar.extractfile(member).read()[:10] == b"x" * 10


def test_async_commit_is_accepted_and_the_one_turn_rule_holds():
    w = ProgramBench(TASK, check_task=False)
    with pytest.raises(ValueError, match="max_calls_per_turn must be at least step_limit"):
        w.check_experiment(options(commit="async"))          # one tick = one turn = the episode
    w.check_experiment(options(commit="async", max_rounds=90, max_calls_per_turn=40))
    assert w._max_rounds == 90
    Experiment(name="pb", world=ProgramBench(TASK, check_task=False),
               participants=[LLMAgent(model="fake:programbench_echo")],
               options={"commit": "async", "max_calls_per_turn": 1000})


SEEN: dict[str, list[dict]] = {}


class Commander(Participant):
    """A scripted participant: `before_s` asleep, its commands in order (results kept in
    `SEEN[agent]`), `after_s` asleep, then `end_turn`; later turns end at once."""

    def __init__(self, commands: list[str], before_s: float = 0.0, after_s: float = 0.0) -> None:
        self.commands, self.before_s, self.after_s = list(commands), before_s, after_s
        self.done = False

    async def turn(self, view, tools):
        if not self.done:
            self.done = True
            await asyncio.sleep(self.before_s)
            for c in self.commands:
                res = await tools.call("bash", {"command": c})
                SEEN.setdefault(self.agent, []).append(res.result)
            await asyncio.sleep(self.after_s)
        await tools.call("end_turn", {})
        return TurnUsage()


def _async_run(tmp_path, participants, **options):
    world = ProgramBench(TASK, workspace=str(tmp_path / "ws"), user=None, lockdown=False,
                         check_task=False)
    exp = Experiment(name="pb-async", world=world, participants=participants,
                     options={"commit": "async", "max_calls_per_turn": 20, **options})
    return exp.run(seed=0, out=tmp_path / "runs")


def test_async_two_agents_commands_overlap(tmp_path):
    stamp = "echo start $(date +%s.%N); sleep 0.4; echo end $(date +%s.%N)"
    run = _async_run(tmp_path, [Commander([stamp]), Commander([stamp])], rounds_per_agent=1)
    assert run.meta["status"] == "ended" and run.meta["last_round"] == 2
    acts = [e for e in run.events if e["type"] == "action_committed"]
    assert sorted(a["agent"] for a in acts) == ["a000", "a001"]
    spans = []
    for a in acts:
        lines = dict(line.split() for line in a["feedback"]["output"].splitlines())
        spans.append((float(lines["start"]), float(lines["end"])))
    assert max(s for s, _ in spans) < min(e for _, e in spans)   # both ran at the same time
    assert Run.load(run.dir).replay()["score"] == run.score


def test_async_submission_freezes_the_workspace_and_ends_the_run(tmp_path):
    SEEN.clear()
    slow = "echo $$ > b.pid; sleep 5; touch late"
    run = _async_run(tmp_path, [Commander([f"echo {SUBMIT_MARKER}"], before_s=0.5, after_s=1.0),
                                Commander([slow])], rounds_per_agent=2)
    assert run.meta["end_reason"] == "terminal" and run.score["submitted"]
    # B's ticks closed after the submission did not end the run: the submitter's turn did
    ended = [(e["round"], e["agent"]) for e in run.events if e["type"] == "turn_ended"]
    assert ended[-1] == (run.meta["last_round"], "a000") and len(ended) == 3
    fb = SEEN["a001"][0]["feedback"]
    assert fb["returncode"] == -9 and "a000 submitted the final output" in fb["exception_info"]
    ws = tmp_path / "ws"
    assert not _alive(int((ws / "b.pid").read_text()))
    time.sleep(0.2)
    assert not (ws / "late").exists()
    store = SnapshotStore(run.dir)
    latest = store.latest()
    assert latest.round == run.meta["last_round"]                # the final tick is snapshotted
    assert workspace_tar_of(store.load(latest)["world"]) == workspace_tar(ws)[0]
    assert Run.load(run.dir).replay()["score"] == run.score


ORPHAN = "echo $$ > o.pid; sleep 2; echo changed > late"
AT_SNAPSHOT: list[bool] = []   # was the orphan alive when the runner took each snapshot


class Watched(ProgramBench):
    """ProgramBench that notes, at every snapshot, whether the orphan (`o.pid`) still runs."""

    entry_point = None

    def snapshot(self) -> bytes:
        pid_file = Path(self.workspace) / "o.pid"
        if pid_file.exists() and pid_file.read_text().strip():
            AT_SNAPSHOT.append(_alive(int(pid_file.read_text())))
        return super().snapshot()


def _orphan_world(tmp_path) -> Watched:
    return Watched(TASK, workspace=str(tmp_path / "ws"), user=None, lockdown=False,
                   check_task=False, command_timeout_s=0.1, step_limit=20)


def _check_orphan_killed_before_the_final_snapshot(tmp_path, run) -> None:
    ws = tmp_path / "ws"
    assert AT_SNAPSHOT and AT_SNAPSHOT[-1] is False        # killed before the final snapshot
    pid = int((ws / "o.pid").read_text())
    assert not _alive(pid)
    store = SnapshotStore(run.dir)
    latest = store.latest()
    assert latest.round == run.meta["last_round"]
    time.sleep(2.2)
    assert not (ws / "late").exists()                       # and it never wrote its file
    assert workspace_tar_of(store.load(latest)["world"]) == workspace_tar(ws)[0]
    assert Run.load(run.dir).replay()["score"] == run.score


def test_async_run_ended_by_the_per_agent_cap_kills_a_timed_out_command(tmp_path):
    """The run ends `max_rounds` at tick 1 (rounds_per_agent 1), long before `max_rounds` 10:
    the world's `end_run`, not a guess in `end_round`, kills the timed-out command."""
    SEEN.clear()
    AT_SNAPSHOT.clear()
    exp = Experiment(name="orphan-async", world=_orphan_world(tmp_path),
                     participants=[Commander([ORPHAN])],
                     options={"commit": "async", "rounds_per_agent": 1, "max_calls_per_turn": 20})
    run = exp.run(seed=0, max_rounds=10, out=tmp_path / "runs")
    assert (run.meta["end_reason"], run.meta["last_round"]) == ("max_rounds", 1)
    assert SEEN["a000"][0]["feedback"]["returncode"] == -1  # it timed out and was left running
    _check_orphan_killed_before_the_final_snapshot(tmp_path, run)


def orphan_then_end(request, rng):
    """Fake script: the orphan command once, then `end_turn` (every turn after that too)."""
    from swarmlab.providers.fake import _assistant_calls, _response

    done = sum(1 for m in request.messages for name, _ in _assistant_calls(m) if name == "bash")
    if done == 0:
        return _response(request, rng, [("bash", {"command": ORPHAN})])
    return _response(request, rng, [("end_turn", {})])


def test_immediate_run_ended_by_the_soft_budget_kills_a_timed_out_command(tmp_path):
    from swarmlab import Budget
    from swarmlab.providers.fake import FakeProvider

    AT_SNAPSHOT.clear()
    exp = Experiment(name="orphan-soft", world=_orphan_world(tmp_path),
                     participants=[LLMAgent(model=f"fake:{__name__}:orphan_then_end",
                                            max_tokens=256)],
                     budget=Budget(soft_usd=0.0001),
                     providers={"fake": FakeProvider(pricing={"*": (10.0, 50.0, 1.0)})},
                     options={"commit": "immediate", "max_calls_per_turn": 20})
    run = exp.run(seed=0, max_rounds=10, out=tmp_path / "runs")
    assert (run.meta["end_reason"], run.meta["last_round"]) == ("soft_budget", 1)
    fb = [e["feedback"] for e in run.events if e["type"] == "action_committed"]
    assert fb[0]["returncode"] == -1 and "timed out" in fb[0]["exception_info"]
    _check_orphan_killed_before_the_final_snapshot(tmp_path, run)


def test_submission_closed_exists_only_under_async(tmp_path):
    w = make(tmp_path)
    assert "submission_closed" not in pickle_meta(w.snapshot())
    w = ProgramBench(TASK, workspace=str(tmp_path / "ws2"), user=None, lockdown=False,
                     check_task=False)
    w.set_commit_mode("async")
    w.reset(random.Random(0), ["a000", "a001"])
    w.begin_round(1)
    sh(w, f"echo {SUBMIT_MARKER}")
    assert w.submitted and not w.terminal()
    w.close_turn("a001", 1)
    assert not w.terminal()
    w.close_turn("a000", 1)
    assert w.terminal() and pickle_meta(w.snapshot())["submission_closed"] is True


def pickle_meta(blob: bytes) -> dict:
    import pickle

    return pickle.loads(blob)["meta"]
