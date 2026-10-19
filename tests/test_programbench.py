"""ProgramBench world (temp-dir local backend, mocked docker), its lockdown seam and the fake
script `programbench_echo`. Nothing here needs Docker, the task images, the network or the
`programbench` package (the tests that read the package skip without it)."""
from __future__ import annotations

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


def sh(w: ProgramBench, command: str, agent: str = "a000") -> dict:
    out = w.bash(agent, command)
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
    assert w.bash("a000", "ls").accepted is False
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


def test_templates_are_the_installed_mini_swe_agent_config():
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


def test_requires_immediate_commit():
    with pytest.raises(ValueError, match="requires commit=immediate"):
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
    with pytest.raises(ValueError, match="requires commit=immediate"):
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


def test_docker_backend_commands(tmp_path, monkeypatch):
    (tmp_path / "src").mkdir()
    (tmp_path / "src" / "main.c").write_text("int main;\n")
    tar_bytes, _ = workspace_tar(tmp_path / "src")
    fake = FakeDocker(tar_bytes)
    monkeypatch.setattr(subprocess, "run", fake)
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
    w.end_round(1)                                               # max_rounds 1: capture, remove
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
    monkeypatch.setattr(subprocess, "run", fake)
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

    monkeypatch.setattr(subprocess, "run", failing)
    w = ProgramBench(TASK, backend="docker", check_task=False)
    w.check_experiment(options(max_calls_per_turn=1000))
    w.reset(random.Random(0), ["a000"])
    w.begin_round(1)
    sh(w, "ls")
    with pytest.raises(RuntimeError, match="workspace tar failed: tar: boom"):
        w.end_round(1)
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
