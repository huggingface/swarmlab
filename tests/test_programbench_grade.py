"""`swarmlab programbench grade`: the local container class the official evaluator runs on (temp
dirs, plus one end-to-end evaluation of the package's fixture task with the real `Evaluator`),
the submission extraction from a dry-run run dir, the grade-job plan, the launch (mocked hf
CLI), the score write-back and what `Run.load`, `report` and `export` show afterwards, and
bootstrap.sh's grade branch against stubs. Nothing is launched; no Docker, image or dataset."""
from __future__ import annotations

import io
import json
import os
import pickle
import shlex
import shutil
import subprocess
import sys
import tarfile
import time
import zipfile
from pathlib import Path

import pytest
from typer.testing import CliRunner

from swarmlab import Experiment, Run
from swarmlab.cli import app
from swarmlab.jobs import grade as g
from swarmlab.jobs import launch as jl
from swarmlab.jobs import programbench as pbj
from swarmlab.participants import LLMAgent
from swarmlab.report import write_report
from swarmlab.snapshot import SnapshotStore
from swarmlab.spec import SpecError
from swarmlab.world.programbench import ProgramBench

from .test_jobs import COMMIT, NOW, REPO, STUB_HF, FakeHF

TASK = "abishekvashok__cmatrix.5c082c6"
IMAGE = "programbench/abishekvashok_1776_cmatrix.5c082c6:task_cleanroom_v6"
cli = CliRunner()


# ---- LocalEvalEnvironment ---------------------------------------------------------------------
def machine(tmp_path, **kw):
    ws = tmp_path / "ws"
    ws.mkdir(exist_ok=True)
    kw.setdefault("environ", {"PATH": os.environ["PATH"], "HOME": str(tmp_path),
                              "HF_TOKEN": "secret", "ANTHROPIC_KEY": "k", "SWARMLAB_TAG": "t",
                              "PROGRAMBENCH_BLOB_DIR": "/x", "KEEP_ME": "1"})
    return g.LocalMachine(tmp_path / "store", workspace=str(ws),
                          keep=(str(tmp_path / "stash"),), **kw)


def env(m, image="base:1", **kw):
    return g.LocalEvalEnvironment(image=image, cwd=m.workspace, timeout=600, cpus=3,
                                  env=kw.pop("env", {"PYTEST_ADDOPTS": "--max-worker-restart=4"}),
                                  machine=m, **kw)


def test_execute_runs_bash_lc_in_cwd_with_the_container_env(tmp_path):
    m = machine(tmp_path)
    e = env(m)
    r = e.execute("pwd; echo out; echo err >&2; printf 'a\\r\\nb'; exit 3")
    assert r == {"output": f"{m.workspace}\nout\na\nb" + "err\n", "returncode": 3,
                 "exception_info": ""}
    got = e.execute("echo $PYTEST_XDIST_AUTO_NUM_WORKERS $PYTEST_ADDOPTS $KEEP_ME "
                    "${HF_TOKEN:-none} ${ANTHROPIC_KEY:-none} ${SWARMLAB_TAG:-none} "
                    "${PROGRAMBENCH_BLOB_DIR:-none}")["output"]
    assert got == "3 --max-worker-restart=4 1 none none none none\n"
    serial = env(m, env={"PYTEST_XDIST_AUTO_NUM_WORKERS": "1"})
    assert serial.execute("echo $PYTEST_XDIST_AUTO_NUM_WORKERS")["output"] == "1\n"


def test_execute_timeout_kills_the_command_and_reports_the_official_values(tmp_path):
    m = machine(tmp_path)
    e = env(m)
    pidfile = tmp_path / "pid"
    t0 = time.monotonic()
    r = e.execute(f"sleep 30 & echo $! > {pidfile}; echo partial; wait", timeout=1)
    assert time.monotonic() - t0 < 20
    assert r == {"output": "", "returncode": -1, "exception_info": "Command timed out after 1s"}
    pid = int(pidfile.read_text())
    time.sleep(0.2)
    assert not Path(f"/proc/{pid}").exists() or "Z" in Path(f"/proc/{pid}/stat").read_text().split()[2]
    fast = g.LocalEvalEnvironment(image="base:1", cwd=m.workspace, timeout=1, machine=m)
    assert fast.execute("sleep 5")["exception_info"] == "Command timed out after 1s"


def test_cleanup_stops_what_commands_left_running(tmp_path):
    m = machine(tmp_path)
    e = env(m)
    pidfile = tmp_path / "bg"
    assert e.execute(f"(sleep 60 >/dev/null 2>&1 &) ; sleep 60 >/dev/null 2>&1 & echo $! > {pidfile}")[
        "returncode"] == 0
    pid = int(pidfile.read_text())
    assert Path(f"/proc/{pid}").exists()
    e.cleanup()
    time.sleep(0.3)
    assert not Path(f"/proc/{pid}").exists() or \
        Path(f"/proc/{pid}/stat").read_text().rsplit(")", 1)[1].split()[0] == "Z"


def test_copy_in_and_copy_in_tar(tmp_path):
    m = machine(tmp_path)
    e = env(m)
    src = tmp_path / "src"
    (src / "sub").mkdir(parents=True)
    (src / "sub" / "f.txt").write_text("x")
    (src / "run.sh").write_text("#!/bin/sh\n")
    (src / "run.sh").chmod(0o755)
    (src / "link").symlink_to("../../etc/hosts")  # an escaping symlink survives, as via tar
    e.copy_in(src, f"{m.workspace}/copied")
    ws = Path(m.workspace)
    assert (ws / "copied/sub/f.txt").read_text() == "x"
    assert os.access(ws / "copied/run.sh", os.X_OK)
    assert os.readlink(ws / "copied/link") == "../../etc/hosts"
    plug = tmp_path / "plugdir"
    plug.mkdir()
    e.copy_in(src / "run.sh", str(plug / "p.py"))
    assert (plug / "p.py").read_text() == "#!/bin/sh\n"
    buf = io.BytesIO()
    with tarfile.open(fileobj=buf, mode="w:gz") as tar:
        tar.add(src / "sub" / "f.txt", arcname="eval/tests/t.py")
    tgz = tmp_path / "b.tar.gz"
    tgz.write_bytes(buf.getvalue())
    e.copy_in_tar(tgz, f"{m.workspace}/")
    assert (ws / "eval/tests/t.py").read_text() == "x"
    with pytest.raises(RuntimeError, match="tar stream into container failed"):
        e.copy_in_tar(tmp_path / "missing.tar.gz", f"{m.workspace}/")


def test_commit_then_fresh_environment_resets_workspace_and_stash(tmp_path):
    m = machine(tmp_path, base_image="base:1")
    e = env(m)
    ws, stash = Path(m.workspace), tmp_path / "stash"
    e.execute("mkdir -p build && echo artefact > build/x && echo v1 > keep.txt && "
              f"printf bin > {stash} && ln -s keep.txt link")
    ref = e.commit("programbench-compiled/t:abc")
    assert ref == "programbench-compiled/t:abc" and m.images[ref].exists()
    # a branch run mutates everything, consumes the stash, leaves debris
    e.execute(f"rm -rf build && echo v2 > keep.txt && echo junk > .hidden && mv {stash} ./executable")
    assert not stash.exists()
    branch = env(m, image=ref)  # "a fresh container from the committed image"
    assert m.resets == [ref]
    assert sorted(p.name for p in ws.iterdir()) == ["build", "keep.txt", "link"]
    assert (ws / "keep.txt").read_text() == "v1\n" and (ws / "build/x").read_text() == "artefact\n"
    assert os.readlink(ws / "link") == "keep.txt" and stash.read_text() == "bin"
    assert branch.execute("cat keep.txt")["output"] == "v1\n"
    env(m, image="base:1")  # the job's own image: nothing is reset
    assert m.resets == [ref]
    with pytest.raises(RuntimeError, match="local grading runs in base:1"):
        env(m, image="other:2")
    m.remove_image(ref)
    assert ref not in m.images and not list((m.store / "images").iterdir())


def test_docker_shim_reads_a_file_like_docker_cp(tmp_path):
    m = machine(tmp_path)
    e = env(m)
    (Path(m.workspace) / "eval").mkdir()
    (Path(m.workspace) / "eval" / "results.xml").write_text("<testsuites/>")
    dest = tmp_path / "out.xml"
    dest.write_text("")
    cp = subprocess.run([e.executable, "cp", f"{e.container_id}:{m.workspace}/eval/results.xml",
                         str(dest)], capture_output=True, text=True, check=False)
    assert cp.returncode == 0 and dest.read_text() == "<testsuites/>"
    bad = subprocess.run([e.executable, "exec", "x"], capture_output=True, text=True, check=False)
    assert bad.returncode == 2


def test_grade_env_strips_secrets_swarmlab_and_the_venv():
    out = g.grade_env({"PATH": "/venv/bin:/usr/bin", "VIRTUAL_ENV": "/venv", "HF_TOKEN": "x",
                       "UV_PYTHON_INSTALL_DIR": "/p", "HF_HOME": "/h", "LANG": "C.UTF-8",
                       "PYTHONDONTWRITEBYTECODE": "1"}, "/venv/bin")
    assert out == {"PATH": "/usr/bin", "LANG": "C.UTF-8"}


# ---- the real Evaluator on the package's fixture task -------------------------------------------
FIXTURE = "testorg__calculator.abc1234"


@pytest.mark.skipif(shutil.which("git") is None or shutil.which("sha256sum") is None,
                    reason="needs git and sha256sum")
def test_official_evaluator_runs_on_the_local_environment(tmp_path, monkeypatch):
    """The fixture task (bash calculator) graded end to end through `grade()`: the official
    Evaluator compiles, commits, resets, runs the branch with pytest and parses the XML; the test
    results match the package's own recorded evaluation of the same submission."""
    pytest.importorskip("programbench")
    import programbench
    from programbench.eval import eval as pb_eval

    data = Path(programbench.__file__).parent / "data"
    # the hidden-tests layout: <blob dir>/<task>/tests/<branch>.tar.gz
    tests = tmp_path / "tests"
    branch_dir = data / "tasks" / FIXTURE / "tests" / "33128f6b8600"
    (tests / FIXTURE / "tests").mkdir(parents=True)
    with tarfile.open(tests / FIXTURE / "tests" / "33128f6b8600.tar.gz", "w:gz") as tar:
        tar.add(branch_dir / "eval", arcname="eval")
    sub = tmp_path / "submission.tar.gz"
    with zipfile.ZipFile(data / "test_runs" / "correct" / FIXTURE / "submission.zip") as z, \
            tarfile.open(sub, "w:gz") as tar:
        for name in z.namelist():
            info = tarfile.TarInfo(name)
            body = z.read(name)
            info.size, info.mode = len(body), 0o755
            tar.addfile(info, io.BytesIO(body))
    # not root here: no DNS blackhole (the job does it as root), no package index
    monkeypatch.setattr(pb_eval, "block_build_internet_dns", lambda env: None)
    monkeypatch.setattr(pb_eval, "restore_build_internet_dns", lambda env: None)
    shim = tmp_path / "pybin"
    shim.mkdir()
    (shim / "python3").write_text(f"#!/bin/sh\nexec {sys.executable} \"$@\"\n")
    (shim / "python3").chmod(0o755)
    # `bash -lc` is a login shell: /etc/profile may reset PATH, ~/.bash_profile comes after it
    home = tmp_path / "home"
    home.mkdir()
    # (and the task images ship pytest-xdist, this env does not: drop its --max-worker-restart)
    (home / ".bash_profile").write_text(f'export PATH="{shim}:$PATH"\nunset PYTEST_ADDOPTS\n')
    environ = {**os.environ, "PIP_NO_INDEX": "1", "HOME": str(home)}
    monkeypatch.setattr(g, "_venv_bin", lambda: None)
    ws = tmp_path / "workspace"
    ws.mkdir()
    (ws / "stale").write_text("from the image")
    out = tmp_path / "out" / "grade.json"
    doc = g.grade(FIXTURE, sub, tests, out, workspace=str(ws), stash=str(tmp_path / "stash"),
                  environ=environ)
    official = json.loads((data / "test_runs" / "correct" / FIXTURE /
                           f"{FIXTURE}.eval.json").read_text())
    ev = doc["evaluation"]
    assert ev["error_code"] is None and ev["executable_hash"] == official["executable_hash"]
    assert sorted((t["name"], t["status"]) for t in ev["test_results"]) == \
        sorted((t["name"], t["status"]) for t in official["test_results"])
    steps = [s["step"] for s in ev["log"]]
    for step in ("wipe_workspace", "clear_stale_executable", "seed_git", "compile",
                 "copy_executable", "hash_executable", "install_rerunfailures",
                 "restore_executable", "verify_executable_hash", "patch_timeout_method",
                 "run_tests", "results_read"):
        assert step in steps, step
    assert doc["score"] == {"score": 0.5, "passed": 3, "total": 6, "resolved": False,
                            "near_resolved": False, "branches": 1, "compile_ok": True,
                            "error_code": None}
    assert doc["pytest_workers"] >= 1 and doc["programbench_version"] == "1.2.5"
    assert json.loads(out.read_text())["score"] == doc["score"]
    assert (out.parent / f"{FIXTURE}.eval.json").exists()
    # the evaluator's module state is restored
    assert pb_eval.WORKSPACE_DIR == "/workspace"
    assert pb_eval.ContainerEnvironment.__name__ == "ContainerEnvironment"
    assert pb_eval.Evaluator._stashed_executable == g.STASHED_EXECUTABLE


def test_grade_refuses_a_wrong_image_and_unknown_task(tmp_path):
    pytest.importorskip("programbench")
    with pytest.raises(ValueError, match="graded in programbench/abishekvashok_1776"):
        g.grade(TASK, tmp_path / "s.tar.gz", tmp_path, tmp_path / "g.json", image="other:1")
    assert g.main(["--task", "nobody__x.1", "--submission", "s", "--tests", str(tmp_path),
                   "--out", str(tmp_path / "g.json")]) == 1


# ---- a dry-run run dir --------------------------------------------------------------------------
def dry_run(tmp_path) -> Run:
    world = ProgramBench(TASK, workspace=str(tmp_path / "ws"), user=None, lockdown=False,
                         step_limit=5, check_task=False)
    exp = Experiment(name="pb-dry", world=world,
                     participants=[LLMAgent(model="fake:programbench_echo", max_tokens=256)],
                     options={"commit": "immediate", "max_calls_per_turn": 5})
    return exp.run(seed=0, max_rounds=1, out=tmp_path / "runs")


def test_submission_is_the_last_snapshot_workspace(tmp_path):
    run = dry_run(tmp_path)
    sub = pbj.submission_from_run(run.dir)
    assert sub == run.dir / "artifacts" / "submission.tar.gz"
    with tarfile.open(sub, "r:gz") as tar:
        assert sorted(tar.getnames()) == ["compile.sh", "executable"]
        assert tar.extractfile("compile.sh").read().startswith(b"#!/bin/bash")
    assert pbj.run_task(run.dir) == TASK


def test_untouched_workspace_writes_a_marker_and_is_not_graded(tmp_path):
    run = dry_run(tmp_path)
    store = SnapshotStore(run.dir)
    manifest = store.latest()
    blob = pickle.loads(store.blobs.get(manifest.plugins["world"]))
    blob["workspace_tar"] = None
    manifest.plugins.pop("world")
    store.write(manifest, {"world": pickle.dumps(blob)})
    assert pbj.submission_from_run(run.dir) is None
    assert (run.dir / "artifacts" / "submission.untouched").read_bytes() == b""
    with pytest.raises(SpecError, match="nothing of the agent's to grade"):
        pbj.plan_grade(run.dir, git=lambda: (COMMIT, False), environ={"HF_TOKEN": "h"})


def gplan(run, **kw):
    kw.setdefault("environ", {"HF_TOKEN": "h"})
    return pbj.plan_grade(run.dir, now=NOW, git=lambda: (COMMIT, False), **kw)


def test_grade_plan_command_volume_secret_timeout(tmp_path):
    pytest.importorskip("programbench")
    run = dry_run(tmp_path)
    p = gplan(run)
    assert (p.task, p.image, p.flavor, p.branches) == (TASK, IMAGE, "cpu-upgrade", 14)
    assert p.bound_s == 1800 + 900 + 14 * 3600 and p.timeout_s == 12 * 3600
    assert p.tag == f"grade-{run.id.replace('__', '-')}-0123456-20261003-040000"
    cmd = jl.hf_command(p)
    joined = shlex.join(cmd)
    assert ("hf jobs run --detach --flavor cpu-upgrade --timeout 12h --secrets HF_TOKEN "
            f"--name swarmlab-{p.tag}") in joined
    assert cmd[cmd.index("-v") + 1] == \
        "hf://datasets/programbench/ProgramBench-Tests:/root/programbench-tests:ro"
    e = dict(cmd[i + 1].split("=", 1) for i, c in enumerate(cmd) if c == "-e")
    assert e["SWARMLAB_MODE"] == "grade" and e["SWARMLAB_GRADE_TASK"] == TASK
    assert e["SWARMLAB_GRADE_TESTS"] == "/root/programbench-tests"
    assert e["SWARMLAB_STAGE"] == f"hf://buckets/cmpatino/swarmlab-runs/jobs/{p.tag}"
    assert e["SWARMLAB_EXTRAS"] == "programbench" and e["SWARMLAB_IMAGE"] == IMAGE
    assert cmd[-4] == IMAGE and "UV=$(command -v uv)" in cmd[-1]
    text = pbj.describe_grade(p)
    assert "evaluator bound 14.8 h" in text and "capped at 12h" in text
    assert "mounted read-only at /root/programbench-tests (grade job only)" in text
    assert gplan(run, timeout="2h", flavor="cpu-xl").timeout_s == 7200
    assert "below the evaluator's bound" in " ".join(gplan(run, timeout="2h").notes)
    assert gplan(run, environ={}).missing_secrets == ["HF_TOKEN"]


def test_grade_plan_without_the_package_uses_the_cap(tmp_path, monkeypatch):
    run = dry_run(tmp_path)
    monkeypatch.setattr(pbj, "active_branches", lambda task: None)
    p = gplan(run)
    assert p.timeout_s == 12 * 3600 and "branch count is unknown" in p.notes[0]


def test_grade_plan_refusals(tmp_path):
    pytest.importorskip("programbench")
    run = dry_run(tmp_path)
    with pytest.raises(SpecError, match="--bucket must look like"):
        gplan(run, bucket="s3://x")
    meta = json.loads((run.dir / "run.json").read_text())
    (run.dir / "run.json").write_text(json.dumps({**meta, "status": "running"}))
    with pytest.raises(SpecError, match="has not ended"):
        gplan(run)


def test_submit_grade_stages_submission_and_records_the_job(tmp_path):
    pytest.importorskip("programbench")
    run = dry_run(tmp_path)
    p = gplan(run)
    hf = FakeHF()
    row = pbj.submit_grade(p, runner=hf, build=None, ledger=tmp_path / "jobs.jsonl")
    assert [c[1:3] for c in hf.calls] == [["buckets", "sync"], ["jobs", "run"], ["buckets", "cp"]]
    assert hf.staged == ["bootstrap.sh", "plan.json", "submission.tar.gz"]
    assert hf.calls[0][4] == p.stage_remote and hf.calls[1] == jl.hf_command(p)
    job = json.loads((run.dir / "artifacts" / "grade_job.json").read_text())
    assert job["job_id"] == row["job_id"] == "6abd02b3031314b696344afb"
    assert job["kind"] == "grade" and job["stage_remote"] == p.stage_remote
    assert job["secrets"] == ["HF_TOKEN"] and job["mode"] == "grade"
    with pytest.raises(SpecError, match="not launching: HF_TOKEN"):
        pbj.submit_grade(gplan(run, environ={}), runner=FakeHF(), build=None)


# ---- the score write-back -----------------------------------------------------------------------
def grade_doc(task=TASK, **score):
    sc = {"score": 0.75, "passed": 3, "total": 4, "resolved": False, "near_resolved": False,
          "branches": 14, "compile_ok": True, "error_code": None, **score}
    return {"task": task, "image": IMAGE, "programbench_version": "1.2.5", "cpus": 8.0,
            "pytest_workers": 8, "ended_at": "2026-10-15T12:00:00Z", "deviation": g.DEVIATION,
            "score": sc, "evaluation": {"test_results": []}}


def test_write_grade_then_load_report_and_export(tmp_path):
    run = dry_run(tmp_path)
    before = dict(run.score)
    gj = tmp_path / "grade.json"
    gj.write_text(json.dumps(grade_doc()))
    score = pbj.write_grade(run.dir, gj)
    expected = {**before, "graded": True, "score": 0.75, "passed": 3, "total": 4,
                "resolved": False, "near_resolved": False, "branches": 14, "compile_ok": True}
    assert score == expected
    meta = json.loads((run.dir / "run.json").read_text())
    assert meta["score"] == expected and meta["score_world"] == before
    assert meta["grade"]["programbench_version"] == "1.2.5" and meta["grade"]["cpus"] == 8.0
    assert (run.dir / "artifacts" / "grade.json").exists()
    loaded = Run.load(run.dir)  # replay checks the world's own score against score_world
    assert loaded.score == expected and loaded.summary()["score"]["graded"] is True
    # a re-grade replaces the graded keys, the world's score stays the runner's
    gj.write_text(json.dumps(grade_doc(score=1.0, passed=4, resolved=True, near_resolved=True)))
    again = pbj.write_grade(run.dir, gj)
    assert again["score"] == 1.0 and again["resolved"] is True
    assert json.loads((run.dir / "run.json").read_text())["score_world"] == before
    report = write_report(run.dir.parent, None, "t", include_fake=True)
    final = report.split("## Final score", 1)[1]
    header = final.strip().splitlines()[0]
    for key in ("graded", "score", "passed", "total", "resolved", "near_resolved", "branches",
                "compile_ok", "submitted", "steps"):
        assert f" {key} " in header, key
    assert "pb-dry__" in final and "skipped (not ended / unreadable): none" in report
    exported = json.loads((run.export(tmp_path / "exp") / "run.json").read_text())
    assert exported["score"]["graded"] is True and exported["score"]["passed"] == 4


def test_write_grade_refuses_another_task_and_partial_scores(tmp_path):
    run = dry_run(tmp_path)
    gj = tmp_path / "grade.json"
    gj.write_text(json.dumps(grade_doc(task="tomnomnom__gron.88a6234")))
    with pytest.raises(SpecError, match="grades task 'tomnomnom__gron"):
        pbj.write_grade(run.dir, gj)
    bad = grade_doc()
    del bad["score"]["compile_ok"]
    gj.write_text(json.dumps(bad))
    with pytest.raises(SpecError, match="lacks"):
        pbj.write_grade(run.dir, gj)


def test_fetch_grade_downloads_from_the_recorded_stage(tmp_path):
    run = dry_run(tmp_path)
    with pytest.raises(FileNotFoundError, match="grade_job.json does not exist"):
        pbj.fetch_grade(run.dir)
    (run.dir / "artifacts").mkdir(exist_ok=True)
    (run.dir / "artifacts" / "grade_job.json").write_text(json.dumps(
        {"job_id": "j" * 24, "tag": "gt", "stage_remote": "hf://buckets/ns/b/jobs/gt"}))
    src = tmp_path / "remote-grade.json"
    src.write_text(json.dumps(grade_doc()))
    calls = []

    def runner(args):
        calls.append(args)
        shutil.copy(src, args[4])
        return subprocess.CompletedProcess(args, 0, "", "")

    dest = pbj.fetch_grade(run.dir, runner=runner)
    assert calls == [["hf", "buckets", "cp", "hf://buckets/ns/b/jobs/gt/grade.json", str(dest)]]
    assert pbj.write_grade(run.dir, dest)["score"] == 0.75
    assert json.loads((run.dir / "run.json").read_text())["grade"]["job_id"] == "j" * 24

    def failing(args):
        return subprocess.CompletedProcess(args, 1, "", "404")

    (run.dir / "artifacts" / "grade.json").unlink()
    with pytest.raises(RuntimeError, match="no grade.json"):
        pbj.fetch_grade(run.dir, runner=failing)


def test_wait_grade_polls_until_terminal():
    stages = iter(["SCHEDULING", "RUNNING", "RUNNING", "COMPLETED"])
    seen = []
    assert pbj.wait_grade("j", status_fn=lambda j: {"stage": next(stages)}, sleep=lambda s: None,
                          say=seen.append) == "COMPLETED"
    assert seen == ["grade job j: SCHEDULING", "grade job j: RUNNING", "grade job j: COMPLETED"]


# ---- CLI ----------------------------------------------------------------------------------------
def test_cli_grade_prints_plan_and_fetch_grade_writes_score(tmp_path, monkeypatch):
    pytest.importorskip("programbench")
    run = dry_run(tmp_path)
    monkeypatch.setenv("HF_TOKEN", "h")
    monkeypatch.setattr(pbj, "submit_grade", lambda *a, **k: (_ for _ in ()).throw(AssertionError))
    res = cli.invoke(app, ["programbench", "grade", str(run.dir), "--stage-dir",
                           str(tmp_path / "st")], catch_exceptions=False)
    assert res.exit_code == 0, res.output
    assert "hf jobs run --detach --flavor cpu-upgrade --timeout 12h --secrets HF_TOKEN" in res.output
    assert "-v hf://datasets/programbench/ProgramBench-Tests:/root/programbench-tests:ro" in res.output
    assert "not launched" in res.output
    assert (tmp_path / "st" / "submission.tar.gz").read_bytes() == \
        (run.dir / "artifacts" / "submission.tar.gz").read_bytes()
    assert {p.name for p in (tmp_path / "st").iterdir()} >= {"bootstrap.sh", "plan.json",
                                                             "submission.tar.gz"}
    bad = cli.invoke(app, ["programbench", "grade", str(run.dir), "--wait"])
    assert bad.exit_code == 2 and "--wait needs --launch" in bad.stderr
    gj = tmp_path / "g.json"
    gj.write_text(json.dumps(grade_doc()))
    res = cli.invoke(app, ["programbench", "fetch-grade", str(run.dir), "--grade-json", str(gj),
                           "--json"])
    assert res.exit_code == 0, res.output
    data = json.loads(res.stdout)
    assert data["score"]["graded"] is True and data["score"]["score"] == 0.75
    assert Run.load(run.dir).score["passed"] == 3


def test_cli_grade_launch_wait_fetches_and_writes(tmp_path, monkeypatch):
    pytest.importorskip("programbench")
    run = dry_run(tmp_path)
    monkeypatch.setenv("HF_TOKEN", "h")
    monkeypatch.setattr(jl, "checkout_identity", lambda: (COMMIT, False))

    def fake_submit(plan, **kw):
        (Path(plan.run_dir) / "artifacts" / "grade_job.json").write_text(json.dumps(
            {"job_id": "d" * 24, "stage_remote": plan.stage_remote, "tag": plan.tag}))
        return {"job_id": "d" * 24, "url": "u"}

    def fake_fetch(run_dir, **kw):
        p = Path(run_dir) / "artifacts" / "grade.json"
        p.write_text(json.dumps(grade_doc(score=0.5, passed=2)))
        return p

    monkeypatch.setattr(pbj, "submit_grade", fake_submit)
    monkeypatch.setattr(pbj, "wait_grade", lambda job_id, **kw: "COMPLETED")
    monkeypatch.setattr(pbj, "fetch_grade", fake_fetch)
    res = cli.invoke(app, ["programbench", "grade", str(run.dir), "--launch", "--wait", "--json"])
    assert res.exit_code == 0, res.output
    data = json.loads(res.stdout)
    assert data["launched"] and data["job_stage"] == "COMPLETED" and data["score"]["score"] == 0.5
    assert Run.load(run.dir).score["graded"] is True


# ---- bootstrap.sh, grade branch -----------------------------------------------------------------
STUB_UV_GRADE = r'''#!/usr/bin/env bash
echo "uv $*" >> "$STUB_LOG"
if [ "$1" = venv ]; then
  dir="${@: -1}"; mkdir -p "$dir/bin"
  ln -sf "$STUB_DIR/python" "$dir/bin/python"; ln -sf "$STUB_DIR/hf" "$dir/bin/hf"
fi
exit 0
'''

STUB_PY = r'''#!/usr/bin/env bash
# the venv's python: `-m swarmlab.jobs.grade` is logged and answered with a grade.json
echo "python $*" >> "$STUB_LOG"
if [ "$1 $2" = "-m swarmlab.jobs.grade" ]; then
  out=""; prev=""
  for a in "$@"; do [ "$prev" = "--out" ] && out="$a"; prev="$a"; done
  [ -n "$FAKE_GRADE_FAIL" ] && { echo "GRADE ERROR: boom" >&2; exit 1; }
  printf '{"task": "%s", "score": {"score": 1.0}}' "$SWARMLAB_GRADE_TASK" > "$out"
  echo "GRADE ok"
  exit 0
fi
exec "$REAL_PYTHON" "$@"
'''


def _run_grade_bootstrap(tmp_path, **extra):
    stub = tmp_path / "stubs"
    stub.mkdir()
    for name, body in (("hf", STUB_HF), ("uv", STUB_UV_GRADE), ("python", STUB_PY)):
        (stub / name).write_text(body)
        (stub / name).chmod(0o755)
    stage = tmp_path / "stage"
    stage.mkdir()
    (stage / "swarmlab-0.0.1-py3-none-any.whl").write_text("not a real wheel")
    (stage / "submission.tar.gz").write_bytes(b"x")
    env = {"PATH": os.environ["PATH"], "HOME": str(tmp_path), "STUB_LOG": str(tmp_path / "log"),
           "STUB_DIR": str(stub), "FAKE_BUCKET": str(tmp_path / "bucket"),
           "REAL_PYTHON": sys.executable, "SWARMLAB_UV": str(stub / "uv"),
           "SWARMLAB_STAGE_DIR": str(stage), "SWARMLAB_MODE": "grade", "SWARMLAB_TAG": "g1",
           "SWARMLAB_STAGE": "hf://buckets/ns/b/jobs/g1", "SWARMLAB_GRADE_TASK": TASK,
           "SWARMLAB_GRADE_TESTS": "/root/programbench-tests", "SWARMLAB_IMAGE": IMAGE,
           "SWARMLAB_EXTRAS": "programbench", "SWARMLAB_SECRET_NAMES": "HF_TOKEN",
           "JOB_ID": "e" * 24, "TMPDIR": str(tmp_path), "SWARMLAB_JOB_ROOT": str(tmp_path / "root"),
           **extra}
    proc = subprocess.run(["bash", str(REPO / "swarmlab" / "jobs" / "bootstrap.sh")], env=env,
                          capture_output=True, text=True, timeout=120, cwd=tmp_path, check=False)
    return proc, tmp_path / "bucket" / "jobs" / "g1", (tmp_path / "log").read_text()


def test_bootstrap_grade_mode_runs_the_grader_and_uploads_grade_json(tmp_path):
    proc, stage, calls = _run_grade_bootstrap(tmp_path)
    assert proc.returncode == 0, proc.stdout + proc.stderr
    assert (f"python -m swarmlab.jobs.grade --task {TASK} --submission "
            f"{tmp_path}/stage/submission.tar.gz --tests /root/programbench-tests --image {IMAGE} "
            f"--out {tmp_path}/root/tmp/grade.json --sweep-on-timeout") in calls
    assert "swarmlab[programbench] @ file://" in calls and "uv python install 3.12" in calls
    assert "validate" not in calls and "vllm" not in calls.replace("swarmlab", "")
    assert json.loads((stage / "grade.json").read_text())["score"] == {"score": 1.0}
    manifest = json.loads((stage / "job.json").read_text())
    assert manifest["mode"] == "grade" and manifest["status"] == "completed"
    assert manifest["grade"] == {"task": TASK, "score": {"score": 1.0}}
    assert manifest["secrets"] == ["HF_TOKEN"] and manifest["vllm"] is None


def test_bootstrap_grade_failure_exits_14(tmp_path):
    proc, stage, _ = _run_grade_bootstrap(tmp_path, FAKE_GRADE_FAIL="1")
    assert proc.returncode == 14, proc.stdout + proc.stderr
    assert "grading failed" in proc.stdout and not (stage / "grade.json").exists()
    manifest = json.loads((stage / "job.json").read_text())
    assert manifest["status"] == "failed" and manifest["exit_code"] == 14
    assert "GRADE ERROR: boom" in (stage / "logs" / "grade.log").read_text()



def test_sweep_spares_ancestors_and_processes_older_than_the_grade(tmp_path):
    old = subprocess.Popen(["sleep", "60"])
    try:
        protected = g._pids()
        new = subprocess.Popen(["sleep", "60"])
        killed = []
        real_kill = os.kill

        def fake_kill(pid, sig):  # only ever really kill our own new child
            killed.append(pid)
            if pid == new.pid:
                real_kill(pid, sig)

        import swarmlab.jobs.grade as mod

        orig = mod.os.kill
        mod.os.kill = fake_kill
        try:
            g.sweep_processes(protected)
        finally:
            mod.os.kill = orig
        assert new.pid in killed and old.pid not in killed and os.getpid() not in killed
        assert 1 not in killed and os.getppid() not in killed
        assert new.wait(timeout=5) == -9
    finally:
        old.kill()
