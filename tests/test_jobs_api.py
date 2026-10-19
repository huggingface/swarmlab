"""`swarmlab job run` without vLLM (API mode): the plan, the command, the secrets, the
ProgramBench defaults and refusals, and bootstrap.sh's api branch against stub `uv` and `hf`.
Offline; nothing is launched."""
import json
import os
import shlex
import shutil
import subprocess
import sys
from pathlib import Path

import pytest
import yaml
from typer.testing import CliRunner

from swarmlab import Run
from swarmlab.cli import app
from swarmlab.jobs import launch as jl
from swarmlab.spec import SpecError

from .test_jobs import COMMIT, NOW, REPO, STUB_HF, STUB_UV, FakeHF, vllm_spec

TASK = "abishekvashok__cmatrix.5c082c6"
IMAGE = "programbench/abishekvashok_1776_cmatrix.5c082c6:task_cleanroom_v6"
KEYS = {"HF_TOKEN": "hf_x", "ANTHROPIC_API_KEY": "sk-x"}
cli = CliRunner()


def solo_doc(**world):
    """The guide's solo YAML, run inside the task image (backend: local), on Haiku 5.5."""
    return {
        "name": "programbench-solo", "seeds": [0],
        "budget": {"soft_usd": 0, "hard_usd": 5.0},
        "options": {"commit": "immediate", "max_rounds": 1, "max_calls_per_turn": 1000},
        "roles": {"solo": {"tools": ["bash"]}},
        "arms": {"solo": {
            "world": {"type": "programbench",
                      "params": {"task": TASK, "backend": "local", **world}},
            "participants": [{"type": "llm", "count": 1, "role": "solo",
                              "params": {"model": "anthropic:claude-haiku-5-5",
                                         "max_tokens": 8192}}]}},
    }


def write(tmp_path, doc, name="spec.yaml"):
    p = tmp_path / name
    p.write_text(yaml.safe_dump(doc, sort_keys=False))
    return p


def plan(spec, environ=KEYS, **kw):
    return jl.plan_job(spec, now=NOW, git=lambda: (COMMIT, False), environ=environ, **kw)


def env_of(cmd):
    return dict(cmd[i + 1].split("=", 1) for i, c in enumerate(cmd) if c == "-e")


def test_programbench_solo_plan_is_api_mode(tmp_path):
    p = plan(write(tmp_path, solo_doc()))
    assert p.mode == "api" and p.model is None
    assert (p.image, p.flavor, p.timeout_s) == (IMAGE, "cpu-upgrade", 7 * 3600)
    assert p.secrets == ["HF_TOKEN", "ANTHROPIC_API_KEY"] and p.missing_secrets == []
    assert p.extras == ["anthropic", "programbench"]
    assert p.programbench["task"] == TASK and p.programbench["lockdown"] is True
    assert "cas-server-direct.xethub.hf.co" in p.programbench["pin_hosts"]
    cmd = jl.hf_command(p)
    joined = shlex.join(cmd)
    assert cmd[:4] == ["hf", "jobs", "run", "--detach"]
    assert "--flavor cpu-upgrade --timeout 7h --secrets HF_TOKEN --secrets ANTHROPIC_API_KEY" in joined
    assert "-v" not in cmd  # the hidden tests never reach the agent's container
    env = env_of(cmd)
    assert not any(k.startswith("VLLM_") for k in env)
    assert env["SWARMLAB_MODE"] == "api" and env["SWARMLAB_RUNS"] == "solo:0"
    assert env["SWARMLAB_SECRET_NAMES"] == "HF_TOKEN ANTHROPIC_API_KEY"
    assert env["SWARMLAB_EXTRAS"] == "anthropic,programbench"
    assert env["SWARMLAB_IMAGE"] == IMAGE and "HF_HOME" not in env
    assert "sk-x" not in joined and "hf_x" not in joined  # names only, never values
    assert cmd[-4] == IMAGE and cmd[-3:-1] == ["bash", "-c"]
    entry = cmd[-1]
    assert entry.startswith("mkdir -p /opt/swarmlab/stage && UV=$(command -v uv) || { curl -LsSf "
                            "https://astral.sh/uv/install.sh | env UV_INSTALL_DIR=/opt/swarmlab/uv")
    assert '"$UV" tool run --from \'huggingface_hub>=1.32\' hf buckets sync' in entry
    assert entry.endswith('SWARMLAB_UV="$UV" exec bash /opt/swarmlab/stage/bootstrap.sh')
    subprocess.run(["bash", "-n", "-c", entry], check=True)
    # the job spec is the spec's arms as given: no providers override
    doc = yaml.safe_load(p.job_spec_yaml)
    assert not doc.get("providers") and list(doc["arms"]) == ["solo"]
    d = p.as_dict()
    assert d["vllm_serve_argv"] is None and d["vllm_serve_command"] is None
    assert d["secrets"] == ["HF_TOKEN", "ANTHROPIC_API_KEY"] and d["mode"] == "api"
    text = jl.describe(p)
    assert "mode: api (no vLLM)" in text and "vllm " not in text.split("\n", 3)[3]
    assert "secrets (names; the hf CLI reads each value from this environment): HF_TOKEN " \
           "ANTHROPIC_API_KEY" in text
    assert "locks the job down at its first command" in text
    assert "billed by their providers" in text and "7h" in text


def test_anthropic_key_name_follows_the_launching_environment(tmp_path):
    spec = write(tmp_path, solo_doc())
    assert plan(spec, environ={"HF_TOKEN": "h", "ANTHROPIC_KEY": "k"}).secrets == \
        ["HF_TOKEN", "ANTHROPIC_KEY"]
    both = {"HF_TOKEN": "h", "ANTHROPIC_KEY": "k", "ANTHROPIC_API_KEY": "a"}
    assert plan(spec, environ=both).secrets[1] == "ANTHROPIC_API_KEY"


def test_missing_secret_is_noted_and_submit_refuses(tmp_path):
    p = plan(write(tmp_path, solo_doc()), environ={"HF_TOKEN": "h"})
    assert p.missing_secrets == ["ANTHROPIC_API_KEY"]
    assert "ANTHROPIC_API_KEY not set in this environment" in jl.describe(p)
    hf = FakeHF()
    with pytest.raises(SpecError, match="not launching: ANTHROPIC_API_KEY"):
        jl.submit(p, runner=hf, build=None)
    assert hf.calls == []  # nothing uploaded, nothing submitted


def test_submit_api_plan_records_mode_and_secret_names(tmp_path):
    p = plan(write(tmp_path, solo_doc()))
    hf = FakeHF()
    row = jl.submit(p, runner=hf, build=None, ledger=tmp_path / "jobs.jsonl")
    assert hf.calls[1] == jl.hf_command(p)
    assert row["mode"] == "api" and row["secrets"] == ["HF_TOKEN", "ANTHROPIC_API_KEY"]
    staged = json.loads(json.dumps(p.as_dict()))
    assert "sk-x" not in json.dumps(staged)


def test_a_custom_api_key_env_is_declared_and_never_reaches_commands(tmp_path):
    """A key in a variable without a secret-looking name (`MODEL_AUTH`) is passed by name and
    listed in SWARMLAB_SECRET_NAMES, which the agent's and the grader's filters both honour."""
    from swarmlab.jobs.grade import grade_env
    from swarmlab.world.programbench import command_env

    doc = solo_doc()
    doc["providers"] = {"gw": {"type": "openai_compat", "params": {
        "name": "gw", "base_url": "https://gw.example/v1", "api_key_env": "MODEL_AUTH",
        "pricing": {"m": [1.0, 2.0, 0.0]}}}}
    doc["arms"]["solo"]["participants"][0]["params"]["model"] = "gw:m"
    p = plan(write(tmp_path, doc), environ={"HF_TOKEN": "h", "MODEL_AUTH": "dummy-value"})
    assert p.secrets == ["HF_TOKEN", "MODEL_AUTH"]
    cmd = jl.hf_command(p)
    assert "--secrets MODEL_AUTH" in shlex.join(cmd)
    job_env = {**env_of(cmd), "MODEL_AUTH": "dummy-value", "HF_TOKEN": "h"}
    assert job_env["SWARMLAB_SECRET_NAMES"] == "HF_TOKEN MODEL_AUTH"
    for filtered in (command_env(job_env), grade_env(job_env)):
        assert "MODEL_AUTH" not in filtered and "dummy-value" not in filtered.values()


def test_programbench_refusals(tmp_path):
    with pytest.raises(SpecError, match="backend is 'docker'.*cannot run Docker"):
        plan(write(tmp_path, solo_doc(backend="docker")))
    doc = solo_doc()
    other = json.loads(json.dumps(doc["arms"]["solo"]))
    other["world"]["params"]["task"] = "tomnomnom__gron.88a6234"
    doc["arms"]["other"] = other
    with pytest.raises(SpecError, match="different programbench tasks"):
        plan(write(tmp_path, doc))
    assert plan(write(tmp_path, doc), arms=["other"]).programbench["task"] == \
        "tomnomnom__gron.88a6234"
    with pytest.raises(SpecError, match="2 programbench runs in one job"):
        plan(write(tmp_path, solo_doc()), seeds=[0, 1])
    # a programbench world cannot run next to vLLM (the task image has no GPU stack)
    vdoc = solo_doc()
    vdoc["arms"]["solo"]["participants"][0]["params"]["model"] = "vllm:Qwen/Qwen3.5-9B"
    with pytest.raises(SpecError, match="programbench world runs in an API-model job"):
        plan(write(tmp_path, vdoc), model="Qwen/Qwen3.5-9B")


def test_api_mode_explicit_image_flavor_timeout_and_notes(tmp_path):
    p = plan(write(tmp_path, solo_doc(lockdown=False, wall_limit_s=600)), image="my/img:1",
             flavor="cpu-xl")
    assert (p.image, p.flavor) == ("my/img:1", "cpu-xl")
    assert p.timeout_s == 600 + 3600  # wall limit + slack, rounded to 10 min
    assert any("not the task image" in n for n in p.notes)
    assert any("lockdown: false" in n for n in p.notes)
    assert p.rate_usd_per_hour == 1.00
    assert plan(write(tmp_path, solo_doc()), timeout="8h").timeout_s == 8 * 3600


def test_non_programbench_api_spec_uses_the_uv_image_and_round_estimate(tmp_path):
    spec = vllm_spec(tmp_path, model="anthropic:claude-haiku-5-5", n=2)
    p = plan(spec)
    assert p.mode == "api" and p.image == jl.DEFAULT_IMAGE and p.flavor == "cpu-upgrade"
    assert p.programbench is None and p.extras == ["anthropic"]
    assert p.estimate_detail["arms"]["bc"]["per_round_s"] == 34.0
    assert p.timeout_s == 1200  # 1.5 x (600 setup + 2 seeds x 2 rounds x 34 s), up to 10 min
    # --model asks for vLLM: then every model must be vllm:<model>, as before
    with pytest.raises(SpecError, match="only vllm:"):
        plan(spec, model="Qwen/Qwen3.5-9B")
    # a vllm: model in the arms without --model
    with pytest.raises(SpecError, match="pass --model"):
        plan(vllm_spec(tmp_path))


def test_vllm_plans_keep_their_shape(tmp_path):
    p = jl.plan_job(vllm_spec(tmp_path), model="Qwen/Qwen3.5-9B", now=NOW,
                    git=lambda: (COMMIT, False))
    assert p.mode == "vllm" and p.flavor == "a100-large" and p.image == jl.DEFAULT_IMAGE
    assert p.secrets == ["HF_TOKEN"] and p.missing_secrets == [] and p.extras == []
    cmd = jl.hf_command(p)
    assert cmd.count("--secrets") == 1 and env_of(cmd)["VLLM_MODEL"] == "Qwen/Qwen3.5-9B"
    assert "SWARMLAB_MODE" not in env_of(cmd)
    assert cmd[-1].startswith("mkdir -p /opt/swarmlab/stage && uv tool run")


def test_cli_job_run_without_model_prints_api_plan(tmp_path, monkeypatch):
    monkeypatch.setenv("HF_TOKEN", "h")
    monkeypatch.setenv("ANTHROPIC_API_KEY", "a")
    monkeypatch.setattr(jl, "submit", lambda *a, **k: (_ for _ in ()).throw(AssertionError))
    res = cli.invoke(app, ["job", "run", str(write(tmp_path, solo_doc())), "--stage-dir",
                           str(tmp_path / "st")], catch_exceptions=False)
    assert res.exit_code == 0, res.output
    assert "hf jobs run --detach --flavor cpu-upgrade --timeout 7h --secrets HF_TOKEN " \
           "--secrets ANTHROPIC_API_KEY" in res.output
    assert "mode: api" in res.output and "not launched" in res.output
    staged = json.loads((tmp_path / "st" / "plan.json").read_text())
    assert staged["mode"] == "api" and staged["vllm_serve_argv"] is None
    bad = cli.invoke(app, ["job", "run", str(write(tmp_path, solo_doc(backend="docker")))])
    assert bad.exit_code == 2 and "cannot run Docker" in bad.stderr


# ---- bootstrap.sh, api branch -------------------------------------------------------------------
@pytest.mark.skipif(shutil.which("curl") is None, reason="needs curl")
def test_bootstrap_api_mode_skips_vllm(tmp_path):
    stub = tmp_path / "stubs"
    stub.mkdir()
    for name, body in (("hf", STUB_HF), ("uv", STUB_UV)):
        (stub / name).write_text(body)
        (stub / name).chmod(0o755)
    stage = tmp_path / "stage"
    stage.mkdir()
    (stage / "swarmlab-0.0.1-py3-none-any.whl").write_text("not a real wheel")
    (stage / "spec.yaml").write_text(yaml.safe_dump({
        "name": "boot", "options": {"max_rounds": 2},
        "arms": {"scripted": {"world": "flaggame",
                              "participants": [{"type": "evidence_aggregator", "count": 3}]}}}))
    real_python = Path(sys.executable)
    root = tmp_path / "root"
    env = {
        "PATH": os.environ["PATH"], "HOME": str(tmp_path),  # no uv on PATH: SWARMLAB_UV is used
        "STUB_LOG": str(tmp_path / "stub.log"), "STUB_DIR": str(stub),
        "FAKE_BUCKET": str(tmp_path / "bucket"), "REAL_PYTHON": str(real_python),
        "REAL_SWARMLAB": str(real_python.parent / "swarmlab"), "SWARMLAB_UV": str(stub / "uv"),
        "SWARMLAB_STAGE_DIR": str(stage), "SWARMLAB_TAG": "t1", "SWARMLAB_MODE": "api",
        "SWARMLAB_STAGE": "hf://buckets/ns/b/jobs/t1",
        "SWARMLAB_RUNS_REMOTE": "hf://buckets/ns/b/runs/t1", "SWARMLAB_RUNS": "scripted:1",
        "SWARMLAB_GIT_COMMIT": COMMIT, "SWARMLAB_GIT_DIRTY": "0", "SWARMLAB_SYNC_EVERY": "1",
        "SWARMLAB_EXTRAS": "anthropic,programbench",
        "SWARMLAB_SECRET_NAMES": "HF_TOKEN ANTHROPIC_API_KEY", "JOB_ID": "c" * 24,
        "TMPDIR": str(tmp_path), "SWARMLAB_JOB_ROOT": str(root),
    }
    env["PATH"] = os.pathsep.join(p for p in env["PATH"].split(os.pathsep)
                                  if not (Path(p) / "uv").exists())
    proc = subprocess.run(["bash", str(REPO / "swarmlab" / "jobs" / "bootstrap.sh")], env=env,
                          capture_output=True, text=True, timeout=180, cwd=tmp_path, check=False)
    assert proc.returncode == 0, proc.stdout + proc.stderr
    calls = (tmp_path / "stub.log").read_text()
    assert "vllm" not in calls.replace("swarmlab", "")  # no vLLM install, serve or probe
    assert "uv python install 3.12" in calls
    assert f"swarmlab[anthropic,programbench] @ file://{stage}/swarmlab-0.0.1-py3-none-any.whl" in calls
    assert "mode=api" in proc.stdout and "vLLM ready" not in proc.stdout
    assert oct(root.stat().st_mode & 0o777) == "0o700"
    assert (root / "tmp" / "joblogs").is_dir()  # scratch under the root-only dir
    bucket = tmp_path / "bucket"
    r = Run.load(bucket / "runs" / "t1" / "boot__scripted__s1")
    assert r.status == "ended" and r.meta["git_commit"] == COMMIT
    manifest = json.loads((bucket / "jobs" / "t1" / "job.json").read_text())
    assert manifest["mode"] == "api" and manifest["vllm"] is None
    assert manifest["secrets"] == ["HF_TOKEN", "ANTHROPIC_API_KEY"]
    assert manifest["status"] == "completed" and manifest["runs_done"] == ["scripted:1"]


def test_mini_swe_agent_participant_needs_no_extra(tmp_path):
    pytest.importorskip("swarmlab.participants.minisweagent")  # the participant PR (#25)
    doc = solo_doc()
    arm = doc["arms"]["solo"]
    arm["participants"] = [{"type": "mini_swe_agent", "count": 1, "role": "solo",
                            "params": {"model": "anthropic:claude-haiku-4-5"}}]
    doc["options"]["max_calls_per_turn"] = 3000
    p = plan(write(tmp_path, doc))
    assert p.extras == ["anthropic", "programbench"]  # the participant's loop is vendored


def test_api_mode_parallel_schedules_runs_and_refuses_programbench(tmp_path):
    spec = vllm_spec(tmp_path, model="anthropic:claude-haiku-5-5", n=2)
    seq = plan(spec, per_round_s=100, setup_s=60)
    par = plan(spec, per_round_s=100, setup_s=60, parallel=2)
    assert par.mode == "api" and par.parallel == 2 and par.estimate_detail["parallel"] == 2
    assert seq.estimate_s == 60 + 2 * 200 and par.estimate_s == 60 + 200
    assert "SWARMLAB_PARALLEL" not in env_of(jl.hf_command(seq))
    assert env_of(jl.hf_command(par))["SWARMLAB_PARALLEL"] == "2"
    assert "2 runs at once" in jl.describe(par)
    with pytest.raises(SpecError, match="--parallel 2 with a programbench run"):
        plan(write(tmp_path, solo_doc()), parallel=2)
    assert plan(write(tmp_path, solo_doc()), parallel=1).timeout_s == 7 * 3600


EXAMPLE_ASYNC = REPO / "examples" / "programbench_team_async.yaml"


def test_async_programbench_team_example_plans_one_api_job(tmp_path):
    p = plan(EXAMPLE_ASYNC)
    assert p.mode == "api" and p.image == IMAGE and p.flavor == "cpu-upgrade"
    assert [r for r, _ in p.runs] == ["team"]                     # one run per job holds
    assert p.timeout_s == 7 * 3600                                # the 6 h wall limit + 1 h
    detail = p.estimate_detail["arms"]["team"]
    # rounds_per_agent 30 x 3 agents resolves to 90 ticks, bounded by the world's wall limit
    assert (detail["agents"], detail["rounds"], detail["unit"]) == (3, 90, "tick")
    text = jl.describe(p)
    assert "team: N=3, up to 90 ticks (async), bounded by the world's wall limit 6h/run" in text
    assert "an upper bound: the wall limit" in text
    # the round-based solo arm prints as before
    solo = jl.describe(plan(write(tmp_path, solo_doc())))
    assert "solo: N=1, bounded by the world's wall limit 6h/run" in solo
