"""`swarmlab job serve|endpoint|stop` and `job run --endpoint`: a model served by its own HF Job
(swarmlab/jobs/serve.py). Offline: `hf` and HTTP are stubbed, serve.sh runs against a stub server;
nothing is launched."""
import json
import os
import shlex
import shutil
import signal
import subprocess
import sys
import time

import pytest
from typer.testing import CliRunner

from swarmlab import Experiment
from swarmlab.cli import app
from swarmlab.jobs import launch as jl
from swarmlab.jobs import profiles as jp
from swarmlab.jobs import serve as js
from swarmlab.spec import SpecError

from .test_jobs import COMMIT, NOW, REPO, FakeHF, _free_port, vllm_spec
from .test_jobs_api import TASK, env_of, solo_doc, write

cli = CliRunner()
JOB = "0123456789abcdef01234567"
URL = f"https://{JOB}--30000.hf.jobs"
FLINT_EXAMPLE = REPO / "examples" / "programbench_team_async_flint.yaml"
SERVED = {"type": "openai_compat", "params": {
    "name": "served", "base_url_env": "SWARMLAB_ENDPOINT_URL", "api_key_env": "HF_TOKEN",
    "history_reasoning": True, "self_hosted": True, "pricing": {"flint": [0, 0, 0]}}}


def serve_plan(model="RecursiveSI/flint", **kw):
    return js.plan_serve(model, now=NOW, git=lambda: (COMMIT, False),
                         environ={"HF_TOKEN": "hf_x"}, **kw)


def flint_doc(model="served:flint", **world):
    doc = solo_doc(**world)
    doc["providers"] = {"served": SERVED}
    doc["arms"]["solo"]["participants"][0]["params"]["model"] = model
    return doc


def run_plan(spec, endpoint=None, **kw):
    kw.setdefault("environ", {"HF_TOKEN": "hf_x"})
    return jl.plan_job(spec, now=NOW, git=lambda: (COMMIT, False), endpoint=endpoint, **kw)


def endpoint(url=URL, name="flint", **kw):
    return {"url": url, "served_model_name": name, "job_id": JOB, **kw}


# ---- the serve plan ----------------------------------------------------------------------------
def test_flint_kit_profile_table():
    k = jp.KIT_PROFILES["flint"]
    assert (k.model, k.engine, k.port, k.served_model_name) == ("RecursiveSI/flint", "sglang",
                                                               30000, "flint")
    assert k.image == "lmsysorg/sglang:v0.5.16-cu130" and k.image_digest.startswith("sha256:7b6a35df")
    assert k.serve_scripts == {"h200": "serve_h200.sh"} and k.install == "install.sh"
    assert k.ready_paths == ("/health_generate", "/v1/models") and k.probe_extra == {}
    assert k.setup_env["PIP_BREAK_SYSTEM_PACKAGES"] == "1" and k.min_gpu_gb == 141
    assert jp.select_kit_profile("RecursiveSI/flint", "auto") is k
    assert jp.select_kit_profile("recursivesi/FLINT", "flint") is k


def test_flint_serve_plan_command(tmp_path):
    p = serve_plan(profile="flint", flavor="h200")
    cmd = js.hf_command(p)
    joined = shlex.join(cmd)
    assert cmd[:4] == ["hf", "jobs", "run", "--detach"]
    assert "--flavor h200 --timeout 6h --secrets HF_TOKEN" in joined
    assert "--expose 30000" in joined and "--expose-public" not in joined
    assert "hf_x" not in joined   # the secret by name only
    env = env_of(cmd)
    assert env["SWARMLAB_MODE"] == "serve" and env["SWARMLAB_SERVE_NAME"] == "flint"
    assert env["SWARMLAB_SERVE_PORT"] == "30000"
    assert env["SWARMLAB_SERVE_READY_PATHS"] == "/health_generate /v1/models"
    assert env["PIP_BREAK_SYSTEM_PACKAGES"] == "1"
    assert cmd[-4] == "lmsysorg/sglang:v0.5.16-cu130" and cmd[-3:-1] == ["bash", "-c"]
    assert cmd[-1].endswith("exec bash /opt/swarmlab/stage/serve.sh")
    assert "\"$UV\" tool run" in cmd[-1]   # the image's huggingface_hub is never upgraded
    assert p.setup_argvs[1] == ["bash", "/opt/model/sglang/install.sh", "/opt/model/sglang"]
    assert "allow_patterns=['sglang/*']" in p.setup_argvs[0][2]
    assert "snapshot_download('RecursiveSI/flint', local_dir='/opt/model')" in p.setup_argvs[2][2]
    assert p.serve_argv == ["env", "MODEL=/opt/model", "PORT=30000", "bash",
                            "/opt/model/sglang/serve_h200.sh"]
    assert p.rate_usd_per_hour == 5.0 and p.missing_secrets == []
    d = p.as_dict()
    assert d["serve_argv"] == p.serve_argv and d["expose"] == [30000]
    assert d["image_digest"].startswith("sha256:") and json.dumps(d)
    text = js.describe_serve(p)
    assert "$5.00/h while it runs, $30.00 if it runs to the timeout" in text
    assert "swarmlab job stop" in text
    rev = serve_plan(revision="abc123")
    assert all("revision='abc123'" in a[2] for a in (rev.setup_argvs[0], rev.setup_argvs[2]))
    st = js.stage_serve(p, tmp_path / "st")
    assert sorted(x.name for x in st.iterdir()) == ["plan.json", "serve.sh"]


@pytest.mark.parametrize("flavor", ["a100-large", "h200x2", "l40sx1"])
def test_flint_refuses_other_flavors(flavor):
    with pytest.raises(SpecError, match=r"no serve script for flavor .*141 GB.*--flavor h200"):
        serve_plan(flavor=flavor)


def test_sglang_needs_a_kit_profile():
    with pytest.raises(SpecError, match="kit profile only"):
        serve_plan("Qwen/Qwen3.5-9B")
    with pytest.raises(SpecError, match="serves RecursiveSI/flint"):
        serve_plan("Qwen/Qwen3.5-9B", profile="flint")
    with pytest.raises(SpecError, match="no such kit profile"):
        serve_plan(profile="nope")
    with pytest.raises(SpecError, match="--engine"):
        serve_plan(engine="tgi")


def test_vllm_engine_reuses_the_in_job_serve_argv():
    p = serve_plan("Qwen/Qwen3.5-9B", engine="vllm", flavor="a100-large", max_model_len=32768)
    assert p.serve_argv[:3] == ["vllm", "serve", "Qwen/Qwen3.5-9B"]
    i = p.serve_argv.index("--host")
    assert p.serve_argv[i + 1] == "0.0.0.0" and p.serve_argv[p.serve_argv.index("--port") + 1] == "8000"
    assert p.serve_argv[-len(p.vllm_flags.split()):] == p.vllm_flags.split()
    assert p.vllm_flags == jp.PROFILES["Qwen/Qwen3.5-9B"].flags
    assert p.served_model_name == "Qwen/Qwen3.5-9B" and p.ready_paths == ["/health", "/v1/models"]
    assert p.setup_argvs == [["uv", "pip", "install", "--system", "vllm==0.30.0"]]
    assert p.image == jl.DEFAULT_IMAGE and "--expose 8000" in shlex.join(js.hf_command(p))
    with pytest.raises(SpecError, match="no GPU"):
        serve_plan("Qwen/Qwen3.5-9B", engine="vllm", flavor="cpu-upgrade")


def test_submit_serve_records_kind_serve_in_the_ledger(tmp_path):
    p = serve_plan()
    hf = FakeHF()
    row = js.submit_serve(p, runner=hf, ledger=tmp_path / "jobs.jsonl")
    assert hf.staged == ["plan.json", "serve.sh"]   # no wheel
    assert hf.calls[1] == js.hf_command(p)
    led = [json.loads(x) for x in (tmp_path / "jobs.jsonl").read_text().splitlines()]
    assert led == [row]
    assert {k: row[k] for k in ("kind", "engine", "model", "served_model_name", "port", "flavor",
                                "rate_usd_per_hour", "mode")} == {
        "kind": "serve", "engine": "sglang", "model": "RecursiveSI/flint",
        "served_model_name": "flint", "port": 30000, "flavor": "h200", "rate_usd_per_hour": 5.0,
        "mode": "serve"}
    assert js.ledger_row(row["job_id"], tmp_path / "jobs.jsonl") == row
    with pytest.raises(SpecError, match="HF_TOKEN not set"):
        js.submit_serve(js.plan_serve("RecursiveSI/flint", git=lambda: (COMMIT, False),
                                      environ={}), runner=FakeHF())


# ---- the endpoint ------------------------------------------------------------------------------
def inspect_json(stage="RUNNING", urls=(URL,), ports=(30000,), secs=1800):
    return json.dumps({"id": JOB, "flavor": "h200", "status": {
        "stage": stage, "message": None, "expose_urls": list(urls) or None},
        "expose": list(ports), "durations": {"running_secs": secs},
        "url": f"https://huggingface.co/jobs/cmpatino/{JOB}", "endpoint": "https://huggingface.co"})


def inspect_runner(out):
    calls = []

    def run(args):
        calls.append(args)
        return subprocess.CompletedProcess(args, 0, out, "")
    run.calls = calls
    return run


def ledger_with_serve(tmp_path, **kw):
    path = tmp_path / "jobs.jsonl"
    path.write_text(json.dumps({"kind": "launch", "job_id": "x" * 24}) + "\n"
                    + json.dumps({"kind": "serve", "job_id": JOB, "port": 30000,
                                  "served_model_name": "flint", "flavor": "h200",
                                  "timeout_s": 6 * 3600, **kw}) + "\n")
    return path


def test_resolve_endpoint_from_a_stubbed_inspect(tmp_path):
    gets = []

    def get(url, headers):
        gets.append((url, dict(headers)))
        return 200, {"object": "list", "data": [{"id": "flint"}]}

    run = inspect_runner(inspect_json(urls=[f"https://{JOB}--8080.hf.jobs", URL]))
    ep = js.resolve_endpoint(JOB, runner=run, ledger=ledger_with_serve(tmp_path), get=get,
                             environ={"HF_TOKEN": "hf_x"})
    assert run.calls == [["hf", "jobs", "inspect", JOB, "--format", "json"]]
    assert gets == [(f"{URL}/v1/models", {"authorization": "Bearer hf_x"})]
    assert (ep["url"], ep["base_url"], ep["served_model_name"], ep["ready"]) == (
        URL, f"{URL}/v1", "flint", True)
    assert ep["cost_usd"] == 2.5 and ep["remaining_s"] == 6 * 3600 - 1800
    assert "ready true" in js.describe_endpoint(ep) and "$2.50 so far" in js.describe_endpoint(ep)


def test_endpoint_not_ready_or_not_exposed(tmp_path):
    led = ledger_with_serve(tmp_path)
    loading = js.resolve_endpoint(JOB, runner=inspect_runner(inspect_json()), ledger=led,
                                  get=lambda u, h: (502, None), environ={"HF_TOKEN": "hf_x"})
    assert loading["ready"] is False and "HTTP 502" in loading["detail"]
    other = js.resolve_endpoint(JOB, runner=inspect_runner(inspect_json()), ledger=led,
                                get=lambda u, h: (200, {"data": [{"id": "other"}]}), environ={})
    assert other["ready"] is False and "HF_TOKEN is not set" in other["detail"]
    queued = js.resolve_endpoint(JOB, runner=inspect_runner(inspect_json("SCHEDULING", urls=())),
                                 ledger=led, get=lambda u, h: pytest.fail("no URL, no request"))
    assert queued["url"] is None and "SCHEDULING" in queued["detail"]
    # no ledger row: the port from inspect, the served name from the server
    bare = js.resolve_endpoint(JOB, runner=inspect_runner(inspect_json()), ledger=tmp_path / "none",
                               get=lambda u, h: (200, {"data": [{"id": "flint"}]}), environ={})
    assert bare["ready"] is True and bare["served_model_name"] == "flint" and not bare["in_ledger"]


# ---- job run --endpoint ------------------------------------------------------------------------
def test_endpoint_sets_the_provider_url_and_pins_the_host(tmp_path):
    spec = write(tmp_path, flint_doc())
    p = run_plan(spec, endpoint(remaining_s=3600))
    assert p.mode == "api" and p.secrets == ["HF_TOKEN"]
    env = env_of(jl.hf_command(p))
    assert env["SWARMLAB_ENDPOINT_URL"] == f"{URL}/v1" and env["SWARMLAB_ENDPOINT_JOB"] == JOB
    assert env["SWARMLAB_PIN_HOSTS_EXTRA"] == f"{JOB}--30000.hf.jobs"
    assert p.programbench["pin_hosts_extra"] == [f"{JOB}--30000.hf.jobs"]
    assert p.endpoint["base_url"] == f"{URL}/v1" and p.as_dict()["endpoint"]["job_id"] == JOB
    assert any("times out in 60 min" in n for n in p.notes)
    # the URL is in no spec: the job spec and every run's spec_hash ignore the endpoint
    assert JOB not in p.job_spec_yaml
    other = run_plan(spec, endpoint(url="https://ffffffffffffffffffffffff--30000.hf.jobs"))
    assert other.job_spec_sha256 == p.job_spec_sha256
    exp = Experiment.from_yaml(spec, "solo")
    assert exp.provider_for("served:flint")[0].history_reasoning
    text = jl.describe(p)
    assert f"endpoint: {URL}/v1" in text and f"{JOB}--30000.hf.jobs)" in text


def test_endpoint_without_lockdown_pins_nothing(tmp_path):
    p = run_plan(write(tmp_path, flint_doc(lockdown=False)), endpoint())
    assert "SWARMLAB_PIN_HOSTS_EXTRA" not in env_of(jl.hf_command(p))


def test_flint_example_plans_against_an_endpoint():
    p = run_plan(FLINT_EXAMPLE, endpoint())
    assert p.programbench["task"] == TASK
    assert env_of(jl.hf_command(p))["SWARMLAB_ENDPOINT_URL"] == f"{URL}/v1"


def test_endpoint_refusals(tmp_path):
    with pytest.raises(SpecError, match="vllm: models"):
        run_plan(vllm_spec(tmp_path), endpoint())
    with pytest.raises(SpecError, match="--model"):
        run_plan(vllm_spec(tmp_path), endpoint(), model="Qwen/Qwen3.5-9B")
    with pytest.raises(SpecError, match="must be the served model name"):
        run_plan(write(tmp_path, {**flint_doc("served:other"),
                                  "providers": {"served": {**SERVED, "params": {
                                      **SERVED["params"], "pricing": {"other": [0, 0, 0]}}}}}),
                 endpoint())
    with pytest.raises(SpecError, match="no chosen arm calls a provider with base_url_env"):
        run_plan(write(tmp_path, solo_doc(), "plain.yaml"), endpoint(),
                 environ={"HF_TOKEN": "x", "ANTHROPIC_API_KEY": "y"})
    with pytest.raises(SpecError, match="pass one"):
        run_plan(write(tmp_path, flint_doc(), "noep.yaml"))
    with pytest.raises(SpecError, match="no usable URL"):
        run_plan(write(tmp_path, flint_doc(), "nourl.yaml"), endpoint(url=None))


def test_cli_job_run_endpoint_resolves_and_refuses_to_launch_when_not_ready(tmp_path, monkeypatch):
    spec = write(tmp_path, flint_doc())
    seen = {}

    def fake_resolve(job_id, **kw):
        seen["job"] = job_id
        return {"job_id": job_id, "url": URL, "base_url": f"{URL}/v1", "served_model_name": "flint",
                "ready": False, "stage": "RUNNING", "flavor": "h200", "rate_usd_per_hour": 5.0,
                "running_secs": 60, "cost_usd": 0.08, "remaining_s": 9 * 3600, "in_ledger": True,
                "detail": "GET .../v1/models: HTTP 502"}

    monkeypatch.setattr(js, "resolve_endpoint", fake_resolve)
    monkeypatch.setattr(jl, "checkout_identity", lambda: (COMMIT, False))
    monkeypatch.setattr(jl, "submit", lambda *a, **k: pytest.fail("must not submit"))
    monkeypatch.setenv("HF_TOKEN", "hf_x")
    res = cli.invoke(app, ["job", "run", str(spec), "--endpoint", JOB], catch_exceptions=False)
    assert res.exit_code == 0, res.output
    assert seen["job"] == JOB and f"SWARMLAB_ENDPOINT_URL={URL}/v1" in res.output
    assert "ready false" in res.output and "not launched" in res.output
    res = cli.invoke(app, ["job", "run", str(spec), "--endpoint", JOB, "--launch"])
    assert res.exit_code == 2 and "not ready" in res.stderr
    res = cli.invoke(app, ["job", "run", str(spec), "--endpoint-url", URL, "--json"])
    assert res.exit_code == 0 and json.loads(res.stdout)["plan"]["endpoint"]["url"] == URL


def test_cli_job_serve_endpoint_and_stop(tmp_path, monkeypatch):
    monkeypatch.setattr(jl, "checkout_identity", lambda: (COMMIT, False))
    monkeypatch.setenv("HF_TOKEN", "hf_x")
    res = cli.invoke(app, ["job", "serve", "RecursiveSI/flint", "--engine", "sglang", "--profile",
                           "flint", "--flavor", "h200"], catch_exceptions=False)
    assert res.exit_code == 0 and "--expose 30000" in res.output and "not launched" in res.output
    bad = cli.invoke(app, ["job", "serve", "RecursiveSI/flint", "--flavor", "a100-large"])
    assert bad.exit_code == 2 and "141 GB" in bad.stderr
    submitted = {}
    monkeypatch.setattr(js, "submit_serve", lambda plan, **kw: submitted.update(plan=plan, **kw)
                        or {"job_id": JOB, "url": "u"})
    res = cli.invoke(app, ["job", "serve", "RecursiveSI/flint", "--launch", "--ledger",
                           str(tmp_path / "l.jsonl")])
    assert res.exit_code == 0 and f"swarmlab job endpoint {JOB}" in res.output
    assert submitted["plan"].served_model_name == "flint"
    monkeypatch.setattr(js, "resolve_endpoint", lambda job_id, **kw: {
        "job_id": job_id, "url": URL, "base_url": f"{URL}/v1", "served_model_name": "flint",
        "ready": True, "stage": "RUNNING", "flavor": "h200", "rate_usd_per_hour": 5.0,
        "running_secs": 3600, "cost_usd": 5.0, "in_ledger": True, "detail": None})
    res = cli.invoke(app, ["job", "endpoint", JOB, "--json"])
    data = json.loads(res.stdout)
    assert (data["url"], data["served_model_name"], data["ready"]) == (URL, "flint", True)
    cancelled = []
    monkeypatch.setattr(js, "stop", lambda job_id, **kw: cancelled.append(job_id) or "")
    res = cli.invoke(app, ["job", "stop", JOB])
    assert res.exit_code == 0 and cancelled == [JOB] and f"cancelled job {JOB}" in res.output


def test_stop_calls_hf_jobs_cancel():
    run = inspect_runner("")
    js.stop(JOB, runner=run)
    assert run.calls == [["hf", "jobs", "cancel", JOB]]


# ---- serve.sh against a stub server ------------------------------------------------------------
FAKE_SERVER = r'''
import json, os, sys
from http.server import BaseHTTPRequestHandler, HTTPServer
class H(BaseHTTPRequestHandler):
    def log_message(self, *a): pass
    def _send(self, obj):
        body = json.dumps(obj).encode()
        self.send_response(200); self.send_header("content-type", "application/json")
        self.send_header("content-length", str(len(body))); self.end_headers(); self.wfile.write(body)
    def do_GET(self):
        self._send({"data": [{"id": "flint"}]} if self.path == "/v1/models" else {})
    def do_POST(self):
        self.rfile.read(int(self.headers["content-length"]))
        self._send({"choices": [{"finish_reason": "tool_calls", "message": {"content": None,
                    "reasoning_content": "think", "tool_calls": [{"id": "c1", "type": "function",
                    "function": {"name": "guess", "arguments": "{}"}}]}}]})
HTTPServer(("127.0.0.1", int(os.environ["PORT"])), H).serve_forever()
'''


@pytest.mark.skipif(shutil.which("curl") is None, reason="needs curl")
@pytest.mark.parametrize("setup_ok", [True, False])
def test_serve_sh_ready_then_serves_until_sigterm(tmp_path, setup_ok):
    port = _free_port()
    stage, stub = tmp_path / "stage", tmp_path / "stub"
    stage.mkdir()
    stub.mkdir()
    (stub / "server.py").write_text(FAKE_SERVER)
    (stub / "uv").write_text('#!/usr/bin/env bash\necho "uv $*" >> "$STUB_LOG"\n')
    (stub / "uv").chmod(0o755)
    p = serve_plan()
    plan = {**p.as_dict(),
            "setup_argvs": [["touch", str(tmp_path / "setup-ran")]] + ([] if setup_ok else [["false"]]),
            "serve_argv": ["env", f"PORT={port}", sys.executable, str(stub / "server.py")]}
    (stage / "plan.json").write_text(json.dumps(plan))
    env = {**{k: v for k, v in p.env().items()}, "SWARMLAB_SERVE_PORT": str(port),
           "SWARMLAB_SERVE_READY_PATHS": "/health_generate /v1/models", "SWARMLAB_SERVE_WAIT_S": "60",
           "SWARMLAB_STAGE_DIR": str(stage), "SWARMLAB_JOB_TMP": str(tmp_path / "jt"),
           "SWARMLAB_UV": str(stub / "uv"), "STUB_LOG": str(tmp_path / "stub.log"),
           "PATH": os.environ["PATH"], "HOME": str(tmp_path)}
    proc = subprocess.Popen(["bash", str(REPO / "swarmlab" / "jobs" / "serve.sh")], env=env,
                            stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True)
    lines = []
    deadline = time.time() + 60
    while time.time() < deadline:
        line = proc.stdout.readline()
        if not line:
            break
        lines.append(line)
        if "serving until" in line:
            proc.send_signal(signal.SIGTERM)
            break
    out = "".join(lines) + proc.communicate(timeout=60)[0]
    serve_json = json.loads((tmp_path / "jt" / "serve.json").read_text())
    assert (tmp_path / "setup-ran").exists()
    if not setup_ok:
        assert proc.returncode == 11, out
        assert serve_json["status"] == "failed" and "SWARMLAB_SERVE_READY" not in out
        return
    assert proc.returncode == 143, out
    assert f"SWARMLAB_SERVE_READY url=http://127.0.0.1:{port} served_model_name=flint" in out
    assert 'PROBE {"outcome": "ok"' in out
    assert serve_json["status"] == "stopped" and serve_json["ready_s"] is not None
    assert serve_json["probe"]["reasoning_chars"] == 5 and serve_json["served_model_name"] == "flint"
    log = (tmp_path / "stub.log").read_text()
    assert "hf buckets cp" in log and "/serve.json" in log and "logs/server.log" in log
