"""WP8: HF Jobs placement with a co-located vLLM server. Offline: every `hf` call is mocked,
and the bootstrap script runs against stub `uv`, `hf` and `vllm` executables."""
import json
import os
import shutil
import subprocess
import sys
import textwrap
from pathlib import Path

import pytest
import yaml
from typer.testing import CliRunner

from swarmlab import Experiment, Run
from swarmlab.cli import app
from swarmlab.jobs import launch as jl
from swarmlab.jobs import remote
from swarmlab.spec import SpecError, git_identity

REPO = Path(__file__).resolve().parents[1]
MODEL = "Qwen/Qwen3.5-9B"
COMMIT = "0123456789abcdef0123456789abcdef01234567"
NOW = 1_791_000_000.0  # fixed clock -> fixed tag
cli = CliRunner()


def vllm_spec(tmp_path, model=f"vllm:{MODEL}", n=3, extra_arm=None):
    doc = {
        "name": "jobs-test",
        "seeds": [1, 2],
        "options": {"max_rounds": 2},
        "arms": {
            "bc": {
                "world": {"type": "flaggame", "params": {"n_candidates": 8}},
                "participants": [{"type": "llm", "count": n, "params": {
                    "model": model, "max_tokens": 2048,
                    "extra": {"chat_template_kwargs": {"enable_thinking": False}}}}],
                "probes": [{"type": "belief"}],
                "metrics": ["belief.accuracy",
                            {"type": "belief.accuracy", "params": {"source": "probe:belief"}}],
            },
        },
    }
    if extra_arm:
        doc["arms"].update(extra_arm)
    path = tmp_path / "spec.yaml"
    path.write_text(yaml.safe_dump(doc))
    return path


SCRIPTED_ARM = {"scripted": {"world": "flaggame",
                             "participants": [{"type": "evidence_aggregator", "count": 4}]}}


def plan(spec, **kw):
    kw.setdefault("model", MODEL)
    return jl.plan_job(spec, now=NOW, git=lambda: (COMMIT, False), **kw)


# ---- plan, command, determinism -----------------------------------------------------------------
def test_command_is_deterministic_and_complete(tmp_path):
    spec = vllm_spec(tmp_path)
    a, b = plan(spec, timeout="2h"), plan(spec, timeout="2h")
    assert jl.hf_command(a) == jl.hf_command(b)
    assert a.tag == "jobs-test-0123456-20261003-040000"
    cmd = jl.hf_command(a)
    joined = " ".join(cmd)
    assert cmd[:4] == ["hf", "jobs", "run", "--detach"]
    for piece in ("--flavor a100-large", "--timeout 2h", "--secrets HF_TOKEN",
                  "--label experiment=swarmlab", f"--label swarmlab_tag={a.tag}"):
        assert piece in joined
    env = dict(cmd[i + 1].split("=", 1) for i, c in enumerate(cmd) if c == "-e")
    assert env["SWARMLAB_STAGE"] == f"hf://buckets/cmpatino/swarmlab-runs/jobs/{a.tag}"
    assert env["SWARMLAB_RUNS_REMOTE"] == f"hf://buckets/cmpatino/swarmlab-runs/runs/{a.tag}"
    assert env["SWARMLAB_RUNS"] == "bc:1 bc:2"
    assert env["SWARMLAB_GIT_COMMIT"] == COMMIT and env["SWARMLAB_GIT_DIRTY"] == "0"
    assert env["VLLM_MODEL"] == MODEL and env["VLLM_GPU_MEMORY_UTILIZATION"] == "0.9"
    for flag in ("--reasoning-parser qwen3", "--enable-auto-tool-choice",
                 "--tool-call-parser qwen3_coder"):
        assert flag in env["VLLM_FLAGS"]
    assert env["VLLM_WAIT_S"] == "1200" and env["VLLM_MAX_MODEL_LEN"] == "262144"
    assert "N" not in env  # HF Jobs rejects an env var named N
    image, entry = cmd[-4], cmd[-1]
    assert image == jl.DEFAULT_IMAGE and cmd[-3:-1] == ["bash", "-c"]
    assert 'hf buckets sync "$SWARMLAB_STAGE" /opt/swarmlab/stage' in entry
    assert entry.endswith("exec bash /opt/swarmlab/stage/bootstrap.sh")
    assert jl.hf_command(a) == a.as_dict()["hf_command"]


def test_job_spec_injects_self_hosted_vllm_provider(tmp_path):
    p = plan(vllm_spec(tmp_path))
    doc = yaml.safe_load(p.job_spec_yaml)
    prov = doc["providers"]["vllm"]
    assert prov["type"] == "openai_compat"
    assert prov["params"] == {"name": "vllm", "base_url": "http://127.0.0.1:8000/v1",
                              "pricing": {MODEL: [0.0, 0.0, 0.0]}, "concurrency": 64,
                              "timeout_s": 600.0, "self_hosted": True}
    staged = tmp_path / "job.yaml"
    staged.write_text(p.job_spec_yaml)
    exp = Experiment.from_yaml(staged, "bc")
    provider, mid = exp.provider_for(f"vllm:{MODEL}")
    assert provider.self_hosted and provider.model_pricing(mid) == (0.0, 0.0, 0.0)
    assert exp.estimate(1, 2)["usd"] == 0.0
    # the original spec is shipped too, unchanged
    assert p.orig_spec_yaml == (tmp_path / "spec.yaml").read_text()


def test_validation_of_vllm_model_ids(tmp_path):
    with pytest.raises(SpecError, match="--model is"):
        plan(vllm_spec(tmp_path), model="Qwen/Qwen3.5-27B")
    with pytest.raises(SpecError, match="only vllm:"):
        plan(vllm_spec(tmp_path, model="hf:Qwen/Qwen3.5-9B:deepinfra"))
    spec = vllm_spec(tmp_path, extra_arm=SCRIPTED_ARM)
    with pytest.raises(SpecError, match="call no model"):
        plan(spec, arms=["scripted"])
    with pytest.raises(SpecError, match="unknown arm"):
        plan(spec, arms=["nope"])
    with pytest.raises(SpecError, match="hf://buckets/"):
        plan(spec, arms=["bc"], bucket="s3://x")
    # a scripted arm next to a vllm arm is fine (it calls no model)
    assert plan(spec).runs == [("bc", 1), ("bc", 2), ("scripted", 1), ("scripted", 2)]


def test_estimate_overrides_and_short_timeout_note(tmp_path):
    p = plan(vllm_spec(tmp_path), per_round_s=100, setup_s=60, timeout="5m")
    assert p.estimate_s == 60 + 2 * 2 * 100
    assert p.timeout_s == 300 and any("below" in n for n in p.notes)
    assert jl.parse_duration("1.5h") == 5400 and jl.fmt_duration(5400) == "90m"
    with pytest.raises(SpecError):
        jl.parse_duration("soon")


# ---- bootstrap script ---------------------------------------------------------------------------
def test_bootstrap_script_has_expected_pieces():
    s = jl.bootstrap_script()
    for piece in ('"$STAGE_DIR"/swarmlab-*.whl', 'uv pip install -q --python "$VENV/bin/python" "$WHEEL"',
                  'uv pip install --system -q "vllm==$VLLM_VERSION"',
                  '"$VLLM_BIN" serve "$VLLM_MODEL"', '--gpu-memory-utilization',
                  '--max-model-len', '"$BASE/health"', '"$BASE/v1/models"', 'VLLM_WAIT_S',
                  'while sleep "$SYNC_EVERY"; do sync_runs; done',
                  'buckets sync "$OUT" "$SWARMLAB_RUNS_REMOTE"',
                  '--yes --out "$OUT" --json', '"$SWARMLAB_STAGE/job.json"', 'trap'):
        assert piece in s, piece
    assert "exclude" not in s  # the whole run dir (blobs, cache) goes to the bucket
    subprocess.run(["bash", "-n", "-c", s], check=True)


STUB_HF = r'''#!/usr/bin/env bash
# stub hf: buckets sync|cp map hf://buckets/<ns>/<b>/... onto $FAKE_BUCKET/...
echo "hf $*" >> "$STUB_LOG"
map() { case "$1" in hf://buckets/*) echo "$FAKE_BUCKET/${1#hf://buckets/*/*/}";; *) echo "$1";; esac; }
[ "$1" = buckets ] || exit 2
src=$(map "$3"); dst=$(map "$4")
case "$2" in
  sync) mkdir -p "$dst" && cp -r "$src"/. "$dst"/ ;;
  cp) mkdir -p "$(dirname "$dst")" && cp "$src" "$dst" ;;
  *) exit 2 ;;
esac
'''

STUB_UV = r'''#!/usr/bin/env bash
echo "uv $*" >> "$STUB_LOG"
if [ "$1" = venv ]; then
  dir="${@: -1}"; mkdir -p "$dir/bin"
  ln -sf "$REAL_PYTHON" "$dir/bin/python"; ln -sf "$REAL_SWARMLAB" "$dir/bin/swarmlab"
  ln -sf "$STUB_DIR/hf" "$dir/bin/hf"
fi
exit 0
'''

STUB_VLLM = r'''#!/usr/bin/env bash
echo "vllm $*" >> "$STUB_LOG"
port=8000; prev=""
for a in "$@"; do [ "$prev" = "--port" ] && port="$a"; prev="$a"; done
exec "$REAL_PYTHON" "$STUB_DIR/fake_vllm.py" "$port"
'''

FAKE_VLLM = r'''
import json, os, sys
from http.server import BaseHTTPRequestHandler, HTTPServer
MODEL = os.environ["VLLM_MODEL"]
class H(BaseHTTPRequestHandler):
    def log_message(self, *a): pass
    def _send(self, obj):
        body = json.dumps(obj).encode()
        self.send_response(200); self.send_header("content-type", "application/json")
        self.send_header("content-length", str(len(body))); self.end_headers(); self.wfile.write(body)
    def do_GET(self):
        self._send({"data": [{"id": MODEL}]} if self.path == "/v1/models" else {})
    def do_POST(self):
        self.rfile.read(int(self.headers["content-length"]))
        self._send({"choices": [{"finish_reason": "tool_calls", "message": {"content": None,
                    "tool_calls": [{"id": "c1", "type": "function", "function": {
                        "name": "guess", "arguments": "{\"candidate\": \"B\"}"}}]}}],
                    "usage": {"prompt_tokens": 10, "completion_tokens": 5}})
HTTPServer(("127.0.0.1", int(sys.argv[1])), H).serve_forever()
'''


def _free_port():
    import socket
    with socket.socket() as s:
        s.bind(("127.0.0.1", 0))
        return s.getsockname()[1]


def _run_bootstrap(tmp_path, runs):
    """Run bootstrap.sh for real with stub uv/hf/vllm on PATH and a scripted spec."""
    stub = tmp_path / "stubs"
    stub.mkdir()
    for name, body in (("hf", STUB_HF), ("uv", STUB_UV), ("vllm", STUB_VLLM)):
        (stub / name).write_text(body)
        (stub / name).chmod(0o755)
    (stub / "fake_vllm.py").write_text(FAKE_VLLM)
    bucket = tmp_path / "bucket"
    stage = tmp_path / "stage"
    stage.mkdir()
    (stage / "swarmlab-0.0.1-py3-none-any.whl").write_text("not a real wheel")
    (stage / "spec.yaml").write_text(yaml.safe_dump({
        "name": "boot", "options": {"max_rounds": 2},
        "arms": {"scripted": {"world": "flaggame",
                              "participants": [{"type": "evidence_aggregator", "count": 3}]}}}))
    real_python = Path(sys.executable)
    env = {
        "PATH": f"{stub}:{os.environ['PATH']}", "HOME": str(tmp_path),
        "STUB_LOG": str(tmp_path / "stub.log"), "STUB_DIR": str(stub), "FAKE_BUCKET": str(bucket),
        "REAL_PYTHON": str(real_python), "REAL_SWARMLAB": str(real_python.parent / "swarmlab"),
        "SWARMLAB_STAGE_DIR": str(stage), "SWARMLAB_TAG": "t1",
        "SWARMLAB_STAGE": "hf://buckets/ns/b/jobs/t1", "SWARMLAB_RUNS_REMOTE": "hf://buckets/ns/b/runs/t1",
        "SWARMLAB_RUNS": runs, "SWARMLAB_GIT_COMMIT": COMMIT, "SWARMLAB_GIT_DIRTY": "0",
        "SWARMLAB_SYNC_EVERY": "1", "VLLM_MODEL": MODEL, "VLLM_VERSION": "0.30.0",
        "VLLM_MAX_MODEL_LEN": "4096", "VLLM_GPU_MEMORY_UTILIZATION": "0.9",
        "VLLM_FLAGS": "--reasoning-parser qwen3", "VLLM_WAIT_S": "60",
        "VLLM_PORT": str(_free_port()), "JOB_ID": "a" * 24, "TMPDIR": str(tmp_path),
        "SWARMLAB_JOB_ROOT": str(tmp_path / "root"), "SWARMLAB_JOB_TMP": str(tmp_path / "jt"),
    }
    proc = subprocess.run(["bash", str(REPO / "swarmlab" / "jobs" / "bootstrap.sh")], env=env,
                          capture_output=True, text=True, timeout=180, cwd=tmp_path, check=False)
    return proc, bucket, (tmp_path / "stub.log").read_text()


@pytest.mark.skipif(shutil.which("curl") is None, reason="needs curl")
def test_bootstrap_end_to_end_with_stubs(tmp_path):
    proc, bucket, calls = _run_bootstrap(tmp_path, "scripted:1 scripted:2")
    assert proc.returncode == 0, proc.stdout + proc.stderr
    assert "PROBE finish=tool_calls" in proc.stdout and "vLLM ready" in proc.stdout
    assert "vllm serve Qwen/Qwen3.5-9B --served-model-name Qwen/Qwen3.5-9B" in calls
    assert "--max-model-len 4096 --gpu-memory-utilization 0.9 --reasoning-parser qwen3" in calls
    assert "uv pip install --system -q vllm==0.30.0" in calls
    # runs synced to the bucket, complete enough to load locally
    runs = bucket / "runs" / "t1"
    assert sorted(p.name for p in runs.iterdir()) == [  # run dirs + the experiment ledger
        "boot.ledger.jsonl", "boot__scripted__s1", "boot__scripted__s2"]
    r = Run.load(runs / "boot__scripted__s1")
    assert r.status == "ended" and r.meta["git_commit"] == COMMIT
    manifest = json.loads((bucket / "jobs" / "t1" / "job.json").read_text())
    assert manifest["status"] == "completed" and manifest["exit_code"] == 0
    assert manifest["job_id"] == "a" * 24 and manifest["commit"] == COMMIT
    assert manifest["runs_done"] == ["scripted:1", "scripted:2"] and manifest["ended_at"]
    assert manifest["vllm"]["model"] == MODEL and manifest["vllm"]["ready_s"] is not None
    results = (bucket / "jobs" / "t1" / "results.jsonl").read_text().splitlines()
    assert len(results) == 2 and json.loads(results[0])["run_id"] == "boot__scripted__s1"
    assert (bucket / "jobs" / "t1" / "vllm.log").exists()


@pytest.mark.skipif(shutil.which("curl") is None, reason="needs curl")
def test_bootstrap_failed_run_exits_nonzero(tmp_path):
    proc, bucket, _ = _run_bootstrap(tmp_path, "scripted:1 missing-arm:1")
    assert proc.returncode == 1, proc.stdout + proc.stderr
    manifest = json.loads((bucket / "jobs" / "t1" / "job.json").read_text())
    assert manifest["status"] == "failed" and manifest["exit_code"] == 1
    assert manifest["runs_failed"] == ["missing-arm:1"] and manifest["runs_done"] == ["scripted:1"]
    assert (bucket / "runs" / "t1" / "boot__scripted__s1" / "run.json").exists()


# ---- stage and submit ---------------------------------------------------------------------------
def test_stage_writes_files_and_builds_wheel(tmp_path):
    p = plan(vllm_spec(tmp_path))
    built = []

    def fake_build(d):
        built.append(d)
        (d / "swarmlab-0.0.1-py3-none-any.whl").write_text("w")
        return d / "swarmlab-0.0.1-py3-none-any.whl"

    d = jl.stage(p, tmp_path / "stage", build=fake_build)
    assert built == [d]
    assert sorted(x.name for x in d.iterdir()) == [
        "bootstrap.sh", "plan.json", "spec.orig.yaml", "spec.yaml", "swarmlab-0.0.1-py3-none-any.whl"]
    assert (d / "spec.yaml").read_text() == p.job_spec_yaml
    assert json.loads((d / "plan.json").read_text())["tag"] == p.tag


class FakeHF:
    def __init__(self, out="Job started with ID: 6abd02b3031314b696344afb\n", code=0):
        self.calls, self.out, self.code = [], out, code
        self.staged = None

    def __call__(self, args):
        self.calls.append(list(args))
        if args[1:3] == ["buckets", "sync"]:
            self.staged = sorted(p.name for p in Path(args[3]).iterdir())
        if args[1:3] == ["jobs", "run"]:
            return subprocess.CompletedProcess(args, self.code, self.out, "")
        return subprocess.CompletedProcess(args, 0, "", "")


def test_submit_uploads_stage_then_runs_and_records(tmp_path):
    p = plan(vllm_spec(tmp_path))
    hf = FakeHF()
    row = jl.submit(p, runner=hf, build=None, ledger=tmp_path / "jobs.jsonl")
    assert [c[1:3] for c in hf.calls] == [["buckets", "sync"], ["jobs", "run"], ["buckets", "cp"]]
    assert hf.calls[0][4] == p.stage_remote and "spec.yaml" in hf.staged
    assert hf.calls[1] == jl.hf_command(p)
    assert hf.calls[2][4] == f"{p.stage_remote}/launch.json"
    assert row["job_id"] == "6abd02b3031314b696344afb"
    assert row["url"] == "https://huggingface.co/jobs/cmpatino/6abd02b3031314b696344afb"
    ledger = [json.loads(x) for x in (tmp_path / "jobs.jsonl").read_text().splitlines()]
    assert ledger[0]["tag"] == p.tag and ledger[0]["kind"] == "launch"


def test_submit_failures_raise(tmp_path):
    p = plan(vllm_spec(tmp_path))
    with pytest.raises(RuntimeError, match="hf jobs run failed"):
        jl.submit(p, runner=FakeHF(code=1), build=None)
    with pytest.raises(RuntimeError, match="could not parse a job id"):
        jl.submit(p, runner=FakeHF(out="ok"), build=None)


# ---- CLI ----------------------------------------------------------------------------------------
def test_cli_job_run_prints_and_does_not_launch(tmp_path, monkeypatch):
    def boom(*a, **k):
        raise AssertionError("must not submit without --launch")

    monkeypatch.setattr(jl, "submit", boom)
    spec = vllm_spec(tmp_path)
    res = cli.invoke(app, ["job", "run", str(spec), "--model", MODEL, "--timeout", "2h",
                           "--seeds", "3", "--stage-dir", str(tmp_path / "st")], catch_exceptions=False)
    assert res.exit_code == 0, res.output
    assert "hf jobs run --detach --flavor a100-large --timeout 2h" in res.output
    assert "SWARMLAB_RUNS=bc:3" in res.output and "not launched" in res.output
    assert "estimate:" in res.output
    assert (tmp_path / "st" / "spec.yaml").exists() and (tmp_path / "st" / "plan.json").exists()
    bad = cli.invoke(app, ["job", "run", str(spec), "--model", "other/model"])
    assert bad.exit_code == 2 and "--model is" in bad.stderr


def test_cli_job_run_launch_calls_submit(tmp_path, monkeypatch):
    seen = {}

    def fake_submit(plan, **kw):
        seen["plan"] = plan
        return {"job_id": "b" * 24, "url": "u"}

    monkeypatch.setattr(jl, "submit", fake_submit)
    monkeypatch.setattr(jl, "checkout_identity", lambda: (COMMIT, False))
    res = cli.invoke(app, ["job", "run", str(vllm_spec(tmp_path)), "--model", MODEL, "--launch",
                           "--arm", "bc", "--json"])
    assert res.exit_code == 0, res.output
    data = json.loads(res.stdout)
    assert data["launched"] and data["job"]["job_id"] == "b" * 24
    assert seen["plan"].runs == [("bc", 1), ("bc", 2)]
    monkeypatch.setattr(jl, "checkout_identity", lambda: (COMMIT, True))
    dirty = cli.invoke(app, ["job", "run", str(vllm_spec(tmp_path)), "--model", MODEL, "--launch"])
    assert dirty.exit_code == 2 and "uncommitted" in dirty.stderr


# ---- fetch, status ------------------------------------------------------------------------------
def _bucket_with_runs(tmp_path):
    spec = tmp_path / "scripted.yaml"
    spec.write_text(yaml.safe_dump({"name": "fx", "options": {"max_rounds": 2}, "arms": SCRIPTED_ARM}))
    exp = Experiment.from_yaml(spec, "scripted")
    bucket = tmp_path / "bucket"
    old = exp.run(1, out=bucket / "runs" / "fx-0123456-20261001-000000")
    new = exp.run(1, out=bucket / "runs" / "fx-0123456-20261005-000000")
    return bucket, old, new


def test_fetch_from_local_stand_in(tmp_path):
    bucket, _, new = _bucket_with_runs(tmp_path)
    d = remote.fetch(new.id, tmp_path / "home", str(bucket))
    assert d == tmp_path / "home" / new.id
    r = Run.load(d)  # replays: the fetched dir is complete
    assert r.status == "ended" and r.score == new.score
    # newest tag wins; an explicit tag (either form) picks an older one
    assert remote.find_run(new.id, str(bucket)) == ("fx-0123456-20261005-000000", new.id)
    assert remote.find_run(f"fx-0123456-20261001-000000/{new.id}", str(bucket))[0] == (
        "fx-0123456-20261001-000000")
    with pytest.raises(FileNotFoundError):
        remote.fetch("nope__s1", tmp_path / "home", str(bucket))
    res = cli.invoke(app, ["job", "fetch", new.id, "--bucket", str(bucket), "--out",
                           str(tmp_path / "cli"), "--json"])
    assert res.exit_code == 0, res.output
    assert json.loads(res.stdout)["run_id"] == new.id


def test_fetch_remote_uses_hf_cli(tmp_path):
    calls = []
    listing = {
        "runs/": [{"type": "directory", "path": "runs/t-20261001"},
                  {"type": "directory", "path": "runs/t-20261005"}],
        "runs/t-20261005/": [{"type": "directory", "path": "runs/t-20261005/other__s1"}],
        "runs/t-20261001/": [{"type": "directory", "path": "runs/t-20261001/x__a__s1"},
                             {"type": "file", "path": "runs/t-20261001/notes.txt"}],
    }

    def runner(args):
        calls.append(args)
        if args[1:3] == ["buckets", "ls"]:
            key = args[3].removeprefix("hf://buckets/ns/b/")
            return subprocess.CompletedProcess(args, 0, json.dumps(listing.get(key, [])) + "\nHint", "")
        if args[1:3] == ["buckets", "sync"]:
            Path(args[4]).mkdir(parents=True, exist_ok=True)
            (Path(args[4]) / "run.json").write_text("{}")
        return subprocess.CompletedProcess(args, 0, "", "")

    d = remote.fetch("x__a__s1", tmp_path, "hf://buckets/ns/b", runner=runner)
    assert d == tmp_path / "x__a__s1"
    assert calls[-1] == ["hf", "buckets", "sync", "hf://buckets/ns/b/runs/t-20261001/x__a__s1", str(d)]


def test_status_digests_inspect(tmp_path):
    info = [{"id": "c" * 24, "status": {"stage": "RUNNING", "message": None}, "flavor": "a100-large",
             "durations": {"running_secs": 1800}, "labels": {"swarmlab_tag": "t"}, "url": "u"}]

    def runner(args):
        assert args == ["hf", "jobs", "inspect", "c" * 24, "--format", "json"]
        return subprocess.CompletedProcess(args, 0, json.dumps(info), "")

    st = remote.status("c" * 24, runner=runner)
    assert st["stage"] == "RUNNING" and st["cost_usd"] == 1.25 and st["tag"] == "t"


# ---- small pieces -------------------------------------------------------------------------------
def test_git_identity_env_fallback(tmp_path, monkeypatch):
    monkeypatch.setenv("SWARMLAB_GIT_COMMIT", COMMIT)
    monkeypatch.setenv("SWARMLAB_GIT_DIRTY", "0")
    assert git_identity(tmp_path) == (COMMIT, False)
    monkeypatch.delenv("SWARMLAB_GIT_COMMIT")
    assert git_identity(tmp_path) == ("unknown", True)


def test_summary_labels_self_hosted(tmp_path):
    spec = tmp_path / "s.yaml"
    spec.write_text(textwrap.dedent(f"""
        name: sh
        options: {{max_rounds: 1}}
        providers:
          vllm: {{type: openai_compat, params: {{name: vllm, base_url: "http://127.0.0.1:1/v1",
                  pricing: {{"{MODEL}": [0, 0, 0]}}, self_hosted: true}}}}
        arms:
          a: {{world: flaggame, participants: [{{type: evidence_aggregator, count: 2}}]}}
    """))
    run = Experiment.from_yaml(spec, "a").run(1, out=tmp_path / "runs")
    assert run.summary()["self_hosted"] == ["vllm"]
    res = cli.invoke(app, ["run", str(spec), "--out", str(tmp_path / "runs2"), "--seed", "1", "--yes"])
    assert res.exit_code == 0 and "self_hosted ['vllm']" in res.output
