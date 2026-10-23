"""`swarmlab job serve` / `job endpoint` / `job stop`: a model served by its own HF Job.

A ProgramBench run (API mode, CPU flavor, inside the task image) cannot host a GPU model, so the
model is served by a second, GPU job and the run calls it over the jobs proxy. The serving job
IS the server: it sets up the engine, starts it, waits until it is ready, logs
`SWARMLAB_SERVE_READY url=<local url>` and then waits on the server process, so it ends by its
timeout (or `swarmlab job stop`), or when the server exits. It runs `serve.sh` from the stage
dir (`<bucket>/jobs/<tag>/`: `serve.sh`, `plan.json`; no wheel, nothing of swarmlab runs there).

`plan_serve(model, engine=..., flavor=..., timeout=..., profile=...) -> ServePlan` (pure):

- `--engine sglang` serves a model with a kit profile (`profiles.KIT_PROFILES`; `--profile auto`
  finds it by model id): image = the profile's validated tag (`--image` may pin the digest the
  kit's Dockerfile names, `lmsysorg/sglang@sha256:...`, recorded as `image_digest`; the tag is the
  default because the probe ran on it and a digest reference is untested on HF Jobs), setup =
  download the kit (`snapshot_download(model, allow_patterns=["<kit_dir>/*"])`), `bash
  <install> <kit dir>`, download the weights, all into `/opt/model` with the image's own
  `python3` and huggingface_hub (never upgraded: the image's transformers needs < 2.0); serve =
  `env MODEL=/opt/model PORT=<port> bash /opt/model/<kit_dir>/<serve script for the flavor>`. A
  flavor without a serve script in the profile is refused (Flint: h200 only, >= 141 GB per GPU).
  `--revision` pins the repo commit of both downloads.
- `--engine vllm` reuses the in-job vLLM path: the uv image, `uv pip install --system
  vllm==<version>`, and `launch.serve_argv` with the profile/default flags (`profiles.resolve_flags`)
  and `--host 0.0.0.0` (the jobs proxy reaches the container's port, not its loopback). The served
  model name is the HF model id, as in vllm mode.

The plan carries the setup steps and the serve command as argv lists (`setup_argvs`,
`serve_argv`, in plan.json); serve.sh executes exactly those (read NUL-delimited, as bootstrap.sh
does for `vllm_serve_argv`), so the printout is the command that runs. Readiness: every path of
`ready_paths` (sglang kit: `/health_generate` then `/v1/models`; vllm: `/health`, `/v1/models`)
answers 2xx on 127.0.0.1 and `/v1/models` lists the served name, within `SWARMLAB_SERVE_WAIT_S`
(default 1800 s, after setup). Then one tool-calling request (the profile's probe extra) whose
outcome only warns. The port is exposed with `hf jobs run --expose <port>`: the proxy needs an HF
token with read access to the job's namespace; `--expose-public` is never used. `HF_TOKEN` is
passed by name (`--secrets HF_TOKEN`: the kit and weights download, the stage dir).

`submit_serve(plan)` stages, uploads and submits through `launch.submit`; the ledger row
(`runs/jobs.jsonl`) has `kind: serve`, `engine`, `model`, `served_model_name`, `port`, `flavor`,
`rate_usd_per_hour`, `profile`, `image`.

Endpoint (`resolve_endpoint(job_id)`, `swarmlab job endpoint`): `hf jobs inspect <id> --format
json` gives the job's stage and `status.expose_urls` (one URL per exposed port, e.g.
`https://<id>--30000.hf.jobs`; the `url` field is the job's web page, not the endpoint); the URL
for the ledger row's port is taken (else the only one). Readiness is `GET <url>/v1/models` with
`Authorization: Bearer $HF_TOKEN`: ready when it answers 200 and lists the served name (from the
ledger; without a ledger row, the first model listed). The result also has the job's running
seconds, the time left before its timeout (`remaining_s`, from the ledger's `timeout_s`; `job run
--endpoint` notes when it is shorter than the run job's timeout) and its compute cost so far (`rate_usd_per_hour` x hours: the serving cost; the runs that
call it record $0, so there is no per-run attribution).

`swarmlab job run SPEC --endpoint <serve job id>` resolves the endpoint and plans the run job
against it (launch.py, "Self-served endpoint"); `--endpoint-url URL` skips the lookup (offline
plans, tests, a server elsewhere) and then the served name is not checked against the spec.
"""
from __future__ import annotations

import json
import os
import shlex
import time
from collections.abc import Callable, Mapping
from dataclasses import asdict, dataclass, field
from importlib.resources import files
from pathlib import Path
from typing import Any

from ..spec import SpecError
from . import launch as _launch
from . import profiles as jp
from .launch import (
    DEFAULT_BUCKET,
    DEFAULT_GPU_MEMORY_UTILIZATION,
    DEFAULT_IMAGE,
    DEFAULT_MAX_MODEL_LEN,
    DEFAULT_VLLM_VERSION,
    FLAVOR_GPU_GB,
    FLAVOR_USD_PER_HOUR,
    Runner,
    fmt_duration,
    hf_command,
    parse_duration,
    run_cli,
    serve_argv,
    slug,
    submit,
)

ENGINES = ("sglang", "vllm")
SERVE_FLAVOR = "h200"
SERVE_TIMEOUT = "6h"
SERVE_WAIT_S = 1800
MODEL_DIR = "/opt/model"
VLLM_SERVE_PORT = 8000
DEFAULT_LEDGER = "runs/jobs.jsonl"


@dataclass
class ServePlan:
    model: str
    engine: str
    profile: str | None
    flavor: str
    image: str
    image_digest: str | None
    tag: str
    bucket: str
    commit: str
    dirty: bool
    timeout_s: int
    port: int
    served_model_name: str
    revision: str | None
    setup_argvs: list[list[str]]
    ready_paths: list[str]
    probe_extra: dict[str, Any] = field(default_factory=dict)
    setup_env: dict[str, str] = field(default_factory=dict)
    wait_s: int = SERVE_WAIT_S
    notes: list[str] = field(default_factory=list)
    # vllm engine only (launch.serve_argv reads these)
    vllm_version: str | None = None
    vllm_flags: str = ""
    max_model_len: int = DEFAULT_MAX_MODEL_LEN
    gpu_memory_utilization: float = DEFAULT_GPU_MEMORY_UTILIZATION
    kit_serve_argv: list[str] | None = None   # sglang kit: the serve script's command
    host: str = "0.0.0.0"
    mode: str = "serve"
    entry_script: str = "serve.sh"
    secrets: list[str] = field(default_factory=lambda: ["HF_TOKEN"])
    missing_secrets: list[str] = field(default_factory=list)

    @property
    def expose(self) -> list[int]:
        return [self.port]

    @property
    def stage_remote(self) -> str:
        return f"{self.bucket}/jobs/{self.tag}"

    @property
    def rate_usd_per_hour(self) -> float | None:
        return FLAVOR_USD_PER_HOUR.get(self.flavor)

    @property
    def serve_argv(self) -> list[str]:
        return self.kit_serve_argv if self.kit_serve_argv is not None else serve_argv(self)

    def env(self) -> dict[str, str]:
        return {
            "SWARMLAB_MODE": "serve", "SWARMLAB_TAG": self.tag,
            "SWARMLAB_STAGE": self.stage_remote, "SWARMLAB_GIT_COMMIT": self.commit,
            "SWARMLAB_FLAVOR": self.flavor, "SWARMLAB_IMAGE": self.image,
            "SWARMLAB_SERVE_ENGINE": self.engine, "SWARMLAB_SERVE_MODEL": self.model,
            "SWARMLAB_SERVE_NAME": self.served_model_name,
            "SWARMLAB_SERVE_PORT": str(self.port),
            "SWARMLAB_SERVE_READY_PATHS": " ".join(self.ready_paths),
            "SWARMLAB_SERVE_WAIT_S": str(self.wait_s),
            "SWARMLAB_SERVE_PROBE_EXTRA": json.dumps(self.probe_extra, separators=(",", ":"),
                                                     sort_keys=True),
            **self.setup_env,
        }

    def ledger_fields(self) -> dict[str, Any]:
        return {"kind": "serve", "engine": self.engine, "model": self.model,
                "served_model_name": self.served_model_name, "port": self.port,
                "profile": self.profile, "image": self.image, "revision": self.revision}

    def as_dict(self) -> dict[str, Any]:
        d = asdict(self)
        d.update(stage_remote=self.stage_remote, rate_usd_per_hour=self.rate_usd_per_hour,
                 timeout=fmt_duration(self.timeout_s), expose=self.expose,
                 serve_argv=self.serve_argv, serve_command=shlex.join(self.serve_argv),
                 hf_command=hf_command(self))
        return d


def _download(model: str, local_dir: str, revision: str | None,
              patterns: list[str] | None = None) -> list[str]:
    """`python3 -c` argv: huggingface_hub.snapshot_download with the image's own Python."""
    kw = [f"{model!r}", f"local_dir={local_dir!r}"]
    if patterns:
        kw.append(f"allow_patterns={patterns!r}")
    if revision:
        kw.append(f"revision={revision!r}")
    code = f"from huggingface_hub import snapshot_download; print(snapshot_download({', '.join(kw)}))"
    return ["python3", "-c", code]


def plan_serve(model: str, *, engine: str = "sglang", flavor: str | None = None,
               timeout: str | int | None = None, profile: str | None = "auto",
               image: str | None = None, revision: str | None = None,
               bucket: str = DEFAULT_BUCKET, port: int | None = None,
               vllm_version: str = DEFAULT_VLLM_VERSION, vllm_flags: str | None = None,
               max_model_len: int = DEFAULT_MAX_MODEL_LEN,
               gpu_memory_utilization: float = DEFAULT_GPU_MEMORY_UTILIZATION,
               wait_s: int = SERVE_WAIT_S, tag: str | None = None, now: float | None = None,
               git: Callable[[], tuple[str, bool]] | None = None,
               environ: Mapping[str, str] | None = None) -> ServePlan:
    """The serving job for `model` (module doc); SpecError for a refused combination."""
    if engine not in ENGINES:
        raise SpecError(f"--engine must be one of {', '.join(ENGINES)}, got {engine!r}")
    if not bucket.startswith("hf://buckets/"):
        raise SpecError(f"--bucket must look like hf://buckets/<namespace>/<name>, got {bucket!r}")
    flavor = flavor or SERVE_FLAVOR
    timeout_s = parse_duration(timeout if timeout is not None else SERVE_TIMEOUT)
    commit, dirty = (git or _launch.checkout_identity)()
    if tag is None:
        stamp = time.strftime("%Y%m%d-%H%M%S", time.gmtime(time.time() if now is None else now))
        tag = f"serve-{slug(model.rsplit('/', 1)[-1])[:30]}-{commit[:7]}-{stamp}"
    env = os.environ if environ is None else environ
    notes: list[str] = []
    common: dict[str, Any] = {
        "model": model, "engine": engine, "flavor": flavor, "tag": tag,
        "bucket": bucket.rstrip("/"), "commit": commit, "dirty": dirty, "timeout_s": timeout_s,
        "revision": revision, "wait_s": wait_s, "notes": notes,
        "missing_secrets": [s for s in ("HF_TOKEN",) if not env.get(s)]}
    if engine == "sglang":
        kit = jp.select_kit_profile(model, profile)
        script = kit.serve_scripts.get(flavor)
        if script is None:
            gb = FLAVOR_GPU_GB.get(flavor)
            raise SpecError(
                f"profile {kit.name} has no serve script for flavor {flavor!r}"
                + (f" ({gb} GB per GPU)" if gb is not None else "")
                + f": the kit needs >= {kit.min_gpu_gb} GB per replica and ships scripts per GPU "
                f"type; use --flavor {' or '.join(kit.serve_scripts)}")
        kit_path = f"{MODEL_DIR}/{kit.kit_dir}"
        p = port or kit.port
        plan = ServePlan(
            **common, profile=kit.name, image=image or kit.image,
            image_digest=kit.image_digest, port=p, served_model_name=kit.served_model_name,
            setup_argvs=[_download(model, MODEL_DIR, revision, [f"{kit.kit_dir}/*"]),
                         ["bash", f"{kit_path}/{kit.install}", kit_path],
                         _download(model, MODEL_DIR, revision)],
            ready_paths=list(kit.ready_paths), probe_extra=dict(kit.probe_extra),
            setup_env=dict(kit.setup_env),
            kit_serve_argv=["env", f"MODEL={MODEL_DIR}", f"PORT={p}", "bash",
                            f"{kit_path}/{script}"])
        if image is None:
            notes.append(f"image {kit.image} is the tag the kit was validated with; its "
                         f"Dockerfile pins {kit.image.split(':')[0]}@{kit.image_digest} (pass "
                         "--image <that> to pin it; a digest reference is untested on HF Jobs)")
        if kit.note:
            notes.append(f"{kit.name}: {kit.note}")
        return plan
    # vllm: the in-job vLLM serve logic, as a standalone server
    resolved = jp.resolve_flags(model, vllm_flags=vllm_flags, profile=profile)
    plan = ServePlan(
        **common, profile=resolved.profile, image=image or DEFAULT_IMAGE, image_digest=None,
        port=port or VLLM_SERVE_PORT, served_model_name=model,
        setup_argvs=[["uv", "pip", "install", "--system", f"vllm=={vllm_version}"]],
        ready_paths=["/health", "/v1/models"], probe_extra=resolved.probe_extra,
        setup_env={"HF_HOME": "/tmp/hf", "VLLM_USE_FLASHINFER_SAMPLER": "0"},
        vllm_version=vllm_version, vllm_flags=resolved.flags, max_model_len=max_model_len,
        gpu_memory_utilization=gpu_memory_utilization)
    gpu_gb = FLAVOR_GPU_GB.get(flavor)
    if gpu_gb == 0:
        raise SpecError(f"--flavor {flavor} has no GPU; vLLM needs one (e.g. a100-large)")
    if resolved.min_gpu_gb and gpu_gb is not None and gpu_gb < resolved.min_gpu_gb:
        notes.append(f"profile {resolved.profile} needs a {resolved.min_gpu_gb} GB GPU and "
                     f"{flavor} has {gpu_gb} GB per card")
    return plan


def describe_serve(plan: ServePlan) -> str:
    rate = plan.rate_usd_per_hour
    cost = (f"${rate:.2f}/h while it runs, ${rate * plan.timeout_s / 3600:.2f} if it runs to the "
            "timeout" if rate is not None else "rate unknown for this flavor")
    lines = ["hf jobs command:", "  " + shlex.join(hf_command(plan)), "",
             f"serve {plan.model} with {plan.engine}"
             + (f" (profile {plan.profile})" if plan.profile else "")
             + f" on {plan.flavor}, image {plan.image}",
             f"tag {plan.tag}  commit {plan.commit[:12]}{' (dirty)' if plan.dirty else ''}",
             f"stage {plan.stage_remote}/ (serve.sh, plan.json)", "setup:"]
    lines += [f"  {shlex.join(a)}" for a in plan.setup_argvs]
    lines += [f"serve: {shlex.join(plan.serve_argv)}",
              (f"ready: {' then '.join(plan.ready_paths)} on port {plan.port} within "
               f"{plan.wait_s}s, /v1/models listing {plan.served_model_name!r}; then the job "
               "waits on the server (logs SWARMLAB_SERVE_READY)"),
              (f"exposed: port {plan.port} through the jobs proxy (an HF token with read access "
               "to the job's namespace; not public)"),
              (f"timeout {fmt_duration(plan.timeout_s)}; flavor {plan.flavor}: {cost}; the job "
               "idles at that rate until the timeout or `swarmlab job stop <id>`"),
              (f"call it as <provider>:{plan.served_model_name} through an openai_compat provider "
               "with base_url_env (docs/guide/real-models.md); find the URL with `swarmlab job "
               "endpoint <id>`")]
    if plan.missing_secrets:
        lines.append(f"note: {', '.join(plan.missing_secrets)} not set in this environment; "
                     "--launch refuses until set")
    lines += [f"note: {n}" for n in plan.notes]
    return "\n".join(lines)


def serve_script() -> str:
    return (files("swarmlab.jobs") / "serve.sh").read_text()


def stage_serve(plan: ServePlan, directory: Path | str, build: Any = None) -> Path:
    """Write the serving job's stage dir locally: serve.sh and plan.json (no wheel: nothing of
    swarmlab runs in it; `build` is accepted for `launch.submit` and ignored)."""
    d = Path(directory)
    d.mkdir(parents=True, exist_ok=True)
    (d / "serve.sh").write_text(serve_script())
    (d / "plan.json").write_text(json.dumps(plan.as_dict(), indent=1, sort_keys=True))
    return d


def submit_serve(plan: ServePlan, *, hf: str = "hf", runner: Runner = run_cli,
                 ledger: Path | str | None = None) -> dict[str, Any]:
    """Stage, upload, submit; the ledger row has `kind: serve` (module doc)."""
    return submit(plan, hf=hf, runner=runner, build=None, ledger=ledger, stage_fn=stage_serve)


# ---- the endpoint --------------------------------------------------------------------------
HttpGet = Callable[[str, Mapping[str, str]], tuple[int, Any]]


def http_get(url: str, headers: Mapping[str, str]) -> tuple[int, Any]:
    """(status, parsed JSON or None); status 0 on a transport error."""
    import httpx

    try:
        r = httpx.get(url, headers=dict(headers), timeout=20.0, follow_redirects=False)
    except httpx.HTTPError:
        return 0, None
    try:
        return r.status_code, r.json()
    except ValueError:
        return r.status_code, None


def ledger_row(job_id: str, ledger: Path | str | None = DEFAULT_LEDGER) -> dict[str, Any] | None:
    """The last ledger row of `job_id` with `kind: serve`, or None."""
    if ledger is None or not Path(ledger).exists():
        return None
    row = None
    for line in Path(ledger).read_text().splitlines():
        try:
            r = json.loads(line)
        except ValueError:
            continue
        if r.get("job_id") == job_id and r.get("kind") == "serve":
            row = r
    return row


def _inspect(job_id: str, hf: str, runner: Runner) -> dict[str, Any]:
    p = runner([hf, "jobs", "inspect", job_id, "--format", "json"])
    if p.returncode != 0:
        raise RuntimeError(f"hf jobs inspect {job_id} failed ({p.returncode}): "
                           f"{((p.stdout or '') + (p.stderr or ''))[-600:]}")
    raw = json.loads(p.stdout or "{}")
    return raw[0] if isinstance(raw, list) else raw


def expose_urls(info: Mapping[str, Any]) -> list[str]:
    """The exposed URLs of an inspected job (`status.expose_urls`; camelCase tolerated)."""
    st = info.get("status") or {}
    for v in (st.get("expose_urls"), st.get("exposeUrls"), info.get("expose_urls")):
        if isinstance(v, list):
            return [str(u) for u in v]
    return []


def pick_url(urls: list[str], port: int | None) -> str | None:
    """The URL of `port` (`https://<id>--<port>.hf.jobs`), else the only one, else None."""
    if port is not None:
        for u in urls:
            if f"--{port}." in u or u.rstrip("/").endswith(f":{port}"):
                return u.rstrip("/")
    return urls[0].rstrip("/") if len(urls) == 1 else None


def resolve_endpoint(job_id: str, *, hf: str = "hf", runner: Runner = run_cli,
                     ledger: Path | str | None = DEFAULT_LEDGER, get: HttpGet = http_get,
                     environ: Mapping[str, str] | None = None) -> dict[str, Any]:
    """`{job_id, url, base_url, served_model_name, ready, stage, models, ...}` (module doc)."""
    env = os.environ if environ is None else environ
    info = _inspect(job_id, hf, runner)
    row = ledger_row(job_id, ledger) or {}
    exposed = info.get("expose")
    port = row.get("port") or (exposed[0] if isinstance(exposed, list) and exposed else None)
    stage = (info.get("status") or {}).get("stage")
    url = pick_url(expose_urls(info), port)
    name = row.get("served_model_name")
    flavor = info.get("flavor") or row.get("flavor")
    rate = FLAVOR_USD_PER_HOUR.get(flavor or "")
    secs = (info.get("durations") or {}).get("running_secs")
    out: dict[str, Any] = {
        "job_id": info.get("id", job_id), "url": url, "base_url": f"{url}/v1" if url else None,
        "served_model_name": name, "ready": False, "stage": stage, "port": port,
        "flavor": flavor, "rate_usd_per_hour": rate, "running_secs": secs,
        "cost_usd": round(rate * secs / 3600, 4) if rate is not None and secs else None,
        "models": [], "in_ledger": bool(row), "detail": None,
        "remaining_s": (int(row["timeout_s"]) - int(secs or 0)) if row.get("timeout_s") else None}
    if url is None:
        out["detail"] = (f"no exposed URL yet (stage {stage}); the job must be running with "
                         "--expose" if stage else "no exposed URL")
        return out
    headers = {"authorization": f"Bearer {env['HF_TOKEN']}"} if env.get("HF_TOKEN") else {}
    status, body = get(f"{url}/v1/models", headers)
    ids = [m.get("id") for m in (body or {}).get("data", []) if isinstance(m, dict)] \
        if isinstance(body, dict) else []
    out["models"] = ids
    if name is None and ids:
        out["served_model_name"] = name = ids[0]
    out["ready"] = status == 200 and bool(name) and name in ids
    if not out["ready"]:
        out["detail"] = (f"GET {url}/v1/models: "
                         + ("no answer" if status == 0 else f"HTTP {status}")
                         + ("" if status != 200 else f", models {ids}")
                         + ("; HF_TOKEN is not set here" if not headers else ""))
    return out


def describe_endpoint(ep: Mapping[str, Any]) -> str:
    cost = (f"${ep['cost_usd']:.2f} so far (${ep['rate_usd_per_hour']:.2f}/h x "
            f"{(ep['running_secs'] or 0) / 3600:.2f} h)" if ep.get("cost_usd") is not None
            else "cost unknown")
    lines = [f"serve job {ep['job_id']}: stage {ep['stage']}, flavor {ep['flavor']}, {cost}",
             (f"url {ep['url'] or '-'}  served model {ep['served_model_name'] or '?'}  "
              f"ready {str(ep['ready']).lower()}")]
    if ep.get("detail"):
        lines.append(f"  {ep['detail']}")
    if ep.get("base_url"):
        lines.append(f"run against it: swarmlab job run SPEC --endpoint {ep['job_id']} "
                     f"(locally: export SWARMLAB_ENDPOINT_URL={ep['base_url']})")
    if not ep.get("in_ledger"):
        lines.append("  (not in the local ledger: the served name is the server's first model)")
    return "\n".join(lines)


def stop(job_id: str, *, hf: str = "hf", runner: Runner = run_cli) -> str:
    """`hf jobs cancel <id>`; returns its output."""
    p = runner([hf, "jobs", "cancel", job_id])
    out = (p.stdout or "") + (p.stderr or "")
    if p.returncode != 0:
        raise RuntimeError(f"hf jobs cancel {job_id} failed ({p.returncode}): {out[-600:]}")
    return out.strip()
