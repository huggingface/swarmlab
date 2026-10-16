"""Build (and on request submit) an HF Job that serves a model with vLLM and runs a spec against it.

HF Jobs have no inbound networking, so the model server and the swarmlab runner share one job
(DESIGN.md §12: "one job hosts a wave of runs and vLLM when self-hosting"). The pieces:

- `plan_job(spec, model=..., flavor=..., ...) -> JobPlan` (pure, no I/O beyond reading the spec
  and asking git for the commit): validates that every model-backed participant (and probe coder)
  of the chosen arms is `vllm:<model>` with `<model>` == `--model`, builds the **job spec** (the
  YAML with a `providers: {vllm: ...}` override pointing at `http://127.0.0.1:<port>/v1`,
  pricing `(0, 0, 0)`, `self_hosted: true`, a high concurrency and a long per-request timeout),
  checks that every chosen arm of the job spec builds offline, and computes the job tag, the
  bucket paths, the environment and the wall-time estimate.
- `hf_command(plan) -> list[str]`: the exact `hf jobs run` argv. The job's command fetches the
  stage dir (`<bucket>/jobs/<tag>/`: wheel, spec.yaml, bootstrap.sh) with a throwaway hf CLI
  (`uv tool run`; the uv image ships uv) and runs `bash bootstrap.sh` (no exec bit needed). No
  bucket is mounted: run dirs live on local disk and are mirrored with `hf buckets sync`.
- `stage(plan, dir, build_wheel=...)`: writes the stage dir locally: `spec.yaml` (job spec),
  `spec.orig.yaml` (the spec as given), `bootstrap.sh`, `plan.json` and the wheel (`uv build
  --wheel` of the checkout this package was imported from).
- `serve_argv(plan)`: the exact `vllm serve` argv (`["vllm", "serve", model, *launcher_flags,
  *shlex.split(vllm_flags)]`). plan.json carries it as `vllm_serve_argv` (a JSON list) next to
  `vllm_serve_command` (its `shlex.join`), and bootstrap.sh executes that list as read from
  plan.json (NUL-delimited, so a quoted token with spaces or newlines stays one argument), so
  the printed command and the executed argv cannot drift.
- `submit(plan, hf=...)`: stage into a temp dir, upload it, submit, parse the job id, upload
  `launch.json` next to it and append a row to the local ledger. Only `swarmlab job run
  --launch` calls it.

Install path: a wheel built locally at the launching commit and uploaded to the bucket (the repo
is private, so a `git+https` install would need a GitHub token in the job; the wheel needs only
the HF token the job already has for the bucket). The commit is passed as `SWARMLAB_GIT_COMMIT`,
which `git_identity` reads when not inside a checkout, so run.json records it.

Image: `ghcr.io/astral-sh/uv:python3.12-bookworm` + `uv pip install --system vllm==0.30.0`: the
combination that served Qwen/Qwen3.5-9B with the qwen3 reasoning parser and the qwen3_coder
tool parser in the textworld study (a100-large, server up in ~5 min including the install).

Serve flags: `swarmlab/jobs/profiles.py`. `plan_job(vllm_flags=None, profile="auto")` uses
explicit flags if given, else the built-in profile for `--model`, else `DEFAULT_VLLM_FLAGS` (the
Qwen3.5 flags above); `vllm_flags_append` is added to the winner. The readiness probe's request
extra resolves separately: `plan_job(probe_extra={...})` (`--probe-extra`; `{}` sends none) >
the profile's > `DEFAULT_PROBE_EXTRA` only with the default flags under `--profile auto`, else `{}`
(`profiles.resolve_flags` has the details). The plan records the effective flags, where they came
from (`vllm_flags_source`), the profile, the probe extra and where it came from (`probe_extra`,
`probe_extra_source`: `--probe-extra | profile | default | none`, sent as `VLLM_PROBE_EXTRA` /
`VLLM_PROBE_EXTRA_SOURCE` and recorded in the job's job.json), and `describe` prints them with the
flags the launcher always adds (`LAUNCHER_FLAGS`).

Wall-time estimate: `setup_s` (default 600: install + weights download + vLLM start) plus, per
run, `rounds * per_round_s(N)` with `per_round_s(N) = 30 + 2 * N` seconds (a guess: the
router-served Qwen phase-1 runs took ~1 min per round at N=16; a local server should not be
slower). `--per-round` overrides it. With `parallel=N` (`--parallel`, sent as `SWARMLAB_PARALLEL`;
default 1, and then the env and the command are as before) the job runs up to N runs at once
against the one server: bootstrap.sh starts the runs in plan order as slots free up, and the
estimate is the setup plus the makespan of that schedule, assuming concurrent runs do not slow
each other (one vLLM server batches their requests; raise `--per-round` if they contend). The
default timeout is 1.5x the estimate, rounded up to 10 minutes.
"""
from __future__ import annotations

import copy
import hashlib
import json
import math
import re
import shlex
import shutil
import subprocess
import tempfile
import time
from collections.abc import Callable, Sequence
from dataclasses import asdict, dataclass, field
from importlib.resources import files
from pathlib import Path
from typing import Any

import yaml

from ..spec import SpecError, experiment_seeds, git_identity, load_experiment_yaml
from .profiles import DEFAULT_VLLM_FLAGS, LAUNCHER_FLAGS, resolve_flags  # noqa: F401 (re-export)

DEFAULT_BUCKET = "hf://buckets/cmpatino/swarmlab-runs"
DEFAULT_IMAGE = "ghcr.io/astral-sh/uv:python3.12-bookworm"
DEFAULT_VLLM_VERSION = "0.30.0"
DEFAULT_MAX_MODEL_LEN = 262144
DEFAULT_GPU_MEMORY_UTILIZATION = 0.9
DEFAULT_PORT = 8000
DEFAULT_SYNC_EVERY_S = 300
DEFAULT_SETUP_S = 600
DEFAULT_CONCURRENCY = 64
DEFAULT_REQUEST_TIMEOUT_S = 600.0
STAGE_LOCAL = "/opt/swarmlab/stage"
# `hf jobs hardware`, 2026-10-06 (USD per hour); unknown flavors print "?"
FLAVOR_USD_PER_HOUR = {
    "cpu-basic": 0.01, "cpu-upgrade": 0.03, "t4-small": 0.40, "t4-medium": 0.60,
    "l4x1": 0.80, "l4x4": 3.80, "a10g-small": 1.00, "a10g-large": 1.50, "a10g-largex2": 3.00,
    "a10g-largex4": 5.00, "l40sx1": 1.80, "l40sx4": 8.30, "a100-large": 2.50, "a100x4": 10.00,
    "a100x8": 20.00, "h200": 5.00, "h200x2": 10.00, "rtx-pro-6000": 2.75,
}
# memory of ONE GPU in the flavor (GB); the profiles run TP=1, so a multi-GPU flavor does not add
# memory for the model. Unknown flavors get no size note.
FLAVOR_GPU_GB = {
    "cpu-basic": 0, "cpu-upgrade": 0, "t4-small": 16, "t4-medium": 16,
    "l4x1": 24, "l4x4": 24, "a10g-small": 24, "a10g-large": 24, "a10g-largex2": 24,
    "a10g-largex4": 24, "l40sx1": 48, "l40sx4": 48, "a100-large": 80, "a100x4": 80,
    "a100x8": 80, "h200": 141, "h200x2": 141, "rtx-pro-6000": 96,
}
JOB_ID_RE = re.compile(r"\b([0-9a-f]{24})\b")


# ---- small helpers ----------------------------------------------------------------------------
def parse_duration(s: str | int) -> int:
    """'90s' / '30m' / '2h' / '1d' / '1.5h' / bare seconds -> seconds."""
    text = str(s).strip().lower()
    units = {"s": 1, "m": 60, "h": 3600, "d": 86400}
    try:
        if text and text[-1] in units:
            return int(float(text[:-1]) * units[text[-1]])
        return int(float(text))
    except ValueError as e:
        raise SpecError(f"bad duration {s!r}: use e.g. 90m, 2h, 3600") from e


def fmt_duration(secs: int) -> str:
    if secs % 3600 == 0:
        return f"{secs // 3600}h"
    if secs % 60 == 0:
        return f"{secs // 60}m"
    return f"{secs}s"


def slug(text: str) -> str:
    return re.sub(r"[^a-z0-9-]+", "-", text.lower()).strip("-") or "swarmlab"


def per_round_guess(n_agents: int) -> float:
    return 30.0 + 2.0 * n_agents


def arm_models(doc: dict, arm: str) -> list[str]:
    """Every model string an arm calls: participants' `params.model`, probes' `coder_model`."""
    a = doc["arms"][arm]
    out = [g["params"]["model"] for g in a["participants"]
           if isinstance(g["params"].get("model"), str)]
    out += [p["params"]["coder_model"] for p in a["probes"]
            if isinstance(p["params"].get("coder_model"), str)]
    return out


def arm_agents(doc: dict, arm: str) -> int:
    return sum(int(g.get("count", 1)) for g in doc["arms"][arm]["participants"])


def arm_rounds(doc: dict, arm: str) -> int:
    opts = {**doc.get("options", {}), **doc["arms"][arm].get("options", {})}
    if "max_rounds" not in opts:
        raise SpecError(f"arm {arm!r}: no max_rounds in options")
    return int(opts["max_rounds"])


def check_vllm_models(doc: dict, arms: Sequence[str], model: str) -> None:
    """Every model the arms call must be `vllm:<model>`; at least one must exist."""
    problems = []
    seen = 0
    for arm in arms:
        for m in arm_models(doc, arm):
            seen += 1
            prefix, _, mid = m.partition(":")
            if prefix != "vllm":
                problems.append(f"arm {arm!r} uses {m!r}: only vllm:{model} is served in the job "
                                "(the job has no other provider keys)")
            elif mid != model:
                problems.append(f"arm {arm!r} uses {m!r} but --model is {model!r}")
    if problems:
        raise SpecError("; ".join(problems))
    if not seen:
        raise SpecError(f"arms {list(arms)} call no model; nothing for vLLM to serve")


def vllm_provider(model: str, port: int = DEFAULT_PORT, concurrency: int = DEFAULT_CONCURRENCY,
                  timeout_s: float = DEFAULT_REQUEST_TIMEOUT_S) -> dict:
    """The `providers.vllm` override injected into the job spec."""
    return {"type": "openai_compat", "params": {
        "name": "vllm", "base_url": f"http://127.0.0.1:{port}/v1",
        "pricing": {model: [0.0, 0.0, 0.0]}, "concurrency": concurrency,
        "timeout_s": float(timeout_s), "self_hosted": True}}


def job_spec_doc(doc: dict, model: str, arms: Sequence[str] | None = None,
                 **provider_kw: Any) -> dict:
    """The spec the job runs: the in-job vllm provider, and only the chosen `arms` (when given).

    Arms the job does not run are dropped: they may call other models, which the job does not
    serve or price, and the job validates every arm of its spec before it starts."""
    out = copy.deepcopy(doc)
    if arms is not None:
        out["arms"] = {a: out["arms"][a] for a in out["arms"] if a in set(arms)}
    out["providers"] = {**out.get("providers", {}), "vllm": vllm_provider(model, **provider_kw)}
    return out


def dump_yaml(doc: dict) -> str:
    return yaml.safe_dump(doc, sort_keys=False, default_flow_style=False, allow_unicode=True)


# ---- the plan ---------------------------------------------------------------------------------
@dataclass
class JobPlan:
    spec_path: str
    name: str
    tag: str
    model: str
    flavor: str
    image: str
    bucket: str
    commit: str
    dirty: bool
    runs: list[tuple[str, int]]
    timeout_s: int
    estimate_s: int
    estimate_detail: dict[str, Any]
    vllm_version: str
    vllm_flags: str
    max_model_len: int
    gpu_memory_utilization: float
    revision: str | None
    sync_every_s: int
    job_spec_yaml: str
    orig_spec_yaml: str
    notes: list[str] = field(default_factory=list)
    vllm_flags_source: str = "default"
    vllm_profile: str | None = None
    probe_extra: dict[str, Any] = field(default_factory=dict)
    probe_extra_source: str = "default"   # "--probe-extra" | "profile" | "default" | "none"
    port: int = DEFAULT_PORT   # vLLM's port; the job spec's provider URL points at it
    parallel: int = 1   # runs at once in the job (SWARMLAB_PARALLEL)

    @property
    def stage_remote(self) -> str:
        return f"{self.bucket}/jobs/{self.tag}"

    @property
    def runs_remote(self) -> str:
        return f"{self.bucket}/runs/{self.tag}"

    @property
    def job_spec_sha256(self) -> str:
        return hashlib.sha256(self.job_spec_yaml.encode()).hexdigest()

    @property
    def rate_usd_per_hour(self) -> float | None:
        return FLAVOR_USD_PER_HOUR.get(self.flavor)

    @property
    def vllm_serve_argv(self) -> list[str]:
        return serve_argv(self)

    @property
    def vllm_serve_command(self) -> str:
        return serve_command(self)

    def env(self) -> dict[str, str]:
        env = {
            "SWARMLAB_TAG": self.tag,
            "SWARMLAB_STAGE": self.stage_remote,
            "SWARMLAB_RUNS_REMOTE": self.runs_remote,
            "SWARMLAB_RUNS": " ".join(f"{a}:{s}" for a, s in self.runs),
            "SWARMLAB_GIT_COMMIT": self.commit,
            "SWARMLAB_GIT_DIRTY": "1" if self.dirty else "0",
            "SWARMLAB_JOB_SPEC_SHA256": self.job_spec_sha256,
            "SWARMLAB_FLAVOR": self.flavor,
            "SWARMLAB_IMAGE": self.image,
            "SWARMLAB_SYNC_EVERY": str(self.sync_every_s),
            "VLLM_MODEL": self.model,
            "VLLM_PORT": str(self.port),
            "VLLM_VERSION": self.vllm_version,
            "VLLM_MAX_MODEL_LEN": str(self.max_model_len),
            "VLLM_GPU_MEMORY_UTILIZATION": f"{self.gpu_memory_utilization:g}",
            "VLLM_FLAGS": self.vllm_flags,
            "VLLM_FLAGS_SOURCE": self.vllm_flags_source,
            "VLLM_PROFILE": self.vllm_profile or "none",
            "VLLM_PROBE_EXTRA": json.dumps(self.probe_extra, separators=(",", ":"),
                                           sort_keys=True),
            "VLLM_PROBE_EXTRA_SOURCE": self.probe_extra_source,
            "VLLM_WAIT_S": "1200",
            "HF_HOME": "/tmp/hf",
        }
        if self.revision:
            env["VLLM_REVISION"] = self.revision
        if self.parallel > 1:
            env["SWARMLAB_PARALLEL"] = str(self.parallel)
        return env

    def as_dict(self) -> dict[str, Any]:
        d = asdict(self)
        d.pop("job_spec_yaml")
        d.pop("orig_spec_yaml")
        d["runs"] = [f"{a}:{s}" for a, s in self.runs]
        d.update(stage_remote=self.stage_remote, runs_remote=self.runs_remote,
                 job_spec_sha256=self.job_spec_sha256, rate_usd_per_hour=self.rate_usd_per_hour,
                 timeout=fmt_duration(self.timeout_s), hf_command=hf_command(self),
                 vllm_serve_argv=self.vllm_serve_argv,
                 vllm_serve_command=self.vllm_serve_command)
        return d


def plan_job(
    spec: Path | str,
    *,
    model: str,
    flavor: str = "a100-large",
    arms: Sequence[str] | None = None,
    seeds: Sequence[int] | None = None,
    timeout: str | int | None = None,
    bucket: str = DEFAULT_BUCKET,
    image: str = DEFAULT_IMAGE,
    vllm_version: str = DEFAULT_VLLM_VERSION,
    vllm_flags: str | None = None,
    profile: str | None = "auto",
    vllm_flags_append: str | None = None,
    probe_extra: dict[str, Any] | None = None,
    max_model_len: int = DEFAULT_MAX_MODEL_LEN,
    gpu_memory_utilization: float = DEFAULT_GPU_MEMORY_UTILIZATION,
    revision: str | None = None,
    port: int = DEFAULT_PORT,
    concurrency: int = DEFAULT_CONCURRENCY,
    request_timeout_s: float = DEFAULT_REQUEST_TIMEOUT_S,
    sync_every_s: int = DEFAULT_SYNC_EVERY_S,
    per_round_s: float | None = None,
    setup_s: int = DEFAULT_SETUP_S,
    parallel: int = 1,
    tag: str | None = None,
    now: float | None = None,
    git: Callable[[], tuple[str, bool]] | None = None,
) -> JobPlan:
    spec = Path(spec)
    doc = load_experiment_yaml(spec)
    chosen = list(arms) if arms else list(doc["arms"])
    unknown = [a for a in chosen if a not in doc["arms"]]
    if unknown:
        raise SpecError(f"{spec}: unknown arm(s) {unknown}; available: {list(doc['arms'])}")
    seed_list = list(seeds) if seeds else experiment_seeds(doc)
    check_vllm_models(doc, chosen, model)
    if not bucket.startswith("hf://buckets/"):
        raise SpecError(f"--bucket must look like hf://buckets/<namespace>/<name>, got {bucket!r}")
    bucket = bucket.rstrip("/")
    if parallel < 1:
        raise SpecError(f"--parallel must be >= 1, got {parallel}")

    resolved = resolve_flags(model, vllm_flags=vllm_flags, profile=profile,
                             append=vllm_flags_append, probe_extra=probe_extra)
    notes = []
    if resolved.validated_vllm and resolved.validated_vllm != vllm_version:
        notes.append(f"profile {resolved.profile} was validated on vLLM {resolved.validated_vllm}; "
                     f"this job installs {vllm_version} (pass --vllm-version "
                     f"{resolved.validated_vllm} to reproduce the validated setup)")
    if resolved.profile and max_model_len == DEFAULT_MAX_MODEL_LEN:
        notes.append(f"--max-model-len {max_model_len} is the Qwen3.5 default; small GPUs and "
                     "models with a shorter context need a lower one (e.g. --max-model-len 32768)")
    if resolved.source == "--vllm-flags" and resolved.profile:
        notes.append(f"--vllm-flags replaces the {resolved.profile} profile's flags; the "
                     + ("readiness probe still uses the profile's request extra"
                        if resolved.probe_extra_source == "profile"
                        else "readiness probe uses the --probe-extra request extra"))
    gpu_gb = FLAVOR_GPU_GB.get(flavor)
    if resolved.min_gpu_gb and gpu_gb is not None and gpu_gb < resolved.min_gpu_gb:
        fits = sorted((f for f, gb in FLAVOR_GPU_GB.items() if gb >= resolved.min_gpu_gb),
                      key=lambda f: FLAVOR_USD_PER_HOUR.get(f, float("inf")))
        notes.append(f"profile {resolved.profile} needs a {resolved.min_gpu_gb} GB GPU at "
                     f"--max-model-len 32768 (the validated setting) and {flavor} has {gpu_gb} GB "
                     "per card: vLLM will run out of KV cache and exit before it is ready; use "
                     f"{' or '.join(fits[:2])} (a much shorter --max-model-len is unvalidated)")
    if "--tool-call-parser" not in resolved.flags or "--enable-auto-tool-choice" not in resolved.flags:
        notes.append("the serve flags lack --enable-auto-tool-choice/--tool-call-parser: tool calls "
                     "will not be parsed and agent turns will be prose")
    if "vllm" in doc.get("providers", {}):
        notes.append("the spec's own providers.vllm is replaced by the in-job server override")
    job_doc = job_spec_doc(doc, model, arms=chosen, port=port, concurrency=concurrency,
                           timeout_s=request_timeout_s)
    job_yaml = dump_yaml(job_doc)
    _check_job_spec(job_yaml, chosen)

    commit, dirty = (git or checkout_identity)()
    if tag is None:
        stamp = time.strftime("%Y%m%d-%H%M%S", time.gmtime(time.time() if now is None else now))
        tag = f"{slug(doc['name'])}-{commit[:7]}-{stamp}"

    runs = [(a, int(s)) for a in chosen for s in seed_list]
    per_arm = {}
    for a in chosen:
        n = arm_agents(doc, a)
        rounds = arm_rounds(doc, a)
        pr = per_round_s if per_round_s is not None else per_round_guess(n)
        per_arm[a] = {"agents": n, "rounds": rounds, "per_round_s": pr,
                      "per_run_s": int(rounds * pr), "runs": len(seed_list)}
    slots = [0] * parallel   # bootstrap.sh's schedule: each run takes the first free slot
    for a, _ in runs:
        slots[slots.index(min(slots))] += per_arm[a]["per_run_s"]
    total = setup_s + max(slots)
    if parallel > 1:
        notes.append(f"up to {parallel} runs at once share the one vLLM server; the estimate "
                     "assumes they do not slow each other (raise --per-round if they contend)")
    timeout_s = (parse_duration(timeout) if timeout is not None
                 else math.ceil(total * 1.5 / 600.0) * 600)
    if timeout_s < total:
        notes.append(f"timeout {fmt_duration(timeout_s)} is below the {total // 60} min estimate; "
                     "the job may be cut (runs synced so far stay in the bucket, resumable)")
    if dirty:
        notes.append("the checkout has uncommitted changes: the wheel will not match the commit")
    return JobPlan(
        spec_path=str(spec), name=doc["name"], tag=tag, model=model, flavor=flavor, image=image,
        bucket=bucket, commit=commit, dirty=dirty, runs=runs, timeout_s=timeout_s,
        estimate_s=total, estimate_detail={"setup_s": setup_s, "parallel": parallel,
                                           "arms": per_arm},
        vllm_version=vllm_version, vllm_flags=resolved.flags, max_model_len=max_model_len,
        gpu_memory_utilization=gpu_memory_utilization, revision=revision,
        sync_every_s=sync_every_s, job_spec_yaml=job_yaml, orig_spec_yaml=spec.read_text(),
        notes=notes, vllm_flags_source=resolved.source, vllm_profile=resolved.profile,
        probe_extra=resolved.probe_extra, probe_extra_source=resolved.probe_extra_source,
        port=port, parallel=parallel,
    )


def checkout_identity() -> tuple[str, bool]:
    """(commit, dirty) of the checkout this package runs from, ignoring SWARMLAB_GIT_COMMIT."""
    import os

    saved = os.environ.pop("SWARMLAB_GIT_COMMIT", None)
    try:
        return git_identity(Path(__file__).resolve().parent)
    finally:
        if saved is not None:
            os.environ["SWARMLAB_GIT_COMMIT"] = saved


def _check_job_spec(job_yaml: str, arms: Sequence[str]) -> None:
    """Every chosen arm of the job spec must build (offline: the vllm provider is fully priced)."""
    from ..experiment import Experiment

    with tempfile.TemporaryDirectory() as d:
        path = Path(d) / "spec.yaml"
        path.write_text(job_yaml)
        for a in arms:
            try:
                Experiment.from_yaml(path, a)
            except SpecError:
                raise
            except (ValueError, TypeError, ImportError, AttributeError) as e:
                raise SpecError(f"job spec, arm {a!r}: {type(e).__name__}: {e}") from e


# ---- the command ------------------------------------------------------------------------------
def job_entry(plan: JobPlan) -> str:
    """The shell line the job runs: fetch the stage dir, then the bootstrap script."""
    return (f"mkdir -p {STAGE_LOCAL} && uv tool run --from 'huggingface_hub>=1.32' "
            f"hf buckets sync \"$SWARMLAB_STAGE\" {STAGE_LOCAL} && "
            f"exec bash {STAGE_LOCAL}/bootstrap.sh")


def launcher_flags(plan: JobPlan) -> list[str]:
    """The flags the launcher always passes to `vllm serve` (never part of `vllm_flags`)."""
    out = ["--served-model-name", plan.model, "--host", "127.0.0.1", "--port", str(plan.port),
           "--max-model-len", str(plan.max_model_len),
           "--gpu-memory-utilization", f"{plan.gpu_memory_utilization:g}"]
    if plan.revision:
        out = ["--revision", plan.revision, *out]
    assert {f for f in out if f.startswith("--")} <= set(LAUNCHER_FLAGS)
    return out


def serve_argv(plan: JobPlan) -> list[str]:
    """The exact `vllm serve` argv the job runs: plan.json's `vllm_serve_argv`, which bootstrap.sh
    executes as is. The launcher flags go first, then the user flags (shell-split)."""
    return ["vllm", "serve", plan.model, *launcher_flags(plan), *shlex.split(plan.vllm_flags)]


def serve_command(plan: JobPlan) -> str:
    """`serve_argv` shell-quoted (for the printout, plan.json and the job's job.json)."""
    return shlex.join(serve_argv(plan))


def hf_command(plan: JobPlan, hf: str = "hf") -> list[str]:
    cmd = [hf, "jobs", "run", "--detach", "--flavor", plan.flavor,
           "--timeout", fmt_duration(plan.timeout_s), "--secrets", "HF_TOKEN",
           "--name", f"swarmlab-{plan.tag}"[:80],
           "--label", "experiment=swarmlab", "--label", f"swarmlab_tag={plan.tag}"]
    for k, v in plan.env().items():
        cmd += ["-e", f"{k}={v}"]
    return [*cmd, plan.image, "bash", "-c", job_entry(plan)]


def describe(plan: JobPlan) -> str:
    """Human summary: the command, the estimate, the cost bounds, the notes."""
    rate = plan.rate_usd_per_hour
    est_h = plan.estimate_s / 3600
    lines = ["hf jobs command:", "  " + shlex.join(hf_command(plan)), "",
             f"tag {plan.tag}  commit {plan.commit[:12]}{' (dirty)' if plan.dirty else ''}",
             f"stage {plan.stage_remote}/   runs -> {plan.runs_remote}/",
             f"runs ({len(plan.runs)}): " + ", ".join(f"{a}:s{s}" for a, s in plan.runs),
             (f"vllm {plan.vllm_version}: profile {plan.vllm_profile or 'none'}, "
              f"flags from {plan.vllm_flags_source}"),
             f"  flags: {plan.vllm_flags}",
             f"  launcher adds: {shlex.join(launcher_flags(plan))}",
             (f"  probe extra: {json.dumps(plan.probe_extra, sort_keys=True)} "
              f"(from {plan.probe_extra_source})")]
    for a, d in plan.estimate_detail["arms"].items():
        lines.append(f"  {a}: N={d['agents']} x {d['rounds']} rounds x ~{d['per_round_s']:g} s/round"
                     f" = ~{d['per_run_s'] / 60:.0f} min/run x {d['runs']} seed(s)")
    cost = (f"~${rate * est_h:.2f} expected, ${rate * plan.timeout_s / 3600:.2f} worst case "
            f"(${rate:.2f}/h x timeout)" if rate is not None else "rate unknown for this flavor")
    at_once = f", {plan.parallel} runs at once" if plan.parallel > 1 else ""
    lines.append(f"estimate: ~{plan.estimate_s / 60:.0f} min wall (incl. "
                 f"{plan.estimate_detail['setup_s'] // 60} min setup{at_once}); timeout "
                 f"{fmt_duration(plan.timeout_s)}; flavor {plan.flavor}: {cost}")
    lines.append("model calls are self-hosted: the ledger shows $0, the cost is the job's compute")
    lines += [f"note: {n}" for n in plan.notes]
    return "\n".join(lines)


# ---- staging and launch -----------------------------------------------------------------------
def repo_root() -> Path:
    return Path(__file__).resolve().parents[2]


def build_wheel(out_dir: Path) -> Path:
    """`uv build --wheel` of the checkout this package lives in; returns the wheel path."""
    root = repo_root()
    if not (root / "pyproject.toml").exists():
        raise RuntimeError(f"{root} is not a swarmlab checkout; cannot build a wheel")
    subprocess.run(["uv", "build", "--wheel", "--out-dir", str(out_dir), str(root)],
                   check=True, capture_output=True, text=True)
    wheels = sorted(out_dir.glob("swarmlab-*.whl"))
    if not wheels:
        raise RuntimeError(f"uv build produced no wheel in {out_dir}")
    return wheels[-1]


def bootstrap_script() -> str:
    return (files("swarmlab.jobs") / "bootstrap.sh").read_text()


def stage(plan: JobPlan, directory: Path | str,
          build: Callable[[Path], Path] | None = build_wheel) -> Path:
    """Write the stage dir locally (wheel only when `build` is given)."""
    d = Path(directory)
    d.mkdir(parents=True, exist_ok=True)
    (d / "spec.yaml").write_text(plan.job_spec_yaml)
    (d / "spec.orig.yaml").write_text(plan.orig_spec_yaml)
    (d / "bootstrap.sh").write_text(bootstrap_script())
    (d / "plan.json").write_text(json.dumps(plan.as_dict(), indent=1, sort_keys=True))
    if build is not None:
        build(d)
    return d


Runner = Callable[[list[str]], subprocess.CompletedProcess]


def run_cli(args: list[str]) -> subprocess.CompletedProcess:
    return subprocess.run(args, capture_output=True, text=True, check=False)


def submit(plan: JobPlan, *, hf: str = "hf", runner: Runner = run_cli,
           build: Callable[[Path], Path] | None = build_wheel,
           ledger: Path | str | None = None) -> dict[str, Any]:
    """Stage, upload, submit. Returns `{"job_id", "url", "tag", ...}`; raises on failure."""
    with tempfile.TemporaryDirectory() as tmp:
        d = stage(plan, Path(tmp) / plan.tag, build=build)
        p = runner([hf, "buckets", "sync", str(d), plan.stage_remote])
        if p.returncode != 0:
            raise RuntimeError(f"upload of the stage dir failed: {(p.stdout + p.stderr)[-500:]}")
    p = runner(hf_command(plan, hf))
    out = (p.stdout or "") + (p.stderr or "")
    if p.returncode != 0:
        raise RuntimeError(f"hf jobs run failed ({p.returncode}): {out[-800:]}")
    m = JOB_ID_RE.search(out)
    if not m:
        raise RuntimeError("could not parse a job id from `hf jobs run`; the job may have "
                           f"launched: check `hf jobs ls --label swarmlab_tag={plan.tag}`\n{out}")
    job_id = m.group(1)
    row = {"kind": "launch", "ts": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
           "job_id": job_id, "url": f"https://huggingface.co/jobs/{_namespace(plan)}/{job_id}",
           "tag": plan.tag, "flavor": plan.flavor, "timeout_s": plan.timeout_s,
           "estimate_s": plan.estimate_s, "rate_usd_per_hour": plan.rate_usd_per_hour,
           "commit": plan.commit, "runs": [f"{a}:{s}" for a, s in plan.runs],
           "stage_remote": plan.stage_remote, "runs_remote": plan.runs_remote}
    with tempfile.TemporaryDirectory() as tmp:
        f = Path(tmp) / "launch.json"
        f.write_text(json.dumps(row, indent=1))
        runner([hf, "buckets", "cp", str(f), f"{plan.stage_remote}/launch.json"])
    if ledger is not None:
        path = Path(ledger)
        path.parent.mkdir(parents=True, exist_ok=True)
        with path.open("a") as fh:
            fh.write(json.dumps(row, sort_keys=True) + "\n")
    return row


def _namespace(plan: JobPlan) -> str:
    return plan.bucket.removeprefix("hf://buckets/").split("/", 1)[0]


def which_hf() -> str:
    return shutil.which("hf") or "hf"
