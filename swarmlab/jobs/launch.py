"""Build (and on request submit) an HF Job that runs a spec: next to a vLLM server, or against API models.

Two modes (`JobPlan.mode`), chosen by `plan_job` from the spec:

- `vllm` (`--model` given, or some model-backed participant or probe coder of the chosen arms
  calls a `vllm:` model): the job serves the model with vLLM and the runs call it. HF Jobs have
  no inbound networking, so the model server and the swarmlab runner share one job (DESIGN.md
  §12: "one job hosts a wave of runs and vLLM when self-hosting"). Everything below up to "API
  mode" describes this mode, which is unchanged.
- `api` (no `--model` and no `vllm:` model in the chosen arms): no vLLM at all; see "API mode".

The pieces:

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
slower). `--per-round` overrides it. The default timeout is 1.5x the estimate, rounded up to
10 minutes. An arm's rounds are its `max_rounds`, or `rounds_per_agent` x agents when the options
(top level, the arm's merged over them) set it, as `Experiment` resolves it. Under `commit: async`
they are ticks (one turn of one agent each): `--per-round` is then seconds per tick, the plan marks
such an arm `unit: tick` and the printout says "ticks"; the default guess stays the per-round one,
so it is pessimistic there.

Parallel runs (both modes): with `parallel=N` (`--parallel`, sent as `SWARMLAB_PARALLEL`; default
1, and then the env and the command are as before) the job runs up to N runs at once:
bootstrap.sh starts the runs in plan order as slots free up, and the estimate is the setup plus
the makespan of that schedule (`estimate_detail["parallel"]`), assuming concurrent runs do not
slow each other (in vllm mode one server batches their requests; raise `--per-round` if they
contend). In API mode the concurrent runs call their providers at once (each run keeps its own
budget). A plan with a programbench run refuses `parallel > 1` (the run's lockdown and
`/workspace` are job-wide).

API mode
--------
`plan_job(spec)` without `model` when no chosen arm calls a `vllm:` model. The plan has no vLLM
pieces (no providers override, no `VLLM_*` env, no serve argv, no profile; `vllm_serve_argv` is
None in plan.json) and `describe` prints `mode: api`. The job spec is the chosen arms of the
spec as given (its own `providers:` kept).

- Image: `image=` (`--image`), else the task image `programbench/<task>:task_cleanroom_v6` when the
  first chosen arm's world is `programbench`, else `DEFAULT_IMAGE`. A programbench arm whose world
  `backend` is not `local` is refused (a job cannot run Docker; the spec is not rewritten), and so
  are chosen arms naming different programbench tasks (one job runs one image) and a plan with
  more than one programbench run (the task workspace is the image's `/workspace`, which nothing
  resets between runs: one job per arm x seed).
- Flavor: `cpu-upgrade` by default (`vllm` mode keeps `a100-large`).
- Timeout and estimate: a programbench run counts its world's `wall_limit_s` (the 6 h limit),
  plus `API_SLACK_S` (1 h: setup, the final snapshot, the sync), so the official limit gives 7 h;
  other arms use the per-round estimate as in vllm mode (1.5x, rounded up to 10 minutes).
- Secrets (`JobPlan.secrets`, names only, in plan.json and the job's job.json): `HF_TOKEN` (the
  bootstrap's bucket access), plus each model's key variable: the Anthropic provider's
  `ANTHROPIC_API_KEY`, or `ANTHROPIC_KEY` when only that one is set here (the provider reads
  either), and an OpenAI-compatible provider's `api_key_env`. Each is passed as
  `hf jobs run --secrets NAME`, so the hf CLI reads the value from the launching environment; no
  value is ever on the command line, in the plan or in the stage dir. A name unset here is listed
  in `missing_secrets`; `describe` warns and `submit` refuses.
- Extras (`JobPlan.extras`): the wheel is installed as `swarmlab[<extras>]`: `anthropic` when an
  arm calls an Anthropic model, `programbench` for a programbench world (its run-start task check).
  The `mini_swe_agent` participant needs no extra: its agent loop is vendored.
- No volume: the hidden ProgramBench tests must never be in the agent's container (they are
  mounted only in the grade job, `swarmlab/jobs/programbench.py`).
- The entry (`job_entry`) installs uv first when the image lacks it (the task image has Python
  3.10, pip and curl, no uv): `curl -LsSf https://astral.sh/uv/install.sh` into
  `/opt/swarmlab/uv`, without touching PATH or shell profiles, then fetches the stage dir with
  that uv and passes it to bootstrap.sh as `SWARMLAB_UV`. bootstrap.sh installs a managed Python
  3.12 next to it (`uv python install 3.12` into `/opt/swarmlab/python`), creates the venv and
  skips everything vLLM; `SWARMLAB_MODE=api` selects that branch.
- The job runs as root; the programbench world's lockdown happens at the first `bash` command
  (world code, not the bootstrap), and keeps the Hub hosts the bucket sync needs.
"""
from __future__ import annotations

import copy
import hashlib
import json
import math
import os
import re
import shlex
import shutil
import subprocess
import tempfile
import time
from collections.abc import Callable, Mapping, Sequence
from dataclasses import asdict, dataclass, field
from importlib.resources import files
from pathlib import Path
from typing import Any

import yaml

from ..spec import SpecError, experiment_seeds, git_identity, load_experiment_yaml, resolve_rounds
from .profiles import DEFAULT_VLLM_FLAGS, LAUNCHER_FLAGS, resolve_flags  # noqa: F401 (re-export)

DEFAULT_BUCKET = "hf://buckets/cmpatino/swarmlab-runs"
DEFAULT_IMAGE = "ghcr.io/astral-sh/uv:python3.12-bookworm"
DEFAULT_VLLM_VERSION = "0.30.0"
DEFAULT_MAX_MODEL_LEN = 262144
DEFAULT_GPU_MEMORY_UTILIZATION = 0.9
DEFAULT_PORT = 8000
DEFAULT_SYNC_EVERY_S = 300
DEFAULT_SETUP_S = 600
VLLM_FLAVOR = "a100-large"
API_FLAVOR = "cpu-upgrade"
API_SLACK_S = 3600          # a programbench job's timeout: wall_limit_s + this
UV_INSTALLER = "https://astral.sh/uv/install.sh"
UV_LOCAL = "/opt/swarmlab/uv"
DEFAULT_CONCURRENCY = 64
DEFAULT_REQUEST_TIMEOUT_S = 600.0
STAGE_LOCAL = "/opt/swarmlab/stage"
# `hf jobs hardware`, 2026-10-06 (USD per hour); unknown flavors print "?"
FLAVOR_USD_PER_HOUR = {
    "cpu-basic": 0.01, "cpu-upgrade": 0.03, "cpu-xl": 1.00, "cpu-performance": 1.90, "t4-small": 0.40, "t4-medium": 0.60,
    "l4x1": 0.80, "l4x4": 3.80, "a10g-small": 1.00, "a10g-large": 1.50, "a10g-largex2": 3.00,
    "a10g-largex4": 5.00, "l40sx1": 1.80, "l40sx4": 8.30, "a100-large": 2.50, "a100x4": 10.00,
    "a100x8": 20.00, "h200": 5.00, "h200x2": 10.00, "rtx-pro-6000": 2.75,
}
# memory of ONE GPU in the flavor (GB); the profiles run TP=1, so a multi-GPU flavor does not add
# memory for the model. Unknown flavors get no size note.
FLAVOR_GPU_GB = {
    "cpu-basic": 0, "cpu-upgrade": 0, "cpu-xl": 0, "cpu-performance": 0, "t4-small": 16, "t4-medium": 16,
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


def arm_options(doc: dict, arm: str) -> dict:
    """The arm's run options (top-level `options`, the arm's merged over them), with
    `rounds_per_agent` resolved into `max_rounds` as the experiment does (`resolve_rounds`)."""
    opts = {**(doc.get("options") or {}), **(doc["arms"][arm].get("options") or {})}
    return resolve_rounds(opts, arm_agents(doc, arm))


def arm_rounds(doc: dict, arm: str) -> int:
    """`max_rounds` of an arm: rounds, or under `commit: async` ticks (one turn each)."""
    opts = arm_options(doc, arm)
    if opts.get("max_rounds") is None:
        raise SpecError(f"arm {arm!r}: no max_rounds (or rounds_per_agent) in options")
    return int(opts["max_rounds"])


def arm_unit(doc: dict, arm: str) -> str:
    """What one of the arm's `max_rounds` is: `round`, or `tick` (one turn) under async."""
    return "tick" if arm_options(doc, arm).get("commit") == "async" else "round"


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


def job_spec_doc(doc: dict, model: str | None, arms: Sequence[str] | None = None,
                 **provider_kw: Any) -> dict:
    """The spec the job runs: the in-job vllm provider (unless `model` is None: API mode), and
    only the chosen `arms` (when given).

    Arms the job does not run are dropped: they may call other models, which the job does not
    serve or price, and the job validates every arm of its spec before it starts."""
    out = copy.deepcopy(doc)
    if arms is not None:
        out["arms"] = {a: out["arms"][a] for a in out["arms"] if a in set(arms)}
    if model is not None:
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
    model: str | None          # the model vLLM serves; None in API mode
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
    parallel: int = 1          # runs at once in the job (SWARMLAB_PARALLEL)
    mode: str = "vllm"         # "vllm" | "api" (module docstring)
    secrets: list[str] = field(default_factory=lambda: ["HF_TOKEN"])  # names, never values
    missing_secrets: list[str] = field(default_factory=list)          # unset where planned
    extras: list[str] = field(default_factory=list)   # swarmlab extras installed in the job
    programbench: dict[str, Any] | None = None        # {"task", "image", "wall_limit_s", ...}

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
    def vllm_serve_argv(self) -> list[str] | None:
        return serve_argv(self) if self.mode == "vllm" else None

    @property
    def vllm_serve_command(self) -> str | None:
        return serve_command(self) if self.mode == "vllm" else None

    def env(self) -> dict[str, str]:
        if self.mode == "api":
            return {**self._common_env(), "SWARMLAB_MODE": "api",
                    "SWARMLAB_SECRET_NAMES": " ".join(self.secrets),
                    "SWARMLAB_EXTRAS": ",".join(self.extras)}   # HF_HOME: root-only default
        env = {
            **self._common_env(),
            "VLLM_MODEL": self.model or "",
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
        return env

    def _common_env(self) -> dict[str, str]:
        parallel = {"SWARMLAB_PARALLEL": str(self.parallel)} if self.parallel > 1 else {}
        return {
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
            **parallel,
        }

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
    model: str | None = None,
    flavor: str | None = None,
    arms: Sequence[str] | None = None,
    seeds: Sequence[int] | None = None,
    timeout: str | int | None = None,
    bucket: str = DEFAULT_BUCKET,
    image: str | None = None,
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
    environ: Mapping[str, str] | None = None,
) -> JobPlan:
    """The job for `spec`: vLLM mode when `model` is given or an arm calls a `vllm:` model, else
    API mode (module docstring). `environ` (default `os.environ`) is where API mode looks for
    the secrets' values (only whether they are set)."""
    spec = Path(spec)
    doc = load_experiment_yaml(spec)
    chosen = list(arms) if arms else list(doc["arms"])
    unknown = [a for a in chosen if a not in doc["arms"]]
    if unknown:
        raise SpecError(f"{spec}: unknown arm(s) {unknown}; available: {list(doc['arms'])}")
    seed_list = list(seeds) if seeds else experiment_seeds(doc)
    if not bucket.startswith("hf://buckets/"):
        raise SpecError(f"--bucket must look like hf://buckets/<namespace>/<name>, got {bucket!r}")
    bucket = bucket.rstrip("/")
    if parallel < 1:
        raise SpecError(f"--parallel must be >= 1, got {parallel}")
    api = model is None and not any(m.startswith("vllm:") for a in chosen
                                    for m in arm_models(doc, a))
    if not api and model is None:
        raise SpecError("the chosen arms call vllm: models; pass --model <the HF model id vLLM "
                        "serves>")
    commit, dirty = (git or checkout_identity)()
    if tag is None:
        stamp = time.strftime("%Y%m%d-%H%M%S", time.gmtime(time.time() if now is None else now))
        tag = f"{slug(doc['name'])}-{commit[:7]}-{stamp}"
    runs = [(a, int(s)) for a in chosen for s in seed_list]
    common: dict[str, Any] = {
        "spec_path": str(spec), "name": doc["name"], "tag": tag, "bucket": bucket,
        "commit": commit, "dirty": dirty, "runs": runs, "vllm_version": vllm_version,
        "max_model_len": max_model_len, "gpu_memory_utilization": gpu_memory_utilization,
        "revision": revision, "sync_every_s": sync_every_s, "orig_spec_yaml": spec.read_text(),
        "port": port}
    if api:
        plan = _plan_api(doc, chosen, len(seed_list), flavor=flavor, image=image,
                         timeout=timeout, per_round_s=per_round_s, setup_s=setup_s,
                         parallel=parallel, environ=os.environ if environ is None else environ,
                         common=common)
    else:
        assert model is not None
        plan = _plan_vllm(doc, chosen, len(seed_list), model=model, flavor=flavor or VLLM_FLAVOR,
                          image=image or DEFAULT_IMAGE, timeout=timeout, vllm_flags=vllm_flags,
                          profile=profile, vllm_flags_append=vllm_flags_append,
                          probe_extra=probe_extra, concurrency=concurrency,
                          request_timeout_s=request_timeout_s, per_round_s=per_round_s,
                          setup_s=setup_s, parallel=parallel, common=common)
    if dirty:
        plan.notes.append("the checkout has uncommitted changes: the wheel will not match the "
                          "commit")
    return plan


def _estimate(doc: dict, chosen: Sequence[str], runs: Sequence[tuple[str, int]],
              per_round_s: float | None, setup_s: int, parallel: int = 1,
              wall_s: Mapping[str, float] | None = None) -> tuple[int, dict]:
    """(total seconds, per-arm detail): per run, rounds (ticks under async) x the per-round
    guess, or the world's wall limit for the arms in `wall_s` (programbench); the total is the
    setup plus the makespan of bootstrap.sh's schedule of `runs` over `parallel` slots (each run
    takes the first free slot, in plan order; with one slot, the sum)."""
    per_arm = {}
    for a in chosen:
        n, n_seeds = arm_agents(doc, a), sum(1 for r, _ in runs if r == a)
        if wall_s and a in wall_s:
            per_arm[a] = {"agents": n, "wall_limit_s": wall_s[a], "per_run_s": int(wall_s[a]),
                          "runs": n_seeds}
        else:
            rounds = arm_rounds(doc, a)
            pr = per_round_s if per_round_s is not None else per_round_guess(n)
            per_arm[a] = {"agents": n, "rounds": rounds, "per_round_s": pr,
                          "per_run_s": int(rounds * pr), "runs": n_seeds}
            if arm_unit(doc, a) == "tick":  # async: `rounds` are ticks, `per_round_s` per tick
                per_arm[a]["unit"] = "tick"
    slots = [0] * parallel
    for a, _ in runs:
        slots[slots.index(min(slots))] += per_arm[a]["per_run_s"]
    return setup_s + max(slots), per_arm


def _plan_vllm(doc: dict, chosen: list[str], n_seeds: int, *, model: str, flavor: str,
               image: str, timeout: str | int | None, vllm_flags: str | None,
               profile: str | None, vllm_flags_append: str | None,
               probe_extra: dict[str, Any] | None, concurrency: int, request_timeout_s: float,
               per_round_s: float | None, setup_s: int, parallel: int,
               common: dict[str, Any]) -> JobPlan:
    check_vllm_models(doc, chosen, model)
    max_model_len, vllm_version = common["max_model_len"], common["vllm_version"]
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
    job_doc = job_spec_doc(doc, model, arms=chosen, port=common["port"], concurrency=concurrency,
                           timeout_s=request_timeout_s)
    job_yaml = dump_yaml(job_doc)
    pb = [a for a, e in _check_job_spec(job_yaml, chosen).items()
          if getattr(e.world, "name", None) == "programbench"]
    if pb:
        raise SpecError(f"arm(s) {pb}: a programbench world runs in an API-model job inside its "
                        "task image (no vLLM there); call an API model instead of vllm:")
    total, per_arm = _estimate(doc, chosen, common["runs"], per_round_s, setup_s, parallel)
    if parallel > 1:
        notes.append(f"up to {parallel} runs at once share the one vLLM server; the estimate "
                     "assumes they do not slow each other (raise --per-round if they contend)")
    timeout_s = (parse_duration(timeout) if timeout is not None
                 else math.ceil(total * 1.5 / 600.0) * 600)
    if timeout_s < total:
        notes.append(f"timeout {fmt_duration(timeout_s)} is below the {total // 60} min estimate; "
                     "the job may be cut (runs synced so far stay in the bucket, resumable)")
    return JobPlan(
        **common, model=model, flavor=flavor, image=image, timeout_s=timeout_s,
        estimate_s=total, estimate_detail={"setup_s": setup_s, "parallel": parallel,
                                           "arms": per_arm},
        vllm_flags=resolved.flags, job_spec_yaml=job_yaml, notes=notes,
        vllm_flags_source=resolved.source, vllm_profile=resolved.profile,
        probe_extra=resolved.probe_extra, probe_extra_source=resolved.probe_extra_source,
        parallel=parallel)


def _plan_api(doc: dict, chosen: list[str], n_seeds: int, *, flavor: str | None,
              image: str | None, timeout: str | int | None, per_round_s: float | None,
              setup_s: int, parallel: int, environ: Mapping[str, str],
              common: dict[str, Any]) -> JobPlan:
    job_yaml = dump_yaml(job_spec_doc(doc, None, arms=chosen))
    exps = _check_job_spec(job_yaml, chosen)
    notes: list[str] = []
    pb = programbench_arms(exps)
    facts = None
    if pb:
        tasks = sorted({w.task for w in pb.values()})
        if len(tasks) > 1:
            raise SpecError(f"the chosen arms name different programbench tasks {tasks}: one "
                            "job runs one task image; launch one job per task (--arm)")
        runs_pb = [r for r in common["runs"] if r[0] in pb]
        if len(runs_pb) > 1:
            raise SpecError(
                f"{len(runs_pb)} programbench runs in one job "
                f"({', '.join(f'{a}:{s}' for a, s in runs_pb)}): a run works in the image's "
                "own /workspace, which nothing resets between runs, so one job runs one arm x "
                "seed; narrow it with --arm and --seeds")
        if parallel > 1:
            raise SpecError(f"--parallel {parallel} with a programbench run: its lockdown (network, "
                            "sudo) and /workspace are job-wide, so nothing may run beside it; "
                            "drop --parallel or launch the other arms as their own job")
        first = pb[next(iter(pb))]
        facts = {"task": tasks[0], "image": first.image, "wall_limit_s": first.wall_limit_s,
                 "lockdown": first.lockdown, "pin_hosts": list(first.pin_hosts),
                 "user": first.user}
        if not first.lockdown:
            notes.append("the programbench world has lockdown: false: the agent keeps the job's "
                         "network and sudo")
        elif first.user is None:
            notes.append("the programbench world has user: null: the agent's commands run as "
                         "root (the job's user)")
    first_arm = chosen[0]
    default_image = pb[first_arm].image if first_arm in pb else DEFAULT_IMAGE
    image = image or default_image
    if pb and image != facts["image"]:
        notes.append(f"--image {image} is not the task image {facts['image']}")
    flavor = flavor or API_FLAVOR
    wall = {a: float(w.wall_limit_s) for a, w in pb.items()}
    total, per_arm = _estimate(doc, chosen, common["runs"], per_round_s, setup_s, parallel, wall)
    if parallel > 1:
        notes.append(f"up to {parallel} runs at once, each calling its providers (each keeps its own "
                     "budget); the estimate assumes they do not slow each other")
    if timeout is not None:
        timeout_s = parse_duration(timeout)
    elif wall:  # the wall limit + 1 h (setup, final snapshot, sync); other runs at 1.5x
        other = total - setup_s - sum(int(wall[a]) * n_seeds for a in wall)
        timeout_s = math.ceil((sum(wall[a] * n_seeds for a in wall) + API_SLACK_S
                               + 1.5 * other) / 600.0) * 600
    else:
        timeout_s = math.ceil(total * 1.5 / 600.0) * 600
    if timeout_s < total:
        notes.append(f"timeout {fmt_duration(timeout_s)} is below the {total // 60} min estimate; "
                     "the job may be cut (runs synced so far stay in the bucket, resumable)")
    secrets, missing = job_secrets(exps, environ)
    if missing:
        notes.append(f"{', '.join(missing)} not set in this environment: the job reads each "
                     "secret from here (hf jobs run --secrets NAME); --launch refuses until set")
    extras = sorted(({"anthropic"} if any(_is_anthropic(p) for p in _providers(exps)) else set())
                    | ({"programbench"} if pb else set()))
    return JobPlan(
        **common, model=None, flavor=flavor, image=image, timeout_s=timeout_s, estimate_s=total,
        estimate_detail={"setup_s": setup_s, "parallel": parallel, "arms": per_arm},
        vllm_flags="",
        job_spec_yaml=job_yaml, notes=notes, vllm_flags_source="none", vllm_profile=None,
        probe_extra={}, probe_extra_source="none", mode="api", secrets=secrets,
        missing_secrets=missing, extras=extras, programbench=facts, parallel=parallel)


def programbench_arms(exps: Mapping[str, Any]) -> dict[str, Any]:
    """{arm: ProgramBench world} for the arms whose world is programbench; SpecError for one
    whose `backend` is not `local` (a job cannot run Docker; the spec is not rewritten)."""
    from ..world.programbench import ProgramBench

    out = {a: e.world for a, e in exps.items() if isinstance(e.world, ProgramBench)}
    bad = [a for a, w in out.items() if w.backend != "local"]
    if bad:
        raise SpecError(f"arm(s) {bad}: the programbench world's backend is "
                        f"{out[bad[0]].backend!r}, but a job cannot run Docker; set "
                        "world.params.backend: local (the job runs inside the task image)")
    return out


def _models(exp: Any) -> list[str]:
    from ..experiment import participant_model

    models = [participant_model(p) for p in exp.participants]
    models += [p.coder_model() for p in exp.probes]
    models += [r.model for r in exp.roles.values()]  # role model overrides
    return [m for m in models if m is not None]


def _providers(exps: Mapping[str, Any]) -> list[Any]:
    return [e.provider_for(m)[0] for e in exps.values() for m in _models(e)]


def _is_anthropic(provider: Any) -> bool:
    from ..providers.anthropic import AnthropicProvider

    return isinstance(provider, AnthropicProvider)


def job_secrets(exps: Mapping[str, Any], environ: Mapping[str, str]) -> tuple[list[str], list[str]]:
    """(secret names the job needs, those unset in `environ`): HF_TOKEN, then each model's key
    variable (Anthropic: ANTHROPIC_API_KEY, or ANTHROPIC_KEY when only that is set here;
    OpenAI-compatible: its `api_key_env`), in first-seen order."""
    names = ["HF_TOKEN"]
    for p in _providers(exps):
        if _is_anthropic(p):
            name = ("ANTHROPIC_KEY" if not environ.get("ANTHROPIC_API_KEY")
                    and environ.get("ANTHROPIC_KEY") else "ANTHROPIC_API_KEY")
        else:
            name = getattr(p, "api_key_env", None)
        if name and name not in names:
            names.append(name)
    return names, [n for n in names if not environ.get(n)]


def checkout_identity() -> tuple[str, bool]:
    """(commit, dirty) of the checkout this package runs from, ignoring SWARMLAB_GIT_COMMIT."""
    import os

    saved = os.environ.pop("SWARMLAB_GIT_COMMIT", None)
    try:
        return git_identity(Path(__file__).resolve().parent)
    finally:
        if saved is not None:
            os.environ["SWARMLAB_GIT_COMMIT"] = saved


def _check_job_spec(job_yaml: str, arms: Sequence[str]) -> dict[str, Any]:
    """Every chosen arm of the job spec must build (offline: the vllm provider is fully priced);
    returns {arm: Experiment}."""
    from ..experiment import Experiment

    out = {}
    with tempfile.TemporaryDirectory() as d:
        path = Path(d) / "spec.yaml"
        path.write_text(job_yaml)
        for a in arms:
            try:
                out[a] = Experiment.from_yaml(path, a)
            except SpecError:
                raise
            except (ValueError, TypeError, ImportError, AttributeError) as e:
                raise SpecError(f"job spec, arm {a!r}: {type(e).__name__}: {e}") from e
    return out


# ---- the command ------------------------------------------------------------------------------
def job_entry(plan: Any) -> str:
    """The shell line the job runs: fetch the stage dir, then the bootstrap script. In vllm mode
    the uv image's uv fetches it; otherwise (API mode, grade jobs) uv is installed first when the
    image has none (into UV_LOCAL, PATH and shell profiles untouched) and handed to bootstrap.sh
    as SWARMLAB_UV."""
    if plan.mode == "vllm":
        return (f"mkdir -p {STAGE_LOCAL} && uv tool run --from 'huggingface_hub>=1.32' "
                f"hf buckets sync \"$SWARMLAB_STAGE\" {STAGE_LOCAL} && "
                f"exec bash {STAGE_LOCAL}/bootstrap.sh")
    return (f"mkdir -p {STAGE_LOCAL} && UV=$(command -v uv) || {{ curl -LsSf {UV_INSTALLER} | "
            f"env UV_INSTALL_DIR={UV_LOCAL} UV_NO_MODIFY_PATH=1 sh && UV={UV_LOCAL}/uv; }} && "
            f"\"$UV\" tool run --from 'huggingface_hub>=1.32' hf buckets sync \"$SWARMLAB_STAGE\" "
            f"{STAGE_LOCAL} && SWARMLAB_UV=\"$UV\" exec bash {STAGE_LOCAL}/bootstrap.sh")


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


def hf_command(plan: Any, hf: str = "hf") -> list[str]:
    """The exact `hf jobs run` argv of a plan (a `JobPlan`, or a grade plan with the same
    attributes): secrets by name only (`--secrets NAME`: the hf CLI reads the value from the
    launching environment), then volumes (`-v`, none for an agent job), then the env."""
    cmd = [hf, "jobs", "run", "--detach", "--flavor", plan.flavor,
           "--timeout", fmt_duration(plan.timeout_s)]
    for name in plan.secrets:
        cmd += ["--secrets", name]
    cmd += ["--name", f"swarmlab-{plan.tag}"[:80],
            "--label", "experiment=swarmlab", "--label", f"swarmlab_tag={plan.tag}"]
    for vol in getattr(plan, "volumes", ()):
        cmd += ["-v", vol]
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
             f"runs ({len(plan.runs)}): " + ", ".join(f"{a}:s{s}" for a, s in plan.runs)]
    if plan.mode == "vllm":
        lines += [(f"vllm {plan.vllm_version}: profile {plan.vllm_profile or 'none'}, "
                   f"flags from {plan.vllm_flags_source}"),
                  f"  flags: {plan.vllm_flags}",
                  f"  launcher adds: {shlex.join(launcher_flags(plan))}",
                  (f"  probe extra: {json.dumps(plan.probe_extra, sort_keys=True)} "
                   f"(from {plan.probe_extra_source})")]
    else:
        lines += [(f"mode: api (no vLLM): image {plan.image}, flavor {plan.flavor}, "
                   f"wheel extras [{','.join(plan.extras)}]"),
                  ("  secrets (names; the hf CLI reads each value from this environment): "
                   f"{' '.join(plan.secrets)}")]
        pb = plan.programbench
        if pb:
            lines.append(f"  programbench task {pb['task']} (backend local): the world locks the "
                         f"job down at its first command (DNS blackholed, sudo removed; pinned: "
                         f"{', '.join(pb['pin_hosts'])})" if pb["lockdown"] else
                         f"  programbench task {pb['task']} (backend local, lockdown off)")
    for a, d in plan.estimate_detail["arms"].items():
        unit = d.get("unit", "round")
        if "wall_limit_s" in d:
            lines.append(f"  {a}: N={d['agents']}, bounded by the world's wall limit "
                         f"{fmt_duration(int(d['wall_limit_s']))}/run x {d['runs']} seed(s)")
        else:
            lines.append(f"  {a}: N={d['agents']} x {d['rounds']} {unit}s x ~{d['per_round_s']:g}"
                         f" s/{unit} = ~{d['per_run_s'] / 60:.0f} min/run x {d['runs']} seed(s)")
    cost = (f"~${rate * est_h:.2f} expected, ${rate * plan.timeout_s / 3600:.2f} worst case "
            f"(${rate:.2f}/h x timeout)" if rate is not None else "rate unknown for this flavor")
    bounded = any("wall_limit_s" in d for d in plan.estimate_detail["arms"].values())
    at_once = f", {plan.parallel} runs at once" if plan.parallel > 1 else ""
    lines.append(f"estimate: ~{plan.estimate_s / 60:.0f} min wall (incl. "
                 f"{plan.estimate_detail['setup_s'] // 60} min setup{at_once}"
                 f"{'; an upper bound: the wall limit' if bounded else ''}); timeout "
                 f"{fmt_duration(plan.timeout_s)}; flavor {plan.flavor}: {cost}")
    lines.append("model calls are self-hosted: the ledger shows $0, the cost is the job's compute"
                 if plan.mode == "vllm" else
                 "model calls are billed by their providers on top (each run's budget caps them); "
                 "the cost above is the job's compute")
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


def submit(plan: Any, *, hf: str = "hf", runner: Runner = run_cli,
           build: Callable[[Path], Path] | None = build_wheel,
           ledger: Path | str | None = None,
           stage_fn: Callable[..., Path] | None = None) -> dict[str, Any]:
    """Stage, upload, submit. Returns `{"job_id", "url", "tag", ...}`; raises on failure
    (before anything is uploaded when a secret the job needs is unset here). `plan` is a
    `JobPlan`, or a grade plan (`swarmlab/jobs/programbench.py`) with its own `stage_fn`."""
    stage_fn = stage_fn or stage
    if plan.missing_secrets:
        raise SpecError(f"not launching: {', '.join(plan.missing_secrets)} not set in this "
                        "environment (the job reads each secret from here via hf jobs run "
                        "--secrets NAME)")
    with tempfile.TemporaryDirectory() as tmp:
        d = stage_fn(plan, Path(tmp) / plan.tag, build=build)
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
           "estimate_s": getattr(plan, "estimate_s", None),
           "rate_usd_per_hour": plan.rate_usd_per_hour,
           "commit": plan.commit, "runs": [f"{a}:{s}" for a, s in getattr(plan, "runs", [])],
           "stage_remote": plan.stage_remote, "runs_remote": getattr(plan, "runs_remote", None),
           "mode": plan.mode, "secrets": list(plan.secrets)}
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


def _namespace(plan: Any) -> str:
    return plan.bucket.removeprefix("hf://buckets/").split("/", 1)[0]


def which_hf() -> str:
    return shutil.which("hf") or "hf"
