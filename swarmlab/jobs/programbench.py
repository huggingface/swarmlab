"""`swarmlab programbench grade`: grade a finished ProgramBench run in a second HF Job.

The world never grades (swarmlab/world/programbench.py, "Grade seam"); a run ends with
`score.graded: false`. Grading is this separate step, from the operator's machine:

1. `submission_from_run(run_dir)`: the run's last committed snapshot (`snapshots/` manifest of
   `run.json["last_round"]` -> blob `world` -> `workspace_tar_of`) is written to
   `<run>/artifacts/submission.tar.gz`. That snapshot must be of the last round itself. The
   runner always snapshots the round (or async tick) that ends the run, whatever
   `snapshot_every`, so a run that ended normally has it; a run that did not (a hard ceiling or
   a crash discards the round in flight, and resuming such a run needs a snapshot) may have only
   an older one, and grading refuses (SpecError naming both rounds) rather than grade it. A snapshot holding no tar means the image's untouched
   workspace (no command ever ran): `artifacts/submission.untouched` (empty) is written instead
   and grading is refused (there is nothing of the agent's to grade).
2. `plan_grade(run_dir)` -> `GradePlan`: the task from the run's spec
   (`run.spec.world.params.task`); image = the task's clean image
   (`programbench/<task>:task_cleanroom_v6`, the evaluator's default tag); flavor `cpu-upgrade`
   by default; timeout = the evaluator's bounds, `GRADE_SETUP_S` + 900 s compile + 3600 s per
   active branch (`programbench`'s `tests.json`), capped at `GRADE_CAP_S` (12 h; branch retries
   could exceed even the uncapped bound), rounded up to 10 minutes; volume
   `hf://datasets/programbench/ProgramBench-Tests:/root/programbench-tests:ro` (the hidden tests,
   which contain the task's original source: mounted only here, under `/root`, never in the agent's
   job); secret `HF_TOKEN` by name (the stage dir and the upload of grade.json). The stage dir
   `<bucket>/jobs/<tag>/` (tag `grade-<run id>-<commit>-<UTC stamp>`) holds `submission.tar.gz`,
   the swarmlab wheel, `bootstrap.sh` and `plan.json`; the job runs bootstrap.sh in mode `grade`
   (install uv, Python 3.12, `swarmlab[programbench]`, then `python -m swarmlab.jobs.grade`,
   swarmlab/jobs/grade.py) and uploads `<stage>/grade.json`.
3. `submit_grade(plan)` stages, uploads and submits (`launch.submit`), and records the job in
   `<run>/artifacts/grade_job.json` (job id, tag, stage).
4. `fetch_grade(run_dir)` downloads `<stage>/grade.json` (the stage named in grade_job.json) to
   `<run>/artifacts/grade.json`; `write_grade(run_dir, grade_json)` writes the score into the run.

The score write-back is the one documented exception to "the runner owns the score": `run.json`
gets `score = {**score_world, "graded": True, "score", "passed", "total", "resolved",
"near_resolved", "branches", "compile_ok"}` (the official per-instance score from grade.json),
`score_world` = the score the runner wrote (kept from the first write-back, so a re-grade
replaces only the graded keys), and `grade` = where the score came from (job id, tag,
programbench version, CPUs, image, time, the deviation note). Replay (`Run.load`, `swarmlab
report`) checks the world's recomputed score against `score_world` when present. `report`'s
Final score table and `export` read `score` and show the graded keys. Scores are not
cheat-filtered (the benchmark's cheating judge is not public).
"""
from __future__ import annotations

import hashlib
import json
import math
import os
import shlex
import time
from collections.abc import Callable, Mapping
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any

from .._io import atomic_write_bytes
from ..spec import SpecError
from . import launch as _launch
from .launch import (
    DEFAULT_BUCKET,
    FLAVOR_USD_PER_HOUR,
    Runner,
    bootstrap_script,
    build_wheel,
    fmt_duration,
    hf_command,
    parse_duration,
    run_cli,
    slug,
    submit,
)

TESTS_DATASET = "hf://datasets/programbench/ProgramBench-Tests"
TESTS_MOUNT = "/root/programbench-tests"   # under /root (0700): a non-root user cannot traverse
GRADE_FLAVOR = "cpu-upgrade"
COMPILE_S, BRANCH_S = 900, 3600            # the evaluator's compile and per-branch timeouts
GRADE_SETUP_S = 1800                       # uv, Python, the wheel, the image's own start
GRADE_CAP_S = 12 * 3600
GRADED_KEYS = ("score", "passed", "total", "resolved", "near_resolved", "branches", "compile_ok")
TERMINAL_STAGES = {"COMPLETED", "ERROR", "CANCELED", "CANCELLED", "DELETED", "FAILED", "TIMEOUT"}


# ---- the run's side -------------------------------------------------------------------------
def run_task(run_dir: Path | str) -> str:
    """The ProgramBench task of a run (`run.spec.world.params.task`); SpecError otherwise."""
    from ..experiment import Run

    spec = Run(run_dir).spec
    if spec.world.type != "programbench" or "task" not in (spec.world.params or {}):
        raise SpecError(f"{run_dir}: not a programbench run (world {spec.world.type!r})")
    return str(spec.world.params["task"])


def submission_from_run(run_dir: Path | str) -> Path | None:
    """Write the last committed snapshot's workspace tar to `<run>/artifacts/submission.tar.gz`
    and return it; None (and an empty `artifacts/submission.untouched`) when the snapshot holds
    no tar (the image's workspace, untouched)."""
    from ..runner import Runner as RunRunner
    from ..snapshot import SnapshotStore
    from ..world.programbench import workspace_tar_of

    run_dir = Path(run_dir)
    meta = RunRunner._read_meta_at(run_dir)
    store = SnapshotStore(run_dir)
    last = meta.get("last_round")
    manifest = store.latest(max_round=last)
    if manifest is None or "world" not in manifest.plugins:
        raise FileNotFoundError(f"{run_dir}: no world snapshot to grade (the run committed no "
                                "round)")
    if last is not None and manifest.round != last:
        raise SpecError(
            f"{run_dir}: the run's last round is {last} but its last snapshot is of round "
            f"{manifest.round}: the run did not end normally (a hard ceiling or a crash discards "
            "the round in flight; a normal end always snapshots its last round), so the "
            "workspace the agent ended with was never saved; grading the older one would score "
            "the wrong submission. Resume the run to finish it, or rerun it.")
    tar = workspace_tar_of(store.blobs.get(manifest.plugins["world"]))
    art = run_dir / "artifacts"
    art.mkdir(exist_ok=True)
    if tar is None:
        (art / "submission.untouched").write_bytes(b"")
        return None
    out = art / "submission.tar.gz"
    atomic_write_bytes(out, tar)
    return out


def active_branches(task: str) -> int | None:
    """Number of active test branches of `task` (the `programbench` package), None without it."""
    try:
        from programbench.utils.load_data import (  # type: ignore[import-not-found]
            get_active_branches,
            load_all_instances,
        )
    except ImportError:
        return None
    for inst in load_all_instances():
        if inst["instance_id"] == task:
            return len(get_active_branches(inst))
    raise SpecError(f"unknown ProgramBench task {task!r}")


# ---- the plan --------------------------------------------------------------------------------
@dataclass
class GradePlan:
    run_dir: str
    run_id: str
    task: str
    image: str
    flavor: str
    tag: str
    bucket: str
    commit: str
    dirty: bool
    timeout_s: int
    bound_s: int | None
    branches: int | None
    submission: str
    submission_sha256: str
    submission_bytes: int
    notes: list[str] = field(default_factory=list)
    mode: str = "grade"
    secrets: list[str] = field(default_factory=lambda: ["HF_TOKEN"])
    missing_secrets: list[str] = field(default_factory=list)
    volumes: list[str] = field(default_factory=lambda: [f"{TESTS_DATASET}:{TESTS_MOUNT}:ro"])

    @property
    def stage_remote(self) -> str:
        return f"{self.bucket}/jobs/{self.tag}"

    @property
    def rate_usd_per_hour(self) -> float | None:
        return FLAVOR_USD_PER_HOUR.get(self.flavor)

    def env(self) -> dict[str, str]:
        return {
            "SWARMLAB_MODE": "grade",
            "SWARMLAB_TAG": self.tag,
            "SWARMLAB_STAGE": self.stage_remote,
            "SWARMLAB_GIT_COMMIT": self.commit,
            "SWARMLAB_GIT_DIRTY": "1" if self.dirty else "0",
            "SWARMLAB_FLAVOR": self.flavor,
            "SWARMLAB_IMAGE": self.image,
            "SWARMLAB_GRADE_TASK": self.task,
            "SWARMLAB_GRADE_RUN": self.run_id,
            "SWARMLAB_GRADE_TESTS": TESTS_MOUNT,
            "SWARMLAB_EXTRAS": "programbench",
            "SWARMLAB_SECRET_NAMES": " ".join(self.secrets),
        }

    def as_dict(self) -> dict[str, Any]:
        d = asdict(self)
        d.update(stage_remote=self.stage_remote, rate_usd_per_hour=self.rate_usd_per_hour,
                 timeout=fmt_duration(self.timeout_s), hf_command=hf_command(self))
        return d


def plan_grade(run_dir: Path | str, *, flavor: str | None = None,
               timeout: str | int | None = None, bucket: str = DEFAULT_BUCKET,
               tag: str | None = None, now: float | None = None,
               git: Callable[[], tuple[str, bool]] | None = None,
               environ: Mapping[str, str] | None = None) -> GradePlan:
    """Extract the submission and plan the grade job (module docstring); nothing is uploaded."""
    from ..experiment import Run
    from ..world.programbench import image_name

    run_dir = Path(run_dir)
    run = Run(run_dir)
    task = run_task(run_dir)
    if run.status != "ended":
        raise SpecError(f"{run_dir}: the run has not ended (status {run.status!r}); grade a "
                        "finished run")
    sub = submission_from_run(run_dir)
    if sub is None:
        raise SpecError(f"{run_dir}: the last snapshot holds no workspace tar (the image's "
                        "workspace was never touched: no command ran); there is nothing of the "
                        "agent's to grade (marker: artifacts/submission.untouched)")
    if not bucket.startswith("hf://buckets/"):
        raise SpecError(f"--bucket must look like hf://buckets/<namespace>/<name>, got {bucket!r}")
    bucket = bucket.rstrip("/")
    notes: list[str] = []
    n = active_branches(task)
    bound = None if n is None else GRADE_SETUP_S + COMPILE_S + BRANCH_S * n
    if timeout is not None:
        timeout_s = parse_duration(timeout)
    else:
        timeout_s = math.ceil(min(bound or GRADE_CAP_S, GRADE_CAP_S) / 600.0) * 600
    if n is None:
        notes.append(f"the programbench package is not installed here, so the branch count is "
                     f"unknown: timeout {fmt_duration(timeout_s)} (the cap)")
    elif timeout is not None and bound is not None and timeout_s < bound:
        notes.append(f"timeout {fmt_duration(timeout_s)} is below the evaluator's bound "
                     f"({bound / 3600:.1f} h: {GRADE_SETUP_S // 60} min setup + {COMPILE_S} s "
                     f"compile + {n} x {BRANCH_S} s); real grading of a small task takes minutes")
    commit, dirty = (git or _launch.checkout_identity)()
    if dirty:
        notes.append("the checkout has uncommitted changes: the wheel will not match the commit")
    if tag is None:
        stamp = time.strftime("%Y%m%d-%H%M%S", time.gmtime(time.time() if now is None else now))
        tag = f"grade-{slug(run.id)[:40]}-{commit[:7]}-{stamp}"
    env = os.environ if environ is None else environ
    data = sub.read_bytes()
    return GradePlan(
        run_dir=str(run_dir), run_id=run.id, task=task, image=image_name(task),
        flavor=flavor or GRADE_FLAVOR, tag=tag, bucket=bucket, commit=commit, dirty=dirty,
        timeout_s=timeout_s, bound_s=bound, branches=n, submission=str(sub),
        submission_sha256=hashlib.sha256(data).hexdigest(), submission_bytes=len(data),
        notes=notes, missing_secrets=[s for s in ("HF_TOKEN",) if not env.get(s)])


def describe_grade(plan: GradePlan) -> str:
    rate = plan.rate_usd_per_hour
    cost = (f"${rate * plan.timeout_s / 3600:.2f} worst case (${rate:.2f}/h x timeout)"
            if rate is not None else "rate unknown for this flavor")
    bound = (f"evaluator bound {plan.bound_s / 3600:.1f} h ({GRADE_SETUP_S // 60} min setup + "
             f"{COMPILE_S} s compile + {plan.branches} branch(es) x {BRANCH_S} s)"
             if plan.bound_s is not None else "branch count unknown")
    lines = ["hf jobs command:", "  " + shlex.join(hf_command(plan)), "",
             f"grade {plan.run_id}: task {plan.task}, image {plan.image}",
             (f"submission {plan.submission} ({plan.submission_bytes} bytes, sha256 "
              f"{plan.submission_sha256[:12]}) -> {plan.stage_remote}/submission.tar.gz"),
             f"tag {plan.tag}  commit {plan.commit[:12]}{' (dirty)' if plan.dirty else ''}",
             f"tests: {TESTS_DATASET} mounted read-only at {TESTS_MOUNT} (grade job only)",
             (f"timeout {fmt_duration(plan.timeout_s)} ({bound}, capped at "
              f"{fmt_duration(GRADE_CAP_S)}); flavor {plan.flavor}: {cost}"),
             (f"result: {plan.stage_remote}/grade.json; then `swarmlab programbench fetch-grade "
              f"{plan.run_dir}` writes the score into the run")]
    if plan.missing_secrets:
        lines.append(f"note: {', '.join(plan.missing_secrets)} not set in this environment; "
                     "--launch refuses until set")
    lines += [f"note: {n}" for n in plan.notes]
    return "\n".join(lines)


def stage_grade(plan: GradePlan, directory: Path | str,
                build: Callable[[Path], Path] | None = build_wheel) -> Path:
    """Write the grade job's stage dir locally: submission, bootstrap, plan (wheel when `build`)."""
    d = Path(directory)
    d.mkdir(parents=True, exist_ok=True)
    (d / "submission.tar.gz").write_bytes(Path(plan.submission).read_bytes())
    (d / "bootstrap.sh").write_text(bootstrap_script())
    (d / "plan.json").write_text(json.dumps(plan.as_dict(), indent=1, sort_keys=True))
    if build is not None:
        build(d)
    return d


def submit_grade(plan: GradePlan, *, hf: str = "hf", runner: Runner = run_cli,
                 build: Callable[[Path], Path] | None = build_wheel,
                 ledger: Path | str | None = None) -> dict[str, Any]:
    """Stage, upload, submit; record the job in `<run>/artifacts/grade_job.json`."""
    row = submit(plan, hf=hf, runner=runner, build=build, ledger=ledger, stage_fn=stage_grade)
    row = {**row, "kind": "grade", "run_id": plan.run_id, "task": plan.task,
           "submission_sha256": plan.submission_sha256}
    path = Path(plan.run_dir) / "artifacts" / "grade_job.json"
    path.write_text(json.dumps(row, indent=1, sort_keys=True) + "\n")
    return row


# ---- the result -----------------------------------------------------------------------------
def grade_job(run_dir: Path | str) -> dict[str, Any]:
    path = Path(run_dir) / "artifacts" / "grade_job.json"
    if not path.exists():
        raise FileNotFoundError(f"{path} does not exist: launch the grade with `swarmlab "
                                "programbench grade RUN --launch` first (or pass --grade-json)")
    return json.loads(path.read_text())


def wait_grade(job_id: str, *, hf: str = "hf", poll_s: float = 30.0,
               status_fn: Callable[[str], dict] | None = None,
               sleep: Callable[[float], None] = time.sleep,
               say: Callable[[str], None] | None = None) -> str:
    """Poll the job until a terminal stage; returns it."""
    from .remote import status

    get = status_fn or (lambda j: status(j, hf=hf))
    last = None
    while True:
        stage = str(get(job_id).get("stage") or "").upper()
        if stage != last and say is not None:
            say(f"grade job {job_id}: {stage or '?'}")
        last = stage
        if stage in TERMINAL_STAGES:
            return stage
        sleep(poll_s)


def fetch_grade(run_dir: Path | str, *, hf: str = "hf", runner: Runner = run_cli) -> Path:
    """Download `<stage>/grade.json` of the run's grade job to `<run>/artifacts/grade.json`."""
    job = grade_job(run_dir)
    dest = Path(run_dir) / "artifacts" / "grade.json"
    p = runner([hf, "buckets", "cp", f"{job['stage_remote']}/grade.json", str(dest)])
    if p.returncode != 0 or not dest.exists():
        raise RuntimeError(f"no grade.json at {job['stage_remote']}/ yet (the job may still run, "
                           f"or failed: swarmlab job logs {job.get('job_id')}): "
                           f"{((p.stdout or '') + (p.stderr or ''))[-400:]}")
    return dest


def write_grade(run_dir: Path | str, grade_json: Path | str) -> dict[str, Any]:
    """Write grade.json's official score into run.json (module docstring); keeps grade.json in
    `<run>/artifacts/`. Returns the new `score`."""
    from ..runner import Runner as RunRunner

    run_dir = Path(run_dir)
    doc = json.loads(Path(grade_json).read_text())
    task = run_task(run_dir)
    if doc.get("task") != task:
        raise SpecError(f"{grade_json} grades task {doc.get('task')!r}, the run is {task!r}")
    sc = doc.get("score") or {}
    missing = [k for k in GRADED_KEYS if k not in sc]
    if missing:
        raise SpecError(f"{grade_json}: score lacks {missing}")
    kept = run_dir / "artifacts" / "grade.json"
    if Path(grade_json).resolve() != kept.resolve():
        kept.parent.mkdir(exist_ok=True)
        kept.write_text(Path(grade_json).read_text())
    meta = RunRunner._read_meta_at(run_dir)
    world = meta.get("score_world", meta.get("score") or {})
    meta["score_world"] = world
    meta["score"] = {**world, "graded": True, **{k: sc[k] for k in GRADED_KEYS}}
    job = {}
    with_job = run_dir / "artifacts" / "grade_job.json"
    if with_job.exists():
        job = json.loads(with_job.read_text())
    meta["grade"] = {"job_id": job.get("job_id"), "tag": job.get("tag"),
                     "programbench_version": doc.get("programbench_version"),
                     "cpus": doc.get("cpus"), "pytest_workers": doc.get("pytest_workers"),
                     "image": doc.get("image"), "graded_at": doc.get("ended_at"),
                     "grade_json": "artifacts/grade.json", "deviation": doc.get("deviation"),
                     "error_code": sc.get("error_code")}
    text = json.dumps(meta, indent=2, sort_keys=True) + "\n"
    atomic_write_bytes(run_dir / "run.json", text.encode())
    return meta["score"]
