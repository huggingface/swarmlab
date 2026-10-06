"""Look at jobs and bring their runs home: `status`, `logs`, `fetch` (docs/handoff/WP8.md).

All Hub access goes through the `hf` CLI (`runner(argv) -> CompletedProcess`, injectable for
tests). Bucket layout written by a job (see `launch.py`): `<bucket>/jobs/<tag>/` (stage dir, job
manifest `job.json`, `launch.json`, `results.jsonl`, `vllm.log`, `logs/`) and
`<bucket>/runs/<tag>/<run_id>/` (complete run dirs, blobs and inference cache included, so
`Run.load`, `swarmlab view` and `swarmlab fork` work on the fetched copy; a fork beyond the
cached prefix needs a live server again, since the run spec's vllm base_url is the job's
localhost).

`fetch(run_id, out, bucket, tag=None)`: `run_id` may be `<tag>/<run_id>`; without a tag the
newest tag (tags end in a UTC timestamp, so name order is time order) holding that run wins.
`bucket` may also be a local directory with the same layout (a stand-in for tests and for a
bucket already synced to disk); then files are copied instead of downloaded.
"""
from __future__ import annotations

import json
import shutil
import subprocess
from collections.abc import Callable
from pathlib import Path
from typing import Any

from .launch import DEFAULT_BUCKET, FLAVOR_USD_PER_HOUR

Runner = Callable[[list[str]], subprocess.CompletedProcess]


def run_cli(args: list[str]) -> subprocess.CompletedProcess:
    return subprocess.run(args, capture_output=True, text=True, check=False)


def _check(p: subprocess.CompletedProcess, what: str) -> str:
    if p.returncode != 0:
        raise RuntimeError(f"{what} failed ({p.returncode}): {((p.stdout or '') + (p.stderr or ''))[-600:]}")
    return p.stdout or ""


def _is_remote(bucket: str) -> bool:
    return bucket.startswith("hf://")


def list_dirs(bucket: str, prefix: str, *, hf: str = "hf", runner: Runner = run_cli) -> list[str]:
    """Names of the directories directly under `<bucket>/<prefix>/`."""
    if not _is_remote(bucket):
        base = Path(bucket) / prefix
        return sorted(p.name for p in base.iterdir() if p.is_dir()) if base.is_dir() else []
    p = runner([hf, "buckets", "ls", f"{bucket.rstrip('/')}/{prefix}/", "--format", "json"])
    if p.returncode != 0:
        return []
    try:
        rows = json.loads(_json_part(p.stdout or ""))
    except ValueError:
        return []
    return sorted(r["path"].rstrip("/").rsplit("/", 1)[-1] for r in rows
                  if r.get("type") == "directory")


def _json_part(text: str) -> str:
    """The JSON array in `hf ... --format json` output (hints may follow it)."""
    start = text.find("[")
    end = text.rfind("]")
    return text[start:end + 1] if start >= 0 and end > start else text


def find_run(run_id: str, bucket: str, tag: str | None = None, *, hf: str = "hf",
             runner: Runner = run_cli) -> tuple[str, str]:
    """(tag, run_id) of the run to fetch; the newest tag holding it when `tag` is None."""
    if "/" in run_id:
        tag, run_id = run_id.split("/", 1)
    tags = [tag] if tag else sorted(list_dirs(bucket, "runs", hf=hf, runner=runner), reverse=True)
    for t in tags:
        if run_id in list_dirs(bucket, f"runs/{t}", hf=hf, runner=runner):
            return t, run_id
    where = f"runs/{tag}/" if tag else "runs/*/"
    raise FileNotFoundError(f"no run {run_id!r} under {bucket}/{where}")


def fetch(run_id: str, out: Path | str = "runs", bucket: str = DEFAULT_BUCKET,
          tag: str | None = None, *, hf: str = "hf", runner: Runner = run_cli) -> Path:
    """Download `<bucket>/runs/<tag>/<run_id>` to `out/<run_id>`; returns the local dir."""
    t, rid = find_run(run_id, bucket, tag, hf=hf, runner=runner)
    dest = Path(out) / rid
    if not _is_remote(bucket):
        shutil.copytree(Path(bucket) / "runs" / t / rid, dest, dirs_exist_ok=True)
    else:
        dest.mkdir(parents=True, exist_ok=True)
        _check(runner([hf, "buckets", "sync", f"{bucket.rstrip('/')}/runs/{t}/{rid}", str(dest)]),
               "hf buckets sync")
    if not (dest / "run.json").exists():
        raise FileNotFoundError(f"{dest} has no run.json (the run never started, or the sync "
                                "is incomplete)")
    return dest


def status(job_id: str, *, hf: str = "hf", runner: Runner = run_cli) -> dict[str, Any]:
    """`hf jobs inspect` digested: stage, flavor, running seconds, cost so far, labels."""
    raw = json.loads(_check(runner([hf, "jobs", "inspect", job_id, "--format", "json"]),
                            "hf jobs inspect") or "{}")
    info = raw[0] if isinstance(raw, list) else raw
    st = info.get("status") or {}
    flavor = info.get("flavor")
    secs = (info.get("durations") or {}).get("running_secs")
    rate = FLAVOR_USD_PER_HOUR.get(flavor or "")
    labels = info.get("labels") or {}
    return {
        "job_id": info.get("id", job_id), "stage": st.get("stage"), "message": st.get("message"),
        "flavor": flavor, "running_secs": secs,
        "cost_usd": round(rate * secs / 3600, 4) if rate is not None and secs else None,
        "tag": labels.get("swarmlab_tag"), "url": info.get("url"), "labels": labels,
    }


def logs(job_id: str, follow: bool = False, *, hf: str = "hf") -> int:
    """Stream `hf jobs logs` to this terminal; returns its exit code."""
    return subprocess.call([hf, "jobs", "logs", job_id, *(["--follow"] if follow else [])])
