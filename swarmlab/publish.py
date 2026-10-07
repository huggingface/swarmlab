"""Publish exported runs to a Hugging Face dataset repo, and fetch them back
(docs/INTERFACE-M4.md §2-§3). Needs the `hub` extra (`huggingface_hub`).

```
publish(source, repo=None, *, public=False, tag=None, api=None) -> dict
publish_view(run_dir, repo, *, api=None) -> dict          # `swarmlab view RUN_DIR --publish REPO`
fetch_published(repo, run_id, out="runs", *, api=None, force=False) -> Path
```

Repo layout (one **dataset** repo per experiment):

```
README.md                     dataset card: spec per arm, run table, table schemas, layout
index.json                    {"experiment", "export_schema", "runs": {run_id: entry}}
runs/<run_id>/run.json        the export of that run (swarmlab/export.py), i.e. the contents of
runs/<run_id>/tables/...      <run_dir>/export/ placed under runs/<run_id>/
runs/<run_id>/sessions/...
runs/<run_id>/raw/...
runs/<run_id>/view.html       when the run dir has one, or after `view --publish`
```

Decisions where the contract is silent:

- **Source.** `source` is a run directory (has `run.json`) or a directory of run directories.
  Only runs with `status == "ended"` are published; others are reported as skipped. Each run is
  exported into `<run_dir>/export` first unless that export is current (`export.is_current`:
  same log sha256 and export schema).
- **Repo.** Default `<namespace>/<experiment>` with the namespace from `HfApi.whoami()`. Runs of
  several experiments with no `--repo` go to one repo each; an explicit `--repo` with runs of
  several experiments is an error. The repo is created private (`private=not public`); an
  existing repo's visibility is changed only by `--public` (made public), never made private
  again silently. The card carries the `format:agent-traces` tag only when the repo is public
  (the Hub's trace viewer renders public repos only), which can only happen through `--public`.
  `--tag TAG` adds free tags to the card (repeatable).
- **Idempotence.** The remote tree is listed once; a local file is uploaded only when its
  content differs (LFS files compared by sha256, regular files by git blob sha1). `index.json`
  merges the remote index with the published runs and is rendered deterministically (sorted
  keys, no timestamps), as is the card, so re-publishing unchanged runs makes no commit.
  Uploads go in commits of at most `COMMIT_CHUNK` files.
- **Card.** YAML front matter with `configs`: `sessions` (default; `runs/*/sessions/*.jsonl`)
  and one config per table family (`runs/*/tables/<family>.parquet`), so Data Studio loads each
  family as one table across runs and the raw logs are not mistaken for data. The body is
  generated from `index.json` alone, so `publish` and `view --publish` write the same card.
- **fetch-published** downloads `runs/<run_id>/` and rebuilds `<out>/<run_id>/` from `raw/`
  (`run.json`, `events.jsonl`, `discarded.jsonl`, `snapshots/`, `artifacts/`, `blobs/`), puts the
  downloaded export under `<out>/<run_id>/export/` (without its `raw/` copy) and `view.html` next
  to it. An existing non-empty run dir is refused unless `force`.
"""
from __future__ import annotations

import hashlib
import json
import shutil
import tempfile
from collections.abc import Iterable
from pathlib import Path
from typing import Any

import yaml

from .export import EXPORT_SCHEMA, FAMILIES, TABLE_SCHEMAS, export_run, is_current

COMMIT_CHUNK = 1000
REPO_TYPE = "dataset"


class PublishError(ValueError):
    """Bad publish arguments (unknown runs, several experiments for one --repo, ...)."""


def _api(api: Any = None) -> Any:
    if api is not None:
        return api
    try:
        from huggingface_hub import HfApi
    except ImportError as e:  # pragma: no cover - the hub extra is a dev dependency
        raise PublishError("publishing needs the hub extra: uv sync --extra hub") from e
    return HfApi()


# ---- discovery -------------------------------------------------------------------------------
def find_runs(source: Path | str) -> tuple[list[Path], list[str]]:
    """(finished run dirs, skipped descriptions) under `source` (a run dir or a runs dir)."""
    src = Path(source)
    if (src / "run.json").is_file():
        dirs = [src]
    elif src.is_dir():
        dirs = sorted(d for d in src.iterdir() if (d / "run.json").is_file())
    else:
        raise PublishError(f"{src} is neither a run directory nor a directory of runs")
    runs, skipped = [], []
    for d in dirs:
        try:
            status = json.loads((d / "run.json").read_text()).get("status")
        except ValueError:
            skipped.append(f"{d.name} (unreadable run.json)")
            continue
        if status != "ended":
            skipped.append(f"{d.name} (status {status})")
        else:
            runs.append(d)
    return runs, skipped


def ensure_export(run_dir: Path) -> Path:
    out = run_dir / "export"
    if not is_current(run_dir, out):
        export_run(run_dir, out)
    return out


# ---- hashing and remote state ----------------------------------------------------------------
def git_blob_sha1(data: bytes) -> str:
    return hashlib.sha1(b"blob %d\0" % len(data) + data, usedforsecurity=False).hexdigest()


def _lfs_sha(entry: Any) -> str | None:
    lfs = getattr(entry, "lfs", None)
    if lfs is None:
        return None
    return getattr(lfs, "sha256", None) or (lfs.get("sha256") if isinstance(lfs, dict) else None)


def remote_files(api: Any, repo: str) -> dict[str, tuple[str | None, str | None]]:
    """path -> (git blob id, lfs sha256) for every file in the repo (empty if none)."""
    out: dict[str, tuple[str | None, str | None]] = {}
    for entry in api.list_repo_tree(repo, recursive=True, repo_type=REPO_TYPE):
        if getattr(entry, "blob_id", None) is None and getattr(entry, "lfs", None) is None:
            continue  # a folder
        out[entry.path] = (getattr(entry, "blob_id", None), _lfs_sha(entry))
    return out


def _unchanged(data: bytes, remote: tuple[str | None, str | None] | None) -> bool:
    if remote is None:
        return False
    blob_id, lfs = remote
    if lfs:
        return hashlib.sha256(data).hexdigest() == lfs
    return blob_id == git_blob_sha1(data)


def _remote_json(api: Any, repo: str, path: str, files: dict) -> dict | None:
    if path not in files:
        return None
    with tempfile.TemporaryDirectory() as tmp:
        local = api.hf_hub_download(repo, path, repo_type=REPO_TYPE, local_dir=tmp)
        return json.loads(Path(local).read_text())


def _commit(api: Any, repo: str, uploads: dict[str, bytes | Path], files: dict,
            message: str) -> tuple[int, int]:
    """Upload the changed entries of `uploads` (path in repo -> bytes or local path)."""
    from huggingface_hub import CommitOperationAdd

    ops = []
    skipped = 0
    for path in sorted(uploads):
        src = uploads[path]
        data = src if isinstance(src, bytes) else Path(src).read_bytes()
        if _unchanged(data, files.get(path)):
            skipped += 1
            continue
        ops.append(CommitOperationAdd(path_in_repo=path,
                                      path_or_fileobj=data if isinstance(src, bytes) else str(src)))
    for i in range(0, len(ops), COMMIT_CHUNK):
        chunk = ops[i:i + COMMIT_CHUNK]
        part = f" ({i // COMMIT_CHUNK + 1}/{-(-len(ops) // COMMIT_CHUNK)})" if len(ops) > COMMIT_CHUNK else ""
        api.create_commit(repo, chunk, commit_message=message + part, repo_type=REPO_TYPE)
    return len(ops), skipped


# ---- index and card --------------------------------------------------------------------------
def index_entry(run_dir: Path, export_dir: Path) -> dict:
    doc = json.loads((export_dir / "run.json").read_text())
    agents = doc.get("agents") or []
    models = sorted({a["model"] for a in agents if a.get("model")})
    return {
        "run_id": doc["run_id"], "experiment": doc.get("experiment"), "arm": doc.get("arm"),
        "seed": doc.get("seed"), "status": doc.get("status"), "end_reason": doc.get("end_reason"),
        "last_round": doc.get("last_round"), "score": doc.get("score"), "spend": doc.get("spend"),
        "spec_hash": doc.get("spec_hash"), "git_commit": doc.get("git_commit"),
        "dirty": doc.get("dirty"), "parent_run": doc.get("parent_run"),
        "fork_round": doc.get("fork_round"), "agents": len(agents), "models": models,
        "metrics_final": doc.get("metrics_final"),
        "rows": {f: t["rows"] for f, t in (doc.get("tables") or {}).items()},
        "export_schema": doc.get("export_schema"), "blobs": (doc.get("blobs") or {}).get("included"),
        "path": f"runs/{doc['run_id']}", "view": (run_dir / "view.html").exists(),
        "spec": doc.get("spec"),
    }


def _dump(data: Any) -> bytes:
    return (json.dumps(data, indent=2, sort_keys=True, default=str) + "\n").encode()


def _arm_doc(spec: dict) -> dict:
    """An arm's spec with participants collapsed into groups and the seed dropped."""
    groups: list[dict] = []
    for p in spec.get("participants") or []:
        if groups and groups[-1]["type"] == p.get("type") and groups[-1]["params"] == p.get("params"):
            groups[-1]["count"] += 1
        else:
            groups.append({"type": p.get("type"), "count": 1, "params": p.get("params") or {}})
    options = {k: v for k, v in (spec.get("options") or {}).items() if k != "seed"}
    out = {k: spec.get(k) for k in ("world", "medium", "metrics", "probes", "budget", "providers")
           if spec.get(k)}
    return {"participants": groups, **out, "options": options}


def _fmt_score(score: Any) -> str:
    if not isinstance(score, dict):
        return str(score)
    return ", ".join(f"{k}={v:.3g}" if isinstance(v, float) else f"{k}={v}"
                     for k, v in sorted(score.items()))


def render_card(repo: str, index: dict, *, public: bool, tags: Iterable[str] = ()) -> str:
    runs = [index["runs"][k] for k in sorted(index["runs"])]
    experiment = index.get("experiment") or repo.split("/")[-1]
    all_tags = ["swarmlab", "multi-agent", *sorted(set(tags))]
    if public:
        all_tags.append("format:agent-traces")
    configs = [{"config_name": "sessions", "default": True,
                "data_files": [{"split": "train", "path": "runs/*/sessions/*.jsonl"}]}]
    configs += [{"config_name": fam,
                 "data_files": [{"split": "train", "path": f"runs/*/tables/{fam}.parquet"}]}
                for fam in FAMILIES]
    front = yaml.safe_dump({"pretty_name": f"swarmlab: {experiment}", "tags": all_tags,
                            "configs": configs}, sort_keys=False)
    L = ["---", front.rstrip(), "---", "", f"# swarmlab experiment `{experiment}`", "",
         ("Runs of a [swarmlab](https://github.com/cmpatino/swarmlab) experiment: every run's "
          "event log as Parquet tables (one per event family), one pi-format session per agent "
          "(the Hub's agent-traces viewer renders them), and the raw log, snapshots and blobs "
          "from which `swarmlab fetch-published` rebuilds a run directory that `Run.load` "
          "replays."),
         "", "## Arms", ""]
    arms: dict[str, dict] = {}
    for r in runs:
        arms.setdefault(str(r.get("arm")), r.get("spec") or {})
    for arm, spec in arms.items():
        seeds = sorted({r.get("seed") for r in runs if str(r.get("arm")) == arm},
                       key=lambda s: (s is None, s))
        L += [f"### `{arm}`", "", f"Seeds: {', '.join(str(s) for s in seeds)}", "", "```yaml",
              yaml.safe_dump(_arm_doc(spec), sort_keys=False).rstrip(), "```", ""]
    L += ["## Runs", "",
          "| run | arm | seed | end reason | rounds | score | spend (swarm + meas.) | calls | view |",
          "|---|---|---|---|---|---|---|---|---|"]
    for r in runs:
        sp = r.get("spend") or {}
        view = (f"[view.html](https://huggingface.co/datasets/{repo}/resolve/main/"
                f"{r['path']}/view.html)") if r.get("view") else ""
        L.append(f"| `{r['run_id']}` | {r.get('arm')} | {r.get('seed')} | {r.get('end_reason')} | "
                 f"{r.get('last_round')} | {_fmt_score(r.get('score'))} | "
                 f"${float(sp.get('swarm') or 0):.3f} + ${float(sp.get('measurement') or 0):.3f} | "
                 f"{sp.get('calls', 0)} | {view} |")
    L += ["", "`view.html` is a self-contained replay page: download it and open it in a browser.",
          "", "## Layout", "", "```",
          "index.json                     one entry per run (identity, spec, score, spend, row counts)",
          "runs/<run_id>/run.json         identity, spec, score, end reason, spend, metric finals",
          "runs/<run_id>/tables/*.parquet one table per event family (schemas below)",
          "runs/<run_id>/sessions/*.jsonl one pi session (format v3, harness swarmlab) per agent",
          "runs/<run_id>/raw/             events.jsonl (byte-identical), run.json, snapshots, blobs",
          "runs/<run_id>/view.html        replay page (when published)",
          "```", "",
          (f"Export schema `{EXPORT_SCHEMA}`. Every table starts with the key columns "
           "`experiment, arm, seed, run, round, agent`, then `seq` and `ts` of the row's first "
           "event. JSON-valued fields are JSON text. Blob content is inlined up to 64 KiB per "
           "cell, else the cell holds `sha256:<hash>` and the blob is under `raw/blobs/`."),
          "", "## Table schemas", ""]
    for fam, schema in TABLE_SCHEMAS.items():
        L.append(f"- **{fam}**: " + ", ".join(f"`{f.name}` {f.type}" for f in schema))
    L += ["", "## Use", "", "```bash",
          f"swarmlab fetch-published {repo} <run_id> --out runs/   # rebuild a run dir",
          "swarmlab replay runs/<run_id>                            # replays to the recorded score",
          "```", "", "```python", "from datasets import load_dataset",
          f'turns = load_dataset("{repo}", "turns", split="train")', "```", ""]
    return "\n".join(L)


# ---- publish ---------------------------------------------------------------------------------
def _namespace(api: Any) -> str:
    who = api.whoami()
    name = who.get("name") if isinstance(who, dict) else None
    if not name:
        raise PublishError("cannot tell your Hub namespace; pass --repo owner/name")
    return str(name)


def _ensure_repo(api: Any, repo: str, public: bool) -> bool:
    """Create the dataset repo if needed; returns whether it is public afterwards."""
    api.create_repo(repo, repo_type=REPO_TYPE, private=not public, exist_ok=True)
    if public:
        api.update_repo_settings(repo, private=False, repo_type=REPO_TYPE)
        return True
    info = api.repo_info(repo, repo_type=REPO_TYPE)
    return getattr(info, "private", True) is False


def _sync(api: Any, repo: str, public: bool, tags: Iterable[str], entries: dict[str, dict],
          uploads: dict[str, bytes | Path], experiment: str | None, message: str) -> dict:
    is_public = _ensure_repo(api, repo, public)
    files = remote_files(api, repo)
    index = _remote_json(api, repo, "index.json", files) or {"runs": {}}
    index.setdefault("runs", {})
    index["runs"].update(entries)
    index["experiment"] = index.get("experiment") or experiment
    index["export_schema"] = EXPORT_SCHEMA
    uploads = {**uploads, "index.json": _dump(index),
               "README.md": render_card(repo, index, public=is_public, tags=tags).encode()}
    uploaded, unchanged = _commit(api, repo, uploads, files, message)
    return {"repo": repo, "url": f"https://huggingface.co/datasets/{repo}", "public": is_public,
            "uploaded": uploaded, "unchanged": unchanged, "runs": sorted(entries)}


def publish(source: Path | str, repo: str | None = None, *, public: bool = False,
            tag: Iterable[str] | str | None = None, api: Any = None,
            experiment: str | None = None) -> dict:
    """Export (if needed) and upload every finished run under `source` (only those of
    `experiment` when given). Returns a summary."""
    api = _api(api)
    tags = [tag] if isinstance(tag, str) else list(tag or [])
    runs, skipped = find_runs(source)
    if not runs:
        raise PublishError(f"no finished runs under {source}" +
                           (f" (skipped: {', '.join(skipped)})" if skipped else ""))
    by_exp: dict[str, list[Path]] = {}
    for d in runs:
        exp = json.loads((d / "run.json").read_text()).get("experiment") or "swarmlab"
        if experiment is not None and exp != experiment:
            continue
        by_exp.setdefault(exp, []).append(d)
    if not by_exp:
        raise PublishError(f"no finished runs of experiment {experiment!r} under {source}")
    if repo is not None and len(by_exp) > 1:
        raise PublishError(f"--repo names one repo but the runs belong to {sorted(by_exp)}; "
                           "publish one experiment at a time or omit --repo")
    results = []
    for exp, dirs in sorted(by_exp.items()):
        target = repo or f"{_namespace(api)}/{exp}"
        entries: dict[str, dict] = {}
        uploads: dict[str, bytes | Path] = {}
        for d in dirs:
            out = ensure_export(d)
            entry = index_entry(d, out)
            entries[entry["run_id"]] = entry
            for f in sorted(p for p in out.rglob("*") if p.is_file()):
                uploads[f"{entry['path']}/{f.relative_to(out).as_posix()}"] = f
            if (d / "view.html").exists():
                uploads[f"{entry['path']}/view.html"] = d / "view.html"
        results.append(_sync(api, target, public, tags, entries, uploads, exp,
                             f"swarmlab publish: {len(entries)} run(s) of {exp}"))
    return {"ok": True, "repos": results, "skipped": skipped}


def publish_view(run_dir: Path | str, repo: str, *, api: Any = None) -> dict:
    """Upload `<run_dir>/view.html` to `runs/<run_id>/view.html` and relink the card."""
    api = _api(api)
    run_dir = Path(run_dir)
    if not (run_dir / "view.html").exists():
        raise PublishError(f"{run_dir}/view.html does not exist; build it first (swarmlab view)")
    out = ensure_export(run_dir)
    entry = index_entry(run_dir, out)
    uploads: dict[str, bytes | Path] = {f"{entry['path']}/view.html": run_dir / "view.html"}
    return _sync(api, repo, False, [], {entry["run_id"]: entry}, uploads, entry.get("experiment"),
                 f"swarmlab view: {entry['run_id']}")


# ---- fetch -----------------------------------------------------------------------------------
def restore_run(run_prefix: Path, dest: Path) -> Path:
    """Rebuild a run dir at `dest` from a downloaded `runs/<run_id>/` folder."""
    raw = run_prefix / "raw"
    if not (raw / "events.jsonl").exists() or not (raw / "run.json").exists():
        raise PublishError(f"{run_prefix} has no raw/events.jsonl or raw/run.json")
    dest.mkdir(parents=True, exist_ok=True)
    for name in ("events.jsonl", "discarded.jsonl", "run.json"):
        if (raw / name).exists():
            shutil.copyfile(raw / name, dest / name)
    for sub in ("snapshots", "artifacts", "blobs"):
        if (raw / sub).is_dir():
            shutil.copytree(raw / sub, dest / sub, dirs_exist_ok=True)
    exp = dest / "export"
    exp.mkdir(exist_ok=True)
    for item in run_prefix.iterdir():
        if item.name in ("raw", "view.html"):
            continue
        target = exp / item.name
        if item.is_dir():
            shutil.copytree(item, target, dirs_exist_ok=True)
        else:
            shutil.copyfile(item, target)
    if (run_prefix / "view.html").exists():
        shutil.copyfile(run_prefix / "view.html", dest / "view.html")
    return dest


def fetch_published(repo: str, run_id: str, out: Path | str = "runs", *, api: Any = None,
                    force: bool = False) -> Path:
    api = _api(api)
    dest = Path(out) / run_id
    if dest.exists() and any(dest.iterdir()) and not force:
        raise PublishError(f"{dest} exists and is not empty (pass --force to overwrite)")
    with tempfile.TemporaryDirectory() as tmp:
        api.snapshot_download(repo, repo_type=REPO_TYPE, local_dir=tmp,
                              allow_patterns=[f"runs/{run_id}/*"])
        prefix = Path(tmp) / "runs" / run_id
        if not prefix.is_dir():
            raise PublishError(f"{repo} has no run {run_id!r} (see its index.json)")
        if dest.exists() and force:
            shutil.rmtree(dest)
        return restore_run(prefix, dest)
