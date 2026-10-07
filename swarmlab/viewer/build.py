"""Static replay page for one run (docs/INTERFACE.md §17).

`build(run_dir) -> Path` writes `<run_dir>/view.html`: one self-contained page (inline CSS and
vanilla JS, no external requests) that embeds a single JSON blob and folds it per round in the
browser, so viewing needs no Python.

Decisions where the contract is silent:

- Only committed rounds are shown: events with `round` above the last `round_committed` (a
  partial round of a crashed run) are dropped; `run_ended` is kept.
- The embedded blob (`<script type="application/json" id="swarmlab-data">`) is
  ``{"version": 1, "meta": {...}, "agents": [...], "world": {...} | None, "truth": str | None,
  "events": [...]}``. `events` is the logical view minus `seq`/`ts`/`run`, with three
  reductions to keep the page small: `run_started.run_spec` is dropped (the spec summary is in
  `meta`), `tool_returned.result` becomes `{"ok", "error", "summary"}` (a short text), and long
  strings in `tool_called.args` are truncated. `delivery` events gain `content`, the delivered
  text read from the blob store (it can differ from the post text when a policy rewrites it).
- `world` is filled only for a FlagGame (or subclass): the world plugin is rebuilt from the run
  spec and restored from the latest snapshot to read `candidates` and each agent's crop rows.
  Any other world, or a world class that cannot be imported, gives `world = None` and the page
  shows a committed-actions table instead of the flag panel.
- `truth` comes from `run.json` `score["truth"]` if present, else from the restored world's
  `verify()`; the page hides it until "reveal truth" is ticked.
- `world_states` (`{"<round>": state}` or None): `World.render_state()` of a fresh world built
  from the run spec and restored from each committed round's snapshot (the state after that
  round's commit), for the page's "World state" panel; rounds without a snapshot or whose
  state is None are left out, and a world that returns None for every round (FlagGame, which
  has its own panel; any world without the hook), cannot be imported, or fails to restore gives
  None and no panel. States are passed through `json` (non-JSON values become strings).
"""
from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from ..blobs import BlobStore
from ..events import EventLog, logical_view
from ..registry import build as build_plugin
from ..snapshot import SnapshotStore

TEMPLATE = Path(__file__).with_name("template.html")
MAX_ARG_CHARS = 400
MAX_SUMMARY_CHARS = 160
MAX_CONTENT_CHARS = 2000


def _clip(s: str, n: int) -> str:
    return s if len(s) <= n else s[: n - 1] + "…"


def _clip_args(value: Any) -> Any:
    if isinstance(value, str):
        return _clip(value, MAX_ARG_CHARS)
    if isinstance(value, dict):
        return {k: _clip_args(v) for k, v in value.items()}
    if isinstance(value, list):
        return [_clip_args(v) for v in value[:50]]
    return value


def _summarise_result(tool: str | None, result: Any) -> dict:
    if not isinstance(result, dict) or "ok" not in result:
        return {"ok": True, "error": None, "summary": _clip(json.dumps(result, default=str), MAX_SUMMARY_CHARS)}
    inner = result.get("result")
    if isinstance(inner, dict) and isinstance(inner.get("items"), list):
        items = inner["items"]
        ids = [str(i.get("post_id") or i.get("id") or "") for i in items if isinstance(i, dict)]
        text = f"{len(items)} item{'s' if len(items) != 1 else ''}"
        if ids and any(ids):
            text += ": " + ", ".join(ids[:6]) + (", …" if len(ids) > 6 else "")
    elif inner in (None, {}):
        text = ""
    else:
        text = _clip(json.dumps(inner, default=str, separators=(",", ":")), MAX_SUMMARY_CHARS)
    return {"ok": result.get("ok"), "error": result.get("error"), "summary": text}


def _events(run_dir: Path, blobs: BlobStore) -> tuple[list[dict], int]:
    raw = list(logical_view(EventLog(run_dir / "events.jsonl"), exclude=("seq", "ts", "run")))
    last = max((e["round"] for e in raw if e["type"] == "round_committed"), default=0)
    tools: dict[str, str] = {}
    out = []
    for e in raw:
        t = e["type"]
        if e["round"] > last and t != "run_ended":
            continue
        if t == "run_started":
            e = {k: v for k, v in e.items() if k != "run_spec"}
        elif t == "tool_called":
            tools[e["call_id"]] = e["tool"]
            e = {**e, "args": _clip_args(e.get("args"))}
        elif t == "tool_returned":
            e = {**e, "result": _summarise_result(tools.get(e["call_id"]), e.get("result"))}
        elif t == "post":
            e = {**e, "text": _clip(e.get("text") or "", MAX_CONTENT_CHARS)}
        elif t == "delivery":
            try:
                content = blobs.get_text(e["content_hash"])
            except Exception:  # noqa: BLE001 - a missing blob should not break the page
                content = None
            e = {k: v for k, v in e.items() if k != "content_hash"}
            e["content"] = None if content is None else _clip(content, MAX_CONTENT_CHARS)
        out.append(e)
    return out, last


def _world(run_dir: Path, meta: dict, blobs: BlobStore) -> tuple[dict | None, str | None]:
    from ..world.flaggame import FlagGame

    spec = meta.get("spec", {}).get("world") or {}
    try:
        world = build_plugin(spec, "swarmlab.worlds")
    except Exception:  # noqa: BLE001 - inline script worlds may not be importable here
        return None, None
    if not isinstance(world, FlagGame):
        return None, None
    snaps = SnapshotStore(run_dir, blobs)
    manifest = snaps.latest(meta.get("last_round"))
    if manifest is None or "world" not in manifest.plugins:
        return None, None
    world.restore(blobs.get(manifest.plugins["world"]))
    data = {
        "kind": "flaggame",
        "candidates": {n: list(g) for n, g in world.candidates.items()},
        "crops": {a: world.crop_rows(a) for a in world.agents},
    }
    return data, world.verify().get("truth")


def _world_states(run_dir: Path, meta: dict, blobs: BlobStore, last: int) -> dict | None:
    from ..world.base import World

    spec = meta.get("spec", {}).get("world") or {}
    try:
        if type(build_plugin(spec, "swarmlab.worlds")).render_state is World.render_state:
            return None  # no hook: skip restoring every round
    except Exception:  # noqa: BLE001 - inline script worlds may not be importable here
        return None
    snaps = SnapshotStore(run_dir, blobs)
    out: dict[str, Any] = {}
    for r in snaps.list_rounds():
        if not 1 <= r <= last:
            continue
        try:
            manifest = snaps.read(r)
            if "world" not in manifest.plugins:
                continue
            world = build_plugin(spec, "swarmlab.worlds")
            world.restore(blobs.get(manifest.plugins["world"]))
            state = world.render_state()
        except Exception:  # noqa: BLE001 - a world we cannot rebuild here just has no panel
            return None
        if state is None:
            continue
        out[str(r)] = json.loads(json.dumps(state, default=str))
    return out or None


def collect(run_dir: Path | str) -> dict:
    """The data blob the page embeds (see module docstring)."""
    run_dir = Path(run_dir)
    meta = json.loads((run_dir / "run.json").read_text())
    blobs = BlobStore(run_dir / "blobs")
    events, last = _events(run_dir, blobs)
    world, verified_truth = _world(run_dir, meta, blobs)
    score = meta.get("score") or {}
    truth = score.get("truth", verified_truth)
    agents = sorted({a for e in events if e["type"] == "round_started" for a in e["order"]})
    spec = meta.get("spec", {})
    return {
        "version": 1,
        "meta": {
            "run_id": meta.get("run_id"),
            "experiment": meta.get("experiment"),
            "arm": meta.get("arm"),
            "spec_hash": meta.get("spec_hash"),
            "status": meta.get("status"),
            "end_reason": meta.get("end_reason"),
            "last_round": last,
            "score": score,
            "parent_run": meta.get("parent_run"),
            "fork_round": meta.get("fork_round"),
            "world": spec.get("world"),
            "medium": spec.get("medium"),
            "options": spec.get("options"),
        },
        "agents": agents,
        "world": world,
        "world_states": _world_states(run_dir, meta, blobs, last),
        "truth": truth if isinstance(truth, str) else None,
        "events": events,
    }


def build(run_dir: Path | str) -> Path:
    """Write `<run_dir>/view.html` and return its path."""
    run_dir = Path(run_dir)
    data = collect(run_dir)
    blob = json.dumps(data, separators=(",", ":"), ensure_ascii=False)
    blob = blob.replace("<", "\\u003c")  # "<" only occurs inside JSON strings
    title = f"swarmlab · {data['meta']['run_id']}"
    page = TEMPLATE.read_text(encoding="utf-8")
    page = page.replace("__TITLE__", _html_escape(title)).replace("__DATA__", blob)
    out = run_dir / "view.html"
    out.write_text(page, encoding="utf-8")
    return out


def _html_escape(s: str) -> str:
    return s.replace("&", "&amp;").replace("<", "&lt;").replace(">", "&gt;")
