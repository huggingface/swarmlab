"""Export a finished run directory as analysis tables, pi-format sessions and raw copies
(docs/INTERFACE-M4.md §1). Read-only over the run directory.

`export_run(run_dir, out=None) -> Path` (CLI `swarmlab export RUN_DIR [--out DIR]`,
Python `Run.export(out)`) writes, by default into `<run_dir>/export/`:

```
run.json                 identity, spec, score, end reason, spend, metric finals, row counts
tables/<family>.parquet  one table per event family (TABLE_SCHEMAS)
sessions/<agent>.jsonl   one pi session (format v3, `harness: "swarmlab"`) per agent
raw/events.jsonl         byte-identical copy of the log (and raw/discarded.jsonl if present)
raw/run.json, raw/artifacts/, raw/snapshots/, raw/blobs/   what `fetch-published` needs to
                         rebuild a run dir that `Run.load` replays
```

Decisions where the contract is silent:

- **Reading.** The log is read as plain JSON lines (a torn trailing line is ignored, as in
  `EventLog`), not through the typed event union, so event types added later (M3a
  `intervention`, `overflow`) export without a code change.
- **Key columns.** Every table starts with `experiment, arm, seed, run, round, agent`, then `seq`
  and `ts` of the row's anchoring event (for joined families: the first event, e.g.
  `turn_started`, `tool_called`, `inference_attempt`, `round_started`). JSON-valued fields
  (`args`, `result`, `fields`, `parsed`, `feedback`, ...) are JSON text columns.
- **Families.** As listed in the contract, plus `other`: every event whose type no family
  covers (`world_changed`, `budget_changed`, `overflow`, unknown types) as `type` + `payload`
  JSON, so every logged event is represented somewhere. Event types folded into a family row:
  `turn_started`/`turn_ended` -> `turns`, `tool_called`/`tool_returned` -> `tool_calls`,
  `inference_attempt`/`inference_response` -> `inference` (paired by `call_id` in log order; a
  response with no attempt still gets a row), `round_started`/`budget`/`round_committed`/
  `snapshot` -> `rounds` (one row per round), `run_started`/`run_ended` -> `run` (one row).
- **reads.** One row per delivery id read (`read_index` numbers the `read` events of the run);
  a read that returned nothing gives one row with `delivery_id = null`, so empty reads (which the
  M2 reading-behaviour analysis counts) are not lost. Rows = sum(max(1, len(delivery_ids))).
- **Blob inlining.** A blob-referenced cell (`deliveries.content`, `inference.request` /
  `.response`, `probes.question` / `.raw`) holds the blob decoded as UTF-8 when it is at most
  64 KiB (`INLINE_LIMIT`); otherwise (or when not UTF-8) it holds `"sha256:<hash>"`, and the
  blob is guaranteed to be in `raw/blobs/`. A missing blob gives null. The hash column is always
  present next to it. A request blob with image parts (M5 §4, FlagGame `modality="image"`)
  is never inlined: `inference.request` holds `"sha256:<hash>"` at any size and the blob is
  kept in `raw/blobs/` like an oversize one.
- **Raw blobs.** `raw/blobs/` mirrors the run's `blobs/` (shards, `cache/` included) when the
  run directory (log, snapshots, artifacts, blobs) is under 500 MB (`FULL_BLOBS_LIMIT`). Above
  that it holds only the snapshot blobs (needed by `Run.load`) and the blobs too large to
  inline; `run.json["blobs"]` says which. The copy is one walk of each directory and one copy
  per file; a file that cannot be read (EIO on a bucket mount) is reported and the export fails
  with `ExportError` after trying every file, without writing `run.json`.
- **`raw=False`** (`--no-raw`): no `raw/` at all, `run.json["export_raw"] = false` and
  `blobs.included = "none"`; tables and sessions are complete, but `fetch-published` cannot
  rebuild such a run. Raw export reads every blob, which is slow on object storage
  (bucket-mounted run dirs): use `--no-raw` there, or export from a local copy.
- **Unreadable blobs.** A blob that the tables or sessions need and that exists but cannot be
  read (an I/O error, not a missing file) gives a null cell like a missing one and is listed in
  `run.json["unreadable_blobs"]` (`{sha: error}`); with raw copies on, the raw copy then fails
  as above.
- **Sessions** (pi session format v3, https://github.com/earendil-works/pi/blob/main/packages/
  coding-agent/docs/session-format.md, with the Hub's `harness` header field): header
  `{"type":"session","version":3,"id":"<run>/<agent>","timestamp","cwd":"runs/<run>",
  "harness":"swarmlab","name"}`, then `message` entries chained by `parentId`, ids are the first
  8 hex chars of sha256(`<run>/<agent>/<n>`). Messages are `system` (prompt + `toolsAdded`),
  `user`, `assistant` (`text` + `toolCall` blocks, `api`, `provider`, `model`, `usage`,
  `stopReason`, `rawStopReason`) and `toolResult` (`toolCallId`, `toolName`, `isError`).
  - LLM agents (agents with `swarm`-category inference): the system prompt and tools come from
    the first request blob; each request contributes the messages it adds to the previous one
    (the longest suffix/prefix overlap, so window memory works); assistant messages come from the
    response blobs (with usage and cost of the logged response); what follows the last response
    comes from the participant's memory in the last snapshot (unpickled with a loader that
    accepts only builtin types and `random.Random`). Native tool calls are matched in order to
    the round's `tool_called` events, and a result that no later request or the final memory
    holds (window memory drops it; no snapshot) is rebuilt from its `tool_returned` event, so
    every `toolCall` has a `toolResult`. Probe (`measurement`) calls are not part of the
    agent's conversation and are left out (they are in `tables/probes` and `inference`).
    A request blob that cannot be read (deleted, or an I/O error on a bucket mount) adds
    nothing and leaves the overlap reference at the last readable request, so the next request
    contributes only what is new (before this, it was emitted in full: a second `Round 1.` with
    later rounds under it); its response is emitted where the next readable request places the
    assistant message (or right after it, when window memory dropped it). With window memory the
    lost round's user message cannot be recovered and is missing from the session.
  - An image part of a message becomes a text block `[image: PNG <w>×<h>]` (M5 §4,
    `render.image_label`); the pixels stay in the request blobs.
  - pi `usage.cost` holds the logged `cost_usd` in `total` only (the run does not split cost
    by token kind); `api` is `anthropic-messages` for Anthropic and `openai-completions`
    otherwise.
  - Scripted agents: per round a `user` message `Round <r>.` and, per tool call, one assistant
    message with that `toolCall` (`api: "swarmlab-scripted"`, `provider: "scripted"`, `model`:
    the participant type, zero usage) followed by its `toolResult`.
- **Discarded rounds.** `tables/discarded_inference.parquet` holds the `inference_attempt`/
  `inference_response` events of `discarded.jsonl` (rounds aborted at a hard ceiling or lost in
  a crash), with the `inference` columns plus `charged_usd` (0 for a cache hit, `cost_usd` for a
  response, `reserved_usd` for an attempt cancelled in flight: the gate charges the worst case).
  `run.json["spend_discarded_usd"]` is their sum. The ledger (`spend`) includes what hard-ceiling
  aborts charged, so `spend` minus the logged `inference.cost_usd` is about
  `spend_discarded_usd`, not 0. `EXPORT_SCHEMA` 2 added this table.
- `EXPORT_SCHEMA` is recorded in `run.json["export_schema"]`; outputs are deterministic for a
  given run directory and code version (no export timestamp), so re-publishing an unchanged run
  uploads nothing.
"""
from __future__ import annotations

import datetime as dt
import hashlib
import io
import json
import os
import pickle
import shutil
from collections import defaultdict
from collections.abc import Iterator
from pathlib import Path
from typing import Any

import pyarrow as pa
import pyarrow.parquet as pq

from .world.render import image_label

EXPORT_SCHEMA = "swarmlab-export/2"
INLINE_LIMIT = 64 * 1024
IMAGE_MARK = b'"type":"image"'  # an image Part in a canonical-JSON request blob
FULL_BLOBS_LIMIT = 500 * 1024 * 1024
KEY_COLUMNS = ("experiment", "arm", "seed", "run", "round", "agent")

_S, _I, _F, _B = pa.string(), pa.int64(), pa.float64(), pa.bool_()
_KEYS = [("experiment", _S), ("arm", _S), ("seed", _I), ("run", _S), ("round", _I),
         ("agent", _S), ("seq", _I), ("ts", _F)]
_USAGE = [("prompt_tokens", _I), ("completion_tokens", _I), ("cached_prompt_tokens", _I),
          ("reasoning_tokens", _I)]


def _schema(*cols: tuple[str, pa.DataType]) -> pa.Schema:
    return pa.schema([pa.field(n, t) for n, t in (*_KEYS, *cols)])


TABLE_SCHEMAS: dict[str, pa.Schema] = {
    "turns": _schema(("yield_kind", _S), ("calls", _I), *_USAGE, ("cost_usd", _F),
                     ("inference_calls", _I), ("finish_reasons", _S), ("notes", _S),
                     ("usage", _S), ("error", _S), ("private", _S), ("end_seq", _I)),
    "tool_calls": _schema(("call_id", _S), ("tool", _S), ("args", _S), ("ok", _B),
                          ("pending", _B), ("error", _S), ("result", _S), ("return_seq", _I)),
    "posts": _schema(("post_id", _S), ("provisional_id", _S), ("channel", _S), ("text", _S),
                     ("fields", _S)),
    "deliveries": _schema(("post_id", _S), ("recipient", _S), ("delivery_id", _S),
                          ("eligible_round", _I), ("content_hash", _S), ("content", _S)),
    "reads": _schema(("read_index", _I), ("delivery_id", _S)),
    "actions": _schema(("action_id", _S), ("action_name", _S), ("action_args", _S),
                       ("action", _S), ("accepted", _B), ("feedback", _S)),
    "inference": _schema(("call_id", _S), ("category", _S), ("provider", _S), ("model", _S),
                         ("served_by", _S), ("request_hash", _S), ("response_hash", _S),
                         *_USAGE, ("cost_usd", _F), ("reserved_usd", _F), ("latency_s", _F),
                         ("finish_reason", _S), ("cached", _B), ("attempts", _I),
                         ("request", _S), ("response", _S), ("response_seq", _I)),
    "probes": _schema(("probe", _S), ("ok", _B), ("candidate", _S), ("parsed", _S),
                      ("cost_usd", _F), ("question_hash", _S), ("question", _S),
                      ("raw_hash", _S), ("raw", _S)),
    "metrics": _schema(("name", _S), ("value", _F), ("denominator", _I)),
    "interventions": _schema(("intervention", _S), ("op", _S), ("ok", _B), ("affected", _S),
                             ("payload", _S)),
    "rounds": _schema(("order", _S), ("n_agents", _I), ("committed", _B), ("n_posts", _I),
                      ("n_actions", _I), ("n_deliveries", _I), ("spent_swarm", _F),
                      ("spent_measurement", _F), ("reserved", _F), ("calls", _I),
                      ("snapshot", _S), ("commit_seq", _I)),
    "run": _schema(("spec_hash", _S), ("git_commit", _S), ("dirty", _B), ("parent_run", _S),
                   ("fork_round", _I), ("end_reason", _S), ("end_round", _I), ("end_seq", _I),
                   ("end_ts", _F), ("run_spec", _S)),
    "other": _schema(("type", _S), ("payload", _S)),
}
# discarded rounds (discarded.jsonl): the inference columns plus what the ledger was charged
TABLE_SCHEMAS["discarded_inference"] = pa.schema(
    [*TABLE_SCHEMAS["inference"], pa.field("charged_usd", _F)])
FAMILIES = tuple(TABLE_SCHEMAS)

_BASE_FIELDS = {"seq", "run", "round", "agent", "ts", "type"}
_COVERED = {"turn_started", "turn_ended", "tool_called", "tool_returned", "post", "delivery",
            "read", "action_committed", "inference_attempt", "inference_response", "probe",
            "metric", "intervention", "round_started", "round_committed", "budget", "snapshot",
            "run_started", "run_ended"}


# ---- helpers ---------------------------------------------------------------------------------
def _j(value: Any) -> str | None:
    return None if value is None else json.dumps(value, sort_keys=True, ensure_ascii=False)


def read_events(path: Path | str) -> Iterator[dict]:
    """Events of a JSONL log as dicts; a torn or unparsable trailing line is ignored."""
    p = Path(path)
    if not p.exists():
        return
    with open(p, "rb") as f:
        lines = f.read().split(b"\n")
    lines.pop()  # b"" when the file ends with a newline, else a torn write
    for i, raw in enumerate(lines):
        if not raw.strip():
            continue
        try:
            yield json.loads(raw)
        except ValueError:
            if i == len(lines) - 1:
                return
            raise


def _iso(ts: float | None) -> str:
    t = dt.datetime.fromtimestamp(ts or 0.0, dt.UTC)
    return t.isoformat(timespec="milliseconds").replace("+00:00", "Z")


def _ms(ts: float | None) -> int:
    return round((ts or 0.0) * 1000)


class ExportError(RuntimeError):
    """The export could not copy or read part of the run directory (see the message)."""


class _Blobs:
    """Read-only access to a run's blob store plus the inlining rule. A missing blob reads as
    None; any other read error (EIO on a bucket mount, permissions) also reads as None and is
    recorded in `unreadable` so the export reports it."""

    def __init__(self, root: Path) -> None:
        self.root = root
        self.oversize: set[str] = set()
        self.unreadable: dict[str, str] = {}

    def path(self, sha: str) -> Path:
        return self.root / sha[:2] / sha

    def get(self, sha: str | None) -> bytes | None:
        if not sha or len(sha) != 64:
            return None
        try:
            return self.path(sha).read_bytes()
        except FileNotFoundError:
            return None
        except OSError as e:
            self.unreadable[sha] = f"{type(e).__name__}: {e}"
            return None

    def json(self, sha: str | None) -> Any:
        data = self.get(sha)
        if data is None:
            return None
        try:
            return json.loads(data)
        except ValueError:
            return None

    def inline(self, sha: str | None, *, images_by_ref: bool = False) -> str | None:
        data = self.get(sha)
        if data is None:
            return None
        if images_by_ref and IMAGE_MARK in data:
            pass  # M5 §4: a request with image parts is kept by hash only
        elif len(data) <= INLINE_LIMIT:
            try:
                return data.decode("utf-8")
            except UnicodeDecodeError:
                pass
        self.oversize.add(str(sha))
        return f"sha256:{sha}"


class _SafeUnpickler(pickle.Unpickler):
    """Builtin containers and scalars plus `random.Random` (a bound participant's rng):
    participant memory is plain data, and nothing else is ever constructed."""

    def find_class(self, module: str, name: str) -> Any:
        if (module, name) == ("random", "Random"):
            import random

            return random.Random
        raise pickle.UnpicklingError(f"refusing global {module}.{name}")


def _safe_unpickle(data: bytes) -> Any:
    return _SafeUnpickler(io.BytesIO(data)).load()


# ---- tables ----------------------------------------------------------------------------------
def build_tables(events: list[dict], meta: dict, blobs: _Blobs) -> dict[str, list[dict]]:
    """Rows per family (see TABLE_SCHEMAS) from the raw events of one run."""
    spec = meta.get("spec") or {}
    base = {"experiment": meta.get("experiment") or spec.get("experiment"),
            "arm": meta.get("arm"), "seed": (spec.get("options") or {}).get("seed"),
            "run": meta.get("run_id")}
    rows: dict[str, list[dict]] = {f: [] for f in FAMILIES}

    def key(ev: dict) -> dict:
        return {**base, "run": ev.get("run") or base["run"], "round": ev.get("round"),
                "agent": ev.get("agent"), "seq": ev.get("seq"), "ts": ev.get("ts")}

    open_turn: dict[str, dict] = {}
    open_call: dict[str, dict] = {}
    open_inf: dict[str, list[dict]] = defaultdict(list)
    rounds: dict[int, dict] = {}
    run_row: dict | None = None
    read_index = 0

    def round_row(ev: dict) -> dict:
        r = ev["round"]
        if r not in rounds:
            rounds[r] = {**key(ev), "agent": None, "committed": False}
            rows["rounds"].append(rounds[r])
        return rounds[r]

    for ev in events:
        t = ev.get("type")
        if t == "turn_started":
            row = {**key(ev), "private": _j(ev.get("private") or {})}
            open_turn[str(ev.get("agent"))] = row
        elif t == "turn_ended":
            row = open_turn.pop(str(ev.get("agent")), None) or key(ev)
            usage = ev.get("usage") or {}
            row.update(yield_kind=ev.get("yield_kind"), calls=ev.get("calls"),
                       cost_usd=usage.get("cost_usd"), inference_calls=usage.get("inference_calls"),
                       finish_reasons=_j(usage.get("finish_reasons")), notes=_j(usage.get("notes")),
                       usage=_j(usage), error=ev.get("error"), end_seq=ev.get("seq"),
                       **{k: usage.get(k) for k, _ in _USAGE})
            rows["turns"].append(row)
        elif t == "tool_called":
            row = {**key(ev), "call_id": ev.get("call_id"), "tool": ev.get("tool"),
                   "args": _j(ev.get("args") or {})}
            open_call[str(ev.get("call_id"))] = row
            rows["tool_calls"].append(row)
        elif t == "tool_returned":
            row = open_call.pop(str(ev.get("call_id")), None)
            if row is None:
                row = {**key(ev), "call_id": ev.get("call_id")}
                rows["tool_calls"].append(row)
            res = ev.get("result") or {}
            row.update(ok=res.get("ok"), pending=ev.get("pending"), error=res.get("error"),
                       result=_j(res.get("result")), return_seq=ev.get("seq"))
        elif t == "post":
            rows["posts"].append({**key(ev), "post_id": ev.get("post_id"),
                                  "provisional_id": ev.get("provisional_id"),
                                  "channel": ev.get("channel"), "text": ev.get("text"),
                                  "fields": _j(ev.get("fields") or {})})
        elif t == "delivery":
            rows["deliveries"].append({**key(ev), "post_id": ev.get("post_id"),
                                       "recipient": ev.get("recipient"),
                                       "delivery_id": ev.get("delivery_id"),
                                       "eligible_round": ev.get("eligible_round"),
                                       "content_hash": ev.get("content_hash"),
                                       "content": blobs.inline(ev.get("content_hash"))})
        elif t == "read":
            ids = ev.get("delivery_ids") or []
            for did in ids or [None]:
                rows["reads"].append({**key(ev), "read_index": read_index, "delivery_id": did})
            read_index += 1
        elif t == "action_committed":
            act = ev.get("action") or {}
            rows["actions"].append({**key(ev), "action_id": ev.get("action_id"),
                                    "action_name": act.get("name"),
                                    "action_args": _j(act.get("args")), "action": _j(act),
                                    "accepted": ev.get("accepted"),
                                    "feedback": _j(ev.get("feedback") or {})})
        elif t == "inference_attempt":
            row = {**key(ev), "call_id": ev.get("call_id"), "category": ev.get("category"),
                   "provider": ev.get("provider"), "model": ev.get("model"),
                   "request_hash": ev.get("request_hash"), "reserved_usd": ev.get("reserved_usd"),
                   "request": blobs.inline(ev.get("request_hash"), images_by_ref=True)}
            open_inf[str(ev.get("call_id"))].append(row)
            rows["inference"].append(row)
        elif t == "inference_response":
            pending = open_inf.get(str(ev.get("call_id")))
            if pending:
                row = pending.pop(0)
            else:
                row = {**key(ev), "call_id": ev.get("call_id")}
                rows["inference"].append(row)
            usage = ev.get("usage") or {}
            row.update(response_hash=ev.get("response_hash") or None,
                       cost_usd=ev.get("cost_usd"), latency_s=ev.get("latency_s"),
                       served_by=ev.get("served_by"), finish_reason=ev.get("finish_reason"),
                       cached=ev.get("cached"), attempts=ev.get("attempts", 1),
                       response=blobs.inline(ev.get("response_hash")),
                       response_seq=ev.get("seq"), **{k: usage.get(k) for k, _ in _USAGE})
        elif t == "probe":
            parsed = ev.get("parsed") or {}
            cand = parsed.get("candidate") if isinstance(parsed, dict) else None
            rows["probes"].append({**key(ev), "probe": ev.get("probe"), "ok": ev.get("ok"),
                                   "candidate": cand if isinstance(cand, str) else None,
                                   "parsed": _j(parsed), "cost_usd": ev.get("cost_usd"),
                                   "question_hash": ev.get("question_hash") or None,
                                   "question": blobs.inline(ev.get("question_hash")),
                                   "raw_hash": ev.get("raw_hash") or None,
                                   "raw": blobs.inline(ev.get("raw_hash"))})
        elif t == "metric":
            v = ev.get("value")
            rows["metrics"].append({**key(ev), "name": ev.get("name"),
                                    "value": None if v is None else float(v),
                                    "denominator": ev.get("denominator")})
        elif t == "intervention":
            payload = {k: v for k, v in ev.items() if k not in _BASE_FIELDS}
            aff = payload.get("affected")
            rows["interventions"].append({
                **key(ev), "intervention": payload.get("intervention", payload.get("name")),
                "op": payload.get("op"), "ok": payload.get("ok"),
                "affected": _j(aff), "payload": _j(payload)})
        elif t == "round_started":
            rr = round_row(ev)
            rr.update(order=_j(ev.get("order")), n_agents=len(ev.get("order") or []))
        elif t == "budget":
            round_row(ev).update(spent_swarm=ev.get("spent_swarm"),
                                 spent_measurement=ev.get("spent_measurement"),
                                 reserved=ev.get("reserved"), calls=ev.get("calls"))
        elif t == "round_committed":
            round_row(ev).update(committed=True, n_posts=ev.get("n_posts"),
                                 n_actions=ev.get("n_actions"),
                                 n_deliveries=ev.get("n_deliveries"), commit_seq=ev.get("seq"))
        elif t == "snapshot":
            round_row(ev).update(snapshot=ev.get("manifest_path"))
        elif t == "run_started":
            run_row = {**key(ev), "spec_hash": ev.get("spec_hash"),
                       "git_commit": ev.get("git_commit"), "dirty": ev.get("dirty"),
                       "parent_run": ev.get("parent_run"), "fork_round": ev.get("fork_round"),
                       "run_spec": _j(ev.get("run_spec"))}
            rows["run"].append(run_row)
        elif t == "run_ended":
            if run_row is None:
                run_row = key(ev)
                rows["run"].append(run_row)
            # a resumed run may log several run_ended events; the row keeps the last
            run_row.update(end_reason=ev.get("reason"), end_round=ev.get("round"),
                           end_seq=ev.get("seq"), end_ts=ev.get("ts"))
        else:
            payload = {k: v for k, v in ev.items() if k not in _BASE_FIELDS}
            rows["other"].append({**key(ev), "type": t, "payload": _j(payload)})
    rows["turns"].extend(open_turn.values())  # a turn without turn_ended (should not happen)
    return rows


def discarded_inference_rows(run_dir: Path | str, meta: dict,
                             blobs: _Blobs | None = None) -> list[dict]:
    """`discarded_inference` rows: the inference attempts/responses in `discarded.jsonl`.

    `charged_usd` is what the ledger was charged for the call: 0 for a cache hit, the logged
    `cost_usd` for a response, and the reservation (`reserved_usd`, the worst case) for an
    attempt with no response (an in-flight call cancelled when the round was aborted)."""
    run_dir = Path(run_dir)
    blobs = blobs if blobs is not None else _Blobs(run_dir / "blobs")
    events = list(read_events(run_dir / "discarded.jsonl"))
    rows = build_tables(events, meta, blobs)["inference"]
    for row in rows:
        if row.get("response_seq") is None:
            row["charged_usd"] = float(row.get("reserved_usd") or 0.0)
        else:
            row["charged_usd"] = 0.0 if row.get("cached") else float(row.get("cost_usd") or 0.0)
    return rows


def discarded_spend(run_dir: Path | str) -> float:
    """USD charged for inference in discarded rounds (sum of `charged_usd`)."""
    run_dir = Path(run_dir)
    if not (run_dir / "discarded.jsonl").exists():
        return 0.0
    try:
        meta = json.loads((run_dir / "run.json").read_text())
    except (OSError, ValueError):
        meta = {}
    return sum(r["charged_usd"] for r in discarded_inference_rows(run_dir, meta))


def write_table(rows: list[dict], family: str, path: Path) -> int:
    schema = TABLE_SCHEMAS[family]
    names = schema.names
    table = pa.Table.from_pylist([{n: r.get(n) for n in names} for r in rows], schema=schema)
    path.parent.mkdir(parents=True, exist_ok=True)
    pq.write_table(table, path, compression="zstd")
    return table.num_rows


# ---- sessions --------------------------------------------------------------------------------
_STOP = {"tool_use": "toolUse", "tool_calls": "toolUse", "function_call": "toolUse",
         "end_turn": "stop", "stop": "stop", "stop_sequence": "stop", "length": "length",
         "max_tokens": "length"}


def _stop_reason(finish: str | None, has_calls: bool) -> str:
    if finish and finish.startswith("error"):
        return "error"
    if finish in _STOP:
        return _STOP[finish]
    return "toolUse" if has_calls else "stop"


def _content_blocks(content: Any) -> str | list[dict]:
    """ChatMessage content (str or Part list) as pi user content."""
    if isinstance(content, str):
        return content
    out: list[dict] = []
    for p in content or []:
        if p.get("type") == "image" or p.get("image_png_b64"):
            out.append({"type": "text", "text": image_label(p.get("image_png_b64"))})  # M5 §4
        else:
            out.append({"type": "text", "text": p.get("text") or ""})
    return out


def _text_of(content: Any) -> str:
    if isinstance(content, str):
        return content
    return "\n".join(p.get("text") or "" for p in content or [] if p.get("type") != "image")


def _usage(usage: dict | None, cost: float | None) -> dict:
    u = usage or {}
    prompt = int(u.get("prompt_tokens") or 0)
    cached = int(u.get("cached_prompt_tokens") or 0)
    out = int(u.get("completion_tokens") or 0)
    d = {"input": max(0, prompt - cached), "output": out, "cacheRead": cached, "cacheWrite": 0,
         "totalTokens": prompt + out,
         "cost": {"input": 0.0, "output": 0.0, "cacheRead": 0.0, "cacheWrite": 0.0,
                  "total": float(cost or 0.0)}}
    if u.get("reasoning_tokens"):
        d["reasoning"] = int(u["reasoning_tokens"])
    return d


class _Session:
    def __init__(self, run_id: str, agent: str) -> None:
        self.run_id = run_id
        self.agent = agent
        self.entries: list[dict] = []
        self._ids: set[str] = set()
        self.tool_names: dict[str, str] = {}

    def _id(self) -> str:
        n = len(self.entries)
        digest = hashlib.sha256(f"{self.run_id}/{self.agent}/{n}".encode()).hexdigest()
        for width in (8, 12, 16, 64):
            if digest[:width] not in self._ids:
                return digest[:width]
        raise AssertionError("unreachable")

    def add(self, message: dict, ts: float | None) -> None:
        eid = self._id()
        self._ids.add(eid)
        parent = self.entries[-1]["id"] if self.entries else None
        message = {**message, "timestamp": _ms(ts)}
        self.entries.append({"type": "message", "id": eid, "parentId": parent,
                             "timestamp": _iso(ts), "message": message})

    # -- message builders ---------------------------------------------------------------------
    def system(self, text: str, tools: list[dict], ts: float | None) -> None:
        self.add({"role": "system", "content": text,
                  "toolsAdded": [{"name": t.get("name"), "description": t.get("description", ""),
                                  "parameters": t.get("parameters") or {}} for t in tools]}, ts)

    def user(self, content: Any, ts: float | None) -> None:
        self.add({"role": "user", "content": _content_blocks(content)}, ts)

    def assistant(self, text: str, calls: list[dict], *, provider: str, model: str, api: str,
                  usage: dict, finish: str | None, ts: float | None) -> None:
        blocks: list[dict] = [{"type": "text", "text": text}] if text else []
        for c in calls:
            cid = str(c.get("call_id"))
            self.tool_names[cid] = str(c.get("name"))
            blocks.append({"type": "toolCall", "id": cid, "name": c.get("name"),
                           "arguments": c.get("args") or {}})
        msg = {"role": "assistant", "content": blocks, "api": api, "provider": provider,
               "model": model, "usage": usage, "stopReason": _stop_reason(finish, bool(calls))}
        if finish:
            msg["rawStopReason"] = finish
        self.add(msg, ts)

    def tool_result(self, call_id: str, text: str, ts: float | None,
                    name: str | None = None) -> None:
        try:
            payload = json.loads(text)
            is_error = not bool(payload.get("ok", True)) if isinstance(payload, dict) else False
        except ValueError:
            is_error = False
        self.add({"role": "toolResult", "toolCallId": call_id,
                  "toolName": name or self.tool_names.get(call_id, ""),
                  "content": [{"type": "text", "text": text}], "isError": is_error}, ts)

    def lines(self, name: str, ts: float | None) -> list[str]:
        header = {"type": "session", "version": 3, "id": f"{self.run_id}/{self.agent}",
                  "timestamp": _iso(ts), "cwd": f"runs/{self.run_id}", "harness": "swarmlab",
                  "name": name}
        return [json.dumps(header, ensure_ascii=False)] + [
            json.dumps(e, ensure_ascii=False) for e in self.entries]


def _overlap(prev: list[dict], cur: list[dict]) -> int:
    """Largest j with cur[:j] == prev[-j:] (how much of `cur` the previous request held)."""
    for j in range(min(len(prev), len(cur)), 0, -1):
        if cur[:j] == prev[len(prev) - j:]:
            return j
    return 0


def _final_memory(run_dir: Path, blobs: _Blobs, agent: str) -> list[dict] | None:
    """[system] + memory of an LLMAgent from the latest snapshot, or None."""
    snaps = sorted((run_dir / "snapshots").glob("*.json"))
    if not snaps:
        return None
    try:
        manifest = json.loads(snaps[-1].read_text())
        data = blobs.get(manifest.get("plugins", {}).get(f"participant:{agent}"))
        state = _safe_unpickle(data) if data is not None else None
    except (OSError, ValueError, pickle.UnpicklingError, EOFError):
        return None
    if not isinstance(state, dict) or not isinstance(state.get("rounds"), list):
        return None
    msgs = [{"role": "system", "content": state["system"], "tool_calls": None,
             "tool_call_id": None}] if state.get("system") else []
    for r in state["rounds"]:
        msgs.extend(r.get("messages") or [])
    return msgs


def _norm(m: dict) -> dict:
    """Comparable form of a ChatMessage dict (request blobs and memory dumps agree on it)."""
    return {"role": m.get("role"), "content": m.get("content"),
            "tool_calls": m.get("tool_calls") or None, "tool_call_id": m.get("tool_call_id")}


def _payload_text(ret: dict | None) -> str:
    """The tool message an LLMAgent writes for a `tool_returned` event."""
    res = (ret or {}).get("result") or {}
    return json.dumps({"ok": res.get("ok"), "result": res.get("result"),
                       "error": res.get("error")}, sort_keys=True)


def _llm_session(sess: _Session, calls: list[tuple[dict, dict | None]], blobs: _Blobs,
                 tail: list[dict] | None, tool_events: dict[int, list[tuple[dict, dict | None]]],
                 end_ts: float | None) -> None:
    """Rebuild an LLM agent's conversation (see the module doc).

    Native tool calls of each response are matched, in order, to the executor's `tool_called`
    events of that round; when a call's result never shows up in a later request or in the
    final memory (window memory drops it at the next round), the result is taken from the
    matching `tool_returned` event.
    """
    prev: list[dict] = []
    first = True
    pending: dict[str, tuple[dict, dict | None] | None] = {}
    cursor: dict[int, int] = defaultdict(int)
    # responses whose request blob could not be read: their assistant message is emitted where
    # the next readable request places it (after the user message of its round), not before
    deferred: list[tuple[dict, dict, dict]] = []
    unseen = 0  # responses emitted since the last readable request (their assistant messages
    #             open that request's new segment and are skipped there)

    def flush(ts: float | None) -> None:
        for cid, pair in list(pending.items()):
            if pair is not None:
                sess.tool_result(cid, _payload_text(pair[1]), (pair[1] or pair[0]).get("ts") or ts)
            else:
                sess.tool_result(cid, json.dumps({"ok": False, "result": {}, "error":
                                                  "arguments are not a JSON object; nothing was "
                                                  "executed"}, sort_keys=True), ts)
        pending.clear()

    def answer(att: dict, resp_ev: dict, resp: dict) -> None:
        nonlocal unseen
        flush(att.get("ts"))
        provider = str(resp.get("provider") or att.get("provider") or "")
        rcalls = list(resp.get("tool_calls") or [])
        sess.assistant(str(resp.get("text") or ""), rcalls,
                       provider=provider, model=str(resp.get("model") or att.get("model") or ""),
                       api="anthropic-messages" if provider == "anthropic" else "openai-completions",
                       usage=_usage(resp_ev.get("usage") or resp.get("usage"),
                                    resp_ev.get("cost_usd")),
                       finish=resp.get("finish_reason"), ts=resp_ev.get("ts"))
        unseen += 1
        rnd = int(att.get("round") or 0)
        evs = tool_events.get(rnd, [])
        for c in rcalls:
            if set(c.get("args") or {}) == {"_raw"}:
                pending[str(c.get("call_id"))] = None
            elif cursor[rnd] < len(evs):
                pending[str(c.get("call_id"))] = evs[cursor[rnd]]
                cursor[rnd] += 1

    def emit(msgs: list[dict], ts: float | None) -> None:
        nonlocal unseen
        skip, unseen = unseen, 0
        for m in msgs:
            role = m["role"]
            if role == "assistant":
                if skip:
                    skip -= 1  # already emitted from its response blob, with usage
                elif deferred:
                    answer(*deferred.pop(0))
                    unseen = 0  # it is in this segment, not in the next request's
                continue
            if role == "tool":
                cid = str(m.get("tool_call_id"))
                pending.pop(cid, None)
                sess.tool_result(cid, _text_of(m["content"]), ts)
            elif role == "user":
                flush(ts)
                sess.user(m["content"], ts)

    for att, resp_ev in calls:
        req = blobs.json(att.get("request_hash"))
        resp = blobs.json((resp_ev or {}).get("response_hash"))
        if req is None:  # unreadable request blob: keep `prev`, so nothing is emitted twice
            if resp_ev is not None and resp is not None:
                deferred.append((att, resp_ev, resp))
            continue
        msgs = [_norm(m) for m in req.get("messages") or []]
        ts = att.get("ts")
        if first:
            if msgs and msgs[0]["role"] == "system":
                sess.system(_text_of(msgs[0]["content"]), req.get("tools") or [], ts)
            first = False
        body = [m for m in msgs if m["role"] != "system"]
        emit(body[_overlap(prev, body):], ts)
        prev = body
        while deferred:  # the request did not hold them (window memory): keep them, in order
            answer(*deferred.pop(0))
        if resp_ev is None or resp is None:
            continue  # no response (provider error or aborted)
        answer(att, resp_ev, resp)
    if tail is not None:
        body = [_norm(m) for m in tail if m.get("role") != "system"]
        j = _overlap(prev, body)
        if j or not prev:  # memory extends the last request (else: an aborted last round)
            emit(body[j:], end_ts)
    while deferred:
        answer(*deferred.pop(0))
    flush(end_ts)


def _scripted_session(sess: _Session, agent_events: list[dict], ptype: str) -> None:
    current_round = None
    calls: dict[str, dict] = {}
    for ev in agent_events:
        t = ev.get("type")
        if t == "turn_started":
            current_round = ev.get("round")
            sess.user(f"Round {current_round}.", ev.get("ts"))
        elif t == "tool_called":
            calls[str(ev.get("call_id"))] = ev
            sess.assistant("", [{"call_id": ev.get("call_id"), "name": ev.get("tool"),
                                 "args": ev.get("args") or {}}],
                           provider="scripted", model=ptype, api="swarmlab-scripted",
                           usage=_usage(None, 0.0), finish="tool_use", ts=ev.get("ts"))
        elif t == "tool_returned":
            res = ev.get("result") or {}
            sess.tool_result(str(ev.get("call_id")), json.dumps(
                {"ok": res.get("ok"), "result": res.get("result"), "error": res.get("error")},
                sort_keys=True), ev.get("ts"))


def build_sessions(run_dir: Path, events: list[dict], meta: dict,
                   blobs: _Blobs) -> dict[str, list[str]]:
    """pi session lines per agent."""
    run_id = str(meta.get("run_id"))
    spec = meta.get("spec") or {}
    participants = spec.get("participants") or []
    agents = [f"a{i:03d}" for i in range(len(participants))]
    by_agent: dict[str, list[dict]] = defaultdict(list)
    for ev in events:
        if ev.get("agent") is not None:
            by_agent[str(ev["agent"])].append(ev)
    for a in by_agent:
        if a not in agents:
            agents.append(a)
    out: dict[str, list[str]] = {}
    for i, agent in enumerate(agents):
        evs = by_agent.get(agent, [])
        pspec = participants[i] if i < len(participants) else {}
        ptype = str(pspec.get("type") or "unknown")
        model = (pspec.get("params") or {}).get("model")
        sess = _Session(run_id, agent)
        # swarm inference calls: attempt + its response, in log order
        pending: dict[str, list[dict]] = defaultdict(list)
        calls: list[tuple[dict, dict | None]] = []
        index: dict[int, int] = {}
        for ev in evs:
            if ev.get("type") == "inference_attempt" and ev.get("category", "swarm") == "swarm":
                index[id(ev)] = len(calls)
                calls.append((ev, None))
                pending[str(ev.get("call_id"))].append(ev)
            elif ev.get("type") == "inference_response" and pending.get(str(ev.get("call_id"))):
                att = pending[str(ev.get("call_id"))].pop(0)
                calls[index[id(att)]] = (att, ev)
        start_ts = evs[0].get("ts") if evs else None
        if calls:
            tool_events = _tool_events_by_round(evs)
            tail = _final_memory(run_dir, blobs, agent)
            last_ts = max((e.get("ts") or 0.0) for e in evs) if evs else None
            _llm_session(sess, calls, blobs, tail, tool_events, last_ts)
            label = model or ptype
        else:
            _scripted_session(sess, evs, ptype)
            label = ptype
        out[agent] = sess.lines(f"{run_id} {agent} ({label})", start_ts)
    return out


def _tool_events_by_round(evs: list[dict]) -> dict[int, list[tuple[dict, dict | None]]]:
    rets = {str(e.get("call_id")): e for e in evs if e.get("type") == "tool_returned"}
    out: dict[int, list[tuple[dict, dict | None]]] = defaultdict(list)
    for e in evs:
        if e.get("type") == "tool_called":
            out[int(e.get("round") or 0)].append((e, rets.get(str(e.get("call_id")))))
    return out


# ---- raw copies ------------------------------------------------------------------------------
def _files(root: Path) -> list[tuple[Path, int]]:
    """(path, size) of every file under `root`, sorted, from one directory walk."""
    out: list[tuple[Path, int]] = []
    if not root.is_dir():
        return out
    stack = [root]
    while stack:
        with os.scandir(stack.pop()) as it:
            for entry in it:
                if entry.is_dir(follow_symlinks=False):
                    stack.append(Path(entry.path))
                elif entry.is_file():
                    out.append((Path(entry.path), entry.stat().st_size))
    return sorted(out)


def _snapshot_blobs(run_dir: Path) -> set[str]:
    out: set[str] = set()
    for p in (run_dir / "snapshots").glob("*.json"):
        try:
            out.update(str(v) for v in (json.loads(p.read_text()).get("plugins") or {}).values())
        except (OSError, ValueError):
            continue
    return out


def _copy_raw(run_dir: Path, raw: Path, blobs: _Blobs) -> tuple[dict, list[str]]:
    """Copy what `fetch-published` needs into `raw/` in one pass over the files; returns the
    blob summary and the files that could not be copied (`"<path>: <error>"`)."""
    errors: list[str] = []

    def copy(src: Path, dst: Path) -> int:
        try:
            dst.parent.mkdir(parents=True, exist_ok=True)
            shutil.copyfile(src, dst)
            return 1
        except OSError as e:
            errors.append(f"{src.relative_to(run_dir)}: {type(e).__name__}: {e}")
            return 0

    raw.mkdir(parents=True, exist_ok=True)
    size = 0
    for name in ("events.jsonl", "discarded.jsonl", "run.json"):
        if (run_dir / name).exists() or name == "events.jsonl":
            copy(run_dir / name, raw / name)
    for sub in ("snapshots", "artifacts"):
        for src, n in _files(run_dir / sub):
            size += n
            copy(src, raw / src.relative_to(run_dir))
    src_root = run_dir / "blobs"
    listed = _files(src_root)
    size += sum(n for _, n in listed) + sum(
        (run_dir / f).stat().st_size for f in ("events.jsonl", "discarded.jsonl")
        if (run_dir / f).exists())
    if not src_root.is_dir():
        return {"included": "none", "files": 0, "bytes": 0}, errors
    if size < FULL_BLOBS_LIMIT:
        mode, wanted = "all", listed
    else:
        mode = "snapshots+oversize"
        keep = _snapshot_blobs(run_dir) | blobs.oversize
        wanted = [(p, n) for p, n in listed if p.name in keep and p.parent.parent == src_root
                  and p.parent.name == p.name[:2]]
    files = nbytes = 0
    for src, n in wanted:
        ok = copy(src, raw / "blobs" / src.relative_to(src_root))
        files += ok
        nbytes += n * ok
    return {"included": mode, "files": files, "bytes": nbytes, "run_dir_bytes": size}, errors


# ---- entry point -----------------------------------------------------------------------------
def _clear(out: Path) -> None:
    """Remove what a previous export wrote (only the paths this module owns)."""
    for sub in ("tables", "sessions", "raw"):
        if (out / sub).is_dir():
            shutil.rmtree(out / sub)
    if (out / "run.json").exists():
        (out / "run.json").unlink()


def _metric_finals(events: list[dict]) -> dict:
    out: dict[str, dict] = {}
    for ev in events:
        if ev.get("type") == "metric":
            out[str(ev.get("name"))] = {"value": ev.get("value"),
                                        "denominator": ev.get("denominator"),
                                        "round": ev.get("round")}
    return out


def export_run(run_dir: Path | str, out: Path | str | None = None, *, raw: bool = True) -> Path:
    """Export one run directory (see module doc); returns the export directory.

    `raw=False` (CLI `--no-raw`) skips `raw/` (tables and sessions only; `run.json["export_raw"]`
    is false). Raises `ExportError` when a file of `raw/` cannot be copied; `run.json` is then
    not written, so the export is never taken as current."""
    run_dir = Path(run_dir)
    meta_path = run_dir / "run.json"
    if not meta_path.exists():
        raise FileNotFoundError(f"{run_dir} is not a run directory (no run.json)")
    meta = json.loads(meta_path.read_text())
    out = Path(out) if out is not None else run_dir / "export"
    if out.resolve() == run_dir.resolve():
        raise ValueError("export out dir must differ from the run dir")
    events = list(read_events(run_dir / "events.jsonl"))
    blobs = _Blobs(run_dir / "blobs")
    _clear(out)
    rows = build_tables(events, meta, blobs)
    rows["discarded_inference"] = discarded_inference_rows(run_dir, meta, blobs)
    counts = {f: write_table(rows[f], f, out / "tables" / f"{f}.parquet") for f in FAMILIES}
    sessions = build_sessions(run_dir, events, meta, blobs)
    (out / "sessions").mkdir(parents=True, exist_ok=True)
    for agent, lines in sessions.items():
        (out / "sessions" / f"{agent}.jsonl").write_text("\n".join(lines) + "\n")
    if raw:
        blob_info, errors = _copy_raw(run_dir, out / "raw", blobs)
        if errors:
            shown = "; ".join(errors[:5]) + (f"; ... ({len(errors)} in all)" if len(errors) > 5 else "")
            raise ExportError(
                f"could not copy {len(errors)} file(s) of {run_dir} into {out / 'raw'}: {shown}. "
                "On a bucket-mounted runs dir, retry from a local copy of the run dir, or export "
                "without raw copies (--no-raw).")
    else:
        blob_info = {"included": "none", "files": 0, "bytes": 0}
    spec = meta.get("spec") or {}
    led = meta.get("ledger") or {}
    ev_bytes = (run_dir / "events.jsonl").read_bytes()
    doc = {
        "export_schema": EXPORT_SCHEMA,
        "run_id": meta.get("run_id"), "experiment": meta.get("experiment"),
        "arm": meta.get("arm"), "seed": (spec.get("options") or {}).get("seed"),
        "spec_hash": meta.get("spec_hash"), "git_commit": meta.get("git_commit"),
        "dirty": meta.get("dirty"), "parent_run": meta.get("parent_run"),
        "fork_round": meta.get("fork_round"), "status": meta.get("status"),
        "end_reason": meta.get("end_reason"), "last_round": meta.get("last_round"),
        "score": meta.get("score"),
        "spend": {k: led.get(k, 0) for k in ("swarm", "measurement", "reserved", "calls")},
        "spend_discarded_usd": sum(r["charged_usd"] for r in rows["discarded_inference"]),
        "budget": meta.get("budget") or spec.get("budget"),
        "metrics_final": _metric_finals(events),
        "agents": [{"agent": f"a{i:03d}", "type": p.get("type"),
                    "model": (p.get("params") or {}).get("model")}
                   for i, p in enumerate(spec.get("participants") or [])],
        "tables": {f: {"rows": counts[f], "path": f"tables/{f}.parquet"} for f in FAMILIES},
        "sessions": sorted(f"sessions/{a}.jsonl" for a in sessions),
        "events": {"n": len(events), "sha256": hashlib.sha256(ev_bytes).hexdigest()},
        "blobs": blob_info,
        "export_raw": raw,
        "unreadable_blobs": dict(sorted(blobs.unreadable.items())),
        "spec": spec,
    }
    (out / "run.json").write_text(json.dumps(doc, indent=2, sort_keys=True, default=str) + "\n")
    return out


def is_current(run_dir: Path | str, out: Path | str | None = None, *, raw: bool = True) -> bool:
    """True when `out` holds an export of the run's current log with this EXPORT_SCHEMA, made
    with the same `raw` choice."""
    run_dir = Path(run_dir)
    out = Path(out) if out is not None else run_dir / "export"
    try:
        doc = json.loads((out / "run.json").read_text())
        sha = hashlib.sha256((run_dir / "events.jsonl").read_bytes()).hexdigest()
    except (OSError, ValueError):
        return False
    return (doc.get("export_schema") == EXPORT_SCHEMA and doc.get("events", {}).get("sha256") == sha
            and doc.get("export_raw", True) == raw)

