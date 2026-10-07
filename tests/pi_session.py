"""A local validator for pi session files (format v3) as the Hub's agent-traces viewer reads them.

Written from https://github.com/earendil-works/pi/blob/main/packages/coding-agent/docs/session-format.md,
.../docs/message-types.md and https://huggingface.co/docs/hub/session-traces-format ("Alternative:
Pi's session format": add a `harness` field to the header). Independent of `swarmlab.export`.

Checked: header first line (`type: "session"`, `version: 3`, string `id`, ISO `timestamp`,
string `cwd`, non-empty `harness`); every later line is an entry with `type`, unique string
`id`, `parentId` null or an earlier id, ISO `timestamp`; `message` entries carry an
AgentMessage whose role-specific required fields have the documented types; content blocks are
`text` / `thinking` / `image` / `toolCall` where allowed; assistant `usage` has the full Usage
shape and `stopReason` is a terminal value; every `toolResult.toolCallId` answers an earlier
`toolCall` of the same session; swarmlab round markers (a user message whose text starts with
`Round <n>.`) are strictly increasing within a session (a repeated or earlier marker means a
conversation was emitted twice).
"""
from __future__ import annotations

import datetime as dt
import json
import re
from pathlib import Path

STOP_REASONS = {"stop", "length", "toolUse", "error", "aborted", "deferred"}  # not "pending"
ENTRY_TYPES = {"message", "model_change", "thinking_level_change", "usage", "compaction",
               "context_edit", "branch_summary", "custom", "custom_message", "label",
               "session_info"}
ROLES = {"system", "user", "assistant", "toolResult", "bashExecution", "custom",
         "branchSummary", "compactionSummary"}


def _iso(v) -> bool:
    if not isinstance(v, str):
        return False
    try:
        dt.datetime.fromisoformat(v)
    except ValueError:
        return False
    return True


def _num(v) -> bool:
    return isinstance(v, (int, float)) and not isinstance(v, bool)


def _text_or_image(blocks, where, errs):
    if isinstance(blocks, str):
        return
    if not isinstance(blocks, list):
        errs.append(f"{where}: content must be a string or a list")
        return
    for b in blocks:
        _block(b, {"text", "image"}, where, errs)


def _block(b, allowed, where, errs):
    if not isinstance(b, dict) or b.get("type") not in allowed:
        errs.append(f"{where}: block {b!r:.80} not one of {sorted(allowed)}")
        return
    t = b["type"]
    if t == "text" and not isinstance(b.get("text"), str):
        errs.append(f"{where}: text block without string text")
    if t == "thinking" and not isinstance(b.get("thinking"), str):
        errs.append(f"{where}: thinking block without string thinking")
    if t == "image" and not (isinstance(b.get("data"), str) and isinstance(b.get("mimeType"), str)):
        errs.append(f"{where}: image block needs data and mimeType")
    if t == "toolCall" and not (isinstance(b.get("id"), str) and isinstance(b.get("name"), str)
                                and isinstance(b.get("arguments"), dict)):
        errs.append(f"{where}: toolCall needs string id, string name, object arguments")


def _usage(u, where, errs):
    if not isinstance(u, dict):
        errs.append(f"{where}: usage missing")
        return
    for k in ("input", "output", "cacheRead", "cacheWrite", "totalTokens"):
        if not _num(u.get(k)):
            errs.append(f"{where}: usage.{k} not a number")
    cost = u.get("cost")
    if not isinstance(cost, dict) or any(not _num(cost.get(k)) for k in
                                         ("input", "output", "cacheRead", "cacheWrite", "total")):
        errs.append(f"{where}: usage.cost incomplete")


def _message(m, where, errs, calls: set[str]):
    if not isinstance(m, dict) or m.get("role") not in ROLES:
        errs.append(f"{where}: unknown role {m.get('role') if isinstance(m, dict) else m!r}")
        return
    if not _num(m.get("timestamp")):
        errs.append(f"{where}: message.timestamp must be epoch milliseconds")
    role = m["role"]
    if role == "system":
        c = m.get("content")
        if not isinstance(c, (str, list)):
            errs.append(f"{where}: system content must be string or text blocks")
        for t in m.get("toolsAdded") or []:
            if not (isinstance(t, dict) and isinstance(t.get("name"), str)):
                errs.append(f"{where}: toolsAdded entry without name")
    elif role == "user":
        _text_or_image(m.get("content"), where, errs)
    elif role == "assistant":
        c = m.get("content")
        if not isinstance(c, list):
            errs.append(f"{where}: assistant content must be a list")
        else:
            for b in c:
                _block(b, {"text", "thinking", "toolCall"}, where, errs)
                if isinstance(b, dict) and b.get("type") == "toolCall":
                    calls.add(b.get("id"))
        for k in ("api", "provider", "model"):
            if not isinstance(m.get(k), str):
                errs.append(f"{where}: assistant.{k} must be a string")
        _usage(m.get("usage"), where, errs)
        if m.get("stopReason") not in STOP_REASONS:
            errs.append(f"{where}: stopReason {m.get('stopReason')!r}")
    elif role == "toolResult":
        if not isinstance(m.get("toolCallId"), str) or m["toolCallId"] not in calls:
            errs.append(f"{where}: toolResult answers unknown toolCallId {m.get('toolCallId')!r}")
        if not isinstance(m.get("toolName"), str):
            errs.append(f"{where}: toolName must be a string")
        if not isinstance(m.get("isError"), bool):
            errs.append(f"{where}: isError must be a bool")
        c = m.get("content")
        if not isinstance(c, list):
            errs.append(f"{where}: toolResult content must be a list")
        else:
            for b in c:
                _block(b, {"text", "image"}, where, errs)


def validate_lines(lines: list[str]) -> list[str]:
    errs: list[str] = []
    if not lines:
        return ["empty file"]
    try:
        head = json.loads(lines[0])
    except ValueError:
        return ["header is not JSON"]
    if head.get("type") != "session":
        errs.append("first line must be the session header")
    if head.get("version") != 3:
        errs.append("header.version must be 3")
    if not isinstance(head.get("id"), str) or not head["id"]:
        errs.append("header.id must be a non-empty string")
    if not _iso(head.get("timestamp")):
        errs.append("header.timestamp must be ISO 8601")
    if not isinstance(head.get("cwd"), str):
        errs.append("header.cwd must be a string")
    if not isinstance(head.get("harness"), str) or not head["harness"]:
        errs.append("header.harness must name the harness")
    ids: set[str] = set()
    calls: set[str] = set()
    last_round = 0
    for n, raw in enumerate(lines[1:], start=2):
        where = f"line {n}"
        try:
            e = json.loads(raw)
        except ValueError:
            errs.append(f"{where}: not JSON")
            continue
        if e.get("type") not in ENTRY_TYPES:
            errs.append(f"{where}: unknown entry type {e.get('type')!r}")
        if not isinstance(e.get("id"), str) or not e["id"] or e["id"] in ids:
            errs.append(f"{where}: id missing or duplicate")
        if "parentId" not in e or (e["parentId"] is not None and e["parentId"] not in ids):
            errs.append(f"{where}: parentId must be null or an earlier entry id")
        if not _iso(e.get("timestamp")):
            errs.append(f"{where}: timestamp must be ISO 8601")
        ids.add(e.get("id"))
        if e.get("type") == "message":
            _message(e.get("message"), where, errs, calls)
            r = _round_marker(e.get("message"))
            if r is not None:
                if r <= last_round:
                    errs.append(f"{where}: round marker 'Round {r}.' after 'Round {last_round}.'")
                last_round = max(last_round, r)
    return errs


def _round_marker(m) -> int | None:
    """n of a user message whose text starts with `Round <n>.`, else None."""
    if not isinstance(m, dict) or m.get("role") != "user":
        return None
    c = m.get("content")
    if isinstance(c, list):
        c = next((b.get("text") for b in c if isinstance(b, dict) and b.get("type") == "text"), "")
    hit = re.match(r"Round (\d+)\.", c or "")
    return int(hit.group(1)) if hit else None


def validate_file(path: Path | str) -> list[str]:
    return validate_lines(Path(path).read_text().splitlines())
