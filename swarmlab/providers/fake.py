"""FakeProvider: deterministic, scripted responses (docs/INTERFACE-M1b.md §1).

`FakeProvider(script=None, pricing=None, concurrency=8)`. A script is a function
`(request: ChatRequest, rng: random.Random) -> ChatResponse`. It is chosen by `script` when given,
else by the request's model id (`"fake:reader"` -> script `reader`), and resolved as: a built-in
name (`flaggame_reader`, alias `reader`), an entry point in group `swarmlab.fake_scripts`, or
`module:function`. The rng for a call is `derive(int(request_hash(request)[:8], 16), "fake")`, so
the fake is a pure function of the request. Pricing defaults to `{"*": (1.0, 5.0, 0.1)}` (every
model id is priced). `calls` counts `complete()` invocations on the instance.

The provider fills in what the script leaves unset: `provider`, `model`, `latency_s = 0`, `cost_usd`
(from `cost(request, usage)`), and, when the script returns all-zero usage,
`prompt_tokens = estimate_prompt_tokens(request)` and
`completion_tokens = max(1, ceil(len(text + tool-call JSON) / 4))`.

Built-in `flaggame_reader` (FlagGame text observations; in the image modality it needs
`image_text_hint=True`, whose text parts joined with newlines form the same listing, and ignores
the images; `FlagGame.check_participants` refuses it otherwise). Within one turn (the messages after the
last `user` message that is not a json-protocol `[tool results]` message):

1. If `read_board` is offered and has not been called this turn: call `post` with
   `"crop:\\n<rows>"` (on `main`, or with no channel when the `post` schema's channel enum does
   not offer `main`) first if `post` is offered and the conversation shows no earlier `post` of a
   crop, then `read_board(limit=200)`.
2. Otherwise: collect every distinct crop in the conversation (any `crop:` block of lowercase rows,
   in any message text or tool result, JSON-escaped newlines included; this covers the
   observation's "Your crop:" and crops read from the board), take the candidates from the latest
   "Candidate flags:" listing, and `guess` the candidate containing the most crops (ties broken by
   the rng over tied names in name order), then `end_turn` (each only if offered).
3. With no tools offered at all (a probe), it answers with text
   `{"candidate": <best>, "confidence": <share of crops it contains>}`.

Tool-call ids are `fake_<8 hex from the rng>`. Under `tool_protocol == "json"` the calls are
returned as a JSON array `[{"name", "args"}]` in `text` instead of `tool_calls`; earlier calls are
recognised in either form.
"""
from __future__ import annotations

import importlib
import json
import math
import random
import re
from collections.abc import Callable
from importlib.metadata import entry_points
from typing import Any, ClassVar

from ..rng import derive
from ..tools import ToolCall
from ..world.flaggame import CROP_HEADER, PREAMBLE, contains
from .base import (
    ChatMessage,
    ChatRequest,
    ChatResponse,
    PricingRow,
    Provider,
    Usage,
    model_id,
    request_hash,
    text_of,
)

Script = Callable[[ChatRequest, random.Random], ChatResponse]

RESULTS_PREFIX = "[tool results]"  # LLMAgent's json-protocol results message (participants/llm.py)
_CROP_RE = re.compile(r"crop:[ \t]*((?:\n[ \t]*[a-z]+[ \t]*)+)")
_HEADER_RE = re.compile(r"^([A-Za-z0-9_]+):$")
_ROW_RE = re.compile(r"^[a-z]+$")


# ---- conversation helpers ----------------------------------------------------------------------
def _assistant_calls(m: ChatMessage) -> list[tuple[str, dict]]:
    if m.role != "assistant":
        return []
    if m.tool_calls:
        return [(tc.name, tc.args) for tc in m.tool_calls]
    try:
        value = json.loads(text_of(m.content))
    except ValueError:
        return []
    if isinstance(value, list):
        return [(c.get("name", ""), c.get("args") or {}) for c in value if isinstance(c, dict)]
    return []


def _texts(request: ChatRequest) -> list[str]:
    out = []
    for m in request.messages:
        out.append(text_of(m.content))
        for _, args in _assistant_calls(m):
            out.append(json.dumps(args))
    return out


def parse_crops(text: str) -> list[tuple[str, ...]]:
    """Every `crop:` block of lowercase rows in `text` (JSON-escaped newlines accepted)."""
    text = text.replace("\\n", "\n")
    crops = []
    for m in _CROP_RE.finditer(text):
        rows = tuple(r.strip() for r in m.group(1).split("\n") if r.strip())
        if rows and len({len(r) for r in rows}) == 1:
            crops.append(rows)
    return crops


def parse_listing(text: str) -> tuple[dict[str, list[str]], list[str]]:
    """Candidates and own crop from the "Candidate flags:" listing inside a larger message."""
    candidates: dict[str, list[str]] = {}
    crop: list[str] = []
    idx = text.rfind(PREAMBLE)
    if idx < 0:
        return candidates, crop
    current: list[str] | None = None
    for raw in text[idx + len(PREAMBLE):].split("\n"):
        line = raw.strip()
        if not line:
            continue
        if line == CROP_HEADER:
            current = crop
        elif _HEADER_RE.match(line):
            current = candidates.setdefault(line[:-1], [])
        elif _ROW_RE.match(line) and current is not None:
            current.append(line)
        elif current is None:
            continue  # prose before the first section (FlagGame image modality framing line)
        else:
            break
    return candidates, crop


def _best(candidates: dict[str, list[str]], crops: list[tuple[str, ...]],
          rng: random.Random) -> tuple[str, float]:
    names = sorted(candidates)
    scores = {n: sum(1 for c in crops if contains(candidates[n], list(c))) for n in names}
    top = max(scores.values())
    choice = rng.choice([n for n in names if scores[n] == top])
    return choice, (top / len(crops) if crops else 0.0)


def _response(request: ChatRequest, rng: random.Random, calls: list[tuple[str, dict]],
              text: str = "") -> ChatResponse:
    tool_calls = [ToolCall(call_id=f"fake_{rng.getrandbits(32):08x}", name=n, args=a) for n, a in calls]
    if request.tool_protocol == "json" and tool_calls:
        text = json.dumps([{"name": n, "args": a} for n, a in calls])
        tool_calls = []
    finish = "tool_use" if tool_calls else "end_turn"
    return ChatResponse(text=text, tool_calls=tool_calls, usage=Usage(), cost_usd=0.0, provider="fake",
                        model=model_id(request), latency_s=0.0, finish_reason=finish)


def flaggame_reader(request: ChatRequest, rng: random.Random) -> ChatResponse:
    tools = {t.name for t in request.tools}
    msgs = request.messages
    # json-protocol tool results come back as user messages starting with "[tool results]"
    last_user = max((i for i, m in enumerate(msgs) if m.role == "user"
                     and not text_of(m.content).startswith(RESULTS_PREFIX)), default=-1)
    this_turn = [name for m in msgs[last_user + 1:] for name, _ in _assistant_calls(m)]
    listing = next((text_of(m.content) for m in reversed(msgs)
                    if m.role == "user" and PREAMBLE in text_of(m.content)), "")
    candidates, own_crop = parse_listing(listing)
    if "read_board" in tools and "read_board" not in this_turn:
        calls: list[tuple[str, dict]] = []
        posted = any(name == "post" and "crop:" in str(args.get("text", ""))
                     for m in msgs for name, args in _assistant_calls(m))
        if "post" in tools and not posted and own_crop:
            # post to `main` when offered, else to the executor's default channel (M3c: a Tree
            # topology offers only group channels)
            enum = next((t.parameters.get("properties", {}).get("channel", {}).get("enum")
                         for t in request.tools if t.name == "post"), None)
            where = {"channel": "main"} if enum is None or "main" in enum else {}
            calls.append(("post", {**where, "text": "crop:\n" + "\n".join(own_crop)}))
        calls.append(("read_board", {"limit": 200}))
        return _response(request, rng, calls)
    crops = sorted({c for t in _texts(request) for c in parse_crops(t)})
    if not tools:
        if not candidates:
            return _response(request, rng, [], json.dumps({"candidate": None, "confidence": 0.0}))
        best, conf = _best(candidates, crops, rng)
        return _response(request, rng, [], json.dumps({"candidate": best, "confidence": round(conf, 3)}))
    calls = []
    if "guess" in tools and candidates:
        best, _ = _best(candidates, crops, rng)
        calls.append(("guess", {"candidate": best}))
    if "end_turn" in tools:
        calls.append(("end_turn", {}))
    return _response(request, rng, calls)


BUILTIN_SCRIPTS: dict[str, Script] = {"flaggame_reader": flaggame_reader, "reader": flaggame_reader}


def resolve_script(name: str) -> Script:
    if name in BUILTIN_SCRIPTS:
        return BUILTIN_SCRIPTS[name]
    for ep in entry_points(group="swarmlab.fake_scripts"):
        if ep.name == name:
            return ep.load()
    if ":" in name:
        module, _, qualname = name.partition(":")
        obj: Any = importlib.import_module(module)
        for part in qualname.split("."):
            obj = getattr(obj, part)
        return obj
    raise ValueError(f"unknown fake script {name!r} (built-ins: {sorted(BUILTIN_SCRIPTS)}, "
                     "entry points in swarmlab.fake_scripts, or module:function)")


class FakeProvider(Provider):
    entry_point: ClassVar[str | None] = "fake"
    name = "fake"
    default_pricing: ClassVar[dict[str, PricingRow]] = {"*": (1.0, 5.0, 0.1)}

    def __init__(self, script: str | None = None, pricing: dict | None = None,
                 concurrency: int = 8) -> None:
        self.script = script
        self._setup(pricing, concurrency)

    def script_for(self, request: ChatRequest) -> Script:
        return resolve_script(self.script or model_id(request))

    async def complete(self, request: ChatRequest) -> ChatResponse:
        self.calls += 1
        h = request_hash(request)
        rng = derive(int(h[:8], 16), "fake")
        resp = self.script_for(request)(request, rng)
        usage = resp.usage
        if usage == Usage():
            body = resp.text + "".join(json.dumps(tc.args) + tc.name for tc in resp.tool_calls)
            usage = Usage(prompt_tokens=self.estimate_prompt_tokens(request),
                          completion_tokens=max(1, math.ceil(len(body) / 4)))
        return resp.model_copy(update={
            "usage": usage, "cost_usd": self.cost(request, usage), "provider": self.name,
            "model": model_id(request), "latency_s": 0.0,
        })
