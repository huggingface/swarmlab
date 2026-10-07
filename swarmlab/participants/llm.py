"""LLMAgent: the in-process model loop (docs/INTERFACE-M1b.md §4).

`LLMAgent(model, system_prompt=None, memory="full", window_rounds=3, max_tokens=2048,
temperature=None, tool_protocol="native", thinking_budget=None, max_calls=None, role="worker",
extra=None, text_tool_fallback=False, context_limit_tokens=None, overflow="drop_oldest",
summary_model=None, system_prompt_append=None, memory_messages=8, report_json=False,
report_fields=None)`, entry point `llm`.

**max_tokens** defaults to 2048 (was 1024). In the 2026-10-06 smoke, Qwen3.5-9B with thinking on
spent the whole 1024-token budget reasoning (`finish_reason="length"`, empty text, no tool call)
on 7 of 12 turns. Reasoning models need room, or thinking turned off through `extra`; a
`length` finish is noted in the turn (below).

**extra** (dict, default `{}`) goes to `ChatRequest.extra` on every request this agent makes,
probes included (`model_request_defaults()` carries it): provider-specific body fields the adapter
merges into the request. For Qwen3 on DeepInfra/vLLM via an OpenAI-compatible provider:
`extra={"chat_template_kwargs": {"enable_thinking": False}}`; OpenAI-style
`{"reasoning_effort": ...}` or `{"reasoning": {...}}` pass the same way. It is part of the request
hash, so it separates cache entries and belongs to the arm's spec.

**text_tool_fallback** (native protocol only, default False because it is an experimental
condition): when a response has no native tool calls but its text spells out calls to offered
tools, they are parsed (`parse_text_tool_calls`: Hermes/Qwen `<tool_call>{...}</tool_call>`
blocks, then `name(key="value", ...)` call syntax, then a JSON call list as in the json
protocol) and executed like json-protocol calls (results come back as a `[tool results]` user
message, since there are no provider tool-call ids). Each trigger is logged (logger
`swarmlab.participants.llm`) and noted in the turn.

**Turn notes.** `turn()` returns an `LLMTurnUsage` (a `TurnUsage` with `finish_reasons`, one per
model call, and `notes`, e.g. `"length"` when a response hit `max_tokens` and
`"text_tool_fallback:<n>"`); the runner puts it into `turn_ended.usage`.

**System prompt.** A Jinja2 template rendered once, at the agent's first turn, with `agent`,
`role`, `description` (`View.description`, i.e. `World.description()`) and `tools`
(`view.tools`, `ToolSchema` objects). The default is `prompts/default_system.j2`;
`system_prompt` replaces it with a template string or `file:<path>` (read at render time; the
path is what goes into `params`). Rendering uses `trim_blocks`/`lstrip_blocks` and
`StrictUndefined`. `system_prompt_append` (text, not a template) adds to the prompt without
copying it: it is the template variable `system_prompt_append` (empty when unset), which the
default template places after the task section and before `Tools:`; a custom `system_prompt`
that never names the variable gets the text appended at its end. It is in `params` (and the spec
hash) only when set, so the default prompt and existing hashes are unchanged. Under `tool_protocol="json"` a fixed JSON-protocol section (with each tool's
parameter schema, since no tools go through the provider API) is appended after the template.

**Roles** (M3c, swarmlab/roles.py). The runner calls `apply_role(role)` right after `bind`. It sets
`role` (the template's role word) to the role's name, `model`, `system_prompt` and `max_tokens`
(from `role.budget`) when the role sets them, and a transient `_role_prompt_append` that the
renderer appends to the rendered template, after its last section (and after any
`system_prompt_append`), before the JSON-protocol section. These are plain attributes or set
again at every bind, so a resumed run sees the same values; params (the participant's spec) do
not change, the role is part of the run spec instead.

**Round message.** One user message per round, a list of `Part`s: `Round {r}.`, the observation
parts unchanged (text and image parts pass through), then, when present, one text part
`Outcomes of your actions last round:` with a line per outcome
(`- <tool> <action_id>: accepted|rejected <feedback JSON>`), and one text part
`Delivered to you:` with a line per pushed item (`- <post_id> (round <eligible_round>): <content>`).

**Loop** (one turn): build a `ChatRequest` (system + memory, `tools=view.tools`, the agent's
settings), `await tools.infer(request)`, then

- native: no tool calls -> the turn ends (`no_tool`; the assistant text, if non-empty, is kept).
  Otherwise the assistant message (text + tool calls) is appended and each call is executed in
  order through `tools.call`; each result becomes one `tool` message whose content is the JSON of
  `{"ok", "result", "error"}`. A call whose arguments did not parse (`{"_raw": ...}`, an
  OpenAI-compatible server's bad JSON) is not executed; it gets a tool message with
  `ok: false` and the parse error so the model can correct itself.
- json: the response text is parsed tolerantly (`parse_tool_json`: code fences and
  `<think>...</think>` blocks stripped, the outermost `[...]` taken, a single object accepted as a
  one-element list, `arguments`/`parameters`/`input` accepted for `args`). Text with no `[`/`{` at
  all is a plain reply and ends the turn (`no_tool`). A malformed array gets **one** error-feedback
  retry per turn; a second malformed reply ends the turn. Results (and the parse error) go back
  as a `user` message starting with `[tool results]` followed by a JSON list, because a json-protocol
  conversation has no provider tool-call ids to attach `tool` messages to (Anthropic and
  OpenAI-compatible APIs reject orphan tool results).
- If `end_turn` is among a response's calls, the remaining calls of that response still run
  (the executor answers them `turn_ended`), then the turn returns.
- `max_calls` (model calls per turn) stops the loop before the runner's cap; the runner's
  `max_calls_per_turn` (tool calls) still raises `TurnCapReached`, which propagates (`cap`). If a
  call raises mid-response, the calls not yet answered get an error tool message first, so the
  stored conversation never has a tool call without a result.

**Memory.** Messages are stored as plain dicts (`ChatMessage.model_dump(mode="json")`) grouped by
round in `self.rounds: list[{"round", "messages"}]`, so the default `Persistable` snapshot works.
`full` keeps everything. `window` keeps the last `window_rounds` rounds: before a turn adds round
r, older rounds are dropped so that rounds r-window_rounds+1..r remain (the current observation is
re-sent every round anyway). The system prompt is always kept.

**memory="received"** (M6, docs/INTERFACE-M6.md §3; the Flag Game paper's pairwise memory). No
conversation is kept across turns: each turn's request is the system prompt plus one user message
built from state, and the turn's own tool traffic (if any) follows it during the turn only. The
message is the current observation's parts (the crop is re-shown every call), then one text part
`Transcript memory (oldest -> newest):` with a line `- <content>` per remembered item (or `[]`
when there is none), then, when the agent has answered before, `Your previous answers (oldest ->
newest): <JSON list>`, then under `report_json` the schema line. There is no `Round r.` line and
no outcome lines (the paper's prompt has neither; under OneSpeaker the round number would leak
the population's step count). Remembered items are delivered board items, ingested from
`View.pushed` at turn start, from `read_board` results during a turn (pull works too) and from
the runner's `prepare_probe` hook before probes; an item is added only when its
`(eligible_round, delivery_id)` is newer than every item ingested so far (so re-pushed items are
not duplicated), and only the last `memory_messages` are kept. Author ids are not shown (pushed
items carry none). Own answers are the accepted `guess` arguments (and, under `report_json`
without a `guess` tool, the reported answer), the last `max(1, memory_messages)` kept. State
(`received`, `received_upto`, `own_answers`, `last_observation`) is plain data in the snapshot.
Pair it with `delivery: push`, `push_consume: true` (swarmlab/medium/board.py) so each item is
pushed once and the newest win. `prepare_probe(view_for)` ingests pending items and, for an agent
that has not acted yet, stores its observation and renders its system prompt. `probe_context()`
is `[system, <the message above without the schema line>]` (`[system]` or `[]` before any
observation). `memory_messages` is in `params` only when `memory="received"` or non-default.

**report_json** (M6 §4; `report_fields` default `["country", "reason"]`, the paper's m=3 format;
`["country"]` is m=1; any memory mode). The system prompt defaults to `prompts/report_json_system.j2` (the paper's
JSON-only rules, the task description, "choose exactly one ... follow the exact output schema";
a `system_prompt` still replaces it, the role text is still appended, the json tool-protocol
section is not). Each turn's message ends with the schema line `Output JSON exactly:
{"country":"<one allowed country>","reason":"<one sentence>"}`. The reply's first JSON object
(reasoning and fences tolerated) is read: the first field (keys case-insensitive) is the answer,
matched to the observation's allowed names case-insensitively; the other fields are kept when
present (non-strings as JSON). The harness then calls `guess(<answer>)` (the guess tool's first
parameter; skipped when the agent has no `guess` tool), `post(text=<the report as compact JSON,
fields in report_fields order>)` and `end_turn`, all in that one step. A reply without a usable
answer, or a rejected guess, gets one corrective user message and one more model call (so at
most 2 calls; `max_calls=1` gives no retry; `calls_per_turn_cap` tells `swarmlab estimate`); a
second failure ends the turn with no post (turn note `report_json:failed:...`). Under the native
protocol a reply with native tool calls is executed as such instead and ends the turn (note
`report_json:native_tools`), so tool-capable models may still act directly; under the json
protocol no tools go through the API and the JSON answer is the only path. `report_json` and
`report_fields` are in `params` only when set.

**Probing.** `probe_context()` returns `[system] + memory` as fresh `ChatMessage` objects (callers
cannot mutate the agent's memory through them); `model_request_defaults()` returns `model,
temperature, max_tokens, thinking_budget, extra`.

`TurnUsage.calls` counts executed tool calls; tokens and cost are filled in by the executor.

**Context limit** (M3a §3; `context_limit_tokens=None`, `overflow="drop_oldest"`,
`summary_model=None`). Before every model request of a turn the prompt (system + memory + tool
schemas) is estimated with the provider's `estimate_prompt_tokens`; the provider is a preset
resolved from the model prefix once per agent (`providers.preset`), falling back to ceil(chars/4)
of the request JSON when no preset can be built (e.g. `vllm` without a base url). Experiment-level
provider overrides are not visible to a participant; every built-in provider uses the base
estimator anyway. If the estimate exceeds the limit:

- `drop_oldest`: whole memory entries are removed from the front until the estimate fits. The
  system prompt and the current round (the last entry) are never dropped; if only those remain
  and it still does not fit, the request is sent as is (the event's `tokens_after` stays over
  the limit). This is checked before each request, so a long current round can trigger further
  drops mid-turn.
- `summarize`: entries are dropped from the front until the estimate plus `SUMMARY_MAX_TOKENS`
  (room for the note) fits, then one `infer(category="measurement")` call on `summary_model`
  (default `DEFAULT_SUMMARY_MODEL`, Claude Haiku 4.5, DESIGN's cheap model; `max_tokens=400`,
  `temperature=0`, no tools) turns the dropped entries into a note of under 200 words that keeps
  beliefs, evidence and decisions. The note is stored as a memory entry
  `{"round": <last dropped round>, "from_round": <first summarised round>, "summary": True,
  "messages": [assistant message]}` (text `[Summary of my earlier rounds a-b]` + the note) in front of
  the remaining rounds, so it persists in snapshots and `probe_context()`. An earlier note is
  dropped first and so is folded into the next one. If the measurement budget refuses the call
  (`MeasurementBudgetReached`), the entries stay dropped without a note and the event carries
  `detail="measurement_budget"`; `HardCeilingReached` propagates. `summary_model` is not priced at
  `Experiment` construction; the gate resolves it at call time.
- `fail_turn`: the turn ends at once by raising `ContextLimitExceeded`, whose `turn_error`
  (`"context_limit"`) the runner writes as `turn_ended.error` with `yield_kind="error"`. Memory
  keeps the current round's messages (what the agent saw), so later turns overflow too unless the
  window memory makes room.

Every overflow (one per trimming, also under `fail_turn` and when nothing could be dropped)
appends `{policy, dropped_rounds, tokens_before, tokens_after[, detail]}` to a transient list the
runner drains after the turn (`drain_overflow()`) into `overflow` events, placed after the turn's
tool events and before `turn_ended`. `dropped_rounds` counts real rounds (a dropped summary note is
not a round). Window trimming happens first, at turn start. A summary note counts as one entry of
the window. Dropped messages are gone from memory by design, so snapshots, resume and forks see
the trimmed memory. `context_limit_tokens`, `overflow` and `summary_model` are in `params` (and the
spec hash) only when set: `overflow` and `summary_model` only together with a limit.
"""
from __future__ import annotations

import ast
import json
import logging
import math
import re
from pathlib import Path
from typing import Any, ClassVar, Literal

import jinja2
from pydantic import Field

from ..budget import MeasurementBudgetReached
from ..probes import strip_reasoning
from ..providers.base import ChatMessage, ChatRequest, split_model, text_of
from ..spec import canonical_json
from ..tools import AgentTools, ToolCall, ToolSchema
from ..view import Part, View
from .base import Participant, TurnUsage

log = logging.getLogger(__name__)

PROMPTS_DIR = Path(__file__).resolve().parent / "prompts"
DEFAULT_SYSTEM_TEMPLATE = PROMPTS_DIR / "default_system.j2"
REPORT_SYSTEM_TEMPLATE = PROMPTS_DIR / "report_json_system.j2"
MEMORY_MODES = ("full", "window", "received")
DEFAULT_REPORT_FIELDS = ("country", "reason")
TRANSCRIPT_HEADER = "Transcript memory (oldest -> newest):"
OWN_ANSWERS_HEADER = "Your previous answers (oldest -> newest):"
REPORT_SCHEMA_PREFIX = "Output JSON exactly: "
_FIELD_PLACEHOLDERS = {"country": "<one allowed country>", "reason": "<one sentence>",
                       "candidate": "<one allowed candidate>"}


def report_schema(fields: list[str] | tuple[str, ...]) -> str:
    """The schema line of a `report_json` turn, e.g. `Output JSON exactly: {"country":"<one
    allowed country>","reason":"<one sentence>"}` (the paper's m=3 format)."""
    body = ",".join(f'"{f}":"{_FIELD_PLACEHOLDERS.get(f, "<" + f + ">")}"' for f in fields)
    return REPORT_SCHEMA_PREFIX + "{" + body + "}"
RESULTS_PREFIX = "[tool results]"

JSON_PROTOCOL_TEXT = (
    "Tool protocol: tools are not available as native function calls. To act, reply with only a "
    'JSON array of tool calls, for example [{{"name": "end_turn", "args": {{}}}}]. Each element has '
    '"name" (one of the tools) and "args" (an object matching that tool\'s parameters). The calls '
    "run in order and their results come back in the next message. A reply without a JSON array "
    "ends your turn.\n\nTool parameters (JSON schema):\n{schemas}"
)

_ENV = jinja2.Environment(trim_blocks=True, lstrip_blocks=True, undefined=jinja2.StrictUndefined,
                          autoescape=False)


def render_system_prompt(template: str | None, *, agent: str, role: str, description: str,
                         tools: list[ToolSchema], append: str | None = None,
                         default: Path = DEFAULT_SYSTEM_TEMPLATE, **variables: Any) -> str:
    """Render `template` (text, `file:<path>`, or None for the default) with the agent's context.

    `append` (`system_prompt_append`) is the template variable `system_prompt_append`; the
    default template puts it after the task section, before the tool list. A custom template
    that does not use the variable gets it appended at the end."""
    if template is None:
        text = default.read_text()
    elif template.startswith("file:"):
        text = Path(template[len("file:"):]).read_text()
    else:
        text = template
    extra = (append or "").strip()
    out = _ENV.from_string(text).render(agent=agent, role=role, description=description,
                                        tools=tools, system_prompt_append=extra,
                                        **variables).strip()
    if extra and "system_prompt_append" not in text:
        out += "\n\n" + extra
    return out


def json_protocol_text(tools: list[ToolSchema]) -> str:
    schemas = "\n".join(f"- {t.name}: {json.dumps(t.parameters, sort_keys=True)}" for t in tools)
    return JSON_PROTOCOL_TEXT.format(schemas=schemas)


class ToolJsonError(ValueError):
    """The model's json-protocol reply looks like an action list but does not parse."""


_THINK_RE = re.compile(r"<think>.*?</think>", re.DOTALL)
_FENCE_RE = re.compile(r"```(?:json)?\s*(.*?)```", re.DOTALL)


def parse_tool_json(text: str) -> list[tuple[str, dict]] | None:
    """Tolerant parse of a json-protocol reply.

    Returns None when the text holds no JSON-looking content (a plain reply), the list of
    `(name, args)` otherwise; raises `ToolJsonError` for malformed content.
    """
    body = _THINK_RE.sub("", text or "").strip()
    fence = _FENCE_RE.search(body)
    if fence:
        body = fence.group(1).strip()
    start_l, start_o = body.find("["), body.find("{")
    if start_l < 0 and start_o < 0:
        return None
    candidates = []
    if start_l >= 0:
        candidates.append(body[start_l:body.rfind("]") + 1])
    if start_o >= 0:
        candidates.append(body[start_o:body.rfind("}") + 1])
    candidates.sort(key=lambda s: body.find(s))
    value: Any = None
    for chunk in candidates:
        try:
            value = json.loads(chunk)
            break
        except ValueError:
            continue
    else:
        raise ToolJsonError("reply is not valid JSON; answer with a JSON array of "
                            '{"name": ..., "args": {...}} objects')
    if isinstance(value, dict):
        value = [value]
    if not isinstance(value, list):
        raise ToolJsonError("expected a JSON array of tool calls")
    out: list[tuple[str, dict]] = []
    for i, item in enumerate(value):
        if not isinstance(item, dict) or not isinstance(item.get("name"), str):
            raise ToolJsonError(f"element {i} is not an object with a string \"name\"")
        args = next((item[k] for k in ("args", "arguments", "parameters", "input") if k in item), {})
        if isinstance(args, str):
            try:
                args = json.loads(args) if args.strip() else {}
            except ValueError as e:
                raise ToolJsonError(f"element {i}: args is not a JSON object") from e
        if not isinstance(args, dict):
            raise ToolJsonError(f"element {i}: args must be a JSON object")
        out.append((item["name"], args))
    return out


_TOOL_CALL_TAG_RE = re.compile(r"<tool_call>\s*(.*?)\s*</tool_call>", re.DOTALL)


def _call_syntax(body: str, names: set[str]) -> list[tuple[str, dict]]:
    """`name(key=value, ...)` calls to known tools, values Python/JSON literals, in text order."""
    out: list[tuple[int, str, dict]] = []
    for name in names:
        for m in re.finditer(rf"(?<![\w.]){re.escape(name)}\s*\(", body):
            depth, end = 0, None
            for i in range(m.end() - 1, len(body)):
                if body[i] == "(":
                    depth += 1
                elif body[i] == ")":
                    depth -= 1
                    if depth == 0:
                        end = i + 1
                        break
            if end is None:
                continue
            snippet = re.sub(r"\s+", " ", body[m.start():end])
            snippet = re.sub(r"\btrue\b", "True", re.sub(r"\bfalse\b", "False", snippet))
            snippet = re.sub(r"\bnull\b", "None", snippet)
            try:
                node = ast.parse(snippet, mode="eval").body
                if not isinstance(node, ast.Call) or node.args:
                    continue
                args = {kw.arg: ast.literal_eval(kw.value) for kw in node.keywords if kw.arg}
            except (SyntaxError, ValueError):
                continue
            out.append((m.start(), name, args))
    return [(n, a) for _, n, a in sorted(out, key=lambda t: t[0])]


def parse_text_tool_calls(text: str, tools: list[ToolSchema]) -> list[tuple[str, dict]]:
    """Tool calls a model wrote as text instead of calling natively (empty list if none).

    Tried in order, the first form that yields calls to offered tools wins: Hermes/Qwen
    `<tool_call>{"name", "arguments"}</tool_call>` blocks; `name(key="value")` call syntax; a JSON
    call list or object (`parse_tool_json`). Reasoning is ignored (`probes.strip_reasoning`). Calls to tools that are
    not offered are dropped.
    """
    names = {t.name for t in tools}
    body = strip_reasoning(text)
    calls: list[tuple[str, dict]] = []
    for block in _TOOL_CALL_TAG_RE.findall(body):
        try:
            calls += parse_tool_json(block) or []
        except ToolJsonError:
            continue
    calls = [c for c in calls if c[0] in names]
    if calls:
        return calls
    calls = _call_syntax(body, names)
    if calls:
        return calls
    try:
        calls = parse_tool_json(body) or []
    except ToolJsonError:
        return []
    return [c for c in calls if c[0] in names]


# ---- context limit (M3a §3) ---------------------------------------------------------------------
OVERFLOW_POLICIES = ("drop_oldest", "summarize", "fail_turn")
DEFAULT_SUMMARY_MODEL = "anthropic:claude-haiku-4-5"
SUMMARY_MAX_TOKENS = 400
SUMMARY_SYSTEM = (
    "You compress the earlier part of an agent's conversation so it fits its context window. "
    "Write one note, in the agent's first person and under 200 words, that preserves: its current "
    "beliefs (with confidence), the evidence behind them (what it saw or was told, by whom, in "
    "which round), and the decisions, actions and commitments it has made. Drop pleasantries and "
    "repetition. Reply with the note only, as plain text."
)


class ContextLimitExceeded(RuntimeError):
    """`overflow="fail_turn"`: the prompt is over the limit; the runner ends the turn as an error."""

    turn_error = "context_limit"


def _render_for_summary(entries: list[dict]) -> str:
    lines: list[str] = []
    for entry in entries:
        head = "Earlier summary" if entry.get("summary") else f"Round {entry.get('round')}"
        lines.append(f"## {head}")
        for raw in entry["messages"]:
            m = ChatMessage.model_validate(raw)
            body = text_of(m.content)
            if not isinstance(m.content, str) and any(p.type == "image" for p in m.content):
                body += "\n[image]"
            for tc in m.tool_calls or []:
                body += f"\n[call] {tc.name}({canonical_json(tc.args)})"
            lines.append(f"{m.role}: {body.strip()}")
    return "\n".join(lines)


class LLMTurnUsage(TurnUsage):
    """`TurnUsage` plus per-turn diagnostics (lands in `turn_ended.usage`)."""

    finish_reasons: list[str] = Field(default_factory=list)
    notes: list[str] = Field(default_factory=list)


def _result_payload(res: Any) -> dict:
    return {"ok": res.ok, "result": res.result, "error": res.error}


class LLMAgent(Participant):
    entry_point: ClassVar[str | None] = "llm"

    def __init__(
        self,
        model: str,
        system_prompt: str | None = None,
        memory: Literal["full", "window", "received"] = "full",
        window_rounds: int = 3,
        max_tokens: int = 2048,
        temperature: float | None = None,
        tool_protocol: Literal["native", "json"] = "native",
        thinking_budget: int | None = None,
        max_calls: int | None = None,
        role: str = "worker",
        extra: dict | None = None,
        text_tool_fallback: bool = False,
        context_limit_tokens: int | None = None,
        overflow: Literal["drop_oldest", "summarize", "fail_turn"] = "drop_oldest",
        summary_model: str | None = None,
        system_prompt_append: str | None = None,
        memory_messages: int = 8,
        report_json: bool = False,
        report_fields: list[str] | None = None,
    ) -> None:
        if memory not in MEMORY_MODES:
            raise ValueError(f"memory must be one of {MEMORY_MODES}, got {memory!r}")
        if isinstance(memory_messages, bool) or not isinstance(memory_messages, int) or memory_messages < 0:
            raise ValueError("memory_messages must be an integer >= 0")
        fields = list(report_fields) if report_fields is not None else list(DEFAULT_REPORT_FIELDS)
        if not fields or not all(isinstance(f, str) and f for f in fields) or len(set(fields)) != len(fields):
            raise ValueError(f"report_fields must be distinct non-empty strings, got {report_fields!r}")
        if tool_protocol not in ("native", "json"):
            raise ValueError(f"tool_protocol must be 'native' or 'json', got {tool_protocol!r}")
        if window_rounds < 1:
            raise ValueError("window_rounds must be >= 1")
        if max_calls is not None and max_calls < 1:
            raise ValueError("max_calls must be >= 1 or None")
        self.model = model
        self.system_prompt = system_prompt
        self.memory = memory
        self.window_rounds = window_rounds
        self.max_tokens = max_tokens
        self.temperature = temperature
        self.tool_protocol = tool_protocol
        self.thinking_budget = thinking_budget
        self.max_calls = max_calls
        self.role = role
        self.extra = dict(extra or {})
        self.text_tool_fallback = bool(text_tool_fallback)
        # ---- context limit (M3a §3): params only when set, so existing spec hashes stay ----
        if context_limit_tokens is not None and context_limit_tokens < 1:
            raise ValueError("context_limit_tokens must be >= 1 or None")
        if overflow not in OVERFLOW_POLICIES:
            raise ValueError(f"overflow must be one of {OVERFLOW_POLICIES}, got {overflow!r}")
        if summary_model is not None:
            split_model(summary_model)
        params = getattr(self, "params", {})
        if context_limit_tokens is None:
            for k in ("context_limit_tokens", "overflow", "summary_model"):
                params.pop(k, None)
        elif summary_model is None:
            params.pop("summary_model", None)
        self.context_limit_tokens = context_limit_tokens
        self.overflow = overflow
        self.summary_model = summary_model
        self._overflow_events: list[dict] = []
        if system_prompt_append is None:  # in params (and the spec hash) only when set
            params.pop("system_prompt_append", None)
        self.system_prompt_append = system_prompt_append
        # ---- M6 §3-§4: params only when set, so existing spec hashes stay ----
        if memory_messages == 8 and memory != "received":
            params.pop("memory_messages", None)
        if not report_json:
            params.pop("report_json", None)
        if report_fields is None:
            params.pop("report_fields", None)
        self.memory_messages = memory_messages
        self.report_json = bool(report_json)
        self.report_fields = fields
        # conversation state (plain data, snapshotted by Persistable)
        self.system: str | None = None
        self.rounds: list[dict] = []
        # memory="received" state (M6 §3): the last `memory_messages` delivered items, the
        # newest delivery key ingested, the agent's own accepted answers, its last observation
        self.received: list[dict] = []
        self.received_upto: list | None = None
        self.own_answers: list[str] = []
        self.last_observation: list[dict] | None = None

    # ---- probing support ---------------------------------------------------------------------
    def probe_context(self) -> list[ChatMessage]:
        msgs = [ChatMessage(role="system", content=self.system)] if self.system else []
        if self.memory == "received":  # M6 §3: the same construction, without the schema line
            if self.last_observation is None:
                return msgs
            return msgs + [self._received_message(schema=False)]
        return msgs + [ChatMessage.model_validate(m) for r in self.rounds for m in r["messages"]]

    # ---- memory="received" (M6 §3) -------------------------------------------------------------
    @property
    def calls_per_turn_cap(self) -> int | None:
        """Most model calls a turn can make (for `swarmlab estimate`): `max_calls`, or 2 under
        `report_json` (one answer plus one retry)."""
        if self.report_json:
            return min(self.max_calls or 2, 2)
        return self.max_calls

    def ingest(self, items: list[dict]) -> int:
        """Add delivered board items (`{delivery_id, post_id, eligible_round, content}`) to the
        received memory: only items newer (by `(eligible_round, delivery_id)`) than every item
        ingested so far, oldest first, keeping the last `memory_messages`. Returns how many were
        added."""
        def key(it: dict) -> list:
            return [int(it.get("eligible_round") or 0), str(it.get("delivery_id") or "")]

        added = 0
        for it in sorted(items, key=key):
            k = key(it)
            if self.received_upto is not None and k <= list(self.received_upto):
                continue
            content = it.get("content")
            if not isinstance(content, str):
                content = json.dumps(content, sort_keys=True)
            self.received.append({"delivery_id": it.get("delivery_id"), "post_id": it.get("post_id"),
                                  "content": content})
            self.received_upto = k
            added += 1
        if self.memory_messages == 0:
            self.received = []
        else:
            self.received = self.received[-self.memory_messages:]
        return added

    def _received_message(self, schema: bool = True) -> ChatMessage:
        """Observation parts, the transcript block, the own-answers line, the schema line."""
        parts = [Part.model_validate(p) for p in self.last_observation or []]
        if self.received:
            block = "\n".join([TRANSCRIPT_HEADER] + [f"- {it['content']}" for it in self.received])
        else:
            block = f"{TRANSCRIPT_HEADER} []"
        parts.append(Part(type="text", text=block))
        if self.own_answers:
            parts.append(Part(type="text", text=f"{OWN_ANSWERS_HEADER} "
                              + json.dumps(self.own_answers, ensure_ascii=False)))
        if schema and self.report_json:
            parts.append(Part(type="text", text=report_schema(self.report_fields)))
        return ChatMessage(role="user", content=parts)

    def _note_answer(self, name: str, args: dict, ok: bool) -> None:
        if name == "guess" and ok and args:
            value = next(iter(args.values()))
            self.own_answers = (self.own_answers + [str(value)])[-max(1, self.memory_messages):]

    def _note_read(self, name: str, result: Any) -> None:
        if self.memory == "received" and name == "read_board" and isinstance(result, dict):
            items = result.get("items")
            if isinstance(items, list):
                self.ingest([i for i in items if isinstance(i, dict)])

    def prepare_probe(self, view_for: Any) -> None:
        """Runner hook before a probe (M6 §3): under `memory="received"`, ingest the items
        delivered since the last turn and, when the agent has not had a turn yet, store its
        observation, so `probe_context()` shows the crop and the memory. `view_for(observe)`
        builds the agent's View (with the world observation only when `observe`)."""
        if self.memory != "received":
            return
        view = view_for(self.last_observation is None)
        if self.system is None:
            self.system = self._render_system(view)
        if self.last_observation is None:
            self.last_observation = [p.model_dump(mode="json") for p in view.observation.parts]
        self.ingest(list(view.pushed))

    def reconfigure(self, **kw: Any) -> None:
        """M3a `Ops.reconfigure`: change `model`, `system_prompt` (re-rendered at the next turn),
        `memory`, `window_rounds` or `max_tokens` from the next turn on. The settings are plain
        attributes, so they are in the snapshot and survive resume; `params` (the spec) is unchanged."""
        allowed = {"model", "system_prompt", "memory", "window_rounds", "max_tokens"}
        unknown = set(kw) - allowed
        if unknown:
            raise ValueError(f"LLMAgent.reconfigure: unknown settings {sorted(unknown)}; allowed {sorted(allowed)}")
        if kw.get("memory", self.memory) not in MEMORY_MODES:
            raise ValueError(f"memory must be one of {MEMORY_MODES}, got {kw['memory']!r}")
        if kw.get("window_rounds", self.window_rounds) < 1:
            raise ValueError("window_rounds must be >= 1")
        for k, v in kw.items():
            setattr(self, k, v)
        if "system_prompt" in kw:
            self.system = None

    def apply_role(self, role: Any) -> None:
        """M3c: apply a `swarmlab.roles.Role` at bind (see the module doc)."""
        self._role_prompt_append = role.prompt_append
        self.role = role.name
        if role.system_prompt is not None:
            self.system_prompt = role.system_prompt
        if role.model is not None:
            self.model = role.model
            self._estimator = None
        if role.budget and "max_tokens" in role.budget:
            self.max_tokens = int(role.budget["max_tokens"])

    def model_request_defaults(self) -> dict:
        return {"model": self.model, "temperature": self.temperature,
                "max_tokens": self.max_tokens, "thinking_budget": self.thinking_budget,
                "extra": dict(self.extra)}

    # ---- context limit (M3a §3) ----------------------------------------------------------------
    def drain_overflow(self) -> list[dict]:
        """Overflow records of the last turn (the runner turns them into `overflow` events)."""
        out, self._overflow_events = list(getattr(self, "_overflow_events", [])), []
        return out

    def _estimate_tokens(self, request: ChatRequest) -> int:
        est = getattr(self, "_estimator", None)
        if est is None:
            from ..providers import preset

            try:
                est = preset(split_model(self.model)[0])
            except Exception:  # noqa: BLE001 - no preset (e.g. vllm without base url): chars/4
                est = False
            self._estimator = est
        if est:
            return est.estimate_prompt_tokens(request)
        return math.ceil(len(canonical_json(request.model_dump(mode="json",
                                                               include={"messages", "tools"}))) / 4)

    async def _enforce_context_limit(self, tools_offered: list[ToolSchema], tools: AgentTools) -> None:
        """Apply the overflow policy if the next request would exceed `context_limit_tokens`."""
        limit = self.context_limit_tokens
        if limit is None:
            return
        before = self._estimate_tokens(self._request(tools_offered))
        if before <= limit:
            return
        record = {"policy": self.overflow, "dropped_rounds": 0, "tokens_before": before,
                  "tokens_after": before}
        if self.overflow == "fail_turn":
            self._overflow_events.append(record)
            raise ContextLimitExceeded(f"prompt estimate {before} > context_limit_tokens {limit}")
        reserve = SUMMARY_MAX_TOKENS if self.overflow == "summarize" else 0
        dropped: list[dict] = []
        est = before
        while len(self.rounds) > 1 and est + reserve > limit:
            dropped.append(self.rounds.pop(0))
            est = self._estimate_tokens(self._request(tools_offered))
        record["dropped_rounds"] = sum(1 for e in dropped if not e.get("summary"))
        if self.overflow == "summarize" and dropped:
            note = await self._summarize(dropped, tools)
            if note is None:
                record["detail"] = "measurement_budget"
            else:
                self.rounds.insert(0, {"round": dropped[-1]["round"], "from_round": note[0],
                                       "summary": True, "messages": [ChatMessage(
                                           role="assistant", content=note[1]).model_dump(mode="json")]})
            est = self._estimate_tokens(self._request(tools_offered))
        record["tokens_after"] = est
        self._overflow_events.append(record)

    async def _summarize(self, dropped: list[dict], tools: AgentTools) -> tuple[int, str] | None:
        first = dropped[0].get("from_round", dropped[0]["round"])
        req = ChatRequest(
            model=self.summary_model or DEFAULT_SUMMARY_MODEL, max_tokens=SUMMARY_MAX_TOKENS,
            temperature=0.0, tools=[], messages=[
                ChatMessage(role="system", content=SUMMARY_SYSTEM),
                ChatMessage(role="user", content=f"Agent: {self.agent}\n\n"
                            + _render_for_summary(dropped)),
            ])
        try:
            resp = await tools.infer(req, category="measurement")
        except MeasurementBudgetReached:
            return None
        text = strip_reasoning(resp.text or "").strip()
        return first, f"[Summary of my earlier rounds {first}-{dropped[-1]['round']}]\n{text}"

    # ---- the loop ----------------------------------------------------------------------------
    def _render_system(self, view: View) -> str:
        report = getattr(self, "report_json", False)
        text = render_system_prompt(self.system_prompt, agent=str(self.agent), role=self.role,
                                    description=view.description, tools=view.tools,
                                    append=getattr(self, "system_prompt_append", None),
                                    default=REPORT_SYSTEM_TEMPLATE if report else DEFAULT_SYSTEM_TEMPLATE,
                                    answer=(getattr(self, "report_fields", None) or ["country"])[0])
        role_append = getattr(self, "_role_prompt_append", None)  # M3c: the role's text, at the end
        if role_append:
            text += "\n\n" + role_append.strip()
        if self.tool_protocol == "json" and not report:
            text += "\n\n" + json_protocol_text(view.tools)
        return text

    @staticmethod
    def round_message(view: View) -> ChatMessage:
        parts = [Part(type="text", text=f"Round {view.round}.")]
        parts += [p.model_copy() for p in view.observation.parts]
        if view.outcomes:
            lines = ["Outcomes of your actions last round:"]
            for o in view.outcomes:
                verdict = "accepted" if o.get("accepted") else "rejected"
                fb = json.dumps(o.get("feedback") or {}, sort_keys=True)
                lines.append(f"- {o.get('tool')} {o.get('action_id')}: {verdict} {fb}")
            parts.append(Part(type="text", text="\n".join(lines)))
        if view.pushed:
            lines = ["Delivered to you:"]
            for item in view.pushed:
                content = item.get("content")
                if not isinstance(content, str):
                    content = json.dumps(content, sort_keys=True)
                lines.append(f"- {item.get('post_id')} (round {item.get('eligible_round')}): {content}")
            parts.append(Part(type="text", text="\n".join(lines)))
        return ChatMessage(role="user", content=parts)

    def _append(self, msg: ChatMessage) -> None:
        self.rounds[-1]["messages"].append(msg.model_dump(mode="json"))

    def _request(self, tools: list[ToolSchema]) -> ChatRequest:
        return ChatRequest(messages=self.probe_context(), tools=tools,
                           tool_protocol=self.tool_protocol, **self.model_request_defaults())

    async def turn(self, view: View, tools: AgentTools) -> LLMTurnUsage:
        if self.system is None:
            self.system = self._render_system(view)
        if self.memory == "window":
            self.rounds = self.rounds[-(self.window_rounds - 1):] if self.window_rounds > 1 else []
        if self.memory == "received":  # M6 §3: no history across turns, one constructed message
            self.ingest(list(view.pushed))
            self.last_observation = [p.model_dump(mode="json") for p in view.observation.parts]
            self.rounds = [{"round": view.round, "messages": []}]
            self._append(self._received_message())
        else:
            self.rounds.append({"round": view.round, "messages": []})
            msg = self.round_message(view)
            if self.report_json:  # M6 §4: the schema line ends every turn's message
                msg.content.append(Part(type="text", text=report_schema(self.report_fields)))
            self._append(msg)
        usage = LLMTurnUsage()
        self._overflow_events = []
        if self.report_json:
            return await self._report_turn(view, tools, usage)
        executed = 0
        model_calls = 0
        retried = False
        while self.max_calls is None or model_calls < self.max_calls:
            await self._enforce_context_limit(view.tools, tools)  # M3a §3 context limit
            resp = await tools.infer(self._request(view.tools))
            model_calls += 1
            usage.finish_reasons.append(resp.finish_reason)
            if resp.finish_reason == "length" and "length" not in usage.notes:
                usage.notes.append("length")  # ran out of max_tokens (often mid-reasoning)
            if self.tool_protocol == "native":
                if not resp.tool_calls:
                    text_calls = (parse_text_tool_calls(resp.text, view.tools)
                                  if self.text_tool_fallback and resp.text else [])
                    if resp.text:
                        self._append(ChatMessage(role="assistant", content=resp.text))
                    if not text_calls:
                        break
                    log.info("%s round %s: text_tool_fallback parsed %d call(s): %s", self.agent,
                             view.round, len(text_calls), [n for n, _ in text_calls])
                    usage.notes.append(f"text_tool_fallback:{len(text_calls)}")
                    n, ended = await self._run_json(text_calls, tools)
                    executed += n
                    if ended:
                        break
                    continue
                self._append(ChatMessage(role="assistant", content=resp.text,
                                         tool_calls=resp.tool_calls))
                n, ended = await self._run_native(resp.tool_calls, tools)
            else:
                self._append(ChatMessage(role="assistant", content=resp.text))
                try:
                    calls = parse_tool_json(resp.text)
                except ToolJsonError as e:
                    if retried:
                        break
                    retried = True
                    self._append(ChatMessage(role="user", content=f"{RESULTS_PREFIX}\n" + json.dumps(
                        [{"ok": False, "error": f"could not parse your tool calls: {e}"}])))
                    continue
                if not calls:
                    break
                n, ended = await self._run_json(calls, tools)
            executed += n
            if ended:
                break
        usage.calls = executed
        return usage

    async def _run_native(self, calls: list[ToolCall], tools: AgentTools) -> tuple[int, bool]:
        answered = 0
        ended = False
        try:
            for tc in calls:
                if "_raw" in tc.args and len(tc.args) == 1:
                    payload = {"ok": False, "result": {},
                               "error": "arguments are not a JSON object; nothing was executed"}
                else:
                    res = await tools.call(tc.name, tc.args)
                    payload = _result_payload(res)
                    self._note_answer(tc.name, tc.args, res.ok)
                    self._note_read(tc.name, res.result)
                    ended = ended or tc.name == "end_turn"
                self._append(ChatMessage(role="tool", content=json.dumps(payload, sort_keys=True),
                                         tool_call_id=tc.call_id))
                answered += 1
        finally:
            for tc in calls[answered:]:  # keep the conversation well-formed after an exception
                self._append(ChatMessage(role="tool", tool_call_id=tc.call_id, content=json.dumps(
                    {"ok": False, "result": {}, "error": "not executed: the turn was cut off"})))
        return answered, ended

    async def _run_json(self, calls: list[tuple[str, dict]], tools: AgentTools) -> tuple[int, bool]:
        results: list[dict] = []
        ended = False
        try:
            for name, args in calls:
                res = await tools.call(name, args)
                self._note_answer(name, args, res.ok)
                self._note_read(name, res.result)
                results.append({"name": name, **_result_payload(res)})
                ended = ended or name == "end_turn"
        finally:
            done = len(results)
            for name, _ in calls[done:]:
                results.append({"name": name, "ok": False, "result": {},
                                "error": "not executed: the turn was cut off"})
            self._append(ChatMessage(role="user", content=f"{RESULTS_PREFIX}\n"
                                     + json.dumps(results, sort_keys=True)))
        return done, ended

    # ---- report_json (M6 §4) -------------------------------------------------------------------
    async def _report_turn(self, view: View, tools: AgentTools, usage: LLMTurnUsage) -> LLMTurnUsage:
        """One JSON answer -> `guess(<answer>)` + `post(text=<json>)` + `end_turn` (see module doc)."""
        offered = {t.name: t for t in view.tools}
        attempts = self.calls_per_turn_cap or 2
        executed = 0
        for attempt in range(attempts):
            await self._enforce_context_limit(view.tools, tools)
            resp = await tools.infer(self._request(view.tools))
            usage.finish_reasons.append(resp.finish_reason)
            if resp.finish_reason == "length" and "length" not in usage.notes:
                usage.notes.append("length")
            if self.tool_protocol == "native" and resp.tool_calls:  # the model used tools itself
                self._append(ChatMessage(role="assistant", content=resp.text, tool_calls=resp.tool_calls))
                n, _ = await self._run_native(resp.tool_calls, tools)
                usage.calls = executed + n
                usage.notes.append("report_json:native_tools")
                return usage
            self._append(ChatMessage(role="assistant", content=resp.text or ""))
            report, error = self._parse_report(resp.text or "")
            results: list[dict] = []
            if report is not None:
                answer = report[self.report_fields[0]]
                guess = offered.get("guess")
                if guess is not None:
                    arg = next(iter(guess.parameters.get("properties") or {"candidate": {}}))
                    res = await tools.call("guess", {arg: answer})
                    executed += 1
                    self._note_answer("guess", {arg: answer}, res.ok)
                    results.append({"name": "guess", **_result_payload(res)})
                    if not res.ok:
                        error = f"answer {answer!r} was rejected: {res.error}"
                else:  # no guess tool (e.g. a blind manager): the answer is still its decision
                    self._note_answer("guess", {"answer": answer}, True)
                if error is None:
                    if "post" in offered:
                        text = json.dumps(report, ensure_ascii=False, separators=(",", ":"))
                        res = await tools.call("post", {"text": text})
                        executed += 1
                        results.append({"name": "post", **_result_payload(res)})
                    await tools.call("end_turn", {})
                    executed += 1
                    if self.memory != "received":
                        self._append(ChatMessage(role="user", content=f"{RESULTS_PREFIX}\n"
                                                 + json.dumps(results, sort_keys=True)))
                    usage.calls = executed
                    return usage
            usage.notes.append(f"report_json:retry:{error}"[:120] if attempt + 1 < attempts
                               else f"report_json:failed:{error}"[:120])
            self._append(ChatMessage(role="user", content=(
                f"Your reply could not be used: {error}. Answer with only the JSON object. "
                + report_schema(self.report_fields))))
        usage.calls = executed
        return usage

    def _canonical(self, answer: str) -> str:
        """`answer` spelled as in the observation's list of allowed names (FlagGame
        `candidate_names`: candidate headers or `Allowed countries`), matched case-insensitively
        after collapsing whitespace; unchanged when nothing matches."""
        from ..world.flaggame import candidate_names
        from ..world.flags_real import match_country

        parts = self.last_observation
        if parts is None and self.rounds and self.rounds[-1]["messages"]:
            parts = self.rounds[-1]["messages"][0].get("content")
        text = "\n".join(p.get("text") or "" for p in parts or [] if isinstance(p, dict))
        names = candidate_names(text)
        return match_country(answer, names) or answer if names else answer

    def _parse_report(self, text: str) -> tuple[dict | None, str | None]:
        """(report with exactly the `report_fields` present, in order; or None, the error)."""
        from ..probes import parse_json_object

        obj = parse_json_object(strip_reasoning(text))
        if obj is None:
            return None, "no JSON object"
        lower = {str(k).lower(): v for k, v in obj.items()}
        key = self.report_fields[0]
        answer = lower.get(key.lower())
        if isinstance(answer, (int, float)) and not isinstance(answer, bool):
            answer = str(answer)
        if not isinstance(answer, str) or not answer.strip():
            return None, f"no string {key!r}"
        report = {key: self._canonical(answer.strip())}
        for f in self.report_fields[1:]:
            v = lower.get(f.lower())
            if v is not None:
                report[f] = v if isinstance(v, str) else json.dumps(v, ensure_ascii=False)
        return report, None
