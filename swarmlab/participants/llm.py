"""LLMAgent: the in-process model loop (docs/INTERFACE-M1b.md §4).

`LLMAgent(model, system_prompt=None, memory="full", window_rounds=3, max_tokens=1024,
temperature=None, tool_protocol="native", thinking_budget=None, max_calls=None, role="worker")`,
entry point `llm`.

**System prompt.** A Jinja2 template rendered once, at the agent's first turn, with `agent`,
`role`, `description` (`View.description`, i.e. `World.description()`) and `tools`
(`view.tools`, `ToolSchema` objects). The default is `prompts/default_system.j2`;
`system_prompt` replaces it with a template string or `file:<path>` (read at render time; the
path is what goes into `params`). Rendering uses `trim_blocks`/`lstrip_blocks` and
`StrictUndefined`. Under `tool_protocol="json"` a fixed JSON-protocol section (with each tool's
parameter schema, since no tools go through the provider API) is appended after the template.

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

**Probing.** `probe_context()` returns `[system] + memory` as fresh `ChatMessage` objects (callers
cannot mutate the agent's memory through them); `model_request_defaults()` returns `model,
temperature, max_tokens, thinking_budget`.

`TurnUsage.calls` counts executed tool calls; tokens and cost are filled in by the executor.
"""
from __future__ import annotations

import json
import re
from pathlib import Path
from typing import Any, ClassVar, Literal

import jinja2

from ..providers.base import ChatMessage, ChatRequest
from ..tools import AgentTools, ToolCall, ToolSchema
from ..view import Part, View
from .base import Participant, TurnUsage

PROMPTS_DIR = Path(__file__).resolve().parent / "prompts"
DEFAULT_SYSTEM_TEMPLATE = PROMPTS_DIR / "default_system.j2"
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
                         tools: list[ToolSchema]) -> str:
    """Render `template` (text, `file:<path>`, or None for the default) with the agent's context."""
    if template is None:
        text = DEFAULT_SYSTEM_TEMPLATE.read_text()
    elif template.startswith("file:"):
        text = Path(template[len("file:"):]).read_text()
    else:
        text = template
    return _ENV.from_string(text).render(agent=agent, role=role, description=description,
                                         tools=tools).strip()


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


def _result_payload(res: Any) -> dict:
    return {"ok": res.ok, "result": res.result, "error": res.error}


class LLMAgent(Participant):
    entry_point: ClassVar[str | None] = "llm"

    def __init__(
        self,
        model: str,
        system_prompt: str | None = None,
        memory: Literal["full", "window"] = "full",
        window_rounds: int = 3,
        max_tokens: int = 1024,
        temperature: float | None = None,
        tool_protocol: Literal["native", "json"] = "native",
        thinking_budget: int | None = None,
        max_calls: int | None = None,
        role: str = "worker",
    ) -> None:
        if memory not in ("full", "window"):
            raise ValueError(f"memory must be 'full' or 'window', got {memory!r}")
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
        # conversation state (plain data, snapshotted by Persistable)
        self.system: str | None = None
        self.rounds: list[dict] = []

    # ---- probing support ---------------------------------------------------------------------
    def probe_context(self) -> list[ChatMessage]:
        msgs = [ChatMessage(role="system", content=self.system)] if self.system else []
        return msgs + [ChatMessage.model_validate(m) for r in self.rounds for m in r["messages"]]

    def model_request_defaults(self) -> dict:
        return {"model": self.model, "temperature": self.temperature,
                "max_tokens": self.max_tokens, "thinking_budget": self.thinking_budget}

    # ---- the loop ----------------------------------------------------------------------------
    def _render_system(self, view: View) -> str:
        text = render_system_prompt(self.system_prompt, agent=str(self.agent), role=self.role,
                                    description=view.description, tools=view.tools)
        if self.tool_protocol == "json":
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

    async def turn(self, view: View, tools: AgentTools) -> TurnUsage:
        if self.system is None:
            self.system = self._render_system(view)
        if self.memory == "window":
            self.rounds = self.rounds[-(self.window_rounds - 1):] if self.window_rounds > 1 else []
        self.rounds.append({"round": view.round, "messages": []})
        self._append(self.round_message(view))
        executed = 0
        model_calls = 0
        retried = False
        while self.max_calls is None or model_calls < self.max_calls:
            resp = await tools.infer(self._request(view.tools))
            model_calls += 1
            if self.tool_protocol == "native":
                if not resp.tool_calls:
                    if resp.text:
                        self._append(ChatMessage(role="assistant", content=resp.text))
                    break
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
        return TurnUsage(calls=executed)

    async def _run_native(self, calls: list[ToolCall], tools: AgentTools) -> tuple[int, bool]:
        answered = 0
        ended = False
        try:
            for tc in calls:
                if "_raw" in tc.args and len(tc.args) == 1:
                    payload = {"ok": False, "result": {},
                               "error": "arguments are not a JSON object; nothing was executed"}
                else:
                    payload = _result_payload(await tools.call(tc.name, tc.args))
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
