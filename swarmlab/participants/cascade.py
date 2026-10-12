"""CascadeWorker: the one-shot worker of the `cascade` world (swarmlab/world/cascade.py).

An `LLMAgent` subclass (entry point `cascade_worker`) that sends the source's prompt instead of
the default framing: the system message is `SYSTEM` with the worker's name (`a003` -> `W03`).
A set `system_prompt` (param, or a role's `system_prompt`) replaces it and is rendered as
`LLMAgent` renders it (Jinja: `agent`, `role`, `description`, `tools` = the tools the worker
offers, `system_prompt_append`, plus `wid`); `system_prompt_append` follows the default prompt
after a blank line (a custom template places it itself or gets it at the end), and a role's
`prompt_append` comes last. The turn's single user message (`turn_message`) is, separated by
blank lines,

    <world observation: documentation + private probe result>
    <board>                      EMPTY_BOARD, or the board header and one paragraph per post
                                 (TOOL_BOARD_LINE under board_channel: tool)
    [<board rule>]               board_rule (`quote` -> QUOTE_RULE, or any text)
    [<pay paragraph>]            REWARD, when reward_last is set
    QUESTION
    INSTRUCTION                  INSTRUCTION_DISCLOSED when disclose_belief
    FORMAT                       SUBMIT_FORMAT under board_channel: tool

The board is the turn's pushed items in delivery order (the world's `requires` asks for push
delivery with `push_limit >= N`). There is no "Round 1." line, tool list or `Delivered to you:`.

Turn loop: `LLMAgent.turn()` with its two steps overridden (llm.py module doc, Turn loop).
`_infer` sets the offered tools (below) and calls the base `_infer`, which keeps the context
limit, the `length` note and the `refusal_retries` re-sends. `_handle_response` reads the answer:
a usable one runs `guess(candidate=<interpretation>)`, `post(text="W03: <board_post>")` (with
`disclose_belief`: `"W03 [committed: <interpretation>]: <board_post>"`) and `end_turn`; an unusable
one gets a correction (a `tool` result per tool call, else a user message) and the note
`cascade:retry:<reason>`, or `cascade:failed:<reason>` on the last call, which leaves the worker
with no commitment and no post. A guess the world does not accept (`accepted: false` or an
`error` in its feedback; the world refuses a second commitment) ends the turn at once with
`cascade:failed:commit rejected: <error>` and no post. Every tool call of the answer gets a
`tool` result in memory: the submit call that was carried out gets
`{"ok": true, "result": {"interpretation", "committed": true, "posted": <post text>}}` (or
`{"ok": false, "error": "commit rejected: ..."}`), any other call a "not executed" error, so the
exported session (swarmlab/export.py) pairs every call with its result. `max_calls` defaults to 2
and is capped at 2. Requests are the
same as the earlier hand-written loop's; from the base class come a `refusal:<category>` note per
refused response and `memory`, kept-thinking and `context_limit_tokens` handling.
`memory="received"` and `report_json` are refused (they replace the prompt or the reply handling).

Board channel `user` (default)
------------------------------
The board is part of the user message. The model is called with no tools and must reply with a
JSON object `{"board_post": ..., "interpretation": "reads_transcript" | "output_only"}` (the
interpretation is matched case-insensitively, with spaces and hyphens read as underscores).

Board channel `tool`
--------------------
The user message says the board is "available through the read_board tool". Before the turn's
first call the transcript gets the worker's `read_board` call (an assistant message with one tool
call, id `read_board_0`) and its result: a `tool` message whose text is exactly the board the
`user` channel would show. The worker writes both itself (no executed `read_board` call, so the
log's `reads` are unchanged); the export shows the call as a `swarmlab-participant` message. The
model is offered `read_board` and `submit(board_post, interpretation)`, and the request requires
`submit` through `ChatRequest.tool_choice = ToolChoice(name="submit")`; the provider's wire format
is the adapter's (providers/base.py, Tool choice). A text reply is still parsed as JSON. Plain JSON
answers after a tool result were unreliable on Qwen3.8-27B (it re-called `read_board`, or replied
empty or in prose). A forced choice needs native tools, so `board_channel: tool` with
`tool_protocol: json` is refused when the worker is built, by the request validation
(`check_tool_choice`), rather than sending a forced choice the model is never offered.
docs/guide/cascade.md has the details; give this arm `max_tokens: 2048`.

Params, besides every `LLMAgent` param:
- `show_board` (default true): false shows EMPTY_BOARD to every worker (no-board control).
- `board_rule`: a rule paragraph after the board; `quote` selects QUOTE_RULE.
- `disclose_belief` (default false): attach the commitment to each post.
- `reward_last`: k; adds the pay paragraph naming the last k workers.
- `n_workers` (default 20): only used to name the last k workers.
- `board_channel` (`user` or `tool`).
- `board_header`: `agents`, `humans` (BOARD_HEADERS) or any text; default BOARD_HEADER.
"""
from __future__ import annotations

import json
import re
from typing import Any, ClassVar

from ..probes import parse_json_object, strip_reasoning
from ..providers import ChatMessage, ChatRequest, ChatResponse, ToolChoice
from ..providers.base import check_tool_choice
from ..tools import AgentTools, ToolCall, ToolSchema
from ..view import View
from ..world import cascade as cw
from .llm import LLMAgent, LLMTurnUsage, render_system_prompt

READ_BOARD = ToolSchema(name="read_board", description="Read the shared worker board.",
                        parameters={"type": "object", "properties": {}})
SUBMIT = ToolSchema(
    name="submit", description="Post your message to the shared board and commit to an interpretation.",
    parameters={"type": "object", "required": ["board_post", "interpretation"], "properties": {
        "board_post": {"type": "string", "description": "The message you post to the shared board."},
        "interpretation": {"type": "string", "enum": list(cw.INTERPRETATIONS)}}})
READ_CALL_ID = "read_board_0"
SUBMIT_CHOICE = ToolChoice(name="submit")


def normalize_interpretation(value: Any) -> str | None:
    v = re.sub(r"[\s\-]+", "_", value.strip().strip("`'\"").lower()) if isinstance(value, str) else None
    return v if v in cw.INTERPRETATIONS else None


def parse_fields(obj: dict) -> tuple[dict | None, str | None]:
    """(`{"board_post", "interpretation"}`, None) or (None, the reason the answer is unusable)."""
    lower = {str(k).lower(): v for k, v in obj.items()}
    post = lower.get("board_post")
    interp = normalize_interpretation(lower.get("interpretation"))
    if not isinstance(post, str) or not post.strip():
        return None, "no string 'board_post'"
    if interp is None:
        return None, f"'interpretation' must be one of {list(cw.INTERPRETATIONS)}"
    return {"board_post": post.strip(), "interpretation": interp}, None


def parse_reply(text: str) -> tuple[dict | None, str | None]:
    """`parse_fields` of the first JSON object in a text reply."""
    obj = parse_json_object(strip_reasoning(text or ""))
    return (None, "no JSON object") if obj is None else parse_fields(obj)


class CascadeWorker(LLMAgent):
    entry_point: ClassVar[str | None] = "cascade_worker"
    description: ClassVar[str] = ("The cascade world's one-shot worker: the source's prompt and "
                                  "board, answered as JSON or a forced submit call")

    def __init__(self, model: str, show_board: bool = True, board_rule: str | None = None,
                 disclose_belief: bool = False, reward_last: int | None = None,
                 n_workers: int = 20, board_channel: str = "user", board_header: str | None = None,
                 **kwargs: Any) -> None:
        kwargs.setdefault("max_calls", 2)
        super().__init__(model=model, **kwargs)
        if self.memory == "received" or self.report_json:
            raise ValueError("cascade_worker does not support memory='received' or report_json")
        if reward_last is not None and not 1 <= reward_last <= n_workers:
            raise ValueError(f"reward_last must be in [1, n_workers={n_workers}], got {reward_last!r}")
        if board_channel not in ("user", "tool"):
            raise ValueError(f"board_channel must be 'user' or 'tool', got {board_channel!r}")
        if board_channel == "tool":  # the forced submit needs native tools (check_tool_choice)
            probe = ChatRequest(model=model, messages=[], tools=[READ_BOARD, SUBMIT],
                                tool_protocol=self.tool_protocol)
            try:
                check_tool_choice(probe.model_copy(update={"tool_choice": SUBMIT_CHOICE}))
            except ValueError as e:
                raise ValueError(f"board_channel 'tool' forces the submit call: {e}") from None
        self.show_board = bool(show_board)
        self.board_rule = cw.QUOTE_RULE if board_rule == "quote" else board_rule
        self.disclose_belief = bool(disclose_belief)
        self.reward_last = reward_last
        self.n_workers = n_workers
        self.board_channel = board_channel
        self.board_header = (cw.BOARD_HEADERS.get(board_header, board_header) if board_header
                             else cw.BOARD_HEADER)

    @property
    def calls_per_turn_cap(self) -> int | None:
        return min(self.max_calls or 2, 2)

    def offered_tools(self) -> list[ToolSchema]:
        """The tools the worker offers the model: none (`user`), `read_board` + `submit` (`tool`)."""
        return [READ_BOARD, SUBMIT] if self.board_channel == "tool" else []

    def _render_system(self, view: View) -> str:
        """SYSTEM with the worker id; a set `system_prompt` replaces it (rendered like LLMAgent's,
        plus the variable `wid`); `system_prompt_append` and a role's `prompt_append` follow."""
        wid = cw.worker_id(self.agent)
        if self.system_prompt is None:
            text = cw.SYSTEM.format(wid=wid)
            extra = (self.system_prompt_append or "").strip()
            if extra:
                text += "\n\n" + extra
        else:
            text = render_system_prompt(self.system_prompt, agent=str(self.agent), role=self.role,
                                        description=view.description, tools=self.offered_tools(),
                                        append=self.system_prompt_append, wid=wid)
        role_append = getattr(self, "_role_prompt_append", None)
        if role_append:
            text += "\n\n" + role_append.strip()
        return text

    def board_text(self, view: View) -> str:
        """The board as the `user` channel shows it (and the `tool` channel returns it)."""
        posts = [str(it.get("content")) for it in view.pushed] if self.show_board else []
        return self.board_header + "\n\n" + "\n\n".join(posts) if posts else cw.EMPTY_BOARD

    def round_message(self, view: View) -> ChatMessage:  # type: ignore[override]
        obs = "\n\n".join(p.text or "" for p in view.observation.parts if p.type == "text")
        tool = self.board_channel == "tool"
        sections = [obs, cw.TOOL_BOARD_LINE if tool else self.board_text(view)]
        if self.board_rule:
            sections.append(self.board_rule)
        if self.reward_last:
            k, n = self.reward_last, self.n_workers
            sections.append(cw.REWARD.format(k=k, first=cw.worker_id(f"a{n - k}"),
                                          last=cw.worker_id(f"a{n - 1}")))
        sections += [cw.QUESTION, cw.INSTRUCTION_DISCLOSED if self.disclose_belief else cw.INSTRUCTION,
                     cw.SUBMIT_FORMAT if tool else cw.FORMAT]
        return ChatMessage(role="user", content="\n\n".join(sections))

    def turn_message(self, view: View) -> ChatMessage:
        return self.round_message(view)

    async def _infer(self, view: View, tools: AgentTools, usage: LLMTurnUsage) -> ChatResponse:
        if self.board_channel == "tool":
            msgs = self.rounds[-1]["messages"]
            if not any(m.get("tool_call_id") == READ_CALL_ID for m in msgs):  # first call: pre-fill
                self._append(ChatMessage(role="assistant", content="", tool_calls=[
                    ToolCall(call_id=READ_CALL_ID, name="read_board", args={})]))
                self._append(ChatMessage(role="tool", content=self.board_text(view),
                                         tool_call_id=READ_CALL_ID))
        offered = self.offered_tools()
        return await super()._infer(view.model_copy(update={"tools": offered}), tools, usage)

    def _request(self, tools: list[ToolSchema], attempt: int = 0) -> ChatRequest:
        req = super()._request(tools, attempt)
        if tools:  # the tool channel: require the submit call (checked in __init__ and by adapters)
            req = req.model_copy(update={"tool_choice": SUBMIT_CHOICE})
        return req

    async def _handle_response(self, resp: ChatResponse, view: View,
                               tools: AgentTools) -> tuple[int, bool]:
        self._append(self._assistant(resp, resp.tool_calls or None))
        sub: ToolCall | None = None
        if resp.tool_calls:
            sub = next((c for c in resp.tool_calls if c.name == "submit"), None)
            reply, error = (parse_fields(sub.args) if sub is not None and "_raw" not in sub.args
                            else (None, "no valid submit call"))
        else:
            reply, error = parse_reply(resp.text or "")
        if reply is not None:
            res = await tools.call("guess", {"candidate": reply["interpretation"]})
            outcome = res.result or {}
            feedback = outcome.get("feedback") or {}
            if res.ok and outcome.get("accepted", True) and "error" not in feedback:
                tag = f" [committed: {reply['interpretation']}]" if self.disclose_belief else ""
                text = f"{cw.worker_id(self.agent)}{tag}: {reply['board_post']}"
                posted = await tools.call("post", {"text": text})
                await tools.call("end_turn", {})
                self._answer_calls(resp, sub, json.dumps({"ok": True, "error": None, "result": {
                    "interpretation": reply["interpretation"], "committed": True,
                    "posted": text if posted.ok else None}}, sort_keys=True))
                return 3, True
            # a refused commitment is final: no post, no retry (the world has decided)
            error = f"commit rejected: {feedback.get('error') or res.error or 'not accepted'}"
            self._turn.usage.notes.append(f"cascade:failed:{error}"[:120])
            self._answer_calls(resp, sub, json.dumps({"ok": False, "result": {}, "error": error},
                                                     sort_keys=True))
            return 1, True
        state = self._turn
        last = state.budget is not None and state.calls >= state.budget
        state.usage.notes.append(f"cascade:{'failed' if last else 'retry'}:{error}"[:120])
        fmt = cw.SUBMIT_FORMAT if self.board_channel == "tool" else cw.FORMAT
        for c in resp.tool_calls:  # every tool call needs a result before the next request
            self._append(ChatMessage(role="tool", tool_call_id=c.call_id,
                                     content=f"Your submit call could not be used: {error}. {fmt}"))
        if not resp.tool_calls:
            self._append(ChatMessage(role="user", content=f"Your reply could not be used: {error}. {fmt}"))
        return 0, False

    def _answer_calls(self, resp: ChatResponse, sub: ToolCall | None, content: str) -> None:
        """A `tool` result for every call of `resp`: `content` for the submit call that was used
        (the outcome of the guess/post it was carried out with), a note for any other."""
        for c in resp.tool_calls:
            text = content if c is sub else json.dumps(
                {"ok": False, "result": {}, "error": "not executed: the turn was carried out with the submit call"},
                sort_keys=True)
            self._append(ChatMessage(role="tool", tool_call_id=c.call_id, content=text))
