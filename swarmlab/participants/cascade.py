"""CascadeWorker: the one-shot worker of the `cascade` world (swarmlab/world/cascade.py).

An `LLMAgent` subclass (entry point `cascade_worker`) that sends the source's prompt instead of
the default framing: the system message is `SYSTEM` with the worker's name (`a003` -> `W03`), and
the turn's single user message is, separated by blank lines,

    <world observation: documentation + private probe result>
    <board>                      EMPTY_BOARD, or the board header and one paragraph per post
                                 (TOOL_BOARD_LINE under board_channel: tool)
    [<board rule>]               board_rule (`quote` -> QUOTE_RULE, or any text)
    [<pay paragraph>]            REWARD, when reward_last is set
    QUESTION
    INSTRUCTION                  INSTRUCTION_DISCLOSED when disclose_belief
    FORMAT                       SUBMIT_FORMAT under board_channel: tool

The board is the turn's pushed items in delivery order, so the arm needs
`medium: {delivery: push, push_limit: >= N}`. There is no "Round 1." line, no tool list and no
`Delivered to you:` section.

Board channel `user` (default)
------------------------------
The board is part of the user message. The model is called with no tools and must reply with a
JSON object `{"board_post": ..., "interpretation": "reads_transcript" | "output_only"}` (the
interpretation is matched case-insensitively, with spaces and hyphens read as underscores).

Board channel `tool`
--------------------
The user message says the board is "available through the read_board tool", and the transcript
already holds the worker's `read_board` call (an assistant message with one tool call, id
`read_board_0`) and its result: a `tool` message whose text is exactly the board the `user`
channel would show. The model is offered `read_board` and `submit(board_post, interpretation)`,
and the request forces `submit` (`tool_choice` naming it, through the request's `extra`). Plain
JSON answers after a tool result were unreliable on Qwen3.8-27B: it re-called `read_board`, or
replied empty or in prose without JSON. A reply that is not a usable `submit` call gets a `tool`
result naming the problem (or, for a text reply, a user message) and one more try. Qwen3.8's chat
template renders a tool result as a user-role turn wrapped in `<tool_response>` tags, so for that
model the channel difference is those tags plus the worker's own preceding tool call. Answers
written inside `submit` arguments run long; give this arm `max_tokens: 2048`.

After a usable answer
---------------------
The worker calls `guess(candidate=<interpretation>)`, then `post(text="W03: <board_post>")` (with
`disclose_belief`: `"W03 [committed: <interpretation>]: <board_post>"`), then `end_turn`. After
two unusable answers the worker has no commitment and no post, and the turn notes
`cascade:failed:<reason>`. Long posts can hit `max_tokens`: on Qwen3.8-27B with
`max_tokens: 1024`, about 5% of `user`-channel board turns are truncated once and about 1% twice.

Params, besides every `LLMAgent` param:
- `show_board` (default true): false renders EMPTY_BOARD for every worker (the no-board control).
  Posts are still made and logged.
- `board_rule`: a rule paragraph after the board; `quote` selects QUOTE_RULE.
- `disclose_belief` (default false): attach the commitment to each post.
- `reward_last`: k; adds the pay paragraph naming the last k workers.
- `n_workers` (default 20): only used to name the last k workers.
- `board_channel` (`user` or `tool`).
- `board_header`: `agents`, `humans` (BOARD_HEADERS) or any header text. Default: BOARD_HEADER,
  "posts by earlier workers".

`max_calls` defaults to 2.
"""
from __future__ import annotations

import re
from typing import Any, ClassVar

from ..probes import parse_json_object, strip_reasoning
from ..providers.base import ChatMessage
from ..tools import ToolCall, ToolSchema
from ..view import View
from ..world.cascade import (
    BOARD_HEADER,
    BOARD_HEADERS,
    EMPTY_BOARD,
    FORMAT,
    INSTRUCTION,
    INSTRUCTION_DISCLOSED,
    INTERPRETATIONS,
    QUESTION,
    QUOTE_RULE,
    REWARD,
    SUBMIT_FORMAT,
    SYSTEM,
    TOOL_BOARD_LINE,
    worker_id,
)
from .llm import LLMAgent, LLMTurnUsage

READ_BOARD = ToolSchema(name="read_board", description="Read the shared worker board.",
                        parameters={"type": "object", "properties": {}})
SUBMIT = ToolSchema(
    name="submit", description="Post your message to the shared board and commit to an interpretation.",
    parameters={"type": "object", "required": ["board_post", "interpretation"], "properties": {
        "board_post": {"type": "string", "description": "The message you post to the shared board."},
        "interpretation": {"type": "string", "enum": list(INTERPRETATIONS)}}})
READ_CALL_ID = "read_board_0"
FORCE_SUBMIT = {"tool_choice": {"type": "function", "function": {"name": "submit"}}}


def normalize_interpretation(value: Any) -> str | None:
    if not isinstance(value, str):
        return None
    v = re.sub(r"[\s\-]+", "_", value.strip().strip("`'\"").lower())
    return v if v in INTERPRETATIONS else None


def parse_fields(obj: dict) -> tuple[dict | None, str | None]:
    """(`{"board_post", "interpretation"}`, None) or (None, the reason the answer is unusable)."""
    lower = {str(k).lower(): v for k, v in obj.items()}
    post = lower.get("board_post")
    interp = normalize_interpretation(lower.get("interpretation"))
    if not isinstance(post, str) or not post.strip():
        return None, "no string 'board_post'"
    if interp is None:
        return None, f"'interpretation' must be one of {list(INTERPRETATIONS)}"
    return {"board_post": post.strip(), "interpretation": interp}, None


def parse_reply(text: str) -> tuple[dict | None, str | None]:
    """`parse_fields` of the first JSON object in a text reply."""
    obj = parse_json_object(strip_reasoning(text or ""))
    if obj is None:
        return None, "no JSON object"
    return parse_fields(obj)


class CascadeWorker(LLMAgent):
    entry_point: ClassVar[str | None] = "cascade_worker"

    def __init__(self, model: str, show_board: bool = True, board_rule: str | None = None,
                 disclose_belief: bool = False, reward_last: int | None = None,
                 n_workers: int = 20, board_channel: str = "user", board_header: str | None = None,
                 **kwargs: Any) -> None:
        kwargs.setdefault("max_calls", 2)
        super().__init__(model=model, **kwargs)
        if reward_last is not None and not 1 <= reward_last <= n_workers:
            raise ValueError(f"reward_last must be in [1, n_workers={n_workers}], got {reward_last!r}")
        if board_channel not in ("user", "tool"):
            raise ValueError(f"board_channel must be 'user' or 'tool', got {board_channel!r}")
        self.show_board = bool(show_board)
        self.board_rule = QUOTE_RULE if board_rule == "quote" else board_rule
        self.disclose_belief = bool(disclose_belief)
        self.reward_last = reward_last
        self.n_workers = n_workers
        self.board_channel = board_channel
        self.board_header = BOARD_HEADERS.get(board_header, board_header) if board_header else BOARD_HEADER

    @property
    def calls_per_turn_cap(self) -> int | None:
        return min(self.max_calls or 2, 2)

    def _render_system(self, view: View) -> str:
        return SYSTEM.format(wid=worker_id(self.agent))

    def board_text(self, view: View) -> str:
        """The board as the `user` channel shows it (and the `tool` channel returns it)."""
        posts = [str(it.get("content")) for it in view.pushed] if self.show_board else []
        return self.board_header + "\n\n" + "\n\n".join(posts) if posts else EMPTY_BOARD

    def board_section(self, view: View) -> str:
        return TOOL_BOARD_LINE if self.board_channel == "tool" else self.board_text(view)

    def round_message(self, view: View) -> ChatMessage:  # type: ignore[override]
        obs = "\n\n".join(p.text or "" for p in view.observation.parts if p.type == "text")
        sections = [obs, self.board_section(view)]
        if self.board_rule:
            sections.append(self.board_rule)
        if self.reward_last:
            k, n = self.reward_last, self.n_workers
            sections.append(REWARD.format(k=k, first=worker_id(f"a{n - k}"),
                                          last=worker_id(f"a{n - 1}")))
        sections += [QUESTION, INSTRUCTION_DISCLOSED if self.disclose_belief else INSTRUCTION,
                     SUBMIT_FORMAT if self.board_channel == "tool" else FORMAT]
        return ChatMessage(role="user", content="\n\n".join(sections))

    def turn_message(self, view: View) -> ChatMessage:
        return self.round_message(view)

    async def turn(self, view: View, tools: Any) -> LLMTurnUsage:
        self.system = self._render_system(view)
        self.rounds = [{"round": view.round, "messages": []}]
        self._append(self.turn_message(view))
        offered: list[ToolSchema] = []
        if self.board_channel == "tool":  # pre-filled: the worker has already read the board
            self._append(ChatMessage(role="assistant", content="", tool_calls=[
                ToolCall(call_id=READ_CALL_ID, name="read_board", args={})]))
            self._append(ChatMessage(role="tool", content=self.board_text(view), tool_call_id=READ_CALL_ID))
            offered = [READ_BOARD, SUBMIT]
        fmt = SUBMIT_FORMAT if offered else FORMAT
        usage = LLMTurnUsage()
        executed = 0
        attempts = self.calls_per_turn_cap or 2
        for attempt in range(attempts):
            req = self._request(offered)
            if offered:
                req.extra = {**req.extra, **FORCE_SUBMIT}
            resp = await tools.infer(req)
            usage.finish_reasons.append(resp.finish_reason)
            if resp.finish_reason == "length" and "length" not in usage.notes:
                usage.notes.append("length")
            if resp.tool_calls:
                self._append(ChatMessage(role="assistant", content=resp.text or "", tool_calls=resp.tool_calls))
                sub = next((c for c in resp.tool_calls if c.name == "submit"), None)
                reply, error = (parse_fields(sub.args) if sub is not None and "_raw" not in sub.args
                                else (None, "no valid submit call"))
            else:
                self._append(ChatMessage(role="assistant", content=resp.text or ""))
                reply, error = parse_reply(resp.text or "")
            if reply is not None:
                res = await tools.call("guess", {"candidate": reply["interpretation"]})
                executed += 1
                if res.ok:
                    tag = f" [committed: {reply['interpretation']}]" if self.disclose_belief else ""
                    await tools.call("post", {"text": f"{worker_id(self.agent)}{tag}: {reply['board_post']}"})
                    await tools.call("end_turn", {})
                    usage.calls = executed + 2
                    return usage
                error = f"commit rejected: {res.error}"
            usage.notes.append(f"cascade:{'retry' if attempt + 1 < attempts else 'failed'}:{error}"[:120])
            if resp.tool_calls:  # every tool call needs a result before the next request
                for c in resp.tool_calls:
                    self._append(ChatMessage(role="tool", tool_call_id=c.call_id,
                                             content=f"Your submit call could not be used: {error}. {fmt}"))
            else:
                self._append(ChatMessage(role="user", content=f"Your reply could not be used: {error}. {fmt}"))
        usage.calls = executed
        return usage
