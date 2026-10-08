"""CascadeWorker: the one-shot worker of the `cascade` world (swarmlab/world/cascade.py).

An `LLMAgent` subclass (entry point `cascade_worker`) that sends the source's prompt instead of
the default framing: the system message is `SYSTEM` with the worker's name (`a003` -> `W03`), and
the turn's single user message is, separated by blank lines,

    <world observation: documentation + private probe result>
    <board>                      EMPTY_BOARD, or BOARD_HEADER and one paragraph per post
    [<board rule>]               board_rule (`quote` -> QUOTE_RULE, or any text)
    [<pay paragraph>]            REWARD, when reward_last is set
    QUESTION
    INSTRUCTION                  INSTRUCTION_DISCLOSED when disclose_belief
    FORMAT

The board is the turn's pushed items in delivery order, so the arm needs
`medium: {delivery: push, push_limit: >= N}`. There is no "Round 1." line, no tool list and no
`Delivered to you:` section. The model is called with no tools and must reply with a JSON object
`{"board_post": ..., "interpretation": "reads_transcript" | "output_only"}` (the interpretation is
matched case-insensitively, with spaces and hyphens read as underscores). On a usable reply the
worker calls `guess(candidate=<interpretation>)`, then `post(text="W03: <board_post>")` (with
`disclose_belief`: `"W03 [committed: <interpretation>]: <board_post>"`), then `end_turn`. On an
unusable reply (no JSON object, missing field, or truncated) it appends the error and FORMAT and
asks once more; after two failures the worker has no commitment and no post, and the turn notes
`cascade:failed:<reason>`. Long posts can hit `max_tokens`: on Qwen3.8-27B with
`max_tokens: 1024`, about 5% of board turns are truncated once and about 1% twice.

Params, besides every `LLMAgent` param: `show_board` (default true; false renders EMPTY_BOARD for
every worker, the no-board control; posts are still made and logged), `board_rule`,
`disclose_belief` (default false), `reward_last` (k; pay paragraph naming the last k workers) and
`n_workers` (default 20; only used to name the last k workers). `max_calls` defaults to 2.
"""
from __future__ import annotations

import re
from typing import Any, ClassVar

from ..probes import parse_json_object, strip_reasoning
from ..providers.base import ChatMessage
from ..view import View
from ..world.cascade import (
    BOARD_HEADER,
    EMPTY_BOARD,
    FORMAT,
    INSTRUCTION,
    INSTRUCTION_DISCLOSED,
    INTERPRETATIONS,
    QUESTION,
    QUOTE_RULE,
    REWARD,
    SYSTEM,
    worker_id,
)
from .llm import LLMAgent, LLMTurnUsage


def normalize_interpretation(value: Any) -> str | None:
    if not isinstance(value, str):
        return None
    v = re.sub(r"[\s\-]+", "_", value.strip().strip("`'\"").lower())
    return v if v in INTERPRETATIONS else None


def parse_reply(text: str) -> tuple[dict | None, str | None]:
    """(`{"board_post", "interpretation"}`, None) or (None, the reason the reply is unusable)."""
    obj = parse_json_object(strip_reasoning(text or ""))
    if obj is None:
        return None, "no JSON object"
    lower = {str(k).lower(): v for k, v in obj.items()}
    post = lower.get("board_post")
    interp = normalize_interpretation(lower.get("interpretation"))
    if not isinstance(post, str) or not post.strip():
        return None, "no string 'board_post'"
    if interp is None:
        return None, f"'interpretation' must be one of {list(INTERPRETATIONS)}"
    return {"board_post": post.strip(), "interpretation": interp}, None


class CascadeWorker(LLMAgent):
    entry_point: ClassVar[str | None] = "cascade_worker"

    def __init__(self, model: str, show_board: bool = True, board_rule: str | None = None,
                 disclose_belief: bool = False, reward_last: int | None = None,
                 n_workers: int = 20, **kwargs: Any) -> None:
        kwargs.setdefault("max_calls", 2)
        super().__init__(model=model, **kwargs)
        if reward_last is not None and not 1 <= reward_last <= n_workers:
            raise ValueError(f"reward_last must be in [1, n_workers={n_workers}], got {reward_last!r}")
        self.show_board = bool(show_board)
        self.board_rule = QUOTE_RULE if board_rule == "quote" else board_rule
        self.disclose_belief = bool(disclose_belief)
        self.reward_last = reward_last
        self.n_workers = n_workers

    @property
    def calls_per_turn_cap(self) -> int | None:
        return min(self.max_calls or 2, 2)

    def _render_system(self, view: View) -> str:
        return SYSTEM.format(wid=worker_id(self.agent))

    def board_section(self, view: View) -> str:
        posts = [str(it.get("content")) for it in view.pushed] if self.show_board else []
        return BOARD_HEADER + "\n\n" + "\n\n".join(posts) if posts else EMPTY_BOARD

    def round_message(self, view: View) -> ChatMessage:  # type: ignore[override]
        obs = "\n\n".join(p.text or "" for p in view.observation.parts if p.type == "text")
        sections = [obs, self.board_section(view)]
        if self.board_rule:
            sections.append(self.board_rule)
        if self.reward_last:
            k, n = self.reward_last, self.n_workers
            sections.append(REWARD.format(k=k, first=worker_id(f"a{n - k}"),
                                          last=worker_id(f"a{n - 1}")))
        sections += [QUESTION, INSTRUCTION_DISCLOSED if self.disclose_belief else INSTRUCTION, FORMAT]
        return ChatMessage(role="user", content="\n\n".join(sections))

    def turn_message(self, view: View) -> ChatMessage:
        return self.round_message(view)

    async def turn(self, view: View, tools: Any) -> LLMTurnUsage:
        self.system = self._render_system(view)
        self.rounds = [{"round": view.round, "messages": []}]
        self._append(self.turn_message(view))
        usage = LLMTurnUsage()
        executed = 0
        attempts = self.calls_per_turn_cap or 2
        for attempt in range(attempts):
            resp = await tools.infer(self._request([]))
            usage.finish_reasons.append(resp.finish_reason)
            if resp.finish_reason == "length" and "length" not in usage.notes:
                usage.notes.append("length")
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
            self._append(ChatMessage(role="user", content=f"Your reply could not be used: {error}. {FORMAT}"))
        usage.calls = executed
        return usage
