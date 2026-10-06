"""The tool executor for one round (docs/INTERFACE.md §6, §12).

`RoundExecutor` is the only mutation path participants have. The runner creates one per round,
calls `begin_turn(agent)` / `end_turn_event(agent, ...)` around each turn, and after all turns
appends `events(agent)` for each agent in seeded order, then commits (`buffered_posts`,
`buffered_actions`) or, in immediate mode, logs what `committed` already applied.

Decisions where the contract is silent:

- Ids are deterministic: call ids `c{round:04d}-{agent}-{n:03d}` (n counts calls in the turn, from 1),
  world action ids `x{round:04d}-{agent}-{n:02d}` (n counts accepted-for-buffering actions, from 0),
  provisional post ids `tmp-{agent}-{n}` (round_end only; n from 0 within the round). Real post ids
  are assigned by the board when the runner calls `board.buffer_post` at commit, in seeded agent
  order, so they never depend on async interleaving.
- Every call logs `tool_called` then `tool_returned`; `read_board` also logs `read` between them.
  `tool_returned.result` is `{"ok", "result", "error"}` (the `ToolResult` minus `call_id` and
  `pending`, which has its own field).
- Cap: the call that makes `calls > max_calls_per_turn` is logged with
  `ok=False, error="cap"` and raises `TurnCapReached`. The cap is checked first, so a participant
  that keeps calling after `end_turn` still hits it.
- `end_turn` never raises: it sets a per-agent flag (`turn_ended(agent)`) and returns `ok=True`.
  Every later call in the same turn is logged and returns `ok=False, error="turn_ended"` without
  side effects. `EndTurn` stays defined in `tools.py` but is not raised.
- `end_turn` is always allowed. The optional per-agent allowlist filters every other tool
  (schemas and calls); a call outside it returns `ok=False, error="not_allowed"`. Unknown tools
  return the same error.
- `read_board(channel=None, limit=50)` returns `{"items": [{delivery_id, post_id, eligible_round,
  content}]}` with verbatim delivered content. Unknown channel or bad args: `ok=False`.
- `post(channel="main", text, fields={})`: `text` must be a string, `channel` a board channel.
- World actions are validated with `world.validate` at call time against current (round-start in
  round_end mode) state; a failed `Ack` returns `ok=False, error=<ack.error>` and nothing is buffered.
  In immediate mode the action is committed at once and the result is
  `{"id", "accepted", "feedback"}` with `pending=False`.
- Status tools answer from the world's current state, which under round_end is round-start state
  because nothing commits during the turns.
- Concurrency: `call` contains no `await`, so under asyncio each call is atomic. During round_end
  turns it only reads shared world state and touches the calling agent's own board inbox
  (`board.read`), so concurrent turns cannot observe each other.
"""
from __future__ import annotations

import random
from collections.abc import Callable
from dataclasses import dataclass, field
from typing import Any, Literal

from .events import (
    Event,
    ReadEvent,
    ToolCalledEvent,
    ToolReturnedEvent,
    TurnEndedEvent,
    TurnStartedEvent,
)
from .ids import ActionId, AgentId, CallId
from .medium.board import Board, Delivery, Post
from .tools import ToolResult, ToolSchema, TurnCapReached
from .world.base import Action, Outcome, World

BOARD_TOOLS = ("read_board", "post")
STATUS_TOOLS = ("my_status", "collective_status")

END_TURN_SCHEMA = ToolSchema(
    name="end_turn",
    description="Finish your turn for this round.",
    parameters={"type": "object", "properties": {}, "additionalProperties": False},
)


def board_schemas(board: Board) -> list[ToolSchema]:
    return [
        ToolSchema(
            name="read_board",
            description="Read unread messages delivered to your inbox (oldest first).",
            parameters={
                "type": "object",
                "properties": {
                    "channel": {"type": "string", "enum": list(board.channels)},
                    "limit": {"type": "integer"},
                },
                "additionalProperties": False,
            },
        ),
        ToolSchema(
            name="post",
            description="Post a message to a board channel.",
            parameters={
                "type": "object",
                "properties": {
                    "channel": {"type": "string", "enum": list(board.channels)},
                    "text": {"type": "string"},
                    "fields": {"type": "object"},
                },
                "required": ["text"],
                "additionalProperties": False,
            },
        ),
    ]


@dataclass
class _AgentState:
    events: list[Event] = field(default_factory=list)
    actions: list[tuple[AgentId, ActionId, Action]] = field(default_factory=list)
    posts: list[tuple[str, str, dict]] = field(default_factory=list)  # (channel, text, fields)
    calls: int = 0
    n_actions: int = 0
    ended: bool = False


@dataclass
class Committed:
    """What immediate mode applied during the round, in application order."""

    posts: list[Post] = field(default_factory=list)
    deliveries: list[Delivery] = field(default_factory=list)
    actions: list[tuple[AgentId, ActionId, Action, Outcome]] = field(default_factory=list)


class RoundExecutor:
    def __init__(
        self,
        *,
        run: str,
        round: int,
        world: World,
        board: Board,
        blobs: Any,
        agents: list[AgentId],
        commit: Literal["round_end", "immediate"] = "round_end",
        max_calls_per_turn: int = 20,
        topology_rng: Callable[[], random.Random] | None = None,
        allowlist: dict[AgentId, set[str]] | None = None,
    ) -> None:
        self.run = run
        self.round = round
        self.world = world
        self.board = board
        self.blobs = blobs
        self.agents = list(agents)
        self.commit_mode = commit
        self.max_calls = max_calls_per_turn
        self._topology_rng = topology_rng or (lambda: random.Random(0))
        self.allowlist = allowlist
        self._state: dict[AgentId, _AgentState] = {}
        self.committed = Committed()
        self._world_tools = {s.name: s for s in world.tool_schemas()}

    # ---- per-agent bookkeeping ---------------------------------------------------------------
    def _st(self, agent: AgentId) -> _AgentState:
        st = self._state.get(agent)
        if st is None:
            st = self._state[agent] = _AgentState()
        return st

    def _ev(self, cls: type[Event], agent: AgentId, **kw: Any) -> Event:
        ev = cls(run=self.run, round=self.round, agent=agent, **kw)
        self._st(agent).events.append(ev)
        return ev

    def begin_turn(self, agent: AgentId, private: dict | None = None) -> None:
        self._ev(TurnStartedEvent, agent, private=dict(private or {}))

    def end_turn_event(
        self, agent: AgentId, yield_kind: str, usage: dict | None = None, error: str | None = None
    ) -> None:
        st = self._st(agent)
        self._ev(TurnEndedEvent, agent, yield_kind=yield_kind, calls=st.calls,
                 usage=dict(usage or {}), error=error)

    def events(self, agent: AgentId) -> list[Event]:
        return list(self._st(agent).events)

    def turn_ended(self, agent: AgentId) -> bool:
        """True once `agent` has called `end_turn` this round."""
        return self._st(agent).ended

    def calls(self, agent: AgentId) -> int:
        return self._st(agent).calls

    def buffered_posts(self, agent: AgentId) -> list[tuple[str, str, dict]]:
        return list(self._st(agent).posts)

    def buffered_actions(self, agent: AgentId) -> list[tuple[AgentId, ActionId, Action]]:
        return list(self._st(agent).actions)

    # ---- ToolExecutor ------------------------------------------------------------------------
    def _allowed(self, agent: AgentId, name: str) -> bool:
        if name == "end_turn":
            return True
        if self.allowlist is None or agent not in self.allowlist:
            return True
        return name in self.allowlist[agent]

    def schemas(self, agent: AgentId) -> list[ToolSchema]:
        out = [*self._world_tools.values(), *board_schemas(self.board), END_TURN_SCHEMA]
        return [s for s in out if self._allowed(agent, s.name)]

    async def call(self, agent: AgentId, name: str, args: dict | None = None) -> ToolResult:
        st = self._st(agent)
        args = dict(args or {})
        st.calls += 1
        call_id = CallId(f"c{self.round:04d}-{agent}-{st.calls:03d}")
        self._ev(ToolCalledEvent, agent, call_id=call_id, tool=name, args=args)
        if st.calls > self.max_calls:
            self._ret(agent, ToolResult(call_id=call_id, ok=False, result={}, error="cap"))
            raise TurnCapReached()
        if st.ended:
            return self._ret(agent, self._err(call_id, "turn_ended"))
        if name == "end_turn":
            st.ended = True
            return self._ret(agent, ToolResult(call_id=call_id, ok=True, result={}))
        if not self._allowed(agent, name):
            return self._ret(agent, self._err(call_id, "not_allowed"))
        if name == "read_board":
            res = self._read_board(agent, call_id, args)
        elif name == "post":
            res = self._post(agent, call_id, args)
        elif name in STATUS_TOOLS and name in self._world_tools:
            res = self._status(agent, call_id, name, args)
        elif name in self._world_tools:
            res = self._world_action(agent, call_id, name, args)
        else:
            res = self._err(call_id, "not_allowed")
        return self._ret(agent, res)

    # ---- helpers -----------------------------------------------------------------------------
    @staticmethod
    def _err(call_id: CallId, error: str) -> ToolResult:
        return ToolResult(call_id=call_id, ok=False, result={}, error=error)

    def _ret(self, agent: AgentId, res: ToolResult) -> ToolResult:
        self._ev(ToolReturnedEvent, agent, call_id=res.call_id,
                 result=res.model_dump(mode="json", exclude={"call_id", "pending"}),
                 pending=res.pending)
        return res

    def _read_board(self, agent: AgentId, call_id: CallId, args: dict) -> ToolResult:
        extra = set(args) - {"channel", "limit"}
        if extra:
            return self._err(call_id, f"bad args: extra={sorted(extra)}")
        channel = args.get("channel")
        limit = args.get("limit", 50)
        if not isinstance(limit, int) or isinstance(limit, bool):
            return self._err(call_id, "bad args: limit must be an integer")
        try:
            items = self.board.read(agent, self.round, channel, limit)
        except ValueError as e:
            return self._err(call_id, str(e))
        self._ev(ReadEvent, agent, delivery_ids=[d.delivery_id for d in items])
        return ToolResult(call_id=call_id, ok=True, result={"items": [
            {"delivery_id": d.delivery_id, "post_id": d.post_id,
             "eligible_round": d.eligible_round, "content": self.board.content(d, self.blobs)}
            for d in items
        ]})

    def _post(self, agent: AgentId, call_id: CallId, args: dict) -> ToolResult:
        extra = set(args) - {"channel", "text", "fields"}
        channel = args.get("channel", "main")
        text = args.get("text")
        fields = args.get("fields") or {}
        if extra or not isinstance(text, str) or not isinstance(fields, dict):
            return self._err(call_id, "bad args: post(channel?, text: str, fields?: object)")
        if channel not in self.board.channels:
            return self._err(call_id, f"unknown channel {channel!r}")
        st = self._st(agent)
        if self.commit_mode == "round_end":
            provisional = f"tmp-{agent}-{len(st.posts)}"
            st.posts.append((channel, text, dict(fields)))
            return ToolResult(call_id=call_id, ok=True, result={"id": provisional}, pending=True)
        post_id = self.board.buffer_post(agent, self.round, channel, text, dict(fields))
        posts, deliveries = self.board.commit(self.round, self.agents, self._topology_rng(), self.blobs)
        self.committed.posts.extend(posts)
        self.committed.deliveries.extend(deliveries)
        return ToolResult(call_id=call_id, ok=True, result={"id": post_id}, pending=False)

    def _status(self, agent: AgentId, call_id: CallId, name: str, args: dict) -> ToolResult:
        if args:
            return self._err(call_id, "bad args: takes no arguments")
        value = self.world.my_status(agent) if name == "my_status" else self.world.collective_status()
        if value is None:
            return self._err(call_id, "not_allowed")
        return ToolResult(call_id=call_id, ok=True, result=dict(value))

    def _world_action(self, agent: AgentId, call_id: CallId, name: str, args: dict) -> ToolResult:
        action = Action(name=name, args=args)
        ack = self.world.validate(agent, action)
        if not ack.ok:
            return self._err(call_id, ack.error or "rejected")
        st = self._st(agent)
        action_id = ActionId(f"x{self.round:04d}-{agent}-{st.n_actions:02d}")
        st.n_actions += 1
        if self.commit_mode == "round_end":
            st.actions.append((agent, action_id, action))
            return ToolResult(call_id=call_id, ok=True, result={"id": action_id}, pending=True)
        outcome = self.world.commit([(agent, action_id, action)])[0]
        outcome.action_id = action_id
        self.committed.actions.append((agent, action_id, action, outcome))
        return ToolResult(call_id=call_id, ok=True, pending=False, result={
            "id": action_id, "accepted": outcome.accepted, "feedback": dict(outcome.feedback)})
