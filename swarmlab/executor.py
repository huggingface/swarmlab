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
  order, so they never depend on async interleaving. The runner records the provisional id on
  the `post` event (`provisional_id`); under immediate commit it equals `post_id`.
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
  `fields` is accepted (scripted participants use it) but not advertised in the `post` schema: a
  free-form object cannot be expressed under Anthropic strict tool use, and an advertised
  `{"type": "object"}` made Haiku reject every request (smoke 2026-10-06). An arm that wants
  typed fields should advertise them with explicit properties.
- Schemas: every schema `schemas()` returns is `ToolSchema.normalized()` (`properties`,
  `required`, `additionalProperties: false` at every object level), so it passes
  `tools.strict_violations` and providers can send it strict.
- World actions are validated with `world.validate` at call time against current (round-start in
  round_end mode) state; a failed `Ack` returns `ok=False, error=<ack.error>` and nothing is buffered.
  `validate` also gets `pending=`, the agent's world actions already buffered this round
  (`pending_actions(agent)`, always empty under immediate commit), so a world can refuse at call
  time what its commit would certainly reject (ColoringGrid `paints_per_round`, FlagGame
  `guess_limit`). `my_status` gets the same `pending=` so it can count them (ColoringGrid
  `paints_left_this_round`). Both are passed only when the world's override accepts a `pending`
  keyword (checked once per executor with `inspect.signature`), so worlds written against
  `validate(agent, action)` / `my_status(agent)` are unaffected.
- `rejected_calls(agent) -> (rejected, answered)` counts this turn's `ok=False` returns; the
  runner uses it to add a `cap:<rejected>` note when a turn that ended by the cap was mostly
  rejected calls.
  In immediate mode the action is committed at once and the result is
  `{"id", "accepted", "feedback"}` with `pending=False`.
- Status tools answer from the world's current state, which under round_end is round-start state
  because nothing commits during the turns.
- Inference (M1b, `infer(agent, request, category)`): call ids `i{round:04d}-{agent}-{n:03d}`
  (n counts this agent's inference calls in the round, from 1); inference calls do not count
  toward `max_calls_per_turn`. The work is done by `swarmlab.inference.Inference` (cache, gate,
  operational events). Per agent the executor sums the *nominal* usage and cost of its "swarm"
  calls (`inference_usage(agent)`), which the runner merges into `turn_ended.usage`. A
  `HardCeilingReached` sets `hard_ceiling` on the executor before propagating, so the runner aborts
  the round even if a participant swallows the exception. Without an `Inference` (unit tests
  that build a bare executor), `infer` raises `RuntimeError`.
- Registry (M3b, swarmlab/medium/registry.py): with a `registry`, the schemas add
  `registry_get(key)`, `registry_put(key, value)`, `registry_cas(key, expected_version, value)`,
  `registry_acquire(key, ttl_rounds)`, `registry_release(key)`. Args are checked at call time
  (`ok=False, error="bad args: ..."`, nothing buffered). Under round_end a write is buffered per
  agent (`buffered_registry(agent)`) and returns `pending=True` with `{"id": op_id}`; `get`
  answers from round-start state plus the caller's own buffered writes. Under immediate commit a
  write applies at once (`committed.registry`) and returns its outcome with `pending=False`, and
  a world action passes the claim check (`commit_world`) before the world commits it
  (`committed.claims`).
- Roles (M3c, docs/INTERFACE-M3c.md §1, swarmlab/roles.py): `roles` maps an agent to its
  *effective* `Role` (the runner's `roles.bind_roles` has already intersected its channel lists
  with the topology's). Checked after the cap and `turn_ended`, a call is `not_allowed` when the
  tool is outside `role.tools`, is a world action under `may_act=False` (status tools are not
  actions), is a registry tool under `registry="none"` or a registry write under `"read"`, is
  `read_board` of a channel outside `channels_read`, or is `post` to a channel outside
  `channels_write` or with `fields` outside `post_fields`. Like every call it is logged
  (`tool_called`/`tool_returned` with `error="not_allowed"`) and has no side effect. Schemas
  follow the same rules: hidden tools are not offered, the `channel` enums list only the
  role's channels (`read_board` / `post` are hidden when that list is empty), and `post`
  advertises `fields` (optional, with enums) when the role has `post_fields`.
  `read_board()` without a channel returns only readable channels' items (oldest first, the
  limit applied after filtering); items on unreadable channels stay unread. `post` without a
  channel goes to `board.topology.default_channel(agent)` when the topology has one (Tree),
  else to `main`. A role `budget.max_calls` replaces `max_calls_per_turn` for that agent.
  `turn_started.role` is the role's name (None for an agent without one). `pushable(agent,
  limit)` is the push-delivery selection with the same channel filter.
- Concurrency: `call` contains no `await`, so under asyncio each call is atomic. During round_end
  turns it only reads shared world state and touches the calling agent's own board inbox
  (`board.read`), so concurrent turns cannot observe each other.
"""
from __future__ import annotations

import inspect
import random
from collections.abc import Callable
from dataclasses import dataclass, field
from typing import Any, Literal

from .budget import HardCeilingReached
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
from .medium.registry import (
    TOOL_OPS,
    ClaimPolicy,
    Registry,
    RegistryOp,
    RegistryOutcome,
    check_op_args,
    commit_world,
)
from .providers.base import ChatRequest, ChatResponse, Usage
from .roles import Role
from .tools import ToolResult, ToolSchema, TurnCapReached
from .world.base import Action, Outcome, World

BOARD_TOOLS = ("read_board", "post")
STATUS_TOOLS = ("my_status", "collective_status")
REGISTRY_TOOLS = ("registry_get", *TOOL_OPS)


def _takes_pending(fn: Callable) -> bool:
    """True when `fn` accepts a `pending` keyword (World.validate / my_status, see base.py)."""
    try:
        params = inspect.signature(fn).parameters
    except (TypeError, ValueError):
        return False
    return "pending" in params or any(p.kind is p.VAR_KEYWORD for p in params.values())


def _obj(props: dict, required: list[str]) -> dict:
    return {"type": "object", "properties": props, "required": required, "additionalProperties": False}


def registry_schemas() -> list[ToolSchema]:
    key = {"type": "string", "description": "Registry key, e.g. cell:3,4"}
    value = {"type": "string"}
    return [
        ToolSchema(name="registry_get", description="Read a registry entry: value, version, owner "
                   "(the agent holding a live claim) and expires_round.", parameters=_obj({"key": key}, ["key"])),
        ToolSchema(name="registry_put", description="Set a registry value (applied at the end of the round).",
                   parameters=_obj({"key": key, "value": value}, ["key", "value"])),
        ToolSchema(name="registry_cas", description="Set a registry value only if the entry's version "
                   "still equals expected_version at the end of the round (0 for a missing key).",
                   parameters=_obj({"key": key, "expected_version": {"type": "integer"}, "value": value},
                                   ["key", "expected_version", "value"])),
        ToolSchema(name="registry_acquire", description="Claim a key for ttl_rounds rounds (this round "
                   "counts as the first). Succeeds at the end of the round only if nobody else holds a "
                   "live claim on it; earlier-scheduled agents win ties. The result arrives next round.",
                   parameters=_obj({"key": key, "ttl_rounds": {"type": "integer"}}, ["key", "ttl_rounds"])),
        ToolSchema(name="registry_release", description="Release your claim on a key.",
                   parameters=_obj({"key": key}, ["key"])),
    ]

END_TURN_SCHEMA = ToolSchema(
    name="end_turn",
    description="Finish your turn for this round.",
    parameters={"type": "object", "properties": {}, "required": [], "additionalProperties": False},
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
                "required": [],
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
    post_ids: list[str] = field(default_factory=list)  # provisional ids, parallel to `posts`
    registry: list[RegistryOp] = field(default_factory=list)  # buffered registry writes (M3b)
    calls: int = 0
    n_actions: int = 0
    ended: bool = False
    n_infer: int = 0
    infer_swarm: int = 0
    usage: Usage = field(default_factory=Usage)
    cost_usd: float = 0.0


@dataclass
class Committed:
    """What immediate mode applied during the round, in application order."""

    posts: list[Post] = field(default_factory=list)
    deliveries: list[Delivery] = field(default_factory=list)
    actions: list[tuple[AgentId, ActionId, Action, Outcome]] = field(default_factory=list)
    registry: list[RegistryOutcome] = field(default_factory=list)  # M3b
    claims: list[dict] = field(default_factory=list)  # M3b: commit_world claim records


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
        inference: Any = None,
        registry: Registry | None = None,
        claim_policy: ClaimPolicy | None = None,
        roles: dict[AgentId, Role] | None = None,
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
        self.inference = inference
        self.registry = registry
        self.claim_policy = claim_policy
        self.roles = dict(roles or {})  # M3c: effective role per agent
        self.hard_ceiling = False
        self._state: dict[AgentId, _AgentState] = {}
        self.committed = Committed()
        self._world_tools = {s.name: s for s in world.tool_schemas()}
        self._validate_pending = _takes_pending(world.validate)
        self._status_pending = _takes_pending(world.my_status)

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
        role = self.roles.get(agent)
        self._ev(TurnStartedEvent, agent, private=dict(private or {}),
                 role=role.name if role is not None and role.name else None)

    def end_turn_event(
        self, agent: AgentId, yield_kind: str, usage: dict | None = None, error: str | None = None
    ) -> None:
        st = self._st(agent)
        self._ev(TurnEndedEvent, agent, yield_kind=yield_kind, calls=st.calls,
                 usage=dict(usage or {}), error=error)

    def note(self, cls: type[Event], agent: AgentId, **kw: Any) -> Event:
        """Buffer a participant-reported event (M3a: `overflow`) with the agent's turn events."""
        return self._ev(cls, agent, **kw)

    def events(self, agent: AgentId) -> list[Event]:
        return list(self._st(agent).events)

    def turn_ended(self, agent: AgentId) -> bool:
        """True once `agent` has called `end_turn` this round."""
        return self._st(agent).ended

    def calls(self, agent: AgentId) -> int:
        return self._st(agent).calls

    def buffered_posts(self, agent: AgentId) -> list[tuple[str, str, dict]]:
        return list(self._st(agent).posts)

    def buffered_post_ids(self, agent: AgentId) -> list[str]:
        """Provisional ids returned to the agent, parallel to `buffered_posts(agent)`."""
        return list(self._st(agent).post_ids)

    def buffered_registry(self, agent: AgentId) -> list[RegistryOp]:
        return list(self._st(agent).registry)

    def buffered_actions(self, agent: AgentId) -> list[tuple[AgentId, ActionId, Action]]:
        return list(self._st(agent).actions)

    def pending_actions(self, agent: AgentId) -> list[Action]:
        """The agent's world actions buffered this round, in call order (empty under immediate)."""
        return [a for _, _, a in self._st(agent).actions]

    def rejected_calls(self, agent: AgentId) -> tuple[int, int]:
        """(rejected, answered) tool calls of `agent` this round: `tool_returned` events with
        `ok=False`, and all `tool_returned` events, both without the call that hit the cap."""
        rets = [e for e in self._st(agent).events if isinstance(e, ToolReturnedEvent)
                and e.result.get("error") != "cap"]
        return sum(1 for e in rets if not e.result.get("ok")), len(rets)

    # ---- ToolExecutor ------------------------------------------------------------------------
    def _allowed(self, agent: AgentId, name: str) -> bool:
        if name == "end_turn":
            return True
        if self.allowlist is not None and agent in self.allowlist and name not in self.allowlist[agent]:
            return False
        return self._role_allows(agent, name)

    def _role_allows(self, agent: AgentId, name: str) -> bool:
        """M3c: the agent's role permits calling `name` at all (channel checks come later)."""
        role = self.roles.get(agent)
        if role is None:
            return True
        if role.tools is not None and name not in role.tools:
            return False
        if name in REGISTRY_TOOLS:
            return role.registry == "write" or (role.registry == "read" and name == "registry_get")
        if name == "read_board":
            return bool(self._channels(agent, "read"))
        if name == "post":
            return bool(self._channels(agent, "write"))
        if name in self._world_tools and name not in STATUS_TOOLS:
            return role.may_act
        return True

    def _channels(self, agent: AgentId, mode: str) -> list[str]:
        """Board channels the agent may read or write."""
        role = self.roles.get(agent)
        if role is None:
            return list(self.board.channels)
        check = role.can_read if mode == "read" else role.can_write
        return [c for c in self.board.channels if check(c)]

    def _max_calls(self, agent: AgentId) -> int:
        role = self.roles.get(agent)
        if role is not None and role.budget and "max_calls" in role.budget:
            return int(role.budget["max_calls"])
        return self.max_calls

    def _board_schemas(self, agent: AgentId) -> list[ToolSchema]:
        out = board_schemas(self.board)
        role = self.roles.get(agent)
        if role is None:
            return out
        read, write = self._channels(agent, "read"), self._channels(agent, "write")
        for s in out:
            props = s.parameters["properties"]
            props["channel"] = {**props["channel"], "enum": read if s.name == "read_board" else write}
            if s.name == "post" and role.post_fields:
                props["fields"] = _obj({k: {"type": "string", "enum": list(v)}
                                        for k, v in role.post_fields.items()}, [])
        return out

    def schemas(self, agent: AgentId) -> list[ToolSchema]:
        reg = registry_schemas() if self.registry is not None else []
        out = [*self._world_tools.values(), *self._board_schemas(agent), *reg, END_TURN_SCHEMA]
        return [s.normalized() for s in out if self._allowed(agent, s.name)]

    def pushable(self, agent: AgentId, limit: int) -> list[Delivery]:
        """Push-delivery items for `agent` this round: `board.pushable` minus unreadable channels."""
        if agent not in self.roles:
            return self.board.pushable(agent, self.round, limit)
        readable = set(self._channels(agent, "read"))
        items = self.board.pushable(agent, self.round, 1 << 30)
        return [d for d in items if self.board.channel_of(d) in readable][: max(limit, 0)]

    async def call(self, agent: AgentId, name: str, args: dict | None = None) -> ToolResult:
        st = self._st(agent)
        args = dict(args or {})
        st.calls += 1
        call_id = CallId(f"c{self.round:04d}-{agent}-{st.calls:03d}")
        self._ev(ToolCalledEvent, agent, call_id=call_id, tool=name, args=args)
        if st.calls > self._max_calls(agent):
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
        elif name in REGISTRY_TOOLS and self.registry is not None:
            res = self._registry(agent, call_id, name, args)
        elif name in STATUS_TOOLS and name in self._world_tools:
            res = self._status(agent, call_id, name, args)
        elif name in self._world_tools:
            res = self._world_action(agent, call_id, name, args)
        else:
            res = self._err(call_id, "not_allowed")
        return self._ret(agent, res)

    async def infer(self, agent: AgentId, request: ChatRequest, category: str = "swarm") -> ChatResponse:
        if self.inference is None:
            raise RuntimeError("inference is not available in this executor")
        st = self._st(agent)
        st.n_infer += 1
        call_id = f"i{self.round:04d}-{agent}-{st.n_infer:03d}"
        try:
            resp, nominal = await self.inference.infer(agent=agent, round=self.round, call_id=call_id,
                                                       request=request, category=category)
        except HardCeilingReached:
            self.hard_ceiling = True
            raise
        if category == "swarm":
            st.infer_swarm += 1
            st.usage = st.usage + nominal.usage
            st.cost_usd += nominal.cost_usd
        return resp

    def inference_usage(self, agent: AgentId) -> dict | None:
        """Summed nominal usage of this agent's swarm inference calls this round, or None."""
        st = self._st(agent)
        if not st.infer_swarm:
            return None
        return {**st.usage.model_dump(), "cost_usd": st.cost_usd, "inference_calls": st.infer_swarm}

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
        if agent in self.roles:  # M3c: channel permissions
            if channel is not None and channel in self.board.channels and \
                    channel not in self._channels(agent, "read"):
                return self._err(call_id, "not_allowed")
            if channel is None:
                return self._read_readable(agent, call_id, limit)
        try:
            items = self.board.read(agent, self.round, channel, limit)
        except ValueError as e:
            return self._err(call_id, str(e))
        return self._read_result(agent, call_id, items)

    def _read_readable(self, agent: AgentId, call_id: CallId, limit: int) -> ToolResult:
        """`read_board()` under a role: the oldest `limit` unread items on readable channels."""
        readable = self._channels(agent, "read")
        chosen = [d for d in self.board.pushable(agent, self.round, 1 << 30)
                  if self.board.channel_of(d) in readable][: max(limit, 0)]
        counts: dict[str, int] = {}
        for d in chosen:
            ch = self.board.channel_of(d)
            counts[ch] = counts.get(ch, 0) + 1
        # per channel, board.read returns that channel's oldest items: exactly the chosen ones
        items = [d for ch, n in counts.items() for d in self.board.read(agent, self.round, ch, n)]
        items.sort(key=lambda d: (d.eligible_round, d.delivery_id))
        return self._read_result(agent, call_id, items)

    def _read_result(self, agent: AgentId, call_id: CallId, items: list[Delivery]) -> ToolResult:
        self._ev(ReadEvent, agent, delivery_ids=[d.delivery_id for d in items])
        return ToolResult(call_id=call_id, ok=True, result={"items": [
            {"delivery_id": d.delivery_id, "post_id": d.post_id,
             "eligible_round": d.eligible_round, "content": self.board.content(d, self.blobs)}
            for d in items
        ]})

    def _post(self, agent: AgentId, call_id: CallId, args: dict) -> ToolResult:
        extra = set(args) - {"channel", "text", "fields"}
        channel = args.get("channel")
        if channel is None:
            default = getattr(self.board.topology, "default_channel", None)
            channel = (default(agent) if callable(default) else None) or "main"
        text = args.get("text")
        fields = args.get("fields") or {}
        if extra or not isinstance(text, str) or not isinstance(fields, dict):
            return self._err(call_id, "bad args: post(channel?, text: str, fields?: object)")
        if channel not in self.board.channels:
            return self._err(call_id, f"unknown channel {channel!r}")
        role = self.roles.get(agent)
        if role is not None:  # M3c: channel and typed-field permissions
            if channel not in self._channels(agent, "write"):
                return self._err(call_id, "not_allowed")
            if role.post_fields is not None and any(
                    k not in role.post_fields or v not in role.post_fields[k] for k, v in fields.items()):
                return self._err(call_id, "not_allowed")
        st = self._st(agent)
        if self.commit_mode == "round_end":
            provisional = f"tmp-{agent}-{len(st.posts)}"
            st.posts.append((channel, text, dict(fields)))
            st.post_ids.append(provisional)
            return ToolResult(call_id=call_id, ok=True, result={"id": provisional}, pending=True)
        post_id = self.board.buffer_post(agent, self.round, channel, text, dict(fields))
        posts, deliveries = self.board.commit(self.round, self.agents, self._topology_rng(), self.blobs)
        self.committed.posts.extend(posts)
        self.committed.deliveries.extend(deliveries)
        return ToolResult(call_id=call_id, ok=True, result={"id": post_id}, pending=False)

    def _status(self, agent: AgentId, call_id: CallId, name: str, args: dict) -> ToolResult:
        if args:
            return self._err(call_id, "bad args: takes no arguments")
        if name == "my_status":
            value = (self.world.my_status(agent, pending=self.pending_actions(agent))
                     if self._status_pending else self.world.my_status(agent))
        else:
            value = self.world.collective_status()
        if value is None:
            return self._err(call_id, "not_allowed")
        return ToolResult(call_id=call_id, ok=True, result=dict(value))

    def _world_action(self, agent: AgentId, call_id: CallId, name: str, args: dict) -> ToolResult:
        action = Action(name=name, args=args)
        ack = (self.world.validate(agent, action, pending=self.pending_actions(agent))
               if self._validate_pending else self.world.validate(agent, action))
        if not ack.ok:
            return self._err(call_id, ack.error or "rejected")
        st = self._st(agent)
        action_id = ActionId(f"x{self.round:04d}-{agent}-{st.n_actions:02d}")
        st.n_actions += 1
        if self.commit_mode == "round_end":
            st.actions.append((agent, action_id, action))
            return ToolResult(call_id=call_id, ok=True, result={"id": action_id}, pending=True)
        outs, claims = commit_world(self.world, [(agent, action_id, action)], registry=self.registry,
                                    policy=self.claim_policy, round=self.round)
        outcome = outs[0]
        outcome.action_id = action_id
        self.committed.claims.extend(claims)
        self.committed.actions.append((agent, action_id, action, outcome))
        return ToolResult(call_id=call_id, ok=True, pending=False, result={
            "id": action_id, "accepted": outcome.accepted, "feedback": dict(outcome.feedback)})

    def _registry(self, agent: AgentId, call_id: CallId, name: str, args: dict) -> ToolResult:
        assert self.registry is not None
        op_name = "get" if name == "registry_get" else TOOL_OPS[name]
        op, error = check_op_args(op_name, args)
        if op is None:
            return self._err(call_id, error or "bad args")
        st = self._st(agent)
        if op_name == "get":
            pending = st.registry if self.commit_mode == "round_end" else []
            return ToolResult(call_id=call_id, ok=True,
                              result=self.registry.get(op.key, self.round, pending))
        op = op.model_copy(update={"agent": agent,
                                   "op_id": f"g{self.round:04d}-{agent}-{len(st.registry):02d}"})
        st.registry.append(op)
        if self.commit_mode == "round_end":
            return ToolResult(call_id=call_id, ok=True, result={"id": op.op_id}, pending=True)
        outcome = self.registry.apply(op, self.round)
        self.committed.registry.append(outcome)
        return ToolResult(call_id=call_id, ok=True, pending=False,
                          result={"id": op.op_id, "accepted": outcome.ok, "feedback": outcome.feedback()})
