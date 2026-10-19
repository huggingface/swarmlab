"""Board topologies (docs/INTERFACE.md §9): who receives a post.

Every topology carries the channel protocol of the `Topology` base class (`reset_agents`,
`channels`, `channel_members`, `channel_permissions`, `default_channel`). `Broadcast`, `Gossip`,
`Groups` and `Star` keep its defaults (no extra channels, no restrictions, posts default to
`main`); `Tree` and `Rooms` override all five.

Decisions where the contract is silent:

- `Gossip(k)` is directed: each author gets `k` partners for the round, drawn with
  `rng.sample` from the other live agents (in the order of `agents`), and all of that author's
  posts in the round go to those partners. The whole round's assignment is computed once, on
  the first `recipients` call of the round, from a *clone* of the passed rng, so the schedule is a
  pure function of the rng state the runner derives from `("topology", round)`. It does not
  matter whether the runner passes one rng per commit or reuses one across several single-item
  commits (immediate mode). The cached assignment is not part of the snapshot. If `k` exceeds
  the number of other agents, every other agent is a partner. An author not in `agents` (not
  live) gets no recipients.
  This is the Flag Game paper's "gossip" protocol (pairwise: one speaker, one listener) at
  `k=1`: in each round every speaker's posts reach exactly one listener, drawn afresh each round.
  It is directed and not a matching: a listener may hear from several speakers in a round (or
  from none), and A->B does not imply B->A. There is no separate `Pairwise` class; `gossip`
  with `params: {k: 1}` is the pairwise protocol.
- `Groups(size)` partitions by agent index: `a000..a{size-1}` are group 0, and so on. Recipients
  are the live members of the author's group, minus the author.
- No topology ever returns the author.
- `Star(center="a000")` (WP16, the Flag Game paper's "manager" protocol together with a blind
  manager: `FlagGame(blind_agents=1)` and the `manager` role). A member's post goes only to the
  center (none when the center is not live); the center's posts go to every live member. No
  channels and no permissions: it changes deliveries only, so a member that calls `read_board`
  sees only the center's posts (and its own, as on any board). An agent list without the center
  delivers member posts to nobody. Pure function of params and the agent list; no snapshot.
- `Tree(groups, coordinators=None, top=None)` (docs/INTERFACE-M3c.md §2). With `groups` an int
  the agents (in agent order, `top` excluded) are split into that many contiguous groups whose
  sizes differ by at most one (earlier groups get the extra agent); explicit lists are used as
  given and every agent but `top` must be in exactly one group. The coordinator of group i is
  `coordinators[i]` or its first member. Channels are `group:<i>` and `coordinators`.
  Membership: `group:<i>` = the group's members (its coordinator included); `coordinators` = the
  coordinators plus `top`. `recipients(post)` = the live members of the post's channel minus the
  author; a post on any other channel (e.g. the board's `main`) is treated as a post on the
  author's default channel (`group:<i>`, or `coordinators` for `top`), which is also what the
  executor uses when `post` is called without a channel. The layout is computed from the full
  agent list given to `reset_agents` (the runner calls it via `roles.bind_roles` at bind, so
  killed agents never reshuffle groups); before that, from the agents passed to `recipients`.
  The layout is a cache, never snapshotted (the topology is a pure function of params and the
  agent list). Tree's `channel_permissions(agent)` returns `(read, write)` channel lists (the
  base default is None, unrestricted), intersected with the agent's role by `bind_roles`.
- `Rooms(rooms={name: [agents]})`: named rooms with fixed membership. Each room is a board
  channel (added by `bind_roles`, like Tree's); an agent may be in several rooms or in none.
  `channel_permissions(agent)` = the rooms it is in, for both read and write, so with or without
  a role an agent can only read and post in its rooms (a role can narrow this further, never
  widen it), and an agent in no room has no board tools. `recipients(post)` = the live members of
  the post's room minus the author, so a member of two rooms reaches only the room it posts on:
  information crosses rooms only when a shared member repeats it. `default_channel(agent)` is the
  agent's room when it is in exactly one, else None, and the executor rejects `post` without a
  channel from an agent in several rooms (`bad args` naming its rooms) rather than picking one.
  A post on a channel that is not a room goes to the author's default room, or to nobody. Room
  names are any non-empty strings; each room lists at least one agent, each agent at most once;
  `reset_agents` rejects members that are not agents of the run. Room names are exposed in
  alphabetical order (`channels()`, `channel_permissions()`, hence the board's channel list and
  the `post` enum), so equivalent mappings behave identically and a reconstructed run keeps its
  channel order. Pure function of params; no snapshot.
"""
from __future__ import annotations

import random
import re
from typing import TYPE_CHECKING, ClassVar

from ..ids import AgentId
from .base import Topology

if TYPE_CHECKING:
    from .board import Post


class Broadcast(Topology):
    """Everyone but the author (the base-class default)."""

    entry_point: ClassVar[str | None] = "broadcast"
    description: ClassVar[str] = "Every post reaches everyone but its author"

    def __init__(self) -> None:
        pass


class Gossip(Topology):
    """Per round, each author's posts go to `k` partners drawn from the round's topology rng."""

    entry_point: ClassVar[str | None] = "gossip"
    description: ClassVar[str] = "Each round, each author's posts go to k partners drawn at random"
    _skip_in_snapshot: ClassVar[tuple[str, ...]] = ("params", "_schedule")

    def __init__(self, k: int = 1) -> None:
        if k < 1:
            raise ValueError(f"Gossip k must be >= 1, got {k}")
        self.k = k
        self._schedule: tuple[int, tuple[AgentId, ...], dict[AgentId, list[AgentId]]] | None = None

    def schedule(self, agents: list[AgentId], round: int, rng: random.Random) -> dict[AgentId, list[AgentId]]:
        """The round's partner assignment, computed once per (round, agents) and then reused."""
        key = tuple(agents)
        cached = self._schedule
        if cached is not None and cached[0] == round and cached[1] == key:
            return cached[2]
        draw = random.Random()
        draw.setstate(rng.getstate())
        assignment: dict[AgentId, list[AgentId]] = {}
        for agent in agents:
            others = [a for a in agents if a != agent]
            assignment[agent] = draw.sample(others, min(self.k, len(others)))
        self._schedule = (round, key, assignment)
        return assignment

    def recipients(self, post: Post, agents: list[AgentId], round: int, rng: random.Random) -> list[AgentId]:
        partners = self.schedule(agents, round, rng).get(post.agent, [])
        return [a for a in partners if a != post.agent]


class Groups(Topology):
    """Fixed groups by agent index; a post reaches the author's group minus the author."""

    entry_point: ClassVar[str | None] = "groups"
    description: ClassVar[str] = "Fixed groups by agent index; a post reaches the author's group"

    def __init__(self, size: int) -> None:
        if size < 1:
            raise ValueError(f"Groups size must be >= 1, got {size}")
        self.size = size

    def group(self, agent: AgentId) -> int:
        return int(agent[1:]) // self.size

    def recipients(self, post: Post, agents: list[AgentId], round: int, rng: random.Random) -> list[AgentId]:
        g = self.group(post.agent)
        return [a for a in agents if a != post.agent and self.group(a) == g]


class Star(Topology):
    """Hub and spokes: members' posts reach only the center; the center's posts reach everyone."""

    entry_point: ClassVar[str | None] = "star"
    description: ClassVar[str] = (
        "Hub and spokes: members reach only the center, the center reaches all"
    )
    _skip_in_snapshot: ClassVar[tuple[str, ...]] = ("params", "center")

    def __init__(self, center: str = "a000") -> None:
        if not isinstance(center, str) or not center:
            raise ValueError(f"Star center must be an agent id, got {center!r}")
        self.center = AgentId(center)

    def recipients(self, post: Post, agents: list[AgentId], round: int, rng: random.Random) -> list[AgentId]:
        if post.agent == self.center:
            return [a for a in agents if a != self.center]
        return [self.center] if self.center in agents else []


AGENT_ID_PATTERN = r"^a[0-9]{3,4}$"


class Direct(Topology):
    """Direct messages: every post names one addressee (`post(to=..., text)`) and reaches only it.

    The `to` argument is a plain string checked against the run's agents, not an enum of agent
    ids: at hundreds of agents an enum would add thousands of tokens to every request and swamp
    the communication cost being measured. A post without `to`, to an unknown agent or to the
    author is refused with a tool error, so nothing is delivered silently to nobody.
    """

    entry_point: ClassVar[str | None] = "direct"
    description: ClassVar[str] = "Direct messages: each post names one addressee (`to`) and reaches only it"
    _skip_in_snapshot: ClassVar[tuple[str, ...]] = ("params", "agents")

    def __init__(self) -> None:
        self.agents: list[AgentId] = []

    def reset_agents(self, agents: list[AgentId]) -> None:
        self.agents = list(agents)

    def post_properties(self, agent: AgentId) -> dict[str, dict]:
        return {"to": {"type": "string", "pattern": AGENT_ID_PATTERN,
                       "description": "the id of the one agent who receives this message, e.g. a003"}}

    def post_required(self) -> list[str]:
        return ["to"]

    def check_post(self, agent: AgentId, channel: str, extra: dict) -> str | None:
        to = extra.get("to")
        if not isinstance(to, str) or not to:
            return "post needs `to`: the id of the agent who receives the message"
        if to == agent:
            return "`to` must be another agent, not yourself"
        if self.agents and to not in self.agents:
            return f"unknown agent {to!r}; agents are a000..{self.agents[-1]}"
        return None

    def recipients(self, post: Post, agents: list[AgentId], round: int, rng: random.Random) -> list[AgentId]:
        to = post.fields.get("to")
        return [AgentId(to)] if to in agents and to != post.agent else []


CHANNEL_NAME_PATTERN = r"^[A-Za-z0-9_-]{1,40}$"


class SelfOrganized(Topology):
    """Agent-managed channels: agents create, join and leave named channels while the run goes on.

    - A post to a channel name the board does not have yet creates that channel (up to
      `max_channels`); posting to a channel makes the author a member. A post reaches the
      channel's members at commit time, minus the author.
    - `join_channel(channel)` and `leave_channel(channel)` change membership at once (within the
      round, before the commit), so a member who joins in round r receives the posts committed at
      the end of round r. Joining a channel that does not exist yet creates it.
    - `list_channels()` returns every channel with its member count and whether the caller is a
      member; `list_channels(channel=X)` returns X's members.
    - `main` is an ordinary channel with no members at the start: a team that moves everyone into
      it has chosen all-to-all for itself.

    Membership is plain state, snapshotted with the topology, so resume and replay rebuild it.
    The tool calls are logged like every tool call (`tool_calls` export table), which is the
    record of how the rooms formed.
    """

    entry_point: ClassVar[str | None] = "self_organized"
    description: ClassVar[str] = "Agent-managed channels: agents create, join and leave named channels"
    dynamic_channels: ClassVar[bool] = True

    def __init__(self, max_channels: int = 1024, list_limit: int = 100) -> None:
        if max_channels < 1:
            raise ValueError(f"max_channels must be >= 1, got {max_channels}")
        if list_limit < 1:
            raise ValueError(f"list_limit must be >= 1, got {list_limit}")
        self.max_channels = max_channels
        self.list_limit = list_limit
        self.members: dict[str, list[AgentId]] = {}

    def reset_agents(self, agents: list[AgentId]) -> None:
        self.members = {}

    def channels(self) -> list[str]:
        return sorted(self.members)

    def channel_members(self, channel: str) -> list[AgentId]:
        return list(self.members.get(channel, []))

    def default_channel(self, agent: AgentId) -> str | None:
        return None  # every post names its channel

    def _ensure(self, channel: str) -> str | None:
        if not isinstance(channel, str) or not re.fullmatch(CHANNEL_NAME_PATTERN, channel):
            return "channel names are 1-40 letters, digits, '_' or '-'"
        if channel not in self.members:
            if channel != "main" and len([c for c in self.members if c != "main"]) >= self.max_channels:
                return f"too many channels (at most {self.max_channels})"
            self.members[channel] = []
        return None

    def _join(self, agent: AgentId, channel: str) -> str | None:
        err = self._ensure(channel)
        if err is None and agent not in self.members[channel]:
            self.members[channel].append(agent)
        return err

    def check_post(self, agent: AgentId, channel: str, extra: dict) -> str | None:
        return self._join(agent, channel)  # posting creates the channel and joins its author

    def tool_schemas(self, agent: AgentId) -> list:
        from ..tools import ToolSchema

        chan = {"type": "string", "pattern": CHANNEL_NAME_PATTERN}
        return [
            ToolSchema(name="join_channel",
                       description="Join a channel (creating it if it does not exist) to receive its posts.",
                       parameters={"type": "object", "properties": {"channel": chan},
                                   "required": ["channel"], "additionalProperties": False}),
            ToolSchema(name="leave_channel", description="Leave a channel; you stop receiving its posts.",
                       parameters={"type": "object", "properties": {"channel": chan},
                                   "required": ["channel"], "additionalProperties": False}),
            ToolSchema(name="list_channels",
                       description=("List the channels with their member counts, or, given a channel, "
                                    "its members."),
                       parameters={"type": "object", "properties": {"channel": chan},
                                   "required": [], "additionalProperties": False}),
        ]

    def call_tool(self, agent: AgentId, name: str, args: dict, round: int) -> tuple[bool, dict | str]:
        channel = args.get("channel")
        if set(args) - {"channel"}:
            return False, f"bad args: {name}(channel)"
        if name == "join_channel":
            err = self._join(agent, channel)
            return (False, err) if err else (True, {"channel": channel,
                                                    "members": len(self.members[channel])})
        if name == "leave_channel":
            if channel not in self.members or agent not in self.members[channel]:
                return False, f"you are not a member of {channel!r}"
            self.members[channel].remove(agent)
            return True, {"channel": channel, "members": len(self.members[channel])}
        if name == "list_channels":
            if channel is not None:
                if channel not in self.members:
                    return False, f"unknown channel {channel!r}"
                return True, {"channel": channel, "members": list(self.members[channel])}
            rows = sorted(self.members.items(), key=lambda kv: (-len(kv[1]), kv[0]))
            return True, {"channels": [{"channel": c, "members": len(m), "joined": agent in m}
                                       for c, m in rows[: self.list_limit]],
                          "total": len(rows)}
        return False, f"unknown topology tool {name!r}"

    def recipients(self, post: Post, agents: list[AgentId], round: int, rng: random.Random) -> list[AgentId]:
        members = set(self.members.get(post.channel, []))
        return [a for a in agents if a != post.agent and a in members]


class Tree(Topology):
    """Groups with coordinators and an optional coordinator-of-coordinators (`top`)."""

    entry_point: ClassVar[str | None] = "tree"
    description: ClassVar[str] = (
        "Groups with coordinators and an optional coordinator-of-coordinators"
    )
    _skip_in_snapshot: ClassVar[tuple[str, ...]] = ("params", "groups", "coordinators", "top")

    def __init__(self, groups: list[list[AgentId]] | int,
                 coordinators: list[AgentId] | None = None, top: AgentId | None = None) -> None:
        if isinstance(groups, bool) or not isinstance(groups, (int, list, tuple)):
            raise TypeError(f"Tree groups must be an int or a list of agent lists, got {groups!r}")
        if isinstance(groups, int):
            if groups < 1:
                raise ValueError(f"Tree groups must be >= 1, got {groups}")
            self.groups: list[list[AgentId]] | int = groups
        else:
            if not groups or any(not g for g in groups):
                raise ValueError("Tree groups must be non-empty lists of agents")
            self.groups = [[AgentId(a) for a in g] for g in groups]
            flat = [a for g in self.groups for a in g]
            if len(set(flat)) != len(flat):
                raise ValueError("an agent is in more than one Tree group")
        n = groups if isinstance(groups, int) else len(groups)
        if coordinators is not None and len(coordinators) != n:
            raise ValueError(f"Tree needs one coordinator per group ({n}), got {len(coordinators)}")
        self.coordinators = None if coordinators is None else [AgentId(a) for a in coordinators]
        self.top = None if top is None else AgentId(top)
        if isinstance(self.groups, list):
            for i, g in enumerate(self.groups):
                if self.coordinators is not None and self.coordinators[i] not in g:
                    raise ValueError(f"coordinator {self.coordinators[i]} is not in group {i}")
                if self.top is not None and self.top in g:
                    raise ValueError(f"top {self.top} must not be in a group")
        self._layout: tuple[tuple[AgentId, ...], list[list[AgentId]]] | None = None

    @property
    def n_groups(self) -> int:
        return self.groups if isinstance(self.groups, int) else len(self.groups)

    def reset_agents(self, agents: list[AgentId]) -> None:
        """Fix the layout from the run's full agent list."""
        self._layout = None
        self._members(list(agents))

    def _members(self, agents: list[AgentId] | None = None) -> list[list[AgentId]]:
        if self._layout is not None:
            return self._layout[1]
        if agents is None:
            raise RuntimeError("Tree layout is not set: call reset_agents(agents) first")
        if isinstance(self.groups, list):
            members = [list(g) for g in self.groups]
            covered = {a for g in members for a in g} | ({self.top} if self.top else set())
            missing = [a for a in agents if a not in covered]
            if missing:
                raise ValueError(f"agents {missing} are in no Tree group")
        else:
            pool = [a for a in agents if a != self.top]
            k = self.groups
            if len(pool) < k:
                raise ValueError(f"Tree(groups={k}) needs at least {k} agents besides top, got {len(pool)}")
            base, extra = divmod(len(pool), k)
            members, i = [], 0
            for g in range(k):
                size = base + (1 if g < extra else 0)
                members.append(pool[i:i + size])
                i += size
        if self.coordinators is not None:
            for i, c in enumerate(self.coordinators):
                if c not in members[i]:
                    raise ValueError(f"coordinator {c} is not in group {i}")
        self._layout = (tuple(agents), members)
        return members

    def coordinator(self, group: int) -> AgentId:
        return self.coordinators[group] if self.coordinators is not None else self._members()[group][0]

    def group_of(self, agent: AgentId) -> int | None:
        for i, g in enumerate(self._members()):
            if agent in g:
                return i
        return None

    def channels(self) -> list[str]:
        return [f"group:{i}" for i in range(self.n_groups)] + ["coordinators"]

    def channel_members(self, channel: str) -> list[AgentId]:
        if channel == "coordinators":
            out = [self.coordinator(i) for i in range(self.n_groups)]
            return out + ([self.top] if self.top is not None else [])
        if channel.startswith("group:"):
            return list(self._members()[int(channel[len("group:"):])])
        return []

    def channel_permissions(self, agent: AgentId) -> tuple[list[str], list[str]]:
        chans = [c for c in self.channels() if agent in self.channel_members(c)]
        return list(chans), list(chans)

    def default_channel(self, agent: AgentId) -> str | None:
        g = self.group_of(agent)
        if g is not None:
            return f"group:{g}"
        return "coordinators" if agent == self.top else None

    def recipients(self, post: Post, agents: list[AgentId], round: int, rng: random.Random) -> list[AgentId]:
        self._members(list(agents))
        channel = post.channel if post.channel in self.channels() else self.default_channel(post.agent)
        if channel is None:
            return []
        members = set(self.channel_members(channel))
        return [a for a in agents if a != post.agent and a in members]


class Rooms(Topology):
    """Named rooms with fixed membership; an agent may be in several rooms or in none.

    Room names are exposed in alphabetical order whatever the order of the `rooms` mapping, so
    equivalent mappings behave identically (spec hashing sorts keys) and a run reconstructed from
    run.json keeps its channel order.
    """

    entry_point: ClassVar[str | None] = "rooms"
    description: ClassVar[str] = "Named rooms with fixed membership; an agent may be in several rooms or none"
    _skip_in_snapshot: ClassVar[tuple[str, ...]] = ("params", "rooms")

    def __init__(self, rooms: dict[str, list[AgentId]]) -> None:
        if not isinstance(rooms, dict) or not rooms:
            raise TypeError(f"Rooms rooms must be a non-empty mapping of room -> agents, got {rooms!r}")
        for name in rooms:
            if not isinstance(name, str) or not name:
                raise ValueError(f"room names must be non-empty strings, got {name!r}")
        self.rooms: dict[str, list[AgentId]] = {}  # canonical: sorted by room name
        for name, members in sorted(rooms.items()):
            if isinstance(members, str) or not isinstance(members, (list, tuple)) or not members:
                raise ValueError(f"room {name!r} must list at least one agent, got {members!r}")
            if len(set(members)) != len(members):
                raise ValueError(f"room {name!r} lists an agent twice")
            self.rooms[name] = [AgentId(a) for a in members]

    def reset_agents(self, agents: list[AgentId]) -> None:
        """Check that every member is one of the run's agents."""
        unknown = sorted({a for m in self.rooms.values() for a in m} - set(agents))
        if unknown:
            raise ValueError(f"Rooms members {unknown} are not agents of this run")

    def channels(self) -> list[str]:
        return list(self.rooms)

    def channel_members(self, channel: str) -> list[AgentId]:
        return list(self.rooms.get(channel, []))

    def channel_permissions(self, agent: AgentId) -> tuple[list[str], list[str]]:
        chans = [c for c, members in self.rooms.items() if agent in members]
        return list(chans), list(chans)

    def default_channel(self, agent: AgentId) -> str | None:
        chans = self.channel_permissions(agent)[1]
        return chans[0] if len(chans) == 1 else None

    def recipients(self, post: Post, agents: list[AgentId], round: int, rng: random.Random) -> list[AgentId]:
        channel = post.channel if post.channel in self.rooms else self.default_channel(post.agent)
        if channel is None:
            return []
        members = set(self.rooms[channel])
        return [a for a in agents if a != post.agent and a in members]

