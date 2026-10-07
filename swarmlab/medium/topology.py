"""Board topologies (docs/INTERFACE.md §9): who receives a post.

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
- `Groups(size)` partitions by agent index: `a000..a{size-1}` are group 0, and so on. Recipients
  are the live members of the author's group, minus the author.
- No topology ever returns the author.
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
  agent list). `channel_permissions(agent)` returns `(read, write)` channel lists, intersected
  with the agent's role by `bind_roles`.
"""
from __future__ import annotations

import random
from typing import TYPE_CHECKING, ClassVar

from ..ids import AgentId
from .base import Topology

if TYPE_CHECKING:
    from .board import Post


class Broadcast(Topology):
    """Everyone but the author (the base-class default)."""

    entry_point: ClassVar[str | None] = "broadcast"

    def __init__(self) -> None:
        pass


class Gossip(Topology):
    """Per round, each author's posts go to `k` partners drawn from the round's topology rng."""

    entry_point: ClassVar[str | None] = "gossip"
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

    def __init__(self, size: int) -> None:
        if size < 1:
            raise ValueError(f"Groups size must be >= 1, got {size}")
        self.size = size

    def group(self, agent: AgentId) -> int:
        return int(agent[1:]) // self.size

    def recipients(self, post: Post, agents: list[AgentId], round: int, rng: random.Random) -> list[AgentId]:
        g = self.group(post.agent)
        return [a for a in agents if a != post.agent and self.group(a) == g]


class Tree(Topology):
    """Groups with coordinators and an optional coordinator-of-coordinators (`top`)."""

    entry_point: ClassVar[str | None] = "tree"
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


TOPOLOGIES: dict[str, type[Topology]] = {
    "broadcast": Broadcast,
    "gossip": Gossip,
    "groups": Groups,
    "tree": Tree,
}
