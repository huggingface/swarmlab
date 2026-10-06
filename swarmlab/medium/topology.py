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


TOPOLOGIES: dict[str, type[Topology]] = {
    "broadcast": Broadcast,
    "gossip": Gossip,
    "groups": Groups,
}
