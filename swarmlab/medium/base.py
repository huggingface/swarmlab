"""Policy and Topology base classes (docs/INTERFACE.md §9).

A policy is a function of reader, post, and round; it knows nothing about inboxes or snapshots.

Besides `recipients`, `Topology` carries the channel protocol that `roles.bind_roles` and the
executor call on every topology: `reset_agents`, `channels`, `channel_members`,
`channel_permissions`, `default_channel`. The defaults describe a topology that adds no
channels and restricts nobody, with posts going to `main` (Broadcast, Gossip, Groups, Star);
`Tree` overrides them all.
"""
from __future__ import annotations

import random
from typing import TYPE_CHECKING

from ..base import Persistable, Plugin
from ..ids import AgentId

if TYPE_CHECKING:
    from .board import Post


class Policy(Persistable, Plugin):
    def apply(self, reader: AgentId, post: Post, round: int) -> tuple[int, str] | None:
        """Return (eligible_round, content) or None to withhold. Default: next round, unchanged."""
        return round + 1, post.text


class Topology(Persistable, Plugin):
    def recipients(self, post: Post, agents: list[AgentId], round: int, rng: random.Random) -> list[AgentId]:
        """Default: broadcast to everyone but the author."""
        return [a for a in agents if a != post.agent]

    def reset_agents(self, agents: list[AgentId]) -> None:
        """Called once at bind (by `roles.bind_roles`) with the run's full agent list.

        A topology that lays out channels from the agents fixes the layout here, so later kills
        never reshuffle it. Default: no-op.
        """

    def channels(self) -> list[str]:
        """Channels this topology adds to the board (after `main`) at bind. Default: none."""
        return []

    def channel_members(self, channel: str) -> list[AgentId]:
        """The agents of one of this topology's channels. Default: none."""
        return []

    def channel_permissions(self, agent: AgentId) -> tuple[list[str], list[str]] | None:
        """`(read, write)` channel lists for `agent`, or None for unrestricted.

        `bind_roles` intersects a tuple with the agent's role lists (intersection wins; an agent
        without a role gets an anonymous role carrying only these limits). None leaves the role,
        if any, to apply alone. Default: None.
        """
        return None

    def default_channel(self, agent: AgentId) -> str | None:
        """The channel a `post` without a `channel` argument goes to. Default: `"main"`.

        None means the agent must name a channel: the executor answers such a call with
        `bad args: post needs a channel, one of [<writable channels>]`.
        """
        return "main"
