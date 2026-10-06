"""Policy and Topology base classes (docs/INTERFACE.md §9).

A policy is a function of reader, post, and round; it knows nothing about inboxes or snapshots.
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
