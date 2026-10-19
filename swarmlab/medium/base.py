"""Policy and Topology base classes (docs/INTERFACE.md §9).

A policy is a function of reader, post, and round; it knows nothing about inboxes or snapshots.

Besides `recipients`, `Topology` carries the channel protocol that `roles.bind_roles` and the
executor call on every topology: `reset_agents`, `channels`, `channel_members`,
`channel_permissions`, `default_channel`. The defaults describe a topology that adds no
channels and restricts nobody, with posts going to `main` (Broadcast, Gossip, Groups, Star);
`Tree` overrides them all. Nothing enforces `channel_members` on its own: `recipients` decides
who a post is delivered to and `channel_permissions` decides who may read and write each channel,
so a topology that wants membership to matter must use it from those two (as `Tree` does).
"""
from __future__ import annotations

import random
from typing import TYPE_CHECKING, ClassVar

from ..base import Persistable, Plugin
from ..ids import AgentId

if TYPE_CHECKING:
    from .board import Post


class Policy(Persistable, Plugin):
    description: ClassVar[str] = ""  # one line for `swarmlab plugins` (swarmlab/registry.py)

    def apply(self, reader: AgentId, post: Post, round: int) -> tuple[int, str] | None:
        """Return (eligible_round, content) or None to withhold. Default: next round, unchanged."""
        return round + 1, post.text


class Topology(Persistable, Plugin):
    description: ClassVar[str] = ""  # one line for `swarmlab plugins` (swarmlab/registry.py)

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
        """The agents of one of this topology's channels. Default: none.

        Descriptive only: no caller enforces it. Overriding it alone changes neither routing nor
        access; `recipients` decides delivery and `channel_permissions` decides read/write, so a
        custom topology must consult its own membership from those (as `Tree` does).
        """
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

    # -- addressed posts and agent-managed channels (comm-protocols) ------------------------

    #: True when agents create channels by posting to a new name: the `channel` argument of
    #: `post` and `read_board` is then a free string (no enum) and the executor adds unknown
    #: names to the board (`Board.add_channel`) instead of rejecting them.
    dynamic_channels: ClassVar[bool] = False

    def post_properties(self, agent: AgentId) -> dict[str, dict]:
        """Extra JSON-schema properties of `post` for `agent` (e.g. `to`). Default: none.

        The executor advertises them, accepts them as `post` arguments and stores their values
        in the post's `fields`, where `recipients` can read them.
        """
        return {}

    def post_required(self) -> list[str]:
        """Names among `post_properties` that every `post` must give. Default: none."""
        return []

    def check_post(self, agent: AgentId, channel: str, extra: dict) -> str | None:
        """An error message when `agent` may not post `extra` to `channel`, else None.

        Called by the executor before the post is buffered; the message is returned to the agent
        as `bad args: <message>`. Default: everything is allowed.
        """
        return None

    def tool_schemas(self, agent: AgentId) -> list:
        """Topology tools offered to `agent` besides `read_board` and `post` (e.g. joining a
        channel). Default: none."""
        return []

    def call_tool(self, agent: AgentId, name: str, args: dict, round: int) -> tuple[bool, dict | str]:
        """Run topology tool `name`: `(True, result)` or `(False, error message)`."""
        return False, f"unknown topology tool {name!r}"
