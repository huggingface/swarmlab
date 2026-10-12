"""INTERFACE §9: the channel protocol on the `Topology` base class (defaults and custom overrides)."""
from __future__ import annotations

import random
from typing import ClassVar

from swarmlab import Board, Experiment, Participant, TurnUsage
from swarmlab.blobs import BlobStore
from swarmlab.executor import RoundExecutor
from swarmlab.ids import AgentId, agent_id
from swarmlab.medium.base import Topology
from swarmlab.medium.board import Post
from swarmlab.medium.topology import Broadcast, Gossip, Groups, Star, Tree
from swarmlab.rng import derive
from swarmlab.roles import Role, bind_roles
from swarmlab.world.flaggame import FlagGame

AGENTS = [agent_id(i) for i in range(3)]


class Sides(Topology):
    """Two extra channels, no permissions, and no default channel: every post must name one."""

    channel_names: ClassVar[list[str]] = ["left", "right"]

    def __init__(self) -> None:
        self.seen: list[AgentId] | None = None

    def reset_agents(self, agents: list[AgentId]) -> None:
        self.seen = list(agents)

    def channels(self) -> list[str]:
        return list(self.channel_names)

    def default_channel(self, agent: AgentId) -> str | None:
        return None

    def recipients(self, post: Post, agents: list[AgentId], round: int, rng: random.Random) -> list[AgentId]:
        return [a for a in agents if a != post.agent]


def executor(tmp_path, board: Board, roles: dict[AgentId, Role]) -> RoundExecutor:
    world = FlagGame()
    world.reset(derive(1, "world"), AGENTS)
    return RoundExecutor(run="r", round=1, world=world, board=board,
                         blobs=BlobStore(tmp_path / "blobs"), agents=AGENTS,
                         topology_rng=lambda: derive(1, "topology", 1), roles=roles)


def test_base_defaults():
    for t in (Topology(), Broadcast(), Gossip(), Groups(size=2), Star()):
        assert t.reset_agents(AGENTS) is None
        assert t.channels() == []
        assert t.channel_members("main") == []
        assert t.channel_permissions(AGENTS[0]) is None
        assert t.default_channel(AGENTS[0]) == "main"


def test_bind_roles_adds_declared_channels_without_roles():
    board = Board(topology=Sides())
    eff = bind_roles(board, {a: object() for a in AGENTS}, [])
    assert board.channels == ["main", "left", "right"]
    assert board.topology.seen == AGENTS
    assert eff == {}  # no role and no permissions: nobody gets an effective role


def test_bind_roles_leaves_default_topology_alone():
    board = Board()
    assert bind_roles(board, {a: object() for a in AGENTS}, []) == {}
    assert board.channels == ["main"]


async def test_post_without_channel_needs_one_when_default_is_none(tmp_path):
    board = Board(topology=Sides())
    bind_roles(board, {a: object() for a in AGENTS}, [])
    ex = executor(tmp_path, board, {AGENTS[1]: Role(name="r", channels_write=["right"])})
    a, b = AGENTS[0], AGENTS[1]
    ex.begin_turn(a)
    res = await ex.call(a, "post", {"text": "hi"})
    assert not res.ok and res.error == "bad args: post needs a channel, one of ['main', 'left', 'right']"
    assert ex.buffered_posts(a) == []
    assert (await ex.call(a, "post", {"channel": "left", "text": "hi"})).ok
    assert ex.buffered_posts(a) == [("left", "hi", {})]
    ex.begin_turn(b)  # a role narrows the list to the agent's writable channels
    assert (await ex.call(b, "post", {"text": "hi"})).error == \
        "bad args: post needs a channel, one of ['right']"


def post_schema(ex: RoundExecutor, agent: AgentId) -> dict:
    return next(s for s in ex.schemas(agent) if s.name == "post").parameters


def test_post_schema_requires_channel_when_default_is_none(tmp_path):
    board = Board(topology=Sides())
    eff = bind_roles(board, {a: object() for a in AGENTS}, [])
    params = post_schema(executor(tmp_path, board, eff), AGENTS[0])
    assert params["required"] == ["channel", "text"]
    assert params["properties"]["channel"]["enum"] == ["main", "left", "right"]
    ex = executor(tmp_path, board, {AGENTS[1]: Role(name="r", channels_write=["right"])})
    params = post_schema(ex, AGENTS[1])  # with a role: still required, enum narrowed
    assert params["required"] == ["channel", "text"]
    assert params["properties"]["channel"]["enum"] == ["right"]


def test_post_schema_channel_optional_with_a_default(tmp_path):
    board = Board(topology=Tree(groups=[AGENTS[:2], AGENTS[2:]]))
    eff = bind_roles(board, {a: object() for a in AGENTS}, [])
    assert board.topology.default_channel(AGENTS[0]) is not None
    assert post_schema(executor(tmp_path, board, eff), AGENTS[0])["required"] == ["text"]
    board = Board()
    assert post_schema(executor(tmp_path, board, {}), AGENTS[0])["required"] == ["text"]


class SidePoster(Participant):
    async def turn(self, view, tools):
        bare = await tools.call("post", {"text": "x"})
        named = await tools.call("post", {"channel": "left", "text": f"{self.agent} r{view.round}"})
        self.results = getattr(self, "results", []) + [(bare.error, named.ok)]
        await tools.call("end_turn", {})
        return TurnUsage(calls=3)


def test_custom_topology_in_a_run(tmp_path):
    exp = Experiment(name="sides", world=FlagGame(), participants=[SidePoster() for _ in AGENTS],
                     medium=Board(topology=Sides()))
    run = exp.run(seed=1, max_rounds=1, out=tmp_path)
    posts = [e for e in run.events if e["type"] == "post"]
    assert sorted(p["agent"] for p in posts) == AGENTS
    assert {p["channel"] for p in posts} == {"left"}
    errors = [e["result"]["error"] for e in run.events if e["type"] == "tool_returned"
              and e["result"]["error"] and e["result"]["error"].startswith("bad args")]
    assert errors == ["bad args: post needs a channel, one of ['main', 'left', 'right']"] * 3
