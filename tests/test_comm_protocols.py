"""The `direct` (DM) and `self_organized` (agent-managed channels) topologies."""
from __future__ import annotations

import pytest

from swarmlab import Board, Experiment, Participant, Run, TurnUsage
from swarmlab.ids import agent_id
from swarmlab.medium.board import Post
from swarmlab.medium.topology import Direct, SelfOrganized
from swarmlab.registry import resolve
from swarmlab.rng import derive
from swarmlab.roles import bind_roles
from swarmlab.world.flaggame import FlagGame

from .test_roles import executor

AGENTS = [agent_id(i) for i in range(6)]


def post(agent: str, channel: str = "main", **fields) -> Post:
    return Post(post_id="p0001-0000", round=1, agent=agent, channel=channel, text="hi", fields=fields)


def bound(topology) -> Board:
    board = Board(topology=topology)
    bind_roles(board, {a: None for a in AGENTS}, [])
    return board


# ---- direct ------------------------------------------------------------------------------------
def test_direct_recipients_and_registry():
    d = Direct()
    d.reset_agents(AGENTS)
    rng = derive(0, "topology", 1)
    assert d.recipients(post("a000", to="a003"), AGENTS, 1, rng) == ["a003"]
    assert d.recipients(post("a000"), AGENTS, 1, rng) == []
    assert d.recipients(post("a000", to="a000"), AGENTS, 1, rng) == []
    assert d.recipients(post("a000", to="a003"), [a for a in AGENTS if a != "a003"], 1, rng) == []
    assert resolve("direct", "swarmlab.topologies") is Direct


async def test_direct_post_schema_and_checks(tmp_path):
    ex = executor(tmp_path, {}, board=bound(Direct()))
    schema = next(s for s in ex.schemas("a000") if s.name == "post")
    props = schema.parameters["properties"]
    assert set(schema.parameters["required"]) == {"to", "text"}
    assert "enum" not in props["to"] and props["to"]["pattern"]  # no per-agent enum (token cost)
    assert (await ex.call("a000", "post", {"text": "x"})).error.startswith("bad args: post needs `to`")
    assert "yourself" in (await ex.call("a000", "post", {"to": "a000", "text": "x"})).error
    assert "unknown agent" in (await ex.call("a000", "post", {"to": "a099", "text": "x"})).error
    res = await ex.call("a000", "post", {"to": "a002", "text": "x"})
    assert res.ok, res.error


# ---- self_organized ----------------------------------------------------------------------------
def test_self_organized_membership_tools():
    t = SelfOrganized()
    t.reset_agents(AGENTS)
    assert t.channels() == []
    assert t.call_tool("a000", "join_channel", {"channel": "zone-1"}, 1) == (True, {"channel": "zone-1",
                                                                                    "members": 1})
    assert t.call_tool("a001", "join_channel", {"channel": "zone-1"}, 1)[0]
    assert t.check_post("a002", "zone-1", {}) is None  # posting joins
    assert t.channel_members("zone-1") == ["a000", "a001", "a002"]
    rng = derive(0, "topology", 1)
    assert t.recipients(post("a001", "zone-1"), AGENTS, 1, rng) == ["a000", "a002"]
    assert t.call_tool("a001", "leave_channel", {"channel": "zone-1"}, 1)[0]
    assert not t.call_tool("a001", "leave_channel", {"channel": "zone-1"}, 1)[0]
    assert t.recipients(post("a000", "zone-1"), AGENTS, 1, rng) == ["a002"]
    ok, listing = t.call_tool("a003", "list_channels", {}, 1)
    assert ok and listing == {"channels": [{"channel": "zone-1", "members": 2, "joined": False}],
                              "total": 1}
    assert t.call_tool("a003", "list_channels", {"channel": "zone-1"}, 1) == (
        True, {"channel": "zone-1", "members": ["a000", "a002"]})
    assert not t.call_tool("a000", "join_channel", {"channel": "bad name!"}, 1)[0]


def test_self_organized_channel_cap_and_snapshot():
    t = SelfOrganized(max_channels=2)
    t.reset_agents(AGENTS)
    assert t.check_post("a000", "x", {}) is None and t.check_post("a000", "y", {}) is None
    assert "too many" in t.check_post("a000", "z", {})
    assert t.check_post("a000", "main", {}) is None  # main never counts toward the cap
    fresh = SelfOrganized(max_channels=2)
    fresh.restore(t.snapshot())
    assert fresh.channel_members("x") == ["a000"] and fresh.channels() == ["main", "x", "y"]


async def test_self_organized_executor_schemas_and_dynamic_channels(tmp_path):
    board = bound(SelfOrganized())
    ex = executor(tmp_path, {}, board=board)
    names = {s.name for s in ex.schemas("a000")}
    assert {"join_channel", "leave_channel", "list_channels", "post", "read_board"} <= names
    post_schema = next(s for s in ex.schemas("a000") if s.name == "post")
    assert "enum" not in post_schema.parameters["properties"]["channel"]
    assert set(post_schema.parameters["required"]) == {"channel", "text"}
    assert (await ex.call("a000", "post", {"text": "x"})).error.startswith("bad args: post needs a channel")
    assert (await ex.call("a000", "post", {"channel": "zone-3", "text": "x"})).ok
    assert "zone-3" in board.channels and "zone-3" not in board.spec()["params"]["channels"]
    assert (await ex.call("a001", "join_channel", {"channel": "zone-3"})).ok
    assert (await ex.call("a001", "read_board", {"channel": "zone-3"})).ok
    bad = await ex.call("a001", "join_channel", {"channel": "x", "extra": 1})
    assert not bad.ok and bad.error.startswith("bad args")


# ---- in runs -----------------------------------------------------------------------------------
class DMTalker(Participant):
    """Sends one DM to the next agent each round, then reads."""

    async def turn(self, view, tools):
        n = int(self.agent[1:])
        await tools.call("post", {"to": agent_id((n + 1) % len(AGENTS)), "text": f"{self.agent} r{view.round}"})
        await tools.call("read_board", {})
        await tools.call("end_turn", {})
        return TurnUsage(calls=3)


class Roomer(Participant):
    """Even agents talk in `even`, odd agents in `odd`; a005 joins `even` in round 2."""

    async def turn(self, view, tools):
        n = int(self.agent[1:])
        if self.agent == "a005" and view.round == 2:
            await tools.call("join_channel", {"channel": "even"})
        await tools.call("post", {"channel": "even" if n % 2 == 0 else "odd", "text": f"{self.agent} r{view.round}"})
        await tools.call("read_board", {})
        await tools.call("end_turn", {})
        return TurnUsage(calls=4)


def deliveries(run: Run) -> dict[str, set[str]]:
    posts = {e["post_id"]: (e["agent"], e["round"]) for e in run.events if e["type"] == "post"}
    out: dict[str, set[str]] = {}
    for e in run.events:
        if e["type"] == "delivery":
            out.setdefault(posts[e["post_id"]], set()).add(e["recipient"])
    return out


def run_with(tmp_path, topology, part, rounds: int = 2) -> Run:
    exp = Experiment(name="comm", world=FlagGame(), participants=[part() for _ in AGENTS],
                     medium=Board(topology=topology), metrics=["comm.posts_per_round"])
    return exp.run(seed=1, max_rounds=rounds, out=tmp_path / "runs")


def test_direct_run_delivers_only_to_addressee(tmp_path):
    run = run_with(tmp_path, Direct(), DMTalker)
    got = deliveries(run)
    assert got[("a000", 1)] == {"a001"} and got[("a005", 2)] == {"a000"}
    assert all(len(r) == 1 for r in got.values())


def test_self_organized_run_and_late_join(tmp_path):
    run = run_with(tmp_path, SelfOrganized(), Roomer)
    got = deliveries(run)
    assert got[("a000", 1)] == {"a002", "a004"}
    assert got[("a001", 1)] == {"a003", "a005"}
    assert got[("a000", 2)] == {"a002", "a004", "a005"}  # a005 joined `even` in round 2


@pytest.mark.parametrize("topology", [Direct, SelfOrganized])
def test_runs_replay(tmp_path, topology):
    part = DMTalker if topology is Direct else Roomer
    run = run_with(tmp_path, topology(), part, rounds=3)
    assert Run.load(run.dir).replay()["metrics_checked"] > 0
