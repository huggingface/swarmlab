"""The Rooms topology: named rooms with fixed, possibly overlapping membership."""
from __future__ import annotations

import pytest

from swarmlab import Board, Experiment, Participant, Run, TurnUsage
from swarmlab.ids import agent_id
from swarmlab.medium.board import Post
from swarmlab.medium.topology import TOPOLOGIES, Rooms
from swarmlab.rng import derive
from swarmlab.roles import Role, assign, bind_roles
from swarmlab.spec import load_experiment_yaml
from swarmlab.world.flaggame import FlagGame

from .helpers import logical
from .test_roles import executor

AGENTS = [agent_id(i) for i in range(6)]
# a001 bridges red and blue; a005 is in no room
ROOMS = {"red": ["a000", "a001"], "blue": ["a001", "a002", "a003"], "green": ["a004"]}


def post(agent: str, channel: str = "main") -> Post:
    return Post(post_id="p0001-0000", round=1, agent=agent, channel=channel, text="hi")


def rooms(spec: dict | None = None) -> Rooms:
    r = Rooms(rooms=spec or ROOMS)
    r.reset_agents(AGENTS)
    return r


def test_channels_members_and_permissions():
    r = rooms()
    assert r.channels() == ["red", "blue", "green"]
    assert r.channel_members("blue") == ["a001", "a002", "a003"]
    assert r.channel_members("main") == []
    assert r.channel_permissions("a000") == (["red"], ["red"])
    assert r.channel_permissions("a001") == (["red", "blue"], ["red", "blue"])
    assert r.channel_permissions("a005") == ([], [])


def test_default_channel_only_for_single_room_agents():
    r = rooms()
    assert r.default_channel("a000") == "red"
    assert r.default_channel("a004") == "green"
    assert r.default_channel("a001") is None  # in two rooms: must name one
    assert r.default_channel("a005") is None  # in none


def test_recipients_follow_the_post_channel():
    r = rooms()
    rng = derive(0, "topology", 1)
    # the bridge's post on red reaches red only, not its other room
    assert r.recipients(post("a001", "red"), AGENTS, 1, rng) == ["a000"]
    assert r.recipients(post("a001", "blue"), AGENTS, 1, rng) == ["a002", "a003"]
    assert r.recipients(post("a002", "blue"), AGENTS, 1, rng) == ["a001", "a003"]
    assert r.recipients(post("a004", "green"), AGENTS, 1, rng) == []
    # a channel that is not a room: the author's single room, else nobody
    assert r.recipients(post("a000"), AGENTS, 1, rng) == ["a001"]
    assert r.recipients(post("a001"), AGENTS, 1, rng) == []
    assert r.recipients(post("a005"), AGENTS, 1, rng) == []
    live = [a for a in AGENTS if a != "a003"]
    assert r.recipients(post("a002", "blue"), live, 1, rng) == ["a001"]


def test_validation():
    for bad in ({}, {"red": []}, {"": ["a000"]}, {"red": ["a000", "a000"]}, ["a000"]):
        with pytest.raises((TypeError, ValueError)):
            Rooms(rooms=bad)
    with pytest.raises(ValueError, match="a009"):
        Rooms(rooms={"red": ["a000", "a009"]}).reset_agents(AGENTS)
    assert TOPOLOGIES["rooms"] is Rooms


def test_spec_is_snapshot_free_and_rebuilds():
    r = rooms()
    assert r.spec() == {"type": "rooms", "params": {"rooms": ROOMS}}
    board = Board.from_spec(Board(topology=r.spec()).spec())
    assert isinstance(board.topology, Rooms) and board.topology.rooms == ROOMS
    fresh = Rooms(rooms=ROOMS)
    fresh.restore(r.snapshot())
    fresh.reset_agents(AGENTS)
    assert fresh.channel_permissions("a001") == r.channel_permissions("a001")


async def test_executor_offers_each_agent_its_rooms(tmp_path):
    board = Board(topology=Rooms(rooms=ROOMS))
    roles = bind_roles(board, {a: None for a in AGENTS}, [])
    assert board.channels == ["main", "red", "blue", "green"]
    ex = executor(tmp_path, roles, board=board)

    def enums(agent: str) -> dict[str, list[str]]:
        return {s.name: s.parameters["properties"]["channel"]["enum"]
                for s in ex.schemas(agent) if s.name in ("post", "read_board")}

    assert enums("a001") == {"read_board": ["red", "blue"], "post": ["red", "blue"]}
    assert enums("a000") == {"read_board": ["red"], "post": ["red"]}
    assert enums("a005") == {}
    res = await ex.call("a001", "post", {"text": "x"})
    assert res.error == "bad args: post needs a channel, one of ['red', 'blue']"
    assert (await ex.call("a001", "post", {"channel": "main", "text": "x"})).error == "not_allowed"
    assert (await ex.call("a000", "post", {"text": "x"})).ok


# ---- in runs ---------------------------------------------------------------------------------
class RoomTalker(Participant):
    """Posts once without a channel and once on every room it may write, then reads."""

    async def turn(self, view, tools):
        await tools.call("post", {"text": f"{self.agent} default r{view.round}"})
        for s in view.tools:
            if s.name == "post":
                for ch in s.parameters["properties"]["channel"].get("enum") or []:
                    await tools.call("post", {"channel": ch, "text": f"{self.agent} {ch} r{view.round}"})
        await tools.call("read_board", {})
        await tools.call("end_turn", {})
        return TurnUsage(calls=4)


def rooms_experiment(roles: dict | None = None) -> Experiment:
    parts = [RoomTalker() for _ in AGENTS]
    for i, name in (roles or {}).items():
        assign(parts[i], name)
    return Experiment(name="rooms", world=FlagGame(), participants=parts,
                      medium=Board(topology=Rooms(rooms=ROOMS)),
                      roles={n: Role(name=n, channels_write=["red"]) for n in (roles or {}).values()},
                      metrics=["comm.read_rate", "comm.posts_per_round"])


def deliveries(run: Run) -> dict[str, set[str]]:
    out: dict[str, set[str]] = {}
    for e in run.events:
        if e["type"] == "delivery":
            out.setdefault(e["post_id"], set()).add(e["recipient"])
    return out


def returned(run: Run, agent: str, tool: str) -> list[dict]:
    called = {e["call_id"]: e for e in run.events if e["type"] == "tool_called" and e["agent"] == agent}
    return [e for e in run.events if e["type"] == "tool_returned" and e["call_id"] in called
            and called[e["call_id"]]["tool"] == tool]


def test_rooms_run_deliveries(tmp_path):
    run = rooms_experiment().run(seed=1, max_rounds=2, out=tmp_path)
    posts = [e for e in run.events if e["type"] == "post"]
    assert {p["channel"] for p in posts} == {"red", "blue", "green"}
    got = deliveries(run)
    for p in posts:
        assert got.get(p["post_id"], set()) == set(ROOMS[p["channel"]]) - {p["agent"]}
    by_agent = {(p["agent"], p["channel"]) for p in posts}
    assert {("a001", "red"), ("a001", "blue")} <= by_agent
    assert not any(a == "a005" for a, _ in by_agent)
    # the bridge's channel-less post is rejected with a hint; single-room agents' go to their room
    rejected = [r for r in returned(run, "a001", "post") if not r["result"]["ok"]]
    assert rejected and "red" in rejected[0]["result"]["error"] and "blue" in rejected[0]["result"]["error"]
    assert all(r["result"]["ok"] for r in returned(run, "a000", "post"))
    # an agent in no room has no board access
    lonely = returned(run, "a005", "post") + returned(run, "a005", "read_board")
    assert lonely and all(r["result"]["error"] == "not_allowed" for r in lonely)


def test_role_narrows_room_permissions(tmp_path):
    run = rooms_experiment(roles={1: "red_only"}).run(seed=1, max_rounds=1, out=tmp_path)
    posted = {p["channel"] for p in run.events if p["type"] == "post" and p["agent"] == "a001"}
    assert posted == {"red"}
    # still a member of two rooms, so a channel-less post is rejected
    assert any(not r["result"]["ok"] for r in returned(run, "a001", "post"))


def test_rooms_run_is_deterministic_and_replays(tmp_path):
    exp = rooms_experiment()
    a = exp.run(seed=3, max_rounds=3, out=tmp_path / "a")
    b = exp.run(seed=3, max_rounds=3, out=tmp_path / "b")
    assert logical(a) == logical(b)
    loaded = Run.load(a.dir)
    assert loaded.replay()["metrics_checked"] > 0
    assert deliveries(loaded) == deliveries(a)


def test_yaml_rooms_round_trip(tmp_path):
    path = tmp_path / "x.yaml"
    path.write_text("""
name: r
options: {max_rounds: 1}
arms:
  rooms:
    world: flaggame
    participants:
      - {type: "tests.test_rooms_topology:RoomTalker", count: 6}
    medium:
      topology: {type: rooms, params: {rooms: {red: [a000, a001], blue: [a001, a002, a003], green: [a004]}}}
""")
    exp = Experiment.from_yaml(path, "rooms")
    assert isinstance(exp.medium.topology, Rooms) and exp.medium.topology.rooms == ROOMS
    out = tmp_path / "back.yaml"
    exp.to_yaml(out)
    doc = load_experiment_yaml(out)
    assert doc["arms"]["rooms"]["medium"]["topology"] == {"type": "rooms", "params": {"rooms": ROOMS}}
    assert Experiment.from_yaml(out, "rooms").spec_hash(0, 1) == exp.spec_hash(0, 1)
