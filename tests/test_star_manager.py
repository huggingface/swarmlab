"""WP16 §2: the Star topology and the `manager` role (the Flag Game paper's manager protocol)."""
from __future__ import annotations

import asyncio

from swarmlab import Board, Experiment, Run
from swarmlab.blobs import BlobStore
from swarmlab.executor import RoundExecutor
from swarmlab.ids import agent_id
from swarmlab.medium.board import Post
from swarmlab.medium.topology import TOPOLOGIES, Gossip, Star
from swarmlab.participants import EvidenceAggregator
from swarmlab.rng import derive
from swarmlab.roles import BUILTIN_ROLES, MANAGER, assign, resolve_role
from swarmlab.world.flaggame import FlagGame

from .helpers import Chatter, logical

AGENTS = [agent_id(i) for i in range(5)]


def post(agent: str) -> Post:
    return Post(post_id="p0001-0000", round=1, agent=agent, channel="main", text="hi")


def test_star_recipients():
    s = Star()
    rng = derive(1, "topology", 1)
    assert s.recipients(post("a000"), AGENTS, 1, rng) == AGENTS[1:]
    for a in AGENTS[1:]:
        assert s.recipients(post(a), AGENTS, 1, rng) == ["a000"]
    other = Star(center="a003")
    assert other.recipients(post("a003"), AGENTS, 1, rng) == ["a000", "a001", "a002", "a004"]
    assert other.recipients(post("a001"), AGENTS, 1, rng) == ["a003"]
    # center not live: members reach nobody; dead members are not delivered to
    assert s.recipients(post("a002"), AGENTS[1:], 1, rng) == []
    assert s.recipients(post("a000"), ["a000", "a002"], 1, rng) == ["a002"]
    assert TOPOLOGIES["star"] is Star
    assert Board(topology="star").spec()["params"]["topology"] == {"type": "star",
                                                                   "params": {"center": "a000"}}


def test_gossip_k1_is_one_listener_per_speaker():
    """Pairwise protocol: each speaker reaches exactly one listener per round, redrawn per round."""
    g = Gossip(k=1)
    rounds = [g.schedule(AGENTS, r, derive(1, "topology", r)) for r in range(1, 6)]
    for sched in rounds:
        assert all(len(p) == 1 and p[0] != a for a, p in sched.items())
    assert len({tuple(sorted((a, p[0]) for a, p in s.items())) for s in rounds}) > 1


def deliveries(run: Run) -> dict[str, set[str]]:
    posts = {e["post_id"]: e["agent"] for e in run.events if e["type"] == "post"}
    out: dict[str, set[str]] = {}
    for e in run.events:
        if e["type"] == "delivery":
            out.setdefault(posts[e["post_id"]], set()).add(e["recipient"])
    return out


def test_star_run_deliveries_and_replay(tmp_path):
    exp = Experiment(name="star", world=FlagGame(), participants=[Chatter()] * 4,
                     medium=Board(topology="star"), metrics=["comm.read_rate"])
    run = exp.run(seed=1, max_rounds=2, out=tmp_path / "a")
    got = deliveries(run)
    assert got["a000"] == {"a001", "a002", "a003"}
    for a in ("a001", "a002", "a003"):
        assert got[a] == {"a000"}
    again = exp.run(seed=1, max_rounds=2, out=tmp_path / "b")
    assert logical(again) == logical(run)
    assert Run.load(run.dir).replay()["metrics_checked"] > 0


def test_manager_role_is_builtin():
    assert BUILTIN_ROLES["manager"] is MANAGER and resolve_role("manager") is MANAGER
    assert MANAGER.may_act is False
    assert MANAGER.channels_read is None and MANAGER.channels_write is None
    assert MANAGER.tools is None and MANAGER.registry == "write"
    text = MANAGER.prompt_append
    assert "single summary" in text and "which candidate the evidence supports and why" in text
    assert resolve_role("manager", {"may_act": True}).may_act is True


def test_manager_permissions(tmp_path):
    world = FlagGame(blind_agents=1, blind_may_guess=True)  # may_act alone must block guess
    world.reset(derive(1, "world"), AGENTS)
    ex = RoundExecutor(run="r", round=1, world=world, board=Board(topology="star"),
                       blobs=BlobStore(tmp_path / "blobs"), agents=AGENTS, roles={"a000": MANAGER})
    names = [s.name for s in ex.schemas("a000")]
    assert "guess" not in names and {"post", "read_board", "my_status", "end_turn"} <= set(names)
    res = asyncio.run(ex.call("a000", "guess", {"candidate": "A"}))
    assert res.error == "not_allowed"
    assert asyncio.run(ex.call("a000", "post", {"text": "summary"})).ok
    assert asyncio.run(ex.call("a000", "read_board", {})).ok


def test_manager_protocol_run(tmp_path):
    parts = [EvidenceAggregator() for _ in range(4)]
    assign(parts[0], "manager")
    exp = Experiment(name="manager", world=FlagGame(blind_agents=1), participants=parts,
                     medium=Board(topology="star"),
                     metrics=["belief.accuracy", "belief.consensus", "comm.read_rate"])
    run = exp.run(seed=1, max_rounds=3, out=tmp_path)
    roles = {e["agent"]: e["role"] for e in run.events if e["type"] == "turn_started"}
    assert roles["a000"] == "manager" and roles["a001"] is None
    guessed = {e["agent"] for e in run.events if e["type"] == "action_committed"}
    assert "a000" not in guessed and guessed
    assert {n for _, _, n in run.metrics["belief.accuracy"]} == {3}
    got = deliveries(run)
    assert all(rs == {"a000"} for a, rs in got.items() if a != "a000")
    assert Run.load(run.dir).replay()["metrics_checked"] > 0
