"""M3c §2: the Tree topology (unit level and in runs: deliveries, determinism, replay, resume)."""
from __future__ import annotations

from pathlib import Path

import pytest

from swarmlab import Board, Experiment, Participant, Run, TurnUsage
from swarmlab.events import parse_event
from swarmlab.ids import agent_id
from swarmlab.medium.board import Post
from swarmlab.medium.topology import TOPOLOGIES, Tree
from swarmlab.rng import derive
from swarmlab.roles import assign
from swarmlab.spec import load_experiment_yaml
from swarmlab.world.flaggame import FlagGame

from .helpers import logical

AGENTS = [agent_id(i) for i in range(10)]


def post(agent: str, channel: str = "main") -> Post:
    return Post(post_id="p0001-0000", round=1, agent=agent, channel=channel, text="hi")


def tree(**kw) -> Tree:
    t = Tree(**kw)
    t.reset_agents(AGENTS)
    return t


def test_int_groups_split_by_index_and_top_is_excluded():
    t = tree(groups=3, top="a009")
    assert [t.channel_members(f"group:{i}") for i in range(3)] == [
        ["a000", "a001", "a002"], ["a003", "a004", "a005"], ["a006", "a007", "a008"]]
    assert t.channel_members("coordinators") == ["a000", "a003", "a006", "a009"]
    assert t.channels() == ["group:0", "group:1", "group:2", "coordinators"]
    # uneven: earlier groups get the extra agent
    u = tree(groups=3)
    assert [len(u.channel_members(f"group:{i}")) for i in range(3)] == [4, 3, 3]


def test_channel_permissions_and_default_channel():
    t = tree(groups=3, top="a009", coordinators=["a001", "a004", "a007"])
    assert t.channel_permissions("a000") == (["group:0"], ["group:0"])
    assert t.channel_permissions("a004") == (["group:1", "coordinators"], ["group:1", "coordinators"])
    assert t.channel_permissions("a009") == (["coordinators"], ["coordinators"])
    assert t.default_channel("a005") == "group:1" and t.default_channel("a009") == "coordinators"


def test_recipients_by_channel():
    t = tree(groups=3, top="a009")
    rng = derive(0, "topology", 1)
    assert t.recipients(post("a001", "group:0"), AGENTS, 1, rng) == ["a000", "a002"]
    assert t.recipients(post("a001"), AGENTS, 1, rng) == ["a000", "a002"]  # main -> own group
    assert t.recipients(post("a003", "coordinators"), AGENTS, 1, rng) == ["a000", "a006", "a009"]
    assert t.recipients(post("a009"), AGENTS, 1, rng) == ["a000", "a003", "a006"]
    live = [a for a in AGENTS if a != "a002"]  # a killed agent never reshuffles groups
    assert t.recipients(post("a001", "group:0"), live, 1, rng) == ["a000"]


def test_explicit_groups_and_validation():
    t = Tree(groups=[["a000", "a001"], ["a002", "a003"]], coordinators=["a001", "a002"])
    t.reset_agents(AGENTS[:4])
    assert t.channel_members("coordinators") == ["a001", "a002"]
    with pytest.raises(ValueError, match="no Tree group"):
        Tree(groups=[["a000"]]).reset_agents(AGENTS[:2])
    with pytest.raises(ValueError, match="one coordinator per group"):
        Tree(groups=2, coordinators=["a000"])
    with pytest.raises(ValueError, match="not in group"):
        Tree(groups=[["a000"], ["a001"]], coordinators=["a001", "a000"])
    with pytest.raises(ValueError):
        Tree(groups=0)
    assert TOPOLOGIES["tree"] is Tree


def test_spec_is_snapshot_free_and_rebuilds():
    t = tree(groups=3, top="a009")
    assert t.spec() == {"type": "tree", "params": {"groups": 3, "coordinators": None, "top": "a009"}}
    board = Board.from_spec(Board(topology=t.spec()).spec())
    assert isinstance(board.topology, Tree) and board.topology.top == "a009"
    fresh = Tree(groups=3, top="a009")
    fresh.restore(t.snapshot())
    fresh.reset_agents(AGENTS)
    assert fresh.channel_members("group:1") == t.channel_members("group:1")


# ---- in runs ---------------------------------------------------------------------------------
class TreeTalker(Participant):
    """Posts to its default channel; coordinators (and top) also post on `coordinators`."""

    async def turn(self, view, tools):
        await tools.call("post", {"text": f"{self.agent} r{view.round}"})
        if any("coordinators" in (s.parameters["properties"]["channel"].get("enum") or [])
               for s in view.tools if s.name == "post"):
            await tools.call("post", {"channel": "coordinators", "text": f"{self.agent} up r{view.round}"})
        await tools.call("read_board", {})
        await tools.call("end_turn", {})
        return TurnUsage(calls=4)


def tree_experiment(n: int = 10) -> Experiment:
    parts = [TreeTalker() for _ in range(n)]
    for i in (0, 3, 6):
        assign(parts[i], "coordinator")
    assign(parts[9], "coordinator")
    return Experiment(name="tree", world=FlagGame(), participants=parts,
                      medium=Board(topology=Tree(groups=3, top="a009")),
                      metrics=["comm.read_rate", "comm.posts_per_round"])


def deliveries(run: Run) -> dict[tuple[int, str], set[str]]:
    posts = {e["post_id"]: e for e in run.events if e["type"] == "post"}
    out: dict[tuple[int, str], set[str]] = {}
    for e in run.events:
        if e["type"] == "delivery":
            p = posts[e["post_id"]]
            out.setdefault((p["round"], p["post_id"]), set()).add(e["recipient"])
    return out


def test_tree_run_deliveries(tmp_path):
    run = tree_experiment().run(seed=1, max_rounds=2, out=tmp_path)
    t = Tree(groups=3, top="a009")
    t.reset_agents(AGENTS)
    posts = [e for e in run.events if e["type"] == "post"]
    assert posts and {p["channel"] for p in posts} == {"group:0", "group:1", "group:2", "coordinators"}
    got = deliveries(run)
    for p in posts:
        expect = set(t.channel_members(p["channel"])) - {p["agent"]}
        assert got.get((p["round"], p["post_id"]), set()) == expect
        if p["agent"] == "a009":
            assert p["channel"] == "coordinators"  # top's default channel
    # top reads only coordinator traffic
    top_posts = {e["post_id"] for e in run.events if e["type"] == "delivery" and e["recipient"] == "a009"}
    assert top_posts and all(next(p for p in posts if p["post_id"] == pid)["channel"] == "coordinators"
                             for pid in top_posts)
    # turn_started records the role
    roles = {e["agent"]: e["role"] for e in run.events if e["type"] == "turn_started"}
    assert roles["a000"] == "coordinator" and roles["a001"] is None


def test_tree_run_is_deterministic_and_replays(tmp_path):
    exp = tree_experiment()
    a = exp.run(seed=3, max_rounds=3, out=tmp_path / "a")
    b = exp.run(seed=3, max_rounds=3, out=tmp_path / "b")
    assert logical(a) == logical(b)
    loaded = Run.load(a.dir)  # rebuilt from the spec: roles and tree come back from run.json
    assert loaded.replay()["metrics_checked"] > 0
    assert deliveries(loaded) == deliveries(a)


def _truncate_mid_round(run_dir: Path, round: int) -> None:
    lines = (run_dir / "events.jsonl").read_bytes().splitlines(keepends=True)
    for i, line in enumerate(lines):
        ev = parse_event(line)
        if ev.type == "tool_called" and ev.round == round:
            (run_dir / "events.jsonl").write_bytes(b"".join(lines[: i + 1]) + lines[i + 1][:10])
            return
    raise AssertionError("round not found")


def test_tree_resume_mid_run_matches_uninterrupted(tmp_path):
    exp = tree_experiment()
    ref = exp.run(seed=2, max_rounds=4, out=tmp_path / "ref")
    run = exp.run(seed=2, max_rounds=4, out=tmp_path / "crash")
    _truncate_mid_round(run.dir, 3)
    resumed = Run(run.dir).resume()  # experiment rebuilt from the spec
    assert resumed.status == "ended"
    assert logical(resumed) == logical(ref) and resumed.score == ref.score
    Run.load(run.dir)


def test_tree_fork_continues_like_parent(tmp_path):
    exp = tree_experiment()
    parent = exp.run(seed=4, max_rounds=4, out=tmp_path)
    child = parent.fork(2).run()
    tail = [e for e in deliveries(child).items() if e[0][0] > 2]
    assert tail == [e for e in deliveries(parent).items() if e[0][0] > 2]


def test_yaml_tree_params_round_trip(tmp_path):
    path = tmp_path / "x.yaml"
    path.write_text("""
name: t
options: {max_rounds: 1}
arms:
  tree:
    world: flaggame
    participants:
      - {type: "tests.test_tree_topology:TreeTalker", count: 9}
    medium: {topology: {type: tree, params: {groups: 3}}}
""")
    exp = Experiment.from_yaml(path, "tree")
    assert isinstance(exp.medium.topology, Tree) and exp.medium.topology.groups == 3
    out = tmp_path / "back.yaml"
    exp.to_yaml(out)
    doc = load_experiment_yaml(out)
    assert doc["arms"]["tree"]["medium"]["topology"] == {
        "type": "tree", "params": {"groups": 3, "coordinators": None, "top": None}}
    assert Experiment.from_yaml(out, "tree").spec_hash(0, 1) == exp.spec_hash(0, 1)
