import hashlib

import pytest

from swarmlab.ids import agent_id
from swarmlab.medium.base import Policy, Topology
from swarmlab.medium.board import Board, DelayPolicy, Delivery, Post
from swarmlab.medium.topology import Gossip, Groups
from swarmlab.rng import derive

AGENTS = [agent_id(i) for i in range(4)]


class MemBlobs:
    def __init__(self):
        self.data: dict[str, bytes] = {}

    def put(self, data: bytes) -> str:
        sha = hashlib.sha256(data).hexdigest()
        self.data[sha] = data
        return sha

    def get(self, sha: str) -> bytes:
        return self.data[sha]


class OddAgentsSeeNothing(Policy):
    def apply(self, reader, post, round):
        if int(reader[1:]) % 2:
            return None                      # withhold
        return round + 1, post.text          # available next round, unchanged


class WithholdFrom(Policy):
    def __init__(self, reader: str):
        self.reader = reader

    def apply(self, reader, post, round):
        return None if reader == self.reader else (round + 1, post.text)


class Shout(Policy):
    def apply(self, reader, post, round):
        return round + 1, post.text.upper()


class Prefix(Policy):
    def __init__(self, prefix: str):
        self.prefix = prefix

    def apply(self, reader, post, round):
        return round + 2, self.prefix + post.text


class CountingPolicy(Policy):
    def __init__(self):
        self.calls = 0

    def apply(self, reader, post, round):
        self.calls += 1
        return round + 1, post.text


def commit(board: Board, rnd: int, blobs=None, agents=AGENTS):
    return board.commit(rnd, agents, derive(0, "topology", rnd), blobs or MemBlobs())


def test_post_and_delivery_ids_and_records():
    board = Board()
    blobs = MemBlobs()
    assert board.buffer_post("a001", 1, "main", "x", {"k": 1}) == "p0001-0000"
    assert board.buffer_post("a000", 1, "main", "y", None) == "p0001-0001"
    posts, dels = commit(board, 1, blobs)
    assert [p.post_id for p in posts] == ["p0001-0000", "p0001-0001"]
    assert [p.agent for p in posts] == ["a001", "a000"]   # insertion order preserved
    assert posts[0].fields == {"k": 1} and posts[1].fields == {}
    assert [d.delivery_id for d in dels] == [f"d0001-{n:05d}" for n in range(6)]
    assert all(d.eligible_round == 2 and d.read_round is None for d in dels)
    assert board.content(dels[0], blobs) == "x"
    assert board.buffer_post("a000", 2, "main", "z", {}) == "p0002-0000"


def test_unknown_channel_raises():
    board = Board(channels=("main", "ops"))
    board.buffer_post("a000", 1, "ops", "ok", {})
    with pytest.raises(ValueError):
        board.buffer_post("a000", 1, "nope", "x", {})
    with pytest.raises(ValueError):
        board.read("a001", 1, channel="nope")


def test_unknown_topology_raises():
    with pytest.raises(ValueError):
        Board(topology="ring")


def test_broadcast_author_excluded():
    board = Board()
    board.buffer_post("a002", 1, "main", "hi", {})
    _, dels = commit(board, 1)
    assert sorted(d.recipient for d in dels) == ["a000", "a001", "a003"]


class IncludesAuthor(Topology):
    def recipients(self, post, agents, round, rng):
        return list(agents)


def test_author_excluded_even_if_topology_returns_it():
    board = Board(topology=IncludesAuthor())
    board.buffer_post("a002", 1, "main", "hi", {})
    _, dels = commit(board, 1)
    assert "a002" not in {d.recipient for d in dels}
    assert len(dels) == 3


def test_gossip_board_commit():
    agents = [agent_id(i) for i in range(10)]
    board = Board(topology="gossip")
    assert isinstance(board.topology, Gossip)
    for a in agents:
        board.buffer_post(a, 1, "main", f"from {a}", {})
    board.buffer_post("a003", 1, "main", "again", {})
    _, dels = board.commit(1, agents, derive(7, "topology", 1), MemBlobs())
    by_post = {}
    for d in dels:
        by_post.setdefault(d.post_id, []).append(d.recipient)
    assert all(len(r) == 1 for r in by_post.values())
    assert by_post["p0001-0003"] == by_post["p0001-0010"]   # a003's two posts, same partner
    assert all(r[0] != f"a{int(pid[-4:]):03d}" for pid, r in by_post.items() if pid != "p0001-0010")

    # same seed -> same schedule on a fresh board
    board2 = Board(topology=Gossip(k=1))
    for a in agents:
        board2.buffer_post(a, 1, "main", f"from {a}", {})
    board2.buffer_post("a003", 1, "main", "again", {})
    _, dels2 = board2.commit(1, agents, derive(7, "topology", 1), MemBlobs())
    assert [d.model_dump() for d in dels] == [d.model_dump() for d in dels2]


def test_groups_board_commit():
    board = Board(topology=Groups(size=2))
    board.buffer_post("a000", 1, "main", "g0", {})
    board.buffer_post("a003", 1, "main", "g1", {})
    _, dels = commit(board, 1)
    assert [(d.post_id, d.recipient) for d in dels] == [("p0001-0000", "a001"), ("p0001-0001", "a002")]


def test_delay_policy_all_readers():
    board = Board(policies=[DelayPolicy(rounds=2)])
    board.buffer_post("a000", 1, "main", "x", {})
    _, dels = commit(board, 1)
    assert {d.eligible_round for d in dels} == {4}
    assert board.read("a001", 3) == []
    assert len(board.read("a001", 4)) == 1


def test_delay_policy_specific_readers():
    board = Board(policies=[DelayPolicy(rounds=3, readers=["a002"])])
    board.buffer_post("a000", 1, "main", "x", {})
    _, dels = commit(board, 1)
    assert {d.recipient: d.eligible_round for d in dels} == {"a001": 2, "a002": 5, "a003": 2}


def test_withholding_policy():
    board = Board(policies=[WithholdFrom("a001")])
    board.buffer_post("a000", 1, "main", "x", {})
    _, dels = commit(board, 1)
    assert sorted(d.recipient for d in dels) == ["a002", "a003"]
    assert board.inbox("a001") == []


def test_first_none_short_circuits():
    counter = CountingPolicy()
    board = Board(policies=[WithholdFrom("a001"), counter])
    board.buffer_post("a000", 1, "main", "x", {})
    commit(board, 1)
    assert counter.calls == 2   # not called for the withheld reader


def test_multiple_policies_combine():
    blobs = MemBlobs()
    board = Board(policies=[Prefix(">"), DelayPolicy(rounds=0), Shout()])
    board.buffer_post("a000", 1, "main", "hey", {})
    _, dels = commit(board, 1, blobs)
    assert {d.eligible_round for d in dels} == {3}          # max of 3, 2, 2
    assert {board.content(d, blobs) for d in dels} == {"HEY"}   # last content wins

    board = Board(policies=[Shout(), Prefix(">")])
    board.buffer_post("a000", 1, "main", "hey", {})
    _, dels = commit(board, 1, blobs)
    assert {board.content(d, blobs) for d in dels} == {">hey"}


def test_odd_agents_see_nothing_design_example():
    board = Board(topology="broadcast", policies=[OddAgentsSeeNothing()])
    board.buffer_post("a000", 1, "main", "x", {})
    board.buffer_post("a001", 1, "main", "y", {})
    _, dels = commit(board, 1)
    assert sorted((d.post_id, d.recipient) for d in dels) == [
        ("p0001-0000", "a002"), ("p0001-0001", "a000"), ("p0001-0001", "a002"),
    ]
    assert board.read("a001", 2) == [] and board.read("a003", 2) == []
    assert len(board.read("a002", 2)) == 2


def test_pull_read_marks_and_respects_eligibility_and_limit():
    board = Board()
    for n in range(5):
        board.buffer_post("a000", 1, "main", f"m{n}", {})
    commit(board, 1)
    assert board.read("a001", 1) == []                     # not yet eligible
    first = board.read("a001", 2, limit=3)
    assert [d.post_id for d in first] == ["p0001-0000", "p0001-0001", "p0001-0002"]
    assert all(d.read_round == 2 for d in first)
    rest = board.read("a001", 3, limit=50)
    assert [d.post_id for d in rest] == ["p0001-0003", "p0001-0004"]
    assert board.read("a001", 4) == []
    assert [d.read_round for d in board.inbox("a001")] == [2, 2, 2, 3, 3]
    assert board.read("a002", 2, limit=0) == []


def test_read_orders_by_eligible_round_then_id():
    board = Board(policies=[DelayPolicy(rounds=2, readers=["a001"])])
    board.buffer_post("a000", 1, "main", "late", {})
    commit(board, 1)                                       # eligible 4 for a001
    board.buffer_post("a000", 2, "main", "plain", {})
    board2_policy = board.policies
    board.policies = []
    commit(board, 2)                                       # eligible 3
    board.policies = board2_policy
    assert [d.post_id for d in board.read("a001", 5)] == ["p0002-0000", "p0001-0000"]


def test_read_channel_filter():
    board = Board(channels=["main", "ops"])
    board.buffer_post("a000", 1, "main", "m", {})
    board.buffer_post("a000", 1, "ops", "o", {})
    commit(board, 1)
    ops = board.read("a001", 2, channel="ops")
    assert [d.post_id for d in ops] == ["p0001-0001"]
    assert board.channel_of(ops[0]) == "ops"
    assert [d.post_id for d in board.read("a001", 2)] == ["p0001-0000"]


def test_pushable_does_not_mark():
    board = Board(delivery="push", push_limit=2)
    for n in range(3):
        board.buffer_post("a000", 1, "main", f"m{n}", {})
    commit(board, 1)
    assert board.pushable("a001", 1, 2) == []
    p1 = board.pushable("a001", 2, 2)
    p2 = board.pushable("a001", 2, 2)
    assert [d.post_id for d in p1] == [d.post_id for d in p2] == ["p0001-0000", "p0001-0001"]
    assert all(d.read_round is None for d in board.inbox("a001"))
    assert len(board.read("a001", 2)) == 3


def test_returned_deliveries_are_copies():
    board = Board()
    board.buffer_post("a000", 1, "main", "x", {})
    _, dels = commit(board, 1)
    dels[0].read_round = 99
    board.inbox("a001")[0].read_round = 99
    assert board.inbox(dels[0].recipient)[0].read_round is None


def test_immediate_mode_same_round_and_unique_ids():
    board = Board(commit_mode="immediate")
    assert board.buffer_post("a000", 3, "main", "a", {}) == "p0003-0000"
    _, d1 = commit(board, 3)
    assert {d.eligible_round for d in d1} == {3}
    assert len(board.read("a001", 3)) == 1
    assert board.buffer_post("a001", 3, "main", "b", {}) == "p0003-0001"
    _, d2 = commit(board, 3)
    assert not {d.delivery_id for d in d1} & {d.delivery_id for d in d2}


def test_immediate_mode_with_policies():
    board = Board(commit_mode="immediate", policies=[OddAgentsSeeNothing(), DelayPolicy(2, ["a002"])])
    board.buffer_post("a000", 3, "main", "a", {})
    _, dels = commit(board, 3)
    # default-returning policies keep same-round visibility; explicit delays are absolute
    assert {d.recipient: d.eligible_round for d in dels} == {"a002": 6}
    board = Board(commit_mode="immediate", policies=[OddAgentsSeeNothing()])
    board.buffer_post("a001", 3, "main", "a", {})
    _, dels = commit(board, 3)
    assert {d.recipient: d.eligible_round for d in dels} == {"a000": 3, "a002": 3}


def test_buffer_cleared_after_commit():
    board = Board()
    board.buffer_post("a000", 1, "main", "x", {})
    assert len(board.buffered) == 1
    posts, _ = commit(board, 1)
    assert len(posts) == 1 and board.buffered == []
    posts, dels = commit(board, 2)
    assert posts == [] and dels == []


def test_snapshot_restore_round_trip():
    blobs = MemBlobs()
    board = Board(topology=Gossip(k=2), policies=[DelayPolicy(1, ["a003"])], channels=["main", "x"])
    for a in AGENTS:
        board.buffer_post(a, 1, "x", f"hi {a}", {"a": a})
    commit(board, 1, blobs)
    board.read("a000", 2)
    board.buffer_post("a001", 2, "main", "pending", {})
    blob = board.snapshot()

    other = Board(topology=Gossip(k=2), policies=[DelayPolicy(1, ["a003"])], channels=["main", "x"])
    other.restore(blob)
    for a in AGENTS:
        assert [d.model_dump() for d in other.inbox(a)] == [d.model_dump() for d in board.inbox(a)]
    assert other.buffered == board.buffered
    assert other.snapshot() == blob
    assert other.buffer_post("a002", 2, "main", "next", {}) == "p0002-0001"
    assert [d.post_id for d in other.read("a001", 2, channel="x")] == [
        d.post_id for d in board.read("a001", 2, channel="x")
    ]


class Stateful(Policy):
    def __init__(self, tag: str = "a"):
        self.tag = tag
        self.seen = 0

    def apply(self, reader, post, round):
        self.seen += 1
        return round + 1, post.text


def test_restore_nested_state_only_when_spec_matches():
    board = Board(policies=[Stateful("a")])
    board.buffer_post("a000", 1, "main", "x", {})
    commit(board, 1)
    blob = board.snapshot()
    same = Board(policies=[Stateful("a")])
    same.restore(blob)
    assert same.policies[0].seen == 3
    edited = Board(policies=[Stateful("b")])
    edited.restore(blob)
    assert edited.policies[0].seen == 0 and edited.policies[0].tag == "b"
    assert len(edited.inbox("a001")) == 1


def test_spec_round_trip_nested():
    board = Board(topology="gossip", policies=[DelayPolicy(2, ["a001"]), OddAgentsSeeNothing()],
                  channels=("main", "ops"), delivery="push", push_limit=5)
    spec = board.spec()
    assert spec == {
        "type": "board",
        "params": {
            "topology": {"type": "gossip", "params": {"k": 1}},
            "delivery": "push",
            "push_limit": 5,
            "policies": [
                {"type": "delay", "params": {"rounds": 2, "readers": ["a001"]}},
                {"type": "tests.test_board:OddAgentsSeeNothing", "params": {}},
            ],
            "channels": ["main", "ops"],
        },
    }
    rebuilt = Board.from_spec(spec)
    assert isinstance(rebuilt.topology, Gossip)
    assert isinstance(rebuilt.policies[1], OddAgentsSeeNothing)
    assert rebuilt.spec() == spec
    assert Board.from_spec(spec["params"]).spec() == spec
    assert Board().spec()["params"]["topology"] == {"type": "broadcast", "params": {}}
    assert Board(topology=Groups(size=3)).spec()["params"]["topology"] == {
        "type": "groups", "params": {"size": 3}}


def test_models():
    p = Post(post_id="p0001-0000", round=1, agent="a000", channel="main", text="t")
    assert p.fields == {}
    d = Delivery(delivery_id="d0001-00000", post_id=p.post_id, recipient="a001",
                 eligible_round=2, content_hash="h")
    assert d.read_round is None
