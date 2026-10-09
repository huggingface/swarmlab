from swarmlab.ids import agent_id
from swarmlab.medium.board import Post
from swarmlab.medium.topology import TOPOLOGIES, Broadcast, Gossip, Groups
from swarmlab.rng import derive

AGENTS = [agent_id(i) for i in range(8)]


def post(agent: str, n: int = 0) -> Post:
    return Post(post_id=f"p0001-{n:04d}", round=1, agent=agent, channel="main", text="hi")


def test_broadcast_everyone_but_author():
    assert Broadcast().recipients(post("a003"), AGENTS, 1, derive(0, "topology", 1)) == [
        a for a in AGENTS if a != "a003"
    ]


def test_gossip_k_partners_never_author():
    topo = Gossip(k=2)
    rng = derive(5, "topology", 1)
    for a in AGENTS:
        rec = topo.recipients(post(a), AGENTS, 1, rng)
        assert len(rec) == 2 and a not in rec and len(set(rec)) == 2
        assert set(rec) <= set(AGENTS)


def test_gossip_deterministic_given_seed():
    def schedule(seed: int, rnd: int) -> dict:
        topo = Gossip(k=1)
        rng = derive(seed, "topology", rnd)
        return {a: topo.recipients(post(a), AGENTS, rnd, rng) for a in AGENTS}

    assert schedule(1, 3) == schedule(1, 3)
    # different seeds or rounds almost surely differ for 8 agents
    assert schedule(1, 3) != schedule(2, 3) or schedule(1, 3) != schedule(1, 4)


def test_gossip_stable_across_posts_in_round_and_call_order():
    topo = Gossip(k=1)
    rng = derive(9, "topology", 2)
    first = topo.recipients(post("a001", 0), AGENTS, 2, rng)
    rng.random()  # the runner consuming the rng must not change the round's schedule
    for n in range(1, 5):
        assert topo.recipients(post("a001", n), AGENTS, 2, rng) == first
    # querying authors in another order gives the same assignment
    other = Gossip(k=1)
    rng2 = derive(9, "topology", 2)
    for a in reversed(AGENTS):
        other.recipients(post(a), AGENTS, 2, rng2)
    assert other.recipients(post("a001"), AGENTS, 2, rng2) == first
    # a fresh rng from the same label (e.g. repeated immediate-mode commits) agrees too
    assert Gossip(k=1).recipients(post("a001"), AGENTS, 2, derive(9, "topology", 2)) == first


def test_gossip_new_round_recomputes():
    topo = Gossip(k=1)
    seen = set()
    for rnd in range(1, 10):
        seen.add(tuple(topo.recipients(post("a000"), AGENTS, rnd, derive(4, "topology", rnd))))
    assert len(seen) > 1


def test_gossip_k_larger_than_population():
    rec = Gossip(k=10).recipients(post("a000"), AGENTS[:3], 1, derive(0, "topology", 1))
    assert sorted(rec) == ["a001", "a002"]


def test_gossip_cache_not_in_snapshot():
    topo = Gossip(k=1)
    topo.recipients(post("a000"), AGENTS, 1, derive(0, "topology", 1))
    fresh = Gossip(k=1)
    assert topo.snapshot() == fresh.snapshot()


def test_groups_by_index():
    topo = Groups(size=3)
    rng = derive(0, "topology", 1)
    assert topo.recipients(post("a000"), AGENTS, 1, rng) == ["a001", "a002"]
    assert topo.recipients(post("a004"), AGENTS, 1, rng) == ["a003", "a005"]
    assert topo.recipients(post("a007"), AGENTS, 1, rng) == ["a006"]
    # only live agents
    assert topo.recipients(post("a004"), ["a003", "a004"], 1, rng) == ["a003"]


def test_specs_and_registry():
    assert Broadcast().spec() == {"type": "broadcast", "params": {}}
    assert Gossip().spec() == {"type": "gossip", "params": {"k": 1}}
    assert Groups(size=4).spec() == {"type": "groups", "params": {"size": 4}}
    assert set(TOPOLOGIES) == {"broadcast", "gossip", "groups", "star", "tree", "rooms"}
