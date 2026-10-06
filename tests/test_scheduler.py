from swarmlab.ids import agent_id
from swarmlab.rng import derive
from swarmlab.scheduler import Scheduler, SeededShuffle

AGENTS = [agent_id(i) for i in range(10)]


def test_base_scheduler_keeps_order():
    assert Scheduler().order(1, AGENTS, derive(0, "schedule", 1)) == AGENTS


def test_seeded_shuffle_is_a_deterministic_permutation():
    s = SeededShuffle()
    live = list(AGENTS)
    a = s.order(3, live, derive(7, "schedule", 3))
    b = s.order(3, live, derive(7, "schedule", 3))
    assert a == b
    assert sorted(a) == AGENTS
    assert live == AGENTS  # input not mutated


def test_seeded_shuffle_uses_the_passed_rng():
    s = SeededShuffle()
    orders = {tuple(s.order(r, AGENTS, derive(7, "schedule", r))) for r in range(1, 6)}
    assert len(orders) > 1
    assert s.spec() == {"type": "seeded_shuffle", "params": {}}


def test_scheduler_snapshot_roundtrip():
    s = SeededShuffle()
    t = SeededShuffle()
    t.restore(s.snapshot())
    assert t.order(1, AGENTS, derive(1, "schedule", 1)) == s.order(1, AGENTS, derive(1, "schedule", 1))
