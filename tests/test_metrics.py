import math

import pytest

from swarmlab.events import (
    ActionCommittedEvent,
    DeliveryEvent,
    PostEvent,
    ReadEvent,
    RoundStartedEvent,
    TurnStartedEvent,
)
from swarmlab.metrics.base import get
from swarmlab.metrics.belief import Accuracy, Consensus, Entropy, Polarization
from swarmlab.metrics.comm import Hops, PostsPerRound, ReadRate


def guess(agent, cand, accepted=True, round=1):
    return ActionCommittedEvent(run="r", round=round, agent=agent, action_id="x",
                                action={"name": "guess", "args": {"candidate": cand}},
                                accepted=accepted, feedback={})


def feed(metric, events):
    for e in events:
        metric.update(e)
    return metric.value()


def test_registry_resolves_entry_points():
    for name, cls in [("belief.accuracy", Accuracy), ("belief.consensus", Consensus),
                      ("belief.polarization", Polarization), ("belief.entropy", Entropy),
                      ("comm.read_rate", ReadRate), ("comm.posts_per_round", PostsPerRound),
                      ("comm.hops", Hops)]:
        m = get(name)
        assert isinstance(m, cls) and m.name == name and m.spec()["type"] == name
    assert get("belief.polarization", threshold=0.3).spec()["params"] == {"threshold": 0.3}


def test_belief_metrics_use_latest_accepted_guess():
    evs = [guess("a000", "A"), guess("a001", "A"), guess("a002", "B"), guess("a003", "C"),
           guess("a003", "A", round=2), guess("a002", "C", accepted=False, round=2),
           ActionCommittedEvent(run="r", round=2, agent="a004", action_id="y",
                                action={"name": "paint", "args": {}}, accepted=True)]
    acc = Accuracy()
    acc.set_truth({"truth": "A"})
    assert acc.needs_truth() and not Consensus().needs_truth()
    assert feed(acc, evs) == (0.75, 4)
    assert feed(Consensus(), evs) == (0.75, 4)
    assert feed(Polarization(), evs) == (2.0, 4)      # A 75%, B 25%
    assert feed(Polarization(threshold=0.3), evs) == (1.0, 4)
    h, n = feed(Entropy(), evs)
    assert n == 4 and math.isclose(h, -(0.75 * math.log2(0.75) + 0.25 * math.log2(0.25)))


def test_belief_metrics_empty():
    for m in (Accuracy(), Consensus(), Polarization(), Entropy()):
        assert m.value() == (None, 0)
    assert feed(Entropy(), [guess("a000", "A")]) == (0.0, 1)


def test_belief_denominator_is_all_live_agents_with_explicit_none():
    """B2: 4 live agents, 2 guess: agents without a guess are a 'none' category."""
    evs = [guess("a000", "A"), guess("a001", "B"), guess("a009", "A")]  # a009 is not live
    live = ["a000", "a001", "a002", "a003"]

    def run(m):
        m.set_agents(live)
        return feed(m, evs)

    acc = Accuracy()
    acc.set_truth({"truth": "A"})
    assert run(acc) == (0.25, 4)
    assert run(Consensus()) == (0.25, 4)            # never-guessers lower consensus
    assert run(Polarization()) == (2.0, 4)          # A, B at 25%; "none" (50%) is not a camp
    assert run(Polarization(threshold=0.3)) == (0.0, 4)
    h, n = run(Entropy())
    assert n == 4 and math.isclose(h, 1.5)          # {A: .25, B: .25, none: .5}
    nobody = Consensus()
    nobody.set_agents(live)
    assert nobody.value() == (0.0, 4)
    silent = Entropy()
    silent.set_agents(live)
    assert silent.value() == (0.0, 4)
    empty = Accuracy()
    empty.set_agents([])
    assert empty.value() == (None, 0)


def rs(r):
    return RoundStartedEvent(run="r", round=r, order=[])


def ts(r, a):
    return TurnStartedEvent(run="r", round=r, agent=a)


def read(r, a, ids):
    return ReadEvent(run="r", round=r, agent=a, delivery_ids=ids)


def post(r, a, pid):
    return PostEvent(run="r", round=r, agent=a, post_id=pid, channel="main", text="t")


def deliv(r, pid, did, to):
    return DeliveryEvent(run="r", round=r, agent=to, post_id=pid, recipient=to, delivery_id=did,
                         eligible_round=r + 1, content_hash="0" * 64)


def test_read_rate_and_posts_per_round_are_per_round():
    r1 = [rs(1), ts(1, "a000"), read(1, "a000", []), read(1, "a000", []), ts(1, "a001"),
          post(1, "a001", "p1")]
    rr, ppr = ReadRate(), PostsPerRound()
    assert feed(rr, r1) == (0.5, 2)
    assert feed(ppr, r1) == (1.0, 2)
    r2 = [rs(2), ts(2, "a000"), ts(2, "a001")]
    assert feed(rr, r2) == (0.0, 2)
    assert feed(ppr, r2) == (0.0, 2)
    assert ReadRate().value() == (None, 0)


def test_hops_chain():
    h = Hops()
    assert h.value() == (None, 0)
    # round 1: a000 posts p1 delivered to a001
    feed(h, [rs(1), ts(1, "a000"), post(1, "a000", "p1"), deliv(1, "p1", "d1", "a001")])
    assert h.value() == (1.0, 1)
    # round 2: a001 reads d1, then posts p2 -> hop 2; delivered to a002
    feed(h, [rs(2), ts(2, "a001"), read(2, "a001", ["d1"]), post(2, "a001", "p2"),
             deliv(2, "p2", "d2", "a002")])
    assert h.value() == (2.0, 1)
    # round 3: a002 reads d2 and posts p3 -> hop 3
    feed(h, [rs(3), ts(3, "a002"), read(3, "a002", ["d2"]), post(3, "a002", "p3")])
    assert h.value() == (3.0, 1)


def test_hops_immediate_same_round_reads_resolve_lazily():
    h = Hops()
    # immediate mode log shape: turns (with reads) first, then posts, then deliveries
    evs = [rs(1), ts(1, "a000"), ts(1, "a001"), read(1, "a001", ["d1"]),
           post(1, "a000", "p1"), post(1, "a001", "p2"), deliv(1, "p1", "d1", "a001")]
    assert feed(h, evs) == (2.0, 2)
    state = h.snapshot()
    h.update(rs(2))
    assert h.value() == (2.0, 0)
    h2 = Hops()
    h2.restore(state)
    assert h2.value() == (2.0, 2)


@pytest.mark.parametrize("cls", [Consensus, Hops, ReadRate])
def test_metric_snapshot_roundtrip(cls):
    m = cls()
    feed(m, [rs(1), ts(1, "a000"), guess("a000", "A"), read(1, "a000", [])])
    n = cls()
    n.restore(m.snapshot())
    assert n.value() == m.value()
