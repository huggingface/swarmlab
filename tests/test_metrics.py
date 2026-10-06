import math

import pytest

from swarmlab.events import (
    ActionCommittedEvent,
    DeliveryEvent,
    PostEvent,
    ReadEvent,
    RoundStartedEvent,
    ToolCalledEvent,
    ToolReturnedEvent,
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


def post(r, a, pid, ack=None):
    return PostEvent(run="r", round=r, agent=a, post_id=pid, provisional_id=ack, channel="main",
                     text="t")


def pcall(r, a, ack):
    """The post tool call and its ack, as the executor logs them inside the turn."""
    cid = f"c{r}-{a}-{ack}"
    return [ToolCalledEvent(run="r", round=r, agent=a, call_id=cid, tool="post", args={"text": "t"}),
            ToolReturnedEvent(run="r", round=r, agent=a, call_id=cid,
                              result={"ok": True, "result": {"id": ack}, "error": None}, pending=True)]


def deliv(r, pid, did, to, eligible=None):
    return DeliveryEvent(run="r", round=r, agent=to, post_id=pid, recipient=to, delivery_id=did,
                         eligible_round=r + 1 if eligible is None else eligible, content_hash="0" * 64)


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
    feed(h, [rs(1), ts(1, "a000"), *pcall(1, "a000", "t0"), post(1, "a000", "p1", "t0"),
             deliv(1, "p1", "d1", "a001")])
    assert h.value() == (1.0, 0)  # nothing read yet
    # round 2: a001 reads d1, then posts p2 -> hop 2; delivered to a002
    feed(h, [rs(2), ts(2, "a001"), read(2, "a001", ["d1"]), *pcall(2, "a001", "t0"),
             post(2, "a001", "p2", "t0"), deliv(2, "p2", "d2", "a002")])
    assert h.value() == (2.0, 1)
    # round 3: a002 reads d2 and posts p3 -> hop 3
    feed(h, [rs(3), ts(3, "a002"), read(3, "a002", ["d2"]), *pcall(3, "a002", "t0"),
             post(3, "a002", "p3", "t0")])
    assert h.value() == (3.0, 2)


def test_hops_ignores_reads_after_the_post_in_the_same_turn():
    """A6: post then read in one turn: the read does not feed that post."""
    h = Hops()
    feed(h, [rs(1), ts(1, "a000"), *pcall(1, "a000", "t0"), post(1, "a000", "p1", "t0"),
             deliv(1, "p1", "d1", "a001")])
    feed(h, [rs(2), ts(2, "a001"), *pcall(2, "a001", "t0"), read(2, "a001", ["d1"]),
             post(2, "a001", "p2", "t0")])
    assert h.value() == (1.0, 1)
    # without call events (older logs) only reads in earlier rounds count
    old = Hops()
    feed(old, [rs(1), post(1, "a000", "p1"), deliv(1, "p1", "d1", "a001"),
               rs(2), read(2, "a001", ["d1"]), post(2, "a001", "p2")])
    assert old.value() == (1.0, 1)


def test_hops_immediate_same_round_reads_resolve_lazily():
    h = Hops()
    # immediate mode log shape: turns (with reads) first, then posts, then deliveries
    evs = [rs(1), ts(1, "a000"), *pcall(1, "a000", "p1"), ts(1, "a001"), read(1, "a001", ["d1"]),
           *pcall(1, "a001", "p2"), post(1, "a000", "p1", "p1"), post(1, "a001", "p2", "p2"),
           deliv(1, "p1", "d1", "a001", eligible=1)]
    assert feed(h, evs) == (2.0, 1)
    state = h.snapshot()
    h.update(rs(2))
    assert h.value() == (2.0, 1)
    h2 = Hops()
    h2.restore(state)
    assert h2.value() == (2.0, 1)


@pytest.mark.parametrize("cls", [Consensus, Hops, ReadRate])
def test_metric_snapshot_roundtrip(cls):
    m = cls()
    feed(m, [rs(1), ts(1, "a000"), guess("a000", "A"), read(1, "a000", [])])
    n = cls()
    n.restore(m.snapshot())
    assert n.value() == m.value()


def _hops(tmp_path, participant, commit, name):
    from swarmlab import Board, Experiment
    from swarmlab.world.flaggame import FlagGame

    exp = Experiment(name=name, world=FlagGame(), medium=Board(topology="broadcast"),
                     participants=[participant] * 3, metrics=["comm.hops"])
    run = exp.run(seed=0, max_rounds=5, out=tmp_path, commit=commit)
    run.replay()
    return [(e["value"], e["denominator"]) for e in run.events if e["type"] == "metric"]


def test_hops_post_then_read_round_end(tmp_path):
    """A6 end to end, the reviewer's post-then-read participant, 3 agents, broadcast, round_end.

    Relay chain: a round-r post can only reflect reads made in rounds < r (its own turn reads
    after posting), and a round-r post is first readable in round r+1. So one hop costs two rounds:
      r1 posts: 1 (nothing read yet)           r1 reads: nothing eligible
      r2 posts: 1 (r1 read nothing)            r2 reads: r1 posts (hop 1)
      r3 posts: 2 (read hop-1 posts in r2)     r3 reads: r2 posts (hop 1)
      r4 posts: 2 (best read so far is hop 1)  r4 reads: r3 posts (hop 2)
      r5 posts: 3
    giving [1, 1, 2, 2, 3]. The review note quoted [1, 1, 2, 3, 4], which would need a hop-2 post
    to be read before the round-4 posts; under round_end a round-3 post is first readable in
    round 4, after that round's posts. The old code gave [1, 2, 3, 4, 5] (it let the same-turn read
    feed the post). Denominator: posts read at least once (3 per round from round 2 on, cumulative).
    """
    from .helpers import PostThenRead

    got = _hops(tmp_path, PostThenRead(), "round_end", "ptr")
    assert [v for v, _ in got] == [1.0, 1.0, 2.0, 2.0, 3.0]
    assert [n for _, n in got] == [0, 3, 6, 9, 12]


def test_hops_read_then_post_round_end(tmp_path):
    """Reading before posting relays every round: [1, 2, 3, 4, 5]."""
    from .helpers import ReadThenPost

    got = _hops(tmp_path, ReadThenPost(), "round_end", "rtp")
    assert [v for v, _ in got] == [1.0, 2.0, 3.0, 4.0, 5.0]
