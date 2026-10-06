"""Sequential commit policy: same-round visibility, one agent at a time."""
from swarmlab import Board, Experiment
from swarmlab.world.flaggame import FlagGame

from .helpers import Chatter


def test_immediate_posts_are_readable_in_the_same_round(tmp_path):
    exp = Experiment(name="chat", world=FlagGame(), participants=[Chatter()] * 4, medium=Board(),
                     metrics=["comm.hops", "comm.read_rate"])
    run = exp.run(seed=2, max_rounds=2, commit="immediate", out=tmp_path)
    events = list(run.events)
    r1 = [e for e in events if e["round"] == 1]
    order = next(e["order"] for e in r1 if e["type"] == "round_started")
    reads = {e["agent"]: e["delivery_ids"] for e in r1 if e["type"] == "read"}
    # the i-th agent in the order sees the i earlier posts of this round
    for i, a in enumerate(order):
        assert len(reads[a]) == i
    deliveries = [e for e in r1 if e["type"] == "delivery"]
    assert deliveries and all(d["eligible_round"] == 1 for d in deliveries)
    # tool results were not pending
    rets = [e for e in r1 if e["type"] == "tool_returned"]
    assert rets and not any(e["pending"] for e in rets)
    kinds = {e["yield_kind"] for e in events if e["type"] == "turn_ended"}
    assert kinds == {"no_tool"}
    # hops: the last poster read posts of earlier posters who read earlier ones
    hops = {r: v for r, v, _ in run.metrics["comm.hops"]}
    assert hops[1] == 4.0


def test_round_end_posts_are_next_round(tmp_path):
    exp = Experiment(name="chat", world=FlagGame(), participants=[Chatter()] * 4)
    run = exp.run(seed=2, max_rounds=2, out=tmp_path)
    events = list(run.events)
    assert all(len(e["delivery_ids"]) == 0 for e in events if e["type"] == "read" and e["round"] == 1)
    assert all(len(e["delivery_ids"]) == 3 for e in events if e["type"] == "read" and e["round"] == 2)
    rets = [e for e in events if e["type"] == "tool_returned" and e["round"] == 1]
    posts = [e for e in rets if e["result"]["result"].get("id", "").startswith("tmp-")]
    assert len(posts) == 4 and all(e["pending"] for e in posts)
