"""World tools that return an awaitable (swarmlab/world/base.py, Async tools): the executor awaits
them, other agents' turns run meanwhile under `commit: async`, sync tools are unaffected, and a
cancellation at run end records nothing."""
from __future__ import annotations

import asyncio
import json
from pathlib import Path

from swarmlab import Experiment, Run
from swarmlab.events import logical_view, parse_event
from swarmlab.ids import ActionId, AgentId
from swarmlab.medium.registry import settle_outcomes
from swarmlab.participants.base import Participant, TurnUsage
from swarmlab.view import text_observation
from swarmlab.world.base import Action, Outcome, World, tool


class SyncTotal(World):
    """A shared counter with a synchronous `add`."""

    def reset(self, rng, agents):
        self.total = 0
        self.applied: list[str] = []

    def observe(self, agent):
        return text_observation(f"total={self.total}")

    @tool("add", "Add n to the total", {"n": "integer"})
    def add(self, agent, n: int) -> Outcome:
        self.total += n
        self.applied.append(agent)
        return Outcome(accepted=True, feedback={"total": self.total})

    def score(self):
        return {"total": self.total}


class AsyncTotal(SyncTotal):
    """`add` is `async def`: it awaits `delays[agent]` event-loop steps (a stand-in for a
    subprocess), forever for the agents in `hang`, then adds."""

    def __init__(self, delays: dict | None = None, hang: list | None = None) -> None:
        self.delays = dict(delays or {})
        self.hang = list(hang or [])

    @tool("add", "Add n to the total", {"n": "integer"})
    async def add(self, agent, n: int) -> Outcome:
        if agent in self.hang:
            await asyncio.Event().wait()  # cancelled at run end
        for _ in range(self.delays.get(agent, 0)):
            await asyncio.sleep(0)
        self.total += n
        self.applied.append(agent)
        return Outcome(accepted=True, feedback={"total": self.total})


class Adder(Participant):
    """Calls `add(1)` then `end_turn` every turn."""

    async def turn(self, view, tools):
        await tools.call("add", {"n": 1})
        await tools.call("end_turn", {})
        return TurnUsage(calls=2)


def _events(run_dir: Path) -> list[dict]:
    return [json.loads(line) for line in (run_dir / "events.jsonl").read_text().splitlines()]


def test_async_tools_interleave_and_commit_in_completion_order(tmp_path):
    world = AsyncTotal(delays={"a000": 6, "a001": 1})
    exp = Experiment(name="atotal", world=world, participants=[Adder(), Adder()])
    run = exp.run(seed=0, max_rounds=10, commit="async", out=tmp_path)
    assert run.meta["end_reason"] == "max_rounds"
    events = _events(run.dir)
    commits = [e for e in events if e["type"] == "action_committed"]
    # a000 called add first (agent order) but its add finished after a001's
    assert commits[0]["agent"] == "a001"
    # true completion order across agents: the running totals in log order are 1, 2, 3, ...
    totals = [e["feedback"]["total"] for e in commits]
    assert totals == list(range(1, len(totals) + 1)) and totals[-1] == run.score["total"]
    # while a000 awaited, a001 closed ticks: a000's first commit lands in a later tick, and its
    # action id still carries the tick the call was made in (1)
    first_a000 = next(e for e in commits if e["agent"] == "a000")
    assert first_a000["round"] > 1 and first_a000["action_id"].startswith("x0001-a000")
    # the tool_returned of the awaited call carries the outcome
    rets = [e for e in events if e["type"] == "tool_returned" and e["agent"] == "a000"
            and e["result"]["result"].get("id") == first_a000["action_id"]]
    assert rets and rets[0]["result"]["result"]["feedback"] == first_a000["feedback"]
    assert Run.load(run.dir).replay()["last_round"] == 10


def test_an_async_tool_that_never_suspends_logs_exactly_like_a_sync_tool(tmp_path):
    for commit in ("round_end", "immediate"):
        views = []
        for world in (SyncTotal(), AsyncTotal()):  # AsyncTotal() never awaits anything
            exp = Experiment(name=f"same-{commit}", world=world, participants=[Adder()] * 3)
            run = exp.run(seed=1, max_rounds=3, commit=commit, out=tmp_path / type(world).__name__)
            views.append([e for e in logical_view(_events_typed(run.dir))
                          if e["type"] not in ("run_started",)])
            assert run.score == {"total": 9}
        assert views[0] == views[1]


def _events_typed(run_dir: Path):
    return [parse_event(line) for line in (run_dir / "events.jsonl").read_text().splitlines()]


def test_world_commit_keeps_order_with_async_and_sync_actions_mixed():
    w = AsyncTotal(delays={"a000": 3})
    w.reset(None, [])
    acts = [(AgentId(a), ActionId(f"x{i}"), Action(name="add", args={"n": 1}))
            for i, a in enumerate(["a000", "a001", "a000"])]
    outs = w.commit(acts)
    assert w.applied == []  # nothing ran yet: the first tool is async, the rest are deferred
    outs = asyncio.run(settle_outcomes(outs))
    assert w.applied == ["a000", "a001", "a000"]
    assert [o.feedback["total"] for o in outs] == [1, 2, 3]
    assert [o.action_id for o in outs] == ["x0", "x1", "x2"]
    # a synchronous world still gets plain outcomes from commit
    s = SyncTotal()
    s.reset(None, [])
    assert all(isinstance(o, Outcome) for o in s.commit(acts))


def test_async_tools_under_round_end_are_awaited_in_commit_order(tmp_path):
    world = AsyncTotal(delays={"a000": 5, "a001": 1, "a002": 3})
    exp = Experiment(name="rend", world=world, participants=[Adder()] * 3)
    run = exp.run(seed=2, max_rounds=2, commit="round_end", out=tmp_path)
    events = _events(run.dir)
    for r in (1, 2):
        order = next(e["order"] for e in events if e["type"] == "round_started" and e["round"] == r)
        committed = [e["agent"] for e in events if e["type"] == "action_committed" and e["round"] == r]
        assert committed == order  # seeded order, not completion order
    totals = [e["feedback"]["total"] for e in events if e["type"] == "action_committed"]
    assert totals == [1, 2, 3, 4, 5, 6]


def test_cancellation_while_an_async_tool_awaits_records_nothing(tmp_path):
    world = AsyncTotal(hang=["a000"])
    exp = Experiment(name="hang", world=world, participants=[Adder(), Adder()])
    run = exp.run(seed=0, max_rounds=3, commit="async", out=tmp_path)
    assert run.meta["end_reason"] == "max_rounds" and run.meta["last_round"] == 3
    events = _events(run.dir)
    assert {e["agent"] for e in events if e["type"] == "action_committed"} == {"a001"}
    assert not any(e.get("agent") == "a000" for e in events
                   if e["type"] in ("turn_started", "tool_called", "tool_returned", "turn_ended"))
    assert run.score == {"total": 3}  # a000's add never applied
    assert Run.load(run.dir).replay()["score"] == {"total": 3}
