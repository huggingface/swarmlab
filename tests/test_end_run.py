"""`World.end_run(reason)` (swarmlab/world/base.py): called once on every path that ends a run,
after the final round's `end_round` and the end decision (and, under async, after the in-flight
turns have unwound), before the final budget event, snapshot, score and `run_ended`."""
from __future__ import annotations

import asyncio
import pickle

import pytest

from swarmlab import Experiment, Run
from swarmlab.budget import HardCeilingReached
from swarmlab.participants.base import Participant, TurnUsage
from swarmlab.snapshot import SnapshotStore
from swarmlab.view import text_observation
from swarmlab.world.base import Outcome, World, tool

CALLS: list[str] = []   # end_run calls of runs whose world copy the test cannot read back


class Recording(World):
    """A counter that records its lifecycle calls in its own state; `end_run` writes `ended`."""

    def __init__(self, goal: int = 0, crash_at: int = 0) -> None:
        self.goal, self.crash_at = goal, crash_at

    def reset(self, rng, agents):
        self.total = 0
        self.busy = 0
        self.calls: list[str] = []
        self.ended: str | None = None

    def observe(self, agent):
        return text_observation(f"total={self.total}")

    @tool("add", "Add n to the total", {"n": "integer"})
    def add(self, agent, n: int) -> Outcome:
        self.total += n
        return Outcome(accepted=True, feedback={"total": self.total})

    @tool("hold", "Hold the world busy", {})
    async def hold(self, agent) -> Outcome:
        self.busy += 1
        try:
            await asyncio.Event().wait()
        finally:
            await asyncio.sleep(0)
            self.busy -= 1
        return Outcome(accepted=True, feedback={})

    def end_round(self, round):
        if self.crash_at and round == self.crash_at:
            raise RuntimeError("boom")
        self.calls.append(f"end_round:{round}")

    def terminal(self):
        return bool(self.goal) and self.total >= self.goal

    def end_run(self, reason):
        CALLS.append(reason)
        self.calls.append(f"end_run:{reason}:busy={self.busy}")
        self.ended = reason

    def score(self):
        return {"total": self.total, "ended": self.ended, "busy": self.busy}


class AsyncEnding(Recording):
    async def end_run(self, reason):
        await asyncio.sleep(0)
        super().end_run(reason)


class Adder(Participant):
    async def turn(self, view, tools):
        await tools.call("add", {"n": 1})
        await tools.call("end_turn", {})
        return TurnUsage(calls=2)


class Holder(Participant):
    async def turn(self, view, tools):
        await tools.call("hold", {})
        return TurnUsage(calls=1)


class Ceiling(Participant):
    async def turn(self, view, tools):
        if view.round == 2:
            raise HardCeilingReached("test ceiling")
        await tools.call("add", {"n": 1})
        return TurnUsage(calls=1)


def _final_world(run) -> dict:
    store = SnapshotStore(run.dir)
    latest = store.latest()
    assert latest.round == run.meta["last_round"]
    return pickle.loads(store.load(latest)["world"])


@pytest.mark.parametrize("commit", ["round_end", "immediate"])
def test_end_run_comes_after_the_last_end_round_and_before_the_final_snapshot(tmp_path, commit):
    run = Experiment(name="end", world=Recording(), participants=[Adder(), Adder()]).run(
        seed=0, max_rounds=2, commit=commit, out=tmp_path)
    assert run.end_reason == "max_rounds"
    final = _final_world(run)
    assert final["calls"] == ["end_round:1", "end_round:2", "end_run:max_rounds:busy=0"]
    assert run.score == {"total": 4, "ended": "max_rounds", "busy": 0}
    assert Run.load(run.dir).replay()["score"] == run.score
    types = [e["type"] for e in run.events]
    assert types[-4:] == ["budget", "round_committed", "snapshot", "run_ended"]


def test_end_run_gets_the_terminal_reason(tmp_path):
    run = Experiment(name="end", world=Recording(goal=3), participants=[Adder(), Adder()]).run(
        seed=0, max_rounds=9, commit="immediate", out=tmp_path)
    assert run.end_reason == "terminal" and run.score["ended"] == "terminal"


def test_async_end_run_is_awaited_after_the_in_flight_turns_unwound(tmp_path):
    # a000 holds the world busy forever; a001's turn closes tick 1, the last (max_rounds), so
    # a000's turn is cancelled and its hold undone before end_run and the final snapshot
    run = Experiment(name="end", world=AsyncEnding(), participants=[Holder(), Adder()]).run(
        seed=0, max_rounds=1, commit="async", out=tmp_path)
    assert run.end_reason == "max_rounds" and run.meta["last_round"] == 1
    final = _final_world(run)
    assert final["calls"] == ["end_round:1", "end_run:max_rounds:busy=0"]
    assert run.score == {"total": 1, "ended": "max_rounds", "busy": 0}
    assert Run.load(run.dir).replay()["score"] == run.score


def test_end_run_on_hard_ceiling_and_on_a_crash(tmp_path):
    CALLS.clear()
    run = Experiment(name="end", world=Recording(), participants=[Ceiling()]).run(
        seed=0, max_rounds=5, commit="immediate", out=tmp_path / "ceiling")
    assert run.end_reason == "hard_ceiling" and CALLS == ["hard_ceiling"]
    CALLS.clear()
    with pytest.raises(RuntimeError, match="boom"):
        Experiment(name="end", world=Recording(crash_at=2), participants=[Adder()]).run(
            seed=0, max_rounds=5, commit="immediate", out=tmp_path / "crash")
    assert CALLS == ["crash"]


def test_end_run_is_called_once_per_run(tmp_path):
    CALLS.clear()
    run = Experiment(name="end", world=Recording(), participants=[Adder()]).run(
        seed=0, max_rounds=3, commit="round_end", out=tmp_path)
    assert run.end_reason == "max_rounds" and CALLS == ["max_rounds"]
