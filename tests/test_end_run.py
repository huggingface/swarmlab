"""`World.end_run(reason)` (swarmlab/world/base.py): called once on every path that ends a run,
after the final round's `end_round` and the end decision (and, under async, after the in-flight
turns have unwound), before the final budget event, snapshot, score and `run_ended`."""
from __future__ import annotations

import asyncio
import json
import pickle

import pytest

from swarmlab import Budget, Experiment, Run
from swarmlab.budget import HardCeilingReached
from swarmlab.participants.base import Participant, TurnUsage
from swarmlab.providers.base import ChatMessage, ChatRequest
from swarmlab.providers.fake import FakeProvider
from swarmlab.snapshot import SnapshotStore
from swarmlab.view import text_observation
from swarmlab.world.base import Outcome, World, tool

from .helpers import TEST_PRICING

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


# ---- a run that ends at a round boundary without running a round (INTERFACE §8, §13) ----------
@pytest.mark.parametrize("commit", ["round_end", "async"])
def test_a_fork_ending_at_its_boundary_refreshes_the_snapshot_with_end_run(tmp_path, commit):
    # the fork copies round 1's snapshot (ended=None); max_rounds=1 already holds, so the child
    # runs no round: end_run's state must reach the round-1 snapshot as well as the score
    parent = Experiment(name="end", world=Recording(), participants=[Adder()]).run(
        seed=0, max_rounds=2, commit=commit, out=tmp_path)
    copied = (parent.dir / "snapshots" / "000001.json").read_bytes()
    child = parent.fork(at_round=1).run(max_rounds=1)
    assert child.end_reason == "max_rounds" and child.meta["last_round"] == 1
    final = _final_world(child)
    assert final["calls"] == ["end_round:1", "end_run:max_rounds:busy=0"]
    assert child.score == {"total": 1, "ended": "max_rounds", "busy": 0}
    assert Run.load(child.dir).replay()["score"] == child.score
    # rewritten in place: same file, same log_seq (round 1's round_committed), no new snapshot
    # event (the copied one names the same file), no new round
    store = SnapshotStore(child.dir)
    assert store.list_rounds() == [1]
    manifest = store.read(1)
    assert manifest.log_seq == json.loads(copied)["log_seq"]
    events = list(child.events_all)
    assert [e.seq for e in events if e.type == "round_committed"] == [manifest.log_seq]
    assert [e.round for e in events if e.type == "snapshot"] == [1]
    assert events[-1].type == "run_ended"
    # forking the ended child at its last round restores the state after end_run
    grandchild = Run.load(child.dir).fork(at_round=1).run(max_rounds=2)
    assert _final_world(grandchild)["calls"] == [
        "end_round:1", "end_run:max_rounds:busy=0", "end_round:2", "end_run:max_rounds:busy=0"]


class PaidCeiling(Participant):
    """Round 1 pays for one fake model call and adds 1; round 2 hits the hard ceiling."""

    async def turn(self, view, tools):
        if view.round == 2:
            raise HardCeilingReached("test ceiling")
        await tools.infer(ChatRequest(model="fake:reader", max_tokens=8, messages=[
            ChatMessage(role="user", content=f"Round {view.round}.")]))
        await tools.call("add", {"n": 1})
        return TurnUsage(calls=1)


@pytest.mark.parametrize("commit", ["round_end", "async"])
def test_a_resume_whose_soft_budget_already_holds_ends_with_a_refreshed_snapshot(tmp_path, commit):
    # the hard ceiling leaves round 1's snapshot without end_run (ended=None); the resume's soft
    # budget is already spent, so it ends at the boundary with soft_budget and no new round
    run = Experiment(name="end", world=Recording(), participants=[PaidCeiling()],
                     providers={"fake": FakeProvider(pricing=TEST_PRICING)}).run(
        seed=0, max_rounds=5, commit=commit, out=tmp_path)
    assert run.end_reason == "hard_ceiling" and run.meta["last_round"] == 1
    assert _final_world(run)["ended"] is None
    spent = run.spend["swarm"]
    assert spent > 0
    resumed = run.resume(budget=Budget(soft_usd=spent / 2))
    assert resumed.end_reason == "soft_budget" and resumed.meta["last_round"] == 1
    assert _final_world(resumed)["calls"] == ["end_round:1", "end_run:soft_budget:busy=0"]
    assert resumed.score == {"total": 1, "ended": "soft_budget", "busy": 0}
    assert Run.load(resumed.dir).replay()["score"] == resumed.score
    events = list(resumed.events)
    assert [e["round"] for e in events if e["type"] == "snapshot"] == [1]
    assert [e["round"] for e in events if e["type"] == "round_committed"] == [1]
