"""The `wait` tool (swarmlab/executor.py, `WaitHub`): parks an async turn until a new post is
delivered to the agent, its timeout, or nobody left could post; refused outside `commit: async`."""
from __future__ import annotations

import asyncio
import json
import threading
from pathlib import Path

import pytest

from swarmlab import Experiment, Run
from swarmlab.blobs import BlobStore
from swarmlab.executor import RoundExecutor, WaitHub
from swarmlab.medium.board import Board
from swarmlab.participants.base import Participant, TurnUsage
from swarmlab.roles import Role, assign

from .test_async_tools import SyncTotal

ROLES = {"lead": Role(name="lead"),
         "worker": Role(name="worker", tools=["wait", "read_board", "post", "add"])}
RESULTS: dict[str, list] = {}


class Lead(Participant):
    """First turn: `delay` loop steps (or `sleep_s` seconds), then posts "go" (when `post`);
    later turns just end."""

    def __init__(self, delay: int = 5, sleep_s: float = 0.0, post: bool = True) -> None:
        self.delay, self.sleep_s, self.post = delay, sleep_s, post
        self.turns = 0

    async def turn(self, view, tools):
        self.turns += 1
        if self.turns == 1:
            for _ in range(self.delay):
                await asyncio.sleep(0)
            if self.sleep_s:
                await asyncio.sleep(self.sleep_s)
            if self.post:
                await tools.call("post", {"text": "go"})
        await tools.call("end_turn", {})
        return TurnUsage(calls=2)


class Waiter(Participant):
    """`wait(timeout_s)` `n` times, then `read_board`, then `end_turn`; results go to RESULTS."""

    def __init__(self, timeout_s: float | None = None, n: int = 1) -> None:
        self.timeout_s, self.n = timeout_s, n

    async def turn(self, view, tools):
        args = {} if self.timeout_s is None else {"timeout_s": self.timeout_s}
        for _ in range(self.n):
            res = await tools.call("wait", args)
            RESULTS.setdefault(self.agent, []).append(("wait", res.ok, res.result, res.error))
        res = await tools.call("read_board", {})
        RESULTS.setdefault(self.agent, []).append(("read_board", res.ok, res.result, res.error))
        await tools.call("end_turn", {})
        return TurnUsage(calls=self.n + 2)


def _exp(participants, roles, name="wait", **options) -> Experiment:
    parts = [assign(p, r) for p, r in zip(participants, roles, strict=True)]
    return Experiment(name=name, world=SyncTotal(), participants=parts, roles=ROLES,
                      options={"commit": "async", **options})


def _events(run_dir: Path) -> list[dict]:
    return [json.loads(line) for line in (run_dir / "events.jsonl").read_text().splitlines()]


def _waits(events, agent):
    calls = {e["call_id"]: e for e in events if e["type"] == "tool_called" and e["agent"] == agent}
    return [e["result"] for e in events if e["type"] == "tool_returned"
            and calls.get(e["call_id"], {}).get("tool") == "wait"]


def test_wait_returns_the_lead_post_and_read_board_then_has_nothing_new(tmp_path):
    RESULTS.clear()
    run = _exp([Lead(), Waiter()], ["lead", "worker"]).run(seed=0, max_rounds=4, out=tmp_path)
    assert run.meta["end_reason"] == "max_rounds"
    (kind, ok, res, _), (kind2, ok2, res2, _) = RESULTS["a001"][:2]
    assert kind == "wait" and ok and res["timed_out"] is False and res["waited_s"] >= 0
    assert [p["content"] for p in res["posts"]] == ["go"]
    assert set(res["posts"][0]) == {"delivery_id", "post_id", "eligible_round", "content"}
    assert kind2 == "read_board" and ok2 and res2["items"] == []  # the wait was a read
    events = _events(run.dir)
    reads = [e for e in events if e["type"] == "read" and e["agent"] == "a001"]
    assert reads[0]["delivery_ids"] == [res["posts"][0]["delivery_id"]]
    assert _waits(events, "a001")[0]["result"] == res  # logged with the result
    # replay is exact: it never waits (it never calls tools)
    assert Run.load(run.dir).replay()["last_round"] == 4


def test_wait_times_out(tmp_path):
    RESULTS.clear()
    exp = _exp([Lead(delay=0, sleep_s=0.3, post=False), Waiter(timeout_s=0.05)], ["lead", "worker"])
    run = exp.run(seed=0, max_rounds=2, out=tmp_path)
    assert run.meta["end_reason"] == "max_rounds"
    _, ok, res, _ = RESULTS["a001"][0]
    assert ok and res["timed_out"] is True and res["posts"] == [] and res["waited_s"] >= 0.05


def test_wait_returns_at_once_when_nobody_left_could_post(tmp_path):
    RESULTS.clear()
    # alone: returns at once
    run = _exp([Waiter()], ["worker"]).run(seed=0, max_rounds=2, out=tmp_path / "solo")
    assert run.meta["end_reason"] == "max_rounds"
    assert all(r[2]["timed_out"] and r[2]["posts"] == [] for r in RESULTS["a000"] if r[0] == "wait")
    # two waiters: the second to wait finds the other waiting and returns at once
    RESULTS.clear()
    run = _exp([Waiter(), Waiter()], ["worker", "worker"]).run(seed=0, max_rounds=3,
                                                               out=tmp_path / "two")
    assert run.meta["end_reason"] == "max_rounds"
    second = [r for r in RESULTS["a001"] if r[0] == "wait"]
    assert second and all(r[2]["timed_out"] and r[2]["posts"] == [] for r in second)
    # the other agent finishing (rounds_per_agent) releases a waiter that nobody can post to
    RESULTS.clear()
    exp = _exp([Waiter(), Lead(post=False)], ["worker", "lead"], rounds_per_agent=1)
    run = exp.run(seed=0, out=tmp_path / "finish")
    assert run.meta["end_reason"] == "max_rounds" and run.meta["last_round"] == 2
    assert RESULTS["a000"][0][2]["timed_out"] is True


def test_wait_counts_against_max_calls_per_turn(tmp_path):
    RESULTS.clear()
    run = _exp([Waiter(n=3)], ["worker"], max_calls_per_turn=2).run(seed=0, max_rounds=1,
                                                                     out=tmp_path)
    ended = [e for e in _events(run.dir) if e["type"] == "turn_ended"]
    assert ended[0]["yield_kind"] == "cap" and ended[0]["calls"] == 3


def test_cancellation_at_run_end_releases_a_wait(tmp_path):
    RESULTS.clear()
    run = _exp([Lead(post=False), Waiter()], ["lead", "worker"]).run(seed=0, max_rounds=3,
                                                                    out=tmp_path)
    assert run.meta["end_reason"] == "max_rounds" and run.meta["last_round"] == 3
    assert "a001" not in RESULTS  # still waiting when the run ended
    events = _events(run.dir)
    assert not any(e.get("agent") == "a001" for e in events
                   if e["type"] in ("turn_started", "tool_called", "turn_ended"))
    assert Run.load(run.dir).replay()["last_round"] == 3


@pytest.mark.parametrize("commit", ["immediate", "round_end"])
def test_wait_is_refused_outside_async(commit):
    with pytest.raises(ValueError, match="role 'worker'.*wait.*commit: async"):
        Experiment(name="w", world=SyncTotal(), participants=[assign(Waiter(), "worker")],
                   roles=ROLES, options={"commit": commit})
    # the run's own commit override is checked too
    exp = _exp([Waiter()], ["worker"])
    with pytest.raises(ValueError, match="role 'worker'"):
        exp.check_run(commit=commit)


def test_wait_schema_is_offered_only_when_the_role_lists_it_under_async(tmp_path):
    world = SyncTotal()
    world.reset(None, ["a000", "a001"])
    board = Board()
    roles = {"a000": ROLES["worker"], "a001": Role(name="all")}

    def names(waits, agent):
        ex = RoundExecutor(run="r", round=1, world=world, board=board,
                           blobs=BlobStore(tmp_path / "b"), agents=["a000", "a001"],
                           commit="immediate", roles=roles, waits=waits)
        return {s.name: s for s in ex.schemas(agent)}

    hub = WaitHub()
    offered = names(hub, "a000")
    assert offered["wait"].description == ("Wait until a new message is delivered to you (or the "
                                           "timeout); returns the new messages.")
    assert offered["wait"].strict_violations() == []
    assert "wait" not in names(hub, "a001")  # tools: None does not include wait
    assert "wait" not in names(None, "a000")  # no hub: not an async run


def test_llm_agent_received_memory_ingests_wait_posts():
    from swarmlab.participants import LLMAgent

    agent = LLMAgent(model="fake:reader", memory="received")
    item = {"delivery_id": "d0001-00000", "post_id": "p0001-0000", "eligible_round": 1,
            "content": "go"}
    agent._note_read("wait", {"posts": [item], "timed_out": False, "waited_s": 0.0})
    assert [m["content"] for m in agent.received] == ["go"]


# ---- permits: a parked wait gives its concurrency slot back (WaitHub, Permits) ----------------

HUBS: list[WaitHub] = []


class _Hub(WaitHub):
    def __init__(self, *args, **kw) -> None:
        super().__init__(*args, **kw)
        HUBS.append(self)


def _run(exp: Experiment, monkeypatch, deadline_s: float = 30.0, **kw):
    """`exp.run(**kw)` in a thread, failing (not hanging) when it is not done in `deadline_s`;
    records the run's WaitHub in HUBS."""
    import swarmlab.runner

    HUBS.clear()
    monkeypatch.setattr(swarmlab.runner, "WaitHub", _Hub)
    out: list = []
    t = threading.Thread(target=lambda: out.append(exp.run(**kw)), daemon=True)
    t.start()
    t.join(deadline_s)
    assert not t.is_alive(), "the run hung"
    assert out, "the run raised"
    return out[0]


class Chatter(Participant):
    """Posts "n" every turn, after a few loop steps."""

    async def turn(self, view, tools):
        for _ in range(3):
            await asyncio.sleep(0)
        await tools.call("post", {"text": f"{view.turn}"})
        await tools.call("end_turn", {})
        return TurnUsage(calls=2)


def test_a_parked_wait_lets_the_poster_take_the_only_slot(tmp_path, monkeypatch):
    RESULTS.clear()
    exp = _exp([Waiter(), Lead()], ["worker", "lead"], concurrency=1, rounds_per_agent=1)
    run = _run(exp, monkeypatch, seed=0, out=tmp_path)
    assert run.meta["end_reason"] == "max_rounds" and run.meta["last_round"] == 2
    _, ok, res, _ = RESULTS["a000"][0]
    assert ok and res["timed_out"] is False and [p["content"] for p in res["posts"]] == ["go"]
    assert HUBS[0].free_permits() == 1 and HUBS[0].running == set()
    assert Run.load(run.dir).replay()["last_round"] == 2


def test_waiters_holding_every_slot_let_a_queued_poster_in(tmp_path, monkeypatch):
    RESULTS.clear()
    exp = _exp([Waiter(), Waiter(), Lead()], ["worker", "worker", "lead"], concurrency=2,
               rounds_per_agent=1)
    run = _run(exp, monkeypatch, seed=0, out=tmp_path)
    assert run.meta["end_reason"] == "max_rounds" and run.meta["last_round"] == 3
    for a in ("a000", "a001"):
        _, ok, res, _ = RESULTS[a][0]
        assert ok and res["timed_out"] is False and [p["content"] for p in res["posts"]] == ["go"]
    assert HUBS[0].free_permits() == 2
    assert run.score == {"total": 0}


def test_a_wait_cancelled_at_run_end_with_its_permit_released_leaks_nothing(tmp_path, monkeypatch):
    RESULTS.clear()
    exp = _exp([Lead(post=False), Waiter()], ["lead", "worker"], concurrency=1)
    run = _run(exp, monkeypatch, seed=0, max_rounds=3, out=tmp_path)
    assert run.meta["end_reason"] == "max_rounds" and run.meta["last_round"] == 3
    assert "a001" not in RESULTS  # parked (its permit given up) when the run ended
    assert not any(e.get("agent") == "a001" for e in _events(run.dir)
                   if e["type"] in ("turn_started", "tool_called", "turn_ended"))
    assert HUBS[0].free_permits() == 1 and HUBS[0].running == set()
    assert Run.load(run.dir).replay()["last_round"] == 3


def test_many_waits_leave_every_permit_free(tmp_path, monkeypatch):
    RESULTS.clear()
    exp = _exp([Chatter(), Waiter(), Waiter()], ["lead", "worker", "worker"], concurrency=1)
    run = _run(exp, monkeypatch, seed=0, max_rounds=30, out=tmp_path)
    assert run.meta["end_reason"] == "max_rounds" and run.meta["last_round"] == 30
    waits = [r for a in ("a001", "a002") for r in RESULTS.get(a, []) if r[0] == "wait"]
    assert len(waits) >= 10 and any(not r[2]["timed_out"] for r in waits)
    assert HUBS[0].free_permits() == 1 and HUBS[0].running == set()


def test_hub_release_is_idempotent_and_cancelled_acquire_holds_nothing():
    async def go():
        hub = WaitHub(1)
        await hub.acquire("a")
        hub.release("a")
        hub.release("a")  # not held any more: no second release
        assert hub.free_permits() == 1
        await hub.acquire("a")
        blocked = asyncio.create_task(hub.acquire("b"))
        await asyncio.sleep(0)
        blocked.cancel()
        with pytest.raises(asyncio.CancelledError):
            await blocked
        hub.release("b")  # cancelled while acquiring: holds nothing
        hub.finished("a")  # a loop that ends gives its permit back
        assert hub.free_permits() == 1
        assert WaitHub().free_permits() is None  # unbounded

    asyncio.run(go())


# ---- hub membership follows the agent loops --------------------------------------------------

class KillTotal(SyncTotal):
    """`killed_at(r)` returns `kills[r]`."""

    def __init__(self, kills: dict[int, list[str]]) -> None:
        self.kills = kills

    def killed_at(self, round):
        return list(self.kills.get(round, []))


def test_an_agent_killed_when_the_first_tick_opens_is_not_a_poster(tmp_path, monkeypatch):
    RESULTS.clear()
    exp = Experiment(name="kill1", world=KillTotal({1: ["a001"]}),
                     participants=[assign(Waiter(), "worker"), assign(Lead(), "lead")],
                     roles=ROLES, options={"commit": "async", "rounds_per_agent": 1})
    run = _run(exp, monkeypatch, seed=0, out=tmp_path)
    _, ok, res, _ = RESULTS["a000"][0]
    assert ok and res["timed_out"] is True and res["waited_s"] < 1
    assert HUBS[0].running == set()
    assert run.meta["end_reason"] == "max_rounds" and run.meta["last_round"] == 1


def test_a_later_kill_of_the_only_other_agent_releases_a_waiter(tmp_path, monkeypatch):
    RESULTS.clear()
    exp = Experiment(name="kill2", world=KillTotal({2: ["a001"]}),
                     participants=[assign(Waiter(), "worker"), assign(Lead(post=False), "lead")],
                     roles=ROLES, options={"commit": "async"})
    run = _run(exp, monkeypatch, seed=0, max_rounds=3, out=tmp_path)
    assert run.meta["end_reason"] == "max_rounds" and run.meta["last_round"] == 3
    first = RESULTS["a000"][0][2]
    assert first["timed_out"] is True and first["posts"] == []
    starts = [e for e in _events(run.dir) if e["type"] == "round_started"]
    assert [e["order"] for e in starts] == [["a001"], ["a000"], ["a000"]]
    assert HUBS[0].running == set() and HUBS[0].free_permits() == 32
    assert Run.load(run.dir).replay()["last_round"] == 3
