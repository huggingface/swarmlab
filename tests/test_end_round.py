"""`World.end_round` and the `round_resolved` event."""
from __future__ import annotations

import json

import pytest
from typer.testing import CliRunner

from swarmlab import (
    Board,
    Experiment,
    Outcome,
    Participant,
    Run,
    TurnUsage,
    World,
    text_observation,
    tool,
)
from swarmlab.cli import app
from swarmlab.events import parse_event
from swarmlab.interventions import InjectPost, PatchPrivate
from swarmlab.metrics.base import Metric
from swarmlab.snapshot import SnapshotStore

from .helpers import logical


class Counter(World):
    """Counts `tick` actions; `end_round` resolves the round (moves `pending` into `total`)."""

    def __init__(self, resolve: bool = True) -> None:
        self.resolve = resolve

    def reset(self, rng, agents):
        self.agents = list(agents)
        self.pending, self.first, self.total = 0, None, 0
        self.calls: list[tuple[str, int]] = []
        self.round = 0

    def begin_round(self, round):
        self.round = round

    def observe(self, agent):
        return text_observation(f"total {self.total}")

    @tool("tick", "Tick once.", {})
    def tick(self, agent):
        if self.first is None:
            self.first = str(agent)
        self.pending += 1
        return Outcome(accepted=True)

    def end_round(self, round):
        n, first = self.pending, self.first
        self.total += n
        self.pending, self.first = 0, None
        self.calls.append(("end_round", round))
        return {"n": n, "first": first, "total": self.total} if self.resolve else None

    def patch_private(self, agent, data):
        self.calls.append(("patch_private", self.round))

    def score(self):
        return {"total": self.total}


class Ticker(Participant):
    """Ticks `1 + (agent index + round) % 3` times, then ends its turn."""

    async def turn(self, view, tools):
        k = 1 + (int(str(self.agent)[1:]) + view.round) % 3
        for _ in range(k):
            await tools.call("tick", {})
        await tools.call("end_turn", {})
        return TurnUsage(calls=k + 1)


class ResolvedTotal(Metric):
    """Folds `round_resolved` events: the world's total, without re-deriving it from actions."""

    entry_point = None
    name = "counter.total"

    def __init__(self) -> None:
        self.total, self.rounds = 0, 0

    def update(self, event):
        if event.type == "round_resolved":
            self.total += event.data["n"]
            self.rounds += 1

    def value(self):
        return (float(self.total) if self.rounds else None), self.rounds


def experiment(resolve=True, metrics=None, interventions=None, n=4):
    return Experiment(name="cnt", world=Counter(resolve=resolve), participants=[Ticker()] * n,
                      medium=Board(topology="broadcast"),
                      metrics=[ResolvedTotal()] if metrics is None else metrics,
                      interventions=interventions or [])


def of_type(run, t):
    return [e for e in run.events if e["type"] == t]


def expected_n(r, n=4):
    return sum(1 + (i + r) % 3 for i in range(n))


@pytest.mark.parametrize("commit", ["round_end", "immediate"])
def test_round_resolved_logged_once_per_round(tmp_path, commit):
    run = experiment().run(seed=3, max_rounds=5, commit=commit, out=tmp_path)
    resolved = of_type(run, "round_resolved")
    assert [e["round"] for e in resolved] == [1, 2, 3, 4, 5]
    assert [e["data"]["n"] for e in resolved] == [expected_n(r) for r in range(1, 6)]
    assert all(e["agent"] is None for e in resolved)
    # first ticker is the first agent of the round's order (actions commit in order)
    orders = {e["round"]: e["order"] for e in of_type(run, "round_started")}
    assert all(e["data"]["first"] == orders[e["round"]][0] for e in resolved)
    # position: right after the last action_committed, before the round's tail
    for r in range(1, 6):
        types = [e["type"] for e in run.events if e["round"] == r]
        i = types.index("round_resolved")
        assert types[i - 1] == "action_committed" and "action_committed" not in types[i:]
        assert types.index("turn_ended") < i and types.index("metric") > i
    assert run.score == {"total": sum(expected_n(r) for r in range(1, 6))}


@pytest.mark.parametrize("commit", ["round_end", "immediate"])
def test_metric_folding_round_resolved_matches_score_and_replays(tmp_path, commit):
    run = experiment().run(seed=3, max_rounds=5, commit=commit, out=tmp_path)
    store = SnapshotStore(run.dir)
    metric = {r: v for r, v, _ in run.metrics["counter.total"]}
    for r in range(1, 6):  # the state end_round left is in that round's snapshot
        world = Counter()
        world.restore(store.load(store.read(r))["world"])
        assert world.pending == 0 and world.calls[-1] == ("end_round", r)
        assert metric[r] == world.score()["total"]
    assert run.replay()["metrics_checked"] == 5
    res = CliRunner().invoke(app, ["replay", str(run.dir), "--json"])
    assert res.exit_code == 0, res.output
    assert json.loads(res.stdout)["replay"] == "ok"


def test_replay_does_not_call_end_round(tmp_path, monkeypatch):
    run = experiment().run(seed=3, max_rounds=3, out=tmp_path)

    def boom(self, round):
        raise AssertionError("replay called end_round")

    monkeypatch.setattr(Counter, "end_round", boom)
    Run.load(run.dir)


def test_fork_and_resume_continue_from_the_resolved_state(tmp_path):
    exp = experiment()
    ref = exp.run(seed=4, max_rounds=6, out=tmp_path / "ref")
    child = ref.fork(at_round=3).run()
    view = lambda run: [e for e in logical(run, ("seq", "ts", "run")) if e["type"] == "round_resolved"]
    assert view(child) == view(ref) and child.score == ref.score

    crashed = exp.run(seed=4, max_rounds=6, out=tmp_path / "crash")
    lines = (crashed.dir / "events.jsonl").read_bytes().splitlines(keepends=True)
    cut = next(i for i, ln in enumerate(lines)
               if (ev := parse_event(ln)).type == "tool_called" and ev.round == 4)
    (crashed.dir / "events.jsonl").write_bytes(b"".join(lines[: cut + 1]))
    resumed = Run(crashed.dir).resume()
    assert logical(resumed) == logical(ref) and resumed.score == ref.score
    Run.load(resumed.dir)


@pytest.mark.parametrize("commit", ["round_end", "immediate"])
def test_round_resolved_is_deterministic(tmp_path, commit):
    a = experiment().run(seed=7, max_rounds=4, commit=commit, out=tmp_path / "a")
    b = experiment().run(seed=7, max_rounds=4, commit=commit, out=tmp_path / "b")
    pick = lambda run: [e for e in logical(run) if e["type"] == "round_resolved"]
    assert pick(a) and pick(a) == pick(b)
    assert logical(a) == logical(b)


@pytest.mark.parametrize("commit", ["round_end", "immediate"])
def test_none_logs_nothing_and_leaves_the_log_unchanged(tmp_path, commit):
    on = experiment(resolve=True, metrics=[]).run(seed=2, max_rounds=3, commit=commit,
                                                   out=tmp_path / "on")
    off = experiment(resolve=False, metrics=[]).run(seed=2, max_rounds=3, commit=commit,
                                                     out=tmp_path / "off")
    assert not of_type(off, "round_resolved") and off.score == on.score
    strip = lambda run, drop: [e for e in logical(run) if e["type"] not in drop]
    drop = {"run_started", "round_resolved"}  # run_started carries the world's params
    assert strip(off, drop) == strip(on, drop)


def test_end_round_runs_before_round_end_interventions(tmp_path):
    ivs = [PatchPrivate(agent="a001", data={"x": 1}, at_round=2), InjectPost(text="hi", at_round=2)]
    run = experiment(interventions=ivs).run(seed=1, max_rounds=3, out=tmp_path)
    world = Counter()
    store = SnapshotStore(run.dir)
    world.restore(store.load(store.read(3))["world"])
    assert world.calls == [("end_round", 1), ("end_round", 2), ("patch_private", 2), ("end_round", 3)]
    r2 = [e for e in run.events if e["round"] == 2]
    types = [e["type"] for e in r2]
    assert types.index("round_resolved") < types.index("intervention")
    assert [e["op"] for e in r2 if e["type"] == "intervention"] == ["patch_private", "inject_post"]
    # the injected post is not part of round 2's resolution (only the agents' ticks are)
    assert next(e for e in r2 if e["type"] == "round_resolved")["data"]["n"] == expected_n(2)


def test_export_puts_round_resolved_in_other(tmp_path):
    import pyarrow.parquet as pq

    from swarmlab.export import export_run

    run = experiment().run(seed=3, max_rounds=2, out=tmp_path)
    out = export_run(run.dir)
    rows = [r for r in pq.read_table(out / "tables" / "other.parquet").to_pylist()
            if r["type"] == "round_resolved"]
    assert [r["round"] for r in rows] == [1, 2]
    payload = json.loads(rows[0]["payload"])
    assert set(payload) == {"data"}
    assert payload["data"]["n"] == payload["data"]["total"] == expected_n(1)


class Ballot(World):
    """Agents `vote` for a/b/c; `end_round` resolves the round to the majority (ties: first name)."""

    def reset(self, rng, agents):
        self.agents = list(agents)
        self.ballots: dict[str, int] = {}
        self.winners: list[str] = []

    def observe(self, agent):
        return text_observation(f"winners {self.winners}")

    @tool("vote", "Vote for a choice.", {"choice": {"type": "string", "enum": ["a", "b", "c"]}})
    def vote(self, agent, choice):
        self.ballots[choice] = self.ballots.get(choice, 0) + 1
        return Outcome(accepted=True)

    def end_round(self, round):
        counts, self.ballots = dict(sorted(self.ballots.items())), {}
        winner = max(counts, key=lambda c: (counts[c], -ord(c)))
        self.winners.append(winner)
        return {"counts": counts, "winner": winner}

    def score(self):
        return {"a_wins": self.winners.count("a"), "rounds": len(self.winners)}


class Voter(Participant):
    """Votes once or twice, the choice following agent index and round."""

    async def turn(self, view, tools):
        i = int(str(self.agent)[1:])
        k = 1 + (i * view.round) % 2
        for j in range(k):
            await tools.call("vote", {"choice": "abc"[(i + j + view.round * (i % 2)) % 3]})
        await tools.call("end_turn", {})
        return TurnUsage(calls=k + 1)


class MajorityCheck(Metric):
    """Accumulates the round's committed votes from `action_committed`; on `round_resolved` checks
    they are complete (equal to the resolution's counts) and that the winner is their majority."""

    entry_point = None
    name = "ballot.checked"

    def __init__(self) -> None:
        self.votes: dict[str, int] = {}
        self.checked = 0

    def update(self, event):
        if event.type == "action_committed" and event.action["name"] == "vote" and event.accepted:
            c = event.action["args"]["choice"]
            self.votes[c] = self.votes.get(c, 0) + 1
        elif event.type == "round_resolved":
            votes, self.votes = dict(sorted(self.votes.items())), {}
            assert votes == event.data["counts"], (votes, event.data)
            assert event.data["winner"] == max(votes, key=lambda c: (votes[c], -ord(c)))
            self.checked += 1

    def value(self):
        return float(self.checked), self.checked


def ballot(n=5):
    return Experiment(name="ballot", world=Ballot(), participants=[Voter()] * n,
                      medium=Board(topology="broadcast"), metrics=[MajorityCheck()])


@pytest.mark.parametrize("commit", ["round_end", "immediate"])
def test_metric_sees_all_committed_votes_before_round_resolved(tmp_path, commit):
    run = ballot().run(seed=5, max_rounds=4, commit=commit, out=tmp_path)
    assert [v for _, v, _ in run.metrics["ballot.checked"]] == [1.0, 2.0, 3.0, 4.0]
    assert len({e["data"]["winner"] for e in of_type(run, "round_resolved")}) > 1
    assert run.replay()["metrics_checked"] == 4
    res = CliRunner().invoke(app, ["replay", str(run.dir), "--json"])
    assert res.exit_code == 0, res.output
    assert json.loads(res.stdout)["replay"] == "ok"

