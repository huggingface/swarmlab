"""`commit: async` (swarmlab/runner.py, Async commit): every agent loops over its own turns, all
at once, with no round barrier; Minesweeper's per-turn accounting; replay, view, report, resume."""
from __future__ import annotations

import asyncio
import copy
import json
from pathlib import Path

import pytest
from typer.testing import CliRunner

from swarmlab import Budget, Experiment, Run
from swarmlab.blobs import BlobStore
from swarmlab.cli import app
from swarmlab.participants import LLMAgent
from swarmlab.participants.base import Participant, TurnUsage
from swarmlab.participants.scripted import EvidenceAggregator
from swarmlab.providers.fake import FakeProvider
from swarmlab.rng import derive
from swarmlab.snapshot import SnapshotStore
from swarmlab.spec import spec_hash
from swarmlab.view import text_observation
from swarmlab.world.base import Outcome, World, tool
from swarmlab.world.coloring import ColoringGrid
from swarmlab.world.flaggame import FlagGame
from swarmlab.world.minesweeper import HIDDEN, Minesweeper, deduce

from .helpers import TEST_PRICING

REPO = Path(__file__).resolve().parent.parent
ASYNC_SPEC = REPO / "examples" / "minesweeper_async.yaml"
ROUND_SPEC = REPO / "examples" / "minesweeper.yaml"
AGENTS = [f"a{i:03d}" for i in range(4)]
cli = CliRunner()

# what the Slow participants did, in real (event loop) order; reset by each test that uses them
TRACE: list[tuple] = []
INFLIGHT: set[str] = set()


class Total(World):
    """A shared counter: `add(n)` commits at once under async; the observation shows the total."""

    def reset(self, rng, agents):
        self.total = 0

    def observe(self, agent):
        return text_observation(f"total={self.total}", total=self.total)

    @tool("add", "Add n to the total", {"n": "integer"})
    def add(self, agent, n: int) -> Outcome:
        self.total += n
        return Outcome(accepted=True, feedback={"total": self.total})

    def score(self):
        return {"total": self.total}


class Slow(Participant):
    """Adds 1, then waits `steps` event-loop steps (a stand-in for model latency), then ends."""

    def __init__(self, steps: int = 1) -> None:
        self.steps = steps

    async def turn(self, view, tools):
        seen = int(view.observation.parts[0].text.split("=")[1])
        TRACE.append(("start", self.agent, seen, sorted(INFLIGHT)))
        INFLIGHT.add(self.agent)
        try:
            await tools.call("add", {"n": 1})
            TRACE.append(("added", self.agent))
            for _ in range(self.steps):
                await asyncio.sleep(0)
            await tools.call("end_turn", {})
        finally:
            INFLIGHT.discard(self.agent)
        return TurnUsage(calls=2)


def _events(run_dir: Path) -> list[dict]:
    return [json.loads(line) for line in (run_dir / "events.jsonl").read_text().splitlines()]


def _check_tick_layout(events: list[dict], last: int) -> None:
    """One agent per tick, ticks 1..last, every event's round = the round of the next commit."""
    starts = [e for e in events if e["type"] == "round_started"]
    assert [e["round"] for e in starts] == list(range(1, last + 1))
    assert all(len(e["order"]) == 1 for e in starts)
    pending = []
    for e in events:
        if e["type"] in ("run_started", "run_ended", "inference_attempt", "inference_response"):
            continue
        pending.append(e["round"])
        if e["type"] == "round_committed":
            assert set(pending) == {e["round"]}, pending
            pending = []
        elif e["type"] == "snapshot":
            pending = []
    assert sorted(e["seq"] for e in events) == [e["seq"] for e in events]


# ---- 1. overlap and immediate visibility -----------------------------------------------------

def test_async_turns_overlap_and_every_commit_is_visible_at_once(tmp_path):
    TRACE.clear()
    INFLIGHT.clear()
    exp = Experiment(name="total", world=Total(), participants=[Slow(steps=6), Slow(steps=1)])
    run = exp.run(seed=0, max_rounds=8, commit="async", out=tmp_path)
    assert run.meta["end_reason"] == "max_rounds" and run.meta["last_round"] == 8
    adds, overlapped = 0, False
    for entry in TRACE:
        if entry[0] == "added":
            adds += 1
            continue
        _, agent, seen, inflight = entry
        assert seen == adds  # every add so far, in-flight turns' included, is in the observation
        overlapped |= agent == "a001" and "a000" in inflight
    assert overlapped  # a001 started a turn while a000's was in flight, and saw a000's add
    events = _events(run.dir)
    _check_tick_layout(events, 8)
    orders = [e["order"][0] for e in events if e["type"] == "round_started"]
    assert orders.count("a001") > orders.count("a000") >= 1  # the faster agent closes more ticks
    # true commit order across agents: the running totals in log order are 1, 2, 3, ...
    totals = [e["feedback"]["total"] for e in events if e["type"] == "action_committed"]
    assert totals == list(range(1, len(totals) + 1)) and totals[-1] == run.score["total"]
    # a turn cancelled at the end is not a tick: one turn_ended per tick
    assert sum(e["type"] == "turn_ended" for e in events) == 8
    assert Run.load(run.dir).replay()["last_round"] == 8


def test_rounds_per_agent_is_a_per_agent_turn_cap(tmp_path):
    TRACE.clear()
    INFLIGHT.clear()
    exp = Experiment(name="capped", world=Total(), participants=[Slow(steps=6), Slow(steps=1)],
                     options={"commit": "async", "rounds_per_agent": 3})
    run = exp.run(seed=0, out=tmp_path)
    assert run.meta["end_reason"] == "max_rounds" and run.meta["last_round"] == 6
    orders = [e["order"][0] for e in _events(run.dir) if e["type"] == "round_started"]
    assert orders.count("a000") == orders.count("a001") == 3


# ---- 2. Minesweeper under async ----------------------------------------------------------------

def ms_async(seed=0, agents=AGENTS, **kw) -> Minesweeper:
    w = Minesweeper(**kw)
    w.set_commit_mode("async")
    w.reset(derive(seed, "world"), list(agents))
    w.begin_round(1)
    return w


def _hidden_safe(w):
    return [(x, y) for y in range(w.height) for x in range(w.width)
            if w.board[y][x] == HIDDEN and not w.minefield[y][x]]


def _newly_provable_case():
    """(world, s, c): a002 revealing provably safe `s` makes hidden `c` provably safe."""
    for seed in range(40):
        w = ms_async(seed)
        safe0, _ = deduce(w.board)
        for s in sorted(safe0):
            probe = copy.deepcopy(w)
            probe.observe("a002")
            probe.reveal("a002", *s)
            safe1, _ = deduce(probe.board)
            new = [c for c in sorted(safe1 - safe0) if probe.board[c[1]][c[0]] == HIDDEN]
            if new:
                return w, s, new[0]
    raise AssertionError("no board where a reveal proves a new cell")


def test_minesweeper_async_judges_a_reveal_against_the_board_the_agent_observed():
    w, s, c = _newly_provable_case()
    w.observe("a000")                       # a000's turn starts: c is not provable yet
    w.observe("a002")
    assert w.reveal("a002", *s).feedback["result"] == "opened"
    fresh = copy.deepcopy(w)
    w.reveal("a000", *c)                    # judged against a000's turn-start board: a guess
    w.close_turn("a000", 1)
    data = w.end_round(1)
    assert data["agent"] == "a000" and data["agent_turns"] == 1
    assert (data["reveals"], data["productive"], data["certain"], data["guesses"]) == (1, 1, 0, 1)
    assert data["ideal_reveals"] == 1       # min(reveals_per_round, safe cells a000 saw)
    fresh.observe("a000")                   # a new turn sees s open, which proves c: certain
    fresh.reveal("a000", *c)
    fresh.close_turn("a000", 1)
    assert fresh.end_round(1)["certain"] == 1
    w.close_turn("a002", 2)                 # a002's turn: its own tally, its own board
    d2 = w.end_round(2)
    assert d2["agent"] == "a002" and d2["certain"] == 1 and d2["guesses"] == 0


def test_minesweeper_async_reveal_cap_is_per_turn_not_per_tick():
    w = ms_async(seed=1)
    safe = sorted(deduce(w.board)[0])
    assert len(safe) >= 3
    w.observe("a000")
    assert w.reveal("a000", *safe[0]).accepted
    w.begin_round(2)                        # another agent's turn closed a tick: same turn
    out = w.reveal("a000", *safe[1])
    assert not out.accepted and out.feedback["error"] == "reveals_per_round"
    assert out.feedback["detail"] == "at most 1 reveal per turn"
    w.observe("a000")                       # the next turn: a fresh budget
    assert w.reveal("a000", *safe[1]).accepted
    assert "at most 1 cell per turn" in w.description()


def test_minesweeper_async_collision_versus_stale():
    w = ms_async(seed=2)
    x, y = min(deduce(w.board)[0])
    w.observe("a000")
    w.observe("a001")
    w.reveal("a001", x, y)                  # opened after a000 observed: a collision for a000
    assert w.reveal("a000", x, y).feedback == {"result": "already_revealed", "x": x, "y": y,
                                               "collision": True}
    w.observe("a000")                       # a000 now sees it open: stale
    assert w.reveal("a000", x, y).feedback["collision"] is False
    w.close_turn("a000", 1)
    d = w.end_round(1)
    assert (d["collisions"], d["stale"]) == (0, 1)


def test_minesweeper_async_ideal_is_per_turn():
    w = ms_async(seed=0, safe_opening=False)   # nothing provable: one forced guess
    w.observe("a000")
    w.close_turn("a000", 1)
    assert w.end_round(1)["ideal_reveals"] == 1
    w = ms_async(seed=0, reveals_per_round=3)
    n_safe = len(deduce(w.board)[0])
    w.observe("a001")
    w.close_turn("a001", 1)
    d = w.end_round(1)
    assert d["ideal_reveals"] == min(3, n_safe) and d["provably_safe"] == n_safe
    assert d["n_agents"] == 4 and d["reveals"] == 0


def test_minesweeper_round_modes_keep_their_state_layout():
    w = Minesweeper()
    w.reset(derive(0, "world"), AGENTS)
    assert not hasattr(w, "turn_view") and w.commit_mode == "round_end"


def _replay_reveals(run, events):
    """Re-apply the logged world actions, in log order, to a fresh world: each feedback must
    come out the same, so the log holds the true commit order."""
    spec = run.spec.world
    w = Minesweeper(**spec.params)
    w.reset(derive(run.spec.options.seed, "world"), [f"a{i:03d}" for i in range(len(run.spec.participants))])
    for i, e in enumerate(e for e in events if e["type"] == "action_committed"):
        w.begin_round(i + 1)                # a fresh per-round budget for every action
        out = getattr(w, e["action"]["name"])(e["agent"], **e["action"]["args"])
        got = {k: v for k, v in out.feedback.items() if k != "collision"}
        want = {k: v for k, v in e["feedback"].items() if k != "collision"}
        assert got == want, (e["seq"], got, want)
    return w


@pytest.mark.parametrize("arm", ["random4", "allocator8"])
def test_minesweeper_async_run_end_to_end(tmp_path, arm):
    exp = Experiment.from_yaml(ASYNC_SPEC, arm)
    run = exp.run(0, out=tmp_path)
    n = len(exp.participants)
    meta, sc = run.meta, run.score
    assert meta["end_reason"] == "terminal" and sc["won"] == 1
    assert sc["cleared_round"] == meta["last_round"] < 30 * n
    events = _events(run.dir)
    _check_tick_layout(events, meta["last_round"])
    resolved = [e for e in events if e["type"] == "round_resolved"]
    orders = [e["order"][0] for e in events if e["type"] == "round_started"]
    assert [e["round"] for e in resolved] == list(range(1, meta["last_round"] + 1))
    assert [e["data"]["agent"] for e in resolved] == orders  # round_resolved is the closing turn
    assert all(e["data"]["reveals"] <= 1 and e["data"]["agent_turns"] == 1 for e in resolved)
    assert sc["collisions"] > 0             # the provider's `yields` made the turns overlap
    # model calls cancelled at the end are charged at their reservation, and run.json has them
    paid = sum(e["cost_usd"] for e in events if e["type"] == "inference_response")
    answered = {e["call_id"] for e in events if e["type"] == "inference_response"}
    cut = [e for e in events if e["type"] == "inference_attempt" and e["call_id"] not in answered]
    assert meta["ledger"]["reserved"] == 0
    assert meta["ledger"]["swarm"] == pytest.approx(paid + sum(e["reserved_usd"] for e in cut))
    met = {}
    for e in events:
        if e["type"] == "metric":
            met.setdefault(e["name"], []).append(e["value"])
    assert met["minesweeper.collisions"][-1] == sc["collisions"]
    assert met["minesweeper.mine_hits"][-1] == sc["mine_hits"]
    assert met["minesweeper.allocation_efficiency"][-1] == pytest.approx(sc["allocation_efficiency"])
    assert met["minesweeper.productive_rate"][-1] == pytest.approx(
        sc["productive_reveals"] / meta["last_round"])  # per agent-turn
    final = _replay_reveals(run, events)
    snaps = SnapshotStore(run.dir)
    restored = Minesweeper(**run.spec.world.params)
    restored.restore(snaps.load(snaps.latest())["world"])
    assert final.board == restored.board and final._safe_left() == 0
    assert Run.load(run.dir).replay()["score"] == sc
    # the same run again is identical: `yields` is deterministic
    again = exp.run(0, out=tmp_path / "again")
    strip = lambda evs: [{k: v for k, v in e.items() if k not in ("ts", "run_spec")}
                         for e in evs if e["type"] not in ("inference_attempt", "inference_response")]
    assert strip(_events(again.dir)) == strip(events)


# ---- 3. CLI: run, replay, view, report on 4 and 8 agents ----------------------------------------

def test_cli_replay_view_and_report_on_async_runs(tmp_path):
    out = tmp_path / "runs"
    for arm in ("random4", "allocator8"):
        res = cli.invoke(app, ["run", str(ASYNC_SPEC), "--arm", arm, "--seed", "1", "--out", str(out),
                               "--yes"])
        assert res.exit_code == 0, res.output
    for run_dir in sorted(p for p in out.iterdir() if p.is_dir()):
        res = cli.invoke(app, ["replay", str(run_dir), "--json"])
        assert res.exit_code == 0, res.output
        assert json.loads(res.output)["replay"] == "ok"
        meta = json.loads((run_dir / "run.json").read_text())
        replayed = Run(run_dir).replay()
        assert replayed["metrics_checked"] > 0 and replayed["score"] == meta["score"]
        assert replayed["last_round"] == meta["last_round"]
    res = cli.invoke(app, ["view", str(out / "minesweeper-async__allocator8__s1")])
    assert res.exit_code == 0, res.output
    assert (out / "minesweeper-async__allocator8__s1" / "view.html").stat().st_size > 1000
    res = cli.invoke(app, ["report", str(out), "--include-fake"])
    assert res.exit_code == 0, res.output
    assert "## Minesweeper: outcome, mines and allocation" in res.output
    assert res.output.count("| minesweeper-async__") >= 2


# ---- 4. resume -----------------------------------------------------------------------------------

def test_resume_of_an_async_run_restarts_the_agent_loops_from_the_snapshot(tmp_path):
    run = Experiment.from_yaml(ASYNC_SPEC, "allocator4").run(2, out=tmp_path)
    assert run.meta["last_round"] > 8
    log = run.dir / "events.jsonl"
    lines = log.read_bytes().splitlines(keepends=True)
    cut = next(i for i, line in enumerate(lines)
               if json.loads(line)["type"] == "round_committed" and json.loads(line)["round"] == 6)
    assert json.loads(lines[cut + 1])["type"] == "snapshot"
    prefix = b"".join(lines[:cut + 2])
    log.write_bytes(prefix + b"".join(lines[cut + 2:cut + 9]) + lines[cut + 9][:15])  # crash in tick 7
    resumed = Run(run.dir).resume()
    assert resumed.status == "ended" and resumed.end_reason == "terminal"
    assert resumed.score["won"] == 1 and resumed.score["cleared_round"] == resumed.meta["last_round"]
    assert log.read_bytes().startswith(prefix)
    discarded = [json.loads(line) for line in (run.dir / "discarded.jsonl").read_text().splitlines()]
    logical = [e for e in discarded if e["type"] not in ("inference_attempt", "inference_response")]
    assert logical and {e["round"] for e in logical} == {7}
    events = _events(run.dir)
    _check_tick_layout(events, resumed.meta["last_round"])
    assert Run.load(run.dir).replay()["score"] == resumed.score


def test_hard_ceiling_under_async_ends_at_the_last_tick_and_resumes(tmp_path):
    exp = Experiment(
        name="ms-ceiling", world=Minesweeper(on_mine="continue"),
        participants=[LLMAgent(model="fake:minesweeper_random", max_tokens=256)] * 4,
        metrics=["minesweeper.cleared", "minesweeper.collisions"], budget=Budget(hard_usd=0.2),
        providers={"fake": FakeProvider(pricing=TEST_PRICING, yields=[0, 10])})
    run = exp.run(seed=0, max_rounds=200, commit="async", out=tmp_path)
    assert run.end_reason == "hard_ceiling" and run.meta["last_round"] >= 1
    events = _events(run.dir)
    last = run.meta["last_round"]
    assert [e["round"] for e in events if e["type"] == "round_committed"][-1] == last
    assert not [e for e in events if e["round"] > last and e["type"] not in (
        "inference_attempt", "inference_response")]
    resumed = run.resume(budget=Budget(hard_usd=50.0))
    assert resumed.end_reason == "terminal" and resumed.score["won"] == 1
    Run.load(run.dir).replay()


def test_estimate_counts_one_turn_per_tick():
    exp = Experiment.from_yaml(ASYNC_SPEC, "random4")
    rounds = Experiment.from_yaml(ROUND_SPEC, "random")  # same 4 agents, one turn each per round
    assert exp.estimate(0, 40)["calls"] * 4 == rounds.estimate(0, 40)["calls"]


def _texts(content) -> list[str]:
    if isinstance(content, str):
        return [content]
    return [part.get("text") or "" for part in content if part.get("type") == "text"]


def test_llm_agents_get_turn_based_prompts_under_async(tmp_path):
    run = Experiment.from_yaml(ASYNC_SPEC, "allocator4").run(0, max_rounds=12, out=tmp_path)
    blobs = BlobStore(run.dir / "blobs")
    events = _events(run.dir)
    heads: dict[str, list[int]] = {}
    for e in events:
        if e["type"] != "inference_attempt":
            continue
        req = json.loads(blobs.get(e["request_hash"]))
        system = req["messages"][0]
        assert system["role"] == "system"
        text = "\n".join(_texts(system["content"]))
        assert "You act in turns, at your own pace, at the same time as the other agents." in text
        assert "Call `end_turn` when you are done for this turn;" in text
        assert "runs in rounds" not in text and "for this round" not in text
        assert {t["name"]: t["description"] for t in req["tools"]}["end_turn"] == "Finish your turn."
        user = [m for m in req["messages"] if m["role"] == "user"][-1]
        head = _texts(user["content"])[0]
        assert head.startswith("Turn ") and "Round" not in head
        heads.setdefault(e["agent"], []).append(int(head.split()[1].rstrip(".")))
    done = {a: sum(1 for e in events if e["type"] == "turn_ended" and e["agent"] == a) for a in heads}
    for agent, numbers in heads.items():  # the agent's own turn count, not the global tick
        assert numbers == list(range(1, len(numbers) + 1)), (agent, numbers)
        assert len(numbers) in (done[agent], done[agent] + 1)  # + a turn cancelled at the end
    assert max(len(n) for n in heads.values()) < 12
    res = cli.invoke(app, ["prompts", str(ASYNC_SPEC), "--arm", "random4"])
    assert res.exit_code == 0, res.output
    assert "You act in turns, at your own pace" in res.output and "runs in rounds" not in res.output
    assert "--- turn 1 user message ---\nTurn 1." in res.output
    assert "- end_turn: Finish your turn.\n" in res.output
    res = cli.invoke(app, ["prompts", str(ROUND_SPEC), "--arm", "random"])
    assert "The environment runs in rounds." in res.output and "--- round 1 user message ---\nRound 1." in res.output
    assert "- end_turn: Finish your turn for this round.\n" in res.output


# ---- 5. spec hashes and refusals -----------------------------------------------------------------

def test_existing_spec_hashes_are_unchanged():
    want = {"random": "c0ad3850b1e73319d584a41efa52d45004008d3bb00750c2af819e15e3bb2227",
            "allocator": "74697908b0da90fb63e6652acf08acd62c49be5ee110c8bdc2de3cedfe45ba17"}
    for arm, exp in Experiment.arms_from_yaml(ROUND_SPEC).items():
        opts = exp._options(0, None)
        assert spec_hash(exp.to_spec(**opts.model_dump())) == want[arm]
    assert FakeProvider().spec()["params"] == {"script": None, "pricing": None, "concurrency": 8}


def test_round_based_worlds_refuse_async_at_construction():
    with pytest.raises(ValueError, match=r"world FlagGame requires commit in \[round_end, "
                                         r"immediate\], got async"):
        Experiment(name="f", world=FlagGame(), participants=[EvidenceAggregator()] * 2,
                   options={"commit": "async"})
    flag = Experiment(name="f", world=FlagGame(), participants=[EvidenceAggregator()] * 2)
    with pytest.raises(ValueError, match="requires commit in"):
        flag.run(seed=0, max_rounds=2, commit="async")
    with pytest.raises(ValueError, match="world ColoringGrid requires commit=round_end, got async"):
        Experiment(name="c", world=ColoringGrid(), participants=[EvidenceAggregator()] * 3,
                   options={"commit": "async"})
    with pytest.raises(ValueError, match="commit: async does not support options.scheduler"):
        Experiment(name="t", world=Total(), participants=[Slow()] * 2,
                   options={"commit": "async", "scheduler": "one_speaker"})
    with pytest.raises(ValueError, match="commit: async does not support probes"):
        Experiment(name="t", world=FlagGame(), participants=[EvidenceAggregator()] * 2,
                   probes=["belief"]).check_run(commit="async")
