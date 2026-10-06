"""M1b acceptance 1 (determinism, replay, resume after SIGKILL) and 6 (isolation and delivery)
with real `LLMAgent(model="fake:reader")` participants."""
import json
import os
import pickle
import re
import signal
import subprocess
import sys
from pathlib import Path

import pytest

from swarmlab import Board, Run
from swarmlab.events import EventLog, parse_event
from swarmlab.medium.board import DelayPolicy
from swarmlab.participants import LLMAgent
from swarmlab.probes import BeliefProbe
from swarmlab.providers.base import text_of
from swarmlab.providers.fake import FakeProvider
from swarmlab.snapshot import SnapshotStore
from swarmlab.world.flaggame import FlagGame

from .helpers import PAUSE_AT_ENV, PAUSE_ENV, PausingFlagGame, llm_agent_experiment, logical

REPO = Path(__file__).resolve().parent.parent
N, ROUNDS, SEED, PAUSE_AT = 8, 6, 7, 4

CHILD = """
import sys
from tests.helpers import PausingFlagGame, llm_agent_experiment
from swarmlab.probes import BeliefProbe
llm_agent_experiment({n}, world=PausingFlagGame(), probes=[BeliefProbe()], name="acc1").run(
    seed={seed}, max_rounds={rounds}, out=sys.argv[1])
"""


def experiment(world=None):
    return llm_agent_experiment(N, world=world or PausingFlagGame(), probes=[BeliefProbe()], name="acc1")


@pytest.fixture
def provider_calls(monkeypatch):
    calls: list = []
    original = FakeProvider.complete

    async def counting(self, request):
        calls.append(request)
        return await original(self, request)

    monkeypatch.setattr(FakeProvider, "complete", counting)
    return calls


def test_acceptance_1_determinism_and_replay(tmp_path, provider_calls):
    a = experiment().run(seed=SEED, max_rounds=ROUNDS, out=tmp_path / "a")
    b = experiment().run(seed=SEED, max_rounds=ROUNDS, out=tmp_path / "b")
    assert a.end_reason == "max_rounds"
    assert logical(a) == logical(b)
    assert a.score == b.score and a.spend == b.spend and a.metrics == b.metrics
    assert sum(1 for e in a.events if e["type"] == "turn_ended") == N * ROUNDS
    made = len(provider_calls)
    assert made == 2 * a.spend["calls"]
    loaded = Run.load(a.dir)
    assert len(provider_calls) == made  # replay makes zero provider calls
    assert loaded.metrics == a.metrics and loaded.score == a.score


def test_acceptance_1_resume_after_sigkill_hits_cache(tmp_path, provider_calls):
    out = tmp_path / "crashed"
    marker = tmp_path / "paused"
    env = {**os.environ, PAUSE_ENV: str(marker), PAUSE_AT_ENV: str(PAUSE_AT), "PYTHONPATH": str(REPO)}
    stderr = tmp_path / "child.stderr"
    with open(stderr, "wb") as err:
        proc = subprocess.Popen(
            [sys.executable, "-c", CHILD.format(n=N, seed=SEED, rounds=ROUNDS), str(out)],
            cwd=REPO, env=env, stdout=subprocess.DEVNULL, stderr=err)
    run_dir = out / f"acc1__s{SEED}"
    try:
        from .test_recovery import _wait_for

        _wait_for(marker, proc, stderr)
    finally:
        os.kill(proc.pid, signal.SIGKILL)
        proc.wait()
    log = run_dir / "events.jsonl"
    assert EventLog(log).last_committed()[0] == PAUSE_AT - 1
    crashed = [parse_event(x) for x in log.read_bytes().splitlines()]
    paid = [e for e in crashed if e.type == "inference_response" and e.round == PAUSE_AT and not e.cached]
    assert len(paid) >= N  # every turn of the crashed round finished before the kill

    resumed = Run(run_dir, experiment=experiment()).resume()
    assert resumed.end_reason == "max_rounds"
    events = list(resumed.events_all)
    attempts = {e.call_id: e for e in events if e.type == "inference_attempt" and e.round == PAUSE_AT}
    swarm = [e for e in events if e.type == "inference_response" and e.round == PAUSE_AT
             and attempts[e.call_id].category == "swarm"]
    assert len(swarm) == len(paid) and all(e.cached for e in swarm)
    # after the resume only round PAUSE_AT's probes (never sent before the kill) and later rounds
    # reached the provider
    later = [r for r in provider_calls if not r.tools]
    assert len(later) == N * (ROUNDS - PAUSE_AT + 1)

    reference = experiment().run(seed=SEED, max_rounds=ROUNDS, out=tmp_path / "clean")
    assert logical(resumed) == logical(reference)
    assert resumed.score == reference.score and resumed.metrics == reference.metrics
    assert resumed.spend == reference.spend
    before = len(provider_calls)
    Run.load(run_dir)
    assert len(provider_calls) == before


def _bare(name: str, text: str) -> bool:
    return re.search(rf"(?<![A-Za-z0-9]){re.escape(name)}(?![A-Za-z0-9])", text) is not None


def test_acceptance_6_isolation(tmp_path, provider_calls, monkeypatch):
    views = []
    original_turn = LLMAgent.turn

    async def recording(self, view, tools):
        views.append(view)
        return await original_turn(self, view, tools)

    monkeypatch.setattr(LLMAgent, "turn", recording)
    for seed in (1, 2):
        provider_calls.clear()
        views.clear()
        world = FlagGame()
        exp = llm_agent_experiment(6, world=world, probes=[BeliefProbe()], name=f"iso{seed}",
                                   medium=Board(topology="gossip"))
        run = exp.run(seed=seed, max_rounds=4, out=tmp_path)
        truth = run.score["truth"]
        names = [chr(ord("A") + i) for i in range(8)]  # FlagGame's default candidate names
        assert truth in names
        assert views and all(v.observation.private == {} for v in views)
        texts = [v.model_dump_json() for v in views]
        for req in provider_calls:
            for m in req.messages:
                if m.role != "assistant":  # the agents' own outputs may name any candidate
                    texts.append(text_of(m.content))
        returned = [json.dumps(e["result"]) for e in run.events if e["type"] == "tool_returned"]
        questions = [BeliefProbe().question(f"a{i:03d}", r) for i in range(6) for r in range(1, 5)]
        for text in texts + returned + questions:
            low = text.lower()
            assert "truth" not in low and "correct" not in low and "crop_y" not in low
            stripped = text.replace("\\n", "\n")
            for n in names:
                stripped = stripped.replace(f"\n{n}:\n", "\n")
            assert not _bare(truth, stripped), text[:200]


def test_acceptance_6_delayed_delivery(tmp_path, provider_calls):
    exp = llm_agent_experiment(4, medium=Board(topology="broadcast", policies=[DelayPolicy(rounds=2)]))
    run = exp.run(seed=3, max_rounds=5, out=tmp_path)
    events = list(run.events)
    eligible = {e["delivery_id"]: e["eligible_round"] for e in events if e["type"] == "delivery"}
    posted = {e["post_id"]: e["round"] for e in events if e["type"] == "post"}
    assert posted and set(posted.values()) == {1}  # the reader posts its crop once
    assert set(eligible.values()) == {4}
    reads = [(e["round"], d) for e in events if e["type"] == "read" for d in e["delivery_ids"]]
    assert reads and all(r >= eligible[d] for r, d in reads)
    assert {r for r, _ in reads} == {4}
    # what the agents' models saw from the board matches: nothing before round 4
    for req in provider_calls:
        rnd = max(int(m.content[0].text.split()[1].rstrip(".")) for m in req.messages
                  if m.role == "user" and not isinstance(m.content, str))
        board_items = [json.loads(text_of(m.content)) for m in req.messages if m.role == "tool"]
        items = [i for b in board_items for i in (b.get("result") or {}).get("items", [])]
        assert all(i["eligible_round"] <= rnd for i in items)
        if rnd < 4:
            assert not items


def test_acceptance_6_push_delivery_reaches_the_round_message(tmp_path):
    exp = llm_agent_experiment(3, medium=Board(topology="broadcast", delivery="push"))
    run = exp.run(seed=2, max_rounds=2, out=tmp_path)
    store = SnapshotStore(run.dir)
    state = pickle.loads(store.load(store.read(2))["participant:a000"])
    r2 = state["rounds"][1]["messages"][0]["content"]
    pushed = [p["text"] for p in r2 if (p["text"] or "").startswith("Delivered to you:")]
    assert pushed and "crop:" in pushed[0]
