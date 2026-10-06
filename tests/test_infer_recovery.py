"""Cache part of M1b acceptance 1: determinism, replay without provider calls, and resume after
SIGKILL mid-round serving the completed turns' requests from the cache."""
import os
import signal
import subprocess
import sys
from pathlib import Path

import pytest

from swarmlab import Run
from swarmlab.events import EventLog, parse_event
from swarmlab.providers.fake import FakeProvider

from .helpers import PAUSE_ENV, PausingFlagGame, llm_experiment, logical

REPO = Path(__file__).resolve().parent.parent
SEED, ROUNDS, N = 5, 9, 6

CHILD = """
import sys
from tests.helpers import PausingFlagGame, llm_experiment
llm_experiment({n}, world=PausingFlagGame()).run(seed={seed}, max_rounds={rounds}, out=sys.argv[1])
"""


@pytest.fixture
def provider_calls(monkeypatch):
    calls: list = []
    original = FakeProvider.complete

    async def counting(self, request):
        calls.append(request)
        return await original(self, request)

    monkeypatch.setattr(FakeProvider, "complete", counting)
    return calls


def test_two_runs_identical_and_replay_calls_nothing(tmp_path, provider_calls):
    a = llm_experiment(N).run(seed=SEED, max_rounds=4, out=tmp_path / "a")
    b = llm_experiment(N).run(seed=SEED, max_rounds=4, out=tmp_path / "b")
    assert logical(a) == logical(b)
    assert a.score == b.score and a.spend == b.spend
    made = len(provider_calls)
    assert made == 2 * a.spend["calls"]
    loaded = Run.load(a.dir)
    assert len(provider_calls) == made
    assert loaded.metrics == a.metrics


def test_sigkill_mid_round_then_resume_hits_cache(tmp_path, provider_calls):
    out = tmp_path / "crashed"
    marker = tmp_path / "paused"
    env = {**os.environ, PAUSE_ENV: str(marker), "PYTHONPATH": str(REPO)}
    stderr = tmp_path / "child.stderr"
    with open(stderr, "wb") as err:
        proc = subprocess.Popen(
            [sys.executable, "-c", CHILD.format(n=N, seed=SEED, rounds=ROUNDS), str(out)],
            cwd=REPO, env=env, stdout=subprocess.DEVNULL, stderr=err)
    run_dir = out / "llm__s5"
    try:
        from .test_recovery import _wait_for

        _wait_for(marker, proc, stderr)
    finally:
        os.kill(proc.pid, signal.SIGKILL)
        proc.wait()
    log = run_dir / "events.jsonl"
    assert EventLog(log).last_committed()[0] == 6
    crashed = [parse_event(x) for x in log.read_bytes().splitlines()]
    r7_misses = [e for e in crashed if e.type == "inference_response" and e.round == 7 and not e.cached]
    assert len(r7_misses) == 2 * N  # every round-7 turn completed before the kill

    resumed = Run(run_dir, experiment=llm_experiment(N, world=PausingFlagGame())).resume()
    assert resumed.end_reason == "max_rounds"
    events = list(resumed.events_all)
    r7 = [e for e in events if e.type == "inference_response" and e.round == 7]
    assert len(r7) == 2 * N and all(e.cached for e in r7)
    # only rounds 8 and 9 called the provider
    assert len(provider_calls) == 2 * N * (ROUNDS - 7)

    reference = llm_experiment(N, world=PausingFlagGame()).run(seed=SEED, max_rounds=ROUNDS,
                                                               out=tmp_path / "clean")
    assert logical(resumed) == logical(reference)
    assert resumed.score == reference.score and resumed.metrics == reference.metrics
    # spend survives the crash: round 7 was paid once, before the kill
    assert resumed.spend == reference.spend
    before = len(provider_calls)
    Run.load(run_dir)
    assert len(provider_calls) == before
