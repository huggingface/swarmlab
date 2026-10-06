"""Acceptance 2: SIGKILL during round 7, resume, and match an uninterrupted run."""
import json
import os
import signal
import subprocess
import sys
import time
from pathlib import Path

import pytest

from swarmlab import Run
from swarmlab.events import EventLog, parse_event
from swarmlab.runner import RecoveryError

from .helpers import PAUSE_ENV, PausingFlagGame, flag_experiment, logical

REPO = Path(__file__).resolve().parent.parent
SEED, ROUNDS = 7, 10

CHILD = """
import sys
from tests.helpers import PausingFlagGame, flag_experiment
flag_experiment(8, world=PausingFlagGame()).run(seed={seed}, max_rounds={rounds}, out=sys.argv[1])
"""


def _wait_for(path: Path, proc: subprocess.Popen, stderr: Path, timeout: float = 60.0) -> None:
    deadline = time.monotonic() + timeout
    while not path.exists():
        if proc.poll() is not None:
            raise AssertionError(f"child exited early: {stderr.read_text()}")
        if time.monotonic() > deadline:
            proc.kill()
            raise AssertionError("child never reached round 7")
        time.sleep(0.02)


def test_sigkill_in_round_7_then_resume(tmp_path):
    out = tmp_path / "crashed"
    marker = tmp_path / "paused"
    env = {**os.environ, PAUSE_ENV: str(marker), "PYTHONPATH": str(REPO)}
    stderr = tmp_path / "child.stderr"
    with open(stderr, "wb") as err:
        proc = subprocess.Popen(
            [sys.executable, "-c", CHILD.format(seed=SEED, rounds=ROUNDS), str(out)],
            cwd=REPO, env=env, stdout=subprocess.DEVNULL, stderr=err,
        )
    run_dir = out / "flag__s7"
    log = run_dir / "events.jsonl"
    try:
        _wait_for(marker, proc, stderr)
        events = [parse_event(line) for line in log.read_bytes().splitlines()]
        assert any(e.type == "turn_started" and e.round == 7 for e in events)
        assert not any(e.type == "round_committed" and e.round == 7 for e in events)
    finally:
        os.kill(proc.pid, signal.SIGKILL)
        proc.wait()
    assert EventLog(log).last_committed()[0] == 6

    loaded = Run.load(run_dir)              # replays the partial log
    assert loaded.status == "running"
    resumed = loaded.resume()
    assert resumed.status == "ended" and resumed.end_reason == "max_rounds"

    reference = flag_experiment(8, world=PausingFlagGame()).run(seed=SEED, max_rounds=ROUNDS,
                                                                 out=tmp_path / "clean")
    assert logical(resumed) == logical(reference)
    assert resumed.score == reference.score
    assert resumed.metrics == reference.metrics
    discarded = [json.loads(line) for line in (run_dir / "discarded.jsonl").read_text().splitlines()]
    assert discarded and all(e["round"] == 7 for e in discarded)
    assert any(e["type"] == "turn_started" for e in discarded)
    Run.load(run_dir)  # the resumed run replays cleanly


def _truncate_mid_round(run_dir: Path, round: int) -> None:
    """Simulate a crash: keep the log up to the first tool call of `round`."""
    lines = (run_dir / "events.jsonl").read_bytes().splitlines(keepends=True)
    for i, line in enumerate(lines):
        ev = parse_event(line)
        if ev.type == "tool_called" and ev.round == round:
            (run_dir / "events.jsonl").write_bytes(b"".join(lines[: i + 1]) + lines[i + 1][:10])
            return
    raise AssertionError("round not found")


def test_resume_from_a_crash_in_round_1(tmp_path):
    exp = flag_experiment(4)
    ref = exp.run(seed=1, max_rounds=3, out=tmp_path / "ref")
    run = exp.run(seed=1, max_rounds=3, out=tmp_path / "crash")
    _truncate_mid_round(run.dir, 1)
    resumed = Run(run.dir).resume()
    assert logical(resumed) == logical(ref) and resumed.score == ref.score


def test_resume_of_an_ended_run_is_a_no_op(tmp_path):
    run = flag_experiment(4).run(seed=1, max_rounds=2, out=tmp_path)
    before = (run.dir / "events.jsonl").read_bytes()
    assert run.resume().status == "ended"
    assert (run.dir / "events.jsonl").read_bytes() == before


def test_resume_without_a_snapshot_at_the_commit_is_refused(tmp_path):
    run = flag_experiment(4).run(seed=1, max_rounds=4, snapshot_every=2, out=tmp_path)
    _truncate_mid_round(run.dir, 4)
    with pytest.raises(RecoveryError, match="snapshot_every"):
        Run(run.dir).resume()
