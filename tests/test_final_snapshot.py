"""The round (or tick) that ends a run is always snapshotted, whatever `snapshot_every` says, so
the run's last snapshot is always `last_round` (swarmlab/runner.py, step 7)."""
from __future__ import annotations

import json
from pathlib import Path

from swarmlab import Experiment, Run
from swarmlab.events import logical_view, parse_event

from .helpers import flag_experiment
from .test_async import Slow, Total


def _snapshots(run_dir: Path) -> list[int]:
    return sorted(int(p.stem) for p in (run_dir / "snapshots").glob("*.json"))


def _snapshot_events(run_dir: Path) -> list[int]:
    return [json.loads(line)["round"] for line in (run_dir / "events.jsonl").read_text().splitlines()
            if json.loads(line)["type"] == "snapshot"]


def test_async_run_snapshots_its_last_tick(tmp_path):
    exp = Experiment(name="snap", world=Total(), participants=[Slow(steps=3), Slow(steps=1)])
    run = exp.run(seed=0, max_rounds=7, commit="async", snapshot_every=5, out=tmp_path)
    assert run.meta["last_round"] == 7
    assert _snapshots(run.dir) == _snapshot_events(run.dir) == [5, 7]
    assert Run.load(run.dir).replay()["score"] == run.score  # the score is checked at tick 7


def test_async_run_ended_by_the_per_agent_cap_snapshots_its_last_tick(tmp_path):
    exp = Experiment(name="snapcap", world=Total(), participants=[Slow(steps=3), Slow(steps=1)],
                     options={"commit": "async", "rounds_per_agent": 3, "snapshot_every": 4})
    run = exp.run(seed=0, out=tmp_path)
    assert run.meta["end_reason"] == "max_rounds" and run.meta["last_round"] == 6
    assert _snapshots(run.dir) == [4, 6]


def test_round_end_run_snapshots_its_last_round(tmp_path):
    run = flag_experiment(4).run(seed=1, max_rounds=3, snapshot_every=2, out=tmp_path / "two")
    assert run.meta["last_round"] == 3
    assert _snapshots(run.dir) == _snapshot_events(run.dir) == [2, 3]
    assert Run.load(run.dir).replay()["score"] == run.score
    # nothing else changes: the log is the snapshot_every=1 log minus the round-1 snapshot
    every = flag_experiment(4).run(seed=1, max_rounds=3, snapshot_every=1, out=tmp_path / "one")

    def view(d: Path, drop_round_1_snapshot: bool) -> list[dict]:
        evs = [parse_event(line) for line in (d / "events.jsonl").read_text().splitlines()]
        out = [e for e in logical_view(evs, exclude=("seq", "ts", "run"))
               if e["type"] != "run_started"]
        if drop_round_1_snapshot:
            out = [e for e in out if not (e["type"] == "snapshot" and e["round"] == 1)]
        return out

    assert view(run.dir, False) == view(every.dir, True)


class TerminalTotal(Total):
    """Ends `terminal` once the total reaches 5."""

    def terminal(self):
        return self.total >= 5


def test_a_terminal_round_is_snapshotted(tmp_path):
    exp = Experiment(name="term", world=TerminalTotal(), participants=[Slow(), Slow()])
    run = exp.run(seed=0, max_rounds=20, snapshot_every=10, out=tmp_path / "rounds")
    assert run.meta["end_reason"] == "terminal" and run.meta["last_round"] == 3
    assert _snapshots(run.dir) == [3]
    run = exp.run(seed=0, max_rounds=20, snapshot_every=10, commit="async", out=tmp_path / "ticks")
    assert run.meta["end_reason"] == "terminal" and run.meta["last_round"] == 5
    assert _snapshots(run.dir) == [5]


def test_max_rounds_that_is_not_a_multiple_is_snapshotted(tmp_path):
    run = flag_experiment(4).run(seed=2, max_rounds=5, snapshot_every=10, out=tmp_path)
    assert _snapshots(run.dir) == [5]
