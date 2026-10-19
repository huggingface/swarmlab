"""Compatibility fixture: a representative `round_end` and `immediate` run with scripted
participants must keep its logical log and its snapshots byte for byte (one digest per logical
event, minus `ts`, `git_commit` and `dirty`, and the sha256 of every snapshot manifest, which
names every plugin blob). The fixture was recorded with the final-round snapshot rule in force
(runner step 7), so with `snapshot_every=2` and `max_rounds=3` it expects snapshots 2 and 3: round
3 is the one intentional addition over older runs. Regenerate (only for an intended change, and
say why in the commit) with `SWARMLAB_REGEN_FIXTURES=1 pytest tests/test_compat_fixture.py`."""
from __future__ import annotations

import hashlib
import json
import os
from pathlib import Path

import pytest

from swarmlab.events import logical_view, parse_event

from .helpers import flag_experiment

FIXTURE = Path(__file__).parent / "fixtures" / "compat_logs.json"
DROP = ("ts", "git_commit", "dirty")


def _digest(run_dir: Path) -> dict:
    evs = [parse_event(line) for line in (run_dir / "events.jsonl").read_text().splitlines()]
    events = []
    for e in logical_view(evs, exclude=DROP):
        blob = json.dumps(e, sort_keys=True, separators=(",", ":")).encode()
        events.append([e["seq"], e["type"], e["round"], hashlib.sha256(blob).hexdigest()[:16]])
    snaps = {p.stem: hashlib.sha256(p.read_bytes()).hexdigest()
             for p in sorted((run_dir / "snapshots").glob("*.json"))}
    meta = json.loads((run_dir / "run.json").read_text())
    return {"events": events, "snapshots": snaps,
            "run": {k: meta[k] for k in ("spec_hash", "score", "last_round", "end_reason")}}


def _dump(got: dict) -> str:
    """One line per logical event, so a diff of the fixture shows the events that moved."""
    lines = ["{"]
    for i, (commit, d) in enumerate(sorted(got.items())):
        evs = ",\n".join(f"   {json.dumps(e)}" for e in d["events"])
        lines.append(f' "{commit}": {{\n  "events": [\n{evs}\n  ],\n'
                     f'  "run": {json.dumps(d["run"], sort_keys=True)},\n'
                     f'  "snapshots": {json.dumps(d["snapshots"], sort_keys=True)}\n'
                     f' }}{"," if i < len(got) - 1 else ""}')
    return "\n".join(lines) + "\n}\n"


def _runs(tmp_path: Path) -> dict:
    out = {}
    for commit in ("round_end", "immediate"):
        run = flag_experiment(4, name=f"compat-{commit}").run(
            seed=1, max_rounds=3, snapshot_every=2, commit=commit, out=tmp_path / commit)
        out[commit] = _digest(run.dir)
    return out


def test_round_based_runs_match_the_compat_fixture(tmp_path):
    got = _runs(tmp_path)
    if os.environ.get("SWARMLAB_REGEN_FIXTURES"):
        FIXTURE.write_text(_dump(got))
        pytest.skip("fixture regenerated")
    want = json.loads(FIXTURE.read_text())
    for commit in ("round_end", "immediate"):
        assert got[commit]["run"] == want[commit]["run"], commit
        assert got[commit]["snapshots"] == want[commit]["snapshots"], commit
        for g, w in zip(got[commit]["events"], want[commit]["events"], strict=False):
            assert g == w, f"{commit}: first differing logical event {g} (want {w})"
        assert len(got[commit]["events"]) == len(want[commit]["events"]), commit
        assert list(want[commit]["snapshots"]) == ["000002", "000003"]
