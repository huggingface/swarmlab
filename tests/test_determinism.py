"""Acceptance 1: same experiment and seed -> identical logical views and scores."""
import json

import pytest

from swarmlab import Board, DelayPolicy, Run

from .helpers import flag_experiment, logical


@pytest.mark.parametrize("commit", ["round_end", "immediate"])
def test_two_runs_are_identical(tmp_path, commit):
    exp = flag_experiment(8, medium=Board(topology="gossip", policies=[DelayPolicy(1, readers=["a001"])]))
    a = exp.run(seed=11, max_rounds=10, commit=commit, out=tmp_path / "one")
    b = exp.run(seed=11, max_rounds=10, commit=commit, out=tmp_path / "two")
    va, vb = logical(a), logical(b)
    assert len(va) > 100
    assert va == vb
    assert a.score == b.score
    assert a.metrics == b.metrics
    # snapshots are byte-identical too (deterministic pickles and manifests)
    for r in range(1, 11):
        name = f"snapshots/{r:06d}.json"
        assert (a.dir / name).read_bytes() == (b.dir / name).read_bytes()
    meta_a, meta_b = json.loads((a.dir / "run.json").read_text()), json.loads((b.dir / "run.json").read_text())
    assert meta_a["spec_hash"] == meta_b["spec_hash"]
    assert a.status == "ended" and a.end_reason == "max_rounds"


def test_different_seed_differs(tmp_path):
    exp = flag_experiment(8)
    a = exp.run(seed=1, max_rounds=3, out=tmp_path)
    b = exp.run(seed=2, max_rounds=3, out=tmp_path)
    assert logical(a, exclude=("seq", "ts", "run")) != logical(b, exclude=("seq", "ts", "run"))


def test_replay_matches_and_detects_tampering(tmp_path):
    from swarmlab.runner import ReplayMismatch

    run = flag_experiment(8).run(seed=3, max_rounds=5, out=tmp_path)
    loaded = Run.load(run.dir)
    assert loaded.score == run.score and loaded.status == "ended"
    report = loaded.replay()
    assert report["metrics_checked"] == 5 * 7 and report["score"] == run.score
    meta = json.loads((run.dir / "run.json").read_text())
    meta["score"]["accuracy"] = -1
    (run.dir / "run.json").write_text(json.dumps(meta))
    with pytest.raises(ReplayMismatch):
        Run.load(run.dir)


def test_run_dir_layout(tmp_path):
    run = flag_experiment(4).run(seed=0, max_rounds=2, out=tmp_path)
    assert run.id == "flag__s0" and run.dir == tmp_path / "flag__s0"
    for p in ["run.json", "events.jsonl", "artifacts/spec.yaml", "artifacts/git.txt",
              "snapshots/000001.json", "snapshots/000002.json"]:
        assert (run.dir / p).exists(), p
    first = next(iter(run.events_all))
    assert first.type == "run_started" and first.run_spec["experiment"] == "flag"
    assert first.spec_hash == json.loads((run.dir / "run.json").read_text())["spec_hash"]
    with pytest.raises(FileExistsError):
        flag_experiment(4).run(seed=0, max_rounds=2, out=tmp_path)
