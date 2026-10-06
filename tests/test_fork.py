"""Acceptance 3: fork at round 4 reproduces rounds 1-4 and continues live."""
import pytest

from swarmlab import Board, DelayPolicy, Experiment, Run
from swarmlab.participants import Silent

from .helpers import flag_experiment, logical


def committed_prefix(view, at_round):
    out = []
    for e in view:
        out.append(e)
        if e["type"] == "round_committed" and e["round"] == at_round:
            return out
    raise AssertionError("round not found")


def test_fork_reproduces_prefix_and_runs_live(tmp_path):
    parent = flag_experiment(8).run(seed=5, max_rounds=10, out=tmp_path)
    child = parent.fork(at_round=4).run()
    assert child.id == "flag__s5__f4_1" and child.dir.parent == parent.dir.parent
    assert child.status == "ended"

    # log prefix byte-identical through round 4's round_committed (+ its snapshot line)
    p_lines = (parent.dir / "events.jsonl").read_bytes().splitlines(keepends=True)
    c_lines = (child.dir / "events.jsonl").read_bytes().splitlines(keepends=True)
    p_view = logical(parent, exclude=("seq", "ts", "run"))
    prefix = committed_prefix(p_view, 4)
    n = len(prefix) + 1  # + snapshot line
    assert c_lines[:n] == p_lines[:n]
    # snapshot 4 byte-identical
    snap = "snapshots/000004.json"
    assert (child.dir / snap).read_bytes() == (parent.dir / snap).read_bytes()

    c_view = logical(child, exclude=("seq", "ts", "run"))
    assert c_view[: len(prefix)] == prefix
    started = c_view[n]
    assert started["type"] == "run_started"
    assert started["parent_run"] == parent.id and started["fork_round"] == 4
    # rounds after 4 ran live under the fork's run id, continuing the parent's seqs
    c_events = list(child.events_all)
    assert c_events[n].seq == c_events[n - 1].seq + 1
    live = [e for e in c_events[n + 1:]]
    assert live and all(e.run == child.id and e.round > 4 for e in live)
    assert any(e.type == "turn_started" and e.round == 5 for e in live)
    # same experiment and seed: the live continuation equals the parent's rounds 5-10
    assert c_view[n + 1:] == p_view[len(prefix) + 1:]
    assert child.score == parent.score
    # forks replay cleanly and number upwards
    Run.load(child.dir)
    second = parent.fork(4).run()
    assert second.id == "flag__s5__f4_2"


def test_fork_with_edited_experiment(tmp_path):
    exp = flag_experiment(8, medium=Board(topology="broadcast"))
    parent = exp.run(seed=6, max_rounds=8, out=tmp_path / "p")
    edited = Experiment(
        name="flag", world=exp.world,
        participants=list(exp.participants[:4]) + [Silent()] * 4,
        medium=Board(topology="broadcast", policies=[DelayPolicy(2)]),
        metrics=["belief.consensus", "comm.read_rate", "belief.entropy"],
    )
    child = parent.fork(3, experiment=edited).run(out=tmp_path / "f")
    assert child.dir.parent == tmp_path / "f"
    meta = child.meta
    assert meta["restored"]["world"] is True
    assert meta["restored"]["participants"] == ["a000", "a001", "a002", "a003"]
    assert sorted(meta["restored"]["metrics"]) == ["belief.consensus", "belief.entropy", "comm.read_rate"]
    assert child.spec.medium.policies[0].type == "delay"
    # silent agents never call read_board after the fork
    reads = {e["agent"] for e in child.events if e["type"] == "read" and e["round"] > 3}
    assert reads == {"a000", "a001", "a002", "a003"}
    after = {name for name, vals in child.metrics.items() if any(r > 3 for r, _, _ in vals)}
    assert after == {"belief.consensus", "comm.read_rate", "belief.entropy"}
    Run.load(child.dir)


def test_fork_requires_a_committed_round(tmp_path):
    parent = flag_experiment(4).run(seed=1, max_rounds=3, out=tmp_path)
    with pytest.raises(ValueError, match="not committed"):
        parent.fork(5).run()
