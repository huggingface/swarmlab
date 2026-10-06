"""Review B3, B4 (post ack link), D2 (run.json first), C2 (notebooks, spec mismatch)."""
import asyncio
import json

import pytest

from swarmlab import Board, Experiment, Run
from swarmlab.events import EventLog, InferenceAttemptEvent, TurnStartedEvent, logical_view
from swarmlab.runner import Runner
from swarmlab.spec import RunOptions
from swarmlab.world.flaggame import FlagGame

from .helpers import OPERATIONAL_HOOK, Chatter, Inferrer, flag_experiment, logical


def _exp(n=3, name="op"):
    return Experiment(name=name, world=FlagGame(), medium=Board(), participants=[Inferrer()] * n,
                      metrics=["comm.posts_per_round"])


def test_operational_event_is_appended_immediately_and_not_logical(tmp_path):
    exp = _exp()
    runner = Runner(tmp_path / "run", exp, RunOptions(seed=0, max_rounds=2))
    on_disk = []

    def fn(agent, rnd):
        seq = runner.log_operational(InferenceAttemptEvent(
            run=runner.run_id, round=rnd, agent=agent, call_id=f"i-{agent}-{rnd}", provider="p",
            model="m", request_hash="h", reserved_usd=0.01))
        # on disk before the turn (and the round) finishes
        lines = runner.log_path.read_text().splitlines()
        on_disk.append(json.loads(lines[-1])["seq"] == seq)

    OPERATIONAL_HOOK["fn"] = fn
    try:
        runner.live()
    finally:
        OPERATIONAL_HOOK.clear()
    assert on_disk == [True] * 6
    events = list(EventLog(runner.log_path))
    r1 = [e for e in events if e.round == 1]
    kinds = [e.type for e in r1]
    first_turn = kinds.index("turn_started")
    assert kinds[0] == "round_started"
    assert kinds[1:first_turn] == ["inference_attempt"] * 3  # before the round's buffered events
    assert "inference_attempt" not in kinds[first_turn:]
    assert [e.seq for e in events] == list(range(len(events)))
    assert not [e for e in logical_view(events) if e["type"] == "inference_attempt"]
    # logical content equals a run without operational events
    plain = _exp().run(seed=0, max_rounds=2, out=tmp_path / "plain")
    ex = ("seq", "ts", "run")
    assert list(logical_view(events, ex)) == list(logical_view(plain.events_all, ex))
    Run(runner.dir).replay()
    with pytest.raises(ValueError):
        runner.log_operational(TurnStartedEvent(run="r", round=1))


@pytest.mark.parametrize("commit", ["round_end", "immediate"])
def test_post_event_links_the_ack_id(tmp_path, commit):
    exp = Experiment(name="ack", world=FlagGame(), medium=Board(), participants=[Chatter()] * 3)
    run = exp.run(seed=2, max_rounds=2, out=tmp_path, commit=commit)
    calls = {e["call_id"]: e for e in run.events if e["type"] == "tool_called"}
    acks = {}
    for e in run.events:
        if e["type"] == "tool_returned" and calls[e["call_id"]]["tool"] == "post":
            acks[(e["round"], calls[e["call_id"]]["args"]["text"])] = e["result"]["result"]["id"]
    posts = [e for e in run.events if e["type"] == "post"]
    assert len(posts) == 6
    for p in posts:
        assert p["provisional_id"] == acks[(p["round"], p["text"])]
        if commit == "round_end":
            assert p["provisional_id"].startswith(f"tmp-{p['agent']}-")
        else:
            assert p["provisional_id"] == p["post_id"]


def test_run_json_is_written_before_run_started(tmp_path, monkeypatch):
    exp = flag_experiment(3)
    ref = exp.run(seed=5, max_rounds=2, out=tmp_path / "ref")
    seen = {}

    def crash(self, round):
        seen["meta"] = (self.dir / "run.json").exists()
        raise KeyboardInterrupt  # killed between run.json and run_started

    monkeypatch.setattr(Runner, "_run_started", crash)
    with pytest.raises(KeyboardInterrupt):
        exp.run(seed=5, max_rounds=2, out=tmp_path / "crash")
    monkeypatch.undo()
    assert seen["meta"]
    crashed = tmp_path / "crash" / ref.dir.name
    resumed = Run(crashed).resume()
    assert resumed.status == "ended" and logical(resumed) == logical(ref)


def test_run_inside_a_running_event_loop(tmp_path):
    exp = flag_experiment(3)

    async def main():
        direct = exp.run(seed=1, max_rounds=2, out=tmp_path / "direct")  # notebook-style call
        threaded = await asyncio.to_thread(exp.run, seed=1, max_rounds=2, out=tmp_path / "thread")
        return direct, threaded

    direct, threaded = asyncio.run(main())
    ref = exp.run(seed=1, max_rounds=2, out=tmp_path / "ref")
    assert logical(direct) == logical(ref) == logical(threaded)


def test_existing_dir_with_a_different_spec_names_both_hashes(tmp_path):
    exp = flag_experiment(3)
    first = exp.run(seed=1, max_rounds=2, out=tmp_path)
    old = json.loads((first.dir / "run.json").read_text())["spec_hash"]
    with pytest.raises(FileExistsError) as same:
        exp.run(seed=1, max_rounds=2, out=tmp_path)
    assert "different" not in str(same.value)
    with pytest.raises(FileExistsError) as diff:
        exp.run(seed=1, max_rounds=2, out=tmp_path, commit="immediate")
    msg = str(diff.value)
    new = Runner(first.dir, exp, RunOptions(seed=1, max_rounds=2, commit="immediate"))
    assert "different configuration" in msg and old in msg
    from swarmlab.spec import spec_hash

    assert spec_hash(new.spec) in msg
