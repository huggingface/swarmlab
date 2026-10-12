"""Runs whose participant forwards options through `**kwargs` replay and fork, also from a
run.json holding the older nested form (`{"kwargs": {...}}`)."""
import json

from swarmlab import Board, Experiment, Run
from swarmlab.participants import EvidenceAggregator
from swarmlab.world.flaggame import FlagGame


class Loud(EvidenceAggregator):
    def __init__(self, volume: int = 1) -> None:
        self.volume = volume


class Forwarding(Loud):
    def __init__(self, tag: str = "t", **kwargs) -> None:
        super().__init__(**kwargs)
        self.tag = tag


def run_forwarding(out):
    exp = Experiment(name="kw", world=FlagGame(), participants=[Forwarding(volume=3)] * 4,
                     medium=Board(topology="gossip"), metrics=["belief.consensus"])
    return exp.run(seed=2, max_rounds=5, out=out)


def test_run_json_params_are_flat_and_replay(tmp_path):
    run = run_forwarding(tmp_path)
    params = [p["params"] for p in run.meta["spec"]["participants"]]
    assert params == [{"tag": "t", "volume": 3}] * 4
    loaded = Run.load(run.dir)
    assert loaded.experiment.participants[0].volume == 3


def test_nested_run_json_replays_and_forks_restoring_participants(tmp_path):
    run = run_forwarding(tmp_path)
    meta_path = run.dir / "run.json"
    meta = json.loads(meta_path.read_text())
    old_hash = meta["spec_hash"]
    for p in meta["spec"]["participants"]:
        p["params"] = {"tag": "t", "kwargs": {"volume": 3}}
    meta_path.write_text(json.dumps(meta))

    loaded = Run.load(run.dir)
    assert loaded.experiment.participants[0].volume == 3
    assert Run(run.dir).meta["spec_hash"] == old_hash

    child = Run(run.dir).fork(3).run()
    assert child.status == "ended"
    assert len(child.meta["restored"]["participants"]) == 4
