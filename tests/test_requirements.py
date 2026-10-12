"""World requirements (`Requirements`, `World.requires`, `World.check_experiment`): checked at
`Experiment` construction and again at run start with the resolved run options."""
import json
from typing import Any, ClassVar

import pytest
from typer.testing import CliRunner

from swarmlab import Board, Experiment, Requirements, Run
from swarmlab.cli import app
from swarmlab.participants import EvidenceAggregator
from swarmlab.worlds import FlagGame


class PushWorld(FlagGame):
    requires: ClassVar[Requirements] = Requirements(
        commit="immediate", delivery="push", push_limit="n_agents", n_agents=(2, None))


class RoundsWorld(FlagGame):
    requires: ClassVar[Requirements] = Requirements(max_rounds="rounds")

    def __init__(self, rounds: int = 3) -> None:
        super().__init__()
        self.rounds = rounds


class FixedOrderWorld(FlagGame):
    requires: ClassVar[Requirements] = Requirements(scheduler="swarmlab.scheduler:Scheduler")


class SpeakerByName(FlagGame):
    requires: ClassVar[Requirements] = Requirements(scheduler="one_speaker")


class SpeakerByPath(FlagGame):
    requires: ClassVar[Requirements] = Requirements(scheduler="swarmlab.scheduler:OneSpeaker")


class EvenWorld(PushWorld):
    """An imperative check on top of the declared requirements."""

    calls: ClassVar[list] = []

    def check_experiment(self, experiment: Any) -> None:
        EvenWorld.calls.append(len(experiment.participants))
        super().check_experiment(experiment)
        if len(experiment.participants) % 2:
            raise ValueError("EvenWorld needs an even number of agents")


def exp(world, n=4, medium=None, **kw) -> Experiment:
    return Experiment(name="req", world=world, participants=[EvidenceAggregator()] * n,
                      medium=medium if medium is not None else Board(), **kw)


def push_ok(world=None, n=4) -> Experiment:
    return exp(world or PushWorld(), n, Board(delivery="push", push_limit=n),
               options={"commit": "immediate"})


def test_every_violated_field_is_named_in_one_message():
    with pytest.raises(ValueError) as e:
        exp(PushWorld(), 1, Board(push_limit=0))
    msg = str(e.value)
    assert "validation error" not in msg  # the plain ValueError, not pydantic's wrapper
    for part in ("world PushWorld requires commit=immediate, got round_end",
                 "world PushWorld requires delivery=push, got pull",
                 "world PushWorld requires push_limit>=1 (n_agents), got 0",
                 "world PushWorld requires n_agents>=2, got 1"):
        assert part in msg
    with pytest.raises(ValueError, match="push_limit>=4 .*got 3") as e:
        exp(PushWorld(), 4, Board(delivery="push", push_limit=3), options={"commit": "immediate"})
    assert "commit" not in str(e.value) and "delivery" not in str(e.value)


def test_a_matching_experiment_is_accepted_and_runs(tmp_path):
    run = push_ok().run(seed=1, max_rounds=2, out=tmp_path)
    assert run.status == "ended"


def test_run_time_overrides_are_checked_at_run_start(tmp_path):
    with pytest.raises(ValueError, match="requires commit=immediate, got round_end"):
        push_ok().run(seed=1, max_rounds=2, commit="round_end", out=tmp_path)


def test_max_rounds_names_a_world_attribute(tmp_path):
    with pytest.raises(ValueError, match=r"requires max_rounds=3 \(world.rounds\), got 5"):
        exp(RoundsWorld(rounds=3), options={"max_rounds": 5})
    with pytest.raises(ValueError, match="max_rounds=3"):  # rounds_per_agent resolves first
        exp(RoundsWorld(rounds=3), options={"rounds_per_agent": 2})
    exp(RoundsWorld(rounds=8), options={"rounds_per_agent": 2})
    # unknown at construction: checked when run() gives it
    late = exp(RoundsWorld(rounds=3))
    with pytest.raises(ValueError, match=r"max_rounds=3 \(world.rounds\), got 5"):
        late.run(seed=1, max_rounds=5, out=tmp_path / "a")
    assert late.run(seed=1, max_rounds=3, out=tmp_path / "b").status == "ended"


def test_scheduler_is_matched_by_resolved_class(tmp_path):
    with pytest.raises(ValueError, match="requires scheduler=swarmlab.scheduler:Scheduler, "
                                         "got swarmlab.scheduler:SeededShuffle"):
        exp(FixedOrderWorld())  # unset = SeededShuffle
    with pytest.raises(ValueError, match="got swarmlab.scheduler:OneSpeaker"):
        exp(FixedOrderWorld(), options={"scheduler": "one_speaker"})
    fixed = exp(FixedOrderWorld(), options={"scheduler": "swarmlab.scheduler:Scheduler"})
    assert fixed.run(seed=1, max_rounds=2, out=tmp_path).status == "ended"
    for world in (SpeakerByName(), SpeakerByPath()):
        for spelling in ("one_speaker", "swarmlab.scheduler:OneSpeaker",
                         {"type": "one_speaker", "params": {"listeners": 1}}):
            exp(world, options={"scheduler": spelling})
        with pytest.raises(ValueError, match="requires scheduler="):
            exp(world)


def test_an_override_adds_its_own_check_and_runs_once_per_construction():
    EvenWorld.calls.clear()
    push_ok(EvenWorld(), 4)
    assert EvenWorld.calls == [4]  # exactly once per construction
    with pytest.raises(ValueError, match="even number"):
        push_ok(EvenWorld(), 3)
    with pytest.raises(ValueError, match="requires commit=immediate"):  # super() still checks
        exp(EvenWorld(), 4, Board(delivery="push", push_limit=4))


def test_from_spec_checks_and_round_trips(tmp_path):
    good = push_ok()
    spec = good.to_spec(seed=1, max_rounds=2, commit="immediate")
    back = Experiment.from_spec(spec)
    assert back.options == {"max_rounds": 2, "commit": "immediate"}
    assert back.to_spec(seed=1, max_rounds=2, commit="immediate") == spec
    with pytest.raises(ValueError, match="requires commit=immediate, got round_end"):
        Experiment.from_spec(good.to_spec(seed=1, max_rounds=2))


def test_fork_resume_and_load_of_a_valid_run(tmp_path):
    run = push_ok().run(seed=2, max_rounds=3, out=tmp_path)
    loaded = Run(run.dir)  # rebuilds the experiment with from_spec (options from the spec)
    assert loaded.experiment.options["commit"] == "immediate"
    child = loaded.fork(at_round=1).run(out=tmp_path / "forks")
    assert child.status == "ended"
    assert Run.load(run.dir).status == "ended"
    assert loaded.resume().status == "ended"


SPEC = """\
name: req
options: {{max_rounds: 2{extra}}}
arms:
  A:
    world: {{type: "tests.test_requirements:PushWorld"}}
    participants: [{{type: evidence_aggregator, count: 4}}]
    medium: {{delivery: push, push_limit: {limit}}}
"""


def test_validate_cli_runs_the_requirement_check(tmp_path):
    bad = tmp_path / "bad.yaml"
    bad.write_text(SPEC.format(extra="", limit=2))
    res = CliRunner().invoke(app, ["validate", str(bad), "--json"])
    assert res.exit_code == 2, res.output
    err = json.loads(res.stdout)["error"]
    assert "world PushWorld requires commit=immediate, got round_end" in err
    assert "world PushWorld requires push_limit>=4 (n_agents), got 2" in err
    good = tmp_path / "good.yaml"
    good.write_text(SPEC.format(extra=", commit: immediate", limit=4))
    res = CliRunner().invoke(app, ["validate", str(good), "--json"])
    assert res.exit_code == 0, res.output
