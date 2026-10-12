"""World requirements: the declared `Requirements` (`World.requires`) are checked early (at
`Experiment` construction, `check_run`, estimate, CLI estimate/run) and again at run start; the
runtime hook `World.check_experiment` (participant-dependent checks) runs only at run start, after
bind and roles, before any inference."""
import json
from typing import Any, ClassVar

import pytest
from typer.testing import CliRunner

from swarmlab import Board, Experiment, Requirements, Run
from swarmlab.cli import app
from swarmlab.participants import EvidenceAggregator
from swarmlab.participants.base import Participant, TurnUsage
from swarmlab.roles import assign
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


def test_an_override_runs_at_run_start_only(tmp_path):
    EvenWorld.calls.clear()
    odd = push_ok(EvenWorld(), 3)  # construction checks the declared requirements only
    odd.check_run(max_rounds=2)
    assert EvenWorld.calls == []
    with pytest.raises(ValueError, match="even number"):
        odd.run(seed=1, max_rounds=2, out=tmp_path / "odd")
    assert EvenWorld.calls == [3]
    with pytest.raises(ValueError, match="requires commit=immediate"):  # declared: early
        exp(EvenWorld(), 4, Board(delivery="push", push_limit=4))
    assert push_ok(EvenWorld(), 4).run(seed=1, max_rounds=2, out=tmp_path / "even").status == "ended"


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


# ---- participant compatibility is checked on the bound, role-applied participants -----------------
VISION = "fake:tests.helpers:script_end_turn_first"  # a fake script that does not read text grids


def role_exp(proto_model: str, role_model: str) -> Experiment:
    from swarmlab.participants import LLMAgent
    from swarmlab.providers.fake import FakeProvider
    from swarmlab.roles import Role, assign

    from .helpers import TEST_PRICING

    return Experiment(name="img", world=FlagGame(modality="image"),
                      participants=[assign(LLMAgent(model=proto_model), "seer") for _ in range(2)],
                      medium=Board(topology="broadcast"), roles={"seer": Role(name="seer", model=role_model)},
                      providers={"fake": FakeProvider(pricing=TEST_PRICING)})


@pytest.fixture
def inference_calls(monkeypatch):
    from swarmlab.providers.fake import FakeProvider

    seen: list = []
    original = FakeProvider.complete

    async def recording(self, request):
        seen.append(request)
        return await original(self, request)

    monkeypatch.setattr(FakeProvider, "complete", recording)
    return seen


def test_a_role_model_that_fixes_a_rejected_prototype_runs(tmp_path, inference_calls):
    good = role_exp("fake:reader", VISION)
    good.check_run(max_rounds=1)
    assert good.run(seed=1, max_rounds=1, out=tmp_path).status == "ended"
    assert [p.model for p in good.participants] == ["fake:reader"] * 2  # prototypes untouched
    assert inference_calls  # it really ran


def test_a_role_model_that_breaks_an_accepted_prototype_is_refused_at_run_start(tmp_path, inference_calls):
    bad = role_exp(VISION, "fake:reader")  # construction checks declared settings only
    bad.check_run(max_rounds=1)
    with pytest.raises(ValueError, match=r"participants \['a000', 'a001'\].*image_text_hint=True"):
        bad.run(seed=1, max_rounds=1, out=tmp_path)
    assert inference_calls == []  # refused before any provider call
    assert [p.model for p in bad.participants] == [VISION] * 2  # prototypes untouched


class NeedsBinding(Participant):
    """A valid participant whose `apply_role` needs binding state (`self.agent`)."""

    labels: ClassVar[list] = []

    def apply_role(self, role):
        self.label = f"{self.agent}:{role.name}"
        NeedsBinding.labels.append((self.label, self.rng.random()))

    async def turn(self, view, tools):
        await tools.call("end_turn", {})
        return TurnUsage()


@pytest.mark.parametrize("world", [FlagGame(), PushWorld()], ids=["flaggame", "requires"])
def test_apply_role_runs_only_after_bind(tmp_path, world):
    NeedsBinding.labels.clear()
    push = isinstance(world, PushWorld)
    built = Experiment(name="custom", world=world,
                       participants=[assign(NeedsBinding(), "worker") for _ in range(2)],
                       medium=Board(delivery="push", push_limit=2) if push else Board(),
                       options={"commit": "immediate"} if push else {})
    built.check_run(max_rounds=2)
    assert NeedsBinding.labels == []  # construction and check_run never call participant hooks
    assert all(not hasattr(p, "label") for p in built.participants)
    assert built.run(seed=1, max_rounds=2, out=tmp_path).status == "ended"
    from swarmlab.runner import agent_stream

    # called once per agent after bind: the real agent id and its own rng stream
    assert NeedsBinding.labels == [(f"{a}:worker", agent_stream(1, a).random()) for a in ("a000", "a001")]


# ---- estimate and run check the effective settings, overrides included ---------------------------
ROUNDS_SPEC = """\
name: req
seeds: [1, 2]
options: {max_rounds: 3}
arms:
  A:
    world: {type: "tests.test_requirements:RoundsWorld", params: {rounds: 3}}
    participants: [{type: evidence_aggregator, count: 2}]
"""


def test_estimate_checks_its_max_rounds():
    good = exp(RoundsWorld(rounds=3), options={"max_rounds": 3})
    assert good.estimate(seed=1, max_rounds=3)["rounds"] == 3
    with pytest.raises(ValueError, match=r"requires max_rounds=3 \(world.rounds\), got 5"):
        good.estimate(seed=1, max_rounds=5)
    with pytest.raises(ValueError, match="requires commit=immediate"):
        push_ok().check_run(commit="round_end")
    push_ok().check_run(max_rounds=7)


def test_cli_estimate_and_run_refuse_a_bad_max_rounds_override_up_front(tmp_path):
    spec = tmp_path / "rounds.yaml"
    spec.write_text(ROUNDS_SPEC)
    res = CliRunner().invoke(app, ["estimate", str(spec), "--max-rounds", "5", "--json"])
    assert res.exit_code == 2, res.output
    out = json.loads(res.stdout)
    assert "requires max_rounds=3 (world.rounds), got 5" in out["error"] and "arms" not in out
    res = CliRunner().invoke(app, ["estimate", str(spec), "--max-rounds", "3", "--json"])
    assert res.exit_code == 0, res.output
    assert json.loads(res.stdout)["arms"]["A"]["rounds"] == 3
    res = CliRunner().invoke(app, ["run", str(spec), "--max-rounds", "5", "--out", str(tmp_path / "runs")])
    assert res.exit_code == 2, res.output
    assert "requires max_rounds=3 (world.rounds), got 5" in res.output
    assert "running" not in res.output and not (tmp_path / "runs").exists()
    res = CliRunner().invoke(app, ["run", str(spec), "--max-rounds", "3", "--out", str(tmp_path / "runs")])
    assert res.exit_code == 0, res.output
    assert "running" in res.output
