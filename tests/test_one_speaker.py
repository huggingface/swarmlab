"""M6 §2: OneSpeaker scheduler, rounds_per_agent, estimate (docs/INTERFACE-M6.md)."""
from collections import Counter

import pytest

from swarmlab import Board, Run
from swarmlab.ids import agent_id
from swarmlab.rng import derive
from swarmlab.scheduler import OneSpeaker, SeededShuffle, build_scheduler
from swarmlab.spec import PluginSpec, RunOptions, SpecError, arm_to_runspec, validate_experiment_doc

from .helpers import flag_experiment, logical
from .test_recovery import _truncate_mid_round

AGENTS = [agent_id(i) for i in range(6)]


def gossip1():
    return Board(topology={"type": "gossip", "params": {"k": 1}})


def one_speaker_exp(n=6, **kw):
    return flag_experiment(n, medium=gossip1(), options={"scheduler": "one_speaker"}, **kw)


def test_order_is_one_seeded_live_agent():
    s = OneSpeaker()
    picks = [s.order(r, AGENTS, derive(5, "schedule", r)) for r in range(1, 601)]
    assert all(len(p) == 1 and p[0] in AGENTS for p in picks)
    assert picks == [s.order(r, AGENTS, derive(5, "schedule", r)) for r in range(1, 601)]
    counts = Counter(p[0] for p in picks)
    assert set(counts) == set(AGENTS) and min(counts.values()) > 60  # roughly uniform
    assert s.order(1, [], derive(5, "schedule", 1)) == []
    assert s.turns_per_round(6) == 1 and SeededShuffle().turns_per_round(6) == 6
    assert s.spec() == {"type": "one_speaker", "params": {"listeners": 1}}
    with pytest.raises(ValueError):
        OneSpeaker(listeners=0)


def test_build_scheduler_forms():
    assert isinstance(build_scheduler(None), SeededShuffle)
    assert isinstance(build_scheduler("one_speaker"), OneSpeaker)
    assert build_scheduler({"type": "one_speaker", "params": {"listeners": 2}}).listeners == 2
    assert isinstance(build_scheduler(PluginSpec(type="seeded_shuffle")), SeededShuffle)
    assert isinstance(build_scheduler("swarmlab.scheduler:OneSpeaker"), OneSpeaker)


def test_run_options_dump_unchanged_without_m6_fields():
    plain = RunOptions(seed=1, max_rounds=3).model_dump(mode="json")
    assert not {"scheduler", "rounds_per_agent", "stop_when"} & set(plain)
    opts = RunOptions(seed=1, max_rounds=3, scheduler="one_speaker")
    assert opts.model_dump(mode="json")["scheduler"] == {"type": "one_speaker", "params": {}}


def test_one_turn_per_round_one_delivery_per_post(tmp_path):
    exp = one_speaker_exp()
    run = exp.run(seed=3, max_rounds=12, commit="immediate", out=tmp_path / "a")
    events = list(run.events)
    started = Counter(e["round"] for e in events if e["type"] == "turn_started")
    assert started == {r: 1 for r in range(1, 13)}
    orders = [e["order"] for e in events if e["type"] == "round_started"]
    assert all(len(o) == 1 for o in orders)
    posts = [e for e in events if e["type"] == "post"]
    deliveries = [e for e in events if e["type"] == "delivery"]
    assert posts and len(deliveries) == len(posts)
    by_post = Counter(d["post_id"] for d in deliveries)
    assert all(by_post[p["post_id"]] == 1 for p in posts)
    for p in posts:  # the listener is never the speaker
        assert next(d["recipient"] for d in deliveries if d["post_id"] == p["post_id"]) != p["agent"]
    again = exp.run(seed=3, max_rounds=12, commit="immediate", out=tmp_path / "b")
    assert logical(again) == logical(run)
    Run.load(run.dir)  # replays


def test_resume_after_a_crash(tmp_path):
    exp = one_speaker_exp(4)
    ref = exp.run(seed=1, max_rounds=8, commit="immediate", out=tmp_path / "ref")
    run = exp.run(seed=1, max_rounds=8, commit="immediate", out=tmp_path / "crash")
    _truncate_mid_round(run.dir, 5)
    resumed = Run(run.dir).resume()
    assert logical(resumed) == logical(ref) and resumed.score == ref.score


def test_listeners_must_match_gossip_k(tmp_path):
    exp = flag_experiment(4, medium=Board(topology={"type": "gossip", "params": {"k": 2}}),
                          options={"scheduler": {"type": "one_speaker", "params": {"listeners": 1}}})
    with pytest.raises(ValueError, match="listeners"):
        exp.run(seed=1, max_rounds=2, commit="immediate", out=tmp_path)


def test_probes_ask_every_live_agent(tmp_path):
    from swarmlab.experiment import Experiment
    from swarmlab.participants.llm import LLMAgent
    from swarmlab.world.flaggame import FlagGame

    exp = Experiment(name="p", world=FlagGame(), participants=[LLMAgent("fake:reader")] * 4,
                     medium=gossip1(), probes=[{"type": "belief", "params": {"every": 2}}],
                     options={"scheduler": "one_speaker"})
    run = exp.run(seed=1, max_rounds=4, commit="immediate", out=tmp_path)
    probes = [e for e in run.events if e["type"] == "probe"]
    assert Counter(e["round"] for e in probes) == {2: 4, 4: 4}


def doc(**opts):
    return {"name": "m6", "options": {"max_rounds": 99, **opts},
            "arms": {"a": {"world": "flaggame",
                           "participants": [{"type": "evidence_aggregator", "count": 4}]}}}


def test_rounds_per_agent_sugar():
    spec = arm_to_runspec(doc(rounds_per_agent=10), "a", seed=1)
    assert spec.options.max_rounds == 40 and spec.options.rounds_per_agent == 10
    # an explicit override wins
    assert arm_to_runspec(doc(rounds_per_agent=10), "a", seed=1, max_rounds=7).options.max_rounds == 7
    # absent: unchanged, and not in the dump
    plain = arm_to_runspec(doc(), "a", seed=1)
    assert plain.options.max_rounds == 99 and "rounds_per_agent" not in plain.model_dump(mode="json")["options"]
    with pytest.raises(SpecError):
        arm_to_runspec(doc(rounds_per_agent=0), "a", seed=1)
    validate_experiment_doc(doc(rounds_per_agent=3, scheduler="one_speaker"))


def test_rounds_per_agent_through_yaml(tmp_path):
    import yaml

    from swarmlab.experiment import Experiment

    path = tmp_path / "e.yaml"
    path.write_text(yaml.safe_dump(doc(rounds_per_agent=3)))
    exp = Experiment.from_yaml(path, "a")
    assert exp.options["max_rounds"] == 12
    assert exp._options(1, None).max_rounds == 12


def test_estimate_prices_one_turn_per_round():
    from swarmlab.experiment import Experiment
    from swarmlab.participants.llm import LLMAgent
    from swarmlab.world.flaggame import FlagGame

    def est(options):
        exp = Experiment(name="e", world=FlagGame(), participants=[LLMAgent("fake:x", max_calls=1)] * 8,
                         options=options)
        return exp.estimate(1, 40, calls_per_turn=1)

    every = est({})
    one = est({"scheduler": "one_speaker"})
    assert every["calls"] == 8 * 40 and one["calls"] == pytest.approx(40)
    assert one["usd"] == pytest.approx(every["usd"] / 8)
