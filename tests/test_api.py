"""Acceptance 4: the three DESIGN.md Experimenter-interface examples, and spec round-trips."""

from swarmlab import Experiment, Outcome, Participant, Policy, Run, World, text_observation, tool
from swarmlab.spec import RunSpec, spec_hash


# ---- example 2 fixture (verbatim from DESIGN.md) ---------------------------------------------
class OddAgentsSeeNothing(Policy):
    def apply(self, reader, post, round):
        if int(reader[1:]) % 2:
            return None                      # withhold
        return round + 1, post.text          # available next round, unchanged


# ---- example 3 fixture (verbatim from DESIGN.md) ---------------------------------------------
class Counter(World):
    def reset(self, rng, agents):
        self.total = 0
    def observe(self, agent):
        return text_observation(f"total so far: {self.total}")
    @tool("add", "Add n to the shared total", {"n": "integer"})
    def add(self, agent, n: int) -> Outcome:
        self.total += n
        return Outcome(accepted=True, feedback={"added": n})
    def score(self):
        return {"total": self.total}


class Adder(Participant):
    """Adds `step` every round, then yields without end_turn (no_tool)."""

    def __init__(self, step: int = 1):
        self.step = step

    async def turn(self, view, tools):
        await tools.call(self.agent, "add", {"n": self.step})


def test_example_1_existing_task(tmp_path):
    from swarmlab import Board, Budget
    from swarmlab.participants import EvidenceAggregator
    from swarmlab.worlds import FlagGame

    exp = Experiment(
        name="flag-gossip",
        world=FlagGame(n_candidates=8),
        participants=[EvidenceAggregator()] * 16,
        medium=Board(topology="gossip"),
        metrics=["belief.consensus", "belief.accuracy", "comm.read_rate"],
        budget=Budget(soft_usd=0, hard_usd=0),      # scripted agents spend nothing
    )
    run = exp.run(seed=3, max_rounds=20, out=tmp_path)
    assert set(run.score) == {"accuracy", "n_guessed", "truth"} and run.score["n_guessed"] == 16
    consensus = run.metrics["belief.consensus"]
    assert [r for r, _, _ in consensus] == list(range(1, 21))
    assert all(d == 16 for _, _, d in consensus)
    events = list(run.events)
    assert events[0]["type"] == "run_started" and events[-1]["type"] == "run_ended"
    assert len({e["agent"] for e in events if e["type"] == "turn_started"}) == 16
    page = run.view()
    assert page == run.dir / "view.html" and page.exists()
    fork = run.fork(at_round=4).run()
    assert fork.status == "ended" and fork.meta["fork_round"] == 4
    assert Run.load(run.dir).score == run.score


def test_example_2_change_visibility(tmp_path):
    from swarmlab import Board
    from swarmlab.participants import EvidenceAggregator
    from swarmlab.worlds import FlagGame

    exp = Experiment(name="odd-blind", world=FlagGame(), participants=[EvidenceAggregator()] * 8,
                     medium=Board(topology="broadcast", policies=[OddAgentsSeeNothing()]),
                     metrics=["comm.read_rate"])
    run = exp.run(seed=1, max_rounds=3, out=tmp_path)
    recipients = {e["recipient"] for e in run.events if e["type"] == "delivery"}
    assert recipients == {"a000", "a002", "a004", "a006"}
    reads = {e["agent"]: len(e["delivery_ids"]) for e in run.events if e["type"] == "read" and e["round"] == 2}
    assert all(n == 0 for a, n in reads.items() if int(a[1:]) % 2)
    assert all(n == 7 for a, n in reads.items() if not int(a[1:]) % 2)
    # round-trips through the spec with the inline policy as module:Class
    spec = exp.to_spec(seed=1, max_rounds=3)
    assert spec.medium.policies[0].type == "tests.test_api:OddAgentsSeeNothing"
    assert Experiment.from_spec(spec).to_spec(seed=1, max_rounds=3) == spec


def test_example_3_new_world(tmp_path):
    exp = Experiment(name="count", world=Counter(), participants=[Adder(), Adder(step=2)] * 2)
    run = exp.run(seed=0, max_rounds=5, out=tmp_path)
    assert run.score == {"total": (1 + 2) * 2 * 5}
    outcomes = [e for e in run.events if e["type"] == "action_committed"]
    assert len(outcomes) == 20 and all(o["accepted"] for o in outcomes)
    assert {e["yield_kind"] for e in run.events if e["type"] == "turn_ended"} == {"no_tool"}
    spec = exp.to_spec(seed=0, max_rounds=5)
    assert spec.world.type == "tests.test_api:Counter"
    assert [p.params for p in spec.participants] == [{"step": 1}, {"step": 2}] * 2
    again = Experiment.from_spec(spec)
    assert again.to_spec(seed=0, max_rounds=5) == spec
    # a rebuilt experiment reproduces the run, and resume/fork work from disk alone
    rerun = again.run(seed=0, max_rounds=5, out=tmp_path / "again")
    assert rerun.score == run.score
    loaded = Run.load(run.dir)
    child = loaded.fork(2).run()
    assert child.score == run.score


def test_spec_and_yaml_round_trip(tmp_path):
    from swarmlab import Board, Budget, DelayPolicy
    from swarmlab.metrics.belief import Polarization
    from swarmlab.participants import EvidenceAggregator, Silent
    from swarmlab.worlds import FlagGame

    exp = Experiment(
        name="rt", world=FlagGame(n_candidates=6, guess_limit=3),
        participants=[EvidenceAggregator()] * 3 + [Silent()] * 2,
        medium=Board(topology="gossip", delivery="push", policies=[DelayPolicy(1, readers=["a001"])]),
        metrics=["belief.consensus", Polarization(threshold=0.3)],
        budget=Budget(hard_usd=1.5),
    )
    spec = exp.to_spec(seed=4, max_rounds=6, commit="immediate")
    assert isinstance(spec, RunSpec) and len(spec.participants) == 5
    assert spec.metrics[1].model_dump() == {"type": "belief.polarization", "params": {"threshold": 0.3}}
    rebuilt = Experiment.from_spec(spec)
    assert rebuilt.to_spec(seed=4, max_rounds=6, commit="immediate") == spec
    assert spec_hash(rebuilt.to_spec(seed=4, max_rounds=6, commit="immediate")) == spec_hash(spec)

    path = tmp_path / "exp.yaml"
    exp.options = {"max_rounds": 6, "commit": "immediate"}
    exp.to_yaml(path)
    from_yaml = Experiment.from_yaml(path, "default")
    assert from_yaml.arm == "default"
    yspec = from_yaml.to_spec(seed=4, max_rounds=6, commit="immediate")
    assert yspec.model_dump(exclude={"arm"}) == spec.model_dump(exclude={"arm"})
    run = from_yaml.run(seed=4, out=tmp_path / "runs")      # max_rounds and commit from YAML
    assert run.id == "rt__default__s4"
    assert run.spec.options.commit == "immediate" and run.spec.options.max_rounds == 6


def test_shared_prototype_is_copied_per_agent(tmp_path):
    from swarmlab.participants import EvidenceAggregator
    from swarmlab.worlds import FlagGame

    proto = EvidenceAggregator()
    exp = Experiment(name="proto", world=FlagGame(), participants=[proto] * 16)
    run = exp.run(seed=9, max_rounds=2, out=tmp_path)
    assert not hasattr(proto, "agent")              # the prototype itself is never bound
    assert len({e["agent"] for e in run.events if e["type"] == "post"}) == 16
    # the default medium is not shared between experiments
    other = Experiment(name="x", world=FlagGame(), participants=[proto])
    assert other.medium is not exp.medium
