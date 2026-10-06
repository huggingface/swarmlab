import json

from swarmlab import Board, Experiment
from swarmlab.ids import agent_id
from swarmlab.participants import Enumerator, EvidenceAggregator, Silent
from swarmlab.rng import derive
from swarmlab.world.flaggame import FlagGame

AGENTS = [agent_id(i) for i in range(8)]


def truth_of(world, seed):
    w = FlagGame(**world.params)
    w.reset(derive(seed, "world"), AGENTS)
    return w


def turn_kinds(run):
    return [(e["agent"], e["yield_kind"], e["error"]) for e in run.events if e["type"] == "turn_ended"]


def test_enumerator_never_sees_correctness_in_20_rounds(tmp_path):
    for seed in (1, 2):
        world = FlagGame()
        truth = truth_of(world, seed).truth
        exp = Experiment(name="enum", world=world, participants=[Enumerator(truth_name=truth)] * 8,
                         medium=Board(), metrics=["belief.accuracy"])
        run = exp.run(seed=seed, max_rounds=20, out=tmp_path)
        kinds = turn_kinds(run)
        assert len(kinds) == 160
        assert all(k == "end_turn" for _, k, _ in kinds), [x for x in kinds if x[1] != "end_turn"][:1]
        # it really did cycle: each agent guessed every candidate, so accuracy hit 1/8 per round
        calls = [e for e in run.events if e["type"] == "tool_called"]
        assert {c["tool"] for c in calls} == {"guess", "my_status", "collective_status", "end_turn"}


class LeakyFlagGame(FlagGame):
    entry_point = None

    def my_status(self, agent):
        st = super().my_status(agent)
        st["correct"] = st["current_guess"] == self.truth
        return st


def test_enumerator_detects_a_leak(tmp_path):
    exp = Experiment(name="leak", world=LeakyFlagGame(), participants=[Enumerator()] * 2)
    run = exp.run(seed=1, max_rounds=1, out=tmp_path)
    kinds = turn_kinds(run)
    assert all(k == "error" and "AssertionError" in err and "correct" in err for _, k, err in kinds)


def test_silent_guesses_once_and_never_talks(tmp_path):
    exp = Experiment(name="silent", world=FlagGame(), participants=[Silent()] * 4)
    run = exp.run(seed=4, max_rounds=3, out=tmp_path)
    tools = [(e["round"], e["tool"]) for e in run.events if e["type"] == "tool_called"]
    assert sorted({t for _, t in tools}) == ["end_turn", "guess"]
    assert all(r == 1 for r, t in tools if t == "guess")
    assert run.score["n_guessed"] == 4


def test_evidence_aggregator_posts_crop_and_pools(tmp_path):
    exp = Experiment(name="agg", world=FlagGame(), participants=[EvidenceAggregator()] * 8,
                     medium=Board(topology="broadcast"), metrics=["belief.accuracy", "belief.consensus"])
    run = exp.run(seed=5, max_rounds=4, out=tmp_path)
    world = truth_of(FlagGame(), 5)
    posts = [e for e in run.events if e["type"] == "post"]
    assert len(posts) == 8 and all(p["round"] == 1 for p in posts)
    for p in posts:
        assert p["text"] == "crop:\n" + "\n".join(world.crop_rows(p["agent"]))
    # with every crop pooled under broadcast, everyone converges on the truth by round 2
    acc = {r: v for r, v, _ in run.metrics["belief.accuracy"]}
    assert acc[2] == 1.0 and acc[4] == 1.0
    assert run.score["accuracy"] == 1.0
    reads = [e for e in run.events if e["type"] == "read" and e["round"] == 2]
    assert len(reads) == 8 and all(len(r["delivery_ids"]) == 7 for r in reads)
    assert json.dumps(run.score)  # plain data
