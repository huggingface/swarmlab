"""M1b acceptance 2 (hard ceiling) and 3 (soft budget) with the fake provider."""
import json

import pytest

from swarmlab import Budget, Experiment, Run
from swarmlab.events import logical_view
from swarmlab.spec import load_experiment_yaml

from .helpers import llm_experiment

N, ROUNDS, SEED = 4, 4, 11


def by_type(run, t):
    return [e for e in run.events_all if e.type == t]


def comparable(run):
    """Logical view minus run_started (its spec carries the budget) and budget_changed."""
    return [e for e in logical_view(run.events_all)
            if e["type"] not in ("run_started", "budget_changed")]


def test_hard_ceiling_aborts_mid_round_and_resume_finishes(tmp_path):
    # round 1 costs about 4 x $0.0125; round 2's first calls reserve their worst case and cross
    run = llm_experiment(N, budget=Budget(hard_usd=0.07)).run(seed=SEED, max_rounds=ROUNDS,
                                                              out=tmp_path / "a")
    assert run.status == "ended" and run.end_reason == "hard_ceiling"
    assert run.meta["last_round"] == 1
    evs = list(run.events_all)
    assert [e.round for e in evs if e.type == "round_committed"] == [1]
    assert any(e.type == "round_started" and e.round == 2 for e in evs)
    # mid-round: some of round 2's calls went out before the ceiling refused one
    attempts2 = [e for e in evs if e.type == "inference_attempt" and e.round == 2]
    assert attempts2
    # nothing of round 2 was committed or even logged as a turn
    assert not [e for e in evs if e.round == 2 and e.type in (
        "post", "action_committed", "delivery", "turn_started", "turn_ended", "metric")]
    assert evs[-1].type == "run_ended" and evs[-1].reason == "hard_ceiling"
    # the ledger keeps the spend, round 2's included
    responses = by_type(run, "inference_response")
    spend = run.spend
    assert spend["swarm"] == pytest.approx(sum(r.cost_usd for r in responses))
    assert spend["swarm"] > by_type(run, "budget")[-1].spent_swarm
    assert spend["swarm"] <= 0.07 and spend["reserved"] == 0
    assert run.meta["budget"]["hard_usd"] == 0.07

    resumed = run.resume(budget=Budget(hard_usd=10.0))
    assert resumed.status == "ended" and resumed.end_reason == "max_rounds"
    assert resumed.meta["budget"]["hard_usd"] == 10.0
    assert resumed.spec.budget.hard_usd == 0.07  # the archived spec is unchanged
    changed = by_type(resumed, "budget_changed")
    assert len(changed) == 1 and changed[0].old["hard_usd"] == 0.07 and changed[0].new["hard_usd"] == 10
    discarded = [json.loads(x) for x in (run.dir / "discarded.jsonl").read_text().splitlines()]
    assert {e["type"] for e in discarded} >= {"round_started", "inference_attempt", "run_ended"}
    # completed round-2 calls from the aborted attempt are served from cache on resume
    r2 = [e for e in resumed.events_all if e.type == "inference_response" and e.round == 2]
    assert any(e.cached for e in r2)

    reference = llm_experiment(N, budget=Budget(hard_usd=10.0)).run(seed=SEED, max_rounds=ROUNDS,
                                                                    out=tmp_path / "ref")
    assert comparable(resumed) == comparable(reference)
    assert resumed.score == reference.score
    # total spend is the same: every call was paid once
    assert resumed.spend["swarm"] == pytest.approx(reference.spend["swarm"])
    assert resumed.spend["calls"] == reference.spend["calls"]
    Run.load(resumed.dir)


def test_hard_ceiling_in_round_one_is_resumable(tmp_path):
    run = llm_experiment(N, budget=Budget(hard_usd=0.001)).run(seed=SEED, max_rounds=2,
                                                               out=tmp_path)
    assert run.end_reason == "hard_ceiling" and run.meta["last_round"] == 0
    assert run.spend["calls"] == 0  # the very first reservation was refused
    resumed = run.resume(budget=Budget())
    assert resumed.end_reason == "max_rounds" and resumed.meta["last_round"] == 2


def test_soft_budget_ends_at_round_boundary(tmp_path):
    run = llm_experiment(N, budget=Budget(soft_usd=0.06)).run(seed=SEED, max_rounds=10,
                                                              out=tmp_path / "a")
    assert run.end_reason == "soft_budget"
    evs = list(run.events_all)
    last = run.meta["last_round"]
    assert 1 < last < 10
    # the last round is fully committed, and nothing of the next round started
    assert [e.round for e in evs if e.type == "round_committed"][-1] == last
    assert not [e for e in evs if e.round > last]
    assert sum(1 for e in evs if e.type == "turn_ended" and e.round == last) == N
    budgets = by_type(run, "budget")
    assert budgets[-1].spent_swarm >= 0.06 > budgets[-2].spent_swarm
    assert evs[-1].type == "run_ended" and evs[-1].reason == "soft_budget"
    # resuming without a new budget stops again at once; with one, it continues
    again = run.resume()
    assert again.end_reason == "soft_budget" and again.meta["last_round"] == last
    more = run.resume(budget=Budget(soft_usd=100))
    assert more.end_reason == "max_rounds" and more.meta["last_round"] == 10
    reference = llm_experiment(N).run(seed=SEED, max_rounds=10, out=tmp_path / "ref")
    assert comparable(more) == comparable(reference)


def test_yaml_budget_and_providers(tmp_path):
    path = tmp_path / "exp.yaml"
    path.write_text("""
name: llmyaml
budget: {soft_usd: 0.5, hard_usd: 1.0, measurement_usd: 0.1}
providers:
  fake: {type: fake, params: {pricing: {"*": [10.0, 50.0, 1.0]}}}
options: {max_rounds: 2}
arms:
  A:
    world: flaggame
    participants:
      - {type: "tests.helpers:FakeLLM", count: 3}
""")
    doc = load_experiment_yaml(path)
    assert doc["providers"]["fake"]["type"] == "fake"
    exp = Experiment.from_yaml(path, "A")
    assert exp.budget == Budget(soft_usd=0.5, hard_usd=1.0, measurement_usd=0.1)
    assert exp.providers["fake"].pricing == {"*": (10.0, 50.0, 1.0)}
    spec = exp.to_spec(1, 2)
    assert spec.providers["fake"].type == "fake"
    again = Experiment.from_spec(spec)
    assert again.providers["fake"].pricing == exp.providers["fake"].pricing
    run = exp.run(seed=1, out=tmp_path / "runs")
    assert run.end_reason == "max_rounds" and run.spend["calls"] == 3 * 2 * 2
    exp.to_yaml(tmp_path / "out.yaml")
    assert Experiment.from_yaml(tmp_path / "out.yaml", "A").to_spec(1, 2) == spec.model_copy(
        update={"arm": "A"})


def test_unknown_pricing_is_a_construction_error():
    from swarmlab.providers.base import UnknownModelPricing
    from swarmlab.world.flaggame import FlagGame

    from .helpers import FakeLLM

    with pytest.raises(UnknownModelPricing):
        Experiment(name="x", world=FlagGame(), participants=[FakeLLM(model="hf:Qwen/Unpriced-1B")])
    with pytest.raises(ValueError):
        Experiment(name="x", world=FlagGame(), participants=[FakeLLM(model="nosuch:model")])


def test_estimate():
    from swarmlab.participants import Silent
    from swarmlab.world.flaggame import FlagGame

    from .helpers import FakeLLM

    exp = Experiment(name="e", world=FlagGame(),
                     participants=[FakeLLM(model="anthropic:claude-haiku-4-5")] * 3 + [Silent()])
    est = exp.estimate(seed=1, max_rounds=10)
    per_call = (3000 * 1.0 + 300 * 5.0) / 1e6
    assert est["llm_agents"] == 3 and est["agents"] == 4 and est["calls"] == 60
    assert est["usd"] == pytest.approx(60 * per_call)
    assert est["by_model"] == {"anthropic:claude-haiku-4-5": pytest.approx(60 * per_call)}
