"""The estimate prices each group's own `max_calls`, honours `--calls-per-turn`, and says which
calls-per-turn value it used."""
from .test_cli import invoke

SPEC = """\
name: cpt
seeds: [1]
budget: {hard_usd: 1.0}
providers:
  fake: {type: fake, params: {pricing: {"*": [10.0, 50.0, 1.0]}}}
options: {max_rounds: 2}
arms:
  capped:
    world: {type: flaggame}
    participants: [{type: llm, count: 2, params: {model: "fake:reader", max_calls: 6}}]
  mixed:
    world: {type: flaggame}
    participants:
      - {type: llm, count: 1, params: {model: "fake:reader", max_calls: 4}}
      - {type: llm, count: 1, params: {model: "fake:reader"}}
"""


def test_group_max_calls_is_what_is_priced_and_printed(tmp_path):
    spec = tmp_path / "cpt.yaml"
    spec.write_text(SPEC)
    res, est = invoke("estimate", spec, "--arm", "capped", "--seed", 1, "--json")
    assert res.exit_code == 0, res.output
    assert est["calls_per_turn"] == 6 and est["calls"] == 2 * 2 * 6
    assert est["calls_per_turn_source"] == "participant max_calls, under options.max_calls_per_turn 20"
    res, _ = invoke("estimate", spec, "--arm", "capped", "--seed", 1)
    assert "x 6 calls/turn (participant max_calls" in res.stdout and "20 calls/turn" not in res.stdout
    # --calls-per-turn takes effect (below the cap) and the line says so
    res, est = invoke("estimate", spec, "--arm", "capped", "--seed", 1, "--calls-per-turn", 3, "--json")
    assert est["calls_per_turn"] == 3 and est["calls"] == 12
    assert est["calls_per_turn_source"] == "--calls-per-turn 3"
    res, _ = invoke("estimate", spec, "--arm", "capped", "--seed", 1, "--calls-per-turn", 3)
    assert "x 3 calls/turn (--calls-per-turn 3)" in res.stdout
    # mixed groups: both values are reported, and the table has a calls/turn column
    res, data = invoke("estimate", spec, "--json")
    assert data["arms"]["mixed"]["calls_per_turn"] == [4, 20]
    assert data["arms"]["mixed"]["calls"] == 2 * (4 + 20)
    res, _ = invoke("estimate", spec, "--calls-per-turn", 5)
    assert "calls/turn" in res.stdout.splitlines()[0]
    assert "4-5" in res.stdout and "--calls-per-turn 5" in res.stdout


def test_run_prints_the_same_calls_per_turn(tmp_path):
    spec = tmp_path / "cpt.yaml"
    spec.write_text(SPEC)
    res, _ = invoke("run", spec, "--arm", "capped", "--seed", 1, "--out", tmp_path / "runs", "--yes")
    assert res.exit_code == 0, res.output
    assert "2 model agents x 2 rounds x 6 calls/turn (participant max_calls" in res.stdout
