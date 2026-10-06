"""M1b CLI additions: estimate printout before budgeted runs, `estimate`, resume budget flags."""
import json

from swarmlab import Run

from .test_cli import EXAMPLE, invoke

TINY = """\
name: cli-llm
budget: {hard_usd: 0.0002}
providers:
  fake: {type: fake, params: {pricing: {"*": [10.0, 50.0, 1.0]}}}
options: {max_rounds: 3}
arms:
  A:
    world: {type: flaggame}
    participants: [{type: llm, count: 2, params: {model: "fake:reader", max_tokens: 64}}]
    probes: [belief]
    metrics: [belief.consensus, {type: belief.consensus, params: {source: "probe:belief"}}]
"""


def test_estimate_command():
    res, data = invoke("estimate", EXAMPLE, "--arm", "llm", "--seed", 1, "--json")
    assert res.exit_code == 0, res.output
    assert data["llm_agents"] == 8 and data["rounds"] == 6 and data["calls_per_turn"] == 20
    assert data["probe_calls"] == 48 and data["usd"] > 0
    res, data = invoke("estimate", EXAMPLE, "--arm", "llm", "--calls-per-turn", 2, "--max-rounds", 3, "--json")
    assert data["calls"] == 8 * 3 * 2
    res, _ = invoke("estimate", EXAMPLE, "--arm", "llm")
    assert res.exit_code == 0 and "worst-case $" in res.stdout


def test_run_prints_estimate_only_with_a_budget(tmp_path):
    res, _ = invoke("run", EXAMPLE, "--arm", "llm", "--seed", 1, "--max-rounds", 2, "--out", tmp_path)
    assert res.exit_code == 1 and "--yes" in res.stderr  # a budget asks first; no TTY answers no
    res, _ = invoke("run", EXAMPLE, "--arm", "llm", "--seed", 1, "--max-rounds", 2, "--out", tmp_path,
                    "--yes")
    assert res.exit_code == 0, res.output
    assert res.stdout.startswith("estimate: arm=llm worst-case $")
    res, data = invoke("run", EXAMPLE, "--arm", "llm", "--seed", 2, "--max-rounds", 2, "--out", tmp_path,
                       "--json", "--yes")
    assert res.exit_code == 0 and data["status"] == "ended"  # stdout is still one JSON object
    assert "estimate:" in res.stderr
    res, _ = invoke("run", EXAMPLE, "--arm", "broadcast", "--seed", 1, "--max-rounds", 1, "--out", tmp_path)
    assert "estimate:" not in res.stdout  # zero budget: no printout


def test_resume_with_budget_flags(tmp_path):
    spec = tmp_path / "tiny.yaml"
    spec.write_text(TINY)
    res, data = invoke("run", spec, "--seed", 1, "--out", tmp_path, "--json", "--yes")
    assert res.exit_code == 0, res.output
    assert data["end_reason"] == "hard_ceiling"
    run_dir = data["run_dir"]
    res, data = invoke("resume", run_dir, "--budget-hard", 5, "--budget-measurement", 1, "--json")
    assert res.exit_code == 0, res.output
    assert data["end_reason"] == "max_rounds" and data["last_round"] == 3
    meta = json.loads((Run(run_dir).dir / "run.json").read_text())
    assert meta["budget"] == {"soft_usd": 0.0, "hard_usd": 5.0, "measurement_usd": 1.0}
    assert any(e["type"] == "budget_changed" for e in Run(run_dir).events)
