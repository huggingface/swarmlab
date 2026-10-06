"""M1b CLI additions: estimate printout before budgeted runs, `estimate`, resume budget flags."""
import json

import pytest

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
    assert data["arms"]["llm"]["calls"] == 8 * 3 * 2  # no --seed: every seed, per-arm results
    res, _ = invoke("estimate", EXAMPLE, "--arm", "llm", "--seed", 1)
    assert res.exit_code == 0 and "worst-case $" in res.stdout


def test_estimate_covers_every_arm_and_seed_like_run():
    res, data = invoke("estimate", EXAMPLE, "--json")
    assert res.exit_code == 0, res.output
    assert set(data["arms"]) == {"broadcast", "gossip", "llm"} and data["seeds"] == [1, 2]
    assert data["runs"] == 6 and data["prompt_growth"] == 0
    llm = data["arms"]["llm"]
    assert data["total_usd"] == pytest.approx(2 * llm["usd"]) == pytest.approx(data["total_usd_flat"])
    res, narrowed = invoke("estimate", EXAMPLE, "--arm", "llm", "--json")
    assert set(narrowed["arms"]) == {"llm"} and narrowed["runs"] == 2
    res, one_seed = invoke("estimate", EXAMPLE, "--seed", 1, "--json")
    assert one_seed["seeds"] == [1] and one_seed["runs"] == 3
    res, _ = invoke("estimate", EXAMPLE)
    assert res.exit_code == 0
    assert res.stdout.splitlines()[0].split()[:2] == ["arm", "model"]
    assert "estimate total: $" in res.stdout and "over 6 run(s)" in res.stdout
    res, _ = invoke("estimate", EXAMPLE, "--arm", "nope")
    assert res.exit_code == 2


def test_estimate_prompt_growth():
    res, flat = invoke("estimate", EXAMPLE, "--arm", "llm", "--seed", 1, "--max-rounds", 3,
                       "--calls-per-turn", 1, "--json")
    res, grown = invoke("estimate", EXAMPLE, "--arm", "llm", "--seed", 1, "--max-rounds", 3,
                        "--calls-per-turn", 1, "--prompt-growth", 1000, "--json")
    assert res.exit_code == 0, res.output
    assert grown["usd_flat"] == pytest.approx(flat["usd"]) == pytest.approx(flat["usd_flat"])
    # rounds 1..3 add 0 + 1000 + 2000 prompt tokens per call stream (turn calls and probes)
    p_in = 1.0 / 1e6  # the example's fake pricing: (1, 5, 0.1) USD per M tokens
    streams = 8 * 1 + flat["probe_calls"] // 3  # 8 agents x 1 call/turn, plus one probe each round
    assert grown["usd"] - flat["usd"] == pytest.approx(streams * 3000 * p_in)
    res, data = invoke("estimate", EXAMPLE, "--prompt-growth", 1000, "--json")
    assert data["total_usd"] > data["total_usd_flat"]
    res, _ = invoke("estimate", EXAMPLE, "--prompt-growth", 1000)
    assert "per run +growth" in res.stdout and "with prompt growth 1000 tokens/round: $" in res.stdout
    res, _ = invoke("estimate", EXAMPLE, "--arm", "llm", "--seed", 1, "--prompt-growth", 1000)
    assert "flat (no growth): $" in res.stdout
    res, _ = invoke("estimate", EXAMPLE, "--prompt-growth", -1)
    assert res.exit_code == 2


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
