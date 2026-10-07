"""`estimate --from RUN_DIR` and the measured soft/hard gap warning (field notes item 5)."""
import pytest
import yaml

from swarmlab import Experiment, Run
from swarmlab.experiment import _fit_line

from .test_cli_run_all import invoke

SPEC = """\
name: meas
seeds: [1, 2]
budget: {soft_usd: 0.3, hard_usd: 0.5}
providers:
  paid: {type: fake, params: {pricing: {"*": [10.0, 50.0, 1.0]}}}
options: {max_rounds: 3}
arms:
  A:
    world: {type: flaggame}
    participants:
      - {type: llm, count: 2, params: {model: "paid:reader", max_tokens: 64}}
      - {type: evidence_aggregator, count: 1}
    probes: [belief]
"""


def spec_file(tmp_path, **budget):
    doc = yaml.safe_load(SPEC)
    doc["budget"].update(budget)
    p = tmp_path / "meas.yaml"
    p.write_text(yaml.safe_dump(doc))
    return p


def test_fit_line():
    assert _fit_line([(0, 10.0), (1, 12.0), (2, 14.0)]) == pytest.approx((10.0, 2.0))
    assert _fit_line([(0, 5.0)]) == (5.0, 0.0)


def test_measured_figures_and_estimate_from(tmp_path):
    out = tmp_path / "runs"
    run = Experiment.from_yaml(spec_file(tmp_path), "A").run(seed=1, out=out)
    m = Run(run.dir).measured()
    assert m["run_id"] == run.id and m["rounds"] == 3 and m["turns"] == 6  # model agents only
    assert m["calls"] >= m["turns"] and m["calls_per_turn"] == pytest.approx(m["calls"] / 6)
    assert m["prompt_tokens"] > 0 and m["prompt_growth"] > 0  # full memory: the context grows
    assert m["completion_tokens"] > 0 and m["probe_call_usd"] > 0
    assert [r for r, _ in m["per_round"]] == [1, 2, 3]
    # pricing the spec with the run's own figures reproduces its swarm spend
    res, est = invoke("estimate", spec_file(tmp_path), "--seed", 1, "--from", run.dir, "--json")
    assert res.exit_code == 0, res.output
    assert est["calls_per_turn_source"] == f"measured in {run.id}"
    assert est["usd"] == pytest.approx(run.spend["swarm"] + run.spend["measurement"], rel=0.05)
    worst = invoke("estimate", spec_file(tmp_path), "--seed", 1, "--json")[1]
    assert est["usd"] < worst["usd"]
    res, _ = invoke("estimate", spec_file(tmp_path), "--from", run.dir)
    assert res.exit_code == 0 and f"measured in {run.id}" in res.stdout
    assert "calls/turn, prompt tokens per call" in res.stdout
    res, _ = invoke("estimate", spec_file(tmp_path), "--from", run.dir, "--calls-per-turn", 2)
    assert res.exit_code == 2
    res, _ = invoke("estimate", spec_file(tmp_path), "--from", tmp_path)
    assert res.exit_code == 2 and "no run.json" in res.stderr


def test_gap_warning_uses_a_finished_run_of_the_arm(tmp_path):
    out = tmp_path / "runs"
    # no finished run: the gap is compared with half the worst-case round
    res, data = invoke("estimate", spec_file(tmp_path), "--seed", 1, "--json")
    worst_round = data["usd_per_round"]
    gap = 0.5 * worst_round - 0.001
    spec = spec_file(tmp_path, soft_usd=0.3, hard_usd=0.3 + gap)
    res, _ = invoke("run", spec, "--seed", 1, "--out", out, "--yes")
    assert res.exit_code == 0, res.output
    assert "less than half of one round's estimated worst case" in res.stdout
    # with a finished run of the arm, its measured round cost decides (far below the worst case)
    res, _ = invoke("run", spec, "--seed", 2, "--out", out, "--yes")
    assert res.exit_code == 0, res.output
    assert "warning:" not in res.stdout
    m = Run(out / "meas__A__s1").measured()
    assert m["run_id"] == "meas__A__s1"
    tight = spec_file(tmp_path, soft_usd=0.3, hard_usd=0.3001)
    res, _ = invoke("run", tight, "--seed", 2, "--out", tmp_path / "x", "--yes")
    assert "estimated worst case" in res.stdout  # other --out: nothing measured there
    res, _ = invoke("run", tight, "--seed", 2, "--out", out, "--rerun", "--yes")
    assert "as measured in meas__A__s" in res.stdout
