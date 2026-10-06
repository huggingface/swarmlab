"""The `swarmlab` CLI (docs/INTERFACE.md §16) through typer's CliRunner."""
import json
from pathlib import Path

import pytest
from typer.testing import CliRunner

from swarmlab import Run
from swarmlab.cli import app

EXAMPLE = Path(__file__).resolve().parents[1] / "examples" / "flaggame_m1a.yaml"
SMALL = """\
name: cli-small
options: {max_rounds: 4}
arms:
  A:
    world: {type: flaggame}
    participants: [{type: evidence_aggregator, count: 4}]
    medium: {topology: gossip}
    metrics: [belief.consensus, comm.read_rate]
  B:
    world: {type: flaggame}
    participants: [{type: evidence_aggregator, count: 4}]
    metrics: [belief.consensus]
"""

runner = CliRunner()


def invoke(*args):
    res = runner.invoke(app, [str(a) for a in args])
    return res, (json.loads(res.stdout) if "--json" in args and res.stdout.strip() else None)


@pytest.fixture
def spec(tmp_path):
    p = tmp_path / "small.yaml"
    p.write_text(SMALL)
    return p


def test_validate_example_and_small(spec):
    res, data = invoke("validate", EXAMPLE, "--json")
    assert res.exit_code == 0, res.output
    assert data["ok"] and set(data["arms"]) == {"broadcast", "gossip"}
    assert all(a["n_agents"] == 16 and a["max_rounds"] == 20 for a in data["arms"].values())
    res, _ = invoke("validate", spec)
    assert res.exit_code == 0 and "cli-small" in res.stdout


@pytest.mark.parametrize("text", [
    "name: x\narms: {A: {world: flaggame}}\n",                                     # no participants
    "name: x\narms: {A: {world: flaggame, participants: [nope_agent]}}\n",         # unknown plugin
    "name: x\narms: {A: {world: {type: flaggame, params: {bogus: 1}}, participants: [silent]}}\n",
    "name: [unclosed\n",                                                          # invalid YAML
])
def test_broken_spec_exits_2(tmp_path, text):
    p = tmp_path / "bad.yaml"
    p.write_text(text)
    res, data = invoke("validate", p, "--json")
    assert res.exit_code == 2, res.output
    assert data["ok"] is False and data["exit_code"] == 2
    res, _ = invoke("run", p, "--seed", 1, "--out", tmp_path / "runs")
    assert res.exit_code == 2
    assert "error" in res.stderr


def test_run_unknown_arm_and_missing_arm_exit_2(spec, tmp_path):
    res, _ = invoke("run", spec, "--arm", "Z", "--seed", 1, "--out", tmp_path)
    assert res.exit_code == 2 and "unknown arm" in res.stderr
    res, _ = invoke("run", spec, "--seed", 1, "--out", tmp_path)       # two arms, no --arm
    assert res.exit_code == 2


def test_run_replay_resume_fork_view(spec, tmp_path):
    out = tmp_path / "runs"
    res, data = invoke("run", spec, "--arm", "A", "--seed", 2, "--out", out, "--json")
    assert res.exit_code == 0, res.output
    run_dir = Path(data["run_dir"])
    assert data["run_id"] == "cli-small__A__s2" and run_dir == out / "cli-small__A__s2"
    assert data["last_round"] == 4 and data["end_reason"] == "max_rounds"
    assert len(data["spec_hash"]) == 64 and data["score"]["n_guessed"] == 4
    assert set(data["metrics"]) == {"belief.consensus", "comm.read_rate"}
    assert data["metrics"]["belief.consensus"]["round"] == 4
    # same answer as the Python API
    py = Run(run_dir)
    assert data["score"] == py.score and data["metrics"]["belief.consensus"]["value"] == py.metrics["belief.consensus"][-1][1]

    res, data2 = invoke("replay", run_dir, "--json")
    assert res.exit_code == 0 and data2["replay"] == "ok" and data2["score"] == data["score"]

    res, data3 = invoke("resume", run_dir, "--json")                  # ended run: no-op
    assert res.exit_code == 0 and data3["last_round"] == 4

    res, f = invoke("fork", run_dir, "--at", 2, "--json")
    assert res.exit_code == 0, res.output
    assert f["parent_run"] == "cli-small__A__s2" and f["fork_round"] == 2 and f["last_round"] == 4

    res, f2 = invoke("fork", run_dir, "--at", 2, "--spec", spec, "--max-rounds", 6,
                     "--out", tmp_path / "forks", "--json")
    assert res.exit_code == 0, res.output
    assert f2["run_dir"].startswith(str(tmp_path / "forks")) and f2["last_round"] == 6

    res, v = invoke("view", run_dir, "--json")
    assert res.exit_code == 0 and Path(v["view"]).exists() and v["view"].endswith("view.html")

    res, _ = invoke("run", spec, "--arm", "A", "--seed", 2, "--out", out)   # log exists already
    assert res.exit_code == 1


def test_human_output_and_errors(spec, tmp_path):
    res, _ = invoke("run", spec, "--arm", "B", "--seed", 1, "--max-rounds", 2, "--out", tmp_path)
    assert res.exit_code == 0 and "cli-small__B__s1" in res.stdout and "rounds=2" in res.stdout
    res, data = invoke("replay", tmp_path / "nope", "--json")
    assert res.exit_code == 1 and data["ok"] is False
    res, _ = invoke("fork", tmp_path / "cli-small__B__s1", "--at", 9)
    assert res.exit_code == 1


def test_replay_mismatch_exits_1(spec, tmp_path):
    res, data = invoke("run", spec, "--arm", "B", "--seed", 1, "--out", tmp_path, "--json")
    meta_path = Path(data["run_dir"]) / "run.json"
    meta = json.loads(meta_path.read_text())
    meta["score"]["accuracy"] = 0.123
    meta_path.write_text(json.dumps(meta))
    res, data = invoke("replay", meta_path.parent, "--json")
    assert res.exit_code == 1 and "ReplayMismatch" in data["error"]
