"""The experiment-wide cap across invocations, processes and `run --parallel` (field notes 3)."""
import json
import os
import socket
import time

import pytest
import yaml

from swarmlab import Experiment
from swarmlab.budget import ExperimentLedger
from swarmlab.world.flaggame import FlagGame

from .test_cli_run_all import invoke


class SlowFlagGame(FlagGame):
    """FlagGame whose commit takes a moment, so concurrent runs overlap measurably."""

    def commit(self, actions):
        time.sleep(0.15)
        return super().commit(actions)


# `paid:` is a FakeProvider under another prefix: it stands in for a billed provider
SPEC = """\
name: led
seeds: [1, 2, 3]
budget: {hard_usd: 0.05, total_usd: 0.12}
providers:
  paid: {type: fake, params: {pricing: {"*": [10.0, 50.0, 1.0]}}}
options: {max_rounds: 2}
arms:
  A:
    world: {type: flaggame}
    participants: [{type: llm, count: 2, params: {model: "paid:reader", max_tokens: 64, max_calls: 2}}]
"""


def spec_file(tmp_path, text=SPEC):
    p = tmp_path / "led.yaml"
    p.write_text(text)
    return p


def test_ledger_sums_latest_spend_per_run_instance(tmp_path):
    led = ExperimentLedger(tmp_path, "x")
    led.record("x__s1", "x__s1@1", 0.01, "running", hard=0.05)
    led.record("x__s1", "x__s1@1", 0.03, "ended", hard=0.05)
    led.record("x__s1", "x__s1@2", 0.02, "ended", hard=0.05)  # the same id run again: both count
    assert led.spent() == pytest.approx(0.05)
    # a run in flight in a live process reserves its remaining headroom; a dead one does not
    led.record("x__s2", "x__s2@1", 0.01, "running", hard=0.05)
    assert led.totals() == pytest.approx((0.06, 0.04))
    with led.lock() as f:
        f.write(json.dumps({"ts": 0, "pid": 2 ** 22 + 7, "host": socket.gethostname(),
                            "run_id": "x__s3", "key": "x__s3@1", "spend_usd": 0.01,
                            "status": "running", "hard_usd": 0.05}).encode() + b"\n{torn")
    assert led.totals() == pytest.approx((0.07, 0.04))
    # admission: spend + reserved + hard must fit, and an admitted run reserves at once
    why, checked = led.admit("x__s4", 0.05, 0.2)
    assert why is None and checked == pytest.approx(0.11)
    why, _ = led.admit("x__s5", 0.05, 0.2)
    assert "> total_usd" in why and "reserved by runs still in flight" in why
    led.record("x__s4", "x__s4@admit", 0.0, "failed")  # releases the admitted reservation
    assert led.totals()[1] == pytest.approx(0.04)


def test_two_invocations_share_the_total(tmp_path):
    """Before the ledger, each `--seed` invocation saw only its own runs' spend."""
    spec, out = spec_file(tmp_path), tmp_path / "runs"
    outcomes = []
    for sd in (1, 2, 3):
        res, data = invoke("run", spec, "--seed", sd, "--out", out, "--yes", "--json")
        outcomes.append(res.exit_code)
        if sd < 3:
            assert res.exit_code == 0, res.output
    assert outcomes[2] == 1 and "total cap" in data["error"]
    assert not (out / "led__A__s3").exists()
    led = ExperimentLedger(out, "led")
    assert led.path == out / "led.ledger.jsonl"
    spent = sum(json.loads((out / f"led__A__s{s}" / "run.json").read_text())["ledger"]["swarm"]
                for s in (1, 2))
    assert led.spent() == pytest.approx(spent) and data["capped"]["spent"] == pytest.approx(spent)
    rows = led.rows()
    assert {r["run_id"] for r in rows} == {"led__A__s1", "led__A__s2"}
    assert {"ts", "pid", "spec_hash", "spend_usd", "status"} <= set(rows[-1])
    assert rows[-1]["status"] == "ended" and rows[-1]["pid"] == os.getpid()
    # a run dir from before the ledger existed is backfilled when found
    led.path.unlink()
    res, data = invoke("run", spec, "--seed", 3, "--out", out, "--yes", "--json")
    assert res.exit_code == 1 and data["capped"]["spent"] == pytest.approx(spent)
    assert all(r.get("backfilled") for r in led.rows())


def test_runner_stops_at_the_total_at_a_round_boundary(tmp_path):
    """Spend recorded by another process counts at every round boundary."""
    text = SPEC.replace("max_rounds: 2", "max_rounds: 6").replace("hard_usd: 0.05", "hard_usd: 1")
    exp = Experiment.from_yaml(spec_file(tmp_path, text), "A")
    out = tmp_path / "runs"
    ExperimentLedger(out, "led").record("led__A__s9", "led__A__s9@1", 0.07, "ended")
    run = exp.run(seed=1, out=out)
    assert run.end_reason == "total_budget" and 1 <= run.meta["last_round"] < 6
    assert 0.07 + run.spend["swarm"] >= 0.12
    # resumable like soft_budget once the total allows more
    assert run.meta["status"] == "ended"


PARALLEL = SPEC.replace("seeds: [1, 2, 3]", "seeds: [1, 2]").replace(
    "world: {type: flaggame}", "world: {type: \"tests.test_experiment_ledger:SlowFlagGame\"}"
) + """\
  B:
    world: {type: "tests.test_experiment_ledger:SlowFlagGame"}
    participants: [{type: llm, count: 2, params: {model: "paid:reader", max_tokens: 64, max_calls: 2, temperature: 0.5}}]
"""


def test_parallel_runs_concurrently_under_the_shared_cap(tmp_path):
    spec, out = spec_file(tmp_path, PARALLEL), tmp_path / "runs"
    res, data = invoke("run", spec, "--out", out, "--yes", "--parallel", 2, "--json")
    assert res.exit_code == 1, res.output
    outcomes = {r["run_id"]: r["outcome"] for r in data["runs"]}
    assert outcomes == {"led__A__s1": "ran", "led__A__s2": "ran", "led__B__s1": "capped",
                        "led__B__s2": "capped"}
    assert data["capped"]["skipped"] == ["led__B__s1", "led__B__s2"]
    led = ExperimentLedger(out, "led")
    assert led.spent() <= 0.12
    # the two runs overlapped in time
    span = {}
    for r in led.rows():
        lo, hi = span.get(r["run_id"], (r["ts"], r["ts"]))
        span[r["run_id"]] = (min(lo, r["ts"]), max(hi, r["ts"]))
    (a0, a1), (b0, b1) = span["led__A__s1"], span["led__A__s2"]
    assert a0 < b1 and b0 < a1
    # a larger total lets the rest run, in parallel too, and skips the finished ones
    doc = yaml.safe_load(PARALLEL)
    doc["budget"]["total_usd"] = 1.0
    spec.write_text(yaml.safe_dump(doc))
    res, data = invoke("run", spec, "--out", out, "--yes", "--parallel", 2, "--json")
    assert res.exit_code == 0, res.output
    assert [r["outcome"] for r in data["runs"]] == ["skipped", "skipped", "ran", "ran"]
