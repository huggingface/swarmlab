"""Admission under `budget.total_usd` with runs in flight (the m6-flag-paper refusals).

`fixtures/m6-flag-paper.ledger.jsonl` is the experiment ledger of the m6-flag-paper grid cut at
the moment `swarmlab run --parallel 4` refused `broadcast-128__s4` ("spent $8.9105 + next run's
hard_usd $3 ... ($7.4407 of it reserved by runs still in flight)"), with each run instance's
per-commit `running` rows thinned to its last two. Three runs were in flight then: two
broadcast-128 runs ($3 hard each) and one pairwise-128 run ($1.50).
"""
import json
import os
import shutil
import socket
import threading
import time
from pathlib import Path

import pytest
import yaml

from swarmlab import Experiment
from swarmlab import budget as budget_mod
from swarmlab.budget import ExperimentLedger

from .test_cli_run_all import invoke
from .test_experiment_ledger import SPEC

FIXTURE = Path(__file__).parent / "fixtures" / "m6-flag-paper.ledger.jsonl"
HOST = "r-cmpatino-agent-manager-qdevzw5r-17180-2tzr4"  # the host that wrote the fixture
IN_FLIGHT = {"m6-flag-paper__broadcast-128__s2", "m6-flag-paper__broadcast-128__s3",
             "m6-flag-paper__pairwise-128__s10"}


def saved_ledger(tmp_path) -> ExperimentLedger:
    shutil.copy(FIXTURE, tmp_path / FIXTURE.name)
    return ExperimentLedger(tmp_path, "m6-flag-paper")


def test_saved_ledger_read_later_from_another_host_reserves_nothing(tmp_path, monkeypatch):
    """The fixture's processes are long gone. Read from another host (a restarted container
    gets a new hostname), their `running` rows reserved $7.44 forever and refused the run."""
    monkeypatch.setattr(socket, "gethostname", lambda: "another-host")
    led = saved_ledger(tmp_path)
    state = led.state()
    assert state.spent == pytest.approx(1.4698, abs=1e-4)
    assert state.in_flight == () and state.reserved == 0
    adm = led.admit("m6-flag-paper__broadcast-128__s4", 3.0, 10.0)
    assert adm.refusal is None and adm.checked == pytest.approx(1.4698, abs=1e-4)


def test_saved_ledger_with_its_runs_alive(tmp_path, monkeypatch):
    """The state at the refusal itself: the three runs in flight are counted per run id with
    their remaining headroom; a $3 run has to wait for them (it is not refused for good), a
    $0.05 run is admitted, and the message names the runs in flight."""
    monkeypatch.setattr(socket, "gethostname", lambda: HOST)
    monkeypatch.setattr(budget_mod, "_pid_alive", lambda pid: True)
    led = saved_ledger(tmp_path)
    state = led.state()
    assert {f.run_id for f in state.in_flight} == IN_FLIGHT
    assert state.reserved == pytest.approx(7.4407, abs=1e-4)
    adm = led.admit("m6-flag-paper__broadcast-128__s4", 3.0, 10.0)
    assert adm.refusal and adm.wait  # 1.47 spent + 3 fits once the runs in flight end
    for rid in IN_FLIGHT:
        assert rid in adm.refusal
    assert "m6-flag-paper__broadcast-128__s2 $2.9910" in adm.refusal
    small = led.admit("m6-flag-paper__manager-4__s1", 0.05, 10.0)
    assert small.refusal is None


def test_saved_ledger_compacts_to_one_row_per_run_instance(tmp_path, monkeypatch):
    monkeypatch.setattr(socket, "gethostname", lambda: HOST)
    monkeypatch.setattr(budget_mod, "_pid_alive", lambda pid: True)
    led = saved_ledger(tmp_path)
    before = led.state()
    n_rows = len(led.rows())
    assert led.compact() > 0
    rows = led.rows()
    assert len(rows) < n_rows / 1.5
    assert len({r["key"] for r in rows}) == len(rows)
    after = led.state()
    assert after.spent == pytest.approx(before.spent) and after.reserved == pytest.approx(
        before.reserved)
    assert {f.run_id for f in after.in_flight} == IN_FLIGHT


def _row(led, rid, key, spend, status, hard, pid=None, host=None, ts=None):
    row = {"ts": time.time() if ts is None else ts, "pid": os.getpid() if pid is None else pid,
           "host": socket.gethostname() if host is None else host, "run_id": rid, "key": key,
           "spec_hash": None, "spend_usd": spend, "status": status, "hard_usd": hard}
    with led.lock() as f:
        led._append(f, json.dumps(row) + "\n")


def test_four_runs_in_flight_far_under_the_cap(tmp_path):
    """Nothing may be refused while spend + every reservation fits; a finished run releases
    its reservation; a dead process or a silent host does not reserve."""
    led = ExperimentLedger(tmp_path, "x")
    for i in range(4):
        adm = led.admit(f"x__s{i}", 3.0, 100.0)
        assert adm.refusal is None
        led.record(f"x__s{i}", f"x__s{i}@{i}", 0.01, "running", hard=3.0)
    state = led.state()
    assert len(state.in_flight) == 4 and state.reserved == pytest.approx(4 * 2.99)
    led.record("x__s0", "x__s0@0", 0.02, "ended", hard=3.0)
    state = led.state()
    assert {f.run_id for f in state.in_flight} == {"x__s1", "x__s2", "x__s3"}
    assert state.spent == pytest.approx(0.05) and state.reserved == pytest.approx(3 * 2.99)
    # a process that died on this host, and a host silent for longer than STALE_S
    _row(led, "x__s8", "x__s8@1", 0.5, "running", 3.0, pid=2 ** 22 + 7)
    _row(led, "x__s9", "x__s9@admit", 0.0, "admitted", 3.0, pid=2 ** 22 + 9)
    _row(led, "x__s10", "x__s10@1", 0.5, "running", 3.0, host="elsewhere",
         ts=time.time() - budget_mod.STALE_S - 5)
    # a host that sent a heartbeat recently does reserve
    _row(led, "x__s11", "x__s11@1", 0.5, "running", 3.0, host="elsewhere")
    state = led.state()
    assert {f.run_id for f in state.in_flight} == {"x__s1", "x__s2", "x__s3", "x__s11"}
    assert state.spent == pytest.approx(0.05 + 3 * 0.01 + 1.5)
    assert state.reserved == pytest.approx(3 * 2.99 + 2.5)


def test_a_later_row_of_the_same_run_id_supersedes_a_stale_instance(tmp_path):
    """In-flight state is the latest row per run id: a `running` row left by an instance of
    this very process (live pid) that crashed is released by the `failed` row written for the
    run id, and by a new instance of the same run id."""
    led = ExperimentLedger(tmp_path, "x")
    led.record("x__s1", "x__s1@admit", 0.0, "admitted", hard=1.0)
    led.record("x__s1", "x__s1@100", 0.2, "running", hard=1.0)
    led.record("x__s1", "x__s1@admit", 0.0, "failed")
    state = led.state()
    assert state.in_flight == () and state.spent == pytest.approx(0.2)
    led.record("x__s1", "x__s1@200", 0.1, "running", hard=1.0)
    state = led.state()
    assert [f.key for f in state.in_flight] == ["x__s1@200"]
    assert state.spent == pytest.approx(0.3) and state.reserved == pytest.approx(0.9)


def test_backfilled_unfinished_run_dir_does_not_reserve(tmp_path):
    """A run dir whose run.json still says `running` (its process was killed) is backfilled
    with this process's pid: it must not count as in flight."""
    led = ExperimentLedger(tmp_path, "x")
    led.backfill({"run_id": "x__s1", "started_at": "1", "status": "running",
                  "ledger": {"swarm": 0.3}, "budget": {"hard_usd": 2.0}, "spec": {}})
    state = led.state()
    assert state.in_flight == () and state.spent == pytest.approx(0.3)


PARALLEL4 = (SPEC.replace("seeds: [1, 2, 3]", "seeds: [1, 2, 3, 4]")
             .replace("budget: {hard_usd: 0.05, total_usd: 0.12}",
                      "budget: {hard_usd: 1.0, total_usd: 3.0}")
             .replace("world: {type: flaggame}",
                      "world: {type: \"tests.test_experiment_ledger:SlowFlagGame\"}")
             + """\
  B:
    world: {type: "tests.test_experiment_ledger:SlowFlagGame"}
    participants: [{type: llm, count: 2, params: {model: "paid:reader", max_tokens: 64, max_calls: 2, temperature: 0.5}}]
""")


def test_parallel_runs_wait_for_headroom_instead_of_being_refused(tmp_path):
    """`--parallel 4`, hard $1 each, total $3, actual spend cents: the fourth run used to be
    refused (3 x $1 reserved + $1 > $3) and every later run with it."""
    spec, out = tmp_path / "led.yaml", tmp_path / "runs"
    spec.write_text(PARALLEL4)
    res, data = invoke("run", spec, "--out", out, "--yes", "--parallel", 4, "--json")
    assert res.exit_code == 0, res.output
    assert [r["outcome"] for r in data["runs"]] == ["ran"] * 8
    assert data["capped"] is None
    assert "waiting" in res.stderr and "in flight" in res.stderr
    led = ExperimentLedger(out, "led")
    state = led.state()
    assert state.in_flight == () and state.spent <= 3.0
    # one row per state change (admitted, running, ended), not one per commit
    per_key: dict[str, int] = {}
    for r in led.rows():
        per_key[r["key"]] = per_key.get(r["key"], 0) + 1
    assert max(per_key.values()) <= 2


def test_a_refused_run_does_not_cap_the_runs_after_it(tmp_path):
    """A run that cannot fit even with nothing in flight is capped; a later, smaller run that
    fits still runs."""
    doc = yaml.safe_load(PARALLEL4)
    doc["seeds"] = [1]
    doc["budget"] = {"hard_usd": 0.05, "total_usd": 0.5}
    doc["arms"]["A"]["budget"] = {"hard_usd": 1.0}
    spec, out = tmp_path / "led.yaml", tmp_path / "runs"
    spec.write_text(yaml.safe_dump(doc))
    res, data = invoke("run", spec, "--out", out, "--yes", "--json")
    assert res.exit_code == 1, res.output
    assert [r["outcome"] for r in data["runs"]] == ["capped", "ran"]
    assert data["capped"]["skipped"] == ["led__A__s1"]


def test_refusal_names_the_runs_in_flight(tmp_path):
    doc = yaml.safe_load(PARALLEL4)
    doc["seeds"] = [1]
    doc["budget"] = {"hard_usd": 0.05, "total_usd": 0.6}
    spec, out = tmp_path / "led.yaml", tmp_path / "runs"
    spec.write_text(yaml.safe_dump(doc))
    led = ExperimentLedger(out, "led")
    _row(led, "led__C__s9", "led__C__s9@1", 0.58, "running", 1.0, host="elsewhere", pid=4242)
    res, data = invoke("run", spec, "--arm", "A", "--seed", 1, "--out", out, "--yes", "--json")
    assert res.exit_code == 1, res.output
    assert "led__C__s9 $0.4200 (running, pid 4242 on elsewhere)" in res.stderr
    assert data["capped"]["in_flight"][0]["run_id"] == "led__C__s9"


def test_run_waits_for_a_run_in_another_process(tmp_path, monkeypatch):
    """A run in flight elsewhere holds the headroom; `run` (and `run_all`) wait for it to end
    instead of refusing."""
    monkeypatch.setattr(budget_mod, "ADMIT_POLL_S", 0.05)
    doc = yaml.safe_load(PARALLEL4)
    doc["seeds"] = [1]
    doc["budget"] = {"hard_usd": 1.0, "total_usd": 1.5}
    spec, out = tmp_path / "led.yaml", tmp_path / "runs"
    spec.write_text(yaml.safe_dump(doc))
    led = ExperimentLedger(out, "led")

    def other_process(rid):
        _row(led, rid, f"{rid}@1", 0.01, "running", 1.0, host="elsewhere")
        time.sleep(0.4)
        _row(led, rid, f"{rid}@1", 0.02, "ended", 1.0, host="elsewhere")

    t = threading.Thread(target=other_process, args=("led__C__s9",))
    t.start()
    time.sleep(0.05)
    res, data = invoke("run", spec, "--arm", "A", "--seed", 1, "--out", out, "--yes", "--json")
    t.join()
    assert res.exit_code == 0, res.output
    assert "waiting" in res.stderr and "led__C__s9" in res.stderr
    t = threading.Thread(target=other_process, args=("led__C__s8",))
    t.start()
    time.sleep(0.05)
    runs = Experiment.from_yaml(spec, "B").run_all([1], out=out)
    t.join()
    assert [r.id for r in runs] == ["led__B__s1"]


def test_heartbeat_rows_are_throttled(tmp_path, monkeypatch):
    """A running run writes a `running` row at most every HEARTBEAT_S (here 0: every commit)."""
    monkeypatch.setattr(budget_mod, "HEARTBEAT_S", 0.0)
    doc = yaml.safe_load(PARALLEL4)
    doc["options"]["max_rounds"] = 4
    spec, out = tmp_path / "led.yaml", tmp_path / "runs"
    spec.write_text(yaml.safe_dump(doc))
    Experiment.from_yaml(spec, "A").run(seed=1, out=out)
    rows = [r for r in ExperimentLedger(out, "led").rows() if r["run_id"] == "led__A__s1"]
    assert sum(r["status"] == "running" for r in rows) >= 4
    assert rows[-1]["status"] == "ended"
