"""M6 §6-§7: experiments/m6_flag_paper.yaml (the Flag Game paper's three protocols on real flags)."""
from __future__ import annotations

import json
from pathlib import Path

import pytest
import yaml
from typer.testing import CliRunner

from swarmlab import Experiment
from swarmlab.cli import app
from swarmlab.medium.topology import Broadcast, Gossip, Star
from swarmlab.report import build_report
from swarmlab.roles import role_name
from swarmlab.scheduler import OneSpeaker
from swarmlab.spec import arm_to_runspec, dump_experiment_yaml, load_experiment_yaml

SPEC = Path(__file__).resolve().parents[1] / "experiments" / "m6_flag_paper.yaml"
MODEL = "hf:google/gemma-4-26B-A4B-it:deepinfra"
NS = (4, 16, 128)
cli = CliRunner()


@pytest.fixture(scope="module")
def arms():
    return Experiment.arms_from_yaml(SPEC)


def test_validates_with_nine_arms():
    doc = load_experiment_yaml(SPEC)
    assert doc["seeds"] == list(range(1, 11)) and doc["budget"]["total_usd"] == 15
    assert list(doc["arms"]) == [f"{p}-{n}" for p in ("pairwise", "broadcast", "manager") for n in NS]
    hard = sum(a["budget"]["hard_usd"] for a in doc["arms"].values())
    assert hard <= doc["budget"]["total_usd"]


@pytest.mark.parametrize("protocol", ["pairwise", "broadcast", "manager"])
@pytest.mark.parametrize("n", NS)
def test_arm_builds(arms, protocol, n):
    exp = arms[f"{protocol}-{n}"]
    w = exp.world
    assert (w.flags, w.candidates_mode, w.modality, w.width, w.height, w.crop_w, w.crop_h, w.cell_px) == (
        "real", "names", "image", 24, 16, 6, 4, 25)
    assert w.status_tools == ()
    for p in exp.participants:
        assert p.model == MODEL and p.temperature == 0.2 and p.report_json
        assert p.tool_protocol == "json" and p.memory == "received"
        assert p.max_tokens == (200 if protocol == "pairwise" else 250)
    spec = arm_to_runspec(load_experiment_yaml(SPEC), f"{protocol}-{n}", seed=1)
    topo = exp.medium.topology
    assert exp.medium.delivery == "push" and exp.medium.push_consume
    if protocol == "pairwise":
        assert isinstance(topo, Gossip) and topo.k == 1
        assert spec.options.commit == "immediate" and spec.options.max_rounds == 10 * n
        assert spec.options.scheduler.type == "one_speaker"
        assert spec.options.stop_when.model_dump() == {
            "metric": "belief.consensus@probe:belief", "op": ">=", "value": 1.0, "consecutive": 5}
        assert [p.every for p in exp.probes] == [n]
        assert all(p.memory_messages == 8 for p in exp.participants)
        assert len(exp.participants) == n
    elif protocol == "broadcast":
        assert isinstance(topo, Broadcast) and exp.medium.push_limit == n - 1
        assert spec.options.commit == "round_end" and spec.options.max_rounds == 10
        assert [p.every for p in exp.probes] == [1]
        assert all(p.memory_messages == n - 1 and p.answers_kept == 8 for p in exp.participants)
    else:
        assert isinstance(topo, Star) and topo.center == "a000" and exp.medium.push_limit == n
        assert w.blind_agents == 1 and w.blind_may_guess
        assert len(exp.participants) == n + 1 and not exp.probes
        assert [role_name(p) for p in exp.participants] == ["manager"] + [None] * n
        assert exp.participants[0].memory_messages == n
        assert spec.roles["manager"]["may_act"] is True
        assert spec.options.max_rounds == 10
    exp.spec_hash(1)


@pytest.mark.parametrize("arm", ["pairwise-4", "manager-4"])
def test_prompts_render(arm):
    res = cli.invoke(app, ["prompts", str(SPEC), "--arm", arm])
    assert res.exit_code == 0, res.output
    assert 'Output JSON exactly: {"country":"<one allowed country>","reason":"<one sentence>"}' in res.output
    assert "You must output only valid JSON." in res.output
    assert "Transcript memory (oldest -> newest): []" in res.output
    if arm == "manager-4":
        assert "You are the manager." in res.output and "You have no crop of your own" in res.output


def test_estimate_prices_the_grid():
    res = cli.invoke(app, ["estimate", str(SPEC), "--json"])
    assert res.exit_code == 0, res.output
    data = json.loads(res.output)
    ests = data["arms"]
    assert set(ests) == {f"{p}-{n}" for p in ("pairwise", "broadcast", "manager") for n in NS}
    for n in NS:
        # OneSpeaker: one turn per round, at most 2 model calls per report_json turn
        assert ests[f"pairwise-{n}"]["rounds"] == 10 * n
        assert ests[f"pairwise-{n}"]["calls"] == pytest.approx(10 * n * 2)
        assert ests[f"pairwise-{n}"]["probe_calls"] == 10 * n
        assert ests[f"broadcast-{n}"]["calls"] == 10 * n * 2
        assert ests[f"manager-{n}"]["calls"] == 10 * (n + 1) * 2
    for e in ests.values():
        assert e["usd_flat"] <= e["budget"]["hard_usd"]
    assert data["total_usd_flat"] < 40  # ten seeds x about $3.7 flat worst case per seed


def test_scheduler_builds_for_pairwise():
    from swarmlab.scheduler import build_scheduler

    doc = load_experiment_yaml(SPEC)
    assert isinstance(build_scheduler(doc["arms"]["pairwise-4"]["options"]["scheduler"]), OneSpeaker)


def test_n4_arms_dry_run_on_the_fake_provider(tmp_path):
    """Every N=4 arm end to end with the fake JSON answerer in place of Gemma (no paid call)."""
    raw = yaml.safe_load(SPEC.read_text())
    doc = load_experiment_yaml(SPEC)
    doc["budget"] = {}  # fake prices are nominal; fake-only runs need no caps
    for arm in doc["arms"].values():
        arm["budget"] = {}
        for g in arm["participants"]:
            g["params"]["model"] = "fake:country_reporter"
    fake = tmp_path / "m6_fake.yaml"
    dump_experiment_yaml(doc, fake)
    for arm in ("pairwise-4", "broadcast-4", "manager-4"):
        run = Experiment.from_yaml(fake, arm).run(seed=1, out=tmp_path / "runs")
        assert run.status == "ended"
        if arm == "pairwise-4":
            assert run.end_reason in ("stop_condition", "max_rounds")
            assert max(r for r, _, _ in run.metrics["belief.state"]) <= 40
        else:
            assert run.end_reason == "max_rounds"
    report = build_report(tmp_path / "runs", include_fake=True)
    section = report.split("## Terminal states (Flag Game paper)")[1].split("\n## ")[0]
    for arm in ("pairwise-4", "broadcast-4", "manager-4"):
        assert f"| {arm} (simulated) | 1 |" in section
    assert raw["name"] == "m6-flag-paper"
