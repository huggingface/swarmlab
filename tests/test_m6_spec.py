"""WP16 §3: experiments/m6_flag_vlm.yaml (broadcast / gossip / manager x N in {4, 16, 128})."""
from __future__ import annotations

from pathlib import Path

import pytest
import yaml

from swarmlab import Experiment
from swarmlab.medium.topology import Broadcast, Gossip, Star
from swarmlab.roles import role_name
from swarmlab.spec import load_experiment_yaml, validate_experiment_doc

SPEC = Path(__file__).resolve().parents[1] / "experiments" / "m6_flag_vlm.yaml"
MODEL = "hf:google/gemma-4-26B-A4B-it:deepinfra"
HARD = {4: 0.30, 16: 1.20, 128: 16.0}


@pytest.fixture(scope="module")
def arms():
    return Experiment.arms_from_yaml(SPEC)


def test_validates_and_has_nine_arms(arms):
    doc = load_experiment_yaml(SPEC)
    assert doc["seeds"] == [1] and doc["options"]["max_rounds"] == 10
    assert doc["budget"]["total_usd"] == 30
    assert list(arms) == [f"{p}-{n}" for n in (4, 16, 128) for p in ("bc", "gossip", "manager")]
    assert not any(k.startswith("x-") for k in doc)


@pytest.mark.parametrize("protocol", ["bc", "gossip", "manager"])
@pytest.mark.parametrize("n", [4, 16, 128])
def test_arm_builds(arms, protocol, n):
    exp = arms[f"{protocol}-{n}"]
    assert len(exp.participants) == n
    for p in exp.participants:
        assert p.model == MODEL and p.max_tokens == 1024 and p.max_calls == 5
        assert p.memory == "window" and p.window_rounds == 3
    w = exp.world
    assert w.modality == "image" and w.status_tools == ("my_status",)
    assert exp.budget.hard_usd == HARD[n] and exp.budget.soft_usd == 0
    assert [p.spec()["type"] for p in exp.probes] == ["belief"]
    names = [m.name for m in exp.metrics]
    for base in ("belief.accuracy", "belief.consensus", "belief.polarization", "belief.entropy"):
        assert base in names and f"{base}@probe:belief" in names
    assert {"comm.read_rate", "comm.post_rate"} <= set(names)
    topo = exp.medium.topology
    roles = [role_name(p) for p in exp.participants]
    if protocol == "bc":
        assert isinstance(topo, Broadcast) and w.blind_agents == [] and not any(roles)
    elif protocol == "gossip":
        assert isinstance(topo, Gossip) and topo.k == 1 and w.blind_agents == []
    else:
        assert isinstance(topo, Star) and topo.center == "a000" and w.blind_agents == 1
        assert roles == ["manager"] + [None] * (n - 1)
    exp.spec_hash(1)  # the run spec resolves


def test_arms_share_the_world_draw(arms):
    from swarmlab.rng import derive

    worlds = {}
    for name in ("bc-4", "gossip-4", "manager-4"):
        w = arms[name].world
        w.reset(derive(1, "world"), ["a000", "a001", "a002", "a003"])
        worlds[name] = w
    bc, mgr = worlds["bc-4"], worlds["manager-4"]
    assert bc.verify()["candidates"] == mgr.verify()["candidates"] and bc.truth == mgr.truth
    assert {a: c for a, c in bc.crops.items() if a != "a000"} == mgr.crops


def test_memory_anchor_switches_to_full(tmp_path):
    text = SPEC.read_text().replace("x-memory: &memory { memory: window, window_rounds: 3 }",
                                    "x-memory: &memory { memory: full }")
    doc = validate_experiment_doc(yaml.safe_load(text))
    params = [g["params"] for a in doc["arms"].values() for g in a["participants"]]
    assert all(p["memory"] == "full" for p in params)


def test_x_keys_are_ignored():
    base = {"name": "x", "arms": {"a": {"world": "flaggame",
                                        "participants": [{"type": "silent", "count": 2}]}}}
    assert validate_experiment_doc({**base, "x-anchor": {"anything": 1}}) == \
        validate_experiment_doc(base)


@pytest.mark.parametrize("arm", ["bc-4", "gossip-4", "manager-4"])
def test_small_arms_run_on_the_fake_provider(tmp_path, arm):
    """The N=4 arms play end to end with the model swapped for the fake reader (text hint on)."""
    text = SPEC.read_text().replace(MODEL, "fake:reader").replace(
        "modality: image,", "modality: image, image_text_hint: true,")
    path = tmp_path / "m6.yaml"
    path.write_text(text)
    exp = Experiment.from_yaml(path, arm=arm)
    run = exp.run(seed=1, max_rounds=2, out=tmp_path / "runs")
    assert run.status == "ended"
    n = {n for _, _, n in run.metrics["belief.accuracy"]}
    assert n == ({3} if arm.startswith("manager") else {4})
    if arm.startswith("manager"):
        assert not [e for e in run.events if e["type"] == "action_committed" and e["agent"] == "a000"]
