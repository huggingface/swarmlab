import subprocess
import textwrap

import pytest

from swarmlab import Budget
from swarmlab.spec import (
    MediumSpec,
    PluginSpec,
    RunOptions,
    RunSpec,
    SpecError,
    arm_to_runspec,
    dump_experiment_yaml,
    dump_runspec_yaml,
    git_identity,
    load_experiment_yaml,
    load_runspec_yaml,
    runspec_to_doc,
    spec_hash,
)

YAML = textwrap.dedent(
    """
    name: flag-gossip
    budget: {soft_usd: 1.5}
    options: {max_rounds: 20}
    arms:
      A:
        world: {type: flaggame, params: {n_candidates: 8, height: 8}}
        participants:
          - {type: evidence_aggregator, count: 3}
          - {type: silent, count: 2, params: {x: 1}}
        medium: {topology: gossip, policies: [{type: delay, params: {rounds: 2}}]}
        metrics: [belief.consensus, {type: belief.polarization, params: {threshold: 0.2}}]
      B:
        world: flaggame
        participants: [enumerator]
        options: {commit: immediate}
        budget: {hard_usd: 5}
    """
)


@pytest.fixture
def yaml_path(tmp_path):
    p = tmp_path / "exp.yaml"
    p.write_text(YAML)
    return p


def make_spec(**opt) -> RunSpec:
    return RunSpec(
        experiment="e",
        world=PluginSpec(type="w", params={"b": 1, "a": {"y": 2, "x": 1}}),
        participants=[PluginSpec(type="p")] * 2,
        medium=MediumSpec(),
        metrics=[PluginSpec(type="m")],
        budget=Budget(),
        options=RunOptions(seed=1, max_rounds=5, **opt),
    )


def test_hash_stable_under_key_order():
    s1 = make_spec()
    s2 = make_spec()
    s2.world.params = {"a": {"x": 1, "y": 2}, "b": 1}
    assert spec_hash(s1) == spec_hash(s2)
    assert len(spec_hash(s1)) == 64


def test_hash_changes_with_content():
    assert spec_hash(make_spec()) != spec_hash(make_spec(commit="immediate"))
    s = make_spec()
    s.participants = list(reversed([PluginSpec(type="p"), PluginSpec(type="q")]))
    t = make_spec()
    t.participants = [PluginSpec(type="p"), PluginSpec(type="q")]
    assert spec_hash(s) != spec_hash(t)


def test_load_and_count_expansion(yaml_path):
    doc = load_experiment_yaml(yaml_path)
    spec = arm_to_runspec(doc, "A", seed=3)
    assert spec.experiment == "flag-gossip" and spec.arm == "A"
    assert [p.type for p in spec.participants] == ["evidence_aggregator"] * 3 + ["silent"] * 2
    assert spec.participants[4].params == {"x": 1}
    assert spec.medium.topology == PluginSpec(type="gossip")
    assert spec.medium.policies == [PluginSpec(type="delay", params={"rounds": 2})]
    assert spec.metrics[0] == PluginSpec(type="belief.consensus")
    assert spec.options.seed == 3 and spec.options.max_rounds == 20
    assert spec.options.commit == "round_end"
    assert spec.budget.soft_usd == 1.5


def test_arm_overrides(yaml_path):
    doc = load_experiment_yaml(yaml_path)
    spec = arm_to_runspec(doc, "B", seed=1, max_rounds=7, snapshot_every=None)
    assert spec.options.commit == "immediate"
    assert spec.options.max_rounds == 7 and spec.options.snapshot_every == 1
    assert spec.budget.soft_usd == 1.5 and spec.budget.hard_usd == 5
    assert spec.world == PluginSpec(type="flaggame")
    assert [p.type for p in spec.participants] == ["enumerator"]
    assert spec.medium == MediumSpec()


def test_yaml_round_trips(yaml_path, tmp_path):
    doc = load_experiment_yaml(yaml_path)
    spec = arm_to_runspec(doc, "A", seed=3)
    # RunSpec archive round trip
    dump_runspec_yaml(spec, tmp_path / "spec.yaml")
    back = load_runspec_yaml(tmp_path / "spec.yaml")
    assert back == spec and spec_hash(back) == spec_hash(spec)
    # experiment document round trip (count groups collapse back)
    doc2 = runspec_to_doc(spec)
    assert [g["count"] for g in doc2["arms"]["A"]["participants"]] == [3, 2]
    dump_experiment_yaml(doc2, tmp_path / "exp2.yaml")
    again = arm_to_runspec(load_experiment_yaml(tmp_path / "exp2.yaml"), "A", seed=3)
    assert spec_hash(again) == spec_hash(spec)
    # full document dump/load is idempotent
    dump_experiment_yaml(doc, tmp_path / "exp3.yaml")
    assert load_experiment_yaml(tmp_path / "exp3.yaml") == doc


@pytest.mark.parametrize(
    "text",
    [
        "arms: {A: {world: w, participants: [p]}}",  # missing name
        "name: x\narms: {}",  # no arms
        "name: x\narms: {A: {world: w, participants: []}}",  # no participants
        "name: x\narms: {A: {world: w, participants: [{type: p, count: 0}]}}",
        "name: x\narms: {A: {world: w, participants: [p], bogus: 1}}",
        "name: x\noptions: {max_round: 3}\narms: {A: {world: w, participants: [p]}}",
        "name: x\narms: {A: {world: w, participants: [p], medium: {delivery: carrier}}}",
        "- just a list",
        "name: [unclosed",
    ],
)
def test_invalid_documents(tmp_path, text):
    p = tmp_path / "bad.yaml"
    p.write_text(text)
    with pytest.raises(SpecError):
        load_experiment_yaml(p)


def test_arm_errors(yaml_path):
    doc = load_experiment_yaml(yaml_path)
    with pytest.raises(SpecError):
        arm_to_runspec(doc, "Z", seed=1)
    with pytest.raises(SpecError):
        arm_to_runspec(doc, "A", seed=1, bogus=3)
    nomax = {**doc, "options": {}}
    with pytest.raises(SpecError):
        arm_to_runspec(nomax, "A", seed=1)  # max_rounds missing


def test_git_identity(tmp_path):
    assert git_identity(tmp_path) == ("unknown", True)

    def run(*a):
        subprocess.run(["git", "-C", str(tmp_path), *a], check=True, capture_output=True)

    run("init", "-q")
    (tmp_path / "f.txt").write_text("1")
    run("add", "f.txt")
    run("-c", "user.name=t", "-c", "user.email=t@t", "commit", "-qm", "init")
    commit, dirty = git_identity(tmp_path)
    assert len(commit) == 40 and dirty is False
    (tmp_path / "untracked.txt").write_text("x")
    assert git_identity(tmp_path)[1] is False
    (tmp_path / "f.txt").write_text("2")
    assert git_identity(tmp_path) == (commit, True)
