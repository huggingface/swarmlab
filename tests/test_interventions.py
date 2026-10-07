"""M3a interventions (docs/INTERFACE-M3a.md §1, acceptance §4 items 1-4)."""
from __future__ import annotations

import os
import signal
import subprocess
import sys
from pathlib import Path
from typing import ClassVar

import pytest
import yaml

from swarmlab import Board, Experiment, Participant, Run, TurnUsage
from swarmlab.events import EventLog, parse_event
from swarmlab.interventions import (
    DelayDelivery,
    InjectPost,
    Intervention,
    KillAgents,
    Mute,
    NotSupported,
    PatchPrivate,
    Reconfigure,
    build_intervention,
)
from swarmlab.medium.board import DelayPolicy
from swarmlab.participants import EvidenceAggregator
from swarmlab.snapshot import SnapshotStore
from swarmlab.spec import load_experiment_yaml, spec_hash
from swarmlab.world.flaggame import CROP_CHANGED, FlagGame, parse_observation

from .helpers import (
    PAUSE_AT_ENV,
    PAUSE_ENV,
    Chatter,
    PausingFlagGame,
    llm_agent_experiment,
    logical,
)

REPO = Path(__file__).resolve().parent.parent


class Reader(Participant):
    """Reads the whole board every round and records the observation text, then ends its turn."""

    async def turn(self, view, tools):
        self.texts = getattr(self, "texts", {})
        self.texts[view.round] = view.observation.parts[0].text
        await tools.call("read_board", {"limit": 200})
        await tools.call("end_turn", {})
        return TurnUsage(calls=2)


class SetTruth(Intervention):
    """A custom intervention using the world op."""

    def __init__(self, candidate: str, **trigger) -> None:
        super().__init__(**trigger)
        self.candidate = candidate

    def apply(self, ctx):
        ctx.ops.world("set_truth", name=self.candidate)


class Bogus(Intervention):
    def apply(self, ctx):
        ctx.ops.world("no_such_op", x=1)


class Recorder(Intervention):
    """Logs what it saw through a harmless op so tests can inspect the context."""

    seen: ClassVar[list] = []

    def apply(self, ctx):
        Recorder.seen.append((ctx.round, ctx.rng.random(), dict(ctx.metrics)))
        ctx.ops.delay([], 0)


def experiment(interventions, n=4, participants=None, world=None, metrics=None, name="iv", medium=None):
    return Experiment(
        name=name, world=world if world is not None else FlagGame(),
        participants=participants if participants is not None else [Reader()] * n,
        medium=medium if medium is not None else Board(topology="broadcast"),
        metrics=metrics if metrics is not None else ["belief.consensus", "belief.accuracy"],
        interventions=interventions,
    )


def events_of(run, type_):
    return [e for e in run.events if e["type"] == type_]


def first_read_round(run, agent, delivery_id):
    for e in events_of(run, "read"):
        if e["agent"] == agent and delivery_id in e["delivery_ids"]:
            return e["round"]
    return None


# ---- acceptance 1: inject + delay, determinism, replay -------------------------------------------

def test_acceptance_1_inject_post_reaches_everyone_next_round(tmp_path):
    run = experiment([InjectPost(text="breaking news", at_round=3)]).run(seed=1, max_rounds=6, out=tmp_path)
    ivs = events_of(run, "intervention")
    assert len(ivs) == 1 and ivs[0]["round"] == 3 and ivs[0]["op"] == "inject_post"
    assert ivs[0]["affected"] == [] and ivs[0]["ok"] and ivs[0]["agent"] is None
    post = next(e for e in events_of(run, "post") if e["post_id"] == ivs[0]["post_id"])
    assert post["agent"] == "system" and post["text"] == "breaking news" and post["round"] == 3
    dels = [e for e in events_of(run, "delivery") if e["post_id"] == post["post_id"]]
    assert sorted(d["recipient"] for d in dels) == ["a000", "a001", "a002", "a003"]
    assert all(d["eligible_round"] == 4 and d["round"] == 3 for d in dels)
    for d in dels:
        assert first_read_round(run, d["recipient"], d["delivery_id"]) == 4


def _delay_inject():
    return [DelayDelivery(agents=["a000"], rounds=2, at_round=3),
            InjectPost(text="breaking news", at_round=3)]


def test_acceptance_1_delay_holds_back_one_recipient(tmp_path):
    run = experiment(_delay_inject()).run(seed=1, max_rounds=7, out=tmp_path / "a")
    ivs = events_of(run, "intervention")
    assert [(e["op"], e["agent"], e["affected"]) for e in ivs] == [
        ("delay", "a000", ["a000"]), ("inject_post", None, [])]
    dels = {d["recipient"]: d for d in events_of(run, "delivery") if d["post_id"] == ivs[1]["post_id"]}
    assert dels["a000"]["eligible_round"] == 6
    assert {dels[a]["eligible_round"] for a in ("a001", "a002", "a003")} == {4}
    assert first_read_round(run, "a000", dels["a000"]["delivery_id"]) == 6
    assert first_read_round(run, "a001", dels["a001"]["delivery_id"]) == 4
    # deterministic across two runs
    again = experiment(_delay_inject()).run(seed=1, max_rounds=7, out=tmp_path / "b")
    assert logical(run) == logical(again)


def test_delay_in_immediate_mode_counts_like_delay_policy(tmp_path):
    run = experiment(_delay_inject()).run(seed=1, max_rounds=7, commit="immediate", out=tmp_path)
    post_id = events_of(run, "intervention")[1]["post_id"]
    dels = {d["recipient"]: d for d in events_of(run, "delivery") if d["post_id"] == post_id}
    assert dels["a000"]["eligible_round"] == 6 and dels["a001"]["eligible_round"] == 3
    assert first_read_round(run, "a000", dels["a000"]["delivery_id"]) == 6
    assert first_read_round(run, "a001", dels["a001"]["delivery_id"]) == 4


def test_acceptance_1_replay_reproduces_without_calling_plugins(tmp_path, monkeypatch):
    run = experiment(_delay_inject()).run(seed=1, max_rounds=7, out=tmp_path)

    def boom(*a, **k):
        raise AssertionError("replay called an intervention plugin")

    monkeypatch.setattr(Intervention, "should_fire", boom)
    monkeypatch.setattr(InjectPost, "apply", boom)
    monkeypatch.setattr(DelayDelivery, "apply", boom)
    loaded = Run.load(run.dir)
    assert loaded.metrics == run.metrics and loaded.score == run.score
    assert events_of(loaded, "intervention") == events_of(run, "intervention")


# ---- acceptance 2: kill / revive ------------------------------------------------------------------

def test_acceptance_2_kill_and_revive(tmp_path):
    ivs = [KillAgents(agents=["a001", "a002"], at_round=5),
           KillAgents(agents=["a001", "a002"], mode="revive", at_round=8, name="revive")]
    run = experiment(ivs, participants=[EvidenceAggregator()] * 6).run(seed=3, max_rounds=10, out=tmp_path)
    kills = events_of(run, "intervention")
    assert [(e["round"], e["op"], e["agent"]) for e in kills] == [
        (5, "kill", "a001"), (5, "kill", "a002"), (8, "revive", "a001"), (8, "revive", "a002")]
    for e in events_of(run, "round_started"):
        dead = 6 <= e["round"] <= 8
        assert ("a001" in e["order"]) is not dead and ("a002" in e["order"]) is not dead
        assert len(e["order"]) == (4 if dead else 6)
    turns = {(e["round"], e["agent"]) for e in events_of(run, "turn_started")}
    assert (5, "a001") in turns and (6, "a001") not in turns and (9, "a001") in turns
    for e in events_of(run, "metric"):
        assert e["denominator"] == (4 if 6 <= e["round"] <= 8 else 6), e
    # no deliveries to dead agents for posts committed while they are dead
    assert not [d for d in events_of(run, "delivery")
                if d["recipient"] in ("a001", "a002") and 6 <= d["round"] <= 8]
    assert Run.load(run.dir).metrics == run.metrics  # replay derives the live set from the log


def test_killed_agents_are_not_probed(tmp_path):
    from swarmlab.probes import BeliefProbe

    exp = llm_agent_experiment(4, probes=[BeliefProbe()], name="kp",
                               interventions=[KillAgents(agents=["a003"], at_round=2)])
    run = exp.run(seed=1, max_rounds=3, out=tmp_path)
    probed = {(e["round"], e["agent"]) for e in events_of(run, "probe")}
    assert (1, "a003") in probed and (2, "a003") not in probed and (3, "a003") not in probed
    assert (2, "a000") in probed


def test_kill_random_agents_uses_the_intervention_rng(tmp_path):
    a = experiment([KillAgents(agents=2, at_round=2)], n=6).run(seed=5, max_rounds=3, out=tmp_path / "a")
    b = experiment([KillAgents(agents=2, at_round=2)], n=6).run(seed=5, max_rounds=3, out=tmp_path / "b")
    killed = [e["agent"] for e in events_of(a, "intervention")]
    assert len(killed) == 2 and killed == [e["agent"] for e in events_of(b, "intervention")]


# ---- acceptance 3: patch_private on FlagGame ------------------------------------------------------

def test_acceptance_3_patch_private_changes_the_crop(tmp_path):
    exp = experiment([], world=FlagGame())  # find a crop different from the one reset draws
    probe_run = exp.run(seed=2, max_rounds=1, out=tmp_path / "probe")
    old = next(e["private"] for e in events_of(probe_run, "turn_started") if e["agent"] == "a000")
    new = [0, 0] if [old["crop_y"], old["crop_x"]] != [0, 0] else [1, 1]

    run = experiment([PatchPrivate(agent="a000", data={"crop": new}, at_round=4)]).run(
        seed=2, max_rounds=6, out=tmp_path / "run")
    (ev,) = events_of(run, "intervention")
    assert (ev["round"], ev["op"], ev["agent"], ev["ok"]) == (4, "patch_private", "a000", True)
    priv = {(e["round"], e["agent"]): e["private"] for e in events_of(run, "turn_started")}
    assert priv[(4, "a000")] == old and priv[(5, "a000")] == {"crop_y": new[0], "crop_x": new[1]}
    assert priv[(5, "a001")] == priv[(4, "a001")]
    snaps = SnapshotStore(run.dir)

    def verify_at(r):
        w = FlagGame()
        w.restore(snaps.load(snaps.read(r))["world"])
        return w.verify()

    assert verify_at(3)["crops"]["a000"] == [old["crop_y"], old["crop_x"]]
    assert verify_at(4)["crops"]["a000"] == new
    # the next observation says so (once), and still parses
    w = FlagGame()
    w.restore(snaps.load(snaps.read(4))["world"])
    text = w.observe("a000").parts[0].text
    assert text.endswith("\n\n" + CROP_CHANGED)
    _, crop = parse_observation(text)
    assert crop == w.crop_rows("a000")
    assert CROP_CHANGED not in w.observe("a000").parts[0].text
    assert CROP_CHANGED not in w.observe("a001").parts[0].text


def test_patch_private_observation_reaches_the_participant(tmp_path):
    run = experiment([PatchPrivate(agent="a001", data={"crop": [0, 0]}, at_round=2)]).run(
        seed=4, max_rounds=3, out=tmp_path)
    blob = SnapshotStore(run.dir).load(SnapshotStore(run.dir).read(3))["participant:a001"]
    r = Reader()
    r.restore(blob)
    assert CROP_CHANGED in r.texts[3] and CROP_CHANGED not in r.texts[2]


def test_world_op_set_truth_updates_metrics_and_replays(tmp_path):
    w = FlagGame()
    exp = experiment([], participants=[EvidenceAggregator()] * 4, world=w)
    base = exp.run(seed=1, max_rounds=1, out=tmp_path / "base")
    truth = base.score["truth"]
    other = next(c for c in "ABCDEFGH" if c != truth)
    run = experiment([SetTruth(candidate=other, at_round=2)], participants=[EvidenceAggregator()] * 4).run(
        seed=1, max_rounds=3, out=tmp_path / "run")
    (ev,) = events_of(run, "intervention")
    assert ev["op"] == "world" and ev["result"]["truth"] == other and ev["params"]["name"] == "set_truth"
    assert run.score["truth"] == other
    acc = {e["round"]: e["value"] for e in events_of(run, "metric") if e["name"] == "belief.accuracy"}
    assert acc[3] == run.score["accuracy"]
    Run.load(run.dir)  # replay re-applies the world op to its private world copy


# ---- acceptance 4: metric trigger -----------------------------------------------------------------

def test_acceptance_4_metric_trigger_fires_once(tmp_path):
    when = {"metric": "belief.consensus", "op": ">=", "value": 0.9}
    exp = llm_agent_experiment(6, name="when", interventions=[InjectPost(text="consensus!", when=when)])
    run = exp.run(seed=3, max_rounds=6, out=tmp_path)
    cons = {e["round"]: e["value"] for e in events_of(run, "metric") if e["name"] == "belief.consensus"}
    first = min(r for r, v in cons.items() if v is not None and v >= 0.9)
    assert sum(1 for r, v in cons.items() if v is not None and v >= 0.9) > 1  # holds again later
    ivs = events_of(run, "intervention")
    assert [e["round"] for e in ivs] == [first]
    assert Run.load(run.dir).metrics == run.metrics


def test_when_once_false_fires_every_round_it_holds(tmp_path):
    when = {"metric": "belief.consensus", "op": ">=", "value": 0.0, "once": False}
    run = experiment([Mute(agents=["a000"], rounds=1, when=when)], participants=[EvidenceAggregator()] * 3).run(
        seed=1, max_rounds=3, out=tmp_path)
    assert [e["round"] for e in events_of(run, "intervention")] == [1, 2, 3]


def test_on_event_trigger_and_context(tmp_path):
    Recorder.seen = []
    on_event = {"type": "action_committed", "where": {"action.name": "guess", "agent": "a001"}}
    ivs = [Recorder(on_event=on_event)]
    experiment(ivs, participants=[EvidenceAggregator()] * 3).run(seed=1, max_rounds=3, out=tmp_path / "a")
    assert [s[0] for s in Recorder.seen] == [1]
    _, _, metrics = Recorder.seen[0]
    assert set(metrics) == {"belief.consensus", "belief.accuracy"} and metrics["belief.consensus"][1] == 3
    assert ivs[0].fired is False  # the experiment's prototype is never mutated
    seen = list(Recorder.seen)
    experiment(ivs, participants=[EvidenceAggregator()] * 3).run(seed=1, max_rounds=3, out=tmp_path / "b")
    assert Recorder.seen[1][:2] == seen[0][:2]  # same rng draw per (name, round)


def test_every_trigger(tmp_path):
    run = experiment([Mute(agents=["a000"], rounds=1, every=2, start=1)]).run(seed=1, max_rounds=5, out=tmp_path)
    assert [e["round"] for e in events_of(run, "intervention")] == [1, 3, 5]


# ---- NotSupported and errors -----------------------------------------------------------------------

def test_not_supported_is_logged_as_not_ok(tmp_path):
    ivs = [Reconfigure(agents=["a000"], settings={"model": "x"}, at_round=1), Bogus(at_round=2),
           PatchPrivate(agent="a000", data={"crop": [99, 99]}, at_round=2, name="badcrop")]
    run = experiment(ivs, participants=[EvidenceAggregator()] * 2).run(seed=1, max_rounds=3, out=tmp_path)
    evs = events_of(run, "intervention")
    assert [(e["op"], e["ok"]) for e in evs] == [("reconfigure", False), ("world", False),
                                                 ("patch_private", False)]
    assert evs[0]["error"].startswith("not supported") and evs[1]["error"].startswith("not supported")
    assert evs[2]["error"].startswith("ValueError")
    assert run.end_reason == "max_rounds"
    with pytest.raises(NotSupported):
        Participant().reconfigure(model="x")


def test_llm_agent_reconfigure(tmp_path):
    exp = llm_agent_experiment(2, name="rc", interventions=[
        Reconfigure(agents=["a000"], settings={"max_tokens": 77, "memory": "window"}, at_round=1)])
    run = exp.run(seed=1, max_rounds=2, out=tmp_path)
    assert [e["ok"] for e in events_of(run, "intervention")] == [True]
    store = SnapshotStore(run.dir)
    from swarmlab.participants import LLMAgent

    p = LLMAgent(model="fake:reader")
    p.restore(store.load(store.read(2))["participant:a000"])
    assert p.max_tokens == 77 and p.memory == "window"
    with pytest.raises(ValueError, match="unknown settings"):
        p.reconfigure(temperature=1.0)


# ---- board overlays, snapshot, resume --------------------------------------------------------------

def test_board_overlays_snapshot_round_trip():
    b = Board(topology="broadcast")
    b.set_delay(["a000"], 2, until=5)
    b.mute(["a001"], 4)
    b.set_policies([DelayPolicy(rounds=1)])
    b.set_topology("gossip")
    fresh = Board(topology="broadcast")
    fresh.restore(b.snapshot())
    assert fresh.overlays(4) == {"delays": {"a000": 2}, "mutes": ["a001"]}
    assert fresh.overlays(6) == {"delays": {}, "mutes": []}
    assert [p.spec() for p in fresh.policies] == [DelayPolicy(rounds=1).spec()]
    assert fresh.topology.spec()["type"] == "gossip"
    b.set_delay(["a000"], 0)
    assert b.overlays(4)["delays"] == {}


def test_mute_withholds_but_logs_posts(tmp_path):
    run = experiment([Mute(agents=["a000"], rounds=2, at_round=1)], participants=[Chatter()] * 3).run(
        seed=1, max_rounds=4, out=tmp_path)
    posts = {e["post_id"]: e for e in events_of(run, "post") if e["agent"] == "a000"}
    delivered = {d["post_id"] for d in events_of(run, "delivery")}
    by_round = {p["round"]: pid in delivered for pid, p in posts.items()}
    assert by_round == {1: True, 2: False, 3: False, 4: True}


def test_fork_keeps_fired_once_state(tmp_path):
    when = {"metric": "belief.consensus", "op": ">=", "value": 0.0}
    exp = experiment([Mute(agents=["a000"], rounds=1, when=when)], participants=[EvidenceAggregator()] * 3)
    run = exp.run(seed=1, max_rounds=3, out=tmp_path)
    child = run.fork(2).run()
    assert [e["round"] for e in events_of(child, "intervention")] == [1]


def mute_experiment():
    return experiment([Mute(agents=["a000"], rounds=4, at_round=2)], participants=[Chatter()] * 4,
                      world=PausingFlagGame(), name="mute")


CHILD = """
import sys
from tests.test_interventions import mute_experiment
mute_experiment().run(seed=5, max_rounds=8, out=sys.argv[1])
"""


def test_resume_after_sigkill_with_active_mute(tmp_path):
    from .test_recovery import _wait_for

    out, marker = tmp_path / "crashed", tmp_path / "paused"
    env = {**os.environ, PAUSE_ENV: str(marker), PAUSE_AT_ENV: "4", "PYTHONPATH": str(REPO)}
    stderr = tmp_path / "child.stderr"
    with open(stderr, "wb") as err:
        proc = subprocess.Popen([sys.executable, "-c", CHILD, str(out)], cwd=REPO, env=env,
                                stdout=subprocess.DEVNULL, stderr=err)
    run_dir = out / "mute__s5"
    try:
        _wait_for(marker, proc, stderr)
    finally:
        os.kill(proc.pid, signal.SIGKILL)
        proc.wait()
    assert EventLog(run_dir / "events.jsonl").last_committed()[0] == 3
    snaps = SnapshotStore(run_dir)
    b = Board()
    b.restore(snaps.load(snaps.read(3))["board"])
    assert b.overlays(4)["mutes"] == ["a000"]  # the active mute is in the snapshot
    resumed = Run(run_dir).resume()
    assert resumed.end_reason == "max_rounds"
    reference = mute_experiment().run(seed=5, max_rounds=8, out=tmp_path / "clean")
    assert logical(resumed) == logical(reference)
    posts = {e["post_id"]: e["round"] for e in events_of(resumed, "post") if e["agent"] == "a000"}
    delivered = {d["post_id"] for d in events_of(resumed, "delivery")}
    assert {r: pid in delivered for pid, r in posts.items()} == {
        1: True, 2: True, 3: False, 4: False, 5: False, 6: False, 7: True, 8: True}
    Run.load(run_dir)


# ---- spec and YAML ---------------------------------------------------------------------------------

def test_builtin_spec_round_trip():
    for iv in [InjectPost(text="x", at_round=[2, 3]), DelayDelivery(agents=["a000"], rounds=2, every=3),
               Mute(agents=2, rounds=1, when={"metric": "m", "value": 1}),
               KillAgents(agents=["a1"], mode="revive", on_event={"type": "post"}, name="rv"),
               PatchPrivate(agent="a000", data={"crop": [1, 2]}, at_round=1),
               Reconfigure(agents="a000", settings={"model": "fake:x"}, at_round=1),
               SetTruth(candidate="A", at_round=1)]:
        spec = iv.spec()
        assert "trigger" not in spec["params"]
        again = build_intervention(spec)
        assert again.spec() == spec and again.name == iv.name
    assert InjectPost(text="x", at_round=1).spec() == {
        "type": "inject_post",
        "params": {"text": "x", "author": "system", "channel": "main", "fields": None,
                   "recipients": None, "at_round": 1}}


def test_trigger_validation():
    with pytest.raises(ValueError, match="exactly one trigger"):
        Mute(agents=["a"], rounds=1, at_round=1, every=2)
    with pytest.raises(ValueError, match="no trigger"):
        Mute(agents=["a"], rounds=1)
    with pytest.raises(ValueError, match="when.op"):
        Mute(agents=["a"], rounds=1, when={"metric": "m", "op": "~", "value": 1})
    with pytest.raises(ValueError, match="unique"):
        experiment([Mute(agents=["a"], rounds=1, at_round=1), Mute(agents=["b"], rounds=1, at_round=2)])


YAML = """
name: ivyaml
interventions:
  - {type: inject_post, params: {text: hello, at_round: 2}}
arms:
  plain:
    world: flaggame
    participants: [{type: evidence_aggregator, count: 3}]
  muted:
    world: flaggame
    participants: [{type: evidence_aggregator, count: 3}]
    interventions:
      - {type: mute, params: {agents: [a000], rounds: 2, at_round: 1}}
      - {type: "tests.test_interventions:SetTruth", params: {candidate: A, at_round: 3}}
"""


def test_yaml_round_trip(tmp_path):
    path = tmp_path / "x.yaml"
    path.write_text(YAML)
    doc = load_experiment_yaml(path)
    assert doc["interventions"] == [{"type": "inject_post", "params": {"text": "hello", "at_round": 2}}]
    exp = Experiment.from_yaml(path, "muted")
    assert [type(i) for i in exp.interventions] == [InjectPost, Mute, SetTruth]
    spec = exp.to_spec(seed=1, max_rounds=3)
    assert [i.type for i in spec.interventions] == [
        "inject_post", "mute", "tests.test_interventions:SetTruth"]
    assert Experiment.from_spec(spec).to_spec(seed=1, max_rounds=3) == spec
    out = tmp_path / "back.yaml"
    exp.to_yaml(out)
    assert Experiment.from_yaml(out, "muted").to_spec(seed=1, max_rounds=3) == spec
    run = exp.run(seed=1, max_rounds=3, out=tmp_path / "runs")
    assert [e["op"] for e in events_of(run, "intervention")] == ["mute", "inject_post", "world"]
    assert Run.load(run.dir).score == run.score
    resumed_spec = yaml.safe_load((run.dir / "artifacts" / "spec.yaml").read_text())
    assert len(resumed_spec["interventions"]) == 3


def test_no_interventions_keeps_spec_hash_and_documents(tmp_path):
    path = tmp_path / "x.yaml"
    path.write_text(YAML)
    doc = load_experiment_yaml(path)
    assert "interventions" not in doc["arms"]["plain"]
    exp = experiment([], participants=[EvidenceAggregator()] * 2)
    spec = exp.to_spec(seed=1, max_rounds=2)
    import hashlib

    from swarmlab.spec import canonical_json
    old = spec.model_dump(mode="json")
    old.pop("interventions")
    old.pop("probes")
    assert spec_hash(spec) == hashlib.sha256(canonical_json(old).encode()).hexdigest()


def test_event_parses():
    ev = parse_event({"run": "r", "round": 1, "type": "intervention", "intervention": "m", "op": "mute",
                      "affected": ["a000"], "agent": "a000"})
    assert ev.ok and ev.params == {}
