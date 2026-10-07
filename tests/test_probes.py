"""Probes (docs/INTERFACE-M1b.md §5, §6; acceptance 4 and 5)."""
from collections import Counter
from types import SimpleNamespace

import pytest

from swarmlab import Budget, Experiment, Run
from swarmlab.budget import HardCeilingReached
from swarmlab.executor import RoundExecutor
from swarmlab.metrics.belief import Accuracy, Consensus, Entropy
from swarmlab.participants import EvidenceAggregator, LLMAgent
from swarmlab.probes import BeliefProbe, Probe, build_probe, parse_json_object, probe_messages
from swarmlab.providers.base import ChatMessage
from swarmlab.runner import Runner
from swarmlab.spec import load_experiment_yaml, spec_hash
from swarmlab.tools import ToolCall

from .helpers import TEST_PRICING, llm_agent_experiment, logical

PROBE_METRICS = ["belief.consensus", "belief.accuracy",
                 Consensus(source="probe:belief"), Accuracy(source="probe:belief")]
EXAMPLE = "examples/flaggame_m1a.yaml"


def probe_events(run):
    return [e for e in run.events if e["type"] == "probe"]


def test_acceptance_4_one_probe_per_agent_per_round(tmp_path, monkeypatch):
    original = Runner._probe_one
    checked = []

    async def guarded(self, probe, agent, r, ex):
        before = self.participants[agent].snapshot()
        out = await original(self, probe, agent, r, ex)
        assert self.participants[agent].snapshot() == before  # memory byte-identical
        checked.append((r, agent))
        return out

    monkeypatch.setattr(Runner, "_probe_one", guarded)
    n, rounds = 6, 3
    exp = llm_agent_experiment(n, probes=[BeliefProbe()], metrics=PROBE_METRICS)
    run = exp.run(seed=4, max_rounds=rounds, out=tmp_path)
    assert len(checked) == n * rounds
    evs = probe_events(run)
    assert Counter((e["round"], e["agent"]) for e in evs) == {
        (r, f"a{i:03d}"): 1 for r in range(1, rounds + 1) for i in range(n)}
    assert all(e["ok"] and isinstance(e["parsed"]["candidate"], str) and e["cost_usd"] > 0 for e in evs)
    # charged to measurement, not to the swarm or to turn usage
    attempts = [e for e in run.events_all if e.type == "inference_attempt"]
    assert Counter(a.category for a in attempts)["measurement"] == n * rounds
    assert run.spend["measurement"] > 0
    assert sum(e["cost_usd"] for e in evs) == pytest.approx(run.spend["measurement"])
    turn_calls = sum(e["usage"]["inference_calls"] for e in run.events if e["type"] == "turn_ended")
    assert turn_calls == Counter(a.category for a in attempts)["swarm"]
    # probe-sourced metrics side by side with world metrics
    assert {"belief.consensus", "belief.consensus@probe:belief", "belief.accuracy@probe:belief"} <= set(run.metrics)
    assert all(len(v) == rounds for v in run.metrics.values())
    # probes sit after the commit and before the round's metrics
    kinds = [e["type"] for e in run.events if e["round"] == 2]
    assert kinds.index("probe") > max(i for i, k in enumerate(kinds) if k == "action_committed")
    assert kinds.index("probe") < kinds.index("metric")
    # Run.probes
    by_round = Counter(r for r, _, _, ok in run.probes["belief"] if ok)
    assert by_round == {r: n for r in range(1, rounds + 1)}
    # the question and the raw answer are blobs
    from swarmlab.blobs import BlobStore
    blobs = BlobStore(run.dir / "blobs")
    assert "Which candidate" in blobs.get_text(evs[0]["question_hash"])
    assert '"candidate"' in blobs.get_text(evs[0]["raw_hash"])
    # replay reproduces the probe metrics without any provider call
    assert Run.load(run.dir).metrics == run.metrics


def test_acceptance_5_measurement_cap_stops_probes_not_the_swarm(tmp_path):
    exp = llm_agent_experiment(4, probes=[BeliefProbe()], metrics=PROBE_METRICS,
                               budget=Budget(measurement_usd=0.05))
    run = exp.run(seed=1, max_rounds=5, out=tmp_path)
    assert run.end_reason == "max_rounds" and run.meta["last_round"] == 5
    evs = probe_events(run)
    skipped = [e for e in evs if e["parsed"].get("skipped") == "measurement_budget"]
    assert skipped and all(not e["ok"] for e in skipped)
    stop = skipped[0]["round"]
    assert stop < 5
    assert not [e for e in evs if e["round"] > stop]  # probing stopped for the run
    assert run.spend["measurement"] <= 0.05
    # the swarm kept going after the cap
    assert {t["round"] for t in run.events if t["type"] == "turn_ended"} == set(range(1, 6))
    # skipped agents leave the probe metrics' denominator (they were not asked)
    series = run.metrics["belief.consensus@probe:belief"]
    assert all(den == 4 for r, _, den in series if r < stop)
    n_skipped_at_stop = len([e for e in skipped if e["round"] == stop])
    assert [den for r, _, den in series if r == stop] == [4 - n_skipped_at_stop]
    assert series[-1][2] == 4 - n_skipped_at_stop


def test_scripted_agents_skipped_once_per_run(tmp_path):
    from swarmlab import Board
    from swarmlab.providers.fake import FakeProvider
    from swarmlab.world.flaggame import FlagGame

    exp = Experiment(name="mixed", world=FlagGame(),
                     participants=[LLMAgent(model="fake:reader")] * 2 + [EvidenceAggregator()] * 2,
                     medium=Board(topology="broadcast"), metrics=PROBE_METRICS,
                     probes=[BeliefProbe()], providers={"fake": FakeProvider(pricing=TEST_PRICING)})
    run = exp.run(seed=1, max_rounds=3, out=tmp_path)
    evs = probe_events(run)
    skipped = [e for e in evs if e["parsed"] == {"skipped": "no_context"}]
    assert sorted(e["agent"] for e in skipped) == ["a002", "a003"]
    assert all(e["round"] == 1 and not e["ok"] for e in skipped)
    assert len([e for e in evs if e["ok"]]) == 2 * 3
    child = run.fork(at_round=1).run(out=tmp_path / "forks")
    child_skips = [e for e in probe_events(child) if e["parsed"].get("skipped")]
    assert len(child_skips) == 2  # the copied prefix's, none new
    # scripted agents (never asked) are left out of the probe metrics' denominator
    assert run.metrics["belief.accuracy@probe:belief"][-1][2] == 2


def test_probe_hard_ceiling_commits_the_round_then_ends(tmp_path, monkeypatch):
    original = RoundExecutor.infer

    async def ceiling(self, agent, request, category="swarm"):
        if category == "measurement" and self.round == 2 and not getattr(self, "_resumed", False):
            raise HardCeilingReached("test ceiling")
        return await original(self, agent, request, category)

    monkeypatch.setattr(RoundExecutor, "infer", ceiling)
    exp = llm_agent_experiment(3, probes=[BeliefProbe()], metrics=PROBE_METRICS)
    run = exp.run(seed=1, max_rounds=4, out=tmp_path)
    # the round is kept and the end reason says the ceiling was reached by the probes
    assert run.end_reason == "hard_ceiling_probes" and run.meta["last_round"] == 2
    r2 = [e for e in run.events if e["round"] == 2]
    assert "round_committed" in [e["type"] for e in r2]
    assert {e["parsed"].get("skipped") for e in r2 if e["type"] == "probe"} == {"hard_ceiling"}
    assert run.summary()["probes_skipped"] == {"hard_ceiling": 3}
    # every agent skipped in round 2: no probe answers, no denominator (not three "none"s)
    assert run.metrics["belief.consensus@probe:belief"][1] == (2, None, 0)
    assert run.metrics["belief.consensus@probe:belief"][0][2] == 3
    monkeypatch.setattr(RoundExecutor, "infer", original)
    resumed = Run(run.dir, experiment=exp).resume(budget=Budget(hard_usd=10.0))
    assert resumed.end_reason == "max_rounds" and resumed.meta["last_round"] == 4
    assert {e["round"] for e in probe_events(resumed) if e["ok"]} == {1, 3, 4}


def test_probe_skips_in_report_and_status_line(tmp_path, monkeypatch):
    from typer.testing import CliRunner

    from swarmlab.cli import app
    from swarmlab.report import build_report

    original = RoundExecutor.infer

    async def ceiling(self, agent, request, category="swarm"):
        if category == "measurement" and self.round == 2:
            raise HardCeilingReached("test ceiling")
        return await original(self, agent, request, category)

    monkeypatch.setattr(RoundExecutor, "infer", ceiling)
    exp = llm_agent_experiment(3, probes=[BeliefProbe()], metrics=PROBE_METRICS)
    run = exp.run(seed=1, max_rounds=4, out=tmp_path)
    text = build_report(tmp_path, include_fake=True)
    assert "Probes skipped: 3 over 1 run(s) (hard_ceiling 3)" in text
    # round 2's probe columns have no answers (skipped agents are not counted as "no answer")
    probe_rows = [line for line in text.splitlines() if line.startswith("| probe consensus |")]
    assert probe_rows and probe_rows[0].rstrip(" |").split(" | ")[2] == "n/a"
    res = CliRunner().invoke(app, ["replay", str(run.dir)])
    assert res.exit_code == 0, res.output
    assert "end=hard_ceiling_probes" in res.output
    assert "probes_skipped=3 (hard_ceiling 3)" in res.output


def test_coder_model_extracts_free_text(tmp_path):
    probe = BeliefProbe(coder_model="fake:tests.helpers:script_coder")
    exp = llm_agent_experiment(2, agent_kw={"model": "fake:tests.helpers:script_freetext_belief"},
                               probes=[probe], metrics=PROBE_METRICS)
    run = exp.run(seed=1, max_rounds=2, out=tmp_path)
    evs = probe_events(run)
    assert len(evs) == 4 and all(e["ok"] and e["parsed"]["coded"] for e in evs)
    measurement = [e for e in run.events_all if e.type == "inference_attempt" and e.category == "measurement"]
    assert len(measurement) == 8 and sum(m.model.endswith("script_coder") for m in measurement) == 4
    # without a coder the same answers are parse failures, which count as "none"
    plain = llm_agent_experiment(2, agent_kw={"model": "fake:tests.helpers:script_freetext_belief"},
                                 probes=[BeliefProbe()], metrics=PROBE_METRICS)
    run = plain.run(seed=1, max_rounds=2, out=tmp_path / "plain")
    assert not any(e["ok"] for e in probe_events(run))
    assert run.metrics["belief.consensus@probe:belief"][-1] == (2, 0.0, 2)


def test_every_and_probe_names(tmp_path):
    exp = llm_agent_experiment(2, probes=[BeliefProbe(every=2), BeliefProbe(name="b2")],
                               metrics=["belief.consensus", Consensus(source="probe:b2")])
    run = exp.run(seed=1, max_rounds=4, out=tmp_path)
    assert {r for r, *_ in run.probes["belief"]} == {2, 4}
    assert {r for r, *_ in run.probes["b2"]} == {1, 2, 3, 4}
    with pytest.raises(ValueError, match="unique"):
        llm_agent_experiment(1, probes=[BeliefProbe(), BeliefProbe()])
    with pytest.raises(ValueError):
        BeliefProbe(every=0)


def test_unpriced_coder_model_fails_at_construction():
    from swarmlab.providers.base import UnknownModelPricing

    with pytest.raises(UnknownModelPricing):
        llm_agent_experiment(1, probes=[BeliefProbe(coder_model="hf:some/model")])


def test_belief_parse_and_messages():
    p = BeliefProbe()
    assert p.parse('{"candidate": "C", "confidence": 0.7}') == (True, {"candidate": "C", "confidence": 0.7})
    assert p.parse('<think>{"x": 1}</think>Answer: ```{"candidate": " D "}```') == (True, {"candidate": "D"})
    assert p.parse("C, I think")[0] is False
    assert p.parse('{"candidate": null}')[0] is False
    assert parse_json_object('pre {"a": "}"} post') == {"a": "}"}
    assert "JSON" in p.question("a000", 3)
    ctx = [ChatMessage(role="system", content="s"), ChatMessage(role="user", content="u"),
           ChatMessage(role="assistant", content="", tool_calls=[ToolCall(call_id="1", name="guess",
                                                                          args={"candidate": "A"})]),
           ChatMessage(role="tool", content='{"ok": true}', tool_call_id="1")]
    flat = probe_messages(ctx)
    assert [m.role for m in flat] == ["system", "user", "assistant", "user"]
    assert flat[2].content == '[tool call] guess {"candidate": "A"}' and flat[2].tool_calls is None
    assert flat[3].content.startswith("[tool result]") and flat[3].tool_call_id is None
    assert ctx[2].tool_calls  # the input is untouched


def test_probe_metric_source_rules():
    m = Consensus(source="probe:belief")
    e = Entropy(source="probe:belief")
    assert m.name == "belief.consensus@probe:belief" and e.name == "belief.entropy@probe:belief"
    m.set_agents(["a0", "a1"])

    def ev(agent, parsed, ok, probe="belief"):
        return SimpleNamespace(type="probe", probe=probe, agent=agent, parsed=parsed, ok=ok)

    m.update(ev("a0", {"candidate": "A"}, True))
    m.update(ev("a1", {"candidate": "A"}, True))
    assert m.value() == (1.0, 2)
    m.update(ev("a1", {"error": "no JSON object"}, False))  # failed parse -> none
    assert m.value() == (0.5, 2)
    m.update(ev("a1", {"candidate": "A"}, True))
    m.update(ev("a1", {"skipped": "measurement_budget"}, False))  # skipped: not in the denominator
    m.update(ev("a1", {"candidate": "B"}, True, probe="other"))  # another probe: ignored
    m.update(SimpleNamespace(type="action_committed", agent="a1", accepted=True,
                             action={"name": "guess", "args": {"candidate": "B"}}))
    assert m.value() == (1.0, 1)
    m.update(ev("a0", {"skipped": "hard_ceiling"}, False))
    assert m.value() == (None, 0)
    m.update(ev("a1", {"candidate": "B"}, True))  # answered again: back in
    assert m.value() == (1.0, 1)
    old = Consensus(source="probe:belief")  # metrics revision 1: a skip left the last answer
    old.use_rev(1)
    old.set_agents(["a0", "a1"])
    for x in (ev("a0", {"candidate": "A"}, True), ev("a1", {"candidate": "A"}, True),
              ev("a1", {"skipped": "measurement_budget"}, False)):
        old.update(x)
    assert old.value() == (1.0, 2)
    with pytest.raises(ValueError):
        Consensus(source="probe:")
    with pytest.raises(ValueError):
        Consensus(source="board")
    assert Consensus().spec() == {"type": "belief.consensus", "params": {}}


def test_yaml_probes_round_trip_and_spec_hash(tmp_path):
    doc = load_experiment_yaml(EXAMPLE)
    assert doc["arms"]["llm"]["probes"] == [{"type": "belief", "params": {}}]
    exp = Experiment.from_yaml(EXAMPLE, "llm")
    assert [type(p) for p in exp.probes] == [BeliefProbe]
    spec = exp.to_spec(seed=1, max_rounds=2)
    assert spec.probes[0].type == "belief"
    back = Experiment.from_spec(spec)
    assert back.to_spec(seed=1, max_rounds=2) == spec
    path = tmp_path / "x.yaml"
    back.to_yaml(path)
    assert Experiment.from_yaml(path, "llm").to_spec(seed=1, max_rounds=2).probes == spec.probes
    # a spec without probes hashes as before the field existed
    plain = Experiment.from_yaml(EXAMPLE, "broadcast").to_spec(seed=1, max_rounds=2)
    import hashlib

    from swarmlab.spec import canonical_json
    old = plain.model_dump(mode="json")
    del old["probes"]
    old.pop("interventions", None)  # M3a field, likewise left out when empty
    assert spec_hash(plain) == hashlib.sha256(canonical_json(old).encode()).hexdigest()


def test_build_probe_forms():
    assert isinstance(build_probe("belief"), BeliefProbe)
    assert build_probe({"type": "belief", "params": {"every": 3}}).every == 3
    p = BeliefProbe(name="x")
    assert build_probe(p) is not p and build_probe(p).name == "x"
    with pytest.raises(NotImplementedError):
        Probe().parse("x")


def test_determinism_with_probes(tmp_path):
    exp = llm_agent_experiment(4, probes=[BeliefProbe()], metrics=PROBE_METRICS)
    a = exp.run(seed=3, max_rounds=3, out=tmp_path / "a")
    b = llm_agent_experiment(4, probes=[BeliefProbe()], metrics=PROBE_METRICS).run(
        seed=3, max_rounds=3, out=tmp_path / "b")
    assert logical(a) == logical(b) and a.spend == b.spend
