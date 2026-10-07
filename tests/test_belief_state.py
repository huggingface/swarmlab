"""M6 §5: belief.state classification, stop_when (docs/INTERFACE-M6.md)."""
from collections import Counter
from types import SimpleNamespace

import pytest

from swarmlab import Board, Experiment, Run
from swarmlab.metrics.belief import STATES, Consensus, State, classify_state
from swarmlab.participants import LLMAgent
from swarmlab.world.flaggame import FlagGame

from .helpers import logical


@pytest.mark.parametrize("counts,truth,label", [
    ({"T": 17, "R": 3}, "T", "correct_consensus"),        # s1 = 0.85 exactly
    ({"T": 16, "R": 4}, "T", "fragmented"),               # 0.80 top; the 0.20 rival is no camp
    ({"R": 9, "T": 1}, "T", "wrong_consensus"),           # 0.90 on a rival
    ({"T": 5, "R": 5}, "T", "polarized"),                 # two camps of 0.5
    ({"T": 4, "R": 3, "X": 3}, "T", "polarized"),         # 0.4 / 0.3 / 0.3
    ({"T": 4, "R": 2, "X": 2, "Y": 2}, "T", "fragmented"),  # 0.4 / 0.2 x3: one camp only
    ({"A": 1, "B": 1, "C": 1, "D": 1, "E": 1}, "T", "fragmented"),
    ({"T": 1}, "T", "correct_consensus"),
    ({}, "T", None),
])
def test_classify_hand_computed(counts, truth, label):
    assert classify_state(Counter(counts), truth) == label


def test_thresholds_are_parameters():
    assert classify_state({"T": 8, "R": 2}, "T", consensus=0.75) == "correct_consensus"
    assert classify_state({"T": 8, "R": 2}, "T", camp=0.2) == "polarized"
    with pytest.raises(ValueError):
        State(consensus=0.2, camp=0.3)


def committed(agent, cand, fb=None):
    return SimpleNamespace(type="action_committed", accepted=True, agent=agent,
                           action={"name": "guess", "args": {"candidate": cand}}, feedback=fb or {})


def test_metric_outputs_label_and_one_hot():
    m = State()
    m.set_truth({"truth": "Germany"})
    m.set_agents(["a000", "a001", "a002", "a003", "a004"])
    out = m.outputs()
    assert out[0] == ("belief.state", None, 0, None)  # nobody holds a guess
    assert all(v is None for _, v, _, _ in out[1:])
    for a in ("a000", "a001", "a002"):
        m.update(committed(a, "germany", {"candidate": "Germany"}))  # canonical name from feedback
    m.update(committed("a003", "France"))
    # a004 has no guess: out of the denominator; 3/4 = 0.75 Germany, 0.25 France -> polarized
    out = m.outputs()
    assert out[0] == ("belief.state", float(STATES.index("polarized")), 4, "polarized")
    assert [n for n, *_ in out[1:]] == [f"belief.state.{s}" for s in STATES]
    assert [v for _, v, _, _ in out[1:]] == [0.0, 0.0, 1.0, 0.0]
    m.update(committed("a003", "Germany"))
    assert m.outputs()[0][3] == "correct_consensus"
    probe = State(source="probe:belief")
    assert probe.output_names()[1] == "belief.state.correct_consensus@probe:belief"
    assert "consensus" not in State().spec()["params"]


def paper_like(n=4, stop=None, every=1, rounds=30):
    agent = LLMAgent("fake:country_reporter", memory="received", memory_messages=8, report_json=True)
    options = {"scheduler": "one_speaker", "max_rounds": rounds, "commit": "immediate"}
    if stop:
        options["stop_when"] = stop
    return Experiment(
        name="stop", world=FlagGame(flags="real", candidates="names", canvas=[24, 16], crop=[6, 4]),
        participants=[agent] * n,
        medium=Board(topology={"type": "gossip", "params": {"k": 1}}, delivery="push", push_limit=8,
                     push_consume=True),
        metrics=["belief.state", Consensus(source="probe:belief"), State(source="probe:belief")],
        probes=[{"type": "belief", "params": {"every": every}}], options=options)


STOP = {"metric": "belief.consensus@probe:belief", "op": ">=", "value": 1.0, "consecutive": 5}


def test_stop_when_ends_after_five_consecutive_hits(tmp_path):
    run = paper_like(stop=STOP).run(seed=4, out=tmp_path)
    assert run.end_reason == "stop_condition"
    cons = {r: v for r, v, _ in run.metrics["belief.consensus@probe:belief"]}
    last = max(cons)
    assert last < 30
    assert [cons[r] for r in range(last - 4, last + 1)] == [1.0] * 5
    assert not all(cons[r] == 1.0 for r in range(last - 5, last + 1) if r >= 1)  # first 5-streak
    labels = [e["label"] for e in run.events if e["type"] == "metric" and e["name"] == "belief.state"]
    assert labels and set(labels) <= set(STATES) | {None}
    Run.load(run.dir)  # replays every output of belief.state


def test_stop_when_only_counts_probe_rounds(tmp_path):
    run = paper_like(stop=STOP, every=2, rounds=60).run(seed=4, out=tmp_path)
    assert run.end_reason == "stop_condition"
    last = max(r for r, _, _ in run.metrics["belief.consensus@probe:belief"])
    assert last % 2 == 0


def test_resume_keeps_the_streak(tmp_path):
    from .test_recovery import _truncate_mid_round

    ref = paper_like(stop=STOP).run(seed=4, out=tmp_path / "ref")
    last = max(r for r, _, _ in ref.metrics["belief.consensus@probe:belief"])
    crash = paper_like(stop=STOP).run(seed=4, out=tmp_path / "crash")
    _truncate_mid_round(crash.dir, last - 1)
    resumed = Run(crash.dir).resume()
    assert resumed.end_reason == "stop_condition"
    same = lambda r: [e for e in logical(r) if e["type"] != "budget"]
    assert same(resumed) == same(ref)


def test_no_stop_without_condition(tmp_path):
    run = paper_like(rounds=12).run(seed=4, out=tmp_path)
    assert run.end_reason == "max_rounds"
