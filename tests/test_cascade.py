"""The `cascade` world, the `cascade_worker` participant and experiments/cascade.yaml."""
from __future__ import annotations

import json
import random
from pathlib import Path

import pytest

from swarmlab import Experiment
from swarmlab.ids import AgentId
from swarmlab.participants.cascade import CascadeWorker, parse_reply
from swarmlab.registry import resolve
from swarmlab.rng import derive
from swarmlab.spec import load_experiment_yaml
from swarmlab.view import View
from swarmlab.world.cascade import (
    ACCEPTED,
    EMPTY_BOARD,
    INSTRUCTION,
    INSTRUCTION_DISCLOSED,
    OUTPUT,
    QUOTE_RULE,
    READS,
    REJECTED,
    CascadeWorld,
)

SPEC = Path(__file__).resolve().parents[1] / "experiments" / "cascade.yaml"
AGENTS = [AgentId(f"a{i:03d}") for i in range(20)]


def world(**params) -> CascadeWorld:
    w = CascadeWorld(**params)
    w.reset(derive(3, "world"), list(AGENTS))
    return w


def test_entry_points_resolve():
    assert resolve("cascade", "swarmlab.worlds") is CascadeWorld
    assert resolve("cascade_worker", "swarmlab.participants") is CascadeWorker


def test_signals_are_paired_across_arms_and_the_first_k_are_forced_wrong():
    normal, stress = world(), world(n_wrong_first=4)
    assert normal.drawn_correct == stress.drawn_correct
    wrong = REJECTED  # truth output_only: ACCEPTED points to the truth
    assert [stress.signal[a] for a in AGENTS[:4]] == [wrong] * 4
    assert all(stress.signal[a] == normal.signal[a] for a in AGENTS[4:])
    assert all((normal.signal[a] == ACCEPTED) == normal.drawn_correct[a] for a in AGENTS)
    flipped = world(truth=READS)
    assert all((flipped.signal[a] == REJECTED) == flipped.drawn_correct[a] for a in AGENTS)


def test_signal_accuracy_is_the_parameter():
    hits = total = 0
    for seed in range(200):
        w = CascadeWorld(accuracy=0.7)
        w.reset(derive(seed, "world"), list(AGENTS))
        hits += sum(w.drawn_correct.values())
        total += len(AGENTS)
    assert abs(hits / total - 0.7) < 0.02


@pytest.mark.parametrize("params", [{"truth": "maybe"}, {"accuracy": 1.5}, {"n_wrong_first": -1}])
def test_bad_params_are_refused(params):
    with pytest.raises(ValueError):
        CascadeWorld(**params)


def test_observation_carries_only_the_own_signal():
    w = world(n_wrong_first=4)
    text = w.observe(AGENTS[0]).parts[0].text
    assert text.startswith("TaskBench Platform Documentation")
    assert "Result: REJECTED with code E-INTEGRITY." in text and "ACCEPTED." not in text


def test_guess_records_once_with_feedback_and_scores():
    w = world(n_wrong_first=4)
    out = w.guess(AGENTS[0], READS)
    assert out.accepted and out.feedback == {"interpretation": READS, "signal": REJECTED,
                                             "follows_signal": True}
    assert not w.guess(AGENTS[0], OUTPUT).accepted
    assert not w.guess(AGENTS[1], "both").accepted
    assert w.score() == {"truth": OUTPUT, "n_committed": 1, "accuracy": 0.0, "final_worker": None}
    assert w.verify()["signals"]["W00"] == REJECTED


def view(agent: str, pushed: list[str]) -> View:
    w = world(n_wrong_first=4)
    return View(round=1, agent=AgentId(agent), observation=w.observe(AgentId(agent)), outcomes=[],
                pushed=[{"content": p} for p in pushed], tools=[])


def worker(agent: str = "a007", **params) -> CascadeWorker:
    p = CascadeWorker(model="fake:cascade_herder", **params)
    p.bind(AgentId(agent), random.Random(0))
    return p


def test_prompt_matches_the_source_layout():
    p = worker("a000")
    v = view("a000", [])
    assert p._render_system(v).startswith("You are Worker W00, an autonomous agent")
    text = p.round_message(v).content
    assert EMPTY_BOARD in text and text.index(EMPTY_BOARD) < text.index("QUESTION:")
    assert INSTRUCTION in text and "Round 1." not in text and "Delivered to you" not in text
    board = worker().round_message(view("a007", ["W00: one", "W01: two"])).content
    assert "W00: one\n\nW01: two\n\nQUESTION:" in board


def test_variants_change_only_their_paragraph():
    v = view("a007", ["W00: one"])
    base = worker().round_message(v).content
    assert EMPTY_BOARD in worker(show_board=False).round_message(v).content
    assert QUOTE_RULE in worker(board_rule="quote").round_message(v).content
    disclosed = worker(disclose_belief=True).round_message(v).content
    assert disclosed == base.replace(INSTRUCTION, INSTRUCTION_DISCLOSED)
    paid = worker(reward_last=5).round_message(v).content
    assert "last 5 workers this shift (W15–W19)" in paid
    with pytest.raises(ValueError):
        worker(reward_last=21)


@pytest.mark.parametrize("text,expected", [
    ('{"board_post": "hi", "interpretation": "Output Only"}', {"board_post": "hi", "interpretation": OUTPUT}),
    ('sure: {"board_post": "hi", "interpretation": "reads-transcript"}', {"board_post": "hi", "interpretation": READS}),
    ('{"board_post": "hi", "interpretation": "both"}', None),
    ('{"board_post": "", "interpretation": "output_only"}', None),
    ('{"board_post": "hi", "interpretation": "output_on', None),
])
def test_parse_reply(text, expected):
    assert parse_reply(text)[0] == expected


def test_spec_arms_build():
    doc = load_experiment_yaml(SPEC)
    assert doc["seeds"] == list(range(1, 41))
    for arm in doc["arms"]:
        Experiment.from_yaml(SPEC, arm)


def commits(run) -> list[dict]:
    events = [json.loads(line) for line in (Path(run.dir) / "events.jsonl").read_text().splitlines()]
    return [e for e in events if e["type"] == "action_committed" and e["accepted"]]


def posts(run) -> list[str]:
    events = [json.loads(line) for line in (Path(run.dir) / "events.jsonl").read_text().splitlines()]
    return [e["text"] for e in events if e["type"] == "post"]


def test_dry_board_run_cascades_and_no_board_follows_signals(tmp_path):
    board = Experiment.from_yaml(SPEC, "board_stress_dry").run(seed=3, out=tmp_path)
    acts = commits(board)
    assert len(acts) == 20 and [a["agent"] for a in acts] == [str(a) for a in AGENTS]
    assert {a["feedback"]["interpretation"] for a in acts} == {READS}  # the herders cascade
    texts = posts(board)
    assert texts[0].startswith("W00: My probe came back REJECTED.") and len(texts) == 20
    alone = Experiment.from_yaml(SPEC, "noboard_stress_dry").run(seed=3, out=tmp_path)
    assert all(a["feedback"]["follows_signal"] for a in commits(alone))
    assert board.score["accuracy"] == 0 and alone.score["accuracy"] > 0


def test_disclose_tags_the_post(tmp_path):
    doc = load_experiment_yaml(SPEC)
    arm = doc["arms"]["board_stress_dry"]
    arm["participants"][0]["params"]["disclose_belief"] = True
    spec = tmp_path / "spec.yaml"
    import yaml
    spec.write_text(yaml.safe_dump({**doc, "arms": {"d": arm}}))
    run = Experiment.from_yaml(spec, "d").run(seed=3, out=tmp_path / "runs")
    assert posts(run)[0].startswith("W00 [committed: reads_transcript]: ")
