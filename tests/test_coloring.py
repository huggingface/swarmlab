"""ColoringGrid, painters, coloring and claim metrics (M3b)."""
from __future__ import annotations

from pathlib import Path

import pytest

from swarmlab import Board, Experiment, Run
from swarmlab.ids import agent_id
from swarmlab.participants.base import Participant
from swarmlab.participants.scripted import QueuePainter, RandomPainter, RowMajorPainter
from swarmlab.rng import derive
from swarmlab.viewer.build import build
from swarmlab.world.base import Action
from swarmlab.world.coloring import ColoringGrid, needed_cells, parse_observation

from .helpers import logical

A, B, C = agent_id(0), agent_id(1), agent_id(2)
REPO = Path(__file__).resolve().parent.parent
GRID_METRICS = ["coloring.coverage", "coloring.duplicate_paints", "coloring.wrong_paints",
                "coloring.parallel_efficiency"]
CLAIM_METRICS = ["claims.violations", "claims.held"]


def world(**kw):
    w = ColoringGrid(**{"height": 3, "width": 4, **kw})
    w.reset(derive(0, "world"), [A, B, C])
    w.begin_round(1)
    return w


def paint(agent, x, y, color, n=0):
    return (agent, f"x-{agent}-{n}", Action(name="paint", args={"x": x, "y": y, "color": color}))


def test_observation_round_trip_and_reset_determinism():
    w = world()
    text = w.observe(A).parts[0].text
    target, current, colours = parse_observation(text)
    assert target == ["".join(r) for r in w.target] and len(target) == 3 and len(target[0]) == 4
    assert current == ["...."] * 3 and colours == ["r", "g", "b", "y"]
    assert w.observe(A).private == {}
    assert len(needed_cells(target, current)) == 12
    assert world().target == w.target
    other = ColoringGrid(height=3, width=4)
    other.reset(derive(1, "world"), [A])
    assert other.target != w.target


def test_validate_bounds_and_colour():
    w = world()
    assert w.validate(A, Action(name="paint", args={"x": 3, "y": 2, "color": "r"})).ok
    assert not w.validate(A, Action(name="paint", args={"x": 4, "y": 0, "color": "r"})).ok
    assert not w.validate(A, Action(name="paint", args={"x": 0, "y": -1, "color": "r"})).ok
    assert "unknown color" in w.validate(A, Action(name="paint", args={"x": 0, "y": 0, "color": "z"})).error
    assert not w.validate(A, Action(name="paint", args={"x": 0, "y": 0})).ok


def test_conflict_rule_last_writer_wins_and_accounting():
    w = world()
    good = w.target[0][0]
    bad = next(c for c in w.colours if c != good)
    outs = w.commit([paint(A, 0, 0, good), paint(B, 0, 0, bad), paint(C, 1, 0, w.target[0][1])])
    assert [o.feedback for o in outs] == [{"result": "overwritten"}, {"result": "painted"},
                                          {"result": "painted"}]
    assert all(o.accepted for o in outs)
    assert w.grid[0][0] == bad
    s = w.score()
    assert (s["paints"], s["useful_paints"], s["wrong_paints"], s["overwritten_paints"]) == (3, 2, 1, 1)
    assert (s["correct"], s["wrong"], s["unpainted"]) == (1, 1, 10)
    # next round: a duplicate (correct cell painted again) and the paint limit
    w.begin_round(2)
    outs = w.commit([paint(C, 1, 0, w.target[0][1]), paint(C, 2, 0, w.target[0][2], 1)])
    assert outs[0].accepted and outs[1].feedback == {"error": "paints_per_round", "detail": "at most 1 paint per round"} and not outs[1].accepted
    s = w.score()
    assert s["duplicate_paints"] == 1 and s["wasted_paints"] == 2 and s["paints"] == 4


def test_status_tools_and_toggles():
    w = world()
    w.commit([paint(A, 0, 0, w.target[0][0])])
    assert w.my_status(A) == {"paints": 1, "cells_mine": 1, "paints_left_this_round": 0}
    assert w.my_status(B) == {"paints": 0, "cells_mine": 0, "paints_left_this_round": 1}
    assert w.collective_status() == {"coverage": 1 / 12, "cells_matching": 1, "cells": 12}
    quiet = ColoringGrid(status_tools=())
    names = [s.name for s in quiet.tool_schemas()]
    assert names == ["paint"]
    assert {s.name for s in w.tool_schemas()} == {"paint", "my_status", "collective_status"}


def test_terminal_and_target_changes():
    w = world(target_changes=[[2, [[0, 0, "r"], [1, 0, "g"]]]])

    def fill():
        for y in range(3):
            for x in range(4):
                w.grid[y][x] = w.target[y][x]

    assert not w.terminal()
    fill()
    assert w.coverage() == 1.0 and not w.terminal()  # a target change is still scheduled
    w.begin_round(2)
    assert w.target[0][:2] == ["r", "g"] and w.verify()["target"][0][:2] == "rg"
    fill()
    assert w.terminal()
    with pytest.raises(ValueError):
        ColoringGrid(target_changes=[[2, [[9, 9, "r"]]]])


def test_claim_key_cells_and_zones():
    act = Action(name="paint", args={"x": 3, "y": 2, "color": "r"})
    assert world().claim_key(A, act) == "cell:3,2"
    z = world(zones=[1, 2])
    assert z.claim_key(A, act) == "zone:1,0"
    assert z.claim_key(A, Action(name="paint", args={"x": 1, "y": 2, "color": "r"})) == "zone:0,0"
    assert "Zones: 1 x 2" in z.observe(A).parts[0].text
    assert world().claim_key(A, Action(name="my_status", args={})) is None


def test_snapshot_keeps_state_not_config():
    w = world()
    w.commit([paint(A, 0, 0, w.target[0][0])])
    other = ColoringGrid(height=3, width=4, paints_per_round=2)
    other.restore(w.snapshot())
    assert other.grid == w.grid and other.score() == w.score() and other.paints_per_round == 2


# ---- runs -------------------------------------------------------------------------------------------

def coloring(name, participants, board=None, n=16, world_kw=None, metrics=None):
    board = board or Board()
    return Experiment(
        name=name, world=ColoringGrid(**{"height": 8, "width": 8, **(world_kw or {})}),
        participants=[participants] * n, medium=board,
        metrics=metrics if metrics is not None else GRID_METRICS + (CLAIM_METRICS if board.registry else []))


def values(run, name):
    return [v for _, v, _ in run.metrics[name]]


def test_row_major_painters_duplicate_work(tmp_path):
    run = coloring("rm", RowMajorPainter()).run(seed=0, max_rounds=5, out=tmp_path)
    assert run.score["correct"] == 5 and run.score["duplicate_paints"] == 75
    assert values(run, "coloring.coverage") == [k / 64 for k in range(1, 6)]
    assert values(run, "coloring.parallel_efficiency") == [1 / 16] * 5
    assert values(run, "coloring.duplicate_paints")[-1] == 75.0


def test_metrics_equal_score(tmp_path):
    run = coloring("rnd", RandomPainter(), n=6, world_kw={"palette": 3}).run(
        seed=3, max_rounds=8, out=tmp_path)
    s = run.score
    assert values(run, "coloring.coverage")[-1] == s["coverage"]
    assert values(run, "coloring.wrong_paints")[-1] == s["wrong_paints"] > 0
    assert values(run, "coloring.duplicate_paints")[-1] == s["duplicate_paints"]
    assert run.metrics["coloring.wrong_paints"][-1][2] == s["paints"]
    Run.load(run.dir)  # replay reproduces every metric and the score


def test_advisory_counts_violations_enforced_rejects(tmp_path):
    adv = coloring("adv", QueuePainter(), Board(registry=True)).run(seed=0, max_rounds=20, out=tmp_path)
    enf = coloring("enf", QueuePainter(), Board(registry=True, claim_policy="enforced")).run(
        seed=0, max_rounds=20, out=tmp_path)
    for run in (adv, enf):
        assert run.end_reason == "terminal" and run.score["coverage"] == 1.0
    assert values(adv, "claims.violations")[-1] > 0
    assert adv.score["duplicate_paints"] == values(adv, "claims.violations")[-1]
    # the same seed draws the same choices; enforced rejects exactly the violating paints
    assert values(enf, "claims.violations") == values(adv, "claims.violations")
    assert enf.score["duplicate_paints"] == 0 and enf.score["paints"] == 64
    claims = [e for e in enf.events if e["type"] == "claim"]
    assert claims and all(e["rejected"] == e["violation"] for e in claims)
    rejected = {e["action_id"] for e in claims if e["rejected"]}
    acts = {e["action_id"]: e for e in enf.events if e["type"] == "action_committed"}
    assert all(not acts[a]["accepted"] and acts[a]["feedback"]["error"] == "not_claimed" for a in rejected)
    assert values(enf, "claims.held")[0] > 0


def test_enforced_rejects_paints_without_a_claim(tmp_path):
    run = coloring("rm-enf", RowMajorPainter(), Board(registry=True, claim_policy="enforced"), n=3).run(
        seed=0, max_rounds=3, out=tmp_path)
    acts = [e for e in run.events if e["type"] == "action_committed"]
    assert len(acts) == 9 and not any(e["accepted"] for e in acts)
    assert run.score["coverage"] == 0.0 and values(run, "claims.violations") == [0.0] * 3
    adv = coloring("rm-adv", RowMajorPainter(), Board(registry=True), n=3).run(
        seed=0, max_rounds=3, out=tmp_path)
    assert adv.score["correct"] == 3


def test_worker_failures_kill_agents(tmp_path):
    exp = coloring("fail", QueuePainter(), Board(registry=True), n=4,
                   world_kw={"worker_failures": [[3, ["a000", "a001"]]]})
    run = exp.run(seed=1, max_rounds=5, out=tmp_path)
    started = {e["round"]: e["order"] for e in run.events if e["type"] == "round_started"}
    assert len(started[2]) == 4 and sorted(started[3]) == ["a002", "a003"]
    kills = [e for e in run.events if e["type"] == "intervention"]
    assert [(e["round"], e["agent"], e["op"], e["intervention"]) for e in kills] == [
        (3, "a000", "kill", "worker_failures"), (3, "a001", "kill", "worker_failures")]
    assert run.metrics["claims.held"][2][2] == 2  # denominators exclude the dead
    Run.load(run.dir)


def test_target_change_in_a_run(tmp_path):
    exp = coloring("chg", QueuePainter(), Board(registry=True), n=16,
                   world_kw={"target_changes": [[3, [[0, 0, "r"], [7, 7, "g"], [3, 3, "b"]]]]})
    run = exp.run(seed=2, max_rounds=20, out=tmp_path)
    assert run.end_reason == "terminal" and run.score["coverage"] == 1.0
    assert values(run, "coloring.coverage")[-1] == 1.0
    Run.load(run.dir)


def test_determinism_and_fork(tmp_path):
    def exp():
        return coloring("det", QueuePainter(ttl_rounds=3), Board(registry=True), n=8)

    a = exp().run(seed=5, max_rounds=6, out=tmp_path / "a")
    b = exp().run(seed=5, max_rounds=6, out=tmp_path / "b")
    strip = ("seq", "ts", "run")
    assert logical(a, strip) == logical(b, strip) and a.score == b.score
    child = a.fork(at_round=2).run()
    tail = [e for e in logical(a, strip) if e["round"] > 2]
    assert [e for e in logical(child, strip) if e["round"] > 2 and e["type"] != "run_started"] == tail


def test_viewer_builds_for_a_coloring_run(tmp_path):
    from .test_viewer import page_data

    run = coloring("view", QueuePainter(), Board(registry=True), n=4).run(seed=0, max_rounds=3, out=tmp_path)
    path = build(run.dir)
    data = page_data(path)
    assert data["world"] is None
    assert {"registry", "claim", "action_committed"} <= {e["type"] for e in data["events"]}


def test_fake_painter_llm_arm(tmp_path):
    from swarmlab.participants import LLMAgent
    from swarmlab.providers.fake import FakeProvider

    exp = Experiment(name="llm", world=ColoringGrid(height=4, width=4),
                     participants=[LLMAgent(model="fake:painter", max_tokens=64)] * 3,
                     metrics=GRID_METRICS, providers={"fake": FakeProvider()})
    run = exp.run(seed=0, max_rounds=4, out=tmp_path)
    acts = [e for e in run.events if e["type"] == "action_committed"]
    assert len(acts) == 12 and all(e["action"]["name"] == "paint" for e in acts)
    assert run.score["correct"] >= 4 and run.score["wrong_paints"] == 0


def test_example_spec_loads_and_runs_scripted_arms(tmp_path):
    arms = Experiment.arms_from_yaml(REPO / "examples" / "coloring_s0.yaml")
    assert {"row_major", "queue_advisory", "queue_enforced"} <= set(arms)
    assert arms["queue_enforced"].medium.claim_policy.type_name() == "enforced"
    run = arms["queue_enforced"].run(seed=0, out=tmp_path)
    assert run.score["coverage"] == 1.0


# ---- paint limit at call time, status, cap notes (paint-limit change) -------------------------------

def coloring_executor(tmp_path, commit="round_end", **kw):
    from swarmlab.blobs import BlobStore
    from swarmlab.executor import RoundExecutor

    w = world(**kw)
    board = Board()
    board.commit_mode = commit
    ex = RoundExecutor(run="r", round=1, world=w, board=board, blobs=BlobStore(tmp_path / "blobs"),
                       agents=[A, B, C], commit=commit, topology_rng=lambda: derive(0, "topology", 1))
    return w, ex


def cell(w, i):
    y, x = divmod(i, w.width)
    return {"x": x, "y": y, "color": w.target[y][x]}


async def test_round_end_refuses_a_paint_beyond_the_limit_at_call_time(tmp_path):
    w, ex = coloring_executor(tmp_path)
    first = await ex.call(A, "paint", cell(w, 0))
    assert first.ok and first.pending
    second = await ex.call(A, "paint", cell(w, 1))
    assert not second.ok and not second.pending
    assert second.error == ("paints_per_round: at most 1 paint per round; this one would be "
                            "rejected at commit")
    assert [a.args for _, _, a in ex.buffered_actions(A)] == [cell(w, 0)]  # not buffered
    ret = ex.events(A)[-1]
    assert ret.type == "tool_returned" and ret.result["ok"] is False
    assert (await ex.call(B, "paint", cell(w, 1))).ok  # the limit is per agent
    # malformed paints say why
    bad = await ex.call(C, "paint", {"x": 9, "y": 0, "color": "r"})
    assert bad.error.startswith("out_of_bounds: ")
    assert (await ex.call(C, "paint", {"x": 0, "y": 0, "color": "z"})).error.startswith("bad_color: ")


async def test_round_end_limit_counts_pending_paints_up_to_paints_per_round(tmp_path):
    w, ex = coloring_executor(tmp_path, paints_per_round=2)
    assert (await ex.call(A, "my_status", {})).result["paints_left_this_round"] == 2
    assert (await ex.call(A, "paint", cell(w, 0))).ok
    assert (await ex.call(A, "my_status", {})).result["paints_left_this_round"] == 1
    assert (await ex.call(A, "paint", cell(w, 1))).ok
    assert (await ex.call(A, "my_status", {})).result["paints_left_this_round"] == 0
    third = await ex.call(A, "paint", cell(w, 2))
    assert third.error.startswith("paints_per_round: at most 2 paints per round")
    assert len(ex.buffered_actions(A)) == 2
    assert (await ex.call(B, "my_status", {})).result["paints_left_this_round"] == 2


async def test_immediate_mode_paint_limit_is_unchanged(tmp_path):
    w, ex = coloring_executor(tmp_path, commit="immediate")
    first = await ex.call(A, "paint", cell(w, 0))
    assert first.ok and not first.pending and first.result["accepted"]
    assert (await ex.call(A, "my_status", {})).result["paints_left_this_round"] == 0
    second = await ex.call(A, "paint", cell(w, 1))  # the world's commit rejects it, as before
    assert second.ok and not second.pending and not second.result["accepted"]
    assert second.result["feedback"] == {"error": "paints_per_round", "detail": "at most 1 paint per round"}


def test_validate_without_pending_keeps_old_behaviour():
    w = world()
    act = Action(name="paint", args=cell(w, 0))
    assert w.validate(A, act).ok
    assert not w.validate(A, act, pending=[act]).ok
    assert w.validate(A, act, pending=[Action(name="other", args={})]).ok


async def test_executor_passes_pending_only_to_worlds_that_accept_it(tmp_path):
    from swarmlab.blobs import BlobStore
    from swarmlab.executor import RoundExecutor

    class OldStyle(ColoringGrid):
        def validate(self, agent, action):  # written against the old signature
            return super().validate(agent, action)

        def my_status(self, agent):
            return super().my_status(agent)

    w = OldStyle(height=3, width=4)
    w.reset(derive(0, "world"), [A])
    w.begin_round(1)
    ex = RoundExecutor(run="r", round=1, world=w, board=Board(), blobs=BlobStore(tmp_path / "b"),
                       agents=[A])
    assert (await ex.call(A, "paint", cell(w, 0))).ok
    assert (await ex.call(A, "paint", cell(w, 1))).ok  # no pre-check: buffered, rejected at commit
    assert (await ex.call(A, "my_status", {})).result["paints_left_this_round"] == 1


def test_description_states_the_limit_and_wasted_paints():
    text = ColoringGrid().description()
    assert "at most 1 cell per round" in text and "refused" in text
    assert "already has its target colour wastes your paint" in text
    assert "at most 3 cells per round" in ColoringGrid(paints_per_round=3).description()


def test_commit_feedback_says_why():
    w = world()
    outs = w.commit([paint(A, 9, 0, "r"), paint(A, 0, 0, "z", 1), paint(B, 0, 0, w.target[0][0]),
                     paint(B, 1, 0, w.target[0][1], 1), paint(C, 0, 0, w.target[0][0])])
    assert [o.feedback.get("error") or o.feedback.get("result") for o in outs] == [
        "out_of_bounds", "bad_color", "overwritten", "paints_per_round", "painted"]


class Spinner(Participant):
    """Keeps painting until the cap: every call after the first is refused."""

    async def turn(self, view, tools):
        while True:
            await tools.call("paint", {"x": 0, "y": 0, "color": "r"})


class Busy(Participant):
    """Keeps asking for its status until the cap: every call succeeds."""

    async def turn(self, view, tools):
        while True:
            await tools.call("my_status", {})


def test_cap_note_marks_spinning_turns_and_report_counts_rejected_calls(tmp_path):
    from swarmlab.report import build_report

    out = tmp_path / "runs"
    exp = Experiment(name="cap", world=ColoringGrid(height=2, width=2),
                     participants=[Spinner(), Busy()], metrics=GRID_METRICS)
    run = exp.run(seed=0, max_rounds=2, out=out, max_calls_per_turn=5)
    ended = {e["agent"]: e for e in run.events if e["type"] == "turn_ended" and e["round"] == 1}
    spinner, busy = ended[agent_id(0)], ended[agent_id(1)]
    assert spinner["yield_kind"] == "cap" and spinner["usage"]["notes"] == ["cap:4"]
    assert busy["yield_kind"] == "cap" and "notes" not in busy["usage"]
    Run.load(run.dir)  # replay reproduces the run
    text = build_report(out, "cap")
    assert "| rejected tool calls |" in text
    assert "| 8/20 |" in text  # per round: spinner 4 of 5 answered calls refused, busy 0 of 5


def test_m3_vllm_spec_caps_model_calls_per_turn():
    import yaml  # the spec needs a vLLM base_url to build, so read it as data

    doc = yaml.safe_load((REPO / "experiments" / "m3_coloring_vllm.yaml").read_text())
    assert set(doc["arms"]) == {"s0", "claims-advisory", "claims-enforced"}
    for arm in doc["arms"].values():
        assert [g["params"]["max_calls"] for g in arm["participants"]] == [4]
