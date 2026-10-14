"""ColoringGrid with crossed zones, painters, coloring, relay and claim metrics (M3b)."""
from __future__ import annotations

import inspect
import re
from pathlib import Path

import pytest

from swarmlab import Board, Experiment, Run
from swarmlab.events import (
    ActionCommittedEvent,
    DeliveryEvent,
    PostEvent,
    ReadEvent,
    RoundStartedEvent,
)
from swarmlab.ids import agent_id
from swarmlab.metrics.coloring import RelayStatus
from swarmlab.participants.base import Participant
from swarmlab.participants.scripted import (
    QueuePainter,
    RandomPainter,
    RelayPainter,
    RowMajorPainter,
)
from swarmlab.rng import derive
from swarmlab.viewer.build import build
from swarmlab.world.base import Action
from swarmlab.world.coloring import (
    ColoringGrid,
    board_hash,
    format_report,
    generate_zones,
    join_names,
    needed_cells,
    paintable_needed_cells,
    parse_observation,
    parse_report,
    parse_zones,
    zone_at,
    zone_size,
)

from .helpers import logical

A, B, C, D = agent_id(0), agent_id(1), agent_id(2), agent_id(3)
REPO = Path(__file__).resolve().parent.parent
GRID_METRICS = ["coloring.coverage", "coloring.duplicate_paints", "coloring.wrong_paints",
                "coloring.parallel_efficiency"]
CLAIM_METRICS = ["claims.violations", "claims.held"]
ZONE_METRICS = ["coloring.relay_status", "coloring.zone_paint_refusals",
                "coloring.posts_by_agent", "coloring.reads_by_agent"]
# `world()` is the default board of seed 0 (8 x 15, three 5 x 2 zones, Z0 G0 -> G1, Z1 G1 -> G2,
# Z2 G2 -> G0); zone cells always come from `w.crossed`. Its row 0 is entirely open (asserted in
# `world()`): the paint tests below use row 0 cells, as do the runs of seed 0 on the default board.


def world(**kw):
    w = ColoringGrid(**kw)
    w.reset(derive(0, "world"), [A, B, C])
    w.begin_round(1)
    if (w.height, w.width, w.n_zones) == (8, 15, 3):
        assert all(zone_at(w.crossed, x, 0) is None for x in range(w.width))  # row 0 is open
    return w


def zone(w, zid):
    return next(z for z in w.crossed if z["zone_id"] == zid)


def span(z):
    """The coordinates of a zone line: `x=5..9, y=4..5`."""
    return f"x={z['x']}..{z['x'] + z['width'] - 1}, y={z['y']}..{z['y'] + z['height'] - 1}"


def text(w, agent=A):
    return w.observe(agent).parts[0].text


def paint(agent, x, y, color, n=0):
    return (agent, f"x-{agent}-{n}", Action(name="paint", args={"x": x, "y": y, "color": color}))


def zone_cells(z):
    return [(x, y) for y in range(z["y"], z["y"] + z["height"])
            for x in range(z["x"], z["x"] + z["width"])]


# ---- boards and zones -------------------------------------------------------------------------------

def test_board_id_is_a_content_hash_of_the_generated_board():
    ids = []
    for seed in (1, 2, 3):
        w, again = ColoringGrid(), ColoringGrid()
        w.reset(derive(seed, "world"), [A, B, C])
        again.reset(derive(seed, "world"), [A, B, C])
        assert re.fullmatch(r"[0-9a-f]{10}", w.board_id)
        assert again.board_id == w.board_id and again.target == w.target and again.crossed == w.crossed
        assert w.score()["board"] == w.verify()["board"] == w.render_state()["board"] == w.board_id
        assert w.board_id == board_hash(w.target, w.crossed)
        ids.append(w.board_id)
    assert len(set(ids)) == 3
    w = world()
    zones = [dict(z) for z in w.crossed]
    zones[0]["x"] += 1 if zones[0]["x"] == 0 else -1
    assert board_hash(w.target, zones) != w.board_id  # the zones are part of the hash
    two = ColoringGrid(groups=2, zones=3)
    two.reset(derive(0, "world"), [A, B, C])
    assert two.target == w.target and two.board_id != w.board_id  # so are their groups


def test_param_errors():
    with pytest.raises(ValueError, match="groups must be >= 2"):
        ColoringGrid(groups=1)
    with pytest.raises(ValueError, match="zones must be >= 1"):
        ColoringGrid(zones=0)
    with pytest.raises(ValueError, match="zone_size must be"):
        ColoringGrid(zone_size=[5])
    with pytest.raises(ValueError, match="zone_size must be"):
        ColoringGrid(zone_size=[0, 2])
    with pytest.raises(ValueError, match="unknown status tools"):
        ColoringGrid(status_tools=["nope"])
    with pytest.raises(ValueError, match="cannot hold 3 zones"):
        ColoringGrid(height=1, width=2)
    with pytest.raises(ValueError, match="cannot hold 5 zones"):
        ColoringGrid(height=2, width=2, zones=5)
    with pytest.raises(ValueError, match="at least 3 agents"):
        ColoringGrid().reset(derive(1, "world"), [A, B])
    with pytest.raises(ValueError, match="at least 4 agents"):
        ColoringGrid(groups=4).reset(derive(1, "world"), [A, B, C])
    assert "board" not in inspect.signature(ColoringGrid).parameters


@pytest.mark.parametrize("hw,n_zones,groups,max_size,size", [
    ((8, 8), 3, 3, (5, 2), (5, 2)), ((3, 4), 3, 3, (5, 2), (4, 1)), ((2, 2), 3, 3, (5, 2), (1, 1)),
    ((2, 6), 3, 3, (5, 2), (2, 2)), ((8, 15), 3, 3, (5, 2), (5, 2)), ((20, 20), 3, 3, (5, 2), (5, 2)),
    ((8, 15), 2, 2, (5, 2), (5, 2)), ((8, 15), 6, 3, (5, 2), (5, 2)), ((8, 15), 4, 4, (5, 2), (5, 2)),
    ((4, 6), 6, 3, (5, 2), (2, 2)), ((2, 2), 4, 2, (5, 2), (1, 1)), ((9, 9), 3, 3, (3, 3), (3, 3)),
    ((8, 15), 3, 3, (3, 3), (3, 3)), ((5, 5), 4, 4, (3, 3), (2, 2)), ((2, 15), 3, 3, (5, 2), (5, 2)),
])
def test_generated_zones_fit_shrink_and_follow_the_cycle(hw, n_zones, groups, max_size, size):
    h, w_ = hw
    assert zone_size(h, w_, n_zones, *max_size) == size
    for seed in range(20):
        zones = generate_zones(derive(seed, "world"), h, w_, n_zones, groups, size)
        cells = [c for z in zones for c in zone_cells(z)]
        assert len(cells) == len(set(cells)) == n_zones * size[0] * size[1]
        assert all(0 <= x < w_ and 0 <= y < h for x, y in cells)
        assert [(z["zone_id"], z["viewer_group"], z["painter_group"]) for z in zones] == [
            (f"Z{i}", i % groups, (i + 1) % groups) for i in range(n_zones)]
        assert all(set(z) == {"zone_id", "x", "y", "width", "height", "viewer_group",
                              "painter_group"} for z in zones)


def test_random_board_keeps_the_old_target_draws_and_is_deterministic():
    for kw, (h, w_) in [({"height": 3, "width": 4}, (3, 4)), ({}, (8, 15)), ({"groups": 4}, (8, 15))]:
        w = ColoringGrid(**kw)
        w.reset(derive(0, "world"), [A, B, C, D])
        rng = derive(0, "world")
        assert w.target == [[rng.choice("rgby") for _ in range(w_)] for _ in range(h)]  # drawn first
        again = ColoringGrid(**kw)
        again.reset(derive(0, "world"), [A, B, C, D])
        assert again.target == w.target and again.crossed == w.crossed
        assert again.board_id == w.board_id


@pytest.mark.parametrize("kw,n", [({}, 3), ({}, 5), ({"groups": 2}, 2), ({"groups": 2}, 3),
                                  ({"groups": 4, "zones": 4}, 4), ({"groups": 4, "zones": 4}, 6),
                                  ({"groups": 3, "zones": 6}, 3), ({"groups": 3, "zones": 1}, 3),
                                  ({"groups": 2, "zones": 5, "height": 4, "width": 6}, 2),
                                  ({"zone_size": [3, 3]}, 3)])
def test_every_cell_is_paintable_and_viewers_differ_from_painters(kw, n):
    agents = [agent_id(i) for i in range(n)]
    for seed in range(5):
        w = ColoringGrid(**kw)
        w.reset(derive(seed, "world"), agents)
        groups = w.verify()["groups"]
        assert sorted(groups) == agents and len(w.crossed) == w.n_zones
        for z in w.crossed:
            assert z["viewer_group"] != z["painter_group"]
            assert any(g == z["painter_group"] for g in groups.values())
            assert any(g == z["viewer_group"] for g in groups.values())
        for y in range(w.height):
            for x in range(w.width):
                assert any(w.validate(a, Action(name="paint", args={"x": x, "y": y, "color": "r"})).ok
                           for a in agents), (kw, n, seed, x, y)


def test_two_groups_see_one_zone_and_paint_the_other():
    w = ColoringGrid(groups=2)
    w.reset(derive(0, "world"), [A, B, C])
    w.begin_round(1)
    assert [(z["zone_id"], z["viewer_group"], z["painter_group"]) for z in w.crossed] == [
        ("Z0", 0, 1), ("Z1", 1, 0)]
    assert w.groups == {A: 0, B: 1, C: 0}
    t = text(w, A)
    assert "Groups: G0 = a000, a002; G1 = a001" in t
    assert ("You are a000, in group G0. Your group sees the targets of Z0 and must report them on "
            "the message board for G1, which paints Z0. Only your group may paint Z1; its targets "
            "are shown to G1, so you must get them from the board.") in t
    assert "split into 2 groups, G0 and G1, and the grid holds 2 zones, Z0 and Z1, of" in w.description()


def test_more_zones_than_groups_and_one_zone():
    w = ColoringGrid(groups=3, zones=6)
    w.reset(derive(0, "world"), [A, B, C])
    w.begin_round(1)
    assert [z["zone_id"] for z in w.crossed] == [f"Z{i}" for i in range(6)]
    t = text(w, A)
    assert t.count("Your group sees the targets of") == 2 and t.count("Only your group may paint") == 2
    assert "sees the targets of Z0 " in t and "sees the targets of Z3 " in t
    assert "may paint Z2;" in t and "may paint Z5;" in t
    assert "the grid holds 6 zones, Z0, Z1, Z2, Z3, Z4 and Z5, of 5 x 2 cells" in w.description()
    one = ColoringGrid(zones=1)
    one.reset(derive(0, "world"), [A, B, C])
    assert "the grid holds 1 zone, Z0, of 5 x 2 cells" in one.description()
    assert one.description_for(C) == one.description() + " You are a002, in group G2."  # no duties
    assert join_names(["A"]) == "A" and join_names(["A", "B"]) == "A and B"
    assert join_names(["A", "B", "C"]) == "A, B and C"


def test_zone_size_param_sets_the_zones_and_the_report_length():
    w = ColoringGrid(zone_size=[3, 3])
    w.reset(derive(0, "world"), [A, B, C])
    assert all((z["width"], z["height"]) == (3, 3) for z in w.crossed)
    d = w.description()
    assert "of 3 x 3 cells" in d and '"<zone id> <colour 1> <colour 2> ... <colour 9>"' in d
    assert "Zones (3 columns x 3 rows each;" in text(w, A)
    small = ColoringGrid(height=4, width=6, zones=6)  # 5 x 2 zones do not fit six times: 2 x 2
    small.reset(derive(0, "world"), [A, B, C])
    assert (small.zone_w, small.zone_h) == (2, 2) and "of 2 x 2 cells" in small.description()
    assert small.score()["open_correct"] == 0  # the zones cover the whole grid: no open cells


# ---- observation --------------------------------------------------------------------------------------

def test_observation_masks_the_zones_a_group_cannot_see():
    w = world()
    zones = {z["zone_id"]: z for z in w.crossed}
    for agent, seen in [(A, "Z0"), (B, "Z1"), (C, "Z2")]:
        target, current, colours = parse_observation(text(w, agent))
        assert current == ["." * 15] * 8 and colours == ["r", "g", "b", "y"]
        for zid, z in zones.items():
            shown = [target[y][x] for x, y in zone_cells(z)]
            want = [w.target[y][x] for x, y in zone_cells(z)]
            assert shown == (want if zid == seen else ["?"] * 10), (agent, zid)
        open_ = [(x, y) for y in range(8) for x in range(15) if zone_at(w.crossed, x, y) is None]
        assert len(open_) == 90 and all(target[y][x] == w.target[y][x] for x, y in open_)
        assert w.observe(agent).private == {"group": int(agent[1:]) % 3}
    t = text(w, A)
    assert f"Z0 at {span(zones['Z0'])}: targets visible to G0, painted by G1" in t
    assert f"Z2 at {span(zones['Z2'])}: targets visible to G2, painted by G0" in t
    assert "Groups: G0 = a000; G1 = a001; G2 = a002" in t
    assert ("You are a000, in group G0. Your group sees the targets of Z0 and must report them on "
            "the message board for G1, which paints Z0. Only your group may paint Z2; its targets "
            "are shown to G2, so you must get them from the board.") in t
    info = parse_zones(t)
    assert info["agent"] == "a000" and info["group"] == 0 and info["groups"][1] == ["a001"]
    assert [(z["zone_id"], z["viewers"], z["painter_group"]) for z in info["zones"]] == [
        ("Z0", [0], 1), ("Z1", [1], 2), ("Z2", [2], 0)]
    assert [{k: z[k] for k in ("x", "y", "width", "height")} for z in info["zones"]] == [
        {k: z[k] for k in ("x", "y", "width", "height")} for z in w.crossed]
    assert len(needed_cells(*parse_observation(t)[:2])) == 120 - 20  # hidden cells skipped
    assert len(paintable_needed_cells(t)) == 90  # open only: Z0 is not ours, Z2 is hidden


def test_current_grid_is_fully_visible():
    w = world()
    z1 = next(z for z in w.crossed if z["zone_id"] == "Z1")
    x, y = zone_cells(z1)[0]
    w.commit([paint(C, x, y, w.target[y][x])])  # G2 paints Z1
    _, current, _ = parse_observation(text(w, A))  # G0 cannot see Z1's targets, sees its paint
    assert current[y][x] == w.target[y][x]


def test_oracle_shows_a_zone_to_its_painters_too():
    w = world(oracle=True)
    target, _, _ = parse_observation(text(w, B))  # G1 paints Z0 and views Z1
    z0 = next(z for z in w.crossed if z["zone_id"] == "Z0")
    z2 = next(z for z in w.crossed if z["zone_id"] == "Z2")
    assert [target[y][x] for x, y in zone_cells(z0)] == [w.target[y][x] for x, y in zone_cells(z0)]
    assert [target[y][x] for x, y in zone_cells(z2)] == ["?"] * 10  # still hidden
    t = text(w, B)
    assert f"Z0 at {span(z0)}: targets visible to G0 and G1 (its painters), painted by G1" in t
    assert parse_zones(t)["zones"][0]["viewers"] == [0, 1]
    assert len(paintable_needed_cells(t)) == 100
    assert "shown to two groups" in w.description() and "have to reach" not in w.description()


def test_description_states_rules_format_and_board_timing():
    w = world()
    d = w.description()
    for phrase in ("at most 10 cells per round", "refused", "already has its target colour wastes",
                   ("split into 3 groups, G0, G1 and G2, and the grid holds 3 zones, Z0, Z1 and Z2, "
                    "of 5 x 2 cells."),
                   "Cells outside the zones are open: anyone may paint them",
                   "a paint inside a zone by an agent outside its painting group is refused",
                   '"<zone id> <colour 1> <colour 2> ... <colour 10>"',
                   "row-major order", "delivered when the round ends",
                   "read with read_board from the next round on"):
        assert phrase in d, phrase
    # no concrete colour sequence that a model could copy as if it were a zone's targets
    assert "red green blue" not in d
    assert "at most 1 cell per round" in ColoringGrid(paints_per_round=1).description()
    assert w.description_for(C).startswith(d) and "You are a002, in group G2." in w.description_for(C)
    assert ColoringGrid().description_for(A) == ColoringGrid().description()  # before reset


# ---- validation and commit ----------------------------------------------------------------------------

def test_validate_bounds_and_colour():
    w = world()
    assert w.validate(A, Action(name="paint", args={"x": 14, "y": 0, "color": "r"})).ok
    assert not w.validate(A, Action(name="paint", args={"x": 15, "y": 0, "color": "r"})).ok
    assert not w.validate(A, Action(name="paint", args={"x": 0, "y": -1, "color": "r"})).ok
    assert "unknown color" in w.validate(A, Action(name="paint", args={"x": 0, "y": 0, "color": "z"})).error
    assert not w.validate(A, Action(name="paint", args={"x": 0, "y": 0})).ok


def test_zone_paint_refused_at_call_time_names_only_the_painter_group():
    w = world()
    z0 = zone_cells(zone(w, "Z0"))
    for (x, y), color in [(z0[0], "r"), (z0[-1], "g")]:  # Z0 cells, painted by G1
        for agent in (A, C):
            ack = w.validate(agent, Action(name="paint", args={"x": x, "y": y, "color": color}))
            assert not ack.ok
            assert ack.error == (f"zone_paint_refused: cell ({x}, {y}) is in zone Z0, which only "
                                 "group G1 may paint")
            for word in ("red", "green", "blue", "yellow", "correct", "wrong", "target"):
                assert word not in ack.error
        assert w.validate(B, Action(name="paint", args={"x": x, "y": y, "color": color})).ok
    (x2, y2), (x1, y1), (x0, y0) = zone_cells(zone(w, "Z2"))[0], zone_cells(zone(w, "Z1"))[0], z0[0]
    assert w.validate(A, Action(name="paint", args={"x": x2, "y": y2, "color": "r"})).ok  # Z2: G0's
    assert w.validate(C, Action(name="paint", args={"x": x1, "y": y1, "color": "r"})).ok  # Z1: G2's
    assert w.validate(A, Action(name="paint", args={"x": 0, "y": 0, "color": "r"})).ok  # open
    full = [Action(name="paint", args={"x": i, "y": 0, "color": "r"}) for i in range(10)]
    assert w.validate(B, Action(name="paint", args={"x": x0, "y": y0, "color": "r"}),
                      pending=full).error.startswith("paints_per_round")


def test_zone_paint_refused_again_at_commit_without_using_the_limit():
    w = world(paints_per_round=1)
    x, y = zone_cells(zone(w, "Z0"))[0]
    outs = w.commit([paint(A, x, y, w.target[y][x]), paint(A, 0, 0, w.target[0][0], 1)])
    assert outs[0].accepted is False and outs[0].feedback == {
        "error": "zone_paint_refused",
        "detail": f"cell ({x}, {y}) is in zone Z0, which only group G1 may paint"}
    assert outs[1].accepted and outs[1].feedback == {"result": "painted"}
    assert w.grid[y][x] == "." and w.score()["zone_paint_refusals"] == 1
    assert w.score()["paints"] == 1


def test_conflict_rule_last_writer_wins_and_accounting():
    w = world(paints_per_round=1)
    good = w.target[0][0]
    bad = next(c for c in w.colours if c != good)
    outs = w.commit([paint(A, 0, 0, good), paint(B, 0, 0, bad), paint(C, 1, 0, w.target[0][1])])
    assert [o.feedback for o in outs] == [{"result": "overwritten"}, {"result": "painted"},
                                          {"result": "painted"}]
    assert w.grid[0][0] == bad
    s = w.score()
    assert (s["paints"], s["useful_paints"], s["wrong_paints"], s["overwritten_paints"]) == (3, 2, 1, 1)
    assert (s["correct"], s["wrong"], s["unpainted"]) == (1, 1, 118)
    w.begin_round(2)
    outs = w.commit([paint(C, 1, 0, w.target[0][1]), paint(C, 2, 0, w.target[0][2], 1)])
    assert outs[0].accepted and not outs[1].accepted
    assert outs[1].feedback == {"error": "paints_per_round", "detail": "at most 1 paint per round"}
    s = w.score()
    assert s["duplicate_paints"] == 1 and s["wasted_paints"] == 2 and s["paints"] == 4


def test_commit_feedback_says_why():
    w = world(paints_per_round=1)
    x, y = zone_cells(zone(w, "Z0"))[0]
    outs = w.commit([paint(A, 15, 0, "r"), paint(A, 0, 0, "z", 1), paint(B, 0, 0, w.target[0][0]),
                     paint(B, 1, 0, w.target[0][1], 1), paint(C, 0, 0, w.target[0][0]),
                     paint(C, x, y, "r", 1)])
    assert [o.feedback.get("error") or o.feedback.get("result") for o in outs] == [
        "out_of_bounds", "bad_color", "overwritten", "paints_per_round", "painted",
        "zone_paint_refused"]


def test_score_keys_and_zone_counts():
    w = world()
    s = w.score()
    for key in ("coverage", "correct", "wrong", "unpainted", "paints", "useful_paints",
                "duplicate_paints", "wrong_paints", "overwritten_paints", "wasted_paints",
                "board", "zones_completed", "crossed_correct", "open_correct", "zone_paint_refusals"):
        assert key in s, key
    assert (s["zones_completed"], s["crossed_correct"], s["open_correct"]) == (0, 0, 0)
    z0 = next(z for z in w.crossed if z["zone_id"] == "Z0")
    for x, y in zone_cells(z0):
        w.grid[y][x] = w.target[y][x]
    w.grid[0][0] = w.target[0][0]
    z2 = next(z for z in w.crossed if z["zone_id"] == "Z2")
    x, y = zone_cells(z2)[0]
    w.grid[y][x] = w.target[y][x]
    s = w.score()
    assert (s["zones_completed"], s["crossed_correct"], s["open_correct"], s["correct"]) == (1, 11, 1, 12)
    assert ColoringGrid().score()["zones_completed"] == 0  # before reset
    v = w.verify()
    assert v["board"] == w.board_id and v["groups"] == {A: 0, B: 1, C: 2} and len(v["crossed_zones"]) == 3
    assert v["oracle"] is False and "zones" not in v


def test_status_tools_are_off_by_default_and_leak_free():
    assert [s.name for s in world().tool_schemas()] == ["paint"]
    w = world(status_tools=["my_status", "collective_status"])
    w.commit([paint(A, 0, 0, w.target[0][0])])
    assert w.my_status(A) == {"paints": 1, "paints_left_this_round": 9}
    assert w.my_status(B) == {"paints": 0, "paints_left_this_round": 10}
    assert w.collective_status() == {"cells_painted": 1, "cells": 120}
    assert {s.name for s in w.tool_schemas()} == {"paint", "my_status", "collective_status"}


def test_terminal_and_target_changes():
    w = world(target_changes=[[2, [[0, 0, "r"], [1, 0, "g"]]]])

    def fill():
        for y in range(8):
            for x in range(15):
                w.grid[y][x] = w.target[y][x]

    assert not w.terminal()
    fill()
    assert w.coverage() == 1.0 and not w.terminal()  # a target change is still scheduled
    w.begin_round(2)
    assert w.target[0][:2] == ["r", "g"] and w.verify()["target"][0][:2] == "rg"
    fill()
    assert w.terminal()
    with pytest.raises(ValueError):
        ColoringGrid(target_changes=[[2, [[99, 9, "r"]]]])


def test_claim_keys_are_cells():
    w = world()
    assert w.claim_key(A, Action(name="paint", args={"x": 3, "y": 2, "color": "r"})) == "cell:3,2"
    assert w.claim_key(A, Action(name="paint", args={"x": 5, "y": 4, "color": "r"})) == "cell:5,4"
    assert w.claim_key(A, Action(name="my_status", args={})) is None


def test_snapshot_keeps_state_not_config():
    w = world()
    w.commit([paint(A, 0, 0, w.target[0][0])])
    other = ColoringGrid(paints_per_round=2, oracle=True)
    other.restore(w.snapshot())
    assert other.grid == w.grid and other.score() == w.score() and other.paints_per_round == 2
    assert other.crossed == w.crossed and other.groups == w.groups and other.board_id == w.board_id
    assert other.oracle is True and other.n_groups == 3
    assert text(other, A) != text(w, A)  # oracle (config) now unmasks Z2 for G0


def test_requirements_round_end_and_one_agent_per_group():
    with pytest.raises(ValueError, match=r"requires n_agents>=3 \(world.n_groups\), got 2"):
        Experiment(name="x", world=ColoringGrid(), participants=[RelayPainter()] * 2)
    with pytest.raises(ValueError, match=r"requires n_agents>=4 \(world.n_groups\), got 3"):
        Experiment(name="x", world=ColoringGrid(groups=4), participants=[RelayPainter()] * 3)
    Experiment(name="x", world=ColoringGrid(groups=2), participants=[RelayPainter()] * 2)
    Experiment(name="x", world=ColoringGrid(groups=4, zones=4), participants=[RelayPainter()] * 4)
    with pytest.raises(ValueError, match="commit=round_end"):
        Experiment(name="x", world=ColoringGrid(), participants=[RelayPainter()] * 3,
                   options={"commit": "immediate"})
    Experiment(name="x", world=ColoringGrid(), participants=[RelayPainter()] * 4)


# ---- relay reports and the relay-status metric --------------------------------------------------------

def test_parse_report_is_loose():
    want = list("rgbyrgbyrg")
    assert parse_report(format_report("Z0", want), "Z0", 10) == [want]
    assert parse_report("z0: R, G, B, Y, R; G B Y R G!", "Z0", 10) == [want]
    assert parse_report("Zone Z0 targets -> Red green BLUE yellow red green blue yellow red green "
                        "blue blue", "Z0", 10) == [want]  # only the first ten count
    assert parse_report("Z0 rows: rgbyr / gbyrg", "Z0", 10) == [want]  # rows copied from the grid
    assert parse_report("Z0 painted by G1: r g b y r g b y r g", "Z0", 10) == [want]  # 'by', G1 skipped
    assert parse_report("Z0 red green blue", "Z0", 10) == []  # too few
    assert parse_report("Z0 red green blue", "Z0", 3) == [list("rgb")]  # a 3-cell zone
    assert parse_report("Z10 r g b y r g b y r g", "Z1", 10) == []  # not the zone id
    assert parse_report("Z10 r g b y r g b y r g", "Z10", 10) == [want]
    assert parse_report("Z1 r r r r r r r r r r and Z0 r g b y r g b y r g", "Z0", 10) == [want]


def test_parse_zones_reads_the_zone_lines():
    lines = [
        "Target grid (2 rows x 3 columns; x = column 0-2, y = row 0-1; \"?\" = hidden from you):",
        "r??", "g??", "", "Groups: G0 = a000, a002; G1 = a001",
        "Zones (2 columns x 2 rows each; cells outside the zones are open: anyone may paint them):",
        "Z0 at x=1..2, y=0..1: targets visible to G0, painted by G1",
        "Z10 at x=0..0, y=0..0: targets visible to G1 and G0 (its painters), painted by G0",
        "Z1 type A at x=0..0, y=0..0: targets visible to G1, painted by G0",  # the old format
        "You are a002, in group G0. Only your group may paint Z1."]
    info = parse_zones("\n".join(lines))
    assert info["zones"] == [
        {"zone_id": "Z0", "x": 1, "y": 0, "width": 2, "height": 2, "viewers": [0], "painter_group": 1},
        {"zone_id": "Z10", "x": 0, "y": 0, "width": 1, "height": 1, "viewers": [1, 0],
         "painter_group": 0}]
    assert info["groups"] == {0: ["a000", "a002"], 1: ["a001"]}
    assert (info["agent"], info["group"]) == ("a002", 0)


def relay_world():
    w = world()
    z0 = next(z for z in w.crossed if z["zone_id"] == "Z0")
    return w, z0, [w.target[y][x] for x, y in zone_cells(z0)]


def fold(w, events):
    m = RelayStatus()
    m.set_truth(w.verify())
    m.set_agents([A, B, C])
    for ev in [RoundStartedEvent(run="r", round=1, order=[A, B, C]), *events]:
        m.update(ev)
    return m


def post(agent, pid, text_, rnd=1):
    return PostEvent(run="r", round=rnd, agent=agent, post_id=pid, channel="main", text=text_)


def deliver(pid, did, to):
    return DeliveryEvent(run="r", round=1, post_id=pid, recipient=to, delivery_id=did,
                         eligible_round=2, content_hash="h")


def read(agent, *dids):
    return ReadEvent(run="r", round=2, agent=agent, delivery_ids=list(dids))


def test_relay_status_buckets_on_constructed_logs():
    w, z0, want = relay_world()
    report = format_report("Z0", want)
    assert fold(w, []).statuses()["Z0"] == "viewer_never_posted"
    assert fold(w, [post(B, "p1", report)]).statuses()["Z0"] == "viewer_never_posted"  # B: not G0
    assert fold(w, [post(A, "p1", "hello")]).statuses()["Z0"] == "viewer_posted_wrong_or_unparsed"
    wrong = format_report("Z0", ["r" if c != "r" else "g" for c in want])
    assert fold(w, [post(A, "p1", wrong)]).statuses()["Z0"] == "viewer_posted_wrong_or_unparsed"
    base = [post(A, "p1", "hello"), post(A, "p2", report), deliver("p2", "d1", B),
            deliver("p2", "d2", C), deliver("p1", "d3", B)]
    assert fold(w, base).statuses()["Z0"] == "painter_never_read_correct_report"
    assert fold(w, [*base, read(C, "d2"), read(B, "d3")]).statuses()["Z0"] == (
        "painter_never_read_correct_report")  # C is not G1; B read only the wrong post
    m = fold(w, [*base, read(B, "d1")])
    assert m.statuses()["Z0"] == "painter_read_but_painted_wrong"
    assert m.value() == (1.0, 3)
    paints = [ActionCommittedEvent(run="r", round=2, agent=B, action_id=f"x{i}", accepted=True,
                                   action={"name": "paint", "args": {"x": x, "y": y, "color": c}},
                                   feedback={"result": "painted"})
              for i, ((x, y), c) in enumerate(zip(zone_cells(z0), want, strict=True))]
    m = fold(w, [*base, read(B, "d1"), *paints])
    assert m.statuses() == {"Z0": "complete", "Z1": "viewer_never_posted", "Z2": "viewer_never_posted"}
    outs = {n: (v, d, lab) for n, v, d, lab in m.outputs()}
    assert outs["coloring.relay_status"] == (1.0, 3, None)
    assert outs["coloring.relay_status.Z0"] == (0.0, 1, "complete")
    assert outs["coloring.relay_status.Z1"] == (1.0, 1, "viewer_never_posted")
    assert m.output_names() == ["coloring.relay_status", "coloring.relay_status.Z0",
                                "coloring.relay_status.Z1", "coloring.relay_status.Z2"]
    # completed without any relay still counts as complete
    assert fold(w, paints).statuses()["Z0"] == "complete"


# ---- runs ---------------------------------------------------------------------------------------------

def relay_exp(name="relay", post_=True, n=3, metrics=None, **world_kw):
    return Experiment(name=name, world=ColoringGrid(**world_kw),
                      participants=[RelayPainter(post=post_)] * n,
                      metrics=metrics if metrics is not None else ["coloring.coverage", *ZONE_METRICS])


@pytest.mark.parametrize("seed", range(1, 6))
def test_relay_painters_complete_every_board(tmp_path, seed):
    run = relay_exp().run(seed=seed, max_rounds=20, out=tmp_path)
    s = run.score
    assert run.end_reason == "terminal" and re.fullmatch(r"[0-9a-f]{10}", s["board"])
    assert s["coverage"] == 1.0 and run.meta["last_round"] <= 20
    crossed = 3 * 5 * 2
    assert (s["zones_completed"], s["crossed_correct"], s["open_correct"]) == (3, crossed, 120 - crossed)
    assert s["zone_paint_refusals"] == 0 and s["wrong_paints"] == 0
    last = {n: run.metrics[n][-1][1] for n in run.metrics}
    assert last["coloring.relay_status"] == 3.0
    assert last["coloring.relay_status.Z0"] == 0.0  # complete
    assert [last[f"coloring.posts_by_agent.{a}"] for a in (A, B, C)] == [1.0, 1.0, 1.0]
    Run.load(run.dir)  # replay reproduces every metric (labels included) and the score


def test_never_posting_relay_painters_leave_every_zone_unpainted(tmp_path):
    for seed in range(1, 6):
        run = relay_exp("silent", post_=False).run(seed=seed, max_rounds=20, out=tmp_path)
        s = run.score
        assert run.end_reason == "max_rounds"
        assert (s["zones_completed"], s["crossed_correct"], s["open_correct"]) == (0, 0, 90)
        w = ColoringGrid()
        w.restore(_last_world(run))
        assert all(w.grid[y][x] == "." for z in w.crossed for x, y in zone_cells(z))
        assert set(_labels(run).values()) == {"viewer_never_posted"}


def _last_world(run):
    from swarmlab.snapshot import SnapshotStore

    snaps = SnapshotStore(run.dir)
    return snaps.load(snaps.latest())["world"]


def _labels(run):
    out = {}
    for ev in run.events:
        if ev["type"] == "metric" and ev["name"].startswith("coloring.relay_status."):
            out[ev["name"]] = ev.get("label")
    return out


def test_relay_painters_with_six_agents_split_work(tmp_path):
    run = relay_exp("six", n=6).run(seed=2, max_rounds=20, out=tmp_path)
    assert run.end_reason == "terminal" and run.score["duplicate_paints"] == 0
    assert run.score["zones_completed"] == 3


def test_relay_painters_on_a_small_board(tmp_path):
    run = relay_exp("gen", height=8, width=8).run(seed=3, max_rounds=20, out=tmp_path)
    assert run.end_reason == "terminal" and re.fullmatch(r"[0-9a-f]{10}", run.score["board"])
    Run.load(run.dir)


@pytest.mark.parametrize("world_kw,n", [({"groups": 2}, 2), ({"groups": 2}, 5),
                                        ({"groups": 4, "zones": 4}, 4),
                                        ({"groups": 4, "zones": 4}, 6),
                                        ({"groups": 3, "zones": 6}, 3),
                                        ({"groups": 2, "zones": 3, "zone_size": [3, 3]}, 3)])
def test_relay_painters_complete_other_group_and_zone_counts(tmp_path, world_kw, n):
    run = relay_exp("gm", n=n, **world_kw).run(seed=1, max_rounds=20, out=tmp_path)
    s = run.score
    w = ColoringGrid(**world_kw)
    zones, (zw, zh) = w.n_zones, (w.zone_w, w.zone_h)
    assert run.end_reason == "terminal" and s["coverage"] == 1.0
    assert (s["zones_completed"], s["crossed_correct"]) == (zones, zones * zw * zh)
    assert s["open_correct"] == 120 - zones * zw * zh
    assert s["zone_paint_refusals"] == 0 and s["wrong_paints"] == 0
    last = {name: run.metrics[name][-1][1] for name in run.metrics}
    assert last["coloring.relay_status"] == float(zones)
    assert [k for k in last if k.startswith("coloring.relay_status.")] == [
        f"coloring.relay_status.Z{i}" for i in range(zones)]
    Run.load(run.dir)


def test_crossed_zone_determinism_replay_and_fork(tmp_path):
    a = relay_exp("det").run(seed=4, max_rounds=6, out=tmp_path / "a")
    b = relay_exp("det").run(seed=4, max_rounds=6, out=tmp_path / "b")
    strip = ("seq", "ts", "run")
    assert logical(a, strip) == logical(b, strip) and a.score == b.score
    assert type(a).load(a.dir).replay()["metrics_checked"] > 0
    child = a.fork(at_round=2).run()
    tail = [e for e in logical(a, strip) if e["round"] > 2]
    assert [e for e in logical(child, strip) if e["round"] > 2 and e["type"] != "run_started"] == tail


def coloring(name, participants, board=None, n=16, world_kw=None, metrics=None):
    board = board or Board()
    return Experiment(
        name=name,
        world=ColoringGrid(**{"height": 8, "width": 8, "paints_per_round": 1, "oracle": True,
                              **(world_kw or {})}),
        participants=[participants] * n, medium=board,
        metrics=metrics if metrics is not None else GRID_METRICS + (CLAIM_METRICS if board.registry else []))


def values(run, name):
    return [v for _, v, _ in run.metrics[name]]


def test_row_major_painters_duplicate_work(tmp_path):
    exp = Experiment(name="rm", world=ColoringGrid(paints_per_round=1),
                     participants=[RowMajorPainter()] * 15, metrics=GRID_METRICS)
    run = exp.run(seed=0, max_rounds=5, out=tmp_path)  # seed 0: row 0 is open, everyone paints its next cell
    assert run.score["correct"] == 5 and run.score["duplicate_paints"] == 70
    assert values(run, "coloring.coverage") == [k / 120 for k in range(1, 6)]
    assert values(run, "coloring.parallel_efficiency") == [1 / 15] * 5
    assert values(run, "coloring.duplicate_paints")[-1] == 70.0


def test_painters_without_the_board_never_paint_hidden_zones(tmp_path):
    exp = Experiment(name="rm", world=ColoringGrid(paints_per_round=10),
                     participants=[RowMajorPainter(paints=10)] * 3,
                     metrics=GRID_METRICS + ZONE_METRICS)
    run = exp.run(seed=1, max_rounds=20, out=tmp_path)
    s = run.score
    assert s["open_correct"] == 90 and s["crossed_correct"] == 0 and s["zone_paint_refusals"] == 0
    assert values(run, "coloring.zone_paint_refusals")[-1] == 0.0


def test_metrics_equal_score(tmp_path):
    run = coloring("rnd", RandomPainter(), n=6, world_kw={"palette": 3}).run(
        seed=3, max_rounds=8, out=tmp_path)
    s = run.score
    assert values(run, "coloring.coverage")[-1] == s["coverage"]
    assert values(run, "coloring.wrong_paints")[-1] == s["wrong_paints"] > 0
    assert values(run, "coloring.duplicate_paints")[-1] == s["duplicate_paints"]
    assert run.metrics["coloring.wrong_paints"][-1][2] == s["paints"]
    Run.load(run.dir)


class ZoneTrespasser(Participant):
    """Paints the first cell of every zone each turn, whatever its group."""

    async def turn(self, view, tools):
        info = parse_zones("\n".join(p.text or "" for p in view.observation.parts))
        for z in info["zones"]:
            await tools.call("paint", {"x": z["x"], "y": z["y"], "color": "r"})
        await tools.call("end_turn", {})


def test_zone_paint_refusals_metric_counts_call_time_refusals(tmp_path):
    exp = Experiment(name="tres", world=ColoringGrid(), participants=[ZoneTrespasser()] * 3,
                     metrics=["coloring.zone_paint_refusals"])
    run = exp.run(seed=1, max_rounds=2, out=tmp_path)
    assert values(run, "coloring.zone_paint_refusals") == [6.0, 12.0]  # 2 of 3 zones per agent
    assert run.metrics["coloring.zone_paint_refusals"][-1][2] == 18
    assert run.score["zone_paint_refusals"] == 0  # refused at call time: never reached the world
    rets = [e for e in run.events if e["type"] == "tool_returned"
            and (e["result"].get("error") or "").startswith("zone_paint_refused")]
    assert len(rets) == 12
    Run.load(run.dir)


def test_advisory_counts_violations_enforced_rejects(tmp_path):
    adv = coloring("adv", QueuePainter(), Board(registry=True)).run(seed=0, max_rounds=20, out=tmp_path)
    enf = coloring("enf", QueuePainter(), Board(registry=True, claim_policy="enforced")).run(
        seed=0, max_rounds=20, out=tmp_path)
    for run in (adv, enf):
        assert run.end_reason == "terminal" and run.score["coverage"] == 1.0
    assert values(adv, "claims.violations")[-1] > 0
    assert adv.score["duplicate_paints"] == values(adv, "claims.violations")[-1]
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
    assert adv.score["correct"] >= 3


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


def test_render_state_lists_the_zones():
    w = world()
    st = w.render_state()
    assert st["board"] == w.board_id and [z["zone"] for z in st["zones"]] == ["Z0", "Z1", "Z2"]
    assert "type" not in st["zones"][0]
    assert st["zones"][0]["seen_by"] == "G0" and st["zones"][0]["painted_by"] == "G1"
    assert len(st["grid"]) == 8 and len(st["target"][0]) == 15


def test_fake_painter_llm_arm(tmp_path):
    from swarmlab.participants import LLMAgent
    from swarmlab.providers.fake import FakeProvider

    exp = Experiment(name="llm", world=ColoringGrid(height=4, width=4, oracle=True),
                     participants=[LLMAgent(model="fake:painter", max_tokens=64)] * 3,
                     metrics=GRID_METRICS, providers={"fake": FakeProvider()})
    run = exp.run(seed=0, max_rounds=4, out=tmp_path)
    acts = [e for e in run.events if e["type"] == "action_committed"]
    assert len(acts) == 12 and all(e["action"]["name"] == "paint" for e in acts)
    assert all(e["accepted"] for e in acts)  # it never paints another group's zone
    assert run.score["correct"] >= 4 and run.score["wrong_paints"] == 0


def test_example_specs_load_and_run_scripted_arms(tmp_path):
    arms = Experiment.arms_from_yaml(REPO / "examples" / "coloring_s0.yaml")
    assert {"row_major", "queue_advisory", "queue_enforced"} <= set(arms)
    assert arms["queue_enforced"].medium.claim_policy.type_name() == "enforced"
    run = arms["queue_enforced"].run(seed=0, out=tmp_path)
    assert run.score["coverage"] == 1.0
    zones = Experiment.arms_from_yaml(REPO / "examples" / "crossed_zones.yaml")
    assert set(zones) == {"zones"}
    run = zones["zones"].run(seed=3, out=tmp_path)
    assert re.fullmatch(r"[0-9a-f]{10}", run.score["board"]) and run.end_reason == "terminal"


def test_report_shows_zone_scores_and_relay_table(tmp_path):
    from swarmlab.report import build_report

    out = tmp_path / "runs"
    Experiment.from_yaml(REPO / "examples" / "crossed_zones.yaml", "zones").run_all(
        [1, 2], out=out)
    relay_exp("crossed-zones-silent", post_=False, n=4, groups=4, zones=4).run_all(
        [1], max_rounds=3, out=out)
    text_ = build_report(out, include_fake=True)
    final = text_.split("## Final score")[1]
    head = next(ln for ln in final.splitlines() if ln.startswith("| run "))
    for key in ("board", "zones_completed", "crossed_correct", "open_correct", "zone_paint_refusals"):
        assert f"| {key} |" in head or head.endswith(f"| {key} |"), key
    assert "## Crossed zones (relay status at the end of each run)" in text_
    # one column per zone id over the runs: the 3-zone runs show n/a under Z3
    assert "| arm | run | board | Z0 | Z1 | Z2 | Z3 | zones done |" in text_
    assert re.search(r"\| zones \| crossed-zones__zones__s1 \| [0-9a-f]{10} \| complete \| complete \| "
                     r"complete \| n/a \| 3 \| 1 / 1 / 1 \|", text_)
    assert ("viewer_never_posted | viewer_never_posted | viewer_never_posted | viewer_never_posted "
            "| 0 | 0 / 0 / 0 / 0 |") in text_
    assert "| coloring.relay_status.Z0 |" in text_


# ---- paint limit at call time, status, cap notes (paint-limit change) -------------------------------

def coloring_executor(tmp_path, commit="round_end", **kw):
    from swarmlab.blobs import BlobStore
    from swarmlab.executor import RoundExecutor

    w = world(**{"paints_per_round": 1, **kw})
    board = Board()
    board.commit_mode = commit
    ex = RoundExecutor(run="r", round=1, world=w, board=board, blobs=BlobStore(tmp_path / "blobs"),
                       agents=[A, B, C], commit=commit, topology_rng=lambda: derive(0, "topology", 1))
    return w, ex


def cell(w, i):
    y, x = divmod(i, w.width)  # i < 15: row 0, all open on seed 0's board (`world()`)
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
    bad = await ex.call(C, "paint", {"x": 99, "y": 0, "color": "r"})
    assert bad.error.startswith("out_of_bounds: ")
    assert (await ex.call(C, "paint", {"x": 0, "y": 0, "color": "z"})).error.startswith("bad_color: ")
    x, y = zone_cells(zone(w, "Z0"))[0]
    refused = await ex.call(C, "paint", {"x": x, "y": y, "color": "r"})
    assert refused.error.startswith("zone_paint_refused: ") and not ex.buffered_actions(C)


async def test_round_end_limit_counts_pending_paints_up_to_paints_per_round(tmp_path):
    w, ex = coloring_executor(tmp_path, paints_per_round=2, status_tools=["my_status"])
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
    w, ex = coloring_executor(tmp_path, commit="immediate", status_tools=["my_status"])
    first = await ex.call(A, "paint", cell(w, 0))
    assert first.ok and not first.pending and first.result["accepted"]
    assert (await ex.call(A, "my_status", {})).result["paints_left_this_round"] == 0
    second = await ex.call(A, "paint", cell(w, 1))  # the world's commit rejects it, as before
    assert second.ok and not second.pending and not second.result["accepted"]
    assert second.result["feedback"] == {"error": "paints_per_round", "detail": "at most 1 paint per round"}


def test_validate_without_pending_keeps_old_behaviour():
    w = world(paints_per_round=1)
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

    w = OldStyle(paints_per_round=1, status_tools=["my_status"])
    w.reset(derive(0, "world"), [A, B, C])
    w.begin_round(1)
    ex = RoundExecutor(run="r", round=1, world=w, board=Board(), blobs=BlobStore(tmp_path / "b"),
                       agents=[A, B, C])
    assert (await ex.call(A, "paint", cell(w, 0))).ok
    assert (await ex.call(A, "paint", cell(w, 1))).ok  # no pre-check: buffered, rejected at commit
    assert (await ex.call(A, "my_status", {})).result["paints_left_this_round"] == 1


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


class Idle(Participant):
    async def turn(self, view, tools):
        await tools.call("end_turn", {})


def test_cap_note_marks_spinning_turns_and_report_counts_rejected_calls(tmp_path):
    from swarmlab.report import build_report

    out = tmp_path / "runs"
    exp = Experiment(name="cap", world=ColoringGrid(paints_per_round=1, status_tools=["my_status"]),
                     participants=[Spinner(), Busy(), Idle()], metrics=GRID_METRICS)
    run = exp.run(seed=0, max_rounds=2, out=out, max_calls_per_turn=5)
    ended = {e["agent"]: e for e in run.events if e["type"] == "turn_ended" and e["round"] == 1}
    spinner, busy = ended[agent_id(0)], ended[agent_id(1)]
    assert spinner["yield_kind"] == "cap" and spinner["usage"]["notes"] == ["cap:4"]
    assert busy["yield_kind"] == "cap" and "notes" not in busy["usage"]
    Run.load(run.dir)
    text_ = build_report(out, "cap")
    assert "| rejected tool calls |" in text_
    assert "| 8/22 |" in text_  # spinner 4 of 5 refused, busy 0 of 5, idle 0 of 1, per round
