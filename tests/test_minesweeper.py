"""Minesweeper world, its metrics and the fake scripts `minesweeper_random` / `minesweeper_allocator`."""
from __future__ import annotations

import json
import random
from pathlib import Path
from types import SimpleNamespace

import pytest

from swarmlab import Experiment, Run
from swarmlab.participants import LLMAgent
from swarmlab.providers.fake import resolve_script
from swarmlab.registry import resolve
from swarmlab.report import build_report
from swarmlab.rng import derive
from swarmlab.world.base import Action
from swarmlab.world.minesweeper import (
    EXPLODED,
    FLAG,
    HIDDEN,
    AllocationEfficiency,
    Cleared,
    Collisions,
    MineHits,
    Minesweeper,
    ProductiveRate,
    column_labels,
    deduce,
    fake_allocator,
    fake_random,
    parse_observation,
)

AGENTS = [f"a{i:03d}" for i in range(4)]
REPO = Path(__file__).resolve().parent.parent
SPEC = REPO / "examples" / "minesweeper.yaml"
METRIC_NAMES = ["minesweeper.cleared", "minesweeper.collisions", "minesweeper.productive_rate",
                "minesweeper.allocation_efficiency", "minesweeper.mine_hits"]
METRIC_CLASSES = (Cleared, Collisions, ProductiveRate, AllocationEfficiency, MineHits)
PRESETS = [(9, 9, 10), (16, 16, 40), (16, 30, 99)]   # beginner, intermediate, expert (h, w, mines)


def make(seed=0, round=1, agents=AGENTS, **kw):
    w = Minesweeper(**kw)
    w.reset(derive(seed, "world"), list(agents))
    w.begin_round(round)
    return w


def rows(w):
    return ["".join(r) for r in w.board]


def cells(w, mine, state=HIDDEN):
    return [(x, y) for y in range(w.height) for x in range(w.width)
            if w.minefield[y][x] == mine and w.board[y][x] == state]


def next_round(w):
    data = w.end_round(w.round)
    w.begin_round(w.round + 1)
    return data


# ---- construction and reset ------------------------------------------------------------------

@pytest.mark.parametrize("kw", [{"height": 0}, {"width": 100}, {"height": 2.5}, {"mines": 0},
                                {"height": 3, "width": 3, "mines": 9}, {"on_mine": "explode"},
                                {"reveals_per_round": 0}, {"height": True}])
def test_constructor_rejects_bad_params(kw):
    with pytest.raises(ValueError):
        Minesweeper(**kw)


def test_reset_is_seeded_and_opens_a_safe_flood_filled_start():
    a, b, c = make(seed=3), make(seed=3), make(seed=4)
    assert a.minefield == b.minefield and a.board == b.board
    assert a.minefield != c.minefield
    assert sum(map(sum, a.minefield)) == 10
    opened = [(x, y) for y in range(9) for x in range(9) if a.board[y][x] != HIDDEN]
    assert opened and a.initial_opened == len(opened)
    assert not any(a.minefield[y][x] for x, y in opened)
    for x, y in opened:                       # a flood fill: every 0 has all its neighbours open
        if a.board[y][x] == ".":
            assert all(a.board[ny][nx] != HIDDEN for nx, ny in a._nbrs(x, y))
    assert make(seed=3, safe_opening=False).minefield == a.minefield   # the rng is not shifted
    assert all(c == HIDDEN for r in make(seed=3, safe_opening=False).board for c in r)


@pytest.mark.parametrize("h,w,m", PRESETS)
def test_difficulty_presets_reset_observe_and_parse(h, w, m):
    world = make(seed=1, height=h, width=w, mines=m)
    assert (len(world.minefield), len(world.minefield[0]), sum(map(sum, world.minefield))) == (h, w, m)
    text = world.observe("a000").parts[0].text
    parsed = parse_observation(text)
    assert parsed["rows"] == rows(world)
    assert parsed["mines"] == m and parsed["flags"] == 0 and parsed["exploded"] == 0
    assert parsed["hidden"] == sum(r.count(HIDDEN) for r in rows(world))
    assert text.startswith(f"Board ({w} columns x {h} rows; x = column 0-{w - 1}, y = row 0-{h - 1};")
    assert all(ln in text for ln in column_labels(w))
    assert world.verify()["safe_cells"] == h * w - m


def test_column_labels():
    assert column_labels(9) == ["012345678"]
    assert column_labels(12) == ["          11", "012345678901"]


# ---- observation, task text and the information boundary --------------------------------------

def test_observation_is_public_and_shows_no_mines():
    w = make(seed=2)
    texts = {w.observe(a).parts[0].text for a in AGENTS}
    assert len(texts) == 1                                  # identical for every agent
    obs = w.observe("a000")
    assert obs.private == {} and len(obs.parts) == 1
    assert set("".join(parse_observation(obs.parts[0].text)["rows"])) <= set("#F.*12345678")
    assert parse_observation("prefix\n" + obs.parts[0].text + "\nsuffix")["rows"] == rows(w)
    assert parse_observation("no board here") == {}


def test_parse_reads_rows_made_only_of_digits():
    w = make(seed=2)
    w.board[3] = list("123211111")
    assert parse_observation(w.observe("a000").parts[0].text)["rows"][3] == "123211111"


def test_description_states_the_task_only():
    team = make().description()
    assert "one of 4 agents (a000, a001, a002, a003)" in team
    assert "9 columns and 9 rows and hides 10 mines" in team
    assert "at most 1 cell per round; flagging and unflagging are free" in team
    assert "Revealing a mine ends the game" in team and "The team wins" in team
    for word in ("same time", "end of the round", "order", "post", "message", "board proves"):
        assert word not in team, word                       # how rounds work is the spec's business
    solo = make(agents=["a000"]).description()
    assert "You are playing a game of Minesweeper" in solo and "You win" in solo
    assert "You may reveal at most 1 cell" in solo and "visible to every agent" not in solo
    alt = make(on_mine="continue", free_flags=False, reveals_per_round=2).description()
    assert "the cell shows * and the game goes on" in alt
    assert "at most 2 actions per round (reveal, flag or unflag)" in alt


# ---- actions ---------------------------------------------------------------------------------

def test_reveal_opens_floods_and_wins():
    w = make(seed=5, agents=["a000"], reveals_per_round=1000)
    zero = next((x, y) for x, y in cells(w, False) if w.numbers[y][x] == 0)
    out = w.reveal("a000", x=zero[0], y=zero[1])
    assert out.accepted and out.feedback["result"] == "opened" and out.feedback["cells_opened"] > 1
    assert set(out.feedback) == {"result", "x", "y", "value", "cells_opened"}
    for x, y in cells(w, False):
        w.reveal("a000", x=x, y=y)
    assert w.won() and w.terminal() and w.cleared_round == 1
    sc = w.score()
    assert sc["won"] == 1 and sc["cleared"] == 1.0 and sc["mine_hits"] == 0


def test_mine_ends_the_game_by_default():
    w = make(seed=1)
    mx, my = cells(w, True)[0]
    sx, sy = cells(w, False)[0]
    out = w.reveal("a000", x=mx, y=my)
    assert out.accepted and out.feedback == {"result": "exploded", "x": mx, "y": my}
    assert w.lost and w.terminal() and w.board[my][mx] == EXPLODED
    late = w.reveal("a001", x=sx, y=sy)                     # later in the same commit batch
    assert not late.accepted and late.feedback["error"] == "game_over"
    assert w.score()["lost"] == 1 and w.score()["first_mine_round"] == 1


def test_mine_is_a_penalty_with_on_mine_continue():
    w = make(seed=1, on_mine="continue")
    mx, my = cells(w, True)[0]
    assert w.reveal("a000", x=mx, y=my).feedback["result"] == "exploded"
    assert not w.lost and not w.terminal() and w.first_mine_round == 1
    sx, sy = cells(w, False)[0]
    assert w.reveal("a001", x=sx, y=sy).feedback["result"] == "opened"
    assert w.observe("a000").parts[0].text.count(EXPLODED) == 2   # legend + the cell
    assert parse_observation(w.observe("a000").parts[0].text)["exploded"] == 1


def test_flags_are_shared_unchecked_and_block_reveals():
    w = make(seed=1)
    sx, sy = cells(w, False)[0]
    out = w.flag("a000", x=sx, y=sy)                         # a wrong flag: feedback cannot tell
    assert out.accepted and out.feedback == {"result": "flagged", "x": sx, "y": sy}
    assert w.board[sy][sx] == FLAG and "F" in w.observe("a003").parts[0].text
    blocked = w.reveal("a001", x=sx, y=sy)
    assert not blocked.accepted and blocked.feedback["error"] == "flagged"
    assert w.round_used.get("a001", 0) == 0                 # a refused reveal keeps the budget
    assert w.flag("a002", x=sx, y=sy).feedback["error"] == "not_hidden"
    assert w.unflag("a002", x=sx, y=sy).accepted            # anyone may remove a flag
    assert w.unflag("a002", x=sx, y=sy).feedback["error"] == "not_flagged"
    w.flag("a000", x=sx, y=sy)
    mx, my = cells(w, True)[0]
    w.flag("a000", x=mx, y=my)
    sc = w.score()
    assert (sc["correct_flags"], sc["wrong_flags"], sc["flags"], sc["unflags"]) == (1, 1, 3, 1)


def test_reveal_limit_at_validate_and_commit():
    w = make(seed=1)
    (ax, ay), (bx, by) = cells(w, False)[:2]
    first = Action(name="reveal", args={"x": ax, "y": ay})
    assert w.validate("a000", first).ok
    ack = w.validate("a000", Action(name="reveal", args={"x": bx, "y": by}), [first])
    assert not ack.ok and ack.error.startswith("reveals_per_round")
    assert w.validate("a000", Action(name="flag", args={"x": bx, "y": by}), [first]).ok  # free
    assert w.validate("a000", Action(name="reveal", args={"x": 9, "y": 0})).error.startswith("out_of_bounds")
    assert not w.validate("a000", Action(name="reveal", args={"x": 1})).ok
    assert w.reveal("a000", x=ax, y=ay).accepted
    late = w.reveal("a000", x=bx, y=by)
    assert not late.accepted and late.feedback["error"] == "reveals_per_round"
    next_round(w)
    assert w.reveal("a000", x=bx, y=by).accepted            # the budget is per round
    strict = make(seed=1, free_flags=False)
    assert strict.flag("a000", x=ax, y=ay).accepted
    assert strict.flag("a000", x=bx, y=by).feedback["error"] == "reveals_per_round"


def test_collision_in_the_same_round_and_stale_reveal_later():
    w = make(seed=1)
    x, y = cells(w, False)[0]
    assert w.reveal("a000", x=x, y=y).feedback["result"] == "opened"
    again = w.reveal("a001", x=x, y=y)
    assert again.accepted and again.feedback == {"result": "already_revealed", "x": x, "y": y,
                                                 "same_round": True}
    data = next_round(w)
    assert data["collisions"] == 1 and data["stale"] == 0
    assert w.reveal("a002", x=x, y=y).feedback["same_round"] is False
    assert w.end_round(2)["stale"] == 1
    assert w.claim_key("a000", Action(name="reveal", args={"x": 3, "y": 4})) == "cell:3,4"


# ---- deduction and accounting -----------------------------------------------------------------

def test_deduce_single_clue_and_subset_rules():
    assert deduce(["#1", "11"]) == (set(), {(0, 0)})
    assert deduce(["##", ".."]) == ({(0, 0), (1, 0)}, set())
    # 1-2-1: only the subset rule decides it (the corners are mines, the middle is safe)
    assert deduce(["###", "121", "..."]) == ({(1, 0)}, {(0, 0), (2, 0)})
    assert deduce(["F1"]) == (set(), {(0, 0)})                # a flag is not trusted: deduced ...
    assert deduce(["F."]) == ({(0, 0)}, set())                # ... and a wrong one is proven safe
    assert deduce(["*#", "11"]) == ({(1, 0)}, set())          # ... an exploded mine is


def test_deduce_is_sound_on_played_boards():
    for seed in range(8):
        for h, w_, m in PRESETS:
            w = make(seed=seed, height=h, width=w_, mines=m, on_mine="continue",
                     reveals_per_round=10_000)
            rng = random.Random(seed)
            for _ in range(15):
                safe, mines = deduce(w.board)
                assert all(not w.minefield[y][x] for x, y in safe)
                assert all(w.minefield[y][x] for x, y in mines)
                hidden = cells(w, False) + cells(w, True)
                if not hidden or w.won():
                    break
                x, y = min(safe) if safe else rng.choice(hidden)
                w.reveal("a000", x=x, y=y)


def test_mine_hits_are_classified_against_the_round_start_board():
    w = make(seed=1, on_mine="continue")
    m1, m2, m3 = cells(w, True)[:3]
    w.round_mines, w.round_safe = [m1], [cells(w, False)[0]]
    w.reveal("a000", x=m1[0], y=m1[1])                       # proven a mine: a reasoning error
    w.reveal("a001", x=m2[0], y=m2[1])                       # safe cells were available
    w.round_safe = []
    w.reveal("a002", x=m3[0], y=m3[1])                       # nothing was provably safe
    data = w.end_round(1)
    assert (data["provable_mine_hits"], data["unforced_mine_hits"], data["forced_mine_hits"]) == (1, 1, 1)
    assert data["mine_hits"] == 3 and data["ideal_reveals"] == 1
    sc = w.score()
    assert (sc["provable_mine_hits"], sc["unforced_mine_hits"], sc["forced_mine_hits"]) == (1, 1, 1)


def test_end_round_tally_and_metrics_fold_equal_score():
    w = make(seed=6, on_mine="continue")
    metrics = [cls() for cls in METRIC_CLASSES]
    rng = random.Random(0)
    for r in range(1, 40):
        assert w.round == r
        for a in AGENTS:
            choice = w.round_safe or cells(w, False) + cells(w, True)
            if not choice:
                break
            x, y = rng.choice(choice)
            w.reveal(a, x=x, y=y)
        data = w.end_round(r)
        assert data["ideal_reveals"] == (min(4, data["provably_safe"]) or 1)
        assert data["reveals"] == data["productive"] + data["collisions"] + data["stale"] + data["mine_hits"]
        assert data["productive"] == data["certain"] + data["guesses"]
        ev = SimpleNamespace(type="round_resolved", data=data)
        for m in metrics:
            m.update(ev)
        sc = w.score()
        vals = {m.name: m.value() for m in metrics}
        assert vals["minesweeper.cleared"] == (sc["cleared"], w.safe_cells())
        assert vals["minesweeper.collisions"] == (float(sc["collisions"]), sc["reveals"])
        assert vals["minesweeper.mine_hits"] == (float(sc["mine_hits"]), sc["reveals"])
        assert vals["minesweeper.productive_rate"] == (sc["productive_reveals"] / (4 * r), 4 * r)
        assert vals["minesweeper.allocation_efficiency"] == (sc["allocation_efficiency"], sc["ideal_reveals"])
        if w.terminal():
            break
        w.begin_round(r + 1)
    assert w.won() and w.score()["cleared_round"] == r


def test_snapshot_restore_round_trip_and_size_check():
    w = make(seed=2)
    x, y = cells(w, False)[0]
    w.reveal("a000", x=x, y=y)
    w.flag("a001", *cells(w, True)[0])
    next_round(w)
    blob = w.snapshot()
    fresh = Minesweeper()
    fresh.restore(blob)
    assert fresh.board == w.board and fresh.minefield == w.minefield and fresh.score() == w.score()
    assert fresh.observe("a000") == w.observe("a000")
    for other in (Minesweeper(height=10), Minesweeper(mines=11)):
        with pytest.raises(ValueError, match="cannot change across a fork or resume"):
            other.restore(blob)


def test_render_state_shows_hidden_mines_to_the_evaluator_only():
    w = make(seed=2)
    st = w.render_state()
    assert sum(r.count("M") for r in st["grid"]) == 10
    assert set(st["palette"]) >= {HIDDEN, FLAG, EXPLODED, ".", "M", "1", "8"}
    assert st["provably safe at round start"] == len(w.round_safe)


# ---- plumbing: entry points, fake scripts, runs, report, forks ---------------------------------

def test_entry_points_resolve():
    assert resolve("minesweeper", "swarmlab.worlds") is Minesweeper
    for name, cls in zip(METRIC_NAMES, METRIC_CLASSES):
        assert resolve(name, "swarmlab.metrics") is cls
        assert cls.type_name() == name == cls.name
    assert resolve_script("minesweeper_random") is fake_random
    assert resolve_script("minesweeper_allocator") is fake_allocator
    assert Minesweeper(height=16, width=30, mines=99).spec() == {"type": "minesweeper", "params": {
        "height": 16, "width": 30, "mines": 99, "on_mine": "end", "reveals_per_round": 1,
        "free_flags": True, "safe_opening": True}}


def _events(run):
    return [json.loads(line) for line in (run.dir / "events.jsonl").read_text().splitlines()]


@pytest.mark.parametrize("arm", ["random", "allocator"])
def test_example_spec_dry_run_end_to_end(tmp_path, arm):
    run = Experiment.from_yaml(SPEC, arm).run(0, out=tmp_path)
    assert run.meta["status"] == "ended" and run.meta["end_reason"] == "terminal"
    sc = run.meta["score"]
    assert sc["won"] == 1 and sc["cleared"] == 1.0 and sc["cleared_round"] == run.meta["last_round"]
    events = _events(run)
    resolved = [e for e in events if e["type"] == "round_resolved"]
    assert [e["round"] for e in resolved] == list(range(1, run.meta["last_round"] + 1))
    for r in range(1, run.meta["last_round"] + 1):          # resolved after the round's commits
        types = [e["type"] for e in events if e["round"] == r]
        assert "action_committed" not in types[types.index("round_resolved"):]
    met = {}
    for e in events:
        if e["type"] == "metric":
            met.setdefault(e["name"], []).append(e["value"])
    assert met["minesweeper.cleared"][-1] == 1.0
    assert met["minesweeper.collisions"][-1] == sc["collisions"]
    assert met["minesweeper.mine_hits"][-1] == sc["mine_hits"]
    assert met["minesweeper.allocation_efficiency"][-1] == pytest.approx(sc["allocation_efficiency"])
    posts = [e for e in events if e["type"] == "post"]
    assert bool(posts) == (arm == "allocator")
    for e in events:                                        # agents never see evaluator state
        if e["type"] in ("tool_returned", "action_committed", "turn_started"):
            body = json.dumps(e.get("result") or e.get("feedback") or e)
            for k in ("minefield", "round_safe", "provable", "ideal", "allocation_efficiency"):
                assert k not in body, (k, e["type"])
    assert Run.load(run.dir).replay()["score"] == run.score


def test_allocator_collides_less_than_random_on_the_same_boards(tmp_path):
    totals = {}
    for arm in ("random", "allocator"):
        exp = Experiment.from_yaml(SPEC, arm)
        runs = [exp.run(seed, out=tmp_path / arm) for seed in range(4)]
        totals[arm] = sum(r.score["collisions"] for r in runs)
    assert totals["allocator"] < totals["random"]


def api_experiment(**world):
    return Experiment(name="ms-api", world=Minesweeper(**world),
                      participants=[LLMAgent(model="fake:minesweeper_random", max_tokens=256)] * 4,
                      metrics=METRIC_NAMES)


def test_python_api_run_with_the_standard_rule_replays(tmp_path):
    exp = api_experiment(height=16, width=16, mines=40)
    run = exp.run(seed=1, max_rounds=80, out=tmp_path)
    sc = run.score
    assert run.meta["end_reason"] == "terminal" and (sc["won"] or sc["lost"])
    if sc["lost"]:
        assert sc["first_mine_round"] == run.meta["last_round"] and sc["cleared_round"] is None
    assert Run.load(run.dir).replay()["score"] == sc


def test_report_section(tmp_path):
    runs = [Experiment.from_yaml(SPEC, arm).run(0, out=tmp_path / "out") for arm in ("random", "allocator")]
    text = build_report(tmp_path / "out", include_fake=True)
    assert text.index("## Final score") < text.index("## Minesweeper: outcome, mines and allocation")
    section = text.split("## Minesweeper: outcome, mines and allocation")[1].split("\n## ")[0]
    lines = [ln for ln in section.splitlines() if ln.startswith("| ")]
    assert lines[0] == ("| arm | run | outcome | cleared | cleared_round | first_mine_round | mine_hits "
                        "| provable/unforced/forced | collisions | allocation_efficiency |")
    assert len(lines) == 3
    for run in runs:
        sc = run.score
        row = next(ln for ln in lines if f"| {run.id} |" in ln)
        assert f"| won | 1.000 | {sc['cleared_round']} |" in row
        assert row.endswith(f"| {sc['collisions']} | {sc['allocation_efficiency']:.3f} |")


def test_fork_with_changed_rule_continues_and_changed_size_is_refused(tmp_path):
    exp = api_experiment(on_mine="continue")
    parent = exp.run(seed=2, max_rounds=3, out=tmp_path / "p")

    def edited(world):
        return Experiment(name=exp.name, world=world, participants=list(exp.participants),
                          metrics=METRIC_NAMES)

    child = parent.fork(1, experiment=edited(Minesweeper(on_mine="end", reveals_per_round=2))).run(
        out=tmp_path / "f", max_rounds=4)
    assert child.meta["restored"]["world"] is True and child.meta["status"] == "ended"
    with pytest.raises(ValueError, match="cannot change across a fork or resume"):
        parent.fork(1, experiment=edited(Minesweeper(height=12, width=12, mines=22))).run(
            out=tmp_path / "g")
