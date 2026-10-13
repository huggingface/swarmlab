"""ColoringGrid with crossed zones (DESIGN.md component 4; M3b, crossed zones).

A shared world: agents paint cells with `paint(x, y, color)` until the current grid equals a
target grid. The grid holds three **crossed zones**: each zone's target colours are shown to one
group of agents and the zone may be painted only by a different group, so the targets have to be
relayed over the message board. Cells outside the zones ("open" cells) are visible to and
paintable by everyone. There is no switch to turn the zones off: a coloring grid without them
never restricts visibility or painting and so cannot test communication. `oracle: true` is the
control condition in which a zone's painting group also sees its targets.

Grid, colours, boards
---------------------
`height` rows by `width` columns; `x` is the column (0 .. width-1), `y` the row (0 .. height-1).
Colours are the first `palette` letters of `COLOURS` ("rgbykwopcmnt", as in FlagGame), so
`2 <= palette <= 12`; an unpainted cell is ".". The defaults (8 x 15, palette 4: r g b y) select
the five fixed boards in `swarmlab/world/data/coloring_boards.json` (ids "00".."04", the boards of
the first crossed-zone experiments; colour names map to letters red=r, green=g, blue=b, yellow=y).

- `board: "auto"` (default): on an 8 x 15 grid with palette 4, fixed board `(seed - 1) % 5` of
  the run seed (seeds 1..5 -> boards 00..04). The world sees the run seed through the rng it is
  reset with: `swarmlab.rng.derive(seed, "world")` records `root = (seed, "world")` on the stream,
  and `reset` reads it. A stream without `root` (a plain `random.Random`) picks the board with
  `rng.randrange(5)`. On any other size or palette, "auto" generates the board (as "random").
- `board: "00".."04"`: that fixed board (needs 8 x 15 and palette 4, else `ValueError`).
- `board: "random"`: the target is drawn cell by cell in row-major order from the world rng (as the
  coloring world always did), then the zones are placed from the same stream: three
  non-overlapping `zone_width x zone_height` rectangles fully inside the grid (rejection
  sampling, then a shuffled aligned lattice when the grid is tight). The zone size is 5 x 2, or on
  grids too small for three of those the largest `w x h` with `w <= 5`, `h <= 2` such that three
  fit on an aligned lattice (`(width // w) * (height // h) >= 3`); a grid of fewer than 3 cells is
  a `ValueError`.

Zones and groups
----------------
Agent i (reset's agent order a000, a001, ...) is in group `G(i % 3)`; at least 3 agents are
required (`requires.n_agents`; `reset` also raises), groups differ in size by at most one. Zones
are `Z0` (type A: targets shown to G0, painted by G1), `Z1` (type B: G1 -> G2) and `Z2` (type C:
G2 -> G0); the fixed boards encode the same cycle (checked at load). Each group therefore sees one
zone's targets and is the only group allowed to paint another zone.

Observation format (exact; `parse_observation` and `parse_zones` invert it)
----------------------------------------------------------------------------
One text part:

    Target grid (8 rows x 15 columns; x = column 0-14, y = row 0-7; "?" = hidden from you):
    <row 0 of the target, "?" for every cell of a zone whose targets this agent cannot see>
    ...
    <blank>
    Current grid ("." = unpainted):
    <row 0 of the current grid (always fully visible)>
    ...
    <blank>
    Colours: r g b y
    Colour names: r = red, g = green, b = blue, y = yellow
    <blank>
    Groups: G0 = a000; G1 = a001; G2 = a002
    Zones (5 columns x 2 rows each; cells outside the zones are open: anyone may paint them):
    Z0 type A at x=5..9, y=4..5: targets visible to G0, painted by G1
    Z1 type B at x=0..4, y=1..2: targets visible to G1, painted by G2
    Z2 type C at x=5..9, y=6..7: targets visible to G2, painted by G0
    You are a000, in group G0. <your duties: the zone you see and report, the zone you paint>

With `oracle: true` the zone lines read `targets visible to G0 and G1 (its painters)` and the
painting group's target rows show the zone. `observation.private` is `{"group": <0..2>}`.

Actions and commit
------------------
- `paint(x, y, color)`: `validate` rejects out-of-bounds cells (`error="out_of_bounds: ..."`),
  unknown colours (`"bad_color: ..."`) and, for a well-formed cell inside a zone whose painting
  group is not the agent's, `"zone_paint_refused: cell (x, y) is in zone Z0, which only group G1
  may paint"` (the message names the zone and its painting group only, never a colour or whether
  the paint would be right). It also refuses at call time a paint the commit would reject for the
  limit: when the agent's buffered `pending` paints reach `paints_per_round` (default 10), the
  error is `"paints_per_round: at most N paints per round; this one would be rejected at commit"`
  and the executor does not buffer it. Under an `enforced` claim policy a buffered paint may later
  be rejected `not_claimed` without reaching the world; the pre-check counts every buffered paint
  all the same.
- Commit (`paint` again checks everything): a paint is refused with feedback `error` =
  `out_of_bounds`, `bad_color`, `zone_paint_refused` (counted in `zone_paint_refusals`; a zone
  refusal does not use up the agent's per-round paints) or `paints_per_round` (with a human
  `detail`). At most `paints_per_round` paints per agent are applied per round, in commit order.
- Conflict rule: paints are applied in commit (seeded) order, so the last writer of a cell wins.
  An accepted paint's feedback is `{"result": "painted"}`, or `{"result": "overwritten"}` when a
  later paint in the same commit batch painted the same cell.
- Paint accounting (evaluator-side, in commit order): a paint is `wrong` when its colour is not
  the target's, a `duplicate` when the cell already matched the target and it paints the target
  colour again, `useful` when it turns a non-matching cell into a matching one; `wasted` is every
  applied paint that is not useful.
- `claim_key(agent, paint) -> "cell:<x>,<y>"` for the registry and claim policies (advisory /
  enforced work unchanged on top of the zones; a zone refusal happens in the world, after the
  claim check).

Run settings (`requires`): `commit: round_end` (the task text says posts are delivered when the
round ends, readable from the next round on) and at least 3 agents.

Dynamics
--------
- `target_changes: [[round, [[x, y, color], ...]], ...]` changes the target at the start of that
  round (`begin_round`, before anyone observes), so agents see the new target in that round
  (masked like the rest of the target).
- `worker_failures: [[round, [agent, ...]], ...]`: `killed_at(round)` returns those agents and
  the runner removes them from the live set at the start of that round, logging an
  `intervention` event `intervention="worker_failures", op="kill"` per agent.

Status tools (off by default; `status_tools` lists the ones to offer). The original versions
leaked hidden targets (`cells_mine` and `coverage` count matches against cells the agent cannot
see), so both are leak-free now: `my_status` -> `{"paints", "paints_left_this_round"}` (applied
paints; `paints_per_round` minus this round's applied paints (immediate) and buffered paints
(round_end), with `pending=` passed by the executor as for `validate`); `collective_status` ->
`{"cells_painted", "cells"}` (what the visible current grid already shows).

Evaluator side
--------------
`terminal()` is true when coverage is 1.0 and no target change is scheduled after the current
round. `score()` -> coverage, correct (matching cells), wrong (painted, not matching), unpainted,
paints, useful_paints, duplicate_paints, wrong_paints, overwritten_paints, wasted_paints, plus
board (board id or "random"), zones_completed (zones whose cells all match, 0..3),
crossed_correct (matching zone cells, of 3 x zone size: 30 on the fixed boards), open_correct
(matching open cells: 90) and zone_paint_refusals (commit-time zone refusals; call-time refusals
never reach the world, `validate` has no side effects, so they are counted by the metric
`coloring.zone_paint_refusals` from the log). Per-zone relay diagnostics need the board's post,
delivery and read events, which the world never sees: they are the metric `coloring.relay_status`
(swarmlab/metrics/coloring.py) and this world's `report_sections` table.
`verify()` -> target (as of now), initial_target, target_changes, colours, height, width,
paints_per_round, board, crossed_zones (list of zone dicts), groups (agent -> group), oracle.

`render_state()` (the replay page's world panel) -> `grid` (current), `target`, `palette`, the
zones as a table, coverage, cells_matching, cells, round and paint counts.

Snapshots carry game state only (grid, target, zones, groups, painters, counters, round);
constructor config is kept on restore, as in FlagGame.
"""
from __future__ import annotations

import json
import random
import re
from collections.abc import Sequence
from functools import cache
from importlib.resources import files
from typing import Any, ClassVar

from ..ids import ActionId, AgentId
from ..view import Observation, text_observation
from .base import Ack, Action, Outcome, Requirements, World, tool

COLOURS = "rgbykwopcmnt"
COLOUR_NAMES = {"r": "red", "g": "green", "b": "blue", "y": "yellow", "k": "black", "w": "white",
                "o": "orange", "p": "purple", "c": "cyan", "m": "magenta", "n": "brown", "t": "teal"}
NAME_TO_LETTER = {v: k for k, v in COLOUR_NAMES.items()}
PALETTE_CSS = {"r": "#d62728", "g": "#2ca02c", "b": "#1f77b4", "y": "#f2c80f", "k": "#111111",
               "w": "#ffffff", "o": "#ff7f0e", "p": "#7b4fa0", "c": "#17becf", "m": "#e377c2",
               "n": "#8c564b", "t": "#0f766e"}  # the replay page's colours for COLOURS
UNPAINTED = "."
HIDDEN = "?"
TARGET_HEADER = "Target grid"
CURRENT_HEADER = 'Current grid ("." = unpainted):'
COLOURS_PREFIX = "Colours:"
N_GROUPS = 3
ZONE_IDS = ("Z0", "Z1", "Z2")
ZONE_TYPES = ("A", "B", "C")
ZONE_W, ZONE_H = 5, 2
FIXED_SIZE = (8, 15)  # (height, width) of the fixed boards
FIXED_PALETTE = 4
BOARD_IDS = ("00", "01", "02", "03", "04")
RELAY_STATUSES = ("complete", "viewer_never_posted", "viewer_posted_wrong_or_unparsed",
                  "painter_never_read_correct_report", "painter_read_but_painted_wrong")


def zone_cycle(i: int) -> tuple[int, int]:
    """(viewer group, painter group) of zone i: A G0->G1, B G1->G2, C G2->G0."""
    return i % N_GROUPS, (i + 1) % N_GROUPS


@cache
def fixed_boards() -> dict[str, dict]:
    """The fixed boards by id: `{"targets": [row strings of letters], "zones": [zone dicts]}`."""
    raw = json.loads(files("swarmlab.world").joinpath("data/coloring_boards.json").read_text())
    out: dict[str, dict] = {}
    for b in raw:
        if (b["height"], b["width"]) != FIXED_SIZE or len(b["targets"]) != b["height"]:
            raise ValueError(f"board {b['board']}: not {FIXED_SIZE[1]} x {FIXED_SIZE[0]}")
        rows = ["".join(NAME_TO_LETTER[c] for c in row) for row in b["targets"]]
        zones = [dict(z) for z in b["zones"]]
        for i, z in enumerate(zones):
            if (z["zone_id"], z["type"], (z["viewer_group"], z["painter_group"])) != (
                    ZONE_IDS[i], ZONE_TYPES[i], zone_cycle(i)):
                raise ValueError(f"board {b['board']}: zone {i} breaks the A/B/C cycle: {z}")
        _check_zones(zones, b["height"], b["width"])
        out[b["board"]] = {"targets": rows, "zones": zones}
    if tuple(sorted(out)) != BOARD_IDS:
        raise ValueError(f"fixed boards must be {BOARD_IDS}, got {sorted(out)}")
    return out


def _cells(z: dict) -> list[tuple[int, int]]:
    """A zone's cells in row-major order."""
    return [(x, y) for y in range(z["y"], z["y"] + z["height"])
            for x in range(z["x"], z["x"] + z["width"])]


def _check_zones(zones: list[dict], height: int, width: int) -> None:
    seen: set[tuple[int, int]] = set()
    for z in zones:
        cells = _cells(z)
        if any(not (0 <= x < width and 0 <= y < height) for x, y in cells) or seen & set(cells):
            raise ValueError(f"zone {z['zone_id']} is outside the grid or overlaps another")
        seen |= set(cells)


def zone_size(height: int, width: int) -> tuple[int, int]:
    """(zone width, zone height) for a grid: 5 x 2, or the largest w x h (w <= 5, h <= 2) of which
    three fit on an aligned lattice; ValueError for a grid of fewer than 3 cells."""
    sizes = sorted(((w, h) for w in range(ZONE_W, 0, -1) for h in range(ZONE_H, 0, -1)),
                   key=lambda s: (-s[0] * s[1], -s[0]))
    for w, h in sizes:
        if (width // w) * (height // h) >= N_GROUPS:
            return w, h
    raise ValueError(f"a {width} x {height} grid cannot hold {N_GROUPS} zones (needs >= 3 cells)")


def generate_zones(rng: random.Random, height: int, width: int) -> list[dict]:
    """Three non-overlapping zones placed from `rng` (see the module docstring)."""
    zw, zh = zone_size(height, width)

    def zone(i: int, x: int, y: int) -> dict:
        v, p = zone_cycle(i)
        return {"zone_id": ZONE_IDS[i], "type": ZONE_TYPES[i], "x": x, "y": y, "width": zw,
                "height": zh, "viewer_group": v, "painter_group": p}

    for _ in range(200):
        placed: list[dict] = []
        used: set[tuple[int, int]] = set()
        for i in range(N_GROUPS):
            for _ in range(50):
                z = zone(i, rng.randrange(width - zw + 1), rng.randrange(height - zh + 1))
                if not used & set(_cells(z)):
                    placed.append(z)
                    used |= set(_cells(z))
                    break
            else:
                break
        if len(placed) == N_GROUPS:
            return placed
    slots = [(i * zw, j * zh) for j in range(height // zh) for i in range(width // zw)]
    rng.shuffle(slots)
    return [zone(i, x, y) for i, (x, y) in enumerate(slots[:N_GROUPS])]


# ---- relay reports (the recommended board format, parsed loosely) ----------------------------------

def format_report(zone_id: str, letters: Sequence[str]) -> str:
    """The recommended relay post: `Z0 red green blue ...` (colour names, row-major)."""
    return " ".join([zone_id, *(COLOUR_NAMES.get(c, c) for c in letters)])


def _colour_tokens(text: str) -> list[str]:
    """Colour letters in `text`, in order: full colour names and single letters r/g/b/y/...
    (any case), plus runs of 3+ palette letters such as `rgbyr` (a target row copied from the
    observation) read letter by letter. A token is a run of letters not touching a digit, so group
    and zone names (G1, Z0) are never colours. Everything else is ignored, punctuation included."""
    out: list[str] = []
    for tok in re.findall(r"(?<![a-z0-9])[a-z]+(?![a-z0-9])", text.lower()):
        if tok in NAME_TO_LETTER:
            out.append(NAME_TO_LETTER[tok])
        elif len(tok) == 1 and tok in COLOURS:
            out.append(tok)
        elif len(tok) >= 3 and set(tok) <= set(COLOURS):
            out.extend(tok)
    return out


def parse_report(text: str, zone_id: str, n: int = ZONE_W * ZONE_H) -> list[list[str]]:
    """Every reading of `text` as a report on `zone_id`: for each occurrence of the zone id (as a
    token, any case), the first `n` colour tokens after it (`_colour_tokens`), when there are `n`.
    A report is correct when one of the readings equals the zone's targets in row-major order."""
    low = text.lower()
    out: list[list[str]] = []
    for m in re.finditer(rf"(?<![a-z0-9]){re.escape(zone_id.lower())}(?![0-9])", low):
        toks = _colour_tokens(low[m.end():])[:n]
        if len(toks) == n:
            out.append(toks)
    return out


# ---- observation parsing (scripted painters, fake model scripts) -----------------------------------

def parse_observation(text: str) -> tuple[list[str], list[str], list[str]]:
    """(target rows, current rows, colours) from a ColoringGrid observation (or a larger text
    containing one; the last occurrence wins). Hidden target cells stay "?"."""
    idx = text.rfind(TARGET_HEADER)
    if idx < 0:
        return [], [], []
    target: list[str] = []
    current: list[str] = []
    colours: list[str] = []
    section: list[str] | None = None
    for raw in text[idx:].split("\n"):
        line = raw.strip()
        if line.startswith(TARGET_HEADER):
            section = target
        elif line == CURRENT_HEADER:
            section = current
        elif line.startswith(COLOURS_PREFIX):
            colours = line[len(COLOURS_PREFIX):].split()
            section = None
        elif not line:
            section = None
        elif section is not None:
            section.append(line)
    return target, current, colours


_ZONE_RE = re.compile(r"^(Z\d+) type (\w) at x=(\d+)\.\.(\d+), y=(\d+)\.\.(\d+): targets visible "
                      r"to G(\d+)(?: and G(\d+) \(its painters\))?, painted by G(\d+)$")


def parse_zones(text: str) -> dict:
    """The zone and group lines of the last observation in `text`: `{"zones": [{"zone_id", "type",
    "x", "y", "width", "height", "viewers": [groups], "painter_group"}], "groups": {group: [agents]},
    "agent": own id or None, "group": own group or None}`."""
    idx = text.rfind(TARGET_HEADER)
    part = text[idx:] if idx >= 0 else text
    zones: list[dict] = []
    groups: dict[int, list[str]] = {}
    agent = group = None
    for raw in part.split("\n"):
        line = raw.strip()
        m = _ZONE_RE.match(line)
        if m:
            x0, x1, y0, y1 = (int(m.group(k)) for k in (3, 4, 5, 6))
            viewers = [int(m.group(7))] + ([int(m.group(8))] if m.group(8) else [])
            zones.append({"zone_id": m.group(1), "type": m.group(2), "x": x0, "y": y0,
                          "width": x1 - x0 + 1, "height": y1 - y0 + 1, "viewers": viewers,
                          "painter_group": int(m.group(9))})
        elif line.startswith("Groups:"):
            for chunk in line[len("Groups:"):].split(";"):
                name, _, members = chunk.partition("=")
                if name.strip().startswith("G"):
                    groups[int(name.strip()[1:])] = [a.strip() for a in members.split(",") if a.strip()]
        else:
            m = re.match(r"^You are (\S+), in group G(\d+)\.", line)
            if m:
                agent, group = m.group(1), int(m.group(2))
    return {"zones": zones, "groups": groups, "agent": agent, "group": group}


def zone_at(zones: Sequence[dict], x: int, y: int) -> dict | None:
    """The zone (as in `parse_zones` or `verify()`) holding cell (x, y), else None."""
    for z in zones:
        if z["x"] <= x < z["x"] + z["width"] and z["y"] <= y < z["y"] + z["height"]:
            return z
    return None


def needed_cells(target: list[str], current: list[str]) -> list[tuple[int, int, str]]:
    """(x, y, colour) of every cell whose current colour differs from a visible target, row-major
    (hidden "?" target cells are skipped)."""
    return [(x, y, target[y][x]) for y in range(len(target)) for x in range(len(target[y]))
            if target[y][x] != HIDDEN
            and (y >= len(current) or x >= len(current[y]) or current[y][x] != target[y][x])]


def paintable_needed_cells(text: str) -> list[tuple[int, int, str]]:
    """`needed_cells` of the observation in `text`, minus the cells of zones the observing agent's
    group may not paint."""
    target, current, _ = parse_observation(text)
    info = parse_zones(text)
    out = []
    for x, y, c in needed_cells(target, current):
        z = zone_at(info["zones"], x, y)
        if z is None or z["painter_group"] == info["group"]:
            out.append((x, y, c))
    return out


class ColoringGrid(World):
    """Shared grid with three crossed zones: targets seen by one group, painted by another."""

    entry_point: ClassVar[str | None] = "coloring"
    name = "coloring"
    requires: ClassVar[Requirements] = Requirements(commit="round_end", n_agents=(N_GROUPS, None))

    def __init__(self, height: int = 8, width: int = 15, palette: int = 4, board: str = "auto",
                 oracle: bool = False,
                 target_changes: list | tuple = (),
                 worker_failures: list | tuple = (),
                 paints_per_round: int = 10,
                 status_tools: list[str] | tuple[str, ...] = ()) -> None:
        if height < 1 or width < 1:
            raise ValueError("height and width must be >= 1")
        if not 2 <= palette <= len(COLOURS):
            raise ValueError(f"palette must be in 2..{len(COLOURS)}, got {palette}")
        if paints_per_round < 1:
            raise ValueError("paints_per_round must be >= 1")
        board = str(board)
        if board not in ("auto", "random", *BOARD_IDS):
            raise ValueError(f"board must be 'auto', 'random' or one of {list(BOARD_IDS)}, got {board!r}")
        fixed_ok = (height, width) == FIXED_SIZE and palette == FIXED_PALETTE
        if board in BOARD_IDS and not fixed_ok:
            raise ValueError(f"fixed board {board} needs height={FIXED_SIZE[0]}, width={FIXED_SIZE[1]}, "
                             f"palette={FIXED_PALETTE}")
        bad_status = set(status_tools) - {"my_status", "collective_status"}
        if bad_status:
            raise ValueError(f"unknown status tools {sorted(bad_status)}")
        self.height, self.width = height, width
        self.colours = list(COLOURS[:palette])
        self.board = board
        self.oracle = bool(oracle)
        self.zone_w, self.zone_h = (ZONE_W, ZONE_H) if fixed_ok else zone_size(height, width)
        self.target_changes = [(int(r), [(int(x), int(y), str(c)) for x, y, c in cells])
                               for r, cells in target_changes]
        for r, cells in self.target_changes:
            for x, y, c in cells:
                if not (0 <= x < width and 0 <= y < height) or c not in self.colours:
                    raise ValueError(f"target change at round {r}: bad cell {(x, y, c)}")
        self.worker_failures = [(int(r), [AgentId(a) for a in agents]) for r, agents in worker_failures]
        self.paints_per_round = paints_per_round
        self.status_tools = tuple(status_tools)
        # state (snapshotted)
        self.agents: list[AgentId] = []
        self.groups: dict[str, int] = {}
        self.board_id: str | None = None
        self.crossed: list[dict] = []
        self.target: list[list[str]] = []
        self.initial_target: list[list[str]] = []
        self.grid: list[list[str]] = []
        self.painter: list[list[str | None]] = []
        self.round = 0
        self.paint_counts: dict[str, int] = {}
        self.stats = {"paints": 0, "useful_paints": 0, "duplicate_paints": 0, "wrong_paints": 0,
                      "overwritten_paints": 0, "wasted_paints": 0}
        self.zone_paint_refusals = 0
        self.round_paints: dict[str, int] = {}

    _skip_in_snapshot: ClassVar[tuple[str, ...]] = (
        "params", "height", "width", "colours", "board", "oracle", "zone_w", "zone_h",
        "target_changes", "worker_failures", "paints_per_round", "status_tools",
    )

    # ---- lifecycle ---------------------------------------------------------------------------
    def _uses_fixed(self) -> bool:
        if self.board in BOARD_IDS:
            return True
        return (self.board == "auto" and (self.height, self.width) == FIXED_SIZE
                and len(self.colours) == FIXED_PALETTE)

    def reset(self, rng: random.Random, agents: list[AgentId]) -> None:
        if len(agents) < N_GROUPS:
            raise ValueError(f"ColoringGrid needs at least {N_GROUPS} agents (one per group), "
                             f"got {len(agents)}")
        self.agents = list(agents)
        self.groups = {str(a): i % N_GROUPS for i, a in enumerate(self.agents)}
        if self._uses_fixed():
            if self.board in BOARD_IDS:
                bid = self.board
            else:
                root = getattr(rng, "root", None)
                seed = root[0] if isinstance(root, tuple) and root else None
                idx = (seed - 1) % len(BOARD_IDS) if isinstance(seed, int) else rng.randrange(len(BOARD_IDS))
                bid = BOARD_IDS[idx]
            fb = fixed_boards()[bid]
            self.board_id = bid
            self.target = [list(r) for r in fb["targets"]]
            self.crossed = [dict(z) for z in fb["zones"]]
        else:
            self.board_id = "random"
            self.target = [[rng.choice(self.colours) for _ in range(self.width)]
                           for _ in range(self.height)]
            self.crossed = generate_zones(rng, self.height, self.width)
        self.initial_target = [list(r) for r in self.target]
        self.grid = [[UNPAINTED] * self.width for _ in range(self.height)]
        self.painter = [[None] * self.width for _ in range(self.height)]
        self.round = 0
        self.paint_counts = {}
        self.round_paints = {}
        self.zone_paint_refusals = 0
        for k in self.stats:
            self.stats[k] = 0

    def begin_round(self, round: int) -> None:
        self.round = round
        self.round_paints = {}
        for r, cells in self.target_changes:
            if r == round:
                for x, y, c in cells:
                    self.target[y][x] = c

    def killed_at(self, round: int) -> list[AgentId]:
        return [a for r, agents in self.worker_failures if r == round for a in agents]

    # ---- zones ---------------------------------------------------------------------------------
    def _zone_at(self, x: int, y: int) -> dict | None:
        return zone_at(self.crossed, x, y)

    def _can_view(self, agent: AgentId, z: dict) -> bool:
        g = self.groups.get(str(agent))
        return g == z["viewer_group"] or (self.oracle and g == z["painter_group"])

    def _zone_refusal(self, agent: AgentId, x: int, y: int) -> str | None:
        z = self._zone_at(x, y)
        if z is None or self.groups.get(str(agent)) == z["painter_group"]:
            return None
        return (f"cell ({x}, {y}) is in zone {z['zone_id']}, which only group "
                f"G{z['painter_group']} may paint")

    def _members(self, g: int) -> list[str]:
        return [a for a in self.agents if self.groups.get(str(a)) == g]

    # ---- agent-facing ------------------------------------------------------------------------
    def _zone_cells_n(self) -> int:
        return self.zone_w * self.zone_h

    def description(self) -> str:
        n = self.paints_per_round
        k = self._zone_cells_n()
        names = [COLOUR_NAMES[c] for c in self.colours]
        example = " ".join(["Z0", *(names[i % len(names)] for i in range(k))])
        if self.oracle:
            visibility = (
                "Each zone's target colours are shown to two groups, one that only sees them and the "
                "zone's painting group; to everyone else the zone's cells show \"?\" in the target "
                "grid. Only the painting group may paint a zone: a paint inside a zone by an agent "
                "outside its painting group is refused. Cells outside the zones are open: anyone "
                "may paint them."
            )
        else:
            visibility = (
                "Each zone's target colours are shown only to one group (to everyone else the zone's "
                "cells show \"?\" in the target grid), and only a different group may paint the zone: "
                "a paint inside a zone by an agent outside its painting group is refused. Cells "
                "outside the zones are open: anyone may paint them. A zone's targets therefore "
                "have to reach its painting group over the message board."
            )
        return (
            "You are one of several agents colouring a shared grid. Each round you see the target "
            "grid and the current grid. Use paint(x, y, color) to paint a cell (x = column, "
            f"y = row, both from 0) with a colour letter. You may paint at most {n} "
            f"cell{'s' if n != 1 else ''} per round; paints are applied at the end of the round, "
            "and a paint beyond the limit is refused. Painting a cell that already has its target "
            "colour wastes your paint. When several agents paint the same cell in one round, the "
            "last one applied wins. The task is done when the current grid equals the target. "
            "The agents are split into three groups, G0, G1 and G2, and the grid holds three "
            f"zones, Z0, Z1 and Z2, of {self.zone_w} x {self.zone_h} cells. " + visibility + " "
            f"Recommended format for a post about a zone: the zone id followed by its {k} target "
            "colours in row-major order (the top row left to right, then the next row), for "
            f"example \"{example}\"; other wording works too. Board posts are delivered when the "
            "round ends, so a post can be read with read_board from the next round on."
        )

    def _duties(self, agent: AgentId) -> str:
        g = self.groups.get(str(agent))
        if g is None:
            return ""
        sees = [z for z in self.crossed if z["viewer_group"] == g]
        paints = [z for z in self.crossed if z["painter_group"] == g]
        parts = [f"You are {agent}, in group G{g}."]
        for z in sees:
            if self.oracle:
                parts.append(f"Your group sees the targets of {z['zone_id']}, which "
                             f"G{z['painter_group']} paints (G{z['painter_group']} sees them too).")
            else:
                parts.append(f"Your group sees the targets of {z['zone_id']} and must report them on "
                             f"the message board for G{z['painter_group']}, which paints "
                             f"{z['zone_id']}.")
        for z in paints:
            if self.oracle:
                parts.append(f"Only your group may paint {z['zone_id']}; its targets are shown to "
                             "your group as well.")
            else:
                parts.append(f"Only your group may paint {z['zone_id']}; its targets are shown to "
                             f"G{z['viewer_group']}, so you must get them from the board.")
        return " ".join(parts)

    def description_for(self, agent: AgentId) -> str:
        duties = self._duties(agent)
        return self.description() + (f" {duties}" if duties else "")

    def observe(self, agent: AgentId) -> Observation:
        header = (f"{TARGET_HEADER} ({self.height} rows x {self.width} columns; "
                  f"x = column 0-{self.width - 1}, y = row 0-{self.height - 1}; "
                  f"\"{HIDDEN}\" = hidden from you):")
        lines = [header]
        for y, row in enumerate(self.target):
            cells = []
            for x, c in enumerate(row):
                z = self._zone_at(x, y)
                cells.append(c if z is None or self._can_view(agent, z) else HIDDEN)
            lines.append("".join(cells))
        lines += ["", CURRENT_HEADER]
        lines += ["".join(r) for r in self.grid]
        lines += ["", f"{COLOURS_PREFIX} {' '.join(self.colours)}",
                  "Colour names: " + ", ".join(f"{c} = {COLOUR_NAMES[c]}" for c in self.colours)]
        if self.crossed:
            lines += ["", "Groups: " + "; ".join(f"G{g} = {', '.join(self._members(g))}"
                                                 for g in range(N_GROUPS)),
                      (f"Zones ({self.zone_w} columns x {self.zone_h} rows each; cells outside "
                       "the zones are open: anyone may paint them):")]
            for z in self.crossed:
                v, p = z["viewer_group"], z["painter_group"]
                seen = f"G{v} and G{p} (its painters)" if self.oracle else f"G{v}"
                lines.append(f"{z['zone_id']} type {z['type']} at x={z['x']}..{z['x'] + z['width'] - 1}, "
                             f"y={z['y']}..{z['y'] + z['height'] - 1}: targets visible to {seen}, "
                             f"painted by G{p}")
            duties = self._duties(agent)
            if duties:
                lines.append(duties)
        g = self.groups.get(str(agent))
        return text_observation("\n".join(lines), **({"group": g} if g is not None else {}))

    def _cell_error(self, x: Any, y: Any, color: Any) -> tuple[str, str] | None:
        """(reason code, message) when the paint is malformed, else None."""
        if not isinstance(x, int) or isinstance(x, bool) or not isinstance(y, int) or isinstance(y, bool):
            return "out_of_bounds", "x and y must be integers"
        if not (0 <= x < self.width and 0 <= y < self.height):
            return "out_of_bounds", f"cell ({x}, {y}) is outside the {self.width} x {self.height} grid"
        if color not in self.colours:
            return "bad_color", f"unknown color {color!r}; colours: {' '.join(self.colours)}"
        return None

    def _limit_text(self) -> str:
        n = self.paints_per_round
        return f"at most {n} paint{'s' if n != 1 else ''} per round"

    def validate(self, agent: AgentId, action: Action, pending: Sequence[Action] = ()) -> Ack:
        ack = super().validate(agent, action)
        if not ack.ok or action.name != "paint":
            return ack
        x, y = action.args.get("x"), action.args.get("y")
        err = self._cell_error(x, y, action.args.get("color"))
        if err:
            return Ack(ok=False, error=f"{err[0]}: {err[1]}")
        assert isinstance(x, int) and isinstance(y, int)  # `_cell_error` checked them
        refusal = self._zone_refusal(agent, x, y)
        if refusal:
            return Ack(ok=False, error=f"zone_paint_refused: {refusal}")
        if sum(1 for a in pending if a.name == "paint") >= self.paints_per_round:
            return Ack(ok=False, error=f"paints_per_round: {self._limit_text()}; this one would be "
                                       "rejected at commit")
        return ack

    def _paints_used(self, agent: AgentId, pending: Sequence[Action]) -> int:
        """Paints `agent` has applied this round (immediate) plus those buffered (round_end)."""
        return self.round_paints.get(agent, 0) + sum(1 for a in pending if a.name == "paint")

    @tool("paint", "Paint cell (x = column, y = row) with a colour letter.",
          {"x": "integer", "y": "integer", "color": "string"})
    def paint(self, agent: AgentId, x: int, y: int, color: str) -> Outcome:
        err = self._cell_error(x, y, color)
        if err:
            return Outcome(accepted=False, feedback={"error": err[0], "detail": err[1]})
        if agent not in self.agents:
            return Outcome(accepted=False, feedback={"error": "unknown agent"})
        refusal = self._zone_refusal(agent, x, y)
        if refusal:
            self.zone_paint_refusals += 1
            return Outcome(accepted=False, feedback={"error": "zone_paint_refused", "detail": refusal})
        n = self.round_paints.get(agent, 0)
        if n >= self.paints_per_round:
            return Outcome(accepted=False, feedback={"error": "paints_per_round",
                                                     "detail": self._limit_text()})
        self.round_paints[agent] = n + 1
        want = self.target[y][x]
        before = self.grid[y][x]
        self.stats["paints"] += 1
        if color != want:
            self.stats["wrong_paints"] += 1
        elif before == want:
            self.stats["duplicate_paints"] += 1
        else:
            self.stats["useful_paints"] += 1
        if not (color == want and before != want):
            self.stats["wasted_paints"] += 1
        self.grid[y][x] = color
        self.painter[y][x] = agent
        self.paint_counts[agent] = self.paint_counts.get(agent, 0) + 1
        return Outcome(accepted=True, feedback={"result": "painted"})

    def commit(self, actions: list[tuple[AgentId, ActionId, Action]]) -> list[Outcome]:
        outcomes = super().commit(actions)
        last: dict[tuple[int, int], int] = {}
        for i, ((_, _, action), out) in enumerate(zip(actions, outcomes, strict=True)):
            if action.name == "paint" and out.accepted:
                last[(action.args["x"], action.args["y"])] = i
        for i, ((_, _, action), out) in enumerate(zip(actions, outcomes, strict=True)):
            if action.name == "paint" and out.accepted and last[(action.args["x"], action.args["y"])] != i:
                out.feedback = {"result": "overwritten"}
                self.stats["overwritten_paints"] += 1
        return outcomes

    def claim_key(self, agent: AgentId, action: Action) -> str | None:
        if action.name != "paint":
            return None
        x, y = action.args.get("x"), action.args.get("y")
        if not isinstance(x, int) or not isinstance(y, int):
            return None
        return f"cell:{x},{y}"

    def _matching(self) -> int:
        return sum(1 for y in range(self.height) for x in range(self.width)
                   if self.grid[y][x] == self.target[y][x])

    def my_status(self, agent: AgentId, pending: Sequence[Action] = ()) -> dict | None:
        if "my_status" not in self.status_tools:
            return None
        left = max(0, self.paints_per_round - self._paints_used(agent, pending))
        return {"paints": self.paint_counts.get(agent, 0), "paints_left_this_round": left}

    def collective_status(self) -> dict | None:
        if "collective_status" not in self.status_tools:
            return None
        painted = sum(1 for row in self.grid for c in row if c != UNPAINTED)
        return {"cells_painted": painted, "cells": self.height * self.width}

    # ---- evaluator-only ----------------------------------------------------------------------
    def coverage(self) -> float:
        return self._matching() / (self.height * self.width) if self.grid else 0.0

    def terminal(self) -> bool:
        if not self.grid or any(r > self.round for r, _ in self.target_changes):
            return False
        return self._matching() == self.height * self.width

    def _zone_counts(self) -> tuple[int, int, int]:
        """(zones completed, matching zone cells, matching open cells)."""
        done = crossed = 0
        in_zone: set[tuple[int, int]] = set()
        for z in self.crossed:
            cells = _cells(z)
            in_zone |= set(cells)
            ok = sum(1 for x, y in cells if self.grid[y][x] == self.target[y][x])
            crossed += ok
            done += int(ok == len(cells))
        open_ok = sum(1 for y in range(self.height) for x in range(self.width)
                      if (x, y) not in in_zone and self.grid[y][x] == self.target[y][x])
        return done, crossed, open_ok

    def score(self) -> dict:
        cells = self.height * self.width
        if not self.grid:
            return {"coverage": 0.0, "correct": 0, "wrong": 0, "unpainted": cells, **self.stats,
                    "board": self.board_id, "zones_completed": 0, "crossed_correct": 0,
                    "open_correct": 0, "zone_paint_refusals": 0}
        correct = self._matching()
        unpainted = sum(1 for row in self.grid for c in row if c == UNPAINTED)
        done, crossed, open_ok = self._zone_counts()
        return {"coverage": correct / cells, "correct": correct, "wrong": cells - correct - unpainted,
                "unpainted": unpainted, **self.stats, "board": self.board_id,
                "zones_completed": done, "crossed_correct": crossed, "open_correct": open_ok,
                "zone_paint_refusals": self.zone_paint_refusals}

    def verify(self) -> dict:
        return {"target": ["".join(r) for r in self.target],
                "initial_target": ["".join(r) for r in self.initial_target],
                "target_changes": [[r, [list(c) for c in cells]] for r, cells in self.target_changes],
                "colours": list(self.colours), "height": self.height, "width": self.width,
                "paints_per_round": self.paints_per_round, "board": self.board_id,
                "crossed_zones": [dict(z) for z in self.crossed], "groups": dict(self.groups),
                "oracle": self.oracle}

    def render_state(self) -> dict | None:
        """Target and current grid and the zones for the replay page (see `World.render_state`)."""
        if not self.grid:
            return None
        cells = self.height * self.width
        done, crossed, open_ok = self._zone_counts()
        return {"grid": [list(r) for r in self.grid], "target": [list(r) for r in self.target],
                "palette": {**{c: PALETTE_CSS[c] for c in self.colours}, UNPAINTED: "transparent"},
                "zones": [{"zone": z["zone_id"], "type": z["type"], "x": z["x"], "y": z["y"],
                           "width": z["width"], "height": z["height"],
                           "seen_by": f"G{z['viewer_group']}", "painted_by": f"G{z['painter_group']}"}
                          for z in self.crossed],
                "board": self.board_id, "coverage": round(self._matching() / cells, 4),
                "cells_matching": self._matching(), "cells": cells, "zones_completed": done,
                "crossed_correct": crossed, "open_correct": open_ok, "round": self.round,
                "paints": self.stats["paints"], "wrong_paints": self.stats["wrong_paints"],
                "duplicate_paints": self.stats["duplicate_paints"]}

    def report_sections(self, arms: dict[str, list[Any]]) -> list[str]:
        """A "Crossed zones" table: per run, the board, each zone's final relay status (the
        `coloring.relay_status` fold over the run's events) and posts / read_board calls per
        agent (in agent order)."""
        from ..metrics.coloring import relay_summary

        rows = []
        for arm, runs in sorted(arms.items()):
            for run in runs:
                try:
                    s = relay_summary(run)
                except Exception as e:  # noqa: BLE001 - one bad run must not hide the others
                    rows.append([arm, run.id, f"n/a ({type(e).__name__}: {e})", "", "", "", "", "", ""])
                    continue
                rows.append([arm, run.id, s["board"], *(s["status"].get(z, "n/a") for z in ZONE_IDS),
                             s["zones_completed"], " / ".join(str(v) for v in s["posts"].values()),
                             " / ".join(str(v) for v in s["reads"].values())])
        if not rows:
            return []
        head = ["arm", "run", "board", *ZONE_IDS, "zones done", "posts per agent",
                "read_board calls per agent"]
        out = ["## Crossed zones (relay status at the end of each run)", "",
               "| " + " | ".join(head) + " |", "|" + "---|" * len(head)]
        out += ["| " + " | ".join(str(c).replace("|", "\\|") for c in r) + " |" for r in rows]
        return out + ["", "Statuses: " + ", ".join(RELAY_STATUSES) + " (see `coloring.relay_status`).", ""]


# ---- fake LLM script (entry point `painter` in swarmlab.fake_scripts; model "fake:painter") -----

def fake_painter(request: Any, rng: random.Random) -> Any:
    """FakeProvider script for ColoringGrid: paint one random cell per turn that the agent can
    paint and whose visible target differs from the current grid.

    Once per turn, paint such a cell (from the latest observation in the conversation: hidden "?"
    targets and zones of other painting groups are skipped) with its target colour, then
    `end_turn`. It never reads or posts, so zones it cannot see stay unpainted. A probe (no tools)
    gets the text `{"coverage": <share of matching cells among the visible targets>}`. The rng is
    a function of the request, so agents with identical prompts pick the same cell."""
    from ..providers.base import text_of
    from ..providers.fake import RESULTS_PREFIX, _assistant_calls, _response

    tools = {t.name for t in request.tools}
    msgs = request.messages
    last_user = max((i for i, m in enumerate(msgs) if m.role == "user"
                     and not text_of(m.content).startswith(RESULTS_PREFIX)), default=-1)
    this_turn = [name for m in msgs[last_user + 1:] for name, _ in _assistant_calls(m)]
    obs = next((text_of(m.content) for m in reversed(msgs)
                if m.role == "user" and TARGET_HEADER in text_of(m.content)), "")
    if not tools:
        target, current, _ = parse_observation(obs)
        cells = sum(1 for r in target for c in r if c != HIDDEN)
        cov = (cells - len(needed_cells(target, current))) / cells if cells else 0.0
        return _response(request, rng, [], json.dumps({"coverage": round(cov, 3)}))
    need = paintable_needed_cells(obs)
    calls: list[tuple[str, dict]] = []
    if "paint" in tools and "paint" not in this_turn and need:
        x, y, c = rng.choice(need)
        calls.append(("paint", {"x": x, "y": y, "color": c}))
    if "end_turn" in tools:
        calls.append(("end_turn", {}))
    return _response(request, rng, calls)
