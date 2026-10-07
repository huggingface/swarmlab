"""ColoringGrid, plain variant (DESIGN.md component 4; M3b).

A shared world: every agent sees the full target grid and the current grid, and paints cells with
`paint(x, y, color)`. The swarm's job is to make the current grid equal the target. It is the
framework-validation task for the registry and claim policies (S0: 16 agents, full visibility).

Grid and colours
----------------
`height` rows by `width` columns; `x` is the column (0 .. width-1), `y` the row (0 .. height-1).
Colours are the first `palette` letters of `COLOURS` ("rgbykwopcmnt", as in FlagGame), so
`2 <= palette <= 12`; an unpainted cell is ".". `reset` draws the target cell by cell in
row-major order from the world rng; the current grid starts unpainted.

Observation format (exact; `parse_observation` inverts it)
----------------------------------------------------------
One text part:

    Target grid (8 rows x 8 columns; x = column 0-7, y = row 0-7):
    <row 0 of the target>
    ...
    <blank>
    Current grid ("." = unpainted):
    <row 0 of the current grid>
    ...
    <blank>
    Colours: r g b y

plus, when zones are on, a final line `Zones: 2 x 2 (zone of a cell: zone:<x * zx // width>,<y * zy // height>)`.
`observation.private` is empty (nothing is hidden in the plain variant).

Actions and commit
------------------
- `paint(x, y, color)`: `validate` rejects out-of-bounds cells (`error="out_of_bounds: ..."`)
  and unknown colours (`"bad_color: ..."`) at call time.
  At most `paints_per_round` paints per agent are applied per round (counted at commit in commit
  order; further ones get `accepted=False, feedback={"error": "paints_per_round", "detail"}`).
  `validate(agent, action, pending)` also refuses at call time a paint the commit would reject
  for the limit: when the agent's buffered `pending` paints (round_end only) already reach
  `paints_per_round`, the error is `"paints_per_round: at most N
  paint(s) per round; this one would be rejected at commit"` and the executor does not buffer
  it. Under immediate commit `pending` is empty and the world's own commit already enforces the
  limit (the call returns `ok=True` with `accepted=False`), so nothing changes there. Caveat: under an `enforced` claim policy a buffered paint
  may later be rejected `not_claimed` without reaching the world, and the refused paint would
  then have been applied; the pre-check counts every buffered paint all the same (the limit is
  stated to agents as "paints per round", not "applied paints").
- Commit-time feedback says why a paint was not applied: `error` is `paints_per_round`,
  `out_of_bounds` or `bad_color` (with a human `detail`); an applied paint that a later one
  replaced says `{"result": "overwritten"}`. The LLM participant renders these in its
  "Outcomes of your actions last round" lines.
- Conflict rule: paints are applied in commit (seeded) order, so the last writer of a cell wins.
  An accepted paint's feedback is `{"result": "painted"}`, or `{"result": "overwritten"}` when a
  later paint in the same commit batch painted the same cell. Under immediate commit every paint
  is its own batch, so nothing is reported overwritten.
- Paint accounting (evaluator-side, in commit order): a paint is `wrong` when its colour is not
  the target's, a `duplicate` when the cell already matched the target and it paints the target
  colour again, `useful` when it turns a non-matching cell into a matching one; `wasted` is every
  applied paint that is not useful (wrong + duplicate + correct-colour repaints of cells that
  changed back in between, which cannot happen without wrong paints).
- `claim_key(agent, paint) -> "cell:<x>,<y>"`, or `"zone:<zx>,<zy>"` when `zones=[zy_count,
  zx_count]` splits the grid into blocks (the minimal "crossed zones" variant: claims cover
  regions, and an agent's neighbouring paints share one claim). Off by default.

Dynamics
--------
- `target_changes: [[round, [[x, y, color], ...]], ...]` changes the target at the start of that
  round (`begin_round`, before anyone observes), so agents see the new target in that round.
- `worker_failures: [[round, [agent, ...]], ...]`: `killed_at(round)` returns those agents and
  the runner removes them from the live set at the start of that round (they take no turn from
  that round on), logging an `intervention` event `intervention="worker_failures", op="kill"`
  per agent, the same as the `kill_agents` intervention. A failure at round 1 is applied before
  round 1's turns.

Status tools (both on by default; `status_tools` toggles): `my_status` -> `{"paints", "cells_mine",
"paints_left_this_round"}` (applied paints; cells whose current colour is this agent's paint and
matches the target; `paints_per_round` minus this round's applied paints (immediate) and
buffered paints (round_end)). The
executor passes the buffered actions as `my_status(agent, pending=...)` (the same hook as
`validate`; least invasive: no executor-to-world callback and no world state touched during
turns), so under round_end the count drops as the agent paints;
`collective_status` -> `{"coverage", "cells_matching", "cells"}`. The target is visible to every
agent in this variant, so these reveal nothing an agent could not compute.

`terminal()` is true when coverage is 1.0 and no target change is scheduled after the current
round. `score()` -> coverage, correct (matching cells), wrong (painted, not matching), unpainted,
paints, useful_paints, duplicate_paints, wrong_paints, overwritten_paints, wasted_paints.
`verify()` -> target (as of now), initial_target, target_changes, colours, sizes, zones.

Snapshots carry game state only (grid, target, painters, counters, round); constructor config is
kept on restore, as in FlagGame.
"""
from __future__ import annotations

import random
from collections.abc import Sequence
from typing import Any, ClassVar

from ..ids import ActionId, AgentId
from ..view import Observation, text_observation
from .base import Ack, Action, Outcome, World, tool

COLOURS = "rgbykwopcmnt"
UNPAINTED = "."
TARGET_HEADER = "Target grid"
CURRENT_HEADER = 'Current grid ("." = unpainted):'
COLOURS_PREFIX = "Colours:"


def parse_observation(text: str) -> tuple[list[str], list[str], list[str]]:
    """(target rows, current rows, colours) from a ColoringGrid observation (or a larger text
    containing one; the last occurrence wins)."""
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


def needed_cells(target: list[str], current: list[str]) -> list[tuple[int, int, str]]:
    """(x, y, colour) of every cell whose current colour differs from the target, row-major."""
    return [(x, y, target[y][x]) for y in range(len(target)) for x in range(len(target[y]))
            if y >= len(current) or x >= len(current[y]) or current[y][x] != target[y][x]]


class ColoringGrid(World):
    entry_point: ClassVar[str | None] = "coloring"
    name = "coloring"

    def __init__(self, height: int = 8, width: int = 8, palette: int = 4,
                 zones: list[int] | None = None,
                 target_changes: list | tuple = (),
                 worker_failures: list | tuple = (),
                 paints_per_round: int = 1,
                 status_tools: list[str] | tuple[str, ...] = ("my_status", "collective_status")) -> None:
        if height < 1 or width < 1:
            raise ValueError("height and width must be >= 1")
        if not 2 <= palette <= len(COLOURS):
            raise ValueError(f"palette must be in 2..{len(COLOURS)}, got {palette}")
        if paints_per_round < 1:
            raise ValueError("paints_per_round must be >= 1")
        if zones is not None and (len(zones) != 2 or min(zones) < 1):
            raise ValueError("zones must be [zone rows, zone columns], each >= 1")
        self.height, self.width = height, width
        self.colours = list(COLOURS[:palette])
        self.zones = None if zones is None else [int(zones[0]), int(zones[1])]
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
        self.target: list[list[str]] = []
        self.initial_target: list[list[str]] = []
        self.grid: list[list[str]] = []
        self.painter: list[list[str | None]] = []
        self.round = 0
        self.paint_counts: dict[str, int] = {}
        self.stats = {"paints": 0, "useful_paints": 0, "duplicate_paints": 0, "wrong_paints": 0,
                      "overwritten_paints": 0, "wasted_paints": 0}
        self.round_paints: dict[str, int] = {}

    _skip_in_snapshot: ClassVar[tuple[str, ...]] = (
        "params", "height", "width", "colours", "zones", "target_changes", "worker_failures",
        "paints_per_round", "status_tools",
    )

    # ---- lifecycle ---------------------------------------------------------------------------
    def reset(self, rng: random.Random, agents: list[AgentId]) -> None:
        self.agents = list(agents)
        self.target = [[rng.choice(self.colours) for _ in range(self.width)] for _ in range(self.height)]
        self.initial_target = [list(r) for r in self.target]
        self.grid = [[UNPAINTED] * self.width for _ in range(self.height)]
        self.painter = [[None] * self.width for _ in range(self.height)]
        self.round = 0
        self.paint_counts = {}
        self.round_paints = {}
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

    # ---- agent-facing ------------------------------------------------------------------------
    def description(self) -> str:
        n = self.paints_per_round
        return (
            "You are one of several agents colouring a shared grid. Each round you see the target "
            "grid and the current grid. Use paint(x, y, color) to paint a cell (x = column, "
            f"y = row, both from 0). You may paint at most {n} cell{'s' if n != 1 else ''} per "
            "round; paints are applied at the end of the round, and a paint beyond the limit is "
            "refused. Painting a cell that already has its target colour wastes your paint. When "
            "several agents paint the same cell in one round, the last one applied wins. The task "
            "is done when the current grid equals the target."
        )

    def observe(self, agent: AgentId) -> Observation:
        header = (f"{TARGET_HEADER} ({self.height} rows x {self.width} columns; "
                  f"x = column 0-{self.width - 1}, y = row 0-{self.height - 1}):")
        lines = [header]
        lines += ["".join(r) for r in self.target]
        lines += ["", CURRENT_HEADER]
        lines += ["".join(r) for r in self.grid]
        lines += ["", f"{COLOURS_PREFIX} {' '.join(self.colours)}"]
        if self.zones is not None:
            zy, zx = self.zones
            lines.append(f"Zones: {zy} x {zx} (zone of a cell: zone:<x * {zx} // {self.width}>,"
                         f"<y * {zy} // {self.height}>)")
        return text_observation("\n".join(lines))

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
        err = self._cell_error(action.args.get("x"), action.args.get("y"), action.args.get("color"))
        if err:
            return Ack(ok=False, error=f"{err[0]}: {err[1]}")
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
        if self.zones is None:
            return f"cell:{x},{y}"
        zy, zx = self.zones
        return f"zone:{x * zx // self.width},{y * zy // self.height}"

    def _matching(self) -> int:
        return sum(1 for y in range(self.height) for x in range(self.width)
                   if self.grid[y][x] == self.target[y][x])

    def my_status(self, agent: AgentId, pending: Sequence[Action] = ()) -> dict | None:
        if "my_status" not in self.status_tools:
            return None
        mine = sum(1 for y in range(self.height) for x in range(self.width)
                   if self.painter and self.painter[y][x] == agent and self.grid[y][x] == self.target[y][x])
        left = max(0, self.paints_per_round - self._paints_used(agent, pending))
        return {"paints": self.paint_counts.get(agent, 0), "cells_mine": mine,
                "paints_left_this_round": left}

    def collective_status(self) -> dict | None:
        if "collective_status" not in self.status_tools:
            return None
        cells = self.height * self.width
        m = self._matching() if self.grid else 0
        return {"coverage": m / cells, "cells_matching": m, "cells": cells}

    # ---- evaluator-only ----------------------------------------------------------------------
    def coverage(self) -> float:
        return self._matching() / (self.height * self.width) if self.grid else 0.0

    def terminal(self) -> bool:
        if not self.grid or any(r > self.round for r, _ in self.target_changes):
            return False
        return self._matching() == self.height * self.width

    def score(self) -> dict:
        cells = self.height * self.width
        if not self.grid:
            return {"coverage": 0.0, "correct": 0, "wrong": 0, "unpainted": cells, **self.stats}
        correct = self._matching()
        unpainted = sum(1 for row in self.grid for c in row if c == UNPAINTED)
        return {"coverage": correct / cells, "correct": correct, "wrong": cells - correct - unpainted,
                "unpainted": unpainted, **self.stats}

    def verify(self) -> dict:
        return {"target": ["".join(r) for r in self.target],
                "initial_target": ["".join(r) for r in self.initial_target],
                "target_changes": [[r, [list(c) for c in cells]] for r, cells in self.target_changes],
                "colours": list(self.colours), "height": self.height, "width": self.width,
                "zones": self.zones, "paints_per_round": self.paints_per_round}


# ---- fake LLM script (entry point `painter` in swarmlab.fake_scripts; model "fake:painter") -----

def fake_painter(request: Any, rng: random.Random) -> Any:
    """A FakeProvider script for ColoringGrid: once per turn, paint a random cell that differs
    from the target (from the latest observation in the conversation) with its target colour,
    then `end_turn`. A probe (no tools) gets the text `{"coverage": <share of matching cells>}`.
    The rng is a function of the request, so agents with identical prompts pick the same cell."""
    import json

    from ..providers.base import text_of
    from ..providers.fake import RESULTS_PREFIX, _assistant_calls, _response

    tools = {t.name for t in request.tools}
    msgs = request.messages
    last_user = max((i for i, m in enumerate(msgs) if m.role == "user"
                     and not text_of(m.content).startswith(RESULTS_PREFIX)), default=-1)
    this_turn = [name for m in msgs[last_user + 1:] for name, _ in _assistant_calls(m)]
    obs = next((text_of(m.content) for m in reversed(msgs)
                if m.role == "user" and TARGET_HEADER in text_of(m.content)), "")
    target, current, _ = parse_observation(obs)
    need = needed_cells(target, current)
    if not tools:
        cells = sum(len(r) for r in target)
        cov = (cells - len(need)) / cells if cells else 0.0
        return _response(request, rng, [], json.dumps({"coverage": round(cov, 3)}))
    calls: list[tuple[str, dict]] = []
    if "paint" in tools and "paint" not in this_turn and need:
        x, y, c = rng.choice(need)
        calls.append(("paint", {"x": x, "y": y, "color": c}))
    if "end_turn" in tools:
        calls.append(("end_turn", {}))
    return _response(request, rng, calls)
