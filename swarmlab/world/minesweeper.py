"""Minesweeper: one shared board, one hidden minefield, the swarm reveals every safe cell.

A collaboration task whose difficulty is set by the board: `height` x `width` cells hiding
`mines` mines. Every agent sees the whole board; agents reveal cells, flag cells they believe
are mines, and win as a team when every safe cell is revealed. The interesting questions are
about the team: whether agents split the work (two agents revealing the same cell in one round
waste a reveal), share deductions, and avoid revealing cells the board already proves are mines.

Constructor
-----------
`Minesweeper(height=9, width=9, mines=10, on_mine="end", reveals_per_round=1, free_flags=True,
safe_opening=True)`

- `height`, `width`: rows and columns, each 1..99. The classic difficulties are 9 x 9 with 10
  mines (beginner), 16 x 16 with 40 (intermediate) and 16 rows x 30 columns with 99 (expert).
- `mines`: 1 .. height * width - 1.
- `on_mine`: `"end"` (the standard rule: revealing a mine loses the game and the run ends) or
  `"continue"` (the cell shows `*`, the hit is counted and play goes on, so every game can be
  cleared and the round it was cleared in is always defined).
- `reveals_per_round`: reveals per agent per round (>= 1); per turn under `commit: async`.
- `free_flags`: flags and unflags do not count toward `reveals_per_round` (False: they do).
- `safe_opening`: `reset` opens a safe start, the first cell of a seeded shuffle whose number is
  0 (a safe cell with a number when there is no 0), flood-filled. False: the board starts hidden.

`reset` places the mines with the world rng (a seeded sample of the cells), so the same seed gives
the same minefield and opening in every arm of an experiment.

Observation format (exact; `parse_observation` inverts it)
----------------------------------------------------------
One text part, e.g. for a 12 x 9 board with 15 mines after the opening:

    Board (12 columns x 9 rows; x = column 0-11, y = row 0-8; # hidden, F flagged, . empty, 1-8 adjacent mines, * exploded mine):
                11
      012345678901
    0 ############
    1 ############
    2 ###22111####
    3 ###1...12###
    4 ###11...1###
    5 ####1...112#
    6 ####1.....1#
    7 ####1211..11
    8 #######1....

    Mines: 15  Flags: 0  Exploded: 0  Hidden: 64

The tens line appears only when `width > 10` (tens digit above every column >= 10). Row labels are
right-aligned to the width of the largest row index. `observation.private` is empty: the board is
public and identical for every agent.

Actions and commit
------------------
- `reveal(x, y)`: opens the cell; a 0 flood-fills its neighbours. Feedback (agent-visible):
  `{"result": "opened", "x", "y", "value", "cells_opened"}`, `{"result": "exploded", "x", "y"}`,
  `{"result": "already_revealed", "x", "y", "collision"}` (see below), or an error: `flagged`
  (unflag it first), `out_of_bounds`, `reveals_per_round`, `game_over` (the game is already won
  or lost, possibly earlier in the same commit batch; no action is accepted after that).
- `flag(x, y)` / `unflag(x, y)`: mark or unmark a hidden cell. Flags are shared board state (every
  agent sees them, anyone may remove one), never checked against the minefield: feedback is
  `flagged` / `unflagged` or the error `not_hidden` / `not_flagged`, never whether a flag is right.
- `validate(agent, action, pending)` refuses at call time an out-of-bounds cell and a reveal (or,
  with `free_flags: False`, any action) beyond `reveals_per_round` given the agent's buffered
  `pending` actions; the commit enforces the same limit in commit order.
- A repeat reveal of an open cell is a **collision** (`collision: true`) when another agent's
  reveal opened it (directly or by its flood) after this agent last observed the board, and
  **stale** otherwise (the cell was visible to the agent, was in the opening, or the agent's own
  earlier reveal opened it). The runner observes at the start of every turn, and `observe`
  records how many reveals the agent has seen, so the rule needs no knowledge of the commit mode:
  under `commit: round_end` actions apply together at the end of the round in the seeded commit
  order, nobody sees the round's reveals while choosing, and a cell revealed twice in a round is
  a collision; under `commit: immediate` each turn sees the earlier turns' results, so repeating
  one is stale. Unit code that calls the actions without `observe` is judged as round_end.
- `claim_key(agent, action) -> "cell:<x>,<y>"`, so the registry and claim policies apply.

The world has no requirements (`requires` is empty): any commit mode (`async` included, see
below), scheduler, topology and agent count is a legitimate variation. Its task text (`description()`) therefore says nothing
about turns, their order, when actions apply or who reads a post; an experiment states how its
rounds work in `system_prompt_append`, as `examples/minesweeper.yaml` does.

Provable cells and allocation accounting (evaluator-only)
---------------------------------------------------------
`deduce(rows)` returns the hidden cells the visible board proves safe and proves mined, by
iterated single-clue rules and the pairwise subset rule (a lower bound on what exact inference
finds; flags are ignored because they may be wrong). `begin_round` stores it for the board at the
start of the round (`round_safe`, `round_mines`). Then, per round:

- `ideal_reveals`: what a perfectly coordinated team could open without guessing,
  `min(n_agents * reveals_per_round, |round_safe|)`, or 1 (one forced guess) when nothing is
  provably safe; 0 in a round that starts with the game already over (the runner always plays
  round 1, even when the opening cleared the board). Under `commit: immediate` later turns see
  more than the round-start board, so this and the classification below are judged against the
  round-start board, a lower bound.
- each reveal that opened a cell is `certain` (it was in `round_safe`) or a `guess`;
- each mine hit is `provable` (the cell was in `round_mines`: a reasoning error), `unforced`
  (provably safe cells were available) or `forced` (nothing was provably safe).

`end_round(r)` returns the round's tally and the board's progress, logged as a `round_resolved`
event (agents never see it): `n_agents`, `provably_safe`, `provable_mines`, `ideal_reveals`,
`reveals`, `productive`, `certain`, `guesses`, `collisions`, `stale`, `cells_opened`,
`mine_hits`, `provable_mine_hits`, `unforced_mine_hits`, `forced_mine_hits`, `opened`,
`safe_cells`, `cleared`, `won`, `lost`. `n_agents` is the agents given at reset (agents killed
by an intervention are not subtracted).

`terminal()` is true once the game is won, or lost under `on_mine: "end"`.
`score()` -> `won`, `lost`, `cleared` (share of safe cells revealed), `cleared_round` (the round
the last safe cell opened, or None), `first_mine_round` (the round of the first mine hit: the
round the game is lost under the standard rule, or None), `allocation_efficiency`
(productive reveals / ideal reveals; it can exceed 1 when several agents guess safely in a round
whose ideal counted one guess), `correct_flags`, `wrong_flags` and the cumulative counters
`reveals`, `productive_reveals`, `certain_reveals`, `guess_reveals`, `ideal_reveals`,
`cells_opened`, `collisions`, `stale_reveals`, `mine_hits`, `provable_mine_hits`,
`unforced_mine_hits`, `forced_mine_hits`, `flags`, `unflags`, `refused`.
`verify()` -> the minefield (rows of `x` and `.`), the sizes, `safe_cells` and `initial_opened`.
`render_state()` -> the board as the replay page's grid (hidden mines shown as `M`), plus the
round's provable counts and the score.

Metrics (entry points; all fold the `round_resolved` events, so `swarmlab replay` reproduces
them): `minesweeper.cleared`, `minesweeper.collisions`, `minesweeper.productive_rate`,
`minesweeper.allocation_efficiency`, `minesweeper.mine_hits` (see each class).

`report_sections` adds one table to `swarmlab report`: per run, the outcome, rounds to clear,
first mine round, mine hits by kind, collisions and allocation efficiency.

Async mode (`commit: async`, swarmlab/runner.py)
------------------------------------------------
No rounds: every agent loops over its own turns, concurrently, each action applying at once; a
tick is one completed turn (`begin_round(t)` when tick t opens, `close_turn(agent, t)` and
`end_round(t)` when the agent's turn closes it). The runner sets `commit_mode` before `reset`,
and only then does the world keep per-turn state (`turn_view`, `turn_tally`, `closing`, created
by `reset` in async mode only, so round-based snapshots are unchanged):

- `observe(agent)` starts the agent's turn: its `reveals_per_round` budget is reset (a per-turn
  cap), its tally zeroed, and `deduce` is run on the board it is shown (cached for the world and
  recomputed only when a reveal changed the board since the last one; flags never change it).
  `begin_round` only sets the round (tick) number: other agents' turns are in flight.
- A reveal is judged against what the agent observed at its turn start, not a shared round
  start: `certain` / `guess` by that turn's provably safe cells, a mine hit `provable` /
  `unforced` / `forced` by that turn's provable mines and safe cells. Collisions keep the rule
  above (another agent's reveal opened the cell after this agent last observed); a cell the
  agent saw open, or opened itself earlier, is stale.
- `end_round(t)` returns the closing agent's turn: its tally (the reveals of that turn, which
  may have committed during earlier ticks), `agent`, `agent_turns: 1` (`productive_rate` divides
  by agent-turns, `n_agents` per round otherwise), `provably_safe` / `provable_mines` as that
  agent saw them, and `ideal_reveals = min(reveals_per_round, provably safe cells it saw)`, or 1
  when it saw none (a forced guess), or 0 when it observed a finished game; so
  `allocation_efficiency` keeps its meaning, per turn. `round_safe` / `round_mines` (the replay
  page's counts) become that agent's turn-start sets.
- `cleared_round` / `first_mine_round` are ticks (the tick open when the cell opened); the run
  ends at the close of that same tick, so `cleared_round == last_round` for a won game.
- The cumulative counters of `score()` count every committed action, as in the other modes;
  the `round_resolved` tallies (and the metrics folded from them) count completed turns. They
  differ only by the commits of turns cancelled when the run ends (or lost at a resume) while
  in flight; the fake scripts act only after their one model call, so for them they agree.
- The task text says "per turn" instead of "per round" for the reveal limit.

Fake scripts (no network, no spend): `fake:minesweeper_random` reveals a random provably safe
cell each turn (when there is none, a random hidden cell not proven to be a mine); `fake:minesweeper_allocator` makes
agent k take the k-th provably safe cell in row-major order and post its pick when `post` is
offered. Neither flags. They are plumbing checks and uncoordinated / coordinated references.
Both read the board from the latest observation in the prompt, so under async an agent picks
from the board of its own turn start (the allocator's k-th cell is the k-th of what it saw).

Snapshots and forks
-------------------
Snapshots carry game state only (minefield, board, round tallies, counters); constructor params
come from the spec a fork or resume runs under. A snapshot whose board size or mine count differs
from the configured one is refused at restore. `on_mine`, `reveals_per_round` and `free_flags`
may change across a fork; `safe_opening` only acts at reset.
"""
from __future__ import annotations

import random
import re
from collections.abc import Sequence
from typing import Any, ClassVar

from ..ids import AgentId
from ..metrics.base import Metric
from ..view import Observation, text_observation
from .base import Ack, Action, Outcome, World, tool

HIDDEN, FLAG, EXPLODED, EMPTY = "#", "F", "*", "."
BOARD_HEADER = "Board ("
LEGEND = "# hidden, F flagged, . empty, 1-8 adjacent mines, * exploded mine"
MAX_SIDE = 99
# Replay-page colours: blues darken with the number; hidden mines (evaluator view) are "M".
PALETTE_CSS = {HIDDEN: "#9ca3af", FLAG: "#f59e0b", EXPLODED: "#dc2626", EMPTY: "#f9fafb",
               "M": "#4b5563", "1": "#bfdbfe", "2": "#93c5fd", "3": "#60a5fa", "4": "#3b82f6",
               "5": "#2563eb", "6": "#1d4ed8", "7": "#1e40af", "8": "#1e3a8a"}
_SUMMARY_RE = re.compile(r"^Mines: (\d+)  Flags: (\d+)  Exploded: (\d+)  Hidden: (\d+)$")
_ROW_RE = re.compile(r"^ *(\d+) (\S+)$")
_TALLY_KEYS = ("reveals", "productive", "certain", "guesses", "collisions", "stale",
               "cells_opened", "mine_hits", "provable_mine_hits", "unforced_mine_hits",
               "forced_mine_hits")


# ---- pure helpers ----------------------------------------------------------------------------

def neighbours(x: int, y: int, width: int, height: int) -> list[tuple[int, int]]:
    """The up to 8 cells around (x, y), row by row."""
    return [(x + dx, y + dy) for dy in (-1, 0, 1) for dx in (-1, 0, 1)
            if (dx or dy) and 0 <= x + dx < width and 0 <= y + dy < height]


def deduce(rows: Sequence[Sequence[str]]) -> tuple[set[tuple[int, int]], set[tuple[int, int]]]:
    """(provably safe, provably mined) unknown cells of a visible board, as (x, y) sets.

    `rows` are board rows of observation symbols. Unknown cells are hidden or flagged ones (flags
    are ignored: they may be wrong, and a flood fill never opens a flagged cell, so an empty `.`
    cell next to a flag is a 0 clue that proves it safe); exploded cells count as known mines.
    Rules, iterated to a fixed point: a clue whose remaining mines are 0 makes its unknown
    neighbours safe, one whose remaining mines equal its unknown neighbours makes them mines; when
    neither applies, for two clues with unknown sets A strictly inside B, the cells of B - A hold
    (need_B - need_A) mines, so they are all safe or all mines at the extremes. Sound, not
    complete.
    """
    height = len(rows)
    width = len(rows[0]) if height else 0
    safe: set[tuple[int, int]] = set()
    mines: set[tuple[int, int]] = set()
    clues = [(x, y, 0 if rows[y][x] == EMPTY else int(rows[y][x])) for y in range(height)
             for x in range(width) if rows[y][x] == EMPTY or rows[y][x].isdigit()]
    nbrs = {(x, y): neighbours(x, y, width, height) for x, y, _ in clues}
    while True:
        cons: dict[frozenset, int] = {}
        for x, y, n in clues:
            unknown = frozenset(p for p in nbrs[(x, y)] if rows[p[1]][p[0]] in (HIDDEN, FLAG)
                                and p not in safe and p not in mines)
            if unknown:
                known = sum(1 for p in nbrs[(x, y)] if p in mines or rows[p[1]][p[0]] == EXPLODED)
                cons[unknown] = n - known
        new_safe: set = set()
        new_mines: set = set()
        for cells, need in cons.items():
            if need == 0:
                new_safe |= cells
            elif need == len(cells):
                new_mines |= cells
        if not new_safe and not new_mines:
            items = list(cons.items())
            for a, need_a in items:
                for b, need_b in items:
                    if a < b:
                        rest, need = b - a, need_b - need_a
                        if need == 0:
                            new_safe |= rest
                        elif need == len(rest):
                            new_mines |= rest
        new_safe -= safe
        new_mines -= mines
        if not new_safe and not new_mines:
            return safe, mines
        safe |= new_safe
        mines |= new_mines


def column_labels(width: int) -> list[str]:
    """The label lines above the board columns (a tens line only when width > 10)."""
    ones = "".join(str(x % 10) for x in range(width))
    if width <= 10:
        return [ones]
    return ["".join(str(x // 10) if x >= 10 else " " for x in range(width)), ones]


def parse_observation(text: str) -> dict[str, Any]:
    """Board rows and counters from a Minesweeper observation (or a larger text containing one;
    the last occurrence wins): `{"rows", "mines", "flags", "exploded", "hidden"}`, or {}."""
    idx = text.rfind(BOARD_HEADER)
    if idx < 0:
        return {}
    rows: list[str] = []
    out: dict[str, Any] = {}
    for line in text[idx:].split("\n")[1:]:
        m = _ROW_RE.match(line)
        if m and int(m.group(1)) == len(rows):
            rows.append(m.group(2))
            continue
        s = _SUMMARY_RE.match(line.strip())
        if s:
            out = dict(zip(("mines", "flags", "exploded", "hidden"), map(int, s.groups())))
            break
    return {"rows": rows, **out} if rows else {}


# ---- world ------------------------------------------------------------------------------------

class Minesweeper(World):
    """A shared Minesweeper board: reveal every safe cell of one hidden minefield; size and mines set the difficulty."""

    entry_point: ClassVar[str | None] = "minesweeper"
    name = "minesweeper"

    def __init__(self, height: int = 9, width: int = 9, mines: int = 10, on_mine: str = "end",
                 reveals_per_round: int = 1, free_flags: bool = True,
                 safe_opening: bool = True) -> None:
        for key, v in (("height", height), ("width", width)):
            if not isinstance(v, int) or isinstance(v, bool) or not 1 <= v <= MAX_SIDE:
                raise ValueError(f"{key} must be an integer in 1..{MAX_SIDE}, got {v!r}")
        if not isinstance(mines, int) or isinstance(mines, bool) or not 1 <= mines < height * width:
            raise ValueError(f"mines must be an integer in 1..{height * width - 1} for a "
                             f"{width} x {height} board, got {mines!r}")
        if on_mine not in ("end", "continue"):
            raise ValueError(f"on_mine must be 'end' or 'continue', got {on_mine!r}")
        if not isinstance(reveals_per_round, int) or reveals_per_round < 1:
            raise ValueError(f"reveals_per_round must be a positive integer, got {reveals_per_round!r}")
        self.height, self.width, self.mines = height, width, mines
        self.on_mine = on_mine
        self.reveals_per_round = reveals_per_round
        self.free_flags = bool(free_flags)
        self.safe_opening = bool(safe_opening)
        # ---- state (snapshotted) ----
        self.agents: list[str] = []
        self.minefield: list[list[bool]] = []
        self.numbers: list[list[int]] = []
        self.board: list[list[str]] = []          # what agents see
        self.seq = 0                              # reveals committed so far that changed the board
        self.opened_seq: list[list[int]] = []     # seq of the reveal that opened a cell (-1: hidden; 0: opening)
        self.opened_by: list[list[str | None]] = []   # agent whose reveal opened it (None: the opening)
        self.seen_seq: dict[str, int] = {}        # seq when each agent last observed the board, this round
        self.round_start_seq = 0
        self.round_live = False                   # the game was still on at the start of the round
        self.round = 0
        self.lost = False
        self.cleared_round: int | None = None
        self.first_mine_round: int | None = None
        self.initial_opened = 0
        self.round_safe: list[tuple[int, int]] = []
        self.round_mines: list[tuple[int, int]] = []
        self.round_used: dict[str, int] = {}      # limited actions per agent this round
        self.tally: dict[str, int] = dict.fromkeys(_TALLY_KEYS, 0)   # this round
        self.stats: dict[str, int] = {}
        self._zero_stats()

    _skip_in_snapshot: ClassVar[tuple[str, ...]] = (
        "params", "height", "width", "mines", "on_mine", "reveals_per_round", "free_flags",
        "safe_opening")

    def _zero_stats(self) -> None:
        self.stats = dict.fromkeys(
            ("reveals", "productive_reveals", "certain_reveals", "guess_reveals", "ideal_reveals",
             "cells_opened", "collisions", "stale_reveals", "mine_hits", "provable_mine_hits",
             "unforced_mine_hits", "forced_mine_hits", "flags", "unflags", "refused"), 0)

    def restore(self, blob: bytes) -> None:
        """Restore game state, then refuse a board whose size or mine count is not the configured one."""
        super().restore(blob)
        self._deduced = None
        h, w = len(self.minefield), len(self.minefield[0]) if self.minefield else 0
        n = sum(map(sum, self.minefield))
        if self.minefield and (h, w, n) != (self.height, self.width, self.mines):
            raise ValueError(
                f"minesweeper: the snapshot has a {w} x {h} board with {n} mines but this world is "
                f"configured as {self.width} x {self.height} with {self.mines}; height, width and "
                "mines cannot change across a fork or resume")

    # ---- lifecycle ---------------------------------------------------------------------------
    def reset(self, rng: random.Random, agents: list[AgentId]) -> None:
        self.agents = list(agents)
        cells = [(x, y) for y in range(self.height) for x in range(self.width)]
        mined = set(rng.sample(cells, self.mines))
        self.minefield = [[(x, y) in mined for x in range(self.width)] for y in range(self.height)]
        self.numbers = [[sum(self.minefield[ny][nx] for nx, ny in self._nbrs(x, y))
                         for x in range(self.width)] for y in range(self.height)]
        self.board = [[HIDDEN] * self.width for _ in range(self.height)]
        self.seq = 0
        self.opened_seq = [[-1] * self.width for _ in range(self.height)]
        self.opened_by = [[None] * self.width for _ in range(self.height)]
        self.seen_seq, self.round_start_seq, self.round_live = {}, 0, False
        self.round, self.lost = 0, False
        self.cleared_round = self.first_mine_round = None
        self.round_safe, self.round_mines, self.round_used = [], [], {}
        self.tally = dict.fromkeys(_TALLY_KEYS, 0)
        self._zero_stats()
        self._deduced = None
        if self.commit_mode == "async":  # created only then, so other modes' snapshots are as before
            self.turn_view: dict[str, dict] = {}    # per agent: what it saw at its turn start
            self.turn_tally: dict[str, dict] = {}   # per agent: its turn's tally so far
            self.closing: str | None = None         # the agent whose turn closes the tick
        order = cells[:]
        rng.shuffle(order)       # drawn even when unused, so safe_opening never shifts the rng
        self.initial_opened = 0
        if self.safe_opening:
            start = next(((x, y) for x, y in order if not self.minefield[y][x] and self.numbers[y][x] == 0),
                         None) or next((x, y) for x, y in order if not self.minefield[y][x])
            self.initial_opened = self._open(*start, None)
            if self.won():
                self.cleared_round = 0

    def begin_round(self, round: int) -> None:
        self.round = round
        if self.commit_mode == "async":  # a tick: turns in flight keep their per-turn state
            return
        self.round_used, self.seen_seq, self.round_start_seq = {}, {}, self.seq
        self.tally = dict.fromkeys(_TALLY_KEYS, 0)
        self.round_live = not self.terminal()
        if not self.round_live:
            self.round_safe, self.round_mines = [], []
            return
        safe, mines = deduce(self.board)
        self.round_safe = sorted(safe, key=lambda p: (p[1], p[0]))
        self.round_mines = sorted(mines, key=lambda p: (p[1], p[0]))

    def close_turn(self, agent: AgentId, round: int) -> None:
        """Async commit: `agent`'s turn closes tick `round` (resolved by `end_round`)."""
        self.closing = agent

    def _end_turn(self, round: int) -> dict:
        """Async commit: tick `round` is the closing agent's turn, judged against the board it
        observed at its turn start."""
        n, agent = len(self.agents), self.closing
        seen = self.turn_view.get(agent) or {"safe": [], "mines": [], "live": False}
        tally = self.turn_tally.pop(agent, None) or dict.fromkeys(_TALLY_KEYS, 0)
        ideal = (min(self.reveals_per_round, len(seen["safe"])) or 1) if seen["live"] else 0
        self.stats["ideal_reveals"] += ideal
        self.round_safe = [tuple(p) for p in seen["safe"]]   # the replay page's provable counts
        self.round_mines = [tuple(p) for p in seen["mines"]]
        self.closing = None
        opened = self.safe_cells() - self._safe_left()
        return {"round": round, "agent": agent, "n_agents": n, "agent_turns": 1,
                "provably_safe": len(seen["safe"]), "provable_mines": len(seen["mines"]),
                "ideal_reveals": ideal, **tally, "opened": opened, "safe_cells": self.safe_cells(),
                "cleared": opened / self.safe_cells(), "won": self.won(), "lost": self.lost}

    def end_round(self, round: int) -> dict:
        """Tally round `round` against the round-start board (logged as round_resolved)."""
        if self.commit_mode == "async":
            return self._end_turn(round)
        n = len(self.agents)
        ideal = (min(n * self.reveals_per_round, len(self.round_safe)) or 1) if self.round_live else 0
        self.stats["ideal_reveals"] += ideal
        opened = self.safe_cells() - self._safe_left()
        return {"round": round, "n_agents": n, "provably_safe": len(self.round_safe),
                "provable_mines": len(self.round_mines), "ideal_reveals": ideal, **self.tally,
                "opened": opened, "safe_cells": self.safe_cells(),
                "cleared": opened / self.safe_cells(), "won": self.won(), "lost": self.lost}

    def terminal(self) -> bool:
        return self.lost or self.won()

    # ---- helpers -----------------------------------------------------------------------------
    def _nbrs(self, x: int, y: int) -> list[tuple[int, int]]:
        return neighbours(x, y, self.width, self.height)

    def _open(self, x: int, y: int, by: str | None) -> int:
        """Open a safe cell, flood-filling from 0s, as reveal `self.seq` by agent `by` (None: the
        opening); returns the number of cells opened."""
        stack, opened = [(x, y)], 0
        while stack:
            cx, cy = stack.pop()
            if self.board[cy][cx] != HIDDEN:
                continue
            n = self.numbers[cy][cx]
            self.board[cy][cx] = EMPTY if n == 0 else str(n)
            self.opened_seq[cy][cx], self.opened_by[cy][cx] = self.seq, by
            opened += 1
            if n == 0:
                stack.extend(self._nbrs(cx, cy))
        return opened

    def safe_cells(self) -> int:
        return self.height * self.width - self.mines

    def _safe_left(self) -> int:
        return sum(1 for y in range(self.height) for x in range(self.width)
                   if not self.minefield[y][x] and self.board[y][x] in (HIDDEN, FLAG))

    def won(self) -> bool:
        return bool(self.board) and self._safe_left() == 0

    def _limited(self, name: str) -> bool:
        return name == "reveal" or not self.free_flags

    def _cell_error(self, x: Any, y: Any) -> str | None:
        if not isinstance(x, int) or isinstance(x, bool) or not isinstance(y, int) or isinstance(y, bool):
            return "out_of_bounds: x and y must be integers"
        if not (0 <= x < self.width and 0 <= y < self.height):
            return (f"out_of_bounds: ({x}, {y}) is outside the board (x = column 0-{self.width - 1}, "
                    f"y = row 0-{self.height - 1})")
        return None

    def _per(self) -> str:
        return "turn" if self.commit_mode == "async" else "round"

    def _limit_text(self) -> str:
        n = self.reveals_per_round
        what = "reveal" if self.free_flags else "action"
        return f"at most {n} {what}{'s' if n != 1 else ''} per {self._per()}"

    # ---- agent-facing ------------------------------------------------------------------------
    def description(self) -> str:
        n, k = len(self.agents), self.reveals_per_round
        team = (f"You are one of {n} agents ({', '.join(self.agents)}) playing one shared game of "
                "Minesweeper." if n > 1 else "You are playing a game of Minesweeper.")
        flags = ("Flags are visible to every agent, and anyone may remove one. " if n > 1 else "")
        per = self._per()
        limit = (f"Each agent may reveal at most {k} cell{'s' if k != 1 else ''} per {per}; flagging "
                 "and unflagging are free." if self.free_flags else
                 f"Each agent may take at most {k} action{'s' if k != 1 else ''} per {per} (reveal, "
                 "flag or unflag).")
        if n == 1:
            limit = limit.replace("Each agent", "You")
        mine = ("Revealing a mine ends the game: it is lost." if self.on_mine == "end" else
                "Revealing a mine is a penalty: the cell shows * and the game goes on.")
        win = "The team wins" if n > 1 else "You win"
        return (
            f"{team} The board has {self.width} columns and {self.height} rows and hides "
            f"{self.mines} mines. A revealed number is how many of the neighbouring cells "
            "(up to 8) hold mines. reveal(x, y) opens a cell (x = column, y = row, both from 0); "
            "revealing a cell with no neighbouring mines also opens its neighbours. flag(x, y) "
            "and unflag(x, y) mark and unmark a hidden cell you believe is a mine; a flagged cell "
            f"cannot be revealed until it is unflagged. {flags}{limit} {mine} {win} when every "
            "cell without a mine is revealed."
        )

    def observe(self, agent: AgentId) -> Observation:
        # The runner observes at the start of every turn (after the earlier turns' commits under
        # `commit: immediate`), so this records what the agent has seen; it only classifies repeat
        # reveals (collision vs stale) and never changes what any agent is shown.
        self.seen_seq[agent] = self.seq
        if self.commit_mode == "async":
            self._start_turn(agent)
        label_w = len(str(self.height - 1))
        pad = " " * (label_w + 1)
        lines = [(f"{BOARD_HEADER}{self.width} columns x {self.height} rows; x = column "
                  f"0-{self.width - 1}, y = row 0-{self.height - 1}; {LEGEND}):")]
        lines += [pad + lab for lab in column_labels(self.width)]
        lines += [f"{y:>{label_w}} {''.join(row)}" for y, row in enumerate(self.board)]
        flat = [c for row in self.board for c in row]
        lines += ["", (f"Mines: {self.mines}  Flags: {flat.count(FLAG)}  "
                       f"Exploded: {flat.count(EXPLODED)}  Hidden: {flat.count(HIDDEN)}")]
        return text_observation("\n".join(lines))

    def _deduce_now(self) -> tuple[list[tuple[int, int]], list[tuple[int, int]]]:
        """`deduce` of the current board (sorted row-major), recomputed only when a reveal changed
        the board since the last call (flags never change it: `deduce` ignores them)."""
        memo = getattr(self, "_deduced", None)
        if memo is None or memo[0] != self.seq:
            safe, mines = deduce(self.board)
            key = lambda p: (p[1], p[0])
            memo = self._deduced = (self.seq, sorted(safe, key=key), sorted(mines, key=key))
        return memo[1], memo[2]

    def _start_turn(self, agent: AgentId) -> None:
        """Async commit, at the agent's observation (its turn start): a fresh reveal budget and
        tally, and the provable cells of the board it is shown, against which its reveals are
        judged."""
        self.round_used.pop(agent, None)
        self.turn_tally[agent] = dict.fromkeys(_TALLY_KEYS, 0)
        live = not self.terminal()
        safe, mines = self._deduce_now() if live else ([], [])
        self.turn_view[agent] = {"safe": list(safe), "mines": list(mines), "live": live}

    def _judged(self, agent: AgentId) -> tuple[Sequence[tuple[int, int]], Sequence[tuple[int, int]], dict]:
        """(provably safe, provably mined, tally) a reveal by `agent` is judged and counted
        against: under async the agent's turn-start board and turn tally, else the round's."""
        if self.commit_mode == "async" and agent in self.turn_view:
            seen = self.turn_view[agent]
            return ([tuple(p) for p in seen["safe"]], [tuple(p) for p in seen["mines"]],
                    self.turn_tally.setdefault(agent, dict.fromkeys(_TALLY_KEYS, 0)))
        return self.round_safe, self.round_mines, self.tally

    def validate(self, agent: AgentId, action: Action, pending: Sequence[Action] = ()) -> Ack:
        ack = super().validate(agent, action)
        if not ack.ok:
            return ack
        err = self._cell_error(action.args.get("x"), action.args.get("y"))
        if err:
            return Ack(ok=False, error=err)
        if self._limited(action.name) and \
                sum(1 for a in pending if self._limited(a.name)) >= self.reveals_per_round:
            return Ack(ok=False, error=f"reveals_per_round: {self._limit_text()}; this one would be "
                                       "rejected at commit")
        return ack

    def _refuse(self, error: str, detail: str | None = None, **cell: int) -> Outcome:
        self.stats["refused"] += 1
        fb: dict[str, Any] = {"error": error, **cell}
        if detail:
            fb["detail"] = detail
        return Outcome(accepted=False, feedback=fb)

    def _check(self, x: Any, y: Any) -> Outcome | None:
        """Commit-time checks shared by the actions: a refusal, or None."""
        err = self._cell_error(x, y)
        if err:
            return self._refuse("out_of_bounds", err.split(": ", 1)[1])
        if self.terminal():          # also later in the commit batch that ended the game
            why = "the game is lost: a mine was revealed" if self.lost else "the game is won"
            return self._refuse("game_over", why, x=x, y=y)
        return None

    def _count(self, agent: AgentId, name: str, x: int, y: int) -> Outcome | None:
        """Count a limited action against the agent's round budget (a refusal when it is spent).
        Called after the cell checks, so a refused action never uses up the budget."""
        if self._limited(name):
            if self.round_used.get(agent, 0) >= self.reveals_per_round:
                return self._refuse("reveals_per_round", self._limit_text(), x=x, y=y)
            self.round_used[agent] = self.round_used.get(agent, 0) + 1
        return None

    @tool("reveal", "Reveal cell (x = column, y = row).", {"x": "integer", "y": "integer"})
    def reveal(self, agent: AgentId, x: int, y: int) -> Outcome:
        if (out := self._check(x, y)) is not None:
            return out
        cell = self.board[y][x]
        if cell == FLAG:
            return self._refuse("flagged", "unflag the cell before revealing it", x=x, y=y)
        if (out := self._count(agent, "reveal", x, y)) is not None:
            return out
        safe_cells, mine_cells, tally = self._judged(agent)
        tally["reveals"] += 1
        self.stats["reveals"] += 1
        if cell != HIDDEN:
            # A collision: another agent's reveal opened the cell after this agent last saw the
            # board (the same round under round_end). Stale: it was already visible to this agent,
            # or its own earlier reveal opened it.
            seen = self.seen_seq.get(agent, self.round_start_seq)
            collision = self.opened_by[y][x] not in (None, agent) and self.opened_seq[y][x] > seen
            tally["collisions" if collision else "stale"] += 1
            self.stats["collisions" if collision else "stale_reveals"] += 1
            return Outcome(accepted=True, feedback={"result": "already_revealed", "x": x, "y": y,
                                                    "collision": collision})
        self.seq += 1
        if self.minefield[y][x]:
            self.board[y][x] = EXPLODED
            self.opened_seq[y][x], self.opened_by[y][x] = self.seq, agent
            kind = ("provable" if (x, y) in mine_cells else
                    "unforced" if safe_cells else "forced")
            for key in ("mine_hits", f"{kind}_mine_hits"):
                tally[key] += 1
                self.stats[key] += 1
            if self.first_mine_round is None:
                self.first_mine_round = self.round
            self.lost = self.lost or self.on_mine == "end"
            return Outcome(accepted=True, feedback={"result": "exploded", "x": x, "y": y})
        kind = "certain" if (x, y) in safe_cells else "guess"
        opened = self._open(x, y, agent)
        for key, stat in (("productive", "productive_reveals"),
                          ("certain" if kind == "certain" else "guesses", f"{kind}_reveals")):
            tally[key] += 1
            self.stats[stat] += 1
        tally["cells_opened"] += opened
        self.stats["cells_opened"] += opened
        if self.cleared_round is None and self.won():
            self.cleared_round = self.round
        return Outcome(accepted=True, feedback={"result": "opened", "x": x, "y": y,
                                                "value": self.board[y][x], "cells_opened": opened})

    @tool("flag", "Flag hidden cell (x = column, y = row) as a suspected mine.",
          {"x": "integer", "y": "integer"})
    def flag(self, agent: AgentId, x: int, y: int) -> Outcome:
        if (out := self._check(x, y)) is not None:
            return out
        if self.board[y][x] != HIDDEN:
            return self._refuse("not_hidden", "only a hidden cell can be flagged", x=x, y=y)
        if (out := self._count(agent, "flag", x, y)) is not None:
            return out
        self.board[y][x] = FLAG
        self.stats["flags"] += 1
        return Outcome(accepted=True, feedback={"result": "flagged", "x": x, "y": y})

    @tool("unflag", "Remove the flag from cell (x = column, y = row).",
          {"x": "integer", "y": "integer"})
    def unflag(self, agent: AgentId, x: int, y: int) -> Outcome:
        if (out := self._check(x, y)) is not None:
            return out
        if self.board[y][x] != FLAG:
            return self._refuse("not_flagged", "the cell has no flag", x=x, y=y)
        if (out := self._count(agent, "unflag", x, y)) is not None:
            return out
        self.board[y][x] = HIDDEN
        self.stats["unflags"] += 1
        return Outcome(accepted=True, feedback={"result": "unflagged", "x": x, "y": y})

    def claim_key(self, agent: AgentId, action: Action) -> str | None:
        x, y = action.args.get("x"), action.args.get("y")
        return f"cell:{x},{y}" if isinstance(x, int) and isinstance(y, int) else None

    # ---- evaluator-only ----------------------------------------------------------------------
    def score(self) -> dict:
        if not self.board:
            return {"won": 0, "lost": 0, "cleared": 0.0, "cleared_round": None,
                    "first_mine_round": None, "allocation_efficiency": None, "correct_flags": 0,
                    "wrong_flags": 0, **self.stats}
        flags = [(x, y) for y in range(self.height) for x in range(self.width) if self.board[y][x] == FLAG]
        right = sum(self.minefield[y][x] for x, y in flags)
        ideal = self.stats["ideal_reveals"]
        return {"won": int(self.won()), "lost": int(self.lost),
                "cleared": (self.safe_cells() - self._safe_left()) / self.safe_cells(),
                "cleared_round": self.cleared_round, "first_mine_round": self.first_mine_round,
                "allocation_efficiency": self.stats["productive_reveals"] / ideal if ideal else None,
                "correct_flags": right, "wrong_flags": len(flags) - right, **self.stats}

    def verify(self) -> dict:
        return {"minefield": ["".join("x" if m else "." for m in row) for row in self.minefield],
                "height": self.height, "width": self.width, "mines": self.mines,
                "safe_cells": self.safe_cells(), "initial_opened": self.initial_opened}

    def render_state(self) -> dict | None:
        """The board (hidden mines as `M`) for the replay page, with the round's provable counts."""
        if not self.board:
            return None
        grid = [["M" if self.minefield[y][x] and self.board[y][x] == HIDDEN else self.board[y][x]
                 for x in range(self.width)] for y in range(self.height)]
        sc = self.score()
        return {"grid": grid, "palette": PALETTE_CSS, "round": self.round,
                "provably safe at round start": len(self.round_safe),
                "provable mines at round start": len(self.round_mines),
                **{k: sc[k] for k in ("won", "lost", "cleared", "cleared_round", "first_mine_round",
                                      "mine_hits", "collisions", "wrong_flags")}}

    def report_sections(self, arms: dict[str, list[Any]]) -> list[str]:
        from ..report import NA, f, table

        rows = []
        for arm, runs in arms.items():
            for run in runs:
                sc = run.score if isinstance(run.score, dict) else {}
                hits = "/".join(str(sc.get(k, NA)) for k in
                                ("provable_mine_hits", "unforced_mine_hits", "forced_mine_hits"))
                outcome = "won" if sc.get("won") else "lost" if sc.get("lost") else "unfinished"
                rows.append([arm, run.id, outcome, f(sc.get("cleared")),
                             sc.get("cleared_round") if sc.get("cleared_round") is not None else NA,
                             sc.get("first_mine_round") if sc.get("first_mine_round") is not None else NA,
                             sc.get("mine_hits", NA), hits, sc.get("collisions", NA),
                             f(sc.get("allocation_efficiency"))])
        head = ["arm", "run", "outcome", "cleared", "cleared_round", "first_mine_round",
                "mine_hits", "provable/unforced/forced", "collisions", "allocation_efficiency"]
        return ["## Minesweeper: outcome, mines and allocation", "",
                ("`cleared_round` is the round the last safe cell opened; `first_mine_round` is the "
                 "round the game is lost under the standard rule. A provable hit revealed a cell the "
                 "board proved to be a mine; an unforced one guessed while provably safe cells were "
                 "available. `allocation_efficiency` is productive reveals over what a perfectly "
                 "coordinated team could open without guessing."), "", *table(head, rows), ""]


# ---- metrics ----------------------------------------------------------------------------------

class _MinesFold(Metric):
    """Folds the world's `round_resolved` events (one per round, see `Minesweeper.end_round`)."""

    entry_point: ClassVar[str | None] = None

    def __init__(self) -> None:
        self.last: dict = {}
        self.sums: dict[str, float] = {}
        self.agent_rounds = 0

    def update(self, event: Any) -> None:
        if getattr(event, "type", None) == "round_resolved":
            self.last = dict(event.data)
            # async ticks are one agent-turn each (`agent_turns`); a round is `n_agents` turns
            self.agent_rounds += int(event.data.get("agent_turns", event.data["n_agents"]))
            for k in ("ideal_reveals", *_TALLY_KEYS):
                self.sums[k] = self.sums.get(k, 0) + event.data[k]


class Cleared(_MinesFold):
    entry_point: ClassVar[str | None] = "minesweeper.cleared"
    name = "minesweeper.cleared"
    description = "share of safe cells revealed (denominator: safe cells)"

    def value(self) -> tuple[float | None, int]:
        return (self.last["cleared"], self.last["safe_cells"]) if self.last else (None, 0)


class Collisions(_MinesFold):
    entry_point: ClassVar[str | None] = "minesweeper.collisions"
    name = "minesweeper.collisions"
    description = "cumulative reveals of a cell already opened earlier in the same round (denominator: reveals)"

    def value(self) -> tuple[float | None, int]:
        return (float(self.sums["collisions"]), int(self.sums["reveals"])) if self.last else (None, 0)


class ProductiveRate(_MinesFold):
    entry_point: ClassVar[str | None] = "minesweeper.productive_rate"
    name = "minesweeper.productive_rate"
    description = "reveals that opened a cell per agent-round so far (denominator: agent-rounds)"

    def value(self) -> tuple[float | None, int]:
        if not self.agent_rounds:
            return None, 0
        return self.sums["productive"] / self.agent_rounds, self.agent_rounds


class AllocationEfficiency(_MinesFold):
    entry_point: ClassVar[str | None] = "minesweeper.allocation_efficiency"
    name = "minesweeper.allocation_efficiency"
    description = "productive reveals over what a coordinated team could open without guessing (denominator: ideal reveals)"

    def value(self) -> tuple[float | None, int]:
        ideal = int(self.sums.get("ideal_reveals", 0))
        return (self.sums["productive"] / ideal, ideal) if ideal else (None, 0)


class MineHits(_MinesFold):
    entry_point: ClassVar[str | None] = "minesweeper.mine_hits"
    name = "minesweeper.mine_hits"
    description = "cumulative mines revealed (denominator: reveals)"

    def value(self) -> tuple[float | None, int]:
        return (float(self.sums["mine_hits"]), int(self.sums["reveals"])) if self.last else (None, 0)


# ---- fake LLM scripts (`fake:minesweeper_random`, `fake:minesweeper_allocator`) -----------------

def _fake(request: Any, rng: random.Random, allocate: bool) -> Any:
    import json

    from ..providers.base import text_of
    from ..providers.fake import RESULTS_PREFIX, _assistant_calls, _response

    tools = {t.name for t in request.tools}
    msgs = request.messages
    obs = next((text_of(m.content) for m in reversed(msgs)
                if m.role == "user" and BOARD_HEADER in text_of(m.content)), "")
    rows = parse_observation(obs).get("rows", [])
    safe, mines = deduce(rows) if rows else (set(), set())
    if not tools:
        return _response(request, rng, [], json.dumps({"provably_safe": len(safe)}))
    last_user = max((i for i, m in enumerate(msgs) if m.role == "user"
                     and not text_of(m.content).startswith(RESULTS_PREFIX)), default=-1)
    this_turn = [name for m in msgs[last_user + 1:] for name, _ in _assistant_calls(m)]
    calls: list[tuple[str, dict]] = []
    if rows and "reveal" not in this_turn and "reveal" in tools:
        ordered = sorted((p for p in safe if rows[p[1]][p[0]] == HIDDEN), key=lambda p: (p[1], p[0]))
        hidden = [(x, y) for y, r in enumerate(rows) for x, c in enumerate(r)
                  if c == HIDDEN and (x, y) not in mines]
        if allocate and ordered:
            system = next((text_of(m.content) for m in msgs if m.role == "system"), "")
            m = re.search(r"\bagent a(\d+)\b", system)
            x, y = ordered[(int(m.group(1)) if m else 0) % len(ordered)]
        else:
            x, y = rng.choice(ordered or hidden or [(0, 0)])
        calls.append(("reveal", {"x": x, "y": y}))
        if allocate and "post" in tools:
            calls.append(("post", {"text": f"revealing ({x}, {y})"}))
    if "end_turn" in tools:
        calls.append(("end_turn", {}))
    return _response(request, rng, calls)


def fake_random(request: Any, rng: random.Random) -> Any:
    """Fake script: reveal a random provably safe cell each turn (a random unproven one when none)."""
    return _fake(request, rng, allocate=False)


def fake_allocator(request: Any, rng: random.Random) -> Any:
    """Fake script: agent k reveals the k-th provably safe cell (row-major) and posts its pick."""
    return _fake(request, rng, allocate=True)
