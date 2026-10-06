"""FlagGame, text variant (docs/INTERFACE.md §8).

A hidden "truth" flag is one of `n_candidates` named candidate flags. Each agent privately sees
one `crop_h x crop_w` window of the truth (position withheld) and records guesses with the
`guess` tool. Nothing agent-visible ever says whether a guess is right; `score()` and `verify()`
are evaluator-only.

Flags
-----
A flag is a list of `height` strings of length `width`; each character is a colour letter.
The palette is the first `palette` letters of `COLOURS` ("rgbykwopcmnt": red, green, blue,
yellow, black/k, white, orange, purple, cyan, magenta, brown/n, teal), so `palette <= 12`.
Colours are lowercase so they never collide with "letters" candidate names.

Structured flags (truth and distractors) are one of: horizontal stripes (2-4 bands), vertical
stripes (2-4 bands), 2x2 blocks, 2x3 blocks (2 rows by 3 columns). Bands/blocks split the grid
as evenly as possible; neighbouring bands/blocks never share a colour.

The rival is a copy of the truth with `k = max(1, round((1 - rival_similarity) * height * width))`
distinct cells recoloured, each to a different colour than it had, so the rival always differs
from the truth (even at `rival_similarity=1.0`). Crops that avoid every recoloured cell are
contained in both truth and rival: that is how "the rival is favoured by some crops".
Distractors are fresh structured flags; generation retries until all candidates are distinct
(ValueError after 1000 attempts, e.g. if `n_candidates` exceeds what the palette allows).

Candidates are shuffled and then named in order: "letters" -> A, B, C, ... (max 26),
"numbers" -> 1, 2, 3, ... The name of the truth is therefore uniformly random.

Randomness
----------
`reset(rng, agents)` receives one stream (the runner passes `derive(seed, "world")`). Draws, in
order: truth, rival, distractors, shuffle, then one 64-bit value `s_i = rng.getrandbits(64)` per
agent in the order of `agents`. Agent i's crop is drawn from `derive(s_i, "private", agent_i)`.
Crops are thus a pure function of the world rng and the agent list (the contract's
`("private", agent)` root is honoured as a label under a world-derived seed, since `reset` has no
access to the run seed).

Observation format (exact; `parse_observation` inverts it)
----------------------------------------------------------
One text part, lines separated by "\\n":

    Candidate flags:
    <blank>
    A:
    <row 1 of A>
    ...
    <row height of A>
    <blank>
    B:
    ...
    <blank>
    Your crop:
    <crop row 1>
    ...
    <crop row crop_h>

Candidates are listed in name order. Every row is the colour letters with no separators. A
section header is a line ending in ":"; "Candidate flags:" is the preamble and "Your crop:" opens
the crop section. `observation.private == {"crop_y": y, "crop_x": x}` (top-left of the crop).

Decisions where the contract is silent
--------------------------------------
- Guesses for unknown candidates and guesses beyond `guess_limit` are rejected by `validate`; the
  `guess` method re-checks both (returning `accepted=False`) because several guesses buffered in
  one round are validated against round-start state. Guesses from agents not passed to `reset`
  are rejected the same way.
- `collective_status()["guess_counts"]` lists every candidate name (zeros included), in name
  order, so the key set carries no information.
- `my_status` for an agent with no state (including before `reset`) returns
  `{"current_guess": None, "guesses_made": 0}` when enabled.
- `score()["accuracy"]` divides by all agents given to `reset` (0.0 when there are none).
- Snapshots carry game state only. Constructor config (the kwargs) is skipped, so a restored or
  forked world keeps the config it was constructed with (e.g. a fork may change `guess_limit`).
"""
from __future__ import annotations

import random
import string
from typing import Any, ClassVar

from ..ids import AgentId
from ..rng import derive
from ..view import Observation, text_observation
from .base import Ack, Action, Outcome, World, tool

COLOURS = "rgbykwopcmnt"
PREAMBLE = "Candidate flags:"
CROP_HEADER = "Your crop:"
_LAYOUTS = ("h_stripes", "v_stripes", "blocks_2x2", "blocks_2x3")
_MAX_ATTEMPTS = 1000

Grid = list[str]


# ---- helpers usable by scripted participants -------------------------------------------------
def parse_observation(text: str) -> tuple[dict[str, list[str]], list[str]]:
    """Invert `FlagGame.observe`: return (candidates by name, crop rows).

    See the module docstring for the exact format. Blank lines are ignored.
    """
    candidates: dict[str, list[str]] = {}
    crop: list[str] = []
    current: list[str] | None = None
    for raw in text.split("\n"):
        line = raw.strip()
        if not line or line == PREAMBLE:
            continue
        if line == CROP_HEADER:
            current = crop
        elif line.endswith(":"):
            current = candidates.setdefault(line[:-1], [])
        elif current is not None:
            current.append(line)
        else:
            raise ValueError(f"row outside any section: {line!r}")
    return candidates, crop


def contains(grid: list[str], crop: list[str]) -> bool:
    """True if `crop` appears as a contiguous sub-grid of `grid`."""
    h, w = len(crop), len(crop[0]) if crop else 0
    if h == 0 or h > len(grid) or w > len(grid[0]):
        return h == 0
    for y in range(len(grid) - h + 1):
        for x in range(len(grid[0]) - w + 1):
            if all(grid[y + i][x : x + w] == crop[i] for i in range(h)):
                return True
    return False


def candidates_containing(candidates: dict[str, list[str]], crop: list[str]) -> list[str]:
    """Names (in the order of `candidates`) whose grid contains `crop` as a sub-grid."""
    return [name for name, grid in candidates.items() if contains(grid, crop)]


# ---- generation -------------------------------------------------------------------------------
def _splits(n: int, parts: int) -> list[int]:
    """Index -> part number, splitting range(n) into `parts` near-equal contiguous runs."""
    return [min(parts - 1, i * parts // n) for i in range(n)]


def _structured_flag(rng: random.Random, height: int, width: int, colours: str) -> Grid:
    layout = rng.choice(_LAYOUTS)
    if layout == "h_stripes":
        rows_n, cols_n = rng.randint(2, min(4, height)), 1
    elif layout == "v_stripes":
        rows_n, cols_n = 1, rng.randint(2, min(4, width))
    elif layout == "blocks_2x2":
        rows_n, cols_n = min(2, height), min(2, width)
    else:
        rows_n, cols_n = min(2, height), min(3, width)
    block: list[list[str]] = []
    for by in range(rows_n):
        row: list[str] = []
        for bx in range(cols_n):
            banned = set()
            if bx:
                banned.add(row[bx - 1])
            if by:
                banned.add(block[by - 1][bx])
            row.append(rng.choice([c for c in colours if c not in banned]))
        block.append(row)
    ys, xs = _splits(height, rows_n), _splits(width, cols_n)
    return ["".join(block[ys[y]][xs[x]] for x in range(width)) for y in range(height)]


def _recolour(rng: random.Random, grid: Grid, k: int, colours: str) -> Grid:
    height, width = len(grid), len(grid[0])
    cells = rng.sample(range(height * width), k)
    out = [list(r) for r in grid]
    for c in cells:
        y, x = divmod(c, width)
        out[y][x] = rng.choice([col for col in colours if col != out[y][x]])
    return ["".join(r) for r in out]


def _names(kind: str, n: int) -> list[str]:
    if kind == "letters":
        if n > 26:
            raise ValueError("candidate_names='letters' supports at most 26 candidates")
        return list(string.ascii_uppercase[:n])
    if kind == "numbers":
        return [str(i + 1) for i in range(n)]
    raise ValueError(f"unknown candidate_names {kind!r}; use 'letters' or 'numbers'")


# ---- world ------------------------------------------------------------------------------------
_CONFIG = (
    "height", "width", "palette", "n_candidates", "rival_similarity", "crop_h", "crop_w",
    "candidate_names", "status_tools", "guess_limit",
)


class FlagGame(World):
    entry_point: ClassVar[str | None] = "flaggame"
    name: str = "flaggame"
    _skip_in_snapshot: ClassVar[tuple[str, ...]] = ("params", *_CONFIG)

    def __init__(
        self,
        height: int = 8,
        width: int = 12,
        palette: int = 6,
        n_candidates: int = 8,
        rival_similarity: float = 0.85,
        crop_h: int = 3,
        crop_w: int = 4,
        candidate_names: str = "letters",
        status_tools: tuple[str, ...] | list[str] = ("my_status", "collective_status"),
        guess_limit: int | None = None,
    ) -> None:
        if not 2 <= palette <= len(COLOURS):
            raise ValueError(f"palette must be in 2..{len(COLOURS)}")
        if n_candidates < 2:
            raise ValueError("n_candidates must be >= 2 (truth and rival)")
        if not (1 <= crop_h <= height and 1 <= crop_w <= width):
            raise ValueError("crop must fit inside the flag")
        if height < 2 or width < 2:
            raise ValueError("flag must be at least 2x2")
        if not 0.0 <= rival_similarity <= 1.0:
            raise ValueError("rival_similarity must be in [0, 1]")
        unknown = set(status_tools) - {"my_status", "collective_status"}
        if unknown:
            raise ValueError(f"unknown status tools {sorted(unknown)}")
        _names(candidate_names, n_candidates)  # validates
        self.height, self.width, self.palette = height, width, palette
        self.n_candidates, self.rival_similarity = n_candidates, rival_similarity
        self.crop_h, self.crop_w = crop_h, crop_w
        self.candidate_names = candidate_names
        self.status_tools = tuple(status_tools)
        self.guess_limit = guess_limit
        # game state (plain Python data only)
        self.agents: list[str] = []
        self.candidates: dict[str, list[str]] = {}
        self.truth: str | None = None
        self.rival: str | None = None
        self.crops: dict[str, tuple[int, int]] = {}
        self.guesses: dict[str, str] = {}
        self.guesses_made: dict[str, int] = {}

    # ---- required -----------------------------------------------------------------------------
    def reset(self, rng: random.Random, agents: list[AgentId]) -> None:
        colours = COLOURS[: self.palette]
        h, w = self.height, self.width
        truth = _structured_flag(rng, h, w, colours)
        k = max(1, round((1 - self.rival_similarity) * h * w))
        rival = _recolour(rng, truth, k, colours)
        flags = [truth, rival]
        attempts = 0
        while len(flags) < self.n_candidates:
            attempts += 1
            if attempts > _MAX_ATTEMPTS:
                raise ValueError("could not generate enough distinct candidate flags")
            f = _structured_flag(rng, h, w, colours)
            if f not in flags:
                flags.append(f)
        order = list(range(len(flags)))
        rng.shuffle(order)
        names = _names(self.candidate_names, self.n_candidates)
        self.candidates = {names[pos]: flags[i] for pos, i in enumerate(order)}
        self.truth = names[order.index(0)]
        self.rival = names[order.index(1)]
        seeds = [rng.getrandbits(64) for _ in agents]
        self.agents = [str(a) for a in agents]
        self.crops = {}
        for agent, s in zip(self.agents, seeds, strict=True):
            r = derive(s, "private", agent)
            self.crops[agent] = (r.randint(0, h - self.crop_h), r.randint(0, w - self.crop_w))
        self.guesses = {}
        self.guesses_made = {}

    def crop_rows(self, agent: AgentId) -> list[str]:
        y, x = self.crops[agent]
        truth = self.candidates[self.truth]  # type: ignore[index]
        return [row[x : x + self.crop_w] for row in truth[y : y + self.crop_h]]

    def observe(self, agent: AgentId) -> Observation:
        lines = [PREAMBLE]
        for name, grid in self.candidates.items():
            lines += ["", f"{name}:", *grid]
        lines += ["", CROP_HEADER, *self.crop_rows(agent)]
        y, x = self.crops[agent]
        return text_observation("\n".join(lines), crop_y=y, crop_x=x)

    def score(self) -> dict:
        right = sum(1 for a in self.agents if self.guesses.get(a) == self.truth)
        return {
            "accuracy": right / len(self.agents) if self.agents else 0.0,
            "n_guessed": sum(1 for a in self.agents if a in self.guesses),
            "truth": self.truth,
        }

    # ---- actions ------------------------------------------------------------------------------
    def _guess_error(self, agent: AgentId, candidate: Any) -> str | None:
        if agent not in self.crops:
            return "unknown agent"
        if not isinstance(candidate, str) or candidate not in self.candidates:
            return f"unknown candidate {candidate!r}"
        if self.guess_limit is not None and self.guesses_made.get(agent, 0) >= self.guess_limit:
            return "guess limit reached"
        return None

    @tool("guess", "Record your current guess of which candidate the flag is", {"candidate": "string"})
    def guess(self, agent: AgentId, candidate: str) -> Outcome:
        err = self._guess_error(agent, candidate)
        if err is not None:
            return Outcome(accepted=False, feedback={"error": err})
        self.guesses[agent] = candidate
        self.guesses_made[agent] = self.guesses_made.get(agent, 0) + 1
        return Outcome(accepted=True, feedback={"recorded": True})

    def validate(self, agent: AgentId, action: Action) -> Ack:
        ack = super().validate(agent, action)
        if not ack.ok or action.name != "guess":
            return ack
        err = self._guess_error(agent, action.args["candidate"])
        return Ack(ok=False, error=err) if err else ack

    # ---- status -------------------------------------------------------------------------------
    def my_status(self, agent: AgentId) -> dict | None:
        if "my_status" not in self.status_tools:
            return None
        return {
            "current_guess": self.guesses.get(agent),
            "guesses_made": self.guesses_made.get(agent, 0),
        }

    def collective_status(self) -> dict | None:
        if "collective_status" not in self.status_tools:
            return None
        counts = {name: 0 for name in self.candidates}
        for g in self.guesses.values():
            counts[g] += 1
        return {"guess_counts": counts, "agents_with_guess": len(self.guesses)}

    # ---- evaluator-only -----------------------------------------------------------------------
    def verify(self) -> dict:
        return {
            "truth": self.truth,
            "rival": self.rival,
            "candidates": {n: list(g) for n, g in self.candidates.items()},
            "crops": {a: [y, x] for a, (y, x) in self.crops.items()},
        }
