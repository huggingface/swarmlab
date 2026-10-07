"""FlagGame, text and image variants (docs/INTERFACE.md §8, docs/INTERFACE-M5.md §1).

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

Structured flags are one of: horizontal stripes (2-4 bands), vertical stripes (2-4 bands), 2x2
blocks, 2x3 blocks (2 rows by 3 columns). Bands/blocks split the grid as evenly as possible;
neighbouring bands/blocks never share a colour.

Twin pairs (review A1: no shortcut from the observation alone). Candidates are `n_candidates // 2`
near-twin pairs (`n_candidates` must be even and >= 2). A pair is a structured flag plus a variant
in which `min(rival_edits, number of bands/blocks)` distinct whole bands/blocks are recoloured, one
after another, each to a palette colour different from its own and from its current neighbours.
So both members are clean structured flags of the same layout and the variant always differs.
The truth is a uniformly random member of a uniformly random pair and the rival is its twin. Since
every candidate has a twin generated the same way and the truth's pair and side are drawn after
generation, no similarity or cleanliness heuristic over the candidates singles out the truth.
Crops that avoid the recoloured bands are contained in both truth and rival: that is how "the
rival is favoured by some crops". Generation redraws a pair until every candidate is distinct
(ValueError after 1000 attempts, e.g. if `n_candidates` exceeds what the palette allows), and
`palette >= 3` is required so a band between two differently coloured neighbours can change.

Candidates are shuffled and then named in order: "letters" -> A, B, C, ... (max 26),
"numbers" -> 1, 2, 3, ... The name of the truth is therefore uniformly random.

Randomness
----------
`reset(rng, agents)` receives one stream (the runner passes `derive(seed, "world")`). Draws, in
order: the pairs (base then variant, pair by pair), the truth's pair index, the truth's side
(base or variant), shuffle, then one 64-bit value `s_i = rng.getrandbits(64)` per
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

Image modality (M5)
-------------------
`FlagGame(modality="image", cell_px=12, image_text_hint=False)`. `observe()` returns, in order:

1. a text part, two lines: `Candidate flags:` and the framing sentence (`IMAGE_INTRO`), which
   names the candidates in order and says that the order of the images is the labelling;
2. one image part per candidate, in name order (`grid_to_png` of the grid, no label drawn);
3. the text part `Your crop:`;
4. one image part for the crop, at the same `cell_px`;
5. after a `patch_private` crop move, the text part `Your crop has changed.` (once).

With `image_text_hint=True` the grids are given as text too: after each candidate image a text
part `<name>:\n<rows>`, after the crop image a text part with the crop rows. Joining the text
parts with "\n" then gives a listing that `parse_observation` (and the fake provider's reader)
reads exactly like the text variant (they skip the prose line before the first section).
`private` is unchanged. `description()` says the candidates and the crop are images.

Decisions where the contract is silent
--------------------------------------
- `modality`, `cell_px` and `image_text_hint` are part of `spec()` only when they differ from
  their defaults, so every text-mode run keeps its spec hash. `image_text_hint` is ignored in
  text mode (allowed, so an arm grid can vary `modality` alone).
- Images are base64 PNG (`Part.image_png_b64`); rendered PNGs are cached per grid in a transient
  attribute (`_png_cache`, never snapshotted).
- `check_participants(participants, experiment)` (called by the runner after bind, fresh runs and
  forks) raises `ValueError` in image mode without `image_text_hint` when a participant reads the
  text grids: scripted FlagGame participants (`reads_text_observation = True`) and `LLMAgent`s
  whose model resolves to the fake provider's `flaggame_reader` script.
- Guesses for unknown candidates and guesses beyond `guess_limit` are rejected by `validate`; the
  `guess` method re-checks both (returning `accepted=False`). Under round_end the executor passes
  the agent's buffered actions as `validate(..., pending=...)`, and a guess is refused at call
  time when recorded plus buffered guesses already reach `guess_limit` (the commit would reject
  it); a direct `validate` call without `pending` checks round-start state only. Guesses from agents not passed to `reset`
  are rejected the same way.
- `collective_status()["guess_counts"]` lists every candidate name (zeros included), in name
  order, so the key set carries no information.
- `my_status` for an agent with no state (including before `reset`) returns
  `{"current_guess": None, "guesses_made": 0}` when enabled.
- `score()["accuracy"]` divides by all agents given to `reset` (0.0 when there are none).
- `crop_overrides={agent: [y, x]}` (M3a §2, paired runs) replaces those agents' crop positions
  after the normal reset draws, so every other draw (candidates, truth, other crops) is unchanged.
  Positions are validated against the flag at construction, agent ids at `reset`. The param is
  part of `spec()` only when given, so runs without it keep their spec hash.
- Blind agents (WP16, the Flag Game paper's manager protocol): `blind_agents` is a list of agent
  ids or an int n (the first n agents of the `reset` list, i.e. by index). Crops are still drawn
  for every agent (so the other agents' crops, the candidates and the truth are what they would be
  without blind agents) and then dropped for the blind ones. A blind agent's observation lists
  the candidates exactly as usual and, in place of the crop section, the line `BLIND_NOTE`
  ("You have no crop of your own; rely on what others report."); in image mode the framing line
  drops "your crop follows" (`IMAGE_INTRO_BLIND`) and `BLIND_NOTE` is the last text part.
  `private` is `{}`. Unless `blind_may_guess=True`, `allows_tool(agent, "guess")` is False for
  them (the executor neither offers nor runs `guess`) and `guess`/`validate` reject their guesses
  ("blind agents may not guess"). Blind agents are left out of `score()` (numerator and
  `accuracy`'s denominator, and `n_guessed`; `n_blind` is added) and named by
  `excluded_from_belief()`, so belief metrics leave them out too; `verify()["crops"]` omits them
  and `verify()["blind"]` lists them. `description()` gains one sentence saying that some agents
  have no crop. `patch_private` and `crop_overrides` on a blind agent are a `ValueError`; an int
  or list naming every agent (nobody sighted) or an unknown agent is a `ValueError` at `reset`.
  Both params are left out of `spec()` when unset (empty / 0 / False), so spec hashes are
  unchanged.
- Snapshots carry game state only. Constructor config (the kwargs) is skipped, so a restored or
  forked world keeps the config it was constructed with (e.g. a fork may change `guess_limit`).
"""
from __future__ import annotations

import base64
import random
import re
import string
from collections.abc import Sequence
from typing import Any, ClassVar

from ..ids import AgentId
from ..rng import derive
from ..view import Observation, Part, text_observation
from .base import Ack, Action, Outcome, World, tool
from .render import PALETTE, grid_to_png

COLOURS = "rgbykwopcmnt"
PREAMBLE = "Candidate flags:"
CROP_HEADER = "Your crop:"
CROP_CHANGED = "Your crop has changed."
BLIND_NOTE = "You have no crop of your own; rely on what others report."
IMAGE_INTRO = ("Candidates {names} are shown as images in that order (no label is drawn inside an "
               "image: the order is the labelling); your crop follows. Each colour cell is "
               "{cell_px}x{cell_px} pixels in every image.")
IMAGE_INTRO_BLIND = ("Candidates {names} are shown as images in that order (no label is drawn inside "
                     "an image: the order is the labelling). Each colour cell is "
                     "{cell_px}x{cell_px} pixels in every image.")
_IMAGE_INTRO_RE = re.compile(r"^Candidates (.+?) are shown as images in that order")
MODALITIES = ("text", "image")
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
        if not line or line in (PREAMBLE, CROP_CHANGED, BLIND_NOTE):
            continue
        if line == CROP_HEADER:
            current = crop
        elif line.endswith(":"):
            current = candidates.setdefault(line[:-1], [])
        elif current is not None:
            current.append(line)
        elif " " in line:
            continue  # prose before the first section (the image modality's framing line)
        else:
            raise ValueError(f"row outside any section: {line!r}")
    return candidates, crop


def candidate_names(text: str) -> list[str]:
    """Candidate names from an observation's text: the section headers, else (image modality
    without the text hint) the names listed in the framing line."""
    try:
        names = list(parse_observation(text)[0])
    except ValueError:
        names = []
    if names:
        return names
    for line in text.split("\n"):
        m = _IMAGE_INTRO_RE.match(line.strip())
        if m:
            return [n.strip() for n in m.group(1).split(",") if n.strip()]
    return []


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


def _structured_blocks(
    rng: random.Random, height: int, width: int, colours: str
) -> list[list[str]]:
    """A random layout as a rows_n x cols_n block grid of colours (neighbours differ)."""
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
    return block


def _render(block: list[list[str]], height: int, width: int) -> Grid:
    ys, xs = _splits(height, len(block)), _splits(width, len(block[0]))
    return ["".join(block[ys[y]][xs[x]] for x in range(width)) for y in range(height)]


def _neighbours(block: list[list[str]], by: int, bx: int) -> set[str]:
    out = set()
    for dy, dx in ((-1, 0), (1, 0), (0, -1), (0, 1)):
        y, x = by + dy, bx + dx
        if 0 <= y < len(block) and 0 <= x < len(block[0]):
            out.add(block[y][x])
    return out


def _variant(rng: random.Random, block: list[list[str]], edits: int, colours: str) -> list[list[str]] | None:
    """Recolour `edits` distinct whole blocks (clamped to the block count); None if impossible."""
    out = [list(r) for r in block]
    cells = [(y, x) for y in range(len(out)) for x in range(len(out[0]))]
    rng.shuffle(cells)
    done = 0
    for y, x in cells:
        if done == min(edits, len(cells)):
            break
        options = [c for c in colours if c != out[y][x] and c not in _neighbours(out, y, x)]
        if options:
            out[y][x] = rng.choice(options)
            done += 1
    return out if done == min(edits, len(cells)) else None


def _twin_pair(rng: random.Random, height: int, width: int, colours: str, edits: int) -> tuple[Grid, Grid]:
    for _ in range(_MAX_ATTEMPTS):
        block = _structured_blocks(rng, height, width, colours)
        var = _variant(rng, block, edits, colours)
        if var is not None:
            return _render(block, height, width), _render(var, height, width)
    raise ValueError("could not generate a twin pair")


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
    "height", "width", "palette", "n_candidates", "rival_edits", "crop_h", "crop_w",
    "candidate_names", "status_tools", "guess_limit", "crop_overrides",
    "modality", "cell_px", "image_text_hint", "blind_agents", "blind_may_guess",
)
_M5_DEFAULTS = {"modality": "text", "cell_px": 12, "image_text_hint": False}


def _check_overrides(overrides: dict[str, list[int]] | None, max_y: int,
                     max_x: int) -> dict[str, tuple[int, int]]:
    """Validate `crop_overrides` ({agent: [y, x]}, top-left inside the flag) into tuples."""
    out: dict[str, tuple[int, int]] = {}
    for agent, pos in (overrides or {}).items():
        if (not isinstance(pos, (list, tuple)) or len(pos) != 2
                or not all(isinstance(v, int) and not isinstance(v, bool) for v in pos)):
            raise ValueError(f"crop_overrides[{agent!r}] must be [y, x], got {pos!r}")
        y, x = pos
        if not (0 <= y <= max_y and 0 <= x <= max_x):
            raise ValueError(f"crop_overrides[{agent!r}] = {[y, x]} is outside 0..{max_y} x 0..{max_x}")
        out[str(agent)] = (y, x)
    return out


def _check_blind(blind: list[str] | int | None) -> list[str] | int:
    """Validate `blind_agents`: None/[] -> [], an int >= 0, or a list of distinct agent ids."""
    if blind is None:
        return []
    if isinstance(blind, int) and not isinstance(blind, bool):
        if blind < 0:
            raise ValueError(f"blind_agents must be >= 0, got {blind}")
        return blind
    if not isinstance(blind, (list, tuple)) or not all(isinstance(a, str) for a in blind):
        raise ValueError(f"blind_agents must be a list of agent ids or an int, got {blind!r}")
    if len(set(blind)) != len(blind):
        raise ValueError(f"blind_agents lists an agent twice: {list(blind)}")
    return [str(a) for a in blind]


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
        rival_edits: int = 1,
        crop_h: int = 3,
        crop_w: int = 4,
        candidate_names: str = "letters",
        status_tools: tuple[str, ...] | list[str] = ("my_status", "collective_status"),
        guess_limit: int | None = None,
        crop_overrides: dict[str, list[int]] | None = None,
        modality: str = "text",
        cell_px: int = 12,
        image_text_hint: bool = False,
        blind_agents: list[str] | int | None = None,
        blind_may_guess: bool = False,
    ) -> None:
        if not 3 <= palette <= len(COLOURS):
            raise ValueError(f"palette must be in 3..{len(COLOURS)}")
        if n_candidates < 2 or n_candidates % 2:
            raise ValueError("n_candidates must be even and >= 2 (candidates come in twin pairs)")
        if not (1 <= crop_h <= height and 1 <= crop_w <= width):
            raise ValueError("crop must fit inside the flag")
        if height < 2 or width < 2:
            raise ValueError("flag must be at least 2x2")
        if not isinstance(rival_edits, int) or rival_edits < 1:
            raise ValueError("rival_edits must be an integer >= 1")
        unknown = set(status_tools) - {"my_status", "collective_status"}
        if unknown:
            raise ValueError(f"unknown status tools {sorted(unknown)}")
        _names(candidate_names, n_candidates)  # validates
        if modality not in MODALITIES:
            raise ValueError(f"modality must be one of {MODALITIES}, got {modality!r}")
        if not isinstance(cell_px, int) or isinstance(cell_px, bool) or cell_px < 1:
            raise ValueError("cell_px must be an integer >= 1")
        self.modality, self.cell_px, self.image_text_hint = modality, cell_px, bool(image_text_hint)
        self.blind_agents = _check_blind(blind_agents)
        self.blind_may_guess = bool(blind_may_guess)
        self.height, self.width, self.palette = height, width, palette
        self.n_candidates, self.rival_edits = n_candidates, rival_edits
        self.crop_h, self.crop_w = crop_h, crop_w
        self.candidate_names = candidate_names
        self.status_tools = tuple(status_tools)
        self.guess_limit = guess_limit
        self.crop_overrides = _check_overrides(crop_overrides, height - crop_h, width - crop_w)
        if crop_overrides is None and isinstance(getattr(self, "params", None), dict):
            # absent from spec() when unset, so runs without overrides keep their spec hash
            self.params.pop("crop_overrides", None)
        if isinstance(getattr(self, "params", None), dict):
            # WP16: absent from spec() when unset, so runs without blind agents keep their hash
            if not self.blind_agents:
                self.params.pop("blind_agents", None)
            if not self.blind_may_guess:
                self.params.pop("blind_may_guess", None)
        if isinstance(getattr(self, "params", None), dict):
            # M5: absent from spec() at their defaults, so text-mode spec hashes are unchanged
            for k, default in _M5_DEFAULTS.items():
                if k in self.params and self.params[k] == default and type(self.params[k]) is type(default):
                    self.params.pop(k)
        # game state (plain Python data only)
        self.agents: list[str] = []
        self.candidates: dict[str, list[str]] = {}
        self.truth: str | None = None
        self.rival: str | None = None
        self.crops: dict[str, tuple[int, int]] = {}
        self.blind: list[str] = []
        self.guesses: dict[str, str] = {}
        self.guesses_made: dict[str, int] = {}

    # ---- required -----------------------------------------------------------------------------
    def reset(self, rng: random.Random, agents: list[AgentId]) -> None:
        colours = COLOURS[: self.palette]
        h, w = self.height, self.width
        pairs: list[tuple[Grid, Grid]] = []
        seen: list[Grid] = []
        attempts = 0
        while len(pairs) < self.n_candidates // 2:
            attempts += 1
            if attempts > _MAX_ATTEMPTS:
                raise ValueError("could not generate enough distinct candidate flags")
            base, var = _twin_pair(rng, h, w, colours, self.rival_edits)
            if base in seen or var in seen:
                continue
            pairs.append((base, var))
            seen += [base, var]
        flags = [f for pair in pairs for f in pair]  # pair i -> indices 2i, 2i+1
        pair = rng.randrange(len(pairs))
        side = rng.randrange(2)
        truth_i, rival_i = 2 * pair + side, 2 * pair + 1 - side
        order = list(range(len(flags)))
        rng.shuffle(order)
        names = _names(self.candidate_names, self.n_candidates)
        self.candidates = {names[pos]: flags[i] for pos, i in enumerate(order)}
        self.truth = names[order.index(truth_i)]
        self.rival = names[order.index(rival_i)]
        seeds = [rng.getrandbits(64) for _ in agents]
        self.agents = [str(a) for a in agents]
        self.crops = {}
        for agent, s in zip(self.agents, seeds, strict=True):
            r = derive(s, "private", agent)
            self.crops[agent] = (r.randint(0, h - self.crop_h), r.randint(0, w - self.crop_w))
        unknown = sorted(set(self.crop_overrides) - set(self.agents))
        if unknown:
            raise ValueError(f"crop_overrides names agents not in the game: {unknown}")
        for agent, (y, x) in self.crop_overrides.items():
            self.crops[agent] = (y, x)
        self.blind = self._resolve_blind(self.agents)
        bad = sorted(set(self.crop_overrides) & set(self.blind))
        if bad:
            raise ValueError(f"crop_overrides names blind agents: {bad}")
        for agent in self.blind:
            del self.crops[agent]
        self.guesses = {}
        self.guesses_made = {}

    def _resolve_blind(self, agents: list[str]) -> list[str]:
        spec = getattr(self, "blind_agents", [])
        if isinstance(spec, int):
            if spec >= len(agents) and spec:
                raise ValueError(f"blind_agents={spec} leaves no agent with a crop "
                                 f"({len(agents)} agents)")
            return list(agents[:spec])
        unknown = sorted(set(spec) - set(agents))
        if unknown:
            raise ValueError(f"blind_agents names agents not in the game: {unknown}")
        if agents and set(agents) <= set(spec):
            raise ValueError("blind_agents names every agent: nobody has a crop")
        return [a for a in agents if a in spec]

    def is_blind(self, agent: AgentId) -> bool:
        return str(agent) in getattr(self, "blind", ())

    def sighted(self) -> list[str]:
        """The agents with a crop, in reset order."""
        return [a for a in self.agents if not self.is_blind(a)]

    def allows_tool(self, agent: AgentId, name: str) -> bool:
        return not (name == "guess" and self.is_blind(agent) and not self.blind_may_guess)

    def excluded_from_belief(self) -> list[AgentId]:
        return [AgentId(a) for a in getattr(self, "blind", ())]

    def crop_rows(self, agent: AgentId) -> list[str]:
        y, x = self.crops[agent]
        truth = self.candidates[self.truth]  # type: ignore[index]
        return [row[x : x + self.crop_w] for row in truth[y : y + self.crop_h]]

    def _png_b64(self, grid: list[str]) -> str:
        cache = self.__dict__.setdefault("_png_cache", {})
        key = (self.cell_px, tuple(grid))
        if key not in cache:
            cache[key] = base64.b64encode(grid_to_png(list(grid), PALETTE, self.cell_px)).decode()
        return cache[key]

    def _observe_image(self, agent: AgentId) -> Observation:
        names = ", ".join(self.candidates)
        blind = self.is_blind(agent)
        intro = IMAGE_INTRO_BLIND if blind else IMAGE_INTRO
        parts = [Part(type="text", text=PREAMBLE + "\n"
                      + intro.format(names=names, cell_px=self.cell_px))]
        for name, grid in self.candidates.items():
            parts.append(Part(type="image", image_png_b64=self._png_b64(grid)))
            if self.image_text_hint:
                parts.append(Part(type="text", text="\n".join([f"{name}:", *grid])))
        if blind:
            parts.append(Part(type="text", text=BLIND_NOTE))
            return Observation(parts=parts, private={})
        crop = self.crop_rows(agent)
        parts += [Part(type="text", text=CROP_HEADER), Part(type="image", image_png_b64=self._png_b64(crop))]
        if self.image_text_hint:
            parts.append(Part(type="text", text="\n".join(crop)))
        if agent in getattr(self, "crops_changed", ()):
            self.crops_changed = [a for a in self.crops_changed if a != agent]
            parts.append(Part(type="text", text=CROP_CHANGED))
        y, x = self.crops[agent]
        return Observation(parts=parts, private={"crop_y": y, "crop_x": x})

    def observe(self, agent: AgentId) -> Observation:
        if getattr(self, "modality", "text") == "image":
            return self._observe_image(agent)
        lines = [PREAMBLE]
        for name, grid in self.candidates.items():
            lines += ["", f"{name}:", *grid]
        if self.is_blind(agent):
            return text_observation("\n".join([*lines, "", BLIND_NOTE]))
        lines += ["", CROP_HEADER, *self.crop_rows(agent)]
        if agent in getattr(self, "crops_changed", ()):  # M3a patch_private: said once
            self.crops_changed = [a for a in self.crops_changed if a != agent]
            lines += ["", CROP_CHANGED]
        y, x = self.crops[agent]
        return text_observation("\n".join(lines), crop_y=y, crop_x=x)

    def score(self) -> dict:
        agents = self.sighted()  # blind agents are not scored (WP16)
        right = sum(1 for a in agents if self.guesses.get(a) == self.truth)
        out = {
            "accuracy": right / len(agents) if agents else 0.0,
            "n_guessed": sum(1 for a in agents if a in self.guesses),
            "truth": self.truth,
        }
        if getattr(self, "blind", None):
            out["n_blind"] = len(self.blind)
        return out

    # ---- actions ------------------------------------------------------------------------------
    def _guess_error(self, agent: AgentId, candidate: Any) -> str | None:
        if self.is_blind(agent):
            if not self.blind_may_guess:
                return "blind agents may not guess"
        elif agent not in self.crops:
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

    def validate(self, agent: AgentId, action: Action, pending: Sequence[Action] = ()) -> Ack:
        ack = super().validate(agent, action)
        if not ack.ok or action.name != "guess":
            return ack
        err = self._guess_error(agent, action.args["candidate"])
        if err is None and self.guess_limit is not None:
            buffered = sum(1 for a in pending if a.name == "guess")
            if self.guesses_made.get(agent, 0) + buffered >= self.guess_limit:
                err = (f"guess limit reached: at most {self.guess_limit} guesses; this one would be "
                       "rejected at commit")
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

    def description(self) -> str:
        limit = (f" You may record at most {self.guess_limit} guesses."
                 if self.guess_limit is not None else "")
        if getattr(self, "blind_agents", None):
            limit += (" Some agents have no crop of their own (their observation says so) and "
                      "rely on what others report.")
        if getattr(self, "modality", "text") == "image":
            hint = (" Each image is also written out as a grid of colour letters."
                    if self.image_text_hint else "")
            return (
                f"There are {self.n_candidates} candidate flags, each shown as an image of coloured "
                "cells, and exactly one of them is the hidden flag. You privately see a "
                f"{self.crop_h}x{self.crop_w}-cell crop of the hidden flag, shown as an image at "
                "the same scale, at an undisclosed position, and more than one candidate may "
                f"contain your crop.{hint} Record which candidate you believe is the hidden flag "
                "with the `guess` tool; only your latest guess counts and you may change it in any "
                f"round.{limit} You are never told whether a guess is right."
            )
        return (
            f"There are {self.n_candidates} candidate flags, each a grid of colour letters, and "
            "exactly one of them is the hidden flag. You privately see a "
            f"{self.crop_h}x{self.crop_w} crop of the hidden flag at an undisclosed position, and "
            "more than one candidate may contain your crop. Record which candidate you believe is "
            "the hidden flag with the `guess` tool; only your latest guess counts and you may "
            f"change it in any round.{limit} You are never told whether a guess is right."
        )

    # ---- M5: participants that read the text grids ------------------------------------------------
    def check_participants(self, participants: dict, experiment: Any = None) -> None:
        """Image mode without `image_text_hint`: refuse participants that read the text grids."""
        if getattr(self, "modality", "text") != "image" or self.image_text_hint:
            return
        bad = sorted(str(a) for a, p in participants.items() if reads_text_grids(p, experiment))
        if bad:
            raise ValueError(
                f"FlagGame(modality='image') without image_text_hint=True gives no text grids, but "
                f"participants {bad} read them (scripted FlagGame participants and the fake "
                "provider's reader script); set image_text_hint=True or use a vision model")

    # ---- interventions (M3a, docs/INTERFACE-M3a.md §1) --------------------------------------------
    def patch_private(self, agent: AgentId, data: dict) -> None:
        """`{"crop": [y, x]}` moves the agent's crop; its next observation (only) ends with a blank
        line and the line `Your crop has changed.` after the crop rows (`parse_observation` skips it)."""
        if set(data) != {"crop"}:
            raise ValueError(f"FlagGame.patch_private takes {{'crop': [y, x]}}, got keys {sorted(data)}")
        if self.is_blind(agent):
            raise ValueError(f"{agent} is blind (has no crop to move)")
        if str(agent) not in self.crops:
            raise ValueError(f"unknown agent {agent!r}")
        y, x = (int(v) for v in data["crop"])
        if not (0 <= y <= self.height - self.crop_h and 0 <= x <= self.width - self.crop_w):
            raise ValueError(f"crop {[y, x]} does not fit a {self.height}x{self.width} flag")
        self.crops[str(agent)] = (y, x)
        changed = getattr(self, "crops_changed", [])
        self.crops_changed = changed + ([str(agent)] if str(agent) not in changed else [])

    def intervene(self, name: str, /, **args: Any) -> dict:
        """`set_truth(name=...)` makes another candidate the hidden flag (the rival swaps with the
        truth when the new truth is the rival, else it is unchanged)."""
        if name != "set_truth":
            return super().intervene(name, **args)
        new = args.get("name")
        if set(args) != {"name"} or new not in self.candidates:
            raise ValueError(f"set_truth needs name=<candidate>, got {args}")
        if new == self.rival:
            self.rival = self.truth
        self.truth = new
        return {"truth": self.truth, "rival": self.rival}

    # ---- evaluator-only -----------------------------------------------------------------------
    def verify(self) -> dict:
        out = {
            "truth": self.truth,
            "rival": self.rival,
            "candidates": {n: list(g) for n, g in self.candidates.items()},
            "crops": {a: [y, x] for a, (y, x) in self.crops.items()},
        }
        if getattr(self, "blind", None):
            out["blind"] = list(self.blind)
        return out


def reads_text_grids(participant: Any, experiment: Any = None) -> bool:
    """True for a participant that can only play FlagGame from the text grids: scripted FlagGame
    participants (`reads_text_observation = True`) and model participants whose model resolves to
    the fake provider's `flaggame_reader` script."""
    if getattr(participant, "reads_text_observation", False):
        return True
    model = getattr(participant, "model", None)
    if not isinstance(model, str) or ":" not in model:
        return False
    from ..providers.base import ChatMessage, ChatRequest
    from ..providers.fake import BUILTIN_SCRIPTS, FakeProvider, flaggame_reader

    provider = None
    if experiment is not None and hasattr(experiment, "provider_for"):
        try:
            provider = experiment.provider_for(model)[0]
        except Exception:  # noqa: BLE001 - an unresolvable model is reported elsewhere
            return False
    elif model.split(":", 1)[0] == "fake":
        provider = FakeProvider()
    if not isinstance(provider, FakeProvider):
        return False
    try:
        script = provider.script_for(ChatRequest(model=model, messages=[
            ChatMessage(role="user", content="")]))
    except Exception:  # noqa: BLE001
        return False
    return script is flaggame_reader or script is BUILTIN_SCRIPTS.get("reader")
