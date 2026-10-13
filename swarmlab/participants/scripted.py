"""Scripted participants for FlagGame (docs/INTERFACE.md §10).

All three draw randomness only from the rng passed to `bind` (the runner passes
`derive(seed, "agent", agent)`), keep plain Python state (pickled by `Persistable`), and parse
the world's observation with `swarmlab.world.flaggame.parse_observation` (the text parts joined
with newlines). In FlagGame's image modality they need `image_text_hint=True` (M5): they set
`reads_text_observation = True` and `FlagGame.check_experiment` refuses them otherwise.

- `Silent`: in round 1 guesses a candidate containing its crop (ties broken by its rng), then
  `end_turn`. Later rounds: `end_turn` only. Never reads or posts.
- `EvidenceAggregator`: round 1 posts `"crop:\\n" + "\\n".join(crop_rows)`. Every round it calls
  `read_board(limit=200)`, parses every item whose content starts with `crop:` into a crop
  (the remaining non-empty lines), keeps every crop it has seen (its own included, duplicates
  kept since each comes from a different post), guesses the candidate contained in the most
  crops (ties broken by its rng, choosing from the tied names in name order), then `end_turn`.
- `Enumerator(truth_name=None)`: test oracle. Guesses candidate `(round - 1 + i) % n` in name
  order (i = agent index, so agents spread over candidates), calls both status tools when they
  are offered, and raises `AssertionError` if the view or any tool result reveals correctness:
  the words "correct" or "truth" anywhere (case-insensitive) in the view or a result, or the
  truth name as a bare token outside the symmetric places every candidate name appears
  (candidate headers in the observation, `guess_counts` keys, and the agent's own
  `current_guess` echo), following the WP2 oracle test. The truth name is checked in the
  observation text and tool results, not in the view's numeric fields.
"""
from __future__ import annotations

import json
import re
from typing import Any, ClassVar

from ..tools import AgentTools
from ..view import View
from ..world.flaggame import candidates_containing, contains, parse_observation
from .base import Participant, TurnUsage

CROP_PREFIX = "crop:"
_INTRO_LINE_RE = re.compile(r"^Candidates .+ are shown as images in that order.*$", re.MULTILINE)


def _observation_text(view: View) -> str:
    return "\n".join(p.text or "" for p in view.observation.parts if p.type == "text")


class Silent(Participant):
    entry_point: ClassVar[str | None] = "silent"
    description: ClassVar[str] = (
        "FlagGame: guesses from its own crop in round 1; never reads or posts"
    )
    reads_text_observation: ClassVar[bool] = True  # M5: needs image_text_hint in image mode

    async def turn(self, view: View, tools: AgentTools) -> TurnUsage:
        calls = 0
        if view.round == 1:
            candidates, crop = parse_observation(_observation_text(view))
            names = candidates_containing(candidates, crop) or sorted(candidates)
            await tools.call("guess", {"candidate": self.rng.choice(names)})
            calls += 1
        await tools.call("end_turn", {})
        return TurnUsage(calls=calls + 1)


class EvidenceAggregator(Participant):
    entry_point: ClassVar[str | None] = "evidence_aggregator"
    description: ClassVar[str] = (
        "FlagGame: posts its crop, reads crops, guesses the best-matching candidate"
    )
    reads_text_observation: ClassVar[bool] = True

    def bind(self, agent: Any, rng: Any) -> None:
        super().bind(agent, rng)
        self.crops: list[list[str]] = []
        self.posted = False

    async def turn(self, view: View, tools: AgentTools) -> TurnUsage:
        candidates, crop = parse_observation(_observation_text(view))
        if not self.crops:
            self.crops.append(list(crop))
        calls = 0
        if not self.posted:
            calls += 1
            await tools.call("post", {"channel": "main", "text": CROP_PREFIX + "\n" + "\n".join(crop)})
            self.posted = True
        res = await tools.call("read_board", {"limit": 200})
        for item in res.result.get("items", []) if res.ok else []:
            content = item.get("content", "")
            if content.startswith(CROP_PREFIX):
                rows = [r.strip() for r in content[len(CROP_PREFIX):].split("\n") if r.strip()]
                if rows and len({len(r) for r in rows}) == 1:
                    self.crops.append(rows)
        names = sorted(candidates)
        scores = {n: sum(1 for c in self.crops if contains(candidates[n], c)) for n in names}
        best = max(scores.values())
        choice = self.rng.choice([n for n in names if scores[n] == best])
        await tools.call("guess", {"candidate": choice})
        await tools.call("end_turn", {})
        return TurnUsage(calls=calls + 3)


def _bare(name: str, text: str) -> bool:
    return re.search(rf"(?<![A-Za-z0-9]){re.escape(name)}(?![A-Za-z0-9])", text) is not None


class Enumerator(Participant):
    entry_point: ClassVar[str | None] = "enumerator"
    description: ClassVar[str] = (
        "FlagGame test oracle: cycles guesses, fails if anything reveals the truth"
    )
    reads_text_observation: ClassVar[bool] = True

    def __init__(self, truth_name: str | None = None) -> None:
        self.truth_name = truth_name

    def _check_words(self, label: str, data: Any) -> None:
        low = json.dumps(data, sort_keys=True, default=str).lower()
        for word in ("correct", "truth"):
            if word in low:
                raise AssertionError(f"{self.agent}: {label} contains {word!r}: {low[:200]}")

    def _check_name(self, label: str, text: str) -> None:
        if self.truth_name is not None and _bare(self.truth_name, text):
            raise AssertionError(f"{self.agent}: {label} names the truth: {text[:200]}")

    async def turn(self, view: View, tools: AgentTools) -> TurnUsage:
        self._check_words("view", view.model_dump(mode="json"))
        text = _observation_text(view)
        candidates, _ = parse_observation(text)
        names = sorted(candidates)
        stripped = _INTRO_LINE_RE.sub("", text)  # M5: the framing line lists every name
        for n in names:
            stripped = stripped.replace(f"\n{n}:\n", "\n")
        self._check_name("observation", stripped)
        self._check_name("outcomes", json.dumps(view.outcomes, sort_keys=True))
        self._check_name("tools", json.dumps([t.model_dump() for t in view.tools], sort_keys=True))
        offered = {t.name for t in view.tools}
        idx = int(self.agent[1:]) if self.agent[1:].isdigit() else 0
        results = [await tools.call("guess",
                                    {"candidate": names[(view.round - 1 + idx) % len(names)]})]
        for status in ("my_status", "collective_status"):
            if status in offered:
                results.append(await tools.call(status, {}))
        for res in results:
            data = res.model_dump(mode="json")
            self._check_words("tool result", data)
            body = dict(data.get("result") or {})
            if "guess_counts" in body:
                counts = body.pop("guess_counts")
                if sorted(counts) != names:
                    raise AssertionError(f"{self.agent}: guess_counts keys are not all candidates")
                body["guess_counts"] = sorted(counts.values())
            body.pop("current_guess", None)
            self._check_name("tool result", json.dumps({**data, "result": body}, sort_keys=True))
        await tools.call("end_turn", {})
        return TurnUsage(calls=len(results) + 1)


# ---- ColoringGrid painters (M3b, crossed zones; swarmlab/world/coloring.py) --------------------
#
# Each paints at most `paints` cells per turn, chosen from its own observation of the target and
# the current grid (round-start state), then calls `end_turn`. Every painter skips cells it may not
# paint (zones of another painting group) and cells whose target it cannot see ("?"), so the first
# three never paint a zone whose targets are hidden from them: they never read or post.
#
# - `RowMajorPainter`: the first such cells, in row-major order, whose colour differs from the
#   target. Every such agent picks the same cells: the duplicate-work baseline.
# - `QueuePainter(ttl_rounds=1)`: the work-queue baseline. Picks such cells at random (its rng)
#   among those whose registry entry shows no live claim by another agent (`registry_get` at round
#   start; at most `max_checks` reads per turn), then `registry_acquire("cell:x,y", ttl_rounds)`
#   (the world's `claim_key`) and paints them in the same turn. Registry writes commit before world
#   actions, so under `enforced` the paint of an agent that lost the acquire race is rejected;
#   under `advisory` it goes through and counts as a claim violation.
# - `RandomPainter`: random paintable cells with random palette colours: the noise baseline.
# - `RelayPainter(paints=10, post=True)`: the crossed-zones relay baseline. Every turn it calls
#   `read_board(limit=200)` and keeps, for the zone its group paints, the first report it can
#   parse (`parse_report`: the zone id and the zone's cell count of colours); once (its first
#   turn) it posts the targets of the zone its group sees in the recommended format
#   (`Z0 red green ...`) when `post` is true. It then paints, up to `paints` cells: first the
#   cells of its paintable zone that differ from the parsed report (split among its group's
#   members by cell index), then the open cells it is responsible for (open cells in row-major
#   order, cell k belongs to agent k mod n in agent order). `post: false` is the never-posting
#   control: nobody learns a hidden zone, so the zones stay unpainted.


def _coloring_obs(view: View) -> tuple[list[str], list[str], list[str]]:
    from ..world.coloring import parse_observation as parse_coloring

    return parse_coloring(_observation_text(view))


def _paintable(view: View) -> list[tuple[int, int, str]]:
    from ..world.coloring import paintable_needed_cells

    return paintable_needed_cells(_observation_text(view))


class RowMajorPainter(Participant):
    entry_point: ClassVar[str | None] = "row_major_painter"
    description: ClassVar[str] = (
        "ColoringGrid: paints the first `paints` wrong cells it may paint, row by row"
    )

    def __init__(self, paints: int = 1) -> None:
        self.paints = paints

    async def turn(self, view: View, tools: AgentTools) -> TurnUsage:
        calls = 0
        for x, y, c in _paintable(view)[: self.paints]:
            await tools.call("paint", {"x": x, "y": y, "color": c})
            calls += 1
        await tools.call("end_turn", {})
        return TurnUsage(calls=calls + 1)


class QueuePainter(Participant):
    entry_point: ClassVar[str | None] = "queue_painter"
    description: ClassVar[str] = (
        "ColoringGrid: claims random wrong cells in the registry, then paints"
    )

    def __init__(self, paints: int = 1, ttl_rounds: int = 1, max_checks: int = 4) -> None:
        self.paints = paints
        self.ttl_rounds = ttl_rounds
        self.max_checks = max_checks

    async def turn(self, view: View, tools: AgentTools) -> TurnUsage:
        need = _paintable(view)
        self.rng.shuffle(need)
        calls = 0
        checks = 0
        done = 0
        for x, y, c in need:
            if done >= self.paints:
                break
            key = f"cell:{x},{y}"
            if checks < self.max_checks:
                checks += 1
                calls += 1
                res = await tools.call("registry_get", {"key": key})
                owner = res.result.get("owner") if res.ok else None
                if owner not in (None, self.agent):
                    continue
            await tools.call("registry_acquire", {"key": key, "ttl_rounds": self.ttl_rounds})
            await tools.call("paint", {"x": x, "y": y, "color": c})
            calls += 2
            done += 1
        await tools.call("end_turn", {})
        return TurnUsage(calls=calls + 1)


class RandomPainter(Participant):
    entry_point: ClassVar[str | None] = "random_painter"
    description: ClassVar[str] = (
        "ColoringGrid: paints `paints` random paintable cells with random colours"
    )

    def __init__(self, paints: int = 1) -> None:
        self.paints = paints

    async def turn(self, view: View, tools: AgentTools) -> TurnUsage:
        from ..world.coloring import parse_zones, zone_at

        text = _observation_text(view)
        target, _, colours = _coloring_obs(view)
        info = parse_zones(text)
        cells = [(x, y) for y in range(len(target)) for x in range(len(target[y]))
                 if (z := zone_at(info["zones"], x, y)) is None or z["painter_group"] == info["group"]]
        calls = 0
        if cells and colours:
            for _ in range(self.paints):
                x, y = self.rng.choice(cells)
                await tools.call("paint", {"x": x, "y": y, "color": self.rng.choice(colours)})
                calls += 1
        await tools.call("end_turn", {})
        return TurnUsage(calls=calls + 1)


class RelayPainter(Participant):
    entry_point: ClassVar[str | None] = "relay_painter"
    description: ClassVar[str] = (
        "ColoringGrid: posts its zone's targets, reads its painting zone's, paints its share"
    )

    def __init__(self, paints: int = 10, post: bool = True) -> None:
        self.paints = paints
        self.post = post

    def bind(self, agent: Any, rng: Any) -> None:
        super().bind(agent, rng)
        self.known: dict[str, list[str]] = {}
        self.posted = False

    async def turn(self, view: View, tools: AgentTools) -> TurnUsage:
        from ..world.coloring import format_report, needed_cells, parse_report, parse_zones, zone_at

        text = _observation_text(view)
        target, current, _ = _coloring_obs(view)
        info = parse_zones(text)
        g = info["group"]
        everyone = sorted(a for members in info["groups"].values() for a in members)
        mates = sorted(info["groups"].get(g, []))
        zones = info["zones"]
        mine = [z for z in zones if z["painter_group"] == g]
        calls = 1
        res = await tools.call("read_board", {"limit": 200})
        for item in res.result.get("items", []) if res.ok else []:
            for z in mine:
                n = z["width"] * z["height"]
                if z["zone_id"] not in self.known:
                    reports = parse_report(item.get("content", ""), z["zone_id"], n)
                    if reports:
                        self.known[z["zone_id"]] = reports[0]
        if self.post and not self.posted:
            for z in zones:
                if g in z["viewers"] and z["painter_group"] != g:
                    cols = [target[y][x] for y in range(z["y"], z["y"] + z["height"])
                            for x in range(z["x"], z["x"] + z["width"])]
                    if "?" not in cols:
                        await tools.call("post", {"channel": "main",
                                                  "text": format_report(z["zone_id"], cols)})
                        calls += 1
            self.posted = True
        todo: list[tuple[int, int, str]] = []
        for z in mine:
            want = self.known.get(z["zone_id"])
            if want is None:
                continue
            cells = [(x, y) for y in range(z["y"], z["y"] + z["height"])
                     for x in range(z["x"], z["x"] + z["width"])]
            for k, ((x, y), c) in enumerate(zip(cells, want, strict=True)):
                if mates and mates[k % len(mates)] == self.agent and current[y][x] != c:
                    todo.append((x, y, c))
        open_need = {(x, y): c for x, y, c in needed_cells(target, current)
                     if zone_at(zones, x, y) is None}
        open_cells = [(x, y) for y in range(len(target)) for x in range(len(target[y]))
                      if zone_at(zones, x, y) is None]
        for k, (x, y) in enumerate(open_cells):
            if everyone and everyone[k % len(everyone)] == self.agent and (x, y) in open_need:
                todo.append((x, y, open_need[(x, y)]))
        for x, y, c in todo[: self.paints]:
            await tools.call("paint", {"x": x, "y": y, "color": c})
            calls += 1
        await tools.call("end_turn", {})
        return TurnUsage(calls=calls + 1)
