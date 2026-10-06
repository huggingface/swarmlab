"""Scripted participants for FlagGame (docs/INTERFACE.md §10).

All three draw randomness only from the rng passed to `bind` (the runner passes
`derive(seed, "agent", agent)`), keep plain Python state (pickled by `Persistable`), and parse
the world's observation with `swarmlab.world.flaggame.parse_observation`.

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


def _observation_text(view: View) -> str:
    return "\n".join(p.text or "" for p in view.observation.parts if p.type == "text")


class Silent(Participant):
    entry_point: ClassVar[str | None] = "silent"

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
        stripped = text
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
