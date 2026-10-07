"""ColoringGrid and claim metrics (DESIGN.md §10; M3b).

The grid metrics need the truth (`ColoringGrid.verify()`: initial target, target changes, sizes,
`paints_per_round`) and fold the log: `round_started` applies the round's target changes and
counts the round's live agents (`len(order)`), each accepted `paint` in `action_committed`
(result `painted` or `overwritten`) is applied in log order, which is the world's commit order,
with the same accounting as the world (swarmlab/world/coloring.py), so the final values equal
`score()`.

- `coloring.coverage`: matching cells / cells; denominator cells.
- `coloring.duplicate_paints`, `coloring.wrong_paints`: cumulative counts; denominator the
  applied paints so far.
- `coloring.parallel_efficiency`: (matching cells / agent-rounds) / `paints_per_round`, i.e. the
  correct cells per agent-round relative to the single-agent ideal of `paints_per_round` correct
  cells per round; denominator agent-rounds so far (live agents summed over rounds).

Claim metrics fold `claim` and `registry` events (no truth needed):

- `claims.violations`: cumulative number of world actions on a key whose live owner was another
  agent (under `enforced` those actions were also rejected); denominator the keyed actions so far.
- `claims.held`: live claims after this round's commit (acquired, not released, not expired);
  denominator the round's live agents.
"""
from __future__ import annotations

from typing import Any, ClassVar

from .base import Metric


class _GridFold(Metric):
    def __init__(self) -> None:
        self.truth: dict | None = None
        self.target: list[list[str]] | None = None
        self.grid: list[list[str]] | None = None
        self.paints = 0
        self.duplicates = 0
        self.wrong = 0
        self.agent_rounds = 0

    def needs_truth(self) -> bool:
        return True

    def set_truth(self, truth: dict) -> None:
        self.truth = dict(truth)
        if self.target is None:
            self.target = [list(r) for r in truth.get("initial_target", truth.get("target", []))]
            self.grid = [["."] * len(r) for r in self.target]

    def update(self, event: Any) -> None:
        t = getattr(event, "type", None)
        if self.target is None or self.grid is None:
            return
        if t == "round_started":
            self.agent_rounds += len(event.order)
            for r, cells in (self.truth or {}).get("target_changes", []):
                if r == event.round:
                    for x, y, c in cells:
                        self.target[y][x] = c
        elif t == "action_committed" and event.accepted and event.action.get("name") == "paint":
            if event.feedback.get("result") not in ("painted", "overwritten"):
                return
            args = event.action.get("args", {})
            x, y, c = args["x"], args["y"], args["color"]
            want, before = self.target[y][x], self.grid[y][x]
            self.paints += 1
            if c != want:
                self.wrong += 1
            elif before == want:
                self.duplicates += 1
            self.grid[y][x] = c

    def _cells(self) -> int:
        return sum(len(r) for r in self.target or [])

    def _matching(self) -> int:
        if not self.target or not self.grid:
            return 0
        return sum(1 for tr, gr in zip(self.target, self.grid, strict=True)
                   for a, b in zip(tr, gr, strict=True) if a == b)


class Coverage(_GridFold):
    entry_point: ClassVar[str | None] = "coloring.coverage"
    name = "coloring.coverage"

    def value(self) -> tuple[float | None, int]:
        cells = self._cells()
        return (self._matching() / cells if cells else None), cells


class DuplicatePaints(_GridFold):
    entry_point: ClassVar[str | None] = "coloring.duplicate_paints"
    name = "coloring.duplicate_paints"

    def value(self) -> tuple[float | None, int]:
        return float(self.duplicates), self.paints


class WrongPaints(_GridFold):
    entry_point: ClassVar[str | None] = "coloring.wrong_paints"
    name = "coloring.wrong_paints"

    def value(self) -> tuple[float | None, int]:
        return float(self.wrong), self.paints


class ParallelEfficiency(_GridFold):
    entry_point: ClassVar[str | None] = "coloring.parallel_efficiency"
    name = "coloring.parallel_efficiency"

    def value(self) -> tuple[float | None, int]:
        if not self.agent_rounds:
            return None, 0
        ppr = int((self.truth or {}).get("paints_per_round", 1)) or 1
        return self._matching() / self.agent_rounds / ppr, self.agent_rounds


class ClaimViolations(Metric):
    entry_point: ClassVar[str | None] = "claims.violations"
    name = "claims.violations"

    def __init__(self) -> None:
        self.violations = 0
        self.keyed = 0

    def update(self, event: Any) -> None:
        if getattr(event, "type", None) == "claim":
            self.keyed += 1
            self.violations += int(event.violation)

    def value(self) -> tuple[float | None, int]:
        return float(self.violations), self.keyed


class ClaimsHeld(Metric):
    entry_point: ClassVar[str | None] = "claims.held"
    name = "claims.held"

    def __init__(self) -> None:
        self.claims: dict[str, int] = {}  # key -> expires_round of its live claim
        self.round = 0
        self.live = 0

    def update(self, event: Any) -> None:
        t = getattr(event, "type", None)
        if t == "round_started":
            self.round = event.round
            self.live = len(event.order)
        elif t == "registry" and event.ok and event.op in ("acquire", "release"):
            if event.owner is not None and event.expires_round is not None:
                self.claims[event.key] = event.expires_round
            else:
                self.claims.pop(event.key, None)

    def value(self) -> tuple[float | None, int]:
        return float(sum(1 for exp in self.claims.values() if exp >= self.round)), self.live
