"""Belief metrics over committed world guesses (docs/INTERFACE.md §14, DESIGN.md §10).

Each metric tracks the latest accepted `guess` per agent from `action_committed` events
(`action["name"] == "guess"`, `accepted` true, belief = `action["args"]["candidate"]`).

Denominator (review B2): **all live agents**, which the runner supplies through
`set_agents(agents)` at reset and whenever the live list changes (replay derives it from each
`round_started.order`). A live agent with no committed guess is an explicit `"none"` belief.
Guesses by agents that are not live are ignored. With no live agents the value is None (denominator
0). Standalone use without `set_agents` falls back to the agents that have guessed.

- `belief.accuracy`: share of live agents whose guess equals `truth["truth"]` (needs truth). This
  equals FlagGame's `score()["accuracy"]` after every round.
- `belief.consensus`: the largest share of live agents holding one candidate ("none" excluded
  from the numerator, included in the denominator); 0.0 when nobody has guessed.
- `belief.polarization(threshold=0.2)`: number of candidates (not "none") held by at least
  `threshold` of the live agents (DESIGN.md: "beliefs above a threshold share"), as a float.
- `belief.entropy`: Shannon entropy in bits of the belief distribution including "none".
"""
from __future__ import annotations

import math
from collections import Counter
from typing import Any, ClassVar

from .base import Metric

NONE = "none"


class _BeliefMetric(Metric):
    def __init__(self) -> None:
        self.beliefs: dict[str, str] = {}
        self.agents: list[str] | None = None

    def set_agents(self, agents: list[Any]) -> None:
        self.agents = sorted(str(a) for a in agents)

    def update(self, event: Any) -> None:
        if getattr(event, "type", None) != "action_committed" or not event.accepted:
            return
        action = event.action or {}
        if action.get("name") != "guess":
            return
        candidate = (action.get("args") or {}).get("candidate")
        if isinstance(candidate, str) and event.agent is not None:
            self.beliefs[event.agent] = candidate

    def _distribution(self) -> tuple[Counter[str], int]:
        """(counts of candidate beliefs among live agents, number of live agents without one)."""
        agents = self.agents if self.agents is not None else sorted(self.beliefs)
        counts: Counter[str] = Counter(self.beliefs[a] for a in agents if a in self.beliefs)
        return counts, len(agents) - sum(counts.values())

    def value(self) -> tuple[float | None, int]:
        counts, none = self._distribution()
        n = sum(counts.values()) + none
        if n == 0:
            return None, 0
        return self._value(counts, none, n) + 0.0, n  # + 0.0 normalises -0.0

    def _value(self, counts: Counter[str], none: int, n: int) -> float:
        raise NotImplementedError


class Accuracy(_BeliefMetric):
    entry_point: ClassVar[str | None] = "belief.accuracy"
    name = "belief.accuracy"

    def __init__(self) -> None:
        super().__init__()
        self.truth: str | None = None

    def needs_truth(self) -> bool:
        return True

    def set_truth(self, truth: dict) -> None:
        self.truth = truth.get("truth")

    def _value(self, counts: Counter[str], none: int, n: int) -> float:
        return counts.get(self.truth, 0) / n if self.truth is not None else 0.0


class Consensus(_BeliefMetric):
    entry_point: ClassVar[str | None] = "belief.consensus"
    name = "belief.consensus"

    def _value(self, counts: Counter[str], none: int, n: int) -> float:
        return max(counts.values(), default=0) / n


class Polarization(_BeliefMetric):
    entry_point: ClassVar[str | None] = "belief.polarization"
    name = "belief.polarization"

    def __init__(self, threshold: float = 0.2) -> None:
        super().__init__()
        self.threshold = threshold

    def _value(self, counts: Counter[str], none: int, n: int) -> float:
        return float(sum(1 for c in counts.values() if c / n >= self.threshold))


class Entropy(_BeliefMetric):
    entry_point: ClassVar[str | None] = "belief.entropy"
    name = "belief.entropy"

    def _value(self, counts: Counter[str], none: int, n: int) -> float:
        parts = [*counts.values(), none]
        return -sum((c / n) * math.log2(c / n) for c in parts if c)
