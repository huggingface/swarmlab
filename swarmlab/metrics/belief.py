"""Belief metrics over committed world guesses (docs/INTERFACE.md §14, DESIGN.md §10).

Each metric tracks the latest accepted `guess` per agent from `action_committed` events
(`action["name"] == "guess"`, `accepted` true, belief = `action["args"]["candidate"]`).
Denominator: agents with a committed guess. With no guesses the value is None.

- `belief.accuracy`: share of guessing agents whose guess equals `truth["truth"]` (needs truth).
- `belief.consensus`: share of guessing agents on the modal belief.
- `belief.polarization(threshold=0.2)`: number of beliefs held by at least `threshold` of the
  guessing agents (DESIGN.md: "beliefs above a threshold share"), as a float. 1.0 means one
  camp, 2.0 two sizeable camps, and so on.
- `belief.entropy`: Shannon entropy in bits of the belief distribution.

M1a has no dead agents, so every guessing agent counts.
"""
from __future__ import annotations

import math
from collections import Counter
from typing import Any, ClassVar

from .base import Metric


class _BeliefMetric(Metric):
    def __init__(self) -> None:
        self.beliefs: dict[str, str] = {}

    def update(self, event: Any) -> None:
        if getattr(event, "type", None) != "action_committed" or not event.accepted:
            return
        action = event.action or {}
        if action.get("name") != "guess":
            return
        candidate = (action.get("args") or {}).get("candidate")
        if isinstance(candidate, str) and event.agent is not None:
            self.beliefs[event.agent] = candidate

    def _counts(self) -> Counter[str]:
        return Counter(self.beliefs.values())

    def value(self) -> tuple[float | None, int]:
        n = len(self.beliefs)
        if n == 0:
            return None, 0
        return self._value(self._counts(), n), n

    def _value(self, counts: Counter[str], n: int) -> float:
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

    def _value(self, counts: Counter[str], n: int) -> float:
        return counts.get(self.truth, 0) / n if self.truth is not None else 0.0


class Consensus(_BeliefMetric):
    entry_point: ClassVar[str | None] = "belief.consensus"
    name = "belief.consensus"

    def _value(self, counts: Counter[str], n: int) -> float:
        return max(counts.values()) / n


class Polarization(_BeliefMetric):
    entry_point: ClassVar[str | None] = "belief.polarization"
    name = "belief.polarization"

    def __init__(self, threshold: float = 0.2) -> None:
        super().__init__()
        self.threshold = threshold

    def _value(self, counts: Counter[str], n: int) -> float:
        return float(sum(1 for c in counts.values() if c / n >= self.threshold))


class Entropy(_BeliefMetric):
    entry_point: ClassVar[str | None] = "belief.entropy"
    name = "belief.entropy"

    def _value(self, counts: Counter[str], n: int) -> float:
        h = -sum((c / n) * math.log2(c / n) for c in counts.values())
        return h + 0.0  # normalise -0.0
