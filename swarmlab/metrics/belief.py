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

M1b `source` param (docs/INTERFACE-M1b.md §6), on every belief metric: `"world"` (default, the
committed guesses above) or `"probe:<name>"`, which reads `probe` events of that probe instead:
`ok` with a string `parsed["candidate"]` sets the agent's belief, any other answer (a failed
parse) resets it to `"none"`, and skipped probes (`parsed["skipped"]`: scripted agents, an
exhausted measurement budget) leave it unchanged. The metric's `name` gains the suffix
`@probe:<name>` (`belief.consensus@probe:belief`), so the same entry point can run twice side by
side; the runner requires unique names, not unique entry points. Same denominator rules.
`source="world"` is left out of `params`, so M1a specs and their hashes are unchanged.
"""
from __future__ import annotations

import math
from collections import Counter
from typing import Any, ClassVar

from .base import Metric

NONE = "none"


def _check_source(source: str) -> str:
    if source != "world" and not (source.startswith("probe:") and len(source) > len("probe:")):
        raise ValueError(f"source must be 'world' or 'probe:<name>', got {source!r}")
    return source


class _BeliefMetric(Metric):
    def __init__(self, source: str = "world") -> None:
        self._init(source)

    def _init(self, source: str) -> None:
        self.source = _check_source(source)
        if source == "world" and isinstance(getattr(self, "params", None), dict):
            self.params.pop("source", None)  # the default stays out of the spec (M1a spec hashes)
        self.probe = source[len("probe:"):] if source != "world" else None
        base = type(self).name
        self.name = base if self.probe is None else f"{base}@{source}"
        self.beliefs: dict[str, str] = {}
        self.agents: list[str] | None = None

    def set_agents(self, agents: list[Any]) -> None:
        self.agents = sorted(str(a) for a in agents)

    def update(self, event: Any) -> None:
        if self.probe is not None:
            self._update_probe(event)
            return
        if getattr(event, "type", None) != "action_committed" or not event.accepted:
            return
        action = event.action or {}
        if action.get("name") != "guess":
            return
        candidate = (action.get("args") or {}).get("candidate")
        if isinstance(candidate, str) and event.agent is not None:
            self.beliefs[event.agent] = candidate

    def _update_probe(self, event: Any) -> None:
        if getattr(event, "type", None) != "probe" or event.probe != self.probe or event.agent is None:
            return
        parsed = event.parsed or {}
        if "skipped" in parsed:  # not asked (scripted agent, budget): the last answer stands
            return
        candidate = parsed.get("candidate")
        if event.ok and isinstance(candidate, str):
            self.beliefs[event.agent] = candidate
        else:  # a failed parse counts as "none"
            self.beliefs.pop(event.agent, None)

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
    description = "share of live agents whose belief is the truth (world guesses or probes)"

    def __init__(self, source: str = "world") -> None:
        self._init(source)
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
    description = "largest share of live agents holding one candidate (world guesses or probes)"

    def _value(self, counts: Counter[str], none: int, n: int) -> float:
        return max(counts.values(), default=0) / n


class Polarization(_BeliefMetric):
    entry_point: ClassVar[str | None] = "belief.polarization"
    name = "belief.polarization"
    description = "number of candidates held by at least `threshold` (0.2) of live agents"

    def __init__(self, threshold: float = 0.2, source: str = "world") -> None:
        self._init(source)
        self.threshold = threshold

    def _value(self, counts: Counter[str], none: int, n: int) -> float:
        return float(sum(1 for c in counts.values() if c / n >= self.threshold))


class Entropy(_BeliefMetric):
    entry_point: ClassVar[str | None] = "belief.entropy"
    name = "belief.entropy"
    description = "Shannon entropy (bits) of the live agents' beliefs, 'none' included"

    def _value(self, counts: Counter[str], none: int, n: int) -> float:
        parts = [*counts.values(), none]
        return -sum((c / n) * math.log2(c / n) for c in parts if c)
