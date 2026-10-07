"""Belief metrics over committed world guesses (docs/INTERFACE.md §14, DESIGN.md §10).

Each metric tracks the latest accepted `guess` per agent from `action_committed` events
(`action["name"] == "guess"`, `accepted` true, belief = `action["args"]["candidate"]`).

Denominator (review B2): **all live agents**, which the runner supplies through
`set_agents(agents)` at reset and whenever the live list changes (replay derives it from each
`round_started.order`). A live agent with no committed guess is an explicit `"none"` belief.
Guesses by agents that are not live are ignored. Agents the world names in
`World.excluded_from_belief()` (FlagGame's blind agents, WP16) are not in the list the runner
passes, so they count in neither numerator nor denominator. With no live agents the value is None (denominator
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
parse) resets it to `"none"`. A skipped probe (`parsed["skipped"]`: a scripted agent without a
probe context, an exhausted measurement budget, the hard ceiling, a provider error) takes the
agent out of the denominator (and the numerator) until its next answered probe: nobody asked
it, so it is neither `"none"` nor its older answer. Before metrics revision 2 (runs whose
`run.json` has no `metrics_rev`, replayed with `use_rev(1)`) a skip left the last answer
standing and an agent that never answered counted as `"none"`. The metric's `name` gains the suffix
`@probe:<name>` (`belief.consensus@probe:belief`), so the same entry point can run twice side by
side; the runner requires unique names, not unique entry points. Same denominator rules, less
the skipped agents.
`source="world"` is left out of `params`, so M1a specs and their hashes are unchanged.

M6 (docs/INTERFACE-M6.md §5): a guess's belief is the world's canonical name when the
`action_committed` feedback carries one (`feedback["candidate"]`, FlagGame with real flags, where
guesses match country names case-insensitively), else `args["candidate"]` as before. The live list
for replay comes from `round_started.live` when logged (OneSpeaker), else `order`.

`belief.state(consensus=0.85, camp=0.25, source="world")` is the Flag Game paper's terminal
classification, computed every round over the live (non-blind) agents **that hold a guess**
(`"none"` and skipped agents are out of numerator and denominator): `s1` = the top candidate's
share, `correct_consensus` if `s1 >= consensus` (with 1e-9 slack) on the truth, `wrong_consensus`
if on another candidate, `polarized` if `s1 < consensus` and at least two candidates each hold
`>= camp`, else `fragmented` (`classify_state`). It emits (`outputs()`) the event `belief.state`
with `label` = the class and `value` = its index in `STATES` (0..3; None and no label when nobody
holds a guess), plus one 0/1 series per class, `belief.state.<class>` (all None when nobody holds
a guess), each with the same denominator. With a probe source every name gains `@probe:<name>`.
`consensus`/`camp` are in `params` only when not at their defaults.
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
    belief_population: ClassVar[bool] = True  # WP16: the runner leaves out excluded_from_belief()

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
        self.skipped: set[str] = set()  # probe source: agents whose latest probe was skipped
        self._rev1 = False

    def use_rev(self, rev: int) -> None:
        self._rev1 = rev < 2

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
        candidate = (getattr(event, "feedback", None) or {}).get("candidate")  # M6: canonical name
        if not isinstance(candidate, str):
            candidate = (action.get("args") or {}).get("candidate")
        if isinstance(candidate, str) and event.agent is not None:
            self.beliefs[event.agent] = candidate

    def _update_probe(self, event: Any) -> None:
        if getattr(event, "type", None) != "probe" or event.probe != self.probe or event.agent is None:
            return
        parsed = event.parsed or {}
        if "skipped" in parsed:  # not asked (scripted agent, budget, error)
            if not self._rev1:  # revision 1: the last answer stands
                self.skipped.add(event.agent)
            return
        self.skipped.discard(event.agent)
        candidate = parsed.get("candidate")
        if event.ok and isinstance(candidate, str):
            self.beliefs[event.agent] = candidate
        else:  # a failed parse counts as "none"
            self.beliefs.pop(event.agent, None)

    def _distribution(self) -> tuple[Counter[str], int]:
        """(counts of candidate beliefs among live agents, number of live agents without one)."""
        agents = self.agents if self.agents is not None else sorted(self.beliefs)
        if self.skipped:
            agents = [a for a in agents if a not in self.skipped]
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


STATES = ("correct_consensus", "wrong_consensus", "polarized", "fragmented")


def classify_state(counts: Counter[str] | dict[str, int], truth: str | None,
                   consensus: float = 0.85, camp: float = 0.25) -> str | None:
    """The paper's class of a belief distribution (counts per candidate, no "none"); None if
    empty. Ties for the top share only matter below `consensus` (two candidates cannot both
    reach a consensus threshold above 0.5)."""
    n = sum(counts.values())
    if n == 0:
        return None
    eps = 1e-9
    top_name, top = max(sorted(counts.items()), key=lambda kv: kv[1])
    if top / n >= consensus - eps:
        return "correct_consensus" if top_name == truth else "wrong_consensus"
    if sum(1 for c in counts.values() if c / n >= camp - eps) >= 2:
        return "polarized"
    return "fragmented"


class State(_BeliefMetric):
    entry_point: ClassVar[str | None] = "belief.state"
    name = "belief.state"
    description = ("Flag Game paper class of the guess distribution: correct/wrong consensus, "
                   "polarized, fragmented (label, plus one-hot belief.state.<class> series)")

    def __init__(self, consensus: float = 0.85, camp: float = 0.25, source: str = "world") -> None:
        if not (0 < camp <= consensus <= 1):
            raise ValueError("belief.state needs 0 < camp <= consensus <= 1")
        self._init(source)
        self.consensus, self.camp = float(consensus), float(camp)
        if isinstance(getattr(self, "params", None), dict):
            for k, default in (("consensus", 0.85), ("camp", 0.25)):
                if self.params.get(k) == default:
                    self.params.pop(k)
        self.truth: str | None = None

    def needs_truth(self) -> bool:
        return True

    def set_truth(self, truth: dict) -> None:
        self.truth = truth.get("truth")

    def label(self) -> tuple[str | None, int]:
        counts, _ = self._distribution()
        return classify_state(counts, self.truth, self.consensus, self.camp), sum(counts.values())

    def value(self) -> tuple[float | None, int]:
        label, n = self.label()
        return (None if label is None else float(STATES.index(label))), n

    def output_names(self) -> list[str]:
        suffix = self.name[len("belief.state"):]  # "" or "@probe:<name>"
        return [self.name] + [f"belief.state.{s}{suffix}" for s in STATES]

    def outputs(self) -> list[tuple[str, float | None, int, str | None]]:
        label, n = self.label()
        names = self.output_names()
        out = [(names[0], None if label is None else float(STATES.index(label)), n, label)]
        out += [(nm, None if label is None else float(label == s), n, None)
                for nm, s in zip(names[1:], STATES, strict=True)]
        return out
