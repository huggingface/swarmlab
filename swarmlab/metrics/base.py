"""Metric base class and registry (docs/INTERFACE.md §14).

Additions after M1a:

- `description` (class attribute): one line shown by `swarmlab metrics` (`catalog()`) and
  `swarmlab plugins` (`swarmlab.registry.catalog`).
- `METRICS_REV` and `Metric.use_rev(rev)`: the revision of metric semantics a run was recorded
  with. New runs record `metrics_rev: METRICS_REV` in `run.json`; a run without it is revision 1
  and the runner calls `use_rev(1)` on every metric when it replays, resumes or reports on such a
  run, so a metric whose meaning changed (rev 2: `comm.posts_per_round` per agent instead of
  swarm-wide, probe-sourced belief metrics leaving skipped agents out of the denominator) folds
  old logs the way they were written and `replay` still matches. The default is a no-op.
- M6: `outputs()` lists the metric events one metric emits per round as `(name, value,
  denominator, label)`; the default is the single `(self.name, *self.value(), None)`. A metric
  with several outputs (`belief.state`: its state code with a string `label`, plus four one-hot
  series) overrides it; the runner logs one `metric` event per output and replay checks each.
- `round_resolved` events: a world whose `end_round(r)` returns a dict has it logged as a
  `round_resolved` event (right after the round's world commit), and metrics are fed it like any
  other logical event of the round. A metric that reports the world's per-round resolution folds
  it (`event.type == "round_resolved"`, `event.data`) instead of recomputing it from the
  `action_committed` events; replay feeds it the same logged event, so live and replay agree.
"""
from __future__ import annotations

from collections.abc import Callable
from importlib.metadata import entry_points
from typing import Any, ClassVar

from ..base import Persistable, Plugin
from ..registry import catalog as group_catalog

METRICS_REV = 2


class Metric(Persistable, Plugin):
    name: str = "metric"
    description: str = ""

    def use_rev(self, rev: int) -> None:
        """Fold with the semantics of metrics revision `rev` (see the module doc). No-op default."""

    def update(self, event: Any) -> None:
        raise NotImplementedError

    def value(self) -> tuple[float | None, int]:
        raise NotImplementedError

    def outputs(self) -> list[tuple[str, float | None, int, str | None]]:
        """M6: the metric events of this round, `(name, value, denominator, label)`."""
        value, denom = self.value()
        return [(self.name, value, denom, None)]

    def needs_truth(self) -> bool:
        return False

    def set_truth(self, truth: dict) -> None:
        pass

    # True for metrics over agents' beliefs (belief.*): the runner passes them the live agents
    # minus `World.excluded_from_belief()` (WP16) instead of every live agent.
    belief_population: ClassVar[bool] = False

    def set_agents(self, agents: list[Any]) -> None:
        """The runner calls this at reset and whenever the live agent list changes. No-op default."""

    @staticmethod
    def from_fn(name: str, fn: Callable[[list[Any]], tuple[float | None, int]]) -> Metric:
        """One-liner metric: fn receives all logical events seen so far."""

        class _FnMetric(Metric):
            entry_point = None

            def __init__(self) -> None:
                self.name = name
                self.events: list[Any] = []

            def update(self, event: Any) -> None:
                self.events.append(event)

            def value(self) -> tuple[float | None, int]:
                return fn(self.events)

        _FnMetric.__qualname__ = f"FnMetric[{name}]"
        return _FnMetric()


def catalog() -> list[tuple[str, str]]:
    """(entry-point name, description) of every registered metric, sorted by name."""
    return [(name, desc) for name, desc, _ in group_catalog("swarmlab.metrics")]


def get(name: str, **params: Any) -> Metric:
    """Resolve a metric by entry-point name, or `module:Class`."""
    for ep in entry_points(group="swarmlab.metrics"):
        if ep.name == name:
            cls = ep.load()
            return cls(**params)
    if ":" in name:
        import importlib

        mod, _, qual = name.partition(":")
        cls = importlib.import_module(mod)
        for part in qual.split("."):
            cls = getattr(cls, part)
        return cls(**params)
    raise KeyError(f"unknown metric {name!r}")
