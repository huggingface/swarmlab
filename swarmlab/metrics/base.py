"""Metric base class and registry (docs/INTERFACE.md §14)."""
from __future__ import annotations

from collections.abc import Callable
from importlib.metadata import entry_points
from typing import Any

from ..base import Persistable, Plugin


class Metric(Persistable, Plugin):
    name: str = "metric"

    def update(self, event: Any) -> None:
        raise NotImplementedError

    def value(self) -> tuple[float | None, int]:
        raise NotImplementedError

    def needs_truth(self) -> bool:
        return False

    def set_truth(self, truth: dict) -> None:
        pass

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
