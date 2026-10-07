"""Build plugins from `{"type", "params"}` specs (docs/INTERFACE.md §0).

`resolve(type_name, group)` finds the class: first an entry point named `type_name` in `group`,
then `module:Qual.Name` import. `build(spec, group)` instantiates it with `params`.
Groups used by `Experiment.from_spec`: `swarmlab.worlds`, `swarmlab.participants`,
`swarmlab.metrics`; the board resolves `swarmlab.topologies` and `swarmlab.policies` itself.
"""
from __future__ import annotations

import importlib
from collections.abc import Mapping
from importlib.metadata import entry_points
from typing import Any

GROUPS = (
    "swarmlab.worlds",
    "swarmlab.topologies",
    "swarmlab.policies",
    "swarmlab.participants",
    "swarmlab.metrics",
    "swarmlab.claim_policies",
)


def resolve(type_name: str, group: str) -> type:
    for ep in entry_points(group=group):
        if ep.name == type_name:
            return ep.load()
    if ":" in type_name:
        module, _, qualname = type_name.partition(":")
        obj: Any = importlib.import_module(module)
        for part in qualname.split("."):
            obj = getattr(obj, part)
        return obj
    raise ValueError(f"unknown {group} plugin {type_name!r} (no entry point, not module:Class)")


def build(spec: Mapping[str, Any] | Any, group: str) -> Any:
    """Instantiate a plugin from a dict or a `PluginSpec`-like object with `type`/`params`."""
    if not isinstance(spec, Mapping):
        spec = {"type": spec.type, "params": dict(spec.params)}
    cls = resolve(spec["type"], group)
    return cls(**dict(spec.get("params") or {}))
