"""Build plugins from `{"type", "params"}` specs (docs/INTERFACE.md §0).

`resolve(type_name, group)` finds the class: first an entry point named `type_name` in `group`,
then `module:Qual.Name` import. `build(spec, group)` instantiates it with `params`.
Groups used by `Experiment.from_spec`: `swarmlab.worlds`, `swarmlab.participants`,
`swarmlab.metrics`; the board resolves `swarmlab.topologies` and `swarmlab.policies` itself.

`add_import_dir(path)` puts a directory at the front of `sys.path` (moving it there if it is
already on it), so `module:Class` plugins in files next to an experiment YAML import without
`PYTHONPATH`. `Experiment.from_yaml` calls it with the YAML's directory, and the runner records
that directory in `run.json["import_dir"]` so `Run.experiment` (replay, resume, fork, view of a
run dir) can do the same from any working directory.
"""
from __future__ import annotations

import importlib
import sys
from collections.abc import Mapping
from importlib.metadata import entry_points
from pathlib import Path
from typing import Any

GROUPS = (
    "swarmlab.worlds",
    "swarmlab.topologies",
    "swarmlab.policies",
    "swarmlab.participants",
    "swarmlab.metrics",
    "swarmlab.claim_policies",
    "swarmlab.schedulers",
)


def add_import_dir(path: Path | str) -> str:
    """Put `path` (a directory) at the front of `sys.path`; returns it as a resolved string."""
    d = str(Path(path).resolve())
    if not sys.path or sys.path[0] != d:
        while d in sys.path:
            sys.path.remove(d)
        sys.path.insert(0, d)
        importlib.invalidate_caches()
    return d


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
