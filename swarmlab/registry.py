"""Build plugins from `{"type", "params"}` specs (docs/INTERFACE.md §0).

`resolve(type_name, group)` finds the class: first an entry point named `type_name` in `group`,
then `module:Qual.Name` import. `normalize(spec, group)` returns the spec as a plain
`{"type", "params"}` dict in the form `Plugin.spec()` writes now; `build(spec, group)` instantiates
the class with the normalised `params`. Both take a mapping or a `PluginSpec`-like object.
`flat_params(cls, params)` is the lower-level step `normalize` applies: it accepts the form
`Plugin.spec()` wrote before it flattened `**name` arguments (`{"kwargs": {...}}`, nested under
the parameter's name), so older run.json files still rebuild and compare equal (the runner's fork
normalises the parent's participant specs this way). `normalize` leaves `params` as given when
the type does not resolve; `build` raises then.
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
import inspect
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
    cls, norm = _resolved(spec, group)
    return cls(**norm["params"])


def normalize(spec: Mapping[str, Any] | Any, group: str) -> dict[str, Any]:
    """`spec` (a dict or `PluginSpec`-like) as `{"type", "params"}` with `flat_params` applied;
    `params` unchanged when the type does not resolve."""
    try:
        return _resolved(spec, group)[1]
    except (ImportError, AttributeError, ValueError):
        t, params = _parts(spec)
        return {"type": t, "params": params}


def _parts(spec: Mapping[str, Any] | Any) -> tuple[str, dict[str, Any]]:
    if isinstance(spec, Mapping):
        return spec["type"], dict(spec.get("params") or {})
    return spec.type, dict(spec.params or {})


def _resolved(spec: Mapping[str, Any] | Any, group: str) -> tuple[type, dict[str, Any]]:
    t, params = _parts(spec)
    cls = resolve(t, group)
    return cls, {"type": t, "params": flat_params(cls, params)}


def flat_params(cls: Any, params: Mapping[str, Any]) -> dict[str, Any]:
    """`params` with a dict under `cls`'s `**name` parameter merged into the top level."""
    out = dict(params)
    try:
        sig = inspect.signature(cls)
    except (TypeError, ValueError):
        return out
    var_kw = next((p.name for p in sig.parameters.values() if p.kind is p.VAR_KEYWORD), None)
    if var_kw is not None and isinstance(out.get(var_kw), dict):
        out.update(out.pop(var_kw))
    return out
