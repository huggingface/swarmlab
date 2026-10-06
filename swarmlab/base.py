"""Plugin plumbing shared by every extension point (docs/INTERFACE.md §0, §8-§14).

`Persistable` gives snapshot/restore for ordinary Python state so plugin authors never touch
checkpoint manifests. `Plugin` records constructor kwargs as `params` and exposes `spec()`.
"""
from __future__ import annotations

import inspect
import pickle
from typing import Any, ClassVar


class Persistable:
    """Default persistence: pickle of __dict__ minus `params`, every attribute whose name starts
    with `_` (transient caches), and any name in `_skip_in_snapshot`. Override for non-plain state.
    """

    _skip_in_snapshot: ClassVar[tuple[str, ...]] = ("params",)

    def snapshot(self) -> bytes:
        skip = self._skip_in_snapshot
        state = {
            k: v for k, v in self.__dict__.items()
            if k != "params" and not k.startswith("_") and k not in skip
        }
        return pickle.dumps(state, protocol=pickle.HIGHEST_PROTOCOL)

    def restore(self, blob: bytes) -> None:
        self.__dict__.update(pickle.loads(blob))


class Plugin:
    """Records constructor kwargs so `spec()` can serialise the instance for hashing and rebuild.

    Subclasses either call `super().__init__(**kwargs)` or rely on `__init_subclass__` wrapping,
    which captures the bound arguments of the subclass's own `__init__` automatically.
    `entry_point` is the name under the plugin's entry-point group; None means `module:Class`.
    It is honoured only when defined on the class itself, so a subclass of a registered plugin
    serialises as its own `module:Class`, never as its parent (INTERFACE §3).
    """

    entry_point: ClassVar[str | None] = None
    params: dict[str, Any]

    def __init_subclass__(cls, **kw: Any) -> None:
        super().__init_subclass__(**kw)
        init = cls.__dict__.get("__init__")
        if init is None:
            return
        sig = inspect.signature(init)

        def wrapped(self: Plugin, *args: Any, **kwargs: Any) -> None:
            bound = sig.bind(self, *args, **kwargs)
            bound.apply_defaults()
            params = dict(bound.arguments)
            params.pop("self", None)
            # keep the outermost (most derived) constructor's params
            if not hasattr(self, "params"):
                self.params = params
            init(self, *args, **kwargs)

        wrapped.__signature__ = sig  # type: ignore[attr-defined]
        wrapped.__doc__ = init.__doc__
        wrapped.__name__ = "__init__"
        cls.__init__ = wrapped  # type: ignore[method-assign]

    def __init__(self, **kwargs: Any) -> None:
        if not hasattr(self, "params"):
            self.params = dict(kwargs)

    @classmethod
    def type_name(cls) -> str:
        ep = cls.__dict__.get("entry_point")
        if ep:
            return ep
        return f"{cls.__module__}:{cls.__qualname__}"

    def spec(self) -> dict[str, Any]:
        return {"type": self.type_name(), "params": _plain(getattr(self, "params", {}))}


def _plain(value: Any) -> Any:
    """Make params JSON-serialisable: nested plugins become their spec()."""
    if isinstance(value, Plugin):
        return value.spec()
    if isinstance(value, dict):
        return {k: _plain(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [_plain(v) for v in value]
    return value
