"""World base class (docs/INTERFACE.md §8).

Required: reset, observe, score, and at least one @tool-decorated action. Everything else has a
working default. Nothing agent-visible may reveal correctness; score() and verify() are
evaluator-only and called by the runner alone.
"""
from __future__ import annotations

import random
from collections.abc import Callable

from pydantic import BaseModel

from ..base import Persistable, Plugin
from ..ids import ActionId, AgentId
from ..tools import ToolSchema
from ..view import Observation


class Action(BaseModel):
    name: str
    args: dict


class Ack(BaseModel):
    """Pre-commit sanity check only; no mutation happens at validate time."""

    ok: bool
    error: str | None = None


class Outcome(BaseModel):
    """Agent-visible result of a committed action. Never carries correctness."""

    accepted: bool
    feedback: dict = {}
    action_id: ActionId | None = None  # filled in by the runner


# The parameters of a tool without arguments: a complete, strict-compatible object schema.
NO_ARGS: dict = {"type": "object", "properties": {}, "required": [], "additionalProperties": False}

_JSON_TYPES = {"integer", "number", "string", "boolean", "array", "object"}


def tool(name: str, description: str, params: dict[str, str | dict] | None = None) -> Callable:
    """Mark a World method as an agent-callable action.

    `params` maps argument name to a JSON-schema type name ("integer") or a full property schema.
    The decorated method has signature (self, agent: AgentId, **args) -> Outcome.
    """
    props: dict[str, dict] = {}
    for k, v in (params or {}).items():
        props[k] = {"type": v} if isinstance(v, str) else dict(v)
        if isinstance(v, str) and v not in _JSON_TYPES:
            raise ValueError(f"unknown JSON type {v!r} for param {k!r}")
    schema = ToolSchema(
        name=name,
        description=description,
        parameters={"type": "object", "properties": props, "required": list(props), "additionalProperties": False},
    ).normalized()

    def deco(fn: Callable) -> Callable:
        fn.__swarmlab_tool__ = schema  # type: ignore[attr-defined]
        return fn

    return deco


class World(Persistable, Plugin):
    name: str = "world"

    # ---- required -------------------------------------------------------------------------
    def reset(self, rng: random.Random, agents: list[AgentId]) -> None:
        raise NotImplementedError

    def observe(self, agent: AgentId) -> Observation:
        raise NotImplementedError

    def score(self) -> dict:
        raise NotImplementedError

    # ---- defaults -------------------------------------------------------------------------
    def _actions(self) -> dict[str, Callable]:
        out: dict[str, Callable] = {}
        for cls in type(self).__mro__:
            for attr, fn in vars(cls).items():
                schema = getattr(fn, "__swarmlab_tool__", None)
                if schema is not None and schema.name not in out:
                    out[schema.name] = getattr(self, attr)
        return out

    def tool_schemas(self) -> list[ToolSchema]:
        schemas = [fn.__swarmlab_tool__ for fn in self._actions().values()]  # type: ignore[attr-defined]
        if self.my_status(AgentId("a000")) is not None:
            schemas.append(ToolSchema(name="my_status", description="Status of your own work.",
                                      parameters=NO_ARGS))
        if self.collective_status() is not None:
            schemas.append(ToolSchema(name="collective_status", description="Status of the swarm's work.",
                                      parameters=NO_ARGS))
        return [s.normalized() for s in schemas]

    def validate(self, agent: AgentId, action: Action) -> Ack:
        fn = self._actions().get(action.name)
        if fn is None:
            return Ack(ok=False, error=f"unknown action {action.name!r}")
        schema = fn.__swarmlab_tool__.parameters  # type: ignore[attr-defined]
        missing = [k for k in schema.get("required", []) if k not in action.args]
        extra = [k for k in action.args if k not in schema.get("properties", {})]
        if missing or extra:
            return Ack(ok=False, error=f"bad args: missing={missing} extra={extra}")
        return Ack(ok=True)

    def commit(self, actions: list[tuple[AgentId, ActionId, Action]]) -> list[Outcome]:
        """Apply actions in the given (seeded) order. Override for conflict semantics."""
        fns = self._actions()
        outcomes: list[Outcome] = []
        for agent, action_id, action in actions:
            fn = fns.get(action.name)
            if fn is None:
                out = Outcome(accepted=False, feedback={"error": "unknown action"})
            else:
                out = fn(agent, **action.args)
            out.action_id = action_id
            outcomes.append(out)
        return outcomes

    def my_status(self, agent: AgentId) -> dict | None:
        return None

    def collective_status(self) -> dict | None:
        return None

    def terminal(self) -> bool:
        return False

    def description(self) -> str:
        """Agent-facing task description (M1b), used by prompt templates; "" when not provided.

        Must not reveal correctness, and should describe the task only: how agents coordinate
        (for example whether to read the board) is the experiment's business.
        """
        return ""

    def verify(self) -> dict:
        return {}
