"""World base class (docs/INTERFACE.md §8).

Required: reset, observe, score, and at least one @tool-decorated action. Everything else has a
working default. Nothing agent-visible may reveal correctness; score() and verify() are
evaluator-only and called by the runner alone.
"""
from __future__ import annotations

import random
from collections.abc import Callable, Sequence

from pydantic import BaseModel

from ..base import Persistable, Plugin
from ..ids import ActionId, AgentId
from ..interventions import NotSupported
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

    def validate(self, agent: AgentId, action: Action, pending: Sequence[Action] = ()) -> Ack:
        """Check `action` before it is buffered (round_end) or committed (immediate).

        `pending` is the agent's own world actions already buffered this round, in call order
        (always empty under immediate commit, where every action has already been committed).
        A world uses it to refuse at call time what its commit would certainly reject, such as a
        per-round limit. The executor passes `pending` only to overrides that accept it, so
        worlds written against `validate(agent, action)` keep working.
        """
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

    def my_status(self, agent: AgentId, pending: Sequence[Action] = ()) -> dict | None:
        """The agent's own status, or None (the tool is not exposed). `pending` is as in
        `validate`, passed only to overrides that accept it."""
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

    def render_state(self) -> dict | None:
        """The world's state for the replay page's "World state" panel, or None (no panel).

        Evaluator-side (agents never see it, so it may show the truth). Called by the viewer
        builder on a fresh instance restored from each round's snapshot (the state after that
        round's commit), so it must depend on snapshotted state and constructor config only.
        Return JSON-able data; the page renders, per key:

        - `grid`, and any other key holding a list of equal-length lists of scalars when the dict
          has `palette` (or `colors`), a mapping from cell value to CSS colour: a coloured grid
          (unknown cell values in grey; FlagGame/ColoringGrid letters have default colours);
        - a list of flat dicts: a small table; a flat dict: key/value rows;
        - a scalar: a key/value row; anything else: JSON text.
        """
        return None

    # ---- per-agent hooks (WP16: Flag Game blind agents) ----------------------------------------
    def allows_tool(self, agent: AgentId, name: str) -> bool:
        """False hides the world tool `name` from `agent` (not offered; a call is `not_allowed`).
        Default: every world tool is offered to every agent."""
        return True

    def excluded_from_belief(self) -> list[AgentId]:
        """Agents left out of belief metrics' populations (numerator and denominator), e.g. agents
        with no private evidence of their own. The runner removes them from the live list it
        passes to belief metrics' `set_agents`. Default: none."""
        return []

    # ---- round, claim and failure hooks (M3b; swarmlab/medium/registry.py) ----------------------
    def begin_round(self, round: int) -> None:
        """Called by the runner at the start of every round, before any observation (no-op)."""

    def claim_key(self, agent: AgentId, action: Action) -> str | None:
        """The registry key of the resource `action` works on, or None (not claimable)."""
        return None

    def killed_at(self, round: int) -> list[AgentId]:
        """Agents that fail at the start of `round` (the runner removes them from the live set)."""
        return []

    # ---- intervention hooks (M3a, docs/INTERFACE-M3a.md §1) -----------------------------------
    def patch_private(self, agent: AgentId, data: dict) -> None:
        """Change `agent`'s private information; deliver it as new evidence in its next observation."""
        raise NotSupported(f"{type(self).__name__} does not support patch_private")

    def intervene(self, name: str, /, **args: object) -> dict:
        """A world-specific change (e.g. FlagGame `set_truth`); returns a JSON-able result."""
        raise NotSupported(f"{type(self).__name__} does not support intervention {name!r}")
