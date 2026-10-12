"""World base class (docs/INTERFACE.md §8).

Required: reset, observe, score, and at least one @tool-decorated action. Everything else has a
working default. Nothing agent-visible may reveal correctness; score() and verify() are
evaluator-only and called by the runner alone.

`swarmlab plugins` shows the first line of a world class's docstring: `World` cannot carry the
`description` class attribute other plugin bases have, because `description()` is the
agent-facing task text (M1b). Give every world class a one-line docstring summary.

Tool schemas. `@tool` stores a `_ToolSpec(name, description, params)` on the method as
`__swarmlab_tool__`. The name is always static (it is what `_actions()` keys on); `description`,
`params` as a whole, or any single param value may be a callable of the world instance, so a
schema can follow constructor config (an enum of `n` sites). `World.tool_schema(name)` resolves a
spec into a normalized `ToolSchema` for this instance. It is the one authoritative hook:
`tool_schemas()` and the default `validate()` both call `self.tool_schema(name)`, so an override of
it is what the model is shown and what is checked.
A fully static spec is resolved (and its JSON types checked) once, at decoration time; a dynamic
one is resolved on every call and checked then. Nothing is cached per instance.

Callback lifecycle. Schema callables run whenever a schema is built: at least once per round for
advertising (the executor reads `tool_schemas()` once per round) and again on every `validate`.
They must therefore be pure, side-effect-free functions of constructor config (and of state fixed
at `reset`, which never changes within a run), never of per-round state; otherwise the schema
advertised and the one validated could differ. Per-round schema changes are not supported.

Default validation checks argument names (required and no extras) and, for every property of the
resolved schema with an `enum`, that the value is one of it. JSON types, ranges and anything else
are the world author's job inside the action.
"""
from __future__ import annotations

import random
from collections.abc import Callable, Sequence
from dataclasses import dataclass
from typing import TYPE_CHECKING, Any

from pydantic import BaseModel

from ..base import Persistable, Plugin
from ..ids import ActionId, AgentId
from ..interventions import NotSupported
from ..tools import ToolSchema
from ..view import Observation

if TYPE_CHECKING:  # swarmlab.experiment imports this module
    from ..experiment import Run


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


ParamValue = str | dict | Callable[[Any], str | dict]
Params = dict[str, ParamValue] | Callable[[Any], dict[str, str | dict]]


def _property(key: str, value: str | dict) -> dict:
    if isinstance(value, str):
        if value not in _JSON_TYPES:
            raise ValueError(f"unknown JSON type {value!r} for param {key!r}")
        return {"type": value}
    return dict(value)


@dataclass(frozen=True)
class _ToolSpec:
    """What `@tool` records on a method; `schema(world)` resolves it for one instance."""

    name: str
    description: str | Callable[[Any], str]
    params: Params | None
    static: ToolSchema | None = None  # the resolved schema when nothing is callable

    def schema(self, world: Any) -> ToolSchema:
        if self.static is not None:
            return self.static
        desc = self.description(world) if callable(self.description) else self.description
        params = self.params(world) if callable(self.params) else (self.params or {})
        props = {k: _property(k, v(world) if callable(v) else v) for k, v in params.items()}
        return ToolSchema(
            name=self.name,
            description=desc,
            parameters={"type": "object", "properties": props, "required": list(props),
                        "additionalProperties": False},
        ).normalized()


def tool(name: str, description: str | Callable[[Any], str], params: Params | None = None) -> Callable:
    """Mark a World method as an agent-callable action.

    `params` maps argument name to a JSON-schema type name ("integer") or a full property schema;
    every argument is required. The decorated method has signature
    (self, agent: AgentId, **args) -> Outcome.

    `description`, `params`, or any single value of a `params` dict may instead be a callable
    taking the world instance (`self`) and returning the same shape, for schemas that depend on
    constructor config: `{"site": lambda self: {"type": "string", "enum": self.sites}}`. Static
    parts are checked here (an unknown JSON type raises ValueError); callable parts are checked
    when the schema is built (`World.tool_schema`), with the same error.

    Callables run whenever a schema is built: at least once per round (advertising) and on every
    `validate`. They must be pure and side-effect free, depending only on constructor config (or
    state fixed at `reset`), never on per-round state, so the advertised and validated schemas
    agree. Per-round schema changes are out of scope.
    """
    dynamic = callable(description) or callable(params)
    if not callable(params):
        for k, v in (params or {}).items():
            if callable(v):
                dynamic = True
            else:
                _property(k, v)
    spec = _ToolSpec(name, description, params)
    if not dynamic:
        spec = _ToolSpec(name, description, params, static=spec.schema(None))

    def deco(fn: Callable) -> Callable:
        fn.__swarmlab_tool__ = spec  # type: ignore[attr-defined]
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

    def tool_schema(self, name: str) -> ToolSchema:
        """The normalized schema of world tool `name` for this instance (KeyError if unknown).

        The one authoritative hook: `tool_schemas()` and the default `validate()` both call it, so
        an override (for example adding an argument) is advertised and validated alike.
        """
        return self._actions()[name].__swarmlab_tool__.schema(self)  # type: ignore[attr-defined]

    def tool_schemas(self) -> list[ToolSchema]:
        schemas = [self.tool_schema(name) for name in self._actions()]
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

        The default checks, against `self.tool_schema(action.name)`, argument names only (every
        required one present, no extras) and `enum` membership (a value outside a property's
        `enum` is refused). JSON types, ranges and anything else are the action's job.
        """
        if action.name not in self._actions():
            return Ack(ok=False, error=f"unknown action {action.name!r}")
        schema = self.tool_schema(action.name).parameters
        props = schema.get("properties", {})
        missing = [k for k in schema.get("required", []) if k not in action.args]
        extra = [k for k in action.args if k not in props]
        if missing or extra:
            return Ack(ok=False, error=f"bad args: missing={missing} extra={extra}")
        for k, v in action.args.items():
            allowed = props[k].get("enum")
            if allowed is not None and v not in allowed:
                return Ack(ok=False, error=f"bad args: {k} must be one of {allowed}")
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

    def report_sections(self, arms: dict[str, list[Run]]) -> list[str]:
        """Evaluator-side Markdown lines appended to `swarmlab report` after its generic
        "Final score" table (which already lists every scalar key of the recorded `score()`).

        Called once per world type in the report: runs are grouped by `run.spec.world.type`,
        and this world is built (not reset, so use constructor config only) from the first
        reconstructed run of its group. `arms` maps arm name to that arm's finished runs of this
        world type only (`Run`: `score`, `metrics`, `events`, `spec`, ...; an arm without such a
        run is absent); read per-run state from the runs. A run whose plugins did not import
        when the report was built is a plain `Run` read from run.json (no replay; its
        `experiment` cannot be built). With several world types in one report, the lines are
        placed under a `### World <type>` heading. Default: no lines.
        """
        return []

    # ---- per-agent hooks (WP16: Flag Game blind agents) ----------------------------------------
    def allows_tool(self, agent: AgentId, name: str) -> bool:
        """False hides the world tool `name` from `agent` (not offered; a call is `not_allowed`).
        Default: every world tool is offered to every agent."""
        return True

    def description_for(self, agent: AgentId) -> str:
        """The task description shown to `agent` (what the runner puts in `View.description`).
        Default: `description()` for everyone; override when agents' tasks differ."""
        return self.description()

    def excluded_from_belief(self) -> list[AgentId]:
        """Agents left out of belief metrics' populations (numerator and denominator), e.g. agents
        with no private evidence of their own. The runner removes them from the live list it
        passes to belief metrics' `set_agents`. Default: none."""
        return []

    # ---- round, claim and failure hooks (M3b; swarmlab/medium/registry.py) ----------------------
    def begin_round(self, round: int) -> None:
        """Called by the runner at the start of every round, before any observation (no-op)."""

    def end_round(self, round: int) -> dict | None:
        """Called by the runner once per round right after the world commit (default: None).

        Under `commit: round_end` it runs after `commit_world`; under `commit: immediate` after
        the last turn's commit, at the same point of the round. Either way it runs before the
        round's commit events (`post*`, `delivery*`, `action_committed*`) are logged, and before
        interventions, probes, metrics, the budget event and the snapshot. The world may mutate
        its state here (resolve the round from the committed actions); the snapshot that follows
        captures it, so resume and fork see the resolved state. Round-end interventions are not
        visible to it: the round's outcome follows from the committed actions alone.

        A returned JSON-able dict is logged as a `round_resolved` event (`data` field), which
        metrics may fold; None logs nothing. Evaluator-only: never shown to agents. Replay does
        not call it (it folds the logged event and restores the snapshot).
        """
        return None

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
