"""Tool schemas, calls, results, the ToolExecutor protocol, and AgentTools (docs/INTERFACE.md §6).

The executor is the only path that mutates world, board, or registry. The runner implements it
(swarmlab/executor.py) and never hands it to a participant: each turn receives an `AgentTools`
handle bound to one agent. The handle keeps the executor in a name-mangled slot and exposes
only `agent`, `schemas()`, `call(name, args)` and (M1b) `infer(request, *, category="swarm")`, so a participant has no API to act or read as
another agent and no path to the world or board objects. (Python cannot make that airtight
against deliberate introspection; the point is that no supported or accidental path exists.)
"""
from __future__ import annotations

from typing import TYPE_CHECKING, Any, Protocol

from pydantic import BaseModel

from .ids import AgentId, CallId

if TYPE_CHECKING:
    from .providers.base import ChatRequest, ChatResponse


def _normalize_object(schema: dict) -> dict:
    """A copy of an object schema with `properties`, `required` and `additionalProperties: false`.

    Nested property schemas (and array `items`) are normalised recursively when they are objects
    that declare `properties` or `additionalProperties: false`. A nested object with neither, or one
    that explicitly allows extra keys (`additionalProperties` true or a schema), is a deliberate
    free-form object and is left as is; it makes the tool non-strict (see `strict_violations`).
    """
    out = dict(schema)
    if out.get("type") == "object" and out.get("additionalProperties", False) is False:
        out["properties"] = {k: _normalize_property(v)
                             for k, v in (out.get("properties") or {}).items()}
        out["required"] = [k for k in (out.get("required") or []) if k in out["properties"]]
        out["additionalProperties"] = False
    return out


def _normalize_property(schema: dict) -> dict:
    if not isinstance(schema, dict):
        return schema
    out = dict(schema)
    if out.get("type") == "object":
        if "properties" not in out and "additionalProperties" not in out:
            return out  # free-form object (JSON-schema default): left open, makes the tool non-strict
        return _normalize_object(out)
    if out.get("type") == "array" and isinstance(out.get("items"), dict):
        out["items"] = _normalize_property(out["items"])
    return out


def strict_violations(schema: dict, path: str = "$") -> list[str]:
    """Where `schema` breaks Anthropic strict tool-use rules (empty list = strict-compatible).

    Checked: every `type: object` (at any depth, including array `items` and `anyOf`/`oneOf`/
    `allOf` branches) has a `properties` dict, a `required` list naming only declared properties,
    and `additionalProperties: false`.
    """
    out: list[str] = []
    if not isinstance(schema, dict):
        return out
    if schema.get("type") == "object":
        props = schema.get("properties")
        req = schema.get("required")
        if not isinstance(props, dict):
            out.append(f"{path}: object without properties")
            props = {}
        if not isinstance(req, list):
            out.append(f"{path}: object without a required list")
        elif any(k not in props for k in req):
            out.append(f"{path}: required names undeclared properties")
        if schema.get("additionalProperties") is not False:
            out.append(f"{path}: additionalProperties is not false")
        for k, v in props.items():
            out += strict_violations(v, f"{path}.{k}")
    if isinstance(schema.get("items"), dict):
        out += strict_violations(schema["items"], f"{path}[]")
    for key in ("anyOf", "oneOf", "allOf"):
        for i, branch in enumerate(schema.get(key) or []):
            out += strict_violations(branch, f"{path}.{key}[{i}]")
    return out


class ToolSchema(BaseModel):
    name: str
    description: str
    parameters: dict  # JSON schema for the arguments

    def normalized(self) -> ToolSchema:
        """A copy whose `parameters` is a complete object schema.

        Top level: `type: object`, `properties` (default `{}`), `required` (default `[]`, names
        not in `properties` dropped) and `additionalProperties: false`; nested objects likewise
        unless they explicitly allow extra keys. Idempotent. Every schema the harness emits
        (world `@tool`s, board, status and `end_turn` tools) is already normalised; provider
        adapters call this again defensively.
        """
        params = dict(self.parameters or {})
        params.setdefault("type", "object")
        params.setdefault("additionalProperties", False)
        return self.model_copy(update={"parameters": _normalize_object(params)})

    def strict_violations(self) -> list[str]:
        return strict_violations(self.parameters)


class ToolCall(BaseModel):
    call_id: CallId
    name: str
    args: dict


class ToolResult(BaseModel):
    call_id: CallId
    ok: bool
    result: dict
    pending: bool = False
    error: str | None = None


class EndTurn(Exception):
    """Kept for backward compatibility; no longer raised (end_turn() returns ok=True)."""


class TurnCapReached(Exception):
    """Raised by the executor when max_calls_per_turn is exceeded."""


class ToolExecutor(Protocol):
    def schemas(self, agent: AgentId) -> list[ToolSchema]: ...
    async def call(self, agent: AgentId, name: str, args: dict) -> ToolResult: ...
    async def infer(self, agent: AgentId, request: Any, category: str = "swarm") -> Any: ...


class AgentTools:
    """The agent-bound tool handle a participant receives in `turn(view, tools)`."""

    __slots__ = ("__agent", "__executor")

    def __init__(self, executor: ToolExecutor, agent: AgentId) -> None:
        self.__executor = executor
        self.__agent = agent

    @property
    def agent(self) -> AgentId:
        return self.__agent

    def schemas(self) -> list[ToolSchema]:
        return self.__executor.schemas(self.__agent)

    async def call(self, name: str, args: dict | None = None) -> ToolResult:
        return await self.__executor.call(self.__agent, name, dict(args or {}))

    async def infer(self, request: ChatRequest, *, category: str = "swarm") -> ChatResponse:
        """Run one model call through the harness (gate, ledger, cache, operational events).

        `category` is "swarm" for the agent's own turn and "measurement" for probes.
        """
        return await self.__executor.infer(self.__agent, request, category)

    def __repr__(self) -> str:
        return f"AgentTools(agent={self.__agent!r})"
