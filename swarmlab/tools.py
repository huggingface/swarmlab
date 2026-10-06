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


class ToolSchema(BaseModel):
    name: str
    description: str
    parameters: dict  # JSON schema for the arguments


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
