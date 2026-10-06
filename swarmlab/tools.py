"""Tool schemas, calls, results, and the ToolExecutor protocol (docs/INTERFACE.md §6).

The executor is the only path that mutates world, board, or registry. Participants call it;
the runner implements it (swarmlab/executor.py, WP4).
"""
from typing import Protocol

from pydantic import BaseModel

from .ids import AgentId, CallId


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
    """Raised by the executor when the participant calls end_turn()."""


class TurnCapReached(Exception):
    """Raised by the executor when max_calls_per_turn is exceeded."""


class ToolExecutor(Protocol):
    def schemas(self, agent: AgentId) -> list[ToolSchema]: ...
    async def call(self, agent: AgentId, name: str, args: dict) -> ToolResult: ...
