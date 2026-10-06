"""World protocol (docs/INTERFACE.md §8).

Nothing returned by tools, observe, validate, commit, my_status, or collective_status may reveal
correctness. score() and verify() are evaluator-only and called by the runner alone.
"""
import random
from typing import Protocol

from pydantic import BaseModel

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

    action_id: ActionId
    accepted: bool
    feedback: dict


class World(Protocol):
    name: str

    def reset(self, rng: random.Random, params: dict, agents: list[AgentId]) -> None: ...
    def tools(self, agent: AgentId) -> list[ToolSchema]: ...
    def observe(self, agent: AgentId) -> Observation: ...
    def validate(self, agent: AgentId, action: Action) -> Ack: ...
    def commit(self, actions: list[tuple[AgentId, ActionId, Action]]) -> list[Outcome]: ...
    def my_status(self, agent: AgentId) -> dict: ...
    def collective_status(self) -> dict: ...
    def score(self) -> dict: ...
    def terminal(self) -> bool: ...
    def verify(self) -> dict: ...
    def snapshot(self) -> bytes: ...
    def restore(self, blob: bytes) -> None: ...
