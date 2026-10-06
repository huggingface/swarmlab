"""What an agent sees at the start of its turn (docs/INTERFACE.md §7)."""
from typing import Literal

from pydantic import BaseModel

from .ids import AgentId
from .tools import ToolSchema


class Part(BaseModel):
    type: Literal["text", "image"]
    text: str | None = None
    image_png_b64: str | None = None


class Observation(BaseModel):
    parts: list[Part]
    private: dict = {}  # world-defined, evaluator-only; never shared with other agents


class View(BaseModel):
    round: int
    agent: AgentId
    observation: Observation
    outcomes: list[dict]  # this agent's action_committed feedback from the previous round
    pushed: list[dict]  # delivered inbox items when delivery == "push", else []
    tools: list[ToolSchema]
