"""What an agent sees at the start of its turn (docs/INTERFACE.md §7)."""
from typing import Any, Literal

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
    description: str = ""  # M1b: World.description(), the task text for prompt templates


def describe(world: Any, agent: str) -> str:
    """`world.description_for(agent)` when the world has it (WP16), else `world.description()`."""
    per_agent = getattr(world, "description_for", None)
    return per_agent(agent) if callable(per_agent) else world.description()


def text_observation(text: str, **private: Any) -> Observation:
    return Observation(parts=[Part(type="text", text=text)], private=dict(private))
