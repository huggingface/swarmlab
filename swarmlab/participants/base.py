"""Participant base class (docs/INTERFACE.md §10)."""
from __future__ import annotations

import random

from pydantic import BaseModel

from ..base import Persistable, Plugin
from ..ids import AgentId
from ..tools import ToolExecutor
from ..view import View


class TurnUsage(BaseModel):
    calls: int = 0
    prompt_tokens: int = 0
    completion_tokens: int = 0
    cost_usd: float = 0.0


class Participant(Persistable, Plugin):
    agent: AgentId
    rng: random.Random

    def bind(self, agent: AgentId, rng: random.Random) -> None:
        self.agent = agent
        self.rng = rng

    async def turn(self, view: View, tools: ToolExecutor) -> TurnUsage:
        raise NotImplementedError
