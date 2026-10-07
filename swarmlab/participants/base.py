"""Participant base class (docs/INTERFACE.md §10)."""
from __future__ import annotations

import random

from pydantic import BaseModel

from ..base import Persistable, Plugin
from ..ids import AgentId
from ..interventions import NotSupported
from ..tools import AgentTools
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

    async def turn(self, view: View, tools: AgentTools) -> TurnUsage:
        raise NotImplementedError

    def reconfigure(self, **kw: object) -> None:
        """Change settings from the next turn on (M3a `Ops.reconfigure`); default: not supported."""
        raise NotSupported(f"{type(self).__name__} does not support reconfigure")
