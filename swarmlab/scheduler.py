"""Turn order per round (docs/INTERFACE.md §11).

`Scheduler.order(round, live, rng)` returns the live agents in the order their turns are run
(immediate mode) and their buffers are committed (both modes). The runner passes
`derive(seed, "schedule", round)` as `rng`; a scheduler must draw only from that stream.

Decisions where the contract is silent:

- The base class returns `live` unchanged (agent order), so a subclass that wants a fixed order
  needs no code. `SeededShuffle` copies `live` and shuffles it with the passed rng.
- The scheduler is not part of `RunSpec` in M1a; the runner always uses `SeededShuffle`. It is
  still snapshotted under the key `"scheduler"` so stateful schedulers can be added later.
"""
from __future__ import annotations

import random
from typing import ClassVar

from .base import Persistable, Plugin
from .ids import AgentId


class Scheduler(Persistable, Plugin):
    def order(self, round: int, live: list[AgentId], rng: random.Random) -> list[AgentId]:
        return list(live)


class SeededShuffle(Scheduler):
    entry_point: ClassVar[str | None] = "seeded_shuffle"

    def order(self, round: int, live: list[AgentId], rng: random.Random) -> list[AgentId]:
        out = list(live)
        rng.shuffle(out)
        return out
