"""Turn order per round (docs/INTERFACE.md §11).

`Scheduler.order(round, live, rng)` returns the live agents in the order their turns are run
(immediate mode) and their buffers are committed (both modes). The runner passes
`derive(seed, "schedule", round)` as `rng`; a scheduler must draw only from that stream.

Decisions where the contract is silent:

- The base class returns `live` unchanged (agent order), so a subclass that wants a fixed order
  needs no code. `SeededShuffle` copies `live` and shuffles it with the passed rng.
- The scheduler is not part of `RunSpec` in M1a; the runner always uses `SeededShuffle`. It is
  still snapshotted under the key `"scheduler"` so stateful schedulers can be added later.

M6 (docs/INTERFACE-M6.md §2): `OneSpeaker(listeners=1)`, entry point `one_speaker` (group
`swarmlab.schedulers`), returns exactly one live agent per round, `rng.choice(live)` (uniform), so
a round is one asynchronous step of the Flag Game paper's pairwise protocol. The listener is not
chosen here: with `commit: immediate` and `topology: gossip(k=1)` the round's gossip partner draw
gives the speaker's single listener. Selection (decisions where the contract is silent):

- `RunOptions.scheduler` (a plugin spec, `None` = `SeededShuffle`) picks the scheduler; it is left
  out of the serialised options while `None`, so earlier spec hashes are unchanged.
  `build_scheduler(spec)` resolves it (entry point, then `module:Class`).
- `listeners` is checked, not used: the runner refuses a `gossip` topology whose `k` differs from
  `listeners` (any other topology is allowed and decides the recipients itself).
- `turns_per_round(n)` says how many turns a round runs for `n` live agents (`n` by default, 1
  here); `swarmlab estimate` uses it to price a round.
- With no live agents `order` returns `[]`.
"""
from __future__ import annotations

import random
from typing import Any, ClassVar

from .base import Persistable, Plugin
from .ids import AgentId


class Scheduler(Persistable, Plugin):
    def turns_per_round(self, n: int) -> int:
        """Turns a round runs with `n` live agents (used by `swarmlab estimate`)."""
        return n

    def order(self, round: int, live: list[AgentId], rng: random.Random) -> list[AgentId]:
        return list(live)


class SeededShuffle(Scheduler):
    entry_point: ClassVar[str | None] = "seeded_shuffle"

    def order(self, round: int, live: list[AgentId], rng: random.Random) -> list[AgentId]:
        out = list(live)
        rng.shuffle(out)
        return out


class OneSpeaker(Scheduler):
    """One uniformly drawn live agent acts per round (the Flag Game paper's pairwise step)."""

    entry_point: ClassVar[str | None] = "one_speaker"

    def __init__(self, listeners: int = 1) -> None:
        if not isinstance(listeners, int) or isinstance(listeners, bool) or listeners < 1:
            raise ValueError(f"OneSpeaker listeners must be an integer >= 1, got {listeners!r}")
        self.listeners = listeners

    def order(self, round: int, live: list[AgentId], rng: random.Random) -> list[AgentId]:
        return [rng.choice(list(live))] if live else []

    def turns_per_round(self, n: int) -> int:
        return min(1, n)


SCHEDULERS: dict[str, type[Scheduler]] = {"seeded_shuffle": SeededShuffle, "one_speaker": OneSpeaker}


def build_scheduler(spec: Any = None) -> Scheduler:
    """A scheduler from `RunOptions.scheduler` (`None` -> `SeededShuffle`, a name, a
    `{type, params}` mapping or a `PluginSpec`)."""
    if spec is None:
        return SeededShuffle()
    if isinstance(spec, Scheduler):
        return spec
    if isinstance(spec, str):
        spec = {"type": spec, "params": {}}
    elif not isinstance(spec, dict):
        spec = {"type": spec.type, "params": dict(spec.params)}
    cls = SCHEDULERS.get(spec["type"])
    if cls is None:
        from .registry import resolve

        cls = resolve(spec["type"], "swarmlab.schedulers")
    sched = cls(**dict(spec.get("params") or {}))
    if not isinstance(sched, Scheduler):
        raise TypeError(f"{spec!r} does not build a Scheduler")
    return sched
