"""Turn order per round (docs/INTERFACE.md §11).

`Scheduler.order(round, live, rng)` returns the live agents in the order their turns are run
(immediate mode) and their buffers are committed (both modes). The runner passes
`derive(seed, "schedule", round)` as `rng`; a scheduler must draw only from that stream.

Decisions where the contract is silent:

- The base class returns `live` unchanged (agent order), so a subclass that wants a fixed order
  needs no code. `SeededShuffle` copies `live` and shuffles it with the passed rng.
- The base class is registered as `fixed` (entry point in `swarmlab.schedulers`): every live agent
  acts every round, in agent order a000, a001, ...; `Scheduler().spec()` is
  `{"type": "fixed", "params": {}}`. A run.json holding the older `swarmlab.scheduler:Scheduler`
  still resolves through `module:Class`. Subclasses keep their own name or `module:Class`
  (`entry_point` counts only where it is defined, swarmlab/base.py).
- The scheduler is not part of `RunSpec` in M1a; the runner always uses `SeededShuffle`. It is
  still snapshotted under the key `"scheduler"` so stateful schedulers can be added later.

M6 (docs/INTERFACE-M6.md §2): `OneSpeaker(listeners=1)`, entry point `one_speaker` (group
`swarmlab.schedulers`), returns exactly one live agent per round, `rng.choice(live)` (uniform), so
a round is one asynchronous step of the Flag Game paper's pairwise protocol. The listener is not
chosen here: with `commit: immediate` and `topology: gossip(k=1)` the round's gossip partner draw
gives the speaker's single listener. Selection (decisions where the contract is silent):

- `RunOptions.scheduler` (a plugin spec, `None` = `SeededShuffle`) picks the scheduler; it is left
  out of the serialised options while `None`, so earlier spec hashes are unchanged.
  `build_scheduler(spec)` resolves it through `registry.build(spec, "swarmlab.schedulers")` (entry
  point, then `module:Class`); there is no separate name table. An unknown name raises the
  registry's `ValueError`; a class that is not a `Scheduler` raises `TypeError`.
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
from .registry import build


class Scheduler(Persistable, Plugin):
    """Every live agent acts every round, in agent order (entry point `fixed`)."""

    entry_point: ClassVar[str | None] = "fixed"
    description: ClassVar[str] = "Every live agent acts every round, in agent order a000, a001, ..."

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


def build_scheduler(spec: Any = None) -> Scheduler:
    """A scheduler from `RunOptions.scheduler`: `None` -> `SeededShuffle`, a `Scheduler` instance
    as is, else a name, a `{type, params}` mapping or a `PluginSpec`, built by
    `registry.build(spec, "swarmlab.schedulers")` (entry point, then `module:Class`)."""
    if spec is None:
        return SeededShuffle()
    if isinstance(spec, Scheduler):
        return spec
    if isinstance(spec, str):
        spec = {"type": spec, "params": {}}
    sched = build(spec, "swarmlab.schedulers")
    if not isinstance(sched, Scheduler):
        raise TypeError(f"scheduler {spec!r} builds a {type(sched).__qualname__}, not a Scheduler")
    return sched
