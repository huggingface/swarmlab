"""Importable test fixtures (plugins must live at module level so `module:Class` specs resolve)."""
from __future__ import annotations

import os
import time
from pathlib import Path

from swarmlab import Board, Experiment, Participant, TurnUsage
from swarmlab.participants import EvidenceAggregator
from swarmlab.world.flaggame import FlagGame

PAUSE_ENV = "SWARMLAB_TEST_PAUSE"


class PausingFlagGame(FlagGame):
    """FlagGame that blocks inside its 7th commit when $SWARMLAB_TEST_PAUSE names a marker file.

    The 7th commit happens after round 7's turn events are in the log and before its
    `round_committed`, which gives the recovery test a deterministic kill window.
    """

    def commit(self, actions):
        self.commits = getattr(self, "commits", 0) + 1
        marker = os.environ.get(PAUSE_ENV)
        if marker and self.commits == 7:
            Path(marker).write_text("paused")
            time.sleep(120)
        return super().commit(actions)


class Chatter(Participant):
    """Posts its agent id, reads the board, then yields without end_turn (no_tool)."""

    async def turn(self, view, tools):
        await tools.call("post", {"text": f"hello from {self.agent} r{view.round}"})
        res = await tools.call("read_board", {})
        self.seen = getattr(self, "seen", []) + [i["content"] for i in res.result["items"]]
        return TurnUsage(calls=2)


def flag_experiment(n_agents: int = 8, *, world=None, medium=None, name: str = "flag", **kw) -> Experiment:
    return Experiment(
        name=name,
        world=world if world is not None else FlagGame(),
        participants=[EvidenceAggregator()] * n_agents,
        medium=medium if medium is not None else Board(topology="gossip"),
        metrics=["belief.consensus", "belief.accuracy", "belief.entropy", "belief.polarization",
                 "comm.read_rate", "comm.posts_per_round", "comm.hops"],
        **kw,
    )


def logical(run, exclude=("seq", "ts")):
    return list(run_events(run, exclude))


def run_events(run, exclude=("seq", "ts")):
    from swarmlab.events import logical_view

    return logical_view(run.events_all, exclude=exclude)


class LouderAggregator(EvidenceAggregator):
    """Subclass of a registered participant; must serialise as itself, not as its parent."""

    def __init__(self, volume: int = 2) -> None:
        self.volume = volume


class BigFlag(FlagGame):
    """Subclass of the registered world with no constructor of its own."""


OPERATIONAL_HOOK: dict = {}


class Inferrer(Participant):
    """Calls OPERATIONAL_HOOK["fn"](agent, round) mid-turn (stands in for M1b's tools.infer)."""

    async def turn(self, view, tools):
        fn = OPERATIONAL_HOOK.get("fn")
        if fn is not None:
            fn(self.agent, view.round)
        await tools.call("post", {"text": f"{self.agent} r{view.round}"})
        await tools.call("end_turn", {})
        return TurnUsage(calls=2)
