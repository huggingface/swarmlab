"""Convenience import path used in DESIGN.md (`from swarmlab.worlds import FlagGame`)."""
from .world.base import World
from .world.flaggame import FlagGame

__all__ = ["FlagGame", "World"]
