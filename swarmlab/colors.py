"""The colour-letter palette shared by the FlagGame image renderer and the replay viewer.

One source: `swarmlab/world/render.py` turns these into RGB for PNG crops and candidates, and
`swarmlab/viewer/build.py` injects them into the page's `COLOURS` table, so an image an agent saw
and the grid the viewer draws use the same colours. Letters follow `flaggame.COLOURS`.
"""
from __future__ import annotations

PALETTE_HEX: dict[str, str] = {
    "r": "#d62728",  # red
    "g": "#2ca02c",  # green
    "b": "#1f77b4",  # blue
    "y": "#f2c80f",  # yellow
    "k": "#111111",  # black
    "w": "#ffffff",  # white
    "o": "#ff7f0e",  # orange
    "p": "#7b4fa0",  # purple
    "c": "#17becf",  # cyan
    "m": "#e377c2",  # magenta
    "n": "#8c564b",  # brown
    "t": "#0f766e",  # teal
}


def hex_to_rgb(value: str) -> tuple[int, int, int]:
    v = value.lstrip("#")
    return int(v[0:2], 16), int(v[2:4], 16), int(v[4:6], 16)


PALETTE_RGB: dict[str, tuple[int, int, int]] = {k: hex_to_rgb(v) for k, v in PALETTE_HEX.items()}
