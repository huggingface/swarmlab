"""Seed derivation. Every component draws from its own stream derived from the run seed.

Fixed label roots (see docs/INTERFACE.md §2): ("world",), ("private", agent), ("schedule", round),
("topology", round), ("agent", agent), ("scripted", agent). Changing one component's draws must
never change another's, which holds because each stream is seeded independently.
"""
import hashlib
import random


def derive(seed: int, *labels: str | int) -> random.Random:
    h = hashlib.sha256()
    h.update(str(seed).encode())
    for label in labels:
        h.update(b"\x00")
        h.update(str(label).encode())
    return random.Random(int.from_bytes(h.digest()[:8], "big"))
