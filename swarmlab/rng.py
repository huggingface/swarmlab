"""Seed derivation. Every component draws from its own stream derived from the run seed.

Fixed label roots (see docs/INTERFACE.md §2): ("world",), ("private", agent), ("schedule", round),
("topology", round), ("agent", agent), ("scripted", agent). Changing one component's draws must
never change another's, which holds because each stream is seeded independently.

`derive` returns a plain `random.Random` that also records where it came from as an instance
attribute `root = (seed, *labels)`. A component that needs the run seed
itself, not only a stream (ColoringGrid's `board: auto` maps seed 1..5 to its fixed boards 00..04),
reads `getattr(rng, "root", None)` and falls back to drawing from the stream when it is absent
(a `random.Random` not made by `derive`). `root` is not part of the generator state:
`getstate`/`setstate`, copies and pickles carry only the stream (`random.Random.__reduce__`), so
pickled participant rngs are unchanged.
"""
import hashlib
import random


def derive(seed: int, *labels: str | int) -> random.Random:
    h = hashlib.sha256()
    h.update(str(seed).encode())
    for label in labels:
        h.update(b"\x00")
        h.update(str(label).encode())
    rng = random.Random(int.from_bytes(h.digest()[:8], "big"))
    rng.root = (seed, *labels)  # type: ignore[attr-defined]
    return rng
