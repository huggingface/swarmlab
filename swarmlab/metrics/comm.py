"""Communication metrics (docs/INTERFACE.md §14, DESIGN.md §10).

All three are per-round folds: counters reset on `round_started`, and the denominator is the
number of turns (`turn_started` events) in the current round.

- `comm.read_rate`: share of this round's turns that issued at least one `read` (a `read_board`
  call, even one returning nothing). None when there were no turns.
- `comm.posts_per_round`: number of `post` events committed this round, as a float.
- `comm.hops`: propagation depth, cumulative over the run. A post's hop count is
  1 + the maximum hop count of any post whose delivery its author had read up to and including
  the round of the post (1 if the author read nothing). Reads are resolved to posts through
  `delivery` events; under immediate commit a same-round read may reference a delivery logged
  later in the round, so hop counts of a round's posts are resolved lazily (in `value()` without
  mutating state, and folded into state at the next `round_started`). Value: maximum hop count
  of all posts so far, None before the first post.
"""
from __future__ import annotations

from typing import Any, ClassVar

from .base import Metric


class _PerRound(Metric):
    def __init__(self) -> None:
        self.turns = 0

    def update(self, event: Any) -> None:
        t = getattr(event, "type", None)
        if t == "round_started":
            self.turns = 0
            self._reset_round()
        elif t == "turn_started":
            self.turns += 1
        self._update(t, event)

    def _reset_round(self) -> None:
        pass

    def _update(self, t: str | None, event: Any) -> None:
        pass


class ReadRate(_PerRound):
    entry_point: ClassVar[str | None] = "comm.read_rate"
    name = "comm.read_rate"

    def __init__(self) -> None:
        super().__init__()
        self.readers: list[str] = []

    def _reset_round(self) -> None:
        self.readers = []

    def _update(self, t: str | None, event: Any) -> None:
        if t == "read" and event.agent not in self.readers:
            self.readers.append(event.agent)

    def value(self) -> tuple[float | None, int]:
        if self.turns == 0:
            return None, 0
        return len(self.readers) / self.turns, self.turns


class PostsPerRound(_PerRound):
    entry_point: ClassVar[str | None] = "comm.posts_per_round"
    name = "comm.posts_per_round"

    def __init__(self) -> None:
        super().__init__()
        self.posts = 0

    def _reset_round(self) -> None:
        self.posts = 0

    def _update(self, t: str | None, event: Any) -> None:
        if t == "post":
            self.posts += 1

    def value(self) -> tuple[float | None, int]:
        return float(self.posts), self.turns


class Hops(_PerRound):
    entry_point: ClassVar[str | None] = "comm.hops"
    name = "comm.hops"

    def __init__(self) -> None:
        super().__init__()
        self.delivery_post: dict[str, str] = {}
        self.post_hop: dict[str, int] = {}
        self.reads: dict[str, list[tuple[int, str]]] = {}  # agent -> [(round, delivery_id)]
        self.pending: list[tuple[str, str, int]] = []      # (post_id, author, round)

    def _reset_round(self) -> None:
        self.post_hop = self._resolved()
        self.pending = []

    def _update(self, t: str | None, event: Any) -> None:
        if t == "read":
            self.reads.setdefault(event.agent, []).extend((event.round, d) for d in event.delivery_ids)
        elif t == "delivery":
            self.delivery_post[event.delivery_id] = event.post_id
        elif t == "post":
            self.pending.append((event.post_id, event.agent, event.round))

    def _resolved(self) -> dict[str, int]:
        hops = dict(self.post_hop)
        for post_id, author, rnd in self.pending:
            best = 0
            for r, d in self.reads.get(author, []):
                if r > rnd:
                    continue
                src = self.delivery_post.get(d)
                if src is not None and src in hops:
                    best = max(best, hops[src])
            hops[post_id] = best + 1
        return hops

    def value(self) -> tuple[float | None, int]:
        hops = self._resolved()
        return (float(max(hops.values())) if hops else None), self.turns
