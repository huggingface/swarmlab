"""Communication metrics (docs/INTERFACE.md §14, DESIGN.md §10).

`read_rate` and `posts_per_round` are per-round folds: counters reset on `round_started`.

- `comm.read_rate`: share of this round's turns that issued at least one `read` (a `read_board`
  call, even one returning nothing). None when there were no turns.
- `comm.posts_per_round`: number of `post` events committed this round, as a float.
- `comm.hops`: propagation depth, cumulative over the run (review A6). A post's hop count is
  1 + the maximum hop count of the posts its author had read *before making it* (1 if none):
  a read counts only if it happened earlier than the post call, i.e. in an earlier round or
  earlier in the same turn. The cut is taken at the `post` tool call (the number of the author's
  `read` events seen so far) and linked to the committed `post` event through its
  `provisional_id`, which equals the id the call's ack returned (logs without it fall back to
  reads in earlier rounds). A read also counts only for deliveries with
  `eligible_round <= read round`. Reads are resolved to posts through `delivery` events; under
  immediate commit a same-round read may reference a delivery logged later in the round, so hop
  counts of a round's posts are resolved lazily (in `value()` without mutating state, and folded
  into state at the next `round_started`). Value: the running maximum hop count of all posts so
  far (None before the first post). Denominator: the number of posts read at least once so far
  (not turns: the value is cumulative, so a per-round turn count would be meaningless).

The other two metrics' denominator is the number of turns (`turn_started` events) in the round.
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
        self.delivery_eligible: dict[str, int] = {}
        self.post_hop: dict[str, int] = {}
        self.reads: dict[str, list[tuple[int, str]]] = {}  # agent -> [(round, delivery_id)]
        self.call_cut: dict[str, int] = {}                  # post call_id -> author's reads so far
        self.ack_cut: dict[tuple[int, str], int] = {}       # (round, ack id) -> cut
        self.pending: list[tuple[str, str, int, int | None]] = []  # (post_id, author, round, cut)

    def _reset_round(self) -> None:
        self.post_hop = self._resolved()
        self.pending = []
        self.call_cut = {}
        self.ack_cut = {}

    def _update(self, t: str | None, event: Any) -> None:
        if t == "read":
            self.reads.setdefault(event.agent, []).extend((event.round, d) for d in event.delivery_ids)
        elif t == "tool_called" and event.tool == "post":
            self.call_cut[event.call_id] = len(self.reads.get(event.agent, []))
        elif t == "tool_returned" and event.call_id in self.call_cut:
            res = event.result or {}
            ack = (res.get("result") or {}).get("id") if res.get("ok") else None
            if isinstance(ack, str):
                self.ack_cut[(event.round, ack)] = self.call_cut[event.call_id]
        elif t == "delivery":
            self.delivery_post[event.delivery_id] = event.post_id
            self.delivery_eligible[event.delivery_id] = event.eligible_round
        elif t == "post":
            cut = self.ack_cut.get((event.round, getattr(event, "provisional_id", None) or ""))
            self.pending.append((event.post_id, event.agent, event.round, cut))

    def _resolved(self) -> dict[str, int]:
        hops = dict(self.post_hop)
        for post_id, author, rnd, cut in self.pending:
            reads = self.reads.get(author, [])
            before = reads[:cut] if cut is not None else [x for x in reads if x[0] < rnd]
            best = 0
            for r, d in before:
                src = self.delivery_post.get(d)
                if src is None or src not in hops or self.delivery_eligible.get(d, r + 1) > r:
                    continue
                best = max(best, hops[src])
            hops[post_id] = best + 1
        return hops

    def value(self) -> tuple[float | None, int]:
        hops = self._resolved()
        read_posts = {
            self.delivery_post[d] for rs in self.reads.values() for _, d in rs if d in self.delivery_post
        }
        return (float(max(hops.values())) if hops else None), len(read_posts)
