"""ColoringGrid and claim metrics (DESIGN.md §10; M3b).

The grid metrics need the truth (`ColoringGrid.verify()`: initial target, target changes, sizes,
`paints_per_round`) and fold the log: `round_started` applies the round's target changes and
counts the round's live agents (`len(order)`), each accepted `paint` in `action_committed`
(result `painted` or `overwritten`) is applied in log order, which is the world's commit order,
with the same accounting as the world (swarmlab/world/coloring.py), so the final values equal
`score()`.

- `coloring.coverage`: matching cells / cells; denominator cells.
- `coloring.duplicate_paints`, `coloring.wrong_paints`: cumulative counts; denominator the
  applied paints so far.
- `coloring.parallel_efficiency`: (matching cells / agent-rounds) / `paints_per_round`, i.e. the
  correct cells per agent-round relative to the single-agent ideal of `paints_per_round` correct
  cells per round; denominator agent-rounds so far (live agents summed over rounds).

Crossed-zone metrics (the world's zones, groups and targets come from `verify()`):

- `coloring.relay_status`: the per-zone relay diagnostics the world cannot compute itself (it
  never sees the board). Main output: the number of zones whose painting group has read a correct
  report (denominator 3). Outputs `coloring.relay_status.Z0` / `.Z1` / `.Z2`: the zone's status
  as a label (value = its index in `RELAY_STATUSES`, denominator 1), first match wins:
  `complete` (every zone cell matches its target, however that came about),
  `viewer_never_posted` (no agent of the zone's viewing group has posted anything),
  `viewer_posted_wrong_or_unparsed` (they posted, but no post is a correct report),
  `painter_never_read_correct_report` (a correct report exists, but no agent of the painting
  group received it through `read_board`), `painter_read_but_painted_wrong` (the painting group
  read a correct report and the zone is still not complete: painted wrong, or not at all).
  A correct report is a post by a viewing-group agent in which the zone id (a token, any case) is
  followed by the zone's targets (as of the post's round) in row-major order: the first `n` colour
  tokens after the id (n = zone cells: 10), where a colour token is a full colour name or a single
  palette letter (any case), or a run of 3+ palette letters read letter by letter
  (`swarmlab.world.coloring.parse_report`); punctuation, other words and letters touching a
  digit (G1) are skipped. A read is a
  `read` event of a painting-group agent whose delivery ids include a delivery of such a post.
- `coloring.zone_paint_refusals`: cumulative paints refused with `zone_paint_refused`, at call
  time (a `tool_returned` whose error starts with it) or at commit (`action_committed` feedback);
  denominator the `paint` calls so far.
- `coloring.posts_by_agent` and `coloring.reads_by_agent`: cumulative `post` events and `read`
  events (one per `read_board` call) swarm-wide (main output, denominator the agents), and per
  agent as `coloring.posts_by_agent.a000`, ... (denominator 1). The agents are every agent the
  runner has passed to `set_agents` (dead agents keep their series).

`relay_summary(run)` recomputes the final relay statuses and per-agent counts of a finished run
from its events (the world's `report_sections` table).

Claim metrics fold `claim` and `registry` events (no truth needed):

- `claims.violations`: cumulative number of world actions on a key whose live owner was another
  agent (under `enforced` those actions were also rejected); denominator the keyed actions so far.
- `claims.held`: live claims after this round's commit (acquired, not released, not expired);
  denominator the round's live agents.
"""
from __future__ import annotations

from typing import Any, ClassVar

from ..world.coloring import RELAY_STATUSES, ZONE_IDS, parse_report
from .base import Metric


class _GridFold(Metric):
    def __init__(self) -> None:
        self.truth: dict | None = None
        self.target: list[list[str]] | None = None
        self.grid: list[list[str]] | None = None
        self.paints = 0
        self.duplicates = 0
        self.wrong = 0
        self.agent_rounds = 0

    def needs_truth(self) -> bool:
        return True

    def set_truth(self, truth: dict) -> None:
        self.truth = dict(truth)
        if self.target is None:
            self.target = [list(r) for r in truth.get("initial_target", truth.get("target", []))]
            self.grid = [["."] * len(r) for r in self.target]

    def update(self, event: Any) -> None:
        t = getattr(event, "type", None)
        if self.target is None or self.grid is None:
            return
        if t == "round_started":
            self.agent_rounds += len(event.order)
            for r, cells in (self.truth or {}).get("target_changes", []):
                if r == event.round:
                    for x, y, c in cells:
                        self.target[y][x] = c
        elif t == "action_committed" and event.accepted and event.action.get("name") == "paint":
            if event.feedback.get("result") not in ("painted", "overwritten"):
                return
            args = event.action.get("args", {})
            x, y, c = args["x"], args["y"], args["color"]
            want, before = self.target[y][x], self.grid[y][x]
            self.paints += 1
            if c != want:
                self.wrong += 1
            elif before == want:
                self.duplicates += 1
            self.grid[y][x] = c

    def _cells(self) -> int:
        return sum(len(r) for r in self.target or [])

    def _matching(self) -> int:
        if not self.target or not self.grid:
            return 0
        return sum(1 for tr, gr in zip(self.target, self.grid, strict=True)
                   for a, b in zip(tr, gr, strict=True) if a == b)


class Coverage(_GridFold):
    entry_point: ClassVar[str | None] = "coloring.coverage"
    name = "coloring.coverage"
    description = "share of grid cells matching the target"

    def value(self) -> tuple[float | None, int]:
        cells = self._cells()
        return (self._matching() / cells if cells else None), cells


class DuplicatePaints(_GridFold):
    entry_point: ClassVar[str | None] = "coloring.duplicate_paints"
    name = "coloring.duplicate_paints"
    description = "cumulative paints of a cell that already had the right colour"

    def value(self) -> tuple[float | None, int]:
        return float(self.duplicates), self.paints


class WrongPaints(_GridFold):
    entry_point: ClassVar[str | None] = "coloring.wrong_paints"
    name = "coloring.wrong_paints"
    description = "cumulative paints with a colour the target does not have there"

    def value(self) -> tuple[float | None, int]:
        return float(self.wrong), self.paints


class ParallelEfficiency(_GridFold):
    entry_point: ClassVar[str | None] = "coloring.parallel_efficiency"
    name = "coloring.parallel_efficiency"
    description = "correct cells per agent-round relative to the single-agent ideal"

    def value(self) -> tuple[float | None, int]:
        if not self.agent_rounds:
            return None, 0
        ppr = int((self.truth or {}).get("paints_per_round", 1)) or 1
        return self._matching() / self.agent_rounds / ppr, self.agent_rounds


class RelayStatus(_GridFold):
    entry_point: ClassVar[str | None] = "coloring.relay_status"
    name = "coloring.relay_status"
    description = "zones whose painters read a correct target report; per-zone relay status labels"

    def __init__(self) -> None:
        super().__init__()
        self.zones: list[dict] = []
        self.groups: dict[str, int] = {}
        self.posted: set[str] = set()            # zone ids whose viewing group posted anything
        self.correct_posts: dict[str, list[str]] = {}  # zone id -> post ids of correct reports
        self.delivery_post: dict[str, str] = {}
        self.reads: list[tuple[str, str]] = []    # (agent, delivery id)

    def set_truth(self, truth: dict) -> None:
        super().set_truth(truth)
        self.zones = [dict(z) for z in truth.get("crossed_zones", [])]
        self.groups = {str(a): int(g) for a, g in truth.get("groups", {}).items()}

    def output_names(self) -> list[str]:
        return [self.name] + [f"{self.name}.{z}" for z in ZONE_IDS]

    def update(self, event: Any) -> None:
        super().update(event)
        t = getattr(event, "type", None)
        if t == "post":
            g = self.groups.get(str(event.agent))
            for z in self.zones:
                if g != z["viewer_group"]:
                    continue
                self.posted.add(z["zone_id"])
                want = self._zone_target(z)
                if want and want in parse_report(event.text, z["zone_id"], len(want)):
                    self.correct_posts.setdefault(z["zone_id"], []).append(event.post_id)
        elif t == "delivery":
            self.delivery_post[event.delivery_id] = event.post_id
        elif t == "read":
            self.reads.extend((str(event.agent), d) for d in event.delivery_ids)

    def _zone_cells(self, z: dict) -> list[tuple[int, int]]:
        return [(x, y) for y in range(z["y"], z["y"] + z["height"])
                for x in range(z["x"], z["x"] + z["width"])]

    def _zone_target(self, z: dict) -> list[str]:
        if self.target is None:
            return []
        return [self.target[y][x] for x, y in self._zone_cells(z)]

    def statuses(self) -> dict[str, str]:
        """Zone id -> relay status (see the module docstring)."""
        out: dict[str, str] = {}
        for z in self.zones:
            zid = z["zone_id"]
            cells = self._zone_cells(z)
            if self.target is not None and self.grid is not None and all(
                    self.grid[y][x] == self.target[y][x] for x, y in cells):
                out[zid] = "complete"
            elif zid not in self.posted:
                out[zid] = "viewer_never_posted"
            elif not self.correct_posts.get(zid):
                out[zid] = "viewer_posted_wrong_or_unparsed"
            elif not self._read_correct(z):
                out[zid] = "painter_never_read_correct_report"
            else:
                out[zid] = "painter_read_but_painted_wrong"
        return out

    def _read_correct(self, z: dict) -> bool:
        good = set(self.correct_posts.get(z["zone_id"], []))
        return any(self.groups.get(a) == z["painter_group"] and self.delivery_post.get(d) in good
                   for a, d in self.reads)

    def value(self) -> tuple[float | None, int]:
        if not self.zones:
            return None, 0
        return float(sum(1 for z in self.zones if self._read_correct(z))), len(self.zones)

    def outputs(self) -> list[tuple[str, float | None, int, str | None]]:
        value, denom = self.value()
        st = self.statuses()
        out: list[tuple[str, float | None, int, str | None]] = [(self.name, value, denom, None)]
        for zid in ZONE_IDS:
            label = st.get(zid)
            out.append((f"{self.name}.{zid}",
                        None if label is None else float(RELAY_STATUSES.index(label)),
                        0 if label is None else 1, label))
        return out


class ZonePaintRefusals(Metric):
    entry_point: ClassVar[str | None] = "coloring.zone_paint_refusals"
    name = "coloring.zone_paint_refusals"
    description = "cumulative paints refused because the painter's group may not paint that zone"

    def __init__(self) -> None:
        self.refusals = 0
        self.calls = 0
        self.paint_calls: set[str] = set()

    def update(self, event: Any) -> None:
        t = getattr(event, "type", None)
        if t == "tool_called" and event.tool == "paint":
            self.calls += 1
            self.paint_calls.add(event.call_id)
        elif t == "tool_returned" and event.call_id in self.paint_calls:
            self.paint_calls.discard(event.call_id)
            err = (event.result or {}).get("error") or ""
            if not (event.result or {}).get("ok", True) and err.startswith("zone_paint_refused"):
                self.refusals += 1
        elif (t == "action_committed" and not event.accepted
              and event.feedback.get("error") == "zone_paint_refused"):
            self.refusals += 1

    def value(self) -> tuple[float | None, int]:
        return float(self.refusals), self.calls


class _PerAgentCount(Metric):
    event_type: ClassVar[str] = ""

    def __init__(self) -> None:
        self.agents: list[str] = []
        self.counts: dict[str, int] = {}

    def set_agents(self, agents: list[Any]) -> None:
        for a in agents:
            if str(a) not in self.agents:
                self.agents.append(str(a))
        self.agents.sort()

    def update(self, event: Any) -> None:
        if getattr(event, "type", None) == self.event_type and event.agent is not None:
            a = str(event.agent)
            self.counts[a] = self.counts.get(a, 0) + 1

    def output_names(self) -> list[str]:
        return [self.name] + [f"{self.name}.{a}" for a in self.agents]

    def value(self) -> tuple[float | None, int]:
        return float(sum(self.counts.values())), len(self.agents)

    def outputs(self) -> list[tuple[str, float | None, int, str | None]]:
        total, n = self.value()
        out: list[tuple[str, float | None, int, str | None]] = [(self.name, total, n, None)]
        return out + [(f"{self.name}.{a}", float(self.counts.get(a, 0)), 1, None) for a in self.agents]


class PostsByAgent(_PerAgentCount):
    entry_point: ClassVar[str | None] = "coloring.posts_by_agent"
    name = "coloring.posts_by_agent"
    description = "cumulative board posts, swarm-wide and per agent"
    event_type = "post"


class ReadsByAgent(_PerAgentCount):
    entry_point: ClassVar[str | None] = "coloring.reads_by_agent"
    name = "coloring.reads_by_agent"
    description = "cumulative read_board calls, swarm-wide and per agent"
    event_type = "read"


def relay_summary(run: Any) -> dict:
    """Final relay statuses and per-agent posts / read_board calls of a finished `Run`, folded
    from its events with the truth of its world rebuilt as the runner builds it:
    `{"board", "status": {zone id: status}, "zones_completed", "posts": {agent: n},
    "reads": {agent: n}}`."""
    import copy

    from ..ids import agent_id
    from ..rng import derive

    exp = run.experiment
    agents = [agent_id(i) for i in range(len(exp.participants))]
    world = copy.deepcopy(exp.world)
    world.reset(derive(run.spec.options.seed, "world"), list(agents))
    truth = world.verify()
    relay, posts, reads = RelayStatus(), PostsByAgent(), ReadsByAgent()
    relay.set_truth(truth)
    folds: list[Metric] = [relay, posts, reads]
    for m in folds:
        m.set_agents(list(agents))
    for ev in run.events_all:
        if ev.type in ("round_started", "action_committed", "post", "delivery", "read"):
            for m in folds:
                m.update(ev)
    status = relay.statuses()
    return {"board": truth.get("board"), "status": status,
            "zones_completed": sum(1 for v in status.values() if v == "complete"),
            "posts": {a: posts.counts.get(a, 0) for a in posts.agents},
            "reads": {a: reads.counts.get(a, 0) for a in reads.agents}}


class ClaimViolations(Metric):
    entry_point: ClassVar[str | None] = "claims.violations"
    name = "claims.violations"
    description = "cumulative world actions on a key another live agent had claimed"

    def __init__(self) -> None:
        self.violations = 0
        self.keyed = 0

    def update(self, event: Any) -> None:
        if getattr(event, "type", None) == "claim":
            self.keyed += 1
            self.violations += int(event.violation)

    def value(self) -> tuple[float | None, int]:
        return float(self.violations), self.keyed


class ClaimsHeld(Metric):
    entry_point: ClassVar[str | None] = "claims.held"
    name = "claims.held"
    description = "live registry claims after the round's commit"

    def __init__(self) -> None:
        self.claims: dict[str, int] = {}  # key -> expires_round of its live claim
        self.round = 0
        self.live = 0

    def update(self, event: Any) -> None:
        t = getattr(event, "type", None)
        if t == "round_started":
            self.round = event.round
            self.live = len(event.order)
        elif t == "registry" and event.ok and event.op in ("acquire", "release"):
            if event.owner is not None and event.expires_round is not None:
                self.claims[event.key] = event.expires_round
            else:
                self.claims.pop(event.key, None)

    def value(self) -> tuple[float | None, int]:
        return float(sum(1 for exp in self.claims.values() if exp >= self.round)), self.live
