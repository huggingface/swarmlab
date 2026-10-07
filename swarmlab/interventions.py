"""Interventions: spec-declared perturbations of a running swarm (docs/INTERFACE-M3a.md §1).

An `Intervention` is a plugin with a trigger (chosen by constructor kwargs) and an operation
(`apply(ctx)`, which calls `ctx.ops.*`). The runner evaluates every intervention once per round,
after the board and world commit and before probes, in spec order (`fire_interventions`).

Triggers (exactly one of the constructor kwargs, or none when a subclass overrides `should_fire`):

- `at_round: int | list[int]`; `every: int` with `start: int = 1` (fires at start, start+every, ...);
- `when: {"metric", "op", "value", "once": True}`: compares `ctx.metrics[metric][0]` (a value of
  None never fires); `on_event: {"type", "where": {}, "once": True}`: some logical event of this
  round so far has that `type` and every `where` item equal to the event's field. `where` keys may
  be dotted paths into nested dicts (`"action.name": "guess"`). With `once` the trigger fires at
  most once per run (the flag is in the snapshot).

Decisions where the contract is silent:

- `ctx.metrics` is computed lazily on first access: a deep copy of each run metric folded over the
  round's logical events so far (so this round's world commit counts, this round's probes do not,
  because probes run after interventions). The run's metrics are never touched.
- `ctx.events` are the round's logical events so far as typed `Event`s, including `intervention`
  events written by earlier interventions of the same round.
- `name` defaults to the class's entry point (built-ins) or class name; names must be unique per
  run. `spec()` flattens a `**trigger` constructor argument into `params` and leaves out trigger
  kwargs at their defaults, so `{type: mute, params: {agents: [a000], rounds: 2, at_round: 3}}`
  round-trips.
- Events: one `intervention` event per affected agent for `delay`, `mute`, `kill`, `revive`,
  `patch_private`, `reconfigure`; one event with `affected=[]` for `inject_post`, `set_policies`,
  `set_topology`, `world`. Hook failures (`NotSupported`, or any exception raised by the world /
  participant hook or by the board for a bad argument) are logged with `ok=False` and `error`, and
  the run continues; an exception raised by `apply` itself outside `ops` propagates (the round is
  lost and the run is resumable from the last commit).
- `inject_post`: posts by an author that is a run agent go through the topology over the live
  agents; any other author (default `"system"`) reaches every live agent. `recipients` restricts
  the fan-out to those live agents (bypassing the topology). The post and its deliveries are
  committed in the current round and logged as ordinary `post` / `delivery` events (before the
  `intervention` event), so they are eligible next round under round_end commit.
- `delay(agents, rounds, *, duration=None)`: `duration` bounds the commit rounds affected
  (`round + duration`); None means until cleared, and `delay(agents, 0)` clears.
- `mute(agents, rounds)`: posts committed by these agents in this round (after the mute, i.e. only
  injected ones) through round + rounds are withheld.
- `kill` / `revive` change `runner.live_agents` immediately: the agents are excluded from this
  round's probes and from turns, deliveries and metric denominators from the next round (metrics
  are told the new live list at the next round start, which is what replay reconstructs from
  `round_started.order`). `revive` keeps the agent order of the run.
- `patch_private` / `world` re-send the truth to truth-needing metrics afterwards; replay re-applies
  them (from the logged `params`) to its private world copy, never calling intervention plugins.
- `reconfigure` relies on the participant keeping its settings as snapshotted attributes
  (`LLMAgent` does), so a reconfiguration survives resume.

`NotSupported` is the error the default world / participant hooks raise.
"""
from __future__ import annotations

import copy
import inspect
import operator
import pickle
import random
from collections.abc import Callable, Mapping, Sequence
from typing import TYPE_CHECKING, Any, ClassVar

from .base import Persistable, Plugin, _plain
from .ids import AgentId
from .rng import derive

if TYPE_CHECKING:
    from .events import Event
    from .runner import Runner


class NotSupported(Exception):
    """A world or participant does not implement an intervention hook."""


GROUP = "swarmlab.interventions"
TRIGGER_DEFAULTS: dict[str, Any] = {"name": None, "at_round": None, "every": None, "start": 1,
                                    "when": None, "on_event": None}
_OPS: dict[str, Callable[[float, float], bool]] = {
    ">": operator.gt, ">=": operator.ge, "<": operator.lt, "<=": operator.le,
    "==": operator.eq, "!=": operator.ne,
}


def _lookup(data: Any, path: str) -> Any:
    for part in path.split("."):
        if not isinstance(data, Mapping) or part not in data:
            return _MISSING
        data = data[part]
    return data


_MISSING = object()


class Intervention(Persistable, Plugin):
    name: str

    def __init__(self, *, name: str | None = None, at_round: int | list[int] | None = None,
                 every: int | None = None, start: int = 1, when: dict | None = None,
                 on_event: dict | None = None) -> None:
        self.name = name or type(self).__dict__.get("entry_point") or type(self).__name__
        given = [k for k, v in (("at_round", at_round), ("every", every), ("when", when),
                                ("on_event", on_event)) if v is not None]
        if len(given) > 1:
            raise ValueError(f"intervention {self.name!r}: give exactly one trigger, got {given}")
        if not given and type(self).should_fire is Intervention.should_fire:
            raise ValueError(f"intervention {self.name!r}: no trigger (at_round, every, when, on_event)")
        self._kind = given[0] if given else None
        if at_round is not None:
            self._rounds = {int(at_round)} if isinstance(at_round, int) else {int(r) for r in at_round}
        if every is not None:
            if every < 1:
                raise ValueError("every must be >= 1")
            self._every, self._start = int(every), int(start)
        if when is not None:
            unknown = set(when) - {"metric", "op", "value", "once"}
            if unknown or "metric" not in when or "value" not in when:
                raise ValueError(f"when needs metric, op, value (and optional once), got {when}")
            if when.get("op", ">=") not in _OPS:
                raise ValueError(f"when.op must be one of {sorted(_OPS)}, got {when.get('op')!r}")
            self._when = {"op": ">=", "once": True, **when}
        if on_event is not None:
            unknown = set(on_event) - {"type", "where", "once"}
            if unknown or "type" not in on_event:
                raise ValueError(f"on_event needs type (and optional where, once), got {on_event}")
            self._on_event = {"where": {}, "once": True, **on_event}
        # run state (the snapshot)
        self.fired = False
        self.fire_count = 0
        self.last_round: int | None = None
        self.state: dict = {}

    # ---- spec ---------------------------------------------------------------------------------
    def spec(self) -> dict[str, Any]:
        params = dict(getattr(self, "params", {}))
        for p in inspect.signature(type(self).__init__).parameters.values():
            if p.kind is inspect.Parameter.VAR_KEYWORD and isinstance(params.get(p.name), dict):
                params.update(params.pop(p.name))
        for k, default in TRIGGER_DEFAULTS.items():
            if k in params and params[k] == default:
                params.pop(k)
        return {"type": self.type_name(), "params": _plain(params)}

    # ---- trigger ------------------------------------------------------------------------------
    @property
    def once(self) -> bool:
        if self._kind == "when":
            return bool(self._when["once"])
        if self._kind == "on_event":
            return bool(self._on_event["once"])
        return False

    def should_fire(self, ctx: InterventionContext) -> bool:
        if self.once and self.fired:
            return False
        r = ctx.round
        if self._kind == "at_round":
            return r in self._rounds
        if self._kind == "every":
            return r >= self._start and (r - self._start) % self._every == 0
        if self._kind == "when":
            value = ctx.metrics.get(self._when["metric"], (None, 0))[0]
            return value is not None and _OPS[self._when["op"]](value, self._when["value"])
        if self._kind == "on_event":
            where = self._on_event["where"] or {}
            for ev in ctx.events:
                if ev.type != self._on_event["type"]:
                    continue
                data = ev.model_dump(mode="json")
                if all(_lookup(data, k) == v for k, v in where.items()):
                    return True
            return False
        raise NotImplementedError(f"{type(self).__name__} has no trigger; override should_fire")

    def apply(self, ctx: InterventionContext) -> None:
        raise NotImplementedError

    def mark_fired(self, round: int) -> None:
        self.fired = True
        self.fire_count += 1
        self.last_round = round

    # ---- persistence --------------------------------------------------------------------------
    def snapshot(self) -> bytes:
        return pickle.dumps({"fired": self.fired, "fire_count": self.fire_count,
                             "last_round": self.last_round, "state": self.state},
                            protocol=pickle.HIGHEST_PROTOCOL)

    def restore(self, blob: bytes) -> None:
        data = pickle.loads(blob)
        self.fired, self.fire_count = data["fired"], data["fire_count"]
        self.last_round, self.state = data["last_round"], data["state"]


class InterventionContext:
    """What an intervention sees in round `round` (committed state only)."""

    def __init__(self, *, round: int, live: list[AgentId], agents: list[AgentId], events: list[Event],
                 rng: random.Random, ops: Ops, metrics: Callable[[], dict] | dict) -> None:
        self.round, self.live, self.agents, self.events = round, live, agents, events
        self.rng, self.ops = rng, ops
        self._metrics = metrics

    @property
    def metrics(self) -> dict[str, tuple[float | None, int]]:
        if callable(self._metrics):
            self._metrics = self._metrics()
        return self._metrics


class Ops:
    """The only mutation surface; owned by the runner, one per (intervention, round)."""

    def __init__(self, runner: Runner, intervention: str, round: int) -> None:
        self._runner = runner
        self._name = intervention
        self._round = round

    # ---- logging ------------------------------------------------------------------------------
    def _log(self, op: str, affected: Sequence[str] | None, *, ok: bool = True, error: str | None = None,
             params: dict | None = None, result: dict | None = None, post_id: str | None = None) -> None:
        from .events import InterventionEvent

        p = _jsonable(params or {})
        res = _jsonable(result or {})
        targets = [None] if not affected else [str(a) for a in affected]
        for a in targets:
            self._runner._append(InterventionEvent, self._round, a, intervention=self._name, op=op,
                                 affected=[] if a is None else [a], ok=ok, error=error, params=p,
                                 result=res, post_id=post_id)

    def _agents(self, agents: Sequence[str] | str) -> list[AgentId]:
        return [AgentId(agents)] if isinstance(agents, str) else [AgentId(a) for a in agents]

    # ---- board --------------------------------------------------------------------------------
    def inject_post(self, text: str, *, author: str = "system", channel: str = "main",
                    fields: dict | None = None, recipients: Sequence[str] | None = None) -> str | None:
        from .events import DeliveryEvent, PostEvent

        run, r = self._runner, self._round
        board = run.board
        params = {"text": text, "author": author, "channel": channel, "fields": fields or {},
                  "recipients": None if recipients is None else [str(a) for a in recipients]}
        live = list(run.live_agents)
        if recipients is not None:
            wanted = {str(a) for a in recipients}
            rec: list[AgentId] | None = [a for a in live if a in wanted]
        else:
            rec = None if author in run.agents else live
        if board.buffered:
            raise RuntimeError("inject_post with a non-empty board buffer")
        try:
            board.buffer_post(AgentId(author), r, channel, text, fields)
        except ValueError as e:
            self._log("inject_post", [], ok=False, error=str(e), params=params)
            return None
        posts, deliveries = board.commit(r, live, derive(run.options.seed, "topology", r), run.blobs,
                                         recipients=rec)
        for p in posts:
            run._append(PostEvent, r, p.agent, post_id=p.post_id, provisional_id=p.post_id,
                        channel=p.channel, text=p.text, fields=dict(p.fields))
        for d in deliveries:
            run._append(DeliveryEvent, r, d.recipient, post_id=d.post_id, recipient=d.recipient,
                        delivery_id=d.delivery_id, eligible_round=d.eligible_round,
                        content_hash=d.content_hash)
        post_id = posts[0].post_id
        self._log("inject_post", [], params=params, post_id=post_id)
        return post_id

    def delay(self, agents: Sequence[str] | str, rounds: int, *, duration: int | None = None) -> None:
        targets = self._agents(agents)
        until = None if duration is None else self._round + duration
        self._runner.board.set_delay(targets, rounds, until)
        self._log("delay", targets, params={"rounds": rounds, "duration": duration, "until": until})

    def mute(self, agents: Sequence[str] | str, rounds: int) -> None:
        targets = self._agents(agents)
        self._runner.board.mute(targets, self._round + rounds)
        self._log("mute", targets, params={"rounds": rounds, "until": self._round + rounds})

    def set_policies(self, policies: Sequence[Any]) -> None:
        try:
            self._runner.board.set_policies(policies)
        except (ValueError, TypeError) as e:
            self._log("set_policies", [], ok=False, error=str(e), params={"policies": _plain(list(policies))})
            return
        self._log("set_policies", [], params={"policies": [p.spec() for p in self._runner.board.policies]})

    def set_topology(self, topology: Any) -> None:
        try:
            self._runner.board.set_topology(topology)
        except (ValueError, TypeError) as e:
            self._log("set_topology", [], ok=False, error=str(e), params={"topology": _plain(topology)})
            return
        self._log("set_topology", [], params={"topology": self._runner.board.topology.spec()})

    # ---- live set -----------------------------------------------------------------------------
    def kill(self, agents: Sequence[str] | str) -> None:
        run = self._runner
        targets = self._agents(agents)
        unknown = [a for a in targets if a not in run.agents]
        dead = set(targets)
        run.live_agents = [a for a in run.live_agents if a not in dead]
        self._log_each("kill", targets, unknown)

    def revive(self, agents: Sequence[str] | str) -> None:
        run = self._runner
        targets = self._agents(agents)
        unknown = [a for a in targets if a not in run.agents]
        back = set(run.live_agents) | set(targets)
        run.live_agents = [a for a in run.agents if a in back]
        self._log_each("revive", targets, unknown)

    def _log_each(self, op: str, targets: list[AgentId], unknown: list[AgentId]) -> None:
        for a in targets:
            if a in unknown:
                self._log(op, [a], ok=False, error=f"unknown agent {a!r}")
            else:
                self._log(op, [a])

    # ---- world and participants ---------------------------------------------------------------
    def patch_private(self, agent: str, data: dict) -> None:
        run = self._runner
        try:
            run.world.patch_private(AgentId(agent), dict(data))
        except Exception as e:  # noqa: BLE001 - hook failures are logged, the run goes on
            self._log("patch_private", [agent], ok=False, error=_err(e), params={"data": data})
            return
        run._set_truth()
        self._log("patch_private", [agent], params={"data": data})

    def world(self, name: str, /, **args: Any) -> dict | None:
        run = self._runner
        params = {"name": name, "args": args}
        try:
            result = run.world.intervene(name, **args)
        except Exception as e:  # noqa: BLE001
            self._log("world", [], ok=False, error=_err(e), params=params)
            return None
        run._set_truth()
        self._log("world", [], params=params, result=result if isinstance(result, dict) else {})
        return result

    def reconfigure(self, agent: str, **kw: Any) -> None:
        run = self._runner
        participant = run.participants.get(AgentId(agent))
        try:
            if participant is None:
                raise ValueError(f"unknown agent {agent!r}")
            participant.reconfigure(**kw)
        except Exception as e:  # noqa: BLE001
            self._log("reconfigure", [agent], ok=False, error=_err(e), params=kw)
            return
        self._log("reconfigure", [agent], params=kw)


def _err(e: BaseException) -> str:
    kind = "not supported" if isinstance(e, NotSupported) else type(e).__name__
    return f"{kind}: {e}"


def _jsonable(value: Any) -> Any:
    import json

    return json.loads(json.dumps(_plain(value), default=str))


# ---- runner hook ------------------------------------------------------------------------------

def _fold_metrics(runner: Runner) -> dict[str, tuple[float | None, int]]:
    from .events import parse_event
    from .runner import NOT_FED

    fed = [parse_event(ev.model_dump_json()) for ev in runner._round_events if ev.type not in NOT_FED]
    out: dict[str, tuple[float | None, int]] = {}
    for m in runner.metrics:
        c = copy.deepcopy(m)
        for ev in fed:
            c.update(ev)
        out[c.name] = c.value()
    return out


def fire_interventions(runner: Runner, round: int) -> None:
    """Evaluate every intervention once, in spec order, after the round's commit."""
    for iv in runner.interventions:
        ops = Ops(runner, iv.name, round)
        ctx = InterventionContext(
            round=round, live=list(runner.live_agents), agents=list(runner.agents),
            events=list(runner._round_events),
            rng=derive(runner.options.seed, "intervention", iv.name, round), ops=ops,
            metrics=lambda: _fold_metrics(runner),
        )
        if iv.should_fire(ctx):
            iv.apply(ctx)
            iv.mark_fired(round)


def replay_world_op(world: Any, event: Any) -> bool:
    """Re-apply a logged world-changing intervention to a private world copy (replay)."""
    if not event.ok or event.op not in ("patch_private", "world"):
        return False
    if event.op == "patch_private":
        world.patch_private(AgentId(event.agent), dict(event.params["data"]))
    else:
        world.intervene(event.params["name"], **event.params.get("args", {}))
    return True


# ---- construction -----------------------------------------------------------------------------

def build_intervention(spec: Intervention | str | Mapping[str, Any] | Any) -> Intervention:
    """An `Intervention` from an instance (deep-copied), a type name, or a `{type, params}` spec."""
    if isinstance(spec, Intervention):
        return copy.deepcopy(spec)
    from .registry import resolve

    if isinstance(spec, str):
        spec = {"type": spec, "params": {}}
    elif not isinstance(spec, Mapping):
        spec = {"type": spec.type, "params": dict(spec.params)}
    cls = resolve(spec["type"], GROUP)
    obj = cls(**dict(spec.get("params") or {}))
    if not isinstance(obj, Intervention):
        raise TypeError(f"{spec['type']!r} is not an Intervention")
    return obj


def check_names(interventions: Sequence[Intervention]) -> None:
    names = [i.name for i in interventions]
    if len(set(names)) != len(names):
        raise ValueError(f"intervention names must be unique (pass name=...), got {names}")


# ---- built-ins --------------------------------------------------------------------------------

def _pick(ctx: InterventionContext, agents: Sequence[str] | str | int) -> list[str]:
    """A list of agents, or `k` agents drawn from the live set with the intervention's rng."""
    if isinstance(agents, int):
        return sorted(ctx.rng.sample(list(ctx.live), min(agents, len(ctx.live))))
    return [agents] if isinstance(agents, str) else list(agents)


class InjectPost(Intervention):
    entry_point: ClassVar[str | None] = "inject_post"

    def __init__(self, text: str, author: str = "system", channel: str = "main",
                 fields: dict | None = None, recipients: list[str] | None = None, **trigger: Any) -> None:
        super().__init__(**trigger)
        self.text, self.author, self.channel = text, author, channel
        self.fields, self.recipients = dict(fields or {}), recipients

    def apply(self, ctx: InterventionContext) -> None:
        ctx.ops.inject_post(self.text, author=self.author, channel=self.channel, fields=self.fields,
                            recipients=self.recipients)


class DelayDelivery(Intervention):
    entry_point: ClassVar[str | None] = "delay_delivery"

    def __init__(self, agents: list[str] | str | int, rounds: int, duration: int | None = None,
                 **trigger: Any) -> None:
        super().__init__(**trigger)
        self.agents, self.rounds, self.duration = agents, rounds, duration

    def apply(self, ctx: InterventionContext) -> None:
        ctx.ops.delay(_pick(ctx, self.agents), self.rounds, duration=self.duration)


class Mute(Intervention):
    entry_point: ClassVar[str | None] = "mute"

    def __init__(self, agents: list[str] | str | int, rounds: int, **trigger: Any) -> None:
        super().__init__(**trigger)
        self.agents, self.rounds = agents, rounds

    def apply(self, ctx: InterventionContext) -> None:
        ctx.ops.mute(_pick(ctx, self.agents), self.rounds)


class KillAgents(Intervention):
    """`mode="kill"` (default) or `"revive"`; `agents` a list or a count of random live agents."""

    entry_point: ClassVar[str | None] = "kill_agents"

    def __init__(self, agents: list[str] | str | int, mode: str = "kill", **trigger: Any) -> None:
        super().__init__(**trigger)
        if mode not in ("kill", "revive"):
            raise ValueError(f"mode must be 'kill' or 'revive', got {mode!r}")
        if mode == "revive" and isinstance(agents, int):
            raise ValueError("revive needs explicit agents")
        self.agents, self.mode = agents, mode

    def apply(self, ctx: InterventionContext) -> None:
        targets = _pick(ctx, self.agents)
        (ctx.ops.kill if self.mode == "kill" else ctx.ops.revive)(targets)


class PatchPrivate(Intervention):
    entry_point: ClassVar[str | None] = "patch_private"

    def __init__(self, agent: str, data: dict, **trigger: Any) -> None:
        super().__init__(**trigger)
        self.agent, self.data = agent, dict(data)

    def apply(self, ctx: InterventionContext) -> None:
        ctx.ops.patch_private(self.agent, self.data)


class Reconfigure(Intervention):
    """`settings` are passed as `Participant.reconfigure(**settings)` to each of `agents`."""

    entry_point: ClassVar[str | None] = "reconfigure"

    def __init__(self, agents: list[str] | str, settings: dict, **trigger: Any) -> None:
        super().__init__(**trigger)
        self.agents, self.settings = agents, dict(settings)

    def apply(self, ctx: InterventionContext) -> None:
        for a in _pick(ctx, self.agents):
            ctx.ops.reconfigure(a, **self.settings)


BUILTINS: dict[str, type[Intervention]] = {
    c.entry_point: c for c in (InjectPost, DelayDelivery, Mute, KillAgents, PatchPrivate, Reconfigure)
    if c.entry_point
}
