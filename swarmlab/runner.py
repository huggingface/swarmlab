"""The runner: live, replay, resume, fork (docs/INTERFACE.md §12).

`Runner(run_dir, experiment, options, *, parent=None, run_id=None)` owns one run directory:

```
run.json  events.jsonl  discarded.jsonl  blobs/  snapshots/<round:06d>.json  artifacts/{spec.yaml,git.txt}
```

Per-run state is built from deep copies of the experiment's world, medium (board), metrics and
one deep copy of each participant prototype per agent (agent i is `agent_id(i)`), so an
`Experiment` can be run many times. Participants are bound with `derive(seed, "agent", agent)`,
the world is reset with `derive(seed, "world")`, metrics that `needs_truth()` get
`set_truth(world.verify())`. The scheduler is always `SeededShuffle` in M1a.

Round `r` (phase-commit, `commit == "round_end"`):

1. `order = scheduler.order(r, live, derive(seed, "schedule", r))`; log `round_started`.
2. Turns run concurrently (asyncio, `Semaphore(options.concurrency)`); each turn builds its
   `View` (observation with `private` emptied, the private dict going to `turn_started.private`
   instead; last round's outcomes, pushed items when `board.delivery == "push"`,
   tool schemas) and awaits `participant.turn(view, AgentTools(executor, agent))`. A normal return -> `end_turn` if
   the agent called `end_turn()` during the turn, else `no_tool`; `TurnCapReached` -> `cap`; any
   other exception -> `error` (traceback in `turn_ended.error`). An exception wins over an earlier
   `end_turn()`. A returned `TurnUsage` goes to `turn_ended.usage` (also after `end_turn`).
3. After all turns, each agent's buffered events are appended in `order`.
4. Buffered posts go to `board.buffer_post` in `order` (each agent's in call order), then
   `board.commit(r, live, derive(seed, "topology", r), blobs)`; log `post*`, `delivery*`.
5. `world.commit(actions in order)`; log `action_committed*`; outcomes become next round's
   `View.outcomes` as `{"action_id", "tool", "accepted", "feedback"}` dicts.
5b. (M1b, WP7) Probes: for every probe due this round (`r % every == 0`) each live agent is
   probed after the commit and before the metrics, so `probe` events are part of the round,
   fed to metrics (probe-sourced belief metrics see round r's answers in round r), covered by
   the round's `budget` event, and dropped with the round on a crash (resume re-asks them; the
   cache answers the ones already paid). See swarmlab/probes.py for the request, skip and budget
   rules. A `HardCeilingReached` from a probe does not abort the round: the round commits and
   the run then ends with `run_ended(hard_ceiling)` at round r.
6. Metrics are folded over this round's logical events (every event from `round_started`
   through the last `action_committed`, parsed back from JSON so live and replay feed identical
   objects); log `metric*`.
7. If `r % snapshot_every == 0` the snapshot manifest is written *before* `round_committed`
   (its `log_seq` is the seq that `round_committed` will get), so a committed round always has
   its snapshot on disk. Then `round_committed`, `run.json`, and the `snapshot` event
   (`manifest_path` is relative to the run dir, `snapshots/<round:06d>.json`).

Immediate mode (`commit == "immediate"`) runs turns one at a time in `order`; each `post` and
world action commits at once inside the executor. The resulting `post*`, `delivery*`,
`action_committed*` events are logged after the turn events in application order, so the log has
the same shape in both modes.

Termination: after each commit, `world.terminal()` -> `run_ended(terminal)`, else
`r == max_rounds` -> `run_ended(max_rounds)`, else (M1b) `soft_usd > 0` and
`ledger.spent["swarm"] >= soft_usd` -> `run_ended(soft_budget)`, all checked at the round boundary.

Budget and inference (M1b, docs/INTERFACE-M1b.md §2-§3):

- Per run the runner owns a `Ledger`, a `Gate` (providers = `experiment.resolved_providers()`,
  shared with the experiment, never copied) and an `Inference` service (blob store, the
  `blobs/cache/` record/replay cache, operational logging via `log_operational`). Each round's
  executor gets the service, so `AgentTools.infer` works; `self.executor` is the current round's
  executor (WP7's probe hook calls `executor.infer(agent, request, "measurement")` after commit).
- The effective budget is `run.json["budget"]` when present (set by `resume(budget=...)`), else the
  spec's. `run.json` also holds `ledger` (`{swarm, measurement, reserved, calls}`) and `ledger_seq`
  (the log seq the ledger accounts up to). The ledger is in every snapshot as `plugins["ledger"]`.
- A `budget` event (logical) is appended after the round's `metric*` events and before
  `round_committed`; metrics are not fed `budget`/`budget_changed`.
- `HardCeilingReached` from any turn cancels the other turns (an `asyncio.TaskGroup`), discards
  the round's buffered events and commits, appends `run_ended(hard_ceiling)` at round r - 1 and
  writes `run.json` (spend kept, score of the last commit). A participant that swallows the
  exception does not prevent this (the executor flags it).
- `turn_ended.usage` is the participant's `TurnUsage` updated with the executor's nominal
  inference totals (`prompt_tokens`, `completion_tokens`, `cached_prompt_tokens`,
  `reasoning_tokens`, `cost_usd`, `inference_calls`) when the agent made swarm inference calls.
- `replay()` builds a gate but never calls it; it raises `ReplayMismatch` if the gate's
  `provider_calls` is non-zero and reports it as `provider_calls`.
- `fork()` also copies the parent's cache entries for rounds <= the fork round and the request and
  response blobs referenced by the copied operational events; the child restores the parent's
  ledger from the fork snapshot.

`run.json` (rewritten atomically after every commit): run_id, experiment, arm, spec, spec_hash,
git_commit, dirty, parent_run, fork_round, restored (what a fork restored), status
("running" | "ended"), end_reason, last_round (last committed round), score (`world.score()`
after the last commit).

Recovery (`resume(budget=None)`): if the log has `run_ended` with a reason other than
`soft_budget`/`hard_ceiling`, nothing to do (budget-ended runs are resumed like crashed ones). Else find the last
`round_committed` (round c); keep the log through it plus a directly following `snapshot` /
`run_started` event; move every dropped event (logical and operational) to `discarded.jsonl`;
restore all plugins, `outcomes_prev` and `live` from snapshot c; re-append the `snapshot` event if
the crash lost it; continue live from c + 1. M1a requires a snapshot at c (always true with
`snapshot_every == 1`), else `RecoveryError`. With no committed round the log is cut back to
`run_started` and the run restarts from round 1.

Replay (`replay()`) never calls participants or the world's mutating paths except `reset` on a
private copy (to obtain the truth for metrics) and `restore` (to recompute the score). It folds
the metrics over the log's logical events and checks every logged `metric` event, then restores
the world from the latest snapshot at or before the last committed round and checks `score()`
against `run.json`. Any difference raises `ReplayMismatch`.

Fork (`fork(at_round, experiment=None, out=None)`): the child dir is
`<out or parent's parent dir>/<parent_id>__f<at_round>_<n>` (first free n from 1). The parent's
`events.jsonl` is copied byte-identically up to and including round `at_round`'s
`round_committed` line and the `snapshot` line right after it (so copied events still carry the
parent's run id in `run`; this is intended), the manifest `snapshots/<at_round>.json` is copied
byte-identically with its plugin blobs and the content blobs of every copied delivery. Then the
child appends its own `run_started` (round = at_round, `parent_run`, `fork_round`), restores and
runs live from `at_round + 1` under `experiment` (default: the parent's). Restore rules under an
edited experiment: the board restores itself (WP3: inboxes always, nested plugin state when the
nested spec matches); the world is restored when its `type` equals the parent's (WP2's FlagGame
keeps its own constructor config across restore, so e.g. a changed `guess_limit` applies to the
continuation; a world whose snapshot carries config would revert it); a participant is restored
only when its spec equals the parent's spec for that agent, else it starts fresh (bound, no
memory); a metric is restored when its spec is in the parent's metric list, else it starts fresh
at the fork round. The number of participants must not change.

Registry and claims (M3b, swarmlab/medium/registry.py): when the board's `registry` is on, the
runner owns a `Registry` (snapshot key `plugins["registry"]`, restored whenever present) and the
board's claim policy, and hands both to each round's executor. Each round starts with
`world.begin_round(r)` and then removes `world.killed_at(r)` from the live set (logged as
`intervention` events, `intervention="worker_failures"`, `op="kill"`) before the order is drawn.
Commit order under phase-commit: posts, then registry writes (in `order`, each agent's in call
order; `registry*` events), then world actions through the claim check (`claim*` events, then
`action_committed*`). Registry outcomes join `View.outcomes` after the world-action outcomes.

Fork at round 0 (M3a §2): no round-0 snapshot is ever written (the first is after round 1), so
`fork(0, experiment)` copies nothing from the parent: the child re-resets the world with the
parent's seed under the (possibly edited) experiment, binds fresh participants, and runs from
round 1 with the same schedule/topology streams. Its log starts with its own
`run_started(round=0, parent_run, fork_round=0)`; `run.json["restored"]` is
`{"world": False, "participants": [], "metrics": [], "reset": True}`. With the same experiment it
reproduces the parent's logical view (apart from `run` and `run_started`).

Paired-run repeats (M3a §2): `RunOptions.repeat` (default 0, omitted from the spec when 0) selects
the agent streams. Repeat 0 is a plain run. Repeat i > 0 binds agent `a` with
`derive(seed, "repeat", i, "agent", a)` and gives every inference request that has no `seed` one
derived from `derive(seed, "repeat", i, "agent", a, call_id)` (`RepeatSeeded`), so the sampling of
seeded providers, the fake provider and the request cache differ per repeat. World, schedule and
topology streams never depend on the repeat.
"""
from __future__ import annotations

import asyncio
import copy
import json
import random
import shutil
import traceback
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING, Any

from ._io import atomic_write_bytes
from .blobs import BlobStore
from .budget import Gate, HardCeilingReached, Ledger, MeasurementBudgetReached
from .events import (
    OPERATIONAL_TYPES,
    ActionCommittedEvent,
    BudgetChangedEvent,
    BudgetEvent,
    ClaimEvent,
    DeliveryEvent,
    Event,
    EventLog,
    MetricEvent,
    OverflowEvent,
    PostEvent,
    ProbeEvent,
    RegistryEvent,
    RoundCommittedEvent,
    RoundStartedEvent,
    RunEndedEvent,
    RunStartedEvent,
    SnapshotEvent,
    parse_event,
    write_jsonl,
)
from .executor import RoundExecutor
from .ids import AgentId, agent_id, fork_run_id
from .ids import run_id as make_run_id
from .inference import Inference, InferenceCache
from .interventions import (
    Intervention,
    Ops,
    build_intervention,
    check_names,
    fire_interventions,
    replay_world_op,
)
from .medium.registry import Registry, commit_world
from .metrics.base import Metric
from .metrics.base import get as get_metric
from .probes import CODER_SYSTEM, Probe, build_probe, probe_messages
from .providers.base import ChatMessage, ChatRequest, ProviderError
from .rng import derive
from .roles import agent_roles, bind_roles
from .scheduler import SeededShuffle
from .snapshot import SnapshotManifest, SnapshotStore
from .spec import (
    Budget,
    PluginSpec,
    RunOptions,
    RunSpec,
    dump_runspec_yaml,
    git_identity,
    spec_hash,
)
from .tools import AgentTools, TurnCapReached
from .view import View

if TYPE_CHECKING:
    from .experiment import Experiment

# events metrics are fed (every logical event of a round up to the commit bookkeeping)
NOT_FED = frozenset({"run_started", "metric", "round_committed", "snapshot", "run_ended", "budget",
                     "budget_changed"})
BUDGET_REASONS = ("soft_budget", "hard_ceiling")  # run_ended reasons that resume() continues from
REPO_DIR = Path(__file__).resolve().parent.parent


class ReplayMismatch(RuntimeError):
    """Replaying the log does not reproduce a logged metric or the recorded score."""


class RecoveryError(RuntimeError):
    """The run directory cannot be resumed."""


@dataclass
class ForkOrigin:
    parent_run: str
    at_round: int
    parent_spec: RunSpec


def _jsonable(value: Any) -> Any:
    return json.loads(json.dumps(value, default=str))


def _run_coro(coro: Any) -> Any:
    """`asyncio.run(coro)`, or, inside an already-running event loop (a notebook), the same in a
    worker thread; the caller blocks until the run finishes either way."""
    try:
        asyncio.get_running_loop()
    except RuntimeError:
        return asyncio.run(coro)
    with ThreadPoolExecutor(max_workers=1, thread_name_prefix="swarmlab-runner") as pool:
        return pool.submit(asyncio.run, coro).result()


def build_metric(m: str | Metric) -> Metric:
    return get_metric(m) if isinstance(m, str) else copy.deepcopy(m)


def metric_spec(m: Metric) -> PluginSpec:
    return PluginSpec(**m.spec())


def agent_stream(seed: int, agent: str, repeat: int = 0) -> random.Random:
    """Agent `agent`'s stream: `derive(seed, "agent", agent)`, and for paired-run repeat i > 0
    `derive(seed, "repeat", i, "agent", agent)` (M3a §2); repeat 0 is exactly a plain run."""
    if repeat == 0:
        return derive(seed, "agent", agent)
    return derive(seed, "repeat", repeat, "agent", agent)


class RepeatSeeded:
    """Inference wrapper for repeat i > 0: a request without a `seed` gets one drawn from
    `derive(seed, "repeat", i, "agent", agent, call_id)`. Sampling is thereby re-derived per
    repeat (seeded providers and the fake provider answer differently; the request hash and so
    the cache key differ) while staying a pure function of the run, so resume hits the cache."""

    def __init__(self, inner: Inference, seed: int, repeat: int) -> None:
        self.inner = inner
        self.seed = seed
        self.repeat = repeat

    async def infer(self, *, agent: str | None, round: int, call_id: str, request: ChatRequest,
                    category: str = "swarm") -> Any:
        if request.seed is None:
            s = derive(self.seed, "repeat", self.repeat, "agent", agent or "", call_id).getrandbits(31)
            request = request.model_copy(update={"seed": s})
        return await self.inner.infer(agent=agent, round=round, call_id=call_id, request=request,
                                      category=category)

    def __getattr__(self, name: str) -> Any:
        return getattr(self.inner, name)


class Runner:
    def __init__(
        self,
        run_dir: Path | str,
        experiment: Experiment,
        options: RunOptions,
        *,
        parent: ForkOrigin | None = None,
        run_id: str | None = None,
    ) -> None:
        self.dir = Path(run_dir)
        self.experiment = experiment
        self.options = options
        self.parent = parent
        self.spec: RunSpec = experiment.to_spec(**options.model_dump())
        meta = self._read_meta()
        if meta is not None:
            self.run_id = meta["run_id"]
        else:
            self.run_id = run_id or make_run_id(experiment.name, experiment.arm, options.seed)
        self.parent_run = parent.parent_run if parent else (meta or {}).get("parent_run")
        self.fork_round = parent.at_round if parent else (meta or {}).get("fork_round")
        self.restored: dict[str, Any] = (meta or {}).get("restored") or {}
        if meta is not None:
            self.budget = Budget.model_validate(meta.get("budget") or meta["spec"]["budget"])
        else:
            self.budget = experiment.budget
        self._aborted_score: Any = None
        self.status = "running"
        self.end_reason: str | None = None
        self.last_round = 0
        self._round_events: list[Event] = []

    # ---- paths and metadata ------------------------------------------------------------------
    @property
    def log_path(self) -> Path:
        return self.dir / "events.jsonl"

    def _read_meta(self) -> dict | None:
        p = self.dir / "run.json"
        return json.loads(p.read_text()) if p.exists() else None

    @staticmethod
    def _read_meta_at(run_dir: Path | str) -> dict:
        p = Path(run_dir) / "run.json"
        if not p.exists():
            raise FileNotFoundError(f"{run_dir} is not a run directory (no run.json)")
        return json.loads(p.read_text())

    def _write_meta(self) -> None:
        git_commit, dirty = self._git
        data = {
            "run_id": self.run_id,
            "experiment": self.spec.experiment,
            "arm": self.spec.arm,
            "spec": self.spec.model_dump(mode="json"),
            "spec_hash": spec_hash(self.spec),
            "git_commit": git_commit,
            "dirty": dirty,
            "parent_run": self.parent_run,
            "fork_round": self.fork_round,
            "restored": self.restored,
            "status": self.status,
            "end_reason": self.end_reason,
            "last_round": self.last_round,
            "score": self._aborted_score if self._aborted_score is not None
            else _jsonable(self.world.score()),
            "budget": self.budget.model_dump(mode="json"),
            "ledger": self.ledger.to_dict(),
            "ledger_seq": self.log.next_seq - 1 if getattr(self, "log", None) is not None else -1,
        }
        text = json.dumps(data, indent=2, sort_keys=True) + "\n"
        atomic_write_bytes(self.dir / "run.json", text.encode())

    def _write_artifacts(self) -> None:
        art = self.dir / "artifacts"
        art.mkdir(parents=True, exist_ok=True)
        dump_runspec_yaml(self.spec, art / "spec.yaml")
        git_commit, dirty = self._git
        atomic_write_bytes(art / "git.txt", f"commit: {git_commit}\ndirty: {str(dirty).lower()}\n".encode())

    # ---- state -------------------------------------------------------------------------------
    def _build(self) -> None:
        exp = self.experiment
        self.agents: list[AgentId] = [agent_id(i) for i in range(len(exp.participants))]
        self.world = copy.deepcopy(exp.world)
        self.board = copy.deepcopy(exp.medium)
        self.board.commit_mode = self.options.commit
        self.scheduler = SeededShuffle()
        self.metrics: list[Metric] = [build_metric(m) for m in exp.metrics]
        names = [m.name for m in self.metrics]
        if len(set(names)) != len(names):
            raise ValueError(f"metric names must be unique, got {names}")
        self.participants = {a: copy.deepcopy(p) for a, p in zip(self.agents, exp.participants, strict=True)}
        self.probes: list[Probe] = [build_probe(p) for p in exp.probes]
        pnames = [p.name for p in self.probes]
        if len(set(pnames)) != len(pnames):
            raise ValueError(f"probe names must be unique, got {pnames}")
        self.interventions: list[Intervention] = [build_intervention(i) for i in exp.interventions]
        check_names(self.interventions)  # M3a
        self.registry = Registry() if getattr(self.board, "registry", False) else None  # M3b
        self._probes_stopped = False
        self._probe_hard_ceiling = False
        self._probe_no_context: set[tuple[str, str]] = set()
        self.dir.mkdir(parents=True, exist_ok=True)
        self.blobs = BlobStore(self.dir / "blobs")
        self.snapshots = SnapshotStore(self.dir, self.blobs)
        self.live_agents: list[AgentId] = list(self.agents)
        self.outcomes_prev: dict[str, list[dict]] = {}
        self._git = git_identity(REPO_DIR)
        self.ledger = Ledger()
        self.gate = Gate(exp.resolved_providers(), self.ledger, self.budget)
        self.cache = InferenceCache(self.dir / "blobs")
        self.inference = Inference(run_id=self.run_id, gate=self.gate, blobs=self.blobs,
                                   cache=self.cache, log_operational=self.log_operational)
        if self.options.repeat:
            self.inference = RepeatSeeded(self.inference, self.options.seed, self.options.repeat)

    def _init_fresh(self) -> None:
        seed = self.options.seed
        for a, p in self.participants.items():
            p.bind(a, agent_stream(seed, a, self.options.repeat))
        self.roles = bind_roles(self.board, self.participants, agent_roles(  # M3c
            self.spec.roles, self.spec.participant_roles, len(self.agents)))
        self.world.reset(derive(seed, "world"), list(self.agents))
        self._set_truth()
        self._set_agents()

    def _set_agents(self) -> None:
        """Tell metrics the live agent list (at reset, restore, and whenever it changes)."""
        self._metric_agents = list(self.live_agents)
        for m in self.metrics:
            m.set_agents(list(self.live_agents))

    def _set_truth(self) -> None:
        if any(m.needs_truth() for m in self.metrics):
            truth = self.world.verify()
            for m in self.metrics:
                if m.needs_truth():
                    m.set_truth(truth)

    def _plugin_blobs(self) -> dict[str, bytes]:
        out = {"world": self.world.snapshot(), "board": self.board.snapshot(),
               "scheduler": self.scheduler.snapshot(), "ledger": self.ledger.snapshot()}
        for a, p in self.participants.items():
            out[f"participant:{a}"] = p.snapshot()
        for m in self.metrics:
            out[f"metric:{m.name}"] = m.snapshot()
        for iv in self.interventions:
            out[f"intervention:{iv.name}"] = iv.snapshot()
        if self.registry is not None:  # M3b
            out["registry"] = self.registry.snapshot()
        return out

    def _restore(self, manifest: SnapshotManifest, parent_spec: RunSpec | None) -> None:
        """Restore every plugin from `manifest`; `parent_spec` (forks) enables the match rules."""
        blobs = self.snapshots.load(manifest)
        same = parent_spec is None
        if parent_spec is not None and len(parent_spec.participants) != len(self.agents):
            raise ValueError("a fork cannot change the number of participants")
        restored: dict[str, Any] = {"world": False, "participants": [], "metrics": []}
        if same or parent_spec.world.type == self.spec.world.type:
            self.world.restore(blobs["world"])
            restored["world"] = True
        self.board.restore(blobs["board"])
        self.scheduler.restore(blobs["scheduler"])
        if "ledger" in blobs:
            self.ledger.restore(blobs["ledger"])
            self.ledger.reserved_nano = {c: 0 for c in self.ledger.reserved_nano}
        for i, (a, p) in enumerate(self.participants.items()):
            key = f"participant:{a}"
            if key in blobs and (same or parent_spec.participants[i] == self.spec.participants[i]):
                p.restore(blobs[key])
                restored["participants"].append(a)
        for m in self.metrics:
            key = f"metric:{m.name}"
            if key in blobs and (same or metric_spec(m) in parent_spec.metrics):
                m.restore(blobs[key])
                restored["metrics"].append(m.name)
        for iv in self.interventions:  # M3a: same rule as metrics
            key = f"intervention:{iv.name}"
            if key in blobs and (same or PluginSpec(**iv.spec()) in parent_spec.interventions):
                iv.restore(blobs[key])
        if self.registry is not None and "registry" in blobs:  # M3b
            self.registry.restore(blobs["registry"])
        self._set_truth()
        self.outcomes_prev = {a: list(v) for a, v in manifest.outcomes_prev.items()}
        self.live_agents = [AgentId(a) for a in manifest.live]
        self._set_agents()
        self.last_round = manifest.round
        if not same:
            self.restored = restored

    # ---- logging -----------------------------------------------------------------------------
    def _append(self, cls: type[Event], round: int, agent: str | None = None, **kw: Any) -> Event:
        ev = cls(run=self.run_id, round=round, agent=agent, **kw)
        self._log_event(ev)
        return ev

    def _log_event(self, ev: Event) -> None:
        self.log.append(ev)
        if ev.type not in OPERATIONAL_TYPES:
            self._round_events.append(ev)

    def log_operational(self, event: Event) -> int:
        """Append an operational event (`inference_attempt`, `inference_response`) immediately.

        Unlike turn events, which the executor buffers per agent until the round's turns finish,
        operational events go straight to the log, so an attempt is on disk before the model is
        called and spend survives an aborted round. They are never fed to metrics and
        `logical_view` drops them. Returns the assigned seq.
        """
        if event.type not in OPERATIONAL_TYPES:
            raise ValueError(f"{event.type!r} is not an operational event")
        return self.log.append(event)

    def _run_started(self, round: int) -> None:
        git_commit, dirty = self._git
        self._append(RunStartedEvent, round, spec_hash=spec_hash(self.spec), git_commit=git_commit,
                     dirty=dirty, run_spec=self.spec.model_dump(mode="json"),
                     parent_run=self.parent_run, fork_round=self.fork_round)

    # ---- modes -------------------------------------------------------------------------------
    def live(self) -> Runner:
        """Start the run from round 0 (or, for a fork child, from the fork round)."""
        if self.log_path.exists() and self.log_path.stat().st_size and self.parent is None:
            meta = self._read_meta() or {}
            old_hash, new_hash = meta.get("spec_hash"), spec_hash(self.spec)
            if old_hash is not None and old_hash != new_hash:
                raise FileExistsError(
                    f"{self.dir} already holds a run of a different configuration "
                    f"(existing spec_hash {old_hash}, this run's spec_hash {new_hash}); the run id "
                    "covers only name, arm and seed, so use another out dir, name or arm"
                )
            raise FileExistsError(f"{self.log_path} already exists; use resume() or a new out dir")
        self._build()
        self._write_artifacts()
        self.log = EventLog(self.log_path)
        try:
            self._init_fresh()
            if self.parent is None:
                start = 1
            elif self.parent.at_round == 0:  # M3a §2: a fork at round 0 is a fresh reset
                self.restored = {"world": False, "participants": [], "metrics": [], "reset": True}
                start = 1
            else:
                manifest = self.snapshots.read(self.parent.at_round)
                self._restore(manifest, self.parent.parent_spec)
                start = self.parent.at_round + 1
            self._write_meta()  # before run_started: the dir is always resumable or removable
            self._run_started(start - 1)
            _run_coro(self._loop(start))
        finally:
            self.log.close()
        return self

    def resume(self, budget: Budget | None = None) -> Runner:
        """Recover after a crash (§12 `recover()`) and continue live.

        A run that ended with `soft_budget` or `hard_ceiling` is resumable too: its `run_ended`
        (and, after a hard ceiling, the aborted round's events) go to `discarded.jsonl` like a
        crashed round's. `budget`, when given, replaces the effective budget (stored as
        `run.json["budget"]`; the archived spec and its hash are unchanged) and is logged as a
        `budget_changed` event right after the recovered commit point. A run that ended for any
        other reason is left untouched.
        """
        meta = self._read_meta()
        if meta is None:
            raise RecoveryError(f"{self.dir} has no run.json")
        self.spec = RunSpec.model_validate(meta["spec"])
        self._build()
        self.log = EventLog(self.log_path)
        try:
            events = list(self.log)
            ended = [e for e in events if e.type == "run_ended"]
            if ended and ended[-1].reason not in BUDGET_REASONS:
                self.status, self.end_reason = "ended", meta.get("end_reason")
                self.last_round = meta.get("last_round", 0)
                return self
            self._init_fresh()
            start = self.recover(events, meta)
            if budget is not None:
                old = self.budget
                self.budget = self.gate.budget = budget
                self._append(BudgetChangedEvent, start - 1, old=old.model_dump(mode="json"),
                             new=budget.model_dump(mode="json"))
                self.log.sync()
                self._write_meta()
            _run_coro(self._loop(start))
        finally:
            self.log.close()
        return self

    def recover(self, events: list[Event] | None = None, meta: dict | None = None) -> int:
        """Truncate to the last commit, restore its snapshot; return the next round to run.

        Spend survives: the ledger is `run.json["ledger"]` (accurate up to `run.json["ledger_seq"]`)
        plus every operational event after that seq, kept or dropped: an `inference_response`
        charges its `cost_usd` to its attempt's category, an attempt without a response (in flight
        at the crash) charges its `reserved_usd`, and every non-cached attempt counts as a call.
        A `budget_changed` event directly after the commit point is kept.
        """
        events = list(self.log) if events is None else events
        meta = self._read_meta() if meta is None else meta
        ledger = self._recovered_ledger(events, meta or {})
        lc = self.log.last_committed()
        if lc is None:
            starts = [e.seq for e in events if e.type == "run_started"]
            if not starts:
                if self.parent_run is not None:
                    raise RecoveryError("fork child has no run_started event")
                # crashed between writing run.json and appending run_started: start over
                dropped = self.log.truncate_after(-1)
                if dropped:
                    write_jsonl(self.dir / "discarded.jsonl", dropped)
                self.last_round = 0
                self._write_meta()
                self._run_started(0)
                return 1
            keep = starts[-1]
            committed = None
            for e in events:
                if e.seq == keep + 1 and e.type == "budget_changed":
                    keep = e.seq
        else:
            committed, keep = lc
            for e in events:
                if e.seq <= keep:
                    continue
                if e.seq == keep + 1 and (
                    (e.type == "snapshot" and e.round == committed)
                    or e.type in ("run_started", "budget_changed")
                ):
                    keep = e.seq
                else:
                    break
        dropped = self.log.truncate_after(keep)
        if dropped:
            write_jsonl(self.dir / "discarded.jsonl", dropped)
        if committed is None:
            self.last_round = 0
            self._adopt_ledger(ledger)
            self._write_meta()
            return 1
        manifest = self.snapshots.latest(max_round=committed)
        if manifest is None or manifest.round != committed:
            raise RecoveryError(
                f"no snapshot for committed round {committed}; M1a resume requires snapshot_every == 1"
            )
        self._restore(manifest, None)
        self._adopt_ledger(ledger)
        kept = [e for e in events if e.seq <= keep]
        has_snap = any(e.type == "snapshot" and e.round == committed for e in kept)
        if not has_snap:
            self._append(SnapshotEvent, committed, manifest_path=self._manifest_rel(committed))
        self._write_meta()
        return committed + 1

    def _recovered_ledger(self, events: list[Event], meta: dict) -> Ledger:
        ledger = Ledger.from_dict(meta.get("ledger"))
        since = meta.get("ledger_seq", -1) if meta.get("ledger") is not None else -1
        attempts = {}
        responses = {}
        for e in events:
            if e.seq <= since:
                continue
            if e.type == "inference_attempt":
                attempts[e.call_id] = e
            elif e.type == "inference_response":
                responses[e.call_id] = e
        for call_id, a in attempts.items():
            r = responses.get(call_id)
            if r is not None and r.cached:
                continue
            ledger.calls += 1
            ledger.charge(a.category, r.cost_usd if r is not None else a.reserved_usd)
        return ledger

    def _adopt_ledger(self, ledger: Ledger) -> None:
        self.ledger.spent_nano = dict(ledger.spent_nano)
        self.ledger.reserved_nano = {c: 0 for c in ledger.reserved_nano}
        self.ledger.calls = ledger.calls

    def fork(self, at_round: int, experiment: Experiment | None = None,
             out: Path | str | None = None, *, max_rounds: int | None = None) -> Runner:
        """Copy the parent's prefix and snapshot at `at_round` into a new run and continue live."""
        meta = self._read_meta()
        if meta is None:
            raise ValueError(f"{self.dir} is not a run directory")
        parent_spec = RunSpec.model_validate(meta["spec"])
        parent_id = meta["run_id"]
        if at_round == 0:
            return self._fork_at_zero(parent_spec, parent_id, experiment, out, max_rounds)
        raw = self.log_path.read_bytes()
        lines = raw.splitlines(keepends=True)
        end = None
        for i, line in enumerate(lines):
            ev = parse_event(line)
            if ev.type == "round_committed" and ev.round == at_round:
                end = i + 1
                if end < len(lines):
                    nxt = parse_event(lines[end])
                    if nxt.type == "snapshot" and nxt.round == at_round:
                        end += 1
                break
        if end is None:
            raise ValueError(f"round {at_round} is not committed in {self.dir}")
        src_snaps = SnapshotStore(self.dir)
        if at_round not in src_snaps.list_rounds():
            raise ValueError(f"no snapshot for round {at_round} in {self.dir}")
        child_id, child_dir = self._fork_dir(parent_id, at_round, out)
        child_dir.mkdir(parents=True)
        prefix = lines[:end]
        (child_dir / "events.jsonl").write_bytes(b"".join(prefix))
        dst_snaps = SnapshotStore(child_dir)
        shutil.copyfile(src_snaps.path(at_round), dst_snaps.path(at_round))
        manifest = src_snaps.read(at_round)
        shas = set(manifest.plugins.values())
        for line in prefix:
            ev = parse_event(line)
            if ev.type == "delivery":
                shas.add(ev.content_hash)
            elif ev.type == "inference_attempt":
                shas.add(ev.request_hash)
            elif ev.type == "inference_response" and ev.response_hash:
                shas.add(ev.response_hash)
        InferenceCache(self.dir / "blobs").copy_to(InferenceCache(child_dir / "blobs"), at_round)
        for sha in sorted(shas):
            if not src_snaps.blobs.path(sha).exists():
                continue
            dst = dst_snaps.blobs.path(sha)
            dst.parent.mkdir(parents=True, exist_ok=True)
            shutil.copyfile(src_snaps.blobs.path(sha), dst)
        options = parent_spec.options
        if max_rounds is not None:
            options = options.model_copy(update={"max_rounds": max_rounds})
        if experiment is None:
            experiment = self.experiment
        child = Runner(child_dir, experiment, options,
                       parent=ForkOrigin(parent_id, at_round, parent_spec), run_id=child_id)
        return child.live()

    def _fork_dir(self, parent_id: str, at_round: int, out: Path | str | None) -> tuple[str, Path]:
        out_dir = Path(out) if out is not None else self.dir.parent
        n = 1
        while (out_dir / fork_run_id(parent_id, at_round, n)).exists():
            n += 1
        child_id = fork_run_id(parent_id, at_round, n)
        return child_id, out_dir / child_id

    def _fork_at_zero(self, parent_spec: RunSpec, parent_id: str, experiment: Experiment | None,
                      out: Path | str | None, max_rounds: int | None) -> Runner:
        """Fork at round 0 (M3a §2). Runs never write a round-0 snapshot, so nothing is copied:
        the child re-resets the world with the parent's seed (and repeat) under `experiment`
        (possibly edited), binds fresh participants, uses the same schedule and topology streams,
        and its log starts with its own `run_started(round=0, parent_run, fork_round=0)`."""
        if experiment is None:
            experiment = self.experiment
        if len(experiment.participants) != len(parent_spec.participants):
            raise ValueError("a fork cannot change the number of participants")
        child_id, child_dir = self._fork_dir(parent_id, 0, out)
        child_dir.mkdir(parents=True)
        options = parent_spec.options
        if max_rounds is not None:
            options = options.model_copy(update={"max_rounds": max_rounds})
        child = Runner(child_dir, experiment, options,
                       parent=ForkOrigin(parent_id, 0, parent_spec), run_id=child_id)
        return child.live()

    def replay(self) -> dict:
        """Fold metrics over the log and recompute the score; raise ReplayMismatch on a difference."""
        meta = self._read_meta()
        if meta is None:
            raise ValueError(f"{self.dir} is not a run directory")
        self.spec = RunSpec.model_validate(meta["spec"])
        self._build()
        self.world.reset(derive(self.options.seed, "world"), list(self.agents))
        self._set_truth()
        self._set_agents()
        fork_round = meta.get("fork_round") or 0
        restored = meta.get("restored") or {}
        if fork_round:
            blobs = self.snapshots.load(self.snapshots.read(fork_round))
            for m in self.metrics:
                if m.name in restored.get("metrics", []):
                    m.restore(blobs[f"metric:{m.name}"])
            self._set_truth()
        log = EventLog(self.log_path)
        by_name = {m.name: m for m in self.metrics}
        last_committed = 0
        checked = 0
        for ev in log:
            if ev.type in OPERATIONAL_TYPES or ev.round <= fork_round:
                continue
            if ev.type == "round_committed":
                last_committed = ev.round
            if ev.type == "round_started" and sorted(ev.order) != sorted(self._metric_agents):
                self.live_agents = [AgentId(a) for a in ev.order]
                self._set_agents()
            if ev.type == "intervention" and replay_world_op(self.world, ev):  # M3a
                self._set_truth()
            if ev.type not in NOT_FED:
                for m in self.metrics:
                    m.update(ev)
            elif ev.type == "metric" and ev.name in by_name:
                value, denom = by_name[ev.name].value()
                if (value, denom) != (ev.value, ev.denominator):
                    raise ReplayMismatch(
                        f"round {ev.round} metric {ev.name}: log has ({ev.value}, {ev.denominator}), "
                        f"replay gives ({value}, {denom})"
                    )
                checked += 1
        log.close()
        if self.gate.provider_calls:
            raise ReplayMismatch(f"replay made {self.gate.provider_calls} provider calls")
        last_committed = max(last_committed, fork_round)
        score = None
        manifest = self.snapshots.latest(max_round=last_committed)
        if manifest is not None and manifest.round == meta.get("last_round"):
            world_ok = manifest.round != fork_round or restored.get("world", True)
            if world_ok:
                self.world.restore(self.snapshots.load(manifest)["world"])
                score = _jsonable(self.world.score())
                if score != meta.get("score"):
                    raise ReplayMismatch(
                        f"score after round {manifest.round}: run.json has {meta.get('score')}, "
                        f"replay gives {score}"
                    )
        return {"last_round": last_committed, "metrics_checked": checked, "score": score,
                "provider_calls": self.gate.provider_calls}

    # ---- the round loop ----------------------------------------------------------------------
    async def _loop(self, start: int) -> None:
        self._probe_no_context = {
            (e.probe, str(e.agent)) for e in self.log
            if e.type == "probe" and (e.parsed or {}).get("skipped") == "no_context"
        }
        r = start
        while True:
            reason = self._end_reason(r - 1)
            if reason is not None:
                self._end(r - 1, reason)
                return
            try:
                await self._round(r)
            except HardCeilingReached:
                self._abort_round(r)
                return
            if self._probe_hard_ceiling:  # a probe hit the ceiling after round r was committed
                self._end(r, "hard_ceiling")
                return
            r += 1

    # ---- probes (docs/INTERFACE-M1b.md §5; details in swarmlab/probes.py) ------------------------
    async def _probe_round(self, r: int, order: list[AgentId], ex: RoundExecutor) -> None:
        for probe in self.probes:
            if r % max(1, probe.every) or self._probes_stopped:
                continue
            targets: list[AgentId] = []
            for a in order:
                if a not in self.live_agents:
                    continue
                if not callable(getattr(self.participants[a], "probe_context", None)):
                    if (probe.name, str(a)) not in self._probe_no_context:
                        self._probe_no_context.add((probe.name, str(a)))
                        self._append(ProbeEvent, r, a, probe=probe.name, question_hash="",
                                     raw_hash="", parsed={"skipped": "no_context"}, ok=False)
                    continue
                targets.append(a)
            if not targets:
                continue
            results = await asyncio.gather(*(self._probe_one(probe, a, r, ex) for a in targets))
            for a, fields in zip(targets, results, strict=True):
                self._append(ProbeEvent, r, a, **fields)

    async def _probe_one(self, probe: Probe, agent: AgentId, r: int, ex: RoundExecutor) -> dict:
        """One agent's probe: the `probe` event fields (budget exceptions become skipped events)."""
        participant = self.participants[agent]
        question = probe.question(agent, r)
        q_hash = self.blobs.put_text(question)
        req = ChatRequest(messages=probe_messages(participant.probe_context())
                          + [ChatMessage(role="user", content=question)], tools=[],
                          **participant.model_request_defaults())
        cost = 0.0
        names = probe.candidates_from_context(participant.probe_context())
        try:
            resp = await ex.infer(agent, req, "measurement")
            cost += self.gate.provider_for(req.model).cost(req, resp.usage)
            raw = resp.text
            ok, parsed = probe.parse(raw, names)
            coder = probe.coder_model()
            if not ok and coder:
                creq = ChatRequest(model=coder, max_tokens=256, temperature=0.0, tools=[], messages=[
                    ChatMessage(role="system", content=CODER_SYSTEM),
                    ChatMessage(role="user", content=f"Question:\n{question}\n\nReply:\n{raw}"),
                ])
                cresp = await ex.infer(agent, creq, "measurement")
                cost += self.gate.provider_for(creq.model).cost(creq, cresp.usage)
                ok, parsed = probe.parse(cresp.text, names)
                parsed = {**parsed, "coded": True}
        except MeasurementBudgetReached:
            self._probes_stopped = True
            return {"probe": probe.name, "question_hash": q_hash, "raw_hash": "",
                    "parsed": {"skipped": "measurement_budget"}, "ok": False, "cost_usd": cost}
        except HardCeilingReached:
            self._probes_stopped = True
            self._probe_hard_ceiling = True
            return {"probe": probe.name, "question_hash": q_hash, "raw_hash": "",
                    "parsed": {"skipped": "hard_ceiling"}, "ok": False, "cost_usd": cost}
        except ProviderError as e:
            # A probe call that exhausted its retries must not abort the round: log and move on.
            return {"probe": probe.name, "question_hash": q_hash, "raw_hash": "",
                    "parsed": {"skipped": "provider_error", "error": str(e)[:200]},
                    "ok": False, "cost_usd": cost}
        return {"probe": probe.name, "question_hash": q_hash, "raw_hash": self.blobs.put_text(raw),
                "parsed": _jsonable(parsed), "ok": ok, "cost_usd": cost}

    def _end_reason(self, committed: int) -> str | None:
        if committed >= 1 and self.world.terminal():
            return "terminal"
        if committed >= self.options.max_rounds:
            return "max_rounds"
        soft = self.budget.soft_usd
        if soft > 0 and self.ledger.spent["swarm"] >= soft:
            return "soft_budget"
        return None

    def _abort_round(self, r: int) -> None:
        """Hard ceiling inside round r: drop the round's buffers, end resumably at r - 1.

        The log keeps `round_started(r)` and the round's operational events (spend audit); none
        of the round's turn, post, delivery or action events are written. In-memory plugin state
        may hold partial round-r changes, so `run.json` keeps the score of the last commit;
        `resume()` restores everything from the snapshot of round r - 1.
        """
        meta = self._read_meta() or {}
        self._aborted_score = meta.get("score")
        self._end(r - 1, "hard_ceiling")

    def _end(self, round: int, reason: str) -> None:
        self._append(RunEndedEvent, round, reason=reason)
        self.log.sync()
        self.status, self.end_reason = "ended", reason
        self._write_meta()

    def _manifest_rel(self, round: int) -> str:
        return self.snapshots.path(round).relative_to(self.dir).as_posix()

    async def _turn(self, agent: AgentId, ex: RoundExecutor, round: int,
                    sem: asyncio.Semaphore | None) -> None:
        if sem is not None:
            async with sem:
                await self._turn_inner(agent, ex, round)
        else:
            await self._turn_inner(agent, ex, round)

    async def _turn_inner(self, agent: AgentId, ex: RoundExecutor, round: int) -> None:
        obs = self.world.observe(agent)
        ex.begin_turn(agent, _jsonable(obs.private))
        obs = obs.model_copy(update={"private": {}})  # private never reaches the participant
        pushed: list[dict] = []
        if self.board.delivery == "push":
            pushed = [
                {"delivery_id": d.delivery_id, "post_id": d.post_id,
                 "eligible_round": d.eligible_round, "content": self.board.content(d, self.blobs)}
                for d in ex.pushable(agent, self.board.push_limit)  # M3c: role channel filter
            ]
        view = View(round=round, agent=agent, observation=obs,
                    outcomes=list(self.outcomes_prev.get(agent, [])), pushed=pushed,
                    tools=ex.schemas(agent), description=self.world.description())
        usage: dict = {}
        error = None
        try:
            result = await self.participants[agent].turn(view, AgentTools(ex, agent))
            kind = "end_turn" if ex.turn_ended(agent) else "no_tool"
            if result is not None and hasattr(result, "model_dump"):
                usage = result.model_dump(mode="json")
        except HardCeilingReached:
            raise
        except TurnCapReached:
            kind = "cap"
        except Exception as e:  # noqa: BLE001 - any participant failure ends its turn as "error"
            kind = "error"
            error = getattr(e, "turn_error", None) or traceback.format_exc()  # M3a: e.g. context_limit
        if ex.hard_ceiling:  # the participant swallowed it; the round is aborted all the same
            raise HardCeilingReached(f"{agent}: hard ceiling reached during the turn")
        drain = getattr(self.participants[agent], "drain_overflow", None)  # M3a §3 context limit
        for fields in drain() if callable(drain) else ():
            ex.note(OverflowEvent, agent, **fields)
        inferred = ex.inference_usage(agent)
        if inferred is not None:
            usage = {**usage, **inferred}
        ex.end_turn_event(agent, kind, usage, error)

    async def _round(self, r: int) -> None:
        seed = self.options.seed
        self._round_events = []
        self.world.begin_round(r)  # M3b: world dynamics, then scheduled worker failures
        failed = [a for a in self.world.killed_at(r) if a in self.live_agents]
        if failed:
            Ops(self, "worker_failures", r).kill(failed)
        if self.live_agents != getattr(self, "_metric_agents", None):
            self._set_agents()
        order = self.scheduler.order(r, list(self.live_agents), derive(seed, "schedule", r))
        self._append(RoundStartedEvent, r, order=list(order))
        ex = RoundExecutor(
            run=self.run_id, round=r, world=self.world, board=self.board, blobs=self.blobs,
            agents=list(self.live_agents), commit=self.options.commit,
            max_calls_per_turn=self.options.max_calls_per_turn,
            topology_rng=lambda: derive(seed, "topology", r), inference=self.inference,
            registry=self.registry, claim_policy=self.board.claim_policy,
            roles=self.roles,  # M3c
        )
        self.executor = ex
        if self.options.commit == "round_end":
            sem = asyncio.Semaphore(max(1, self.options.concurrency))
            try:
                async with asyncio.TaskGroup() as tg:  # a HardCeilingReached cancels the siblings
                    for a in order:
                        tg.create_task(self._turn(a, ex, r, sem))
            except* HardCeilingReached as eg:
                raise eg.exceptions[0] from None
        else:
            for a in order:
                await self._turn(a, ex, r, None)
        for a in order:
            for ev in ex.events(a):
                self._log_event(ev)
        # commit
        provisional: dict[str, str] = {}
        if self.options.commit == "round_end":
            for a in order:
                for (channel, text, fields), tmp in zip(ex.buffered_posts(a), ex.buffered_post_ids(a),
                                                        strict=True):
                    provisional[self.board.buffer_post(a, r, channel, text, fields)] = tmp
            posts, deliveries = self.board.commit(r, list(self.live_agents),
                                                  derive(seed, "topology", r), self.blobs)
            reg = (self.registry.commit(r, [op for a in order for op in ex.buffered_registry(a)])
                   if self.registry is not None else [])  # M3b: registry before world actions
            actions = [x for a in order for x in ex.buffered_actions(a)]
            outcomes, claims = commit_world(self.world, actions, registry=self.registry,
                                            policy=self.board.claim_policy, round=r)
            applied = [(a, aid, act, out) for (a, aid, act), out in zip(actions, outcomes, strict=True)]
        else:
            posts, deliveries = ex.committed.posts, ex.committed.deliveries
            applied = ex.committed.actions
            reg, claims = ex.committed.registry, ex.committed.claims
        for p in posts:
            self._append(PostEvent, r, p.agent, post_id=p.post_id,
                         provisional_id=provisional.get(p.post_id, p.post_id), channel=p.channel,
                         text=p.text, fields=dict(p.fields))
        for d in deliveries:
            self._append(DeliveryEvent, r, d.recipient, post_id=d.post_id, recipient=d.recipient,
                         delivery_id=d.delivery_id, eligible_round=d.eligible_round,
                         content_hash=d.content_hash)
        for o in reg:  # M3b
            self._append(RegistryEvent, r, o.agent, op_id=o.op_id, op=o.op, key=o.key, ok=o.ok,
                         version=o.version, owner=o.owner, expires_round=o.expires_round,
                         value=_jsonable(o.value), error=o.error)
        for c in claims:
            self._append(ClaimEvent, r, c["agent"], **{k: v for k, v in c.items() if k != "agent"})
        self.outcomes_prev = {a: [] for a in self.live_agents}
        for a, aid, act, out in applied:
            feedback = _jsonable(out.feedback)
            self._append(ActionCommittedEvent, r, a, action_id=aid, action=act.model_dump(mode="json"),
                         accepted=out.accepted, feedback=feedback)
            self.outcomes_prev.setdefault(a, []).append(
                {"action_id": aid, "tool": act.name, "accepted": out.accepted, "feedback": feedback})
        for o in reg:  # M3b
            self.outcomes_prev.setdefault(o.agent, []).append(_jsonable(o.view_outcome()))
        # interventions (M3a: after the commit, before probes, in spec order)
        fire_interventions(self, r)
        # probes (after the commit, before metrics, so probe-sourced metrics see this round)
        await self._probe_round(r, order, ex)
        # metrics
        fed = [parse_event(ev.model_dump_json()) for ev in self._round_events if ev.type not in NOT_FED]
        for m in self.metrics:
            for ev in fed:
                m.update(ev)
        for m in self.metrics:
            value, denom = m.value()
            self._append(MetricEvent, r, name=m.name, value=value, denominator=denom)
        led = self.ledger.spent
        self._append(BudgetEvent, r, spent_swarm=led["swarm"], spent_measurement=led["measurement"],
                     reserved=self.ledger.reserved, calls=self.ledger.calls)
        # snapshot (before the commit marker), commit marker, run.json, snapshot event
        snap = r % max(1, self.options.snapshot_every) == 0
        if snap:
            manifest = SnapshotManifest(
                run=self.run_id, round=r, log_seq=self.log.next_seq,
                outcomes_prev=self.outcomes_prev, live=list(self.live_agents),
            )
            self.snapshots.write(manifest, self._plugin_blobs())
        self._append(RoundCommittedEvent, r, n_posts=len(posts), n_actions=len(applied),
                     n_deliveries=len(deliveries))
        self.last_round = r
        self._write_meta()
        if snap:
            self._append(SnapshotEvent, r, manifest_path=self._manifest_rel(r))
