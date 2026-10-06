# swarmlab M1a interface contract

Scope: the first working slice from DESIGN.md. Scripted participants, phase-commit runner with tool executor, event log with commit markers and per-round snapshots, board with inboxes, FlagGame text variant, replay, resume, fork, minimal viewer, and the public `Experiment` API. No model is called in M1a. Everything here is binding for implementers; anything not here is the implementer's choice and must be documented in the module docstring.

Conventions: Python 3.12+ (the dev env runs 3.14), `pydantic` v2 models for data crossing a boundary, abstract base classes with working defaults for plugins, `asyncio` in the runner. Rounds are 1-based; round 0 is the initial state. All randomness comes from `swarmlab.rng`, never from the global `random`.

Guiding test (DESIGN.md, Experimenter interface): a scientist changes the hypothesis without touching execution machinery. Plugins are small classes; persistence, action ordering, validation, and status tools have defaults on the base classes.

## 0. Public API

```python
class Experiment(BaseModel, arbitrary_types_allowed=True):
    name: str
    world: World
    participants: list[Participant]          # prototypes; one fresh copy is bound per agent
    medium: Board = Board()
    metrics: list[str | Metric] = []
    budget: Budget = Budget()
    probes: list = []; interventions: list = []   # M1b
    arm: str | None = None; options: RunOptions | None = None   # set when built from YAML; run() falls back to them

    def run(self, seed: int, max_rounds: int, *, commit: Literal["round_end","immediate"] = "round_end",
            max_calls_per_turn: int = 20, snapshot_every: int = 1, out: Path | str = "runs",
            concurrency: int = 32) -> "Run": ...
    def to_spec(self, seed, max_rounds, **run_options) -> RunSpec: ...   # resolved, hashable
    @classmethod
    def from_spec(cls, spec: RunSpec) -> "Experiment": ...                # via entry points
    @classmethod
    def from_yaml(cls, path, arm: str) -> "Experiment": ...
    def to_yaml(self, path) -> None: ...

class Run:
    dir: Path; spec: RunSpec; id: RunId
    score: dict                                 # final evaluator-side world score
    metrics: dict[str, list[tuple[int, float | None, int]]]   # name -> [(round, value, denominator)]
    events: Iterator[Event]                     # logical view by default; events_all for operational too
    status: Literal["running","ended"]; end_reason: str | None
    def view(self) -> Path: ...                 # builds view.html, returns its path
    def fork(self, at_round: int, experiment: Experiment | None = None) -> "ForkHandle": ...   # ForkHandle.run(out=None) -> Run
    def resume(self) -> "Run": ...
    @classmethod
    def load(cls, dir) -> "Run": ...            # replays the log; never calls plugins' mutating paths
```

`participants=[EvidenceAggregator()] * 16` is valid: the runner deep-copies each prototype per agent and calls `bind`. Every plugin instance exposes `spec() -> dict` as `{"type": <entry-point name>, "params": {...}}` built from its constructor kwargs; `to_spec` serialises the experiment from those, and `from_spec` rebuilds it through entry points. A plugin defined inline in a script (no entry point) is allowed for Python use; `to_spec` records `type` as `module:Class` and `from_spec` imports it, so identity and replay still work on the same code tree.

YAML is the serialised `Experiment` plus arms: `Experiment.from_yaml(path, arm)` and the CLI build the same object and call `run`.

## 1. Package layout

```
swarmlab/
  __init__.py      re-exports: Experiment, Run, Budget, Board, Policy, Topology, World, Participant,
                   Metric, Outcome, Action, tool, text_observation
  ids.py           AgentId, RunId, PostId, ActionId, CallId, DeliveryId         [done]
  rng.py           derive(seed, *labels) -> random.Random                       [done]
  base.py          Persistable mixin (pickle-based snapshot/restore), Plugin (spec())  [done]
  spec.py          Budget, RunSpec, spec_hash(), YAML load/dump
  events.py        Event union, EventLog, logical_view()
  blobs.py         BlobStore
  tools.py         ToolSchema, ToolCall, ToolResult, ToolExecutor, EndTurn, TurnCapReached  [done]
  view.py          Part, Observation, View, text_observation()                  [done]
  world/base.py    World base class, Action, Ack, Outcome, @tool                [done]
  world/flaggame.py
  medium/base.py   Policy, Topology base classes                                [done]
  medium/board.py  Board, Post, Delivery, DelayPolicy
  medium/topology.py  Broadcast, Gossip, Groups
  participants/base.py  Participant base class, TurnUsage                       [done]
  participants/scripted.py  Silent, EvidenceAggregator, Enumerator
  scheduler.py     Scheduler base, SeededShuffle
  executor.py      RoundExecutor
  runner.py        Runner: live, replay, resume, fork
  experiment.py    Experiment, Run
  snapshot.py      SnapshotManifest, SnapshotStore
  metrics/base.py  Metric base class, registry                                  [done]
  metrics/belief.py, metrics/comm.py
  viewer/build.py
  cli.py
tests/
```
Files marked `[done]` exist on `main` and are the shared vocabulary; extend them only by adding, never by renaming.

## 2. Identifiers and randomness

- `AgentId` is `a` + zero-padded index (`a000`). `RunId` is `<experiment>__s<seed>` for a Python-defined experiment, `<experiment>__<arm>__s<seed>` from YAML; a fork appends `__f<round>_<n>`.
- `rng.derive(seed, *labels) -> random.Random`. Fixed label roots: `("world",)`, `("private", agent)`, `("schedule", round)`, `("topology", round)`, `("agent", agent)`, `("scripted", agent)`.

## 3. Spec

```python
class Budget(BaseModel):
    soft_usd: float = 0.0; hard_usd: float = 0.0; measurement_usd: float = 0.0   # M1a: recorded, not enforced
class PluginSpec(BaseModel): type: str; params: dict = {}
class MediumSpec(BaseModel):
    topology: PluginSpec = PluginSpec(type="broadcast")
    delivery: Literal["pull", "push"] = "pull"; push_limit: int = 20
    policies: list[PluginSpec] = []; channels: list[str] = ["main"]
class RunOptions(BaseModel):
    seed: int; max_rounds: int; commit: Literal["round_end", "immediate"] = "round_end"
    max_calls_per_turn: int = 20; snapshot_every: int = 1; concurrency: int = 32
class RunSpec(BaseModel):
    experiment: str; arm: str | None = None
    world: PluginSpec; participants: list[PluginSpec]      # one entry per agent, in agent order
    medium: MediumSpec; metrics: list[PluginSpec]; budget: Budget; options: RunOptions
```

`Plugin.type_name()` uses `entry_point` only when it is defined on the class itself (`cls.__dict__`), so a subclass of a registered plugin serialises as `module:Class`, never as its parent. `spec_hash(run_spec)` is SHA-256 of canonical JSON (sorted keys, no whitespace). Run identity is `(git_commit, dirty, spec_hash)`; the runner records all three on `run_started` and archives the serialised spec to `artifacts/spec.yaml`. YAML files hold `name`, `arms: {arm: {world, participants: [{type, count, params}], medium, metrics}}`, `budget`, and `options`; `participants[].count` expands to repeated entries.

## 4. Events

Every event has `seq: int` (dense, assigned on append), `run: RunId`, `round: int`, `agent: AgentId | None`, `ts: float` (informational), `type: str`.

| type | fields | when |
|---|---|---|
| `run_started` | `spec_hash, git_commit, dirty, run_spec, parent_run, fork_round` | once |
| `round_started` | `order: list[AgentId]` | per round |
| `turn_started` | `private: dict` (the observation's evaluator-only data) | per agent turn |
| `tool_called` | `call_id, tool, args` | each tool call inside a turn |
| `tool_returned` | `call_id, result, pending: bool` | each tool return |
| `inference_attempt` | `call_id, provider, model, request_hash, reserved_usd` | M1b; operational |
| `inference_response` | `call_id, response_hash, usage, cost_usd, latency_s` | M1b; operational |
| `turn_ended` | `yield_kind: "no_tool"|"end_turn"|"cap"|"error", calls, usage` | per agent turn |
| `read` | `delivery_ids` | each board read, inside the turn |
| `post` | `post_id, provisional_id, channel, text, fields` | at commit, per accepted post |
| `delivery` | `post_id, recipient, delivery_id, eligible_round, content_hash` | at commit, per fan-out |
| `action_committed` | `action_id, action, accepted, feedback` | at commit, per buffered action |
| `world_changed` | `payload` (opaque) | at commit, optional |
| `metric` | `name, value, denominator` | after fold, per metric |
| `round_committed` | `n_posts, n_actions, n_deliveries` | last logical event of the round |
| `snapshot` | `manifest_path` | after `round_committed`, every `snapshot_every` |
| `run_ended` | `reason: "terminal"|"max_rounds"|"soft_budget"|"hard_ceiling"|"error"` | once |

**Logical versus operational.** `inference_attempt`, `inference_response`, and `ts` are operational; `logical_view(events)` strips them and also drops `seq`, because operational events are appended as they happen and shift later sequence numbers. Acceptance tests compare logical views.

**Order rule.** Turns run concurrently in live mode, but their events are buffered per agent and appended at commit in the round's seeded order, each agent's events in call order; then `post*`, `delivery*`, `action_committed*`, `world_changed`, `metric*`, `round_committed`, `snapshot`. Operational events are appended to the log immediately by the runner (never buffered), so an `inference_attempt` exists on disk before the model is called and spend survives an aborted round.

`EventLog(path)`: `append(event) -> seq`, `__iter__`, `last_committed() -> (round, seq) | None`, `truncate_after(seq)`. One JSONL file, fsync on `round_committed`.

## 5. Blobs

`BlobStore(dir)`: `put(bytes) -> sha256`, `get(sha) -> bytes`. Delivered content, inference bodies, and snapshot plugin state live here; events reference blobs by hash.

## 6. Tools and the executor

`ToolSchema`, `ToolCall`, `ToolResult`, `TurnCapReached` are in `tools.py` as shipped. The executor is the only mutation path. **A participant never receives the round executor itself.** It receives an agent-bound handle `AgentTools` with `agent`, `schemas() -> list[ToolSchema]`, `call(name, args) -> ToolResult`, and (M1b) `infer(request) -> response`; the handle carries the agent identity, so a participant cannot act or read as another agent and has no path to the world or board objects. `end_turn()` does not raise: it marks the turn ended and returns `ok=True`; any later call in the same turn returns `ok=False, error="turn_ended"`. `TurnCapReached` still raises after `max_calls_per_turn` calls. (`EndTurn` remains defined for backward compatibility but is no longer raised.) Namespaces: world tools from `World.tool_schemas()`; board tools `read_board(channel?, limit?)` and `post(channel, text, fields?)`; `my_status()` and `collective_status()` if the world enables them; `end_turn()` always. A call outside the agent's allowlist returns `ok=False, error="not_allowed"` and is logged.

Under `commit == "round_end"`: `post` and world actions return `pending=True` with `{"id": ...}`; `read_board` returns eligible unread inbox items (`eligible_round <= round`), marks them read, logs `read`; status tools answer from round-start state. Under `commit == "immediate"`: every call applies at once and returns the real outcome with `pending=False`.

## 7. View

`Part`, `Observation`, `View` as shipped in `view.py`, plus `text_observation(text: str, **private) -> Observation`. **`Observation.private` never reaches a participant**: the runner copies the observation with `private={}` into the view and records the private dict on the `turn_started` event (`private` field) for the evaluator and viewer.

## 8. World

`world/base.py` as shipped defines `Action`, `Ack`, `Outcome(accepted, feedback, action_id=None)`, the `@tool(name, description, params)` decorator, and the `World` base class:

```python
class World(Persistable, Plugin):
    # required
    def reset(self, rng: random.Random, agents: list[AgentId]) -> None: ...
    def observe(self, agent: AgentId) -> Observation: ...
    def score(self) -> dict: ...                       # evaluator-only
    # actions: methods decorated with @tool(name, description, params); signature (self, agent, **args) -> Outcome
    # defaults, override when needed
    def validate(self, agent, action) -> Ack            # checks the tool exists and args match the schema
    def commit(self, actions: list[tuple[AgentId, ActionId, Action]]) -> list[Outcome]
                                                        # calls the decorated method per action, in the given order
    def my_status(self, agent) -> dict | None           # None (default) means the tool is not exposed
    def collective_status(self) -> dict | None
    def terminal(self) -> bool                          # False
    def verify(self) -> dict                            # {}
    def tool_schemas(self) -> list[ToolSchema]          # from @tool decorations plus enabled status tools
    # snapshot()/restore() from Persistable: pickle of __dict__ minus `params`
```
Constructor kwargs are stored on `self.params` and returned by `spec()`. Nothing returned by `observe`, `validate`, `commit`, status tools, or tool schemas may reveal correctness; `score` and `verify` are called only by the runner.

### FlagGame (text variant)

Constructor kwargs with defaults: `height=8, width=12, palette=6, n_candidates=8, rival_edits=1, crop_h=3, crop_w=4, candidate_names="letters"`, `status_tools=("my_status","collective_status")`, `guess_limit=None`.

**No shortcut from the observation alone.** Candidates are generated as `n_candidates / 2` near-twin pairs: each pair is a structured flag plus a variant with `rival_edits` bands or blocks recoloured (both members are clean structured flags). The truth is a random member of a random pair and its twin is the rival. Every candidate therefore has a twin, and no similarity or cleanliness heuristic singles out the truth. A test asserts that a crop-free heuristic (choose among the most similar pair, then the cleaner member) scores within sampling noise of chance over 400 seeds.

- `reset` draws the pairs as above from `("world",)`; candidates are shuffled and named, the mapping is hidden. Each agent's crop is a random `crop_h × crop_w` window of the truth from a per-agent stream derived from the world rng.
- `observe`: one text part with the named candidate grids and the agent's crop rendered the same way, position withheld; `private` holds the crop coordinates.
- `@tool("guess", ...)`: records the latest guess; `Outcome.feedback == {"recorded": True}`. Guess limit enforced in `validate`.
- `my_status` → `{"current_guess", "guesses_made"}`; `collective_status` → `{"guess_counts", "agents_with_guess"}`. Both return `None` when disabled via `status_tools`.
- `terminal` is False; `score` → `{"accuracy", "n_guessed", "truth"}`; `verify` → truth, rival, candidates, crop positions.

## 9. Medium

`medium/base.py` as shipped:

```python
class Policy(Persistable, Plugin):
    def apply(self, reader: AgentId, post: "Post", round: int) -> tuple[int, str] | None:
        return round + 1, post.text          # default: available next round, unchanged
class Topology(Persistable, Plugin):
    def recipients(self, post: "Post", agents: list[AgentId], round: int, rng: random.Random) -> list[AgentId]:
        return [a for a in agents if a != post.agent]   # default: broadcast
```

`medium/board.py`:
```python
class Post(BaseModel):     post_id, round, agent, channel, text, fields: dict = {}
class Delivery(BaseModel): delivery_id, post_id, recipient, eligible_round, content_hash, read_round: int | None = None
class Board(Persistable, Plugin):
    def __init__(self, topology: str | Topology = "broadcast", delivery="pull", push_limit=20,
                 policies: list[Policy] = (), channels=("main",)) ...
    def buffer_post(self, agent, round, channel, text, fields) -> PostId
    def commit(self, round, agents, rng, blobs) -> tuple[list[Post], list[Delivery]]
    def read(self, agent, round, channel=None, limit=50) -> list[Delivery]    # eligible, unread; marks read_round
    def pushable(self, agent, round, limit) -> list[Delivery]
```
`commit` turns buffered posts into `Post` records in seeded order, computes recipients (author never included), applies policies in order (the first returning `None` withholds; otherwise the last content and the maximum eligible round win), stores content in blobs, creates one `Delivery` per recipient. There is no global cursor. Topologies: `Broadcast`, `Gossip(k=1)` (per round each agent's posts go to `k` partners from `("topology", round)`), `Groups(size)`. Policy: `DelayPolicy(rounds, readers=None)`. Under `commit == "immediate"` the default eligible round is `round`.

## 10. Participants

`participants/base.py` as shipped:
```python
class TurnUsage(BaseModel): calls=0; prompt_tokens=0; completion_tokens=0; cost_usd=0.0
class Participant(Persistable, Plugin):
    agent: AgentId; rng: random.Random
    def bind(self, agent, rng) -> None            # default stores both; override to add state
    async def turn(self, view: View, tools: AgentTools) -> TurnUsage: ...   # required
```
A turn ends when `turn` returns or `TurnCapReached` propagates. `yield_kind` is `end_turn` if `end_turn()` was called during the turn (the returned `TurnUsage` is kept), `cap` on `TurnCapReached`, `error` on any other exception (traceback in `turn_ended.error`), else `no_tool`.

Participants never hold provider clients, locks, or other shared services: they are deep-copied per agent and pickled per round. Inference (M1b) goes through `tools.infer(request)`, which the runner owns: it applies the admission gate, logs `inference_attempt` before and `inference_response` after the call, charges the ledger, and serves the record/replay cache. `Persistable` skips `params` and every attribute whose name starts with `_`, so transient caches can be kept out of snapshots.

Scripted participants in M1a: `Silent` (round 1 guesses the candidate containing its crop, never posts or reads, `end_turn`), `EvidenceAggregator` (round 1 posts `"crop: <rows>"`; every round reads the board, pools crops, guesses the most consistent candidate, `end_turn`), `Enumerator` (cycles candidates, reads both status tools, raises if any tool result or view field contains `"correct"`, `"truth"`, or the truth name supplied out of band by the test).

## 11. Scheduler

```python
class Scheduler(Persistable, Plugin):
    def order(self, round, live, rng) -> list[AgentId]: ...
class SeededShuffle(Scheduler)   # shuffles live with ("schedule", round)
```
The commit policy is a run option read by the runner. In M1a the scheduler is always `SeededShuffle` and is not part of the run spec.

## 12. Runner

`Runner(run_dir, experiment, options)` executes `live()`, `replay()`, `resume()`, `fork(at_round, experiment=None)` exactly as specified below. `Experiment.run` and `Run` wrap it; nothing else calls it. `run.json` is written before `run_started` so a directory is always resumable or removable. `Experiment.run` works inside an already-running event loop (notebooks) by running the runner in a worker thread.

Phase-commit round `r`:
```
order = scheduler.order(r, live, derive(seed,"schedule",r)); log round_started(order)
for each agent concurrently (semaphore = options.concurrency):
    view = View(r, agent, world.observe(agent), outcomes_prev[agent], board.pushable(...) if push else [], executor.schemas(agent))
    run participant.turn(view, executor) with a per-agent event buffer; executor buffers posts and actions
append each agent's buffered events in `order`
posts, deliveries = board.commit(r, agents, derive(seed,"topology",r), blobs); log post*, delivery*
outcomes = world.commit(actions in `order`, each agent's in call order); log action_committed*; outcomes_prev = by agent
fold metrics on this round's logical events; log metric*
if r % snapshot_every == 0: write snapshot (so every committed round has one on disk); log round_committed; log snapshot
```
Sequential (`immediate`): agents run one at a time in `order`; every tool call applies immediately through single-item `board.commit` / `world.commit`; `read_board` sees deliveries with `eligible_round <= r` including same-round posts.

`recover()`: `last_committed()`; `truncate_after(seq)`; load the latest snapshot with `round <= committed`; if older than the last commit, replay logical events between them into plugins (M1a may require `snapshot_every == 1` and document it). Operational events from the discarded round go to `discarded.jsonl`.

`fork(at_round, experiment)`: new run dir, `run_started` with `parent_run` and `fork_round`, copy snapshot `at_round` and the log prefix up to that round's `round_committed`, continue live under the (possibly new) experiment. Plugin state is restored from the snapshot when the plugin matches: the world when its *type* matches (worlds keep their own constructor settings across restore, so a fork may change e.g. `guess_limit`), participants when their full `spec()` matches, metrics when present in the parent's list; what was restored is recorded in `run.json["restored"]`.

## 13. Snapshot

```python
class SnapshotManifest(BaseModel):
    run: RunId; round: int; log_seq: int
    plugins: dict[str, str]       # "world" | "board" | "scheduler" | f"participant:{agent}" | f"metric:{name}" -> blob sha
    outcomes_prev: dict[AgentId, list[dict]]; live: list[AgentId]
```
`SnapshotStore(run_dir)`: `write(manifest, blobs) -> path`, `latest(max_round=None)`, `load(manifest) -> dict[str, bytes]`. Manifests at `snapshots/<round:06d>.json`. Plugins never see the manifest; they only implement `snapshot()`/`restore()`, which `Persistable` provides by default.

## 14. Metrics

`metrics/base.py` as shipped:
```python
class Metric(Persistable, Plugin):
    name: str
    def update(self, event: Event) -> None: ...                  # required
    def value(self) -> tuple[float | None, int]: ...             # (value, denominator); required
    def needs_truth(self) -> bool: return False                  # runner injects world.verify() via set_truth
    def set_truth(self, truth: dict) -> None: ...
registry: get(name) -> Metric via entry points; Metric.from_fn(name, fn) for one-liners
```
M1a metrics: `belief.accuracy` (needs truth), `belief.consensus`, `belief.polarization(threshold=0.2)`, `belief.entropy`, `comm.read_rate`, `comm.posts_per_round`, `comm.hops`. Denominators: **all live agents** for belief metrics, with agents that have no committed guess counted as an explicit `none` category (so `belief.accuracy` equals the world's `score()["accuracy"]`, and agents that never guess lower consensus rather than raise it); turns for comm metrics.

## 15. Run directory

```
runs/<run_id>/  run.json  events.jsonl  discarded.jsonl  blobs/<sha>  snapshots/<round>.json  artifacts/{spec.yaml,git.txt}   (lock.txt deferred to M4 publish)  view.html
```

## 16. CLI (M1a subset)

```
swarmlab validate spec.yaml
swarmlab run spec.yaml --arm A --seed 1 [--max-rounds N] [--out runs/]
swarmlab replay RUN_DIR | resume RUN_DIR | fork RUN_DIR --at R [--spec edited.yaml] | view RUN_DIR
```
`--json` on every command; exit 0 success, 2 validation error, 1 otherwise. The CLI builds an `Experiment` and calls the same `run`/`Run` API as Python.

Setup additions (after M1b; additive):
```
swarmlab doctor [SPEC...] [--offline]            # Python, extras, keys, provider reachability, git; exit 1 on a failed check
swarmlab models [--provider hf|anthropic] [--tools] [--search S] [--refresh]   # model catalog with USD per M tokens
swarmlab init [NAME] [--dir D] [--force]         # starter NAME.yaml + NAME.py (FlagGame, broadcast vs gossip, fake LLM agents)
swarmlab run spec.yaml [--arm A] [--seed N] [--max-rounds N] [--out runs/] [--yes] [--rerun]
```
`run` without `--arm`/`--seed` runs every arm x every seed of the YAML's optional top-level `seeds:` (default `[0]`), sequentially into `runs/<run_id>`. It prints the per-arm and total worst-case estimate first, asks for confirmation when any budget is non-zero unless `--yes` (declining exits 1 before anything runs), skips a run whose directory holds the same `spec_hash` ("exists, skipping"; `--rerun` writes `runs/<run_id>__r<N>` instead), and ends with a table of run id, outcome, end reason, score and spend. A directory holding a different `spec_hash` fails that run (exit 1) unless `--rerun`. With one arm and `--seed` the output is the single-run summary as before. Python: `Experiment.arms_from_yaml(path)`, `Experiment.run_all(seeds, max_rounds, out=...) -> list[Run]`, `Experiment.estimate(..., seeds=[...])` (adds `runs`, `total_usd`), `Run.summary()`, `Run.load(dir)` or `Run.load("runs", run_id)`.

## 17. Viewer (minimal)

`viewer.build(run_dir) -> view.html`: self-contained page with a round slider; per round the Flag Game candidate grids and each agent's current guess, the board as committed, each agent's inbox with delivered content and read marks, and the per-agent event list. Built from the log and snapshots only. Vanilla JS, no external requests.

## 18. Acceptance tests

1. **Determinism**: two `exp.run(seed=…)` of the same experiment into different dirs produce identical `logical_view` sequences and identical `score`.
2. **Recovery**: run to round 6; SIGKILL during round 7 after at least one turn; `Run.load(dir).resume()`; logical view and score equal an uninterrupted run; `discarded.jsonl` holds the partial round's operational events.
3. **Fork**: `run.fork(at_round=4).run()` reproduces rounds 1 to 4 of the parent exactly in the logical view, the snapshot at 4 is byte-identical, rounds after 4 run live.
4. **API**: the three examples in DESIGN.md's Experimenter interface run as written (with `Counter` and `OddAgentsSeeNothing` as test fixtures) and round-trip through `to_spec`/`from_spec`.

Unit tests must cover: `rng.derive` independence, spec hashing and YAML round-trip, board commit with each topology and the delay policy, FlagGame rival and crop generation, executor pending semantics, log truncation, `Persistable` round-trip, the `@tool` decorator and default `commit`, and the Enumerator never seeing correctness in a 20-round run.

## 19. Work packages

- **WP1 core**: `spec.py, events.py, blobs.py, snapshot.py` plus unit tests (incl. `rng`, `ids`, `base`).
- **WP2 world**: `world/flaggame.py` plus tests; may add helpers to `world/base.py` without changing shipped names.
- **WP3 medium**: `medium/board.py, medium/topology.py` plus tests.
- **WP4 runner and API**: `scheduler.py, executor.py, runner.py, experiment.py, participants/scripted.py, metrics/belief.py, metrics/comm.py` plus the four acceptance tests.
- **WP5 surface**: `cli.py, viewer/build.py`, README, `examples/flaggame_m1a.yaml`, `examples/*.py` mirroring the three DESIGN examples.

WP1 to WP3 start together against this document and the shipped base files. WP4 starts when WP1 lands and stubs WP2 and WP3 until they land. WP5 after WP4.
