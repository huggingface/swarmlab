# swarmlab M1a interface contract

Scope: the first working slice from DESIGN.md. Scripted participants, phase-commit runner with tool executor, event log with commit markers and per-round snapshots, board with inboxes, FlagGame text variant, replay, resume, fork, minimal viewer. No model is called in M1a. Everything here is binding for implementers; anything not here is the implementer's choice and must be documented in the module docstring.

Conventions: Python 3.12, `pydantic` v2 models for all data crossing a boundary, `typing.Protocol` for plugin interfaces, `asyncio` in the runner. Rounds are 1-based; round 0 is the initial state. All randomness comes from `swarmlab.rng`, never from the global `random`.

## 1. Package layout

```
swarmlab/
  __init__.py
  ids.py           AgentId, RunId, PostId, ActionId, CallId, DeliveryId (NewType str)
  rng.py           derive(seed, *labels) -> random.Random
  spec.py          Experiment, Arm, RunSpec, resolve(), spec_hash()
  events.py        Event union, EventLog
  blobs.py         BlobStore (content-addressed)
  tools.py         ToolSchema, ToolCall, ToolResult, ToolExecutor protocol
  view.py          View, Observation, Part
  world/base.py    World protocol, Action, Ack, Outcome
  world/flaggame.py
  medium/board.py  Board, Post, Delivery, Topology, VisibilityPolicy
  medium/topology.py  broadcast, gossip, groups
  participants/base.py  Participant protocol, TurnUsage
  participants/scripted.py  Silent, EvidenceAggregator, Enumerator
  scheduler.py     Scheduler protocol, SeededShuffle
  executor.py      RoundExecutor (the ToolExecutor implementation)
  runner.py        Runner: live, replay, resume, fork
  snapshot.py      SnapshotManifest, SnapshotStore
  metrics/base.py  Metric protocol, registry
  metrics/belief.py, metrics/comm.py
  viewer/build.py  run dir -> single html
  cli.py
tests/
  test_determinism.py  test_recovery.py  test_fork.py  + unit tests per module
```

## 2. Identifiers and randomness

- `AgentId` is `a` + zero-padded index (`a000`). `RunId` is `<experiment>__<arm>__s<seed>`; a fork appends `__f<round>_<n>`.
- `rng.derive(seed: int, *labels: str) -> random.Random` hashes `(seed, labels)` with SHA-256 to seed a `random.Random`. Fixed label roots: `("world",)`, `("private", agent)`, `("schedule", round)`, `("topology", round)`, `("agent", agent)`, `("scripted", agent)`. Changing one component's draws must not change another's.

## 3. Spec

```python
class Budget(BaseModel):
    soft_usd: float = 0.0        # M1a: unused, carried for schema stability
    hard_usd: float = 0.0
    measurement_usd: float = 0.0

class WorldSpec(BaseModel):   type: str; params: dict = {}
class MediumSpec(BaseModel):
    topology: str = "broadcast"          # entry-point name
    topology_params: dict = {}
    delivery: Literal["pull", "push"] = "pull"
    push_limit: int = 20                 # items pushed into the view when delivery == "push"
    policies: list[dict] = []            # visibility policies, applied in order
    channels: list[str] = ["main"]
class ParticipantGroup(BaseModel):
    count: int; type: str; role: str = "worker"; model: str | None = None; params: dict = {}
class SchedulerSpec(BaseModel):
    commit: Literal["round_end", "immediate"] = "round_end"
    max_rounds: int
    max_calls_per_turn: int = 20
    snapshot_every: int = 1
class Arm(BaseModel):
    world: WorldSpec; medium: MediumSpec; participants: list[ParticipantGroup]
    scheduler: SchedulerSpec; metrics: list[str] = []
    probes: list[dict] = []; interventions: list[dict] = []   # M1b
class Experiment(BaseModel):
    name: str; seeds: list[int]; arms: dict[str, Arm]; budget: Budget = Budget()
class RunSpec(BaseModel):     # fully resolved, one arm × one seed
    experiment: str; arm: str; seed: int; arm_spec: Arm; budget: Budget
```

`resolve(experiment: Experiment, arm: str, seed: int) -> RunSpec`. `spec_hash(run_spec) -> str` is SHA-256 of the canonical JSON (sorted keys, no whitespace). Run identity is `(git_commit, dirty, spec_hash)`; the runner records all three on `RunStarted` and archives the raw YAML to `artifacts/spec.yaml`.

Participant indices are assigned in group order, so group 1's agents are `a000..`, group 2 continues.

## 4. Events

Every event has `seq: int` (dense, 0-based, assigned by the log on append), `run: RunId`, `round: int`, `agent: AgentId | None`, `ts: float` (wall clock, informational), `type: str`. Payload fields per type:

| type | fields | when |
|---|---|---|
| `run_started` | `spec_hash, git_commit, dirty, seed, run_spec, parent_run, fork_round` | once |
| `round_started` | `order: list[AgentId]` | per round |
| `turn_started` | | per agent turn |
| `tool_called` | `call_id, tool, args` | each tool call inside a turn |
| `tool_returned` | `call_id, result, pending: bool` | each tool return |
| `inference_attempt` | `call_id, provider, model, request_hash, reserved_usd` | M1b; operational |
| `inference_response` | `call_id, response_hash, usage, cost_usd, latency_s` | M1b; operational |
| `turn_ended` | `yield_kind: Literal["no_tool","end_turn","cap","error"], calls: int, usage` | per agent turn |
| `read` | `delivery_ids: list[DeliveryId]` | each board read, inside the turn |
| `post` | `post_id, channel, text, fields` | at commit, per accepted post |
| `delivery` | `post_id, recipient, delivery_id, eligible_round, content_hash` | at commit, per fan-out |
| `action_committed` | `action_id, action, accepted: bool, feedback: dict` | at commit, per buffered action |
| `world_changed` | `payload: dict` (world-defined, opaque) | at commit, optional |
| `metric` | `name, value, denominator` | after fold, per metric |
| `round_committed` | `n_posts, n_actions, n_deliveries` | per round, last event of the round |
| `snapshot` | `manifest_path` | after `round_committed`, every `snapshot_every` |
| `run_ended` | `reason: Literal["terminal","max_rounds","soft_budget","hard_ceiling","error"]` | once |

**Logical versus operational.** `inference_attempt`, `inference_response`, and the `ts` field are operational. `logical_view(events)` strips them. Acceptance tests compare logical views.

**Order rule for determinism.** Turns run concurrently in live mode, but their events are buffered per agent and appended to the log at commit in the round's seeded order, agent by agent, each agent's events in call order. Then `post`, `delivery`, `action_committed`, `world_changed`, `metric`, `round_committed`, `snapshot`. Operational events may be appended as they happen.

`EventLog(path)` offers `append(event) -> seq`, `__iter__`, `last_committed() -> (round, seq) | None`, `truncate_after(seq)`. Storage is one JSONL file, one event per line, fsync on `round_committed`.

## 5. Blobs

`BlobStore(dir)`: `put(bytes) -> sha256`, `get(sha) -> bytes`. Delivered content, inference requests and responses, and snapshot plugin state live here. Events reference blobs by hash.

## 6. Tools

```python
class ToolSchema(BaseModel): name: str; description: str; parameters: dict  # JSON schema
class ToolCall(BaseModel):   call_id: CallId; name: str; args: dict
class ToolResult(BaseModel): call_id: CallId; ok: bool; result: dict; pending: bool = False; error: str | None = None

class ToolExecutor(Protocol):
    def schemas(self, agent: AgentId) -> list[ToolSchema]: ...
    async def call(self, agent: AgentId, name: str, args: dict) -> ToolResult: ...
```

The executor is the only mutation path. Tool namespaces: world tools come from `World.tools(agent)`; board tools are `read_board(channel?, limit?)`, `post(channel, text, fields?)`; registry tools are M3; `end_turn()` is always present. A call outside the agent's allowlist returns `ok=False, error="not_allowed"` and is logged.

Under `commit == "round_end"`: `post` and world actions return `pending=True` with `{"id": ...}`; `read_board` returns eligible undelivered inbox items for this round (items whose `eligible_round <= round`), marks them read, and logs a `read` event; status tools answer from round-start state plus the agent's own buffered actions where the world supports it. Under `commit == "immediate"`: every call applies at once and returns the real outcome with `pending=False`.

## 7. View

```python
class Part(BaseModel):  type: Literal["text", "image"]; text: str | None = None; image_png_b64: str | None = None
class Observation(BaseModel): parts: list[Part]; private: dict = {}   # `private` is world-defined, never shared
class View(BaseModel):
    round: int; agent: AgentId
    observation: Observation
    outcomes: list[dict]            # this agent's action_committed feedback from the previous round
    pushed: list[dict]              # delivered inbox items when delivery == "push", else []
    tools: list[ToolSchema]
```

## 8. World

```python
class Action(BaseModel): name: str; args: dict
class Ack(BaseModel):    ok: bool; error: str | None = None          # pre-commit sanity only
class Outcome(BaseModel): action_id: ActionId; accepted: bool; feedback: dict  # agent-visible; never correctness

class World(Protocol):
    name: str
    def reset(self, rng: random.Random, params: dict, agents: list[AgentId]) -> None: ...
    def tools(self, agent: AgentId) -> list[ToolSchema]: ...        # world actions + enabled status tools
    def observe(self, agent: AgentId) -> Observation: ...
    def validate(self, agent: AgentId, action: Action) -> Ack: ...  # no mutation
    def commit(self, actions: list[tuple[AgentId, ActionId, Action]]) -> list[Outcome]: ...  # already in seeded order
    def my_status(self, agent: AgentId) -> dict: ...
    def collective_status(self) -> dict: ...
    def score(self) -> dict: ...          # evaluator-only
    def terminal(self) -> bool: ...
    def verify(self) -> dict: ...         # hidden truth, evaluator-only
    def snapshot(self) -> bytes: ...
    def restore(self, blob: bytes) -> None: ...
```

Status tools are exposed only if `params.status_tools` lists them (`["my_status", "collective_status"]` by default). Nothing returned by `tools`, `observe`, `validate`, `commit`, `my_status`, or `collective_status` may reveal correctness; `score` and `verify` are called only by the runner.

### FlagGame (text variant)

Params with defaults: `height=8, width=12, palette=6, n_candidates=8, rival_similarity=0.85, crop_h=3, crop_w=4, candidate_names="letters"` (`A..H`), `status_tools=["my_status","collective_status"]`, `guess_limit=None`.

- `reset` draws the truth flag from `("world",)` as a structured colour grid (horizontal or vertical stripes, or blocks), the rival by copying the truth and recolouring a `1 - rival_similarity` fraction of cells, and distractors as fresh structured flags. Candidates are shuffled and named; the mapping is hidden state. Each agent's crop is a random `crop_h × crop_w` window from `("private", agent)`.
- `observe` returns one text part: the candidate list (name plus full grid rendered as rows of colour letters) and the agent's crop rendered the same way with its position withheld. The `private` dict holds the crop coordinates for the evaluator.
- Tools: `guess(candidate: str)`. `validate` checks the name exists and the guess limit. `commit` records the latest guess per agent; `Outcome.feedback` is `{"recorded": true}` only.
- `my_status` → `{"current_guess": str | None, "guesses_made": int}`. `collective_status` → `{"guess_counts": {name: int}, "agents_with_guess": int}`.
- `terminal` is always False; the run ends on `max_rounds`.
- `score` → `{"accuracy": share of live agents whose latest guess is the truth, "n_guessed": int, "truth": name}`. `verify` → `{"truth": name, "rival": name, "candidates": ..., "crops": {agent: (y, x)}}`.

## 9. Medium: board, inboxes, topology, policies

```python
class Post(BaseModel):     post_id: PostId; round: int; agent: AgentId; channel: str; text: str; fields: dict = {}
class Delivery(BaseModel): delivery_id: DeliveryId; post_id: PostId; recipient: AgentId; eligible_round: int; content_hash: str; read_round: int | None = None

class Topology(Protocol):
    def recipients(self, post: Post, agents: list[AgentId], round: int, rng: random.Random) -> list[AgentId]: ...
class VisibilityPolicy(Protocol):
    def apply(self, reader: AgentId, post: Post, round: int) -> tuple[int, str] | None: ...
    # returns (eligible_round, content) or None to withhold

class Board:
    def buffer_post(self, agent, round, channel, text, fields) -> PostId
    def commit(self, round, agents, topology, policies, rng) -> tuple[list[Post], list[Delivery]]
    def read(self, agent, round, channel=None, limit=50) -> list[Delivery]   # eligible, unread; marks read_round
    def pushable(self, agent, round, limit) -> list[Delivery]             # same set, for push delivery
    def snapshot(self) -> bytes ; def restore(self, blob) -> None
```

`commit` turns buffered posts into `Post` records in seeded order, computes recipients per topology (the author is never a recipient), applies policies in order (the first policy returning `None` withholds; otherwise the last transformed content and the maximum eligible round win), stores content in the blob store, and creates one `Delivery` per recipient. The default eligible round is `round + 1`. Deliveries are the medium's inboxes; a global cursor does not exist.

Topologies in M1a: `broadcast` (all agents), `gossip(k=1)` (per round, each agent's posts go to `k` partners drawn from `("topology", round)`; the schedule is a pure function of the seed), `groups(size)` (fixed groups by index). Policy in M1a: `delay(rounds, readers=None)` sets `eligible_round = round + 1 + rounds`. Policies are entry points so M1b can add transforms.

## 10. Participants

```python
class TurnUsage(BaseModel): calls: int = 0; prompt_tokens: int = 0; completion_tokens: int = 0; cost_usd: float = 0.0
class Participant(Protocol):
    def bind(self, agent: AgentId, rng: random.Random, params: dict) -> None: ...
    async def turn(self, view: View, tools: ToolExecutor) -> TurnUsage: ...
    def snapshot(self) -> bytes: ...
    def restore(self, blob: bytes) -> None: ...
```

A turn ends when `turn` returns or when the executor raises `TurnCapReached` after `max_calls_per_turn` tool calls or `EndTurn` after `end_turn()`. The runner records `yield_kind` as `end_turn` if `end_turn()` was called, `cap` on the cap, `error` on an exception, else `no_tool`.

Scripted participants in M1a, all in `participants/scripted.py`:
- `Silent`: round 1 guesses the candidate whose grid contains its crop (ties broken by rng); never posts or reads; `end_turn()`.
- `EvidenceAggregator`: round 1 posts `"crop: <rows>"`; every round reads the board, collects all crops seen, guesses the candidate consistent with the most crops (ties by rng); `end_turn()`.
- `Enumerator`: cycles through candidates one guess per round, reads `my_status` and `collective_status` every round, and raises if any tool result or view field ever contains the strings `"correct"`, `"truth"`, or the truth name supplied out of band by the test. Used only by the isolation test.

## 11. Scheduler

```python
class Scheduler(Protocol):
    def order(self, round: int, live: list[AgentId], rng: random.Random) -> list[AgentId]: ...
```
`SeededShuffle` shuffles `live` with `("schedule", round)`. The commit policy lives on the runner, read from `SchedulerSpec.commit`.

## 12. Runner

`Runner(run_dir, run_spec)` builds plugins from entry points, then executes one of:

- `live()`: rounds until `terminal`, `max_rounds`, or budget. 
- `replay()`: iterates the log, feeds every event to metrics and the viewer builder, calls no plugin method that mutates; verifies that the final `score` recomputed from snapshots equals the recorded one.
- `resume()`: `recover()` then `live()`.
- `fork(at_round, new_run_spec=None)`: new run dir, `run_started` with `parent_run` and `fork_round`, copy snapshot `at_round` and the log prefix up to that round's `round_committed`, then `live()` under the new spec.

Phase-commit round `r`:

```
order = scheduler.order(r, live, derive(seed, "schedule", r));  log round_started(order)
for each agent concurrently (bounded by a semaphore from params.concurrency, default 32):
    view = View(r, agent, world.observe(agent), outcomes_prev[agent], board.pushable(...) if push else [], executor.schemas(agent))
    run participant.turn(view, executor) with per-agent event buffer; executor buffers posts and actions
append each agent's buffered events in `order`
posts, deliveries = board.commit(r, agents, topology, policies, derive(seed, "topology", r)); log post*, delivery*
outcomes = world.commit(actions in `order`, each agent's actions in call order); log action_committed*; outcomes_prev = by agent
log world_changed? ; fold metrics; log metric*
log round_committed; if r % snapshot_every == 0: write snapshot; log snapshot
```

Sequential (`immediate`) round: same, but agents run one at a time in `order`, every tool call applies immediately through `board.commit`/`world.commit` of a single-item list, and `read_board` sees deliveries whose `eligible_round <= r` including same-round posts (default policy under `immediate` sets `eligible_round = round`).

`recover()`: find `last_committed()`; `truncate_after(seq)`; load the latest snapshot with `round <= committed round`; if the snapshot is older than the last commit, replay the logical events between them into plugins (M1a may assert `snapshot_every == 1` and skip this path, documenting it). Operational events from the discarded round stay in a sidecar `discarded.jsonl` so spend is retained for accounting.

## 13. Snapshot

```python
class SnapshotManifest(BaseModel):
    run: RunId; round: int; log_seq: int   # seq of the round_committed event
    plugins: dict[str, str]                # "world" | "board" | f"participant:{agent}" | "scheduler" | "metrics" -> blob sha
    outcomes_prev: dict[AgentId, list[dict]]
    live: list[AgentId]
```
`SnapshotStore(run_dir)`: `write(manifest, blobs) -> path`, `latest(max_round=None) -> SnapshotManifest | None`, `load(manifest) -> dict[str, bytes]`. Manifests at `snapshots/<round:06d>.json`.

## 14. Metrics

```python
class Metric(Protocol):
    name: str
    def update(self, event: Event) -> None: ...
    def value(self) -> tuple[float | None, int]:  ...   # (value, denominator)
    def snapshot(self) -> bytes ; def restore(self, blob) -> None
```
M1a metrics, fed only logical events: `belief.accuracy` (needs truth; the runner injects `verify()` into belief metrics at reset, never into events agents can see), `belief.consensus`, `belief.polarization(threshold=0.2)`, `belief.entropy`, `comm.read_rate` (share of turns with at least one `read`), `comm.posts_per_round`, `comm.hops` (per post: rounds from commit to first read, averaged). Denominator is live agents with a committed guess for belief metrics, or turns for comm metrics.

## 15. Run directory

```
runs/<run_id>/
  run.json          RunSpec + identity + status (updated at each commit)
  events.jsonl      the log
  discarded.jsonl   operational events from recovered rounds
  blobs/<sha>
  snapshots/<round>.json
  artifacts/spec.yaml, git.txt, lock.txt
  view.html         produced by `viewer.build`
```

## 16. CLI (M1a subset)

```
swarmlab validate spec.yaml
swarmlab run spec.yaml --arm A --seed 1 [--out runs/]
swarmlab replay RUN_DIR
swarmlab resume RUN_DIR
swarmlab fork RUN_DIR --at R [--spec edited.yaml] [--out runs/]
swarmlab view RUN_DIR
```
All commands accept `--json` and print a single JSON object on stdout; exit code 0 on success, 2 on validation error, 1 otherwise.

## 17. Viewer (minimal)

`viewer.build(run_dir) -> view.html`: a single self-contained page with a round slider; per round, the Flag Game candidate grids and each agent's current guess, the board as committed that round, each agent's inbox with delivered content and read marks, and the per-agent event list for the round. Built from the log and snapshots only. Vanilla JS, no external requests.

## 18. Acceptance tests

1. **Determinism**: two `live()` runs of the same `RunSpec` into different dirs produce identical `logical_view` sequences and identical final `score()`.
2. **Recovery**: run to round 6; kill the process (SIGKILL from the test harness) during round 7 after at least one agent's turn; `resume()`; the logical view and final score equal an uninterrupted run; `discarded.jsonl` holds the partial round's operational events.
3. **Fork**: `fork(at_round=4)` with the same spec reproduces rounds 1 to 4 of the parent exactly in the logical view and the snapshot at 4 is byte-identical to the parent's; rounds after 4 run live.

Unit tests must cover: `rng.derive` independence, spec hashing, board commit with each topology and the delay policy, FlagGame rival and crop generation, the executor's pending semantics, log truncation, and the Enumerator never seeing correctness in a 20-round run.

## 19. Work packages

- **WP1 core**: `ids, rng, spec, events, blobs, snapshot` plus their unit tests.
- **WP2 world**: `world/base, world/flaggame` plus tests.
- **WP3 medium**: `medium/board, medium/topology`, policies, plus tests.
- **WP4 runner**: `tools, view, executor, scheduler, runner, participants/base, participants/scripted, metrics/*` plus the three acceptance tests.
- **WP5 surface**: `cli, viewer/build`, README, example spec `examples/flaggame_m1a.yaml`.

WP1 to WP3 are independent and start together against this document. WP4 starts when WP1 lands and stubs WP2 and WP3 until they land. WP5 starts after WP4.
