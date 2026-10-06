# WP4 handoff: runner, executor, scheduler, Experiment/Run API, scripted participants, metrics

Branch `wp4`. `pytest -q -W error` (171 tests, about 20 s) and `ruff check swarmlab tests` are clean.

## What landed

| file | public names |
|---|---|
| `swarmlab/scheduler.py` | `Scheduler`, `SeededShuffle` ("seeded_shuffle") |
| `swarmlab/executor.py` | `RoundExecutor`, `Committed`, `board_schemas`, `END_TURN_SCHEMA` |
| `swarmlab/runner.py` | `Runner`, `ForkOrigin`, `ReplayMismatch`, `RecoveryError` |
| `swarmlab/experiment.py` | `Experiment`, `Run`, `ForkHandle` |
| `swarmlab/registry.py` | `resolve(type, group)`, `build(spec, group)` (entry point, then `module:Class`) |
| `swarmlab/participants/scripted.py` | `Silent`, `EvidenceAggregator`, `Enumerator(truth_name=None)` |
| `swarmlab/metrics/belief.py` | `Accuracy`, `Consensus`, `Polarization(threshold=0.2)`, `Entropy` |
| `swarmlab/metrics/comm.py` | `ReadRate`, `PostsPerRound`, `Hops` |
| `swarmlab/worlds.py` | alias module so DESIGN's `from swarmlab.worlds import FlagGame` works |
| `swarmlab/participants/__init__.py` | re-exports the scripted participants (DESIGN's `from swarmlab.participants import EvidenceAggregator`) |
| `swarmlab/__init__.py` | now also exports `Experiment, Run, Board, DelayPolicy` |

Changes to shipped files, as the handoffs called for: `ids.run_id(experiment, arm: str | None, seed)`
gives `<experiment>__s<seed>` when `arm` is None; `events.TurnEndedEvent` gained
`error: str | None = None` (the traceback when `yield_kind == "error"`).

Tests: `test_executor, test_scheduler, test_metrics, test_scripted, test_immediate` plus the four
acceptance tests, all passing:
- `test_determinism.py`: two runs (round_end and immediate, gossip and a delay policy) have
  identical logical views, scores, metrics and byte-identical snapshots. Replay detects a tampered score.
- `test_recovery.py`: a subprocess runs 10 rounds; a test world (`tests/helpers.py:PausingFlagGame`)
  blocks inside round 7's world commit, after round 7's `turn_started` lines are in the log, and
  writes a marker file. The test SIGKILLs it, runs `Run.load(dir).resume()`, and compares the logical
  view, score and metrics with an uninterrupted run. Also covered: a crash in round 1, resume of an
  ended run (no-op), and the `snapshot_every != 1` refusal.
- `test_fork.py`: fork at 4 has a byte-identical log prefix and snapshot 4, the same logical rounds 1-4,
  its own `run_started`, and live rounds 5+ (which equal the parent's, since seed and experiment are
  the same). Also covers a fork under an edited experiment, and fork numbering (`_1`, `_2`).
- `test_api.py`: the three DESIGN examples (`Counter`, `OddAgentsSeeNothing` as module-level
  fixtures), `to_spec`/`from_spec` and YAML round-trips, and `[EvidenceAggregator()] * 16`.
  `run.view()` is asserted to raise `NotImplementedError("WP5")` until the viewer lands.

## Decisions where the contract was silent

- **Run directory.** `Experiment.run(out=...)` writes to `<out>/<run_id>`. It raises
  `FileExistsError` if that run's log already exists. `run.json` is rewritten atomically after every
  commit. It holds: `run_id, experiment, arm, spec, spec_hash, git_commit, dirty, parent_run, fork_round,
  restored, status ("running"|"ended"), end_reason, last_round, score`. Artifacts are
  `artifacts/spec.yaml` (RunSpec) and `artifacts/git.txt`. There is no `lock.txt` yet.
- **Ids.** Call ids are `c{round:04d}-{agent}-{n:03d}`. Action ids are `x{round:04d}-{agent}-{n:02d}`.
  Provisional post ids are `tmp-{agent}-{n}`. Real post ids are assigned by `board.buffer_post` at
  commit time, called in seeded order, so they never depend on async interleaving.
- **Event shapes.** The `tool_returned.result` field is `{"ok", "result", "error"}`.
  `delivery.agent` is the recipient. `post.agent` and `action_committed.agent` are the author.
  `metric.agent` is None. `snapshot.manifest_path` is relative to the run dir
  (`snapshots/000004.json`). `View.outcomes` items are `{"action_id", "tool", "accepted", "feedback"}`.
- **Order inside a round.** The order is `round_started`, then each agent's turn events in seeded
  order (`turn_started`, `tool_called`/`read`/`tool_returned`..., `turn_ended`), then `post*`,
  `delivery*`, `action_committed*`, `metric*`, `round_committed`, `snapshot`. Immediate mode logs the
  posts, deliveries and actions it already applied in the same place, in application order.
- **Snapshot before commit marker.** The manifest for round r is written before `round_committed`.
  Its `log_seq` is the seq that `round_committed` gets. So a committed round always has its snapshot
  on disk. Recovery re-appends the `snapshot` event if the crash lost it.
- **discarded.jsonl** receives every event dropped by recovery, not only operational ones. M1a has
  no operational events, and the partial round's logical events are useful for debugging.
- **Metric feed.** Metrics see every logical event of the round except `run_started`, `metric`,
  `round_committed`, `snapshot` and `run_ended`, in log order. Live mode parses the events back
  from JSON so live and replay feed identical objects.
- **Metric definitions.**
  - Belief metrics use the latest accepted `guess` per agent; the denominator is the number of agents with a guess.
  - Polarization is the number of beliefs held by at least `threshold` of the guessers (DESIGN §10).
  - Entropy is in bits.
  - Comm metrics are per round, with denominator = turns this round.
  - `read_rate` is the share of turns with at least one `read_board` call.
  - `posts_per_round` is the number of posts.
  - `hops` is the maximum propagation depth so far: a post's hop count is 1 + the maximum hop count
    of any post its author had read up to that round.
- **Push delivery.** `View.pushed` is `board.pushable(...)` rendered as read_board items. Pushing
  does not mark items read.
- **Scheduler** is always `SeededShuffle`; it is not in `RunSpec`.
- **Participant rng.** Every participant, scripted ones included, gets `derive(seed, "agent", agent)`.
  The `("scripted", agent)` label is unused.
- **Fork restore rules under an edited experiment.**
  - The board restores itself (WP3 rules).
  - The world is restored when its `type` equals the parent's, not its full spec. WP2's FlagGame
    keeps its constructor config across restore, so a fork can change e.g. `guess_limit` and keep the
    guesses. This deviates from "restore only on equal spec" on purpose.
  - Participants are restored only on an equal spec; otherwise they start fresh.
  - Metrics are restored when their spec is in the parent's metric list.
  - What was restored is recorded in `run.json["restored"]`, and replay uses it.
  - Changing the number of participants raises `ValueError`.
- **Fork log.** The child copies the parent's lines through `round_committed` and the following
  `snapshot` line (byte-identical; their `run` field still names the parent). It then appends its own
  `run_started` with `round = at_round`, `parent_run` and `fork_round`. It also copies the content
  blobs of every copied delivery, plus the snapshot's plugin blobs.
- **`Experiment` extras.** These are additive and not in the contract:
  - `arm: str | None` and `options: dict` (YAML run-option defaults).
  - `run(seed, max_rounds=None, ...)` falls back to `options` for every run option.
  - `ForkHandle.run(out=None, *, max_rounds=None)`.
  - `Run.replay()` and `Run.meta`.
  - `Run.events` yields `logical_view` dicts; `Run.events_all` yields typed events.
- **Budgets** are recorded on the spec only. `run_ended` reasons used are `terminal` and `max_rounds`.
  A runner crash (not a participant error) propagates and leaves the run resumable; it does not log
  `run_ended(error)`.

## For WP5 (CLI, viewer)

Build and run from YAML (the CLI should do exactly this):

```python
from swarmlab import Experiment, Run
exp = Experiment.from_yaml("examples/flaggame_m1a.yaml", arm="A")      # SpecError on bad YAML -> exit 2
run = exp.run(seed=1, max_rounds=args.max_rounds, out=args.out)       # max_rounds None -> YAML options
Run.load(run_dir)            # replay; raises swarmlab.runner.ReplayMismatch on divergence
Run(run_dir).resume()        # resume
Run(run_dir).fork(R, experiment=Experiment.from_yaml(edited, arm) if edited else None).run(out=None)
```

`swarmlab validate spec.yaml` can call `swarmlab.spec.load_experiment_yaml`. It can then call
`Experiment.from_yaml` for each arm, which resolves every plugin.

The viewer is wired as `Run.view()`: `from swarmlab.viewer.build import build; build(run_dir) -> Path`.
Until `swarmlab/viewer/build.py` exists, it raises `NotImplementedError("WP5")`; update the
`test_example_1` assertion when it lands.

Where the viewer finds state per round r:
- **Log** (`events.jsonl`, `swarmlab.events.EventLog`): events with `round == r`. This includes
  `round_started.order`, the per-agent turn events (tool calls, results, `read.delivery_ids`),
  `post` (text), `delivery` (recipient, `eligible_round`, `content_hash` -> `blobs/<sha[:2]>/<sha>`
  via `BlobStore.get`), `action_committed` (guesses), and `metric`.
- **Snapshots** (`SnapshotStore(run_dir).read(r)`, one per round when `snapshot_every == 1`):
  `load(manifest)["world"]` is FlagGame state. Restore it into `FlagGame()`, or `pickle.loads` it for
  the dict with `candidates, truth, guesses, crops`. `["board"]` is the board state; restore it into
  `Board()` and call `board.inbox(agent)` for deliveries with `read_round` marks. Participant and
  metric blobs are keyed `participant:<agent>` and `metric:<name>`.
- For a fork, rounds up to `fork_round` (in `run.json`) are the parent's copied events; their `run`
  field names the parent.
- `run.json` has the final score and the status.

## Post-review changes (2026-10-06, `docs/notes/m1a-review-2026-10-06.md`)

One commit per finding on branch `fix-m1a-review`.

- **A2 identity.** `Plugin.type_name()` honours `entry_point` only from `cls.__dict__`. Subclasses of
  registered plugins serialise as `module:Class`, and `tests/helpers.py` no longer sets
  `entry_point = None`. `Persistable.snapshot()` skips `params` and every `_`-prefixed attribute.
- **B1 AgentTools.** `participant.turn(view, tools)` now gets `swarmlab.tools.AgentTools(executor,
  agent)`, which exposes `agent`, `schemas()` and `call(name, args)` only. The executor is held in a
  name-mangled slot; the handle has no `__dict__`. Participants call `tools.call("guess", {...})`.
  The `RoundExecutor` API (`ex.call(agent, ...)`) is unchanged and runner-internal.
- **A4 end_turn.** `end_turn` sets a per-agent flag (`RoundExecutor.turn_ended(agent)`) and returns
  `ok=True`. Later calls are logged and return `ok=False, error="turn_ended"`. The cap is checked
  first, so a loop after `end_turn` still ends as `cap`. Yield kinds:
  - `cap` on `TurnCapReached`.
  - `error` on any other exception. An exception wins over an earlier `end_turn`.
  - On a normal return, `end_turn` if the flag is set, else `no_tool`. The returned `TurnUsage` is
    kept in both cases.
  `EndTurn` is still defined but is never raised. The scripted participants now return real
  `TurnUsage(calls=...)`.
- **A5 private.** The runner observes first, logs `turn_started.private`, and gives the participant
  a copy of the observation with `private={}`. The viewer reads crops from the world snapshot, so
  it needed no change.
- **A1 FlagGame.** Candidates are `n_candidates // 2` twin pairs (`rival_edits`, default 1, whole
  bands recoloured). `n_candidates` must be even and `palette >= 3`. See WP2.md.
  - The crop-free heuristic now scores about chance (`test_no_crop_free_shortcut`).
  - With 16 pooled crops the truth is unique in about 95% of seeds (it was about 90% before). In
    the other seeds a candidate from another pair contains every crop, because crops carry no
    position. Seed 11 of `flaggame_m1a.yaml` is such a seed: broadcast accuracy there is 0.625,
    not 1.0.
- **B2 belief.** `Metric.set_agents(agents)` is a no-op by default. The runner calls it after reset
  and restore, and at round start whenever `live_agents` changed. Replay calls it at start and
  whenever `round_started.order` changes the live set. The belief denominator is all live agents,
  with `"none"` for agents that have not guessed. `belief.accuracy` equals `score()["accuracy"]`
  every round.
- **B3 operational events.** `Runner.log_operational(event)` appends `inference_attempt` and
  `inference_response` straight to the log, bypassing the per-agent buffer. It raises for any
  other type. M1b's `tools.infer` should call it.
- **B4 post ids.** `post.provisional_id` is the id the author's ack returned: `tmp-<agent>-<n>`
  under round_end, equal to `post_id` under immediate. The executor keeps
  `buffered_post_ids(agent)` parallel to `buffered_posts(agent)`. The viewer shows
  `(ack tmp-...)` next to the post id.
- **D2 run.json first.** `live()` now writes `run.json`, then `run_started`. On `resume()` of a
  directory whose log has no `run_started`, the runner moves any stray events to
  `discarded.jsonl`, appends `run_started` and starts from round 1.
- **C2.**
  - Inside a running event loop, `live()` and `resume()` run the round loop with `asyncio.run` in
    a worker thread and block until it finishes.
  - Re-running into a directory that holds a different `spec_hash` raises a `FileExistsError` that
    names both hashes.
- **A6 comm.hops.** A read feeds a post only if it came before the post call. The cut is the
  author's read count at the `post` `tool_called`, linked through the ack id to
  `post.provisional_id`. A read also counts only if `eligible_round <= read round`. The value is
  the running max, and the denominator is the number of posts read at least once.
  - The reviewer's post-then-read case (3 agents, broadcast, round_end) gives `[1, 1, 2, 2, 3]`,
    not the `[1, 1, 2, 3, 4]` in the review. A round-r post reflects only reads from rounds before
    r, and it is first readable in round r+1, so each hop costs two rounds. The test docstring
    walks through the chain.
  - Read-then-post gives `[1, 2, 3, 4, 5]`.
