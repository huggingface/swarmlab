# WP1 handoff: core (spec, events, blobs, snapshot)

Branch `wp1`. 58 tests in `tests/` (`test_rng, test_ids, test_base, test_spec, test_events,
test_blobs, test_snapshot`); `pytest -q -W error` and `ruff check swarmlab tests` are clean.

## What landed

| file | public names |
|---|---|
| `swarmlab/spec.py` | `Budget, PluginSpec, MediumSpec, RunOptions, RunSpec, SpecError, spec_hash, canonical_json, load_experiment_yaml, validate_experiment_doc, arm_to_runspec, runspec_to_doc, dump_experiment_yaml, dump_runspec_yaml, load_runspec_yaml, git_identity` |
| `swarmlab/events.py` | `Event` + 17 subclasses (`RunStartedEvent ... RunEndedEvent`), `AnyEvent`, `EVENT_CLASSES`, `OPERATIONAL_TYPES`, `parse_event, logical_view, EventLog, EventLogCorrupt, write_jsonl` |
| `swarmlab/blobs.py` | `BlobStore, sha256_hex` |
| `swarmlab/snapshot.py` | `SnapshotManifest, SnapshotStore` |
| `swarmlab/_io.py` | `atomic_write_bytes, fsync_dir` (internal helper) |
| `swarmlab/__init__.py` | now also exports `Budget` |

## Signatures WP4 will call

```python
# spec.py
spec_hash(run_spec: RunSpec) -> str                         # sha256 hex
load_experiment_yaml(path) -> dict                          # normalised doc; raises SpecError
arm_to_runspec(doc: dict, arm: str, seed: int, **option_overrides) -> RunSpec
                                                            # None-valued overrides ignored
runspec_to_doc(run_spec: RunSpec, arm: str | None = None) -> dict   # one-arm doc, counts collapsed
dump_experiment_yaml(doc: dict, path) -> Path               # for Experiment.to_yaml
dump_runspec_yaml(run_spec, path) -> Path                   # artifacts/spec.yaml
load_runspec_yaml(path) -> RunSpec
git_identity(repo_dir=".") -> tuple[str, bool]              # ("unknown", True) outside a repo

# events.py
EventLog(path)
  .append(event) -> int            # assigns seq (0-based, dense), sets event.seq too
  .extend(events) -> list[int]
  .__iter__() -> Iterator[Event]   # typed; stops at a torn tail
  .read_from(seq) -> Iterator[Event]
  .last_committed() -> tuple[int, int] | None    # (round, seq)
  .truncate_after(seq) -> list[Event]            # returns the discarded events
  .next_seq; .sync(); .close(); context manager
logical_view(events, exclude=("seq", "ts")) -> Iterator[dict]
write_jsonl(path, events, append=True)           # for discarded.jsonl
parse_event(dict | str) -> Event

# blobs.py
BlobStore(dir).put(data: bytes, *, sync=False) -> str; .get(sha) -> bytes (KeyError if missing)
  .has(sha); .put_text(str); .get_text(sha); .path(sha)

# snapshot.py
SnapshotStore(run_dir, blobs: BlobStore | None = None)   # default blobs = run_dir/"blobs"
  .write(manifest, plugin_blobs: dict[str, bytes]) -> Path  # fills manifest.plugins in place
  .latest(max_round=None) -> SnapshotManifest | None
  .read(round) -> SnapshotManifest; .load(manifest) -> dict[str, bytes]; .list_rounds() -> list[int]
  .path(round) -> Path                                     # snapshots/<round:06d>.json
```

## Decisions where the contract was silent

- **Event classes** are named `<CamelType>Event` (e.g. `PostEvent`, avoiding a clash with
  `medium.board.Post`). Models use `extra="forbid"`. `agent` defaults to None; `ts` defaults to
  `time.time()`; `seq` defaults to -1 until appended. Fields with obvious empties default to
  them (`args`, `fields`, `feedback`, `usage` = `{}`, `pending` = False, `payload` = None).
  `run_spec` on `run_started` is a plain dict (use `run_spec.model_dump(mode="json")`).
  `action` on `action_committed` is a dict (`Action.model_dump()`).
- **logical_view yields plain dicts and drops `seq` as well as `ts`** by default, because
  operational events are appended as they happen and shift later seqs. For the fork acceptance
  test compare with `exclude=("seq", "ts", "run")` (the fork has a different run id) and skip
  the `run_started` event (it carries `parent_run`/`fork_round`).
- **seq is 0-based.** `truncate_after(-1)` empties the log.
- **fsync** happens on `round_committed` appends, `sync()` and `close()`. Each append is one
  buffered write + flush of one complete line.
- **Torn tail:** reading stops before a final line that lacks `\n` or does not parse; reading
  never changes the file. The first `append` after opening truncates the torn bytes. A bad line
  followed by more data raises `EventLogCorrupt`.
- **truncate_after** keeps the original bytes of retained lines (rewrite is temp + rename +
  fsync) and returns the dropped events, so `recover()` can do
  `write_jsonl(run_dir/"discarded.jsonl", [e for e in dropped if e.type in OPERATIONAL_TYPES])`.
- **Blob layout** is sharded: `blobs/<sha[:2]>/<sha>`. `put` is atomic and idempotent;
  `sync=True` fsyncs (SnapshotStore always uses it; message content need not).
- **Snapshot manifests** are deterministic JSON (sorted keys, indent 2, trailing newline), so two
  identical runs write byte-identical manifests. Note the manifest contains `run`, so a fork's
  own re-written manifest differs from its parent's; copy the parent's file if byte identity is
  required, or compare `plugins`.
- **YAML shape:** strings are accepted wherever a plugin is expected (`world: flaggame`,
  `topology: gossip`, `metrics: [belief.consensus]`). Arms may carry their own `options` and
  `budget`, merged over the top-level ones (needed for commit policy as an arm). Option
  precedence: defaults < top-level `options` < arm `options` < `seed` < overrides.
  `max_rounds` must come from somewhere or `arm_to_runspec` raises `SpecError`.
  Unknown keys anywhere are a `SpecError` (CLI exit 2).
- **git_identity** ignores untracked files when computing `dirty`, so a `runs/` directory inside
  the repo does not mark the tree dirty.

## Things WP4 must know

- `ids.run_id(experiment, arm, seed)` always includes the arm. INTERFACE §2 says a Python-defined
  experiment's id is `<experiment>__s<seed>`; build that string in `experiment.py` (or add a
  helper to `ids.py`), WP1 did not change `ids.py`.
- `Experiment.to_yaml` can be `dump_experiment_yaml(runspec_to_doc(self.to_spec(seed, max_rounds)), path)`;
  `Experiment.from_yaml(path, arm)` can be `from_spec(arm_to_runspec(load_experiment_yaml(path), arm, seed=..., max_rounds=...))`.
  The YAML seed is supplied at run time, not stored in `options` by `runspec_to_doc`.
- `RunSpec.medium.topology`/`policies` are `PluginSpec`s; `Board.spec()` params (WP3) map onto
  `MediumSpec` fields by name.
- `Metric.update(event)` will receive typed `Event` objects from `events.py`.

## Changes outside WP1 files

- `swarmlab/world/base.py`: removed the unused `from typing import Any` import so
  `ruff check swarmlab` passes. No names added, renamed or removed.
- `swarmlab/__init__.py`: added the `Budget` export.
