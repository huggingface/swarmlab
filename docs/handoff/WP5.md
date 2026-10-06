# WP5 handoff: CLI, viewer, README, examples

Branch `wp5`. `pytest -q -W error` (182 tests) and `ruff check swarmlab tests examples` are clean.

## What landed

| file | what |
|---|---|
| `swarmlab/cli.py` | typer `app`: `validate`, `run`, `replay`, `resume`, `fork`, `view`; `summary(run)` |
| `swarmlab/viewer/build.py` | `build(run_dir) -> Path` (writes `view.html`), `collect(run_dir) -> dict` (the embedded data) |
| `swarmlab/viewer/template.html` | the page: inline CSS and vanilla JS, `__TITLE__`/`__DATA__` placeholders |
| `swarmlab/experiment.py` | `Run.view()` calls `viewer.build.build` (the `NotImplementedError("WP5")` is gone) |
| `examples/flaggame_m1a.yaml` | DESIGN example 1 as YAML; arms `broadcast` and `gossip`, 16 `evidence_aggregator`, 5 metrics, `max_rounds: 20` |
| `examples/01_run_existing.py`, `02_change_visibility.py`, `03_new_world.py` | the three DESIGN examples |
| `tools/viewer-check/screenshot.js` | Playwright sanity render (not part of pytest); `npm install` there first |
| `tests/test_cli.py`, `tests/test_viewer.py` | 11 tests |
| `tests/test_api.py` | `test_example_1` now asserts `run.view()` returns an existing `view.html` |
| README.md, .gitignore (`node_modules/`) | |

## CLI

- Every command builds an `Experiment` with `Experiment.from_yaml` or opens a `Run`, then calls
  `Experiment.run`, `Run.load` (replay), `Run.resume`, `Run.fork(...).run`, or `Run.view`. It
  never touches the `Runner`.
- `--json` prints one object: `run_dir, run_id, spec_hash, status, end_reason, last_round, score,
  metrics`. `metrics` maps each name to `{value, denominator, round}` for its last logged round.
  Extra keys: `replay: "ok"`, `view: <path>`, and `parent_run`/`fork_round` for forks.
  `validate --json` prints `{ok, spec, name, arms: {arm: {world, n_agents, topology, metrics,
  max_rounds}}}`. On failure, `--json` prints `{ok: false, error, exit_code}` and the error also
  goes to stderr.
- Exit code 2 covers: `SpecError`, an unknown or missing `--arm`, a plugin that cannot be
  resolved or constructed (the `ValueError`/`TypeError`/`ImportError`/`AttributeError` that
  `from_yaml` raises), and a missing `max_rounds`. Exit code 1 covers everything else,
  including `ReplayMismatch`, a run dir that does not exist, and an existing log.
- `--arm` is optional when the YAML has a single arm. For `fork --spec edited.yaml` without
  `--arm`, the CLI uses the parent's arm if the edited YAML has it, else the only arm.
- `--seed` is required for `run`.
- `fork` also takes `--max-rounds`, which is passed to `ForkHandle.run`.

## Viewer data format

The page embeds `<script type="application/json" id="swarmlab-data">`, which holds
`collect(run_dir)`. In it, every `<` is escaped as `<`.

```
{"version": 1,
 "meta": {run_id, experiment, arm, spec_hash, status, end_reason, last_round, score,
          parent_run, fork_round, world: PluginSpec, medium: MediumSpec, options: RunOptions},
 "agents": ["a000", ...],                       # sorted, from round_started.order
 "world": {"kind": "flaggame", "candidates": {name: [rows]}, "crops": {agent: [rows]}} | null,
 "truth": "C" | null,                            # run.json score.truth, else world.verify()
 "events": [...]}                                # logical view of committed rounds, minus seq/ts/run
```

Event reductions:
- `run_started` drops `run_spec`.
- `tool_returned.result` becomes `{ok, error, summary}`. For read_board the summary is
  "N items: post ids"; for anything else it is clipped JSON.
- String args of `tool_called` longer than 400 characters are clipped.
- `delivery` drops `content_hash` and gains `content`, the delivered text from the blob store.

Events after the last `round_committed` are dropped, except `run_ended`.

`world` is filled only when the run's world plugin builds and is a `FlagGame` (or subclass). It
is restored from the latest snapshot. If the world is not a FlagGame, or its class is an inline
`__main__` class that the CLI process cannot import, `world` is null. The page then shows a
committed-actions table instead of the flag and guesses panels.

The page folds the data per round in JS:
- Guesses come from accepted `action_committed` events whose action is `guess`. "changed" means
  the agent's committed guess at round r differs from its guess before round r, so a first
  guess counts as a change.
- An agent's inbox lists the deliveries created up to round r. A delivery's read round is the
  first `read` event that lists it. A delivery that is not read shows "not yet" while its
  eligible round is after r, and "unread" otherwise.
- The action queue pairs `tool_called` with `tool_returned` by `call_id` and shows the
  `turn_ended.yield_kind`.

Navigation and display:
- Navigate with the slider, the prev/next buttons, or the arrow and Home/End keys. A URL hash
  `#r=N` opens round N.
- The truth is shown only when "reveal truth" is ticked; the header score hides `truth` until
  then.
- The page follows light/dark through `prefers-color-scheme`.

Size: a 16-agent, 20-round broadcast run gives about 470 KB, and the gossip arm about 430 KB.

## Mismatches with DESIGN.md / INTERFACE.md

1. DESIGN example 2 is a fragment (`Experiment(..., medium=...)`). `examples/02_change_visibility.py`
   fills the `...` in with example 1's world, participants, metrics and budget.
2. DESIGN example 3 defines `Counter` but no participant, and an `Experiment` needs at least one.
   `examples/03_new_world.py` adds a 4-line `Adder(Participant)` and runs
   `[Adder(), Adder(step=2)] * 2`. The design doc could show that participant.
3. In INTERFACE §0, `Run.fork(...)` is typed `-> Experiment`. The merged code returns a
   `ForkHandle` whose `.run()` gives the child `Run`, which is what DESIGN's
   `run.fork(at_round=4).run()` uses. The type in §0 should be fixed.
4. DESIGN §13 lists `resume RUN [--at 40]`. M1a resume has no `--at`, because forking at a round
   is `fork --at`. DESIGN also lists `dryrun`, `smoke`, `status`, `metrics` and `publish`. They
   are outside the M1a subset (INTERFACE §16) and are not implemented.
5. INTERFACE §15 lists `artifacts/lock.txt`. No package writes it yet (WP4 noted this too).
6. All imports in the DESIGN examples resolve as written (`swarmlab.worlds`,
   `swarmlab.participants`).

## Checks run

- `swarmlab validate examples/flaggame_m1a.yaml` succeeded.
- `swarmlab run ... --arm gossip --seed 3` and `--arm broadcast --seed 3` both finished. Their
  accuracy was 0.875 and 1.0.
- `swarmlab view` worked on both runs.
- All three example scripts ran.
- Playwright screenshots in light and dark mode, plus a non-FlagGame run, rendered with no
  console errors and no external requests.
