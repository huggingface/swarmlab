# WP9 handoff: paired runs from round 0 (M3a §2)

Branch `paired-runs`. `uv run pytest -q -W error` passes 381 tests (372 before, plus 9 in
`tests/test_pair.py`). `uv run ruff check swarmlab tests examples tools` is clean.

## What landed

| file | change |
|---|---|
| `swarmlab/world/flaggame.py` | `FlagGame(crop_overrides: dict[str, list[int]] | None = None)`, constructor and `reset` only |
| `swarmlab/spec.py` | `RunOptions.repeat: int = 0` (dropped from dumps when 0) |
| `swarmlab/runner.py` | fork at round 0 (`_fork_at_zero`, `_fork_dir`); `agent_stream()` and `RepeatSeeded` for repeats |
| `swarmlab/experiment.py` | `Experiment.pair(...) -> PairedResult`, `PairedResult`; `run(..., repeat=None, run_id=None)` |
| `tests/test_pair.py` | crop overrides; fork at 0 (scripted and fake LLM); pair on `fake:reader` (acceptance 5) |
| `tools/paired_haiku.py`, `tools/mixed_swarm.py` | real tests (results in `docs/notes/m3-paired-and-mixed-2026-10-07.md`) |

## Decisions where the contract was silent

**crop_overrides**
- Applied after all normal `reset` draws, so candidates, the truth and other agents' crops are
  identical.
- Positions are checked against the flag at construction (`0 <= y <= height - crop_h`, likewise
  for x). Unknown agent ids raise at `reset`.
- The param is popped from `params` when None, so FlagGame specs and spec hashes without it are
  unchanged.
- It is config, not state (skipped in snapshots). The resulting `crops` are in the snapshot and
  in `verify()`.

**Fork at round 0**
- Runs never write a round-0 snapshot; the first one comes after round 1. So `fork(0, exp)`
  copies nothing.
- The child resets the world with the parent's seed (and repeat) under `exp`, binds fresh
  participants, and runs from round 1 with the same schedule and topology streams.
- Its log starts with its own `run_started(round=0, parent_run, fork_round=0)`, and
  `restored = {"world": False, "participants": [], "metrics": [], "reset": True}`.
- Changing the participant count raises `ValueError`, as for other forks.
- Replay and resume need no special case: `fork_round=0` takes the fresh-run path.

**Repeats**
- `RunOptions.repeat` is in the spec, so resume and replay know it. A custom pydantic serializer
  omits it when 0, so ordinary specs are byte-identical and hash as before.
- Repeat i > 0 binds agent `a` with `derive(seed, "repeat", i, "agent", a)`; repeat 0 keeps
  `derive(seed, "agent", a)`.
- `LLMAgent` never uses its rng, and `participants/llm.py` is off limits, so the runner wraps
  `Inference` in `RepeatSeeded`. For repeat > 0, any request without a `seed` gets one from
  `derive(seed, "repeat", i, "agent", a, call_id)`.
- Effects of the request seed:
  - It changes the request hash. The fake provider then answers differently, and cache keys
    differ per repeat.
  - Seeded OpenAI-compatible providers sample differently.
  - Anthropic ignores it; Anthropic sampling is not reproducible anyway.
  - The seed is a pure function of the run, so a resumed repeat hits its own cache.
- World, schedule and topology streams never depend on the repeat.

**`pair` layout**
- Files go to `<out>/<run_id(seed)>__pair/`: `pair.json` plus one run dir per run.
- Run dirs and run ids are `<run_id>__{base,patched,control}_r<i>`. `run(..., run_id=)` was added
  for this.
- Runs execute in the order base_r0, patched_r0, control_r0, base_r1, and so on.
- The patched runs are plain runs of the patch experiment with the same seed and repeat, not
  `fork(0)` children. The result is the same, and it keeps the three roles symmetric.
- `patch=None` pairs the base with itself.
- A different participant count is a `ValueError`. A second `pair` into the same dir is a
  `FileExistsError`.

**Control pairs**
- A control is an unchanged re-run of the base with the same repeat stream. On the fake provider
  it is identical to the base, as acceptance 5 asks. On real models it measures sampling noise
  under exactly the patched pair's conditions.

**`effect(metric, round=-1)`**
- `diff` is the mean of patched minus base over repeats.
- `control_spread` is the root-mean-square of control minus base, i.e. the spread around the
  zero those diffs should have. It is defined for a single control pair, where it equals |diff|.
  It is None without controls.
- The result also includes `diffs`, `control_diffs`, `n` and `n_control`.
- `round=-1` uses each run's last logged value. Pairs where either value is None are skipped. A
  missing metric or round raises `KeyError`.

**Ownership**
- `PairedResult` is not re-exported from `swarmlab/__init__.py` (another WP's file). Import it
  from `swarmlab.experiment`.
- Runner changes outside the fork path are the two-line repeat hook in `_build` and
  `_init_fresh`, plus the module-level helpers.

## Real-test summary

Details: `docs/notes/m3-paired-and-mixed-2026-10-07.md`. Runs:
`/data/workspaces/ultra-harness/runs/m3-tests/`. Total spend: $3.32.

**Paired Haiku (seed 2, $2.40).**
- Seed 1 was skipped because a003's own crop already contains the rival.
- a003's truth-only crop moved to a window contained in {truth H, rival G}. a003 guessed the
  rival G in round 1, read two truth-only posts in round 2, and held H from then on.
- `effect("belief.consensus")`: diff +0.25, control_spread 0.25, so no effect beyond noise. The
  base run was the outlier at 0.75; patched and control both reached 1.0.
- a003 never posted, so no other agent read its evidence. Base and control already differ in
  round 1.

**Mixed 8 Haiku + 8 Qwen (seed 1, $0.84, plus $0.08 for an aborted 90 s-timeout attempt).**
- Two providers in one run work.
- Haiku: read rate 0.85, accuracy 1.0, probes 48/48, latency median 2.7 s / p90 5.3 s, no retries.
- Qwen on DeepInfra: 32 of 48 turns failed after 3 x 60 s timeouts. 41 of 114 responses needed
  more than one attempt. Latency median 2.4 s, p90 64 s, max 178 s. Read rate 0.02, accuracy
  0.5, probes 47/48.
- Cross-model reads happened both ways: Haiku read a Qwen post 7 times, Qwen read Haiku posts 3
  times.

## Open issues

- `Runner._probe_one` does not catch `ProviderError`, so a probe whose retries are exhausted
  aborts the round. The run stays resumable. This is the probe path, owned by WP7.
- DeepInfra's tail on tool requests is heavy for Qwen3.5-9B. Consider another route or lower
  `hf` concurrency before Qwen-heavy experiments.
