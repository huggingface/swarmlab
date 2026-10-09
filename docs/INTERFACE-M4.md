# swarmlab M4 interface contract: export and publish

Scope: DESIGN.md component 3 (publish layout) and component 11 (viewer publishing), plus the experimenter skill. Builds on INTERFACE.md §3-§5 and §15. Read-only with respect to the runner: everything here consumes finished run directories.

## 1. Export

`swarmlab export RUN_DIR [--out DIR]` and `Run.export(out) -> Path` produce, next to or inside the run dir:

```
export/
  run.json                      identity, spec, score, end reason, spend, metric finals
  tables/<family>.parquet       one Parquet table per event family (see below)
  sessions/<agent>.jsonl        one pi-format session per agent (Hub agent-traces viewer)
  raw/events.jsonl              the log, byte-identical copy
  raw/discarded.jsonl           if present
```

**Tables.** Families: `turns` (turn_started/turn_ended joined: agent, round, yield_kind, calls, usage fields, error), `tool_calls` (tool_called/tool_returned joined: call_id, agent, round, tool, args JSON, ok, pending, result JSON), `posts`, `deliveries`, `reads` (one row per delivery id read: agent, round, delivery_id), `actions` (action_committed), `inference` (attempt/response joined: call_id, agent, round, provider, model, request_hash, response_hash, usage fields, cost_usd, latency_s, cached, attempts; additively `refusal_category`, `reasoning` (inlined like blob text up to 64 KB, else the response blob's `sha256:` hash), `reasoning_kind`, `reasoning_redacted`, schema `swarmlab-export/3`), `probes`, `metrics`, `interventions`, `rounds` (round_started/round_committed/snapshot/budget joined), `run` (run_started/run_ended). Shared key columns first: `experiment, arm, seed, run, round, agent`. Blob-referenced content is inlined as text up to 64 KB per cell, else the hash. Schema version string in `run.json["export_schema"]`.

**Sessions.** pi session format v3, one file per agent: header line `{"type":"session","harness":"swarmlab","id":"<run>/<agent>","name":...}` then one `message` line per conversation message reconstructed from the agent's turns: the system prompt (from the first inference request blob when present), per round a user message (the view text), assistant messages with text and tool calls (additively, reasoning as pi `thinking` blocks with `thinkingSignature`, in the provider's block order), and tool messages with results, in log order. Scripted agents get a session built from their tool calls alone. Read the format spec at https://github.com/earendil-works/pi/blob/main/packages/coding-agent/docs/session-format.md and the Hub page https://huggingface.co/docs/hub/agent-traces; follow them exactly.

## 2. Publish

`swarmlab publish RUNS_DIR_OR_RUN [--repo cmpatino/<experiment>] [--public] [--tag TAG]` and `Experiment.publish(...)`: creates (if needed) one **dataset** repo per experiment, private by default, tagged `format:agent-traces` only when `--public` (the Hub viewer renders public repos only), uploads every finished run's `export/` under `runs/<run_id>/`, writes a dataset card README with the experiment spec, arms, run table (id, arm, seed, end reason, score, spend), and the per-family table schemas, and a top-level `index.json` listing runs. Re-publishing is idempotent (unchanged files skipped). Uses `huggingface_hub` (`hub` extra). `swarmlab fetch-published REPO RUN_ID --out runs/` restores a run dir from the published raw log and blobs so `Run.load` works (blobs are included in `raw/blobs/` when the run is under 500 MB; otherwise only content referenced by the tables).

## 3. Viewer publishing

`swarmlab view RUN_DIR --publish REPO` uploads `view.html` into the same dataset repo under `runs/<run_id>/view.html`, and the dataset card links it. A static Space is out of scope for M4.

## 4. Experimenter skill

`skill/SKILL.md` (plus `AGENTS.md` updates) teaching a coding agent the tested workflow: doctor, init, estimate, dry run with scripted agents, smoke at N=4 with a hard ceiling, second-agent review of the exact prompts each arm sees (`swarmlab prompts SPEC --arm A`, a new command that renders the system prompt and round-1 user message for one agent of each group), launch locally or via `swarmlab job run`, watch, fetch, report (`tools/m2_report.py` generalised into `swarmlab report RUNS_DIR`), publish. Install instructions for Claude Code, Codex, and OpenCode skill directories. Every rule in it must point at a command that exists.

## 5. Acceptance

1. Export of a finished fake-provider run produces every table with the documented key columns; `pyarrow` reads them; row counts match event counts; sessions validate against the pi format (a local validator in tests).
2. Publish to a private dataset repo (integration test behind `SWARMLAB_HUB=1`, otherwise mocked) is idempotent; `fetch-published` restores a run that `Run.load` replays with the recorded score.
3. `swarmlab prompts` renders without a model call; `swarmlab report` reproduces the M2 report on the archived runs.
