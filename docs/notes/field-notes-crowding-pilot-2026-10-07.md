# Field notes from the first external experiment (crowding pilot, 2026-10-07)

Source: another Claude Code session ("ideas-market-implementation") ran a complete experiment on main `ee94878` as an outside user: a custom World (4 search sites, 12 LLM agents, sequential choices under `commit: immediate`), two arms × three seeds on `hf:Qwen/Qwen3.8-27B:cerebras`, fake dry arms, analysis, an interactive replay page rebuilt from the event log, and `swarmlab publish`. Spend $0.96, six clean runs. Experiment dir: `/data/workspaces/ideas-market/crowding-pilot`.

## Worked well
1. World interface small enough: reset/observe/score + one `@tool`; `Outcome.feedback` carried analysis state; `module:Class`, param capture and default snapshot just worked; resume/replay needed no world code.
2. `commit: immediate` gave sequential updated state; seeded shuffle gave matched schedules across arms for free.
3. `system_prompt_append` → `swarmlab prompts --arm` → `diff` hygiene loop; the diff showed exactly the two intended lines.
4. `fake:module:function` dry arms reproduced the analytic greedy path before spending.
5. `replay` as integrity check; skip-by-spec-hash; summary table; budget preflight lines.
6. Event log complete enough to rebuild a replay page; pi-format sessions were the only source of agents' reasoning text (document as the intended path).
7. Cerebras via the router: 0.4 s per call; sequential N=12 ≈ 1 min/run.

## Awkward
1. `module:Class` plugins next to the spec need `PYTHONPATH=.`; CLI should put the spec dir on sys.path.
2. A run where every turn errored (Cerebras rejects `chat_template_kwargs`; 36/36 turns `error`, $0) still ended `max_rounds`, exit 0, score W=0. Wants: distinct end reason / loud summary, a per-arm preflight sending one real request with the arm's exact `extra` and tools, docs note that Cerebras wants `reasoning_effort: "none"` / `disable_reasoning: true` and that Qwen3.8 thinks by default.
3. `budget.total_usd` covers one `swarmlab run` invocation; two parallel processes each got the full cap. Wants `run --parallel N` or persisted experiment spend.
4. `total_usd` requires `hard_usd > 0` on fake dry arms too.
5. Soft/hard gap warning fires on every run (worst case 4–6× reality). Wants `estimate --from RUN_DIR`.
6. `swarmlab prompts` does not call `world.begin_round`, so round-dependent observations differ from live.
7. `report` is FlagGame-shaped; wants a generic protocol-health section for any world.
8. `publish` on a bucket-mounted runs dir hit `[Errno 5]` copying blobs, created no repo, exited 0; export copies every blob into export/raw (slow on object storage).
9. Session export: a duplicated "Round 1." marker with round-2 content under it (baseline seed 3, a008).
10. Metrics are per round; per-decision series had to be rebuilt from `action_committed` feedback; document the `Outcome.feedback` pattern and a `verify()`-based metric example.
11. Viewer has no hook for a world's own state panel; wants `World.render_state()`.
12. Spec-hash mismatch behaviour of `run` undocumented.
13. `doctor` via `uv run --project .` created `.venv` inside the checkout.
