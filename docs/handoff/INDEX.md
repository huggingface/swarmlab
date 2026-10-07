# Handoff index

One line per work package. Contracts are in `docs/INTERFACE*.md`; run notes in `docs/notes/`; open problems in `docs/notes/known-issues.md`. There are no separate fix handoffs: the fix rounds (dogfood fixes, estimate/budget fixes, probe-error handling, report and export fixes) are recorded in the git log and in `docs/notes/dogfood-2026-10-07.md` and `dogfood-boardcue-report-2026-10-07.md`.

| WP | milestone | covers |
|---|---|---|
| [WP1](WP1.md) | M1a | Core: spec and YAML loading, spec hash, event classes and log, blob store, snapshots |
| [WP2](WP2.md) | M1a | FlagGame text variant: crops, rival candidates, guess action, status tools, score |
| [WP3](WP3.md) | M1a | Medium: board, inboxes, deliveries, delay policy, broadcast/gossip/groups topologies |
| [WP4](WP4.md) | M1a | Runner, executor, scheduler, Experiment/Run API, scripted participants, belief and comm metrics, replay, resume, fork |
| [WP5](WP5.md) | M1a | CLI (validate, run, replay, resume, fork, view), static replay viewer, README, examples |
| [WP6](WP6.md) | M1b | Providers (Anthropic, OpenAI-compatible: hf/openai/vllm, fake), budget gate, inference through the harness |
| [WP7](WP7.md) | M1b | LLMAgent, default prompt, belief probe, probe-sourced metrics, real-provider smoke script |
| [WP8](WP8.md) | M2 | HF Jobs placement: vLLM next to the runner in one job, `swarmlab job run/status/logs/fetch`, bucket transfer |
| [WP9](WP9.md) | M3a | Paired runs from round 0 (`Experiment.pair`, fork at round 0, `crop_overrides`, `repeat`) |
| [WP10](WP10.md) | M3a | Interventions: trigger plus operation plugins (inject_post, delay_delivery, mute, kill_agents, patch_private, reconfigure) |
| [WP11](WP11.md) | M3a | LLMAgent context limit with drop_oldest, summarize and fail_turn overflow policies |
| [WP12](WP12.md) | M4 | Export (Parquet tables, pi sessions), Hub publish and fetch-published, report, prompts, experimenter skill |
| [WP13](WP13.md) | M3b | Registry with claim policies (advisory, enforced), ColoringGrid world, coloring and claim metrics, Coloring S0 example |
| [WP14](WP14.md) | M3c | Roles (built-ins plus overrides), executor role enforcement, Tree topology, hierarchy examples |
| [field-notes-report](field-notes-report.md) | field notes | Report protocol health for any world, export/publish robustness and `--no-raw`, session round-marker fix, analysis docs, `World.render_state` viewer panel |
