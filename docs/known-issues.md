# Known issues (kept short; fix when an experiment needs it)

Last checked against main on 2026-10-08. Items fixed since the first version (per-call `--calls-per-turn`/`--prompt-growth` in `estimate`, `system_prompt_append`, the experiment-wide `total_usd` cap, `resume --add-budget`, spend on status lines, probe errors skipping the probe, discarded-round spend in exports, simulated runs in `report`) were removed.

## Models and providers

- `tool_protocol="json"` does not work on Haiku 4.5: the model answers in prose instead of a JSON array (smoke 2026-10-06). Only needed for models without native tool calling.
- DeepInfra (via the HF router) stalls on some tool-call requests for Qwen3.5-9B (p90 112 s, max 343 s observed). The per-request timeout and retry (default 90 s, 2 retries) bound it; consider another route or lower `hf` concurrency, or self-host with `swarmlab job run`.
- Sequential commit (`commit: immediate`) with a high-latency provider is impractical: every call is on the critical path (one N=16 round took about 30 minutes on DeepInfra). Prefer phase-commit for slow providers, or sequential at small N.
- Qwen3.5-9B hits `finish_reason=length` on some turns even with thinking off at `max_tokens=1024`; use 2048.
- The OpenAI-compatible adapter ignores `reasoning_content`; hidden reasoning tokens are neither stored nor counted when thinking is on.
- Haiku `thinking_budget` is unsupported across tool turns (thinking blocks are not carried); left unset.

## Estimates, spend, specs

- `Experiment.estimate` / `swarmlab estimate` assume the per-turn call cap every turn and overstate real spend 4-6x unless you pass `--calls-per-turn` and `--prompt-growth` by hand; there is no mode seeded from a measured run yet.
- `estimate` prices each participant by its own model, not a role's `model` override.
- Total cap (`total_usd`), after the 2026-10-08 admission fix (branch `ledger-fix`, WP6 handoff): a run that fits only once runs in flight end now waits instead of capping every later run, but with `hard_usd` far above real spend (m6-flag-paper: $3 ceilings, about $0.26 real) runs still wait on each other's headroom and `--parallel` gives less concurrency than asked; size `hard_usd` from measured spend. Ledger spend of a run in flight lags by up to 60 s (heartbeat), so the runner's round-boundary `total_budget` check sees other runs' spend that late (admission is unaffected: it reserves their `hard_usd`). Two instances of one run id in flight at once (two `--rerun` invocations) count as one. `experiments/m6_flag_paper.yaml` still has `total_usd: 0` from the workaround in 2979125, and `tests/test_m6_paper_spec.py` still expects one seed and `total_usd: 10` (2 failures on main).
- Fake-provider runs report a non-zero "spend" (the fake pricing table); `swarmlab report`, the `run` table and status lines label it `(simulated)`, `run.json` does not.
- A fork's `Run.spend` includes the parent's prefix spend (correct for the ledger, confusing in tables); "inherited" and "new" are not shown separately.
- Probe-sourced metrics must be declared as `{type: belief.consensus, params: {source: "probe:belief"}}`; the string `belief.consensus@probe:belief` is only the resulting metric name and is rejected in the `metrics:` list.
- `swarmlab validate` rejects specs that use `vllm:` models ("provider 'vllm' needs a base_url") because the base_url is injected only by `swarmlab job run`. Use `job run` (without `--launch`) as the validator for them, or give `providers: {vllm: {type: openai_compat, params: {name: vllm, base_url: ...}}}`.
- The job wall-time estimate (30 + 2N s per round) is 3-4x pessimistic for self-hosted vLLM (measured: N=64 about 40 s/round). Pass `--per-round` or `--timeout` from a measured run.

## Roles, paired runs, registry, viewer

- Roles: a role's `model` override is ignored by `Experiment.estimate`; a fork keeps the parent's participant settings (role overrides in an edited spec do not re-bind them); `reconfigure` interventions do not change roles. `Role` and `Tree` are not re-exported at top level (import from `swarmlab.roles` and `swarmlab.medium.topology`).
- Paired runs: `PairedResult` is not re-exported from `swarmlab` (import from `swarmlab.experiment`).
- Registry: the viewer shows registry state as lists (tool results and actions); the `ColoringGrid` panel (`World.render_state`) draws the grids but not who holds which claim.
- `LLMAgent` `summary_model` (context-limit summaries) is not priced by `Experiment.estimate`.
- `swarmlab preflight` assumes native tool calling: for arms with `tool_protocol: json` / `report_json` it reports "NO tool call parsed" on a valid JSON reply (observed with Qwen3-VL on the m6 real-flag spec). It should send the arm's actual round-1 message and judge the reply by the arm's protocol.
