# Known issues (kept short; fix when an experiment needs it)

Last checked against main on 2026-10-07. Items fixed since the first version (per-call `--calls-per-turn`/`--prompt-growth` in `estimate`, `system_prompt_append`, the experiment-wide `total_usd` cap, `resume --add-budget`, spend on status lines, probe errors skipping the probe, discarded-round spend in exports, simulated runs in `report`) were removed.

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
- Fake-provider runs report a non-zero "spend" (the fake pricing table); `swarmlab report`, the `run` table and status lines label it `(simulated)`, `run.json` does not.
- A fork's `Run.spend` includes the parent's prefix spend (correct for the ledger, confusing in tables); "inherited" and "new" are not shown separately.
- Probe-sourced metrics must be declared as `{type: belief.consensus, params: {source: "probe:belief"}}`; the string `belief.consensus@probe:belief` is only the resulting metric name and is rejected in the `metrics:` list.
- `swarmlab validate` rejects specs that use `vllm:` models ("provider 'vllm' needs a base_url") because the base_url is injected only by `swarmlab job run`. Use `job run` (without `--launch`) as the validator for them, or give `providers: {vllm: {type: openai_compat, params: {name: vllm, base_url: ...}}}`.
- The job wall-time estimate (30 + 2N s per round) is 3-4x pessimistic for self-hosted vLLM (measured: N=64 about 40 s/round). Pass `--per-round` or `--timeout` from a measured run.

## Roles, paired runs, registry, viewer

- Roles: a role's `model` override is ignored by `Experiment.estimate`; a fork keeps the parent's participant settings (role overrides in an edited spec do not re-bind them); `reconfigure` interventions do not change roles. `Role` and `Tree` are not re-exported at top level (import from `swarmlab.roles` and `swarmlab.medium.topology`).
- Paired runs: `PairedResult` is not re-exported from `swarmlab` (import from `swarmlab.experiment`).
- Registry: the viewer shows registry state as lists and has no grid panel for `ColoringGrid`.
- `LLMAgent` `summary_model` (context-limit summaries) is not priced by `Experiment.estimate`.
