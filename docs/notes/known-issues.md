# Known issues (kept short; fix when an experiment needs it)

- `tool_protocol="json"` does not work on Haiku 4.5: the model answers in prose instead of a JSON array (smoke 2026-10-06). Only needed for models without native tool calling.
- Qwen3.5-9B hits `finish_reason=length` on some turns even with thinking off at `max_tokens=1024`; use 2048.
- Fake-provider runs report a non-zero "spend" (the fake pricing table); label it as simulated in the CLI table.
- The OpenAI-compatible adapter ignores `reasoning_content`; hidden reasoning tokens are neither stored nor counted when thinking is on.
- Haiku `thinking_budget` is unsupported across tool turns (thinking blocks are not carried); left unset.
- Sequential commit (`commit: immediate`) with a high-latency provider is impractical: every call is on the critical path, so DeepInfra's tail (p90 112 s, max 343 s observed) made one N=16 round take about 30 minutes. Mitigation is the per-request timeout+retry (default 90 s, 2 retries); the structural answer is to prefer phase-commit for slow providers or run sequential at small N.
- A fork's `Run.spend` includes the parent's prefix spend (correct for the ledger, confusing in tables); show "inherited" and "new" separately.
- Probe-sourced metrics must be declared with `params: {source: "probe:<name>"}`; a string shorthand `belief.consensus@probe:belief` in the metrics list would be friendlier. M2 phase 1 computed them offline (`tools/m2_report.py`) because the spec omitted them.
- `Experiment.estimate` assumes the per-turn call cap every turn and overstates real spend 4-6x; add a mode seeded from a measured run (calls/turn and tokens/round).
- `swarmlab validate` rejects specs that use `vllm:` models ("provider 'vllm' needs a base_url") because the base_url is injected only by `swarmlab job run`. Validate should accept vllm specs with a placeholder, or `job run` should be documented as the validator for them.
- The job wall-time estimate (30 + 2N s per round) is 3-4x pessimistic for self-hosted vLLM (measured: N=64 ≈ 40 s/round). Seed it from a measured run.
