# Known issues (kept short; fix when an experiment needs it)

- `tool_protocol="json"` does not work on Haiku 4.5: the model answers in prose instead of a JSON array (smoke 2026-10-06). Only needed for models without native tool calling.
- Qwen3.5-9B hits `finish_reason=length` on some turns even with thinking off at `max_tokens=1024`; use 2048.
- `swarmlab estimate` requires `--arm`; `swarmlab run` does not. Estimate should run over all arms × seeds like `run`, and should model context growth (`prompt_tokens` grows per round under full memory).
- Fake-provider runs report a non-zero "spend" (the fake pricing table); label it as simulated in the CLI table.
- The OpenAI-compatible adapter ignores `reasoning_content`; hidden reasoning tokens are neither stored nor counted when thinking is on.
- Haiku `thinking_budget` is unsupported across tool turns (thinking blocks are not carried); left unset.
