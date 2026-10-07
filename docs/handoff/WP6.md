# WP6 handoff: providers, budget, inference through the harness

Branch `wp6`. `pytest -q -W error` (244 tests, about 40 s) and `ruff check swarmlab tests examples`
are clean. No real provider is called anywhere in the tests.

## What landed

| file | public names |
|---|---|
| `swarmlab/providers/base.py` | `Usage`, `ChatMessage`, `ChatRequest`, `ChatResponse`, `Provider`, `request_hash`, `request_bytes`, `split_model`, `model_id`, `text_of`, `parse_json_args`, `UnknownModelPricing`, `ProviderError` |
| `swarmlab/providers/anthropic.py` | `AnthropicProvider` ("anthropic"), `build_kwargs`, `parse_message`, `tool_definition`, `THINKING_BUDGET_MODELS` |
| `swarmlab/providers/openai_compat.py` | `OpenAICompatProvider` ("openai_compat"), `preset`, `PRESETS`, `build_body`, `parse_completion` |
| `swarmlab/providers/fake.py` | `FakeProvider` ("fake"), `flaggame_reader`, `resolve_script`, `parse_crops`, `parse_listing` |
| `swarmlab/providers/__init__.py` | `resolve(model, overrides=None) -> (Provider, model_id)`, `preset(prefix)`, `PREFIXES` |
| `swarmlab/budget.py` | `Ledger`, `Gate`, `Reservation`, `BudgetExceeded`, `HardCeilingReached`, `SoftBudgetReached`, `MeasurementBudgetReached` |
| `swarmlab/inference.py` | `Inference` (cache + gate + operational events), `InferenceCache` |
| `swarmlab/events.py` | `BudgetEvent`, `BudgetChangedEvent`, `ProbeEvent`; `inference_attempt.category`; `inference_response.served_by/finish_reason/cached` |
| `swarmlab/tools.py` | `AgentTools.infer(request, *, category="swarm")` |
| `swarmlab/executor.py` | `RoundExecutor(..., inference=None)`, `.infer(agent, request, category)`, `.inference_usage(agent)`, `.hard_ceiling` |
| `swarmlab/runner.py` | budget enforcement, `resume(budget=None)`, ledger in run.json and snapshots, cache copy on fork, replay provider-call check |
| `swarmlab/experiment.py` | `Experiment.providers`, `.provider_for(model)`, `.resolved_providers()`, `.estimate(...)`, `Run.spend`, `Run.resume(budget=None)`, `participant_model(p)` |
| `swarmlab/spec.py` | `RunSpec.providers: dict[str, PluginSpec]`, YAML top-level `providers:` |
| `pyproject.toml` | entry-point groups `swarmlab.providers` (anthropic, openai_compat, fake) and `swarmlab.fake_scripts` (flaggame_reader) |

Tests: `test_providers.py` (hash stability, cost math, Anthropic adapter against a stub client incl.
the typed retry chain, OpenAI-compatible adapter against `httpx.MockTransport`, fake determinism),
`test_budget.py` (gate reserve/release/charge, hard ceiling raises before reserving, in-flight
reservations, measurement cap, cancellation, semaphore), `test_infer.py` (operational
attempt/response events, blobs, cache hit within a run, usage merged into `turn_ended`, replay
with zero calls, fork cache copy), `test_budget_runs.py` (acceptance 2 and 3, YAML
`providers:`/`budget:`, pricing errors at construction, `estimate`), `test_infer_recovery.py`
(cache part of acceptance 1: two identical runs, replay makes no `FakeProvider.complete` call,
SIGKILL inside round 7's commit then resume: all round-7 requests are cache hits, only rounds 8-9
call the provider, logical view, score, metrics and spend equal an uninterrupted run).

Acceptance status: **2** and **3** pass end to end with the fake provider (pricing
`{"*": (10, 50, 1)}` makes a 2-call reader turn cost about $0.012). The **cache/replay part of 1**
passes with `tests/helpers.py:FakeLLM` (a minimal loop over `tools.infer`) standing in for
`LLMAgent`; WP7 should re-run it with `LLMAgent(model="fake:reader")`.

## Signatures WP7 needs

```python
from swarmlab.providers.base import ChatMessage, ChatRequest, ChatResponse, Usage
from swarmlab.tools import ToolCall, ToolSchema
from swarmlab.view import Part

ChatMessage(role="system"|"user"|"assistant"|"tool", content: list[Part] | str,
            tool_calls: list[ToolCall] | None = None, tool_call_id: str | None = None)
ChatRequest(model="fake:reader", messages=[...], tools=view.tools, tool_protocol="native"|"json",
            max_tokens=1024, temperature=None, top_p=None, seed=None, thinking_budget=None, extra={})
ChatResponse: text, tool_calls: list[ToolCall], usage: Usage, cost_usd, provider, model,
              served_by, latency_s, finish_reason, cached

# in Participant.turn(view, tools):
resp = await tools.infer(request)                         # category="swarm"
```

- Tool calls round-trip as `ToolCall(call_id=<provider's id>, name, args)`. Append
  `ChatMessage(role="assistant", content=resp.text, tool_calls=resp.tool_calls)` and one
  `ChatMessage(role="tool", content=<str>, tool_call_id=tc.call_id)` per executed call; the
  Anthropic adapter merges consecutive tool results into one user message itself.
- `finish_reason == "bad_tool_args"` means at least one call has `args == {"_raw": <text>}`
  (non-JSON arguments from an OpenAI-compatible server); feed an error tool message back.
- `tool_protocol="json"`: adapters send no tools; the model's text is the action list. The fake
  answers in that protocol with `text = '[{"name": ..., "args": {...}}]'`.
- **The agent's model defaults** for probes come from WP7's `LLMAgent.model_request_defaults()`
  (`model, temperature, max_tokens, thinking_budget`). The harness side only needs the model
  string: `swarmlab.experiment.participant_model(p)` reads `p.model` or `p.params["model"]`
  (`"<prefix>:<id>"`), and that is what `Experiment` prices at construction. Keep `model` as a
  constructor kwarg / attribute of `LLMAgent`.
- **Probes** run outside turns. After the commit of round r the runner's `self.executor` is still
  round r's executor; call
  `await self.executor.infer(agent, ChatRequest(messages=ctx + [user(q)], tools=[], **defaults), "measurement")`.
  Measurement calls get ids `i{round}-{agent}-{n}` continuing the agent's count, are logged as
  operational events with `category="measurement"`, are cached like swarm calls (so replay and
  resume never re-pay a probe), are charged to `ledger.spent["measurement"]`, and are **not**
  added to `turn_ended.usage`. Catch `swarmlab.budget.MeasurementBudgetReached` (not a
  `HardCeilingReached`) and log `probe(ok=False, parsed={"skipped": "measurement_budget"})`.
  A `HardCeilingReached` from a probe is not handled by the runner yet: decide in WP7 whether a
  probe crossing the ceiling ends the run (probably `run_ended(hard_ceiling)` after the commit).
- Probe events: `ProbeEvent(run, round, agent, probe, question_hash, raw_hash, parsed, ok,
  cost_usd)` exists and is logical. If probes are appended after `round_committed`, note that the
  snapshot and the `budget` event of round r are written before them; the ledger in `run.json` is
  rewritten after each commit, and recovery adds operational events past `run.json["ledger_seq"]`,
  so probe spend is not lost either way. Add `"probe"` handling to the metric feed as needed
  (it is currently fed like any logical event in replay; live feeds only up to `action_committed`).
- The fake's built-in `flaggame_reader` (alias `reader`, so `fake:reader` works): call 1 posts its
  crop once per conversation and reads the board; call 2 guesses the candidate containing the most
  distinct crops seen in the conversation and ends the turn; with no tools it answers
  `{"candidate", "confidence"}` (handy for `BeliefProbe` tests). It parses crops from any
  `crop:` block in message text or tool results (JSON-escaped newlines accepted), so render
  read_board results as JSON or text, either works. With `memory="window"` crops read in earlier
  rounds drop out of its context, which is the intended behaviour.

## Decisions where the contract was silent

- **Attempt logging after admission.** The gate's budget check runs before `inference_attempt` is
  logged, so a refused request leaves no attempt (the contract lists "log attempt, then admit").
  The attempt is still on disk before `provider.complete` is awaited. A provider exception logs
  `inference_response(response_hash="", finish_reason="error:<Type>")` and propagates.
- **Cache.** `blobs/cache/<request_hash>` holds the response JSON as first produced;
  `blobs/cache/index.jsonl` maps hash -> round for fork copies. The request blob is the request's
  canonical JSON, so its sha equals `request_hash`. No in-flight dedupe of identical concurrent
  requests.
- **Nominal usage.** `turn_ended.usage.cost_usd` and token counts are the *nominal* values of the
  responses (the original cost even on a cache hit). This keeps the logical view of a resumed run
  identical to an uninterrupted one. Actual spend lives in the ledger, `budget` events, and the
  operational `inference_response.cost_usd` (0 on hits). `turn_ended.usage` gains
  `cached_prompt_tokens`, `reasoning_tokens` and `inference_calls` only when the agent inferred.
- **Ledger.** It stores integer nano-dollars, so totals do not depend on completion order (the
  `budget` event is logical). `calls` counts dispatched provider calls, not cache hits. A call
  cancelled in flight (hard-ceiling abort) is charged its reservation. A provider error is charged
  nothing. Recovery rebuilds spend as `run.json["ledger"]` plus operational events after
  `run.json["ledger_seq"]` (responses at `cost_usd`, attempts without a response at
  `reserved_usd`), so spend survives SIGKILL and repeated crashes.
- **Budget limits.** Each field is enforced only when > 0. The hard ceiling bounds
  `spent_total + reserved + this call's worst case`. The measurement cap bounds the measurement
  category the same way and raises `MeasurementBudgetReached`. `max_cost` adds `thinking_budget`
  to `max_tokens` (worst case).
- **Hard ceiling bookkeeping.** `run_ended(hard_ceiling)` carries round r - 1, the last commit.
  The log keeps `round_started(r)` and round r's operational events. `run.json` keeps the last
  commit's score. On resume these and `run_ended` move to `discarded.jsonl`. Soft-budget runs are
  resumed the same way; without a larger budget they end again at once.
- **`Run.resume(budget=...)`** stores the budget in `run.json["budget"]` and leaves the archived
  spec and `spec_hash` unchanged (run identity is the configuration that was launched). The
  `budget_changed` event sits right after the recovered commit point, and recovery keeps it.
- **Providers are shared services.** `Experiment` caches resolved presets (`resolved_providers()`)
  and every run of the experiment uses the same instances. Nothing deep-copies or pickles a
  provider. `FakeProvider.calls` therefore counts across runs of one experiment.
- **Pricing.** A `"*"` key prices every model of a provider (the fake's default). The Anthropic
  table is seeded with Haiku 4.5 only. The `hf`/`openai`/`vllm` presets have empty tables, so the
  experimenter passes prices through `Experiment.providers` or YAML `providers:`. An unknown model
  raises `UnknownModelPricing` from `Experiment(...)` itself (raised after pydantic validation, so
  it is not wrapped in a `ValidationError`).
- **Anthropic details.** The SDK's own retries are off (`max_retries=0`), so our chain is the
  only one. `retry-after` is honoured (capped at 60 s). `temperature`/`top_p` are dropped when
  thinking is enabled. `seed` is ignored. The client is created per event loop.
  `strict: true` is set when `additionalProperties: false` and `required` are both present
  (`read_board` and `end_turn` have no `required`, so they are not strict).
- **OpenAI-compatible details.** One `httpx.AsyncClient` per call. Transport errors are retried
  like 429/5xx. Final failures raise `ProviderError(status=...)`.
- **Estimate.** `Experiment.estimate` returns `{arm, agents, llm_agents, rounds, calls, usd,
  by_model, budget}` for the experiment's own arm. The CLI printout is WP7's.
- **Hard ceiling inside an immediate-mode round** aborts the same way. World and board may hold
  partial in-memory changes, which are irrelevant since the run stops and resume restores the
  snapshot.
- **TaskGroup.** Round-end turns now run in an `asyncio.TaskGroup`. A runner bug inside a turn
  task (not a participant exception, which is still caught) now surfaces as an `ExceptionGroup`.

## Changes outside WP6 files

- `swarmlab/events.py` (WP1): the new event classes and fields above. `tests/test_events.py`
  covers them.
- `tests/test_agent_tools.py`: the public surface of `AgentTools` now includes `infer`.
- `tests/helpers.py`: `FakeLLM`, `TwiceInferrer`, `llm_experiment`, `TEST_PRICING`.
- `RunSpec` has a new `providers` field (default `{}`). It is part of the spec hash, so hashes of
  new runs differ from M1a runs of the same configuration. Old `run.json` files still load.
- The CLI has no `--budget` flag on `resume` yet (`Run.resume(budget=...)` only).

## Follow-ups from the first external user (field notes 2026-10-07, branch `field-notes-cli`)

- Errored turns: `run.json` and `Run.summary()` carry `turns_total`, `turns_errored`,
  `first_error`, and `health: "degraded"` when more than half the turns ended `error`; the CLI
  warns, `run` reports the run `errored` and exits 1 (swarmlab/runner.py `TurnTally`).
- `swarmlab preflight SPEC --arm A` (swarmlab/preflight.py): one real request per LLM group with
  the arm's exact `extra` and tools, straight to the provider (no gate, nothing on disk),
  capped by `--max-usd`. Cerebras rejects `chat_template_kwargs`; use
  `extra: {reasoning_effort: "none"}` there (swarmlab/providers/openai_compat.py docstring).
- `budget.total_usd` is enforced over `<out>/<experiment>.ledger.jsonl`
  (`budget.ExperimentLedger`), across processes and `run --parallel N`; new end reason
  `total_budget`. Arms on `fake:` models only are exempt.
- `estimate --from RUN_DIR` prices a spec with `Run.measured()`; `Experiment.estimate` takes
  float figures and `probe_call_usd`.

