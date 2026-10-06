# swarmlab M1b interface contract

Scope (DESIGN.md Build plan, M1b): the first model-spending slice. Provider layer with budget enforcement, inference through the harness (`tools.infer`), the in-process LLM participant, probes answered by the agent's own model, probe-sourced belief metrics, and a broadcast-versus-gossip comparison at small N. Builds on the M1a contract (`INTERFACE.md`); everything there stays binding. Nothing in M1b spends money unless the operator runs the explicitly marked real-provider smoke.

Conventions as in M1a. New package modules:

```
swarmlab/
  providers/__init__.py   registry: resolve("anthropic:claude-haiku-4-5") -> (Provider, model_id)
  providers/base.py       ChatMessage, ChatRequest, ChatResponse, Usage, Provider base, request_hash()
  providers/anthropic.py  AnthropicProvider (official `anthropic` SDK, async client)
  providers/openai_compat.py  OpenAICompatProvider via httpx; presets hf, openai, vllm
  providers/fake.py       FakeProvider (deterministic, scripted)
  budget.py               Ledger, Gate, HardCeilingReached, SoftBudgetReached
  participants/llm.py     LLMAgent
  probes.py               Probe base, BeliefProbe
  metrics/belief.py       gains `source` param ("world" | "probe:<name>")
```

## 1. Provider layer

```python
class Usage(BaseModel): prompt_tokens: int = 0; completion_tokens: int = 0; cached_prompt_tokens: int = 0; reasoning_tokens: int = 0
class ChatMessage(BaseModel):
    role: Literal["system", "user", "assistant", "tool"]
    content: list[Part] | str                 # Part from view.py: text or image_png_b64
    tool_calls: list[ToolCall] | None = None  # assistant messages
    tool_call_id: str | None = None           # tool messages
class ChatRequest(BaseModel):
    model: str                                # "<provider>:<model id>", e.g. "anthropic:claude-haiku-4-5", "hf:Qwen/Qwen3.5-9B"
    messages: list[ChatMessage]
    tools: list[ToolSchema] = []
    tool_protocol: Literal["native", "json"] = "native"
    max_tokens: int = 1024
    temperature: float | None = None; top_p: float | None = None; seed: int | None = None
    thinking_budget: int | None = None        # Anthropic models that take budget_tokens (Haiku 4.5); ignored elsewhere
    extra: dict = {}                          # provider passthrough
class ChatResponse(BaseModel):
    text: str; tool_calls: list[ToolCall]; usage: Usage; cost_usd: float
    provider: str; model: str; served_by: str | None = None     # e.g. HF router's x-inference-provider
    latency_s: float; finish_reason: str; cached: bool = False

class Provider(Plugin):
    name: str                                  # prefix: anthropic | hf | openai | vllm | fake
    pricing: dict[str, tuple[float, float, float]]   # model -> (usd per M prompt, per M completion, per M cached prompt)
    concurrency: int = 8
    async def complete(self, request: ChatRequest) -> ChatResponse: ...
    def estimate_prompt_tokens(self, request) -> int    # chars/4 fallback; providers may override with a tokenizer
    def max_cost(self, request) -> float               # estimate_prompt_tokens * p_in + max_tokens * p_out (+ thinking_budget * p_out)
    def cost(self, request, usage) -> float
```
`request_hash(request) -> sha256` over canonical JSON of the request with `extra` included. Unknown model pricing is an error at `Experiment` construction, not at call time.

- **AnthropicProvider**: official `anthropic` SDK, `AsyncAnthropic`; API key from `ANTHROPIC_API_KEY`, falling back to `ANTHROPIC_KEY` (this environment's variable). Tools map to Anthropic tool definitions with `strict: true` where the schema allows; `tool_choice` is always `auto` (forced tool choice is rejected on current models). Images as base64 image blocks. `thinking_budget` maps to `thinking={"type":"enabled","budget_tokens":N}` only for models that take it (Haiku 4.5); omitted otherwise. Pricing table seeded with `claude-haiku-4-5: (1.00, 5.00, 0.10)`; other Claude models may be added from the API reference, never guessed. Error handling: a typed chain (rate limit -> retry with backoff up to 3 times; other API status errors -> raise). Streaming is not required in M1b (short outputs).
- **OpenAICompatProvider(name, base_url, api_key_env, pricing, concurrency)** over `httpx.AsyncClient`: chat-completions with `tools` and `tool_choice="auto"` when `tool_protocol == "native"`, images as `image_url` data URLs, `seed` passed when given, records `x-inference-provider` (HF router) as `served_by`. Presets: `hf` (`https://router.huggingface.co/v1`, `HF_TOKEN`), `openai` (`https://api.openai.com/v1`, `OPENAI_API_KEY`), `vllm` (`base_url` required, no key). Retries 429/5xx with backoff up to 3 times.
- **FakeProvider(script)**: deterministic. `script` is an entry-point name or `module:function` of `(request: ChatRequest, rng: random.Random) -> ChatResponse`; the rng is derived per call from the request hash so the fake is a pure function of the request. Pricing `(1.0, 5.0, 0.1)` so budget tests have real numbers. Ships with `scripts.flaggame_reader`: reads the board, guesses the candidate consistent with the most crops it can parse from the conversation, calls `end_turn`.

## 2. Budget: ledger and gate

```python
class Ledger(Persistable):
    spent: dict[Literal["swarm", "measurement"], float]; reserved: float; calls: int
    def snapshot()/restore()                   # part of every run snapshot under plugins["ledger"]
class Gate:
    def __init__(self, providers: dict[str, Provider], ledger: Ledger, budget: Budget)
    @asynccontextmanager
    async def admit(self, request, category) -> yields reservation
        # reserve provider.max_cost(request); if ledger.spent_total + ledger.reserved > budget.hard_usd -> raise HardCeilingReached (nothing reserved)
        # measurement category is additionally bounded by budget.measurement_usd
        # per-provider asyncio.Semaphore(provider.concurrency)
        # on exit: release reservation, charge actual cost to the category
```
Runner rules: before starting a round, if `ledger.spent["swarm"] >= budget.soft_usd` (and `soft_usd > 0`) the run ends with `soft_budget` at that boundary. `HardCeilingReached` anywhere in a round aborts the round: the runner cancels the other turns, discards the round's buffers, writes `run_ended(reason="hard_ceiling")`, keeps the ledger (spend already charged) and stops; the run is resumable with `Run.resume(budget=Budget(...))`, which overrides the budget in `run.json` and logs a `budget_changed` event. A `budget` event (logical) is written at every commit with `spent_swarm, spent_measurement, reserved, calls`. `Budget` with all zeros means "no enforcement" (dry runs and fake providers still account).

## 3. Inference through the harness

`AgentTools.infer(request: ChatRequest, *, category="swarm") -> ChatResponse`. The executor:
1. assigns `call_id`, computes `h = request_hash(request)`, stores the request body as a blob;
2. **cache**: if `cache/<h>` exists in the run's blob store (same run id; a fork's dir starts with the parent's cache for rounds <= fork round only), returns the stored response with `cached=True`, logs `inference_attempt` and `inference_response(cached=True, cost_usd=0)`, and charges nothing;
3. otherwise logs `inference_attempt` **immediately** (operational, via `Runner.log_operational`), enters `gate.admit`, calls `provider.complete`, logs `inference_response`, stores the response blob and the cache entry, charges the ledger;
4. accumulates `Usage` and cost per agent per turn; the runner uses the executor's totals for `turn_ended.usage` (the participant's returned `TurnUsage` is merged, not trusted alone).

`replay()` must never call a provider: every request in a recorded run is in the cache by construction. `resume()` after a crash re-runs the discarded round; its requests hit the cache where identical, so recovery costs nothing for turns that already completed.

Events added: `inference_attempt{call_id, provider, model, request_hash, reserved_usd, category}`, `inference_response{call_id, response_hash, usage, cost_usd, latency_s, served_by, finish_reason, cached}` (both operational); `budget{spent_swarm, spent_measurement, reserved, calls}` and `budget_changed{old, new}` (logical); `probe{probe, question_hash, raw_hash, parsed, ok, cost_usd}` (logical).

## 4. LLMAgent

```python
class LLMAgent(Participant):   entry_point = "llm"
    def __init__(self, model: str, system_prompt: str | None = None, memory: Literal["full","window"] = "full",
                 window_rounds: int = 3, max_tokens: int = 1024, temperature: float | None = None,
                 tool_protocol: Literal["native","json"] = "native", thinking_budget: int | None = None,
                 max_calls: int | None = None, role: str = "worker")
```
- **System prompt**: Jinja2 template; the default template (`participants/prompts/default_system.j2`) states the agent's id and role, that it acts in rounds, that actions are tool calls, that `end_turn` ends its turn, and lists the tools from `view.tools` with descriptions. It must not instruct the agent to read the board or to collaborate; that is the experiment's business. `system_prompt` overrides the template text (a string, or `file:<path>`). The rendered prompt is a spec parameter by construction (it is in `params`).
- **Round message**: one user message per round with: `Round {r}.`, the observation parts (text and image parts passed through), the previous round's outcomes rendered as short lines, pushed inbox items if any. Nothing else.
- **Loop**: `infer` -> if `tool_calls`: execute each in order through `tools.call`, append the assistant message and one tool message per result, continue; if no tool calls: the turn ends (`no_tool`). If `end_turn` was among the calls, finish executing that response's calls, then return. Own `max_calls` (if set) stops the loop before the runner's cap; the runner's cap still raises `TurnCapReached`.
- **Tool protocol**: `native` uses the provider's tool API. `json` adds an instruction to the system prompt to answer with a JSON array `[{"name":..., "args":{...}}]`, parses tolerant JSON from the text, and feeds a parse error back as a tool message so the model can retry once per cap.
- **Memory**: `full` keeps every message. `window` keeps the system prompt plus the last `window_rounds` rounds of messages; the observation is re-sent each round anyway, which is the "environment is the memory" policy. The messages list is plain data so `Persistable` works unchanged.
- **Probing support**: `probe_context() -> list[ChatMessage]` returns the current messages (system included); `model_request_defaults() -> dict` returns model, temperature, max_tokens, thinking_budget so probes run on the agent's own model and settings.

Worlds gain an optional `description() -> str` (default `""`) used by the default system prompt to describe the task; FlagGame implements it.

## 5. Probes

```python
class Probe(Plugin):
    name: str; every: int = 1
    def question(self, agent, round) -> str                      # Jinja2 rendered
    def parse(self, text) -> tuple[bool, dict]                   # (ok, parsed)
    def coder_model(self) -> str | None                          # cheap model for free-text parsing; None = parse locally
class BeliefProbe(Probe):   entry_point = "belief"
    # question: "Which candidate do you currently believe the flag is? Answer with JSON {"candidate": "<name>", "confidence": <0..1>} and nothing else."
    # parse: tolerant JSON; ok requires a candidate string
```
Runner, after commit, for every live agent whose participant implements `probe_context()` (scripted agents are skipped and a `probe` event with `ok=False, parsed={"skipped": "no_context"}` is written once per run per agent): build `ChatRequest(messages=context + [user(question)], tools=[], **model_request_defaults())`, call `infer(category="measurement")`, parse, log `probe`. The participant's memory is not modified (assert in tests). If `coder_model` is set and local parsing fails, a second `infer` on the coder model with a fixed extraction prompt is made, also under `measurement`.

## 6. Metrics from probes

Belief metrics take `source: str = "world"`. `"world"` reads committed `guess` actions as in M1a; `"probe:<name>"` reads `probe` events of that probe, using `parsed["candidate"]`. Metric `name` becomes `belief.consensus` for world and `belief.consensus@probe:<name>` for probes, so both can run side by side. Denominator rules as in M1a; a failed parse counts as `none`.

## 7. Spec and YAML

- `ParticipantGroup.type = "llm"` with `params` as the constructor kwargs. The `model` string's prefix selects the provider; an `Experiment.providers: dict[str, Provider] | None` field lets a script override presets (for example a vLLM base url); YAML gets a top-level `providers:` map of `{prefix: {type, params}}`.
- `Budget` is enforced as in §2. YAML `budget:` keys `soft_usd, hard_usd, measurement_usd`.
- `Arm.probes: list[PluginSpec]` is now honoured.

## 8. Public API additions

`Run.spend -> dict` (swarm, measurement, reserved, calls); `Run.resume(budget=None)`; `Run.probes -> dict[name, list[(round, agent, parsed, ok)]]`; `Experiment.estimate(seed, max_rounds, calls_per_turn=2, prompt_tokens=3000, completion_tokens=300) -> dict` giving a rough worst-case dollar figure per arm from the pricing table, printed by the CLI before any run whose budget is non-zero.

## 9. Acceptance tests (M1b)

1. **Fake LLM determinism and replay**: 8 `LLMAgent(model="fake:reader")` agents, 6 rounds; two runs identical in logical view; `Run.load(dir).replay()` makes zero `FakeProvider.complete` calls (counter) and reproduces metrics; `resume()` after SIGKILL mid-round serves the completed turns' requests from cache.
2. **Hard ceiling**: `Budget(hard_usd=tiny)` ends the run with `hard_ceiling` mid-round; no `post`/`action_committed` from that round; the ledger shows the spend; `Run.resume(budget=Budget(hard_usd=large))` continues and finishes.
3. **Soft budget**: `Budget(soft_usd=x)` ends at a round boundary with `soft_budget`; the last round is fully committed.
4. **Probes**: `BeliefProbe` every round on fake LLM agents: one `probe` event per agent per round, charged to `measurement`, participant messages byte-identical before and after probing; `belief.consensus@probe:belief` present alongside `belief.consensus`.
5. **Measurement cap**: `measurement_usd` exhausted -> probes stop with a logged `ok=False, parsed={"skipped":"measurement_budget"}`, the swarm continues.
6. **Isolation and delivery** tests from M1a keep passing with `LLMAgent` on the fake provider.
7. **Real-provider smoke** (not in pytest; `tools/real_smoke.py`, runs only with `SWARMLAB_REAL=1`): N=4, 3 rounds, broadcast, `anthropic:claude-haiku-4-5` and one Qwen on the HF router that supports native tool calling (query `/v1/models` and pick the smallest Qwen3.x >= 7B that advertises tools; record the choice), `Budget(soft_usd=0.40, hard_usd=0.50, measurement_usd=0.10)` each; prints cost per turn, tool-call success rate, and whether the native protocol worked for each model. Operator approval is required before running it.

## 10. Work packages

- **WP6 providers and budget**: `providers/*`, `budget.py`, `infer` on `AgentTools`/executor, cache, new events, runner budget enforcement and resume-with-budget, `Experiment.providers` and `estimate`, YAML `providers:`/`budget:`; acceptance 2, 3, and the cache part of 1 using a minimal fake participant that calls `tools.infer` directly. Loads the `claude-api` skill and reads its Python README and tool-use file before writing the Anthropic adapter.
- **WP7 LLM participant and probes**: `participants/llm.py` and prompt template, `World.description`, `probes.py`, probe runner hook, belief metrics `source`, `Run.probes`, CLI estimate printout, `tools/real_smoke.py`, example `examples/04_llm_flaggame.py` and YAML arm; acceptance 1, 4, 5, 6.

WP7 starts after WP6 lands.
