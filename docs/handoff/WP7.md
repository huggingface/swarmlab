# WP7 handoff: LLM participant, probes, probe metrics, real-provider smoke harness

Branch `wp7`. `pytest -q -W error` passes 279 tests in about 60 s. `ruff check swarmlab tests
examples tools` is clean. No test calls a real provider. `tools/real_smoke.py` has **not** been
run.

## What landed

| file | public names |
|---|---|
| `swarmlab/participants/llm.py` | `LLMAgent` ("llm"), `render_system_prompt`, `json_protocol_text`, `parse_tool_json`, `ToolJsonError`, `RESULTS_PREFIX`, `DEFAULT_SYSTEM_TEMPLATE` |
| `swarmlab/participants/prompts/default_system.j2` | the default system prompt. It ships in the wheel (checked with `uv build`), and `artifacts` is listed in the hatch config. |
| `swarmlab/probes.py` | `Probe`, `BeliefProbe` ("belief"), `build_probe`, `probe_messages`, `parse_json_object`, `BELIEF_QUESTION`, `CODER_SYSTEM` |
| `swarmlab/runner.py` | probe hook `_probe_round`/`_probe_one` (after the world/board commit, before metrics), `hard_ceiling` end after a probe, `View.description` filled |
| `swarmlab/metrics/belief.py` | `source="world" \| "probe:<name>"` on all four belief metrics; the name gets the suffix `@probe:<name>` |
| `swarmlab/world/base.py`, `world/flaggame.py` | `World.description() -> str` (default `""`); FlagGame's 4-sentence task text |
| `swarmlab/view.py` | `View.description: str = ""` |
| `swarmlab/spec.py` | `RunSpec.probes`, arm-level YAML `probes:`; `spec_hash` omits an empty `probes` |
| `swarmlab/experiment.py` | `Experiment.probes` built/validated, coder model priced at construction, `estimate` counts probes and per-agent `max_calls`, `Run.probes` |
| `swarmlab/cli.py` | estimate printout in `run`, `swarmlab estimate`, `resume --budget-soft/--budget-hard/--budget-measurement` |
| `pyproject.toml` | `llm` participant entry point, `swarmlab.probes` group with `belief`, hatch `artifacts` |
| `examples/04_llm_flaggame.py`, `examples/flaggame_m1a.yaml` (`llm` arm) | fake provider by default; switching to Haiku is a one-line change |
| `tools/real_smoke.py` | §9 item 7 harness (`SWARMLAB_REAL=1` only) |

Tests:
- `test_llm_agent.py` (15): native and json loops, json retry, tolerant parsing, `end_turn` first, own `max_calls` and the runner cap, bad native args, window memory, image passthrough, snapshot/restore and fork mid-run, prompt contract, overrides.
- `test_probes.py` (12): acceptance 4 and 5, scripted skip once per run (also across a fork), probe hard ceiling and resume, coder model, `every`/names, metric source rules, YAML round trip and spec hash, determinism.
- `test_llm_acceptance.py` (5): acceptance 1 with 8 x `LLMAgent("fake:reader")` and 6 rounds (two identical runs, replay with zero `FakeProvider.complete` calls, SIGKILL during round 4's commit then resume with every round-4 swarm request cached), and acceptance 6 (isolation over every view, request, tool result and probe question; delay policy; push delivery).
- `test_cli_budget.py` (3).

Acceptance status:

| item | status |
|---|---|
| 1 | Passes with `LLMAgent`. The WP6 version with `FakeLLM` is still there too. |
| 2, 3 | Unchanged from WP6. They pass. |
| 4 | Passes. |
| 5 | Passes. |
| 6 | Passes. |
| 7 | Script delivered, not run. |

## Default system prompt

Template `swarmlab/participants/prompts/default_system.j2`. Below it is rendered for agent a000 on the default FlagGame with the board tools:

```
You are agent a000 with the role "worker" in a multi-agent environment.

Task: There are 8 candidate flags, each a grid of colour letters, and exactly one of them is the hidden flag. You privately see a 3x4 crop of the hidden flag at an undisclosed position, and more than one candidate may contain your crop. Record which candidate you believe is the hidden flag with the `guess` tool; only your latest guess counts and you may change it in any round. You are never told whether a guess is right.

The environment runs in rounds. At the start of each round you receive one message with the round number, your current observation, the outcomes of your actions in the previous round, and any items delivered to you. You act only by calling tools; each tool call returns a result to you, and you may make several calls in one turn. Call `end_turn` when you are done for this round; a reply without any tool call also ends your turn.

Tools:
- guess: Record your current guess of which candidate the flag is
- my_status: Status of your own work.
- collective_status: Status of the swarm's work.
- read_board: Read unread messages delivered to your inbox (oldest first).
- post: Post a message to a board channel.
- end_turn: Finish your turn for this round.
```

It says nothing about reading the board or collaborating. A test enforces this. Under `tool_protocol="json"` the code appends a fixed "Tool protocol" section with each tool's JSON parameter schema.

## Decisions where the contract was silent

- **World description reaches the agent through `View.description`.** The runner fills it from `world.description()` every turn. The prompt is rendered once, at the agent's first turn, and kept in `self.system`. FlagGame's text avoids the words "correct" and "truth" because the Enumerator's leak check scans views for them.
- **Round message.** It is a list of parts:
  - `Round {r}.`
  - the observation parts, unchanged
  - `Outcomes of your actions last round:` with lines `- <tool> <action_id>: accepted|rejected <feedback JSON>`
  - `Delivered to you:` with lines `- <post_id> (round <eligible>): <content>`

  The last two parts appear only when they have content.
- **Tool results.**
  - Native: one `tool` message per call, with content `{"ok", "result", "error"}` as JSON.
  - Native call whose arguments did not parse (`{"_raw": ...}`): it is not executed, and the error goes back to the model.
  - JSON protocol: results come back as one `user` message that starts with `[tool results]` and holds a JSON list. Orphan `tool` messages without provider call ids would be rejected by Anthropic and OpenAI. So the contract's "parse error fed back as a tool message" is delivered this way too.
  - A JSON reply with no `[`/`{` at all ends the turn as `no_tool`.
  - A malformed JSON reply gets one error-feedback retry per turn. A second one ends the turn.
- **`max_calls` counts model calls.** The runner cap counts tool calls. If a turn is cut off mid-response (cap or exception), every unanswered call still gets an error result, so the stored conversation stays valid for the next round. `TurnUsage.calls` is the number of executed tool calls.
- **Window memory.** Rounds before r-window_rounds+1 are dropped before round r's message is added. A probe after round r therefore sees exactly `window_rounds` rounds.
- **Probe placement.** Probes run after `action_committed*` and before `metric*`/`budget`/`round_committed`, in the same round. This means:
  - probe metrics for round r reflect round r's answers
  - replay folds them
  - a crash discards them with the round, and resume re-asks them (from the cache if they were already paid)
- **Probe concurrency.** The agents of one probe are probed concurrently, and the events are logged in seeded order.
- **Probe context is flattened.** `probe_messages` renders assistant tool calls as `[tool call] name {args}` text and tool messages as `[tool result] ...` user text. The probe sends `tools=[]`, and Anthropic rejects `tool_use`/`tool_result` blocks when no tools are defined. The agent's memory is never touched; a test asserts its snapshot bytes before and after each probe.
- **Probe event fields.**
  - `question_hash` and `raw_hash` are blob shas of the question and the raw answer.
  - `cost_usd` is the nominal cost: the original cost even on a cache hit, so resumed and uninterrupted runs have equal logical views.
  - Coder-extracted answers carry `parsed["coded"] = True`.
  - The coder call is `CODER_SYSTEM` + `Question/Reply`, with `max_tokens=256` and `temperature=0`.
- **Budget during probes.**
  - `MeasurementBudgetReached` logs `skipped: measurement_budget` for that agent and stops probing for the rest of the process's run. After a resume, probing is attempted again, so a larger budget resumes it.
  - `HardCeilingReached` from a probe logs `skipped: hard_ceiling`. The round still commits (its turns and world commit are complete), then the run ends with `run_ended(hard_ceiling)` at round r. Resume continues at r + 1 and does not retry round r's skipped probes.
- **Scripted agents.** A participant without a callable `probe_context` gets one `skipped: no_context` event per probe, per agent, per run, in the first probed round. On resume and fork this set is re-derived from the log.
- **Probe metric rules.**
  - `ok` with a string candidate sets the agent's belief.
  - Any other non-skipped answer resets it to `none`.
  - Skipped events leave it unchanged.
  - Scripted agents count as `none` in probe metrics, because they are live agents in the denominator.
- **Metric identity.** `source="world"` is dropped from `params`, so M1a metric specs and spec hashes are unchanged. An empty `probes` is likewise left out of `spec_hash`. The runner already keyed metrics by `name`, so two `belief.consensus` entry points with different `source` coexist. No consumer assumed unique entry-point names (CLI `validate` and the viewer use names).
- **Estimate.**
  - Each LLM agent's `calls_per_turn` is capped at its own `max_calls`.
  - Probes add one call per probed agent per probed round, priced like a turn call (`probe_calls`, `measurement_usd`, included in `usd`).
  - The CLI's `run` printout and `swarmlab estimate` default to `calls_per_turn = options.max_calls_per_turn` (the true worst case). The printout goes to stdout, or to stderr under `--json`.
- **YAML `probes:`** is per arm, a list of `{type, params}` or bare names.

## Changes outside WP7 files

- `swarmlab/providers/fake.py` (WP6): `flaggame_reader` ignores `[tool results]` user messages when it finds the start of the turn, so it plays the json protocol correctly.
- `tests/helpers.py`:
  - `PausingFlagGame` takes `SWARMLAB_TEST_PAUSE_AT` (default 7)
  - new fixtures `llm_agent_experiment`, `ImageFlagGame` and `script_*` fake scripts
- `tests/test_cli.py`: the example YAML now has a third arm, `llm`.

## Real-provider smoke (operator approval required)

```
cd $AM_LOCAL/git/swarmlab-wp7 && export UV_PROJECT_ENVIRONMENT=$AM_LOCAL/envs/swarmlab-wp7
SWARMLAB_REAL=1 uv run python tools/real_smoke.py            # --out runs/real_smoke --seed 1
```

Without `SWARMLAB_REAL=1` it prints the estimate and exits.

Setup: N=4, 3 rounds, broadcast, `LLMAgent(max_tokens=1024, max_calls=6)`, a BeliefProbe every round, and `Budget(soft_usd=0.40, hard_usd=0.50, measurement_usd=0.10)` per arm.

Worst case from `Experiment.estimate(seed=1, max_rounds=3, calls_per_turn=6)`, assuming 3000 prompt and 300 completion tokens per call:

| arm | estimate | calls |
|---|---|---|
| `haiku` (`anthropic:claude-haiku-4-5`) | $0.378 | 72 turn calls + 12 probe calls |
| `qwen` | $0.029 | 72 turn calls + 12 probe calls |
| total | $0.407 | |

The hard ceilings cap the absolute spend at $1.00 (2 x $0.50).

**Qwen choice: `hf:Qwen/Qwen3.5-9B:deepinfra`.** Picked from `GET https://router.huggingface.co/v1/models` on 2026-10-06.
- It is the smallest Qwen3.x of at least 7B with `supports_tools: true` on DeepInfra.
- `Qwen/Qwen3-8B` is 1B smaller but is tool-listed only on nscale.
- It is pinned to DeepInfra with the router's `model:provider` suffix.
- Pricing comes from the router's listed DeepInfra price: input 0.10 and output 0.15 USD per M tokens. No cached-input price is listed, so cached tokens are priced at the input price: `(0.10, 0.15, 0.10)`.

The script prints, per arm:
- spend
- cost per turn
- tool-call success rate and calls by tool
- yield kinds
- guesses accepted
- probe ok rate
- `bad_tool_args` and error counts, and `served_by`
- world and probe accuracy by round
- "native tool protocol worked": at least one tool call returned ok, at least one guess was accepted, and no response had unparseable arguments

## Open points for the operator / next WP

- Qwen3.5 may emit `<think>` text. Both parsers strip `<think>` blocks. `LLMAgent` has no `extra` passthrough, so thinking cannot be turned off from the spec yet (for example `chat_template_kwargs`).
- With `thinking_budget` on Haiku, Anthropic expects thinking blocks to be preserved across tool-use turns. Neither `ChatResponse` nor the adapter carries thinking blocks yet (WP6 scope). Leave `thinking_budget=None` for the smoke.
- `Experiment.estimate`'s 3000-token prompt assumption understates full-memory agents in long runs. Use the hard ceiling as the real guard.
