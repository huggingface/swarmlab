# swarmlab

A scientific testbed for finding the primitives that make heterogeneous groups of LLM agents collaborate well or badly. An experiment is a **World** (the task), a set of **Participants**, a **Medium** (a board with a topology and visibility policies) and **Metrics**; the framework handles tool dispatch, ordering, recording, budgets, snapshots, replay, resume and fork, so a scientist changes the hypothesis without touching the execution machinery. Status (2026-10-07): everything in `docs/DESIGN.md` is built except the image variant of the Flag Game; see "What exists" and "Known issues" below.

## Quickstart

```
git clone https://github.com/cmpatino/swarmlab && cd swarmlab
uv sync --extra dev --extra anthropic     # or: pip install -e .[anthropic]
uv run swarmlab doctor                    # Python, extras, API keys, provider reachability, git

mkdir ~/demo-exp && cd ~/demo-exp         # experiments live outside the repo checkout
uv run --project ~/swarmlab swarmlab init demo      # writes demo.yaml and demo.py
uv run --project ~/swarmlab swarmlab run demo.yaml  # every arm x every seed -> ./runs/<run_id>/
uv run --project ~/swarmlab swarmlab view runs/demo__gossip__s1   # writes the replay page view.html
```
(`~/swarmlab` is wherever you cloned the repo. Alternatively `pip install -e ~/swarmlab[anthropic]` into the project's own environment and call `swarmlab` directly.) Work from a project directory outside the checkout: `run` writes `runs/` under the current directory, and inside the swarmlab checkout it refuses unless you pass `--out`, so runs never end up in the repo. `swarmlab validate demo.yaml` prints each arm's agents, models, caps, probes and metrics. The starter compares a broadcast board with a gossip board on the Flag Game, with 8 LLM agents per arm on the deterministic fake model `fake:reader`: no keys, no network, nothing billed. `run` prints the worst-case estimate per arm and in total, runs each arm with each seed in `seeds:`, skips runs whose directory already holds the same spec (`--rerun` writes `runs/<run_id>__r2`), and ends with a table:

```
run                  outcome  end         score                                 spend
demo__broadcast__s1  ran      max_rounds  accuracy=1, n_guessed=8, truth=G      $0.2577 (simulated)
demo__broadcast__s2  ran      max_rounds  accuracy=1, n_guessed=8, truth=H      $0.2577 (simulated)
demo__gossip__s1     ran      max_rounds  accuracy=0.625, n_guessed=8, truth=G  $0.2383 (simulated)
...
```
`(simulated)` marks runs on `fake:` models only: the spend is computed from a nominal price table and nothing is billed.
`--arm A` and `--seed N` narrow it; `--json` prints one JSON object. `demo.py` is the same experiment in Python (`python demo.py`).

## What exists

- **Worlds**: `flaggame` (belief dynamics: hidden flag, one crop per agent, `guess` action; text crops only), `coloring` (allocation: a grid to paint, with claims and a registry). New worlds subclass `World` (example 3).
- **Topologies** (`medium: {topology: ...}`): `broadcast`, `gossip`, `groups`, `tree` (coordinators over worker groups, `params: {groups: 3}`).
- **Policies**: `delay` (messages arrive k rounds late) and your own `Policy` subclasses (example 2). **Registry** with `advisory` or `enforced` claim policies for the coloring task.
- **Participants**: scripted (`evidence_aggregator`, `enumerator`, `silent`, `row_major_painter`, `queue_painter`, `random_painter`) and `llm` (`LLMAgent`: memory window, `context_limit_tokens` with `drop_oldest`/`summarize`/`fail_turn` overflow, prompt params, `max_calls`, `extra` passthrough).
- **Providers** (model id prefix): `anthropic`, `hf` (HF router), `openai`, `vllm` (self-hosted, normally via `job run`), `fake` (`fake:reader`, `fake:painter`: deterministic, no network).
- **Probes**: `belief` (each agent's own model answers "which candidate?" every round, out of band, billed to `measurement_usd`).
- **Interventions** (`interventions:` in a spec, triggered by `at_round`, `every` or a metric `when`): `inject_post`, `delay_delivery`, `mute`, `kill_agents`, `patch_private`, `reconfigure`. **Paired runs** from round 0: `Experiment.pair(...)` with unchanged-pair controls.
- **Metrics**: `belief.consensus`, `belief.accuracy`, `belief.polarization`, `belief.entropy` (from world guesses, or from probes with `params: {source: "probe:belief"}`); `comm.read_rate`, `comm.post_rate` (share of turns with a post), `comm.posts_per_round` (posts per agent per round), `comm.posts_total` (swarm-wide posts per round), `comm.hops`; `coloring.coverage`, `coloring.duplicate_paints`, `coloring.wrong_paints`, `coloring.parallel_efficiency`; `claims.violations`, `claims.held`. `swarmlab metrics` lists them all with one-line descriptions.
- **Roles** (`roles:` plus `role:` per participant group): built-ins `worker`, `coordinator`, `reviewer`, `skeptic`, `scribe`; each can restrict tools, channels, registry access and world actions, add prompt text, or override the model.
- **Run control**: replay (checks metrics and score from the log), resume, fork at a round (optionally with an edited spec), budgets (soft, hard, measurement, experiment total), per-call timeout and retry.
- **Export and publish**: Parquet tables, pi-format sessions and raw logs; private Hub dataset per experiment; static `view.html`; `fetch-published` restores a run.
- **Jobs**: `swarmlab job run|status|logs|fetch` runs a spec on HF Jobs with vLLM serving the model in the same job (validated at N=256, 83 min, $3.45).

## Examples

Runnable copies live in `examples/`; each writes to `runs/<run_id>/`.

**1. Run an existing task with built-in agents and communication** (`examples/01_run_existing.py`)
```python
from swarmlab import Experiment, Board, Budget
from swarmlab.worlds import FlagGame
from swarmlab.participants import EvidenceAggregator

exp = Experiment(
    name="flag-gossip",
    world=FlagGame(n_candidates=8),
    participants=[EvidenceAggregator()] * 16,
    medium=Board(topology="gossip"),
    metrics=["belief.consensus", "belief.accuracy", "comm.read_rate"],
    budget=Budget(soft_usd=0, hard_usd=0),      # scripted agents spend nothing
)
run = exp.run(seed=3, max_rounds=20)
run.score                         # final evaluator-side score
run.metrics["belief.consensus"]   # [(round, value, denominator), ...]
run.events                        # the logical event trajectory
run.view()                        # builds and returns runs/<id>/view.html
run.fork(at_round=4).run()        # live fork; Run.load(path) replays from disk
runs = exp.run_all([1, 2, 3], max_rounds=20)   # several seeds; skips runs already on disk
[r.summary() for r in runs]       # run id, end reason, score, metrics, spend
```

**2. Change one mechanism, here message visibility** (`examples/02_change_visibility.py`)
```python
from swarmlab import Policy

class OddAgentsSeeNothing(Policy):
    def apply(self, reader, post, round):
        if int(reader[1:]) % 2:
            return None                      # withhold
        return round + 1, post.text          # available next round, unchanged

exp = Experiment(..., medium=Board(topology="broadcast", policies=[OddAgentsSeeNothing()]))
```

**3. A new task with the smallest World interface** (`examples/03_new_world.py`)
```python
from swarmlab import World, Outcome, text_observation, tool

class Counter(World):
    def reset(self, rng, agents):
        self.total = 0
    def observe(self, agent):
        return text_observation(f"total so far: {self.total}")
    @tool("add", "Add n to the shared total", {"n": "integer"})
    def add(self, agent, n: int) -> Outcome:
        self.total += n
        return Outcome(accepted=True, feedback={"added": n})
    def score(self):
        return {"total": self.total}
```
To use it from YAML, save it as `myworld.py` next to the spec and write `world: {type: "myworld:Counter"}` (participants, metrics and probes take `module:Class` the same way). Every command that loads a spec (and `Experiment.from_yaml`) puts the spec's directory at the front of `sys.path` first, so no `PYTHONPATH` is needed; `run.json` records that directory, so `replay`, `resume`, `fork` and `view` of the run dir find the module from any working directory.

**4. LLM agents with a belief probe** (`examples/04_llm_flaggame.py`): `LLMAgent(model="fake:reader")` on the Flag Game with `BeliefProbe()`; set `MODEL = "anthropic:claude-haiku-4-5"` for a real run.

## Switching to real models

- **Model ids** are `provider:model[:served_by]`: `anthropic:claude-haiku-4-5`, `hf:Qwen/Qwen3.5-9B:deepinfra` (the HF router, pinned to DeepInfra; without `:served_by` the router picks), `openai:<model>`, `vllm:<model>` (with a `providers:` base url). Put it in `model:` of each `llm` participant group.
- **Keys** come from the environment: `ANTHROPIC_API_KEY` (or `ANTHROPIC_KEY`), `HF_TOKEN`, `OPENAI_API_KEY`. `swarmlab doctor demo.yaml` checks the keys and extras that spec needs and that the endpoints answer.
- **Prices** are never typed by hand. `swarmlab models [--provider hf|anthropic] [--tools] [--search qwen]` lists models, who serves them, tool support and USD per million tokens. `hf` prices come from the router listing (cached 24 h in `~/.cache/swarmlab/catalog.json`); a bare `hf:Org/Model` is priced at its most expensive listed provider. The price used is written into the run spec. To override, or for a model the catalog does not price, add `providers: {hf: {type: openai_compat, params: {name: hf, pricing: {"Org/Model:prov": [in, out, cached]}}}}`.
- **Timeouts**: each provider call attempt is cut off after `timeout_s` (default 90 s) and timeouts, connection errors, 429 and 5xx are retried up to `max_retries` times (default 2) with jittered 1/2/4 s backoff; the budget reservation is held across retries. A phase-commit round waits for its slowest call, so tighten these for slow-tailed routers: `providers: {hf: {type: openai_compat, params: {name: hf, timeout_s: 60, max_retries: 3}}}` (`type: anthropic` takes the same two params).
- Each `inference_response` event records `attempts`; a call that still fails after its retries raises `ProviderError`.
- **Preflight before spending**: `swarmlab preflight spec.yaml --arm A` sends ONE real request per LLM participant group through the real provider, with the arm's exact `model`, `max_tokens`, `temperature` and `extra`, its system prompt, the tools a run offers (world tools plus `post`, `read_board`, `end_turn`) and a one-line user message. It prints the worst case of those requests first (nothing is sent above `--max-usd`, default $0.05), then per group the model, latency, finish reason, whether a tool call parsed, reasoning tokens if the provider reports them, and on failure the provider's HTTP error text; exit 1 if any group fails or parses no tool call. A provider that rejects a body field otherwise shows up only as a run in which every turn errored.
- **Errored runs are loud**: `run.json` and every run summary carry `turns_total` and `turns_errored` (turns that ended with an exception, e.g. a provider error) and the first error's message. When more than half the turns of a run errored, `run.json["health"]` is `"degraded"`, the CLI prints `WARNING: <run>: 36/36 turns errored (first error: ProviderError: hf: HTTP 400: ...)`, the `run` table shows the run as `errored` and `swarmlab run` exits 1. The end reason is unchanged (such a run still ends `max_rounds`, at $0).
- **Reasoning controls differ by provider.** `extra` is merged into the request body as is, so it must use the serving provider's field names. vLLM, SGLang and DeepInfra serve Qwen3 with `extra: {chat_template_kwargs: {enable_thinking: false}}`; Cerebras rejects `chat_template_kwargs` with HTTP 400 and wants `extra: {reasoning_effort: "none"}`. `Qwen/Qwen3.8-27B` on Cerebras (`hf:Qwen/Qwen3.8-27B:cerebras`) reasons by default (`reasoning_effort` defaults to `high`; accepted values `none`, `low`, `medium`, `high`), and Cerebras asks clients not to send Qwen-native fields (`disable_reasoning`, `enable_thinking`, `thinking_budget`) for it. The HF router documents `reasoning_effort` as a standard chat-completion field whose support and default depend on the provider and model. Run `preflight` after changing `extra`.
- **Budgets** (USD, per run): `soft_usd` ends the run at the next round boundary once agent spend reaches it; `hard_usd` is an absolute ceiling on agent + probe spend (reached during the agents' turns, the round in flight is discarded and the run ends `hard_ceiling`; reached by the belief probes, which run after the round has committed, the round is kept, the unanswered probes are logged as skipped and the run ends `hard_ceiling_probes`; either way its spend still counts, and `swarmlab resume RUN --budget-hard X` continues: X is the run's new total including everything already spent, or `--add-budget D` allows D more from the current spend; `resume` prints the spend so far); `measurement_usd` caps probes. A skipped probe (budget, provider error, scripted agent) is left out of the probe-sourced metrics' denominators, not counted as "none"; status lines show `probes_skipped=N (reason n)` and `swarmlab report` a `Probes skipped:` line per arm. 0 means "not enforced", so set `hard_usd` before using a paid model. These caps are per run and can be set in the top-level `budget:` and per arm (`arms.A.budget: {hard_usd: 0.5}`, merged over the top-level one, so arms can get different caps); `total_usd` (top-level `budget:` only) caps the whole experiment, across every `swarmlab run` invocation and process that writes to the same `--out`: every run appends its spend at each commit to `<out>/<experiment>.ledger.jsonl` (run id, spec hash, spend, status, time, pid), and `swarmlab run` (and `Experiment.run_all`) starts a run only if the ledger's spend plus the headroom of runs still in flight (their `hard_usd` minus what they spent) plus its own `hard_usd` fits under `total_usd`, otherwise it stops and lists the skipped runs (outcome `capped`). A running run also ends with `total_budget` at the next round boundary once the ledger's spend reaches `total_usd` (resumable like `soft_budget`). Spend added by a manual `swarmlab resume` lands in the ledger too (resume itself does not check `total_usd` before starting). Run dirs from before the ledger existed are added to it when `run` finds them; delete the file to forget past spend. `swarmlab run --parallel N` runs up to N runs at once (threads in one process, each with its own experiment, providers and event loop) under the same cap. Arms that bill nothing (every model a `fake:` model, the rest scripted participants, e.g. dry runs) are exempt from `total_usd`: they need no `hard_usd`, are never refused, and their nominal spend does not count toward the total. `run` prints every arm's caps and the total cap before starting, an `existing:` line per run that already exists with its actual spend and per-round cost (not the spec's caps), and warns when `hard_usd - soft_usd` is too small for one round (the soft budget is checked between rounds, so a round can start under it, hit the hard ceiling and be discarded): when a finished run of the same arm exists under `--out`, the bound is that run's measured cost of a last round, and the warning says which run it measured; otherwise it is twice the worst-case cost of one round. When any budget is non-zero `run` asks before starting (`--yes` skips the question); `swarmlab estimate spec.yaml` prints the estimate alone (every arm x seed; `--arm`/`--seed` narrow it, `--prompt-growth TOKENS` models a context that grows each round under full memory; calls per turn default to the runner cap `max_calls_per_turn`, capped by each group's own `max_calls`, `--calls-per-turn C` assumes C instead, and the output states the value used). The worst case overstates real spend several times; `swarmlab estimate spec.yaml --from runs/RUN_ID` prices the spec with that finished run's measurements instead: its model calls per turn, its prompt tokens per call as a line fitted through its rounds (start and growth per round), its completion tokens per call and its probe cost per call.

## Spec reference

<!-- spec-reference:start -->
Every key of an experiment YAML, generated from the pydantic models (`swarmlab spec-reference` prints this list). Unknown keys are errors. Wherever the shape is `{type: NAME, params: {...}}` the value is a plugin: `{type: NAME, params: {...}}`, or the bare string `NAME` for `{type: NAME, params: {}}`.

- `name`: `str` (required): experiment name; run ids are `<name>__<arm>__s<seed>`
- `arms`: `{NAME: {world, participants, medium, metrics, probes, interventions, options, budget, roles}}` (required): the conditions, by arm name; each arm is a full setup (see `arms.<arm>`)
  - `arms.NAME.world`: `{type: NAME, params: {...}}` (required): the task, e.g. `{type: flaggame, params: {n_candidates: 8}}` or `flaggame`
  - `arms.NAME.participants`: `[{type, count, params, role}, ...]` (required): participant groups, in agent order
    - `arms.NAME.participants[].type`: `str` (required): participant type: `llm`, `evidence_aggregator`, `enumerator`, `silent`, ...
    - `arms.NAME.participants[].count`: `int` (default `1`): number of agents in this group (>= 1)
    - `arms.NAME.participants[].params`: `{...}` (optional): constructor params, e.g. `{model: "anthropic:claude-haiku-4-5", max_tokens: 1024}` for `llm`
    - `arms.NAME.participants[].role`: `str | null` (optional): role name (declared in `roles`, or a built-in: worker, coordinator, reviewer, skeptic, scribe)
  - `arms.NAME.medium`: `{topology, delivery, push_limit, policies, channels, registry, claim_policy}` (default: see keys): the message board: topology, policies, registry
    - `arms.NAME.medium.topology`: `{type: NAME, params: {...}}` (default `broadcast`): who receives a post: `broadcast`, `gossip` (params `{k: 1}`: partners per agent per round), `groups`, `tree`; e.g. `{type: gossip, params: {k: 2}}`
    - `arms.NAME.medium.delivery`: `pull | push` (default `pull`): `pull` (agents call `read_board`) or `push` (deliveries come with the turn)
    - `arms.NAME.medium.push_limit`: `int` (default `20`): most items pushed per turn under `delivery: push`
    - `arms.NAME.medium.policies`: `[{type: NAME, params: {...}}, ...]` (optional): visibility policies applied in order, e.g. `[{type: delay, params: {rounds: 1}}]`
    - `arms.NAME.medium.channels`: `[str, ...]` (default `[main]`): board channels
    - `arms.NAME.medium.registry`: `bool` (default `false`): turn on the claim registry tools
    - `arms.NAME.medium.claim_policy`: `{type: NAME, params: {...}}` (default `advisory`): `advisory` or `enforced` (registry claims checked against world actions)
  - `arms.NAME.metrics`: `[{type: NAME, params: {...}}, ...]` (optional): metrics logged every round (`swarmlab metrics` lists them)
  - `arms.NAME.probes`: `[{type: NAME, params: {...}}, ...]` (optional): probes asked after every commit, e.g. `[belief]`
  - `arms.NAME.interventions`: `[{type: NAME, params: {...}}, ...]` (optional): interventions (`inject_post`, `mute`, ...) for this arm
  - `arms.NAME.options`: `{max_rounds, commit, max_calls_per_turn, snapshot_every, concurrency, repeat}` (optional): run options for this arm, merged over the top-level `options`
    - `arms.NAME.options.max_rounds`: `int` (required): rounds per run (required here or as `--max-rounds`)
    - `arms.NAME.options.commit`: `round_end | immediate` (default `round_end`): `round_end` (phase commit) or `immediate` (sequential)
    - `arms.NAME.options.max_calls_per_turn`: `int` (default `20`): tool calls an agent may make per turn
    - `arms.NAME.options.snapshot_every`: `int` (default `1`): write a snapshot every N rounds
    - `arms.NAME.options.concurrency`: `int` (default `32`): concurrent turns
    - `arms.NAME.options.repeat`: `int` (default `0`): paired-run repeat index (0: a plain run)
  - `arms.NAME.budget`: `{soft_usd, hard_usd, measurement_usd}` (optional): per-run caps for this arm, merged over the top-level `budget` (no `total_usd`)
    - `arms.NAME.budget.soft_usd`: `float` (default `0.0`): end the run at the next round boundary once agent spend reaches this (0: off)
    - `arms.NAME.budget.hard_usd`: `float` (default `0.0`): absolute ceiling on agent + probe spend for the run: reached mid-round, the round is discarded (end `hard_ceiling`); reached by the probes after the commit, the round is kept (end `hard_ceiling_probes`) (0: off)
    - `arms.NAME.budget.measurement_usd`: `float` (default `0.0`): cap on probe spend (0: off)
  - `arms.NAME.roles`: `{NAME: {prompt_append, system_prompt, tools, channels_read, channels_write, registry, may_act, model, budget, post_fields}}` (optional): role declarations for this arm, merged over the top-level `roles`
    - `arms.NAME.roles.NAME.prompt_append`: `str | null` (optional): text appended to the system prompt
    - `arms.NAME.roles.NAME.system_prompt`: `str | null` (optional): replaces the system prompt
    - `arms.NAME.roles.NAME.tools`: `[str, ...] | null` (optional): allowlist of tool names (null: all)
    - `arms.NAME.roles.NAME.channels_read`: `[str, ...] | null` (optional): channels the role reads (null: all)
    - `arms.NAME.roles.NAME.channels_write`: `[str, ...] | null` (optional): channels the role writes (null: all)
    - `arms.NAME.roles.NAME.registry`: `none | read | write` (default `write`): registry access: `none`, `read`, `write`
    - `arms.NAME.roles.NAME.may_act`: `bool` (default `true`): may use the world's action tools
    - `arms.NAME.roles.NAME.model`: `str | null` (optional): overrides the participant's model (`prefix:id`)
    - `arms.NAME.roles.NAME.budget`: `{...} | null` (optional): per-turn overrides: `{max_calls, max_tokens}`
    - `arms.NAME.roles.NAME.post_fields`: `{...} | null` (optional): allowed post fields and values: `{field: [values]}`
- `budget`: `{soft_usd, hard_usd, measurement_usd, total_usd}` (default: see keys): per-run caps in USD (each arm's `budget` is merged over it) plus `total_usd`, the cap for the whole experiment
  - `budget.soft_usd`: `float` (default `0.0`): end the run at the next round boundary once agent spend reaches this (0: off)
  - `budget.hard_usd`: `float` (default `0.0`): absolute ceiling on agent + probe spend for the run: reached mid-round, the round is discarded (end `hard_ceiling`); reached by the probes after the commit, the round is kept (end `hard_ceiling_probes`) (0: off)
  - `budget.measurement_usd`: `float` (default `0.0`): cap on probe spend (0: off)
  - `budget.total_usd`: `float` (default `0.0`): top level only: cap on the whole experiment's spend (all arms x seeds, finished and resumed runs included) (0: off)
- `options`: `{max_rounds, commit, max_calls_per_turn, snapshot_every, concurrency, repeat}` (optional): run options for every arm (each arm's `options` is merged over them)
  - `options.max_rounds`: `int` (required): rounds per run (required here or as `--max-rounds`)
  - `options.commit`: `round_end | immediate` (default `round_end`): `round_end` (phase commit) or `immediate` (sequential)
  - `options.max_calls_per_turn`: `int` (default `20`): tool calls an agent may make per turn
  - `options.snapshot_every`: `int` (default `1`): write a snapshot every N rounds
  - `options.concurrency`: `int` (default `32`): concurrent turns
  - `options.repeat`: `int` (default `0`): paired-run repeat index (0: a plain run)
- `providers`: `{NAME: {type: NAME, params: {...}}}` (optional): provider overrides by model prefix, e.g. `hf: {type: openai_compat, params: {name: hf, timeout_s: 60}}`
- `seeds`: `[int, ...]` (optional): seeds `swarmlab run` runs per arm (empty: `[0]`)
- `interventions`: `[{type: NAME, params: {...}}, ...]` (optional): interventions for every arm (an arm's own list is appended)
- `roles`: `{NAME: {prompt_append, system_prompt, tools, channels_read, channels_write, registry, may_act, model, budget, post_fields}}` (optional): role declarations by name, for every arm (an arm's `roles` is merged per name)
  - `roles.NAME.prompt_append`: `str | null` (optional): text appended to the system prompt
  - `roles.NAME.system_prompt`: `str | null` (optional): replaces the system prompt
  - `roles.NAME.tools`: `[str, ...] | null` (optional): allowlist of tool names (null: all)
  - `roles.NAME.channels_read`: `[str, ...] | null` (optional): channels the role reads (null: all)
  - `roles.NAME.channels_write`: `[str, ...] | null` (optional): channels the role writes (null: all)
  - `roles.NAME.registry`: `none | read | write` (default `write`): registry access: `none`, `read`, `write`
  - `roles.NAME.may_act`: `bool` (default `true`): may use the world's action tools
  - `roles.NAME.model`: `str | null` (optional): overrides the participant's model (`prefix:id`)
  - `roles.NAME.budget`: `{...} | null` (optional): per-turn overrides: `{max_calls, max_tokens}`
  - `roles.NAME.post_fields`: `{...} | null` (optional): allowed post fields and values: `{field: [values]}`

Example: a gossip arm where each agent reaches one partner per round, with its own caps:

```yaml
arms:
  gossip:
    world: flaggame
    participants:
      - {type: llm, count: 6, params: {model: "anthropic:claude-haiku-4-5"}}
    medium: {topology: {type: gossip, params: {k: 1}}}
    budget: {soft_usd: 0, hard_usd: 0.5}
```
<!-- spec-reference:end -->

A key in the wrong place or with the wrong shape is an error that names the key path and the shape expected there, e.g. `arms.gossip.medium.topology_params: unknown key 'topology_params'; arms.gossip.medium expects a mapping with keys topology, ...; topology is {type: NAME, params: {...}}, e.g. medium: {topology: {type: gossip, params: {k: 1}}}`.

## Changing what agents are told

An `llm` participant's system prompt is a Jinja2 template, rendered once per agent at its first turn; the default is `swarmlab/participants/prompts/default_system.j2` (who the agent is, the task description, how rounds and tools work, then the tool list). Two params change it, per participant group:

- `system_prompt_append: "One more sentence."` adds plain text after the task section and before `Tools:`, leaving the rest of the default prompt as is. Use it for "one sentence differs" arms. It is recorded in the run spec only when set, so arms without it keep their spec hash.
- `system_prompt: |` replaces the whole template (inline text, or `"file:prompts/mine.j2"`). Template variables: `agent` (id, e.g. `a003`), `role` (the `role` param, default `worker`), `description` (the world's task description, may be empty), `tools` (list with `.name`, `.description`, `.parameters`) and `system_prompt_append` (empty unless set; a custom template that does not use it gets the appended text at its end). Undefined variables are errors.

```yaml
participants:
  - type: llm
    count: 6
    params:
      model: "anthropic:claude-haiku-4-5"
      system_prompt_append: "A shared message board exists: `read_board` shows what other agents have posted."
```
The round message (round number, observation, last round's outcomes, deliveries) is fixed by the agent and the world. Check what each arm actually sends, with no model call, and diff two arms (the preview is round 1: the world is reset and `begin_round(1)` is called before the observation, exactly as in a run; later rounds are not shown):
```
swarmlab prompts exp.yaml --arm default > default.txt
swarmlab prompts exp.yaml --arm board > board.txt
diff default.txt board.txt      # should show only the manipulated sentence
```
Note that the default prompt already lists every tool the world and board offer (`read_board`, `post`, ...), so an "agents are told about the board" arm tests a nudge, not awareness.

## CLI

```
swarmlab doctor [SPEC...] [--offline]    swarmlab models [--provider P] [--tools] [--search S] [--refresh]
swarmlab init [NAME] [--dir D] [--force]  swarmlab validate SPEC    swarmlab spec-reference    swarmlab metrics
swarmlab run SPEC [--arm A] [--seed N] [--max-rounds R] [--out runs/] [--yes] [--rerun] [--parallel N]
swarmlab estimate SPEC [--arm A] [--seed N] [--max-rounds R] [--prompt-growth G] [--calls-per-turn C] [--from RUN_DIR]
swarmlab preflight SPEC [--arm A] [--seed N] [--max-usd 0.05]
swarmlab replay RUN_DIR
swarmlab resume RUN_DIR [--budget-hard X | --add-budget D | --budget-soft X | --budget-measurement X]
swarmlab fork RUN_DIR --at 4 [--spec edited.yaml] [--arm A] [--max-rounds R] [--out DIR]
swarmlab view RUN_DIR [--publish OWNER/REPO]
swarmlab prompts SPEC --arm A [--seed N]  swarmlab report RUNS_DIR [--out report.md] [--stdout] [--include-fake] [--title T]
swarmlab export RUN_DIR [--out DIR]
swarmlab publish RUNS_DIR_OR_RUN [--repo OWNER/REPO] [--public] [--tag T]
swarmlab fetch-published OWNER/REPO RUN_ID [--out runs/] [--force]
swarmlab job run SPEC --model M [--flavor F] [--arm A] [--seeds 1,2] [--timeout 2h] [--per-round S] [--launch]
swarmlab job status JOB_ID    swarmlab job logs JOB_ID [--follow]    swarmlab job fetch RUN_ID [--out runs/]
```
`swarmlab job ...` runs a spec whose models are `vllm:<model>` in an HF Job with vLLM serving the model in the same job, and brings run dirs back from the bucket; `job run` only prints the `hf jobs run` command and the estimate unless `--launch` (docs/handoff/WP8.md).
Every command takes `--json` and then prints one JSON object. Exit codes: 0 success, 2 invalid spec, 1 any other error (including a declined confirmation or a failed run).

## Analysing and publishing

- `swarmlab prompts SPEC --arm A` prints the system prompt and round-1 user message of one agent per participant group, exactly as the model would receive them, without calling a model. Review them (ideally with a second agent) before spending.
- `swarmlab report RUNS_DIR` writes a Markdown report over the finished runs: per-arm accuracy and consensus, trajectories, where the swarm went, probe vs world belief, reading behaviour, tool-protocol health. Simulated runs (only `fake:` models, e.g. dry runs) are listed but left out of the tables and spend totals; `--include-fake` adds them as `<arm> (simulated)`. `--out report.md` writes the file and prints one line (`--stdout` also prints the report).
- `swarmlab export RUN_DIR` (Python `Run.export(out)`) writes `RUN_DIR/export/`: `run.json` (identity, spec, score, spend, `spend_discarded_usd`, metric finals), `tables/<family>.parquet` (turns, tool_calls, posts, deliveries, reads, actions, inference, probes, metrics, interventions, rounds, run, other, and `discarded_inference`: the calls of rounds discarded at a hard ceiling, with `charged_usd`, so the ledger spend = non-cached `inference.cost_usd` + discarded spend; key columns `experiment, arm, seed, run, round, agent`; blob content inlined up to 64 KiB), `sessions/<agent>.jsonl` (one [pi-format](https://github.com/earendil-works/pi/blob/main/packages/coding-agent/docs/session-format.md) session per agent with `harness: "swarmlab"`, so the Hub's agent-traces viewer renders each conversation) and `raw/` (the byte-identical log, snapshots and blobs).
- `swarmlab publish runs/` (Python `Experiment.publish(runs_dir)`) exports what is needed and uploads every finished run to one **private** Hub dataset repo per experiment (`<you>/<experiment>`, or `--repo`): `runs/<run_id>/...`, `index.json`, and a dataset card with the arms' specs, a run table and the table schemas. Re-publishing uploads only changed files. `--public` makes the repo public and adds the `format:agent-traces` tag (the Hub's trace viewer renders public repos only); traces hold every prompt and reply, so read them first. Needs the `hub` extra and `HF_TOKEN`.
- `swarmlab view RUN_DIR --publish OWNER/REPO` uploads `view.html` next to the run and links it from the card.
- `swarmlab fetch-published OWNER/REPO RUN_ID --out runs/` rebuilds the run directory from the published raw log, snapshots and blobs, so `replay`, `view`, `fork` and `Run.load` work on it.

## Experimenter skill

`skill/SKILL.md` teaches a coding agent the tested workflow, from `swarmlab doctor` to `swarmlab publish`, including the first-real-run checklist. Install it for Claude Code (`~/.claude/skills/swarmlab/SKILL.md`), Codex (`~/.agents/skills/swarmlab/SKILL.md`) or OpenCode (`~/.config/opencode/skills/swarmlab/SKILL.md`); the file has the copy command.

## Run directory

```
runs/<run_id>/
  run.json          run id, spec, spec hash, git commit, status, end reason, last round, score
  events.jsonl      the event log; a round is committed once its round_committed line is written
  discarded.jsonl   events dropped by crash recovery
  blobs/            content-addressed delivered content and plugin state
  snapshots/        <round:06d>.json manifests, one per round
  artifacts/        spec.yaml, git.txt
  view.html         the static replay page (after `view`)
  export/           tables, pi sessions and raw copies (after `export` or `publish`)
```
Run ids are `<experiment>__s<seed>` from Python, `<experiment>__<arm>__s<seed>` from YAML, plus `__f<round>_<n>` for a fork.

## Known issues

`docs/notes/known-issues.md` lists open problems: JSON tool protocol on Haiku, DeepInfra tool-call stalls, estimates overstating spend, `validate` rejecting `vllm:` specs, role and paired-run gaps, and no grid panel in the viewer.

## Docs

`docs/DESIGN.md` (why), `docs/INTERFACE.md`, `INTERFACE-M1b.md`, `INTERFACE-M3a.md`, `INTERFACE-M3c.md` and `INTERFACE-M4.md` (the binding contracts), `docs/handoff/INDEX.md` (per work package notes), `docs/notes/` (run reports), `AGENTS.md` (rules for agents working in this repo). Development: `uv run pytest -q` and `uv run ruff check swarmlab tests examples tools`.
