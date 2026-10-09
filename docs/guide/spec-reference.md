# Spec reference

The block below is generated; regenerate it with `swarmlab spec-reference --readme docs/guide/spec-reference.md`. Back to the [docs index](../README.md).

<!-- spec-reference:start -->
Every key of an experiment YAML, generated from the pydantic models (`swarmlab spec-reference` prints this list). Unknown keys are errors. Wherever the shape is `{type: NAME, params: {...}}` the value is a plugin: `{type: NAME, params: {...}}`, or the bare string `NAME` for `{type: NAME, params: {}}`.

- `name`: `str` (required): experiment name; run ids are `<name>__<arm>__s<seed>`
- `arms`: `{NAME: {world, participants, medium, metrics, probes, interventions, options, budget, roles}}` (required): the conditions, by arm name; each arm is a full setup (see `arms.<arm>`)
  - `arms.NAME.world`: `{type: NAME, params: {...}}` (required): the task, e.g. `{type: flaggame, params: {n_candidates: 8}}` or `flaggame`
  - `arms.NAME.participants`: `[{type, count, params, role}, ...]` (required): participant groups, in agent order
    - `arms.NAME.participants[].type`: `str` (required): participant type: `llm`, `evidence_aggregator`, `enumerator`, `silent`, ...
    - `arms.NAME.participants[].count`: `int` (default `1`): number of agents in this group (>= 1)
    - `arms.NAME.participants[].params`: `{...}` (optional): constructor params, e.g. `{model: "anthropic:claude-haiku-4-5", max_tokens: 1024}` for `llm`
    - `arms.NAME.participants[].role`: `str | null` (optional): role name (declared in `roles`, or a built-in: worker, coordinator, reviewer, skeptic, scribe, manager)
  - `arms.NAME.medium`: `{topology, delivery, push_limit, policies, channels, registry, claim_policy, push_consume}` (default: see keys): the message board: topology, policies, registry
    - `arms.NAME.medium.topology`: `{type: NAME, params: {...}}` (default `broadcast`): who receives a post: `broadcast`, `gossip` (params `{k: 1}`: partners per agent per round), `groups`, `star` (params `{center: a000}`: members reach only the center, the center reaches all), `tree`, `rooms` (params `{rooms: {red: [a000, a001], blue: [a001, a002]}}`: named rooms, an agent may be in several or none; a post reaches only its room); e.g. `{type: gossip, params: {k: 2}}`
    - `arms.NAME.medium.delivery`: `pull | push` (default `pull`): `pull` (agents call `read_board`) or `push` (deliveries come with the turn)
    - `arms.NAME.medium.push_limit`: `int` (default `20`): most items pushed per turn under `delivery: push`
    - `arms.NAME.medium.policies`: `[{type: NAME, params: {...}}, ...]` (optional): visibility policies applied in order, e.g. `[{type: delay, params: {rounds: 1}}]`
    - `arms.NAME.medium.channels`: `[str, ...]` (default `[main]`): board channels
    - `arms.NAME.medium.registry`: `bool` (default `false`): turn on the claim registry tools
    - `arms.NAME.medium.claim_policy`: `{type: NAME, params: {...}}` (default `advisory`): `advisory` or `enforced` (registry claims checked against world actions)
    - `arms.NAME.medium.push_consume`: `bool` (default `false`): with `delivery: push`: show the newest `push_limit` unread items and mark every pushed-up-to item read, so each item is pushed once
  - `arms.NAME.metrics`: `[{type: NAME, params: {...}}, ...]` (optional): metrics logged every round (`swarmlab metrics` lists them)
  - `arms.NAME.probes`: `[{type: NAME, params: {...}}, ...]` (optional): probes asked after every commit, e.g. `[belief]`
  - `arms.NAME.interventions`: `[{type: NAME, params: {...}}, ...]` (optional): interventions (`inject_post`, `mute`, ...) for this arm
  - `arms.NAME.options`: `{max_rounds, commit, max_calls_per_turn, snapshot_every, concurrency, repeat, scheduler, rounds_per_agent, stop_when}` (optional): run options for this arm, merged over the top-level `options`
    - `arms.NAME.options.max_rounds`: `int` (required): rounds per run (required here or as `--max-rounds`)
    - `arms.NAME.options.commit`: `round_end | immediate` (default `round_end`): `round_end` (phase commit) or `immediate` (sequential)
    - `arms.NAME.options.max_calls_per_turn`: `int` (default `20`): tool calls an agent may make per turn
    - `arms.NAME.options.snapshot_every`: `int` (default `1`): write a snapshot every N rounds
    - `arms.NAME.options.concurrency`: `int` (default `32`): concurrent turns
    - `arms.NAME.options.repeat`: `int` (default `0`): paired-run repeat index (0: a plain run)
    - `arms.NAME.options.scheduler`: `{type: NAME, params: {...}} | null` (optional): turn order per round: `seeded_shuffle` (default, every live agent) or `one_speaker` (one random live agent per round; pair with `gossip` k=1 and `commit: immediate` for the Flag Game paper's pairwise protocol)
    - `arms.NAME.options.rounds_per_agent`: `int | null` (optional): sugar: `max_rounds = rounds_per_agent x number of agents`
    - `arms.NAME.options.stop_when`: `{metric, op, value, consecutive} | null` (optional): `{metric, op, value, consecutive}`: end the run (`stop_condition`) when the metric compares true at `consecutive` evaluations in a row (probe rounds when the run has probes, else every round)
      - `arms.NAME.options.stop_when.metric`: `str` (required): a logged metric name, e.g. `belief.consensus@probe:belief`
      - `arms.NAME.options.stop_when.op`: `>= | > | <= | < | ==` (default `>=`): comparison: `>=`, `>`, `<=`, `<` or `==`
      - `arms.NAME.options.stop_when.value`: `float` (required): threshold the metric is compared with
      - `arms.NAME.options.stop_when.consecutive`: `int` (default `1`): evaluations in a row the comparison must hold
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
- `options`: `{max_rounds, commit, max_calls_per_turn, snapshot_every, concurrency, repeat, scheduler, rounds_per_agent, stop_when}` (optional): run options for every arm (each arm's `options` is merged over them)
  - `options.max_rounds`: `int` (required): rounds per run (required here or as `--max-rounds`)
  - `options.commit`: `round_end | immediate` (default `round_end`): `round_end` (phase commit) or `immediate` (sequential)
  - `options.max_calls_per_turn`: `int` (default `20`): tool calls an agent may make per turn
  - `options.snapshot_every`: `int` (default `1`): write a snapshot every N rounds
  - `options.concurrency`: `int` (default `32`): concurrent turns
  - `options.repeat`: `int` (default `0`): paired-run repeat index (0: a plain run)
  - `options.scheduler`: `{type: NAME, params: {...}} | null` (optional): turn order per round: `seeded_shuffle` (default, every live agent) or `one_speaker` (one random live agent per round; pair with `gossip` k=1 and `commit: immediate` for the Flag Game paper's pairwise protocol)
  - `options.rounds_per_agent`: `int | null` (optional): sugar: `max_rounds = rounds_per_agent x number of agents`
  - `options.stop_when`: `{metric, op, value, consecutive} | null` (optional): `{metric, op, value, consecutive}`: end the run (`stop_condition`) when the metric compares true at `consecutive` evaluations in a row (probe rounds when the run has probes, else every round)
    - `options.stop_when.metric`: `str` (required): a logged metric name, e.g. `belief.consensus@probe:belief`
    - `options.stop_when.op`: `>= | > | <= | < | ==` (default `>=`): comparison: `>=`, `>`, `<=`, `<` or `==`
    - `options.stop_when.value`: `float` (required): threshold the metric is compared with
    - `options.stop_when.consecutive`: `int` (default `1`): evaluations in a row the comparison must hold
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
