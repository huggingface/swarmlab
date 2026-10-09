# Building experiments

What the framework ships (worlds, topologies, participants, providers, metrics, ...), the Python API, how to write a new world and how to change what agents are told. Back to the [docs index](../README.md).

## Components

- **Worlds**: `flaggame` (belief dynamics: hidden flag, one crop per agent, `guess` action; text or image crops, see [Flag Game modalities](flag-game.md#flag-game-modalities); optional blind agents, see [Flag Game protocols](flag-game.md#flag-game-protocols)), `coloring` (allocation: a grid to paint, with claims and a registry). New worlds subclass `World` (example 3).
- **Topologies** (`medium: {topology: ...}`): `broadcast`, `gossip`, `groups`, `star` (members reach only the center, the center reaches everyone, `params: {center: a000}`), `tree` (coordinators over worker groups, `params: {groups: 3}`), `rooms` (named rooms with fixed, possibly overlapping membership, `params: {rooms: {red: [a000, a001], blue: [a001, a002]}}`; see [Rooms](#rooms)). See [Flag Game protocols](flag-game.md#flag-game-protocols).
- **Policies**: `delay` (messages arrive k rounds late) and your own `Policy` subclasses (example 2). **Registry** with `advisory` or `enforced` claim policies for the coloring task.
- **Participants**: scripted (`evidence_aggregator`, `enumerator`, `silent`, `row_major_painter`, `queue_painter`, `random_painter`) and `llm` (`LLMAgent`: memory window, `context_limit_tokens` with `drop_oldest`/`summarize`/`fail_turn` overflow, prompt params, `max_calls`, `extra` passthrough).
- **Providers** (model id prefix): `anthropic`, `hf` (HF router), `openai`, `vllm` (self-hosted, normally via `job run`), `fake` (`fake:reader`, `fake:painter`: deterministic, no network).
- **Probes**: `belief` (each agent's own model answers "which candidate?" every round, out of band, billed to `measurement_usd`).
- **Interventions** (`interventions:` in a spec, triggered by `at_round`, `every` or a metric `when`): `inject_post`, `delay_delivery`, `mute`, `kill_agents`, `patch_private`, `reconfigure`. **Paired runs** from round 0: `Experiment.pair(...)` with unchanged-pair controls.
- **Metrics**: `belief.consensus`, `belief.accuracy`, `belief.polarization`, `belief.entropy` (from world guesses, or from probes with `params: {source: "probe:belief"}`); `comm.read_rate`, `comm.post_rate` (share of turns with a post), `comm.posts_per_round` (posts per agent per round), `comm.posts_total` (swarm-wide posts per round), `comm.hops`; `coloring.coverage`, `coloring.duplicate_paints`, `coloring.wrong_paints`, `coloring.parallel_efficiency`; `claims.violations`, `claims.held`. `swarmlab metrics` lists them all with one-line descriptions.
- **Roles** (`roles:` plus `role:` per participant group): built-ins `worker`, `coordinator`, `reviewer`, `skeptic`, `scribe`, `manager`; each can restrict tools, channels, registry access and world actions, add prompt text, or override the model.
- **Run control**: replay (checks metrics and score from the log), resume, fork at a round (optionally with an edited spec), budgets (soft, hard, measurement, experiment total), per-call timeout and retry.
- **Export and publish**: Parquet tables, pi-format sessions and raw logs; private Hub dataset per experiment; static `view.html`; `fetch-published` restores a run.
- **Jobs**: `swarmlab job run|status|logs|fetch` runs a spec on HF Jobs with vLLM serving the model in the same job (validated at N=256, 83 min, $3.45).

## Rooms

The `rooms` topology restricts communication to named rooms with fixed membership. Each room becomes a board channel; an agent may be in several rooms (a bridge) or in none.

```yaml
medium:
  topology:
    type: rooms
    params:
      rooms:
        red: [a000, a001, a002]
        blue: [a002, a003, a004]   # a002 is in both rooms
```

- An agent can read and post only in its rooms; the `channel` enum of its `read_board` and `post` tools lists exactly those. An agent in no room gets no board tools.
- A post reaches the other live members of the room it is posted on, nothing else: a002's post on `red` never reaches `blue`. Information crosses rooms only when a shared member repeats it.
- `post` without a channel goes to the agent's room when it is in exactly one; an agent in several rooms must name one (the call fails with `bad args` listing its rooms).
- A role's `channels_read` / `channels_write` can narrow an agent's rooms further, never widen them.
- Membership is fixed for the run. Every listed member must be an agent of the run.

`examples/rooms_flaggame.yaml` compares disjoint rooms with rooms joined by two bridge agents.

## Python examples

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

## Writing a world

- Required: `reset(rng, agents)`, `observe(agent)`, `score()` and at least one `@tool` action returning an `Outcome` (example 3). A world in your own module is used from YAML as `world: {type: "mymodule:MyWorld", params: {...}}`.
- State in ordinary attributes is snapshotted every round (a pickle of the instance's attributes minus `params` and `_`-prefixed ones), so resume, replay and fork need no world code.
- Optional hooks (defaults in `swarmlab/world/base.py`): `description()` (task text for prompts), `validate` and `commit` (refusals and conflict rules), `begin_round(round)`, `terminal()`, `my_status`/`collective_status`, `verify()` (evaluator-only truth for metrics, see [Analysis patterns](analysis.md#analysis-patterns)), `claim_key`, `killed_at`, and `render_state()`.
- `Outcome.feedback` reaches the agent and the `actions` table: put there what you will analyse per decision, never correctness.
- `render_state() -> dict | None` fills the "World state" panel of `view.html`: the builder restores a fresh world from each round's snapshot and shows what it returns. A `grid` (list of equal-length rows) is drawn as coloured cells with `palette` (cell value -> CSS colour; other 2-D lists are drawn too when `palette` is given), a list of flat dicts as a table, other values as key/value rows. `coloring` returns its current and target grids; FlagGame returns None (it has its own panel).

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
