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
demo__broadcast__s1  ran      max_rounds  accuracy=1, n_guessed=8, truth=G      $0.2556
demo__gossip__s1     ran      max_rounds  accuracy=0.875, n_guessed=8, truth=G  $0.2362
...
```
`--arm A` and `--seed N` narrow it; `--json` prints one JSON object. `demo.py` is the same experiment in Python (`python demo.py`).

## What exists

- **Worlds**: `flaggame` (belief dynamics: hidden flag, one crop per agent, `guess` action; text crops only), `coloring` (allocation: a grid to paint, with claims and a registry). New worlds subclass `World` (example 3).
- **Topologies** (`medium: {topology: ...}`): `broadcast`, `gossip`, `groups`, `tree` (coordinators over worker groups, `params: {groups: 3}`).
- **Policies**: `delay` (messages arrive k rounds late) and your own `Policy` subclasses (example 2). **Registry** with `advisory` or `enforced` claim policies for the coloring task.
- **Participants**: scripted (`evidence_aggregator`, `enumerator`, `silent`, `row_major_painter`, `queue_painter`, `random_painter`) and `llm` (`LLMAgent`: memory window, `context_limit_tokens` with `drop_oldest`/`summarize`/`fail_turn` overflow, prompt params, `max_calls`, `extra` passthrough).
- **Providers** (model id prefix): `anthropic`, `hf` (HF router), `openai`, `vllm` (self-hosted, normally via `job run`), `fake` (`fake:reader`, `fake:painter`: deterministic, no network).
- **Probes**: `belief` (each agent's own model answers "which candidate?" every round, out of band, billed to `measurement_usd`).
- **Interventions** (`interventions:` in a spec, triggered by `at_round`, `every` or a metric `when`): `inject_post`, `delay_delivery`, `mute`, `kill_agents`, `patch_private`, `reconfigure`. **Paired runs** from round 0: `Experiment.pair(...)` with unchanged-pair controls.
- **Metrics**: `belief.consensus`, `belief.accuracy`, `belief.polarization`, `belief.entropy` (from world guesses, or from probes with `params: {source: "probe:belief"}`); `comm.read_rate`, `comm.posts_per_round`, `comm.hops`; `coloring.coverage`, `coloring.duplicate_paints`, `coloring.wrong_paints`, `coloring.parallel_efficiency`; `claims.violations`, `claims.held`.
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

**4. LLM agents with a belief probe** (`examples/04_llm_flaggame.py`): `LLMAgent(model="fake:reader")` on the Flag Game with `BeliefProbe()`; set `MODEL = "anthropic:claude-haiku-4-5"` for a real run.

## Switching to real models

- **Model ids** are `provider:model[:served_by]`: `anthropic:claude-haiku-4-5`, `hf:Qwen/Qwen3.5-9B:deepinfra` (the HF router, pinned to DeepInfra; without `:served_by` the router picks), `openai:<model>`, `vllm:<model>` (with a `providers:` base url). Put it in `model:` of each `llm` participant group.
- **Keys** come from the environment: `ANTHROPIC_API_KEY` (or `ANTHROPIC_KEY`), `HF_TOKEN`, `OPENAI_API_KEY`. `swarmlab doctor demo.yaml` checks the keys and extras that spec needs and that the endpoints answer.
- **Prices** are never typed by hand. `swarmlab models [--provider hf|anthropic] [--tools] [--search qwen]` lists models, who serves them, tool support and USD per million tokens. `hf` prices come from the router listing (cached 24 h in `~/.cache/swarmlab/catalog.json`); a bare `hf:Org/Model` is priced at its most expensive listed provider. The price used is written into the run spec. To override, or for a model the catalog does not price, add `providers: {hf: {type: openai_compat, params: {name: hf, pricing: {"Org/Model:prov": [in, out, cached]}}}}`.
- **Timeouts**: each provider call attempt is cut off after `timeout_s` (default 90 s) and timeouts, connection errors, 429 and 5xx are retried up to `max_retries` times (default 2) with jittered 1/2/4 s backoff; the budget reservation is held across retries. A phase-commit round waits for its slowest call, so tighten these for slow-tailed routers: `providers: {hf: {type: openai_compat, params: {name: hf, timeout_s: 60, max_retries: 3}}}` (`type: anthropic` takes the same two params).
- Each `inference_response` event records `attempts`; a call that still fails after its retries raises `ProviderError`.
- **Budgets** (USD, per run): `soft_usd` ends the run at the next round boundary once agent spend reaches it; `hard_usd` is an absolute ceiling on agent + probe spend (the round in flight is discarded, its spend still counts, and `swarmlab resume RUN --budget-hard X` continues: X is the run's new total including everything already spent, or `--add-budget D` allows D more from the current spend; `resume` prints the spend so far); `measurement_usd` caps probes. 0 means "not enforced", so set `hard_usd` before using a paid model. These caps are per run; `total_usd` (top-level `budget:` only) caps the whole experiment: `swarmlab run` (and `Experiment.run_all`) starts a run only if the spend of the runs before it plus its `hard_usd` fits under `total_usd`, otherwise it stops and lists the skipped runs. `run` prints every arm's caps and the total cap before starting, and warns when `hard_usd - soft_usd` is less than one round's estimated cost (the soft budget is checked between rounds, so a round can start under it, hit the hard ceiling and be discarded). When any budget is non-zero `run` asks before starting (`--yes` skips the question); `swarmlab estimate spec.yaml` prints the estimate alone (every arm x seed; `--arm`/`--seed` narrow it, `--prompt-growth TOKENS` models a context that grows each round under full memory; calls per turn default to the runner cap `max_calls_per_turn`, capped by each group's own `max_calls`, `--calls-per-turn C` assumes C instead, and the output states the value used).

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
The round message (round number, observation, last round's outcomes, deliveries) is fixed by the agent and the world. Check what each arm actually sends, with no model call, and diff two arms:
```
swarmlab prompts exp.yaml --arm default > default.txt
swarmlab prompts exp.yaml --arm board > board.txt
diff default.txt board.txt      # should show only the manipulated sentence
```
Note that the default prompt already lists every tool the world and board offer (`read_board`, `post`, ...), so an "agents are told about the board" arm tests a nudge, not awareness.

## CLI

```
swarmlab doctor [SPEC...] [--offline]    swarmlab models [--provider P] [--tools] [--search S] [--refresh]
swarmlab init [NAME] [--dir D] [--force]  swarmlab validate SPEC
swarmlab run SPEC [--arm A] [--seed N] [--max-rounds R] [--out runs/] [--yes] [--rerun]
swarmlab estimate SPEC [--arm A] [--seed N] [--max-rounds R] [--prompt-growth G] [--calls-per-turn C]
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

`docs/notes/known-issues.md` lists open problems: JSON tool protocol on Haiku, DeepInfra tool-call stalls, estimates overstating spend, `comm.posts_per_round` being swarm-wide, `validate` rejecting `vllm:` specs, role and paired-run gaps, and no grid panel in the viewer.

## Docs

`docs/DESIGN.md` (why), `docs/INTERFACE.md`, `INTERFACE-M1b.md`, `INTERFACE-M3a.md`, `INTERFACE-M3c.md` and `INTERFACE-M4.md` (the binding contracts), `docs/handoff/INDEX.md` (per work package notes), `docs/notes/` (run reports), `AGENTS.md` (rules for agents working in this repo). Development: `uv run pytest -q` and `uv run ruff check swarmlab tests examples tools`.
