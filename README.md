# swarmlab

A scientific testbed for finding the primitives that make heterogeneous groups of LLM agents collaborate well or badly. An experiment is a **World** (the task), a set of **Participants**, a **Medium** (a board with a topology and visibility policies) and **Metrics**; the framework handles tool dispatch, ordering, recording, budgets, snapshots, replay, resume and fork, so a scientist changes the hypothesis without touching the execution machinery. Status: M1b (LLM agents on Anthropic, the HF router or any OpenAI-compatible endpoint; probes; budgets).

## Quickstart

```
uv sync --extra dev --extra anthropic     # or: pip install -e .[anthropic]
uv run swarmlab doctor                    # Python, extras, API keys, provider reachability, git
uv run swarmlab init demo                 # writes demo.yaml and demo.py
uv run swarmlab run demo.yaml             # every arm x every seed -> runs/<run_id>/
uv run swarmlab view runs/demo__gossip__s1   # writes the replay page view.html
```
(Inside an activated venv, or after `pip install`, drop the `uv run`.) The starter compares a broadcast board with a gossip board on the Flag Game, with 8 LLM agents per arm on the deterministic fake model `fake:reader`: no keys, no network, nothing billed. `run` prints the worst-case estimate per arm and in total, runs each arm with each seed in `seeds:`, skips runs whose directory already holds the same spec (`--rerun` writes `runs/<run_id>__r2`), and ends with a table:

```
run                  outcome  end         score                                 spend
demo__broadcast__s1  ran      max_rounds  accuracy=1, n_guessed=8, truth=G      $0.2556
demo__gossip__s1     ran      max_rounds  accuracy=0.875, n_guessed=8, truth=G  $0.2362
...
```
`--arm A` and `--seed N` narrow it; `--json` prints one JSON object. `demo.py` is the same experiment in Python (`python demo.py`).

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
- **Budgets** (USD, per run): `soft_usd` ends the run at the next round boundary once agent spend reaches it; `hard_usd` is an absolute ceiling on agent + probe spend (the round in flight is discarded and `swarmlab resume RUN --budget-hard X` continues); `measurement_usd` caps probes. 0 means "not enforced", so set `hard_usd` before using a paid model. When any budget is non-zero `run` asks before starting (`--yes` skips the question); `swarmlab estimate spec.yaml --arm A` prints the estimate alone.

## CLI

```
swarmlab doctor [SPEC...] [--offline]    swarmlab models [--provider P] [--tools] [--search S]
swarmlab init [NAME]                     swarmlab validate SPEC
swarmlab run SPEC [--arm A] [--seed N] [--max-rounds R] [--out runs/] [--yes] [--rerun]
swarmlab estimate SPEC [--arm A]         swarmlab replay RUN_DIR
swarmlab resume RUN_DIR [--budget-hard X]   swarmlab fork RUN_DIR --at 4 [--spec edited.yaml]
swarmlab view RUN_DIR
```
Every command takes `--json` and then prints one JSON object. Exit codes: 0 success, 2 invalid spec, 1 any other error (including a declined confirmation or a failed run).

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
```
Run ids are `<experiment>__s<seed>` from Python, `<experiment>__<arm>__s<seed>` from YAML, plus `__f<round>_<n>` for a fork.

## Docs

`docs/DESIGN.md` (why), `docs/INTERFACE.md` and `docs/INTERFACE-M1b.md` (the binding contracts), `docs/handoff/` (per work package notes), `AGENTS.md` (rules for agents working in this repo). Development: `uv run pytest -q` and `uv run ruff check swarmlab tests examples`.
