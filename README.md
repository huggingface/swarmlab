# swarmlab

A scientific testbed for finding the primitives that make heterogeneous groups of LLM agents collaborate well or badly. An experiment is a **World** (the task), a set of **Participants**, a **Medium** (a board with a topology and visibility policies) and **Metrics**; the framework handles tool dispatch, ordering, recording, snapshots, replay, resume and fork, so a scientist changes the hypothesis without touching the execution machinery. Status: M1a (scripted participants only; nothing calls a model yet).

## Install and develop

```
uv sync --extra dev                 # set UV_PROJECT_ENVIRONMENT to keep the venv outside the repo
uv run pytest -q
uv run ruff check swarmlab tests examples
```

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

## CLI

The CLI builds the same `Experiment` from YAML (`examples/flaggame_m1a.yaml` has arms `broadcast` and `gossip`) and calls the same API.

```
swarmlab validate examples/flaggame_m1a.yaml
swarmlab run examples/flaggame_m1a.yaml --arm gossip --seed 3 [--max-rounds N] [--out runs/]
swarmlab replay RUN_DIR                  # re-fold metrics and score from the log; exit 1 on mismatch
swarmlab resume RUN_DIR                  # continue from the last committed round
swarmlab fork RUN_DIR --at 4 [--spec edited.yaml --arm A] [--out DIR]
swarmlab view RUN_DIR                    # writes RUN_DIR/view.html
```
Every command takes `--json` and then prints one JSON object (run dir, run id, spec hash, score, end reason, last round, final metric values). Exit codes: 0 success, 2 invalid spec, 1 any other error.

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

`docs/DESIGN.md` (why), `docs/INTERFACE.md` (the binding M1a contract), `docs/handoff/` (per work package notes), `AGENTS.md` (rules for agents working in this repo).
