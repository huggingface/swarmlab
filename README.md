# swarmlab

<p align="center">
  <b>A scientific testbed for collaboration primitives in heterogeneous LLM-agent swarms</b>
</p>

<p align="center">
  <a href="LICENSE"><img alt="License" src="https://img.shields.io/badge/license-Apache%202.0-blue"></a>
  <img alt="Python" src="https://img.shields.io/badge/python-3.12%2B-blue">
  <a href="docs/README.md"><img alt="Docs" src="https://img.shields.io/badge/docs-guide-green"></a>
</p>

## Overview

When many LLM agents work on one problem, does the group find the truth, herd onto a wrong answer, split into camps, or never agree? The answer depends less on the model than on the **collaboration primitives** around it: who hears whom, when messages arrive, what roles agents play, what they remember. swarmlab exists to find the primitives that matter, one controlled experiment at a time.

An experiment has four parts:

- **World**: the task, with a ground truth the agents cannot see (for example, a hidden flag where each agent holds one crop).
- **Participants**: LLM agents from any provider, scripted baselines, or a mix of both.
- **Medium**: the message board, meaning its topology, delivery rules and visibility policies.
- **Metrics**: what gets measured every round, such as consensus, accuracy, polarization or read rate.

The framework handles everything else: tool dispatch, turn order, logging, budgets, snapshots, replay, resume and fork. To test a new hypothesis, you change one line of YAML and leave the execution code alone.

## Highlights

- **Tasks with a ground truth.** The [Flag Game](https://arxiv.org/abs/2609.19124) tests belief dynamics and Coloring tests allocation. You can add your own `World` in about 15 lines.
- **Communication is the variable you change.** Pick a topology (`broadcast`, `gossip`, `groups`, `star`, `tree`), add delay or custom visibility policies, use a claim registry, and assign roles such as manager, skeptic or reviewer.
- **Mixed swarms.** Use Anthropic, Hugging Face Inference Providers, OpenAI, self-hosted vLLM or deterministic fake models, and mix them within one arm.
- **Built-in measurement.** Belief consensus, accuracy, polarization and entropy; out-of-band belief probes; communication metrics; interventions in the middle of a run; paired runs.
- **Reproducible.** Every event is logged. Replay re-derives the score from the log, and you can resume a run or fork it at any round with an edited spec.
- **No surprise bills.** Estimates and a preflight check come before you spend anything. Each run has soft and hard caps, and a ledger enforces one cap across the whole experiment.
- **Scales on HF Jobs.** vLLM serves the model inside the same job. A 256-agent swarm ran in 83 minutes for $3.45.
- **Easy to share.** Export Parquet tables and pi-format agent traces, publish them as a Hub dataset, and open a static replay page for any run.
- **Agent-friendly.** Every command takes `--json`, and a [skill](skill/SKILL.md) teaches Claude Code, Codex or OpenCode the tested experiment workflow.

## Installation

```bash
git clone https://github.com/cmpatino/swarmlab && cd swarmlab
uv sync --extra anthropic --extra hub    # or: pip install -e ".[anthropic,hub]"
uv run swarmlab doctor                   # checks Python, extras, API keys, provider reachability
```

The extras are `anthropic` (the Anthropic provider), `hub` (publishing to the Hugging Face Hub) and `dev` (tests and linting).

## Quick Start

### Run the starter experiment (free)

```bash
mkdir ~/demo-exp && cd ~/demo-exp                   # experiments live outside the checkout
uv run --project ~/swarmlab swarmlab init demo      # writes demo.yaml and demo.py
uv run --project ~/swarmlab swarmlab run demo.yaml  # every arm x every seed -> ./runs/<run_id>/
uv run --project ~/swarmlab swarmlab view runs/demo__gossip__s1   # writes the replay page view.html
```

The starter compares a broadcast board with a gossip board on the Flag Game, using 8 agents per arm on the deterministic `fake:reader` model. It needs no API keys and no network, and nothing is billed.

```
run                  outcome  end         score                                 spend
demo__broadcast__s1  ran      max_rounds  accuracy=1, n_guessed=8, truth=G      $0.2577 (simulated)
demo__gossip__s1     ran      max_rounds  accuracy=0.625, n_guessed=8, truth=G  $0.2383 (simulated)
...
```

### Write an experiment in Python

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
    budget=Budget(soft_usd=0, hard_usd=0),
)
run = exp.run(seed=3, max_rounds=20)
run.metrics["belief.consensus"]   # [(round, value, denominator), ...]
run.fork(at_round=4).run()        # same history up to round 4, then diverge
```

### Change one mechanism

Here, a custom visibility policy hides every message from odd-numbered agents:

```python
from swarmlab import Policy

class OddAgentsSeeNothing(Policy):
    def apply(self, reader, post, round):
        if int(reader[1:]) % 2:
            return None                      # withhold the message
        return round + 1, post.text          # deliver it next round, unchanged

exp = Experiment(..., medium=Board(topology="broadcast", policies=[OddAgentsSeeNothing()]))
```

### Add a new task

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

### Switch to a real model

```yaml
participants:
  - {type: llm, count: 8, params: {model: "hf:google/gemma-4-26B-A4B-it:deepinfra"}}
budget: {hard_usd: 0.50, total_usd: 5}
```

```bash
swarmlab estimate exp.yaml     # worst-case spend for every arm and seed
swarmlab preflight exp.yaml    # one real request per participant group, for a few cents
swarmlab run exp.yaml
```

Read [Real models and budgets](docs/guide/real-models.md) before your first paid run. It covers model ids, keys, prices, timeouts and every budget cap.

## Command Line Interface

| command | what it does |
|---|---|
| `swarmlab doctor` | checks your environment, keys and providers |
| `swarmlab init NAME` | writes a starter experiment |
| `swarmlab validate SPEC` | shows each arm's agents, models, caps and metrics |
| `swarmlab estimate SPEC` / `preflight SPEC` | prices a spec / sends one real request per group |
| `swarmlab prompts SPEC --arm A` | prints exactly what the agents will see, with no model call |
| `swarmlab run SPEC` | runs every arm x seed under your caps |
| `swarmlab replay` / `resume` / `fork` RUN | re-derives, continues or branches a run |
| `swarmlab view RUN` | builds the static replay page |
| `swarmlab report RUNS` | writes a Markdown report across runs |
| `swarmlab export` / `publish` | writes Parquet tables and agent traces, and uploads a private Hub dataset |
| `swarmlab job run SPEC --model M` | runs a spec on HF Jobs with vLLM in the same job |

Every command takes `--json`. The full flag list is in [CLI and run directory](docs/guide/cli.md).

## Documentation

The [documentation index](docs/README.md) links every page: [getting started](docs/guide/getting-started.md), [building experiments](docs/guide/building-experiments.md), [the Flag Game](docs/guide/flag-game.md), [real models and budgets](docs/guide/real-models.md), [spec reference](docs/guide/spec-reference.md), [CLI](docs/guide/cli.md) and [analysing and publishing](docs/guide/analysis.md). The design rationale is in [`docs/DESIGN.md`](docs/DESIGN.md), and open problems are in [`docs/known-issues.md`](docs/known-issues.md).

**Coding agents:** read [`AGENTS.md`](AGENTS.md) first. To run experiments, follow [`skill/SKILL.md`](skill/SKILL.md).

## Development

```bash
uv sync --extra dev --extra anthropic
uv run pytest -q
uv run ruff check swarmlab tests examples tools
```

## License

Apache-2.0. See [LICENSE](LICENSE).
