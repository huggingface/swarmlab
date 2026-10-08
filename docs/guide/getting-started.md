# Getting started

Install swarmlab, run the free starter experiment and understand what `swarmlab run` writes. Back to the [docs index](../README.md).

## Install and run the starter

```
git clone https://github.com/cmpatino/swarmlab && cd swarmlab
uv sync --extra dev --extra anthropic     # or: pip install -e .[anthropic]
uv run swarmlab doctor                    # Python, extras, API keys, provider reachability, git

mkdir ~/demo-exp && cd ~/demo-exp         # experiments live outside the repo checkout
uv run --project ~/swarmlab swarmlab init demo      # writes demo.yaml and demo.py
uv run --project ~/swarmlab swarmlab run demo.yaml  # every arm x every seed -> ./runs/<run_id>/
uv run --project ~/swarmlab swarmlab view runs/demo__gossip__s1   # writes the replay page view.html
```
(`~/swarmlab` is wherever you cloned the repo. Alternatively `pip install -e ~/swarmlab[anthropic]` into the project's own environment and call `swarmlab` directly.) `uv run --project ~/swarmlab` uses the checkout's own environment, `~/swarmlab/.venv`, and creates it there on first use from whatever directory you run it in (checked with uv 0.12: the environment belongs to the project, not to the working directory; `.venv/` is git-ignored). To keep the environment out of the checkout, e.g. a read-only or shared clone, set `UV_PROJECT_ENVIRONMENT=/path/to/env` for every `uv` call (`uv sync` and `uv run`). Work from a project directory outside the checkout: `run` writes `runs/` under the current directory, and inside the swarmlab checkout it refuses unless you pass `--out`, so runs never end up in the repo. `swarmlab validate demo.yaml` prints each arm's agents, models, caps, probes and metrics. The starter compares a broadcast board with a gossip board on the Flag Game, with 8 LLM agents per arm on the deterministic fake model `fake:reader`: no keys, no network, nothing billed. `run` prints the worst-case estimate per arm and in total, runs each arm with each seed in `seeds:`, skips runs whose directory already holds the same spec (`--rerun` writes `runs/<run_id>__r2`), and ends with a table:

```
run                  outcome  end         score                                 spend
demo__broadcast__s1  ran      max_rounds  accuracy=1, n_guessed=8, truth=G      $0.2577 (simulated)
demo__broadcast__s2  ran      max_rounds  accuracy=1, n_guessed=8, truth=H      $0.2577 (simulated)
demo__gossip__s1     ran      max_rounds  accuracy=0.625, n_guessed=8, truth=G  $0.2383 (simulated)
...
```
`(simulated)` marks runs on `fake:` models only: the spend is computed from a nominal price table and nothing is billed.
`--arm A` and `--seed N` narrow it; `--json` prints one JSON object.

**Existing run dirs.** A run id is only `<experiment>__<arm>__s<seed>`, so `run` compares the spec hash of what it would run with `runs/<run_id>/run.json`. Same hash: the run is skipped (`exists, skipping`; its spend still counts toward `total_usd`). Different hash (you changed the arm's spec, a cap, the model, a prompt): `run` refuses that run with `FileExistsError` naming both hashes (the dir's and the spec's) and the run's outcome is `failed` (exit 1; other runs still go ahead). Then either pass `--rerun`, which writes the new run beside the old one as `runs/<run_id>__r2` (`__r3`, ... the first free N; also for same-hash repeats), or `--out other/` to start a separate set of run dirs. Nothing is ever overwritten. `budget.total_usd` and `seeds:` are not part of the hash. `demo.py` is the same experiment in Python (`python demo.py`).

## Experimenter skill

`skill/SKILL.md` teaches a coding agent the tested workflow, from `swarmlab doctor` to `swarmlab publish`, including the first-real-run checklist. Install it for Claude Code (`~/.claude/skills/swarmlab/SKILL.md`), Codex (`~/.agents/skills/swarmlab/SKILL.md`) or OpenCode (`~/.config/opencode/skills/swarmlab/SKILL.md`); the file has the copy command.
