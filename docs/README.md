# swarmlab documentation

The [project README](../README.md) is the overview. These pages are the reference: every flag, key, cap and edge case lives here.

## User guide

| page | read it when you want to |
|---|---|
| [Getting started](guide/getting-started.md) | install, run the free starter, understand skipped and repeated run dirs, install the experimenter skill |
| [Building experiments](guide/building-experiments.md) | see example worlds, topologies, participants, providers, probes, interventions, metrics and roles (`swarmlab plugins` lists every component); use the Python API; write a new world; change agents' prompts |
| [The Flag Game](guide/flag-game.md) | run image or text crops, the broadcast / gossip / manager protocols, or the paper replication |
| [Hidden Sites](guide/hidden-sites.md) | run the hidden-site allocation task: parameters, information boundary, matched trial stream, metrics, cost |
| [Crossed zones](guide/building-experiments.md#crossed-zones) | run the coloring world's relay task: groups, zones, generated boards, the oracle control, relay-status diagnostics |
| [Minesweeper](guide/minesweeper.md) | run the shared-board Minesweeper world: parameters and difficulty, information boundary, provable cells and allocation accounting, metrics, cost |
| [ProgramBench](guide/programbench.md) | run one ProgramBench task (rebuild a program from its binary and docs with one bash tool): install, the local and Docker backends, the official semantics kept and the documented deviations, the solo spec shape, the lockstep and async (`commit: async`) team shapes, snapshots, the free dry run, running in HF Jobs (API mode, task image) and grading with the official evaluator (`swarmlab programbench grade`) |
| [Cascade](guide/cascade.md) | run the information-cascade ("grader") scenario: one-shot workers, a noisy private signal, misleading first signals, board variants |
| [Real models and budgets](guide/real-models.md) | pick a model id, set keys, read prices, tune timeouts, preflight, set reasoning controls, self-host with `job run` (vLLM serve profiles), size `soft_usd` / `hard_usd` / `total_usd`, handle refusals, capture reasoning traces |
| [Spec reference](guide/spec-reference.md) | look up any YAML key, its shape and default (generated from the pydantic models) |
| [CLI and run directory](guide/cli.md) | look up a command's flags, exit codes, or a file in `runs/<run_id>/` |
| [Analysing and publishing](guide/analysis.md) | write a report, read the Parquet exports and sessions, write a truth-aware metric, publish to the Hub |

## For coding agents

- Working **on** the framework: start with [`AGENTS.md`](../AGENTS.md), then the binding contracts below.
- Running an **experiment** (design, smoke, launch, analyse, report, publish): follow [`skill/SKILL.md`](../skill/SKILL.md); its checklist names a command for every step and links back here for details.
- Every command takes `--json`; `swarmlab spec-reference` prints the spec reference and `swarmlab metrics` the metric list, so neither needs this page.

## Design and contracts

- [`DESIGN.md`](DESIGN.md): why the framework is shaped the way it is.
- [`INTERFACE.md`](INTERFACE.md) and its extensions [`M1b`](INTERFACE-M1b.md), [`M3a`](INTERFACE-M3a.md), [`M3c`](INTERFACE-M3c.md), [`M4`](INTERFACE-M4.md), [`M5`](INTERFACE-M5.md), [`M6`](INTERFACE-M6.md): the binding contracts.
- [`handoff/INDEX.md`](handoff/INDEX.md): what each work package built.
- [`flag-game-paper-setup.md`](flag-game-paper-setup.md): the Flag Game paper's exact setup, which `INTERFACE-M6.md` implements.

## Known issues

- [`known-issues.md`](known-issues.md): open problems: JSON tool protocol on Haiku, DeepInfra tool-call stalls, estimates overstating spend, `validate` rejecting `vllm:` specs, and role and paired-run gaps.
