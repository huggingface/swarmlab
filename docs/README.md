# swarmlab documentation

The [project README](../README.md) is the overview. These pages are the reference: every flag, key, cap and edge case lives here.

## User guide

| page | read it when you want to |
|---|---|
| [Getting started](guide/getting-started.md) | install, run the free starter, understand skipped and repeated run dirs, install the experimenter skill |
| [Building experiments](guide/building-experiments.md) | see every built-in world, topology, participant, provider, probe, intervention, metric and role; use the Python API; write a new world; change agents' prompts |
| [The Flag Game](guide/flag-game.md) | run image or text crops, the broadcast / gossip / manager protocols, or the paper replication |
| [Hidden Sites](guide/hidden-sites.md) | run the hidden-site allocation task: parameters, information boundary, matched trial stream, metrics, cost |
| [Real models and budgets](guide/real-models.md) | pick a model id, set keys, read prices, tune timeouts, preflight, set reasoning controls, size `soft_usd` / `hard_usd` / `total_usd` |
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

## Results and notes

- [`notes/known-issues.md`](notes/known-issues.md): open problems: JSON tool protocol on Haiku, DeepInfra tool-call stalls, estimates overstating spend, `validate` rejecting `vllm:` specs, and role and paired-run gaps.
- [`notes/m6-results-2026-10-08.md`](notes/m6-results-2026-10-08.md): latest results (Flag Game paper replication on Gemma 4).
- [`notes/`](notes/): every run report and smoke note, dated.
