# CLI and run directory

Every `swarmlab` command and what a run directory holds. `swarmlab COMMAND --help` has the full flag text. Back to the [docs index](../README.md).

## CLI

```
swarmlab doctor [SPEC...] [--offline]    swarmlab models [--provider P] [--tools] [--search S] [--refresh]
swarmlab init [NAME] [--dir D] [--force]  swarmlab validate SPEC    swarmlab spec-reference
swarmlab plugins [GROUP]                  swarmlab metrics
swarmlab run SPEC [--arm A] [--seed N] [--max-rounds R] [--out runs/] [--yes] [--rerun] [--parallel N]
swarmlab estimate SPEC [--arm A] [--seed N] [--max-rounds R] [--prompt-growth G] [--calls-per-turn C] [--from RUN_DIR]
swarmlab preflight SPEC [--arm A] [--seed N] [--max-usd 0.05]
swarmlab replay RUN_DIR
swarmlab resume RUN_DIR [--budget-hard X | --add-budget D | --budget-soft X | --budget-measurement X]
swarmlab fork RUN_DIR --at 4 [--spec edited.yaml] [--arm A] [--max-rounds R] [--out DIR]
swarmlab view RUN_DIR [--publish OWNER/REPO]
swarmlab prompts SPEC --arm A [--seed N]  swarmlab report RUNS_DIR [--out report.md] [--stdout] [--include-fake] [--title T]
swarmlab export RUN_DIR [--out DIR] [--no-raw]
swarmlab publish RUNS_DIR_OR_RUN [--repo OWNER/REPO] [--public] [--tag T] [--no-raw]
swarmlab fetch-published OWNER/REPO RUN_ID [--out runs/] [--force]
swarmlab job run SPEC --model M [--flavor F] [--arm A] [--seeds 1,2] [--timeout 2h] [--per-round S]
                 [--vllm-version V] [--max-model-len L] [--profile auto|none|MODEL] [--vllm-flags F] [--vllm-flags-append F] [--probe-extra JSON] [--launch]
swarmlab job run SPEC [--flavor F] [--image I] [--arm A] [--seeds 0] [--timeout 7h] [--launch]   # API models, no vLLM
swarmlab job status JOB_ID    swarmlab job logs JOB_ID [--follow]    swarmlab job fetch RUN_ID [--out runs/]
swarmlab programbench grade RUN_DIR [--flavor F] [--timeout T] [--launch [--wait]]
swarmlab programbench fetch-grade RUN_DIR [--grade-json FILE]
```
`swarmlab job ...` runs a spec whose models are `vllm:<model>` in an HF Job with vLLM serving the model in the same job, and brings run dirs back from the bucket; `job run` only prints the `hf jobs run` command and the estimate unless `--launch` (docs/handoff/WP8.md). The vLLM serve flags come from `--vllm-flags`, else the built-in profile for `--model`, else the Qwen3.5 default; the readiness probe's request extra comes from `--probe-extra` (a JSON object; `'{}'` sends none), else the profile's, else the Qwen3.5 one only with the default flags under `--profile auto` (none for `--profile none` or `--vllm-flags` on a model not in the table); see [Self-hosting with HF Jobs](real-models.md#self-hosting-with-hf-jobs-vllm-serve-profiles).
Without `--model`, when no arm calls a `vllm:` model, `job run` plans an API-model job instead: no vLLM, a CPU flavor (`cpu-upgrade`), the model's key passed with `--secrets NAME`, and the ProgramBench task image for a programbench world ([Running in HF Jobs](programbench.md#running-in-hf-jobs)). `swarmlab programbench grade` writes a finished ProgramBench run's submission, then prints (with `--launch` submits) a second job that runs the official evaluator against the hidden tests. `fetch-grade` (or `--wait`) writes its score into the run's `run.json` ([Grading](programbench.md#grading)).
`swarmlab plugins` lists every plugin group (worlds, topologies, policies, participants, metrics, claim_policies, schedulers, probes, interventions, fake_scripts) with each entry point's name and one-line description; `swarmlab plugins GROUP` lists one group, and `--json` prints `{"ok": true, "plugins": {group: [{"name", "description", "target"}]}}` (`target` is `module:Class`). It is the complete list the guides defer to. `swarmlab metrics` is the `metrics` group with a note on probe-sourced belief metrics.

`validate` builds every arm, which checks each world's declared requirements (`World.requires`, e.g. a world that needs `commit: immediate` or `delivery: push`); a violation exits 2 naming every unmet setting. `run`, `prompts` and `estimate` build the experiment the same way, so they fail the same way; `run` and `estimate` check again with `--max-rounds` before estimating or starting anything, so an override the world refuses exits 2 once rather than failing every run. These are configuration checks only: participant-dependent compatibility (e.g. FlagGame's image modality refusing text-grid readers, after role models apply) is checked when each run initialises, before any inference, so `validate` and `estimate` may accept a spec that `run` then refuses.

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
  export/           tables, pi sessions and raw copies (after `export` or `publish`)
```
Run ids are `<experiment>__s<seed>` from Python, `<experiment>__<arm>__s<seed>` from YAML, plus `__f<round>_<n>` for a fork.
