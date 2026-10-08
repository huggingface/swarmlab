# CLI and run directory

Every `swarmlab` command and what a run directory holds. `swarmlab COMMAND --help` has the full flag text. Back to the [docs index](../README.md).

## CLI

```
swarmlab doctor [SPEC...] [--offline]    swarmlab models [--provider P] [--tools] [--search S] [--refresh]
swarmlab init [NAME] [--dir D] [--force]  swarmlab validate SPEC    swarmlab spec-reference    swarmlab metrics
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
swarmlab job run SPEC --model M [--flavor F] [--arm A] [--seeds 1,2] [--timeout 2h] [--per-round S] [--launch]
swarmlab job status JOB_ID    swarmlab job logs JOB_ID [--follow]    swarmlab job fetch RUN_ID [--out runs/]
```
`swarmlab job ...` runs a spec whose models are `vllm:<model>` in an HF Job with vLLM serving the model in the same job, and brings run dirs back from the bucket; `job run` only prints the `hf jobs run` command and the estimate unless `--launch` (docs/handoff/WP8.md).
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
