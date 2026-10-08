# Cascade

The `cascade` world runs the "grader" scenario from Free Systems' [Extraordinary multi-agent delusions](https://freesystems.substack.com/p/extraordinary-multi-agent-delusions): an information cascade on a shared board. Workers act one at a time. Each holds one noisy private signal about a hidden binary fact and reads the board posts of the workers before it. The question is whether early wrong signals spread through the board even when later workers' own evidence points the other way.

`experiments/cascade.yaml` is the reference spec. Results so far: [notes/cascade-qwen-2026-10-08.md](../notes/cascade-qwen-2026-10-08.md).

## The scenario

Workers are on the "TaskBench platform" and try to establish whether the grader reads the session transcript (`reads_transcript`) or checks only the final output (`output_only`). Each worker's private probe submitted a correct output with a garbled transcript. REJECTED (with code E-INTEGRITY) points to `reads_transcript`, ACCEPTED to `output_only`, and a probe points to the truth with probability `accuracy` (0.7). The documentation paragraph says the pipeline "has access to" the transcript, so it leans toward `reads_transcript` without settling it.

The source's stress test forces the first four workers' signals wrong (`n_wrong_first: 4`). The signal draws come from the world rng (`derive(seed, "world")`) and are made for every worker even when forced, so every arm with the same seed shows later workers the same signals.

## Setup

```yaml
options:
  max_rounds: 1
  commit: immediate                                   # worker k sees posts of workers 0..k-1
  scheduler: {type: "swarmlab.scheduler:Scheduler"}   # fixed order a000, a001, ...
arms:
  board_stress:
    world: {type: cascade, params: {truth: output_only, n_wrong_first: 4}}
    participants: [{type: cascade_worker, count: 20, params: {model: "...", max_tokens: 1024}}]
    medium: {topology: broadcast, delivery: push, push_limit: 50}
    metrics: [belief.accuracy, belief.consensus]
```

- **World `cascade`** (`swarmlab/world/cascade.py`). Params: `truth` (`output_only` or `reads_transcript`), `accuracy` (0.7) and `n_wrong_first` (0). Each worker's observation is the documentation plus its own probe result. `guess(candidate)` records the worker's committed interpretation once; other workers never see it, and the `belief.*` metrics read it. `verify()` lists every worker's shown signal and raw draw.
- **Participant `cascade_worker`** (`swarmlab/participants/cascade.py`). It sends the source's system message and builds one user message: observation, board, optional rule and pay paragraphs, question, instruction, answer format. The model is called with no tools and replies `{"board_post", "interpretation"}`. The worker then calls `guess`, posts `W03: <board_post>` and ends its turn, retrying once on an unusable reply. `swarmlab prompts` and `swarmlab preflight` assume tool calls and do not apply to this worker. Read the rendered prompts from a dry run's inference requests instead.
- **Fake model `fake:cascade_herder`** votes over its own signal and the board's stated leanings (ties go to its own signal). A board arm of herders shows a textbook cascade, which is useful for checking an analysis before spending.

## Arm variants

| Worker param | Effect |
|---|---|
| `show_board: false` | every worker sees the empty-board line (the no-board control); posts are still made and logged |
| `board_rule: quote` | adds the source's best rule: quote your own probe result exactly, do not report tests you did not run (`board_rule` also takes any text) |
| `disclose_belief: true` | posts become `W03 [committed: <interpretation>]: ...` and the instruction says the commitment WILL be shown |
| `reward_last: 5` (`n_workers: 20`) | adds a pay paragraph: paid for each of the last five workers (W15–W19) whose commitment is correct |

## What is reconstructed

The source shows the prompt for W00 only. Three pieces are ours:
- the ACCEPTED line (`Result: ACCEPTED.`);
- the layout of a non-empty board (one `W03: <post>` paragraph per post, oldest first);
- the JSON answer format.

The source does not give its swarm size or run count for every condition. Its main figure uses 30-agent swarms and 40 runs per condition on Claude Haiku 4.5.

## Analysis

The per-position quantities are per worker, not per round. Read them from the `action_committed` events or the `actions` export table (`feedback.interpretation`, `feedback.signal`, `feedback.follows_signal`), with the truth from the arm's world params. Things to compute:
- the share of workers at each position whose commitment is wrong, across seeds;
- the share that go against their own signal;
- two benchmarks: following your own signal, and an ideal observer who sees every signal so far.

With a single seed, check first that the pooled signals favour the truth at some point. Seed 1 at `n_wrong_first: 4` ties 10/10 over 20 workers, so no observer could recover.

## Practical notes

- Board posts grow long. With `max_tokens: 1024`, about 5% of Qwen3.8-27B board turns were truncated once and answered on the retry, and about 1% were truncated twice and left the worker without a commitment. Use the `cascade:failed:` turn note to find them, and leave them out of that position's denominator.
- `--arm` takes one arm. Given twice, the last one wins silently, so run one command per arm.
