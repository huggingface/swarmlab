# M2 phase-1 Flag Game report

Runs dir: `/home/node/local/runs/m2-vllm`; skipped (not ended / unreadable): none

## Summary

| arm | n | final acc (mean ± sd) | final consensus | rounds to 0.9 cons. (median) | read_rate r2-10 | posts/agent/round | spend/run (swarm + meas.) | calls/run | wall/run |
|---|---|---|---|---|---|---|---|---|---|
| bc-qwen9b-16 | 1 | 0.688 ± 0.000 | 0.688 | never | 0.007 | 0.000 | $0.000 + $0.000 | 418 | 3.2 min |
| bc-qwen9b-64 | 1 | 0.641 ± 0.000 | 0.641 | never | 0.024 | 0.002 | $0.000 + $0.000 | 1708 | 7.0 min |
| gossip-qwen9b-16 | 1 | 0.688 ± 0.000 | 0.688 | never | 0.000 | 0.056 | $0.000 + $0.000 | 452 | 2.6 min |

## Trajectories (mean over seeds)

**bc-qwen9b-16**

| metric | r1 | r2 | r3 | r4 | r5 | r6 | r7 | r8 | r9 | r10 |
|---|---|---|---|---|---|---|---|---|---|---|
| accuracy | 0.25 | 0.62 | 0.69 | 0.69 | 0.69 | 0.75 | 0.69 | 0.69 | 0.69 | 0.69 |
| consensus | 0.25 | 0.62 | 0.69 | 0.69 | 0.69 | 0.75 | 0.69 | 0.69 | 0.69 | 0.69 |
| entropy | 1.88 | 1.50 | 1.37 | 1.37 | 1.50 | 1.19 | 1.37 | 1.37 | 1.50 | 1.37 |
| read_rate | 0.00 | 0.00 | 0.06 | 0.00 | 0.00 | 0.00 | 0.00 | 0.00 | 0.00 | 0.00 |

**bc-qwen9b-64**

| metric | r1 | r2 | r3 | r4 | r5 | r6 | r7 | r8 | r9 | r10 |
|---|---|---|---|---|---|---|---|---|---|---|
| accuracy | 0.41 | 0.55 | 0.59 | 0.62 | 0.61 | 0.59 | 0.61 | 0.61 | 0.62 | 0.64 |
| consensus | 0.41 | 0.55 | 0.59 | 0.62 | 0.61 | 0.59 | 0.61 | 0.61 | 0.62 | 0.64 |
| entropy | 2.00 | 1.66 | 1.63 | 1.42 | 1.55 | 1.51 | 1.44 | 1.44 | 1.42 | 1.42 |
| read_rate | 0.02 | 0.03 | 0.06 | 0.03 | 0.02 | 0.02 | 0.02 | 0.02 | 0.02 | 0.02 |

**gossip-qwen9b-16**

| metric | r1 | r2 | r3 | r4 | r5 | r6 | r7 | r8 | r9 | r10 |
|---|---|---|---|---|---|---|---|---|---|---|
| accuracy | 0.12 | 0.50 | 0.56 | 0.62 | 0.69 | 0.69 | 0.69 | 0.69 | 0.69 | 0.69 |
| consensus | 0.12 | 0.50 | 0.56 | 0.62 | 0.69 | 0.69 | 0.69 | 0.69 | 0.69 | 0.69 |
| entropy | 2.12 | 2.08 | 1.79 | 1.67 | 1.50 | 1.50 | 1.50 | 1.50 | 1.50 | 1.50 |
| read_rate | 0.00 | 0.00 | 0.00 | 0.00 | 0.00 | 0.00 | 0.00 | 0.00 | 0.00 | 0.00 |

## Where the swarm went (final committed guesses)

| run | truth | rival | matches score | guess counts | on rival | on truth | elsewhere |
|---|---|---|---|---|---|---|---|
| m2-vllm-qwen9b__bc-qwen9b-16__s1 | G | F | yes | G(T):11, C:2, F(R):2, A:1 | 0.12 | 0.69 | 0.19 |
| m2-vllm-qwen9b__bc-qwen9b-64__s1 | G | F | yes | G(T):41, C:16, D:3, F(R):3, H:1 | 0.05 | 0.64 | 0.31 |
| m2-vllm-qwen9b__gossip-qwen9b-16__s1 | G | F | yes | G(T):11, C:2, D:1, A:1, F(R):1 | 0.06 | 0.69 | 0.25 |

## Probe vs world belief

**bc-qwen9b-16**

| metric | r1 | r2 | r3 | r4 | r5 | r6 | r7 | r8 | r9 | r10 |
|---|---|---|---|---|---|---|---|---|---|---|
| probe consensus | 0.56 | 0.62 | 0.69 | 0.69 | 0.69 | 0.69 | 0.69 | 0.69 | 0.69 | 0.69 |
| world consensus | 0.25 | 0.62 | 0.69 | 0.69 | 0.69 | 0.75 | 0.69 | 0.69 | 0.69 | 0.69 |
| probe accuracy | 0.56 | 0.62 | 0.69 | 0.69 | 0.69 | 0.69 | 0.69 | 0.69 | 0.69 | 0.69 |
| world accuracy | 0.25 | 0.62 | 0.69 | 0.69 | 0.69 | 0.75 | 0.69 | 0.69 | 0.69 | 0.69 |
| disagreement (probe != guess) | 0.00 | 0.00 | 0.00 | 0.00 | 0.00 | 0.06 | 0.00 | 0.00 | 0.06 | 0.00 |

**bc-qwen9b-64**

| metric | r1 | r2 | r3 | r4 | r5 | r6 | r7 | r8 | r9 | r10 |
|---|---|---|---|---|---|---|---|---|---|---|
| probe consensus | 0.52 | 0.56 | 0.59 | 0.61 | 0.59 | 0.59 | 0.58 | 0.61 | 0.61 | 0.59 |
| world consensus | 0.41 | 0.55 | 0.59 | 0.62 | 0.61 | 0.59 | 0.61 | 0.61 | 0.62 | 0.64 |
| probe accuracy | 0.52 | 0.56 | 0.59 | 0.61 | 0.59 | 0.59 | 0.58 | 0.61 | 0.61 | 0.59 |
| world accuracy | 0.41 | 0.55 | 0.59 | 0.62 | 0.61 | 0.59 | 0.61 | 0.61 | 0.62 | 0.64 |
| disagreement (probe != guess) | 0.00 | 0.02 | 0.03 | 0.03 | 0.05 | 0.00 | 0.05 | 0.02 | 0.03 | 0.06 |

**gossip-qwen9b-16**

| metric | r1 | r2 | r3 | r4 | r5 | r6 | r7 | r8 | r9 | r10 |
|---|---|---|---|---|---|---|---|---|---|---|
| probe consensus | 0.38 | 0.50 | 0.56 | 0.56 | 0.62 | 0.69 | 0.69 | 0.69 | 0.69 | 0.69 |
| world consensus | 0.12 | 0.50 | 0.56 | 0.62 | 0.69 | 0.69 | 0.69 | 0.69 | 0.69 | 0.69 |
| probe accuracy | 0.38 | 0.50 | 0.56 | 0.56 | 0.62 | 0.69 | 0.69 | 0.69 | 0.69 | 0.69 |
| world accuracy | 0.12 | 0.50 | 0.56 | 0.62 | 0.69 | 0.69 | 0.69 | 0.69 | 0.69 | 0.69 |
| disagreement (probe != guess) | 0.00 | 0.00 | 0.00 | 0.06 | 0.06 | 0.00 | 0.00 | 0.00 | 0.00 | 0.00 |

## Reading behaviour

| arm | read-turn share r1 | read-turn share r2 | read-turn share r3 | read-turn share r4 | read-turn share r5 | read-turn share r6 | read-turn share r7 | read-turn share r8 | read-turn share r9 | read-turn share r10 | deliveries/read | never-read agents |
|---|---|---|---|---|---|---|---|---|---|---|---|---|
| bc-qwen9b-16 | 0.00 | 0.00 | 0.06 | 0.00 | 0.00 | 0.00 | 0.00 | 0.00 | 0.00 | 0.00 | 0.0 | 0.94 |
| bc-qwen9b-64 | 0.02 | 0.05 | 0.06 | 0.03 | 0.02 | 0.02 | 0.02 | 0.02 | 0.02 | 0.02 | 0.4 | 0.91 |
| gossip-qwen9b-16 | 0.00 | 0.00 | 0.00 | 0.00 | 0.00 | 0.00 | 0.00 | 0.00 | 0.00 | 0.00 | n/a | 1.00 |

## Tool protocol health

| arm | yield kinds | finish reasons (turn usage) | length (responses) | errored turns | cache hits/responses |
|---|---|---|---|---|---|
| bc-qwen9b-16 | {'no_tool': 10, 'end_turn': 150} | {'length': 9, 'tool_calls': 248, 'stop': 1} | 9 | 0 | 0/418 |
| bc-qwen9b-64 | {'end_turn': 625, 'no_tool': 15} | {'tool_calls': 1053, 'length': 14, 'stop': 1} | 14 | 0 | 0/1708 |
| gossip-qwen9b-16 | {'no_tool': 12, 'end_turn': 148} | {'length': 9, 'tool_calls': 280, 'stop': 3} | 9 | 0 | 0/452 |

