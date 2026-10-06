# M2 phase-1 Flag Game report

Runs dir: `/home/node/local/runs/m2`; skipped (not ended / unreadable): none

## Summary

| arm | n | final acc (mean ± sd) | final consensus | rounds to 0.9 cons. (median) | read_rate r2-10 | posts/agent/round | spend/run (swarm + meas.) | calls/run | wall/run |
|---|---|---|---|---|---|---|---|---|---|
| ? | 1 | 1.000 ± 0.000 | 1.000 | 2 | 1.000 | 0.250 | $0.126 + $0.034 | 40 | 1.7 min |
| bc-haiku | 1 | 1.000 ± 0.000 | 1.000 | 2 | 0.722 | 0.062 | $2.472 + $0.992 | 465 | 2.6 min |
| bc-qwen | 3 | 0.625 ± 0.108 | 0.625 | never | 0.009 | 0.002 | $0.162 + $0.093 | 413 | 10.2 min |
| gossip-qwen | 3 | 0.542 ± 0.201 | 0.604 | never | 0.002 | 0.002 | $0.167 + $0.095 | 417 | 11.1 min |

## Trajectories (mean over seeds)

**?**

| metric | r1 | r2 | r3 | r4 | r5 | r6 | r7 | r8 | r9 | r10 |
|---|---|---|---|---|---|---|---|---|---|---|
| accuracy | 0.75 | 1.00 | 1.00 | n/a | n/a | n/a | n/a | n/a | n/a | n/a |
| consensus | 0.75 | 1.00 | 1.00 | n/a | n/a | n/a | n/a | n/a | n/a | n/a |
| entropy | n/a | n/a | n/a | n/a | n/a | n/a | n/a | n/a | n/a | n/a |
| read_rate | 0.00 | 1.00 | 1.00 | n/a | n/a | n/a | n/a | n/a | n/a | n/a |

**bc-haiku**

| metric | r1 | r2 | r3 | r4 | r5 | r6 | r7 | r8 | r9 | r10 |
|---|---|---|---|---|---|---|---|---|---|---|
| accuracy | 0.81 | 0.94 | 1.00 | 1.00 | 1.00 | 1.00 | 1.00 | 1.00 | 1.00 | 1.00 |
| consensus | 0.81 | 0.94 | 1.00 | 1.00 | 1.00 | 1.00 | 1.00 | 1.00 | 1.00 | 1.00 |
| entropy | 0.70 | 0.34 | 0.00 | 0.00 | 0.00 | 0.00 | 0.00 | 0.00 | 0.00 | 0.00 |
| read_rate | 0.19 | 0.81 | 0.88 | 0.69 | 0.69 | 0.69 | 0.69 | 0.69 | 0.69 | 0.69 |

**bc-qwen**

| metric | r1 | r2 | r3 | r4 | r5 | r6 | r7 | r8 | r9 | r10 |
|---|---|---|---|---|---|---|---|---|---|---|
| accuracy | 0.40 | 0.58 | 0.58 | 0.62 | 0.65 | 0.65 | 0.65 | 0.65 | 0.62 | 0.62 |
| consensus | 0.40 | 0.58 | 0.58 | 0.62 | 0.65 | 0.65 | 0.65 | 0.65 | 0.62 | 0.62 |
| entropy | 1.93 | 1.63 | 1.70 | 1.45 | 1.39 | 1.47 | 1.37 | 1.47 | 1.49 | 1.41 |
| read_rate | 0.00 | 0.04 | 0.00 | 0.02 | 0.00 | 0.02 | 0.00 | 0.00 | 0.00 | 0.00 |

**gossip-qwen**

| metric | r1 | r2 | r3 | r4 | r5 | r6 | r7 | r8 | r9 | r10 |
|---|---|---|---|---|---|---|---|---|---|---|
| accuracy | 0.38 | 0.52 | 0.54 | 0.52 | 0.52 | 0.50 | 0.50 | 0.50 | 0.50 | 0.54 |
| consensus | 0.38 | 0.56 | 0.58 | 0.58 | 0.58 | 0.56 | 0.56 | 0.56 | 0.56 | 0.60 |
| entropy | 1.67 | 1.60 | 1.54 | 1.41 | 1.53 | 1.59 | 1.52 | 1.59 | 1.52 | 1.39 |
| read_rate | 0.02 | 0.00 | 0.00 | 0.00 | 0.00 | 0.02 | 0.00 | 0.00 | 0.00 | 0.00 |

## Where the swarm went (final committed guesses)

| run | truth | rival | matches score | guess counts | on rival | on truth | elsewhere |
|---|---|---|---|---|---|---|---|
| m2-seq-check__s1 | G | F | yes | G(T):4 | 0.00 | 1.00 | 0.00 |
| m2-flaggame__bc-haiku__s1 | G | F | yes | G(T):16 | 0.00 | 1.00 | 0.00 |
| m2-flaggame__bc-qwen__s1 | G | F | yes | G(T):11, C:2, A:1, D:1, F(R):1 | 0.06 | 0.69 | 0.25 |
| m2-flaggame__bc-qwen__s2 | H | G | yes | H(T):8, G(R):6, C:1, B:1 | 0.38 | 0.50 | 0.12 |
| m2-flaggame__bc-qwen__s3 | G | E | yes | G(T):11, A:3, B:2 | 0.00 | 0.69 | 0.31 |
| m2-flaggame__gossip-qwen__s1 | G | F | yes | G(T):11, C:3, D:2 | 0.00 | 0.69 | 0.31 |
| m2-flaggame__gossip-qwen__s2 | H | G | yes | G(R):8, H(T):5, C:2, A:1 | 0.50 | 0.31 | 0.19 |
| m2-flaggame__gossip-qwen__s3 | G | E | yes | G(T):10, B:3, A:3 | 0.00 | 0.62 | 0.38 |

## Probe vs world belief

**?**

| metric | r1 | r2 | r3 | r4 | r5 | r6 | r7 | r8 | r9 | r10 |
|---|---|---|---|---|---|---|---|---|---|---|
| probe consensus | 0.75 | 1.00 | 1.00 | n/a | n/a | n/a | n/a | n/a | n/a | n/a |
| world consensus | 0.75 | 1.00 | 1.00 | n/a | n/a | n/a | n/a | n/a | n/a | n/a |
| probe accuracy | 0.75 | 1.00 | 1.00 | n/a | n/a | n/a | n/a | n/a | n/a | n/a |
| world accuracy | 0.75 | 1.00 | 1.00 | n/a | n/a | n/a | n/a | n/a | n/a | n/a |
| disagreement (probe != guess) | 0.00 | 0.00 | 0.00 | n/a | n/a | n/a | n/a | n/a | n/a | n/a |

**bc-haiku**

| metric | r1 | r2 | r3 | r4 | r5 | r6 | r7 | r8 | r9 | r10 |
|---|---|---|---|---|---|---|---|---|---|---|
| probe consensus | 0.81 | 0.94 | 1.00 | 1.00 | 1.00 | 1.00 | 1.00 | 1.00 | 1.00 | 1.00 |
| world consensus | 0.81 | 0.94 | 1.00 | 1.00 | 1.00 | 1.00 | 1.00 | 1.00 | 1.00 | 1.00 |
| probe accuracy | 0.81 | 0.94 | 1.00 | 1.00 | 1.00 | 1.00 | 1.00 | 1.00 | 1.00 | 1.00 |
| world accuracy | 0.81 | 0.94 | 1.00 | 1.00 | 1.00 | 1.00 | 1.00 | 1.00 | 1.00 | 1.00 |
| disagreement (probe != guess) | 0.00 | 0.00 | 0.00 | 0.00 | 0.00 | 0.00 | 0.00 | 0.00 | 0.00 | 0.00 |

**bc-qwen**

| metric | r1 | r2 | r3 | r4 | r5 | r6 | r7 | r8 | r9 | r10 |
|---|---|---|---|---|---|---|---|---|---|---|
| probe consensus | 0.56 | 0.58 | 0.56 | 0.62 | 0.65 | 0.65 | 0.65 | 0.65 | 0.60 | 0.62 |
| world consensus | 0.40 | 0.58 | 0.58 | 0.62 | 0.65 | 0.65 | 0.65 | 0.65 | 0.62 | 0.62 |
| probe accuracy | 0.56 | 0.58 | 0.56 | 0.62 | 0.65 | 0.65 | 0.65 | 0.65 | 0.60 | 0.62 |
| world accuracy | 0.40 | 0.58 | 0.58 | 0.62 | 0.65 | 0.65 | 0.65 | 0.65 | 0.62 | 0.62 |
| disagreement (probe != guess) | 0.00 | 0.02 | 0.04 | 0.02 | 0.00 | 0.04 | 0.02 | 0.04 | 0.06 | 0.02 |

**gossip-qwen**

| metric | r1 | r2 | r3 | r4 | r5 | r6 | r7 | r8 | r9 | r10 |
|---|---|---|---|---|---|---|---|---|---|---|
| probe consensus | 0.56 | 0.52 | 0.58 | 0.54 | 0.58 | 0.56 | 0.54 | 0.56 | 0.56 | 0.56 |
| world consensus | 0.38 | 0.56 | 0.58 | 0.58 | 0.58 | 0.56 | 0.56 | 0.56 | 0.56 | 0.60 |
| probe accuracy | 0.48 | 0.48 | 0.54 | 0.48 | 0.52 | 0.50 | 0.48 | 0.50 | 0.50 | 0.50 |
| world accuracy | 0.38 | 0.52 | 0.54 | 0.52 | 0.52 | 0.50 | 0.50 | 0.50 | 0.50 | 0.54 |
| disagreement (probe != guess) | 0.06 | 0.08 | 0.00 | 0.04 | 0.02 | 0.06 | 0.04 | 0.04 | 0.06 | 0.04 |

## Reading behaviour

| arm | read-turn share r1 | read-turn share r2 | read-turn share r3 | read-turn share r4 | read-turn share r5 | read-turn share r6 | read-turn share r7 | read-turn share r8 | read-turn share r9 | read-turn share r10 | deliveries/read | never-read agents |
|---|---|---|---|---|---|---|---|---|---|---|---|---|
| ? | 0.00 | 1.00 | 1.00 | n/a | n/a | n/a | n/a | n/a | n/a | n/a | 0.9 | 0.00 |
| bc-haiku | 0.19 | 0.81 | 0.88 | 0.69 | 0.69 | 0.69 | 0.69 | 0.69 | 0.69 | 0.69 | 1.0 | 0.06 |
| bc-qwen | 0.00 | 0.04 | 0.00 | 0.02 | 0.00 | 0.02 | 0.00 | 0.00 | 0.00 | 0.00 | 0.0 | 0.94 |
| gossip-qwen | 0.02 | 0.00 | 0.00 | 0.00 | 0.00 | 0.02 | 0.00 | 0.00 | 0.00 | 0.00 | 0.0 | 0.96 |

## Tool protocol health

| arm | yield kinds | finish reasons (turn usage) | length (responses) | errored turns | cache hits/responses |
|---|---|---|---|---|---|
| ? | {'end_turn': 12} | {'tool_use': 28} | 0 | 0 | 0/40 |
| bc-haiku | {'end_turn': 160} | {'tool_use': 304, 'max_tokens': 1} | 0 | 0 | 0/465 |
| bc-qwen | {'end_turn': 459, 'no_tool': 21} | {'tool_calls': 737, 'length': 21} | 21 | 0 | 0/1238 |
| gossip-qwen | {'end_turn': 461, 'no_tool': 19} | {'tool_calls': 752, 'length': 17, 'stop': 2} | 17 | 0 | 0/1251 |

