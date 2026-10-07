# swarmlab report

Runs dir: `runs`; skipped (not ended / unreadable): none

## Summary

| arm | n | final acc (mean ± sd) | final consensus | rounds to 0.9 cons. (median) | read_rate r2-5 | posts/agent/round | spend/run (swarm + meas.) | calls/run | wall/run |
|---|---|---|---|---|---|---|---|---|---|
| board | 1 | 1.000 ± 0.000 | 1.000 | 3 | 1.000 | 0.700 | $0.634 + $0.000 | 92 | 3.8 min |
| default | 1 | 1.000 ± 0.000 | 1.000 | 2 | 0.667 | 0.000 | $0.329 + $0.000 | 63 | 0.8 min |

## Trajectories (mean over seeds)

**board**

| metric | r1 | r2 | r3 | r4 | r5 |
|---|---|---|---|---|---|
| accuracy | 0.83 | 0.83 | 1.00 | 1.00 | 1.00 |
| consensus | 0.83 | 0.83 | 1.00 | 1.00 | 1.00 |
| entropy | n/a | n/a | n/a | n/a | n/a |
| read_rate | 0.33 | 1.00 | 1.00 | 1.00 | 1.00 |

**default**

| metric | r1 | r2 | r3 | r4 | r5 |
|---|---|---|---|---|---|
| accuracy | 0.83 | 1.00 | 1.00 | 1.00 | 1.00 |
| consensus | 0.83 | 1.00 | 1.00 | 1.00 | 1.00 |
| entropy | n/a | n/a | n/a | n/a | n/a |
| read_rate | 0.33 | 0.83 | 0.83 | 0.50 | 0.50 |

## Where the swarm went (final committed guesses)

| run | truth | rival | matches score | guess counts | on rival | on truth | elsewhere |
|---|---|---|---|---|---|---|---|
| boardcue__board__s1 | G | F | yes | G(T):6 | 0.00 | 1.00 | 0.00 |
| boardcue__default__s1 | G | F | yes | G(T):6 | 0.00 | 1.00 | 0.00 |

## Probe vs world belief

**board**

| metric | r1 | r2 | r3 | r4 | r5 |
|---|---|---|---|---|---|
| probe consensus | n/a | n/a | n/a | n/a | n/a |
| world consensus | 0.83 | 0.83 | 1.00 | 1.00 | 1.00 |
| probe accuracy | n/a | n/a | n/a | n/a | n/a |
| world accuracy | 0.83 | 0.83 | 1.00 | 1.00 | 1.00 |
| disagreement (probe != guess) | n/a | n/a | n/a | n/a | n/a |

**default**

| metric | r1 | r2 | r3 | r4 | r5 |
|---|---|---|---|---|---|
| probe consensus | n/a | n/a | n/a | n/a | n/a |
| world consensus | 0.83 | 1.00 | 1.00 | 1.00 | 1.00 |
| probe accuracy | n/a | n/a | n/a | n/a | n/a |
| world accuracy | 0.83 | 1.00 | 1.00 | 1.00 | 1.00 |
| disagreement (probe != guess) | n/a | n/a | n/a | n/a | n/a |

## Reading behaviour

| arm | read-turn share r1 | read-turn share r2 | read-turn share r3 | read-turn share r4 | read-turn share r5 | deliveries/read | never-read agents |
|---|---|---|---|---|---|---|---|
| board | 0.33 | 1.00 | 1.00 | 1.00 | 1.00 | 3.3 | 0.00 |
| default | 0.33 | 0.83 | 0.83 | 0.50 | 0.50 | 0.0 | 0.00 |

## Tool protocol health

| arm | yield kinds | finish reasons (turn usage) | length (responses) | errored turns | cache hits/responses |
|---|---|---|---|---|---|
| board | {'end_turn': 30} | {'tool_use': 84} | 0 | 0 | 9/84 |
| default | {'end_turn': 30} | {'tool_use': 63} | 0 | 0 | 0/63 |

