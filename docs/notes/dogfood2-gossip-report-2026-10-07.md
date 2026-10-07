# swarmlab report

Runs dir: `runs`; skipped (not ended / unreadable): none

## Summary

| arm | n | final acc (mean ± sd) | final consensus | rounds to 0.9 cons. (median) | read_rate r2-5 | posts/agent/round | spend/run (swarm + meas.) | calls/run | wall/run |
|---|---|---|---|---|---|---|---|---|---|
| broadcast | 1 | 1.000 ± 0.000 | 1.000 | 3 | 1.000 | 0.367 | $0.405 + $0.117 | 98 | 1.4 min |
| gossip | 1 | 1.000 ± 0.000 | 1.000 | 3 | 0.958 | 0.300 | $0.371 + $0.098 | 95 | 1.0 min |

## Trajectories (mean over seeds)

**broadcast**

| metric | r1 | r2 | r3 | r4 | r5 |
|---|---|---|---|---|---|
| accuracy | 0.00 | 0.83 | 1.00 | 1.00 | 1.00 |
| consensus | 0.00 | 0.83 | 1.00 | 1.00 | 1.00 |
| entropy | n/a | n/a | n/a | n/a | n/a |
| read_rate | 0.33 | 1.00 | 1.00 | 1.00 | 1.00 |

**gossip**

| metric | r1 | r2 | r3 | r4 | r5 |
|---|---|---|---|---|---|
| accuracy | 0.33 | 0.83 | 1.00 | 1.00 | 1.00 |
| consensus | 0.33 | 0.83 | 1.00 | 1.00 | 1.00 |
| entropy | n/a | n/a | n/a | n/a | n/a |
| read_rate | 0.33 | 1.00 | 1.00 | 1.00 | 0.83 |

## Where the swarm went (final committed guesses)

| run | truth | rival | matches score | guess counts | on rival | on truth | elsewhere |
|---|---|---|---|---|---|---|---|
| gossipvsbcast__broadcast__s1 | G | F | yes | G(T):6 | 0.00 | 1.00 | 0.00 |
| gossipvsbcast__gossip__s1 | G | F | yes | G(T):6 | 0.00 | 1.00 | 0.00 |

## Probe vs world belief

**broadcast**

| metric | r1 | r2 | r3 | r4 | r5 |
|---|---|---|---|---|---|
| probe consensus | 0.83 | 0.83 | 1.00 | 0.67 | 1.00 |
| world consensus | 0.00 | 0.83 | 1.00 | 1.00 | 1.00 |
| probe accuracy | 0.83 | 0.83 | 1.00 | 0.67 | 1.00 |
| world accuracy | 0.00 | 0.83 | 1.00 | 1.00 | 1.00 |
| disagreement (probe != guess) | n/a | 0.00 | 0.00 | 0.33 | 0.00 |

**gossip**

| metric | r1 | r2 | r3 | r4 | r5 |
|---|---|---|---|---|---|
| probe consensus | 0.83 | 0.83 | 1.00 | 1.00 | 0.50 |
| world consensus | 0.33 | 0.83 | 1.00 | 1.00 | 1.00 |
| probe accuracy | 0.83 | 0.83 | 1.00 | 1.00 | 0.50 |
| world accuracy | 0.33 | 0.83 | 1.00 | 1.00 | 1.00 |
| disagreement (probe != guess) | 0.00 | 0.00 | 0.00 | 0.00 | 0.50 |

## Reading behaviour

| arm | read-turn share r1 | read-turn share r2 | read-turn share r3 | read-turn share r4 | read-turn share r5 | deliveries/read | never-read agents |
|---|---|---|---|---|---|---|---|
| broadcast | 0.33 | 1.00 | 1.00 | 1.00 | 1.00 | 2.1 | 0.00 |
| gossip | 0.33 | 1.00 | 1.00 | 1.00 | 0.83 | 0.4 | 0.00 |

## Tool protocol health

| arm | yield kinds | finish reasons (turn usage) | length (responses) | errored turns | cache hits/responses |
|---|---|---|---|---|---|
| broadcast | {'no_tool': 6, 'end_turn': 24} | {'max_tokens': 6, 'tool_use': 64} | 0 | 0 | 0/98 |
| gossip | {'end_turn': 26, 'no_tool': 4} | {'max_tokens': 7, 'tool_use': 61} | 0 | 0 | 0/95 |

