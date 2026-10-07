# M6 Flag Game VLM grid (synthetic candidates)

Runs dir: `/home/node/local/runs/m6`; skipped (not ended / unreadable): none

## Summary

| arm | n | final acc (mean ± sd) | final consensus | rounds to 0.9 cons. (median) | read_rate r2-10 | posts/agent/round | spend/run (swarm + meas.) | calls/run | wall/run |
|---|---|---|---|---|---|---|---|---|---|
| bc-128 | 1 | 0.555 ± 0.000 | 0.555 | never | 0.000 | 0.003 | $1.336 + $0.657 | 3847 | 43.5 min |
| bc-16 | 1 | 0.750 ± 0.000 | 0.750 | never | 0.000 | 0.000 | $0.166 + $0.082 | 480 | 13.6 min |
| bc-4 | 1 | 0.750 ± 0.000 | 0.750 | 7 | 0.000 | 0.000 | $0.042 + $0.021 | 120 | 6.1 min |
| gossip-16 | 1 | 0.625 ± 0.000 | 0.625 | never | 0.000 | 0.000 | $0.167 + $0.082 | 481 | 13.0 min |
| gossip-4 | 1 | 0.750 ± 0.000 | 0.750 | never | 0.000 | 0.000 | $0.042 + $0.021 | 120 | 5.2 min |
| manager-16 | 1 | 0.533 ± 0.000 | 0.533 | never | 0.181 | 0.113 | $0.191 + $0.083 | 519 | 20.1 min |
| manager-4 | 1 | 0.667 ± 0.000 | 0.667 | never | 0.250 | 0.000 | $0.041 + $0.020 | 120 | 4.8 min |

## Trajectories (mean over seeds)

**bc-128**

| metric | r1 | r2 | r3 | r4 | r5 | r6 | r7 | r8 | r9 | r10 |
|---|---|---|---|---|---|---|---|---|---|---|
| accuracy | 0.01 | 0.37 | 0.27 | 0.41 | 0.44 | 0.53 | 0.47 | 0.53 | 0.45 | 0.55 |
| consensus | 0.54 | 0.37 | 0.34 | 0.41 | 0.44 | 0.53 | 0.47 | 0.53 | 0.45 | 0.55 |
| entropy | 1.56 | 1.92 | 1.89 | 2.23 | 1.89 | 1.86 | 2.07 | 1.91 | 1.88 | 1.78 |
| read_rate | 0.00 | 0.00 | 0.00 | 0.00 | 0.00 | 0.00 | 0.00 | 0.00 | 0.00 | 0.00 |
| belief.accuracy@probe:belief | 0.01 | 0.34 | 0.27 | 0.40 | 0.44 | 0.53 | 0.48 | 0.53 | 0.47 | 0.55 |
| belief.consensus@probe:belief | 0.54 | 0.34 | 0.33 | 0.40 | 0.44 | 0.53 | 0.48 | 0.53 | 0.47 | 0.55 |
| belief.entropy@probe:belief | 1.56 | 1.92 | 1.91 | 2.24 | 1.89 | 1.84 | 2.01 | 1.91 | 1.86 | 1.78 |
| belief.polarization | 2.00 | 3.00 | 3.00 | 2.00 | 2.00 | 2.00 | 2.00 | 2.00 | 2.00 | 2.00 |
| belief.polarization@probe:belief | 2.00 | 3.00 | 3.00 | 2.00 | 2.00 | 2.00 | 2.00 | 2.00 | 2.00 | 2.00 |
| comm.post_rate | 0.00 | 0.00 | 0.00 | 0.00 | 0.00 | 0.03 | 0.00 | 0.00 | 0.00 | 0.00 |
| post_rate | 0.00 | 0.00 | 0.00 | 0.00 | 0.00 | 0.03 | 0.00 | 0.00 | 0.00 | 0.00 |

**bc-16**

| metric | r1 | r2 | r3 | r4 | r5 | r6 | r7 | r8 | r9 | r10 |
|---|---|---|---|---|---|---|---|---|---|---|
| accuracy | 0.06 | 0.50 | 0.44 | 0.56 | 0.56 | 0.69 | 0.69 | 0.62 | 0.50 | 0.75 |
| consensus | 0.44 | 0.50 | 0.44 | 0.56 | 0.56 | 0.69 | 0.69 | 0.62 | 0.50 | 0.75 |
| entropy | 1.68 | 1.65 | 1.85 | 1.97 | 1.59 | 1.37 | 1.37 | 1.63 | 1.65 | 1.06 |
| read_rate | 0.00 | 0.00 | 0.00 | 0.00 | 0.00 | 0.00 | 0.00 | 0.00 | 0.00 | 0.00 |
| belief.accuracy@probe:belief | 0.06 | 0.50 | 0.38 | 0.56 | 0.56 | 0.69 | 0.69 | 0.62 | 0.50 | 0.75 |
| belief.consensus@probe:belief | 0.44 | 0.50 | 0.38 | 0.56 | 0.56 | 0.69 | 0.69 | 0.62 | 0.50 | 0.75 |
| belief.entropy@probe:belief | 1.68 | 1.65 | 1.91 | 1.97 | 1.59 | 1.37 | 1.37 | 1.63 | 1.65 | 1.06 |
| belief.polarization | 2.00 | 2.00 | 2.00 | 1.00 | 2.00 | 1.00 | 1.00 | 1.00 | 2.00 | 1.00 |
| belief.polarization@probe:belief | 2.00 | 2.00 | 3.00 | 1.00 | 2.00 | 1.00 | 1.00 | 1.00 | 2.00 | 1.00 |
| comm.post_rate | 0.00 | 0.00 | 0.00 | 0.00 | 0.00 | 0.00 | 0.00 | 0.00 | 0.00 | 0.00 |
| post_rate | 0.00 | 0.00 | 0.00 | 0.00 | 0.00 | 0.00 | 0.00 | 0.00 | 0.00 | 0.00 |

**bc-4**

| metric | r1 | r2 | r3 | r4 | r5 | r6 | r7 | r8 | r9 | r10 |
|---|---|---|---|---|---|---|---|---|---|---|
| accuracy | 0.00 | 0.75 | 0.25 | 0.75 | 0.50 | 0.75 | 1.00 | 0.75 | 0.50 | 0.75 |
| consensus | 0.50 | 0.75 | 0.75 | 0.75 | 0.50 | 0.75 | 1.00 | 0.75 | 0.50 | 0.75 |
| entropy | 1.50 | 0.81 | 0.81 | 0.81 | 1.50 | 0.81 | 0.00 | 0.81 | 1.50 | 0.81 |
| read_rate | 0.00 | 0.00 | 0.00 | 0.00 | 0.00 | 0.00 | 0.00 | 0.00 | 0.00 | 0.00 |
| belief.accuracy@probe:belief | 0.00 | 0.50 | 0.25 | 0.75 | 0.50 | 0.75 | 1.00 | 0.75 | 0.50 | 0.75 |
| belief.consensus@probe:belief | 0.50 | 0.50 | 0.75 | 0.75 | 0.50 | 0.75 | 1.00 | 0.75 | 0.50 | 0.75 |
| belief.entropy@probe:belief | 1.50 | 1.00 | 0.81 | 0.81 | 1.50 | 0.81 | 0.00 | 0.81 | 1.50 | 0.81 |
| belief.polarization | 3.00 | 2.00 | 2.00 | 2.00 | 3.00 | 2.00 | 1.00 | 2.00 | 3.00 | 2.00 |
| belief.polarization@probe:belief | 3.00 | 2.00 | 2.00 | 2.00 | 3.00 | 2.00 | 1.00 | 2.00 | 3.00 | 2.00 |
| comm.post_rate | 0.00 | 0.00 | 0.00 | 0.00 | 0.00 | 0.00 | 0.00 | 0.00 | 0.00 | 0.00 |
| post_rate | 0.00 | 0.00 | 0.00 | 0.00 | 0.00 | 0.00 | 0.00 | 0.00 | 0.00 | 0.00 |

**gossip-16**

| metric | r1 | r2 | r3 | r4 | r5 | r6 | r7 | r8 | r9 | r10 |
|---|---|---|---|---|---|---|---|---|---|---|
| accuracy | 0.12 | 0.56 | 0.25 | 0.38 | 0.62 | 0.62 | 0.56 | 0.75 | 0.50 | 0.62 |
| consensus | 0.56 | 0.56 | 0.38 | 0.38 | 0.62 | 0.62 | 0.56 | 0.75 | 0.50 | 0.62 |
| entropy | 1.67 | 1.42 | 1.56 | 2.06 | 1.50 | 1.50 | 1.67 | 1.31 | 1.77 | 1.67 |
| read_rate | 0.00 | 0.00 | 0.00 | 0.00 | 0.00 | 0.00 | 0.00 | 0.00 | 0.00 | 0.00 |
| belief.accuracy@probe:belief | 0.12 | 0.50 | 0.25 | 0.38 | 0.62 | 0.62 | 0.56 | 0.75 | 0.50 | 0.62 |
| belief.consensus@probe:belief | 0.56 | 0.50 | 0.38 | 0.38 | 0.62 | 0.62 | 0.56 | 0.75 | 0.50 | 0.62 |
| belief.entropy@probe:belief | 1.67 | 1.48 | 1.56 | 2.06 | 1.50 | 1.50 | 1.67 | 1.31 | 1.77 | 1.67 |
| belief.polarization | 1.00 | 2.00 | 3.00 | 2.00 | 1.00 | 1.00 | 1.00 | 1.00 | 2.00 | 1.00 |
| belief.polarization@probe:belief | 1.00 | 2.00 | 3.00 | 2.00 | 1.00 | 1.00 | 1.00 | 1.00 | 2.00 | 1.00 |
| comm.post_rate | 0.00 | 0.00 | 0.00 | 0.00 | 0.00 | 0.00 | 0.00 | 0.00 | 0.00 | 0.00 |
| post_rate | 0.00 | 0.00 | 0.00 | 0.00 | 0.00 | 0.00 | 0.00 | 0.00 | 0.00 | 0.00 |

**gossip-4**

| metric | r1 | r2 | r3 | r4 | r5 | r6 | r7 | r8 | r9 | r10 |
|---|---|---|---|---|---|---|---|---|---|---|
| accuracy | 0.00 | 0.75 | 0.50 | 0.75 | 0.50 | 0.50 | 0.75 | 0.50 | 0.75 | 0.75 |
| consensus | 0.50 | 0.75 | 0.50 | 0.75 | 0.50 | 0.50 | 0.75 | 0.50 | 0.75 | 0.75 |
| entropy | 1.50 | 0.81 | 1.50 | 0.81 | 1.50 | 1.50 | 0.81 | 1.50 | 0.81 | 0.81 |
| read_rate | 0.00 | 0.00 | 0.00 | 0.00 | 0.00 | 0.00 | 0.00 | 0.00 | 0.00 | 0.00 |
| belief.accuracy@probe:belief | 0.00 | 0.75 | 0.50 | 0.75 | 0.50 | 0.50 | 0.75 | 0.50 | 0.75 | 0.75 |
| belief.consensus@probe:belief | 0.50 | 0.75 | 0.50 | 0.75 | 0.50 | 0.50 | 0.75 | 0.50 | 0.75 | 0.75 |
| belief.entropy@probe:belief | 1.50 | 0.81 | 1.50 | 0.81 | 1.50 | 1.50 | 0.81 | 1.50 | 0.81 | 0.81 |
| belief.polarization | 3.00 | 2.00 | 3.00 | 2.00 | 3.00 | 3.00 | 2.00 | 3.00 | 2.00 | 2.00 |
| belief.polarization@probe:belief | 3.00 | 2.00 | 3.00 | 2.00 | 3.00 | 3.00 | 2.00 | 3.00 | 2.00 | 2.00 |
| comm.post_rate | 0.00 | 0.00 | 0.00 | 0.00 | 0.00 | 0.00 | 0.00 | 0.00 | 0.00 | 0.00 |
| post_rate | 0.00 | 0.00 | 0.00 | 0.00 | 0.00 | 0.00 | 0.00 | 0.00 | 0.00 | 0.00 |

**manager-16**

| metric | r1 | r2 | r3 | r4 | r5 | r6 | r7 | r8 | r9 | r10 |
|---|---|---|---|---|---|---|---|---|---|---|
| accuracy | 0.07 | 0.47 | 0.33 | 0.60 | 0.67 | 0.60 | 0.67 | 0.53 | 0.60 | 0.53 |
| consensus | 0.67 | 0.47 | 0.40 | 0.60 | 0.67 | 0.60 | 0.67 | 0.53 | 0.60 | 0.53 |
| entropy | 1.38 | 1.69 | 1.78 | 1.47 | 1.24 | 1.55 | 1.24 | 1.67 | 1.55 | 1.64 |
| read_rate | 0.06 | 0.19 | 0.12 | 0.19 | 0.19 | 0.19 | 0.19 | 0.19 | 0.19 | 0.19 |
| belief.accuracy@probe:belief | 0.07 | 0.47 | 0.33 | 0.60 | 0.67 | 0.60 | 0.67 | 0.60 | 0.60 | 0.53 |
| belief.consensus@probe:belief | 0.67 | 0.47 | 0.47 | 0.60 | 0.67 | 0.60 | 0.67 | 0.60 | 0.60 | 0.53 |
| belief.entropy@probe:belief | 1.38 | 1.69 | 1.69 | 1.69 | 1.24 | 1.55 | 1.24 | 1.37 | 1.69 | 1.77 |
| belief.polarization | 2.00 | 2.00 | 3.00 | 2.00 | 2.00 | 2.00 | 2.00 | 3.00 | 2.00 | 2.00 |
| belief.polarization@probe:belief | 2.00 | 2.00 | 2.00 | 2.00 | 2.00 | 2.00 | 2.00 | 3.00 | 2.00 | 2.00 |
| comm.post_rate | 0.00 | 0.06 | 0.06 | 0.06 | 0.12 | 0.12 | 0.19 | 0.19 | 0.19 | 0.12 |
| post_rate | 0.00 | 0.06 | 0.06 | 0.06 | 0.12 | 0.12 | 0.19 | 0.19 | 0.19 | 0.12 |

**manager-4**

| metric | r1 | r2 | r3 | r4 | r5 | r6 | r7 | r8 | r9 | r10 |
|---|---|---|---|---|---|---|---|---|---|---|
| accuracy | 0.00 | 0.67 | 0.33 | 0.67 | 0.67 | 0.33 | 0.67 | 0.67 | 0.67 | 0.67 |
| consensus | 0.33 | 0.67 | 0.67 | 0.67 | 0.67 | 0.33 | 0.67 | 0.67 | 0.67 | 0.67 |
| entropy | 1.58 | 0.92 | 0.92 | 0.92 | 0.92 | 1.58 | 0.92 | 0.92 | 0.92 | 0.92 |
| read_rate | 0.25 | 0.25 | 0.25 | 0.25 | 0.25 | 0.25 | 0.25 | 0.25 | 0.25 | 0.25 |
| belief.accuracy@probe:belief | 0.00 | 0.67 | 0.33 | 0.67 | 0.67 | 0.33 | 0.67 | 0.67 | 0.67 | 0.67 |
| belief.consensus@probe:belief | 0.33 | 0.67 | 0.67 | 0.67 | 0.67 | 0.33 | 0.67 | 0.67 | 0.67 | 0.67 |
| belief.entropy@probe:belief | 1.58 | 0.92 | 0.92 | 0.92 | 0.92 | 1.58 | 0.92 | 0.92 | 0.92 | 0.92 |
| belief.polarization | 3.00 | 2.00 | 2.00 | 2.00 | 2.00 | 3.00 | 2.00 | 2.00 | 2.00 | 2.00 |
| belief.polarization@probe:belief | 3.00 | 2.00 | 2.00 | 2.00 | 2.00 | 3.00 | 2.00 | 2.00 | 2.00 | 2.00 |
| comm.post_rate | 0.00 | 0.00 | 0.00 | 0.00 | 0.00 | 0.00 | 0.00 | 0.00 | 0.00 | 0.00 |
| post_rate | 0.00 | 0.00 | 0.00 | 0.00 | 0.00 | 0.00 | 0.00 | 0.00 | 0.00 | 0.00 |

## Where the swarm went (final committed guesses)

| run | truth | rival | matches score | guess counts | on rival | on truth | elsewhere |
|---|---|---|---|---|---|---|---|
| m6-flag-vlm__bc-128__s1 | G | F | yes | G(T):71, H:32, E:14, A:5, F(R):3, D:2, C:1 | 0.02 | 0.55 | 0.42 |
| m6-flag-vlm__bc-16__s1 | G | F | yes | G(T):12, E:2, H:2 | 0.00 | 0.75 | 0.25 |
| m6-flag-vlm__bc-4__s1 | G | F | yes | G(T):3, A:1 | 0.00 | 0.75 | 0.25 |
| m6-flag-vlm__gossip-16__s1 | G | F | yes | G(T):10, H:2, E:2, A:1, D:1 | 0.00 | 0.62 | 0.38 |
| m6-flag-vlm__gossip-4__s1 | G | F | yes | G(T):3, E:1 | 0.00 | 0.75 | 0.25 |
| m6-flag-vlm__manager-16__s1 | G | F | yes | G(T):8, H:4, E:2, F(R):1 | 0.06 | 0.50 | 0.44 |
| m6-flag-vlm__manager-4__s1 | G | F | yes | G(T):2, E:1 | 0.00 | 0.50 | 0.50 |

## Terminal states (Flag Game paper)

Share of runs per class of the endpoint distribution (s1 >= 0.85 consensus, camps >= 0.25); terminal truth mass is the mean share of the endpoint on the truth. Endpoint: the manager's final decision when a blind agent guessed, else the last belief-probe round, else the final committed guesses.

| arm | runs | correct consensus | wrong consensus | polarized | fragmented | terminal truth mass | endpoint |
|---|---|---|---|---|---|---|---|
| bc-128 | 1 | 0.00 | 0.00 | 1.00 | 0.00 | 0.555 | probes 1 |
| bc-16 | 1 | 0.00 | 0.00 | 0.00 | 1.00 | 0.750 | probes 1 |
| bc-4 | 1 | 0.00 | 0.00 | 1.00 | 0.00 | 0.750 | probes 1 |
| gossip-16 | 1 | 0.00 | 0.00 | 0.00 | 1.00 | 0.625 | probes 1 |
| gossip-4 | 1 | 0.00 | 0.00 | 1.00 | 0.00 | 0.750 | probes 1 |
| manager-16 | 1 | 0.00 | 0.00 | 1.00 | 0.00 | 0.533 | probes 1 |
| manager-4 | 1 | 0.00 | 0.00 | 1.00 | 0.00 | 0.667 | probes 1 |

## Probe vs world belief

**bc-128**

| metric | r1 | r2 | r3 | r4 | r5 | r6 | r7 | r8 | r9 | r10 |
|---|---|---|---|---|---|---|---|---|---|---|
| probe consensus | 0.54 | 0.34 | 0.33 | 0.40 | 0.44 | 0.53 | 0.48 | 0.53 | 0.47 | 0.55 |
| world consensus | 0.54 | 0.37 | 0.34 | 0.41 | 0.44 | 0.53 | 0.47 | 0.53 | 0.45 | 0.55 |
| probe accuracy | 0.01 | 0.34 | 0.27 | 0.40 | 0.44 | 0.53 | 0.48 | 0.53 | 0.47 | 0.55 |
| world accuracy | 0.01 | 0.37 | 0.27 | 0.41 | 0.44 | 0.53 | 0.47 | 0.53 | 0.45 | 0.55 |
| disagreement (probe != guess) | 0.00 | 0.05 | 0.05 | 0.01 | 0.00 | 0.02 | 0.02 | 0.00 | 0.02 | 0.00 |

Probes skipped: none.

**bc-16**

| metric | r1 | r2 | r3 | r4 | r5 | r6 | r7 | r8 | r9 | r10 |
|---|---|---|---|---|---|---|---|---|---|---|
| probe consensus | 0.44 | 0.50 | 0.38 | 0.56 | 0.56 | 0.69 | 0.69 | 0.62 | 0.50 | 0.75 |
| world consensus | 0.44 | 0.50 | 0.44 | 0.56 | 0.56 | 0.69 | 0.69 | 0.62 | 0.50 | 0.75 |
| probe accuracy | 0.06 | 0.50 | 0.38 | 0.56 | 0.56 | 0.69 | 0.69 | 0.62 | 0.50 | 0.75 |
| world accuracy | 0.06 | 0.50 | 0.44 | 0.56 | 0.56 | 0.69 | 0.69 | 0.62 | 0.50 | 0.75 |
| disagreement (probe != guess) | 0.00 | 0.00 | 0.06 | 0.00 | 0.00 | 0.00 | 0.00 | 0.00 | 0.00 | 0.00 |

Probes skipped: none.

**bc-4**

| metric | r1 | r2 | r3 | r4 | r5 | r6 | r7 | r8 | r9 | r10 |
|---|---|---|---|---|---|---|---|---|---|---|
| probe consensus | 0.50 | 0.50 | 0.75 | 0.75 | 0.50 | 0.75 | 1.00 | 0.75 | 0.50 | 0.75 |
| world consensus | 0.50 | 0.75 | 0.75 | 0.75 | 0.50 | 0.75 | 1.00 | 0.75 | 0.50 | 0.75 |
| probe accuracy | 0.00 | 0.50 | 0.25 | 0.75 | 0.50 | 0.75 | 1.00 | 0.75 | 0.50 | 0.75 |
| world accuracy | 0.00 | 0.75 | 0.25 | 0.75 | 0.50 | 0.75 | 1.00 | 0.75 | 0.50 | 0.75 |
| disagreement (probe != guess) | 0.00 | 0.25 | 0.00 | 0.00 | 0.00 | 0.00 | 0.00 | 0.00 | 0.00 | 0.00 |

Probes skipped: none.

**gossip-16**

| metric | r1 | r2 | r3 | r4 | r5 | r6 | r7 | r8 | r9 | r10 |
|---|---|---|---|---|---|---|---|---|---|---|
| probe consensus | 0.56 | 0.50 | 0.38 | 0.38 | 0.62 | 0.62 | 0.56 | 0.75 | 0.50 | 0.62 |
| world consensus | 0.56 | 0.56 | 0.38 | 0.38 | 0.62 | 0.62 | 0.56 | 0.75 | 0.50 | 0.62 |
| probe accuracy | 0.12 | 0.50 | 0.25 | 0.38 | 0.62 | 0.62 | 0.56 | 0.75 | 0.50 | 0.62 |
| world accuracy | 0.12 | 0.56 | 0.25 | 0.38 | 0.62 | 0.62 | 0.56 | 0.75 | 0.50 | 0.62 |
| disagreement (probe != guess) | 0.00 | 0.06 | 0.00 | 0.00 | 0.00 | 0.00 | 0.00 | 0.00 | 0.00 | 0.00 |

Probes skipped: none.

**gossip-4**

| metric | r1 | r2 | r3 | r4 | r5 | r6 | r7 | r8 | r9 | r10 |
|---|---|---|---|---|---|---|---|---|---|---|
| probe consensus | 0.50 | 0.75 | 0.50 | 0.75 | 0.50 | 0.50 | 0.75 | 0.50 | 0.75 | 0.75 |
| world consensus | 0.50 | 0.75 | 0.50 | 0.75 | 0.50 | 0.50 | 0.75 | 0.50 | 0.75 | 0.75 |
| probe accuracy | 0.00 | 0.75 | 0.50 | 0.75 | 0.50 | 0.50 | 0.75 | 0.50 | 0.75 | 0.75 |
| world accuracy | 0.00 | 0.75 | 0.50 | 0.75 | 0.50 | 0.50 | 0.75 | 0.50 | 0.75 | 0.75 |
| disagreement (probe != guess) | 0.00 | 0.00 | 0.00 | 0.00 | 0.00 | 0.00 | 0.00 | 0.00 | 0.00 | 0.00 |

Probes skipped: none.

**manager-16**

| metric | r1 | r2 | r3 | r4 | r5 | r6 | r7 | r8 | r9 | r10 |
|---|---|---|---|---|---|---|---|---|---|---|
| probe consensus | 0.62 | 0.44 | 0.44 | 0.56 | 0.62 | 0.56 | 0.62 | 0.56 | 0.56 | 0.50 |
| world consensus | 0.67 | 0.47 | 0.40 | 0.60 | 0.67 | 0.60 | 0.67 | 0.53 | 0.60 | 0.53 |
| probe accuracy | 0.06 | 0.44 | 0.31 | 0.56 | 0.62 | 0.56 | 0.62 | 0.56 | 0.56 | 0.50 |
| world accuracy | 0.07 | 0.47 | 0.33 | 0.60 | 0.67 | 0.60 | 0.67 | 0.53 | 0.60 | 0.53 |
| disagreement (probe != guess) | 0.00 | 0.00 | 0.07 | 0.07 | 0.07 | 0.00 | 0.00 | 0.07 | 0.07 | 0.07 |

Probes skipped: none.

**manager-4**

| metric | r1 | r2 | r3 | r4 | r5 | r6 | r7 | r8 | r9 | r10 |
|---|---|---|---|---|---|---|---|---|---|---|
| probe consensus | 0.25 | 0.50 | 0.50 | 0.50 | 0.50 | 0.25 | 0.50 | 0.50 | 0.50 | 0.50 |
| world consensus | 0.33 | 0.67 | 0.67 | 0.67 | 0.67 | 0.33 | 0.67 | 0.67 | 0.67 | 0.67 |
| probe accuracy | 0.00 | 0.50 | 0.25 | 0.50 | 0.50 | 0.25 | 0.50 | 0.50 | 0.50 | 0.50 |
| world accuracy | 0.00 | 0.67 | 0.33 | 0.67 | 0.67 | 0.33 | 0.67 | 0.67 | 0.67 | 0.67 |
| disagreement (probe != guess) | 0.00 | 0.00 | 0.00 | 0.00 | 0.00 | 0.00 | 0.00 | 0.00 | 0.00 | 0.00 |

Probes skipped: none.

## Reading behaviour

| arm | read-turn share r1 | read-turn share r2 | read-turn share r3 | read-turn share r4 | read-turn share r5 | read-turn share r6 | read-turn share r7 | read-turn share r8 | read-turn share r9 | read-turn share r10 | deliveries/read | never-read agents |
|---|---|---|---|---|---|---|---|---|---|---|---|---|
| bc-128 | 0.00 | 0.00 | 0.00 | 0.00 | 0.00 | 0.00 | 0.00 | 0.00 | 0.00 | 0.00 | n/a | 1.00 |
| bc-16 | 0.00 | 0.00 | 0.00 | 0.00 | 0.00 | 0.00 | 0.00 | 0.00 | 0.00 | 0.00 | n/a | 1.00 |
| bc-4 | 0.00 | 0.00 | 0.00 | 0.00 | 0.00 | 0.00 | 0.00 | 0.00 | 0.00 | 0.00 | n/a | 1.00 |
| gossip-16 | 0.00 | 0.00 | 0.00 | 0.00 | 0.00 | 0.00 | 0.00 | 0.00 | 0.00 | 0.00 | n/a | 1.00 |
| gossip-4 | 0.00 | 0.00 | 0.00 | 0.00 | 0.00 | 0.00 | 0.00 | 0.00 | 0.00 | 0.00 | n/a | 1.00 |
| manager-16 | 0.06 | 0.19 | 0.12 | 0.19 | 0.19 | 0.19 | 0.19 | 0.19 | 0.19 | 0.19 | 0.8 | 0.81 |
| manager-4 | 0.25 | 0.25 | 0.25 | 0.25 | 0.25 | 0.25 | 0.25 | 0.25 | 0.25 | 0.25 | 0.0 | 0.75 |

## Tool protocol health

| arm | yield kinds | finish reasons (turn usage) | length (responses) | errored turns | cache hits/responses | max_tokens (responses) | rejected tool calls |
|---|---|---|---|---|---|---|---|
| bc-128 | {'end_turn': 1276, 'no_tool': 4} | {'tool_calls': 2563, 'stop': 4} | 0 | 0 | 0/3847 | 0 | 0/2563 |
| bc-16 | {'end_turn': 159, 'no_tool': 1} | {'tool_calls': 319, 'stop': 1} | 0 | 0 | 0/480 | 0 | 0/319 |
| bc-4 | {'end_turn': 40} | {'tool_calls': 80} | 0 | 0 | 0/120 | 0 | 0/80 |
| gossip-16 | {'end_turn': 159, 'no_tool': 1} | {'tool_calls': 320, 'stop': 1} | 0 | 0 | 0/481 | 0 | 0/320 |
| gossip-4 | {'end_turn': 40} | {'tool_calls': 80} | 0 | 0 | 0/120 | 0 | 0/80 |
| manager-16 | {'end_turn': 160} | {'tool_calls': 359} | 0 | 0 | 0/519 | 0 | 0/359 |
| manager-4 | {'end_turn': 40} | {'tool_calls': 80} | 0 | 0 | 0/120 | 0 | 0/80 |

## Protocol health

**bc-128**

| measure | value |
|---|---|
| runs / turns | 1 / 1280 |
| turn end kinds | end_turn 1276, no_tool 4 |
| turns errored | 0 of 1280 |
| finish reasons (responses) | tool_calls 2563, stop 1284 |
| inference responses, probes included (all / cache hits) | 3847 / 0 |
| retries (responses retried / extra attempts) | 2 / 2 |
| latency s, uncached (median / p90 / max) | 3.68 / 8.57 / 96.34 |
| model calls per turn, swarm (mean / max) | 2.01 / 4 |
| cost per round, swarm + measurement (mean / max) | $0.199 / $0.225 |
| tool calls rejected / answered | 0 / 2563 |
| world actions not accepted / committed | 0 / 1282 |
| probes skipped | none |

**bc-16**

| measure | value |
|---|---|
| runs / turns | 1 / 160 |
| turn end kinds | end_turn 159, no_tool 1 |
| turns errored | 0 of 160 |
| finish reasons (responses) | tool_calls 319, stop 161 |
| inference responses, probes included (all / cache hits) | 480 / 0 |
| retries (responses retried / extra attempts) | 1 / 1 |
| latency s, uncached (median / p90 / max) | 6.11 / 14.75 / 89.47 |
| model calls per turn, swarm (mean / max) | 2.00 / 2 |
| cost per round, swarm + measurement (mean / max) | $0.025 / $0.027 |
| tool calls rejected / answered | 0 / 319 |
| world actions not accepted / committed | 0 / 160 |
| probes skipped | none |

**bc-4**

| measure | value |
|---|---|
| runs / turns | 1 / 40 |
| turn end kinds | end_turn 40 |
| turns errored | 0 of 40 |
| finish reasons (responses) | tool_calls 80, stop 40 |
| inference responses, probes included (all / cache hits) | 120 / 0 |
| retries (responses retried / extra attempts) | 0 / 0 |
| latency s, uncached (median / p90 / max) | 5.81 / 12.62 / 73.33 |
| model calls per turn, swarm (mean / max) | 2.00 / 2 |
| cost per round, swarm + measurement (mean / max) | $0.006 / $0.007 |
| tool calls rejected / answered | 0 / 80 |
| world actions not accepted / committed | 0 / 40 |
| probes skipped | none |

**gossip-16**

| measure | value |
|---|---|
| runs / turns | 1 / 160 |
| turn end kinds | end_turn 159, no_tool 1 |
| turns errored | 0 of 160 |
| finish reasons (responses) | tool_calls 320, stop 161 |
| inference responses, probes included (all / cache hits) | 481 / 0 |
| retries (responses retried / extra attempts) | 3 / 3 |
| latency s, uncached (median / p90 / max) | 5.87 / 15.44 / 95.21 |
| model calls per turn, swarm (mean / max) | 2.01 / 3 |
| cost per round, swarm + measurement (mean / max) | $0.025 / $0.028 |
| tool calls rejected / answered | 0 / 320 |
| world actions not accepted / committed | 0 / 161 |
| probes skipped | none |

**gossip-4**

| measure | value |
|---|---|
| runs / turns | 1 / 40 |
| turn end kinds | end_turn 40 |
| turns errored | 0 of 40 |
| finish reasons (responses) | tool_calls 80, stop 40 |
| inference responses, probes included (all / cache hits) | 120 / 0 |
| retries (responses retried / extra attempts) | 0 / 0 |
| latency s, uncached (median / p90 / max) | 5.69 / 12.68 / 53.72 |
| model calls per turn, swarm (mean / max) | 2.00 / 2 |
| cost per round, swarm + measurement (mean / max) | $0.006 / $0.007 |
| tool calls rejected / answered | 0 / 80 |
| world actions not accepted / committed | 0 / 40 |
| probes skipped | none |

**manager-16**

| measure | value |
|---|---|
| runs / turns | 1 / 160 |
| turn end kinds | end_turn 160 |
| turns errored | 0 of 160 |
| finish reasons (responses) | tool_calls 359, stop 160 |
| inference responses, probes included (all / cache hits) | 519 / 0 |
| retries (responses retried / extra attempts) | 2 / 2 |
| latency s, uncached (median / p90 / max) | 5.71 / 15.70 / 165.19 |
| model calls per turn, swarm (mean / max) | 2.24 / 5 |
| cost per round, swarm + measurement (mean / max) | $0.027 / $0.031 |
| tool calls rejected / answered | 0 / 359 |
| world actions not accepted / committed | 0 / 140 |
| probes skipped | none |

**manager-4**

| measure | value |
|---|---|
| runs / turns | 1 / 40 |
| turn end kinds | end_turn 40 |
| turns errored | 0 of 40 |
| finish reasons (responses) | tool_calls 80, stop 40 |
| inference responses, probes included (all / cache hits) | 120 / 0 |
| retries (responses retried / extra attempts) | 0 / 0 |
| latency s, uncached (median / p90 / max) | 5.46 / 10.37 / 55.64 |
| model calls per turn, swarm (mean / max) | 2.00 / 2 |
| cost per round, swarm + measurement (mean / max) | $0.006 / $0.007 |
| tool calls rejected / answered | 0 / 80 |
| world actions not accepted / committed | 0 / 30 |
| probes skipped | none |

