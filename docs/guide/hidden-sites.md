# Hidden Sites

An allocation world with a strict information boundary: agents must pool what they learn through the board to find out where effort pays, and then spread out instead of crowding the best site. Module: `swarmlab/world/hidden_sites.py` (its docstring is the exact reference). Example spec: `examples/hidden_sites.yaml`. Back to the [docs index](../README.md).

## The task

`n_sites` sites labelled A, B, C, ... have hidden success probabilities that stay fixed for the whole run. Every round each site offers one fresh discovery. Agents start unassigned. On its turn an agent may read the board, keep its site or move with `choose_site(site)`, and post. After every agent has had its turn, each agent holding a site makes one independent attempt there (choosing a site is not an attempt; an unassigned agent makes none), and the team scores one point per site where at least one attempt succeeded. More agents at a site raise its chance of scoring with diminishing returns:

    expected points W_r = sum_k [1 - (1 - p_k) ** n_k]   (end-of-round counts n_k)
    actual points   R_r = number of sites with at least one success

With 12 agents and the default probabilities, the references per round are 1.000 (everyone at the best site), 1.828 (independent uniform picks) and 2.635 (the optimum: 2, 3, 4, 3 agents on the four best sites); `score()["references"]` reports them for the run's assignment.

The world is meant for `options: {commit: immediate}` (one turn at a time, in a seeded order drawn anew every round), so later agents in a round can read earlier posts. The task text (`description()`) states the agent and round counts in words and has no other numbers.

## Parameters

`world: {type: hidden_sites, params: {...}}`

| param | default | meaning |
|---|---|---|
| `probabilities` | `[0.60, 0.35, 0.25, 0.15, 0.10, 0.06, 0.03, 0.01]` | one success probability per site, each in (0, 1) |
| `n_sites` | 8 | 1..26 (labels A..Z); must equal `len(probabilities)` |
| `rounds` | 20 | the round count agents are told ("Round r of 20."); the run still ends at `max_rounds`, so set both |
| `show_history` | true | adds the agent's own attempts per site to the observation |
| `permute` | true | shuffle the probabilities over the labels with the world rng at reset |

## Information boundary

The observation is one text part:

    Round 2 of 20.
    Your current position: C
    Your previous attempt: site C, failure
    Your attempts so far: C: 1 attempt, 0 successes
    Team points last round: 2
    Team points so far: 2

Agents never see the probabilities or their order, how many agents hold any site, other agents' positions or attempts, expected points or references. `choose_site` returns only `{"position", "previous"}`; a second choice in the same round is rejected with `{"error": "already_chose_this_round"}` (visible in the log as `accepted=False`); there are no status tools. Everything about the others comes from what they post.

## Trial stream and matched arms

Agent `a`'s attempt at site `k` in round `r` succeeds iff `derive(trial_key, "trial", r, a, k).random() < p_k`, with `trial_key` drawn from the world rng at reset. The outcome of every possible attempt is fixed by the seed, independent of tool-call counts, turn order and who else is at the site. Two arms with the same seed (say, a control and a one-sentence `system_prompt_append` treatment) therefore share the hidden assignment, the turn orders and the coin for every attempt: differences between them come from where agents stand, not from luck. `verify()` returns the assignment and the trial key, so metrics recompute every attempt.

Round r is resolved from the end-of-round positions when round r+1 begins; the last round is resolved lazily by `score()` and the metrics, so all 20 rounds are scored and `swarmlab replay` reproduces them.

## Metrics

All four need the truth and fold `round_started` plus accepted `choose_site` feedback; the denominator is the number of agents holding a site.

| metric | value |
|---|---|
| `hidden.expected_W` | W of the current round's end-of-round allocation |
| `hidden.expected_cum` | sum of W over the rounds so far (a natural primary outcome) |
| `hidden.actual_points` | R of the current round |
| `hidden.actual_cum` | sum of R so far |

`score()` also reports `W_by_round`, `R_by_round`, the final `allocation`, `unassigned`, the hidden `assignment` and the references. The replay page's world panel (`render_state`) shows each site's hidden p, its count and whether it scored.

## Fake script

`fake:hidden_chooser` is a free, deterministic plumbing check: agent aNNN takes site NNN mod n_sites (12 agents on 8 sites: 2, 2, 2, 2, 1, 1, 1, 1 over A..H), posts its site in round 1 and tries a second choice that must be rejected (12 rejections, round 1 only).

## Cost

12 agents on `hf:Qwen/Qwen3.8-27B:ovhcloud`, 20 rounds, `memory: window` with `window_rounds: 6`, `max_calls: 8`, `max_tokens: 1024`, reasoning off cost about $5.40 per run (2026-10-08/09; the one-sentence reminder arm $5.86). Most of it is prompt tokens: agents post long tallies of who is where and every later turn reads them. Budget about `soft_usd: 5.50`, `hard_usd: 6.00` per run and set `total_usd` for the experiment; a shorter window or fewer rounds cut it.
