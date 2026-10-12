# Hidden Sites

An allocation world with a strict information boundary: agents must pool what they learn through the board to find out where effort pays, and then spread out instead of crowding the best site. Module: `swarmlab/world/hidden_sites.py` (its docstring is the exact reference). Example spec: `examples/hidden_sites.yaml`. Back to the [docs index](../README.md).

## The task

`n_sites` sites labelled A, B, C, ... have hidden success probabilities that stay fixed for the whole run. Every round each site offers one fresh discovery. Agents start unassigned. On its turn an agent may read the board, keep its site or move with `choose_site(site)` (at most one accepted choice per round), and post. At the end of the round, each agent holding a site makes one independent attempt there (choosing a site is not an attempt; an unassigned agent makes none), and the team scores one point per site where at least one attempt succeeded. More agents at a site raise its chance of scoring with diminishing returns:

    expected points W_r = sum_k [1 - (1 - p_k) ** n_k]   (end-of-round counts n_k)
    actual points   R_r = number of sites with at least one success

With 12 agents and the default probabilities, the references per round are 1.000 (everyone at the best site), 1.828 (independent uniform picks) and 2.635 (the optimum: 2, 3, 4, 3 agents on the four best sites); `score()["references"]` reports them for the run's assignment. The optimum comes from an exact greedy (each next agent goes where it adds the most expected points; exact because a site's gains shrink with every agent added), so it is instant for any number of agents and sites.

The world requires `options: {commit: immediate}` and `max_rounds` equal to its `rounds`; `swarmlab validate`, `estimate` and `run` refuse anything else. With the default scheduler and a broadcast board, as in the example, agents take turns one at a time in a seeded order drawn anew every round, and a post is readable by the agents whose turns come later in the same round and by everyone in later rounds. Other schedulers (`fixed`, `one_speaker`, ...) and topologies are legitimate variations, so the world's task text (`description()`) says nothing about turns, their order or who reads a post: it describes the task, the sites, the attempts, the one-choice-per-round rule, what an agent is told after each round, and that the others' positions are learned only through the board. It states the agent and round counts in words and has no other numbers. Tell agents how your experiment's rounds work with `system_prompt_append` (see [Changing what agents are told](building-experiments.md#changing-what-agents-are-told)); `examples/hidden_sites.yaml` does this for the default scheduler and broadcast in both arms, and the reminder arm adds its one treatment sentence after it.

## Parameters

`world: {type: hidden_sites, params: {...}}`

| param | default | meaning |
|---|---|---|
| `probabilities` | `[0.60, 0.35, 0.25, 0.15, 0.10, 0.06, 0.03, 0.01]` | one success probability per site, each in (0, 1) |
| `n_sites` | 8 | 1..26 (labels A..Z); must equal `len(probabilities)` |
| `rounds` | 20 | the round count agents are told ("Round r of 20."); the run ends at `max_rounds`, which must equal it |
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

Agent `a`'s attempt at site `k` in round `r` succeeds iff `derive(trial_key, "trial", r, a, k).random() < p_k`, with `trial_key` drawn from the world rng at reset. The outcome of every possible attempt is fixed by the seed, independent of tool-call counts, turn order and who else is at the site. Two arms with the same seed (say, a control and a one-sentence `system_prompt_append` treatment) therefore share the hidden assignment, the turn orders and the coin for every attempt: differences between them come from where agents stand, not from luck. `verify()` returns the assignment and the trial key.

Round r is resolved by the world's `end_round` hook, right after the round's last commit, from the positions then in force. The resolution (`W`, `R`, the site counts and each attempt's outcome) is logged as a `round_resolved` event, after the round's `action_committed` events; agents never see it.

## Forks and resumes

The world's snapshot carries only game state (the hidden assignment, trial key, positions, this round's choices and the resolved rounds); constructor params come from the spec the fork or resume runs under. A fork into the same world type may change `rounds` (with `max_rounds` to match), `show_history` and `permute` (which only acts at reset, so it changes nothing after the fork round). It may not change the site set: restoring a snapshot whose sites differ from the configured `n_sites` labels raises `ValueError` before any turn. Different `probabilities` over the same number of sites are not detected and have no effect, since the hidden assignment comes from the snapshot.

## Metrics

All four fold the `round_resolved` events, so `swarmlab replay` reproduces them; the denominator is the number of agents holding a site.

| metric | value |
|---|---|
| `hidden.expected_W` | W of the current round's end-of-round allocation |
| `hidden.expected_cum` | sum of W over the rounds so far (a natural primary outcome) |
| `hidden.actual_points` | R of the current round |
| `hidden.actual_cum` | sum of R so far |

`score()` also reports `W_by_round`, `R_by_round`, the final `allocation`, `unassigned`, the hidden `assignment` and the references. The replay page's world panel (`render_state`) shows each site's hidden p, its count and whether it scored. `swarmlab report` adds a table after the Final score table with each run's `expected_cum`, `actual_cum` and the per-round W of the three references.

## Fake script

`fake:hidden_chooser` is a free, deterministic plumbing check: agent aNNN takes site NNN mod n_sites (12 agents on 8 sites: 2, 2, 2, 2, 1, 1, 1, 1 over A..H), posts its site in round 1 and tries a second choice that must be rejected (12 rejections, round 1 only).

## Cost

12 agents on `hf:Qwen/Qwen3.8-27B:ovhcloud`, 20 rounds, `memory: window` with `window_rounds: 6`, `max_calls: 8`, `max_tokens: 1024`, reasoning off cost about $5.40 per run (2026-10-08/09; the one-sentence reminder arm $5.86). Most of it is prompt tokens: agents post long tallies of who is where and every later turn reads them. Budget about `soft_usd: 5.50`, `hard_usd: 6.00` per run and set `total_usd` for the experiment; a shorter window or fewer rounds cut it.
