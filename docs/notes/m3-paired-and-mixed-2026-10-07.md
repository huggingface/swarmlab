# M3a real tests: paired Haiku run and mixed Haiku+Qwen swarm (2026-10-07)

Branch `paired-runs`. Run dirs: `/data/workspaces/ultra-harness/runs/m3-tests/` (copied from
`$AM_LOCAL/runs/m3-tests/`). Each has `report.json` written by its tool. Total spend: **$3.32**.

| run | swarm | measurement | total |
|---|---|---|---|
| paired base (`paired-haiku__base__s2__base_r0`) | 0.582 | 0.242 | 0.824 |
| paired patched (`..._patched_r0`) | 0.567 | 0.212 | 0.779 |
| paired control (`..._control_r0`) | 0.585 | 0.212 | 0.798 |
| mixed swarm, timeout 90 s, aborted in round 2 (`mixed-swarm__s1__timeout90-aborted`) | 0.068 | 0.015 | 0.083 |
| mixed swarm, timeout 60 s (`mixed-swarm__s1`) | 0.618 | 0.223 | 0.841 |

## a. Paired Haiku run (`tools/paired_haiku.py`)

Setup: FlagGame N=8, 6 rounds, broadcast, `LLMAgent(anthropic:claude-haiku-4-5, max_tokens=1024)`,
belief probe every round, `Budget(1.2 / 1.5 / 0.4)` per run. `Experiment.pair(seed, 6, patch=...,
repeats=1, control=True)`.

**Seed.** Seed 1 was skipped: a003's own crop there is already contained in five candidates,
the rival among them, so moving it to a rival-contained window would not change what kind of
evidence it has. The tool rule: a003's own crop must not contain the rival, and the new window
must be contained in the rival. Seed 2 qualifies. Truth H, rival G. a003's crop `[5, 2]`
(`kkbb` x3) is contained only in H. It moved to `[0, 1]` (`rrry` x3), which is contained in
exactly {G, H}. In this world 4 of 8 crops are unique to the truth.

**Trajectories.** Values are rounds 1-6. World guesses; probe values are close and are in `report.json`.

| | accuracy | consensus |
|---|---|---|
| base | .50 .38 .63 .75 .75 .75 | .50 .63 .63 .75 .75 .75 |
| patched | .50 .75 1 1 1 1 | .50 .75 1 1 1 1 |
| control | .50 .75 1 1 1 1 | .50 .75 1 1 1 1 |

**a003's guesses (held per round).**
- base: H H H H H H
- control: H H H H H H
- patched: **G** H H H H H

The probe answers match the guesses in every run. With the ambiguous crop, a003 picked the rival
in round 1. In round 2 it read two posts, from a000 and a001, which carried truth-only evidence.
It switched to H and stayed there.

**Effect.** For `belief.consensus` at the last round: `diff = +0.25` and `control_spread = 0.25`.
Accuracy and both probe metrics give the same numbers. The patched-minus-base difference equals
the control-minus-base difference. **No effect can be told apart from sampling noise with one
pair.** The base run happened to be the outlier: a006 held G throughout, and a005 drifted to C.

**Did the patched crop change anyone else's belief?**
- a003 posted nothing in any run, so no reads of its posts exist (`reads_of_target_posts = []`
  in all three runs). The only channel left was `collective_status` guess counts: a003's G in
  round 1 was visible in round 2.
- Other agents whose held-guess trajectory differs from base: 5 in patched (a000, a004, a005,
  a006, a007) and 4 in control (a000, a005, a006, a007). Nothing points to the patch.
- Base and control already differ in round 1 (a000 G vs H, a005 H vs G), before any agent could
  see another. Haiku's turn-level sampling noise is large at N=8.

**Takeaways.**
- The pairing machinery works end to end on a real model: same world and schedule, the override
  applied to one agent, the control run, and `effect()`.
- Detecting a one-crop effect at N=8 needs many repeats. A single control diff of 0.25 means one
  pair cannot resolve effects below roughly 0.25.
- The patched agent must post for the patch to travel through the board. Here it did not, so the
  test measured a003's own belief only.

## b. Mixed swarm (`tools/mixed_swarm.py`)

Setup: one run of 16 agents on FlagGame seed 1, 6 rounds, broadcast board, belief probe every
round, `Budget(1.5 / 2.0 / 0.5)`.
- a000-a007: Haiku 4.5, `max_tokens=1024`.
- a008-a015: `hf:Qwen/Qwen3.5-9B:deepinfra`, `max_tokens=2048`, `enable_thinking=False`.

**DeepInfra tail, and the switch to 60 s.** The first attempt used the default `timeout_s=90`.
Round 1 finished, but 4 of the 8 Qwen turns ended `error`: `ProviderError` after 3 attempts,
272 s each. Other Qwen calls succeeded at 79-109 s. The 186 s and 272 s successes came on the
third attempt. I aborted it in round 2 (kept as `mixed-swarm__s1__timeout90-aborted`, $0.08) and
reran with `providers={"hf": OpenAICompatProvider("hf", ..., timeout_s=60)}`, CLI flag
`--timeout 60`. The results below are from the 60 s run.

| | Haiku (a000-a007) | Qwen (a008-a015) |
|---|---|---|
| turns | 48 | 48 |
| turn yield kinds | end_turn 48 | **error 32**, end_turn 15, no_tool 1 |
| read rate (turns with read_board) | 0.854 | 0.021 (1 turn) |
| posts | 3 | 1 |
| final accuracy (held guess = truth G) | 8/8 = 1.0 | 4/8 = 0.5 (2 hold D, 2 never guessed) |
| probe parse | 48/48 | 47/48 |
| latency median / p90 / max (successful responses) | 2.7 / 5.3 / 9.5 s | 2.4 / 63.6 / 178 s |
| responses with attempts > 1 | 0 of 147 | 41 of 114 (attempts 1:73, 2:5, 3:36) |
| responses that failed (retries exhausted) | 0 | 32, all `hf: timeout after 60 s ... gave up after 3 attempt(s)` |
| cost | $0.817 | $0.025 |

Swarm-level results:
- World accuracy by round: .44 .44 .50 .69 .75 .75.
- Probe accuracy by round: .69 .69 .81 .81 .88 .88.
- Final score 0.75, 14 of 16 agents guessed.

**Cross-model reads happened in both directions.** Counts are reads that returned a post by the
other group:
- Haiku reading Qwen: 7. The only Qwen post, a015's in round 5 ("My crop matches candidate G at
  rows 5-7, cols 1-4."), was read by 7 Haiku agents in round 6.
- Qwen reading Haiku: 3. a010 read 3 Haiku posts in round 4.
- Haiku reading Haiku: 21.

**Two providers in one run** worked. The gate, ledger, budget and per-prefix providers handled
both providers; Haiku had no retries.

**Timeouts against DeepInfra's tail.**
- The timeout bounds the damage: a stalled Qwen turn costs at most 3 x 60 s plus backoff. The
  run finished in about 30 min instead of hanging.
- At 60 s the tail is too heavy for this model and route: 32 of 48 Qwen turns (67%) failed,
  concentrated in rounds 1-3 (8, 8, 7 failures; then 4, 3, 2).
- Stalls hit requests with tools. Successful retried responses had as few as 16 completion
  tokens after a 63 s first attempt, so this looks like server-side queueing or stalls, not long
  generations.
- Probe calls (no tools) almost never stalled: 47 of 48 parsed.
- With a 90 s timeout some of these calls succeed (79-109 s), at about 4.5 min per failed turn.

Follow-ups:
- For Qwen on the HF router, either pick a different provider route or lower concurrency for
  `hf`. The provider default is 8 in flight.
- A failed turn ends that agent's round, so Qwen agents effectively skipped most of rounds 1-3.
  That is why the Qwen read rate is 0.021.
- A `ProviderError` inside a probe call is not caught by `Runner._probe_one`, which catches only
  the budget exceptions. It would abort the round and leave the run resumable. It did not happen
  here, but at this failure rate it could. Not fixed on this branch: the probe path is outside
  WP9's files.
