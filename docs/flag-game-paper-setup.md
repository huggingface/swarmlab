# Flag Game (Pavlova & Tanaka, arXiv 2609.19124v1, 16 Sep 2026): exact setup
Source: full text of https://arxiv.org/html/2609.19124 (fetched with curl, read directly; PDF via WebFetch was unusable). Figures are images, so per-N accuracy values are NOT in the text. Demo: https://flag-game-demo.vercel.app

## 1. Agents and population
- Models: main runs GPT-4o and GPT-5.4; "smaller validation runs" Claude Haiku 4.5 and Claude Sonnet 4.6. About $25,000 total API cost.
- N: "4 to 128". Table 3 lists N = 4, 8, 16, 32, 64, 128. Protocol/composition/patching studies use N=8; pairwise alpha sweep N=16 (GPT-4o).
- Seeds (Table 2): population scaling 40 seeds per (model, N), pairwise, m=3, no social-awareness prompt. Protocol side-by-side 60 matched seeds per condition (N=8, m=3). Pairwise social-awareness: GPT-4o, N=16, m=3, 28 seeds per alpha. Broadcast social-awareness/composition: N=8, m=3, 30 seeds per (alpha, composition). Memory-conflict probe: 36 trials. (Table 3 percentages are multiples of 1/38 or 1/37, which hints at 38 or 37 usable GPT-4o trials per N; this is my inference, not stated.)
- Rounds: Tmax = kappa*N ("some constant kappa"; kappa value NOT stated). Plots use t/N. Early stop: "if five consecutive probes result in a full country consensus of 100%". "Periodic probes" in pairwise; probe period not stated. Patching comparisons: accuracy "after ten rounds".
- Decoding: temperature 0.2, top-p 1.0. Completion caps: 200 tokens pairwise (population and social-awareness sweeps), 250 tokens broadcast and manager.

## 2. Task
- Each trial: hidden country sampled uniformly from "the country pool of 28 stripe and triangle flags" (real countries; the list of 28 is not given in the text. Examples named: Germany, France, Peru, Yemen, Austria).
- Rendering: "24x16 canvas", crops "6x4" at "render scale 25". Crop image attached with "high image detail" in a multimodal user message. Whole-flag crop is 6x4 of 24x16 = 1/16 of area (my arithmetic). The text says "scale 25" and the abstract-level fetch said "25%"; the exact meaning of scale 25 is ambiguous.
- Crop assignment: "c_i = R_i x, where R_i is the crop randomly assigned to that agent", ranging "from highly ambiguous to strongly diagnostic". Position sampling distribution, overlap and per-agent independence: not stated beyond "randomly assigned".
- Candidates: every prompt lists "Allowed countries: <JSON list>" as text names (no images). The size of the list is not stated separately (the pool is 28; whether the list equals the full 28 is not stated). Exactly one must be chosen.
- Answer format: JSON only. m=1: {"country": ...}. m=3: {"country": "<allowed>", "reason": "<one sentence>"}.

## 3. Protocols
- Pairwise (gossip), asynchronous. "At each interaction t, one speaker and one listener are sampled" (theory section: independently and uniformly). Speaker sees own crop + transcript memory and emits a message (country for m=1; country + reason for m=3). Listener appends it to memory (max H=8). One interaction per step; N interactions = one "round" (t/N). Periodic probes query every agent with private crop + memory; endpoint = empirical distribution of terminal answers. Whether sampling excludes speaker=listener and probe frequency: not stated. Prompt shows "Transcript memory (oldest -> newest)" as a list of speaker messages.
- Broadcast, synchronous. Each round every agent reports a country from its crop and "private memory of its own past final decisions", then "sees the current reports of the other agents". Everyone shares the same public set of current-round reports. Order within round beyond that is not stated. Endpoint = distribution over all agents.
- Manager: N observers (with crops) give country-reason reports; one blind manager sees the reports and "its own prior decisions", never a crop. It emits a country decision each round, which "becomes shared memory for the observers". Endpoint = manager answer (correct iff equals truth). Whether observers see one another's reports directly: not stated (text implies they see only the manager decisions). Manager prompt text: not given.

## 4. Memory
- Pairwise: buffer of at most H=8 prior entries per agent, i.e. received speaker messages (label, or label+reason), oldest -> newest; presumably oldest dropped on overflow (eviction rule not stated explicitly). Own messages are not said to be stored.
- Broadcast: private memory of the agent's own past final decisions (length not stated; "H=8 prior entries per agent" is stated as general). Manager: manager's prior decisions; observers get manager decisions as shared memory.
- Private crop: attached to the same user message every call (all protocol prompts "attach the private crop image to the same user message (unless the agent is a manager)"), so it is re-shown every turn.

## 5. Prompts (Appendix E, Fig. 11 pairwise example)
System: "You must output only valid JSON. No extra keys, no markdown, and no text outside the JSON object. You are one player in a flag identification game. Choose exactly one country from the allowed countries listed in the user message. Follow the exact output schema given in the user message."
User: "All players are identifying the same underlying flag. You always see the same private crop. Transcript memory shows messages you observed from previous interactions with other players. Allowed countries: <JSON list> / Transcript memory (oldest -> newest): <speaker messages, or []> / <optional social-awareness line> / Output JSON exactly: {"country":"<one allowed country>","reason":"<one sentence>"}" plus the crop image.
Probe queries use the same text, differing only in the schema line. No instruction to be truthful or to cooperate is given beyond "all players are identifying the same underlying flag". Broadcast and manager prompts are not reproduced (only that they use JSON-only).
Social-awareness ladder (Table 4, pairwise wording / broadcast wording), by alpha:
- <=.2: "Rely mostly on your own crop and treat transcript memory as weak evidence." / "Rely mostly on your own evidence; treat other agents' country guesses as weak evidence."
- .2-.4: "Give somewhat more weight to your own crop than to transcript memory."
- .4-.6: "Balance your own crop and transcript memory." / "...using their guesses as real evidence."
- .6-.8: "Give somewhat more weight to transcript memory than to your own crop."
- >.8: "Treat transcript memory as strong evidence and update readily toward it." / "Treat other agents' country guesses as strong evidence and update readily toward them."
Default sweeps use no social-awareness line.

## 6. Measurement
- p_final = distribution over agents' final guesses; s1 = max_y p_final(y), y1 = argmax.
- Correct consensus: s1>=0.85, y1=truth. Wrong consensus (= "collective belief collapse"): s1>=0.85, y1!=truth. Polarization: s1<0.85 and at least two countries with mass >=0.25 (truth-rival polarization when camps are truth and a plausible rival). Fragmentation: otherwise. Robustness over consensus 0.75-1.00 and polarization 0.15-0.35 in 0.05 steps (30 combos, Table 3).
- Collective mean accuracy: A_coll = E[p_final(y*)] (terminal truth mass) for pairwise/broadcast; A_mgr = E[1{y_mgr = y*}] for manager. A_init = mean initial isolated accuracy of crop-bearers; A_maj = majority vote of isolated initial guesses; social uplift = A_coll (or A_mgr) - A_init.
- Social circuit attribution: pick an "informative crop" of Germany's flag that elicited Germany "10/10" in ten isolated single-agent probes (no social input). Score S_i = dp_i * E_i, dp_i = (informative-crop accuracy over ten probes) - (agent's original-crop accuracy over ten probes); E_i = temporal closeness = (1/(N-1)) sum_j N/tau_ij(0), tau = earliest message index at which info from i reaches j via time-ordered paths (Pan & Saramaki 2011), unreached contributes 0.
- Patching: replace one agent's crop with the informative crop, hold the communication schedule fixed, measure change in collective mean accuracy. Table 5 (N=8): A4 S=1.81, delta accuracy 0.75 (others 0.00-0.63); baseline 25%; "one game per patch". Paired traces: 25% -> 100% with A4 patched. Repeat "across ten runs", accuracy after ten rounds; for N=8-128 patch 1/8 of agents (one agent at N=4). Mean improvement falls from 40% at N=8 to about 17% at N=128. Whether the schedule is replayed per run with random seeds: not stated beyond "holding the communication schedule fixed".
- Theory side-probe: each initial crop probed 50 times restricted to truth T vs rival R to estimate rival evidence share.
- Memory-conflict probe: one target-country crop + synthetic 8-entry memory with k lure entries and 8-k target entries, shuffled; sweeps 8:0 to 0:8; weak vs strong private-evidence regimes; m in {1,3}.

## 7. Results
- Pairwise, all GPT-4o: collective accuracy above initial, "peaks at the intermediate population size, N=16", declining at larger N; the decline is from polarization, not wrong consensus (wrong consensus falls with N). Mean-field theory (averaged over measured shares) peaks at N=32. Per-N accuracy table: not in text (figure only).
- Table 3 (GPT-4o pairwise, ranges over threshold grid, correct / wrong / polarized / fragmented %): N=4: 50.0-52.6 / 26.3-34.2 / 13.2-23.7 / 0-10.5. N=8: 47.4 / 5.3-23.7 / 26.3-44.7 / 0-21.1. N=16: 50.0-52.6 / 0-10.5 / 18.4-44.7 / 0-31.6. N=32: 42.1-47.4 / 0 / 31.6-55.3 / 0-26.3. N=64: 36.8-42.1 / 0-5.3 / 34.2-55.3 / 0-28.9. N=128: 37.8-40.5 / 0 / 56.8-59.5 / 0-5.4.
- Broadcast social-awareness ladder: terminal truth mass 0.54 -> 0.81. Pairwise alpha sweep has an interior optimum at 0.75.
- Mixed models: broadcast N=8, m=3, GPT-5.4 count 0..8 rest GPT-4o; best teams are mixed (numbers in figure only). Mixed N=8 = 4 GPT-5.4 + 4 GPT-4o. Single-agent: similar crop accuracy, GPT-4o errors more crop-compatible, GPT-5.4 more incompatible (100 isolated runs).
- Protocol sweep N=8: GPT-5.4 manager 0.57 vs GPT-4o manager about 0.48; all-GPT-4o beats all-GPT-5.4 under pairwise, order flipped under broadcast.
- Haiku 4.5: under strong private evidence (crop uniquely identifies target) GPT-4o, GPT-5.4 and Sonnet 4.6 hold the target; Haiku "abandons the private target as social memory accumulates, a signature of sycophantic override". GPT-5.4 routes mass to other crop-compatible countries (compatibility reasoning); at m=3 Haiku/Sonnet also start giving compatible alternatives.

## 8. Differences from your blog-post summary
- Populations 4-128, three protocols, real flags, GPT-4o/GPT-5.4/Haiku 4.5: confirmed. Also used: Claude Sonnet 4.6 (validation, memory probe).
- Flags are a pool of 28 "stripe and triangle" flags, not arbitrary flags; tiny 24x16 canvas, 6x4 crops (1/16 area), not photo-realistic.
- Haiku is used only in the single-agent memory-conflict probe (and validation), not in full swarms as far as stated.
- "Gossip" is called "pairwise"; asynchronous random speaker/listener.
- Broadcast agents keep private memory of their own past decisions, not an 8-interaction social buffer; the 8-entry buffer (H=8) is the pairwise one (the appendix states H=8 generally).
- Memory is of received messages (m=1 labels or m=3 label+reason), not "social interactions" in a richer sense.
- Extra axes not in the blog summary: message bandwidth m in {1,3}, social-awareness alpha prompt, manager is a blind synthesizer, early stop at five consecutive 100% consensus probes, Tmax=kappa*N.
- The influence procedure is a single designated informative Germany crop patched into one agent; the "10/10" criterion applies to that crop.
