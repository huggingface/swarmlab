# swarmlab: design

Working name `swarmlab` (internal; rename before publishing, candidates checked on 2026-10-05 are in `research/`). Status: design settled 2026-10-06 after a structured review of 79 decisions, including two rounds of independent review; nothing implemented. This is the first of many iterations: fundamentals are fixed here, details are filled in from experience. Background digests in `research/`.

## Purpose

A scientific testbed for finding the primitives that make heterogeneous groups of LLM agents collaborate well or badly. Experimenters vary how agents are coupled (board policies, topology, registry services, hierarchy, interventions) while holding the task and models fixed, and measure swarm-level outcomes (coverage, duplicate work, consensus, idea diversity, cost). Initial scale: hundreds of agents. Reference tasks: the coloring grid (allocation) and the Flag Game (belief dynamics). Single operator; no authentication layer. Separate project from `agent-collabs`, keeping its central principle: every agent action goes through one harness-owned API.

## Core idea

Everything that happens in a run is an **event** in one append-only, sequence-numbered log. The runner advances the world in **rounds**. By default a round is **phase-commit**: every agent's turn runs concurrently against the round-start state, and all writes commit together at round end in seeded order, so a message becomes available to its eligible recipients in the next round, for every topology and every N. Whether a recipient reads it is an experimental outcome. Replay is a fold over the log. Resume is replay-to-commit plus continue. A paired run is two runs from round 0 that differ in one declared way. Interventions and probes are events too.

## Architecture

```
  Spec ──► Runner (Scheduler · Tool Executor) ◄── Manager (ledger · budgets · launch · publish)
             │ │ │
     World   Medium   Participants ──► Providers
             └──┴────────┴──────────────┘
                        ▼
                   Event Log ──► Metrics (within-run folds · experiment-level estimators)
                        │    ──► Observatory (status · replay viewer)
                        └──► HF dataset per experiment (Parquet tables + raw JSONL + per-agent pi-format sessions)
  Interventions & Probes: declared in the spec, fired by round / metric / event pattern
```

### 1. Experiment Spec
YAML referencing Python plugins by entry-point name, with a Python escape hatch for generated arms. Contains arms, seeds, repeats, world and parameters, medium and policies, participants (type, model, role, prompt template, memory policy, budgets), scheduler and commit policy, interventions, probes, metrics, soft budget and hard ceiling. Prompts are conditions, not constants.

**Run identity** is the git commit of the code (with dirty flag) plus the hash of the resolved spec after generation. Everything else the run used is archived as artifacts in the run folder: raw spec, prompt and tool definition files, dependency lock, generated fixtures.

**Randomness**: the run seed derives independent streams for world generation, private-information assignment, scheduler order, each agent's sampling, and scripted agents, so changing one component never shifts another.

### 2. Runner, Scheduler, Tool Executor
Executes one arm for one seed. The runner owns the only path that mutates anything: the **tool executor**. Participants call it for every tool; it validates, applies or buffers, logs, and returns feedback. Participants never touch world, board, or registry directly.

Per round under **phase-commit** (default):

1. `Scheduler.next()` returns the live agents in seeded order (reshuffled each round).
2. Build each agent's **view**: the world observation (always), plus pushed inbox items only if the arm enables push delivery.
3. Run all turns concurrently (bounded by the provider gate). During a turn, reads see the round-start state plus the agent's own buffered writes; writes return an acknowledgement with pending status.
4. Commit: apply buffered world actions, posts, and registry operations in seeded order; conflicts are resolved here (second paint on a cell, second acquire on a key) and the outcome is reported in the agent's next view. Fan posts out to per-agent inboxes per policy.
   Commit semantics are deliberately minimal: within one agent's turn its actions keep their order; across agents, buffers are applied in seeded order; each action succeeds or fails independently; a pending acquire is tentative; a failed action never retracts the posts that followed it. No transactions and no dependency tracking.
5. Fire interventions whose trigger matches; run probes for all agents against committed state.
6. Fold metrics. Write the `round_commit` event. Every k rounds (default 1) write a snapshot.

**Sequential** is the alternative commit policy (`commit: immediate`): agents act one at a time in seeded order with immediate feedback and same-round visibility. Messages are then visible within the same round; propagation is reported in hops computed from read events. The Scheduler interface stays open for async or rate-heterogeneous policies later; nothing depends on wall-clock time.

Modes: **live**, **replay** (no inference; events folded), **resume/fork** (restore the last committed snapshot at or before round r, then live, optionally with an edited spec).

**Termination** on world terminal, max rounds, soft budget, or hard ceiling; the end reason is recorded on the run event. Live mid-run interaction (command queue) is deferred to v2; in v1 interventions are spec-declared or applied by stop-and-fork with an edited spec.

### 3. Event Log, snapshots, recovery
Typed, append-only, sequence-numbered. Families: `run`, `round_commit`, `read`, `decision`, `inference_attempt`, `inference_response`, `post`, `delivery`, `registry`, `act`, `accept`, `reject`, `world`, `overflow`, `intervention`, `probe`, `metric`, `snapshot`.

- Events are authoritative. A round exists only once its `round_commit` is written; on recovery an uncommitted round is discarded and re-executed from the last snapshot. Inference is logged as an attempt before the call and a response after, so lost responses still count as spend. The accepted data-loss window is one round of inference spend.
- Every plugin (world, medium incl. inboxes and registry, scheduler, intervention triggers, participants) implements `snapshot()` and `restore()`. A snapshot is a manifest pointing at a log position plus plugin blobs; checkpoints sync to the bucket at every commit.
- Message content is content-addressed at the message level: a prompt is a list of hashes, so shared prefixes are stored once; snapshots store agent memory as hash lists.
- Local backend: JSONL chunks on local disk with periodic bucket sync. Publish: one Parquet table per event family sharing keys (experiment, arm, seed, run, round, agent), `run.json`, raw JSONL in `raw/`, and one pi-format session file per agent so the Hub's agent-traces viewer renders each conversation. One Hub dataset repo per experiment, private by default. The schema is a public contract.

### 4. World (task plugin)
```
reset(seed) -> state
observe(agent) -> Observation        # text and/or image parts; includes private information
act(agent, action) -> Ack            # agent-visible acknowledgement only
commit(actions) -> outcomes          # applied at round end in seeded order
score() -> dict                      # evaluator-only
terminal() -> bool ; verify() -> hidden ground-truth check
budget(agent) -> remaining
mode: "shared" | "per_agent"
status tools: my_status, collective_status   # read-only, each toggleable per arm, never carry correctness
```
Agent-visible feedback is strictly separated from evaluator-side scoring. Private information is first-class. `calibrate` estimates solo success so experiments sit in the 20 to 60 percent headroom band.

- **ColoringGrid** (shared): cells, colors, optional crossed zones, target changes, worker failures. Validation configuration: 16 agents, plain grid, full visibility, S0 board-only.
- **FlagGame** (per_agent): hidden flag, one random crop per agent with configurable overlap, `guess(candidate)` as a world action that returns an acknowledgement only. The latest committed guess counts; a per-agent guess limit is a parameter (unlimited by default); the game ends on a fixed round count, never on a correct guess. Flags synthetic by default, real as a parameter. Crops text-rendered by default, images as a parameter. Candidates: closed set by default (truth, a rival favoured by some crops, distractors; default 8), open text as a parameter.

### 5. Medium
Two surfaces plus policies, all mutated only through the tool executor at commit.

- **Board**: append-only structured posts (free text plus optional typed fields), channels, topology (broadcast, groups, pairwise gossip, manager-mediated, tree). At commit the medium evaluates each post against each recipient's policy and places it in that recipient's **inbox** with its eligibility round and its transformed content. Reads take eligible items from the agent's own inbox; the `read` event logs the delivered content verbatim. Policy changes apply to posts committed after the change; revocation removes only undelivered items. Delivery is **pull by default**; push-into-view is an arm option using the same inboxes. Fan-out and read load under broadcast are a research object, not something the framework reduces.
- **Registry**: typed, versioned key-value store for coordination state, with five atomic operations resolved at commit in seeded order: `get`, `put`, `compare_and_set` (on version), `acquire(key, ttl_rounds)` (succeeds only if absent or expired), `release`. Expiry in rounds, so dead agents' claims lapse. Whether claims bind the world is an explicit **claim policy** plugin: `advisory` (default; the world ignores the registry and violations are measured) or `enforced` (the world rejects actions on resources the actor does not hold).
- **Policies**: write validators and transforms; read filter `visible(reader, item, round) -> bool | transformed`; delivery mode.

World holds task truth, registry holds coordination state, board holds talk.

### 6. Participants
```
turn(view, tools) -> usage           # runs the agent loop until yield, calling the tool executor
state() / load_state()
```
A **turn** is the agent loop: model call, execute tool calls through the executor, feed results back, repeat until the agent **yields**: a response with no tool call, or an explicit `end_turn` tool; which one is logged. World actions, posts, registry operations, and status reads are tool calls and never end the turn. A large safety cap on model calls per turn (default 20) is a spec parameter. Whether an agent reads the board is its own choice and a measured outcome.

**Context**: a per-agent context limit in tokens with an overflow policy that is an experimental condition (`drop_oldest` default, `summarize`, `fail_turn`), each overflow logged. Size limits on posts, read pages, and tool results, truncation logged.

Types in v1:
- **Scripted**: Python strategies. Dry runs and non-LLM baselines (fixed assignment, work queue, random, a Flag Game evidence aggregator, and a candidate-enumerating oracle probe used in tests).
- **In-process**: our loop. Prompt template (Jinja2), tool schema, memory policy (full history default; k-turn window plus state line as a parameter), per-turn budgets.

Reserved for v2: **harness agents** (Claude Agent SDK, Codex app-server, OpenCode serve; the round prompt is the view; the board CLI is the tool executor over a socket; sessions resume by id; resume from latest round only).

**Roles**: a named bundle of prompt template, tool allowlist, board channel permissions, registry permissions, whether it may act in the world, model, and budget overrides. Hierarchy is a tree topology referencing roles; coordinators post typed assignments and cannot act in the world unless an ablation switch says so. Example library: worker, coordinator, reviewer, skeptic, scribe, plus an example two-level hierarchy spec.

### 7. Providers
One tool schema; native tool calling when supported, constrained JSON text otherwise; the parsed action list is what gets logged. Adapters: HF Inference Providers router, OpenAI, vLLM (chat-completions shape including image parts) and the Anthropic API. Shared: fair admission gate, token counting off the event loop, cost and worst-case cost, served-provider recording, thinking-block handling, and a record/replay cache keyed by (run id, request hash); cross-run sharing only behind a debug flag. The gate enforces the one spending invariant v1 insists on: it reserves the maximum permitted cost of a request before dispatch, releases the difference on response, and refuses dispatch once the hard ceiling is reached (see Manager).

### 8. Interventions and counterfactuals
`trigger` (round, metric condition, event pattern) plus `operation` (inject post, mute or delay or reorder delivery, change topology or visibility, deliver new private information, kill or revive agents, change the target, swap model or prompt, add or remove a coordinator layer). Declared in the spec; each fires as an event.

Two operations exist: **historical replay** (fold, no inference) and **live fork** (restore at round r, continue with fresh sampling, optionally with an edited spec). **Influence** is measured by **paired runs from round 0** that differ in one declared element (e.g. one agent's crop), repeated, with unchanged pairs as the sampling-noise control. A change to private information after round 0 is delivered as new evidence the agent observes; rewriting an agent's memory is not supported in v1 and remains an open question to be shaped by the first experiments.

### 9. Probes
Run once per round after commit, for all agents, against committed state. Out-of-band structured questions; answers are `probe` events, not appended to context by default. A probe measures the agent's belief, so it is **answered by the agent's own model with the agent's permitted context**. Only parsing and coding of free-text answers use a configurable cheap model (default Haiku 4.5). Both are funded from a separate **measurement budget** that counts toward the hard ceiling but never against the swarm's soft budget.

### 10. Metrics
**Within-run folds** over the log, live and offline: coverage, duplicate and wrong work, propagation in hops (from read events) and in rounds, relay events, read rate, rounds-until-first-read, share of turns without board interaction, barrier wait, tokens and dollars per correct unit, overflow counts, claim violations, consensus (share on the modal belief), polarization (beliefs above a threshold share, default 20 percent), diversity (entropy), collapse (consensus above threshold on a wrong belief). Belief metrics use the latest committed guess, count "none" explicitly, treat parse failures as missing, exclude dead agents, and report denominators; computed from world guesses and separately from probe answers.

**Experiment-level estimators** over sets of runs: best@k versus team@k (needs the solo pool), parallel efficiency E(N) (needs the reference N), influence (needs the pairs and controls). N-scaling specs must name their protocol: **fixed evidence per agent** (default for the Flag Game; total evidence grows with N) or **fixed total evidence**.

### 11. Observatory
Live status (CLI JSON and a small page). Static replay viewer loading any run from local disk or a published dataset by URL: world, per-agent action queue, each agent's inbox as delivered, verbatim turn and raw inference, arm-by-seed matrix. Publishable to a static Space.

### 12. Manager
Ledger with reserve-then-result rows. Per run: a **soft budget** (stop cleanly at the next round boundary, resumable) and a **hard ceiling** (the provider gate refuses new calls; the round is aborted and discarded; the run stops resumable from the last commit). Spend counts from attempt events and includes retries, probes, metering, and estimated job compute. The experiment cap in the ledger is the sum of hard ceilings; launches over it are refused. Idempotent launch from the ledger; HF Jobs default with a local option; one job hosts a wave of runs and vLLM when self-hosting; readiness report split into infrastructure versus model compliance; publish-to-Hub.

### 13. CLI and skill
```
swarmlab validate spec.yaml      swarmlab run spec.yaml --arm A --seed 3
swarmlab dryrun spec.yaml        swarmlab resume RUN [--at 40]
swarmlab smoke spec.yaml         swarmlab fork RUN --at 40 --spec edited.yaml
swarmlab status RUN              swarmlab metrics RUN|EXPERIMENT [--add my_metric]
swarmlab view RUN                swarmlab publish EXPERIMENT
```
All commands take `--json`. A skill and `AGENTS.md` ship in the repo and install into Claude Code, Codex, and OpenCode skill directories, encoding the workflow: spec, dry run with scripted agents, smoke at N=3, second-agent review of readiness and of the exact prompts each arm sees, budget reservation, launch, status, analyze, view, publish, resume or fork instead of rerun.

## Integrity
Agent identity is assigned by the harness. Hidden data lives outside the agents' reach. No agent-visible field carries correctness. Reward hacking is measured, not blocked, unless the spec says otherwise.

## Engineering decisions
Python 3.12, uv, minimal core (pydantic, httpx, pyarrow, Jinja2) with extras for the Anthropic SDK, `datasets` and `huggingface_hub`, and Playwright for viewer tests. Personal GitHub repo, Apache-2.0, clone on `$AM_LOCAL` with bundle backups to the workspace. Deterministic tests use scripted agents and a fake provider.

## Build plan
Process: orchestrator writes the interface contract; Opus subagents implement work packages against it; Sonnet handles viewer and docs; a fresh Claude session reviews before the first run that spends on models.

- **M1a** First working slice: spec, phase-commit runner with tool executor, scripted participants, fake provider, event log with commit markers and per-round snapshots, board with inboxes, FlagGame text variant, replay, minimal viewer. Acceptance: (1) two runs of the same seed produce identical committed events; (2) killing the runner mid-round and recovering yields the same committed logical events and reconstructed state as an uninterrupted run (operational events such as attempts may differ; incurred spend is retained); (3) a fork at round r reproduces the parent's state at r exactly before diverging.
- **M1b** First model-spending slice: in-process participant, one real provider, probes answered by the agent's model, two metrics, a broadcast-versus-gossip comparison at small N. Acceptance: (4) a post held back for a reader is delivered when eligible and never lost; (5) the candidate-enumerating oracle probe scores at chance and no agent-visible field carries unauthorized correctness feedback (legitimate private evidence is allowed to help).
- **M2** Flag Game proving experiment: N=16 and N=64, broadcast versus gossip, phase-commit versus sequential as arms, Haiku 4.5 and a Qwen via the router, probes every round, influence via paired runs from round 0 with unchanged-pair controls, scripted evidence aggregator as baseline.
- **M3** Coloring S0 at N=16 with full visibility, as framework validation only. The registry and claim policy are built here, where the coloring task first needs them; remaining providers and richer medium policies arrive when an experiment calls for them.
- **M4** Parquet and pi-format export, Hub publish, static viewer polish, skill and `AGENTS.md`.

## Deferred (v2) and open
Harness-agent participants. Live mid-run command queue. Async scheduler. Thousands of agents (broadcast load is a research object, not a framework problem). Code-execution sandboxes for tasks. Memory rewrite for mid-run counterfactuals. Package rename for publication.
