# swarmlab M6 interface contract: faithful Flag Game replication (arXiv 2609.19124)

Scope: run the paper's three protocols as specified in `docs/flag-game-paper-setup.md` on open VLMs through the HF router. Builds on INTERFACE.md, M1b, M3a (interventions), M3c (roles, Tree/Star), M5 (image modality) and the manager-protocol branch (blind agents, Star topology, manager role). Nothing here changes the spec hash of existing specs.

## 1. World: real flags and name-only candidates

`FlagGame(flags: Literal["synthetic", "real"] = "synthetic", candidates: Literal["grids", "names"] = "grids", canvas: tuple[int, int] = (12, 8), crop: tuple[int, int] = (4, 3))`.

- `flags="real"`: the hidden flag is drawn from a built-in set of 28 stripe-and-triangle national flags defined as colour layouts in `swarmlab/world/flags_real.py` (horizontal and vertical stripes with proportions, a triangle at the hoist where applicable, canton-free; each entry: country name, layout spec, colours as RGB). Rendered onto the `canvas` grid (the paper uses 24×16) by the existing renderer; triangles are rasterised cell by cell (a cell belongs to the triangle if its centre does). The candidate set is the full list of 28 country names. `n_candidates`, `rival_edits` and `candidate_names` are ignored in this mode (documented). `verify()` reports the country name and the layout id.
- `candidates="names"`: the observation carries no candidate images or grids, only the list of allowed names as text (the paper's "Allowed countries" JSON list) and the agent's crop (image part under `modality="image"`, text grid otherwise). `guess(candidate)` accepts a country name (case-insensitive, exact match after normalisation).
- Crop sampling: a `crop`-sized window at a uniformly random position per agent (overlap allowed), from the agent's private stream as today. `cell_px` default 25 when `flags="real"` so a 6×4 crop renders at 150×100 px.
- The task description for `names` mode says: "You are one player in a flag identification game. All players are identifying the same underlying flag. You see a crop of it. Allowed countries: [...]." with no instruction to cooperate or be truthful (the paper's framing).

## 2. Scheduler: one speaker per step

`OneSpeaker(listeners: int = 1)` scheduler (`scheduler.py`, entry point `one_speaker`): each round exactly one live agent acts (seeded from `("schedule", round)`), i.e. a round is one of the paper's asynchronous steps. Used with `commit: immediate` and `topology: gossip(k=1)`, whose partner draw for that round gives the single listener. `options.max_rounds = kappa * N` is written by the spec author; the spec gains `options.rounds_per_agent: int | None` as sugar (`max_rounds = rounds_per_agent * N` when set). Probes under this scheduler run every `probes_every` rounds (existing `every`); the spec sets `every: N`.

## 3. Memory: last H received messages

`LLMAgent(memory="received", memory_messages: int = 8)`: the context is the system prompt, the current observation (crop re-shown every call, as in the paper), and the last `memory_messages` delivered board items (oldest to newest) as a single user-visible block, plus the agent's own previous guesses as a short list. No tool-call history is kept across turns. Works with push delivery (`delivery: push`, `push_limit: memory_messages`) so the agent need not call `read_board`; pull still works. `probe_context()` returns the same construction.

## 4. Message format

A built-in participant prompt variant `report_json=True` on `LLMAgent`: the agent is asked to answer each turn with JSON `{"country": ..., "reason": "<one sentence>"}` (the paper's m=3 format; `report_fields=["country"]` gives m=1). The harness turns that JSON into a `guess(country)` action and a `post(text=json)` in one step (`json` tool protocol path, so it also works on models without native tools), and the round ends. Native-tool models may still call tools directly; `report_json` is a condition, not the default.

## 5. Metrics and stopping

- `belief.state` metric (paper classification): per round, `s1` = share of the top country among live non-blind agents with a guess; `correct_consensus` if `s1 >= 0.85` on the truth, `wrong_consensus` if on another country, `polarized` if `s1 < 0.85` and at least two countries each `>= 0.25`, else `fragmented`. Params `consensus=0.85, camp=0.25`. Emits the label as a string value and a one-hot set of four numeric metrics for plotting. Also from probes (`source="probe:belief"`).
- `options.stop_when: {"metric": "belief.consensus", "op": ">=", "value": 1.0, "consecutive": 5}`: the run ends with reason `stop_condition` when the condition holds for `consecutive` evaluations in a row (evaluated once per probe round when probes are on, else per round).
- Experiment-level: `swarmlab report` gains, for Flag Game runs, the paper's terminal-state shares (correct / wrong / polarized / fragmented) per arm across seeds, and terminal truth mass.

## 6. Spec and estimate

`experiments/m6_flag_paper.yaml`: `flags: real, candidates: names, modality: image, canvas: [24, 16], crop: [6, 4]`, Gemma 4 26B-A4B on DeepInfra (`temperature: 0.2`, `max_tokens: 200/250`), arms `pairwise-{4,16,128}` (OneSpeaker, gossip k=1, immediate, received memory H=8, probes every N rounds, rounds_per_agent 10, stop_when 5 consecutive full-consensus probes), `broadcast-{4,16,128}` (round_end, broadcast, memory: window keeping own decisions, 10 rounds, probe every round) and `manager-{4,16,128}` (Star with blind a000 as manager role, 10 rounds). Seeds `[1]`, `budget.total_usd: 10`, per-arm hard caps from the cost note (`docs/flag-game-paper-setup.md` section 8 of the design discussion: N=128 broadcast ≈ $1.5). `swarmlab estimate` must handle `OneSpeaker` (calls per round = 1 agent).

## 7. Acceptance

1. All 28 real flags render; each rendered flag is distinct; a crop at every position is contained in its flag; `names` mode observation has no grids and lists 28 names.
2. `OneSpeaker` + gossip(k=1) + immediate: exactly one `turn_started` per round, one delivery per post, deterministic across runs, replay/resume OK.
3. `memory="received"` keeps exactly the last H items in order and re-shows the crop; snapshot/resume preserves it.
4. `report_json` turns a JSON answer into a guess plus a post on the fake provider.
5. `belief.state` classification matches hand-computed labels on constructed distributions; `stop_when` ends a fake-provider run with `stop_condition` after 5 consecutive hits.
6. The spec validates, all nine arms build, `swarmlab prompts` renders `pairwise-4` and `manager-4`, `estimate` prices the grid.
