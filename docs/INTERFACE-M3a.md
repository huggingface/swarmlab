# swarmlab M3a interface contract: interventions and context limits

Scope: DESIGN.md component 8 (Interventions) and the context limit from component 6 (Participants). Builds on INTERFACE.md and INTERFACE-M1b.md, which stay binding. Nothing here changes the spec hash of a run that declares no interventions and no context limit.

## 1. Interventions

An intervention is a plugin with a **trigger** and an **operation**. Triggers are evaluated by the runner once per round, after the world and board commit and before probes, in spec order. Every operation writes one `intervention` event per affected agent (or one event with `affected=[]` for swarm-wide operations), and all state an intervention keeps is in its snapshot, so interventions replay and survive resume.

```python
class Intervention(Persistable, Plugin):
    name: str
    # trigger, chosen by constructor kwargs (exactly one of):
    #   at_round: int | list[int]         fire at these rounds
    #   every: int (+ start: int = 1)      fire every k rounds from start
    #   when: {"metric": str, "op": ">"|">="|"<"|"<=", "value": float, "once": bool = True}
    #   on_event: {"type": str, "where": dict, "once": bool = True}   (matches logical events of the round)
    def should_fire(self, ctx: "InterventionContext") -> bool    # provided by the base class from the trigger kwargs
    def apply(self, ctx: "InterventionContext") -> None           # calls ctx.ops.*; required

class InterventionContext:
    round: int; live: list[AgentId]; agents: list[AgentId]
    metrics: dict[str, tuple[float | None, int]]   # this round's fold values
    events: list[Event]                            # this round's logical events so far
    rng: random.Random                             # derive(seed, "intervention", name, round)
    ops: Ops
```

`Ops` is the only mutation surface, owned by the runner:

| op | effect | event payload |
|---|---|---|
| `inject_post(text, *, author="system", channel="main", fields={}, recipients=None)` | commits a post now; deliveries become eligible next round per policy; `recipients` restricts fan-out | `op="inject_post", post_id` |
| `delay(agents, rounds)` | adds `rounds` to the eligible round of deliveries to these agents, for posts committed from now until cleared | `op="delay"` |
| `mute(agents, rounds)` | these agents' posts are withheld from everyone for `rounds` rounds (posts still logged, no deliveries) | `op="mute"` |
| `set_policies(policies)` / `set_topology(topology)` | replaces the board's policies or topology from the next commit | `op="set_policies"` |
| `kill(agents)` / `revive(agents)` | removes agents from or returns them to the live set from the next round; killed agents take no turns and are excluded from metric denominators | `op="kill"` |
| `patch_private(agent, data)` | calls `World.patch_private(agent, data)`; the world must deliver the change as new evidence in the agent's next observation (never a memory rewrite) | `op="patch_private"` |
| `world(name, **args)` | calls `World.intervene(name, **args)` for world-specific changes (e.g. FlagGame `set_truth`) | `op="world"` |
| `reconfigure(agent, **kw)` | calls `Participant.reconfigure(**kw)` (LLMAgent supports `model`, `system_prompt`, `memory`, `window_rounds`, `max_tokens`) from the next turn | `op="reconfigure"` |

World and participant hooks (defaults raise `NotSupported`, which the runner logs as `intervention` with `ok=False`):
```python
World.patch_private(self, agent, data: dict) -> None
World.intervene(self, name: str, **args) -> dict
Participant.reconfigure(self, **kw) -> None
```
FlagGame implements `patch_private(agent, {"crop": [y, x]})` (replaces the crop position; the next observation includes a line `Your crop has changed.`) and `intervene("set_truth", name=...)`.

Built-in interventions (entry-point group `swarmlab.interventions`): `inject_post`, `delay_delivery`, `mute`, `kill_agents`, `patch_private`, `reconfigure`, each taking the trigger kwargs plus its operation's arguments. Spec: `interventions: [{type, params}]` on an arm or experiment.

Determinism: interventions see only committed state and this round's logical events; their rng is derived per (name, round). Two runs with the same seed and spec fire identically.

Snapshot: `plugins["intervention:<name>"]` holds fired-once flags and any counters; the board's active delays and mutes are part of the board snapshot; the live set is already in the manifest.

## 2. Paired runs from round 0

`Experiment.pair(seed, max_rounds, *, patch: Experiment | None, repeats: int = 1, out=...) -> PairedResult` runs the base experiment and a patched variant from round 0 with the same seed (identical world generation and schedule), `repeats` times each with the per-component sampling streams re-derived per repeat (`derive(seed, "repeat", i)` mixed into the agent streams), plus `repeats` unchanged pairs as the noise control when `control=True`. `PairedResult` holds the runs and `effect(metric, round=-1)` returning the mean difference and the control spread. A crop swap is expressed as a FlagGame constructor param `crop_overrides: {agent: [y, x]}`; it is applied after `reset` so everything else stays identical.

`Run.fork(at_round=0)` is also valid and must reproduce the parent's round-0 state exactly (the snapshot at round 0 is written at reset; if a run has no round-0 snapshot the fork re-resets the world with the parent's seed and the possibly edited experiment params).

## 3. Context limit

`LLMAgent(context_limit_tokens: int | None = None, overflow: "drop_oldest" | "summarize" | "fail_turn" = "drop_oldest")`.
Before each request the agent estimates prompt tokens with the provider's estimator. If over the limit: `drop_oldest` removes whole earlier rounds until under the limit (the system prompt and the current round are never dropped); `summarize` replaces the dropped rounds with one assistant note produced by the cheap coder model under the measurement budget; `fail_turn` ends the turn with `yield_kind="error"` and `error="context_limit"`. Every overflow writes an `overflow` event `{agent, policy, dropped_rounds, tokens_before, tokens_after}` (logical). Truncation is an experimental condition: the limit and policy are participant params and therefore in the spec hash.

## 4. Acceptance

1. An `inject_post` at round 3 appears in every agent's inbox at round 4 under broadcast; under a `delay_delivery(agents=[a000], rounds=2)` from round 3 it reaches a000 at round 6. Deterministic across two runs; replay reproduces the `intervention` events without calling plugins.
2. `kill_agents` at round 5 removes agents from turns and denominators; `revive` returns them.
3. `patch_private` on FlagGame changes one agent's crop at round 4; its next observation says so; `verify()` reflects it.
4. A `when: {metric: "belief.consensus", op: ">=", value: 0.9}` trigger fires once in the first round the condition holds, in a fake-provider run.
5. `Experiment.pair` on fake LLM agents: control pairs have identical logical views; the patched pair differs only from the patched agent's first turn onward.
6. Context limit: with `context_limit_tokens` small, `drop_oldest` logs overflows and keeps the run going; `fail_turn` ends turns with the documented error; snapshots and resume preserve dropped history correctly (the dropped messages are gone from the participant's memory, by design).
