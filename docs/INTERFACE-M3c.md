# swarmlab M3c interface contract: roles and hierarchy

Scope: DESIGN.md component 6 (Roles) and the tree topology for hierarchy. Builds on INTERFACE.md (§6 executor allowlist, §9 medium channels), INTERFACE-M3a (interventions may `reconfigure` roles' settings later; not required now), and the registry from M3b. Nothing here changes the spec hash of a run that declares no roles.

## 1. Role

```python
class Role(BaseModel):
    name: str
    prompt_append: str | None = None          # appended after the task section of the system prompt
    system_prompt: str | None = None          # full override (rarely)
    tools: list[str] | None = None            # allowlist of tool names; None = all tools the medium/world expose
    channels_read: list[str] | None = None    # board channels this role may read; None = all
    channels_write: list[str] | None = None   # board channels this role may post to; None = all
    registry: Literal["none", "read", "write"] = "write"
    may_act: bool = True                      # may call world actions
    model: str | None = None                  # overrides the participant's model
    budget: dict | None = None                # per-turn overrides: max_calls, max_tokens
    post_fields: dict | None = None           # typed fields this role may attach to posts (e.g. {"kind": ["assignment","report"]})
```

Roles are declared on the experiment (`roles: {name: Role}`) and assigned per participant group (`ParticipantGroup.role`, already a string field). Enforcement is in the executor: a call outside `tools`, a post to a channel not in `channels_write`, a read of a channel not in `channels_read`, a registry write under `registry != "write"`, or a world action with `may_act=False` returns `ok=False, error="not_allowed"` and is logged. The role name is recorded on `turn_started` (`role` field, additive). The role's `prompt_append` and `model` are applied when the participant is bound (LLMAgent only; scripted participants ignore them).

Built-in role library (`swarmlab/roles.py`, exported): `worker` (defaults), `coordinator` (may_act=False, post_fields kind in {assignment, summary}, prompt_append describing its job: read members' reports, post assignments and a summary, never act in the world), `reviewer` (may_act=False, prompt_append: check reports against evidence, post disagreements), `skeptic` (worker who must quote its own evidence verbatim when posting; prompt_append), `scribe` (may_act=False, posts a running summary each round). Each is a `Role` instance; experiments may override any field.

## 2. Tree topology

`Tree(groups: list[list[AgentId]] | int, coordinators: list[AgentId] | None = None, top: AgentId | None = None)` topology in `medium/topology.py`:
- `groups` as an int means equal-size groups by agent index; each group has one coordinator (the first member by default, or the given list); `top` (optional) is a coordinator-of-coordinators.
- Channels are created by the topology at reset: `group:<i>` for each group and `coordinators` for the coordinator tier. Members read and write `group:<i>` only; coordinators read and write their `group:<i>` and `coordinators`; `top` reads and writes `coordinators` only. These channel permissions are applied as role permissions (the topology returns `channel_permissions(agent) -> (read, write)`, merged with the role's lists, intersection wins).
- `recipients(post)` delivers a post to the members of its channel (author excluded). Posts with no channel default to the author's group channel.
- Everything is deterministic from the agent list; the topology is snapshot-free (pure function of params and agent list) unless given explicit lists.

Hierarchy is therefore a spec, not code: a `Tree` topology plus `coordinator` and `worker` roles. The ablation switch is `may_act=True` on the coordinator role.

## 3. Example library and spec

`examples/hierarchy_flaggame.yaml`: Flag Game N=18, two arms: `flat` (broadcast, all workers) and `tree` (3 groups of 6 with coordinators, roles worker/coordinator), fake provider by default with a note on switching to a real model. `examples/roles_demo.py`: the same in Python, showing a custom role.

## 4. Acceptance

1. A role with `tools=["guess","end_turn"]` cannot read or post; the attempts are logged `not_allowed`; the agent's turns still complete.
2. Under `Tree(groups=3)`, a member's post reaches only its group; a coordinator's post on `coordinators` reaches only coordinators; `top` sees only `coordinators`. Deterministic across two runs; replay reproduces deliveries.
3. `coordinator.may_act=False` blocks `guess`; flipping it in the spec allows it and changes the spec hash.
4. `prompt_append` from the role appears in the rendered system prompt (`swarmlab prompts` shows it); `model` override changes the provider used (fake provider ids in tests).
5. Spec hash unchanged for experiments without `roles`; YAML round-trip of roles and tree params.
