"""Roles (docs/INTERFACE-M3c.md §1): named bundles of prompt, permissions, model and budget.

A `Role` is declared on the experiment (`Experiment.roles`, YAML `roles:`) and assigned to agents
(`assign(participant, name)`, YAML `participants[].role`). The executor enforces its permissions
(swarmlab/executor.py); `LLMAgent.apply_role` applies its prompt, model and `max_tokens` when the
runner binds the participant. The built-in library is `WORKER`, `COORDINATOR`, `REVIEWER`,
`SKEPTIC`, `SCRIBE` (`BUILTIN_ROLES` by name).

Decisions where the contract is silent:

- Assignment is explicit: `assign(p, "coordinator")` records the name on the participant (a
  private attribute, so it is not a constructor param and the participant's own spec does not
  change). `LLMAgent`'s `role` constructor param stays what it was before M3c, the word shown in
  the prompt template; it is not an assignment. At bind, `apply_role` sets it to the role's name.
- Name resolution: a role name resolves to the experiment's `roles[name]`; a name that is not
  declared but names a built-in resolves to the built-in (and is written into the spec). Any other
  name is a `ValueError` (`SpecError` from YAML).
- Overriding a built-in: a YAML role entry (a mapping) whose name is a built-in starts from that
  built-in and replaces the given fields, so `roles: {coordinator: {may_act: true}}` is the
  hierarchy ablation switch. Any other name starts from `Role(name=...)` defaults. In Python, pass
  `COORDINATOR.model_copy(update={...})` or a fresh `Role`. The dict key always wins over the
  role's `name` field.
- Spec: `RunSpec.roles` holds every resolved role (full `Role` dumps, so the hash covers built-in
  prompt text too) and `RunSpec.participant_roles` the role name per agent (None = no role). Both
  are left out of `spec_hash` when empty, so experiments without roles keep their hash. Roles are
  active when the experiment declares any or any participant is assigned one.
- `budget` keys: `max_calls` (the executor's per-turn tool-call cap for that agent, replacing
  `max_calls_per_turn`; going over it is the ordinary `cap` turn end) and `max_tokens`
  (`LLMAgent.max_tokens`). Other keys are a `ValueError`.
- `post_fields` (`{field: [allowed values]}`): when set, a post's `fields` may only use these
  keys and values (`not_allowed` otherwise), and the `post` schema advertises them as an
  optional `fields` object with enums. When unset, `fields` is accepted as before.
- `tools` filters every tool but `end_turn`, which is always allowed. `may_act=False` removes the
  world's action tools (status tools are not actions). `registry`: "none" removes every registry
  tool, "read" keeps `registry_get` only.
- Topology permissions (`Tree.channel_permissions`) are intersected with the role's channel
  lists (None means all) by `bind_roles`; an agent with no role under such a topology gets an
  anonymous effective role (name "") that carries only the channel limits, and its
  `turn_started.role` stays None.
- `reconfigure` interventions do not touch roles (not required by M3c).
"""
from __future__ import annotations

from collections.abc import Mapping, Sequence
from typing import TYPE_CHECKING, Any, Literal

from pydantic import BaseModel, ConfigDict, field_validator

if TYPE_CHECKING:
    from .ids import AgentId
    from .medium.board import Board

BUDGET_KEYS = ("max_calls", "max_tokens")
ROLE_ATTR = "_swarmlab_role"


class Role(BaseModel):
    model_config = ConfigDict(extra="forbid")

    name: str
    prompt_append: str | None = None          # appended to the rendered system prompt
    system_prompt: str | None = None          # full override of LLMAgent.system_prompt (rarely)
    tools: list[str] | None = None            # allowlist of tool names; None = all
    channels_read: list[str] | None = None    # None = all
    channels_write: list[str] | None = None   # None = all
    registry: Literal["none", "read", "write"] = "write"
    may_act: bool = True
    model: str | None = None                  # overrides the participant's model
    budget: dict | None = None                # per-turn overrides: max_calls, max_tokens
    post_fields: dict | None = None           # {field: [allowed values]}

    @field_validator("model")
    @classmethod
    def _model(cls, v: str | None) -> str | None:
        if v is not None and ":" not in v:
            raise ValueError(f"role model must be '<prefix>:<id>', got {v!r}")
        return v

    @field_validator("budget")
    @classmethod
    def _budget(cls, v: dict | None) -> dict | None:
        if v is None:
            return v
        unknown = set(v) - set(BUDGET_KEYS)
        if unknown:
            raise ValueError(f"role budget: unknown keys {sorted(unknown)}; allowed {list(BUDGET_KEYS)}")
        for k, n in v.items():
            if not isinstance(n, int) or isinstance(n, bool) or n < 1:
                raise ValueError(f"role budget {k} must be an integer >= 1, got {n!r}")
        return dict(v)

    @field_validator("post_fields")
    @classmethod
    def _post_fields(cls, v: dict | None) -> dict | None:
        if v is None:
            return v
        out = {}
        for k, vals in v.items():
            if not isinstance(vals, (list, tuple)) or not all(isinstance(x, str) for x in vals):
                raise ValueError(f"post_fields[{k!r}] must be a list of strings")
            out[str(k)] = list(vals)
        return out

    # ---- permission helpers (used by the executor) ----------------------------------------
    def can_read(self, channel: str | None) -> bool:
        return self.channels_read is None or channel in self.channels_read

    def can_write(self, channel: str | None) -> bool:
        return self.channels_write is None or channel in self.channels_write


# ---- built-in library ----------------------------------------------------------------------------
WORKER = Role(name="worker")

COORDINATOR = Role(
    name="coordinator",
    may_act=False,
    post_fields={"kind": ["assignment", "summary"]},
    prompt_append=(
        "Your job: you coordinate your group; you never act in the world yourself. Each round, "
        "read your members' reports on the board, then post assignments (fields kind=assignment) "
        "telling members what to check or do next, and post a short summary (fields "
        "kind=summary) of what your group knows, citing the evidence. Your only influence is "
        "through your posts."
    ),
)

REVIEWER = Role(
    name="reviewer",
    may_act=False,
    prompt_append=(
        "Your job: review. Read the reports on the board and check each claim against the "
        "evidence quoted with it. Post a reply naming every claim you disagree with and why, "
        "quoting the evidence. You never act in the world yourself."
    ),
)

SKEPTIC = Role(
    name="skeptic",
    prompt_append=(
        "When you post, quote your own evidence verbatim (exactly as you observed it) before any "
        "conclusion, and treat others' conclusions as unproven until their evidence is quoted."
    ),
)

SCRIBE = Role(
    name="scribe",
    may_act=False,
    prompt_append=(
        "Your job: scribe. Each round, read the board and post one running summary of what the "
        "group has established so far, what is disputed, and what is still unknown. You never act "
        "in the world yourself."
    ),
)

BUILTIN_ROLES: dict[str, Role] = {r.name: r for r in (WORKER, COORDINATOR, REVIEWER, SKEPTIC, SCRIBE)}


# ---- resolution ------------------------------------------------------------------------------------
def resolve_role(name: str, value: Role | Mapping[str, Any] | None = None) -> Role:
    """`value` as a Role named `name`: a Role is renamed; a mapping overrides the built-in of
    that name (or `Role(name=name)` defaults); None is the built-in (KeyError if there is none)."""
    if isinstance(value, Role):
        return value if value.name == name else value.model_copy(update={"name": name})
    if value is None:
        if name not in BUILTIN_ROLES:
            raise KeyError(f"unknown role {name!r}; built-ins: {sorted(BUILTIN_ROLES)}")
        return BUILTIN_ROLES[name]
    base = BUILTIN_ROLES[name].model_dump() if name in BUILTIN_ROLES else {}
    return Role.model_validate({**base, **dict(value), "name": name})


def merge_role_docs(*layers: Mapping[str, Mapping[str, Any]]) -> dict[str, dict]:
    """Field-merge YAML role mappings (later layers win per field)."""
    out: dict[str, dict] = {}
    for layer in layers:
        for name, fields in (layer or {}).items():
            out[name] = {**out.get(name, {}), **dict(fields or {})}
    return out


def assign(participant: Any, role: str | Role | None) -> Any:
    """Assign a role (by name) to a participant prototype; returns the participant."""
    name = role.name if isinstance(role, Role) else role
    participant.__dict__[ROLE_ATTR] = name
    return participant


def role_name(participant: Any) -> str | None:
    """The role name assigned with `assign`, or None."""
    value = getattr(participant, "__dict__", {}).get(ROLE_ATTR)
    return value if isinstance(value, str) else None


def spec_roles(roles: Mapping[str, Role | Mapping[str, Any]],
               names: Sequence[str | None]) -> tuple[dict[str, dict], list[str | None]]:
    """(`RunSpec.roles`, `RunSpec.participant_roles`) for declared roles and per-agent names.

    Empty pair when nothing is declared or assigned (the spec then hashes as before M3c).
    """
    if not roles and not any(names):
        return {}, []
    resolved = {k: resolve_role(k, v) for k, v in roles.items()}
    for n in names:
        if n and n not in resolved:
            if n not in BUILTIN_ROLES:
                raise ValueError(f"participant role {n!r} is neither declared in roles "
                                 f"({sorted(resolved)}) nor a built-in ({sorted(BUILTIN_ROLES)})")
            resolved[n] = BUILTIN_ROLES[n]
    return {k: r.model_dump(mode="json") for k, r in resolved.items()}, list(names)


def roles_from_spec(roles: Mapping[str, Mapping[str, Any]]) -> dict[str, Role]:
    return {k: Role.model_validate({**dict(v), "name": k}) for k, v in roles.items()}


def agent_roles(roles: Mapping[str, Mapping[str, Any]], names: Sequence[str | None],
                n_agents: int) -> list[Role | None]:
    """The role of each agent (None = no role) from the spec fields."""
    built = roles_from_spec(roles)
    names = list(names) + [None] * (n_agents - len(names))
    return [built[n] if n else None for n in names[:n_agents]]


def _intersect(a: list[str] | None, b: list[str] | None) -> list[str] | None:
    if a is None:
        return None if b is None else list(b)
    if b is None:
        return list(a)
    return [c for c in a if c in b]


def bind_roles(board: Board, participants: Mapping[AgentId, Any],
               roles: Sequence[Role | None]) -> dict[AgentId, Role]:
    """Prepare roles for a run (called by the runner right after `bind`).

    Resets a topology that lays out channels (`Tree`) with the full agent list and adds its
    channels to the board; calls `apply_role(role)` on each participant that has a role and the
    hook; returns the effective role per agent (role lists intersected with the topology's
    channel permissions) for the executor. Agents with neither a role nor topology permissions
    are left out.
    """
    agents = list(participants)
    topology = board.topology
    reset = getattr(topology, "reset_agents", None)
    if callable(reset):
        reset(agents)
        for ch in topology.channels():
            if ch not in board.channels:
                board.channels.append(ch)
    perms = getattr(topology, "channel_permissions", None)
    out: dict[AgentId, Role] = {}
    for a, role in zip(agents, list(roles) + [None] * (len(agents) - len(roles)), strict=False):
        if role is not None:
            apply = getattr(participants[a], "apply_role", None)
            if callable(apply):
                apply(role)
        limits = perms(a) if callable(perms) else None
        if role is None and limits is None:
            continue
        eff = role or Role(name="")
        if limits is not None:
            read, write = limits
            eff = eff.model_copy(update={"channels_read": _intersect(eff.channels_read, read),
                                         "channels_write": _intersect(eff.channels_write, write)})
        out[a] = eff
    return out
