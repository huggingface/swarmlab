"""The registry and claim policies (DESIGN.md component 5; M3b).

The registry is a typed, versioned key-value store for coordination state. Agents reach it only
through the tool executor (`registry_get`, `registry_put`, `registry_cas`, `registry_acquire`,
`registry_release`) when the arm's medium sets `registry: true`. World truth stays in the world,
talk on the board; the registry holds who is doing what.

Entries
-------
`key -> {"value", "version", "owner", "expires_round"}`. A missing key has version 0, no value
and no owner. `value` is any JSON value (the tool schemas advertise a string, the type agents
can always send under strict tool use; scripted participants may store numbers, lists or
objects). Keys are non-empty strings of at most `MAX_KEY_CHARS`; a value's JSON is at most
`MAX_VALUE_CHARS`. Both are checked when the tool is called, so a bad op is never buffered.

Operations and their semantics (all writes bump `version` by one when they succeed):

- `get(key)`: a read; never buffered, never committed, no `registry` event. Under phase-commit
  it sees round-start state plus the caller's own pending writes, applied tentatively in call
  order (`"pending": true` in the result when any of them touched the key); under immediate
  commit it sees the current state.
- `put(key, value)`: sets the value. Always succeeds (ownership does not guard values; use CAS).
- `compare_and_set(key, expected_version, value)`: sets the value iff the entry's version equals
  `expected_version` at commit (0 for a missing key); else `error="version_mismatch"`.
- `acquire(key, ttl_rounds)`: succeeds iff the key has no live owner, or the caller is the live
  owner (a renewal, which moves the expiry). A claim acquired in round r with ttl k is live in
  rounds r .. r + k - 1, so `expires_round = r + k - 1` is the last round it holds and a claim
  of a dead agent lapses on its own. Else `error="held"` with the holder and its expiry.
- `release(key)`: succeeds iff the caller is the live owner; clears owner and expiry. Else
  `error="not_owner"`.

Commit order. The runner commits the round in this order: posts, registry ops, world actions.
Registry ops are applied in the round's seeded agent order, each agent's ops in call order, so
of two agents acquiring the same free key in one round the one scheduled earlier wins and the
other gets `ok=False, error="held"`. Every committed op writes one `registry` event
(`op_id, op, key, ok, version, owner, expires_round, value, error`; `version`/`owner`/
`expires_round` describe the entry after the op), and its outcome reaches the agent's next view
as `{"action_id": op_id, "tool": "registry_<op>", "accepted": ok, "feedback": {...}}`, next to
world-action outcomes. Op ids are `g{round:04d}-{agent}-{n:02d}` (n counts the agent's buffered
registry writes in the round, from 0). Under immediate commit each write applies at once with
the same event and its outcome in the tool result.

Claim policies
--------------
A claim policy links the registry to world validation. The world maps an action to a resource
key with `World.claim_key(agent, action) -> str | None` (default None: not claimable). Because
registry ops commit before world actions, a claim acquired this round protects an action this
round. For each world action with a key, at commit and in seeded order, the runner (via
`commit_world`) reads the key's live owner and asks the policy whether the action may proceed:

- `advisory` (default): always; the world ignores the registry.
- `enforced`: only when the actor is the live owner; otherwise the action is not passed to the
  world and its outcome is `accepted=False, feedback={"error": "not_claimed", "key", "owner"}`.

Under either policy each keyed action writes one `claim` event before the `action_committed`
events: `action_id, key, owner (live owner or None), held (owner == actor), violation (owner is
another agent), rejected, policy`. `claims.violations` counts violations (attempted actions on a
key someone else holds, so under `enforced` they are also rejected), `claims.held` the live
claims after the round's commit. Claims are only checked when the registry is on.

Persistence: `Registry` is `Persistable` (the entries dict; nothing is pending between rounds
because buffered ops live in the round's executor). The runner stores it as `plugins["registry"]`
only when the registry is on, so other runs' manifests are unchanged.
"""
from __future__ import annotations

import copy
import json
from collections.abc import Sequence
from typing import Any, ClassVar

from pydantic import BaseModel

from ..base import Persistable, Plugin
from ..ids import ActionId, AgentId

MAX_KEY_CHARS = 200
MAX_VALUE_CHARS = 4000
WRITE_OPS = ("put", "compare_and_set", "acquire", "release")
TOOL_OPS = {
    "registry_put": "put",
    "registry_cas": "compare_and_set",
    "registry_acquire": "acquire",
    "registry_release": "release",
}


class RegistryOp(BaseModel):
    op_id: str
    agent: AgentId
    op: str
    key: str
    value: Any = None
    expected_version: int | None = None
    ttl_rounds: int | None = None


class RegistryOutcome(BaseModel):
    op_id: str
    agent: AgentId
    op: str
    key: str
    ok: bool
    version: int
    owner: AgentId | None = None
    expires_round: int | None = None
    value: Any = None
    error: str | None = None

    def feedback(self) -> dict:
        out = {"key": self.key, "version": self.version, "owner": self.owner,
               "expires_round": self.expires_round}
        if self.error is not None:
            out["error"] = self.error
        return out

    def view_outcome(self) -> dict:
        tool = {v: k for k, v in TOOL_OPS.items()}[self.op]
        return {"action_id": self.op_id, "tool": tool, "accepted": self.ok, "feedback": self.feedback()}


def _empty() -> dict:
    return {"value": None, "version": 0, "owner": None, "expires_round": None}


def check_op_args(op: str, args: dict) -> tuple[RegistryOp | None, str | None]:
    """Validate tool args for `op` ("get" or a write); returns (partial op, error)."""
    allowed = {
        "get": {"key"},
        "put": {"key", "value"},
        "compare_and_set": {"key", "expected_version", "value"},
        "acquire": {"key", "ttl_rounds"},
        "release": {"key"},
    }[op]
    extra = set(args) - allowed
    missing = allowed - set(args)
    if extra or missing:
        return None, f"bad args: missing={sorted(missing)} extra={sorted(extra)}"
    key = args["key"]
    if not isinstance(key, str) or not key or len(key) > MAX_KEY_CHARS:
        return None, f"bad args: key must be a non-empty string of at most {MAX_KEY_CHARS} chars"
    if "value" in args:
        try:
            size = len(json.dumps(args["value"]))
        except (TypeError, ValueError):
            return None, "bad args: value must be JSON"
        if size > MAX_VALUE_CHARS:
            return None, f"bad args: value longer than {MAX_VALUE_CHARS} chars as JSON"
    for name in ("expected_version", "ttl_rounds"):
        if name in args:
            v = args[name]
            if not isinstance(v, int) or isinstance(v, bool) or v < (1 if name == "ttl_rounds" else 0):
                low = 1 if name == "ttl_rounds" else 0
                return None, f"bad args: {name} must be an integer >= {low}"
    return RegistryOp(op_id="", agent=AgentId(""), op=op, key=key, value=args.get("value"),
                      expected_version=args.get("expected_version"),
                      ttl_rounds=args.get("ttl_rounds")), None


class Registry(Persistable, Plugin):
    entry_point: ClassVar[str | None] = "registry"

    def __init__(self) -> None:
        self.entries: dict[str, dict] = {}

    # ---- reads -------------------------------------------------------------------------------
    @staticmethod
    def _live_owner(entry: dict, round: int) -> AgentId | None:
        exp = entry.get("expires_round")
        if entry.get("owner") is None or exp is None or exp < round:
            return None
        return entry["owner"]

    def owner(self, key: str, round: int) -> AgentId | None:
        """The live owner of `key` in `round`, or None."""
        entry = self.entries.get(key)
        return None if entry is None else self._live_owner(entry, round)

    def get(self, key: str, round: int, pending: Sequence[RegistryOp] = ()) -> dict:
        """Agent-facing read: committed state plus `pending` (the caller's own buffered writes)."""
        entries = {key: copy.deepcopy(self.entries[key])} if key in self.entries else {}
        touched = False
        for op in pending:
            if op.key == key:
                self._apply(entries, op, round)
                touched = True
        entry = entries.get(key)
        if entry is None:
            out = {"key": key, "exists": False, **_empty()}
        else:
            out = {"key": key, "exists": True, "value": entry["value"], "version": entry["version"],
                   "owner": self._live_owner(entry, round),
                   "expires_round": entry["expires_round"] if self._live_owner(entry, round) else None}
        if touched:
            out["pending"] = True
        return out

    def live_claims(self, round: int) -> dict[str, dict]:
        return {k: {"owner": e["owner"], "expires_round": e["expires_round"]}
                for k, e in sorted(self.entries.items()) if self._live_owner(e, round) is not None}

    def registry_state(self, round: int | None = None) -> dict[str, dict]:
        """Every entry (sorted by key); with `round`, `live` says whether its claim holds then."""
        out = {}
        for k, e in sorted(self.entries.items()):
            row = dict(e)
            if round is not None:
                row["live"] = self._live_owner(e, round) is not None
            out[k] = row
        return out

    # ---- writes ------------------------------------------------------------------------------
    def _apply(self, entries: dict[str, dict], op: RegistryOp, round: int) -> RegistryOutcome:
        entry = entries.get(op.key)
        cur = entry if entry is not None else _empty()
        live = self._live_owner(cur, round)
        error = None
        new = dict(cur)
        if op.op == "put":
            new["value"] = op.value
        elif op.op == "compare_and_set":
            if cur["version"] != op.expected_version:
                error = "version_mismatch"
            else:
                new["value"] = op.value
        elif op.op == "acquire":
            if live is not None and live != op.agent:
                error = "held"
            else:
                new["owner"] = op.agent
                new["expires_round"] = round + int(op.ttl_rounds or 1) - 1
        elif op.op == "release":
            if live != op.agent:
                error = "not_owner"
            else:
                new["owner"] = None
                new["expires_round"] = None
        else:
            raise ValueError(f"unknown registry op {op.op!r}")
        if error is None:
            new["version"] = cur["version"] + 1
            entries[op.key] = new
            cur = new
        live = self._live_owner(cur, round)
        return RegistryOutcome(
            op_id=op.op_id, agent=op.agent, op=op.op, key=op.key, ok=error is None,
            version=cur["version"], owner=live, expires_round=cur["expires_round"] if live else None,
            value=op.value if op.op in ("put", "compare_and_set") else None, error=error,
        )

    def apply(self, op: RegistryOp, round: int) -> RegistryOutcome:
        """Commit one op now (immediate mode)."""
        return self._apply(self.entries, op, round)

    def commit(self, round: int, ops: Sequence[RegistryOp]) -> list[RegistryOutcome]:
        """Commit buffered ops in the given (seeded agent, then call) order."""
        return [self._apply(self.entries, op, round) for op in ops]


# ---- claim policies ----------------------------------------------------------------------------

class ClaimPolicy(Persistable, Plugin):
    """Decides whether a world action on a claimable resource may proceed."""

    def allows(self, agent: AgentId, key: str, owner: AgentId | None) -> bool:
        raise NotImplementedError


class Advisory(ClaimPolicy):
    entry_point: ClassVar[str | None] = "advisory"

    def allows(self, agent: AgentId, key: str, owner: AgentId | None) -> bool:
        return True


class Enforced(ClaimPolicy):
    entry_point: ClassVar[str | None] = "enforced"

    def allows(self, agent: AgentId, key: str, owner: AgentId | None) -> bool:
        return owner == agent


CLAIM_POLICIES: dict[str, type[ClaimPolicy]] = {"advisory": Advisory, "enforced": Enforced}


def build_claim_policy(spec: Any) -> ClaimPolicy:
    """A policy from an instance (copied), a name, or a `{type, params}` mapping / PluginSpec."""
    if isinstance(spec, ClaimPolicy):
        return copy.deepcopy(spec)
    if isinstance(spec, str):
        spec = {"type": spec, "params": {}}
    elif not isinstance(spec, dict):
        spec = {"type": spec.type, "params": dict(spec.params)}
    from .board import build_plugin

    policy = build_plugin(spec, CLAIM_POLICIES, "swarmlab.claim_policies")
    if not isinstance(policy, ClaimPolicy):
        raise TypeError(f"{spec['type']!r} is not a ClaimPolicy")
    return policy


def commit_world(world: Any, actions: list[tuple[AgentId, ActionId, Any]], *, registry: Registry | None,
                 policy: ClaimPolicy | None, round: int) -> tuple[list[Any], list[dict]]:
    """`world.commit(actions)` with the claim check; returns (outcomes, claim records).

    Without a registry this is exactly `world.commit(actions)`. Outcomes are in action order;
    rejected actions never reach the world. A claim record is the `claim` event's fields plus
    `agent`.
    """
    if registry is None:
        return world.commit(actions), []
    from ..world.base import Outcome

    policy = policy or Advisory()
    claims: list[dict] = []
    passed: list[int] = []
    outcomes: list[Any] = [None] * len(actions)
    for i, (agent, action_id, action) in enumerate(actions):
        key = world.claim_key(agent, action)
        if key is None:
            passed.append(i)
            continue
        owner = registry.owner(key, round)
        ok = policy.allows(agent, key, owner)
        claims.append({"agent": agent, "action_id": action_id, "key": key, "owner": owner,
                       "held": owner == agent, "violation": owner is not None and owner != agent,
                       "rejected": not ok, "policy": policy.type_name()})
        if ok:
            passed.append(i)
        else:
            outcomes[i] = Outcome(accepted=False, action_id=action_id,
                                  feedback={"error": "not_claimed", "key": key, "owner": owner})
    for i, out in zip(passed, world.commit([actions[i] for i in passed]), strict=True):
        outcomes[i] = out
    return outcomes, claims
