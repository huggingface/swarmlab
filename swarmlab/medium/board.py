"""The board: posts, per-recipient inboxes, delivery policies (docs/INTERFACE.md §9).

Lifecycle per round: the executor calls `buffer_post` for each `post` tool call; the runner calls
`commit(round, agents, rng, blobs)` once (phase-commit) or once per post (immediate), which fans
buffered posts out to per-recipient inboxes; `read` (pull) and `pushable` (push) select from an
agent's own inbox. There is no global cursor.

Decisions where the contract is silent:

- Ids. Post ids are `p{round:04d}-{n:04d}` and delivery ids `d{round:04d}-{n:05d}`, with `n`
  counting from 0 within a round in buffer / fan-out order. The counters reset when the round
  changes, not at `commit`, so several single-item commits in one round (immediate mode) never
  reuse an id. Post ids are assigned at `buffer_post` time, so the caller must buffer posts in a
  deterministic order for ids to be reproducible.
- Policies. For each recipient, policies run in order; the first `None` withholds the post from
  that reader (no `Delivery` is created). Otherwise the maximum eligible round and the last
  policy's content win. With no policies the eligible round is `round + 1`. In immediate mode
  (`commit_mode == "immediate"`) the eligible round is `round` when no policy moved it, i.e. when
  there are no policies or every policy returned the default `round + 1`; an explicit delay such
  as `DelayPolicy(rounds=2)` keeps its absolute round. A policy returning a round earlier than
  the commit round is clamped to the commit round.
- `commit_mode` is a run option (`RunOptions.commit`), not part of `MediumSpec`: the runner sets
  `board.commit_mode`, and `spec()` omits it.
- `spec()` always emits the topology as a nested plugin spec (`{"type", "params"}`), even when the
  board was built from a name, and policies as a list of nested specs, matching `MediumSpec`.
  `Board.from_spec` rebuilds nested plugins from those specs (local names first, then the
  `swarmlab.topologies` / `swarmlab.policies` entry points, then `module:Class` import). The
  constructor also accepts such a dict for `topology` and for each policy.
- Channels. `buffer_post` and `read` raise `ValueError` for an unknown channel. A delivery's
  channel is the channel of its post, tracked in an internal map so that `Delivery` keeps exactly
  the contract's fields.
- `read`/`pushable`/`inbox` return copies; `read` marks the stored deliveries as read.
- Persistence. `snapshot()` stores plain data only: the buffer and inboxes as dicts, the id
  counters, the post-to-channel map, and each nested plugin's `(spec, snapshot bytes)`. Config
  (topology choice, policies, channels, delivery, push_limit, commit_mode) is not restored:
  the board keeps the plugins it was constructed with, and a nested plugin's saved state is
  restored only when its `spec()` equals the saved one. A fork under an edited medium therefore
  continues with the new topology and policies and the old inboxes.

Intervention overlays (M3a, docs/INTERFACE-M3a.md §1), owned by the runner's `Ops`:

- `set_delay(agents, rounds, until=None)`: deliveries to these recipients of posts committed in
  rounds <= `until` (forever when None) get `rounds` added to their eligible round, after the
  policies (counting from `round + 1` in immediate mode too, like `DelayPolicy`). Setting replaces an agent's previous delay; `rounds == 0` clears it.
- `mute(agents, until)`: posts by these authors committed in rounds <= `until` are returned by
  `commit` (and logged) but fanned out to nobody. A later mute extends, never shortens.
- `set_policies(policies)` / `set_topology(topology)` replace the configured plugins and record the
  replacement's spec, so it survives snapshot/restore (it wins over the constructor's config).
- `commit(..., recipients=...)`: an explicit recipient list bypasses the topology (used for
  injected posts whose author is not an agent); policies and overlays still apply.
- All of this is in `snapshot()` (keys absent from pre-M3a snapshots restore as empty).

Registry config (M3b, swarmlab/medium/registry.py): `registry: bool = False` and
`claim_policy` (a name, `{type, params}` or `ClaimPolicy`; default "advisory") are medium
settings carried by the board so that `MediumSpec` round-trips through `Board(**params)`. The
board only stores them; the runner builds the `Registry` and the policy. `spec()` emits them only
when they differ from the defaults, so existing specs and hashes are unchanged. A non-advisory
claim policy without the registry is a `ValueError`.
"""
from __future__ import annotations

import importlib
import pickle
import random
from collections.abc import Mapping, Sequence
from importlib.metadata import entry_points
from typing import Any, ClassVar, Literal, Protocol

from pydantic import BaseModel

from ..base import Persistable, Plugin
from ..ids import AgentId, DeliveryId, PostId
from .base import Policy, Topology
from .topology import TOPOLOGIES


class BlobStoreLike(Protocol):
    """What the board needs from `swarmlab.blobs.BlobStore` (INTERFACE.md §5)."""

    def put(self, data: bytes) -> str: ...
    def get(self, sha: str) -> bytes: ...


class Post(BaseModel):
    post_id: PostId
    round: int
    agent: AgentId
    channel: str
    text: str
    fields: dict = {}


class Delivery(BaseModel):
    delivery_id: DeliveryId
    post_id: PostId
    recipient: AgentId
    eligible_round: int
    content_hash: str
    read_round: int | None = None


class DelayPolicy(Policy):
    """Delay posts by `rounds` extra rounds for `readers` (all readers when None)."""

    entry_point: ClassVar[str | None] = "delay"

    def __init__(self, rounds: int, readers: list[AgentId] | None = None) -> None:
        if rounds < 0:
            raise ValueError(f"DelayPolicy rounds must be >= 0, got {rounds}")
        self.rounds = rounds
        self.readers = None if readers is None else list(readers)

    def apply(self, reader: AgentId, post: Post, round: int) -> tuple[int, str] | None:
        if self.readers is None or reader in self.readers:
            return round + 1 + self.rounds, post.text
        return super().apply(reader, post, round)


POLICIES: dict[str, type[Policy]] = {"delay": DelayPolicy}


def build_plugin(spec: Mapping[str, Any], local: Mapping[str, type], group: str) -> Any:
    """Instantiate `{"type", "params"}` via a local name, an entry point, or `module:Class`."""
    type_name = spec["type"]
    params = dict(spec.get("params", {}))
    cls: Any = local.get(type_name)
    if cls is None:
        found = [ep for ep in entry_points(group=group) if ep.name == type_name]
        if found:
            cls = found[0].load()
        elif ":" in type_name:
            module, _, qualname = type_name.partition(":")
            cls = importlib.import_module(module)
            for part in qualname.split("."):
                cls = getattr(cls, part)
        else:
            raise ValueError(f"unknown {group} plugin {type_name!r}")
    return cls(**params)


def _topology(value: str | Topology | Mapping[str, Any]) -> Topology:
    if isinstance(value, Topology):
        return value
    if isinstance(value, str):
        if value not in TOPOLOGIES:
            raise ValueError(f"unknown topology {value!r}; known: {sorted(TOPOLOGIES)}")
        return TOPOLOGIES[value]()
    return build_plugin(value, TOPOLOGIES, "swarmlab.topologies")


def _policy(value: Policy | Mapping[str, Any]) -> Policy:
    if isinstance(value, Policy):
        return value
    return build_plugin(value, POLICIES, "swarmlab.policies")


class Board(Persistable, Plugin):
    entry_point: ClassVar[str | None] = "board"

    def __init__(
        self,
        topology: str | Topology = "broadcast",
        delivery: Literal["pull", "push"] = "pull",
        push_limit: int = 20,
        policies: Sequence[Policy] = (),
        channels: Sequence[str] = ("main",),
        commit_mode: Literal["round_end", "immediate"] = "round_end",
        registry: bool = False,
        claim_policy: Any = "advisory",
    ) -> None:
        if delivery not in ("pull", "push"):
            raise ValueError(f"delivery must be 'pull' or 'push', got {delivery!r}")
        if commit_mode not in ("round_end", "immediate"):
            raise ValueError(f"commit_mode must be 'round_end' or 'immediate', got {commit_mode!r}")
        if not channels:
            raise ValueError("a board needs at least one channel")
        self.topology: Topology = _topology(topology)
        self.delivery = delivery
        self.push_limit = push_limit
        self.policies: list[Policy] = [_policy(p) for p in policies]
        self.channels: list[str] = list(channels)
        self.commit_mode = commit_mode
        from .registry import build_claim_policy

        self.registry = bool(registry)
        self.claim_policy = build_claim_policy(claim_policy)
        if not self.registry and self.claim_policy.type_name() != "advisory":
            raise ValueError("a claim policy other than 'advisory' needs `registry: true`")
        self._buffer: list[Post] = []
        self._inboxes: dict[AgentId, list[Delivery]] = {}
        self._post_channel: dict[PostId, str] = {}
        self._post_seq: tuple[int, int] = (0, 0)       # (round, next n)
        self._delivery_seq: tuple[int, int] = (0, 0)
        # intervention overlays (M3a)
        self._delays: dict[AgentId, tuple[int, int | None]] = {}   # agent -> (rounds, until)
        self._mutes: dict[AgentId, int] = {}                        # author -> last muted round
        self._policy_override: list[dict] | None = None
        self._topology_override: dict | None = None

    # -- spec ------------------------------------------------------------------------------

    def spec(self) -> dict[str, Any]:
        return {
            "type": self.type_name(),
            "params": {
                "topology": self.topology.spec(),
                "delivery": self.delivery,
                "push_limit": self.push_limit,
                "policies": [p.spec() for p in self.policies],
                "channels": list(self.channels),
                **({"registry": True} if self.registry else {}),
                **({"claim_policy": self.claim_policy.spec()}
                   if self.claim_policy.spec() != {"type": "advisory", "params": {}} else {}),
            },
        }

    @classmethod
    def from_spec(cls, spec: Mapping[str, Any]) -> Board:
        """Rebuild from `spec()` output (or a bare params dict); nested specs become plugins."""
        params = dict(spec.get("params", {})) if "type" in spec else dict(spec)
        return cls(**params)

    # -- writes ----------------------------------------------------------------------------

    def _check_channel(self, channel: str) -> None:
        if channel not in self.channels:
            raise ValueError(f"unknown channel {channel!r}; channels: {self.channels}")

    @staticmethod
    def _next(seq: tuple[int, int], round: int) -> tuple[int, tuple[int, int]]:
        n = seq[1] if seq[0] == round else 0
        return n, (round, n + 1)

    def buffer_post(
        self, agent: AgentId, round: int, channel: str, text: str, fields: dict | None = None
    ) -> PostId:
        self._check_channel(channel)
        n, self._post_seq = self._next(self._post_seq, round)
        post_id = PostId(f"p{round:04d}-{n:04d}")
        self._buffer.append(
            Post(post_id=post_id, round=round, agent=agent, channel=channel, text=text,
                 fields=dict(fields or {}))
        )
        return post_id

    def _resolve(self, reader: AgentId, post: Post, round: int) -> tuple[int, str] | None:
        default = round + 1
        if not self.policies:
            return (round if self.commit_mode == "immediate" else default), post.text
        eligible: int | None = None
        content = post.text
        for policy in self.policies:
            result = policy.apply(reader, post, round)
            if result is None:
                return None
            eligible = result[0] if eligible is None else max(eligible, result[0])
            content = result[1]
        assert eligible is not None
        if self.commit_mode == "immediate" and eligible == default:
            eligible = round
        return max(eligible, round), content

    def commit(
        self, round: int, agents: list[AgentId], rng: random.Random, blobs: BlobStoreLike,
        recipients: list[AgentId] | None = None,
    ) -> tuple[list[Post], list[Delivery]]:
        posts = list(self._buffer)
        deliveries: list[Delivery] = []
        for post in posts:
            self._post_channel[post.post_id] = post.channel
            if self._mutes.get(post.agent, -1) >= round:
                continue  # muted author: the post is logged, nobody receives it
            readers = (list(recipients) if recipients is not None
                       else self.topology.recipients(post, list(agents), round, rng))
            for reader in readers:
                if reader == post.agent:
                    continue
                resolved = self._resolve(reader, post, round)
                if resolved is None:
                    continue
                eligible, content = resolved
                extra = self._delay_for(reader, round)
                if extra:  # like DelayPolicy, an explicit delay counts from round + 1 in both modes
                    eligible = max(eligible, round + 1) + extra
                n, self._delivery_seq = self._next(self._delivery_seq, round)
                delivery = Delivery(
                    delivery_id=DeliveryId(f"d{round:04d}-{n:05d}"),
                    post_id=post.post_id,
                    recipient=reader,
                    eligible_round=eligible,
                    content_hash=blobs.put(content.encode()),
                )
                self._inboxes.setdefault(reader, []).append(delivery)
                deliveries.append(delivery.model_copy())
        self._buffer.clear()
        return posts, deliveries

    # -- intervention overlays (M3a) ---------------------------------------------------------

    def _delay_for(self, reader: AgentId, round: int) -> int:
        rounds, until = self._delays.get(reader, (0, None))
        return rounds if until is None or round <= until else 0

    def set_delay(self, agents: Sequence[AgentId], rounds: int, until: int | None = None) -> None:
        if rounds < 0:
            raise ValueError(f"delay rounds must be >= 0, got {rounds}")
        for a in agents:
            if rounds == 0:
                self._delays.pop(AgentId(a), None)
            else:
                self._delays[AgentId(a)] = (rounds, until)

    def mute(self, agents: Sequence[AgentId], until: int) -> None:
        for a in agents:
            self._mutes[AgentId(a)] = max(until, self._mutes.get(AgentId(a), until))

    def overlays(self, round: int) -> dict[str, Any]:
        """Active delays and mutes for commits in `round` (for tests and the viewer)."""
        return {
            "delays": {a: r for a in sorted(self._delays) if (r := self._delay_for(a, round))},
            "mutes": sorted(a for a, until in self._mutes.items() if until >= round),
        }

    def set_policies(self, policies: Sequence[Policy | Mapping[str, Any]]) -> None:
        self.policies = [_policy(p) for p in policies]
        self._policy_override = [p.spec() for p in self.policies]

    def set_topology(self, topology: str | Topology | Mapping[str, Any]) -> None:
        self.topology = _topology(topology)
        self._topology_override = self.topology.spec()

    # -- reads -----------------------------------------------------------------------------

    def _select(self, agent: AgentId, round: int, channel: str | None, limit: int) -> list[Delivery]:
        items = [
            d for d in self._inboxes.get(agent, [])
            if d.read_round is None and d.eligible_round <= round
            and (channel is None or self._post_channel.get(d.post_id) == channel)
        ]
        items.sort(key=lambda d: (d.eligible_round, d.delivery_id))
        return items[: max(limit, 0)]

    def read(
        self, agent: AgentId, round: int, channel: str | None = None, limit: int = 50
    ) -> list[Delivery]:
        """Eligible unread deliveries, oldest first; marks them read in `round`."""
        if channel is not None:
            self._check_channel(channel)
        items = self._select(agent, round, channel, limit)
        for d in items:
            d.read_round = round
        return [d.model_copy() for d in items]

    def pushable(self, agent: AgentId, round: int, limit: int) -> list[Delivery]:
        """What `read` would return (any channel), without marking anything read."""
        return [d.model_copy() for d in self._select(agent, round, None, limit)]

    def content(self, delivery: Delivery, blobs: BlobStoreLike) -> str:
        return blobs.get(delivery.content_hash).decode()

    def channel_of(self, delivery: Delivery) -> str | None:
        return self._post_channel.get(delivery.post_id)

    def inbox(self, agent: AgentId) -> list[Delivery]:
        """Every delivery to `agent`, read or not, in fan-out order (for the viewer)."""
        return [d.model_copy() for d in self._inboxes.get(agent, [])]

    @property
    def buffered(self) -> list[Post]:
        return list(self._buffer)

    # -- persistence -----------------------------------------------------------------------

    def snapshot(self) -> bytes:
        state = {
            "buffer": [p.model_dump(mode="json") for p in self._buffer],
            "inboxes": {a: [d.model_dump(mode="json") for d in ds] for a, ds in self._inboxes.items()},
            "post_channel": dict(self._post_channel),
            "post_seq": self._post_seq,
            "delivery_seq": self._delivery_seq,
            "topology": (self.topology.spec(), self.topology.snapshot()),
            "policies": [(p.spec(), p.snapshot()) for p in self.policies],
            "delays": {a: list(v) for a, v in self._delays.items()},
            "mutes": dict(self._mutes),
            "policy_override": self._policy_override,
            "topology_override": self._topology_override,
        }
        return pickle.dumps(state, protocol=pickle.HIGHEST_PROTOCOL)

    def restore(self, blob: bytes) -> None:
        state = pickle.loads(blob)
        self._buffer = [Post.model_validate(p) for p in state["buffer"]]
        self._inboxes = {
            AgentId(a): [Delivery.model_validate(d) for d in ds] for a, ds in state["inboxes"].items()
        }
        self._post_channel = dict(state["post_channel"])
        self._post_seq = tuple(state["post_seq"])
        self._delivery_seq = tuple(state["delivery_seq"])
        self._delays = {AgentId(a): (int(v[0]), v[1]) for a, v in state.get("delays", {}).items()}
        self._mutes = {AgentId(a): int(v) for a, v in state.get("mutes", {}).items()}
        if state.get("topology_override") is not None:
            self.set_topology(state["topology_override"])
        if state.get("policy_override") is not None:
            self.set_policies(state["policy_override"])
        topo_spec, topo_state = state["topology"]
        if topo_spec == self.topology.spec():
            self.topology.restore(topo_state)
        for policy, (pol_spec, pol_state) in zip(self.policies, state["policies"]):
            if pol_spec == policy.spec():
                policy.restore(pol_state)
