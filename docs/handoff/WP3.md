# WP3 handoff: medium (board, topologies, delay policy)

Branch `wp3`. Files: `swarmlab/medium/board.py`, `swarmlab/medium/topology.py`,
`tests/test_board.py`, `tests/test_topology.py`. Also removed one unused import in
`swarmlab/world/base.py` so `ruff check swarmlab` is clean (separate commit; identical to what any
other WP would do).

## Signatures WP4 needs

```python
from swarmlab.medium.board import Board, Post, Delivery, DelayPolicy, BlobStoreLike, build_plugin
from swarmlab.medium.topology import Broadcast, Gossip, Groups, TOPOLOGIES

class Post(BaseModel):     post_id: PostId; round: int; agent: AgentId; channel: str; text: str; fields: dict = {}
class Delivery(BaseModel): delivery_id: DeliveryId; post_id: PostId; recipient: AgentId
                           eligible_round: int; content_hash: str; read_round: int | None = None

class Board(Persistable, Plugin):            # entry_point "board"
    def __init__(self, topology: str | Topology = "broadcast", delivery: Literal["pull","push"] = "pull",
                 push_limit: int = 20, policies: Sequence[Policy] = (), channels: Sequence[str] = ("main",),
                 commit_mode: Literal["round_end","immediate"] = "round_end")
    topology: Topology; policies: list[Policy]; channels: list[str]
    delivery: str; push_limit: int; commit_mode: str           # plain attributes; runner may set commit_mode
    def buffer_post(self, agent, round, channel, text, fields=None) -> PostId     # ValueError on unknown channel
    def commit(self, round, agents, rng, blobs) -> tuple[list[Post], list[Delivery]]
    def read(self, agent, round, channel=None, limit=50) -> list[Delivery]       # marks read_round=round
    def pushable(self, agent, round, limit) -> list[Delivery]                     # does not mark
    def content(self, delivery, blobs) -> str
    def channel_of(self, delivery) -> str | None
    def inbox(self, agent) -> list[Delivery]                                      # all, read or not
    buffered: list[Post]                                                          # property, copy
    def spec(self) -> dict                    # {"type":"board","params":{topology:{type,params}, delivery,
                                              #   push_limit, policies:[{type,params}], channels:[...]}}
    @classmethod
    def from_spec(cls, spec_or_params: Mapping) -> Board
    def snapshot(self) -> bytes; def restore(self, blob: bytes) -> None

class DelayPolicy(Policy):                   # entry_point "delay"
    def __init__(self, rounds: int, readers: list[AgentId] | None = None)
```

`blobs` is anything with `put(bytes) -> str` and `get(str) -> bytes` (`BlobStoreLike`); the real
`swarmlab.blobs.BlobStore` from WP1 fits.

## Runner wiring (INTERFACE §12)

- `board.commit_mode = options.commit` before round 1 (it is not in `spec()`, see below).
- Phase-commit: `posts, deliveries = board.commit(r, live_agents, derive(seed, "topology", r), blobs)`,
  then log `post` per `Post` and `delivery` per `Delivery` (event fields map 1:1;
  `content_hash` is the blob sha).
- Immediate: `buffer_post` then `commit` per post; same-round eligibility follows from
  `commit_mode`. Passing a fresh `derive(seed,"topology",r)` each time or one shared rng both give the
  same gossip schedule.
- Push: `board.pushable(agent, r, board.push_limit)` for `View`; `read_board` maps to
  `board.read(agent, r, channel, limit)`; log `read` with the returned delivery ids, and get
  verbatim content with `board.content(d, blobs)`.
- **Post id determinism**: ids are assigned in `buffer_post` call order. If the executor calls
  `buffer_post` from concurrently running turns, ids depend on async interleaving. Either buffer posts
  per agent in the executor and call `buffer_post` in seeded `order` at commit time (the id returned
  to the agent as `pending` then has to be assigned after the fact), or ensure turns are interleaved
  deterministically. This is WP4's call; the board just numbers in call order.
- `Experiment.medium: Board = Board()` as a pydantic default may be shared across instances;
  deep-copy the medium per run (`copy.deepcopy(board)` works).
- `swarmlab/__init__.py` does not re-export `Board` yet; WP4 can add it alongside `Experiment`.

## Decisions where the contract was silent

- **Ids**: `p{round:04d}-{n:04d}`, `d{round:04d}-{n:05d}`, `n` from 0 per round; counters reset when the
  round changes, not at commit, so immediate mode never reuses ids.
- **Policy combination**: per recipient, policies in order; the first `None` withholds (no
  `Delivery`, later policies not called). Otherwise max eligible round, last content. Eligible
  rounds below the commit round are clamped to it.
- **Immediate mode + policies**: eligible round is `round` when there are no policies or when every
  policy returned the default `round + 1` ("no policy touched it"); an explicit delay keeps its
  absolute value (e.g. `DelayPolicy(2)` in round 3 gives 6). So DESIGN example 2 keeps same-round
  visibility for even agents in immediate mode.
- **Author exclusion** is enforced by the board too, even if a custom topology returns the author.
- **Gossip(k)** is directed: each author gets `k` distinct partners (`rng.sample` over the other agents
  in `agents` order), all of its posts that round go to them. The assignment is computed once per
  `(round, agents)` from a clone of the passed rng (so it does not consume the runner's rng) and cached
  outside the snapshot. Author not in `agents` gets no recipients.
- **Groups(size)**: group = `int(agent[1:]) // size`; recipients are live group members minus author.
- **Channels**: unknown channel in `buffer_post` or `read` raises `ValueError` (the executor should map
  it to an error tool result). `Delivery` keeps exactly the contract's fields; the channel is tracked
  in an internal post-to-channel map (`channel_of`).
- **Returned objects are copies**; only `read` mutates stored deliveries.
- **spec()**: topology always emitted as a nested plugin spec (also when built from a name), policies
  as nested specs; matches `MediumSpec`. `commit_mode` is omitted because it is `RunOptions.commit`.
  `from_spec` resolves nested types by local name (`broadcast/gossip/groups`, `delay`), then entry
  points (`swarmlab.topologies`, `swarmlab.policies`), then `module:Class` import; the constructor
  also accepts those dict specs for `topology` and each policy.
- **Persistence** (overrides the Persistable default): the snapshot is a pickle of plain data only
  (buffer and inboxes as JSON-mode dicts, id counters, post-to-channel map) plus each nested plugin's
  `(spec(), snapshot())`. No pydantic objects or sets are pickled, so bytes are stable across
  processes. Restore never replaces the board's topology/policies/config; nested plugin state is
  restored only when the saved spec equals the current plugin's spec. A fork under an edited medium
  thus continues with the new rules and the old inboxes ("policy changes apply to posts committed
  after the change", DESIGN §5).
- The board keeps no list of committed posts; the event log holds them.
