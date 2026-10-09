"""Typed events and the append-only JSONL event log (docs/INTERFACE.md §4).

Decisions where the contract is silent:

- One pydantic class per event type, named `<CamelCaseType>Event` (e.g. `RoundCommittedEvent`).
  `AnyEvent` is the discriminated union on `type`; `parse_event(dict | str)` validates one.
  `EVENT_CLASSES` maps type string -> class. Models forbid unknown fields.
- `seq` is 0-based and dense. `EventLog.append` assigns it (overwriting whatever the event held,
  and setting it on the passed object). `ts` defaults to `time.time()` at construction.
- `agent` defaults to None; `run` and `round` are required.
- `logical_view(events, exclude=("seq", "ts"))` yields **plain dicts**
  (`model_dump(mode="json", exclude=...)`) for every non-operational event. `seq` is excluded by
  default as well as `ts`, because operational events (inference attempts/responses) are
  interleaved into the log as they happen and shift the seq of later logical events; logical
  identity is the order and content of logical events, not their log position. Pass
  `exclude=("ts",)` to keep seq, or add `"run"` to compare a fork with its parent. Accepts typed
  events or dicts.
- Durability: each append is one `write` of one complete line followed by a flush; the file is
  fsynced when a `round_committed` event is appended (and on `sync()`/`close()`).
- Torn tail: a trailing line without a newline terminator, or that fails to parse, is treated
  as a partial write from a crash: iteration stops before it, and the next `append` (or
  `truncate_after`) first cuts the file back to the last complete line. Reading never modifies
  the file. A malformed line *followed by valid lines* is real corruption and raises
  `EventLogCorrupt`.
- M1b: `inference_attempt` gained `category`, `inference_response` gained `served_by`,
  `finish_reason`, `cached`; new logical types `budget`, `budget_changed`, `probe`.
- M3a: new logical type `overflow` (`OverflowEvent`, LLM context limit), written into the agent's
  turn events after its tool events and before `turn_ended`.
- M3a: new logical type `intervention` (`InterventionEvent`).
- M3b: new logical types `registry` (`RegistryEvent`, one per committed registry write) and
  `claim` (`ClaimEvent`, one per world action on a claimable resource when the registry is on);
  see swarmlab/medium/registry.py.
- M3c: `turn_started` gained `role` (the agent's role name, None without one).
- M6: `metric` gained `label` (a string, e.g. `belief.state`'s class), omitted when None.
- M6: `round_started` gained `live` (the live agents), written only when the scheduler's `order`
  is not a permutation of them (OneSpeaker); absent, `order` is the live list as before.
- Refusals: `inference_response` gained `refusal_category` (the response's
  `Refusal.category` when `finish_reason == "refusal"`), omitted when None.
- `truncate_after(seq)` returns the discarded events (parsed) so the runner can move the
  operational ones to `discarded.jsonl`; the rewrite is atomic (temp file + rename + fsync).
"""
from __future__ import annotations

import os
import time
from collections.abc import Iterable, Iterator
from pathlib import Path
from typing import Annotated, Any, Literal, Self

from pydantic import BaseModel, ConfigDict, Field, TypeAdapter, ValidationError, model_serializer

from ._io import atomic_write_bytes

# ---- event models ----------------------------------------------------------------------------


class Event(BaseModel):
    model_config = ConfigDict(extra="forbid")
    seq: int = -1
    run: str
    round: int
    agent: str | None = None
    ts: float = Field(default_factory=time.time)
    type: str


class RunStartedEvent(Event):
    type: Literal["run_started"] = "run_started"
    spec_hash: str
    git_commit: str
    dirty: bool
    run_spec: dict
    parent_run: str | None = None
    fork_round: int | None = None


class RoundStartedEvent(Event):
    type: Literal["round_started"] = "round_started"
    order: list[str]
    live: list[str] | None = None  # M6: the live agents, only when `order` does not list them all

    @model_serializer(mode="wrap")
    def _drop_unset_live(self, handler: Any) -> Any:
        data = handler(self)
        if isinstance(data, dict) and data.get("live", 0) is None:
            data.pop("live")
        return data

    @property
    def live_agents(self) -> list[str]:
        """The round's live agents: `live` when logged, else `order` (every scheduler before M6
        lists every live agent)."""
        return list(self.live) if self.live is not None else list(self.order)


class TurnStartedEvent(Event):
    type: Literal["turn_started"] = "turn_started"
    private: dict = {}  # the observation's evaluator-only data, stripped from the agent's view
    role: str | None = None  # M3c: the agent's role name (None without a role)


class ToolCalledEvent(Event):
    type: Literal["tool_called"] = "tool_called"
    call_id: str
    tool: str
    args: dict = {}


class ToolReturnedEvent(Event):
    type: Literal["tool_returned"] = "tool_returned"
    call_id: str
    result: dict
    pending: bool = False


class InferenceAttemptEvent(Event):
    type: Literal["inference_attempt"] = "inference_attempt"
    call_id: str
    provider: str
    model: str
    request_hash: str
    reserved_usd: float = 0.0
    category: Literal["swarm", "measurement"] = "swarm"


class InferenceResponseEvent(Event):
    type: Literal["inference_response"] = "inference_response"
    call_id: str
    response_hash: str
    usage: dict = {}
    cost_usd: float = 0.0
    latency_s: float = 0.0
    served_by: str | None = None
    finish_reason: str = ""
    cached: bool = False
    attempts: int = 1  # provider attempts (retries + 1); 1 on cache hits
    refusal_category: str | None = None  # a refusal's category; omitted when None

    @model_serializer(mode="wrap")
    def _drop_unset_refusal(self, handler: Any) -> Any:
        data = handler(self)
        if isinstance(data, dict) and data.get("refusal_category", 0) is None:
            data.pop("refusal_category")
        return data


class TurnEndedEvent(Event):
    type: Literal["turn_ended"] = "turn_ended"
    yield_kind: Literal["no_tool", "end_turn", "cap", "error"]
    calls: int
    usage: dict = {}
    error: str | None = None  # traceback text when yield_kind == "error" (added by WP4)


class ReadEvent(Event):
    type: Literal["read"] = "read"
    delivery_ids: list[str]


class PostEvent(Event):
    type: Literal["post"] = "post"
    post_id: str
    provisional_id: str | None = None  # the id the author's post() ack returned (tmp-... under round_end)
    channel: str
    text: str
    fields: dict = {}


class DeliveryEvent(Event):
    type: Literal["delivery"] = "delivery"
    post_id: str
    recipient: str
    delivery_id: str
    eligible_round: int
    content_hash: str


class ActionCommittedEvent(Event):
    type: Literal["action_committed"] = "action_committed"
    action_id: str
    action: dict
    accepted: bool
    feedback: dict = {}


class WorldChangedEvent(Event):
    type: Literal["world_changed"] = "world_changed"
    payload: Any = None


class MetricEvent(Event):
    type: Literal["metric"] = "metric"
    name: str
    value: float | None
    denominator: int
    label: str | None = None  # M6: a categorical metric's value (belief.state); omitted when None

    @model_serializer(mode="wrap")
    def _drop_unset_label(self, handler: Any) -> Any:
        data = handler(self)
        if isinstance(data, dict) and data.get("label", 0) is None:
            data.pop("label")
        return data


class RoundCommittedEvent(Event):
    type: Literal["round_committed"] = "round_committed"
    n_posts: int
    n_actions: int
    n_deliveries: int


class SnapshotEvent(Event):
    type: Literal["snapshot"] = "snapshot"
    manifest_path: str


class BudgetEvent(Event):
    """Ledger state at a commit (M1b; logical). Written before `round_committed`."""

    type: Literal["budget"] = "budget"
    spent_swarm: float
    spent_measurement: float
    reserved: float
    calls: int


class BudgetChangedEvent(Event):
    """`Run.resume(budget=...)` replaced the effective budget (M1b; logical)."""

    type: Literal["budget_changed"] = "budget_changed"
    old: dict
    new: dict


class ProbeEvent(Event):
    """A probe's answer for one agent (M1b, written by WP7's probe hook; logical)."""

    type: Literal["probe"] = "probe"
    probe: str
    question_hash: str
    raw_hash: str
    parsed: dict = {}
    ok: bool
    cost_usd: float = 0.0


class OverflowEvent(Event):
    """An LLM participant's prompt exceeded its context limit (M3a §3; logical).

    One per trimming: `dropped_rounds` earlier rounds were removed (0 under `fail_turn`, or when
    nothing could be dropped); token counts are the provider's estimate before and after.
    `detail` is set when the policy could not run as configured (e.g. `"measurement_budget"`:
    the summarize call was refused, the rounds were dropped without a note).
    """

    type: Literal["overflow"] = "overflow"
    policy: Literal["drop_oldest", "summarize", "fail_turn"]
    dropped_rounds: int
    tokens_before: int
    tokens_after: int
    detail: str | None = None
class InterventionEvent(Event):
    """An intervention operation (M3a, docs/INTERFACE-M3a.md §1; logical).

    One event per affected agent (`agent` and `affected == [agent]`), or one event with
    `affected == []` and `agent` None for a swarm-wide operation. `ok` is False when the world or
    participant hook raised (`NotSupported` or another error, text in `error`). `params` are the
    operation's arguments as JSON; `result` is what the hook returned (`World.intervene`).
    """

    type: Literal["intervention"] = "intervention"
    intervention: str
    op: str
    affected: list[str] = []
    ok: bool = True
    post_id: str | None = None
    params: dict = {}
    result: dict = {}
    error: str | None = None


class RegistryEvent(Event):
    """A committed registry write (M3b; logical). `version`, `owner` (live owner) and
    `expires_round` describe the entry after the op; `value` is the written value (put/CAS)."""

    type: Literal["registry"] = "registry"
    op_id: str
    op: str
    key: str
    ok: bool
    version: int
    owner: str | None = None
    expires_round: int | None = None
    value: Any = None
    error: str | None = None


class ClaimEvent(Event):
    """The claim check of one world action on a claimable resource (M3b; logical)."""

    type: Literal["claim"] = "claim"
    action_id: str
    key: str
    owner: str | None = None
    held: bool
    violation: bool
    rejected: bool
    policy: str


class RunEndedEvent(Event):
    type: Literal["run_ended"] = "run_ended"
    reason: Literal["terminal", "max_rounds", "soft_budget", "hard_ceiling", "hard_ceiling_probes",
                    "total_budget", "error", "stop_condition"]


_ALL = (
    RunStartedEvent, RoundStartedEvent, TurnStartedEvent, ToolCalledEvent, ToolReturnedEvent,
    InferenceAttemptEvent, InferenceResponseEvent, TurnEndedEvent, ReadEvent, PostEvent,
    DeliveryEvent, ActionCommittedEvent, WorldChangedEvent, MetricEvent, RoundCommittedEvent,
    SnapshotEvent, RunEndedEvent, BudgetEvent, BudgetChangedEvent, ProbeEvent,
    OverflowEvent, InterventionEvent, RegistryEvent, ClaimEvent,
)
EVENT_CLASSES: dict[str, type[Event]] = {c.model_fields["type"].default: c for c in _ALL}

AnyEvent = Annotated[
    RunStartedEvent | RoundStartedEvent | TurnStartedEvent | ToolCalledEvent | ToolReturnedEvent
    | InferenceAttemptEvent | InferenceResponseEvent | TurnEndedEvent | ReadEvent | PostEvent
    | DeliveryEvent | ActionCommittedEvent | WorldChangedEvent | MetricEvent | RoundCommittedEvent
    | SnapshotEvent | RunEndedEvent | BudgetEvent | BudgetChangedEvent | ProbeEvent
    | OverflowEvent | InterventionEvent | RegistryEvent | ClaimEvent,
    Field(discriminator="type"),
]
_ADAPTER: TypeAdapter[Event] = TypeAdapter(AnyEvent)

OPERATIONAL_TYPES: frozenset[str] = frozenset({"inference_attempt", "inference_response"})


def parse_event(data: dict | str | bytes) -> Event:
    """Validate one event from a dict or a JSON string into its typed class."""
    if isinstance(data, (str, bytes)):
        return _ADAPTER.validate_json(data)
    return _ADAPTER.validate_python(data)


def logical_view(
    events: Iterable[Event | dict], exclude: Iterable[str] = ("seq", "ts")
) -> Iterator[dict]:
    """Logical events as plain dicts, without operational events or the excluded fields."""
    drop = set(exclude)
    for ev in events:
        etype = ev["type"] if isinstance(ev, dict) else ev.type
        if etype in OPERATIONAL_TYPES:
            continue
        d = dict(ev) if isinstance(ev, dict) else ev.model_dump(mode="json")
        yield {k: v for k, v in d.items() if k not in drop}


# ---- the log ---------------------------------------------------------------------------------


class EventLogCorrupt(RuntimeError):
    """A malformed line is followed by valid lines; not a torn tail from a crash."""


class EventLog:
    """One JSONL file of typed events with dense sequence numbers."""

    def __init__(self, path: Path | str) -> None:
        self.path = Path(path)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self._fh: Any = None
        valid_end, last_seq = self._scan()
        self._next_seq = last_seq + 1
        self._valid_end = valid_end

    # -- scanning ------------------------------------------------------------------------------
    def _lines(self) -> Iterator[tuple[int, bytes, Event]]:
        """Yield (end offset, raw line, event) per complete valid line; stop at a torn tail."""
        if not self.path.exists():
            return
        with open(self.path, "rb") as f:
            offset = 0
            for raw in f:
                if not raw.endswith(b"\n"):
                    return  # torn trailing write
                try:
                    ev = parse_event(raw)
                except (ValidationError, ValueError):
                    if f.read(1):
                        raise EventLogCorrupt(
                            f"{self.path}: malformed line at byte {offset} followed by more data"
                        ) from None
                    return  # newline-terminated but unparsable last line: treat as torn
                offset += len(raw)
                yield offset, raw, ev

    def _scan(self) -> tuple[int, int]:
        end, last = 0, -1
        for end, _, ev in self._lines():
            last = ev.seq
        return end, last

    def _repair(self) -> None:
        """Cut a torn tail so the next append starts on a fresh line."""
        if self.path.exists() and self.path.stat().st_size != self._valid_end:
            self._close_fh()
            with open(self.path, "r+b") as f:
                f.truncate(self._valid_end)
                f.flush()
                os.fsync(f.fileno())

    # -- writing -------------------------------------------------------------------------------
    def _open_fh(self) -> Any:
        if self._fh is None:
            self._repair()
            self._fh = open(self.path, "ab")  # noqa: SIM115 - long-lived append handle
        return self._fh

    def _close_fh(self) -> None:
        if self._fh is not None:
            self._fh.close()
            self._fh = None

    def append(self, event: Event) -> int:
        """Assign the next seq, write one JSON line, fsync on round_committed. Returns seq."""
        fh = self._open_fh()
        event.seq = self._next_seq
        line = event.model_dump_json().encode("utf-8") + b"\n"
        fh.write(line)
        fh.flush()
        if event.type == "round_committed":
            os.fsync(fh.fileno())
        self._next_seq += 1
        self._valid_end += len(line)
        return event.seq

    def extend(self, events: Iterable[Event]) -> list[int]:
        return [self.append(ev) for ev in events]

    def sync(self) -> None:
        if self._fh is not None:
            self._fh.flush()
            os.fsync(self._fh.fileno())

    def close(self) -> None:
        if self._fh is not None:
            self.sync()
        self._close_fh()

    def __del__(self) -> None:
        # flush-and-close without fsync; call close() for a durable shutdown
        fh = getattr(self, "_fh", None)
        if fh is not None:
            fh.close()

    def __enter__(self) -> Self:
        return self

    def __exit__(self, *exc: object) -> None:
        self.close()

    # -- reading -------------------------------------------------------------------------------
    @property
    def next_seq(self) -> int:
        return self._next_seq

    def __iter__(self) -> Iterator[Event]:
        if self._fh is not None:
            self._fh.flush()
        for _, _, ev in self._lines():
            yield ev

    def read_from(self, seq: int) -> Iterator[Event]:
        """Events with `event.seq >= seq`, in log order."""
        return (ev for ev in self if ev.seq >= seq)

    def last_committed(self) -> tuple[int, int] | None:
        """(round, seq) of the last `round_committed` event, or None."""
        last = None
        for ev in self:
            if ev.type == "round_committed":
                last = (ev.round, ev.seq)
        return last

    def truncate_after(self, seq: int) -> list[Event]:
        """Keep events with `event.seq <= seq` (atomic rewrite); return the discarded ones."""
        if self._fh is not None:
            self._fh.flush()
        kept: list[bytes] = []
        dropped: list[Event] = []
        last = -1
        for _, raw, ev in self._lines():
            if ev.seq <= seq:
                kept.append(raw)  # original bytes, so kept lines are byte-identical
                last = ev.seq
            else:
                dropped.append(ev)
        self._close_fh()
        data = b"".join(kept)
        atomic_write_bytes(self.path, data, sync=True)
        self._valid_end = len(data)
        self._next_seq = last + 1
        return dropped


def write_jsonl(path: Path | str, events: Iterable[Event], *, append: bool = True) -> None:
    """Write events verbatim (seq kept) to a side file such as `discarded.jsonl`."""
    with open(path, "ab" if append else "wb") as f:
        f.writelines(ev.model_dump_json().encode("utf-8") + b"\n" for ev in events)
        f.flush()
        os.fsync(f.fileno())
