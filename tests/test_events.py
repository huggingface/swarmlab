import json

import pytest

from swarmlab.events import (
    EVENT_CLASSES,
    OPERATIONAL_TYPES,
    ActionCommittedEvent,
    BudgetChangedEvent,
    BudgetEvent,
    DeliveryEvent,
    Event,
    EventLog,
    EventLogCorrupt,
    InferenceAttemptEvent,
    InferenceResponseEvent,
    MetricEvent,
    PostEvent,
    ProbeEvent,
    ReadEvent,
    RoundCommittedEvent,
    RoundStartedEvent,
    RunEndedEvent,
    RunStartedEvent,
    SnapshotEvent,
    ToolCalledEvent,
    ToolReturnedEvent,
    TurnEndedEvent,
    TurnStartedEvent,
    WorldChangedEvent,
    logical_view,
    parse_event,
    write_jsonl,
)

R = "e__s1"


def one_of_each() -> list[Event]:
    return [
        RunStartedEvent(run=R, round=0, spec_hash="h", git_commit="c", dirty=False,
                        run_spec={"x": 1}),
        RoundStartedEvent(run=R, round=1, order=["a001", "a000"]),
        TurnStartedEvent(run=R, round=1, agent="a001"),
        ToolCalledEvent(run=R, round=1, agent="a001", call_id="c1", tool="guess",
                        args={"candidate": "B"}),
        InferenceAttemptEvent(run=R, round=1, agent="a001", call_id="c0", provider="fake",
                              model="m", request_hash="rq", reserved_usd=0.1),
        InferenceResponseEvent(run=R, round=1, agent="a001", call_id="c0", response_hash="rs",
                               usage={"prompt_tokens": 3}, cost_usd=0.01, latency_s=0.2),
        ToolReturnedEvent(run=R, round=1, agent="a001", call_id="c1", result={"id": "x1"},
                          pending=True),
        ReadEvent(run=R, round=1, agent="a001", delivery_ids=["d1"]),
        TurnEndedEvent(run=R, round=1, agent="a001", yield_kind="end_turn", calls=2,
                       usage={"calls": 2}),
        PostEvent(run=R, round=1, agent="a001", post_id="p1", channel="main", text="hi",
                  fields={"k": 1}),
        DeliveryEvent(run=R, round=1, post_id="p1", recipient="a000", delivery_id="d2",
                      eligible_round=2, content_hash="0" * 64),
        ActionCommittedEvent(run=R, round=1, agent="a001", action_id="x1",
                             action={"name": "guess", "args": {"candidate": "B"}},
                             accepted=True, feedback={"recorded": True}),
        WorldChangedEvent(run=R, round=1, payload=[1, {"a": 2}]),
        MetricEvent(run=R, round=1, name="belief.consensus", value=None, denominator=0),
        RoundCommittedEvent(run=R, round=1, n_posts=1, n_actions=1, n_deliveries=1),
        SnapshotEvent(run=R, round=1, manifest_path="snapshots/000001.json"),
        RunEndedEvent(run=R, round=1, reason="max_rounds"),
        BudgetEvent(run=R, round=1, spent_swarm=0.5, spent_measurement=0.1, reserved=0.0, calls=3),
        BudgetChangedEvent(run=R, round=1, old={"hard_usd": 1.0}, new={"hard_usd": 2.0}),
        ProbeEvent(run=R, round=1, agent="a000", probe="belief", question_hash="q", raw_hash="r",
                   parsed={"candidate": "A"}, ok=True, cost_usd=0.001),
    ]


def test_every_type_has_a_class():
    expected = {
        "run_started", "round_started", "turn_started", "tool_called", "tool_returned",
        "inference_attempt", "inference_response", "turn_ended", "read", "post", "delivery",
        "action_committed", "world_changed", "metric", "round_committed", "snapshot", "run_ended",
        "budget", "budget_changed", "probe",
    }
    assert set(EVENT_CLASSES) == expected
    assert {e.type for e in one_of_each()} == expected
    assert OPERATIONAL_TYPES == {"inference_attempt", "inference_response"}


def test_append_iter_round_trip(tmp_path):
    evs = one_of_each()
    log = EventLog(tmp_path / "events.jsonl")
    seqs = [log.append(e) for e in evs]
    assert seqs == list(range(len(evs)))
    assert [e.seq for e in evs] == seqs
    back = list(log)
    assert back == evs
    assert [type(b) for b in back] == [type(e) for e in evs]
    # a fresh instance reads the same and continues the sequence
    log2 = EventLog(tmp_path / "events.jsonl")
    assert list(log2) == evs
    assert log2.append(TurnStartedEvent(run=R, round=2, agent="a000")) == len(evs)


def test_parse_and_validation():
    e = parse_event({"type": "read", "run": R, "round": 1, "delivery_ids": []})
    assert isinstance(e, ReadEvent) and e.agent is None
    assert isinstance(parse_event(e.model_dump_json()), ReadEvent)
    with pytest.raises(ValueError):
        parse_event({"type": "nope", "run": R, "round": 1})
    with pytest.raises(ValueError):
        parse_event({"type": "read", "run": R, "round": 1, "delivery_ids": [], "extra": 1})
    with pytest.raises(ValueError):
        TurnEndedEvent(run=R, round=1, yield_kind="bored", calls=0)


def committed_log(path, rounds=3) -> EventLog:
    log = EventLog(path)
    log.append(RunStartedEvent(run=R, round=0, spec_hash="h", git_commit="c", dirty=False,
                               run_spec={}))
    for r in range(1, rounds + 1):
        log.append(RoundStartedEvent(run=R, round=r, order=["a000"]))
        log.append(TurnStartedEvent(run=R, round=r, agent="a000"))
        log.append(RoundCommittedEvent(run=R, round=r, n_posts=0, n_actions=0, n_deliveries=0))
    return log


def test_last_committed(tmp_path):
    assert EventLog(tmp_path / "empty.jsonl").last_committed() is None
    log = committed_log(tmp_path / "e.jsonl")
    assert log.last_committed() == (3, 9)
    log.append(RoundStartedEvent(run=R, round=4, order=[]))
    assert log.last_committed() == (3, 9)


def test_truncate_after(tmp_path):
    path = tmp_path / "e.jsonl"
    log = committed_log(path)
    log.append(RoundStartedEvent(run=R, round=4, order=["a000"]))
    log.append(InferenceAttemptEvent(run=R, round=4, agent="a000", call_id="c", provider="p",
                                     model="m", request_hash="h"))
    before = path.read_bytes().splitlines(keepends=True)
    _, seq = log.last_committed()
    dropped = log.truncate_after(seq)
    assert [d.type for d in dropped] == ["round_started", "inference_attempt"]
    assert path.read_bytes() == b"".join(before[: seq + 1])  # kept lines byte-identical
    assert [e.seq for e in log] == list(range(seq + 1))
    assert log.append(RoundStartedEvent(run=R, round=4, order=[])) == seq + 1
    assert EventLog(path).next_seq == seq + 2
    log.truncate_after(-1)
    assert list(log) == [] and log.next_seq == 0


def test_read_from(tmp_path):
    log = committed_log(tmp_path / "e.jsonl", rounds=2)
    assert [e.seq for e in log.read_from(4)] == [4, 5, 6]


@pytest.mark.parametrize("tail", [b'{"seq": 10, "run": "e__s1", "rou', b"garbage\n"])
def test_torn_trailing_line(tmp_path, tail):
    path = tmp_path / "e.jsonl"
    committed_log(path).close()
    good = path.read_bytes()
    with open(path, "ab") as f:
        f.write(tail)
    log = EventLog(path)
    assert len(list(log)) == 10
    assert log.last_committed() == (3, 9)
    assert path.read_bytes() == good + tail  # reading does not modify the file
    assert log.append(RoundStartedEvent(run=R, round=4, order=[])) == 10
    assert path.read_bytes().startswith(good)
    assert [e.seq for e in EventLog(path)] == list(range(11))


def test_midfile_corruption_raises(tmp_path):
    path = tmp_path / "e.jsonl"
    committed_log(path).close()
    lines = path.read_bytes().splitlines(keepends=True)
    lines[3] = b"not json\n"
    path.write_bytes(b"".join(lines))
    with pytest.raises(EventLogCorrupt):
        EventLog(path)


def test_logical_view(tmp_path):
    evs = one_of_each()
    log = EventLog(tmp_path / "e.jsonl")
    log.extend(evs)
    view = list(logical_view(log))
    assert len(view) == len(evs) - 2
    assert all(v["type"] not in OPERATIONAL_TYPES for v in view)
    assert all("ts" not in v and "seq" not in v for v in view)
    # identical content with different timestamps and interleaved operational events compares equal
    other = [e.model_copy(update={"ts": 0.0}) for e in evs if e.type not in OPERATIONAL_TYPES]
    for i, e in enumerate(other):
        e.seq = i
    assert list(logical_view(other)) == view
    # dict input and custom exclusion
    as_dicts = [json.loads(e.model_dump_json()) for e in evs]
    assert list(logical_view(as_dicts)) == view
    with_seq = list(logical_view(log, exclude=("ts",)))
    assert with_seq[0]["seq"] == 0


def test_write_jsonl(tmp_path):
    evs = one_of_each()[4:6]
    write_jsonl(tmp_path / "discarded.jsonl", evs)
    lines = (tmp_path / "discarded.jsonl").read_text().splitlines()
    assert [parse_event(x) for x in lines] == evs
