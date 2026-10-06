"""LLMAgent (docs/INTERFACE-M1b.md §4) on the fake provider: both tool protocols, memory, yields,
call caps, images, prompts, persistence."""
import json
import pickle

import pytest

from swarmlab import Board, Run
from swarmlab.participants import LLMAgent
from swarmlab.participants.llm import (
    ToolJsonError,
    parse_tool_json,
    render_system_prompt,
)
from swarmlab.providers.base import ChatMessage
from swarmlab.providers.fake import FakeProvider
from swarmlab.snapshot import SnapshotStore
from swarmlab.world.flaggame import FlagGame

from .helpers import ImageFlagGame, llm_agent_experiment, logical


@pytest.fixture
def requests_seen(monkeypatch):
    seen: list = []
    original = FakeProvider.complete

    async def recording(self, request):
        seen.append(request)
        return await original(self, request)

    monkeypatch.setattr(FakeProvider, "complete", recording)
    return seen


def agent_state(run, round, agent="a000"):
    store = SnapshotStore(run.dir)
    return pickle.loads(store.load(store.read(round))[f"participant:{agent}"])


def turns(run):
    return [e for e in run.events if e["type"] == "turn_ended"]


def messages(state):
    return [m for r in state["rounds"] for m in r["messages"]]


def well_formed(msgs):
    """Every native tool call is answered by exactly one tool message with its id."""
    pending: list[str] = []
    for m in msgs:
        if m["role"] == "assistant":
            assert not pending
            pending = [tc["call_id"] for tc in m.get("tool_calls") or []]
        elif m["role"] == "tool":
            assert m["tool_call_id"] == pending.pop(0)
    assert not pending


def test_native_loop_and_conversation_shape(tmp_path):
    run = llm_agent_experiment(4).run(seed=1, max_rounds=2, out=tmp_path)
    assert run.end_reason == "max_rounds"
    assert {t["yield_kind"] for t in turns(run)} == {"end_turn"}
    state = agent_state(run, 1)
    assert state["system"].startswith("You are agent a000")
    roles = [m["role"] for m in messages(state)]
    # round message, post+read_board, two results, guess+end_turn, two results
    assert roles == ["user", "assistant", "tool", "tool", "assistant", "tool", "tool"]
    first = messages(state)[0]["content"]
    assert first[0]["text"] == "Round 1." and "Candidate flags:" in first[1]["text"]
    well_formed(messages(agent_state(run, 2)))
    usage = next(t["usage"] for t in turns(run) if t["agent"] == "a000" and t["round"] == 1)
    assert usage["calls"] == 4 and usage["inference_calls"] == 2 and usage["cost_usd"] > 0
    assert run.score["n_guessed"] == 4
    # round 2's message reports round 1's outcomes
    r2 = agent_state(run, 2)["rounds"][1]["messages"][0]["content"]
    assert any("Outcomes of your actions last round:" in (p["text"] or "") for p in r2)


def test_json_protocol_runs_the_same_game(tmp_path, requests_seen):
    run = llm_agent_experiment(4, agent_kw={"tool_protocol": "json"}).run(
        seed=1, max_rounds=2, out=tmp_path)
    assert {t["yield_kind"] for t in turns(run)} == {"end_turn"}
    assert run.score["n_guessed"] == 4
    assert all(r.tool_protocol == "json" for r in requests_seen)
    state = agent_state(run, 2)
    assert "JSON array of tool calls" in state["system"]
    msgs = messages(state)
    assert not any(m.get("tool_calls") for m in msgs) and not any(m["role"] == "tool" for m in msgs)
    results = [m for m in msgs if m["role"] == "user" and isinstance(m["content"], str)]
    assert results and all(m["content"].startswith("[tool results]") for m in results)


def test_json_parse_error_gets_one_retry(tmp_path):
    ok = llm_agent_experiment(2, agent_kw={"model": "fake:tests.helpers:script_bad_json_then_ok",
                                           "tool_protocol": "json"})
    run = ok.run(seed=1, max_rounds=1, out=tmp_path / "ok")
    assert {t["yield_kind"] for t in turns(run)} == {"end_turn"}
    assert all(t["usage"]["inference_calls"] == 2 for t in turns(run))
    msgs = messages(agent_state(run, 1))
    assert "could not parse your tool calls" in msgs[2]["content"]

    bad = llm_agent_experiment(2, agent_kw={"model": "fake:tests.helpers:script_always_bad_json",
                                            "tool_protocol": "json"})
    run = bad.run(seed=1, max_rounds=1, out=tmp_path / "bad")
    assert {t["yield_kind"] for t in turns(run)} == {"no_tool"}
    assert all(t["usage"]["inference_calls"] == 2 for t in turns(run))


def test_parse_tool_json_is_tolerant():
    assert parse_tool_json("I will wait.") is None
    assert parse_tool_json('```json\n[{"name": "end_turn", "args": {}}]\n```') == [("end_turn", {})]
    assert parse_tool_json('<think>[maybe]</think> {"name": "guess", "arguments": "{\\"candidate\\": \\"A\\"}"}') \
        == [("guess", {"candidate": "A"})]
    assert parse_tool_json('ok: [{"name": "post", "parameters": {"text": "x"}}, {"name": "end_turn"}]') \
        == [("post", {"text": "x"}), ("end_turn", {})]
    assert parse_tool_json("[]") == []
    for bad in ("[{oops", '[{"args": {}}]', '[{"name": "x", "args": [1]}]', "[1, 2]"):
        with pytest.raises(ToolJsonError):
            parse_tool_json(bad)


def test_end_turn_first_finishes_the_response_then_returns(tmp_path):
    exp = llm_agent_experiment(2, agent_kw={"model": "fake:tests.helpers:script_end_turn_first"})
    run = exp.run(seed=1, max_rounds=1, out=tmp_path)
    assert {t["yield_kind"] for t in turns(run)} == {"end_turn"}
    assert all(t["usage"]["inference_calls"] == 1 for t in turns(run))
    returned = [e for e in run.events if e["type"] == "tool_returned"]
    assert [r["result"]["error"] for r in returned if r["agent"] == "a000"] == [None, "turn_ended"]
    assert run.score["n_guessed"] == 0
    msgs = messages(agent_state(run, 1))
    well_formed(msgs)
    assert json.loads(msgs[-1]["content"])["error"] == "turn_ended"


def test_own_max_calls_and_runner_cap(tmp_path):
    own = llm_agent_experiment(2, agent_kw={"model": "fake:tests.helpers:script_status_forever",
                                            "max_calls": 3})
    run = own.run(seed=1, max_rounds=1, out=tmp_path / "own")
    assert {t["yield_kind"] for t in turns(run)} == {"no_tool"}
    assert all(t["usage"]["inference_calls"] == 3 and t["usage"]["calls"] == 3 for t in turns(run))

    cap = llm_agent_experiment(2, agent_kw={"model": "fake:tests.helpers:script_status_forever"})
    run = cap.run(seed=1, max_rounds=2, out=tmp_path / "cap", max_calls_per_turn=4)
    assert {t["yield_kind"] for t in turns(run)} == {"cap"}
    msgs = messages(agent_state(run, 2))
    well_formed(msgs)  # the call cut off by the cap still has a result
    assert "cut off" in msgs[-1]["content"]


def test_unparseable_native_args_are_not_executed(tmp_path):
    exp = llm_agent_experiment(1, agent_kw={"model": "fake:tests.helpers:script_raw_args"})
    run = exp.run(seed=1, max_rounds=1, out=tmp_path)
    assert [e["tool"] for e in run.events if e["type"] == "tool_called"] == ["end_turn"]
    msgs = messages(agent_state(run, 1))
    well_formed(msgs)
    assert "not a JSON object" in json.loads(msgs[2]["content"])["error"]


def test_window_memory_keeps_the_last_rounds(tmp_path, requests_seen):
    exp = llm_agent_experiment(2, agent_kw={"memory": "window", "window_rounds": 2})
    run = exp.run(seed=1, max_rounds=4, out=tmp_path)
    assert [r["round"] for r in agent_state(run, 4)["rounds"]] == [3, 4]
    assert [r["round"] for r in agent_state(run, 1)["rounds"]] == [1]
    for req in requests_seen:
        rounds = [m.content[0].text for m in req.messages
                  if m.role == "user" and not isinstance(m.content, str)]
        assert len(rounds) <= 2 and req.messages[0].role == "system"
    last = requests_seen[-1]
    assert [m.content[0].text for m in last.messages if m.role == "user"] == ["Round 3.", "Round 4."]
    full = llm_agent_experiment(2).run(seed=1, max_rounds=4, out=tmp_path / "full")
    assert [r["round"] for r in agent_state(full, 4)["rounds"]] == [1, 2, 3, 4]


def test_images_pass_through(tmp_path, requests_seen):
    run = llm_agent_experiment(2, world=ImageFlagGame()).run(seed=1, max_rounds=1, out=tmp_path)
    assert run.end_reason == "max_rounds"
    first = requests_seen[0].messages[1]
    assert [p.type for p in first.content] == ["text", "text", "image"]
    assert first.content[2].image_png_b64.startswith("iVBOR")


def test_snapshot_restore_mid_run_and_fork(tmp_path):
    exp = llm_agent_experiment(4)
    run = exp.run(seed=2, max_rounds=4, out=tmp_path)
    store = SnapshotStore(run.dir)
    blob = store.load(store.read(2))["participant:a001"]
    fresh = LLMAgent(model="fake:reader")
    fresh.restore(blob)
    a, b = pickle.loads(fresh.snapshot()), pickle.loads(blob)
    assert a.pop("rng").getstate() == b.pop("rng").getstate() and a == b
    assert [m.model_dump() for m in fresh.probe_context()][1:] == messages(pickle.loads(blob))
    child = run.fork(at_round=2).run(out=tmp_path / "forks")
    parent_rounds = [e for e in logical(run) if e["round"] > 2 and e["type"] not in ("run_started",)]
    child_rounds = [e for e in logical(child) if e["round"] > 2 and e["type"] not in ("run_started",)]
    strip = lambda evs: [{k: v for k, v in e.items() if k != "run"} for e in evs]
    assert strip(child_rounds) == strip(parent_rounds)
    assert Run.load(child.dir).score == run.score


def test_default_prompt_contract(tmp_path):
    from swarmlab.executor import board_schemas

    world = FlagGame()
    tools = world.tool_schemas() + board_schemas(Board())
    text = render_system_prompt(None, agent="a007", role="skeptic", description=world.description(),
                                tools=tools)
    assert "a007" in text and '"skeptic"' in text and "rounds" in text and "`end_turn`" in text
    assert world.description() in text
    for t in tools:
        assert f"- {t.name}: {t.description}" in text
    low = text.lower()
    for banned in ("collaborat", "cooperat", "read the board", "share", "team"):
        assert banned not in low
    assert "board" not in world.description().lower()
    custom = tmp_path / "p.j2"
    custom.write_text("I am {{ agent }} ({{ role }}); tools: {{ tools | map(attribute='name') | join(',') }}")
    assert render_system_prompt(f"file:{custom}", agent="a1", role="r", description="",
                                tools=tools[:1]) == "I am a1 (r); tools: guess"
    assert render_system_prompt("hi {{ agent }}", agent="a2", role="r", description="", tools=[]) == "hi a2"


def test_system_prompt_override_and_params(tmp_path):
    exp = llm_agent_experiment(1, agent_kw={"system_prompt": "Agent {{ agent }}. Tools: {{ tools|length }}."})
    run = exp.run(seed=1, max_rounds=1, out=tmp_path)
    assert agent_state(run, 1)["system"] == "Agent a000. Tools: 6."
    spec = run.spec.participants[0]
    assert spec.type == "llm" and spec.params["system_prompt"].startswith("Agent {{ agent }}")


def test_probe_context_is_a_copy():
    agent = LLMAgent(model="fake:reader")
    agent.system = "sys"
    agent.rounds = [{"round": 1, "messages": [ChatMessage(role="user", content="hi").model_dump(mode="json")]}]
    before = agent.snapshot()
    ctx = agent.probe_context()
    ctx.append(ChatMessage(role="user", content="probe"))
    ctx[1].content = "changed"
    assert agent.snapshot() == before
    assert agent.model_request_defaults() == {"model": "fake:reader", "temperature": None,
                                              "max_tokens": 1024, "thinking_budget": None}


def test_constructor_validation():
    with pytest.raises(ValueError):
        LLMAgent(model="fake:reader", memory="forever")
    with pytest.raises(ValueError):
        LLMAgent(model="fake:reader", tool_protocol="xml")
    with pytest.raises(ValueError):
        LLMAgent(model="fake:reader", max_calls=0)


def test_estimate_honours_own_max_calls():
    exp = llm_agent_experiment(2, agent_kw={"max_calls": 3})
    est = exp.estimate(seed=0, max_rounds=5, calls_per_turn=20)
    assert est["calls"] == 2 * 5 * 3
