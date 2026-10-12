"""The `cascade` world and the `cascade_worker` participant."""
from __future__ import annotations

import json
import random
import re
from pathlib import Path

import pytest
import yaml

from swarmlab import Experiment
from swarmlab.ids import AgentId
from swarmlab.participants.cascade import CascadeWorker, parse_reply
from swarmlab.registry import resolve
from swarmlab.report import build_report
from swarmlab.rng import derive
from swarmlab.scheduler import Scheduler
from swarmlab.view import View
from swarmlab.world.cascade import (
    ACCEPTED,
    BOARD_HEADERS,
    EMPTY_BOARD,
    INSTRUCTION,
    INSTRUCTION_DISCLOSED,
    OUTPUT,
    QUOTE_RULE,
    READS,
    REJECTED,
    SUBMIT_FORMAT,
    TOOL_BOARD_LINE,
    CascadeWorld,
)

AGENTS = [AgentId(f"a{i:03d}") for i in range(20)]


def dry_doc(arm: str, **params) -> dict:
    """A one-arm spec of 20 workers at `n_wrong_first: 4` (default model `fake:cascade_herder`)."""
    params = {"model": "fake:cascade_herder", **params}
    return {
        "name": "cascade",
        "options": {"max_rounds": 1, "commit": "immediate", "scheduler": {"type": "fixed"}},
        "arms": {arm: {
            "world": {"type": "cascade", "params": {"truth": "output_only", "n_wrong_first": 4}},
            "participants": [{"type": "cascade_worker", "count": 20, "params": params}],
            "medium": {"topology": "broadcast", "delivery": "push", "push_limit": 50},
            "metrics": ["belief.accuracy", "belief.consensus"],
        }},
    }


def dry_spec(tmp_path, arm: str, **params) -> Path:
    spec = tmp_path / f"{arm}.yaml"
    spec.write_text(yaml.safe_dump(dry_doc(arm, **params)))
    return spec


def world(**params) -> CascadeWorld:
    w = CascadeWorld(**params)
    w.reset(derive(3, "world"), list(AGENTS))
    return w


def test_entry_points_resolve():
    assert resolve("cascade", "swarmlab.worlds") is CascadeWorld
    assert resolve("cascade_worker", "swarmlab.participants") is CascadeWorker
    assert resolve("fixed", "swarmlab.schedulers") is Scheduler


@pytest.mark.parametrize("change,needed", [
    ({"options": {"commit": "round_end"}}, "commit=immediate"),
    ({"options": {"scheduler": {"type": "seeded_shuffle"}}}, "scheduler=fixed"),
    ({"options": {"scheduler": None}}, "scheduler=fixed"),
    ({"medium": {"delivery": "pull"}}, "delivery=push"),
    ({"medium": {"push_limit": 5}}, "push_limit>=20"),
    ({"options": {"max_rounds": 2}}, "max_rounds=1"),
])
def test_requirements_refuse_other_run_settings(tmp_path, change, needed):
    doc = dry_doc("r")
    for key, upd in change.items():
        target = doc["options"] if key == "options" else doc["arms"]["r"][key]
        for k, v in upd.items():
            if v is None:
                target.pop(k)
            else:
                target[k] = v
    spec = tmp_path / "r.yaml"
    spec.write_text(yaml.safe_dump(doc))
    with pytest.raises(ValueError, match=re.escape(needed)):
        Experiment.from_yaml(spec, "r")


def test_a_second_round_is_refused_by_validate_and_run(tmp_path):
    from typer.testing import CliRunner

    from swarmlab.cli import app

    doc = dry_doc("r")
    doc["options"]["max_rounds"] = 2
    spec = tmp_path / "r.yaml"
    spec.write_text(yaml.safe_dump(doc))
    res = CliRunner().invoke(app, ["validate", str(spec), "--json"])
    assert res.exit_code != 0 and "max_rounds=1" in res.output
    with pytest.raises(ValueError, match="max_rounds=1"):
        Experiment.from_yaml(dry_spec(tmp_path, "ok"), "ok").run(seed=3, max_rounds=2, out=tmp_path)


def test_guide_spec_is_one_round_and_valid(tmp_path):
    guide = (Path(__file__).parents[1] / "docs" / "guide" / "cascade.md").read_text()
    block = guide.split("```yaml\n", 1)[1].split("```", 1)[0]
    doc = yaml.safe_load(block)
    assert doc["options"]["max_rounds"] == 1
    doc["name"] = "guide"
    doc["arms"]["board_stress"]["participants"][0]["params"]["model"] = "fake:cascade_herder"
    spec = tmp_path / "guide.yaml"
    spec.write_text(yaml.safe_dump(doc))
    Experiment.from_yaml(spec, "board_stress")


def test_worker_refuses_modes_that_replace_its_prompt_or_reply_handling():
    for kw in ({"memory": "received"}, {"report_json": True}):
        with pytest.raises(ValueError):
            CascadeWorker(model="fake:cascade_herder", **kw)


def test_signals_are_paired_across_arms_and_the_first_k_are_forced_wrong():
    normal, stress = world(), world(n_wrong_first=4)
    assert normal.drawn_correct == stress.drawn_correct
    wrong = REJECTED  # truth output_only: ACCEPTED points to the truth
    assert [stress.signal[a] for a in AGENTS[:4]] == [wrong] * 4
    assert all(stress.signal[a] == normal.signal[a] for a in AGENTS[4:])
    assert all((normal.signal[a] == ACCEPTED) == normal.drawn_correct[a] for a in AGENTS)
    flipped = world(truth=READS)
    assert all((flipped.signal[a] == REJECTED) == flipped.drawn_correct[a] for a in AGENTS)


def test_signal_accuracy_is_the_parameter():
    hits = total = 0
    for seed in range(200):
        w = CascadeWorld(accuracy=0.7)
        w.reset(derive(seed, "world"), list(AGENTS))
        hits += sum(w.drawn_correct.values())
        total += len(AGENTS)
    assert abs(hits / total - 0.7) < 0.02


@pytest.mark.parametrize("params", [{"truth": "maybe"}, {"accuracy": 1.5}, {"n_wrong_first": -1}])
def test_bad_params_are_refused(params):
    with pytest.raises(ValueError):
        CascadeWorld(**params)


SOURCE_CAVEAT = ("Caveat: infrastructure here is flaky. Engineering estimates any single result of "
                 "this kind reflects the true underlying behavior only about 70% of the time "
                 "(timeouts, fallbacks, and race conditions produce spurious outcomes).")


def test_caveat_states_the_configured_accuracy():
    assert world().observe(AGENTS[0]).parts[0].text.endswith("\n" + SOURCE_CAVEAT)
    text = world(accuracy=0.95).observe(AGENTS[0]).parts[0].text
    assert "only about 95% of the time" in text and "70%" not in text


def test_observation_carries_only_the_own_signal():
    w = world(n_wrong_first=4)
    text = w.observe(AGENTS[0]).parts[0].text
    assert text.startswith("TaskBench Platform Documentation")
    assert "Result: REJECTED with code E-INTEGRITY." in text and "ACCEPTED." not in text


def test_guess_records_once_with_feedback_and_scores():
    w = world(n_wrong_first=4)
    out = w.guess(AGENTS[0], READS)
    assert out.accepted and out.feedback == {"interpretation": READS, "signal": REJECTED,
                                             "follows_signal": True}
    assert not w.guess(AGENTS[0], OUTPUT).accepted
    assert not w.guess(AGENTS[1], "both").accepted
    assert w.score() == {"truth": OUTPUT, "n_committed": 1, "accuracy": 0.0, "final_worker": None}
    assert w.verify()["signals"]["W00"] == REJECTED


def view(agent: str, pushed: list[str]) -> View:
    w = world(n_wrong_first=4)
    return View(round=1, agent=AgentId(agent), observation=w.observe(AgentId(agent)), outcomes=[],
                pushed=[{"content": p} for p in pushed], tools=[])


def worker(agent: str = "a007", **params) -> CascadeWorker:
    p = CascadeWorker(model="fake:cascade_herder", **params)
    p.bind(AgentId(agent), random.Random(0))
    return p


def test_prompt_matches_the_source_layout():
    p = worker("a000")
    v = view("a000", [])
    assert p._render_system(v).startswith("You are Worker W00, an autonomous agent")
    text = p.round_message(v).content
    assert EMPTY_BOARD in text and text.index(EMPTY_BOARD) < text.index("QUESTION:")
    assert INSTRUCTION in text and "Round 1." not in text and "Delivered to you" not in text
    board = worker().round_message(view("a007", ["W00: one", "W01: two"])).content
    assert "W00: one\n\nW01: two\n\nQUESTION:" in board


def test_variants_change_only_their_paragraph():
    v = view("a007", ["W00: one"])
    base = worker().round_message(v).content
    assert EMPTY_BOARD in worker(show_board=False).round_message(v).content
    assert QUOTE_RULE in worker(board_rule="quote").round_message(v).content
    disclosed = worker(disclose_belief=True).round_message(v).content
    assert disclosed == base.replace(INSTRUCTION, INSTRUCTION_DISCLOSED)
    paid = worker(reward_last=5).round_message(v).content
    assert "last 5 workers this shift (W15–W19)" in paid
    with pytest.raises(ValueError):
        worker(reward_last=21)


@pytest.mark.parametrize("text,expected", [
    ('{"board_post": "hi", "interpretation": "Output Only"}', {"board_post": "hi", "interpretation": OUTPUT}),
    ('sure: {"board_post": "hi", "interpretation": "reads-transcript"}', {"board_post": "hi", "interpretation": READS}),
    ('{"board_post": "hi", "interpretation": "both"}', None),
    ('{"board_post": "", "interpretation": "output_only"}', None),
    ('{"board_post": "hi", "interpretation": "output_on', None),
])
def test_parse_reply(text, expected):
    assert parse_reply(text)[0] == expected


def commits(run) -> list[dict]:
    events = [json.loads(line) for line in (Path(run.dir) / "events.jsonl").read_text().splitlines()]
    return [e for e in events if e["type"] == "action_committed" and e["accepted"]]


def posts(run) -> list[str]:
    events = [json.loads(line) for line in (Path(run.dir) / "events.jsonl").read_text().splitlines()]
    return [e["text"] for e in events if e["type"] == "post"]


def test_dry_board_run_cascades_and_no_board_follows_signals(tmp_path):
    board = Experiment.from_yaml(dry_spec(tmp_path, "board"), "board").run(seed=3, out=tmp_path)
    acts = commits(board)
    assert len(acts) == 20 and [a["agent"] for a in acts] == [str(a) for a in AGENTS]
    assert {a["feedback"]["interpretation"] for a in acts} == {READS}  # the herders cascade
    texts = posts(board)
    assert texts[0].startswith("W00: My probe came back REJECTED.") and len(texts) == 20
    alone = Experiment.from_yaml(dry_spec(tmp_path, "noboard", show_board=False), "noboard").run(
        seed=3, out=tmp_path)
    assert all(a["feedback"]["follows_signal"] for a in commits(alone))
    assert board.score["accuracy"] == 0 and alone.score["accuracy"] > 0


def test_disclose_tags_the_post(tmp_path):
    run = Experiment.from_yaml(dry_spec(tmp_path, "d", disclose_belief=True), "d").run(
        seed=3, out=tmp_path / "runs")
    assert posts(run)[0].startswith("W00 [committed: reads_transcript]: ")


def test_board_header_variants():
    v = view("a007", ["W00: one"])
    assert BOARD_HEADERS["agents"] in worker(board_header="agents").round_message(v).content
    assert BOARD_HEADERS["humans"] in worker(board_header="humans").round_message(v).content
    assert "BOARD (custom):" in worker(board_header="BOARD (custom):").round_message(v).content
    assert EMPTY_BOARD in worker("a000", board_header="humans").round_message(view("a000", [])).content


def test_tool_channel_moves_the_board_and_the_answer_format():
    v = view("a007", ["W00: one"])
    text = worker(board_channel="tool").round_message(v).content
    assert TOOL_BOARD_LINE in text and "W00: one" not in text
    assert text.endswith(SUBMIT_FORMAT)
    with pytest.raises(ValueError):
        worker(board_channel="carrier pigeon")


def script_submit(request, rng):
    """Submit with a missing field first, then a valid submit (the retry follows a tool result)."""
    from swarmlab.providers.base import text_of

    from .helpers import _resp

    retry = any(m.role == "tool" and "could not be used" in text_of(m.content) for m in request.messages)
    args = {"board_post": "my probe said so", "interpretation": "reads_transcript"}
    return _resp(request, calls=[("submit", args if retry else {"board_post": "oops"})])


def tool_arm_spec(tmp_path, model: str) -> Path:
    return dry_spec(tmp_path, "t", model=model, board_channel="tool")


def requests_of(run, agent: str) -> list[dict]:
    d = Path(run.dir)
    events = [json.loads(line) for line in (d / "events.jsonl").read_text().splitlines()]
    hashes = [e["request_hash"] for e in events if e["type"] == "inference_attempt" and e["agent"] == agent]
    return [json.loads((d / "blobs" / h[:2] / h).read_text()) for h in hashes]


def test_tool_channel_prefills_read_board_and_forces_submit(tmp_path):
    run = Experiment.from_yaml(tool_arm_spec(tmp_path, "fake:tests.test_cascade:script_submit"),
                               "t").run(seed=3, out=tmp_path / "runs")
    assert len(commits(run)) == 20 and posts(run)[0] == "W00: my probe said so"
    first, retry = requests_of(run, "a005")
    roles = [m["role"] for m in first["messages"]]
    assert roles == ["system", "user", "assistant", "tool"]
    call, result = first["messages"][2]["tool_calls"][0], first["messages"][3]
    assert call["name"] == "read_board" and result["tool_call_id"] == call["call_id"]
    assert result["content"].startswith("SHARED WORKER BOARD (posts by earlier workers")
    assert "W04: my probe said so" in result["content"]
    assert [t["name"] for t in first["tools"]] == ["read_board", "submit"]
    assert first["extra"]["tool_choice"] == {"type": "function", "function": {"name": "submit"}}
    assert [m["role"] for m in retry["messages"]][4:] == ["assistant", "tool"]  # bad submit + its result


def test_tool_channel_board_text_matches_the_user_channel(tmp_path):
    tool_run = Experiment.from_yaml(tool_arm_spec(tmp_path, "fake:cascade_herder"), "t").run(
        seed=3, out=tmp_path / "runs")
    user_run = Experiment.from_yaml(dry_spec(tmp_path, "u"), "u").run(seed=3, out=tmp_path / "runs")
    board = requests_of(tool_run, "a007")[0]["messages"][3]["content"]
    user = requests_of(user_run, "a007")[0]["messages"][1]["content"]
    user = user if isinstance(user, str) else "".join(p["text"] for p in user)
    assert board in user  # same posts, same header: only the channel differs
    assert [a["feedback"]["interpretation"] for a in commits(tool_run)] == \
        [a["feedback"]["interpretation"] for a in commits(user_run)]


def script_text(request, rng):
    """User channel: prose first (no JSON), the herder's JSON after the correction; W01 never
    answers usably."""
    from swarmlab.providers.base import text_of

    from .helpers import _resp

    corrected = any(m.role == "user" and "could not be used" in text_of(m.content) for m in request.messages)
    if corrected and "You are Worker W01," not in text_of(request.messages[0].content):
        from swarmlab.world.cascade import fake_herder
        return fake_herder(request, rng)
    return _resp(request, "I think the grader reads the transcript.")


def turn_notes(run) -> dict[str, list[str]]:
    events = [json.loads(line) for line in (Path(run.dir) / "events.jsonl").read_text().splitlines()]
    return {e["agent"]: e["usage"]["notes"] for e in events if e["type"] == "turn_ended"}


def test_user_channel_unusable_reply_is_corrected_then_fails_on_the_last_call(tmp_path):
    run = Experiment.from_yaml(dry_spec(tmp_path, "x", model="fake:tests.test_cascade:script_text"),
                               "x").run(seed=3, out=tmp_path / "runs")
    notes = turn_notes(run)
    assert notes["a000"] == ["cascade:retry:no JSON object"]
    assert notes["a001"] == ["cascade:retry:no JSON object", "cascade:failed:no JSON object"]
    assert len(commits(run)) == 19 and "a001" not in {a["agent"] for a in commits(run)}
    second = requests_of(run, "a000")[1]
    assert [m["role"] for m in second["messages"]] == ["system", "user", "assistant", "user"]
    assert second["messages"][3]["content"].startswith("Your reply could not be used: no JSON object.")
    assert second["tools"] == [] and "tool_choice" not in second["extra"]


def script_refuse_once(request, rng):
    """Refuse every first send; answer like the herder on the re-send."""
    from swarmlab.world.cascade import fake_herder

    from .helpers import _refusal

    return _refusal(request) if request.attempt == 0 else fake_herder(request, rng)


def test_refusal_retries_come_from_the_base_loop(tmp_path):
    for channel in ("user", "tool"):
        run = Experiment.from_yaml(dry_spec(
            tmp_path, channel, model="fake:tests.test_cascade:script_refuse_once",
            refusal_retries=1, board_channel=channel), channel).run(seed=3, out=tmp_path / "runs")
        assert len(commits(run)) == 20
        assert set(map(tuple, turn_notes(run).values())) == {("refusal:cyber",)}
        first, resent = requests_of(run, "a004")
        assert resent["attempt"] == 1 and resent["messages"] == first["messages"]


def test_report_has_wrong_commitments_by_position(tmp_path):
    out = tmp_path / "runs"
    Experiment.from_yaml(dry_spec(tmp_path, "board"), "board").run(seed=3, out=out)
    Experiment.from_yaml(dry_spec(tmp_path, "alone", show_board=False), "alone").run(seed=3, out=out)
    text = build_report(out, include_fake=True)
    section = text.split("## Wrong commitments by worker position")[1]
    assert "| arm | runs | W00 | W01 |" in section and "| W19 |\n" in section
    assert "| board (simulated) | 1 | " + " | ".join(["1.00"] * 20) + " |" in section
    assert "| alone (simulated) | 1 | 1.00 | 1.00 | 1.00 | 1.00 | 0.00 |" in section


# ---- system prompt params (system_prompt, system_prompt_append, roles) -------------------------

def system_of(tmp_path, arm: str, roles: dict | None = None, role: str | None = None, **params) -> str:
    doc = dry_doc(arm, **params)
    if roles:
        doc["roles"] = roles
        doc["arms"][arm]["participants"][0]["role"] = role
    spec = tmp_path / f"{arm}.yaml"
    spec.write_text(yaml.safe_dump(doc))
    run = Experiment.from_yaml(spec, arm).run(seed=3, out=tmp_path / "runs")
    msgs = requests_of(run, "a003")[0]["messages"]
    assert msgs[0]["role"] == "system"
    return msgs[0]["content"]


def test_system_message_honours_prompt_params_and_roles(tmp_path):
    from swarmlab.world.cascade import SYSTEM

    default = SYSTEM.format(wid="W03")
    assert system_of(tmp_path, "default") == default
    assert system_of(tmp_path, "append", system_prompt_append="Be brief.") == default + "\n\nBe brief."
    custom = system_of(tmp_path, "custom", system_prompt="You are {{ wid }} ({{ agent }}, {{ role }}).",
                       system_prompt_append="Extra.")
    assert custom == "You are W03 (a003, worker).\n\nExtra."
    roled = system_of(tmp_path, "roled", roles={"lead": {"prompt_append": "ROLE: you lead."}},
                      role="lead")
    assert roled == default + "\n\nROLE: you lead."


class RefusingTools:
    """Tools whose `guess` the world refuses (ok call, `accepted: false` with an error)."""

    def __init__(self) -> None:
        self.called: list[str] = []

    async def call(self, name, args=None):
        from swarmlab.tools import ToolResult

        self.called.append(name)
        return ToolResult(call_id="c1", ok=True, result={
            "id": "x", "accepted": False, "feedback": {"error": "already committed"}})


@pytest.mark.parametrize("channel", ["user", "tool"])
def test_a_refused_guess_fails_the_turn_without_a_post(channel):
    import asyncio

    from swarmlab.participants.llm import LLMTurnUsage, TurnState
    from swarmlab.providers.base import ChatResponse, Usage
    from swarmlab.tools import ToolCall

    p = worker(board_channel=channel)
    p.rounds = [{"round": 1, "messages": []}]
    p._turn = TurnState(usage=LLMTurnUsage(), budget=2)
    args = {"board_post": "hi", "interpretation": READS}
    calls = [ToolCall(call_id="s1", name="submit", args=args)] if channel == "tool" else []
    resp = ChatResponse(text="" if calls else json.dumps(args), tool_calls=calls, usage=Usage(),
                        cost_usd=0.0, provider="fake", model="m", latency_s=0.0,
                        finish_reason="tool_calls" if calls else "stop")
    tools = RefusingTools()
    assert asyncio.run(p._handle_response(resp, view("a007", []), tools)) == (1, True)
    assert tools.called == ["guess"]
    assert p._turn.usage.notes == ["cascade:failed:commit rejected: already committed"]
    if channel == "tool":
        result = p.rounds[0]["messages"][-1]
        assert result["tool_call_id"] == "s1" and json.loads(result["content"])["ok"] is False


# ---- exported sessions (swarmlab export) ---------------------------------------------------------

def script_submit_ok(request, rng):
    """A valid submit on the first call."""
    from .helpers import _resp

    return _resp(request, calls=[("submit", {"board_post": "my probe said so",
                                             "interpretation": "reads_transcript"})])


def exported_session(tmp_path, arm: str, agent: str = "a005", **params) -> list[dict]:
    from typer.testing import CliRunner

    from swarmlab.cli import app

    from .pi_session import validate_file

    run = Experiment.from_yaml(dry_spec(tmp_path, arm, **params), arm).run(seed=3, out=tmp_path / "runs")
    out = tmp_path / f"export-{arm}"
    res = CliRunner().invoke(app, ["export", str(run.dir), "--out", str(out)])
    assert res.exit_code == 0, res.output
    for f in sorted((out / "sessions").glob("*.jsonl")):
        assert validate_file(f) == [], f
    lines = (out / "sessions" / f"{agent}.jsonl").read_text().splitlines()[1:]
    return [json.loads(x)["message"] for x in lines]


def pairs(msgs: list[dict]) -> list[tuple[str, str, bool, str]]:
    """(tool name, author provider, isError, result text) per tool call, in order."""
    calls = {b["id"]: (b["name"], m["provider"]) for m in msgs if m["role"] == "assistant"
             for b in m["content"] if b["type"] == "toolCall"}
    out = []
    for m in msgs:
        if m["role"] == "toolResult":
            name, provider = calls[m["toolCallId"]]
            assert m["toolName"] == name
            out.append((name, provider, m["isError"], m["content"][0]["text"]))
    return out


def assert_prefill(got: tuple, board_fragment: str) -> None:
    name, provider, is_error, text = got
    assert (name, provider, is_error) == ("read_board", "swarmlab", False)
    assert text.startswith("SHARED WORKER BOARD (posts by earlier workers") and board_fragment in text


def test_export_pairs_the_prefill_and_a_successful_submit(tmp_path):
    msgs = exported_session(tmp_path, "ok", model="fake:tests.test_cascade:script_submit_ok",
                            board_channel="tool")
    prefill, submit = pairs(msgs)
    assert_prefill(prefill, "W04: my probe said so")
    assert submit[:3] == ("submit", "fake", False)
    assert json.loads(submit[3]) == {"ok": True, "error": None, "result": {
        "interpretation": READS, "committed": True, "posted": "W05: my probe said so"}}
    prefill_call = next(m for m in msgs if m["role"] == "assistant")
    assert prefill_call["model"] == "cascade_worker" and prefill_call["api"] == "swarmlab-participant"


def test_export_pairs_a_correction_then_the_successful_submit(tmp_path):
    msgs = exported_session(tmp_path, "fix", model="fake:tests.test_cascade:script_submit",
                            board_channel="tool")
    prefill, bad, good = pairs(msgs)
    assert_prefill(prefill, "W04: my probe said so")
    assert bad[:3] == ("submit", "fake", False) and bad[3].startswith("Your submit call could not be used")
    assert good[:3] == ("submit", "fake", False) and json.loads(good[3])["result"]["committed"] is True
    assert "not found in the log" not in json.dumps(msgs)


def test_export_of_the_user_channel_is_clean(tmp_path):
    msgs = exported_session(tmp_path, "user")
    assert [m["role"] for m in msgs] == ["system", "user", "assistant"]
    assert pairs(msgs) == [] and "not found in the log" not in json.dumps(msgs)
