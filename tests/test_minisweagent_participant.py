"""The mini_swe_agent participant (swarmlab/participants/minisweagent.py): mini-swe-agent 2.4.6's
loop driven step by step against the ProgramBench world's temp-dir local backend, with fake
model scripts defined here (`fake:tests.test_minisweagent_participant:<script>`). No Docker, no
network, no model keys."""
from __future__ import annotations

import asyncio
import copy
import json
import pickle
import random
import re
import subprocess
import sys
from pathlib import Path
from types import SimpleNamespace

import pytest
import yaml

from swarmlab import Experiment, Run
from swarmlab.budget import Gate, Ledger
from swarmlab.medium.board import Board
from swarmlab.participants.base import Participant, TurnUsage
from swarmlab.participants.minisweagent import (
    FORMAT_ERROR_TEMPLATE,
    MSA_VERSION,
    OBSERVATION_TEMPLATE,
    OFFICIAL_BASH,
    WAIT_RESULT,
    MiniSweAgent,
    command_rules,
    load_minisweagent,
    resource_sentence_of,
    rules_block,
)
from swarmlab.providers.anthropic import tool_definition
from swarmlab.providers.base import ChatMessage, ChatRequest
from swarmlab.providers.fake import FakeProvider, _assistant_calls, _response
from swarmlab.providers.openai_compat import build_body
from swarmlab.registry import build, catalog, describe
from swarmlab.roles import Role, assign
from swarmlab.snapshot import SnapshotStore
from swarmlab.spec import Budget
from swarmlab.tools import ToolResult
from swarmlab.view import View, text_observation
from swarmlab.world import programbench as pb
from swarmlab.world.base import Outcome, World
from swarmlab.world.base import tool as world_tool
from swarmlab.world.programbench import SUBMIT_MARKER, ProgramBench, prompt_parts

TASK = "abishekvashok__cmatrix.5c082c6"
HERE = __name__
GUIDE = Path(__file__).resolve().parents[1] / "docs" / "guide" / "programbench.md"
SUBMIT = f"echo {SUBMIT_MARKER}"


def fake(script: str) -> str:
    return f"fake:{HERE}:{script}"


# ---- fake model scripts ---------------------------------------------------------------------------

def _replies(request) -> int:
    """Model replies so far: assistant messages plus format-error messages (dropped replies)."""
    return sum(1 for m in request.messages if m.role == "assistant"
               or (m.role == "user" and ("Tool call error" in str(m.content)
                                         or "output token limit" in str(m.content))))


def _names(request) -> list[str]:
    return [n for m in request.messages for n, _ in _assistant_calls(m)]


SEEN: list[dict] = []


def recording_echo(request, rng):
    """`fake:programbench_echo`, recording the tools and message roles of each request."""
    SEEN.append({"tools": [t.name for t in request.tools],
                 "roles": [m.role for m in request.messages]})
    return pb.fake_echo(request, rng)


SCHEMAS: list[list[dict]] = []


def recording_schemas(request, rng):
    """`fake:programbench_echo`, recording the full tool schemas of each request."""
    SCHEMAS.append([t.model_dump() for t in request.tools])
    return pb.fake_echo(request, rng)


def tries_to_stop(request, rng):
    """`end_turn`, then text only, then the submission: a solo model that tries to stop."""
    k = _replies(request)
    if k == 0:
        return _response(request, rng, [("end_turn", {})])
    if k == 1:
        return _response(request, rng, [], "I am done.")
    return _response(request, rng, [("bash", {"command": SUBMIT})])


def forever(request, rng):
    """A bash command every step, never a submission."""
    return _response(request, rng, [("bash", {"command": f"echo step {_replies(request) + 1}"})])


def submit_fifth(request, rng):
    """Four commands, then the submission."""
    k = _replies(request)
    return _response(request, rng, [("bash", {"command": SUBMIT if k == 4 else f"echo {k}"})])


def chatty(request, rng):
    """Text only, never a tool call."""
    return _response(request, rng, [], "Let me think about this.")


def flaky(request, rng):
    """Cut off, then an unknown tool, then a command, then the submission."""
    k = _replies(request)
    if k == 0:
        resp = _response(request, rng, [], "I was going to")
        return resp.model_copy(update={"finish_reason": "max_tokens"})
    if k == 1:
        return _response(request, rng, [("post", {"text": "hi"})])
    return _response(request, rng, [("bash", {"command": "ls" if k == 2 else SUBMIT})])


CLOCK = [1_000_000.0]


def slow(request, rng):
    """A command per step; each model call takes 1100 s of the patched clock."""
    CLOCK[0] += 1100
    return forever(request, rng)


def interleaved(request, rng):
    """A command, a board read and another command in one reply, then the submission."""
    if _replies(request) == 0:
        return _response(request, rng, [("bash", {"command": "echo first"}),
                                        ("read_board", {}),
                                        ("bash", {"command": "echo second"})])
    return _response(request, rng, [("bash", {"command": SUBMIT})])


def ends_then_runs(request, rng):
    """`end_turn` and then a command in one reply; the submission next turn."""
    if _replies(request) == 0:
        return _response(request, rng, [("end_turn", {}), ("bash", {"command": "ls"})])
    return _response(request, rng, [("bash", {"command": SUBMIT})])


def three_commands(request, rng):
    """Three commands per reply (the executor's cap cuts the first reply)."""
    k = _replies(request)
    if k == 0:
        return _response(request, rng, [("bash", {"command": f"echo {c}"}) for c in "abc"])
    return _response(request, rng, [("bash", {"command": SUBMIT})])


def waiting_lead(request, rng):
    """A command and `wait_for_workers`, then the submission."""
    if "wait_for_workers" not in _names(request):
        return _response(request, rng, [("bash", {"command": "echo plan"}),
                                        ("wait_for_workers", {})])
    return _response(request, rng, [("bash", {"command": SUBMIT})])


def lead(request, rng):
    """Assign worker-1, wait, read the board, submit once the report is there."""
    names = _names(request)
    if "post" not in names:
        return _response(request, rng, [("post", {"channel": "worker-1",
                                                  "text": "Write compile.sh."}),
                                        ("wait_for_workers", {})])
    reported = any(m.role == "tool" and "worker-1: done" in str(m.content)
                   for m in request.messages)
    if reported:
        return _response(request, rng, [("bash", {"command": SUBMIT})])
    if names[-1] == "read_board":
        return _response(request, rng, [("wait_for_workers", {})])
    return _response(request, rng, [("read_board", {})])


def worker(request, rng):
    """Build compile.sh, then report to the lead."""
    if "bash" not in _names(request):
        return _response(request, rng, [("bash", {"command": pb.FAKE_COMPILE})])
    return _response(request, rng, [("post", {"channel": "lead", "text": "worker-1: done"})])


class Poster(Participant):
    """Posts `texts` on `channel` in the given rounds, then ends its turn."""

    def __init__(self, texts: list[str], rounds: list[int], channel: str = "main") -> None:
        self.texts, self.rounds, self.channel = list(texts), list(rounds), channel

    async def turn(self, view, tools) -> TurnUsage:
        if view.round in self.rounds:
            for text in self.texts:
                await tools.call("post", {"channel": self.channel, "text": text})
        await tools.call("end_turn", {})
        return TurnUsage()


# ---- helpers ----------------------------------------------------------------------------------

def world(tmp_path: Path, **kw) -> ProgramBench:
    return ProgramBench(TASK, workspace=str(tmp_path / "ws"), user=None, lockdown=False,
                        check_task=False, **kw)


def solo(tmp_path: Path, model: str, *, rounds: int = 1, cap: int = 1000, world_kw=None,
         role: str | None = None, **params) -> Run:
    p = MiniSweAgent(model=model, **params)
    roles = {}
    if role:
        assign(p, role)
        roles = {role: Role(name=role, tools=["bash"])}
    exp = Experiment(name="msa", world=world(tmp_path, **(world_kw or {})), participants=[p],
                     roles=roles, options={"commit": "immediate", "max_calls_per_turn": cap})
    return exp.run(seed=0, max_rounds=rounds, out=tmp_path / "runs")


def state(run: Run, agent: str = "a000", round: int | None = None) -> dict:
    store = SnapshotStore(run.dir)
    manifest = store.latest() if round is None else store.read(round)
    return pickle.loads(store.load(manifest)[f"participant:{agent}"])


def roles_of(st: dict) -> list[str]:
    return [m["role"] for m in st["messages"]]


def tools_of(st: dict) -> list[dict]:
    return [m for m in st["messages"] if m["role"] == "tool"]


def turns(run: Run, agent: str = "a000") -> list[dict]:
    return [e for e in run.events if e["type"] == "turn_ended" and e["agent"] == agent]


def inference_rounds(run: Run, agent: str) -> list[int]:
    return [e.round for e in run.events_all if e.type == "inference_attempt" and e.agent == agent]


# ---- the solo episode ---------------------------------------------------------------------------

def test_solo_episode_transcript_has_the_baseline_shape(tmp_path):
    SEEN.clear()
    run = solo(tmp_path, fake("recording_echo"), role="solo", world_kw={"cpus": 8, "memory_gb": 32})
    assert (run.meta["end_reason"], run.score["submitted"], run.score["steps"]) == \
        ("terminal", True, 3)
    st = state(run)
    assert roles_of(st) == ["system", "user", "assistant", "tool", "assistant", "tool",
                            "assistant", "exit"]
    system, instance = prompt_parts()
    msgs = st["messages"]
    assert msgs[0]["content"] == system
    assert msgs[1]["content"] == f"{instance}\n\nThis machine has 8 CPUs and 32 GB of memory."
    w = world(tmp_path, cpus=8, memory_gb=32)
    w.reset(random.Random(0), ["a000"])
    assert f"{msgs[0]['content']}\n\n{msgs[1]['content']}" == w.description()
    call = msgs[2]["tool_calls"][0]
    assert call["function"] == {"name": "bash", "arguments": json.dumps({"command": "ls"})}
    assert msgs[3] == {"role": "tool", "tool_call_id": call["id"],
                       "content": "<returncode>0</returncode>\n<output>\n</output>\n",
                       "extra": {"raw_output": "", "returncode": 0, "exception_info": ""}}
    assert msgs[5]["content"].startswith("<returncode>0</returncode>\n<output>\ncompile.sh\n")
    assert msgs[2]["extra"]["actions"] == [{"command": "ls", "tool_call_id": call["id"]}]
    assert msgs[2]["extra"]["cost"] > 0 and "timestamp" not in msgs[2]["extra"]
    assert msgs[-1] == {"role": "exit", "content": "",
                        "extra": {"exit_status": "Submitted", "submission": ""}}
    assert (st["n_calls"], st["finished"]) == (3, True)
    # what the model was sent: the role's tools, mini-swe-agent's messages without `exit`
    assert [s["tools"] for s in SEEN] == [["bash"]] * 3
    assert SEEN[1]["roles"] == ["system", "user", "assistant", "tool"]
    (ended,) = turns(run)
    assert ended["usage"]["notes"] == ["exit:Submitted"]
    assert ended["usage"]["calls"] == 3 and ended["usage"]["inference_calls"] == 3
    assert Run.load(run.dir).replay()["score"] == run.score


def test_step_limit_warning_and_exit(tmp_path):
    run = solo(tmp_path, fake("forever"), step_limit=22, world_kw={"step_limit": 22})
    st = state(run)
    assert st["n_calls"] == 22 and roles_of(st)[-1] == "exit"
    assert st["messages"][-1]["extra"]["exit_status"] == "LimitsExceeded"
    obs = [m["content"] for m in tools_of(st)]
    assert len(obs) == 22
    assert all("<IMPORTANT>" not in o for o in obs[:2])
    assert "You are now 19 steps away from reaching your limit." in obs[2]
    assert "You are now 0 steps away" in obs[21]
    assert run.meta["end_reason"] == "max_rounds" and not run.score["submitted"]


def test_wall_limit_warning_and_exit(tmp_path, monkeypatch):
    import swarmlab.participants.minisweagent as msa_mod
    from swarmlab.vendor.minisweagent import default

    clock = SimpleNamespace(time=lambda: CLOCK[0])
    monkeypatch.setattr(default, "time", clock)
    monkeypatch.setattr(msa_mod, "time", clock)
    CLOCK[0] = 1_000_000.0
    run = solo(tmp_path, fake("slow"), wall_limit_s=3600)
    st = state(run)
    assert st["messages"][-1]["extra"]["exit_status"] == "TimeExceeded"
    assert st["n_calls"] == 4 and st["elapsed_s"] == 4400
    obs = [m["content"] for m in tools_of(st)]
    assert all("running low on time" not in o for o in obs[:2])
    assert "You have approximately 5 minutes remaining before timeout." in obs[2]


def test_three_format_errors_end_the_episode(tmp_path):
    run = solo(tmp_path, fake("chatty"))
    st = state(run)
    assert roles_of(st) == ["system", "user", "user", "user", "user", "exit"]
    assert st["messages"][-1]["extra"]["exit_status"] == "RepeatedFormatError"
    error = st["messages"][2]["content"]
    assert error.startswith("Tool call error:\n\n<error>\nNo tool calls found in the response.")
    assert st["n_calls"] == 3 and st["cost"] > 0  # format-error replies are billed, as in run()
    assert st["messages"][2]["extra"]["interrupt_type"] == "FormatError"


def test_format_errors_are_the_baseline_messages_and_reset_on_a_clean_step(tmp_path):
    run = solo(tmp_path, fake("flaky"), role="solo")
    st = state(run)
    assert roles_of(st) == ["system", "user", "user", "user", "assistant", "tool", "assistant",
                            "exit"]
    cut, unknown = st["messages"][2]["content"], st["messages"][3]["content"]
    assert cut.startswith("Your previous response reached the output token limit "
                          "(finish_reason=length)")
    # post is not offered in the solo role; the baseline's parser words it this way
    assert "<error>\nUnknown tool 'post'.Missing 'command' argument in bash tool call.\n</error>" \
        in unknown
    assert st["n_consecutive_format_errors"] == 0
    assert st["messages"][-1]["extra"]["exit_status"] == "Submitted"


def _official_bash_tool(monkeypatch) -> dict:
    """mini-swe-agent 2.4.6's `BASH_TOOL`: the installed real package's when present, else the
    vendored copy's."""
    monkeypatch.setenv("MSWEA_SILENT_STARTUP", "1")
    try:
        from minisweagent.models.utils.actions_toolcall import BASH_TOOL
    except ImportError:
        from swarmlab.vendor.minisweagent.actions_toolcall import BASH_TOOL
    return BASH_TOOL


@pytest.mark.parametrize("shape", ["bash role", "board, no role"])
def test_solo_offers_exactly_the_official_bash_tool(tmp_path, monkeypatch, shape):
    # the executor's View lists end_turn (and, with a board and no role, read_board and post)
    SCHEMAS.clear()
    if shape == "bash role":
        run = solo(tmp_path, fake("recording_schemas"), role="solo")
    else:
        run = _team(tmp_path, [MiniSweAgent(model=fake("recording_schemas"))], rounds=1,
                    cap=1000)
    assert run.score["submitted"] and len(SCHEMAS) == 3
    official = _official_bash_tool(monkeypatch)
    for tools in SCHEMAS:
        assert [{"type": "function", "function": t} for t in tools] == [official]
    called = {e["tool"] for e in run.events if e["type"] == "tool_called"}
    assert called == {"bash"}
    # and the adapters send it as the baseline does: unchanged, and on Anthropic not strict
    fn = official["function"]
    assert tool_definition(OFFICIAL_BASH) == {"name": fn["name"], "description": fn["description"],
                                              "input_schema": fn["parameters"]}
    request = ChatRequest(model="hf:m", messages=[ChatMessage(role="user", content="hi")],
                          tools=[OFFICIAL_BASH])
    assert build_body(request)["tools"] == [official]


def test_official_bash_is_the_installed_mini_swe_agent_bash_tool(monkeypatch):
    # A cross-check against the real package, which swarmlab does not install: it runs when a
    # developer has mini-swe-agent 2.4.6 installed. (The vendored copy is checked by hash.)
    monkeypatch.setenv("MSWEA_SILENT_STARTUP", "1")
    pytest.importorskip("minisweagent")
    from minisweagent.models.utils.actions_toolcall import BASH_TOOL

    assert {"type": "function", "function": OFFICIAL_BASH.model_dump()} == BASH_TOOL


def test_solo_end_turn_or_no_tool_call_is_a_format_error_and_the_episode_goes_on(tmp_path):
    run = solo(tmp_path, fake("tries_to_stop"), role="solo")
    st = state(run)
    assert roles_of(st) == ["system", "user", "user", "user", "assistant", "exit"]
    unknown, chat = st["messages"][2]["content"], st["messages"][3]["content"]
    assert unknown.startswith("Tool call error:\n\n<error>\nUnknown tool 'end_turn'.Missing "
                              "'command' argument in bash tool call.\n</error>")
    assert chat.startswith("Tool call error:\n\n<error>\nNo tool calls found in the response.")
    assert st["n_calls"] == 3 and st["messages"][-1]["extra"]["exit_status"] == "Submitted"
    assert [e["tool"] for e in run.events if e["type"] == "tool_called"] == ["bash"]
    (ended,) = turns(run)
    assert ended["yield_kind"] != "end_turn" and ended["usage"]["inference_calls"] == 3
    assert ended["usage"]["notes"] == ["exit:Submitted"] and run.score["submitted"]


# ---- several rounds, snapshots, forks -------------------------------------------------------------

def test_steps_per_turn_spread_the_episode_over_rounds(tmp_path):
    run = solo(tmp_path, fake("submit_fifth"), rounds=5, cap=20, steps_per_turn=2)
    assert run.meta["end_reason"] == "terminal" and run.meta["last_round"] == 3
    per_turn = [e["usage"]["inference_calls"] for e in turns(run)]
    assert per_turn == [2, 2, 1]
    assert [e["usage"]["notes"] for e in turns(run)] == [[], [], ["exit:Submitted"]]
    first = state(run, round=1)
    assert (first["n_calls"], len(first["messages"]), first["finished"]) == (2, 6, False)
    assert first["elapsed_s"] > 0
    final = state(run)
    assert final["n_calls"] == 5 and final["messages"][:6] == first["messages"]
    # a fork from round 1 restores the state, rebuilds the agent and ends the same way
    child = run.fork(1).run(out=tmp_path / "forks")
    assert child.meta["end_reason"] == "terminal"
    assert state(child)["messages"] == final["messages"]
    assert Run.load(run.dir).replay()["score"] == run.score


def test_snapshot_holds_plain_state_and_round_trips(tmp_path):
    run = solo(tmp_path, fake("submit_fifth"), rounds=1, cap=20, steps_per_turn=2,
               world_kw={"step_limit": 2})
    st = state(run)
    assert set(st) == {"model", "steps_per_turn", "step_limit", "wall_limit_s", "max_tokens",
                       "temperature", "extra", "templates", "idle_until_assigned", "wait_tool",
                       "lead_channel", "worker_channels", "messages", "n_calls", "cost",
                       "n_consecutive_format_errors", "elapsed_s", "finished", "assigned",
                       "delivered_upto", "agent", "rng"}
    p = MiniSweAgent(model=fake("submit_fifth"))
    p.restore(pickle.dumps(st))
    assert p.messages == st["messages"] and p.n_calls == 2 and p.exit_status is None
    again = pickle.loads(p.snapshot())
    assert again["messages"] == st["messages"] and again["elapsed_s"] == st["elapsed_s"]


def test_the_wall_clock_is_rebased_on_restore(monkeypatch):
    import swarmlab.participants.minisweagent as msa_mod

    monkeypatch.setattr(msa_mod, "time", SimpleNamespace(time=lambda: 5000.0))
    p = MiniSweAgent(model="fake:x")
    p.bind("a000", random.Random(0))
    p.messages = [{"role": "system", "content": "s"}, {"role": "user", "content": "u"}]
    p.n_calls, p.cost, p.elapsed_s = 7, 0.5, 1200.0
    msa = load_minisweagent()
    turn = msa_mod._Turn(usage=msa_mod.LLMTurnUsage(), loop=None, tools=None, offered=[])
    agent = p._build(_view(), turn, msa)
    assert (agent._start_time, agent.n_calls, agent.cost) == (3800.0, 7, 0.5)
    assert agent.messages is p.messages


# ---- deliveries ---------------------------------------------------------------------------------

def _team(tmp_path, participants, *, delivery="pull", channels=("main",), rounds=2, cap=20,
          **world_kw) -> Run:
    exp = Experiment(name="msa-team", world=world(tmp_path, **world_kw),
                     participants=participants,
                     medium=Board(delivery=delivery, channels=list(channels)),
                     options={"commit": "immediate", "max_calls_per_turn": cap,
                              "scheduler": {"type": "fixed"}})
    return exp.run(seed=0, max_rounds=rounds, out=tmp_path / "runs")


def test_push_deliveries_are_one_user_message_inserted_once(tmp_path):
    run = _team(tmp_path, [Poster(["assignment one", "assignment two"], [1]),
                           MiniSweAgent(model=fake("forever"), steps_per_turn=1)],
                delivery="push")
    st = state(run, "a001")
    assert roles_of(st) == ["system", "user", "user", "assistant", "tool", "assistant", "tool"]
    delivered = st["messages"][2]["content"]
    assert re.fullmatch(r"Delivered to you:\n- \S+ \(round 1\): assignment one\n"
                        r"- \S+ \(round 1\): assignment two", delivered)
    assert st["delivered_upto"][0] == 1


def test_pull_board_reads_run_through_the_adapter_in_call_order(tmp_path):
    run = _team(tmp_path, [Poster(["assignment one"], [1]),
                           MiniSweAgent(model=fake("interleaved"), steps_per_turn=1,
                                        templates="worker")])
    st = state(run, "a001")
    assert not any(str(m.get("content", "")).startswith("Delivered") for m in st["messages"])
    reply = st["messages"][2]
    assert [c["function"]["name"] for c in reply["tool_calls"]] == ["bash", "read_board", "bash"]
    first, read, second = st["messages"][3:6]
    assert [m["tool_call_id"] for m in (first, read, second)] == \
        [c["id"] for c in reply["tool_calls"]]
    assert "first" in first["content"] and "second" in second["content"]
    payload = json.loads(read["content"])
    assert payload["ok"] and [i["content"] for i in payload["result"]["items"]] == \
        ["assignment one"]
    called = [e["tool"] for e in run.events if e["type"] == "tool_called" and e["agent"] == "a001"]
    assert called[:3] == ["bash", "read_board", "bash"]
    assert st["messages"][-1]["extra"]["exit_status"] == "Submitted"


def test_idle_worker_makes_no_model_call_until_assigned_and_after_its_report(tmp_path):
    run = _team(tmp_path, [Poster(["build it"], [2]),
                           MiniSweAgent(model=fake("worker"), steps_per_turn=10,
                                        templates="worker", idle_until_assigned=True)],
                channels=("main", "lead"), rounds=3)
    assert sorted(set(inference_rounds(run, "a001"))) == [2]
    assert len(inference_rounds(run, "a001")) == 2
    ended = turns(run, "a001")
    assert [e["usage"]["notes"] for e in ended] == [["idle"], [], ["idle"]]
    assert [e["usage"].get("inference_calls", 0) for e in ended] == [0, 2, 0]
    called = [(e["round"], e["tool"]) for e in run.events
              if e["type"] == "tool_called" and e["agent"] == "a001"]
    assert called == [(1, "read_board"), (2, "read_board"), (2, "bash"), (2, "post"),
                      (3, "read_board")]
    st = state(run, "a001")
    assert st["assigned"] is False and st["finished"] is False
    assert st["messages"][2]["content"].endswith(": build it")


def test_wait_for_workers_ends_the_turn_and_posts_nothing(tmp_path):
    run = solo(tmp_path, fake("waiting_lead"), rounds=3, cap=20, steps_per_turn=10,
               templates="lead", wait_tool=True)
    assert [e["usage"].get("inference_calls") for e in turns(run)] == [1, 1]
    assert turns(run)[0]["yield_kind"] == "no_tool"
    st = state(run)
    wait = tools_of(st)[1]
    assert wait["content"] == WAIT_RESULT
    assert not [e for e in run.events if e["type"] == "post"]
    assert st["messages"][-1]["extra"]["exit_status"] == "Submitted"


def test_team_shape_from_the_guide(tmp_path):
    spec = _guide_yaml("programbench-team")
    arm = spec["arms"]["team"]
    arm["world"]["params"].update(workspace=str(tmp_path / "ws"), user=None, lockdown=False,
                                  check_task=False, backend="local")
    arm["participants"][0]["params"]["model"] = fake("lead")
    arm["participants"][1]["params"]["model"] = fake("worker")
    spec["options"]["max_rounds"] = 4
    path = tmp_path / "team.yaml"
    path.write_text(yaml.safe_dump(spec))
    run = Experiment.from_yaml(path, "team").run(seed=0, out=tmp_path / "runs")
    assert run.meta["end_reason"] == "terminal" and run.meta["last_round"] == 2
    assert state(run, "a000")["messages"][-1]["extra"]["exit_status"] == "Submitted"
    assert inference_rounds(run, "a002") == inference_rounds(run, "a003") == []
    assert sorted(set(inference_rounds(run, "a001"))) == [1]
    posts = [(e["agent"], e["channel"]) for e in run.events if e["type"] == "post"]
    assert posts == [("a000", "worker-1"), ("a001", "lead")]
    lead_msgs = state(run, "a000")["messages"]
    assert "You lead a team of 3 workers." in lead_msgs[1]["content"]
    assert any(m["role"] == "tool" and m["content"] == WAIT_RESULT for m in lead_msgs)


# ---- refused calls and the executor's cap ---------------------------------------------------------

def test_a_command_after_end_turn_is_refused_and_the_episode_goes_on(tmp_path):
    # end_turn is offered with the lead/worker templates, not with solo
    run = solo(tmp_path, fake("ends_then_runs"), rounds=2, cap=20, templates="lead")
    st = state(run)
    ended, refused = tools_of(st)[:2]
    assert json.loads(ended["content"]) == {"ok": True, "result": {}, "error": None}
    assert refused["content"].startswith(
        "<exception>swarmlab refused the command: turn_ended</exception>\n"
        "<returncode>-1</returncode>")
    assert turns(run)[0]["yield_kind"] == "end_turn"
    assert st["messages"][-1]["extra"]["exit_status"] == "Submitted"


def test_world_wall_limit_ends_the_episode_as_time_exceeded(tmp_path):
    run = solo(tmp_path, fake("forever"), world_kw={"wall_limit_s": 1e-9})
    st = state(run)
    assert st["messages"][-1] == {"role": "exit", "content": "TimeExceeded",
                                  "extra": {"exit_status": "TimeExceeded", "submission": ""}}


def test_another_agents_submission_ends_the_episode(tmp_path):
    run = _team(tmp_path, [MiniSweAgent(model="fake:programbench_echo"),
                           MiniSweAgent(model=fake("forever"))], rounds=1, cap=10,
                step_limit=10)
    st = state(run, "a001")
    assert st["n_calls"] == 1
    assert st["messages"][-1] == {"role": "exit", "content": "",
                                  "extra": {"exit_status": "Submitted", "submission": ""}}


def test_the_executor_cap_ends_the_turn_with_a_well_formed_history(tmp_path):
    run = solo(tmp_path, fake("three_commands"), rounds=2, cap=2)
    assert turns(run)[0]["yield_kind"] == "cap"
    st = state(run, round=1)
    reply = st["messages"][2]
    a, b, c = st["messages"][3:6]
    assert [m["tool_call_id"] for m in (a, b, c)] == [x["id"] for x in reply["tool_calls"]]
    assert "<output>\na\n</output>" in a["content"] and "<output>\nb\n</output>" in b["content"]
    assert c["content"].startswith("<exception>action was not executed</exception>")
    assert state(run)["messages"][-1]["extra"]["exit_status"] == "Submitted"


# ---- a cancelled turn ---------------------------------------------------------------------------

class _BlockingTools:
    """An `AgentTools` stand-in: `infer` holds a real gate reservation; the inference
    (`block="infer"`) or the second command of the reply (`block="bash"`) blocks until `release`."""

    def __init__(self, block: str) -> None:
        provider = FakeProvider(pricing={"*": (1.0, 5.0, 0.1)})
        self.gate = Gate({"fake": provider}, Ledger(), Budget(hard_usd=1.0))
        self.block = block
        self.started, self.release = asyncio.Event(), asyncio.Event()
        self.cancelled: list[str] = []
        self.returned: list[str] = []

    async def _wait(self, what: str) -> None:
        self.started.set()
        try:
            await self.release.wait()
        except asyncio.CancelledError:
            self.cancelled.append(what)
            raise

    async def infer(self, request, *, category="swarm"):
        async with self.gate.admit(request, category) as res:
            self.gate.dispatch(res)
            if self.block == "infer":
                await self._wait("infer")
            res.charge(0.0)
        self.returned.append("infer")
        return _response(request, random.Random(0), [("bash", {"command": "echo one"}),
                                                     ("bash", {"command": "echo two"})])

    async def call(self, name, args=None):
        if self.block == "bash" and args["command"] == "echo two":
            await self._wait("bash")
        self.returned.append(name)
        return ToolResult(call_id="c1", ok=True, result={"accepted": True, "feedback": {
            "output": args["command"][len("echo "):] + "\n", "returncode": 0,
            "exception_info": ""}})


@pytest.mark.parametrize("block", ["infer", "bash"])
async def test_a_cancelled_turn_cancels_and_settles_the_bridge(block):
    p = MiniSweAgent(model="fake:x")
    p.bind("a000", random.Random(0))
    tools = _BlockingTools(block)
    seen: dict = {}
    steps = p._steps

    def recorded_steps(agent, turn, msa):
        seen["turn"] = turn
        try:
            steps(agent, turn, msa)
        finally:
            seen["returned"] = True

    p._steps = recorded_steps
    task = asyncio.create_task(p.turn(_view(), tools))
    await asyncio.wait_for(tools.started.wait(), 10)
    task.cancel()
    with pytest.raises(asyncio.CancelledError):
        await task
    # the bridge operation was cancelled, the step thread has returned, nothing is outstanding
    assert tools.cancelled == [block]
    assert seen["returned"] and seen["turn"].abort.is_set() and seen["turn"].task is None
    assert tools.gate.ledger.reserved == 0
    saved = copy.deepcopy(p.messages)
    tools.release.set()
    await asyncio.sleep(0.1)
    assert p.messages == saved
    if block == "infer":
        assert roles_of({"messages": saved}) == ["system", "user"]
        assert tools.returned == [] and p.n_calls == 1
    else:
        # repaired: every call of the interrupted reply has its result, in call order
        assert roles_of({"messages": saved}) == ["system", "user", "assistant", "tool", "tool"]
        calls = [c["id"] for c in saved[2]["tool_calls"]]
        assert [m["tool_call_id"] for m in saved[3:]] == calls
        assert "<output>\none\n</output>" in saved[3]["content"]
        assert saved[4]["content"].startswith("<exception>action was not executed</exception>")
        assert tools.returned == ["infer", "bash"]
    assert not p.finished


class _CleanupWorld(World):
    """An async `bash` that never completes on its own; its cancellation cleanup awaits."""

    started: asyncio.Event | None = None  # set per test

    def reset(self, rng, agents):
        self.busy = 0
        self.cleaned = False
        self.cleanup_interrupted = False

    def observe(self, agent):
        return text_observation("offline cancellation test")

    @world_tool("bash", "Stub command", {"command": "string"})
    async def bash(self, agent, command):
        self.busy += 1
        type(self).started.set()
        try:
            await asyncio.Event().wait()
        finally:
            try:
                await asyncio.sleep(0)    # the undo awaits (INTERFACE §8, Async tools)
                await asyncio.sleep(0)
                self.busy -= 1
                self.cleaned = True
            except asyncio.CancelledError:
                self.cleanup_interrupted = True
                raise
        return Outcome(accepted=True)

    def score(self):
        return {"busy": self.busy, "cleaned": self.cleaned,
                "cleanup_interrupted": self.cleanup_interrupted}


class _FinishTick(Participant):
    """Ends its turn once the other agent's command is running: that tick ends the run."""

    async def turn(self, view, tools):
        await _CleanupWorld.started.wait()
        await tools.call("end_turn", {})
        return TurnUsage(calls=1)


def test_run_end_cancels_the_bridge_once_and_awaits_the_tools_cleanup(tmp_path):
    """The final tick cancels the executor's pending bridge operation, then the participant's
    loop; the participant must not cancel the bridge task a second time (that interrupted the
    tool's awaiting cleanup) and must wait for it before its state is saved."""
    _CleanupWorld.started = asyncio.Event()
    run = Experiment(name="double-cancel", world=_CleanupWorld(),
                     participants=[MiniSweAgent(model="fake:programbench_echo"), _FinishTick()],
                     ).run(seed=0, max_rounds=1, commit="async", out=tmp_path / "runs")
    assert run.meta["end_reason"] == "max_rounds"
    assert run.score == {"busy": 0, "cleaned": True, "cleanup_interrupted": False}
    store = SnapshotStore(run.dir)
    latest = store.latest()
    assert latest.round == run.meta["last_round"] == 1
    final = pickle.loads(store.load(latest)["world"])
    assert (final["busy"], final["cleaned"], final["cleanup_interrupted"]) == (0, True, False)
    assert Run.load(run.dir).replay()["score"] == run.score
    # the cancelled turn committed nothing and left no turn events; the closing turn is a001's
    assert not [e for e in run.events if e["type"] == "action_committed"]
    assert [e["agent"] for e in run.events if e["type"] == "turn_ended"] == ["a001"]


# ---- templates ------------------------------------------------------------------------------------

def _view(tools=None, description=None, agent="a000") -> View:
    from swarmlab.executor import END_TURN_SCHEMA
    from swarmlab.tools import ToolSchema

    if tools is None:
        tools = [ToolSchema(name="bash", description="Execute a bash command", parameters={
            "type": "object", "properties": {"command": {"type": "string"}},
            "required": ["command"], "additionalProperties": False}), END_TURN_SCHEMA]
    if description is None:
        description = "x\n\nThis machine has 8 CPUs and 32 GB of memory."
    return View(round=1, agent=agent, observation=text_observation("obs"), outcomes=[],
                pushed=[], tools=tools, description=description)


def _post_tool(channels: list[str]):
    from swarmlab.tools import ToolSchema

    return ToolSchema(name="post", description="Post.", parameters={
        "type": "object", "properties": {"channel": {"type": "string", "enum": channels},
                                         "text": {"type": "string"}},
        "required": ["channel", "text"], "additionalProperties": False})


def test_templates_lead_and_worker():
    lead_p = MiniSweAgent(model="fake:x", templates="lead", wait_tool=True)
    lead_p.bind("a000", random.Random(0))
    view = _view(tools=[_post_tool(["worker-1", "worker-2"])])
    system, instance = prompt_parts()
    assert lead_p._render_system(view) == system
    text = lead_p.turn_message(view).content
    assert text.startswith(instance) and "You lead a team of 2 workers." in text
    assert "`worker-1`, `worker-2`" in text and "`wait_for_workers`" in text
    assert text.endswith("\n\nThis machine has 8 CPUs and 32 GB of memory.")
    no_wait = MiniSweAgent(model="fake:x", templates="lead")
    no_wait.bind("a000", random.Random(0))
    assert "wait_for_workers" not in no_wait.turn_message(view).content
    w = MiniSweAgent(model="fake:x", templates="worker")
    w.bind("a001", random.Random(0))
    wview = _view(tools=[_post_tool(["lead", "worker-1"])])
    wsys, wtext = w._render_system(wview), w.turn_message(wview).content
    assert rules_block(system) in wsys and "## Task context" not in wsys + wtext
    assert "on channel `worker-1`" in wtext and "to channel `lead`" in wtext
    assert command_rules(instance) in wtext and "- Submit your changes" not in wtext


def test_template_helpers_and_custom_templates(tmp_path):
    assert resource_sentence_of("a\n\nThis machine has 8 CPUs.") == "This machine has 8 CPUs."
    assert resource_sentence_of("a\n\nsomething else") == ""
    custom = tmp_path / "t.yaml"
    custom.write_text(yaml.safe_dump({"system_template": "S {{ agent }} {{ step_limit }}",
                                      "instance_template": "I {{ worker_channels }}"}))
    p = MiniSweAgent(model="fake:x", templates=f"file:{custom}", step_limit=7)
    p.bind("a003", random.Random(0))
    view = _view(tools=[_post_tool(["lead", "worker-3"])])
    assert p._render_system(view) == "S a003 7"
    assert p.turn_message(view).content == "I ['worker-3']"
    with pytest.raises(ValueError, match="templates must be one of"):
        MiniSweAgent(model="fake:x", templates="nope")


def test_templates_are_the_installed_mini_swe_agent_model_config(monkeypatch):
    # A cross-check against the real package, which swarmlab does not install (the participant
    # uses the vendored copy): it runs when a developer has mini-swe-agent 2.4.6 installed.
    monkeypatch.setenv("MSWEA_SILENT_STARTUP", "1")
    minisweagent = pytest.importorskip("minisweagent")

    path = Path(minisweagent.__file__).parent / "config" / "benchmarks" / "programbench.yaml"
    cfg = yaml.safe_load(path.read_text())
    assert cfg["model"]["observation_template"] == OBSERVATION_TEMPLATE
    assert cfg["model"]["format_error_template"] == FORMAT_ERROR_TEMPLATE
    assert (cfg["agent"]["step_limit"], cfg["agent"]["wall_time_limit_seconds"]) == (1000, 21600)


# ---- spec, plugins, the vendored copy ---------------------------------------------------------------------

def test_params_hold_only_non_defaults_and_round_trip():
    from swarmlab.spec import canonical_json

    plain = MiniSweAgent(model="anthropic:claude-haiku-4-5")
    explicit = MiniSweAgent(model="anthropic:claude-haiku-4-5", steps_per_turn=None,
                            step_limit=1000, wall_limit_s=21600, max_tokens=8192,
                            temperature=None, extra=None, templates="solo",
                            idle_until_assigned=False, wait_tool=False, lead_channel="lead",
                            worker_channels=None)
    assert plain.params == explicit.params == {"model": "anthropic:claude-haiku-4-5"}
    assert canonical_json(plain.spec()) == canonical_json(explicit.spec())
    team = MiniSweAgent(model="fake:x", steps_per_turn=10, templates="worker",
                        idle_until_assigned=True, worker_channels=["w"])
    assert team.params == {"model": "fake:x", "steps_per_turn": 10, "templates": "worker",
                           "idle_until_assigned": True, "worker_channels": ["w"]}
    rebuilt = build(team.spec(), "swarmlab.participants")
    assert isinstance(rebuilt, MiniSweAgent) and rebuilt.spec() == team.spec()
    assert team.spec()["type"] == "mini_swe_agent"
    assert team.calls_per_turn_cap == 10 and plain.calls_per_turn_cap == 1000


@pytest.mark.parametrize("kw", [{"model": "nocolon"}, {"steps_per_turn": 0},
                                {"step_limit": True}, {"wall_limit_s": 0},
                                {"idle_until_assigned": "yes"}, {"lead_channel": ""},
                                {"worker_channels": "w"}, {"wait_tool": True}])
def test_constructor_rejects_bad_params(kw):
    with pytest.raises(ValueError):
        MiniSweAgent(**{"model": "fake:x", **kw})


def test_plugins_lists_it_with_a_one_line_description():
    rows = {name: (desc, target) for name, desc, target in catalog("participants")}
    desc, target = rows["mini_swe_agent"]
    assert target == "swarmlab.participants.minisweagent:MiniSweAgent"
    assert desc == describe(MiniSweAgent) and "\n" not in desc and len(desc) <= 160


VENDOR = Path(__file__).resolve().parents[1] / "swarmlab" / "vendor" / "minisweagent"


def test_vendored_version_is_msa_version():
    from swarmlab.vendor import minisweagent

    assert minisweagent.__version__ == MSA_VERSION
    assert load_minisweagent().DefaultAgent.__module__ == "swarmlab.vendor.minisweagent.default"


def test_vendored_files_carry_the_header_and_need_no_real_package():
    files = sorted(VENDOR.glob("*.py"))
    assert {f.name for f in files} == {"__init__.py", "exceptions.py", "serialize.py",
                                       "openai_multimodal.py", "actions_toolcall.py", "default.py"}
    for f in files:
        assert f.read_text().startswith("# Vendored from mini-swe-agent 2.4.6 (MIT): minisweagent/"), f
    code = ("import sys; import swarmlab.vendor.minisweagent.default, "
            "swarmlab.vendor.minisweagent.actions_toolcall; "
            "print(sorted(m for m in sys.modules if m.split('.')[0] == 'minisweagent'))")
    out = subprocess.run([sys.executable, "-c", code], capture_output=True, text=True, check=True)
    assert out.stdout.strip() == "[]"


def _guide_yaml(name: str) -> dict:
    for block in re.findall(r"```yaml\n(.*?)```", GUIDE.read_text(), re.DOTALL):
        spec = yaml.safe_load(block)
        if spec.get("name") == name and any(
                p.get("type") == "mini_swe_agent"
                for arm in spec["arms"].values() for p in arm["participants"]):
            return spec
    raise AssertionError(f"no mini_swe_agent spec {name!r} in {GUIDE}")


def test_guide_solo_spec_renders_the_official_prompt(tmp_path):
    from swarmlab.prompts_cmd import render_prompts

    spec = _guide_yaml("programbench-solo")
    path = tmp_path / "solo.yaml"
    path.write_text(yaml.safe_dump(spec))
    (row,) = render_prompts(Experiment.from_yaml(path, "solo"))
    system, instance = prompt_parts()
    assert (row["type"], row["tools"]) == ("mini_swe_agent", ["bash"])
    assert row["system"] == system
    assert row["user"] == f"{instance}\n\nThis machine has 20 CPUs and 60 GB of memory."


def test_render_delivered_drops_the_round_when_turn_based():
    from swarmlab.participants.llm import render_delivered

    items = [{"post_id": "p1", "eligible_round": 3, "content": "hi"},
             {"post_id": "p2", "eligible_round": 4, "content": {"k": 1}}]
    assert render_delivered(items) == \
        'Delivered to you:\n- p1 (round 3): hi\n- p2 (round 4): {"k": 1}'
    assert render_delivered(items, turn_based=True) == \
        'Delivered to you:\n- p1: hi\n- p2: {"k": 1}'


# ---- commit: async: the idle worker and wait_for_workers on the core `wait` ------------------------

def _channel_of(request) -> str:
    """The worker's own channel, from its instance message ("on channel `worker-1`")."""
    for m in request.messages:
        found = re.search(r"on channel `([^`]+)`", str(m.content))
        if found:
            return found.group(1)
    return "main"


def team_worker(request, rng):
    """Write `<channel>.txt` (plus compile.sh for worker-1), then report `<channel>: done`."""
    ch = _channel_of(request)
    if "bash" not in _names(request):
        cmd = f"echo {ch} > {ch}.txt" + (f" && {pb.FAKE_COMPILE}" if ch == "worker-1" else "")
        return _response(request, rng, [("bash", {"command": cmd})])
    return _response(request, rng, [("post", {"channel": "lead", "text": f"{ch}: done"})])


def team_lead(request, rng):
    """Assign both workers and wait; wait again until both reports are in; then submit."""
    if "post" not in _names(request):
        return _response(request, rng, [
            ("post", {"channel": "worker-1", "text": "Write worker-1.txt and compile.sh."}),
            ("post", {"channel": "worker-2", "text": "Write worker-2.txt."}),
            ("wait_for_workers", {})])
    tools = " ".join(str(m.content) for m in request.messages if m.role == "tool")
    if "worker-1: done" in tools and "worker-2: done" in tools:
        return _response(request, rng, [("bash", {"command": SUBMIT})])
    return _response(request, rng, [("wait_for_workers", {})])


class LatePoster(Participant):
    """First turn: sleeps `delay_s`, then posts `text` on `channel`; every turn ends at once."""

    def __init__(self, text: str, channel: str, delay_s: float = 0.3) -> None:
        self.text, self.channel, self.delay_s = text, channel, delay_s
        self.turns = 0

    async def turn(self, view, tools) -> TurnUsage:
        self.turns += 1
        if self.turns == 1:
            await asyncio.sleep(self.delay_s)
            await tools.call("post", {"channel": self.channel, "text": self.text})
        await tools.call("end_turn", {})
        return TurnUsage()


ASYNC_ROLES = {"lead": Role(name="lead", tools=["bash", "post", "read_board", "wait"]),
               "worker": Role(name="worker", tools=["bash", "post", "read_board", "wait"])}


def _async_team(tmp_path, participants, roles, *, channels=("main", "lead"), providers=None,
                topology="broadcast", role_defs=None, **options) -> Run:
    parts = [assign(p, r) for p, r in zip(participants, roles, strict=True)]
    exp = Experiment(name="msa-async", world=world(tmp_path), participants=parts,
                     roles={r: (role_defs or ASYNC_ROLES)[r] for r in set(roles)},
                     medium=Board(delivery="pull", channels=list(channels), topology=topology),
                     providers=providers or {},
                     options={"commit": "async", "max_calls_per_turn": 20, **options})
    return exp.run(seed=0, out=tmp_path / "runs")


def _tool_results(run: Run, agent: str, tool: str) -> list[dict]:
    calls = {e["call_id"] for e in run.events
             if e["type"] == "tool_called" and e["agent"] == agent and e["tool"] == tool}
    return [e["result"]["result"] for e in run.events if e["type"] == "tool_returned"
            and e["call_id"] in calls]


def test_async_idle_worker_waits_with_no_model_call_until_the_post(tmp_path):
    run = _async_team(tmp_path, [LatePoster("build it", "main", delay_s=0.3),
                                 MiniSweAgent(model=fake("worker"), steps_per_turn=10,
                                              templates="worker", idle_until_assigned=True)],
                      ["lead", "worker"], rounds_per_agent=2)
    assert run.meta["end_reason"] == "max_rounds"
    ended = turns(run, "a001")
    assert [e["usage"]["notes"] for e in ended] == [[], ["idle"]]
    assert [e["usage"].get("inference_calls", 0) for e in ended] == [2, 0]
    called = [e["tool"] for e in run.events if e["type"] == "tool_called" and e["agent"] == "a001"]
    assert called == ["wait", "bash", "post", "wait"]
    first, second = _tool_results(run, "a001", "wait")
    assert first["waited_s"] >= 0.2 and not first["timed_out"]
    assert [p["content"] for p in first["posts"]] == ["build it"]
    assert second == {"posts": [], "timed_out": True, "waited_s": second["waited_s"]}
    st = state(run, "a001")
    assert re.fullmatch(r"Delivered to you:\n- \S+: build it", st["messages"][2]["content"])
    assert "When you have no assignment, you wait until the lead sends one." in \
        st["messages"][1]["content"]
    assert st["assigned"] is False and st["elapsed_s"] >= 0.2   # the wait is on its clock


def test_async_offered_tools_hide_wait_and_describe_wait_for_workers_per_mode():
    from swarmlab.executor import WAIT_SCHEMA as CORE_WAIT
    from swarmlab.participants.minisweagent import WAIT_SCHEMA, WAIT_SCHEMA_ASYNC

    p = MiniSweAgent(model="fake:x", templates="lead", wait_tool=True)
    view = _view(tools=[*_view().tools, CORE_WAIT])
    async_view = view.model_copy(update={"turn": 1})
    assert [t.name for t in p.offered_tools(view)] == ["bash", "end_turn", "wait_for_workers"]
    assert p.offered_tools(view)[-1] == WAIT_SCHEMA
    assert p.offered_tools(async_view)[-1] == WAIT_SCHEMA_ASYNC
    assert "turn ends" in WAIT_SCHEMA.description
    assert "returns the messages posted to you" in WAIT_SCHEMA_ASYNC.description
    p.bind("a000", random.Random(0))
    lead_view = _view(tools=[_post_tool(["worker-1"])])
    lockstep = p.turn_message(lead_view).content
    assert "your turn ends after that step" in lockstep
    turn_based = p.turn_message(lead_view.model_copy(update={"turn": 1})).content
    assert "it returns them as its result" in turn_based and "turn ends" not in turn_based


def test_async_team_end_to_end_wait_for_workers_returns_the_reports(tmp_path):
    from typer.testing import CliRunner

    from swarmlab.cli import app
    from swarmlab.providers.fake import FakeProvider

    lead_p = MiniSweAgent(model=fake("team_lead"), templates="lead", steps_per_turn=10,
                          wait_tool=True, worker_channels=["worker-1", "worker-2"])
    run = _async_team(tmp_path, [lead_p,
                                 MiniSweAgent(model=fake("team_worker"), templates="worker",
                                              steps_per_turn=10, idle_until_assigned=True,
                                              worker_channels=["worker-1"]),
                                 MiniSweAgent(model=fake("team_worker"), templates="worker",
                                              steps_per_turn=10, idle_until_assigned=True,
                                              worker_channels=["worker-2"])],
                      ["lead", "worker", "worker"],
                      channels=("lead", "worker-1", "worker-2"), rounds_per_agent=5,
                      # the guide's channels: a room per worker shared with the lead, and `lead`
                      topology={"type": "rooms", "params": {"rooms": {
                          "lead": ["a000", "a001", "a002"], "worker-1": ["a000", "a001"],
                          "worker-2": ["a000", "a002"]}}},
                      role_defs={"lead": Role(name="lead", tools=["bash", "post", "read_board",
                                                                  "wait"],
                                              channels_write=["worker-1", "worker-2"]),
                                 "worker": Role(name="worker", tools=["bash", "post",
                                                                      "read_board", "wait"],
                                                channels_read=["worker-1", "worker-2"])},
                      providers={"fake": FakeProvider(yields=[0, 20])}, snapshot_every=1000)
    assert run.meta["end_reason"] == "terminal"
    lead_turns = turns(run, "a000")
    assert len(lead_turns) == 1                       # wait_for_workers did not end the turn
    st = state(run, "a000")
    assert st["messages"][-1]["extra"]["exit_status"] == "Submitted"
    waits = [m for m in tools_of(st) if m.get("extra", {}).get("tool") == "wait_for_workers"]
    reports = [i["content"] for m in waits if m["content"].startswith("{")
               for i in json.loads(m["content"])["result"]["items"]]
    assert sorted(reports) == ["worker-1: done", "worker-2: done"]
    assert all(m["content"] != WAIT_RESULT for m in waits)
    called = [e["tool"] for e in run.events if e["type"] == "tool_called" and e["agent"] == "a000"]
    assert "read_board" not in called and called.count("wait") == len(waits)
    # the workers did the work in the shared workspace, each from its own assignment
    ws = tmp_path / "ws"
    assert (ws / "worker-1.txt").read_text() == "worker-1\n"
    assert (ws / "worker-2.txt").read_text() == "worker-2\n"
    for w in ("a001", "a002"):
        wst = state(run, w)
        assert wst["messages"][2]["content"].startswith("Delivered to you:\n- ")
        assert "(round" not in wst["messages"][2]["content"]
    # the final tick is snapshotted (snapshot_every 1000) and holds the frozen workspace
    store = SnapshotStore(run.dir)
    latest = store.latest()
    assert latest.round == run.meta["last_round"]
    assert pb.workspace_tar_of(store.load(latest)["world"]) == pb.workspace_tar(ws)[0]
    assert Run.load(run.dir).replay()["score"] == run.score
    res = CliRunner().invoke(app, ["report", str(run.dir), "--include-fake"])
    assert res.exit_code == 0, res.output


def test_async_never_assigned_worker_ends_by_its_per_agent_cap(tmp_path):
    run = _async_team(tmp_path, [MiniSweAgent(model=fake("forever"), steps_per_turn=1,
                                              templates="lead"),
                                 MiniSweAgent(model=fake("worker"), steps_per_turn=10,
                                              templates="worker", idle_until_assigned=True)],
                      ["lead", "worker"], rounds_per_agent=3)
    assert run.meta["end_reason"] == "max_rounds" and run.meta["last_round"] == 6
    ended = turns(run, "a001")
    assert [e["usage"]["notes"] for e in ended] == [["idle"]] * 3     # bounded: 3 ticks
    assert inference_rounds(run, "a001") == []
    results = _tool_results(run, "a001", "wait")
    assert len(results) == 3 and all(r["timed_out"] and r["posts"] == [] for r in results)


def test_async_role_without_wait_is_a_clear_error(tmp_path):
    roles = {"lead": Role(name="lead", tools=["post"]),
             "worker": Role(name="worker", tools=["bash", "post", "read_board"])}
    parts = [assign(LatePoster("x", "main", 0), "lead"),
             assign(MiniSweAgent(model=fake("worker"), templates="worker",
                                 idle_until_assigned=True, steps_per_turn=10), "worker")]
    exp = Experiment(name="msa-async", world=world(tmp_path), participants=parts, roles=roles,
                     medium=Board(delivery="pull", channels=["main", "lead"]),
                     options={"commit": "async", "max_calls_per_turn": 20,
                              "rounds_per_agent": 1})
    run = exp.run(seed=0, out=tmp_path / "runs")
    (ended,) = turns(run, "a001")
    assert ended["yield_kind"] == "error"
    assert "list `wait` in its role's tools" in ended["error"]


def test_async_wait_for_workers_says_so_when_no_worker_can_report(tmp_path):
    from swarmlab.participants.minisweagent import WAIT_STALLED

    run = _async_team(tmp_path, [MiniSweAgent(model=fake("waiting_lead"), templates="lead",
                                              steps_per_turn=10, wait_tool=True),
                                 MiniSweAgent(model=fake("worker"), steps_per_turn=10,
                                              templates="worker", idle_until_assigned=True)],
                      ["lead", "worker"], rounds_per_agent=3)
    assert run.meta["end_reason"] == "terminal"
    st = state(run, "a000")
    wait = [m for m in tools_of(st) if m.get("extra", {}).get("tool") == "wait_for_workers"]
    assert [m["content"] for m in wait] == [WAIT_STALLED]   # the idle worker is waiting too
    assert len(turns(run, "a000")) == 1
    assert st["messages"][-1]["extra"]["exit_status"] == "Submitted"


EXAMPLE_ASYNC = Path(__file__).resolve().parents[1] / "examples" / "programbench_team_async.yaml"


def test_guide_async_team_spec_is_the_example_and_validates(tmp_path):
    from typer.testing import CliRunner

    from swarmlab.cli import app
    from swarmlab.prompts_cmd import render_prompts

    spec = _guide_yaml("programbench-team-async")
    assert spec == yaml.safe_load(EXAMPLE_ASYNC.read_text())
    res = CliRunner().invoke(app, ["validate", str(EXAMPLE_ASYNC)])
    assert res.exit_code == 0, res.output
    exp = Experiment.from_yaml(EXAMPLE_ASYNC, "team")
    assert exp.options["commit"] == "async" and "scheduler" not in exp.options
    lead_row, worker_row = render_prompts(exp)
    assert lead_row["turn_based"] and lead_row["tools"] == [
        "bash", "read_board", "post", "end_turn", "wait_for_workers"]
    assert "it returns them as its result" in lead_row["user"]
    assert "`worker-1`, `worker-2`" in lead_row["user"]
    assert "wait" not in worker_row["tools"]
    assert "When you have no assignment, you wait until the lead sends one." in worker_row["user"]


def test_guide_async_team_spec_runs_end_to_end_with_fake_models(tmp_path):
    spec = _guide_yaml("programbench-team-async")
    arm = spec["arms"]["team"]
    arm["world"]["params"].update(workspace=str(tmp_path / "ws"), user=None, lockdown=False,
                                  check_task=False)
    arm["participants"][0]["params"]["model"] = fake("team_lead")
    arm["participants"][1]["params"]["model"] = fake("team_worker")
    spec["providers"] = {"fake": {"type": "fake", "params": {"yields": [0, 20]}}}
    path = tmp_path / "team.yaml"
    path.write_text(yaml.safe_dump(spec))
    run = Experiment.from_yaml(path, "team").run(seed=0, out=tmp_path / "runs")
    assert run.meta["end_reason"] == "terminal" and run.score["submitted"]
    posts = sorted((e["agent"], e["channel"]) for e in run.events if e["type"] == "post")
    assert posts == [("a000", "worker-1"), ("a000", "worker-2"), ("a001", "lead"),
                     ("a002", "lead")]
    assert state(run, "a000")["messages"][-1]["extra"]["exit_status"] == "Submitted"
    store = SnapshotStore(run.dir)
    assert store.latest().round == run.meta["last_round"]
    assert pb.workspace_tar_of(store.load(store.latest())["world"]) == \
        pb.workspace_tar(tmp_path / "ws")[0]
    assert Run.load(run.dir).replay()["score"] == run.score


def test_guidance_is_appended_to_the_team_templates_and_refused_for_solo(tmp_path):
    """`guidance` is a paragraph after the team text (lead and worker), before the resource
    sentence; the solo template is the official text and takes none."""
    for templates in ("lead", "worker"):
        p = MiniSweAgent(model="fake:x", templates=templates, guidance="Delegate the implementation.")
        view = _view(tools=[_post_tool(["lead", "worker-1"])])
        text = p.turn_message(view).content
        assert "Delegate the implementation." in text
        assert text.index("Delegate the implementation.") > text.index("## Working with a team") \
            if templates == "lead" else True
    assert MiniSweAgent(model="fake:x", templates="lead").guidance is None
    with pytest.raises(ValueError, match="guidance needs the lead"):
        MiniSweAgent(model="fake:x", templates="solo", guidance="x")
    with pytest.raises(ValueError, match="non-empty string"):
        MiniSweAgent(model="fake:x", templates="lead", guidance="  ")
