"""Refusals: provider fields, the `attempt` cache key, run health, report, export, and
`LLMAgent(refusal_retries=N)` on fake providers and stub clients (no paid calls)."""
import hashlib
import json
from types import SimpleNamespace

import httpx
import pyarrow.parquet as pq
import pytest

from swarmlab import Run
from swarmlab.cli import health_warning, refusal_warning
from swarmlab.participants import LLMAgent
from swarmlab.providers.anthropic import build_kwargs, parse_refusal
from swarmlab.providers.base import ChatRequest, Refusal, request_bytes, request_hash
from swarmlab.providers.fake import FakeProvider
from swarmlab.providers.openai_compat import build_body
from swarmlab.report import build_report
from swarmlab.runner import refusal_of_turn
from swarmlab.spec import canonical_json

from .helpers import llm_agent_experiment, logical
from .test_cli_run_all import invoke
from .test_llm_agent import agent_state, messages, turns
from .test_providers import GUESS, anthropic_message, compat, completion, req, stub_provider
from .test_recovery import _truncate_mid_round


@pytest.fixture
def provider_calls(monkeypatch):
    seen: list[ChatRequest] = []
    original = FakeProvider.complete

    async def recording(self, request):
        seen.append(request)
        return await original(self, request)

    monkeypatch.setattr(FakeProvider, "complete", recording)
    return seen


def refusing(script: str, n: int = 2, **agent_kw):
    return llm_agent_experiment(n, agent_kw={"model": f"fake:tests.helpers:{script}", **agent_kw})


# ---- provider layer ----------------------------------------------------------------------------------
def test_attempt_is_in_the_hash_only_when_set():
    r = req()
    legacy = {k: v for k, v in r.model_dump(mode="json").items() if k != "attempt"}
    assert request_bytes(r) == canonical_json(legacy).encode()  # pre-`attempt` hashes hold
    again = r.model_copy(update={"attempt": 1})
    assert request_hash(again) != request_hash(r)
    assert json.loads(request_bytes(again))["attempt"] == 1
    assert request_hash(again) == hashlib.sha256(request_bytes(again)).hexdigest()


def test_attempt_is_never_sent():
    a, b = req(tools=[GUESS]), req(tools=[GUESS], attempt=3)
    assert build_kwargs(a) == build_kwargs(b) and "attempt" not in build_kwargs(b)
    a, b = req(model="hf:m", seed=1), req(model="hf:m", seed=1, attempt=3)
    assert build_body(a) == build_body(b) and "attempt" not in build_body(b)


async def test_anthropic_refusal_reads_stop_details():
    msg = anthropic_message(content=[], stop_reason="refusal")
    msg.stop_details = SimpleNamespace(type="refusal", category="cyber", explanation="Exploit code.")
    p = stub_provider([msg])
    resp = await p.complete(req())
    assert resp.finish_reason == "refusal" and resp.text == "" and resp.tool_calls == []
    assert resp.refusal == Refusal(category="cyber", explanation="Exploit code.")
    assert resp.usage.prompt_tokens == 150  # a refusal is still a billed, ordinary response


async def test_anthropic_refusal_without_details_and_partial_output():
    # null stop_details on a refusal; a mid-stream decline left a tool call with broken input
    partial = SimpleNamespace(type="tool_use", id="toolu_1", name="guess", input="{cand")
    msg = anthropic_message(content=[partial], stop_reason="refusal")
    msg.stop_details = None
    resp = await stub_provider([msg]).complete(req())
    assert resp.finish_reason == "refusal"  # wins over bad_tool_args
    assert resp.refusal == Refusal()


async def test_anthropic_stop_details_ignored_unless_refusal():
    msg = anthropic_message(stop_reason="end_turn")
    msg.stop_details = SimpleNamespace(type="refusal", category="bio", explanation="x")
    resp = await stub_provider([msg]).complete(req())
    assert resp.finish_reason == "end_turn" and resp.refusal is None
    plain = await stub_provider([anthropic_message()]).complete(req())  # no stop_details at all
    assert plain.refusal is None
    assert parse_refusal({"stop_reason": "refusal", "stop_details": {"category": None}}) == Refusal()


async def test_openai_content_filter_is_a_refusal():
    def handler(request):
        return httpx.Response(200, json=completion({"content": ""}, finish="content_filter"))

    resp = await compat(handler).complete(req(model="hf:m"))
    assert resp.finish_reason == "refusal" and resp.refusal == Refusal()

    def ok(request):
        return httpx.Response(200, json=completion({"content": "hi"}, finish="stop"))

    assert (await compat(ok).complete(req(model="hf:m"))).refusal is None


# ---- default (refusal_retries=0): today's behaviour, now counted ---------------------------------------
def test_default_keeps_behaviour_and_counts_refused_turns(tmp_path, provider_calls):
    run = refusing("script_always_refuse").run(seed=1, max_rounds=2, out=tmp_path)
    assert all(r.attempt == 0 for r in provider_calls) and len(provider_calls) == 4  # 2 agents x 2
    for t in turns(run):
        assert t["yield_kind"] == "no_tool"
        assert t["usage"]["finish_reasons"] == ["refusal"]
        assert t["usage"]["notes"] == ["refusal:cyber"]
    # the refused text is kept, as before
    assert messages(agent_state(run, 1))[-1] == {
        "role": "assistant", "content": "I can't help with that.", "tool_calls": None,
        "tool_call_id": None}
    responses = [e for e in run.events_all if e.type == "inference_response"]
    assert {(e.finish_reason, e.refusal_category) for e in responses} == {("refusal", "cyber")}
    # run.json and the summary
    meta = json.loads((run.dir / "run.json").read_text())
    assert meta["turns_refused"] == 4 and meta["refusal_categories"] == {"cyber": 4}
    assert "health" not in meta  # refusals do not degrade health
    s = run.summary()
    assert s["turns_refused"] == 4 and s["turns_total"] == 4 and s["turns_errored"] == 0
    assert health_warning(s) is None
    assert refusal_warning(s).startswith(f"WARNING: {run.id}: 4/4 turns refused (cyber 4)")


def test_refusal_event_field_is_omitted_when_none(tmp_path):
    run = llm_agent_experiment(1).run(seed=1, max_rounds=1, out=tmp_path)
    lines = [json.loads(x) for x in (run.dir / "events.jsonl").read_text().splitlines()]
    assert all("refusal_category" not in e for e in lines)
    assert json.loads((run.dir / "run.json").read_text())["turns_refused"] == 0
    assert refusal_warning(run.summary()) is None


def test_uncategorised_refusal_counts_as_none(tmp_path):
    run = refusing("script_refuse_uncategorised", 1).run(seed=1, max_rounds=1, out=tmp_path)
    assert run.summary()["refusal_categories"] == {"none": 1}
    (resp,) = [e for e in run.events_all if e.type == "inference_response"]
    assert resp.finish_reason == "refusal" and resp.refusal_category is None


def test_cli_warns_and_report_and_export_show_refusals(tmp_path):
    out = tmp_path / "runs"
    run = refusing("script_always_refuse").run(seed=1, max_rounds=2, out=out)
    res, _ = invoke("replay", run.dir)
    assert res.exit_code == 0, res.output  # a warning, not a failure
    assert f"WARNING: {run.id}: 4/4 turns refused (cyber 4)" in res.stderr
    text = build_report(out, include_fake=True)
    assert "| turns refused | 4 of 4 (cyber 4) |" in text
    assert "| refused responses, probes included | 4 (cyber 4) |" in text
    exported = run.export(tmp_path / "export")
    table = pq.read_table(exported / "tables" / "inference.parquet").to_pylist()
    assert [(r["finish_reason"], r["refusal_category"]) for r in table] == [("refusal", "cyber")] * 4


def test_runs_without_refusal_fields_are_counted_from_the_log(tmp_path):
    run = refusing("script_always_refuse", 1).run(seed=1, max_rounds=1, out=tmp_path)
    meta = json.loads((run.dir / "run.json").read_text())
    for k in ("turns_refused", "refusal_categories"):
        meta.pop(k)
    (run.dir / "run.json").write_text(json.dumps(meta))
    s = Run(run.dir).summary()
    assert s["turns_refused"] == 1 and s["refusal_categories"] == {"cyber": 1}


def test_refusal_of_turn():
    assert refusal_of_turn({"finish_reasons": ["tool_use", "refusal"],
                            "notes": ["refusal:bio", "refusal:cyber"]}) == "cyber"
    assert refusal_of_turn({"finish_reasons": ["refusal", "tool_use"], "notes": ["refusal:bio"]}) is None
    assert refusal_of_turn({"finish_reasons": ["refusal"]}) == "unknown"  # logs from before notes
    assert refusal_of_turn({}) is None and refusal_of_turn(None) is None


# ---- refusal_retries ---------------------------------------------------------------------------------
def test_retry_resends_the_same_messages_with_a_new_attempt(tmp_path, provider_calls):
    run = refusing("script_refuse_first", 1, refusal_retries=2).run(seed=1, max_rounds=2, out=tmp_path)
    assert run.end_reason == "max_rounds"
    first, second = provider_calls[0], provider_calls[1]
    assert (first.attempt, second.attempt) == (0, 1)
    assert first.messages == second.messages  # the refused reply was not appended
    # every model call of the turn is refused once and re-sent once
    assert [r.attempt for r in provider_calls] == [0, 1] * (len(provider_calls) // 2)
    t = turns(run)[0]
    assert t["yield_kind"] == "end_turn" and t["usage"]["finish_reasons"] == [
        "refusal", "tool_use", "refusal", "tool_use"]
    assert t["usage"]["notes"] == ["refusal:cyber", "refusal:cyber"]
    assert run.summary()["turns_refused"] == 0  # the final responses were answers
    msgs = messages(agent_state(run, 1))
    assert not any(m["role"] == "assistant" and m["content"] == "I can't" for m in msgs)
    # the refusal's partial guess was never executed
    guesses = [e for e in run.events if e["type"] == "tool_called" and e["tool"] == "guess"]
    assert len(guesses) == 2 and run.score["n_guessed"] == 1


def test_retry_budget_is_per_turn_and_then_falls_back(tmp_path, provider_calls):
    run = refusing("script_always_refuse", 1, refusal_retries=2).run(seed=1, max_rounds=1, out=tmp_path)
    assert [r.attempt for r in provider_calls] == [0, 1, 2]
    (t,) = turns(run)
    assert t["usage"]["finish_reasons"] == ["refusal"] * 3 and len(t["usage"]["notes"]) == 3
    assert t["yield_kind"] == "no_tool"
    # with the re-sends used up the last refusal is handled as by default: its text is kept
    assert messages(agent_state(run, 1))[-1]["content"] == "I can't help with that."
    assert run.summary()["refusal_categories"] == {"cyber": 1}


def test_retries_do_not_count_toward_max_calls(tmp_path, provider_calls):
    run = refusing("script_refuse_first", 1, refusal_retries=1, max_calls=1).run(
        seed=1, max_rounds=1, out=tmp_path)
    assert [r.attempt for r in provider_calls] == [0, 1]
    (t,) = turns(run)
    assert t["usage"]["finish_reasons"] == ["refusal", "tool_use"]


def test_retry_in_report_json_mode(tmp_path, provider_calls):
    exp = refusing("script_report_refuse_first", 1, refusal_retries=1, report_json=True)
    run = exp.run(seed=1, max_rounds=1, out=tmp_path)
    assert [r.attempt for r in provider_calls] == [0, 1]
    assert provider_calls[0].messages == provider_calls[1].messages
    (t,) = turns(run)
    assert t["yield_kind"] == "end_turn" and t["usage"]["notes"] == ["refusal:bio"]
    assert not any("report_json" in n for n in t["usage"]["notes"])  # no corrective retry used
    assert [e["tool"] for e in run.events if e["type"] == "tool_called"] == ["guess", "post", "end_turn"]
    assert not any("nowhere" in str(m["content"]) for m in messages(agent_state(run, 1)))


def test_retry_runs_are_deterministic_and_replay_never_calls_a_provider(tmp_path, provider_calls):
    a = refusing("script_refuse_first", refusal_retries=2).run(seed=3, max_rounds=3, out=tmp_path / "a")
    b = refusing("script_refuse_first", refusal_retries=2).run(seed=3, max_rounds=3, out=tmp_path / "b")
    assert logical(a) == logical(b) and a.score == b.score
    made = len(provider_calls)
    assert made == a.spend["calls"] + b.spend["calls"]
    loaded = Run.load(a.dir)  # full replay
    assert len(provider_calls) == made
    assert loaded.metrics == a.metrics and loaded.score == a.score
    # a crash mid-round: resume re-creates the same attempts and every one hits the cache
    _truncate_mid_round(a.dir, 2)
    resumed = Run(a.dir).resume()
    assert len(provider_calls) == made

    def no_budget(run):  # the ledger keeps what the discarded round charged
        return [e for e in logical(run) if e["type"] != "budget"]

    assert no_budget(resumed) == no_budget(b)
    hits = [e for e in resumed.events_all if e.type == "inference_response" and e.round >= 2]
    assert hits and all(e.cached for e in hits if e.round == 2)


def test_refusal_retries_param_and_validation():
    assert "refusal_retries" not in LLMAgent(model="fake:reader").params
    assert LLMAgent(model="fake:reader", refusal_retries=2).params["refusal_retries"] == 2
    a = llm_agent_experiment(1).spec_hash(seed=0, max_rounds=1)
    b = llm_agent_experiment(1, agent_kw={"refusal_retries": 0}).spec_hash(seed=0, max_rounds=1)
    c = llm_agent_experiment(1, agent_kw={"refusal_retries": 1}).spec_hash(seed=0, max_rounds=1)
    assert a == b != c  # the default leaves existing spec hashes alone
    for bad in (-1, True, 1.5):
        with pytest.raises(ValueError, match="refusal_retries"):
            LLMAgent(model="fake:reader", refusal_retries=bad)
