"""M3a §3 / §4 item 6: LLMAgent context limit (`context_limit_tokens`, `overflow`) on the fake
provider. A FlagGame round of `fake:reader` adds about 300 estimated tokens; the first request
of round 1 is about 840 (system prompt, round message, tool schemas)."""
import pickle

import pytest

from swarmlab import Budget, Run
from swarmlab.participants import LLMAgent
from swarmlab.participants.llm import DEFAULT_SUMMARY_MODEL
from swarmlab.providers.base import text_of
from swarmlab.providers.fake import FakeProvider
from swarmlab.snapshot import SnapshotStore

from .helpers import _resp, llm_agent_experiment, logical
from .test_recovery import _truncate_mid_round

SUMMARY = "fake:tests.test_context_limit:script_summary"


def script_summary(request, rng):
    """Summary model: a fixed-shape note naming the rounds it was given."""
    rounds = [line for line in text_of(request.messages[-1].content).splitlines()
              if line.startswith("## ")]
    return _resp(request, "I keep my beliefs. Covered: " + ", ".join(r[3:] for r in rounds))


@pytest.fixture
def requests_seen(monkeypatch):
    seen: list = []
    original = FakeProvider.complete

    async def recording(self, request):
        seen.append(request)
        return await original(self, request)

    monkeypatch.setattr(FakeProvider, "complete", recording)
    return seen


def run_with(tmp_path, n=2, rounds=6, seed=1, name="ctx", budget=None, **agent_kw):
    kw = {"budget": budget} if budget is not None else {}
    exp = llm_agent_experiment(n, agent_kw=agent_kw, name=name, **kw)
    return exp.run(seed=seed, max_rounds=rounds, out=tmp_path)


def overflows(run):
    return [e for e in run.events if e["type"] == "overflow"]


def turns(run):
    return [e for e in run.events if e["type"] == "turn_ended"]


def agent_state(run, round, agent="a000"):
    store = SnapshotStore(run.dir)
    return pickle.loads(store.load(store.read(round))[f"participant:{agent}"])


def swarm_requests(seen):
    return [r for r in seen if r.tools]


def round_labels(request):
    return [m.content[0].text for m in request.messages
            if m.role == "user" and not isinstance(m.content, str)]


# ---- params and spec -----------------------------------------------------------------------------
def test_params_only_when_set():
    plain = LLMAgent(model="fake:reader").spec()["params"]
    assert not {"context_limit_tokens", "overflow", "summary_model"} & set(plain)
    limited = LLMAgent(model="fake:reader", context_limit_tokens=500).spec()["params"]
    assert limited["context_limit_tokens"] == 500 and limited["overflow"] == "drop_oldest"
    assert "summary_model" not in limited
    summ = LLMAgent(model="fake:reader", context_limit_tokens=500, overflow="summarize",
                    summary_model=SUMMARY).spec()["params"]
    assert summ["summary_model"] == SUMMARY
    with pytest.raises(ValueError):
        LLMAgent(model="fake:reader", context_limit_tokens=0)
    with pytest.raises(ValueError):
        LLMAgent(model="fake:reader", context_limit_tokens=10, overflow="truncate")


def test_spec_hash_changes_only_with_a_limit():
    def h(**kw):
        return llm_agent_experiment(2, agent_kw=kw).spec_hash(seed=1, max_rounds=3)

    assert h() == h(overflow="drop_oldest") == h(overflow="summarize", summary_model=SUMMARY)
    assert h() != h(context_limit_tokens=1000) != h(context_limit_tokens=1000, overflow="fail_turn")


def test_no_overflow_events_when_unset(tmp_path):
    run = run_with(tmp_path, rounds=5)
    assert overflows(run) == []
    big = run_with(tmp_path / "big", rounds=5, context_limit_tokens=10**6)
    assert overflows(big) == []
    strip = lambda evs: [e for e in evs if e["type"] != "run_started"]
    assert strip(logical(big)) == strip(logical(run))  # an unreached limit changes nothing


# ---- drop_oldest ---------------------------------------------------------------------------------
def test_drop_oldest_logs_overflows_and_keeps_running(tmp_path, requests_seen):
    """Acceptance 6: small limit, `drop_oldest` logs overflows and the run goes on."""
    run = run_with(tmp_path, rounds=6, context_limit_tokens=1300)
    assert run.end_reason == "max_rounds"
    assert all(t["yield_kind"] == "end_turn" for t in turns(run))
    evs = overflows(run)
    assert evs and {e["agent"] for e in evs} == {"a000", "a001"}
    for e in evs:
        assert e["policy"] == "drop_oldest" and e["dropped_rounds"] >= 1
        assert e["tokens_before"] > 1300 >= e["tokens_after"]
        assert set(e) >= {"agent", "policy", "dropped_rounds", "tokens_before", "tokens_after"}
    # the overflow sits inside the agent's turn: after turn_started, before turn_ended
    seq = [(e["type"], e.get("agent")) for e in run.events]
    i = seq.index(("overflow", evs[0]["agent"]))
    assert ("turn_ended", evs[0]["agent"]) in seq[i:]
    for req in swarm_requests(requests_seen):
        assert req.messages[0].role == "system"
        assert FakeProvider().estimate_prompt_tokens(req) <= 1300
    # dropped rounds are gone from memory (snapshot) and from probe_context
    state = agent_state(run, 6)
    kept = [r["round"] for r in state["rounds"]]
    assert kept[-1] == 6 and kept[0] > 1 and kept == list(range(kept[0], 7))


def test_drop_oldest_never_drops_system_prompt_or_current_round(tmp_path, requests_seen):
    run = run_with(tmp_path, rounds=4, context_limit_tokens=1)
    assert run.end_reason == "max_rounds"
    assert all(t["yield_kind"] == "end_turn" for t in turns(run))
    for req in swarm_requests(requests_seen):
        assert req.messages[0].role == "system"
        labels = round_labels(req)
        assert len(labels) == 1  # only the current round is left
    for e in overflows(run):
        assert e["tokens_after"] > 1  # cannot get under: sent anyway
        if e["round"] == 1:
            assert e["dropped_rounds"] == 0
    assert [r["round"] for r in agent_state(run, 4)["rounds"]] == [4]
    # one overflow per request (each request is over the limit)
    assert len(overflows(run)) == len(swarm_requests(requests_seen))


# ---- summarize -----------------------------------------------------------------------------------
def test_summarize_one_measurement_call_per_overflow_and_inserts_the_note(tmp_path, requests_seen):
    run = run_with(tmp_path, rounds=6, context_limit_tokens=1500, overflow="summarize",
                   summary_model=SUMMARY)
    assert run.end_reason == "max_rounds"
    evs = overflows(run)
    assert evs and all(e["policy"] == "summarize" for e in evs)
    trims = [e for e in evs if e["dropped_rounds"] > 0 or e["tokens_after"] < e["tokens_before"]]
    summary_calls = [r for r in requests_seen if r.model == SUMMARY]
    assert len(summary_calls) == len(trims) == len(evs)
    attempts = [e for e in run.events_all if e.type == "inference_attempt"]
    measurement = [e for e in attempts if e.category == "measurement"]
    assert len(measurement) == len(evs)
    for req in summary_calls:
        assert "under 200 words" in text_of(req.messages[0].content)
        assert not req.tools and req.temperature == 0.0
    # exactly one note in memory, in front, as an assistant message; it persists in snapshots
    state = agent_state(run, 6)
    notes = [r for r in state["rounds"] if r.get("summary")]
    assert len(notes) == 1 and state["rounds"][0] is notes[0]
    msg = notes[0]["messages"]
    assert len(msg) == 1 and msg[0]["role"] == "assistant"
    assert msg[0]["content"].startswith("[Summary of my earlier rounds 1-")
    # later summaries fold the earlier note in
    assert len([e for e in evs if e["agent"] == "a000"]) > 1
    assert "Covered: Earlier summary, Round" in msg[0]["content"]
    # the swarm request after an overflow carries the note right after the system prompt
    after = [r for r in swarm_requests(requests_seen)
             if len(r.messages) > 1 and r.messages[1].role == "assistant"]
    assert after and all(text_of(r.messages[1].content).startswith("[Summary") for r in after)
    # probe_context reflects the trimmed memory
    agent = LLMAgent(model="fake:reader")
    agent.restore(pickle.dumps(state))
    ctx = agent.probe_context()
    assert ctx[0].role == "system" and text_of(ctx[1].content).startswith("[Summary")
    assert sum(1 for m in ctx if m.role == "user" and not isinstance(m.content, str)) == len(
        [r for r in state["rounds"] if not r.get("summary")])


def test_summarize_without_measurement_budget_drops_without_a_note(tmp_path):
    run = run_with(tmp_path, rounds=5, context_limit_tokens=1500, overflow="summarize",
                   summary_model=SUMMARY, budget=Budget(measurement_usd=1e-9))
    assert run.end_reason == "max_rounds"
    evs = overflows(run)
    assert evs and all(e["detail"] == "measurement_budget" for e in evs)
    assert not any(r.get("summary") for r in agent_state(run, 5)["rounds"])


def test_default_summary_model_is_haiku():
    assert DEFAULT_SUMMARY_MODEL == "anthropic:claude-haiku-4-5"


# ---- fail_turn -----------------------------------------------------------------------------------
def test_fail_turn_ends_the_turn_with_context_limit(tmp_path, requests_seen):
    run = run_with(tmp_path, rounds=4, context_limit_tokens=1000, overflow="fail_turn")
    assert run.end_reason == "max_rounds"
    by_round = {}
    for t in turns(run):
        by_round.setdefault(t["round"], []).append(t)
    assert all(t["yield_kind"] == "end_turn" for t in by_round[1])  # round 1 fits (~840-890)
    for r in (2, 3, 4):
        for t in by_round[r]:
            assert t["yield_kind"] == "error" and t["error"] == "context_limit"
            assert t["calls"] == 0
    evs = overflows(run)
    assert len(evs) == 6 and all(e["policy"] == "fail_turn" and e["dropped_rounds"] == 0
                                 and e["tokens_after"] == e["tokens_before"] > 1000 for e in evs)
    # no swarm request was sent for a failed turn
    assert len(swarm_requests(requests_seen)) == 2 * 2
    # memory keeps what the agent saw (nothing is dropped under fail_turn)
    assert [r["round"] for r in agent_state(run, 4)["rounds"]] == [1, 2, 3, 4]


# ---- window interplay ----------------------------------------------------------------------------
def test_window_trims_first_then_the_context_limit(tmp_path, requests_seen):
    # window of 2 rounds stays under ~1250 tokens: a 1300 limit never fires
    quiet = run_with(tmp_path / "q", rounds=5, memory="window", window_rounds=2,
                     context_limit_tokens=1300)
    assert overflows(quiet) == []
    requests_seen.clear()
    # window of 3 rounds (~1520) with the same limit: overflows drop the oldest of the window
    loud = run_with(tmp_path / "l", rounds=5, memory="window", window_rounds=3,
                    context_limit_tokens=1300)
    evs = overflows(loud)
    assert evs and all(e["dropped_rounds"] == 1 for e in evs)
    assert {e["round"] for e in evs} == {3, 4, 5}
    for req in swarm_requests(requests_seen):
        assert len(round_labels(req)) <= 2
    assert [r["round"] for r in agent_state(loud, 5)["rounds"]] == [4, 5]


# ---- determinism, replay, resume -----------------------------------------------------------------
@pytest.mark.parametrize("policy", ["drop_oldest", "summarize", "fail_turn"])
def test_deterministic_across_two_runs_and_replay(tmp_path, policy):
    kw = {"context_limit_tokens": 1300, "overflow": policy}
    if policy == "summarize":
        kw |= {"context_limit_tokens": 1500, "summary_model": SUMMARY}
    a = run_with(tmp_path / "a", n=3, rounds=5, seed=3, **kw)
    b = run_with(tmp_path / "b", n=3, rounds=5, seed=3, **kw)
    assert overflows(a) and logical(a) == logical(b)
    assert a.score == b.score and a.metrics == b.metrics
    loaded = Run.load(a.dir)  # replay: no provider calls, metrics and score check out
    assert loaded.metrics == a.metrics and loaded.score == a.score
    assert overflows(loaded) == overflows(a)


@pytest.mark.parametrize("policy", ["drop_oldest", "summarize"])
def test_resume_after_a_crash_reproduces_overflows(tmp_path, policy):
    kw = {"context_limit_tokens": 1300 if policy == "drop_oldest" else 1500, "overflow": policy}
    if policy == "summarize":
        kw["summary_model"] = SUMMARY
    ref = run_with(tmp_path / "ref", n=2, rounds=6, seed=5, **kw)
    crashed = run_with(tmp_path / "crash", n=2, rounds=6, seed=5, **kw)
    _truncate_mid_round(crashed.dir, 5)
    resumed = Run(crashed.dir).resume()
    # `budget` events differ for a reason unrelated to this feature: the truncation here leaves
    # run.json's ledger at the finished run's totals (a real crash would not)
    no_budget = lambda run: [e for e in logical(run) if e["type"] != "budget"]
    assert no_budget(resumed) == no_budget(ref)
    assert overflows(resumed) == overflows(ref)
    # the snapshot the resume restored from holds the trimmed memory (dropped rounds are gone)
    state = agent_state(ref, 4)
    real = [r["round"] for r in state["rounds"] if not r.get("summary")]
    assert real[-1] == 4 and real[0] > 1
    assert any(r.get("summary") for r in state["rounds"]) == (policy == "summarize")


def test_reader_still_decides_with_trimmed_memory(tmp_path):
    """Sanity: the fake reader keeps guessing (its own crop is in the current round)."""
    run = run_with(tmp_path, rounds=4, context_limit_tokens=1)
    guesses = [e for e in run.events if e["type"] == "tool_called" and e["tool"] == "guess"]
    assert len(guesses) == 2 * 4
