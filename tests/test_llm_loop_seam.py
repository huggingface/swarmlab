"""LLMAgent's model-call seam (`_infer` / `_handle_response`, docs/INTERFACE-M1b.md §4):
subclasses that override one step, and goldens recorded before the loop was decomposed."""
import hashlib
import json
from pathlib import Path

import pytest

from swarmlab.participants import LLMAgent
from swarmlab.spec import canonical_json

from .helpers import llm_agent_experiment, logical
from .test_report_json import experiment as report_experiment

GOLDENS = Path(__file__).parent / "fixtures" / "llm_loop_goldens.json"


def _llm(n=1, **kw):
    return llm_agent_experiment(n, agent_kw=kw)


def _report(script=None, n=1, **kw):
    model = "fake:country_reporter" if script is None else "fake:x"
    return report_experiment(LLMAgent(model, report_json=True, **kw), n=n, script=script)


# name -> (experiment builder, seed, max_rounds); every branch of the turn loop on a fake provider
SCENARIOS = {
    "native_reader": (lambda: _llm(4), 1, 2),
    "native_max_calls": (lambda: _llm(2, model="fake:tests.helpers:script_status_forever", max_calls=3), 1, 1),
    "native_raw_args": (lambda: _llm(1, model="fake:tests.helpers:script_raw_args"), 1, 1),
    "native_end_turn_first": (lambda: _llm(2, model="fake:tests.helpers:script_end_turn_first"), 1, 1),
    "text_tool_fallback": (lambda: _llm(2, model="fake:tests.test_smoke_fixes:script_text_calls",
                                        text_tool_fallback=True), 1, 1),
    "json_reader": (lambda: _llm(2, tool_protocol="json"), 1, 2),
    "json_bad_then_ok": (lambda: _llm(2, model="fake:tests.helpers:script_bad_json_then_ok",
                                      tool_protocol="json"), 1, 1),
    "json_always_bad": (lambda: _llm(1, model="fake:tests.helpers:script_always_bad_json",
                                     tool_protocol="json"), 1, 1),
    "refusal_retries": (lambda: _llm(2, model="fake:tests.helpers:script_refuse_first",
                                     refusal_retries=2), 3, 3),
    "refusal_exhausted": (lambda: _llm(1, model="fake:tests.helpers:script_always_refuse_ending",
                                       refusal_retries=1), 1, 2),
    "refusal_max_calls": (lambda: _llm(1, model="fake:tests.helpers:script_refuse_first",
                                       refusal_retries=1, max_calls=1), 1, 1),
    "refusal_json": (lambda: _llm(1, model="fake:tests.helpers:script_refuse_first",
                                  refusal_retries=1, tool_protocol="json"), 1, 2),
    "report_json": (lambda: _report(n=2), 1, 2),
    "report_json_received": (lambda: _report(n=2, memory="received"), 1, 2),
    "report_json_json_protocol": (lambda: _report(n=2, tool_protocol="json"), 1, 1),
    "report_json_retry": (lambda: _report("tests.test_report_json:bad_then_good"), 1, 1),
    "report_json_failed": (lambda: _report("tests.test_report_json:always_bad"), 1, 1),
    "report_json_one_call": (lambda: _report("tests.test_report_json:always_bad", max_calls=1), 1, 1),
    "report_json_native": (lambda: _report("tests.test_report_json:native_guess"), 1, 1),
    "report_json_refusal": (lambda: _llm(1, model="fake:tests.helpers:script_report_refuse_first",
                                         refusal_retries=1, report_json=True), 1, 2),
}


def fingerprint(run) -> dict:
    """sha256 of the logical events (minus run id and the checkout's git state) and of the
    request hashes, in log order."""
    events = canonical_json(logical(run, exclude=("seq", "ts", "run", "git_commit", "dirty")))
    requests = [e.request_hash for e in run.events_all if e.type == "inference_attempt"]
    return {"logical": hashlib.sha256(events.encode()).hexdigest(),
            "requests": hashlib.sha256(canonical_json(requests).encode()).hexdigest(),
            "n_requests": len(requests)}


def record(out: Path) -> dict:
    """Regenerate the goldens (run once, on the code before the decomposition)."""
    found = {}
    for name, (build, seed, rounds) in SCENARIOS.items():
        found[name] = fingerprint(build().run(seed=seed, max_rounds=rounds, out=out / name))
    return found


@pytest.mark.parametrize("name", sorted(SCENARIOS))
def test_events_and_requests_match_the_pre_seam_goldens(tmp_path, name):
    build, seed, rounds = SCENARIOS[name]
    golden = json.loads(GOLDENS.read_text())[name]
    assert fingerprint(build().run(seed=seed, max_rounds=rounds, out=tmp_path)) == golden


# ---- subclasses that override one step --------------------------------------------------------------
def script_json_object(request, rng):
    """Refuse at attempt 0, then answer with a bare JSON object (no tool calls) naming the guess."""
    from .helpers import _refusal, _resp
    from .test_smoke_fixes import _reader_guess

    if request.attempt == 0:
        return _refusal(request, "cyber", "no", calls=[("guess", {"candidate": "A"})])
    return _resp(request, json.dumps({"candidate": _reader_guess(request, rng)}))


class JsonObjectAgent(LLMAgent):
    """Reply handling only: the reply is one JSON object `{"candidate": ...}`, played as a guess."""

    async def _handle_response(self, resp, view, tools):
        from swarmlab.probes import parse_json_object

        self._append(self._assistant(resp))
        obj = parse_json_object(resp.text or "")
        if not obj or "candidate" not in obj:
            self._turn.usage.notes.append("json_object:none")
            return 0, True
        await tools.call("guess", {"candidate": obj["candidate"]})
        await tools.call("end_turn", {})
        return 2, True


class GuessOnlyAgent(LLMAgent):
    """Model call only: offers the model `guess` and `end_turn` and nothing else."""

    async def _infer(self, view, tools, usage):
        self.inferred = getattr(self, "inferred", 0) + 1
        keep = [t for t in view.tools if t.name in ("guess", "end_turn")]
        return await super()._infer(view.model_copy(update={"tools": keep}), tools, usage)


def _subclass_experiment(agent, n=2):
    from swarmlab import Board, Experiment
    from swarmlab.providers.fake import FakeProvider
    from swarmlab.world.flaggame import FlagGame

    from .helpers import TEST_PRICING

    return Experiment(name="seam", world=FlagGame(), participants=[agent] * n, medium=Board(),
                      metrics=["belief.consensus"], providers={"fake": FakeProvider(pricing=TEST_PRICING)})


def test_overriding_only_handle_response_runs_a_full_experiment(tmp_path):
    from .test_llm_agent import agent_state, messages, turns

    agent = JsonObjectAgent(model="fake:tests.test_llm_loop_seam:script_json_object", refusal_retries=1)
    run = _subclass_experiment(agent).run(seed=1, max_rounds=2, out=tmp_path)
    assert run.end_reason == "max_rounds" and run.score["n_guessed"] == 2
    assert run.spec.participants[0].type == "tests.test_llm_loop_seam:JsonObjectAgent"
    for t in turns(run):
        assert t["yield_kind"] == "end_turn" and t["usage"]["calls"] == 2
        # `_infer` still dropped the refusal and re-sent: the subclass never saw it
        assert t["usage"]["finish_reasons"] == ["refusal", "end_turn"]
        assert t["usage"]["notes"] == ["refusal:cyber"]
    called = [e["tool"] for e in run.events if e["type"] == "tool_called"]
    assert called == ["guess", "end_turn"] * 4
    attempts = [e.attempt for e in run.events_all if e.type == "inference_attempt"]
    assert attempts == [0, 1] * 4
    msgs = messages(agent_state(run, 2))
    assert [m["role"] for m in msgs] == ["user", "assistant"] * 2  # refusals were not appended
    assert json.loads(msgs[1]["content"])["candidate"]
    again = _subclass_experiment(agent).run(seed=1, max_rounds=2, out=tmp_path / "again")
    assert logical(again, exclude=("seq", "ts", "run")) == logical(run, exclude=("seq", "ts", "run"))


def test_overriding_only_infer_is_used(tmp_path, monkeypatch):
    from swarmlab.providers.fake import FakeProvider

    from .test_llm_agent import turns

    seen = []
    original = FakeProvider.complete

    async def recording(self, request):
        seen.append(request)
        return await original(self, request)

    monkeypatch.setattr(FakeProvider, "complete", recording)
    run = _subclass_experiment(GuessOnlyAgent(model="fake:reader")).run(seed=1, max_rounds=2, out=tmp_path)
    swarm = [r for r in seen if r.tools]
    assert len(swarm) == 4 and all([t.name for t in r.tools] == ["guess", "end_turn"] for r in swarm)
    assert [e["tool"] for e in run.events if e["type"] == "tool_called"] == ["guess", "end_turn"] * 4
    assert {t["yield_kind"] for t in turns(run)} == {"end_turn"}
    assert all(t["usage"]["inference_calls"] == 1 for t in turns(run))
