"""Errored runs are loud, and `swarmlab preflight` catches them first (field notes item 2)."""
import json

import httpx
import pytest
import yaml

from swarmlab import Experiment, Run
from swarmlab.providers import openai_compat
from swarmlab.runner import error_headline

from .test_cli_run_all import invoke

MODEL = "Qwen/Qwen3.8-27B:cerebras"
SPEC = {
    "name": "cere",
    "providers": {"hf": {"type": "openai_compat", "params": {
        "name": "hf", "pricing": {MODEL: [0.6, 1.2, 0.6]}, "max_retries": 0}}},
    "options": {"max_rounds": 2},
    "budget": {"hard_usd": 1.0},
    "arms": {
        "bad": {"world": "flaggame", "participants": [{"type": "llm", "count": 3, "params": {
            "model": f"hf:{MODEL}", "max_tokens": 64,
            "extra": {"chat_template_kwargs": {"enable_thinking": False}}}}]},
        "good": {"world": "flaggame", "participants": [
            {"type": "llm", "count": 2, "params": {"model": f"hf:{MODEL}", "max_tokens": 64,
                                                  "extra": {"reasoning_effort": "none"}}},
            {"type": "evidence_aggregator", "count": 1}]},
    },
}


@pytest.fixture
def cerebras(monkeypatch):
    """The HF router answering like Cerebras: `chat_template_kwargs` is a 400."""
    seen = []

    def handler(request):
        body = json.loads(request.content)
        seen.append(body)
        if "chat_template_kwargs" in body:
            return httpx.Response(400, json={"message": "body.chat_template_kwargs: property "
                                             "'chat_template_kwargs' is unsupported"})
        return httpx.Response(200, headers={"x-inference-provider": "cerebras"}, json={
            "choices": [{"finish_reason": "tool_calls", "message": {"content": "", "tool_calls": [
                {"id": "c1", "type": "function",
                 "function": {"name": "end_turn", "arguments": "{}"}}]}}],
            "usage": {"prompt_tokens": 500, "completion_tokens": 20,
                      "completion_tokens_details": {"reasoning_tokens": 7}}})

    real = httpx.AsyncClient
    monkeypatch.setattr(openai_compat.httpx, "AsyncClient",
                        lambda **kw: real(**{**kw, "transport": httpx.MockTransport(handler)}))
    return seen


def spec_file(tmp_path):
    p = tmp_path / "cere.yaml"
    p.write_text(yaml.safe_dump(SPEC))
    return p


def test_error_headline():
    tb = ("Traceback (most recent call last):\n  File \"x.py\", line 1, in f\n    g()\n"
          "swarmlab.providers.base.ProviderError: hf: HTTP 400: bad field\nmore detail\n")
    assert error_headline(tb) == "ProviderError: hf: HTTP 400: bad field"
    assert error_headline("context_limit: too long\nsecond") == "context_limit: too long"
    assert error_headline(None) == ""


def test_all_turns_errored_is_degraded_and_run_exits_1(tmp_path, cerebras):
    out = tmp_path / "runs"
    res, data = invoke("run", spec_file(tmp_path), "--arm", "bad", "--seed", 0, "--out", out,
                       "--yes", "--json")
    assert res.exit_code == 1, res.output
    assert data["end_reason"] == "max_rounds"  # the end reason stays as it is
    assert data["turns_total"] == data["turns_errored"] == 6 and data["health"] == "degraded"
    assert "HTTP 400" in data["first_error"] and "chat_template_kwargs" in data["first_error"]
    assert "WARNING: cere__bad__s0: 6/6 turns errored (first error: ProviderError" in res.stderr
    meta = json.loads((out / "cere__bad__s0" / "run.json").read_text())
    assert meta["health"] == "degraded" and meta["turns_errored"] == 6
    # the multi-run table says errored; replay prints the warning again
    res, data = invoke("run", spec_file(tmp_path), "--out", tmp_path / "all", "--yes", "--json")
    assert res.exit_code == 1
    assert {r["run_id"]: r["outcome"] for r in data["runs"]} == {
        "cere__bad__s0": "errored", "cere__good__s0": "ran"}
    res, _ = invoke("run", spec_file(tmp_path), "--out", tmp_path / "t", "--arm", "bad", "--yes")
    assert "errored" in res.stdout.splitlines()[-1]
    res, _ = invoke("replay", out / "cere__bad__s0")
    assert res.exit_code == 0 and "6/6 turns errored" in res.stderr


def test_healthy_run_reports_turn_counts(tmp_path, cerebras):
    exp = Experiment.from_yaml(spec_file(tmp_path), "good")
    run = exp.run(seed=0, out=tmp_path / "r")
    s = run.summary()
    assert s["turns_total"] == 6 and s["turns_errored"] == 0 and "health" not in s
    # run dirs written before the fields existed are counted from the log
    meta = json.loads((run.dir / "run.json").read_text())
    for k in ("turns_total", "turns_errored"):
        meta.pop(k)
    (run.dir / "run.json").write_text(json.dumps(meta))
    assert Run(run.dir).summary()["turns_total"] == 6


def test_preflight_shows_the_http_error(tmp_path, cerebras):
    res, data = invoke("preflight", spec_file(tmp_path), "--arm", "bad", "--json")
    assert res.exit_code == 1, res.output
    assert "worst case $" in res.stderr  # printed before anything is sent
    (g,) = data["groups"]
    assert g["ok"] is False and "HTTP 400" in g["error"] and "chat_template_kwargs" in g["error"]
    assert g["extra"] == {"chat_template_kwargs": {"enable_thinking": False}}
    assert len(cerebras) == 1  # one request per group, with the arm's exact extra and tools
    body = cerebras[0]
    assert body["chat_template_kwargs"] == {"enable_thinking": False} and body["max_tokens"] == 64
    names = {t["function"]["name"] for t in body["tools"]}
    assert {"guess", "post", "read_board", "end_turn"} <= names


def test_preflight_ok(tmp_path, cerebras):
    res, _ = invoke("preflight", spec_file(tmp_path), "--arm", "good")
    assert res.exit_code == 0, res.output
    assert "tool call parsed: end_turn" in res.stdout and "7 reasoning tokens" in res.stdout
    assert "served by cerebras" in res.stdout and "skipped (no model" in res.stdout
    assert cerebras[0]["reasoning_effort"] == "none" and len(cerebras) == 1


def test_preflight_refuses_above_max_usd(tmp_path, cerebras):
    res, _ = invoke("preflight", spec_file(tmp_path), "--arm", "good", "--max-usd", "0.0000001")
    assert res.exit_code == 1 and "nothing sent" in res.stderr and not cerebras
