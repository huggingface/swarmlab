"""`swarmlab preflight` on the FlagGame image modality (docs/INTERFACE-M5.md §3)."""
import json

import httpx
import pytest

from swarmlab import Board, Experiment
from swarmlab.participants import LLMAgent
from swarmlab.preflight import PROMPT, plan, result_lines, run_preflight
from swarmlab.providers import openai_compat
from swarmlab.providers.openai_compat import OpenAICompatProvider
from swarmlab.worlds import FlagGame

MODEL = "Qwen/Qwen3.5-9B:deepinfra"


def exp(modality):
    return Experiment(name="pf", world=FlagGame(modality=modality),
                      participants=[LLMAgent(model=f"hf:{MODEL}", max_tokens=64)] * 2,
                      medium=Board(topology="broadcast"),
                      providers={"hf": OpenAICompatProvider("hf", pricing={MODEL: (0.1, 0.15, 0.1)},
                                                            max_retries=0)})


@pytest.fixture
def router(monkeypatch):
    """A text-only model behind the router: any image_url part is a 400."""
    seen = []
    state = {"message": "Model does not support image input: content type image_url is not allowed"}

    def handler(request):
        body = json.loads(request.content)
        seen.append(body)
        user = body["messages"][1]["content"]
        if isinstance(user, list) and any(p["type"] == "image_url" for p in user):
            return httpx.Response(400, json={"message": state["message"]})
        return httpx.Response(200, json={
            "choices": [{"finish_reason": "tool_calls", "message": {"content": "", "tool_calls": [
                {"id": "c1", "type": "function", "function": {"name": "end_turn", "arguments": "{}"}}]}}],
            "usage": {"prompt_tokens": 500, "completion_tokens": 20}})

    real = httpx.AsyncClient
    monkeypatch.setattr(openai_compat.httpx, "AsyncClient",
                        lambda **kw: real(**{**kw, "transport": httpx.MockTransport(handler)}))
    monkeypatch.setenv("HF_TOKEN", "hf_test")
    return seen, state


def test_text_mode_preflight_request_is_unchanged(router):
    rows = plan(exp("text"))
    assert rows[0]["request"].messages[1].content == PROMPT and "images" not in rows[0]
    res = run_preflight(rows, max_usd=1.0)
    assert res[0]["ok"] and "images" not in res[0]


def test_image_mode_sends_the_round_one_images(router):
    seen, _ = router
    rows = plan(exp("image"))
    content = rows[0]["request"].messages[1].content
    assert [p.type for p in content].count("image") == 9 == rows[0]["images"]
    assert content[0].text == "Round 1." and content[-1].text == PROMPT
    res = run_preflight(rows, max_usd=1.0)
    assert not res[0]["ok"] and res[0]["image_input_rejected"] and res[0]["images"] == 9
    assert res[0]["error"].startswith("model rejects image input")
    assert "content type image_url" in res[0]["error"]           # the provider's own text is kept
    assert any("model rejects image input" in line for line in result_lines(res))
    assert sum(p["type"] == "image_url" for p in seen[0]["messages"][1]["content"]) == 9


def test_unrelated_errors_are_not_called_image_rejections(router):
    _, state = router
    state["message"] = "rate limited by upstream; quota exhausted"
    res = run_preflight(plan(exp("image")), max_usd=1.0)
    assert not res[0]["ok"] and "image_input_rejected" not in res[0]
    assert not res[0]["error"].startswith("model rejects image input")
