"""OpenAICompatProvider: chat-completions over `httpx.AsyncClient` (docs/INTERFACE-M1b.md §1).

`OpenAICompatProvider(name, base_url=None, api_key_env=None, pricing=None, concurrency=8,
timeout_s=120.0)`. Presets (`preset(name, **overrides)`): `hf` (`https://router.huggingface.co/v1`,
`HF_TOKEN`), `openai` (`https://api.openai.com/v1`, `OPENAI_API_KEY`), `vllm` (`base_url`
required, no key). A preset's `base_url`/`api_key_env` fill in when not given explicitly, so
`OpenAICompatProvider("hf", pricing={...})` is the HF router with prices. Presets ship with an
empty pricing table: open-model prices vary by served provider, so the experimenter supplies them
(an unpriced model is an error at `Experiment` construction).

Mapping:

- Messages: `content` strings pass through; part lists become `{"type": "text"}` and
  `{"type": "image_url", "image_url": {"url": "data:image/png;base64,..."}}` parts. Assistant
  tool calls -> `tool_calls: [{"id", "type": "function", "function": {"name", "arguments": <JSON>}}]`
  (content None when empty). Tool messages -> `{"role": "tool", "tool_call_id", "content": <text>}`.
- `tools: [{"type": "function", "function": {"name", "description", "parameters"}}]` (parameters
  from `ToolSchema.normalized()`) and `tool_choice: "auto"` when `tool_protocol == "native"` and
  tools are present.
- `max_tokens`, `temperature`, `top_p`, `seed` when given; `thinking_budget` is ignored; `extra`
  is merged into the body last, as top-level body fields. This is the provider passthrough:
  Qwen3 on DeepInfra/vLLM/SGLang turns thinking off with
  `extra={"chat_template_kwargs": {"enable_thinking": False}}` (DeepInfra documents exactly this
  body field for Qwen/Qwen3.5-9B; the HF router forwards the body to the provider), and
  OpenAI-style reasoning knobs (`reasoning_effort`, `reasoning: {...}`) pass the same way. A key
  `extra_body` (the OpenAI SDK's spelling) is flattened into the body too.
- Response: `choices[0].message.content` (None -> ""), `tool_calls[].function.arguments` parsed as
  JSON; a non-JSON string becomes `{"_raw": <string>}` and `finish_reason = "bad_tool_args"`.
  Usage from `prompt_tokens`, `completion_tokens`, `prompt_tokens_details.cached_tokens`,
  `completion_tokens_details.reasoning_tokens`. `served_by` = the `x-inference-provider` header.
- Retries: HTTP 429 and 5xx, and transport errors, with exponential backoff (honouring
  `retry-after`, capped at 60 s) up to 3 times; then, or for any other non-2xx, `ProviderError`.
- A fresh `httpx.AsyncClient` is opened per call (cheap at our call rates, and never bound to a
  dead event loop). Tests set `_transport` to an `httpx.MockTransport`.
"""
from __future__ import annotations

import asyncio
import json
import os
import time
from typing import Any, ClassVar

import httpx

from ..tools import ToolCall
from .base import (
    ChatMessage,
    ChatRequest,
    ChatResponse,
    Provider,
    ProviderError,
    Usage,
    model_id,
    parse_json_args,
    text_of,
)

PRESETS: dict[str, dict[str, str | None]] = {
    "hf": {"base_url": "https://router.huggingface.co/v1", "api_key_env": "HF_TOKEN"},
    "openai": {"base_url": "https://api.openai.com/v1", "api_key_env": "OPENAI_API_KEY"},
    "vllm": {"base_url": None, "api_key_env": None},
}
MAX_RETRIES = 3


def _message(m: ChatMessage) -> dict:
    if m.role == "tool":
        return {"role": "tool", "tool_call_id": m.tool_call_id or "", "content": text_of(m.content)}
    if isinstance(m.content, str):
        content: Any = m.content
    else:
        content = []
        for p in m.content:
            if p.type == "image":
                content.append({"type": "image_url",
                                "image_url": {"url": f"data:image/png;base64,{p.image_png_b64 or ''}"}})
            else:
                content.append({"type": "text", "text": p.text or ""})
    out: dict[str, Any] = {"role": m.role, "content": content}
    if m.role == "assistant" and m.tool_calls:
        out["tool_calls"] = [
            {"id": tc.call_id, "type": "function",
             "function": {"name": tc.name, "arguments": json.dumps(tc.args)}}
            for tc in m.tool_calls
        ]
        if not content:
            out["content"] = None
    return out


def build_body(request: ChatRequest) -> dict:
    body: dict[str, Any] = {
        "model": model_id(request),
        "messages": [_message(m) for m in request.messages],
        "max_tokens": request.max_tokens,
    }
    if request.tools and request.tool_protocol == "native":
        body["tools"] = [{"type": "function", "function": {
            "name": t.name, "description": t.description, "parameters": t.parameters}}
            for t in (s.normalized() for s in request.tools)]
        body["tool_choice"] = "auto"
    for key in ("temperature", "top_p", "seed"):
        value = getattr(request, key)
        if value is not None:
            body[key] = value
    extra = dict(request.extra)
    nested = extra.pop("extra_body", None)
    body.update(extra)
    if isinstance(nested, dict):  # OpenAI-SDK habit: extra_body={...} means top-level body fields
        body.update(nested)
    return body


def parse_completion(data: dict) -> tuple[str, list[ToolCall], Usage, str]:
    choice = (data.get("choices") or [{}])[0]
    msg = choice.get("message") or {}
    calls: list[ToolCall] = []
    bad = False
    for i, tc in enumerate(msg.get("tool_calls") or []):
        fn = tc.get("function") or {}
        args, ok = parse_json_args(fn.get("arguments"))
        bad = bad or not ok
        calls.append(ToolCall(call_id=tc.get("id") or f"call_{i}", name=fn.get("name", ""), args=args))
    u = data.get("usage") or {}
    usage = Usage(
        prompt_tokens=int(u.get("prompt_tokens") or 0),
        completion_tokens=int(u.get("completion_tokens") or 0),
        cached_prompt_tokens=int((u.get("prompt_tokens_details") or {}).get("cached_tokens") or 0),
        reasoning_tokens=int((u.get("completion_tokens_details") or {}).get("reasoning_tokens") or 0),
    )
    finish = "bad_tool_args" if bad else str(choice.get("finish_reason") or "")
    return msg.get("content") or "", calls, usage, finish


class OpenAICompatProvider(Provider):
    entry_point: ClassVar[str | None] = "openai_compat"

    def __init__(self, name: str, base_url: str | None = None, api_key_env: str | None = None,
                 pricing: dict | None = None, concurrency: int = 8, timeout_s: float = 120.0) -> None:
        preset = PRESETS.get(name, {})
        self.name = name
        self.base_url = (base_url or preset.get("base_url") or "").rstrip("/")
        if not self.base_url:
            raise ValueError(f"provider {name!r} needs a base_url")
        self.api_key_env = api_key_env if api_key_env is not None else preset.get("api_key_env")
        self.timeout_s = timeout_s
        self._setup(pricing, concurrency)
        self._transport: httpx.AsyncBaseTransport | None = None
        self._backoff_s = 1.0

    def _headers(self) -> dict[str, str]:
        headers = {"content-type": "application/json"}
        key = os.environ.get(self.api_key_env) if self.api_key_env else None
        if key:
            headers["authorization"] = f"Bearer {key}"
        return headers

    def _delay(self, attempt: int, retry_after: str | None) -> float:
        try:
            if retry_after is not None:
                return min(float(retry_after), 60.0)
        except ValueError:
            pass
        return min(self._backoff_s * (2 ** attempt), 60.0)

    async def complete(self, request: ChatRequest) -> ChatResponse:
        self.calls += 1
        body = build_body(request)
        start = time.monotonic()
        async with httpx.AsyncClient(transport=self._transport, timeout=self.timeout_s) as client:
            attempt = 0
            while True:
                try:
                    resp = await client.post(f"{self.base_url}/chat/completions", json=body,
                                             headers=self._headers())
                except httpx.TransportError as e:
                    if attempt >= MAX_RETRIES:
                        raise ProviderError(f"{self.name}: transport error: {e}") from e
                    await asyncio.sleep(self._delay(attempt, None))
                    attempt += 1
                    continue
                if resp.status_code == 429 or resp.status_code >= 500:
                    if attempt >= MAX_RETRIES:
                        raise ProviderError(f"{self.name}: HTTP {resp.status_code} after "
                                            f"{MAX_RETRIES} retries: {resp.text[:500]}",
                                            status=resp.status_code)
                    await asyncio.sleep(self._delay(attempt, resp.headers.get("retry-after")))
                    attempt += 1
                    continue
                if resp.status_code >= 400:
                    raise ProviderError(f"{self.name}: HTTP {resp.status_code}: {resp.text[:500]}",
                                        status=resp.status_code)
                break
        text, calls, usage, finish = parse_completion(resp.json())
        return ChatResponse(
            text=text, tool_calls=calls, usage=usage, cost_usd=self.cost(request, usage),
            provider=self.name, model=model_id(request),
            served_by=resp.headers.get("x-inference-provider"),
            latency_s=time.monotonic() - start, finish_reason=finish,
        )


def preset(name: str, **overrides: Any) -> OpenAICompatProvider:
    if name not in PRESETS:
        raise ValueError(f"unknown OpenAI-compatible preset {name!r}; known: {sorted(PRESETS)}")
    return OpenAICompatProvider(name, **overrides)
