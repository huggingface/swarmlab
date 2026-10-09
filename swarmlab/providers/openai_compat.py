"""OpenAICompatProvider: chat-completions over `httpx.AsyncClient` (docs/INTERFACE-M1b.md §1).

`OpenAICompatProvider(name, base_url=None, api_key_env=None, pricing=None, concurrency=8,
timeout_s=90.0, max_retries=2, self_hosted=None)`. Presets (`preset(name, **overrides)`): `hf`
(`https://router.huggingface.co/v1`, `HF_TOKEN`), `openai` (`https://api.openai.com/v1`, `OPENAI_API_KEY`), `vllm` (`base_url`
required, no key). A preset's `base_url`/`api_key_env` fill in when not given explicitly, so
`OpenAICompatProvider("hf", pricing={...})` is the HF router with prices. Presets ship with an
empty pricing table: open-model prices vary by served provider, so the experimenter supplies them
(an unpriced model is an error at `Experiment` construction).

Self-hosted serving (WP8, HF Jobs): `self_hosted` marks a provider whose calls are paid as compute
time (a GPU job running vLLM), not per token. `None` (the default) means "self-hosted iff the
name is `vllm`". A self-hosted provider is normally priced `(0, 0, 0)`, so the ledger shows $0
while still counting calls and tokens; reports label that $0 as compute-time
(`Run.summary()["self_hosted"]`). The flag is a constructor kwarg, so it is recorded in the run
spec (`providers.<prefix>.params.self_hosted`; left out when None, so older spec hashes hold). `served_by` falls back to the provider name
(`"vllm"`) when the server sends no `x-inference-provider` header and the provider is
self-hosted.

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
  `extra_body` (the OpenAI SDK's spelling) is flattened into the body too. Field names are the
  serving provider's: Cerebras (`hf:<model>:cerebras`) rejects `chat_template_kwargs` with HTTP
  400 (seen in the 2026-10-07 crowding pilot) and documents `reasoning_effort` (`none`, `low`,
  `medium`, `high`; default `high` for Qwen 3.8 27B, which therefore reasons unless told
  `"none"`). `swarmlab preflight` (swarmlab/preflight.py) sends one request with a spec's exact
  `extra` to check it is accepted.
- Response: `choices[0].message.content` (None -> ""), `tool_calls[].function.arguments` parsed as
  JSON; a non-JSON string becomes `{"_raw": <string>}` and `finish_reason = "bad_tool_args"`.
  Usage from `prompt_tokens`, `completion_tokens`, `prompt_tokens_details.cached_tokens`,
  `completion_tokens_details.reasoning_tokens`. `served_by` = the `x-inference-provider` header.
  `finish_reason == "content_filter"` (the server's safety filter declined) becomes
  `"refusal"` with `ChatResponse.refusal = Refusal()` (no category: the API gives none), so
  refusals read the same across providers. `ChatRequest.attempt` is never sent.
- Timeouts and retries (see `base.py`): each attempt is bounded by `timeout_s` (httpx timeout and
  an `asyncio.timeout` around the attempt). `httpx.TimeoutException`, the attempt deadline,
  other `httpx.TransportError`s, HTTP 429 and 5xx are retried up to `max_retries` times with
  jittered exponential backoff (1 s, 2 s, 4 s; `retry-after` honoured, capped at 60 s); then
  `ProviderError(status=..., attempts=...)`. Any other non-2xx raises `ProviderError` at once.
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
    DEFAULT_MAX_RETRIES,
    DEFAULT_TIMEOUT_S,
    ChatMessage,
    ChatRequest,
    ChatResponse,
    Provider,
    ProviderError,
    Refusal,
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
    stop = str(choice.get("finish_reason") or "")
    if stop == "content_filter":
        stop = "refusal"
    finish = stop if stop == "refusal" else "bad_tool_args" if bad else stop
    return msg.get("content") or "", calls, usage, finish


class OpenAICompatProvider(Provider):
    entry_point: ClassVar[str | None] = "openai_compat"

    def __init__(self, name: str, base_url: str | None = None, api_key_env: str | None = None,
                 pricing: dict | None = None, concurrency: int = 8,
                 timeout_s: float = DEFAULT_TIMEOUT_S, max_retries: int = DEFAULT_MAX_RETRIES,
                 self_hosted: bool | None = None) -> None:
        preset = PRESETS.get(name, {})
        self.self_hosted = (name == "vllm") if self_hosted is None else bool(self_hosted)
        if self_hosted is None:  # keep pre-WP8 specs (and their spec_hash) unchanged
            getattr(self, "params", {}).pop("self_hosted", None)
        self.name = name
        self.base_url = (base_url or preset.get("base_url") or "").rstrip("/")
        if not self.base_url:
            raise ValueError(f"provider {name!r} needs a base_url")
        self.api_key_env = api_key_env if api_key_env is not None else preset.get("api_key_env")
        self._setup(pricing, concurrency, timeout_s, max_retries)
        self._transport: httpx.AsyncBaseTransport | None = None

    def _headers(self) -> dict[str, str]:
        headers = {"content-type": "application/json"}
        key = os.environ.get(self.api_key_env) if self.api_key_env else None
        if key:
            headers["authorization"] = f"Bearer {key}"
        return headers

    async def complete(self, request: ChatRequest) -> ChatResponse:
        self.calls += 1
        body = build_body(request)
        start = time.monotonic()
        timed_out = False
        attempt = 0  # failed attempts so far
        async with httpx.AsyncClient(transport=self._transport, timeout=self.timeout_s) as client:
            while True:
                retry_after: str | None = None
                try:
                    async with asyncio.timeout(self.timeout_s):
                        resp = await client.post(f"{self.base_url}/chat/completions", json=body,
                                                 headers=self._headers())
                except (httpx.TimeoutException, TimeoutError) as e:
                    timed_out = True
                    failure = f"timeout after {self.timeout_s:g} s ({type(e).__name__})"
                    status = None
                    err: BaseException | None = e
                except httpx.TransportError as e:
                    failure = f"transport error: {type(e).__name__}: {e}"
                    status = None
                    err = e
                else:
                    if resp.status_code == 429 or resp.status_code >= 500:
                        failure = f"HTTP {resp.status_code}: {resp.text[:500]}"
                        status = resp.status_code
                        retry_after = resp.headers.get("retry-after")
                        err = None
                    elif resp.status_code >= 400:
                        raise ProviderError(f"{self.name}: HTTP {resp.status_code}: {resp.text[:500]}",
                                            status=resp.status_code, attempts=attempt + 1)
                    else:
                        break
                if attempt >= self.max_retries:
                    raise ProviderError(f"{self.name}: {failure} (gave up after {attempt + 1} "
                                        f"attempt(s))", status=status, attempts=attempt + 1) from err
                await asyncio.sleep(self._delay(attempt, retry_after))
                attempt += 1
        text, calls, usage, finish = parse_completion(resp.json())
        return ChatResponse(
            text=text, tool_calls=calls, usage=usage, cost_usd=self.cost(request, usage),
            provider=self.name, model=model_id(request),
            served_by=resp.headers.get("x-inference-provider")
            or (self.name if self.self_hosted else None),
            latency_s=time.monotonic() - start, finish_reason=finish,
            attempts=attempt + 1, retried_after_timeout=timed_out,
            refusal=Refusal() if finish == "refusal" else None,
        )


def preset(name: str, **overrides: Any) -> OpenAICompatProvider:
    if name not in PRESETS:
        raise ValueError(f"unknown OpenAI-compatible preset {name!r}; known: {sorted(PRESETS)}")
    return OpenAICompatProvider(name, **overrides)
