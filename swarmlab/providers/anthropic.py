"""AnthropicProvider: the official `anthropic` SDK, `AsyncAnthropic` (docs/INTERFACE-M1b.md §1).

The SDK is imported lazily (inside `_get_client` and `complete`), so the core package does not
depend on the `anthropic` extra.

Mapping (ChatRequest -> `messages.create` kwargs):

- `system` messages are joined (blank line between) into the top-level `system` parameter.
- `user` content: text parts -> `text` blocks, image parts -> `image` blocks with a base64
  `image/png` source. `assistant`: a `text` block (if non-empty) then one `tool_use` block per
  tool call (`id` = our `ToolCall.call_id`, `input` = args). `tool`: a `tool_result` block
  (`tool_use_id` = `tool_call_id`, content = the message's text). Consecutive messages that map to
  the same Anthropic role are merged into one message, so all tool results of one assistant turn go
  back in a single user message (parallel tool use) and a user message right after tool results
  joins them.
- Tools -> `{"name", "description", "input_schema"}` with the schema passed through
  `ToolSchema.normalized()` first (defensive: the harness already emits normalised schemas), plus
  `"strict": True` only when the normalised schema has no `strict_violations()` at any depth. (The
  2026-10-06 smoke got `400 tools.4.custom: For 'object' type, 'additionalProperties' must be
  explicitly set to false` because `post` was sent strict while its nested `fields: {"type":
  "object"}` was open; a schema with a free-form nested object is now sent non-strict.) `tool_choice` is always `{"type": "auto"}`
  (sent only when tools are present). `tool_protocol == "json"` sends no tools.
- `thinking_budget` -> `thinking={"type": "enabled", "budget_tokens": N}` only for models in
  `THINKING_BUDGET_MODELS` (Haiku 4.5); omitted otherwise. When thinking is enabled,
  `temperature`/`top_p` are not sent (the API does not accept sampling changes with thinking).
  `seed` has no Anthropic equivalent and is ignored. `extra` is merged into the kwargs last.
- Response: `text` = the text blocks joined; `tool_calls` from `tool_use` blocks (an input that
  arrives as a string is parsed with `json.loads`; non-JSON -> `{"_raw": ...}` and
  `finish_reason = "bad_tool_args"`); otherwise `finish_reason` = `stop_reason`.
  `usage.prompt_tokens = input_tokens + cache_read_input_tokens + cache_creation_input_tokens`,
  `cached_prompt_tokens = cache_read_input_tokens`, `completion_tokens = output_tokens`.
  `served_by` is None.
- Errors (typed chain, most specific first): `RateLimitError` -> retry with backoff (honouring
  `retry-after`, capped at 60 s) up to 3 times; other `APIStatusError` -> raise;
  `APIConnectionError` (including timeouts) -> retry up to 3 times. The SDK's own retries are
  disabled (`max_retries=0`) so this chain is the only one.
- API key: `ANTHROPIC_API_KEY`, else `ANTHROPIC_KEY`; with neither set the SDK's own credential
  resolution applies. The client is created per event loop (the runner uses `asyncio.run` per
  live/resume call). Tests replace `_client` with a stub; a stub set that way is always used.
"""
from __future__ import annotations

import asyncio
import os
import time
from typing import Any, ClassVar

from ..tools import ToolCall, ToolSchema
from .base import (
    ChatMessage,
    ChatRequest,
    ChatResponse,
    PricingRow,
    Provider,
    Usage,
    model_id,
    parse_json_args,
    text_of,
)

THINKING_BUDGET_MODELS = frozenset({"claude-haiku-4-5"})
MAX_RETRIES = 3


def tool_definition(schema: ToolSchema) -> dict:
    norm = schema.normalized()
    tool: dict[str, Any] = {"name": norm.name, "description": norm.description,
                            "input_schema": norm.parameters}
    if not norm.strict_violations():
        tool["strict"] = True
    return tool


def _blocks(m: ChatMessage) -> list[dict]:
    if m.role == "tool":
        return [{"type": "tool_result", "tool_use_id": m.tool_call_id or "",
                 "content": text_of(m.content)}]
    out: list[dict] = []
    if isinstance(m.content, str):
        if m.content:
            out.append({"type": "text", "text": m.content})
    else:
        for p in m.content:
            if p.type == "image":
                out.append({"type": "image", "source": {"type": "base64", "media_type": "image/png",
                                                        "data": p.image_png_b64 or ""}})
            elif p.text:
                out.append({"type": "text", "text": p.text})
    if m.role == "assistant":
        for tc in m.tool_calls or []:
            out.append({"type": "tool_use", "id": tc.call_id, "name": tc.name, "input": tc.args})
    return out


def build_kwargs(request: ChatRequest) -> dict:
    """The `messages.create` keyword arguments for `request` (pure; unit-tested)."""
    mid = model_id(request)
    system = "\n\n".join(text_of(m.content) for m in request.messages if m.role == "system")
    messages: list[dict] = []
    for m in request.messages:
        if m.role == "system":
            continue
        role = "assistant" if m.role == "assistant" else "user"
        blocks = _blocks(m)
        if messages and messages[-1]["role"] == role:
            messages[-1]["content"].extend(blocks)
        else:
            messages.append({"role": role, "content": blocks})
    kw: dict[str, Any] = {"model": mid, "max_tokens": request.max_tokens, "messages": messages}
    if system:
        kw["system"] = system
    if request.tools and request.tool_protocol == "native":
        kw["tools"] = [tool_definition(t) for t in request.tools]
        kw["tool_choice"] = {"type": "auto"}
    thinking = request.thinking_budget is not None and mid in THINKING_BUDGET_MODELS
    if thinking:
        kw["thinking"] = {"type": "enabled", "budget_tokens": int(request.thinking_budget or 0)}
    else:
        if request.temperature is not None:
            kw["temperature"] = request.temperature
        if request.top_p is not None:
            kw["top_p"] = request.top_p
    kw.update(request.extra)
    return kw


def _get(obj: Any, name: str, default: Any = None) -> Any:
    if isinstance(obj, dict):
        return obj.get(name, default)
    return getattr(obj, name, default)


def parse_message(message: Any) -> tuple[str, list[ToolCall], Usage, str]:
    """(text, tool_calls, usage, finish_reason) from an SDK `Message` (or a dict of the same shape)."""
    texts: list[str] = []
    calls: list[ToolCall] = []
    bad = False
    for block in _get(message, "content", []) or []:
        btype = _get(block, "type")
        if btype == "text":
            texts.append(_get(block, "text", ""))
        elif btype == "tool_use":
            args, ok = parse_json_args(_get(block, "input"))
            bad = bad or not ok
            calls.append(ToolCall(call_id=_get(block, "id"), name=_get(block, "name"), args=args))
    u = _get(message, "usage")
    cache_read = int(_get(u, "cache_read_input_tokens", 0) or 0)
    cache_write = int(_get(u, "cache_creation_input_tokens", 0) or 0)
    usage = Usage(
        prompt_tokens=int(_get(u, "input_tokens", 0) or 0) + cache_read + cache_write,
        completion_tokens=int(_get(u, "output_tokens", 0) or 0),
        cached_prompt_tokens=cache_read,
    )
    finish = "bad_tool_args" if bad else str(_get(message, "stop_reason", "") or "")
    return "".join(texts), calls, usage, finish


class AnthropicProvider(Provider):
    entry_point: ClassVar[str | None] = "anthropic"
    name = "anthropic"
    default_pricing: ClassVar[dict[str, PricingRow]] = {"claude-haiku-4-5": (1.00, 5.00, 0.10)}

    def __init__(self, pricing: dict | None = None, concurrency: int = 8,
                 timeout_s: float = 120.0) -> None:
        self._setup(pricing, concurrency)
        self.timeout_s = timeout_s
        self._client: Any = None
        self._client_loop: Any = None
        self._backoff_s = 1.0

    def _get_client(self) -> Any:
        loop = asyncio.get_running_loop()
        if self._client is not None and self._client_loop in (None, loop):
            return self._client
        import anthropic

        key = os.environ.get("ANTHROPIC_API_KEY") or os.environ.get("ANTHROPIC_KEY") or None
        self._client = anthropic.AsyncAnthropic(api_key=key, max_retries=0, timeout=self.timeout_s)
        self._client_loop = loop
        return self._client

    def _delay(self, attempt: int, retry_after: Any = None) -> float:
        try:
            if retry_after is not None:
                return min(float(retry_after), 60.0)
        except (TypeError, ValueError):
            pass
        return min(self._backoff_s * (2 ** attempt), 60.0)

    async def complete(self, request: ChatRequest) -> ChatResponse:
        import anthropic

        self.calls += 1
        client = self._get_client()
        kwargs = build_kwargs(request)
        start = time.monotonic()
        attempt = 0
        while True:
            try:
                message = await client.messages.create(**kwargs)
                break
            except anthropic.RateLimitError as e:
                if attempt >= MAX_RETRIES:
                    raise
                headers = getattr(getattr(e, "response", None), "headers", None) or {}
                await asyncio.sleep(self._delay(attempt, headers.get("retry-after")))
            except anthropic.APIStatusError:
                raise
            except anthropic.APIConnectionError:
                if attempt >= MAX_RETRIES:
                    raise
                await asyncio.sleep(self._delay(attempt))
            attempt += 1
        text, calls, usage, finish = parse_message(message)
        return ChatResponse(
            text=text, tool_calls=calls, usage=usage, cost_usd=self.cost(request, usage),
            provider=self.name, model=model_id(request), served_by=None,
            latency_s=time.monotonic() - start, finish_reason=finish,
        )
