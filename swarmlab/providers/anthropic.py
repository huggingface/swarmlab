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
- Tools -> `{"name", "description", "input_schema"}` with the schema exactly as the participant
  gives it (no re-normalisation: what a participant offers is what the model sees; the executor
  and the world normalise the schemas they emit, so harness tools are unchanged on the wire; a
  participant that wants strict mode normalises its own schema), plus `"strict": True` only when
  that schema has no `strict_violations()` at any depth, so a schema without
  `additionalProperties: false` is sent non-strict. (The
  2026-10-06 smoke got `400 tools.4.custom: For 'object' type, 'additionalProperties' must be
  explicitly set to false` because `post` was sent strict while its nested `fields: {"type":
  "object"}` was open; a schema with a free-form nested object is now sent non-strict.) `tool_choice` is `{"type": "auto"}` (sent only when tools
  are present), or `{"type": "tool", "name": <name>}` when `ChatRequest.tool_choice` names a tool
  (checked by `base.check_tool_choice`: the tool must be offered and the protocol native). Forced
  tool choice is rejected with HTTP 400 by Claude Opus 5.5, Sonnet 5.5 and Fable 5.1 (the error
  propagates; no model list is kept here) and by any model with extended thinking on, so a request
  with both `tool_choice` and a `thinking_budget` that this adapter turns into thinking raises
  `ValueError` before sending. `tool_protocol == "json"` sends no tools.
- `thinking_budget` -> `thinking={"type": "enabled", "budget_tokens": N}` only for models in
  `THINKING_BUDGET_MODELS` (Haiku 4.5); omitted otherwise. When thinking is enabled,
  `temperature`/`top_p` are not sent (the API does not accept sampling changes with thinking).
  `seed` has no Anthropic equivalent and is ignored. `extra` is merged into the kwargs last.
- Response: `text` = the text blocks joined; `tool_calls` from `tool_use` blocks (an input that
  arrives as a string is parsed with `json.loads`; non-JSON -> `{"_raw": ...}` and
  `finish_reason = "bad_tool_args"`); otherwise `finish_reason` = `stop_reason`.
- Refusals (`parse_refusal`): `stop_reason == "refusal"` (a safety classifier declined; HTTP 200)
  gives `finish_reason = "refusal"` (it wins over `bad_tool_args`: the content may be a partial
  reply cut off mid-stream) and `ChatResponse.refusal = Refusal(category, explanation)` from
  `stop_details`. The API fills `stop_details` only on refusals and may leave it null even then,
  so it is read only when `stop_reason == "refusal"`, and a null one gives `Refusal()` (both
  fields None). `ChatRequest.attempt` is never sent. The server-side `fallbacks` option is not
  used (passed through `extra` it would silently serve the request from another model).
  `usage.prompt_tokens = input_tokens + cache_read_input_tokens + cache_creation_input_tokens`,
  `cached_prompt_tokens = cache_read_input_tokens`, `cache_write_tokens =
  cache_creation_input_tokens`, `completion_tokens = output_tokens`
  (thinking included), `reasoning_tokens = output_tokens_details.thinking_tokens` when the API
  reports it (the tokens of the raw reasoning, not of the possibly shorter summary returned).
  `served_by` is None.
- Reasoning (`parse_reasoning`): `thinking` blocks' text joined by blank lines is
  `ChatResponse.reasoning`, with `reasoning_kind` `"summary"` when any has text (`thinking:
  {type: adaptive, display: summarized}` in `extra`) and `"omitted"` when all are empty (the
  default display on Claude Opus 5.5 / Sonnet 5.5 / Haiku 5.5 / Fable; reasoning `""`);
  `redacted_thinking` blocks are counted in `reasoning_redacted`. The raw chain of thought is
  never returned. When the response has any thinking block, `provider_content` holds every
  content block in order as a request block (`text`: type, text; `thinking`: type, thinking,
  signature; `redacted_thinking`: type, data; `tool_use`: type, id, name, input; other types:
  their fields minus None), and an assistant message carrying `provider_content` is sent as
  exactly those blocks instead of text + tool calls. Passing thinking blocks back unchanged is
  what the API asks for in tool loops (preserved thinking on Opus 5.5, Sonnet 5.5 and Fable 5.1:
  the model reads its earlier reasoning; leaving the blocks out is accepted but loses it). The
  blocks are bound to the history before them, so a caller that trims history must drop them
  (`LLMAgent` does, see its module doc).
- Timeouts and retries (see `base.py`): the SDK client gets `timeout=timeout_s` and
  `max_retries=0`, and each attempt also runs under `asyncio.timeout(timeout_s)`. Our own loop
  retries `APITimeoutError` (and the attempt deadline), `APIConnectionError`, `RateLimitError`
  (honouring `retry-after`) and 5xx `APIStatusError` up to `max_retries` times with the same
  jittered backoff as the OpenAI-compatible adapter; exhausted retries raise
  `ProviderError(status=..., attempts=...)` chained to the last SDK error. Other
  `APIStatusError`s (4xx) propagate unchanged at once.
- Pricing: `default_pricing` is the catalog's `ANTHROPIC_MODELS` table. For a model with a
  prompt-length tier (`catalog.ANTHROPIC_LONG_PROMPT`: Haiku 5.5 over 100,000 prompt tokens),
  `cost` prices a request whose `usage.prompt_tokens` (cache reads and writes included, as the
  API counts it) is over the threshold entirely at the long row, and `max_cost` does the same with
  the estimated prompt tokens. The tier applies only while the model's row is the built-in one
  (a `pricing:` override for that model replaces both). Both hand the tier row to `base.usage_cost` /
  `base.max_request_cost` together with the cache-write factor (next bullet).
- Prompt caching (provider params `prompt_cache`, `cache_ttl`; docs/INTERFACE-M1b.md §1
  amendment). `prompt_cache="prefix"` (the default; `True` means the same) sends two cache
  breakpoints: an explicit `cache_control` marker on the system text (`system` then goes as one
  text block `[{"type": "text", "text": <joined system>, "cache_control": {...}}]` instead of a
  string), which caches the static prefix tools -> system, and the top-level `cache_control`
  kwarg of `messages.create` (automatic caching: the API puts the breakpoint on the last
  cacheable block and moves it forward as the conversation grows), which caches the history.
  `"auto"` sends only the top-level one; `"none"` (or `False`) sends neither, the behaviour
  before caching existed. A request without system text gets only the top-level breakpoint.
  Both markers are `{"type": "ephemeral", "ttl": cache_ttl}` with `cache_ttl` `"5m"` (default)
  or `"1h"`: the same TTL on both, so the API's two 400s (all 4 breakpoint slots taken, or a
  last-block marker whose TTL differs from the top-level one) and its ordering rule (a
  longer-TTL marker must come before a shorter one) cannot be broken by what this adapter
  sends. `extra` is merged after them, so a `cache_control` (or `system`) in `extra` still
  wins; the effective settings are resolved once per request after that merge
  (`cache_settings`): the body-level `cache_control` dict that ends up on the wire is also put
  on the system block in `"prefix"` mode, so both markers carry the TTL actually sent (a 1h
  override yields 1h on both, never 5m before 1h; the alternative, rejecting the override,
  would make a valid request an error). Precedence, highest first, is what the wire body ends
  up with: `extra["extra_body"]["cache_control"]` (the SDK merges `extra_body` into the
  serialised body after the kwargs, so it beats the kwarg), then `extra["cache_control"]`, then
  the provider's own marker; a shadowed marker is not sent, so it is neither reserved nor
  billed. Markers `extra` writes elsewhere (its own `system` blocks, tools) are sent
  as given, and their order is the caller's to get right. Why on by default: it changes bills,
  never behaviour (the API returns the same output with or without a cache hit), and every
  swarmlab request shape benefits (a fixed system prompt and tool list per agent, a history that
  only grows within a turn and, under `memory="full"`, across turns); a prefix shorter than the
  model's minimum (4096 tokens on Haiku 4.5, 512 on the 5.5 models) silently does not cache and
  costs nothing extra. The cost of being wrong is the write premium on a prefix that is never read
  again (0.25x of its input price at 5 minutes). The setting is a provider param, not a
  `ChatRequest` field, so `request_hash` and the record/replay cache are the same with it on or
  off; the params are kept out of `spec()` while at their defaults, so a spec written before they
  existed hashes as before. Pricing: writes are billed from the response's per-TTL split,
  `usage.cache_creation.ephemeral_5m_input_tokens` at 1.25x and `ephemeral_1h_input_tokens` at 2x
  the input price, passed to `Provider.cost` as their token-weighted factor
  (`response_write_factor`); without the split, at the factor of the TTLs this request sent (the
  largest, if several), or, when it sent no marker, `write_factor`, 1.25 for `cache_ttl "5m"` and
  2.0 for `"1h"` whatever the mode. `max_cost` (the gate's reservation) prices every prompt token
  as a write at the largest factor among the TTLs the request sends (`max_write_factor`), and at
  1x when it sends no marker (`"none"` and nothing in `extra`). The top-level `cache_control` is
  passed as a keyword of `messages.create` (a typed parameter in the locked SDK, anthropic
  1.11.0), not through `extra_body`, which would collide with an `extra_body` in `extra`.
- API key: `ANTHROPIC_API_KEY`, else `ANTHROPIC_KEY`; with neither set the SDK's own credential
  resolution applies. The client is created per event loop (the runner uses `asyncio.run` per
  live/resume call). Tests replace `_client` with a stub; a stub set that way is always used.
"""
from __future__ import annotations

import asyncio
import os
import time
from collections.abc import Iterator
from typing import Any, ClassVar, NamedTuple

from ..tools import ToolCall, ToolSchema
from .base import (
    DEFAULT_MAX_RETRIES,
    DEFAULT_TIMEOUT_S,
    ChatMessage,
    ChatRequest,
    ChatResponse,
    PricingRow,
    Provider,
    ProviderError,
    Refusal,
    Usage,
    check_tool_choice,
    max_request_cost,
    model_id,
    parse_json_args,
    text_of,
    usage_cost,
)
from .catalog import ANTHROPIC_LONG_PROMPT, ANTHROPIC_MODELS

THINKING_BUDGET_MODELS = frozenset({"claude-haiku-4-5"})
PROMPT_CACHE_MODES = ("none", "auto", "prefix")
DEFAULT_PROMPT_CACHE = "prefix"
CACHE_WRITE_FACTORS = {"5m": 1.25, "1h": 2.0}  # cache-write price / input price, per TTL
DEFAULT_CACHE_TTL = "5m"


def prompt_cache_mode(value: bool | str) -> str:
    """`prompt_cache` as one of `PROMPT_CACHE_MODES` (`True` -> `"prefix"`, `False` -> `"none"`)."""
    if value is True:
        return DEFAULT_PROMPT_CACHE
    if value is False:
        return "none"
    if value not in PROMPT_CACHE_MODES:
        raise ValueError(f"prompt_cache must be one of {list(PROMPT_CACHE_MODES)} or a bool, "
                         f"got {value!r}")
    return str(value)


def check_cache_ttl(ttl: str) -> str:
    if ttl not in CACHE_WRITE_FACTORS:
        raise ValueError(f"cache_ttl must be one of {list(CACHE_WRITE_FACTORS)}, got {ttl!r}")
    return ttl


def _ttl(marker: dict) -> str:
    return str(marker.get("ttl", DEFAULT_CACHE_TTL))  # the API's default TTL is 5 minutes


def _markers(value: Any) -> Iterator[dict]:
    """Every `cache_control` dict at any depth of `value` (an `extra` mapping)."""
    if isinstance(value, dict):
        for k, v in value.items():
            if k == "cache_control" and isinstance(v, dict):
                yield v
            else:
                yield from _markers(v)
    elif isinstance(value, (list, tuple)):
        for v in value:
            yield from _markers(v)


def _body_overrides(extra: dict) -> dict:
    """The top-level request-body keys `extra` sends: its own keys (merged into the
    `messages.create` kwargs) overlaid by `extra["extra_body"]`, which the SDK merges into the
    serialised body after the kwargs (shallowly), so its keys are what goes on the wire."""
    body = extra.get("extra_body")
    if not isinstance(body, dict):
        return extra
    return {**{k: v for k, v in extra.items() if k != "extra_body"}, **body}


class CacheSettings(NamedTuple):
    """The cache settings of one request after merging `extra` (module doc)."""

    marker: dict | None  # what this adapter's own breakpoints carry; None in "none" mode
    ttls: frozenset[str]  # the TTL of every cache_control marker sent

    @property
    def write_factor(self) -> float:
        """The largest cache-write factor among `ttls` (an unknown TTL counts as the largest);
        1.0 when no marker is sent."""
        top = max(CACHE_WRITE_FACTORS.values())
        return max((CACHE_WRITE_FACTORS.get(t, top) for t in self.ttls), default=1.0)


def cache_settings(request: ChatRequest, prompt_cache: bool | str = "none",
                   cache_ttl: str = DEFAULT_CACHE_TTL) -> CacheSettings:
    """What `build_kwargs` sends for caching: the body-level `cache_control` dict `extra` ends
    up sending (`extra["extra_body"]`'s over `extra`'s own) replaces the provider's marker on both
    breakpoints; any marker `extra` sends counts."""
    mode, ttl = prompt_cache_mode(prompt_cache), check_cache_ttl(cache_ttl)
    body = _body_overrides(request.extra)
    ttls = {_ttl(m) for m in _markers(body)}
    if mode == "none":
        return CacheSettings(None, frozenset(ttls))
    own = {"type": "ephemeral", "ttl": ttl}
    top = body.get("cache_control", own)
    marker = dict(top) if isinstance(top, dict) else own
    ttls.add(_ttl(marker))  # top level (extra's, already counted) or system block, or both
    return CacheSettings(marker, frozenset(ttls))


def response_write_factor(message: Any, fallback: float) -> float:
    """The cache-write factor of a response: its 5m / 1h write tokens weighted by 1.25 / 2.0
    (`usage.cache_creation`), or `fallback` when the response has no such split."""
    split = _get(_get(message, "usage"), "cache_creation")
    w5 = int(_get(split, "ephemeral_5m_input_tokens", 0) or 0)
    w1 = int(_get(split, "ephemeral_1h_input_tokens", 0) or 0)
    if w5 + w1 <= 0:
        return fallback
    return (w5 * CACHE_WRITE_FACTORS["5m"] + w1 * CACHE_WRITE_FACTORS["1h"]) / (w5 + w1)


def tool_definition(schema: ToolSchema) -> dict:
    tool: dict[str, Any] = {"name": schema.name, "description": schema.description,
                            "input_schema": schema.parameters}
    if not schema.strict_violations():
        tool["strict"] = True
    return tool


def _blocks(m: ChatMessage) -> list[dict]:
    if m.role == "assistant" and m.provider_content is not None:
        return [dict(b) for b in m.provider_content]  # verbatim: thinking signatures bind order
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


def build_kwargs(request: ChatRequest, *, prompt_cache: bool | str = "none",
                 cache_ttl: str = DEFAULT_CACHE_TTL) -> dict:
    """The `messages.create` keyword arguments for `request` (pure; unit-tested).

    `prompt_cache` / `cache_ttl` add the cache breakpoints (module doc); the default here is no
    caching, the plain mapping. `AnthropicProvider.complete` passes the provider's settings."""
    mode = prompt_cache_mode(prompt_cache)
    marker = cache_settings(request, mode, cache_ttl).marker
    mid = model_id(request)
    forced = check_tool_choice(request)
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
        kw["tool_choice"] = {"type": "tool", "name": forced} if forced else {"type": "auto"}
    thinking = request.thinking_budget is not None and mid in THINKING_BUDGET_MODELS
    if thinking and forced:
        raise ValueError(f"tool_choice ({forced!r}) cannot be combined with thinking on Anthropic "
                         "(the API rejects forced tool use with extended thinking); drop "
                         "thinking_budget or tool_choice")
    if thinking:
        kw["thinking"] = {"type": "enabled", "budget_tokens": int(request.thinking_budget or 0)}
    else:
        if request.temperature is not None:
            kw["temperature"] = request.temperature
        if request.top_p is not None:
            kw["top_p"] = request.top_p
    if marker is not None:
        if mode == "prefix" and system:
            kw["system"] = [{"type": "text", "text": system, "cache_control": dict(marker)}]
        kw["cache_control"] = dict(marker)
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
        reasoning_tokens=int(_get(_get(u, "output_tokens_details"), "thinking_tokens", 0) or 0),
        cache_write_tokens=cache_write,
    )
    stop = str(_get(message, "stop_reason", "") or "")
    finish = stop if stop == "refusal" else "bad_tool_args" if bad else stop
    return "".join(texts), calls, usage, finish


def _request_block(block: Any) -> dict:
    """One response content block as the request block that sends it back unchanged."""
    btype = _get(block, "type")
    if btype == "text":
        return {"type": "text", "text": _get(block, "text", "")}
    if btype == "thinking":
        return {"type": "thinking", "thinking": _get(block, "thinking", "") or "",
                "signature": _get(block, "signature", "") or ""}
    if btype == "redacted_thinking":
        return {"type": "redacted_thinking", "data": _get(block, "data", "") or ""}
    if btype == "tool_use":
        return {"type": "tool_use", "id": _get(block, "id"), "name": _get(block, "name"),
                "input": _get(block, "input")}
    raw = block if isinstance(block, dict) else block.model_dump(mode="json")
    return {k: v for k, v in raw.items() if v is not None}


def parse_reasoning(message: Any) -> dict:
    """`ChatResponse` reasoning fields from an SDK `Message` (module doc); {} without thinking."""
    blocks = list(_get(message, "content", []) or [])
    thoughts = [_get(b, "thinking", "") or "" for b in blocks if _get(b, "type") == "thinking"]
    redacted = sum(1 for b in blocks if _get(b, "type") == "redacted_thinking")
    if not thoughts and not redacted:
        return {}
    text = "\n\n".join(t for t in thoughts if t)
    return {"reasoning": text, "reasoning_kind": "summary" if text else "omitted",
            "reasoning_redacted": redacted,
            "provider_content": [_request_block(b) for b in blocks]}


def parse_refusal(message: Any) -> Refusal | None:
    """`Refusal` from `stop_details` when `stop_reason == "refusal"`, else None (module doc)."""
    if _get(message, "stop_reason") != "refusal":
        return None
    details = _get(message, "stop_details")
    if details is None:
        return Refusal()
    category, explanation = _get(details, "category"), _get(details, "explanation")
    return Refusal(category=None if category is None else str(category),
                   explanation=None if explanation is None else str(explanation))


class AnthropicProvider(Provider):
    entry_point: ClassVar[str | None] = "anthropic"
    name = "anthropic"
    default_pricing: ClassVar[dict[str, PricingRow]] = dict(ANTHROPIC_MODELS)

    def __init__(self, pricing: dict | None = None, concurrency: int = 8,
                 timeout_s: float = DEFAULT_TIMEOUT_S, max_retries: int = DEFAULT_MAX_RETRIES,
                 prompt_cache: bool | str = DEFAULT_PROMPT_CACHE,
                 cache_ttl: str = DEFAULT_CACHE_TTL) -> None:
        self._setup(pricing, concurrency, timeout_s, max_retries)
        self.prompt_cache = prompt_cache_mode(prompt_cache)
        self.cache_ttl = check_cache_ttl(cache_ttl)
        self.write_factor = CACHE_WRITE_FACTORS[self.cache_ttl]
        params = getattr(self, "params", {})
        if self.prompt_cache == DEFAULT_PROMPT_CACHE:  # defaults stay out of the spec (hashes)
            params.pop("prompt_cache", None)
        if self.cache_ttl == DEFAULT_CACHE_TTL:
            params.pop("cache_ttl", None)
        self._client: Any = None
        self._client_loop: Any = None

    # ---- pricing (prompt-length tiers) ---------------------------------------------------------
    def _row_for(self, model: str, prompt_tokens: int) -> PricingRow:
        row = self.model_pricing(model)
        mid = model.split(":", 1)[1] if model.startswith(f"{self.name}:") else model
        tier = ANTHROPIC_LONG_PROMPT.get(mid)
        if tier is not None and row == ANTHROPIC_MODELS.get(mid) and prompt_tokens > tier[0]:
            return tier[1]
        return row

    def _cache_settings(self, request: ChatRequest) -> CacheSettings:
        return cache_settings(request, self.prompt_cache, self.cache_ttl)

    def max_write_factor(self, request: ChatRequest) -> float:
        """The largest write factor among the TTLs `request` sends; 1.0 when it sends none."""
        return self._cache_settings(request).write_factor

    def max_cost(self, request: ChatRequest) -> float:
        # the prompt-length tier row for the estimated prompt, every token possibly a cache write
        prompt = self.estimate_prompt_tokens(request)
        return max_request_cost(request, prompt, self._row_for(model_id(request), prompt),
                                self.max_write_factor(request))

    def cost(self, request: ChatRequest, usage: Usage, write_factor: float | None = None) -> float:
        # the prompt-length tier row (long prompts) with the cache-write premium (base.usage_cost)
        factor = self.write_factor if write_factor is None else write_factor
        return usage_cost(usage, self._row_for(model_id(request), usage.prompt_tokens), factor)

    def _get_client(self) -> Any:
        loop = asyncio.get_running_loop()
        if self._client is not None and self._client_loop in (None, loop):
            return self._client
        import anthropic

        key = os.environ.get("ANTHROPIC_API_KEY") or os.environ.get("ANTHROPIC_KEY") or None
        self._client = anthropic.AsyncAnthropic(api_key=key, max_retries=0, timeout=self.timeout_s)
        self._client_loop = loop
        return self._client

    async def complete(self, request: ChatRequest) -> ChatResponse:
        import anthropic

        self.calls += 1
        client = self._get_client()
        kwargs = build_kwargs(request, prompt_cache=self.prompt_cache, cache_ttl=self.cache_ttl)
        start = time.monotonic()
        timed_out = False
        attempt = 0  # failed attempts so far
        while True:
            retry_after: Any = None
            status: int | None = None
            try:
                async with asyncio.timeout(self.timeout_s):
                    message = await client.messages.create(**kwargs)
                break
            except anthropic.APIStatusError as e:
                status = e.status_code
                if not (isinstance(e, anthropic.RateLimitError) or status >= 500):
                    raise
                headers = getattr(getattr(e, "response", None), "headers", None) or {}
                retry_after = headers.get("retry-after")
                failure: str = f"HTTP {status}: {e}"
                err: BaseException = e
            except (anthropic.APITimeoutError, TimeoutError) as e:
                timed_out = True
                failure, err = f"timeout after {self.timeout_s:g} s ({type(e).__name__})", e
            except anthropic.APIConnectionError as e:
                failure, err = f"connection error: {e}", e
            if attempt >= self.max_retries:
                raise ProviderError(f"{self.name}: {failure} (gave up after {attempt + 1} "
                                    f"attempt(s))", status=status, attempts=attempt + 1) from err
            await asyncio.sleep(self._delay(attempt, retry_after))
            attempt += 1
        text, calls, usage, finish = parse_message(message)
        settings = self._cache_settings(request)
        fallback = settings.write_factor if settings.ttls else self.write_factor
        factor = response_write_factor(message, fallback)
        return ChatResponse(
            text=text, tool_calls=calls, usage=usage, cost_usd=self.cost(request, usage, factor),
            provider=self.name, model=model_id(request), served_by=None,
            latency_s=time.monotonic() - start, finish_reason=finish,
            attempts=attempt + 1, retried_after_timeout=timed_out, refusal=parse_refusal(message),
            **parse_reasoning(message),
        )
