"""Provider layer data types and base class (docs/INTERFACE-M1b.md §1).

Decisions where the contract is silent:

- `ChatRequest.model` is `"<prefix>:<model id>"`; `split_model(model)` splits on the first ":".
  Providers receive the whole request and read the model id with `model_id(request)`.
- `request_hash(request)` is SHA-256 of `canonical_json(request.model_dump(mode="json"))` (sorted
  keys, no whitespace; `extra` included). `request_bytes(request)` returns exactly those bytes, so
  storing them in the run's `BlobStore` gives a blob whose sha *is* the request hash.
- Pricing is `model id -> (usd per M prompt, per M completion, per M cached prompt)`. The key
  `"*"` is a wildcard used when the model id has no entry (the fake provider prices every model
  that way). `model_pricing(model_id)` raises `UnknownModelPricing` (a `ValueError`) otherwise.
  Tuples arriving as lists (from YAML or a spec round-trip) are normalised to tuples.
- `estimate_prompt_tokens`: ceil(chars / 4) over every text part and string content, tool-call
  arguments (canonical JSON), tool-message content and the tool schemas (canonical JSON), plus
  1000 per image part.
- `max_cost = (est * p_in + (max_tokens + (thinking_budget or 0)) * p_out) / 1e6` (the thinking
  budget is added even where the provider ignores it: worst case).
- `cost(request, usage) = ((prompt - cached) * p_in + cached * p_cached + completion * p_out) / 1e6`.
  `prompt_tokens` includes cached tokens; `reasoning_tokens` is informational and assumed to be
  part of `completion_tokens` (true for OpenAI-style usage and for Anthropic `output_tokens`).
- Providers are services, not snapshot state: the runner never deep-copies or pickles them.
  `calls` counts `complete()` invocations on the instance (tests use it).
- Timeouts and retries: `timeout_s` (default 90 s) bounds each attempt (the transport timeout
  plus an `asyncio.timeout` deadline around the whole attempt); `max_retries` (default 2, so at
  most 3 attempts) bounds retries of timeouts, transport errors, 429 and 5xx. Backoff is
  `backoff_s * 2**attempt` (1 s, 2 s, 4 s) times a uniform jitter in [0.8, 1.2], or the
  server's `retry-after` when given, capped at 60 s. Retries happen inside `complete()`, so
  the gate's reservation is held across them and released once. `ChatResponse.attempts` and
  `.retried_after_timeout` record what happened; exhausted retries raise `ProviderError` with
  `.attempts`. Both knobs are constructor kwargs of the real providers, so they are part of the
  provider spec (YAML `providers: {hf: {type: openai_compat, params: {name: hf, timeout_s: 60,
  max_retries: 3}}}`).
"""
from __future__ import annotations

import hashlib
import json
import math
import random
from typing import Any, ClassVar, Literal

from pydantic import BaseModel

from ..base import Plugin
from ..spec import canonical_json
from ..tools import ToolCall, ToolSchema
from ..view import Part

PricingRow = tuple[float, float, float]
DEFAULT_TIMEOUT_S = 90.0
DEFAULT_MAX_RETRIES = 2
MAX_BACKOFF_S = 60.0


class UnknownModelPricing(ValueError):
    """A provider has no pricing entry for a model (raised at Experiment construction)."""


class ProviderError(RuntimeError):
    """A provider returned a non-retryable error (or retries were exhausted)."""

    def __init__(self, message: str, *, status: int | None = None, attempts: int = 1) -> None:
        super().__init__(message)
        self.status = status
        self.attempts = attempts


class Usage(BaseModel):
    prompt_tokens: int = 0
    completion_tokens: int = 0
    cached_prompt_tokens: int = 0
    reasoning_tokens: int = 0

    def __add__(self, other: Usage) -> Usage:
        return Usage(**{k: getattr(self, k) + getattr(other, k) for k in Usage.model_fields})


class ChatMessage(BaseModel):
    role: Literal["system", "user", "assistant", "tool"]
    content: list[Part] | str
    tool_calls: list[ToolCall] | None = None  # assistant messages
    tool_call_id: str | None = None  # tool messages


class ChatRequest(BaseModel):
    model: str  # "<provider>:<model id>"
    messages: list[ChatMessage]
    tools: list[ToolSchema] = []
    tool_protocol: Literal["native", "json"] = "native"
    max_tokens: int = 1024
    temperature: float | None = None
    top_p: float | None = None
    seed: int | None = None
    thinking_budget: int | None = None
    extra: dict = {}


class ChatResponse(BaseModel):
    text: str
    tool_calls: list[ToolCall]
    usage: Usage
    cost_usd: float
    provider: str
    model: str
    served_by: str | None = None
    latency_s: float
    finish_reason: str
    cached: bool = False
    attempts: int = 1  # provider attempts this response took (1 = no retry)
    retried_after_timeout: bool = False  # at least one failed attempt was a timeout


def split_model(model: str) -> tuple[str, str]:
    prefix, sep, model_id = model.partition(":")
    if not sep or not prefix or not model_id:
        raise ValueError(f"model must look like '<provider>:<model id>', got {model!r}")
    return prefix, model_id


def model_id(request: ChatRequest) -> str:
    return split_model(request.model)[1]


def request_bytes(request: ChatRequest) -> bytes:
    return canonical_json(request.model_dump(mode="json")).encode()


def request_hash(request: ChatRequest) -> str:
    return hashlib.sha256(request_bytes(request)).hexdigest()


def text_of(content: list[Part] | str) -> str:
    """The text parts of a message joined by newlines (images dropped)."""
    if isinstance(content, str):
        return content
    return "\n".join(p.text or "" for p in content if p.type == "text")


def _normalise_pricing(pricing: dict | None) -> dict[str, PricingRow]:
    out: dict[str, PricingRow] = {}
    for k, v in (pricing or {}).items():
        row = tuple(float(x) for x in v)
        if len(row) != 3:
            raise ValueError(f"pricing for {k!r} must be (prompt, completion, cached prompt) per M")
        out[k] = row  # type: ignore[assignment]
    return out


class Provider(Plugin):
    """Base class: subclasses set `name` and `pricing` and implement `complete`."""

    name: str = ""
    pricing: dict[str, PricingRow]
    concurrency: int = 8
    timeout_s: float = DEFAULT_TIMEOUT_S
    max_retries: int = DEFAULT_MAX_RETRIES
    default_pricing: ClassVar[dict[str, PricingRow]] = {}
    _backoff_s: float = 1.0

    def _setup(self, pricing: dict | None, concurrency: int, timeout_s: float = DEFAULT_TIMEOUT_S,
               max_retries: int = DEFAULT_MAX_RETRIES) -> None:
        self.pricing = {**self.default_pricing, **_normalise_pricing(pricing)}
        self.concurrency = int(concurrency)
        self.timeout_s = float(timeout_s)
        self.max_retries = int(max_retries)
        if self.timeout_s <= 0 or self.max_retries < 0:
            raise ValueError("timeout_s must be > 0 and max_retries >= 0")
        self.calls = 0
        self._jitter = random.Random()  # operational only: never affects logical state

    def _delay(self, attempt: int, retry_after: Any = None) -> float:
        """Seconds to wait before retry number `attempt + 1` (attempt counts from 0)."""
        try:
            if retry_after is not None:
                return min(max(float(retry_after), 0.0), MAX_BACKOFF_S)
        except (TypeError, ValueError):
            pass
        jitter = getattr(self, "_jitter", None) or random.Random()
        return min(self._backoff_s * (2 ** attempt) * jitter.uniform(0.8, 1.2), MAX_BACKOFF_S)

    async def complete(self, request: ChatRequest) -> ChatResponse:
        raise NotImplementedError

    # ---- pricing -----------------------------------------------------------------------------
    def model_pricing(self, model: str) -> PricingRow:
        """Pricing row for a bare model id (or a full "<prefix>:<id>" string)."""
        if ":" in model and model.split(":", 1)[0] == self.name:
            model = model.split(":", 1)[1]
        row = self.pricing.get(model) or self.pricing.get("*")
        if row is None:
            raise UnknownModelPricing(
                f"provider {self.name!r} has no pricing for model {model!r}; known: "
                f"{sorted(self.pricing)}. Pass pricing={{...}} to the provider (Experiment.providers)."
            )
        return row

    def estimate_prompt_tokens(self, request: ChatRequest) -> int:
        chars = 0
        images = 0
        for m in request.messages:
            if isinstance(m.content, str):
                chars += len(m.content)
            else:
                for p in m.content:
                    if p.type == "image":
                        images += 1
                    else:
                        chars += len(p.text or "")
            for tc in m.tool_calls or []:
                chars += len(tc.name) + len(canonical_json(tc.args))
        if request.tools:
            chars += len(canonical_json([t.model_dump(mode="json") for t in request.tools]))
        return math.ceil(chars / 4) + 1000 * images

    def max_cost(self, request: ChatRequest) -> float:
        p_in, p_out, _ = self.model_pricing(model_id(request))
        out_tokens = request.max_tokens + (request.thinking_budget or 0)
        return (self.estimate_prompt_tokens(request) * p_in + out_tokens * p_out) / 1e6

    def cost(self, request: ChatRequest, usage: Usage) -> float:
        p_in, p_out, p_cached = self.model_pricing(model_id(request))
        cached = min(usage.cached_prompt_tokens, usage.prompt_tokens)
        return ((usage.prompt_tokens - cached) * p_in + cached * p_cached
                + usage.completion_tokens * p_out) / 1e6


def parse_json_args(raw: Any) -> tuple[dict, bool]:
    """Tool arguments as a dict: (args, ok). Non-JSON or non-object input -> ({"_raw": raw}, False)."""
    if isinstance(raw, dict):
        return raw, True
    if raw is None or raw == "":
        return {}, True
    try:
        value = json.loads(raw)
    except (TypeError, ValueError):
        return {"_raw": raw}, False
    if isinstance(value, dict):
        return value, True
    return {"_raw": raw}, False
