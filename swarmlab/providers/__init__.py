"""Provider registry (docs/INTERFACE-M1b.md §1).

`resolve(model, overrides=None) -> (Provider, model_id)` splits `"<prefix>:<model id>"` on the first
":" and returns `overrides[prefix]` when present, else a preset built on demand: `anthropic` ->
`AnthropicProvider()`, `hf`/`openai`/`vllm` -> `OpenAICompatProvider(prefix)` (vllm raises
`ValueError` without a base url: pass an override), `fake` -> `FakeProvider()`. An unknown prefix
is a `ValueError`. Each call builds a fresh preset; callers that need one instance per prefix
(the runner's gate, `Experiment`) cache what they resolve. Provider plugins are also registered
under the `swarmlab.providers` entry-point group (`anthropic`, `openai_compat`, `fake`) for YAML.
"""
from __future__ import annotations

from collections.abc import Mapping

from .base import (
    ChatMessage,
    ChatRequest,
    ChatResponse,
    Provider,
    ProviderError,
    UnknownModelPricing,
    Usage,
    request_hash,
    split_model,
)

PREFIXES = ("anthropic", "hf", "openai", "vllm", "fake")


def preset(prefix: str) -> Provider:
    if prefix == "anthropic":
        from .anthropic import AnthropicProvider

        return AnthropicProvider()
    if prefix in ("hf", "openai", "vllm"):
        from .openai_compat import preset as compat_preset

        return compat_preset(prefix)
    if prefix == "fake":
        from .fake import FakeProvider

        return FakeProvider()
    raise ValueError(f"unknown provider prefix {prefix!r}; known: {list(PREFIXES)}")


def resolve(model: str, overrides: Mapping[str, Provider] | None = None) -> tuple[Provider, str]:
    prefix, model_id = split_model(model)
    if overrides and prefix in overrides:
        return overrides[prefix], model_id
    return preset(prefix), model_id


__all__ = [
    "PREFIXES",
    "ChatMessage",
    "ChatRequest",
    "ChatResponse",
    "Provider",
    "ProviderError",
    "UnknownModelPricing",
    "Usage",
    "preset",
    "request_hash",
    "resolve",
    "split_model",
]
