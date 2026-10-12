"""Provider registry (docs/INTERFACE-M1b.md §1).

`resolve(model, overrides=None) -> (Provider, model_id)` splits `"<prefix>:<model id>"` on the first
":" and returns `overrides[prefix]` when present, else a preset built on demand: `anthropic` ->
`AnthropicProvider()`, `hf`/`openai`/`vllm` -> `OpenAICompatProvider(prefix)` (vllm raises
`ValueError` without a base url: pass an override), `fake` -> `FakeProvider()`. An unknown prefix
is a `ValueError`. Each call builds a fresh preset; callers that need one instance per prefix
(the runner's gate, `Experiment`) cache what they resolve. Provider plugins are also registered
under the `swarmlab.providers` entry-point group (`anthropic`, `openai_compat`, `fake`) for YAML.

Pricing from the catalog (`catalog.py`): for the `hf` prefix, `resolve` fills a missing price for
the requested model from the HF router listing (`ensure_pricing`), so `hf:Org/Model:served_by`
works without a hand-typed price. A price given explicitly (provider `pricing=` / YAML
`providers:`) always wins. Models priced this way are remembered on the provider
(`catalog_priced(provider)`), so `Experiment` can pin the price into the run spec. When no price
can be found, `ensure_pricing` raises `UnknownModelPricing` naming the `providers:` override and
`swarmlab models`.
"""
from __future__ import annotations

from collections.abc import Mapping

from .base import (
    ChatMessage,
    ChatRequest,
    ChatResponse,
    PricingRow,
    Provider,
    ProviderError,
    Refusal,
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
    provider = overrides[prefix] if overrides and prefix in overrides else preset(prefix)
    try:
        ensure_pricing(provider, prefix, model_id)
    except UnknownModelPricing:
        pass  # raised where the price is needed (Experiment construction)
    return provider, model_id


def catalog_priced(provider: Provider) -> set[str]:
    """Model ids whose price `ensure_pricing` took from the catalog for this provider instance."""
    return set(getattr(provider, "_catalog_priced", ()))


def ensure_pricing(provider: Provider, prefix: str, model_id: str) -> PricingRow:
    """The provider's price for `model_id`, filled from the catalog for `hf` when missing."""
    try:
        return provider.model_pricing(model_id)
    except UnknownModelPricing:
        pass
    row = None
    if prefix == "hf":
        from .catalog import hf_pricing

        row = hf_pricing(model_id)
    if row is None:
        raise UnknownModelPricing(unknown_pricing_message(prefix, model_id))
    provider.pricing[model_id] = row
    priced = getattr(provider, "_catalog_priced", None)
    if priced is None:
        priced = set()
        provider._catalog_priced = priced  # type: ignore[attr-defined]
    priced.add(model_id)
    return row


def unknown_pricing_message(prefix: str, model_id: str) -> str:
    search = model_id.split(":", 1)[0].split("/")[-1]
    where = {"hf": "the HF router listing (offline, or the model/served_by is not listed there)",
             "anthropic": "the built-in Anthropic table"}.get(prefix, "any built-in table")
    ptype = "anthropic" if prefix == "anthropic" else "openai_compat"
    name = "" if prefix == "anthropic" else f"name: {prefix}, "
    return (
        f"no price for model {prefix}:{model_id}: not found in {where}. "
        f"List models with prices: `swarmlab models --search {search}`. Or give the price "
        f"(USD per million prompt, completion, cached-prompt tokens) in the spec: "
        f"`providers: {{{prefix}: {{type: {ptype}, params: {{{name}pricing: "
        f"{{\"{model_id}\": [0.10, 0.15, 0.10]}}}}}}}}` "
        f"(Python: `Experiment(providers={{...}})`)."
    )


__all__ = [
    "PREFIXES",
    "ChatMessage",
    "ChatRequest",
    "ChatResponse",
    "Provider",
    "ProviderError",
    "Refusal",
    "UnknownModelPricing",
    "Usage",
    "catalog_priced",
    "ensure_pricing",
    "preset",
    "request_hash",
    "resolve",
    "split_model",
]
