"""Model catalog: what can be called, through whom, and at what price (`swarmlab models`).

Sources:

- `hf`: the Hugging Face router listing `GET https://router.huggingface.co/v1/models` (no auth).
  Each model lists its inference providers (`served_by`) with `status`, `context_length`,
  `supports_tools` and `pricing: {"input", "output"}` in USD per million tokens. One
  `CatalogEntry` per (model, served_by); a provider without `pricing` gets `input=output=None`.
- `anthropic`: the static table `ANTHROPIC_MODELS` below. Only models documented in `docs/` are
  listed (today: Claude Haiku 4.5 at 1.00 / 5.00 / 0.10 USD per M prompt / completion / cached).

Decisions where the contract is silent:

- The router listing is cached as `{"fetched_at": <unix s>, "listing": <raw JSON>}` in
  `$SWARMLAB_CACHE_DIR/catalog.json` (default `$XDG_CACHE_HOME/swarmlab` or
  `~/.cache/swarmlab`) for 24 h. A failed fetch falls back to a stale cache. `SWARMLAB_OFFLINE=1`
  (or `offline=True`) never fetches and uses whatever cache exists, however old.
- `hf_pricing(model_id)` prices an `hf` model id as a `(prompt, completion, cached prompt)` row:
  `Org/Model:served_by` uses that provider's listed price; `Org/Model:cheapest` the lowest listed
  (by input + output); a bare `Org/Model` or `Org/Model:fastest` (where the router picks the
  provider) the highest listed, so a budget computed from it is a worst case. The router lists no
  cached-input price, so cached prompt tokens are priced at the input price. Returns None when the
  model or provider is not listed or has no price.
"""
from __future__ import annotations

import json
import os
import time
from collections.abc import Iterable
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

ROUTER_URL = "https://router.huggingface.co/v1/models"
TTL_S = 24 * 3600

# Static Anthropic table: model id -> (USD per M prompt, completion, cached prompt).
ANTHROPIC_MODELS: dict[str, tuple[float, float, float]] = {
    "claude-haiku-4-5": (1.00, 5.00, 0.10),
}


@dataclass(frozen=True)
class CatalogEntry:
    provider: str  # swarmlab prefix: "hf" or "anthropic"
    model: str  # model id without prefix or served_by
    served_by: str | None
    input: float | None  # USD per M prompt tokens
    output: float | None  # USD per M completion tokens
    cached: float | None  # USD per M cached prompt tokens
    tools: bool | None  # supports native tool calling, when the source says
    context_length: int | None = None
    status: str | None = None

    @property
    def spec_id(self) -> str:
        """The string to put in a spec's `model:` field."""
        tail = f":{self.served_by}" if self.served_by else ""
        return f"{self.provider}:{self.model}{tail}"

    def as_dict(self) -> dict[str, Any]:
        return {**asdict(self), "id": self.spec_id}


# ---- cache -----------------------------------------------------------------------------------
def cache_dir() -> Path:
    if os.environ.get("SWARMLAB_CACHE_DIR"):
        return Path(os.environ["SWARMLAB_CACHE_DIR"])
    base = os.environ.get("XDG_CACHE_HOME") or str(Path.home() / ".cache")
    return Path(base) / "swarmlab"


def cache_path() -> Path:
    return cache_dir() / "catalog.json"


def _offline_env() -> bool:
    return os.environ.get("SWARMLAB_OFFLINE", "").strip().lower() in ("1", "true", "yes")


def _read_cache() -> dict | None:
    try:
        data = json.loads(cache_path().read_text())
    except (OSError, ValueError):
        return None
    return data if isinstance(data, dict) and "listing" in data else None


def write_cache(listing: dict) -> None:
    path = cache_path()
    try:
        path.parent.mkdir(parents=True, exist_ok=True)
        tmp = path.with_suffix(".tmp")
        tmp.write_text(json.dumps({"fetched_at": time.time(), "listing": listing}))
        tmp.replace(path)
    except OSError:
        pass  # a read-only home must not break pricing


def fetch_router(timeout_s: float = 15.0) -> dict:
    """One GET of the router listing (raises on network or HTTP errors)."""
    import httpx

    resp = httpx.get(ROUTER_URL, timeout=timeout_s)
    resp.raise_for_status()
    data = resp.json()
    if not isinstance(data, dict) or not isinstance(data.get("data"), list):
        raise ValueError(f"{ROUTER_URL}: unexpected listing shape")  # noqa: TRY004
    return data


def router_listing(*, refresh: bool = False, offline: bool | None = None) -> dict | None:
    """The router listing: fresh cache, else a fetch, else a stale cache, else None."""
    offline = _offline_env() if offline is None else offline
    cached = _read_cache()
    fresh = cached is not None and time.time() - float(cached.get("fetched_at", 0)) < TTL_S
    if cached is not None and (offline or (fresh and not refresh)):
        return cached["listing"]
    if offline:
        return None
    try:
        listing = fetch_router()
    except Exception:  # noqa: BLE001 - any failure falls back to the stale cache
        return cached["listing"] if cached is not None else None
    write_cache(listing)
    return listing


# ---- parsing ---------------------------------------------------------------------------------
def _num(v: Any) -> float | None:
    return float(v) if isinstance(v, (int, float)) and not isinstance(v, bool) else None


def parse_router(listing: dict) -> list[CatalogEntry]:
    """One entry per (model, served_by) of a router listing."""
    out: list[CatalogEntry] = []
    for m in listing.get("data") or []:
        mid = m.get("id")
        if not isinstance(mid, str):
            continue
        for p in m.get("providers") or []:
            price = p.get("pricing") or {}
            p_in, p_out = _num(price.get("input")), _num(price.get("output"))
            out.append(CatalogEntry(
                provider="hf", model=mid, served_by=p.get("provider"),
                input=p_in, output=p_out, cached=p_in,
                tools=p.get("supports_tools") if isinstance(p.get("supports_tools"), bool) else None,
                context_length=p.get("context_length"), status=p.get("status"),
            ))
    return out


def anthropic_entries() -> list[CatalogEntry]:
    return [CatalogEntry(provider="anthropic", model=m, served_by=None, input=r[0], output=r[1],
                         cached=r[2], tools=True)
            for m, r in ANTHROPIC_MODELS.items()]


def entries(provider: str | None = None, *, tools: bool = False, search: str | None = None,
            refresh: bool = False, offline: bool | None = None,
            listing: dict | None = None) -> list[CatalogEntry]:
    """Catalog rows, filtered. `listing` bypasses the cache/fetch (tests)."""
    rows: list[CatalogEntry] = []
    if provider in (None, "anthropic"):
        rows += anthropic_entries()
    if provider in (None, "hf"):
        if listing is None:
            listing = router_listing(refresh=refresh, offline=offline)
        rows += parse_router(listing or {})
    if tools:
        rows = [r for r in rows if r.tools]
    if search:
        needle = search.lower()
        rows = [r for r in rows if needle in r.spec_id.lower()]
    return rows


# ---- pricing ---------------------------------------------------------------------------------
def _priced(rows: Iterable[CatalogEntry]) -> list[CatalogEntry]:
    return [r for r in rows if r.input is not None and r.output is not None]


def hf_pricing(model_id: str, *, listing: dict | None = None,
               offline: bool | None = None) -> tuple[float, float, float] | None:
    """`(prompt, completion, cached)` USD per M for an `hf` model id, or None (see module doc)."""
    if listing is None:
        listing = router_listing(offline=offline)
    if not listing:
        return None
    model, _, suffix = model_id.partition(":")
    rows = [r for r in parse_router(listing) if r.model == model]
    if suffix and suffix not in ("cheapest", "fastest"):
        rows = [r for r in rows if r.served_by == suffix]
    rows = _priced(rows)
    if not rows:
        return None
    def key(r: CatalogEntry) -> float:
        return (r.input or 0.0) + (r.output or 0.0)

    pick = min(rows, key=key) if suffix == "cheapest" else max(rows, key=key)
    p_in, p_out = pick.input or 0.0, pick.output or 0.0
    return (p_in, p_out, pick.cached if pick.cached is not None else p_in)
