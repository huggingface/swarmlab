"""Model catalog (swarmlab/providers/catalog.py) and catalog pricing in `resolve`/`Experiment`."""
import json
import time
from pathlib import Path

import pytest

from swarmlab import Experiment
from swarmlab.participants import LLMAgent
from swarmlab.providers import UnknownModelPricing, catalog, catalog_priced, resolve
from swarmlab.providers.openai_compat import OpenAICompatProvider
from swarmlab.worlds import FlagGame

FIXTURE = Path(__file__).parent / "fixtures" / "router_models.json"


@pytest.fixture
def listing():
    return json.loads(FIXTURE.read_text())


@pytest.fixture
def cached(listing):
    catalog.write_cache(listing)  # conftest points the cache at a temp dir and sets offline
    return listing


def test_parse_router_listing(listing):
    rows = catalog.parse_router(listing)
    by_id = {r.spec_id: r for r in rows}
    di = by_id["hf:Qwen/Qwen3.5-9B:deepinfra"]
    assert (di.input, di.output, di.cached) == (0.1, 0.15, 0.1)
    assert di.tools is True and di.context_length == 262144 and di.status == "live"
    unpriced = by_id["hf:Qwen/Qwen3.5-9B:featherless-ai"]
    assert unpriced.input is None and unpriced.tools is None
    assert {r.model for r in rows} == {"Qwen/Qwen3.5-9B", "Qwen/Qwen3-8B",
                                       "deepseek-ai/DeepSeek-V4.1-Flash"}


def test_entries_filters(listing):
    rows = catalog.entries(tools=True, search="qwen3.5", listing=listing)
    assert rows and all(r.tools and "qwen3.5" in r.spec_id.lower() for r in rows)
    anth = catalog.entries("anthropic", listing=listing)
    assert [(r.spec_id, r.input, r.output, r.cached) for r in anth] == [
        ("anthropic:claude-haiku-4-5", 1.0, 5.0, 0.1)]
    assert all(r.provider == "hf" for r in catalog.entries("hf", listing=listing))


def test_hf_pricing_rules(listing):
    price = lambda m: catalog.hf_pricing(m, listing=listing)
    assert price("Qwen/Qwen3.5-9B:deepinfra") == (0.1, 0.15, 0.1)
    assert price("Qwen/Qwen3.5-9B") == (0.17, 0.25, 0.17)          # router picks: worst case
    assert price("Qwen/Qwen3.5-9B:fastest") == (0.17, 0.25, 0.17)
    assert price("Qwen/Qwen3.5-9B:cheapest") == (0.1, 0.15, 0.1)
    assert price("Qwen/Qwen3.5-9B:featherless-ai") is None          # listed without a price
    assert price("Qwen/Qwen3.5-9B:nowhere") is None
    assert price("Qwen/Unknown-1B") is None


def test_cache_offline_stale_and_fresh(listing, monkeypatch):
    assert catalog.router_listing() is None  # offline, empty cache
    catalog.write_cache(listing)
    assert catalog.router_listing() == listing
    # stale cache + failing fetch: falls back to the stale copy
    data = json.loads(catalog.cache_path().read_text())
    data["fetched_at"] = time.time() - 2 * catalog.TTL_S
    catalog.cache_path().write_text(json.dumps(data))
    monkeypatch.setenv("SWARMLAB_OFFLINE", "0")

    def boom(timeout_s=15.0):
        raise OSError("no network")

    monkeypatch.setattr(catalog, "fetch_router", boom)
    assert catalog.router_listing() == listing
    # a successful fetch refreshes the cache
    fresh = {"object": "list", "data": []}
    monkeypatch.setattr(catalog, "fetch_router", lambda timeout_s=15.0: fresh)
    assert catalog.router_listing() == fresh
    assert json.loads(catalog.cache_path().read_text())["listing"] == fresh


def test_resolve_fills_hf_price_from_catalog(cached):
    provider, mid = resolve("hf:Qwen/Qwen3.5-9B:deepinfra")
    assert mid == "Qwen/Qwen3.5-9B:deepinfra"
    assert provider.model_pricing(mid) == (0.1, 0.15, 0.1)
    assert catalog_priced(provider) == {mid}
    # an explicit price wins over the catalog
    mine = OpenAICompatProvider("hf", pricing={mid: [1, 2, 3]})
    assert resolve("hf:" + mid, {"hf": mine})[0].model_pricing(mid) == (1.0, 2.0, 3.0)
    assert catalog_priced(mine) == set()


def test_experiment_pins_catalog_price_into_spec(cached):
    exp = Experiment(name="x", world=FlagGame(),
                     participants=[LLMAgent(model="hf:Qwen/Qwen3.5-9B:deepinfra")] * 2)
    spec = exp.to_spec(seed=1, max_rounds=2)
    assert spec.providers["hf"].type == "openai_compat"
    assert spec.providers["hf"].params["pricing"] == {"Qwen/Qwen3.5-9B:deepinfra": [0.1, 0.15, 0.1]}
    # the rebuilt experiment no longer needs the catalog
    catalog.cache_path().unlink()
    again = Experiment.from_spec(spec)
    assert again.providers["hf"].model_pricing("Qwen/Qwen3.5-9B:deepinfra") == (0.1, 0.15, 0.1)
    est = exp.estimate(seed=1, max_rounds=1, calls_per_turn=1)
    assert est["usd"] == pytest.approx(2 * (3000 * 0.1 + 300 * 0.15) / 1e6)


def test_unknown_price_message_names_override_and_command(cached):
    with pytest.raises(UnknownModelPricing) as e:
        Experiment(name="x", world=FlagGame(), participants=[LLMAgent(model="hf:Qwen/Unknown-1B")])
    msg = str(e.value)
    assert "providers:" in msg and "swarmlab models" in msg and "Qwen/Unknown-1B" in msg

