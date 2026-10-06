"""Keep the suite offline: the model catalog never fetches and caches under a temp dir."""
import pytest


@pytest.fixture(autouse=True)
def _offline_catalog(tmp_path_factory, monkeypatch):
    monkeypatch.setenv("SWARMLAB_OFFLINE", "1")
    monkeypatch.setenv("SWARMLAB_CACHE_DIR", str(tmp_path_factory.mktemp("swarmlab-cache")))
