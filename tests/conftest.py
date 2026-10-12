"""Keep the suite offline: the model catalog never fetches and caches under a temp dir.

`research_plugin` installs an external plugin distribution for one test: a temporary
`research_plugin-0.1.dist-info` on `sys.path` whose entry point `research_custom` (group
`swarmlab.topologies`) points at `research_topology:ResearchTopology`.
"""
import importlib
import sys
import textwrap

import pytest


@pytest.fixture(autouse=True)
def _offline_catalog(tmp_path_factory, monkeypatch):
    monkeypatch.setenv("SWARMLAB_OFFLINE", "1")
    monkeypatch.setenv("SWARMLAB_CACHE_DIR", str(tmp_path_factory.mktemp("swarmlab-cache")))


@pytest.fixture
def research_plugin(tmp_path, monkeypatch):
    info = tmp_path / "research_plugin-0.1.dist-info"
    info.mkdir()
    (info / "METADATA").write_text("Metadata-Version: 2.1\nName: research_plugin\nVersion: 0.1\n")
    (info / "entry_points.txt").write_text(
        "[swarmlab.topologies]\nresearch_custom = research_topology:ResearchTopology\n")
    (tmp_path / "research_topology.py").write_text(textwrap.dedent("""\
        from swarmlab.medium.base import Topology


        class ResearchTopology(Topology):
            description = "A researcher-provided topology"

            def recipients(self, post, agents, round, rng):
                return []
        """))
    monkeypatch.syspath_prepend(str(tmp_path))
    importlib.invalidate_caches()
    yield tmp_path
    sys.modules.pop("research_topology", None)
    importlib.invalidate_caches()
