"""`swarmlab plugins` and `swarmlab.registry.catalog`: every built-in plugin is listed with a
one-line description (the guard that keeps the listing, and docs that defer to it, complete)."""
import json
import tomllib
from importlib.metadata import EntryPoint, distribution
from pathlib import Path

from typer.testing import CliRunner

from swarmlab import registry
from swarmlab.cli import app
from swarmlab.medium.base import Topology
from swarmlab.metrics.base import catalog as metrics_catalog
from swarmlab.scheduler import Scheduler

PYPROJECT = Path(__file__).resolve().parents[1] / "pyproject.toml"
UNLISTED = {"swarmlab.providers"}  # configured under `providers:`, not listed by `plugins`
runner = CliRunner()


def _pyproject_groups() -> dict[str, dict[str, str]]:
    eps = tomllib.loads(PYPROJECT.read_text())["project"]["entry-points"]
    return {g: dict(v) for g, v in eps.items() if g.startswith("swarmlab.")}


def test_every_pyproject_group_is_listed():
    assert set(_pyproject_groups()) - UNLISTED == set(registry.GROUPS)


def _own_entry_points() -> dict[str, dict[str, str]]:
    """This distribution's installed entry points (other installed plugins are left out)."""
    out: dict[str, dict[str, str]] = {}
    for ep in distribution("swarmlab").entry_points:
        if ep.group.startswith("swarmlab."):
            out.setdefault(ep.group, {})[ep.name] = ep.value
    return out


def test_every_builtin_entry_point_has_a_description():
    # the installed metadata matches pyproject (else re-run `uv sync`)
    assert _own_entry_points() == _pyproject_groups()
    missing = []
    for group, declared in _pyproject_groups().items():
        if group in UNLISTED:
            continue
        rows = {name: (desc, target) for name, desc, target in registry.catalog(group)}
        # catalog lists every built-in (and any external plugin installed alongside)
        assert {n: rows[n][1] for n in declared if n in rows} == declared, group
        missing += [f"{group}:{n}" for n in declared
                    if not rows[n][0] or "cannot load" in rows[n][0]]
        for name in declared:
            assert "\n" not in rows[name][0] and len(rows[name][0]) <= 160, (group, name)
    assert not missing, f"entry points without a one-line description: {missing}"


def test_external_plugin_is_listed_and_built(research_plugin):
    rows = {name: (desc, target) for name, desc, target in registry.catalog("topologies")}
    assert rows["research_custom"] == ("A researcher-provided topology",
                                       "research_topology:ResearchTopology")
    topo = registry.build({"type": "research_custom"}, "swarmlab.topologies")
    assert type(topo).__name__ == "ResearchTopology"
    # the built-in guard still holds with an external plugin installed
    test_every_pyproject_group_is_listed()
    test_every_builtin_entry_point_has_a_description()


def test_describe_falls_back_to_the_docstring():
    class Documented(Topology):
        """Sends to nobody, documented.

        More detail."""

    class Explicit(Topology):
        """Docstring loses."""

        description = "The attribute wins"

    class Bare(Topology):
        pass

    assert registry.describe(Documented) == "Sends to nobody, documented"
    assert registry.describe(Explicit) == "The attribute wins"
    assert registry.describe(Bare) == ""


def test_base_scheduler_is_described():
    assert registry.describe(Scheduler)


def test_catalog_short_group_name_and_metrics_alias():
    assert registry.catalog("topologies") == registry.catalog("swarmlab.topologies")
    assert metrics_catalog() == [(n, d) for n, d, _ in registry.catalog("metrics")]


def test_catalog_lists_a_broken_plugin(monkeypatch):
    broken = EntryPoint("broken", "swarmlab_no_such_module:Thing", "swarmlab.worlds")
    monkeypatch.setattr(registry, "entry_points", lambda group: [broken])
    [(name, desc, target)] = registry.catalog("worlds")
    assert (name, target) == ("broken", "swarmlab_no_such_module:Thing")
    assert desc.startswith("(cannot load: ModuleNotFoundError")


def test_describe_uses_docstring_without_a_description_attribute():
    def script(request, rng):
        """First line.

        More."""

    assert registry.describe(script) == "First line"
    from swarmlab.world.flaggame import FlagGame

    assert registry.describe(FlagGame)  # World.description() is a method: docstring is used


def test_plugins_command_text():
    res = runner.invoke(app, ["plugins"])
    assert res.exit_code == 0, res.output
    for group in registry.GROUPS:
        short = group.removeprefix("swarmlab.")
        assert f"{short} ({group})" in res.output
    assert "  flaggame  " in res.output and "  belief.accuracy  " in res.output


def test_plugins_command_json_and_group_filter():
    res = runner.invoke(app, ["plugins", "--json"])
    assert res.exit_code == 0, res.output
    data = json.loads(res.output)
    assert data["ok"] is True
    assert list(data["plugins"]) == [g.removeprefix("swarmlab.") for g in registry.GROUPS]
    tree = next(p for p in data["plugins"]["topologies"] if p["name"] == "tree")
    assert tree["target"] == "swarmlab.medium.topology:Tree" and tree["description"]

    res = runner.invoke(app, ["plugins", "schedulers", "--json"])
    assert res.exit_code == 0, res.output
    data = json.loads(res.output)
    assert list(data["plugins"]) == ["schedulers"]
    assert {p["name"] for p in data["plugins"]["schedulers"]} >= {"seeded_shuffle", "one_speaker"}

    res = runner.invoke(app, ["plugins", "schedulers"])
    assert res.output.startswith("schedulers (swarmlab.schedulers)\n")
    assert "topologies" not in res.output


def test_plugins_command_unknown_group_exits_2():
    res = runner.invoke(app, ["plugins", "nope", "--json"])
    assert res.exit_code == 2
    assert "unknown plugin group" in json.loads(res.stdout)["error"]


def test_metrics_command_still_lists_metrics():
    res = runner.invoke(app, ["metrics", "--json"])
    assert res.exit_code == 0, res.output
    names = {m["name"] for m in json.loads(res.output)["metrics"]}
    assert "belief.accuracy" in names
    assert "swarmlab plugins" in runner.invoke(app, ["metrics"]).output
