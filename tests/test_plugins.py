"""`swarmlab plugins` and `swarmlab.registry.catalog`: every built-in plugin is listed with a
one-line description (the guard that keeps the listing, and docs that defer to it, complete)."""
import json
import tomllib
from importlib.metadata import EntryPoint
from pathlib import Path

from typer.testing import CliRunner

from swarmlab import registry
from swarmlab.cli import app
from swarmlab.metrics.base import catalog as metrics_catalog

PYPROJECT = Path(__file__).resolve().parents[1] / "pyproject.toml"
UNLISTED = {"swarmlab.providers"}  # configured under `providers:`, not listed by `plugins`
runner = CliRunner()


def _pyproject_groups() -> dict[str, dict[str, str]]:
    eps = tomllib.loads(PYPROJECT.read_text())["project"]["entry-points"]
    return {g: dict(v) for g, v in eps.items() if g.startswith("swarmlab.")}


def test_every_pyproject_group_is_listed():
    assert set(_pyproject_groups()) - UNLISTED == set(registry.GROUPS)


def test_every_builtin_entry_point_has_a_description():
    missing = []
    for group, declared in _pyproject_groups().items():
        if group in UNLISTED:
            continue
        rows = {name: (desc, target) for name, desc, target in registry.catalog(group)}
        # the installed metadata matches pyproject (else re-run `uv sync`)
        assert {n: t for n, (_, t) in rows.items()} == declared, group
        missing += [f"{group}:{n}" for n, (d, _) in rows.items() if not d or "cannot load" in d]
    assert not missing, f"entry points without a one-line description: {missing}"
    for group in registry.GROUPS:
        for name, desc, _ in registry.catalog(group):
            assert "\n" not in desc and len(desc) <= 160, (group, name, desc)


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
