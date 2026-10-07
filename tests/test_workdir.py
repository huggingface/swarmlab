"""Runs stay out of the swarmlab checkout; `init` says where to work; `validate` shows each arm."""
import json
from pathlib import Path

from swarmlab.cli import is_swarmlab_checkout

from .test_cli import invoke

REPO = Path(__file__).resolve().parents[1]
EXAMPLE = REPO / "examples" / "flaggame_m1a.yaml"


def fake_checkout(path: Path) -> Path:
    path.mkdir(parents=True, exist_ok=True)
    (path / "pyproject.toml").write_text('[project]\nname = "swarmlab"\nversion = "0"\n')
    return path


def test_checkout_marker(tmp_path):
    assert is_swarmlab_checkout(REPO)
    assert not is_swarmlab_checkout(tmp_path)
    (tmp_path / "pyproject.toml").write_text('[project]\nname = "my-exp"\n')
    assert not is_swarmlab_checkout(tmp_path)
    (tmp_path / "pyproject.toml").write_text("not toml [")
    assert not is_swarmlab_checkout(tmp_path)


def test_run_refuses_to_write_runs_into_the_checkout(tmp_path, monkeypatch):
    repo = fake_checkout(tmp_path / "swarmlab")
    monkeypatch.chdir(repo)
    res, _ = invoke("run", EXAMPLE, "--arm", "gossip", "--seed", 1, "--max-rounds", 1)
    assert res.exit_code == 2 and "swarmlab repo checkout" in res.stderr and "--out" in res.stderr
    assert not (repo / "runs").exists()
    res, _ = invoke("run", EXAMPLE, "--arm", "gossip", "--seed", 1, "--max-rounds", 1, "--out", "runs")
    assert res.exit_code == 0, res.output  # an explicit --out is the user's choice
    project = tmp_path / "my-exp"
    project.mkdir()
    monkeypatch.chdir(project)
    res, _ = invoke("run", EXAMPLE, "--arm", "gossip", "--seed", 1, "--max-rounds", 1)
    assert res.exit_code == 0 and (project / "runs" / "flaggame-m1a__gossip__s1").is_dir()


def test_init_points_outside_the_checkout(tmp_path, monkeypatch):
    repo = fake_checkout(tmp_path / "swarmlab")
    monkeypatch.chdir(repo)
    res, data = invoke("init", "demo", "--json")
    assert res.exit_code == 0 and data["inside_checkout"] is True
    assert "project directory outside it" in data["text"] and "uv run --project" in data["text"]
    monkeypatch.chdir(tmp_path)
    res, data = invoke("init", "demo2", "--json")
    assert data["inside_checkout"] is False and "note:" not in data["text"]


def test_validate_shows_each_arm(tmp_path):
    res, _ = invoke("validate", EXAMPLE)
    assert res.exit_code == 0, res.output
    lines = res.stdout.splitlines()
    assert lines[1].split()[:5] == ["arm", "agents", "model(s)", "soft", "hard"]
    llm = next(ln for ln in lines if ln.startswith("llm "))
    assert "fake:reader x8" in llm and "$1" in llm and "$0.25" in llm and "belief " in llm
    assert any(ln.startswith("gossip ") and "evidence_aggregator (scripted) x16" in ln for ln in lines)
    res, data = invoke("validate", EXAMPLE, "--json")
    arm = data["arms"]["llm"]
    assert arm["models"] == {"fake:reader": 8} and arm["probes"] == ["belief"]
    assert arm["budget"]["hard_usd"] == 1.0 and data["total_usd"] == 0
    assert json.dumps(data)
