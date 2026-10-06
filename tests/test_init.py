"""`swarmlab init`: the starter files validate and run."""
import json
import runpy

from typer.testing import CliRunner

from swarmlab import Experiment, Run
from swarmlab.cli import app
from swarmlab.spec import experiment_seeds, load_experiment_yaml

runner = CliRunner()


def test_init_writes_valid_yaml_and_python(tmp_path):
    r = runner.invoke(app, ["init", "demo", "--dir", str(tmp_path)])
    assert r.exit_code == 0, r.output
    yaml_path, py_path = tmp_path / "demo.yaml", tmp_path / "demo.py"
    assert yaml_path.exists() and py_path.exists()
    doc = load_experiment_yaml(yaml_path)
    assert list(doc["arms"]) == ["broadcast", "gossip"] and experiment_seeds(doc) == [1, 2]
    assert doc["budget"] == {"soft_usd": 0.0, "hard_usd": 0.0, "measurement_usd": 0.0}
    for arm in doc["arms"].values():
        (group,) = arm["participants"]
        assert group["type"] == "llm" and group["count"] == 8
        assert group["params"]["model"] == "fake:reader"
        assert [p["type"] for p in arm["probes"]] == ["belief"]
    r = runner.invoke(app, ["validate", str(yaml_path), "--json"])
    assert r.exit_code == 0, r.output
    # refuses to overwrite; --force does
    assert runner.invoke(app, ["init", "demo", "--dir", str(tmp_path)]).exit_code == 1
    assert runner.invoke(app, ["init", "demo", "--dir", str(tmp_path), "--force"]).exit_code == 0
    assert runner.invoke(app, ["init", "9bad name", "--dir", str(tmp_path)]).exit_code == 2


def test_init_yaml_runs_and_python_mirrors_it(tmp_path, monkeypatch):
    runner.invoke(app, ["init", "demo", "--dir", str(tmp_path)])
    out = tmp_path / "runs"
    r = runner.invoke(app, ["run", str(tmp_path / "demo.yaml"), "--max-rounds", "2", "--out", str(out),
                            "--json"])
    assert r.exit_code == 0, r.output
    data = json.loads(r.stdout)
    assert [x["run_id"] for x in data["runs"]] == [
        "demo__broadcast__s1", "demo__broadcast__s2", "demo__gossip__s1", "demo__gossip__s2"]
    assert all(x["outcome"] == "ran" and x["end_reason"] == "max_rounds" for x in data["runs"])

    # the Python file builds the same arms (same spec hash as the YAML at the same options)
    monkeypatch.syspath_prepend(str(tmp_path))
    ns = runpy.run_path(str(tmp_path / "demo.py"))  # not __main__: builds, does not run
    for arm, exp in ns["ARMS"].items():
        from_yaml = Experiment.from_yaml(tmp_path / "demo.yaml", arm)
        assert exp.spec_hash(1, 2) == from_yaml.spec_hash(1, 2)
        assert exp.run_id(1) == f"demo__{arm}__s1"
    # and its main block runs (here: every run already exists with the same spec -> skipped)
    monkeypatch.chdir(tmp_path)
    ns["MAX_ROUNDS"] = 2
    exp = ns["ARMS"]["gossip"]
    runs = exp.run_all([1, 2], 2, out=out)
    assert [r.id for r in runs] == ["demo__gossip__s1", "demo__gossip__s2"]
    assert Run.load(out, "demo__gossip__s1").summary()["end_reason"] == "max_rounds"
