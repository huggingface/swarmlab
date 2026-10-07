"""`module:Class` plugins in a file next to the spec resolve without PYTHONPATH (field notes 1)."""
import os
import sys

import pytest

from swarmlab import Experiment, Run

from .test_cli_run_all import invoke

WORLD = '''
from swarmlab import Outcome, Participant, World, text_observation, tool


class Counter(World):
    def reset(self, rng, agents):
        self.total = 0

    def observe(self, agent):
        return text_observation(f"total so far: {self.total}")

    @tool("add", "Add n to the shared total", {"n": "integer"})
    def add(self, agent, n: int) -> Outcome:
        self.total += n
        return Outcome(accepted=True, feedback={"added": n})

    def score(self):
        return {"total": self.total}


class Adder(Participant):
    def __init__(self, step: int = 1):
        self.step = step

    async def turn(self, view, tools):
        await tools.call("add", {"n": self.step})
'''

SPEC = """\
name: counting
options: {max_rounds: 3}
arms:
  A:
    world: {type: "myworld_fn1:Counter"}
    participants: [{type: "myworld_fn1:Adder", count: 2, params: {step: 2}}]
"""


@pytest.fixture
def project(tmp_path, monkeypatch):
    proj = tmp_path / "proj"
    proj.mkdir()
    (proj / "myworld_fn1.py").write_text(WORLD)
    (proj / "exp.yaml").write_text(SPEC)
    elsewhere = tmp_path / "elsewhere"
    elsewhere.mkdir()
    monkeypatch.chdir(elsewhere)
    monkeypatch.setattr(sys, "path", [p for p in sys.path if p not in ("", ".", os.getcwd())])
    monkeypatch.delenv("PYTHONPATH", raising=False)
    sys.modules.pop("myworld_fn1", None)
    yield proj
    sys.modules.pop("myworld_fn1", None)


def test_cli_puts_the_spec_dir_on_sys_path(project, tmp_path):
    res, data = invoke("validate", project / "exp.yaml", "--json")
    assert res.exit_code == 0, res.output
    assert data["arms"]["A"]["world"] == "myworld_fn1:Counter"
    assert sys.path[0] == str(project.resolve())
    out = tmp_path / "runs"
    res, data = invoke("run", project / "exp.yaml", "--seed", 0, "--out", out, "--json")
    assert res.exit_code == 0, res.output
    assert data["score"] == {"total": 12} and data["end_reason"] == "max_rounds"
    run_dir = out / "counting__A__s0"
    # a later process replays the run dir from anywhere: run.json records the spec dir
    sys.path.remove(str(project.resolve()))
    sys.modules.pop("myworld_fn1", None)
    res, data = invoke("replay", run_dir, "--json")
    assert res.exit_code == 0, res.output
    assert Run(run_dir).meta["import_dir"] == str(project.resolve())


def test_from_yaml_puts_the_spec_dir_on_sys_path(project, tmp_path):
    exp = Experiment.from_yaml(project / "exp.yaml", "A")
    assert type(exp.world).__name__ == "Counter"
    assert exp.run(seed=0, out=tmp_path / "r").score == {"total": 12}
