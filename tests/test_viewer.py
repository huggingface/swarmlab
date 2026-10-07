"""The static replay page (docs/INTERFACE.md §17)."""
import json
import re

from swarmlab import Board, Experiment, Run
from swarmlab.participants import EvidenceAggregator
from swarmlab.viewer.build import build, collect
from swarmlab.worlds import FlagGame

from .test_api import Adder, Counter


def page_data(path):
    m = re.search(r'<script type="application/json" id="swarmlab-data">(.*?)</script>',
                  path.read_text(), re.DOTALL)
    return json.loads(m.group(1))


def test_flaggame_view(tmp_path):
    exp = Experiment(name="v", world=FlagGame(), participants=[EvidenceAggregator()] * 16,
                     medium=Board(topology="broadcast"),
                     metrics=["belief.consensus", "belief.accuracy", "comm.read_rate"])
    run = exp.run(seed=5, max_rounds=20, out=tmp_path)
    path = build(run.dir)
    assert path == run.dir / "view.html" and path.exists()
    html = path.read_text()
    assert path.stat().st_size < 3 * 1024 * 1024
    assert run.id in html
    assert not re.search(r'(src|href)=["\']https?:', html)     # self-contained
    data = page_data(path)
    assert data["meta"]["run_id"] == run.id and data["meta"]["last_round"] == 20
    assert data["meta"]["spec_hash"] == run.meta["spec_hash"]
    committed = {e["round"] for e in data["events"] if e["type"] == "round_committed"}
    assert committed == set(range(1, 21))
    for r in range(1, 21):
        types = {e["type"] for e in data["events"] if e["round"] == r}
        assert {"round_started", "turn_started", "tool_called", "turn_ended", "metric"} <= types
    assert data["world"]["kind"] == "flaggame" and len(data["world"]["candidates"]) == 8
    assert set(data["world"]["crops"]) == set(data["agents"]) and len(data["agents"]) == 16
    assert data["truth"] == run.score["truth"]
    deliveries = [e for e in data["events"] if e["type"] == "delivery"]
    assert deliveries and all(d["content"].startswith("crop:") for d in deliveries)
    assert len(deliveries) == 16 * 15


def test_partial_round_is_dropped_and_generic_world(tmp_path):
    run = Experiment(name="count", world=Counter(), participants=[Adder()] * 2).run(
        seed=0, max_rounds=3, out=tmp_path)
    log = run.dir / "events.jsonl"
    lines = log.read_text().splitlines()
    # simulate a crash in round 4: drop run_ended, add an uncommitted round_started
    lines = [ln for ln in lines if '"type":"run_ended"' not in ln]
    lines.append(json.dumps({"seq": len(lines), "run": run.id, "round": 4, "agent": None, "ts": 0.0,
                             "type": "round_started", "order": ["a000", "a001"]}))
    log.write_text("\n".join(lines) + "\n")
    data = collect(run.dir)
    assert data["world"] is None and data["truth"] is None
    assert max(e["round"] for e in data["events"]) == 3
    assert Run(run.dir).view().exists()


def test_world_state_panel_for_coloring_and_a_custom_world(tmp_path):
    """Field notes item 11: `World.render_state()` per committed round, from the snapshots."""
    from pathlib import Path

    from .helpers import site_experiment

    spec = Path(__file__).resolve().parent.parent / "examples" / "coloring_s0.yaml"
    run = Experiment.from_yaml(spec, "row_major").run(seed=1, max_rounds=3, out=tmp_path / "c")
    path = build(run.dir)
    data = page_data(path)
    states = data["world_states"]
    assert sorted(states) == ["1", "2", "3"]
    st = states["3"]
    assert len(st["grid"]) == 8 and len(st["target"]) == 8 and len(st["grid"][0]) == 8
    assert st["palette"]["."] == "transparent" and set(st["palette"]) >= {"r", "g", "b", "y"}
    painted = [sum(c != "." for row in states[k]["grid"] for c in row) for k in ("1", "2", "3")]
    assert painted[0] > 0 and painted == sorted(painted)  # the grid fills round by round
    assert st["cells_matching"] == run.score["correct"]  # the last state is the final world
    assert 'id="statePanel"' in path.read_text()

    run = site_experiment(crash=False).run(seed=1, max_rounds=2, out=tmp_path / "s")
    states = collect(run.dir)["world_states"]
    assert sorted(states) == ["1", "2"] and states["2"]["picks"] == run.score
    assert {"agent", "site"} == set(states["2"]["last_choice"][0])
    assert build(run.dir).exists()

    # no hook (FlagGame has its own panel; Counter has none): no states, no panel
    flag = Experiment(name="v", world=FlagGame(), participants=[EvidenceAggregator()] * 4,
                      medium=Board(topology="broadcast")).run(seed=1, max_rounds=2,
                                                             out=tmp_path / "f")
    assert collect(flag.dir)["world_states"] is None
    count = Experiment(name="count", world=Counter(), participants=[Adder()] * 2).run(
        seed=0, max_rounds=2, out=tmp_path / "n")
    assert collect(count.dir)["world_states"] is None
