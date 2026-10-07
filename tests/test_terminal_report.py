"""M6 §5: `swarmlab report` terminal-state shares and terminal truth mass (Flag Game paper)."""
from collections import Counter

from swarmlab import Board, Experiment
from swarmlab.metrics.belief import STATES
from swarmlab.participants import LLMAgent
from swarmlab.report import build_report, scan, terminal_distribution, world_verify
from swarmlab.roles import MANAGER, assign
from swarmlab.world.flaggame import FlagGame


def world(**kw):
    return FlagGame(flags="real", candidates="names", canvas=[24, 16], crop=[6, 4], **kw)


def agent(**kw):
    return LLMAgent("fake:country_reporter", report_json=True, **kw)


def pairwise(name="pairwise-4"):
    return Experiment(
        name="m6", arm=name, world=world(), participants=[agent(memory="received")] * 4,
        medium=Board(topology={"type": "gossip", "params": {"k": 1}}, delivery="push", push_limit=8,
                     push_consume=True),
        metrics=["belief.state"], probes=[{"type": "belief", "params": {"every": 4}}],
        options={"scheduler": "one_speaker", "max_rounds": 16, "commit": "immediate"})


def manager(name="manager-4"):
    observers = [agent(memory="received")] * 4
    boss = assign(agent(memory="received"), "manager")
    return Experiment(
        name="m6", arm=name, world=world(blind_agents=1, blind_may_guess=True),
        participants=[boss, *observers],
        roles={"manager": MANAGER.model_copy(update={"may_act": True})},
        medium=Board(topology="star", delivery="push", push_limit=8, push_consume=True),
        metrics=["belief.state"], options={"max_rounds": 3})


def test_terminal_states_section(tmp_path):
    runs = [pairwise().run(seed=s, out=tmp_path) for s in (1, 2, 3)]
    mgr = manager().run(seed=1, out=tmp_path)
    text = build_report(tmp_path, include_fake=True)
    assert "## Terminal states (Flag Game paper)" in text
    section = text.split("## Terminal states (Flag Game paper)")[1].split("\n## ")[0]
    rows = {line.split(" | ")[0].lstrip("| "): line for line in section.splitlines()
            if line.startswith("| ") and not line.startswith("| arm")}
    assert set(rows) == {"pairwise-4 (simulated)", "manager-4 (simulated)"}
    cells = rows["pairwise-4 (simulated)"].strip("| \n").split(" | ")
    assert cells[1] == "3" and abs(sum(float(c) for c in cells[2:6]) - 1.0) < 1e-9
    assert cells[7] == "probes 3"
    # the manager's endpoint is its own final decision: a consensus of one
    mcells = rows["manager-4 (simulated)"].strip("| \n").split(" | ")
    assert mcells[7] == "manager 1" and mcells[4] == mcells[5] == "0.00"
    d = scan(mgr)
    dist, kind = terminal_distribution(mgr, d, world_verify(mgr, d["agents"]))
    assert kind == "manager" and sum(dist.values()) == 1
    # pairwise endpoint = the last probe round's answers of all four agents
    d = scan(runs[0])
    dist, kind = terminal_distribution(runs[0], d, world_verify(runs[0], d["agents"]))
    assert kind == "probes" and sum(dist.values()) == 4
    assert set(STATES) and isinstance(dist, Counter)
