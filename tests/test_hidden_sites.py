"""HiddenSites world, its metrics and the fake script `hidden_chooser`."""
from __future__ import annotations

import json
import random
import re
from pathlib import Path

import pytest

from swarmlab import Experiment, Run
from swarmlab.events import parse_event
from swarmlab.participants import LLMAgent
from swarmlab.providers.base import ChatMessage, ChatRequest
from swarmlab.providers.fake import resolve_script
from swarmlab.registry import resolve
from swarmlab.report import build_report
from swarmlab.rng import derive
from swarmlab.tools import ToolSchema
from swarmlab.world import hidden_sites as hs
from swarmlab.world.base import Action
from swarmlab.world.hidden_sites import (
    ALREADY_CHOSE,
    DEFAULT_PROBABILITIES,
    POINTS_PENDING,
    ActualCum,
    ActualPoints,
    ExpectedCum,
    ExpectedW,
    HiddenSites,
    expected_points,
    fake_chooser,
    fake_site,
    optimum_counts,
    parse_agent_id,
    parse_observation,
    references,
    resolve_round,
    trial_success,
)

AGENTS = [f"a{i:03d}" for i in range(12)]
SITES = list("ABCDEFGH")
REPO = Path(__file__).resolve().parent.parent
SPEC = REPO / "examples" / "hidden_sites.yaml"
METRIC_NAMES = ["hidden.expected_W", "hidden.expected_cum", "hidden.actual_points",
                "hidden.actual_cum"]
# the treatment sentence of the experiment this world came from; never part of the task text
REMINDER = ("Communicate with your teammates to avoid overcrowding sites, where additional agents "
            "contribute less to the team.")
COMMON = ("Your team has twelve agents and twenty rounds to maximize total discovery points. Sites "
          "have different, unknown success probabilities that remain fixed throughout the "
          "experiment. Each site offers one discovery per round. Every agent there independently "
          "attempts to find it. More agents increase the chance of finding it, but the team "
          "receives only one point for that discovery, regardless of how many agents find it. "
          "Sites offer fresh discoveries each round. You can communicate through the shared "
          "message board and change sites.")


def make(seed=0, round=1, **kw):
    w = HiddenSites(**kw)
    w.reset(derive(seed, "world"), list(AGENTS))
    w.begin_round(round)
    return w


def obs_text(w, agent):
    obs = w.observe(agent)
    assert len(obs.parts) == 1 and obs.private == {}
    return obs.parts[0].text


def play_round(w, picks):
    """Commit {agent: site} as accepted choices in the current round."""
    for a, s in picks.items():
        assert w.choose_site(a, site=s).accepted


def next_round(w):
    """End the current round as the runner does (resolve it) and begin the next."""
    data = w.end_round(w.round)
    w.begin_round(w.round + 1)
    return data


# ---- references and randomisation ------------------------------------------------------------

def test_references_for_default_multiset():
    ref = references(12, dict(zip(SITES, DEFAULT_PROBABILITIES)))
    assert ref["all_at_best"]["W"] == pytest.approx(0.999983222784, abs=1e-12)
    assert ref["random"]["W"] == pytest.approx(1.8284913265790605, abs=1e-12)
    assert ref["optimum"]["W"] == pytest.approx(2.63484375, abs=1e-12)
    assert tuple(ref["optimum"]["allocation"].values()) == (2, 3, 4, 3, 0, 0, 0, 0)
    counts, w = optimum_counts(12, tuple(DEFAULT_PROBABILITIES))
    assert sum(1 for _ in hs._compositions(12, 8)) == 50388
    assert counts == (2, 3, 4, 3, 0, 0, 0, 0) and w == pytest.approx(2.63484375, abs=1e-12)
    # under a permutation the references follow the labels and keep their values
    w0 = make(seed=5)
    sc = w0.score()["references"]
    best = max(w0.assignment, key=w0.assignment.get)
    assert sc["all_at_best"]["allocation"][best] == 12
    assert sc["optimum"]["W"] == pytest.approx(2.63484375, abs=1e-12)
    assert sc["random"]["W"] == pytest.approx(1.8284913265790605, abs=1e-15)
    assert {w0.assignment[k]: n for k, n in sc["optimum"]["allocation"].items() if n} == \
        {0.60: 2, 0.35: 3, 0.25: 4, 0.15: 3}


def test_matched_permutation_and_trial_key_for_the_same_rng():
    a, b = make(seed=7), make(seed=7)
    assert a.assignment == b.assignment and a.trial_key == b.trial_key
    assert sorted(a.assignment.values()) == sorted(DEFAULT_PROBABILITIES)
    assert 0 <= a.trial_key < 2 ** 64
    c = make(seed=8)
    assert (c.assignment, c.trial_key) != (a.assignment, a.trial_key)
    nop = make(seed=7, permute=False)
    assert list(nop.assignment.values()) == list(DEFAULT_PROBABILITIES)


def test_trial_stream_independent_of_call_counts_and_other_draws():
    key = make(seed=3).trial_key
    first = [trial_success(key, r, a, k, 0.5) for r in (1, 2, 20) for a in AGENTS for k in SITES]
    # drawing lots of unrelated randomness, or other trials first, changes nothing
    random.random()
    for _ in range(1000):
        derive(key, "trial", 99, "a000", "A").random()
    again = [trial_success(key, r, a, k, 0.5) for r in (20, 2, 1) for a in reversed(AGENTS)
             for k in reversed(SITES)]
    assert first == list(reversed(again))
    # independence across (round, agent, site): a fair coin is not constant
    assert 0.35 < sum(first) / len(first) < 0.65
    # an agent's outcome does not depend on who else is at the site or on how many choices ran
    w1, w2 = make(seed=3), make(seed=3)
    play_round(w1, {"a000": "C"})
    w2.choose_site("a005", site="C")
    w2.choose_site("a000", site="C")
    w2.choose_site("a000", site="D")                       # rejected second call
    r1 = resolve_round(1, w1.site_of, w1.assignment, w1.trial_key)
    r2 = resolve_round(1, w2.site_of, w2.assignment, w2.trial_key)
    assert r1["outcomes"]["a000"] == r2["outcomes"]["a000"]


# ---- the one-choice rule and the information boundary ----------------------------------------

def test_one_choice_per_turn_rejected_and_reset_per_round():
    w = make(round=1)
    a = AGENTS[0]
    first = w.choose_site(a, site="A")
    assert first.accepted and first.feedback == {"position": "A", "previous": None}
    second = w.choose_site(a, site="B")
    assert not second.accepted and second.feedback == {"error": ALREADY_CHOSE}
    assert w.site_of[a] == "A"
    assert w.choose_site(AGENTS[1], site="H").accepted          # per agent, not global
    next_round(w)
    assert w.chose_this_round == set()
    again = w.choose_site(a, site="C")
    assert again.accepted and again.feedback == {"position": "C", "previous": "A"}
    assert not w.choose_site(a, site="C").accepted
    b = AGENTS[2]                                              # unknown site: choice not used up
    bad = w.choose_site(b, site="I")
    assert not bad.accepted and bad.feedback["error"] == "unknown_site"
    assert w.choose_site(b, site="D").accepted


def test_feedback_has_no_occupancy_or_counts():
    w = make()
    for i, a in enumerate(AGENTS):
        out = w.choose_site(a, site=SITES[i % 8])
        assert set(out.feedback) == {"position", "previous"}
        assert all(not isinstance(v, (int, float, dict, list)) for v in out.feedback.values())
        assert w.choose_site(a, site="A").feedback == {"error": ALREADY_CHOSE}
    assert w.my_status(AGENTS[0]) is None and w.collective_status() is None
    schemas = w.tool_schemas()
    assert [s.name for s in schemas] == ["choose_site"]
    assert schemas[0].parameters["properties"]["site"]["enum"] == SITES
    assert not re.search(r"\d", schemas[0].description)
    assert schemas[0].description.startswith("Hold site A to H from now on")
    small = HiddenSites(probabilities=[0.5, 0.2, 0.1], n_sites=3)
    s3 = small.tool_schema("choose_site")
    assert s3.parameters["properties"]["site"]["enum"] == ["A", "B", "C"]
    assert s3.description.startswith("Hold site A to C from now on")
    # the default validate refuses a label outside the enum before the action runs
    ack = small.validate(AGENTS[0], Action(name="choose_site", args={"site": "D"}))
    assert not ack.ok and "must be one of ['A', 'B', 'C']" in ack.error


def test_observation_lines_and_no_counts_or_probabilities():
    w = make(seed=2, round=1)
    t1 = obs_text(w, "a000")
    assert t1 == ("Round 1 of 20.\nYour current position: none\nYour previous attempt: none yet\n"
                  "Your attempts so far: none\nTeam points last round: " + POINTS_PENDING +
                  "\nTeam points so far: 0")
    assert re.findall(r"\d+(?:\.\d+)?", t1) == ["1", "20", "0"]
    picks = {a: SITES[i % 8] for i, a in enumerate(AGENTS)}
    play_round(w, picks)
    res1 = resolve_round(1, w.site_of, w.assignment, w.trial_key)
    assert next_round(w) == {k: res1[k] for k in ("W", "R", "counts", "outcomes")}
    w.choose_site("a001", site="E")                 # a move mid-round changes only a001's line
    for a in AGENTS:
        t = obs_text(w, a)
        site, ok = res1["outcomes"][a]
        verdict = "success" if ok else "failure"
        assert t.splitlines() == [
            "Round 2 of 20.", f"Your current position: {'E' if a == 'a001' else site}",
            f"Your previous attempt: site {site}, {verdict}",
            f"Your attempts so far: {site}: 1 attempt, {int(ok)} " + ("success" if ok else "successes"),
            f"Team points last round: {res1['R']}", f"Team points so far: {res1['R']}"]
        low = t.lower()
        for word in ("occupancy", "unassigned", "agents", "probab", "expected", "optim", "rank",
                     "%", "="):
            assert word not in low, word
        assert not re.search(r"\d\.\d", t)              # no decimals anywhere
    p = parse_observation("Round 2.\n" + obs_text(w, "a003"))
    assert p["round"] == 2 and p["rounds"] == 20 and p["position"] == picks["a003"]
    assert p["previous"] == tuple(res1["outcomes"]["a003"]) and p["points_last"] == res1["R"]
    # several rounds: the history line aggregates per site in order of first attempt
    w.choose_site("a000", site="C")
    next_round(w)
    w.choose_site("a000", site="A")
    next_round(w)
    hist = parse_observation(obs_text(w, "a000"))["history"]
    assert re.fullmatch(r"A: 2 attempts, [012] success(es)?; C: 1 attempt, [01] success(es)?", hist)
    # show_history=False drops the line
    w2 = make(seed=2, show_history=False)
    assert "attempts so far" not in obs_text(w2, "a000")
    # `swarmlab prompts` renders after reset + begin_round(1)
    w0 = HiddenSites()
    w0.reset(random.Random(0), list(AGENTS))
    assert obs_text(w0, AGENTS[0]).startswith("Round 1 of 20.\n")


def test_description_has_common_text_and_no_probabilities_or_rankings():
    d = make().description()
    assert d.startswith(COMMON + "\n\n")
    op = d[len(COMMON) + 2:]
    for need in ("eight sites, named A to H", "start unassigned", "makes no attempt",
                 "one at a time", "random order that is drawn anew every round", "choose_site",
                 "at most one choice per turn", "keep your current site", "later in the same round",
                 "everyone in later rounds", "Choosing a site is not an attempt",
                 "exactly one independent attempt", "at the end of the round",
                 "told privately whether your own attempt succeeded", "team's points",
                 "total so far", "Nobody is told where the other agents are"):
        assert need in op, need
    low = d.lower()
    for bad in ("optimal", "optimum", "0.", ".20", "60", "rank", "best site", "worst",
                "highest", "lowest", "crowd", "%"):
        assert bad not in low, bad
    assert not re.search(r"\d", d)                       # no numbers at all
    assert REMINDER not in d
    assert make(seed=9).description() == d               # identical across seeds and arms


# ---- resolution, score and metrics -----------------------------------------------------------

METRICS = (ExpectedW, ExpectedCum, ActualPoints, ActualCum)


def _feed(ms, ev):
    for m in ms:
        m.update(ev)


def test_end_round_resolves_and_metrics_fold_round_resolved():
    rounds = 6
    w = HiddenSites(rounds=rounds)
    w.reset(derive(11, "world"), list(AGENTS))
    ms = [cls() for cls in METRICS]
    assert not any(m.needs_truth() for m in ms)
    rng = random.Random(4)
    for r in range(1, rounds + 1):
        w.begin_round(r)
        for a in rng.sample(AGENTS, len(AGENTS)):
            for _ in range(rng.choice([0, 1, 1, 2])):         # none, one, or a rejected second
                w.choose_site(a, site=rng.choice(SITES))
        assert len(w.points_by_round) == r - 1              # nothing resolved before end_round
        data = w.end_round(r)
        ev = parse_event(json.loads(json.dumps(
            {"type": "round_resolved", "run": "t", "seq": r, "round": r, "data": data})))
        _feed(ms, ev)
        # the runner evaluates metrics and score() after end_round
        sc = w.score()
        vals = {m.name: m.value() for m in ms}
        assigned = sum(1 for s in w.site_of.values() if s is not None)
        assert sum(data["counts"].values()) == assigned
        assert data == {k: v for k, v in resolve_round(r, w.site_of, w.assignment, w.trial_key).items()
                        if k != "round"}
        assert len(sc["W_by_round"]) == r == len(sc["R_by_round"]) == len(w.allocations)
        assert vals["hidden.expected_W"] == (sc["W_by_round"][-1], assigned)
        assert vals["hidden.actual_points"] == (float(sc["R_by_round"][-1]), assigned)
        assert vals["hidden.expected_cum"] == (sc["expected_cum"], assigned)
        assert vals["hidden.actual_cum"][0] == float(sc["actual_cum"])
        assert sc["W_last"] == sc["W_by_round"][-1]
        assert sc["allocation"] == w.counts() and sc["unassigned"] == 12 - assigned
        assert sc["W_by_round"][-1] == expected_points(w.counts(), w.assignment)
    before = w.snapshot()
    assert w.score() == w.score()
    assert w.snapshot() == before                         # score() mutates nothing
    # round 0 (before begin_round(1)) scores nothing, and no metric has a value
    w0 = HiddenSites()
    w0.reset(derive(1, "world"), list(AGENTS))
    s0 = w0.score()
    assert s0["W_by_round"] == [] and s0["expected_cum"] == 0 and s0["W_last"] is None
    assert ExpectedW().value() == (None, 0)


def test_attempt_outcomes_reported_in_next_round_match_trials():
    w = make(seed=21)
    play_round(w, {a: "A" for a in AGENTS[:6]})
    next_round(w)
    for a in AGENTS[:6]:
        ok = trial_success(w.trial_key, 1, a, "A", w.assignment["A"])
        assert w.attempts[a] == [[1, "A", ok]]
    for a in AGENTS[6:]:
        assert w.attempts[a] == []
        assert "Your previous attempt: none yet" in obs_text(w, a)
    expect_r = int(any(w.attempts[a][0][2] for a in AGENTS[:6]))
    assert w.points_by_round == [expect_r]


def test_snapshot_restore_round_trip():
    w = make(seed=4, round=1)
    play_round(w, {"a001": "B", "a007": "C"})
    next_round(w)
    w.choose_site("a001", site="D")
    blob = w.snapshot()
    w2 = HiddenSites()
    w2.restore(blob)
    for attr in ("assignment", "trial_key", "round", "site_of", "chose_this_round", "allocations",
                 "attempts", "points_by_round", "expected_by_round", "agents"):
        assert getattr(w2, attr) == getattr(w, attr), attr
    assert w2.score() == w.score()
    assert obs_text(w2, "a001") == obs_text(w, "a001")
    assert not w2.choose_site("a001", site="A").accepted   # restored set still blocks
    assert w2.choose_site("a002", site="A").accepted
    w2.end_round(2)
    w2.begin_round(3)
    assert w2.chose_this_round == set() and len(w2.allocations) == 2
    assert w2.render_state()["points this round"] == w2.points_by_round[-1]


# ---- fake script and end-to-end --------------------------------------------------------------

def _request(system, user, tools=("choose_site", "read_board", "post", "end_turn")):
    schemas = [ToolSchema(name=t, description=t,
                          parameters={"type": "object", "properties": {}, "required": [],
                                      "additionalProperties": False}) for t in tools]
    return ChatRequest(model="fake:hidden_chooser",
                       messages=[ChatMessage(role="system", content=system),
                                 ChatMessage(role="user", content=user)],
                       tools=schemas)


def test_fake_script_calls():
    assert resolve_script("hidden_chooser") is fake_chooser
    assert [fake_site(a) for a in AGENTS] == list("ABCDEFGHABCD")
    w = make()
    for a in AGENTS:
        sys_text = f'You are agent {a} with the role "worker" in a multi-agent environment.'
        assert parse_agent_id(sys_text) == a
        resp = fake_chooser(_request(sys_text, "Round 1.\n" + obs_text(w, a)), random.Random(0))
        s = fake_site(a)
        assert [(tc.name, tc.args) for tc in resp.tool_calls] == [
            ("choose_site", {"site": s}), ("post", {"text": f"{a}: at {s}"}),
            ("choose_site", {"site": s}), ("end_turn", {})]
    w3 = make(round=3)
    w3.site_of["a004"] = "E"
    resp = fake_chooser(_request("You are agent a004 with ...", "Round 3.\n" + obs_text(w3, "a004")),
                        random.Random(0))
    assert [(tc.name, tc.args) for tc in resp.tool_calls] == [
        ("read_board", {}), ("choose_site", {"site": "E"}), ("end_turn", {})]


def short_spec(tmp_path, rounds=4):
    """The example spec with `rounds` and `max_rounds` set to `rounds`."""
    import yaml

    doc = yaml.safe_load(SPEC.read_text())
    doc["options"]["max_rounds"] = rounds
    for arm in doc["arms"].values():
        arm["world"]["params"]["rounds"] = rounds
    path = tmp_path / "hidden_short.yaml"
    path.write_text(yaml.safe_dump(doc))
    return path


def test_example_spec_dry_run_end_to_end(tmp_path):
    exp = Experiment.from_yaml(short_spec(tmp_path), "team")
    run = exp.run(2, out=tmp_path)
    assert run.meta["status"] == "ended" and run.meta["end_reason"] == "max_rounds"
    events = [json.loads(line) for line in (run.dir / "events.jsonl").read_text().splitlines()]
    sc = run.meta["score"]
    assert sc["allocation"] == dict(zip(SITES, (2, 2, 2, 2, 1, 1, 1, 1)))
    met = {}
    for e in events:
        if e["type"] == "metric":
            met.setdefault(e["name"], []).append(e["value"])
    assert met["hidden.expected_W"] == sc["W_by_round"] and len(sc["W_by_round"]) == 4
    assert met["hidden.actual_points"] == [float(x) for x in sc["R_by_round"]]
    assert met["hidden.actual_cum"][-1] == sc["actual_cum"]
    rej = [e["round"] for e in events if e["type"] == "action_committed" and not e["accepted"]]
    assert rej == [1] * 12
    resolved = [e for e in events if e["type"] == "round_resolved"]
    assert [e["round"] for e in resolved] == [1, 2, 3, 4]
    assert [e["data"]["W"] for e in resolved] == sc["W_by_round"]
    for r in range(1, 5):                     # after the round's last action_committed
        types = [e["type"] for e in events if e["round"] == r]
        assert "action_committed" not in types[types.index("round_resolved"):]
    for e in events:
        if e["type"] in ("tool_returned", "action_committed"):
            body = json.dumps(e.get("result") or e.get("feedback"))
            for k in ("occupancy", "assigned", "unassigned", "allocation", "probabilit", '"W"',
                      "expected", "points"):
                assert k not in body, (k, body)
        if e["type"] == "turn_started":                  # what the model is shown
            shown = json.dumps(e)
            assert "round_resolved" not in shown and str(sc["assignment"]["A"]) not in shown
    assert sum(e["type"] == "post" for e in events) == 12


def test_fake_script_follows_the_site_enum():
    w = HiddenSites(probabilities=[0.5, 0.2, 0.1], n_sites=3)
    w.reset(derive(0, "world"), list(AGENTS))
    w.begin_round(1)
    assert w.tool_schema("choose_site").parameters["properties"]["site"]["enum"] == ["A", "B", "C"]
    schemas = w.tool_schemas() + [ToolSchema(name=t, description=t, parameters={
        "type": "object", "properties": {}, "required": [], "additionalProperties": False})
        for t in ("post", "end_turn")]
    req = ChatRequest(model="fake:hidden_chooser", tools=schemas, messages=[
        ChatMessage(role="system", content="You are agent a004 with ..."),
        ChatMessage(role="user", content=obs_text(w, "a004"))])
    resp = fake_chooser(req, random.Random(0))
    assert resp.tool_calls[0].args == {"site": "B"}          # 4 mod 3 -> B
    assert fake_site("a004", ["A", "B", "C"]) == "B" and fake_site("a004") == "E"


# ---- registry and the Python API ----------------------------------------------------------------

def test_entry_points_resolve():
    assert resolve("hidden_sites", "swarmlab.worlds") is HiddenSites
    for name, cls in zip(METRIC_NAMES, (ExpectedW, ExpectedCum, ActualPoints, ActualCum)):
        assert resolve(name, "swarmlab.metrics") is cls
        assert cls.type_name() == name == cls.name
    assert HiddenSites(rounds=6).spec() == {"type": "hidden_sites", "params": {
        "probabilities": list(DEFAULT_PROBABILITIES), "n_sites": 8, "rounds": 6,
        "show_history": True, "permute": True}}
    assert resolve_script("hidden_chooser") is fake_chooser


def test_python_api_run_replays_one_choice_and_metrics_equal_score(tmp_path):
    rounds = 5
    exp = Experiment(name="hidden-api", world=HiddenSites(rounds=rounds),
                     participants=[LLMAgent(model="fake:hidden_chooser", max_tokens=256)] * 12,
                     metrics=METRIC_NAMES, options={"commit": "immediate"})
    run = exp.run(seed=3, max_rounds=rounds, out=tmp_path)
    assert run.meta["status"] == "ended" and run.meta["end_reason"] == "max_rounds"
    loaded = Run.load(run.dir)                      # replays: raises on any metric/score mismatch
    replayed = loaded.replay()
    assert replayed["score"] == run.score and replayed["last_round"] == rounds
    sc = run.score
    # one choice per agent per round: every second choice is rejected, and only round 1 has them
    committed = [e for e in run.events if e["type"] == "action_committed"
                 and e["action"]["name"] == "choose_site"]
    accepted: dict[tuple[int, str], int] = {}
    for e in committed:
        if e["accepted"]:
            accepted[(e["round"], e["agent"])] = accepted.get((e["round"], e["agent"]), 0) + 1
        else:
            assert e["feedback"] == {"error": ALREADY_CHOSE}
    assert set(accepted.values()) == {1} and len(accepted) == 12 * rounds
    assert [e["round"] for e in committed if not e["accepted"]] == [1] * 12
    # metrics after every round agree with score(), the last round included
    vals = {name: [v for _, v, _ in run.metrics[name]] for name in METRIC_NAMES}
    assert vals["hidden.expected_W"] == sc["W_by_round"] and len(sc["W_by_round"]) == rounds
    assert vals["hidden.actual_points"] == [float(x) for x in sc["R_by_round"]]
    assert vals["hidden.actual_cum"][-1] == float(sc["actual_cum"])
    assert vals["hidden.expected_cum"][-1] == pytest.approx(sc["expected_cum"], abs=1e-12)
    assert sc["allocation"] == dict(zip(SITES, (2, 2, 2, 2, 1, 1, 1, 1)))
    # the outcomes follow the label-derived trial stream, so an identical second run matches
    again = exp.run(seed=3, max_rounds=rounds, out=tmp_path / "again")
    assert again.score == sc
    assert len([e for e in run.events if e["type"] == "round_resolved"]) == rounds


# ---- requirements and report ------------------------------------------------------------------

def api_experiment(rounds=5, **options):
    return Experiment(name="hidden-req", world=HiddenSites(rounds=rounds),
                      participants=[LLMAgent(model="fake:hidden_chooser", max_tokens=256)] * 4,
                      metrics=METRIC_NAMES, options=options)


def test_requirements_reject_round_end_and_mismatched_max_rounds(tmp_path):
    assert HiddenSites.requires.commit == "immediate" and HiddenSites.requires.max_rounds == "rounds"
    with pytest.raises(ValueError, match="requires commit=immediate, got round_end"):
        api_experiment()                                   # unset commit is round_end
    with pytest.raises(ValueError, match="requires commit=immediate, got round_end"):
        api_experiment(commit="round_end", max_rounds=5)
    with pytest.raises(ValueError, match=r"requires max_rounds=5 \(world.rounds\), got 4"):
        api_experiment(commit="immediate", max_rounds=4)
    exp = api_experiment(commit="immediate")              # max_rounds given later, to run
    with pytest.raises(ValueError, match=r"max_rounds=5 \(world.rounds\), got 3"):
        exp.run(seed=1, max_rounds=3, out=tmp_path)
    with pytest.raises(ValueError, match="commit=immediate, got round_end"):
        exp.run(seed=1, max_rounds=5, commit="round_end", out=tmp_path)
    assert not list(tmp_path.iterdir())                    # refused before any run dir


def test_report_section_shows_references(tmp_path):
    path = short_spec(tmp_path, rounds=3)
    runs = [Experiment.from_yaml(path, arm).run(1, out=tmp_path / "out") for arm in ("team", "reminder")]
    text = build_report(tmp_path / "out", include_fake=True)
    assert text.index("## Final score") < text.index("## Hidden Sites: points and references")
    section = text.split("## Hidden Sites: points and references")[1].split("\n## ")[0]
    rows = [ln for ln in section.splitlines() if ln.startswith("| ")]
    assert rows[0] == ("| arm | run | rounds | expected_cum | actual_cum | all_at_best W | random W "
                       "| optimum W |")
    assert len(rows) == 3
    for run in runs:
        sc, ref = run.score, run.score["references"]
        row = next(ln for ln in rows if f"| {run.id} |" in ln)
        assert row.endswith(f"| 3 | {sc['expected_cum']:.3f} | {sc['actual_cum']} | "
                            f"{ref['all_at_best']['W']:.3f} | {ref['random']['W']:.3f} | "
                            f"{ref['optimum']['W']:.3f} |")

