"""M3c §1, §3, §4: roles (executor enforcement, binding, spec, YAML, prompts)."""
from __future__ import annotations

import json
from pathlib import Path

import pytest

from swarmlab import Board, Experiment, Participant, Run, TurnUsage
from swarmlab.blobs import BlobStore
from swarmlab.executor import RoundExecutor
from swarmlab.ids import agent_id
from swarmlab.medium.registry import Registry
from swarmlab.participants import EvidenceAggregator, LLMAgent
from swarmlab.prompts_cmd import render_prompts
from swarmlab.providers.fake import FakeProvider
from swarmlab.rng import derive
from swarmlab.roles import (
    BUILTIN_ROLES,
    COORDINATOR,
    Role,
    assign,
    resolve_role,
    role_name,
)
from swarmlab.spec import SpecError, arm_to_runspec, load_experiment_yaml, spec_hash
from swarmlab.tools import TurnCapReached
from swarmlab.world.flaggame import FlagGame

from .helpers import TEST_PRICING, logical

REPO = Path(__file__).resolve().parent.parent
AGENTS = [agent_id(i) for i in range(4)]


class Prober(Participant):
    """Tries every kind of call each turn and records the errors it got."""

    async def turn(self, view, tools):
        results = {
            "read": await tools.call("read_board", {}),
            "post": await tools.call("post", {"text": f"{self.agent} r{view.round}"}),
            "guess": await tools.call("guess", {"candidate": "A"}),
        }
        self.errors = getattr(self, "errors", []) + [{k: r.error for k, r in results.items()}]
        await tools.call("end_turn", {})
        return TurnUsage(calls=4)


def executor(tmp_path, roles, *, board=None, registry=None, max_calls=20):
    world = FlagGame()
    world.reset(derive(1, "world"), AGENTS)
    board = board or Board()
    return RoundExecutor(run="r", round=1, world=world, board=board,
                         blobs=BlobStore(tmp_path / "blobs"), agents=AGENTS,
                         max_calls_per_turn=max_calls, registry=registry,
                         topology_rng=lambda: derive(1, "topology", 1), roles=roles)


# ---- the Role model and library ----------------------------------------------------------------
def test_builtin_library():
    assert set(BUILTIN_ROLES) == {"worker", "coordinator", "reviewer", "skeptic", "scribe"}
    assert BUILTIN_ROLES["worker"] == Role(name="worker")
    assert COORDINATOR.may_act is False and COORDINATOR.post_fields == {"kind": ["assignment", "summary"]}
    assert BUILTIN_ROLES["reviewer"].may_act is False and BUILTIN_ROLES["scribe"].may_act is False
    assert BUILTIN_ROLES["skeptic"].may_act is True
    assert all(r.prompt_append for n, r in BUILTIN_ROLES.items() if n != "worker")


def test_role_validation_and_resolution():
    with pytest.raises(ValueError):
        Role(name="x", budget={"max_calls": 0})
    with pytest.raises(ValueError):
        Role(name="x", budget={"tokens": 3})
    with pytest.raises(ValueError):
        Role(name="x", model="no-prefix")
    with pytest.raises(ValueError):
        Role(name="x", surprise=True)
    acting = resolve_role("coordinator", {"may_act": True})
    assert acting.may_act is True and acting.prompt_append == COORDINATOR.prompt_append
    assert resolve_role("lead", {"tools": ["end_turn"]}) == Role(name="lead", tools=["end_turn"])
    assert resolve_role("boss", COORDINATOR).name == "boss"
    with pytest.raises(KeyError):
        resolve_role("nobody")


# ---- executor enforcement ----------------------------------------------------------------------
async def test_acceptance_1_tools_allowlist_blocks_board(tmp_path):
    """A role with tools=[guess, end_turn] cannot read or post; attempts log not_allowed."""
    role = Role(name="guesser", tools=["guess", "end_turn"])
    ex = executor(tmp_path, {AGENTS[0]: role})
    a = AGENTS[0]
    assert [s.name for s in ex.schemas(a)] == ["guess", "end_turn"]
    assert (await ex.call(a, "read_board", {})).error == "not_allowed"
    assert (await ex.call(a, "post", {"text": "x"})).error == "not_allowed"
    assert (await ex.call(a, "my_status", {})).error == "not_allowed"
    assert (await ex.call(a, "guess", {"candidate": "A"})).ok
    assert ex.buffered_posts(a) == []
    rets = [e for e in ex.events(a) if e.type == "tool_returned"]
    assert [r.result["error"] for r in rets[:3]] == ["not_allowed"] * 3
    # the other agents are unaffected
    assert "post" in [s.name for s in ex.schemas(AGENTS[1])]


def test_acceptance_1_in_a_run(tmp_path):
    parts = [assign(Prober(), "guesser"), Prober(), Prober()]
    exp = Experiment(name="acc1", world=FlagGame(), participants=parts,
                     roles={"guesser": Role(name="guesser", tools=["guess", "end_turn"])})
    run = exp.run(seed=1, max_rounds=2, out=tmp_path)
    ev = list(run.events)
    ended = [e for e in ev if e["type"] == "turn_ended" and e["agent"] == "a000"]
    assert [e["yield_kind"] for e in ended] == ["end_turn", "end_turn"]
    calls = {e["call_id"]: e["tool"] for e in ev if e["type"] == "tool_called" and e["agent"] == "a000"}
    blocked = [calls[e["call_id"]] for e in ev if e["type"] == "tool_returned" and e["agent"] == "a000"
               and e["result"]["error"] == "not_allowed"]
    assert blocked == ["read_board", "post"] * 2
    assert not [e for e in ev if e["type"] == "post" and e["agent"] == "a000"]
    started = {e["agent"]: e["role"] for e in ev if e["type"] == "turn_started"}
    assert started == {"a000": "guesser", "a001": None, "a002": None}


async def test_may_act_false_blocks_world_actions_not_status(tmp_path):
    ex = executor(tmp_path, {AGENTS[0]: COORDINATOR})
    a = AGENTS[0]
    names = [s.name for s in ex.schemas(a)]
    assert "guess" not in names and "my_status" in names and "post" in names
    assert (await ex.call(a, "guess", {"candidate": "A"})).error == "not_allowed"
    assert (await ex.call(a, "my_status", {})).ok
    assert ex.buffered_actions(a) == []


async def test_channels_read_and_write(tmp_path):
    board = Board(channels=["main", "ops"])
    role = Role(name="r", channels_read=["main"], channels_write=["ops"])
    ex = executor(tmp_path, {AGENTS[0]: role}, board=board)
    a, b = AGENTS[0], AGENTS[1]
    schemas = {s.name: s for s in ex.schemas(a)}
    assert schemas["read_board"].parameters["properties"]["channel"]["enum"] == ["main"]
    assert schemas["post"].parameters["properties"]["channel"]["enum"] == ["ops"]
    assert (await ex.call(a, "post", {"channel": "main", "text": "x"})).error == "not_allowed"
    assert (await ex.call(a, "post", {"text": "x"})).error == "not_allowed"  # default main
    assert (await ex.call(a, "post", {"channel": "ops", "text": "x"})).ok
    assert (await ex.call(a, "read_board", {"channel": "ops"})).error == "not_allowed"
    # deliveries on both channels; read_board() returns only readable ones and leaves the rest unread
    board.buffer_post(b, 0, "ops", "secret")
    board.buffer_post(b, 0, "main", "public")
    board.buffer_post(b, 0, "main", "public2")
    board.commit(0, AGENTS, derive(1, "topology", 0), ex.blobs)
    res = await ex.call(a, "read_board", {"limit": 1})
    assert [i["content"] for i in res.result["items"]] == ["public"]
    res = await ex.call(a, "read_board", {})
    assert [i["content"] for i in res.result["items"]] == ["public2"]
    assert [d.post_id for d in board.inbox(a) if d.read_round is None] == ["p0000-0000"]
    assert [ex.board.content(d, ex.blobs) for d in ex.pushable(a, 10)] == []
    board.buffer_post(b, 0, "main", "p3")
    board.commit(0, AGENTS, derive(1, "topology", 0), ex.blobs)
    assert [ex.board.content(d, ex.blobs) for d in ex.pushable(a, 10)] == ["p3"]


async def test_post_fields_enforced_and_advertised(tmp_path):
    ex = executor(tmp_path, {AGENTS[0]: COORDINATOR})
    a = AGENTS[0]
    post = next(s for s in ex.schemas(a) if s.name == "post")
    assert post.parameters["properties"]["fields"]["properties"]["kind"]["enum"] == ["assignment", "summary"]
    assert post.strict_violations() == []
    assert (await ex.call(a, "post", {"text": "x", "fields": {"kind": "assignment"}})).ok
    assert (await ex.call(a, "post", {"text": "x", "fields": {"kind": "gossip"}})).error == "not_allowed"
    assert (await ex.call(a, "post", {"text": "x", "fields": {"other": "y"}})).error == "not_allowed"
    assert (await ex.call(a, "post", {"text": "plain"})).ok


async def test_registry_permissions(tmp_path):
    reg = Registry()
    ex = executor(tmp_path, {AGENTS[0]: Role(name="r", registry="read"),
                             AGENTS[1]: Role(name="n", registry="none")}, registry=reg)
    a, b = AGENTS[0], AGENTS[1]
    assert [s.name for s in ex.schemas(a) if s.name.startswith("registry")] == ["registry_get"]
    assert [s.name for s in ex.schemas(b) if s.name.startswith("registry")] == []
    assert (await ex.call(a, "registry_get", {"key": "k"})).ok
    assert (await ex.call(a, "registry_put", {"key": "k", "value": "v"})).error == "not_allowed"
    assert (await ex.call(b, "registry_get", {"key": "k"})).error == "not_allowed"
    assert ex.buffered_registry(a) == []
    assert (await ex.call(AGENTS[2], "registry_put", {"key": "k", "value": "v"})).ok


async def test_budget_max_calls_is_the_agents_cap(tmp_path):
    ex = executor(tmp_path, {AGENTS[0]: Role(name="r", budget={"max_calls": 2})}, max_calls=20)
    a = AGENTS[0]
    await ex.call(a, "my_status", {})
    await ex.call(a, "my_status", {})
    with pytest.raises(TurnCapReached):
        await ex.call(a, "my_status", {})
    for _ in range(3):
        assert (await ex.call(AGENTS[1], "my_status", {})).ok


# ---- acceptance 3: the coordinator ablation switch ---------------------------------------------
def coordinator_experiment(**coordinator_fields) -> Experiment:
    parts = [assign(Prober(), "coordinator"), assign(Prober(), "worker"), Prober()]
    roles = {"coordinator": COORDINATOR.model_copy(update=coordinator_fields)} if coordinator_fields else {}
    return Experiment(name="acc3", world=FlagGame(), participants=parts, roles=roles)


def test_acceptance_3_coordinator_may_act_switch(tmp_path):
    blocked = coordinator_experiment().run(seed=1, max_rounds=1, out=tmp_path / "a")
    ev = list(blocked.events)
    assert not [e for e in ev if e["type"] == "action_committed" and e["agent"] == "a000"]
    assert [e for e in ev if e["type"] == "action_committed" and e["agent"] == "a001"]
    acting = coordinator_experiment(may_act=True)
    run = acting.run(seed=1, max_rounds=1, out=tmp_path / "b")
    assert [e for e in run.events if e["type"] == "action_committed" and e["agent"] == "a000"]
    assert coordinator_experiment().spec_hash(1, 1) != acting.spec_hash(1, 1)
    assert coordinator_experiment().spec_hash(1, 1) == coordinator_experiment().spec_hash(1, 1)


# ---- acceptance 4: prompt_append and model override --------------------------------------------
def llm_roles_experiment(role: Role, n: int = 2) -> Experiment:
    parts = [assign(LLMAgent(model="fake:reader", max_tokens=256), role.name)] + [
        LLMAgent(model="fake:reader", max_tokens=256) for _ in range(n - 1)]
    return Experiment(name="acc4", world=FlagGame(), participants=parts, roles={role.name: role},
                      providers={"fake": FakeProvider(pricing=TEST_PRICING),
                                 "alt": FakeProvider(pricing=TEST_PRICING)})


def test_acceptance_4_prompt_append_in_prompts_and_run(tmp_path):
    role = Role(name="lead", prompt_append="ROLE-MARKER: you lead.", tools=["read_board", "end_turn"])
    exp = llm_roles_experiment(role)
    rows = render_prompts(exp, seed=1)
    assert [(r["agent"], r["role"], r["count"]) for r in rows] == [("a000", "lead", 1), ("a001", None, 1)]
    assert rows[0]["system"].endswith("ROLE-MARKER: you lead.")
    assert 'the role "lead"' in rows[0]["system"]
    assert "ROLE-MARKER" not in rows[1]["system"]
    assert rows[0]["tools"] == ["read_board", "end_turn"]
    run = exp.run(seed=1, max_rounds=1, out=tmp_path)
    first = next(e for e in run.events_all if e.type == "inference_attempt" and e.agent == "a000")
    h = first.request_hash
    req = json.loads((run.dir / "blobs" / h[:2] / h).read_text())
    assert req["messages"][0]["content"] == rows[0]["system"]
    assert [t["name"] for t in req["tools"]] == ["read_board", "end_turn"]


def test_prompt_append_combines_with_system_prompt_append():
    agent = LLMAgent(model="fake:reader")
    agent.bind("a000", derive(0, "agent", "a000"))
    agent.system_prompt_append = "ARM-TEXT"  # what the dogfood-fixes kwarg sets
    agent.apply_role(Role(name="lead", prompt_append="ROLE-TEXT"))
    from swarmlab.view import Observation, View

    view = View(round=1, agent="a000", observation=Observation(parts=[]), outcomes=[], pushed=[],
                tools=[], description="d")
    text = agent._render_system(view)
    assert "ARM-TEXT\n\nTools:" in text and text.endswith("ROLE-TEXT")


def test_acceptance_4_model_override_changes_provider(tmp_path):
    role = Role(name="alt", model="alt:flaggame_reader", budget={"max_tokens": 128})
    exp = llm_roles_experiment(role)
    run = exp.run(seed=1, max_rounds=1, out=tmp_path)
    models = {e.agent: e.model for e in run.events_all if e.type == "inference_attempt"}
    assert models == {"a000": "alt:flaggame_reader", "a001": "fake:reader"}
    assert exp.providers["alt"].calls > 0 and exp.providers["fake"].calls > 0
    first = next(e for e in run.events_all if e.type == "inference_attempt" and e.agent == "a000")
    h = first.request_hash
    assert json.loads((run.dir / "blobs" / h[:2] / h).read_text())["max_tokens"] == 128
    # the prototype is untouched; the override lives in the run spec
    assert exp.participants[0].model == "fake:reader"
    assert exp.to_spec(1, 1).roles["alt"]["model"] == "alt:flaggame_reader"


def test_role_model_is_priced_at_construction():
    from swarmlab.providers.base import UnknownModelPricing

    with pytest.raises((UnknownModelPricing, ValueError)):
        Experiment(name="x", world=FlagGame(), participants=[LLMAgent(model="fake:reader")],
                   roles={"r": Role(name="r", model="nosuchprefix:m")})


# ---- acceptance 5: spec hash and YAML ----------------------------------------------------------
def test_acceptance_5_hash_unchanged_without_roles():
    def plain() -> Experiment:
        return Experiment(name="h", world=FlagGame(), participants=[EvidenceAggregator()] * 3)

    spec = plain().to_spec(1, 3)
    assert spec.roles == {} and spec.participant_roles == []
    data = spec.model_dump(mode="json")
    assert "roles" not in data and "participant_roles" not in data  # the pre-M3c shape
    with_roles = spec.model_copy(update={"participant_roles": [None] * 3, "roles": {"worker": {}}})
    assert set(with_roles.model_dump(mode="json")) >= {"roles", "participant_roles"}
    assert spec_hash(with_roles) != spec_hash(spec)
    # a pinned pre-M3c hash: the flaggame_m1a broadcast arm, seed 1
    doc = load_experiment_yaml(REPO / "examples" / "flaggame_m1a.yaml")
    assert "roles" not in doc and all("role" not in g for g in doc["arms"]["broadcast"]["participants"])
    rs = arm_to_runspec(doc, "broadcast", seed=1)
    assert "roles" not in json.dumps(rs.model_dump(mode="json")["participants"])
    # LLMAgent's own `role` param is not an assignment
    exp = Experiment(name="h", world=FlagGame(), participants=[LLMAgent(model="fake:reader", role="boss")])
    assert exp.to_spec(1, 1).participant_roles == []


ROLES_YAML = """
name: roles-yaml
options: {max_rounds: 1}
roles:
  coordinator: {}
  lead: {prompt_append: "Lead the group.", tools: [read_board, post, end_turn]}
arms:
  base:
    world: flaggame
    participants:
      - {type: "tests.test_roles:Prober", role: coordinator}
      - {type: "tests.test_roles:Prober", count: 2}
      - {type: "tests.test_roles:Prober", role: lead}
      - {type: "tests.test_roles:Prober", role: skeptic}
    medium: {topology: {type: tree, params: {groups: 2}}}
  acting:
    world: flaggame
    roles:
      coordinator: {may_act: true}
    participants:
      - {type: "tests.test_roles:Prober", role: coordinator}
      - {type: "tests.test_roles:Prober", count: 4}
"""


def test_acceptance_5_yaml_roles_round_trip(tmp_path):
    path = tmp_path / "r.yaml"
    path.write_text(ROLES_YAML)
    doc = load_experiment_yaml(path)
    rs = arm_to_runspec(doc, "base", seed=0)
    assert rs.participant_roles == ["coordinator", None, None, "lead", "skeptic"]
    assert set(rs.roles) == {"coordinator", "lead", "skeptic"}
    assert rs.roles["coordinator"]["may_act"] is False
    assert arm_to_runspec(doc, "acting", seed=0).roles["coordinator"]["may_act"] is True
    exp = Experiment.from_yaml(path, "base")
    assert [role_name(p) for p in exp.participants] == rs.participant_roles
    spec = exp.to_spec(0, 1)
    assert (spec.roles, spec.participant_roles) == (rs.roles, rs.participant_roles)
    out = tmp_path / "back.yaml"
    exp.to_yaml(out)
    back = load_experiment_yaml(out)
    assert [g.get("role") for g in back["arms"]["base"]["participants"]] == [
        "coordinator", None, "lead", "skeptic"]
    again = Experiment.from_yaml(out, "base")
    assert again.spec_hash(0, 1) == exp.spec_hash(0, 1)
    # the acting arm's flip is in the spec and changes the hash
    unflipped = load_experiment_yaml(path)
    unflipped["arms"]["acting"].pop("roles")
    assert spec_hash(arm_to_runspec(unflipped, "acting", seed=0, max_rounds=1)) != \
        Experiment.from_yaml(path, "acting").spec_hash(0, 1)


def test_yaml_role_errors(tmp_path):
    bad = ROLES_YAML.replace("role: skeptic", "role: nobody")
    path = tmp_path / "bad.yaml"
    path.write_text(bad)
    with pytest.raises(SpecError, match="nobody"):
        load_experiment_yaml(path)
    path.write_text(ROLES_YAML.replace("coordinator: {}", "coordinator: {may_act: maybe}"))
    with pytest.raises(SpecError):
        load_experiment_yaml(path)


def test_yaml_roles_run_resume_and_replay(tmp_path):
    path = tmp_path / "r.yaml"
    path.write_text(ROLES_YAML)
    exp = Experiment.from_yaml(path, "base")
    run = exp.run(seed=1, max_rounds=3, out=tmp_path / "runs")
    ev = list(run.events)
    started = {e["agent"]: e["role"] for e in ev if e["type"] == "turn_started"}
    assert started == {"a000": "coordinator", "a001": None, "a002": None, "a003": "lead", "a004": "skeptic"}
    lead_blocked = [e for e in ev if e["type"] == "tool_returned" and e["agent"] == "a003"
                    and e["result"]["error"] == "not_allowed"]
    assert len(lead_blocked) == 3  # guess, once per round
    loaded = Run.load(run.dir)
    assert logical(loaded) == logical(run)
    # a fresh run of the rebuilt experiment is identical (roles come back from the spec)
    again = loaded.experiment.run(seed=1, max_rounds=3, out=tmp_path / "again")
    assert logical(again) == logical(run)


def test_hierarchy_example_runs(tmp_path):
    arms = Experiment.arms_from_yaml(REPO / "examples" / "hierarchy_flaggame.yaml")
    assert set(arms) == {"flat", "tree"}
    tree = arms["tree"]
    assert [role_name(p) for p in tree.participants].count("coordinator") == 3
    run = tree.run(seed=1, max_rounds=2, out=tmp_path)
    ev = list(run.events)
    coords = {"a000", "a006", "a012"}
    assert not [e for e in ev if e["type"] == "action_committed" and e["agent"] in coords]
    assert {e["channel"] for e in ev if e["type"] == "post"} <= {"group:0", "group:1", "group:2",
                                                                 "coordinators"}
    assert arms["flat"].spec_hash(1, 2) != tree.spec_hash(1, 2)
