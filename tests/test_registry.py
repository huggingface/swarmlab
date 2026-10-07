"""Registry and claim policies (M3b, swarmlab/medium/registry.py)."""
from __future__ import annotations

import pytest

from swarmlab import Board, Experiment, Participant, TurnUsage
from swarmlab.blobs import BlobStore
from swarmlab.executor import RoundExecutor
from swarmlab.ids import agent_id
from swarmlab.medium.registry import (
    Advisory,
    Enforced,
    Registry,
    RegistryOp,
    build_claim_policy,
    commit_world,
)
from swarmlab.rng import derive
from swarmlab.spec import MediumSpec, PluginSpec, RunSpec, spec_hash
from swarmlab.world.base import Action
from swarmlab.world.coloring import ColoringGrid

from .helpers import flag_experiment, logical

A, B, C = agent_id(0), agent_id(1), agent_id(2)


def op(agent, kind, key="k", n=0, **kw):
    return RegistryOp(op_id=f"g-{agent}-{n}", agent=agent, op=kind, key=key, **kw)


# ---- the store ------------------------------------------------------------------------------------

def test_two_acquires_in_one_round_seeded_earlier_wins():
    reg = Registry()
    outs = reg.commit(3, [op(B, "acquire", ttl_rounds=2), op(A, "acquire", ttl_rounds=2)])
    assert [o.ok for o in outs] == [True, False]
    assert outs[0].owner == B and outs[0].expires_round == 4 and outs[0].version == 1
    assert outs[1].error == "held" and outs[1].owner == B and outs[1].expires_round == 4
    assert outs[1].view_outcome() == {
        "action_id": "g-a000-0", "tool": "registry_acquire", "accepted": False,
        "feedback": {"key": "k", "version": 1, "owner": B, "expires_round": 4, "error": "held"}}
    assert reg.owner("k", 3) == B and reg.owner("k", 4) == B


def test_ttl_expiry_lets_another_agent_acquire():
    reg = Registry()
    reg.commit(1, [op(A, "acquire", ttl_rounds=2)])     # live in rounds 1 and 2
    assert reg.owner("k", 2) == A
    assert reg.owner("k", 3) is None
    assert not reg.commit(2, [op(B, "acquire", ttl_rounds=1)])[0].ok
    out = reg.commit(3, [op(B, "acquire", ttl_rounds=1)])[0]
    assert out.ok and out.owner == B and out.expires_round == 3
    assert reg.live_claims(3) == {"k": {"owner": B, "expires_round": 3}}
    assert reg.live_claims(4) == {}


def test_owner_renews_its_claim():
    reg = Registry()
    reg.commit(1, [op(A, "acquire", ttl_rounds=1)])
    out = reg.commit(1, [op(A, "acquire", ttl_rounds=3)])[0]
    assert out.ok and out.expires_round == 3 and out.version == 2


def test_release_by_non_owner_is_rejected():
    reg = Registry()
    reg.commit(1, [op(A, "acquire", ttl_rounds=5)])
    bad, good, again = reg.commit(2, [op(B, "release"), op(A, "release", n=1), op(A, "release", n=2)])
    assert (bad.ok, bad.error, bad.owner) == (False, "not_owner", A)
    assert good.ok and good.owner is None and good.expires_round is None
    assert (again.ok, again.error) == (False, "not_owner")
    assert reg.owner("k", 2) is None
    assert reg.commit(2, [op(B, "acquire", ttl_rounds=1)])[0].ok


def test_cas_version_semantics():
    reg = Registry()
    first, stale = reg.commit(1, [op(A, "compare_and_set", expected_version=0, value="x"),
                                  op(B, "compare_and_set", expected_version=0, value="y")])
    assert first.ok and first.version == 1 and first.value == "x"
    assert (stale.ok, stale.error, stale.version) == (False, "version_mismatch", 1)
    assert reg.get("k", 1)["value"] == "x"
    put = reg.commit(2, [op(B, "put", value={"n": 1})])[0]
    assert put.ok and put.version == 2
    assert reg.commit(2, [op(A, "compare_and_set", expected_version=1, value="z")])[0].error == "version_mismatch"
    assert reg.commit(2, [op(A, "compare_and_set", expected_version=2, value="z")])[0].version == 3
    assert reg.get("missing", 2) == {"key": "missing", "exists": False, "value": None, "version": 0,
                                     "owner": None, "expires_round": None}


def test_get_sees_own_pending_writes_only():
    reg = Registry()
    reg.commit(1, [op(A, "put", value="old")])
    pending = [op(B, "put", value="mine"), op(B, "acquire", n=1, ttl_rounds=2)]
    view = reg.get("k", 2, pending)
    assert view["value"] == "mine" and view["owner"] == B and view["pending"] is True
    assert view["version"] == 3
    assert reg.get("k", 2) == {"key": "k", "exists": True, "value": "old", "version": 1,
                               "owner": None, "expires_round": None}


def test_snapshot_restore_keeps_live_claims():
    reg = Registry()
    reg.commit(1, [op(A, "acquire", ttl_rounds=4), op(B, "put", key="v", value=[1, 2])])
    other = Registry()
    other.restore(reg.snapshot())
    assert other.owner("k", 4) == A and other.get("v", 4)["value"] == [1, 2]
    assert other.registry_state(5) == {
        "k": {"value": None, "version": 1, "owner": A, "expires_round": 4, "live": False},
        "v": {"value": [1, 2], "version": 1, "owner": None, "expires_round": None, "live": False}}


# ---- claim policies -------------------------------------------------------------------------------

def test_claim_policies():
    assert Advisory().allows(A, "k", B) and Advisory().allows(A, "k", None)
    assert Enforced().allows(A, "k", A) and not Enforced().allows(A, "k", B)
    assert not Enforced().allows(A, "k", None)
    assert isinstance(build_claim_policy("enforced"), Enforced)
    assert isinstance(build_claim_policy({"type": "advisory", "params": {}}), Advisory)
    with pytest.raises(ValueError):
        Board(claim_policy="enforced")  # enforced needs the registry


def _world():
    w = ColoringGrid(height=2, width=2)
    w.reset(derive(0, "world"), [A, B])
    w.begin_round(1)
    return w


def test_commit_world_enforced_rejects_unclaimed_and_records_claims():
    w = _world()
    reg = Registry()
    reg.commit(1, [op(A, "acquire", key="cell:0,0", ttl_rounds=1)])
    c00, c11 = w.target[0][0], w.target[1][1]
    actions = [(B, "x1", Action(name="paint", args={"x": 0, "y": 0, "color": c00})),
               (A, "x2", Action(name="paint", args={"x": 0, "y": 0, "color": c00})),
               (B, "x3", Action(name="paint", args={"x": 1, "y": 1, "color": c11}))]
    outs, claims = commit_world(w, actions, registry=reg, policy=Enforced(), round=1)
    assert [o.accepted for o in outs] == [False, True, False]
    assert outs[0].feedback == {"error": "not_claimed", "key": "cell:0,0", "owner": A}
    assert outs[0].action_id == "x1"
    assert [(c["key"], c["owner"], c["held"], c["violation"], c["rejected"]) for c in claims] == [
        ("cell:0,0", A, False, True, True), ("cell:0,0", A, True, False, False),
        ("cell:1,1", None, False, False, True)]
    assert w.grid[0][0] == c00 and w.grid[1][1] == "."


def test_commit_world_advisory_accepts_and_measures():
    w = _world()
    reg = Registry()
    reg.commit(1, [op(A, "acquire", key="cell:0,0", ttl_rounds=1)])
    actions = [(B, "x1", Action(name="paint", args={"x": 0, "y": 0, "color": w.target[0][0]}))]
    outs, claims = commit_world(w, actions, registry=reg, policy=Advisory(), round=1)
    assert outs[0].accepted and claims[0]["violation"] and not claims[0]["rejected"]
    assert commit_world(w, actions, registry=None, policy=None, round=1)[1] == []


# ---- executor -------------------------------------------------------------------------------------

def _executor(tmp_path, commit="round_end", registry=None, policy=None):
    w = _world()
    board = Board(registry=True)
    board.commit_mode = commit
    ex = RoundExecutor(run="r", round=1, world=w, board=board, blobs=BlobStore(tmp_path / "b"),
                       agents=[A, B], commit=commit, registry=registry or Registry(),
                       claim_policy=policy or Advisory())
    return w, ex


async def test_executor_registry_tools_buffer_and_read_own_writes(tmp_path):
    _, ex = _executor(tmp_path)
    names = [s.name for s in ex.schemas(A)]
    assert {"registry_get", "registry_put", "registry_cas", "registry_acquire",
            "registry_release"} <= set(names)
    assert all(not s.strict_violations() for s in ex.schemas(A))
    res = await ex.call(A, "registry_acquire", {"key": "cell:0,0", "ttl_rounds": 2})
    assert res.ok and res.pending and res.result == {"id": "g0001-a000-00"}
    mine = await ex.call(A, "registry_get", {"key": "cell:0,0"})
    theirs = await ex.call(B, "registry_get", {"key": "cell:0,0"})
    assert mine.result["owner"] == A and mine.result["pending"] is True
    assert theirs.result["owner"] is None and "pending" not in theirs.result
    bad = await ex.call(A, "registry_acquire", {"key": "k", "ttl_rounds": 0})
    assert not bad.ok and "ttl_rounds" in bad.error
    bad = await ex.call(A, "registry_put", {"key": "", "value": "v"})
    assert not bad.ok
    assert [o.op for o in ex.buffered_registry(A)] == ["acquire"]
    assert ex.registry.entries == {}


async def test_executor_without_registry_has_no_registry_tools(tmp_path):
    from .test_executor import setup

    _, _, _, ex = setup(tmp_path)
    assert not any(s.name.startswith("registry_") for s in ex.schemas(A))
    res = await ex.call(A, "registry_get", {"key": "k"})
    assert (res.ok, res.error) == (False, "not_allowed")


async def test_executor_immediate_applies_and_enforces(tmp_path):
    w, ex = _executor(tmp_path, commit="immediate", policy=Enforced())
    c = w.target[0][0]
    res = await ex.call(B, "paint", {"x": 0, "y": 0, "color": c})
    assert res.result["accepted"] is False and res.result["feedback"]["error"] == "not_claimed"
    res = await ex.call(A, "registry_acquire", {"key": "cell:0,0", "ttl_rounds": 1})
    assert not res.pending and res.result["accepted"] and res.result["feedback"]["owner"] == A
    res = await ex.call(A, "paint", {"x": 0, "y": 0, "color": c})
    assert res.result["accepted"] is True
    assert [o.ok for o in ex.committed.registry] == [True]
    assert [cl["rejected"] for cl in ex.committed.claims] == [True, False]


# ---- end to end -----------------------------------------------------------------------------------

class Grabber(Participant):
    """Every agent acquires the same key each round with a ttl of 3, and stores a counter."""

    async def turn(self, view, tools):
        self.seen = getattr(self, "seen", []) + [list(view.outcomes)]
        await tools.call("registry_acquire", {"key": "lock", "ttl_rounds": 3})
        got = await tools.call("registry_get", {"key": "count"})
        await tools.call("registry_cas", {"key": "count", "expected_version": got.result["version"],
                                          "value": f"{self.agent}@{view.round}"})
        await tools.call("end_turn", {})
        return TurnUsage(calls=4)


def grab_experiment(name="grab"):
    return Experiment(name=name, world=ColoringGrid(height=2, width=2), participants=[Grabber()] * 3,
                      medium=Board(registry=True), metrics=["claims.held", "claims.violations"])


def events(run, type_):
    return [e for e in run.events if e["type"] == type_]


def test_end_to_end_registry_events_outcomes_and_ordering(tmp_path):
    run = grab_experiment().run(seed=4, max_rounds=5, out=tmp_path)
    order1 = next(e["order"] for e in run.events if e["type"] == "round_started" and e["round"] == 1)
    acq1 = [e for e in events(run, "registry") if e["round"] == 1 and e["op"] == "acquire"]
    assert [e["agent"] for e in acq1] == order1  # committed in seeded order
    assert [e["ok"] for e in acq1] == [True, False, False]
    winner = order1[0]
    assert all(e["owner"] == winner and e["expires_round"] == 3 for e in acq1)
    cas1 = [e for e in events(run, "registry") if e["round"] == 1 and e["op"] == "compare_and_set"]
    assert [e["ok"] for e in cas1] == [True, False, False]
    # rounds 2-3: the winner renews (its claim is live); at round 4 the renewal from round 3 holds
    for r in (2, 3, 4, 5):
        acq = [e for e in events(run, "registry") if e["round"] == r and e["op"] == "acquire"]
        assert [e["agent"] for e in acq if e["ok"]] == [winner]
    # outcomes reach the next view, after world-action outcomes, as registry_* tools
    rows = [e for e in events(run, "registry") if e["round"] == 1 and e["agent"] == order1[1]]
    assert [r["op_id"] for r in rows] == [f"g0001-{order1[1]}-00", f"g0001-{order1[1]}-01"]
    held = [v for _, v, _ in run.metrics["claims.held"]]
    assert held == [1.0] * 5
    # registry events precede action_committed / round_committed in each round
    types = [e["type"] for e in run.events if e["round"] == 1]
    assert types.index("registry") > max(i for i, t in enumerate(types) if t == "turn_ended")
    assert run.dir.joinpath("snapshots", "000001.json").exists()


def test_outcomes_in_next_view(tmp_path):
    import copy

    exp = grab_experiment()
    run = exp.run(seed=4, max_rounds=2, out=tmp_path)
    from swarmlab.snapshot import SnapshotStore

    snaps = SnapshotStore(run.dir)
    manifest = snaps.read(1)
    order1 = next(e["order"] for e in run.events if e["type"] == "round_started" and e["round"] == 1)
    loser = order1[1]
    outs = manifest.outcomes_prev[loser]
    assert [o["tool"] for o in outs] == ["registry_acquire", "registry_cas"]
    assert outs[0]["accepted"] is False and outs[0]["feedback"]["error"] == "held"
    assert outs[0]["feedback"]["owner"] == order1[0]
    p = copy.deepcopy(exp.participants[0])
    p.restore(snaps.load(snaps.read(2))[f"participant:{loser}"])
    assert p.seen[1] == outs


def test_registry_determinism_and_replay(tmp_path):
    a = grab_experiment().run(seed=9, max_rounds=4, out=tmp_path / "a")
    b = grab_experiment().run(seed=9, max_rounds=4, out=tmp_path / "b")
    assert logical(a, exclude=("seq", "ts", "run")) == logical(b, exclude=("seq", "ts", "run"))
    assert type(a).load(a.dir).replay()["metrics_checked"] > 0


def test_fork_restores_live_claims(tmp_path):
    parent = grab_experiment().run(seed=2, max_rounds=6, out=tmp_path)
    child = parent.fork(at_round=3).run()
    strip = ("seq", "ts", "run")
    tail = [e for e in logical(parent, strip) if e["round"] > 3]
    ctail = [e for e in logical(child, strip) if e["round"] > 3 and e["type"] != "run_started"]
    assert ctail == tail
    assert child.meta["restored"]["world"] is True


def test_resume_after_crash_with_live_claims(tmp_path):
    import json

    ref = grab_experiment().run(seed=6, max_rounds=6, out=tmp_path / "ref")
    run = grab_experiment().run(seed=6, max_rounds=6, out=tmp_path / "crash")
    # simulate a crash during round 5: keep the log through round 4's snapshot event
    log = run.dir / "events.jsonl"
    lines = log.read_bytes().splitlines(keepends=True)
    cut = next(i for i, ln in enumerate(lines)
               if json.loads(ln)["type"] == "snapshot" and json.loads(ln)["round"] == 4)
    keep = lines[: cut + 1] + [ln for ln in lines[cut + 1:]
                                if json.loads(ln)["round"] == 5 and json.loads(ln)["type"]
                                in ("round_started", "turn_started", "tool_called")][:5]
    log.write_bytes(b"".join(keep))
    resumed = type(run)(run.dir).resume()
    assert resumed.end_reason == "max_rounds"
    assert logical(resumed, ("seq", "ts", "run")) == logical(ref, ("seq", "ts", "run"))


def test_spec_hash_unchanged_without_registry(tmp_path):
    exp = flag_experiment(2)
    spec = exp.to_spec(seed=1, max_rounds=2)
    assert "registry" not in spec.model_dump(mode="json")["medium"]
    assert "claim_policy" not in spec.model_dump(mode="json")["medium"]
    explicit = RunSpec(**{**spec.model_dump(),
                          "medium": MediumSpec(**spec.medium.model_dump(), registry=False,
                                               claim_policy="advisory")})
    assert spec_hash(explicit) == spec_hash(spec)  # the defaults never reach the hash
    on = Experiment(name="x", world=ColoringGrid(), participants=[Grabber()],
                    medium=Board(registry=True, claim_policy="enforced"))
    ospec = on.to_spec(seed=1, max_rounds=2)
    assert ospec.medium.registry is True and ospec.medium.claim_policy == PluginSpec(type="enforced")
    back = Experiment.from_spec(ospec)
    assert back.medium.registry and isinstance(back.medium.claim_policy, Enforced)
    assert spec_hash(back.to_spec(seed=1, max_rounds=2)) == spec_hash(ospec)
    assert spec_hash(ospec) != spec_hash(
        Experiment(name="x", world=ColoringGrid(), participants=[Grabber()],
                   medium=Board(registry=True)).to_spec(seed=1, max_rounds=2))
