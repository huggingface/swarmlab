import pytest

from swarmlab.ids import agent_id
from swarmlab.registry import build, resolve
from swarmlab.rng import derive
from swarmlab.scheduler import OneSpeaker, Scheduler, SeededShuffle, build_scheduler

AGENTS = [agent_id(i) for i in range(10)]


def test_base_scheduler_keeps_order():
    assert Scheduler().order(1, AGENTS, derive(0, "schedule", 1)) == AGENTS


def test_seeded_shuffle_is_a_deterministic_permutation():
    s = SeededShuffle()
    live = list(AGENTS)
    a = s.order(3, live, derive(7, "schedule", 3))
    b = s.order(3, live, derive(7, "schedule", 3))
    assert a == b
    assert sorted(a) == AGENTS
    assert live == AGENTS  # input not mutated


def test_seeded_shuffle_uses_the_passed_rng():
    s = SeededShuffle()
    orders = {tuple(s.order(r, AGENTS, derive(7, "schedule", r))) for r in range(1, 6)}
    assert len(orders) > 1
    assert s.spec() == {"type": "seeded_shuffle", "params": {}}


def test_scheduler_snapshot_roundtrip():
    s = SeededShuffle()
    t = SeededShuffle()
    t.restore(s.snapshot())
    assert t.order(1, AGENTS, derive(1, "schedule", 1)) == s.order(1, AGENTS, derive(1, "schedule", 1))


class Reversed(Scheduler):
    def order(self, round, live, rng):
        return list(reversed(live))


def test_base_scheduler_is_registered_as_fixed():
    assert Scheduler().spec() == {"type": "fixed", "params": {}}
    assert resolve("fixed", "swarmlab.schedulers") is Scheduler
    assert type(build({"type": "fixed"}, "swarmlab.schedulers")) is Scheduler
    assert type(build_scheduler({"type": "fixed", "params": {}})) is Scheduler
    assert type(build_scheduler("fixed")) is Scheduler
    # the form written before the base class had a name still resolves (module:Class)
    assert type(build_scheduler("swarmlab.scheduler:Scheduler")) is Scheduler


def test_fixed_name_is_not_inherited():
    assert SeededShuffle().spec()["type"] == "seeded_shuffle"
    assert Reversed().spec() == {"type": "tests.test_scheduler:Reversed", "params": {}}


class NotAScheduler:
    pass


def test_build_scheduler_inputs():
    assert type(build_scheduler(None)) is SeededShuffle
    s = SeededShuffle()
    assert build_scheduler(s) is s
    assert type(build_scheduler("seeded_shuffle")) is SeededShuffle
    one = build_scheduler({"type": "one_speaker"})
    assert type(one) is OneSpeaker and one.listeners == 1
    assert build_scheduler({"type": "one_speaker", "params": {"listeners": 2}}).listeners == 2


def test_build_scheduler_accepts_a_plugin_spec():
    from swarmlab.spec import PluginSpec

    assert type(build_scheduler(PluginSpec(type="one_speaker", params={}))) is OneSpeaker


def test_build_scheduler_unknown_name_raises():
    with pytest.raises(ValueError, match="no_such_scheduler"):
        build_scheduler("no_such_scheduler")


def test_build_scheduler_rejects_a_non_scheduler_class():
    with pytest.raises(TypeError, match="not a Scheduler"):
        build_scheduler("tests.test_scheduler:NotAScheduler")
