"""Ledger and admission gate (docs/INTERFACE-M1b.md §2)."""
import asyncio

import pytest

from swarmlab.budget import Gate, HardCeilingReached, Ledger, MeasurementBudgetReached
from swarmlab.providers.base import ChatMessage, ChatRequest
from swarmlab.providers.fake import FakeProvider
from swarmlab.spec import Budget


def request(chars=4000, max_tokens=1000):
    return ChatRequest(model="fake:x", messages=[ChatMessage(role="user", content="x" * chars)],
                       max_tokens=max_tokens)


def gate(**budget):
    provider = FakeProvider(pricing={"*": (1.0, 5.0, 0.1)}, concurrency=2)
    return Gate({"fake": provider}, Ledger(), Budget(**budget)), provider


# max_cost(request()) = (1000 * 1 + 1000 * 5) / 1e6 = 0.006
WORST = 0.006


async def test_reserve_release_and_charge():
    g, _ = gate(hard_usd=1.0)
    async with g.admit(request(), "swarm") as res:
        assert res.amount == pytest.approx(WORST)
        assert g.ledger.reserved == pytest.approx(WORST)
        g.dispatch(res)
        res.charge(0.002)
    assert g.ledger.reserved == 0
    assert g.ledger.spent == {"swarm": pytest.approx(0.002), "measurement": 0.0}
    assert g.ledger.calls == 1 and g.provider_calls == 1
    async with g.admit(request(), "measurement") as res:
        g.dispatch(res)
        res.charge(0.001)
    assert g.ledger.spent["measurement"] == pytest.approx(0.001)
    assert g.ledger.spent_total == pytest.approx(0.003)


async def test_hard_ceiling_raises_before_reserving():
    g, _ = gate(hard_usd=0.01)
    async with g.admit(request(), "swarm") as res:
        g.dispatch(res)
        res.charge(0.005)
    # 0.005 spent + 0.006 worst case > 0.01
    with pytest.raises(HardCeilingReached):
        async with g.admit(request(), "swarm"):
            pytest.fail("must not be admitted")
    assert g.ledger.reserved == 0 and g.ledger.calls == 1
    # a smaller request still fits
    async with g.admit(request(chars=400, max_tokens=100), "swarm") as res:
        assert res.amount < 0.005


async def test_reservations_in_flight_count_toward_the_ceiling():
    g, _ = gate(hard_usd=0.01)
    async with g.admit(request(), "swarm"):
        with pytest.raises(HardCeilingReached):
            async with g.admit(request(), "swarm"):
                pass
    assert g.ledger.reserved == 0 and g.ledger.spent_total == 0


async def test_measurement_cap_is_separate_and_not_a_hard_ceiling():
    g, _ = gate(measurement_usd=0.008)
    async with g.admit(request(), "measurement") as res:
        g.dispatch(res)
        res.charge(0.004)
    with pytest.raises(MeasurementBudgetReached) as e:
        async with g.admit(request(), "measurement"):
            pass
    assert not isinstance(e.value, HardCeilingReached)
    async with g.admit(request(), "swarm") as res:  # the swarm is not bounded by it
        g.dispatch(res)
        res.charge(0.004)
    assert g.ledger.spent == {"swarm": pytest.approx(0.004), "measurement": pytest.approx(0.004)}


async def test_zero_budget_means_no_enforcement_but_accounting():
    g, _ = gate()
    for _ in range(50):
        async with g.admit(request(), "swarm") as res:
            g.dispatch(res)
            res.charge(WORST)
    assert g.ledger.spent["swarm"] == pytest.approx(50 * WORST) and g.ledger.calls == 50


async def test_cancelled_in_flight_call_charges_its_reservation():
    g, _ = gate()
    started = asyncio.Event()

    async def call():
        async with g.admit(request(), "swarm") as res:
            g.dispatch(res)
            started.set()
            await asyncio.sleep(10)

    task = asyncio.create_task(call())
    await started.wait()
    task.cancel()
    with pytest.raises(asyncio.CancelledError):
        await task
    assert g.ledger.reserved == 0 and g.ledger.spent["swarm"] == pytest.approx(WORST)


async def test_provider_error_after_dispatch_charges_nothing():
    g, _ = gate()
    with pytest.raises(RuntimeError):
        async with g.admit(request(), "swarm") as res:
            g.dispatch(res)
            raise RuntimeError("provider failed")
    assert g.ledger.spent_total == 0 and g.ledger.reserved == 0 and g.ledger.calls == 1


async def test_concurrency_semaphore_per_provider():
    g, _ = gate()
    active = 0
    peak = 0

    async def call():
        nonlocal active, peak
        async with g.admit(request(), "swarm"):
            active += 1
            peak = max(peak, active)
            await asyncio.sleep(0.01)
            active -= 1

    await asyncio.gather(*(call() for _ in range(6)))
    assert peak == 2


def test_ledger_persists_and_sums_order_independently():
    a, b = Ledger(), Ledger()
    costs = [0.1, 0.2, 0.3, 1e-7, 0.123456789]
    for c in costs:
        a.charge("swarm", c)
    for c in reversed(costs):
        b.charge("swarm", c)
    assert a.spent == b.spent
    c = Ledger()
    c.restore(a.snapshot())
    assert c.to_dict() == a.to_dict()
    assert Ledger.from_dict(a.to_dict()).to_dict() == a.to_dict()


async def test_unknown_prefix_resolved_lazily():
    g = Gate({}, Ledger(), Budget())
    async with g.admit(request(), "swarm") as res:
        assert res.provider.name == "fake"
    with pytest.raises(ValueError):
        async with g.admit(request(), "other"):
            pass
