"""Budget enforcement: the spend ledger and the admission gate (docs/INTERFACE-M1b.md §2).

Decisions where the contract is silent:

- Each `Budget` field is enforced only when it is > 0; all zeros means no enforcement, but the
  ledger still accounts every call. `soft_usd` is checked by the runner at round boundaries
  against `spent["swarm"]`; `hard_usd` bounds `spent_total + reserved` (both categories);
  `measurement_usd` additionally bounds `spent["measurement"] + reserved measurement`.
- `Gate.admit` checks the hard ceiling first (`HardCeilingReached`), then the measurement cap
  (`MeasurementBudgetReached`), and only then reserves, so a refused request reserves nothing.
  `MeasurementBudgetReached` is *not* a `HardCeilingReached`: the probe runner catches it and the
  swarm continues. The reservation is taken before waiting on the provider's
  `asyncio.Semaphore(provider.concurrency)`.
- On exit the reservation is released and the actual cost (`Reservation.charge(cost)`) is
  charged to the category. If the context exits with an exception after `dispatch()` and before a
  charge (a cancelled in-flight call, e.g. when a hard ceiling aborts the round), the reserved
  amount is charged as the worst-case spend; a provider error after dispatch charges nothing
  extra (the provider raised, so there is no usage to price).
- The ledger stores spend as integer nano-dollars so totals do not depend on the order in which
  concurrent calls finish (the `budget` event is logical). `spent` and `reserved` are float views.
  `calls` counts dispatched provider calls (cache hits are not calls).
- `Gate.provider_calls` counts dispatches through this gate instance; `replay()` asserts it is 0.
- `Budget.total_usd` (experiment-wide) is not the gate's business: `total_cap_refusal(spent,
  next_hard, total)` is the check `swarmlab run` and `Experiment.run_all` make before starting
  each run (spent = ledger swarm + measurement of the runs already done or found on disk). A
  refusal stops the sequence; `run_all` reports it as a `TotalBudgetWarning`.
"""
from __future__ import annotations

import asyncio
from collections.abc import AsyncIterator, Mapping
from contextlib import asynccontextmanager
from typing import Literal

from .base import Persistable
from .providers import resolve
from .providers.base import ChatRequest, Provider, split_model
from .spec import Budget

Category = Literal["swarm", "measurement"]
CATEGORIES: tuple[str, ...] = ("swarm", "measurement")
NANO = 1_000_000_000


def to_nano(usd: float) -> int:
    return round(usd * NANO)


class BudgetExceeded(Exception):
    """Base for budget refusals."""


class HardCeilingReached(BudgetExceeded):
    """The hard ceiling would be exceeded; the gate refused the call (nothing reserved)."""


class SoftBudgetReached(BudgetExceeded):
    """The swarm's soft budget is spent; the run ends at the next round boundary."""


class MeasurementBudgetReached(BudgetExceeded):
    """The measurement budget would be exceeded; probes stop, the swarm continues."""


class TotalBudgetWarning(UserWarning):
    """`Experiment.run_all` stopped before a run because the experiment's total cap was reached."""


def total_cap_refusal(spent: float, next_hard: float, total: float) -> str | None:
    """Why the next run may not start under the experiment-wide `total` cap, or None if it may.

    A run can spend up to its `hard_usd`, so it starts only when `spent + next_hard <= total`;
    without a per-run ceiling (`next_hard <= 0`) the cap cannot be guaranteed and is refused.
    `total <= 0` means no total cap."""
    if total <= 0:
        return None
    if next_hard <= 0:
        return (f"total_usd ${total:g} needs a per-run hard_usd to bound the next run "
                "(hard_usd is 0)")
    if to_nano(spent) + to_nano(next_hard) > to_nano(total):
        return (f"spent ${spent:.4f} + next run's hard_usd ${next_hard:g} = "
                f"${spent + next_hard:.4f} > total_usd ${total:g}")
    return None


def ledger_total(spend: Mapping | None) -> float:
    """Swarm + measurement spend of a run's ledger dict (`Run.spend`, `run.json["ledger"]`)."""
    spend = spend or {}
    return float(spend.get("swarm") or 0) + float(spend.get("measurement") or 0)


class Ledger(Persistable):
    def __init__(self) -> None:
        self.spent_nano: dict[str, int] = {c: 0 for c in CATEGORIES}
        self.reserved_nano: dict[str, int] = {c: 0 for c in CATEGORIES}
        self.calls = 0

    @property
    def spent(self) -> dict[str, float]:
        return {c: v / NANO for c, v in self.spent_nano.items()}

    @property
    def reserved(self) -> float:
        return sum(self.reserved_nano.values()) / NANO

    @property
    def spent_total(self) -> float:
        return sum(self.spent_nano.values()) / NANO

    def charge(self, category: str, usd: float) -> None:
        self.spent_nano[category] = self.spent_nano.get(category, 0) + to_nano(usd)

    def to_dict(self) -> dict:
        spent = self.spent
        return {"swarm": spent["swarm"], "measurement": spent["measurement"],
                "reserved": self.reserved, "calls": self.calls}

    @classmethod
    def from_dict(cls, data: Mapping | None) -> Ledger:
        led = cls()
        if data:
            for c in CATEGORIES:
                led.spent_nano[c] = to_nano(float(data.get(c, 0.0)))
            led.calls = int(data.get("calls", 0))
        return led

    def __repr__(self) -> str:
        return f"Ledger({self.to_dict()})"


class Reservation:
    def __init__(self, provider: Provider, category: str, amount: float) -> None:
        self.provider = provider
        self.category = category
        self.amount = amount
        self.dispatched = False
        self.cost: float | None = None

    def charge(self, usd: float) -> None:
        self.cost = usd


class Gate:
    def __init__(self, providers: dict[str, Provider], ledger: Ledger, budget: Budget) -> None:
        self.providers = providers  # prefix -> provider; presets are added on first use
        self.ledger = ledger
        self.budget = budget
        self.provider_calls = 0
        self._sems: dict[str, asyncio.Semaphore] = {}

    def provider_for(self, model: str) -> Provider:
        prefix, _ = split_model(model)
        if prefix not in self.providers:
            self.providers[prefix] = resolve(model)[0]
        return self.providers[prefix]

    def _sem(self, provider: Provider) -> asyncio.Semaphore:
        # keyed by provider name; recreated per event loop (asyncio.run per live/resume call)
        loop = asyncio.get_running_loop()
        key = f"{provider.name}@{id(loop)}"
        if key not in self._sems:
            self._sems = {k: v for k, v in self._sems.items() if k.endswith(f"@{id(loop)}")}
            self._sems[key] = asyncio.Semaphore(max(1, provider.concurrency))
        return self._sems[key]

    def check(self, category: str, amount: float) -> None:
        """Raise if reserving `amount` for `category` would break a limit."""
        b, led = self.budget, self.ledger
        n = to_nano(amount)
        if b.hard_usd > 0:
            total = sum(led.spent_nano.values()) + sum(led.reserved_nano.values()) + n
            if total > to_nano(b.hard_usd):
                raise HardCeilingReached(
                    f"hard ceiling ${b.hard_usd:.4f}: spent ${led.spent_total:.4f} + reserved "
                    f"${led.reserved:.4f} + this call's worst case ${amount:.4f}")
        if category == "measurement" and b.measurement_usd > 0:
            total = led.spent_nano["measurement"] + led.reserved_nano["measurement"] + n
            if total > to_nano(b.measurement_usd):
                raise MeasurementBudgetReached(
                    f"measurement budget ${b.measurement_usd:.4f} would be exceeded")

    @asynccontextmanager
    async def admit(self, request: ChatRequest, category: str = "swarm") -> AsyncIterator[Reservation]:
        if category not in CATEGORIES:
            raise ValueError(f"category must be one of {CATEGORIES}, got {category!r}")
        provider = self.provider_for(request.model)
        amount = provider.max_cost(request)
        self.check(category, amount)
        res = Reservation(provider, category, amount)
        n = to_nano(amount)
        self.ledger.reserved_nano[category] += n
        try:
            async with self._sem(provider):
                yield res
        except asyncio.CancelledError:
            if res.cost is None and res.dispatched:
                res.cost = amount
            raise
        finally:
            self.ledger.reserved_nano[category] -= n
            if res.cost is not None:
                self.ledger.charge(category, res.cost)

    def dispatch(self, res: Reservation) -> None:
        res.dispatched = True
        self.provider_calls += 1
        self.ledger.calls += 1
