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
  each run (spent = the experiment ledger's spend plus the headroom of runs in flight, below).
  A run refused only for headroom held by runs in flight waits for them; a final refusal skips
  that run (`swarmlab run` goes on with the next one, `run_all` stops: its later seeds have the
  same `hard_usd`) and `run_all` reports it as a `TotalBudgetWarning`.
- Experiment spend across processes (field notes item 3): `ExperimentLedger(out, experiment)` is
  the append-only file `<out>/<experiment>.ledger.jsonl` of rows `{ts, pid, host, run_id, key,
  spec_hash, spend_usd, status, hard_usd}`, where `spend_usd` is the run's ledger total so far
  and `key` identifies one run instance (`<run_id>@<started_at>`: a resume keeps it, a run dir
  deleted and run again gets a new one, so spend already paid is never forgotten). A run writes
  a row per state change (`running` at its start, `ended`, `interrupted`) and, while running, a
  `running` heartbeat row at most every `HEARTBEAT_S` (60 s; at a commit, or from a background
  thread during a long round), not one per commit. Rows are appended under an exclusive
  `fcntl.flock` on the file.
- The state (`ExperimentLedger.state`): the experiment's spend is the sum over keys of each key's
  latest `spend_usd`, over every process and invocation that wrote rows. A run is in flight when
  the latest row of its *run id* is `admitted` or `running` and its writer is live: the pid is
  alive when the row is from this host, a row from another host is live for `STALE_S` (10 min)
  after it was written (the heartbeat keeps it fresh). So a run's `ended`/`interrupted` row, the
  `failed` row written for a run that could not start, or a newer instance of the same run id
  release its reservation, and a process that died (or a host that went silent) reserves
  nothing; its spend still counts. In flight it reserves `hard_usd - spend_usd` (>= 0).
- Admission (`admit(run_id, hard, total)`, under the lock): spend + the headroom of the runs in
  flight + this run's `hard_usd` must fit under `total_usd` (`total_cap_refusal`); when it does,
  an `admitted` row is appended at once, so concurrent starts (`swarmlab run --parallel`, or two
  shells) cannot both take the same headroom. A refusal names every run counted in flight and
  its headroom; `Admission.wait` is true when spend + `hard_usd` alone fits, i.e. the run only
  has to wait for runs in flight to end (callers then retry every `ADMIT_POLL_S` or when one of
  their own runs ends), otherwise the refusal is final.
- Compaction: `compacted(rows)` keeps the latest row per key (its spend) and the latest row per
  run id (its in-flight state); `state` works on it, and `admit` rewrites the file to it
  (atomically, with the new `admitted` row included) once more than half of 200+ rows are
  superseded. `lock` reopens the file when a rewrite replaced it while it waited.
- A run dir found without a ledger row is backfilled (`backfill`); a `running` run.json there
  (a killed process) is recorded `interrupted`. The runner ends a run with `total_budget` at a
  round boundary when the ledger's spend (its own included) reaches `total_usd`
  (swarmlab/runner.py). Rows with `simulated: true` (runs on `fake:` models only: nominal
  prices, nothing billed) are ignored. Delete the file to forget past spend.
"""
from __future__ import annotations

import asyncio
import fcntl
import json
import os
import socket
import time
from collections.abc import AsyncIterator, Iterator, Mapping
from contextlib import asynccontextmanager, contextmanager
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Literal

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


def _pid_alive(pid: int) -> bool:
    if pid <= 0:
        return False
    if pid == os.getpid():
        return True
    try:
        os.kill(pid, 0)
    except ProcessLookupError:
        return False
    except PermissionError:
        return True
    return True


HEARTBEAT_S = 60.0   # a running run writes a `running` row at most this often (and at least)
STALE_S = 600.0      # a row from another host without a newer row for this long is not live
ADMIT_POLL_S = 5.0   # how often a run waiting for headroom re-checks the ledger
IN_FLIGHT_STATUSES = ("admitted", "running")


@dataclass(frozen=True)
class InFlight:
    """A run counted as in flight: the latest row of its run id (module doc)."""

    run_id: str
    key: str
    status: str
    hard_usd: float
    spend_usd: float
    pid: int
    host: str

    @property
    def reserved(self) -> float:
        """Headroom it still holds: `hard_usd - spend_usd`, at least 0."""
        return max(0, to_nano(self.hard_usd) - to_nano(self.spend_usd)) / NANO

    def describe(self) -> str:
        where = "" if self.host == socket.gethostname() else f" on {self.host}"
        return f"{self.run_id} ${self.reserved:.4f} ({self.status}, pid {self.pid}{where})"

    def to_dict(self) -> dict:
        return {"run_id": self.run_id, "key": self.key, "status": self.status,
                "hard_usd": self.hard_usd, "spend_usd": self.spend_usd,
                "reserved_usd": self.reserved, "pid": self.pid, "host": self.host}


@dataclass(frozen=True)
class LedgerState:
    """`spent`: latest `spend_usd` summed over run instances (keys); `in_flight`: the runs
    whose headroom is reserved."""

    spent: float
    in_flight: tuple[InFlight, ...]

    @property
    def reserved(self) -> float:
        return sum(to_nano(f.reserved) for f in self.in_flight) / NANO

    def describe_in_flight(self) -> str:
        return ", ".join(f.describe() for f in self.in_flight)


@dataclass(frozen=True)
class Admission:
    """`ExperimentLedger.admit`'s answer. `wait`: refused only because of runs in flight
    (spend + this run's `hard_usd` fits once they end), so retry when one ends."""

    refusal: str | None
    spent: float
    reserved: float
    in_flight: tuple[InFlight, ...] = ()
    wait: bool = False

    @property
    def checked(self) -> float:
        """Spend + reserved headroom the run was checked against."""
        return self.spent + self.reserved


class ExperimentLedger:
    """`<out>/<experiment>.ledger.jsonl`: the experiment's spend over every run (module doc)."""

    def __init__(self, out: Path | str, experiment: str) -> None:
        self.path = Path(out) / f"{experiment}.ledger.jsonl"

    @contextmanager
    def lock(self) -> Iterator[Any]:
        """The file, open for append under an exclusive `flock`. If `compact` replaced the file
        while this process waited for the lock, the new file is opened and locked instead."""
        self.path.parent.mkdir(parents=True, exist_ok=True)
        while True:
            with open(self.path, "a+b") as f:
                fcntl.flock(f, fcntl.LOCK_EX)
                try:
                    try:
                        current = os.fstat(f.fileno()).st_ino == os.stat(self.path).st_ino
                    except FileNotFoundError:
                        current = False
                    if current:
                        yield f
                        return
                finally:
                    fcntl.flock(f, fcntl.LOCK_UN)

    @staticmethod
    def _append(f: Any, line: str) -> None:
        """Append one row; a torn last line (a killed writer) is closed off first."""
        f.seek(0, os.SEEK_END)
        if f.tell() > 0:
            f.seek(-1, os.SEEK_END)
            if f.read(1) != b"\n":
                line = "\n" + line
        f.write(line.encode("utf-8"))
        f.flush()

    def rows(self) -> list[dict]:
        try:
            text = self.path.read_text(encoding="utf-8")
        except FileNotFoundError:
            return []
        out = []
        for line in text.splitlines():
            try:
                row = json.loads(line)
            except ValueError:  # a torn last line from a killed process
                continue
            if isinstance(row, dict) and "key" in row and "run_id" in row:
                out.append(row)
        return out

    @staticmethod
    def compacted(rows: list[dict]) -> list[dict]:
        """The rows that carry the state, in file order: the latest row per run instance (key),
        which holds its spend, and the latest row per run id, which says whether it is in
        flight. A superseded `admitted` row (spend 0) is dropped."""
        latest_key: dict[str, int] = {}
        latest_run: dict[str, int] = {}
        for i, row in enumerate(rows):
            latest_key[row["key"]] = i
            latest_run[row["run_id"]] = i
        keep = set(latest_run.values())
        keep.update(i for i in latest_key.values() if rows[i].get("status") != "admitted")
        return [rows[i] for i in sorted(keep)]

    @staticmethod
    def _live(row: dict, now: float) -> bool:
        if row.get("host") == socket.gethostname():
            return _pid_alive(int(row.get("pid") or 0))
        return now - float(row.get("ts") or 0) <= STALE_S

    def state(self, exclude: str | None = None, rows: list[dict] | None = None) -> LedgerState:
        """Spend and runs in flight (module doc), leaving out run instance key or run id
        `exclude` and simulated rows."""
        rows = self.compacted(self.rows() if rows is None else rows)
        latest_key: dict[str, dict] = {}
        latest_run: dict[str, dict] = {}
        for row in rows:
            latest_key[row["key"]] = row
            latest_run[row["run_id"]] = row
        spent = sum(to_nano(float(r.get("spend_usd") or 0)) for k, r in latest_key.items()
                    if exclude not in (k, r["run_id"]) and not r.get("simulated"))
        now = time.time()
        in_flight = tuple(
            InFlight(rid, r["key"], r["status"], float(r.get("hard_usd") or 0),
                     float(r.get("spend_usd") or 0), int(r.get("pid") or 0), str(r.get("host")))
            for rid, r in latest_run.items()
            if r.get("status") in IN_FLIGHT_STATUSES and not r.get("simulated")
            and exclude not in (rid, r["key"]) and self._live(r, now))
        return LedgerState(spent / NANO, in_flight)

    def totals(self, exclude: str | None = None, rows: list[dict] | None = None
               ) -> tuple[float, float]:
        """(spend, headroom reserved by runs in flight), leaving out key or run id `exclude`."""
        st = self.state(exclude, rows)
        return st.spent, st.reserved

    def spent(self, exclude: str | None = None) -> float:
        return self.state(exclude).spent

    def run_ids(self) -> set[str]:
        return {r["run_id"] for r in self.rows()}

    def _row(self, run_id: str, key: str, spend: float, status: str, hard: float,
             spec_hash: str | None, **extra: Any) -> str:
        row = {"ts": time.time(), "pid": os.getpid(), "host": socket.gethostname(),
               "run_id": run_id, "key": key, "spec_hash": spec_hash, "spend_usd": spend,
               "status": status, "hard_usd": hard, **extra}
        return json.dumps(row, sort_keys=True) + "\n"

    def record(self, run_id: str, key: str, spend: float, status: str, hard: float = 0.0,
               spec_hash: str | None = None, **extra: Any) -> None:
        line = self._row(run_id, key, spend, status, hard, spec_hash, **extra)
        with self.lock() as f:
            self._append(f, line)

    def _rewrite(self, rows: list[dict]) -> None:
        """Replace the file by `rows` (atomic rename; call under `lock`)."""
        tmp = self.path.with_name(self.path.name + ".compact.tmp")
        with open(tmp, "wb") as f:
            f.write("".join(json.dumps(r, sort_keys=True) + "\n" for r in rows).encode("utf-8"))
            f.flush()
            os.fsync(f.fileno())
        os.replace(tmp, self.path)

    def compact(self) -> int:
        """Rewrite the file keeping only `compacted` rows; returns how many rows were dropped."""
        with self.lock():
            rows = self.rows()
            keep = self.compacted(rows)
            if len(keep) < len(rows):
                self._rewrite(keep)
            return len(rows) - len(keep)

    def backfill(self, meta: Mapping, keys: set[str] | None = None) -> bool:
        """Add a row for a run dir's `run.json` when the ledger has none for that run instance
        (run dirs written before the ledger existed, or copied in); True if one was added. A
        run.json that is not `ended` (its process was killed) is recorded as `interrupted`:
        a live run writes its own rows, so a backfilled one is never in flight."""
        from .spec import unbilled_spec

        key = f"{meta['run_id']}@{meta.get('started_at') or 'legacy'}"
        if key in (keys if keys is not None else {r["key"] for r in self.rows()}):
            return False
        extra = {"simulated": True} if unbilled_spec(meta.get("spec") or {}) else {}
        status = str(meta.get("status") or "ended")
        if status in IN_FLIGHT_STATUSES:
            extra["backfilled_status"] = status
            status = "interrupted"
        self.record(meta["run_id"], key, ledger_total(meta.get("ledger")), status,
                    hard=float((meta.get("budget") or {}).get("hard_usd") or 0),
                    spec_hash=meta.get("spec_hash"), backfilled=True, **extra)
        if keys is not None:
            keys.add(key)
        return True

    def backfill_dir(self, out: Path | str) -> int:
        """`backfill` every run dir directly under `out` whose `run.json` names this
        experiment; returns how many rows were added."""
        name = self.path.name.removesuffix(".ledger.jsonl")
        keys = {r["key"] for r in self.rows()}
        added = 0
        for d in sorted(Path(out).glob("*/run.json")):
            try:
                meta = json.loads(d.read_text())
            except (OSError, ValueError):
                continue
            if meta.get("experiment") == name and "run_id" in meta:
                added += self.backfill(meta, keys)
        return added

    def admit(self, run_id: str, hard: float, total: float, spec_hash: str | None = None
              ) -> Admission:
        """The start check (module doc). An admitted run gets an `admitted` row in the same
        locked step. The file is compacted first when most of its rows are superseded."""
        with self.lock() as f:
            rows = self.rows()
            keep = self.compacted(rows)
            st = self.state(rows=keep)
            why = total_cap_refusal(st.spent + st.reserved, hard, total)
            wait = (why is not None and st.in_flight != ()
                    and total_cap_refusal(st.spent, hard, total) is None)
            if why and st.in_flight:
                why += (f" (${st.reserved:.4f} of it reserved by {len(st.in_flight)} run(s) "
                        f"still in flight: {st.describe_in_flight()})")
            line = None
            if why is None:
                line = self._row(run_id, f"{run_id}@admit", 0.0, "admitted", hard, spec_hash)
            if len(rows) > 200 and len(keep) * 2 < len(rows):
                # one atomic rewrite that already holds the new row: a process that opens the
                # new file sees it; one waiting on the old file reopens (`lock`)
                self._rewrite(keep + ([json.loads(line)] if line else []))
            elif line:
                self._append(f, line)
            return Admission(why, st.spent, st.reserved, st.in_flight, wait)


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
