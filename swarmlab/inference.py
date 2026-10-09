"""Inference through the harness: request blobs, the record/replay cache, gate, operational
events (docs/INTERFACE-M1b.md §3). Runner-internal; participants reach it via `AgentTools.infer`.

Decisions where the contract is silent:

- **Request blob.** The request body is stored as its canonical JSON (`request_bytes`), so the
  blob's sha equals `request_hash(request)`; `inference_attempt.request_hash` addresses it.
- **Cache layout.** `blobs/cache/<request_hash>` holds the response JSON exactly as first produced
  (`cached=False`, its real `cost_usd`), written atomically; `blobs/cache/index.jsonl` gets one
  line `{"hash", "round"}` per entry written, so a fork can copy the entries of rounds <= its fork
  round. The response is also stored as an ordinary blob; `inference_response.response_hash` is
  that blob's sha.
- **Cache hit.** Logs `inference_attempt` (`reserved_usd=0`) and
  `inference_response(cached=True, cost_usd=0)`, charges nothing, makes no provider call, and
  returns the stored response with `cached=True, cost_usd=0`.
- **Miss.** The gate's budget check runs first: a refused request (`HardCeilingReached`,
  `MeasurementBudgetReached`) raises before anything is logged, so every logged attempt was
  admitted. Then, holding the reservation and the provider semaphore, `inference_attempt` is
  appended (operational, immediately, through the runner) before `provider.complete` is awaited.
  A provider exception logs `inference_response(finish_reason="error:<ExceptionType>",
  response_hash="")` and propagates.
- **Nominal usage.** `infer` returns `(response, nominal)`: `nominal` is the response as first
  produced (real usage and `cost_usd`) on hits and misses alike. The executor accumulates nominal
  usage into `turn_ended.usage`, so a resumed run whose turns hit the cache has the same logical
  `turn_ended` events as an uninterrupted one. The ledger records actual spend.
- **Retries** happen inside `provider.complete` while this call holds the gate reservation and
  the provider semaphore, so the reservation is taken once and released once whatever the number
  of attempts. `inference_response.attempts` records the attempts (from `ChatResponse.attempts`,
  or `ProviderError.attempts` on failure).
- **Refusals** are ordinary responses: cached like any other, so re-sending the same request
  returns the same refusal. `inference_response.refusal_category` carries the response's
  `Refusal.category` (hit or miss). A caller that wants a fresh answer re-sends with
  `ChatRequest.attempt` raised (`LLMAgent(refusal_retries=N)`), which is a different hash and so
  a different cache entry; replay and resume reproduce the same attempts from the log's
  requests, so they hit the cache and never call a provider.
- Concurrent identical requests may both miss and both call the provider (no in-flight dedupe);
  the later write wins the cache file, which is harmless for a deterministic provider.
"""
from __future__ import annotations

import json
import shutil
import time
from collections.abc import Callable
from pathlib import Path

from ._io import atomic_write_bytes
from .blobs import BlobStore
from .budget import Gate
from .events import Event, InferenceAttemptEvent, InferenceResponseEvent
from .providers.base import ChatRequest, ChatResponse, request_bytes, request_hash


def _category(resp: ChatResponse) -> str | None:
    return resp.refusal.category if resp.refusal is not None else None


class InferenceCache:
    def __init__(self, blob_dir: Path | str) -> None:
        self.dir = Path(blob_dir) / "cache"

    def path(self, h: str) -> Path:
        return self.dir / h

    def get(self, h: str) -> ChatResponse | None:
        p = self.path(h)
        if not p.exists():
            return None
        return ChatResponse.model_validate_json(p.read_bytes())

    def put(self, h: str, response: ChatResponse, round: int) -> None:
        self.dir.mkdir(parents=True, exist_ok=True)
        atomic_write_bytes(self.path(h), response.model_dump_json().encode())
        with open(self.dir / "index.jsonl", "a") as f:
            f.write(json.dumps({"hash": h, "round": round}, sort_keys=True) + "\n")

    def entries(self) -> list[tuple[str, int]]:
        idx = self.dir / "index.jsonl"
        if not idx.exists():
            return []
        out = []
        for line in idx.read_text().splitlines():
            try:
                d = json.loads(line)
            except ValueError:
                continue  # torn last line after a crash
            out.append((d["hash"], int(d["round"])))
        return out

    def copy_to(self, other: InferenceCache, max_round: int) -> int:
        """Copy entries written in rounds <= max_round into `other`; returns the number copied."""
        n = 0
        lines = []
        for h, r in self.entries():
            if r <= max_round and self.path(h).exists():
                other.dir.mkdir(parents=True, exist_ok=True)
                shutil.copyfile(self.path(h), other.path(h))
                lines.append(json.dumps({"hash": h, "round": r}, sort_keys=True) + "\n")
                n += 1
        if lines:
            with open(other.dir / "index.jsonl", "a") as f:
                f.writelines(lines)
        return n


class Inference:
    def __init__(self, *, run_id: str, gate: Gate, blobs: BlobStore, cache: InferenceCache,
                 log_operational: Callable[[Event], int]) -> None:
        self.run_id = run_id
        self.gate = gate
        self.blobs = blobs
        self.cache = cache
        self.log = log_operational

    async def infer(self, *, agent: str | None, round: int, call_id: str, request: ChatRequest,
                    category: str = "swarm") -> tuple[ChatResponse, ChatResponse]:
        h = request_hash(request)
        self.blobs.put(request_bytes(request))
        provider = self.gate.provider_for(request.model)
        hit = self.cache.get(h)
        if hit is not None:
            self.log(InferenceAttemptEvent(run=self.run_id, round=round, agent=agent, call_id=call_id,
                                           provider=provider.name, model=request.model,
                                           request_hash=h, reserved_usd=0.0, category=category))
            self.log(InferenceResponseEvent(
                run=self.run_id, round=round, agent=agent, call_id=call_id,
                response_hash=self.blobs.put(hit.model_dump_json().encode()),
                usage=hit.usage.model_dump(), cost_usd=0.0, latency_s=0.0,
                served_by=hit.served_by, finish_reason=hit.finish_reason, cached=True,
                refusal_category=_category(hit)))
            return hit.model_copy(update={"cached": True, "cost_usd": 0.0}), hit
        async with self.gate.admit(request, category) as res:
            self.log(InferenceAttemptEvent(run=self.run_id, round=round, agent=agent, call_id=call_id,
                                           provider=provider.name, model=request.model,
                                           request_hash=h, reserved_usd=res.amount, category=category))
            self.gate.dispatch(res)
            start = time.monotonic()
            try:
                resp = await provider.complete(request)
            except Exception as e:
                self.log(InferenceResponseEvent(
                    run=self.run_id, round=round, agent=agent, call_id=call_id, response_hash="",
                    latency_s=time.monotonic() - start, finish_reason=f"error:{type(e).__name__}",
                    attempts=int(getattr(e, "attempts", 1) or 1)))
                raise
            res.charge(resp.cost_usd)
        resp = resp.model_copy(update={"cached": False})
        body = resp.model_dump_json().encode()
        sha = self.blobs.put(body)
        self.log(InferenceResponseEvent(
            run=self.run_id, round=round, agent=agent, call_id=call_id, response_hash=sha,
            usage=resp.usage.model_dump(), cost_usd=resp.cost_usd, latency_s=resp.latency_s,
            served_by=resp.served_by, finish_reason=resp.finish_reason, cached=False,
            attempts=resp.attempts, refusal_category=_category(resp)))
        self.cache.put(h, resp, round)
        return resp, resp
