from __future__ import annotations

import asyncio
import time
from collections import deque
from dataclasses import dataclass
from typing import Protocol

from .registry import WorkerRecord


class NoEligibleWorkers(RuntimeError):
    pass


class QueueFull(RuntimeError):
    pass


class QueueTimeout(RuntimeError):
    pass


class RegistryLike(Protocol):
    def eligible(self, exclude: set[str] | None = None) -> list[WorkerRecord]: ...


@dataclass
class WorkerLease:
    scheduler: Scheduler
    worker: WorkerRecord
    reservation_id: object
    released: bool = False

    async def release(self) -> None:
        if self.released:
            return
        self.released = True
        await self.scheduler._release(self.worker.config.id, self.reservation_id)

    async def __aenter__(self) -> WorkerRecord:
        return self.worker

    async def __aexit__(self, exc_type, exc, tb) -> None:
        await self.release()


class Scheduler:
    def __init__(
        self,
        registry: RegistryLike,
        *,
        queue_capacity: int = 64,
        queue_timeout: float = 30.0,
        availability_poll_interval: float = 0.05,
    ):
        self.registry = registry
        self.queue_capacity = queue_capacity
        self.queue_timeout = queue_timeout
        self.availability_poll_interval = availability_poll_interval
        self._reservations: dict[str, dict[object, float]] = {}
        self._condition = asyncio.Condition()
        self._tickets: deque[object] = deque()

    def _effective_load(self, worker: WorkerRecord) -> int:
        reported = worker.status.inflight if worker.status is not None else 0
        reservations = self._reservations.get(worker.config.id, {})
        if worker.last_success is None:
            unreported = len(reservations)
        else:
            unreported = sum(acquired_at > worker.last_success for acquired_at in reservations.values())
        return reported + unreported

    def _select(self, exclude: set[str]) -> WorkerRecord | None:
        candidates = []
        for worker in self.registry.eligible(exclude):
            if worker.status is None:
                continue
            effective = self._effective_load(worker)
            if effective >= worker.status.capacity:
                continue
            candidates.append(
                (
                    effective / worker.status.capacity,
                    effective,
                    worker.config.tie_rank,
                    worker.config.id,
                    worker,
                )
            )
        if not candidates:
            return None
        return min(candidates, key=lambda item: item[:4])[4]

    async def acquire(self, exclude: set[str] | None = None) -> WorkerLease:
        excluded = exclude or set()
        if not self.registry.eligible(excluded):
            raise NoEligibleWorkers("no eligible Laya workers")
        ticket = object()
        deadline = time.monotonic() + self.queue_timeout
        async with self._condition:
            if len(self._tickets) >= self.queue_capacity:
                raise QueueFull("gateway request queue is full")
            self._tickets.append(ticket)
            try:
                while True:
                    if self._tickets[0] is ticket:
                        selected = self._select(excluded)
                        if selected is not None:
                            self._tickets.popleft()
                            reservation_id = object()
                            worker_id = selected.config.id
                            self._reservations.setdefault(worker_id, {})[reservation_id] = time.monotonic()
                            self._condition.notify_all()
                            return WorkerLease(self, selected, reservation_id)
                        if not self.registry.eligible(excluded):
                            raise NoEligibleWorkers("no eligible Laya workers")
                    remaining = deadline - time.monotonic()
                    if remaining <= 0:
                        raise QueueTimeout("gateway request queue timed out")
                    try:
                        await asyncio.wait_for(
                            self._condition.wait(),
                            timeout=min(remaining, self.availability_poll_interval),
                        )
                    except TimeoutError:
                        if time.monotonic() >= deadline:
                            raise QueueTimeout("gateway request queue timed out") from None
                        continue
            except BaseException:
                if ticket in self._tickets:
                    self._tickets.remove(ticket)
                    self._condition.notify_all()
                raise

    async def _release(self, worker_id: str, reservation_id: object) -> None:
        async with self._condition:
            reservations = self._reservations.get(worker_id)
            if reservations is not None:
                reservations.pop(reservation_id, None)
                if not reservations:
                    self._reservations.pop(worker_id, None)
            self._condition.notify_all()
