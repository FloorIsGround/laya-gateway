from __future__ import annotations

import asyncio
import time
from collections.abc import Awaitable, Callable, Iterable
from dataclasses import dataclass
from typing import Any
from uuid import UUID

from .models import WorkerConfig, WorkerStatus


@dataclass
class WorkerRecord:
    config: WorkerConfig
    client: Any
    status: WorkerStatus | None = None
    state: str = "unknown"
    last_success: float | None = None
    consecutive_successes: int = 0
    consecutive_failures: int = 0
    circuit_open_until: float = 0.0
    instance_changes: int = 0
    last_error: str | None = None


class WorkerRegistry:
    def __init__(
        self,
        workers: Iterable[tuple[WorkerConfig, Any]],
        *,
        clock: Callable[[], float] = time.monotonic,
        probe_interval: float = 2.0,
        stale_after: float = 6.0,
        failures_to_unhealthy: int = 2,
        successes_to_healthy: int = 2,
        on_state_change: Callable[[str, str, str, UUID | None, str], Awaitable[None]] | None = None,
    ):
        self.records = {
            config.id: WorkerRecord(config=config, client=client)
            for config, client in workers
        }
        self.clock = clock
        self.probe_interval = probe_interval
        self.stale_after = stale_after
        self.failures_to_unhealthy = failures_to_unhealthy
        self.successes_to_healthy = successes_to_healthy
        self.on_state_change = on_state_change

    async def probe_once(self, worker_id: str) -> None:
        record = self.records[worker_id]
        prior_state = record.state
        try:
            observed = await record.client.probe()
        except Exception as exc:  # noqa: BLE001 - one broken probe must not stop registry polling
            record.consecutive_failures += 1
            record.consecutive_successes = 0
            record.last_error = type(exc).__name__
            if record.consecutive_failures >= self.failures_to_unhealthy:
                record.state = "unhealthy"
            await self._emit_state_change(record, prior_state, f"probe_failure:{type(exc).__name__}")
            return

        previous_instance = record.status.instance_id if record.status is not None else None
        if previous_instance is not None and previous_instance != observed.instance_id:
            record.instance_changes += 1
        record.status = observed
        record.last_success = self.clock()
        record.consecutive_failures = 0
        record.last_error = None
        if observed.status != "ready":
            record.consecutive_successes = 0
            record.state = "degraded" if observed.status == "degraded" else "unknown"
        else:
            record.consecutive_successes += 1
            if record.consecutive_successes >= self.successes_to_healthy:
                record.state = "healthy"
        await self._emit_state_change(record, prior_state, "status_probe")

    async def _emit_state_change(self, record: WorkerRecord, prior_state: str, reason: str) -> None:
        if self.on_state_change is None or record.state == prior_state:
            return
        instance_id = record.status.instance_id if record.status is not None else None
        await self.on_state_change(record.config.id, prior_state, record.state, instance_id, reason)

    def record_request_failure(self, worker_id: str, *, cooldown: float = 5.0) -> None:
        record = self.records[worker_id]
        record.circuit_open_until = max(record.circuit_open_until, self.clock() + cooldown)

    def eligible(self, exclude: set[str] | None = None) -> list[WorkerRecord]:
        excluded = exclude or set()
        now = self.clock()
        eligible: list[WorkerRecord] = []
        for record in self.records.values():
            if record.config.id in excluded or record.state != "healthy":
                continue
            if record.status is None or record.last_success is None:
                continue
            if now - record.last_success >= self.stale_after:
                continue
            if now < record.circuit_open_until:
                continue
            eligible.append(record)
        return eligible

    async def run(self, stop: asyncio.Event) -> None:
        while not stop.is_set():
            await asyncio.gather(
                *(self.probe_once(worker_id) for worker_id in self.records),
                return_exceptions=True,
            )
            try:
                await asyncio.wait_for(stop.wait(), timeout=self.probe_interval)
            except TimeoutError:
                pass

    def public_snapshot(self) -> list[dict[str, object]]:
        now = self.clock()
        result: list[dict[str, object]] = []
        for record in self.records.values():
            status = record.status
            result.append(
                {
                    "worker_id": record.config.id,
                    "state": record.state,
                    "fresh": record.last_success is not None
                    and now - record.last_success < self.stale_after,
                    "circuit_open": now < record.circuit_open_until,
                    "instance_id": str(status.instance_id) if status else None,
                    "actual_device": status.actual_device if status else None,
                    "capacity": status.capacity if status else None,
                    "max_context": status.max_context if status else None,
                    "head_max_length": status.head_max_length if status else None,
                    "inflight": status.inflight if status else None,
                    "last_error": record.last_error,
                }
            )
        return result
