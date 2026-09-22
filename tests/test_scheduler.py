from __future__ import annotations

import asyncio
from uuid import UUID

import pytest

from laya_gateway.models import WorkerConfig, WorkerStatus
from laya_gateway.registry import WorkerRecord
from laya_gateway.scheduler import NoEligibleWorkers, QueueFull, QueueTimeout, Scheduler


def status(worker_id: str, inflight: int = 0, capacity: int = 2) -> WorkerStatus:
    return WorkerStatus(
        protocol_version=1,
        status="ready",
        worker_id=worker_id,
        instance_id=UUID("2cadb755-3c83-4b48-9e54-54dbf4767836"),
        service_version="0.2.0",
        laya_version="0.3.4",
        model="convaiinnovations/laya",
        desired_device="cuda",
        actual_device="cuda",
        dtype="torch.bfloat16",
        capacity=capacity,
        queue_capacity=64,
        inflight=inflight,
        executing=min(inflight, capacity),
        queued=max(inflight - capacity, 0),
        completed_total=0,
        failed_total=0,
        queue_rejected_total=0,
        uptime_seconds=1,
    )


def record(worker_id: str, tie_rank: int, inflight: int = 0) -> WorkerRecord:
    cfg = WorkerConfig(worker_id, f"http://{worker_id}", "KEY", "secret", tie_rank, "cuda")
    return WorkerRecord(config=cfg, client=None, status=status(worker_id, inflight), state="healthy", last_success=1.0)


class StubRegistry:
    def __init__(self, records):
        self._records = records

    def eligible(self, exclude=None):
        excluded = exclude or set()
        return [record for record in self._records if record.config.id not in excluded]


@pytest.mark.anyio
async def test_scheduler_prefers_115_on_idle_tie_then_113_when_115_busy():
    r115 = record("laya-primary", 0)
    r113 = record("laya-secondary", 1)
    scheduler = Scheduler(StubRegistry([r115, r113]), queue_capacity=4, queue_timeout=0.1)
    first = await scheduler.acquire()
    assert first.worker.config.id == "laya-primary"
    second = await scheduler.acquire()
    assert second.worker.config.id == "laya-secondary"
    await second.release()
    await first.release()


@pytest.mark.anyio
async def test_scheduler_uses_reported_load_and_exclusions():
    scheduler = Scheduler(
        StubRegistry([record("laya-primary", 0, inflight=1), record("laya-secondary", 1)]),
        queue_capacity=4,
        queue_timeout=0.1,
    )
    lease = await scheduler.acquire()
    assert lease.worker.config.id == "laya-secondary"
    await lease.release()
    lease = await scheduler.acquire(exclude={"laya-primary"})
    assert lease.worker.config.id == "laya-secondary"
    await lease.release()


@pytest.mark.anyio
async def test_scheduler_reports_no_eligible_workers():
    scheduler = Scheduler(StubRegistry([]), queue_capacity=1, queue_timeout=0.01)
    with pytest.raises(NoEligibleWorkers):
        await scheduler.acquire()


@pytest.mark.anyio
async def test_scheduler_bounds_waiting_queue():
    busy = record("laya-primary", 0, inflight=2)
    scheduler = Scheduler(StubRegistry([busy]), queue_capacity=0, queue_timeout=0.01)
    with pytest.raises(QueueFull):
        await scheduler.acquire()


@pytest.mark.anyio
async def test_scheduler_adds_new_reservations_to_reported_load():
    r115 = record("laya-primary", 0, inflight=1)
    r113 = record("laya-secondary", 1, inflight=1)
    scheduler = Scheduler(StubRegistry([r115, r113]), queue_capacity=2, queue_timeout=0.02)
    first = await scheduler.acquire()
    assert first.worker.config.id == "laya-primary"
    second = await scheduler.acquire()
    assert second.worker.config.id == "laya-secondary"
    with pytest.raises(QueueTimeout, match="timed out"):
        await scheduler.acquire()
    await first.release()
    await second.release()


@pytest.mark.anyio
async def test_scheduler_rechecks_registry_without_release_notification():
    busy = record("laya-primary", 0, inflight=2)
    registry = StubRegistry([busy])
    scheduler = Scheduler(registry, queue_capacity=2, queue_timeout=0.2)
    acquire_task = asyncio.create_task(scheduler.acquire())
    await asyncio.sleep(0.02)
    busy.status = status("laya-primary", inflight=0)
    lease = await asyncio.wait_for(acquire_task, timeout=0.2)
    assert lease.worker.config.id == "laya-primary"
    await lease.release()
