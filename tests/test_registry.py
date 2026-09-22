from __future__ import annotations

from uuid import UUID

import pytest

from laya_gateway.models import WorkerConfig, WorkerStatus
from laya_gateway.registry import WorkerRegistry

BASE = WorkerStatus(
    protocol_version=1,
    status="ready",
    worker_id="laya-primary",
    instance_id=UUID("2cadb755-3c83-4b48-9e54-54dbf4767836"),
    service_version="0.2.0",
    laya_version="0.3.4",
    model="convaiinnovations/laya",
    desired_device="cuda",
    actual_device="cuda",
    dtype="torch.bfloat16",
    capacity=2,
    queue_capacity=64,
    inflight=0,
    executing=0,
    queued=0,
    completed_total=0,
    failed_total=0,
    queue_rejected_total=0,
    uptime_seconds=1,
)


def config(worker_id: str = "laya-primary") -> WorkerConfig:
    return WorkerConfig(worker_id, f"http://{worker_id}", "KEY", "secret", 0, "cuda")


class FakeClient:
    def __init__(self, outcomes):
        self.outcomes = iter(outcomes)

    async def probe(self):
        outcome = next(self.outcomes)
        if isinstance(outcome, Exception):
            raise outcome
        return outcome


@pytest.mark.anyio
async def test_registry_requires_two_successes_and_tracks_instance_change():
    now = [10.0]
    second_instance = UUID("6800471f-6d1f-44b1-8263-781e2d4c00eb")
    client = FakeClient([BASE, BASE, BASE.model_copy(update={"instance_id": second_instance})])
    registry = WorkerRegistry([(config(), client)], clock=lambda: now[0])
    await registry.probe_once("laya-primary")
    assert registry.eligible() == []
    await registry.probe_once("laya-primary")
    assert [record.config.id for record in registry.eligible()] == ["laya-primary"]
    await registry.probe_once("laya-primary")
    record = registry.records["laya-primary"]
    assert record.status.instance_id == second_instance
    assert record.instance_changes == 1


@pytest.mark.anyio
async def test_registry_marks_stale_and_opens_short_circuit():
    now = [10.0]
    client = FakeClient([BASE, BASE])
    registry = WorkerRegistry([(config(), client)], clock=lambda: now[0], stale_after=6.0)
    await registry.probe_once("laya-primary")
    await registry.probe_once("laya-primary")
    assert registry.eligible()
    registry.record_request_failure("laya-primary", cooldown=5.0)
    assert registry.eligible() == []
    now[0] = 16.0
    assert registry.eligible() == []


@pytest.mark.anyio
async def test_registry_becomes_unhealthy_after_two_failures():
    now = [1.0]
    client = FakeClient([BASE, BASE, RuntimeError("down"), RuntimeError("down")])
    registry = WorkerRegistry([(config(), client)], clock=lambda: now[0])
    await registry.probe_once("laya-primary")
    await registry.probe_once("laya-primary")
    await registry.probe_once("laya-primary")
    assert registry.records["laya-primary"].state == "healthy"
    await registry.probe_once("laya-primary")
    assert registry.records["laya-primary"].state == "unhealthy"


@pytest.mark.anyio
async def test_registry_emits_worker_state_transitions():
    events = []

    async def record_event(worker_id, prior_state, new_state, instance_id, reason):
        events.append((worker_id, prior_state, new_state, instance_id, reason))

    client = FakeClient([BASE, BASE, RuntimeError("down"), RuntimeError("down")])
    registry = WorkerRegistry([(config(), client)], on_state_change=record_event)
    await registry.probe_once("laya-primary")
    await registry.probe_once("laya-primary")
    await registry.probe_once("laya-primary")
    await registry.probe_once("laya-primary")
    assert [(event[1], event[2]) for event in events] == [
        ("unknown", "healthy"),
        ("healthy", "unhealthy"),
    ]
