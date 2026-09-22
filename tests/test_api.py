from __future__ import annotations

from uuid import UUID

import httpx
import pytest

from laya_gateway.api import create_app
from laya_gateway.ledger import InMemoryLedger
from laya_gateway.models import UpstreamResult, Usage, WorkerConfig, WorkerStatus
from laya_gateway.registry import WorkerRecord
from laya_gateway.scheduler import Scheduler
from laya_gateway.worker_client import AmbiguousWorkerFailure, WorkerConnectFailure


def status(worker_id: str) -> WorkerStatus:
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


class FakeClient:
    def __init__(self, outcome):
        self.outcome = outcome
        self.calls = 0

    async def infer(self, endpoint, payload, request_id):
        self.calls += 1
        if isinstance(self.outcome, Exception):
            raise self.outcome
        return self.outcome


class Registry:
    def __init__(self, records):
        self.records = {record.config.id: record for record in records}
        self.failures = []

    def eligible(self, exclude=None):
        excluded = exclude or set()
        return [record for record in self.records.values() if record.config.id not in excluded]

    def record_request_failure(self, worker_id, cooldown=5.0):
        self.failures.append(worker_id)

    def public_snapshot(self):
        return [{"worker_id": worker_id, "state": "healthy"} for worker_id in self.records]


def record(worker_id: str, tie_rank: int, client) -> WorkerRecord:
    config = WorkerConfig(worker_id, f"http://{worker_id}", "KEY", "secret", tie_rank, "cuda")
    return WorkerRecord(config=config, client=client, status=status(worker_id), state="healthy", last_success=1.0)


def success(worker_id: str) -> UpstreamResult:
    usage = Usage(input_tokens=12, output_tokens=0)
    return UpstreamResult(
        200,
        {"model": "laya", "answers": {}, "usage": usage.model_dump()},
        usage,
        worker_id,
        UUID("2cadb755-3c83-4b48-9e54-54dbf4767836"),
    )


def build_app(client_115, client_113):
    registry = Registry([
        record("laya-primary", 0, client_115),
        record("laya-secondary", 1, client_113),
    ])
    ledger = InMemoryLedger()
    scheduler = Scheduler(registry, queue_capacity=8, queue_timeout=0.1)
    return create_app(registry, scheduler, ledger), registry, ledger


@pytest.mark.anyio
async def test_gateway_prefers_115_and_records_usage():
    first = FakeClient(success("laya-primary"))
    second = FakeClient(success("laya-secondary"))
    app, _, ledger = build_app(first, second)
    async with httpx.AsyncClient(transport=httpx.ASGITransport(app=app), base_url="http://test") as client:
        response = await client.post("/v1/systemone", json={"state": "x", "questions": {}})
    assert response.status_code == 200
    assert response.headers["X-Laya-Worker-ID"] == "laya-primary"
    assert response.headers["X-Laya-Attempts"] == "1"
    assert first.calls == 1 and second.calls == 0
    saved = next(iter(ledger.requests.values()))
    assert saved["input_tokens"] == 12
    assert "state" not in saved


@pytest.mark.anyio
async def test_evaluate_alias_forwards_the_compatibility_path():
    first = FakeClient(success("laya-primary"))
    second = FakeClient(success("laya-secondary"))
    app, _, _ = build_app(first, second)
    async with httpx.AsyncClient(transport=httpx.ASGITransport(app=app), base_url="http://test") as client:
        response = await client.post("/v1/evaluate", json={"state": "x", "questions": {}})
    assert response.status_code == 200
    assert first.calls == 1


@pytest.mark.anyio
async def test_gateway_retries_definite_connect_failure_on_other_worker():
    first = FakeClient(WorkerConnectFailure("down"))
    second = FakeClient(success("laya-secondary"))
    app, registry, ledger = build_app(first, second)
    async with httpx.AsyncClient(transport=httpx.ASGITransport(app=app), base_url="http://test") as client:
        response = await client.post("/v1/systemone", json={"state": "x", "questions": {}})
    assert response.status_code == 200
    assert response.headers["X-Laya-Worker-ID"] == "laya-secondary"
    assert response.headers["X-Laya-Attempts"] == "2"
    assert registry.failures == ["laya-primary"]
    assert len(next(iter(ledger.attempts.values()))) == 2


@pytest.mark.anyio
async def test_gateway_does_not_retry_ambiguous_failure():
    first = FakeClient(AmbiguousWorkerFailure("late"))
    second = FakeClient(success("laya-secondary"))
    app, _, ledger = build_app(first, second)
    async with httpx.AsyncClient(transport=httpx.ASGITransport(app=app), base_url="http://test") as client:
        response = await client.post("/v1/systemone", json={"state": "x", "questions": {}})
    assert response.status_code == 504
    assert second.calls == 0
    saved = next(iter(ledger.requests.values()))
    assert saved["status"] == "unknown"


@pytest.mark.anyio
async def test_health_ready_and_backend_snapshot():
    app, _, _ = build_app(FakeClient(success("laya-primary")), FakeClient(success("laya-secondary")))
    async with httpx.AsyncClient(transport=httpx.ASGITransport(app=app), base_url="http://test") as client:
        assert (await client.get("/health")).status_code == 200
        assert (await client.get("/ready")).status_code == 200
        backends = await client.get("/internal/backends")
    assert {item["worker_id"] for item in backends.json()} == {"laya-primary", "laya-secondary"}


@pytest.mark.anyio
async def test_backend_snapshot_is_loopback_only():
    app, _, _ = build_app(FakeClient(success("laya-primary")), FakeClient(success("laya-secondary")))
    transport = httpx.ASGITransport(app=app, client=("192.0.2.50", 12345))
    async with httpx.AsyncClient(transport=transport, base_url="http://test") as client:
        response = await client.get("/internal/backends")
    assert response.status_code == 403
    assert response.json()["error"]["code"] == "loopback_only"


@pytest.mark.anyio
async def test_inference_requires_json_content_type_before_ledger_write():
    app, _, ledger = build_app(FakeClient(success("laya-primary")), FakeClient(success("laya-secondary")))
    async with httpx.AsyncClient(transport=httpx.ASGITransport(app=app), base_url="http://test") as client:
        response = await client.post("/v1/systemone", content=b"{}", headers={"Content-Type": "text/plain"})
    assert response.status_code == 415
    assert ledger.requests == {}


@pytest.mark.anyio
async def test_gateway_rejects_streamed_oversized_body_before_ledger_write():
    app, _, ledger = build_app(FakeClient(success("laya-primary")), FakeClient(success("laya-secondary")))

    async def oversized_chunks():
        for _ in range(5):
            yield b"x" * (64 * 1024)

    async with httpx.AsyncClient(transport=httpx.ASGITransport(app=app), base_url="http://test") as client:
        response = await client.post(
            "/v1/systemone",
            content=oversized_chunks(),
            headers={"Content-Type": "application/json"},
        )
    assert response.status_code == 413
    assert ledger.requests == {}
