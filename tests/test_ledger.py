from __future__ import annotations

from uuid import uuid4

import pytest

from laya_gateway.ledger import InMemoryLedger
from laya_gateway.models import Usage


@pytest.mark.anyio
async def test_ledger_records_logical_request_and_two_attempts():
    ledger = InMemoryLedger()
    request_id = uuid4()
    await ledger.create_request(request_id, "/v1/systemone")
    await ledger.start_attempt(request_id, 1, "laya-primary", None)
    await ledger.finish_attempt(request_id, 1, "connect_failed", error_code="connect_failed")
    await ledger.start_attempt(request_id, 2, "laya-secondary", None)
    await ledger.finish_attempt(
        request_id,
        2,
        "succeeded",
        status_code=200,
        usage=Usage(input_tokens=12, output_tokens=0),
    )
    await ledger.finish_request(
        request_id,
        "succeeded",
        status_code=200,
        worker_id="laya-secondary",
        usage=Usage(input_tokens=12, output_tokens=0),
    )
    request = ledger.requests[request_id]
    assert request["status"] == "succeeded"
    assert request["attempt_count"] == 2
    assert request["input_tokens"] == 12
    assert request["total_tokens"] == 12
    assert len(ledger.attempts[request_id]) == 2


@pytest.mark.anyio
async def test_atomic_finalization_updates_attempt_and_request_together():
    ledger = InMemoryLedger()
    request_id = uuid4()
    usage = Usage(input_tokens=7, output_tokens=0)
    await ledger.create_request(request_id, "/v1/systemone")
    await ledger.start_attempt(request_id, 1, "laya-secondary", None)
    await ledger.finish_attempt_and_request(
        request_id,
        1,
        "succeeded",
        "succeeded",
        status_code=200,
        worker_id="laya-secondary",
        usage=usage,
        attempt_latency_ms=10,
        request_latency_ms=12,
    )
    assert ledger.attempts[request_id][0]["status"] == "succeeded"
    assert ledger.requests[request_id]["status"] == "succeeded"
    assert ledger.requests[request_id]["latency_ms"] == 12


@pytest.mark.anyio
async def test_recovery_marks_stale_records_unknown():
    ledger = InMemoryLedger()
    request_id = uuid4()
    await ledger.create_request(request_id, "/v1/systemone")
    await ledger.start_attempt(request_id, 1, "laya-primary", None)
    recovered = await ledger.recover_stale()
    assert recovered == 1
    assert ledger.requests[request_id]["status"] == "unknown_after_restart"
    assert ledger.attempts[request_id][0]["status"] == "unknown_after_restart"


@pytest.mark.anyio
async def test_ledger_records_worker_state_event_without_secrets():
    ledger = InMemoryLedger()
    instance_id = uuid4()
    await ledger.record_worker_state_event(
        "laya-primary",
        "unknown",
        "healthy",
        instance_id,
        "status_probe",
    )
    assert ledger.worker_state_events == [
        {
            "worker_id": "laya-primary",
            "prior_state": "unknown",
            "new_state": "healthy",
            "worker_instance_id": instance_id,
            "reason": "status_probe",
        }
    ]
