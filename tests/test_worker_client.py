from __future__ import annotations

import httpx
import pytest

from laya_gateway.models import WorkerConfig
from laya_gateway.worker_client import (
    AmbiguousWorkerFailure,
    RetryableWorkerFailure,
    WorkerAuthFailure,
    WorkerClient,
    WorkerProtocolError,
)

STATUS = {
    "protocol_version": 1,
    "status": "ready",
    "worker_id": "laya-primary",
    "instance_id": "2cadb755-3c83-4b48-9e54-54dbf4767836",
    "service_version": "0.2.0",
    "laya_version": "0.3.4",
    "model": "convaiinnovations/laya",
    "desired_device": "cuda",
    "actual_device": "cuda:0",
    "dtype": "torch.bfloat16",
    "capacity": 2,
    "queue_capacity": 64,
    "inflight": 0,
    "executing": 0,
    "queued": 0,
    "completed_total": 10,
    "failed_total": 1,
    "queue_rejected_total": 0,
    "uptime_seconds": 50,
}


def worker() -> WorkerConfig:
    return WorkerConfig(
        id="laya-primary",
        base_url="http://worker.test:8010",
        api_key_env="TEST_KEY",
        api_key="test",
        tie_rank=0,
        expected_device="cuda",
    )


@pytest.mark.anyio
async def test_probe_validates_identity_and_device():
    async def handler(request: httpx.Request) -> httpx.Response:
        assert request.headers["Authorization"] == "Bearer test"
        return httpx.Response(200, json=STATUS)

    async with httpx.AsyncClient(transport=httpx.MockTransport(handler)) as http:
        status = await WorkerClient(worker(), http).probe()
    assert status.worker_id == "laya-primary"
    assert status.capacity == 2


@pytest.mark.anyio
async def test_probe_accepts_context_capabilities_and_rejects_bad_fingerprint():
    context_status = {
        **STATUS,
        "max_context": 4096,
        "head_max_length": 192,
        "tokenizer_fingerprint": "a" * 64,
    }
    async with httpx.AsyncClient(
        transport=httpx.MockTransport(lambda request: httpx.Response(200, json=context_status))
    ) as http:
        observed = await WorkerClient(worker(), http).probe()
    assert observed.max_context == 4096
    assert observed.head_max_length == 192
    assert observed.tokenizer_fingerprint == "a" * 64
    context_status["tokenizer_fingerprint"] = "invalid"
    async with httpx.AsyncClient(
        transport=httpx.MockTransport(lambda request: httpx.Response(200, json=context_status))
    ) as http:
        with pytest.raises(WorkerProtocolError):
            await WorkerClient(worker(), http).probe()


@pytest.mark.anyio
async def test_probe_rejects_wrong_identity():
    bad = {**STATUS, "worker_id": "wrong"}
    async with httpx.AsyncClient(
        transport=httpx.MockTransport(lambda request: httpx.Response(200, json=bad))
    ) as http:
        with pytest.raises(WorkerProtocolError):
            await WorkerClient(worker(), http).probe()


@pytest.mark.anyio
async def test_infer_requires_usage_and_matching_worker_headers():
    body = {"model": "laya", "answers": {}, "usage": {"input_tokens": 12, "output_tokens": 0}}

    async def handler(request: httpx.Request) -> httpx.Response:
        assert request.headers["X-Request-ID"] == "request-1"
        return httpx.Response(
            200,
            json=body,
            headers={
                "X-Request-ID": "request-1",
                "X-Laya-Worker-ID": "laya-primary",
                "X-Laya-Worker-Instance": STATUS["instance_id"],
                "X-Laya-Protocol-Version": "1",
            },
        )

    async with httpx.AsyncClient(transport=httpx.MockTransport(handler)) as http:
        result = await WorkerClient(worker(), http).infer(
            "/v1/systemone", {"state": "x", "questions": {}}, "request-1"
        )
    assert result.usage.input_tokens == 12
    assert result.worker_id == "laya-primary"
    assert result.body == body


@pytest.mark.anyio
async def test_infer_rejects_success_without_valid_usage():
    async with httpx.AsyncClient(
        transport=httpx.MockTransport(
            lambda request: httpx.Response(
                200,
                json={"answers": {}, "usage": {"input_tokens": True, "output_tokens": 0}},
                headers={
                    "X-Request-ID": "r",
                    "X-Laya-Worker-ID": "laya-primary",
                    "X-Laya-Worker-Instance": STATUS["instance_id"],
                    "X-Laya-Protocol-Version": "1",
                },
            )
        )
    ) as http:
        with pytest.raises(WorkerProtocolError):
            await WorkerClient(worker(), http).infer("/v1/systemone", {}, "r")


@pytest.mark.anyio
@pytest.mark.parametrize("status", [401, 403])
async def test_infer_classifies_auth_failure(status: int):
    async with httpx.AsyncClient(
        transport=httpx.MockTransport(lambda request: httpx.Response(status, json={"detail": "no"}))
    ) as http:
        with pytest.raises(WorkerAuthFailure):
            await WorkerClient(worker(), http).infer("/v1/systemone", {}, "r")


@pytest.mark.anyio
async def test_infer_classifies_retryable_operational_failure():
    async with httpx.AsyncClient(
        transport=httpx.MockTransport(
            lambda request: httpx.Response(
                503, json={"detail": {"code": "queue_timeout", "message": "busy"}}
            )
        )
    ) as http:
        with pytest.raises(RetryableWorkerFailure):
            await WorkerClient(worker(), http).infer("/v1/systemone", {}, "r")


@pytest.mark.anyio
async def test_infer_does_not_hide_client_validation_response():
    body = {"detail": [{"loc": ["body"], "msg": "invalid"}]}
    async with httpx.AsyncClient(
        transport=httpx.MockTransport(lambda request: httpx.Response(422, json=body))
    ) as http:
        result = await WorkerClient(worker(), http).infer("/v1/systemone", {}, "r")
    assert result.status_code == 422
    assert result.body == body
    assert result.usage is None


@pytest.mark.anyio
async def test_infer_classifies_read_timeout_as_ambiguous():
    async def handler(request: httpx.Request) -> httpx.Response:
        raise httpx.ReadTimeout("late", request=request)

    async with httpx.AsyncClient(transport=httpx.MockTransport(handler)) as http:
        with pytest.raises(AmbiguousWorkerFailure):
            await WorkerClient(worker(), http).infer("/v1/systemone", {}, "r")


@pytest.mark.anyio
async def test_infer_non_json_5xx_is_retryable():
    async def handler(_: httpx.Request) -> httpx.Response:
        return httpx.Response(502, text="upstream exploded")

    async with httpx.AsyncClient(transport=httpx.MockTransport(handler)) as http:
        with pytest.raises(RetryableWorkerFailure):
            await WorkerClient(worker(), http).infer("/v1/systemone", {}, "r")
