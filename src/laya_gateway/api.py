from __future__ import annotations

import asyncio
import json
import time
from ipaddress import ip_address
from typing import Any
from uuid import uuid4

from fastapi import FastAPI, Request
from fastapi.responses import JSONResponse

from .models import UpstreamResult
from .scheduler import NoEligibleWorkers, QueueFull, QueueTimeout, Scheduler
from .worker_client import (
    AmbiguousWorkerFailure,
    RetryableWorkerFailure,
    WorkerAuthFailure,
    WorkerConnectFailure,
    WorkerProtocolError,
)

MAX_BODY_BYTES = 256 * 1024


def _error(
    status_code: int,
    code: str,
    message: str,
    request_id=None,
    *,
    headers: dict[str, str] | None = None,
) -> JSONResponse:
    body: dict[str, Any] = {"error": {"code": code, "message": message}}
    response_headers = dict(headers or {})
    if request_id is not None:
        body["request_id"] = str(request_id)
        response_headers["X-Request-ID"] = str(request_id)
    return JSONResponse(body, status_code=status_code, headers=response_headers)


def _is_json_content_type(request: Request) -> bool:
    content_type = request.headers.get("content-type", "").split(";", 1)[0].strip().lower()
    return content_type == "application/json" or (
        content_type.startswith("application/") and content_type.endswith("+json")
    )


def _is_loopback_client(request: Request) -> bool:
    if request.client is None:
        return False
    try:
        return ip_address(request.client.host).is_loopback
    except ValueError:
        return request.client.host.lower() == "localhost"


def create_app(registry, scheduler: Scheduler, ledger, *, lifespan=None) -> FastAPI:
    app = FastAPI(title="Laya Gateway", version="0.1.0", lifespan=lifespan)
    app.state.registry = registry
    app.state.scheduler = scheduler
    app.state.ledger = ledger

    @app.get("/health", tags=["operations"])
    async def health() -> dict[str, str]:
        return {"status": "ok"}

    @app.get("/ready", tags=["operations"])
    async def ready():
        database_ready = await ledger.ready()
        workers_ready = bool(registry.eligible())
        if not database_ready or not workers_ready:
            return JSONResponse(
                {"status": "not_ready", "database": database_ready, "workers": workers_ready},
                status_code=503,
            )
        return {"status": "ready", "database": True, "workers": True}

    @app.get("/internal/backends", tags=["operations"])
    async def backends(request: Request):
        if not _is_loopback_client(request):
            return _error(403, "loopback_only", "backend state is available only over loopback")
        return registry.public_snapshot()

    async def infer(endpoint: str, request: Request):
        if not _is_json_content_type(request):
            return _error(415, "unsupported_media_type", "Content-Type must be application/json")

        content_length = request.headers.get("content-length")
        if content_length is not None:
            try:
                if int(content_length) > MAX_BODY_BYTES:
                    return _error(413, "request_too_large", "request body exceeds 256 KiB")
            except ValueError:
                return _error(400, "invalid_content_length", "Content-Length must be an integer")

        raw_buffer = bytearray()
        async for chunk in request.stream():
            if len(raw_buffer) + len(chunk) > MAX_BODY_BYTES:
                return _error(413, "request_too_large", "request body exceeds 256 KiB")
            raw_buffer.extend(chunk)
        raw = bytes(raw_buffer)
        try:
            payload = json.loads(raw)
        except (json.JSONDecodeError, UnicodeDecodeError):
            return _error(400, "invalid_json", "request body must be valid JSON")
        if not isinstance(payload, dict):
            return _error(422, "invalid_request", "request body must be a JSON object")

        request_id = uuid4()
        request_started = time.monotonic()
        await ledger.create_request(request_id, endpoint)
        excluded: set[str] = set()
        attempts = 0

        while attempts < 2:
            try:
                lease = await scheduler.acquire(exclude=excluded)
            except NoEligibleWorkers:
                latency = int((time.monotonic() - request_started) * 1000)
                await ledger.finish_request(
                    request_id,
                    "failed",
                    status_code=503,
                    latency_ms=latency,
                    error_code="no_workers_available",
                )
                return _error(
                    503,
                    "no_workers_available",
                    "no eligible Laya workers",
                    request_id,
                    headers={"Retry-After": "1"},
                )
            except QueueFull:
                latency = int((time.monotonic() - request_started) * 1000)
                await ledger.finish_request(
                    request_id,
                    "failed",
                    status_code=503,
                    latency_ms=latency,
                    error_code="gateway_queue_full",
                )
                return _error(
                    503,
                    "gateway_queue_full",
                    "gateway request queue is full",
                    request_id,
                    headers={"Retry-After": "1"},
                )
            except QueueTimeout:
                latency = int((time.monotonic() - request_started) * 1000)
                await ledger.finish_request(
                    request_id,
                    "failed",
                    status_code=503,
                    latency_ms=latency,
                    error_code="gateway_queue_timeout",
                )
                return _error(
                    503,
                    "gateway_queue_timeout",
                    "gateway request queue timed out",
                    request_id,
                    headers={"Retry-After": "1"},
                )

            attempts += 1
            worker = lease.worker
            worker_id = worker.config.id
            excluded.add(worker_id)
            instance_id = worker.status.instance_id if worker.status else None
            try:
                await ledger.start_attempt(request_id, attempts, worker_id, instance_id)
            except BaseException:
                await lease.release()
                raise
            started = time.monotonic()
            try:
                result: UpstreamResult = await worker.client.infer(endpoint, payload, request_id)
            except (WorkerConnectFailure, RetryableWorkerFailure) as exc:
                attempt_latency = int((time.monotonic() - started) * 1000)
                request_latency = int((time.monotonic() - request_started) * 1000)
                attempt_status = (
                    "connect_failed" if isinstance(exc, WorkerConnectFailure) else "upstream_failed"
                )
                error_code = type(exc).__name__
                registry.record_request_failure(worker_id)
                if attempts >= 2:
                    await ledger.finish_attempt_and_request(
                        request_id,
                        attempts,
                        attempt_status,
                        "failed",
                        status_code=503,
                        worker_id=worker_id,
                        error_code=error_code,
                        attempt_latency_ms=attempt_latency,
                        request_latency_ms=request_latency,
                    )
                    return _error(
                        503,
                        "workers_failed",
                        "all attempted workers failed",
                        request_id,
                        headers={"Retry-After": "1"},
                    )
                await ledger.finish_attempt(
                    request_id,
                    attempts,
                    attempt_status,
                    error_code=error_code,
                    latency_ms=attempt_latency,
                )
                continue
            except AmbiguousWorkerFailure as exc:
                attempt_latency = int((time.monotonic() - started) * 1000)
                request_latency = int((time.monotonic() - request_started) * 1000)
                error_code = type(exc).__name__
                registry.record_request_failure(worker_id)
                await ledger.finish_attempt_and_request(
                    request_id,
                    attempts,
                    "unknown",
                    "unknown",
                    status_code=504,
                    worker_id=worker_id,
                    error_code=error_code,
                    attempt_latency_ms=attempt_latency,
                    request_latency_ms=request_latency,
                )
                return _error(
                    504,
                    "upstream_outcome_unknown",
                    "the worker outcome is unknown; request was not retried",
                    request_id,
                )
            except (WorkerAuthFailure, WorkerProtocolError) as exc:
                attempt_latency = int((time.monotonic() - started) * 1000)
                request_latency = int((time.monotonic() - request_started) * 1000)
                error_code = type(exc).__name__
                registry.record_request_failure(worker_id)
                await ledger.finish_attempt_and_request(
                    request_id,
                    attempts,
                    "protocol_failed",
                    "failed",
                    status_code=502,
                    worker_id=worker_id,
                    error_code=error_code,
                    attempt_latency_ms=attempt_latency,
                    request_latency_ms=request_latency,
                )
                return _error(502, "worker_protocol_error", "worker protocol failure", request_id)
            except asyncio.CancelledError:
                attempt_latency = int((time.monotonic() - started) * 1000)
                request_latency = int((time.monotonic() - request_started) * 1000)
                registry.record_request_failure(worker_id)
                await asyncio.shield(
                    ledger.finish_attempt_and_request(
                        request_id,
                        attempts,
                        "unknown",
                        "unknown",
                        status_code=499,
                        worker_id=worker_id,
                        error_code="client_cancelled_unknown",
                        attempt_latency_ms=attempt_latency,
                        request_latency_ms=request_latency,
                    )
                )
                raise
            finally:
                await lease.release()

            attempt_latency = int((time.monotonic() - started) * 1000)
            request_latency = int((time.monotonic() - request_started) * 1000)
            if result.status_code >= 400:
                await ledger.finish_attempt_and_request(
                    request_id,
                    attempts,
                    "client_error",
                    "failed",
                    status_code=result.status_code,
                    worker_id=worker_id,
                    attempt_latency_ms=attempt_latency,
                    request_latency_ms=request_latency,
                )
            else:
                await ledger.finish_attempt_and_request(
                    request_id,
                    attempts,
                    "succeeded",
                    "succeeded",
                    status_code=result.status_code,
                    worker_id=worker_id,
                    usage=result.usage,
                    instance_id=result.instance_id,
                    attempt_latency_ms=attempt_latency,
                    request_latency_ms=request_latency,
                )
            response_headers = {
                "X-Request-ID": str(request_id),
                "X-Laya-Worker-ID": worker_id,
                "X-Laya-Attempts": str(attempts),
            }
            if result.instance_id is not None:
                response_headers["X-Laya-Worker-Instance"] = str(result.instance_id)
            return JSONResponse(
                result.body,
                status_code=result.status_code,
                headers=response_headers,
            )

        latency = int((time.monotonic() - request_started) * 1000)
        await ledger.finish_request(
            request_id,
            "failed",
            status_code=503,
            latency_ms=latency,
            error_code="workers_failed",
        )
        return _error(503, "workers_failed", "all attempted workers failed", request_id)

    @app.post("/v1/systemone", tags=["inference"])
    async def systemone(request: Request):
        return await infer("/v1/systemone", request)

    @app.post("/v1/evaluate", tags=["inference"])
    async def evaluate(request: Request):
        return await infer("/v1/evaluate", request)

    return app
