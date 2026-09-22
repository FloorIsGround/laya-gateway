from __future__ import annotations

from datetime import UTC, datetime
from typing import Any
from uuid import UUID

from psycopg_pool import AsyncConnectionPool

from .models import Usage


def _utcnow() -> datetime:
    return datetime.now(UTC)


class InMemoryLedger:
    def __init__(self) -> None:
        self.requests: dict[UUID, dict[str, Any]] = {}
        self.attempts: dict[UUID, list[dict[str, Any]]] = {}
        self.worker_state_events: list[dict[str, Any]] = []

    async def open(self) -> None:
        return None

    async def close(self) -> None:
        return None

    async def ready(self) -> bool:
        return True

    async def create_request(self, request_id: UUID, endpoint: str) -> None:
        self.requests[request_id] = {
            "request_id": request_id,
            "endpoint": endpoint,
            "received_at": _utcnow(),
            "status": "received",
            "status_code": None,
            "worker_id": None,
            "input_tokens": None,
            "output_tokens": None,
            "total_tokens": None,
            "latency_ms": None,
            "error_code": None,
            "attempt_count": 0,
        }
        self.attempts[request_id] = []

    async def start_attempt(
        self,
        request_id: UUID,
        attempt_no: int,
        worker_id: str,
        instance_id: UUID | None,
    ) -> None:
        self.attempts[request_id].append(
            {
                "attempt_no": attempt_no,
                "worker_id": worker_id,
                "worker_instance_id": instance_id,
                "started_at": _utcnow(),
                "status": "started",
            }
        )
        self.requests[request_id]["attempt_count"] = attempt_no

    async def finish_attempt(
        self,
        request_id: UUID,
        attempt_no: int,
        attempt_status: str,
        *,
        status_code: int | None = None,
        error_code: str | None = None,
        usage: Usage | None = None,
        instance_id: UUID | None = None,
        latency_ms: int | None = None,
    ) -> None:
        attempt = self.attempts[request_id][attempt_no - 1]
        attempt.update(
            status=attempt_status,
            completed_at=_utcnow(),
            status_code=status_code,
            error_code=error_code,
            input_tokens=usage.input_tokens if usage else None,
            output_tokens=usage.output_tokens if usage else None,
            latency_ms=latency_ms,
        )
        if instance_id is not None:
            attempt["worker_instance_id"] = instance_id

    async def finish_request(
        self,
        request_id: UUID,
        request_status: str,
        *,
        status_code: int,
        worker_id: str | None = None,
        usage: Usage | None = None,
        latency_ms: int | None = None,
        error_code: str | None = None,
    ) -> None:
        request = self.requests[request_id]
        request.update(
            status=request_status,
            status_code=status_code,
            worker_id=worker_id,
            input_tokens=usage.input_tokens if usage else None,
            output_tokens=usage.output_tokens if usage else None,
            total_tokens=usage.total_tokens if usage else None,
            latency_ms=latency_ms,
            error_code=error_code,
            completed_at=_utcnow(),
        )

    async def finish_attempt_and_request(
        self,
        request_id: UUID,
        attempt_no: int,
        attempt_status: str,
        request_status: str,
        *,
        status_code: int,
        worker_id: str,
        error_code: str | None = None,
        usage: Usage | None = None,
        instance_id: UUID | None = None,
        attempt_latency_ms: int | None = None,
        request_latency_ms: int | None = None,
    ) -> None:
        await self.finish_attempt(
            request_id,
            attempt_no,
            attempt_status,
            status_code=status_code,
            error_code=error_code,
            usage=usage,
            instance_id=instance_id,
            latency_ms=attempt_latency_ms,
        )
        await self.finish_request(
            request_id,
            request_status,
            status_code=status_code,
            worker_id=worker_id,
            usage=usage,
            latency_ms=request_latency_ms,
            error_code=error_code,
        )

    async def recover_stale(self) -> int:
        count = 0
        for request_id, request in self.requests.items():
            if request["status"] != "received":
                continue
            count += 1
            request.update(status="unknown_after_restart", completed_at=_utcnow())
            for attempt in self.attempts[request_id]:
                if attempt["status"] == "started":
                    attempt.update(status="unknown_after_restart", completed_at=_utcnow())
        return count

    async def record_worker_state_event(
        self,
        worker_id: str,
        prior_state: str,
        new_state: str,
        instance_id: UUID | None,
        reason: str,
    ) -> None:
        self.worker_state_events.append(
            {
                "worker_id": worker_id,
                "prior_state": prior_state,
                "new_state": new_state,
                "worker_instance_id": instance_id,
                "reason": reason,
            }
        )


class PostgresLedger:
    def __init__(self, database_url: str):
        self.pool = AsyncConnectionPool(database_url, min_size=1, max_size=5, open=False)

    async def open(self) -> None:
        await self.pool.open()
        await self.pool.wait()

    async def close(self) -> None:
        await self.pool.close()

    async def ready(self) -> bool:
        try:
            async with self.pool.connection() as connection:
                await connection.execute("SELECT 1")
            return True
        except Exception:  # noqa: BLE001 - readiness must fail closed for any pool error
            return False

    async def create_request(self, request_id: UUID, endpoint: str) -> None:
        async with self.pool.connection() as connection:
            await connection.execute(
                "INSERT INTO gateway_requests (request_id, endpoint) VALUES (%s, %s)",
                (request_id, endpoint),
            )

    async def start_attempt(self, request_id: UUID, attempt_no: int, worker_id: str, instance_id: UUID | None) -> None:
        async with self.pool.connection() as connection, connection.transaction():
            await connection.execute(
                """INSERT INTO gateway_attempts
                   (request_id, attempt_no, worker_id, worker_instance_id)
                   VALUES (%s, %s, %s, %s)""",
                (request_id, attempt_no, worker_id, instance_id),
            )
            await connection.execute(
                "UPDATE gateway_requests SET attempt_count = %s WHERE request_id = %s",
                (attempt_no, request_id),
            )

    async def finish_attempt(
        self,
        request_id: UUID,
        attempt_no: int,
        attempt_status: str,
        *,
        status_code: int | None = None,
        error_code: str | None = None,
        usage: Usage | None = None,
        instance_id: UUID | None = None,
        latency_ms: int | None = None,
    ) -> None:
        async with self.pool.connection() as connection:
            await connection.execute(
                """UPDATE gateway_attempts SET
                       completed_at = now(), status = %s, status_code = %s,
                       error_code = %s, input_tokens = %s, output_tokens = %s,
                       worker_instance_id = COALESCE(%s, worker_instance_id), latency_ms = %s
                     WHERE request_id = %s AND attempt_no = %s""",
                (
                    attempt_status,
                    status_code,
                    error_code,
                    usage.input_tokens if usage else None,
                    usage.output_tokens if usage else None,
                    instance_id,
                    latency_ms,
                    request_id,
                    attempt_no,
                ),
            )

    async def finish_request(
        self,
        request_id: UUID,
        request_status: str,
        *,
        status_code: int,
        worker_id: str | None = None,
        usage: Usage | None = None,
        latency_ms: int | None = None,
        error_code: str | None = None,
    ) -> None:
        async with self.pool.connection() as connection:
            await connection.execute(
                """UPDATE gateway_requests SET
                       completed_at = now(), status = %s, status_code = %s,
                       selected_worker_id = %s, input_tokens = %s, output_tokens = %s,
                       total_tokens = %s, latency_ms = %s, error_code = %s
                     WHERE request_id = %s""",
                (
                    request_status,
                    status_code,
                    worker_id,
                    usage.input_tokens if usage else None,
                    usage.output_tokens if usage else None,
                    usage.total_tokens if usage else None,
                    latency_ms,
                    error_code,
                    request_id,
                ),
            )

    async def finish_attempt_and_request(
        self,
        request_id: UUID,
        attempt_no: int,
        attempt_status: str,
        request_status: str,
        *,
        status_code: int,
        worker_id: str,
        error_code: str | None = None,
        usage: Usage | None = None,
        instance_id: UUID | None = None,
        attempt_latency_ms: int | None = None,
        request_latency_ms: int | None = None,
    ) -> None:
        async with self.pool.connection() as connection, connection.transaction():
            await connection.execute(
                """UPDATE gateway_attempts SET
                       completed_at = now(), status = %s, status_code = %s,
                       error_code = %s, input_tokens = %s, output_tokens = %s,
                       worker_instance_id = COALESCE(%s, worker_instance_id), latency_ms = %s
                     WHERE request_id = %s AND attempt_no = %s""",
                (
                    attempt_status,
                    status_code,
                    error_code,
                    usage.input_tokens if usage else None,
                    usage.output_tokens if usage else None,
                    instance_id,
                    attempt_latency_ms,
                    request_id,
                    attempt_no,
                ),
            )
            await connection.execute(
                """UPDATE gateway_requests SET
                       completed_at = now(), status = %s, status_code = %s,
                       selected_worker_id = %s, input_tokens = %s, output_tokens = %s,
                       total_tokens = %s, latency_ms = %s, error_code = %s
                     WHERE request_id = %s""",
                (
                    request_status,
                    status_code,
                    worker_id,
                    usage.input_tokens if usage else None,
                    usage.output_tokens if usage else None,
                    usage.total_tokens if usage else None,
                    request_latency_ms,
                    error_code,
                    request_id,
                ),
            )

    async def recover_stale(self) -> int:
        async with self.pool.connection() as connection, connection.transaction():
            cursor = await connection.execute(
                """UPDATE gateway_attempts SET status = 'unknown_after_restart', completed_at = now()
                   WHERE status = 'started'"""
            )
            await connection.execute(
                """UPDATE gateway_requests SET status = 'unknown_after_restart', completed_at = now()
                   WHERE status = 'received'"""
            )
            return cursor.rowcount

    async def record_worker_state_event(
        self,
        worker_id: str,
        prior_state: str,
        new_state: str,
        instance_id: UUID | None,
        reason: str,
    ) -> None:
        async with self.pool.connection() as connection:
            await connection.execute(
                """INSERT INTO worker_state_events
                   (worker_id, prior_state, new_state, worker_instance_id, reason)
                   VALUES (%s, %s, %s, %s, %s)""",
                (worker_id, prior_state, new_state, instance_id, reason),
            )
