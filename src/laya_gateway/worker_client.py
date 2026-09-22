from __future__ import annotations

from typing import Any
from uuid import UUID

import httpx
from pydantic import ValidationError

from .models import UpstreamResult, Usage, WorkerConfig, WorkerStatus


class WorkerError(RuntimeError):
    pass


class WorkerConnectFailure(WorkerError):
    pass


class AmbiguousWorkerFailure(WorkerError):
    pass


class RetryableWorkerFailure(WorkerError):
    pass


class WorkerAuthFailure(WorkerError):
    pass


class WorkerProtocolError(WorkerError):
    pass


def _device_matches(expected: str, actual: str | None) -> bool:
    if actual is None:
        return False
    expected_value = expected.strip().lower()
    actual_value = actual.strip().lower()
    return expected_value == actual_value or (
        expected_value == "cuda" and actual_value.startswith("cuda:")
    )


def _json_object(response: httpx.Response) -> dict[str, Any]:
    try:
        body = response.json()
    except ValueError as exc:
        raise WorkerProtocolError("worker returned non-JSON response") from exc
    if not isinstance(body, dict):
        raise WorkerProtocolError("worker response must be a JSON object")
    return body


class WorkerClient:
    def __init__(self, config: WorkerConfig, http: httpx.AsyncClient):
        self.config = config
        self.http = http

    @property
    def _headers(self) -> dict[str, str]:
        return {"Authorization": f"Bearer {self.config.api_key}"}

    @property
    def _timeout(self) -> httpx.Timeout:
        return httpx.Timeout(
            connect=self.config.connect_timeout_seconds,
            read=self.config.read_timeout_seconds,
            write=self.config.connect_timeout_seconds,
            pool=self.config.connect_timeout_seconds,
        )

    async def probe(self) -> WorkerStatus:
        try:
            response = await self.http.get(
                f"{self.config.base_url}/v1/worker-status",
                headers=self._headers,
                timeout=self._timeout,
            )
        except (httpx.ConnectError, httpx.ConnectTimeout) as exc:
            raise WorkerConnectFailure(str(exc)) from exc
        except httpx.HTTPError as exc:
            raise WorkerConnectFailure(str(exc)) from exc
        if response.status_code in {401, 403}:
            raise WorkerAuthFailure(f"worker returned {response.status_code}")
        if response.status_code != 200:
            raise RetryableWorkerFailure(f"worker status returned {response.status_code}")
        try:
            worker_status = WorkerStatus.model_validate(_json_object(response))
        except ValidationError as exc:
            raise WorkerProtocolError("invalid worker status payload") from exc
        if worker_status.worker_id != self.config.id:
            raise WorkerProtocolError("worker identity does not match configuration")
        if not _device_matches(self.config.expected_device, worker_status.actual_device):
            raise WorkerProtocolError("worker device does not match configuration")
        return worker_status

    async def infer(
        self,
        endpoint: str,
        payload: dict[str, Any],
        request_id: str | UUID,
    ) -> UpstreamResult:
        request_id_value = str(request_id)
        headers = {**self._headers, "X-Request-ID": request_id_value}
        try:
            response = await self.http.post(
                f"{self.config.base_url}{endpoint}",
                headers=headers,
                json=payload,
                timeout=self._timeout,
            )
        except (httpx.ConnectError, httpx.ConnectTimeout) as exc:
            raise WorkerConnectFailure(str(exc)) from exc
        except (httpx.ReadTimeout, httpx.ReadError, httpx.RemoteProtocolError) as exc:
            raise AmbiguousWorkerFailure(str(exc)) from exc
        except httpx.HTTPError as exc:
            raise AmbiguousWorkerFailure(str(exc)) from exc

        if response.status_code in {401, 403}:
            raise WorkerAuthFailure(f"worker returned {response.status_code}")
        if response.status_code >= 500:
            code = None
            try:
                raw_error = response.json()
                detail = raw_error.get("detail") if isinstance(raw_error, dict) else None
                code = detail.get("code") if isinstance(detail, dict) else None
            except ValueError:
                pass
            if response.status_code == 503 and code:
                raise RetryableWorkerFailure(str(code))
            raise RetryableWorkerFailure(f"worker returned {response.status_code}")

        body = _json_object(response)
        if 400 <= response.status_code < 500:
            return UpstreamResult(response.status_code, body, None)
        if not 200 <= response.status_code < 300:
            raise WorkerProtocolError(f"unexpected worker status {response.status_code}")

        if response.headers.get("X-Request-ID") != request_id_value:
            raise WorkerProtocolError("worker request ID does not match")
        worker_id = response.headers.get("X-Laya-Worker-ID")
        if worker_id != self.config.id:
            raise WorkerProtocolError("worker identity header does not match")
        if response.headers.get("X-Laya-Protocol-Version") != "1":
            raise WorkerProtocolError("worker protocol header is missing or unsupported")
        raw_instance = response.headers.get("X-Laya-Worker-Instance")
        if not raw_instance:
            raise WorkerProtocolError("worker instance header is missing")
        try:
            usage = Usage.model_validate(body.get("usage"))
            instance_id = UUID(raw_instance)
        except (ValidationError, TypeError, ValueError) as exc:
            raise WorkerProtocolError("invalid worker usage or instance identity") from exc
        return UpstreamResult(
            status_code=response.status_code,
            body=body,
            usage=usage,
            worker_id=worker_id,
            instance_id=instance_id,
        )
