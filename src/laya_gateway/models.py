from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any
from uuid import UUID

from pydantic import BaseModel, ConfigDict, Field, StrictInt, model_validator


@dataclass(frozen=True)
class WorkerConfig:
    id: str
    base_url: str
    api_key_env: str
    api_key: str = field(repr=False)
    tie_rank: int = 0
    expected_device: str = "cuda"
    connect_timeout_seconds: float = 1.0
    read_timeout_seconds: float = 65.0


class Usage(BaseModel):
    model_config = ConfigDict(extra="forbid")

    input_tokens: StrictInt = Field(ge=0)
    output_tokens: StrictInt = Field(ge=0)

    @property
    def total_tokens(self) -> int:
        return self.input_tokens + self.output_tokens


class WorkerStatus(BaseModel):
    model_config = ConfigDict(extra="forbid")

    protocol_version: StrictInt
    status: str
    worker_id: str
    instance_id: UUID
    service_version: str
    laya_version: str
    model: str
    desired_device: str
    actual_device: str | None
    dtype: str | None
    capacity: StrictInt = Field(gt=0)
    queue_capacity: StrictInt = Field(gt=0)
    inflight: StrictInt = Field(ge=0)
    executing: StrictInt = Field(ge=0)
    queued: StrictInt = Field(ge=0)
    completed_total: StrictInt = Field(ge=0)
    failed_total: StrictInt = Field(ge=0)
    queue_rejected_total: StrictInt = Field(ge=0)
    uptime_seconds: StrictInt = Field(ge=0)

    @model_validator(mode="after")
    def validate_protocol_and_load(self) -> WorkerStatus:
        if self.protocol_version != 1:
            raise ValueError("unsupported worker protocol version")
        if self.executing > self.inflight:
            raise ValueError("executing cannot exceed inflight")
        if self.queued != self.inflight - self.executing:
            raise ValueError("queued must equal inflight minus executing")
        return self


@dataclass(frozen=True)
class UpstreamResult:
    status_code: int
    body: dict[str, Any]
    usage: Usage | None
    worker_id: str | None = None
    instance_id: UUID | None = None
