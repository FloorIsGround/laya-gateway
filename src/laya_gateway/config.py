from __future__ import annotations

import os
import tomllib
from collections.abc import Mapping
from dataclasses import dataclass
from pathlib import Path
from urllib.parse import urlparse

from .models import WorkerConfig


class ConfigError(ValueError):
    pass


@dataclass(frozen=True)
class GatewayConfig:
    workers: tuple[WorkerConfig, ...]


def load_config(path: str | Path, environ: Mapping[str, str] | None = None) -> GatewayConfig:
    environment = os.environ if environ is None else environ
    try:
        raw = tomllib.loads(Path(path).read_text())
    except (OSError, tomllib.TOMLDecodeError) as exc:
        raise ConfigError(f"unable to load worker config: {exc}") from exc
    entries = raw.get("workers")
    if not isinstance(entries, list) or not entries:
        raise ConfigError("at least one [[workers]] entry is required")

    workers: list[WorkerConfig] = []
    ids: set[str] = set()
    urls: set[str] = set()
    for entry in entries:
        if not isinstance(entry, dict):
            raise ConfigError("worker entries must be tables")
        try:
            worker_id = entry["id"]
            base_url = entry["base_url"].rstrip("/")
            api_key_env = entry["api_key_env"]
            tie_rank = entry.get("tie_rank", 0)
            expected_device = entry.get("expected_device", "cuda")
            connect_timeout = float(entry.get("connect_timeout_seconds", 1.0))
            read_timeout = float(entry.get("read_timeout_seconds", 65.0))
        except (KeyError, AttributeError, TypeError, ValueError) as exc:
            raise ConfigError(f"invalid worker entry: {exc}") from exc
        if not isinstance(worker_id, str) or not worker_id.strip():
            raise ConfigError("worker id must be a non-empty string")
        parsed = urlparse(base_url)
        if (
            parsed.scheme not in {"http", "https"}
            or not parsed.netloc
            or parsed.username
            or parsed.password
            or parsed.query
            or parsed.fragment
        ):
            raise ConfigError(f"invalid base URL for {worker_id}")
        if not isinstance(api_key_env, str) or not api_key_env:
            raise ConfigError(f"api_key_env is required for {worker_id}")
        if not isinstance(tie_rank, int) or isinstance(tie_rank, bool) or tie_rank < 0:
            raise ConfigError(f"tie_rank must be a nonnegative integer for {worker_id}")
        if connect_timeout <= 0 or read_timeout <= 0:
            raise ConfigError(f"timeouts must be positive for {worker_id}")
        if worker_id in ids:
            raise ConfigError(f"duplicate worker id: {worker_id}")
        if base_url in urls:
            raise ConfigError(f"duplicate worker URL: {base_url}")
        api_key = environment.get(api_key_env)
        if not api_key:
            raise ConfigError(f"missing worker secret environment variable: {api_key_env}")
        ids.add(worker_id)
        urls.add(base_url)
        workers.append(
            WorkerConfig(
                id=worker_id,
                base_url=base_url,
                api_key_env=api_key_env,
                api_key=api_key,
                tie_rank=tie_rank,
                expected_device=str(expected_device),
                connect_timeout_seconds=connect_timeout,
                read_timeout_seconds=read_timeout,
            )
        )
    return GatewayConfig(tuple(workers))
