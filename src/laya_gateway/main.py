from __future__ import annotations

import asyncio
import logging
import os
from contextlib import asynccontextmanager
from pathlib import Path

import httpx

from .api import create_app
from .config import load_config
from .context import ContextSizer
from .ledger import InMemoryLedger, PostgresLedger
from .registry import WorkerRegistry
from .scheduler import Scheduler
from .worker_client import WorkerClient

LOGGER = logging.getLogger("laya_gateway")


def _positive_int(name: str, default: int) -> int:
    value = int(os.getenv(name, str(default)))
    if value <= 0:
        raise RuntimeError(f"{name} must be positive")
    return value


def _positive_float(name: str, default: float) -> float:
    value = float(os.getenv(name, str(default)))
    if value <= 0:
        raise RuntimeError(f"{name} must be positive")
    return value


def create_application():
    config_path = Path(os.getenv("LAYA_GATEWAY_CONFIG", "config/workers.toml"))
    config = load_config(config_path)
    http = httpx.AsyncClient()
    clients = [(worker, WorkerClient(worker, http)) for worker in config.workers]
    ledger_mode = os.getenv("GATEWAY_LEDGER_MODE", "postgres").strip().lower()
    if ledger_mode == "memory":
        LOGGER.warning("using non-durable in-memory ledger; this mode is for integration testing only")
        ledger = InMemoryLedger()
    elif ledger_mode == "postgres":
        database_url = os.getenv("DATABASE_URL")
        if not database_url:
            raise RuntimeError("DATABASE_URL is required when GATEWAY_LEDGER_MODE=postgres")
        ledger = PostgresLedger(database_url)
    else:
        raise RuntimeError("GATEWAY_LEDGER_MODE must be postgres or memory")

    registry = WorkerRegistry(
        clients,
        probe_interval=_positive_float("GATEWAY_PROBE_INTERVAL_SECONDS", 2.0),
        stale_after=_positive_float("GATEWAY_STALE_AFTER_SECONDS", 6.0),
        on_state_change=ledger.record_worker_state_event,
    )
    scheduler = Scheduler(
        registry,
        queue_capacity=_positive_int("GATEWAY_QUEUE_CAPACITY", 64),
        queue_timeout=_positive_float("GATEWAY_QUEUE_TIMEOUT_SECONDS", 30.0),
    )
    context_sizer = ContextSizer()

    @asynccontextmanager
    async def lifespan(_app):
        await ledger.open()
        recovered = await ledger.recover_stale()
        if recovered:
            LOGGER.warning("marked %s stale gateway attempts unknown after restart", recovered)
        stop = asyncio.Event()
        probe_task = asyncio.create_task(registry.run(stop), name="worker-registry-probes")
        try:
            yield
        finally:
            stop.set()
            try:
                await asyncio.wait_for(probe_task, timeout=registry.probe_interval + 1)
            except TimeoutError:
                probe_task.cancel()
                await asyncio.gather(probe_task, return_exceptions=True)
            await http.aclose()
            await ledger.close()

    return create_app(registry, scheduler, ledger, lifespan=lifespan, context_sizer=context_sizer)


app = create_application()
