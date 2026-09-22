from __future__ import annotations

import os
from pathlib import Path
from urllib.parse import urlencode
from uuid import uuid4

import pytest
from psycopg import AsyncConnection

from laya_gateway.db import apply_migrations
from laya_gateway.ledger import PostgresLedger
from laya_gateway.models import Usage


def _scoped_url(database_url: str, schema: str) -> str:
    separator = "&" if "?" in database_url else "?"
    return database_url + separator + urlencode({"options": f"-csearch_path={schema}"})


@pytest.mark.anyio
async def test_postgres_migrations_atomic_finalization_and_recovery():
    database_url = os.getenv("TEST_DATABASE_URL")
    if not database_url:
        pytest.skip("TEST_DATABASE_URL is not configured")

    schema = f"laya_gateway_test_{uuid4().hex}"
    admin = await AsyncConnection.connect(database_url, autocommit=True)
    await admin.execute(f'CREATE SCHEMA "{schema}"')
    ledger: PostgresLedger | None = None
    try:
        scoped_url = _scoped_url(database_url, schema)
        migrations = Path(__file__).parents[1] / "migrations"
        assert await apply_migrations(scoped_url, migrations) == ["001_initial.sql"]
        assert await apply_migrations(scoped_url, migrations) == []

        ledger = PostgresLedger(scoped_url)
        await ledger.open()
        request_id = uuid4()
        usage = Usage(input_tokens=17, output_tokens=0)
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
            attempt_latency_ms=11,
            request_latency_ms=13,
        )

        async with ledger.pool.connection() as connection:
            request_row = await (
                await connection.execute(
                    """SELECT status, selected_worker_id, input_tokens, output_tokens,
                              total_tokens, latency_ms
                         FROM gateway_requests WHERE request_id = %s""",
                    (request_id,),
                )
            ).fetchone()
            attempt_row = await (
                await connection.execute(
                    "SELECT status, input_tokens, output_tokens, latency_ms FROM gateway_attempts WHERE request_id = %s",
                    (request_id,),
                )
            ).fetchone()
            columns = {
                row[0]
                for row in await (
                    await connection.execute(
                        """SELECT column_name FROM information_schema.columns
                             WHERE table_schema = current_schema()
                               AND table_name IN ('gateway_requests', 'gateway_attempts')"""
                    )
                ).fetchall()
            }

        assert request_row == ("succeeded", "laya-secondary", 17, 0, 17, 13)
        assert attempt_row == ("succeeded", 17, 0, 11)
        assert {"request_body", "response_body", "prompt", "answer", "api_key"}.isdisjoint(columns)

        stale_id = uuid4()
        await ledger.create_request(stale_id, "/v1/evaluate")
        await ledger.start_attempt(stale_id, 1, "laya-primary", None)
        assert await ledger.recover_stale() == 1
        async with ledger.pool.connection() as connection:
            recovered_request = await (
                await connection.execute(
                    "SELECT status FROM gateway_requests WHERE request_id = %s", (stale_id,)
                )
            ).fetchone()
            recovered_attempt = await (
                await connection.execute(
                    "SELECT status FROM gateway_attempts WHERE request_id = %s", (stale_id,)
                )
            ).fetchone()
        assert recovered_request == ("unknown_after_restart",)
        assert recovered_attempt == ("unknown_after_restart",)
    finally:
        if ledger is not None:
            await ledger.close()
        await admin.execute(f'DROP SCHEMA IF EXISTS "{schema}" CASCADE')
        await admin.close()
