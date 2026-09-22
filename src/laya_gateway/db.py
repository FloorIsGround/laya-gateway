from __future__ import annotations

import asyncio
import hashlib
import os
from pathlib import Path

from psycopg import AsyncConnection


async def apply_migrations(database_url: str, migrations_dir: str | Path = "migrations") -> list[str]:
    migration_root = Path(migrations_dir)
    migration_files = sorted(migration_root.glob("*.sql"))
    if not migration_files:
        raise RuntimeError(f"no SQL migrations found in {migration_root}")

    applied_now: list[str] = []
    connection = await AsyncConnection.connect(database_url, autocommit=True)
    try:
        await connection.execute(
            """CREATE TABLE IF NOT EXISTS schema_migrations (
                   version TEXT PRIMARY KEY,
                   sha256 TEXT NOT NULL,
                   applied_at TIMESTAMPTZ NOT NULL DEFAULT now()
               )"""
        )
        rows = await (
            await connection.execute("SELECT version, sha256 FROM schema_migrations")
        ).fetchall()
        applied: dict[str, str] = dict(rows)

        for migration_path in migration_files:
            version = migration_path.name
            sql = migration_path.read_text()
            checksum = hashlib.sha256(sql.encode("utf-8")).hexdigest()
            prior_checksum = applied.get(version)
            if prior_checksum is not None:
                if prior_checksum != checksum:
                    raise RuntimeError(f"applied migration checksum changed: {version}")
                continue
            async with connection.transaction():
                await connection.execute(sql)
                await connection.execute(
                    "INSERT INTO schema_migrations (version, sha256) VALUES (%s, %s)",
                    (version, checksum),
                )
            applied_now.append(version)
    finally:
        await connection.close()
    return applied_now


async def _main() -> None:
    database_url = os.getenv("DATABASE_URL")
    if not database_url:
        raise RuntimeError("DATABASE_URL is required")
    applied = await apply_migrations(database_url)
    if applied:
        print("applied migrations: " + ", ".join(applied))
    else:
        print("database schema is current")


if __name__ == "__main__":
    asyncio.run(_main())
