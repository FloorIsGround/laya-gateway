# Laya Gateway

Internal FastAPI gateway for routing Laya inference requests across workers and recording token usage without retaining prompts or answers.

## Contract

- `POST /v1/systemone`
- `POST /v1/evaluate` (compatibility alias)
- `GET /health`
- `GET /ready`
- `GET /internal/backends`

The gateway forwards only to workers that pass the authenticated worker-status protocol, report the configured identity/device, and remain fresh. Scheduling is least-loaded with deterministic tie rank. The supplied configuration gives `laya-primary` priority when both workers are equally idle.

A definite connection failure or explicit worker failure may be retried once on a different worker. Read timeouts and other ambiguous outcomes are recorded as unknown and are never retried.

## Development

```bash
uv sync --group dev
uv run pytest -q
```

For local integration tests only, copy `.env.example`, set worker keys, and select the non-durable ledger:

```bash
GATEWAY_LEDGER_MODE=memory uv run uvicorn laya_gateway.main:app --host 127.0.0.1 --port 8020
```

The in-memory ledger is not suitable for accounting or production.

## PostgreSQL setup

Create a dedicated database and role, set `DATABASE_URL`, and apply migrations before starting the production service:

```bash
psql "$DATABASE_URL" -v ON_ERROR_STOP=1 -f migrations/001_initial.sql
```

The schema stores logical request IDs, per-worker attempts, timestamps, status codes, worker identity, latency, and usage totals. It does not contain request bodies, response bodies, prompts, answers, or API keys.

## Configuration

Worker identities and URLs are in `config/workers.toml`. API keys are indirect environment references and must not be placed in TOML or committed.

The initial worker URLs are private-LAN HTTP endpoints protected by independent
worker API keys. Do not expose those ports outside the trusted LAN. TLS or a
mutually authenticated service network should be added before crossing a trust
boundary.

- `LAYA_GATEWAY_CONFIG`: worker TOML path
- `DATABASE_URL`: PostgreSQL connection URI
- `GATEWAY_LEDGER_MODE`: `postgres` by default; `memory` only for integration testing
- `GATEWAY_QUEUE_CAPACITY`: central waiting queue bound, default `64`
- `GATEWAY_QUEUE_TIMEOUT_SECONDS`: maximum gateway wait, default `30`
- `GATEWAY_PROBE_INTERVAL_SECONDS`: worker status poll interval, default `2`
- `GATEWAY_STALE_AFTER_SECONDS`: worker status freshness limit, default `6`

## Service

The included user unit binds `127.0.0.1:8020` and runs one gateway process so queue state and local in-flight counts remain authoritative.

```bash
install -Dm644 deploy/laya-gateway.service ~/.config/systemd/user/laya-gateway.service
systemctl --user daemon-reload
systemctl --user enable --now laya-gateway.service
```

Use PostgreSQL mode before exposing the service to real traffic.
