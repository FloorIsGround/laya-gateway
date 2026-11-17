# Laya Gateway

Internal FastAPI gateway for routing Laya inference requests across workers and recording token usage without retaining prompts or answers.

## Contract

- `POST /v1/systemone`
- `POST /v1/evaluate` (compatibility alias)
- `GET /health`
- `GET /ready`
- `GET /internal/backends`

The gateway forwards only to workers that pass the authenticated worker-status protocol, report the configured identity/device, and remain fresh. Scheduling is least-loaded with deterministic tie rank. The example configuration gives `laya-primary` priority when both workers are equally idle.

Before scheduling a valid request, the gateway counts each Laya question row
using its bundled, pinned tokenizer. The longest untruncated row must fit a
worker's reported `max_context`; the gateway retains the configured tie preference
for short requests, while longer requests can use only a capable worker.
When that worker is busy, the request waits for it rather than falling back to
a short worker that would truncate the state. Requests beyond all known
capacities receive `422 context_too_long`. A matching tokenizer fingerprint is
required for context-aware workers; older known Laya 0.3.4 workers remain
eligible at 512 tokens only. Unknown or mismatched tokenizers are not eligible.
Malformed requests are still validated by the worker, preserving its API errors.

The tokenizer file in `src/laya_gateway/laya-tokenizer.json` is from the
Apache-2.0 `convaiinnovations/laya` model revision
`55cf4c4ebb4ebe31b2550e8bdf3bd21b99753851` (SHA-256
`6c8aaa9a542084f2457eab775d4eeb51f92a70c0fd9de28d5edb0ddec3c08d30`).
The gateway uses only the small Rust `tokenizers` runtime, not Torch or the
model weights. Update the file and parity tests together if the worker tokenizer
or Laya input formatter changes.

A definite connection failure or explicit worker failure may be retried once on a different worker. Read timeouts and other ambiguous outcomes are recorded as unknown and are never retried.
Retries use the same context eligibility filter, so an oversized request can
never be sent to a fallback worker with a shorter window.

## Development

```bash
uv sync --group dev
uv run pytest -q
```

Prepare local configuration before starting the gateway:

```bash
cp config/workers.example.toml config/workers.toml
cp .env.example .env
chmod 600 .env
```

Set each worker's URL, `id`, and `expected_device` in `config/workers.toml`.
The identity must match that worker's `LAYA_WORKER_ID`; the device must match
its configured device (for example, `cpu` or `cuda`). Replace the API-key
placeholders in `.env` with the corresponding workers' `LAYA_API_KEY` values.
Both local configuration files are ignored by Git.

For local integration tests only, select the non-durable ledger:

```bash
GATEWAY_LEDGER_MODE=memory uv run --env-file .env uvicorn laya_gateway.main:app --host 127.0.0.1 --port 8020
```

The in-memory ledger is not suitable for accounting or production.

## PostgreSQL setup

Create a dedicated database and role, set `DATABASE_URL` in `.env`, and apply
migrations before starting the production service:

```bash
uv run --env-file .env python -m laya_gateway.db
```

The migration runner records each applied SQL file and checksum in
`schema_migrations`; it refuses to continue if an already-applied migration was
edited. Use `TEST_DATABASE_URL` to run the isolated PostgreSQL integration test
against a disposable database.

The schema stores logical request IDs, per-worker attempts, timestamps, status codes, worker identity, latency, and usage totals. It does not contain request bodies, response bodies, prompts, answers, or API keys.

## Configuration

Worker identities and URLs belong in the local `config/workers.toml`, copied
from `config/workers.example.toml`. API keys are indirect environment references
and must not be placed in TOML or committed.

The example URLs use placeholder hostnames; replace them with your own worker
addresses. Plain HTTP is intended for a trusted LAN protected by independent
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

The included user unit assumes a checkout at `~/projects/laya-gateway`; adjust
its paths if you clone elsewhere. It loads `.env`, binds `127.0.0.1:8020`, and
runs one gateway process so queue state and local in-flight counts remain
authoritative. Prepare the local configuration and database before starting it.

```bash
install -Dm644 deploy/laya-gateway.service ~/.config/systemd/user/laya-gateway.service
systemctl --user daemon-reload
systemctl --user enable --now laya-gateway.service
```

Use PostgreSQL mode before exposing the service to real traffic.

## License

Copyright 2026 FloorIsGround. Original project code is licensed under the
[Apache License, Version 2.0](LICENSE).

The bundled Laya tokenizer is also distributed under Apache-2.0. Its source,
pinned revision, checksum, and attribution are recorded in [NOTICE](NOTICE).
Model weights are obtained separately by workers. Installed dependencies
retain their respective licenses.
