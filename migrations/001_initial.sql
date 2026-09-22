CREATE TABLE gateway_requests (
    request_id UUID PRIMARY KEY,
    endpoint TEXT NOT NULL,
    received_at TIMESTAMPTZ NOT NULL DEFAULT now(),
    completed_at TIMESTAMPTZ,
    status TEXT NOT NULL DEFAULT 'received'
        CHECK (status IN ('received', 'succeeded', 'failed', 'unknown', 'unknown_after_restart')),
    status_code INTEGER,
    selected_worker_id TEXT,
    input_tokens BIGINT CHECK (input_tokens >= 0),
    output_tokens BIGINT CHECK (output_tokens >= 0),
    total_tokens BIGINT CHECK (total_tokens >= 0),
    latency_ms BIGINT CHECK (latency_ms >= 0),
    error_code TEXT,
    attempt_count INTEGER NOT NULL DEFAULT 0 CHECK (attempt_count >= 0)
);

CREATE TABLE gateway_attempts (
    attempt_id BIGSERIAL PRIMARY KEY,
    request_id UUID NOT NULL REFERENCES gateway_requests(request_id),
    attempt_no INTEGER NOT NULL CHECK (attempt_no > 0),
    worker_id TEXT NOT NULL,
    worker_instance_id UUID,
    started_at TIMESTAMPTZ NOT NULL DEFAULT now(),
    completed_at TIMESTAMPTZ,
    status TEXT NOT NULL DEFAULT 'started'
        CHECK (status IN (
            'started', 'succeeded', 'client_error', 'connect_failed',
            'upstream_failed', 'protocol_failed', 'unknown', 'unknown_after_restart'
        )),
    status_code INTEGER,
    error_code TEXT,
    input_tokens BIGINT CHECK (input_tokens >= 0),
    output_tokens BIGINT CHECK (output_tokens >= 0),
    latency_ms BIGINT CHECK (latency_ms >= 0),
    UNIQUE (request_id, attempt_no)
);

CREATE TABLE worker_state_events (
    id BIGSERIAL PRIMARY KEY,
    worker_id TEXT NOT NULL,
    observed_at TIMESTAMPTZ NOT NULL DEFAULT now(),
    prior_state TEXT,
    new_state TEXT NOT NULL,
    worker_instance_id UUID,
    reason TEXT
);

CREATE INDEX gateway_requests_received_at_idx ON gateway_requests (received_at);
CREATE INDEX gateway_attempts_worker_started_idx ON gateway_attempts (worker_id, started_at);
CREATE INDEX worker_state_events_worker_observed_idx ON worker_state_events (worker_id, observed_at);
