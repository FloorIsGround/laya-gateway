from pathlib import Path


def test_initial_migration_has_required_tables_and_no_payload_columns():
    sql = Path("migrations/001_initial.sql").read_text().lower()
    assert "create table gateway_requests" in sql
    assert "create table gateway_attempts" in sql
    assert "create table worker_state_events" in sql
    assert "total_tokens" in sql
    assert "latency_ms" in sql
    assert "error_code" in sql
    assert "unique (request_id, attempt_no)" in sql
    assert "request_body" not in sql
    assert "response_body" not in sql
    assert "api_key" not in sql
