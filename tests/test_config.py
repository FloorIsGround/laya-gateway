from __future__ import annotations

from pathlib import Path

import pytest

from laya_gateway.config import ConfigError, load_config

VALID_TOML = """
[[workers]]
id = "laya-primary"
base_url = "https://192.0.2.10:8010"
api_key_env = "KEY_PRIMARY"
tie_rank = 0
expected_device = "cuda"

[[workers]]
id = "laya-secondary"
base_url = "http://127.0.0.1:8000"
api_key_env = "KEY_SECONDARY"
tie_rank = 1
expected_device = "cuda"
"""


def test_load_config_resolves_secrets_without_exposing_them(tmp_path: Path):
    path = tmp_path / "workers.toml"
    path.write_text(VALID_TOML)
    config = load_config(path, {"KEY_PRIMARY": "secret-a", "KEY_SECONDARY": "secret-b"})
    assert [worker.id for worker in config.workers] == ["laya-primary", "laya-secondary"]
    assert config.workers[0].api_key == "secret-a"
    assert "secret-a" not in repr(config)
    assert config.workers[0].connect_timeout_seconds > 0


@pytest.mark.parametrize(
    "toml",
    [
        VALID_TOML.replace('id = "laya-secondary"', 'id = "laya-primary"'),
        VALID_TOML.replace('base_url = "http://127.0.0.1:8000"', 'base_url = "https://192.0.2.10:8010"'),
        VALID_TOML.replace("tie_rank = 1", "tie_rank = -1"),
    ],
)
def test_load_config_rejects_invalid_worker_sets(tmp_path: Path, toml: str):
    path = tmp_path / "workers.toml"
    path.write_text(toml)
    with pytest.raises(ConfigError):
        load_config(path, {"KEY_PRIMARY": "a", "KEY_SECONDARY": "b"})


def test_load_config_requires_worker_secret(tmp_path: Path):
    path = tmp_path / "workers.toml"
    path.write_text(VALID_TOML)
    with pytest.raises(ConfigError, match="KEY_SECONDARY"):
        load_config(path, {"KEY_PRIMARY": "a"})


def test_load_config_accepts_plain_http_for_private_lan_worker(tmp_path: Path):
    path = tmp_path / "workers.toml"
    path.write_text(VALID_TOML.replace("https://192.0.2.10:8010", "http://192.0.2.10:8010"))
    config = load_config(path, {"KEY_PRIMARY": "a", "KEY_SECONDARY": "b"})
    assert config.workers[0].base_url == "http://192.0.2.10:8010"


def test_load_config_allows_plain_http_for_loopback_worker(tmp_path: Path):
    path = tmp_path / "workers.toml"
    path.write_text(VALID_TOML)
    config = load_config(path, {"KEY_PRIMARY": "a", "KEY_SECONDARY": "b"})
    assert config.workers[1].base_url == "http://127.0.0.1:8000"


def test_load_config_rejects_direct_secret_fields(tmp_path: Path):
    path = tmp_path / "workers.toml"
    path.write_text(
        VALID_TOML.replace('tie_rank = 0', 'tie_rank = 0\napi_key = "do-not-store-here"')
    )
    with pytest.raises(ConfigError, match="unsupported worker fields"):
        load_config(path, {"KEY_PRIMARY": "a", "KEY_SECONDARY": "b"})
