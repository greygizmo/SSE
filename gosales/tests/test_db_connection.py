import types
from pathlib import Path

import pytest

from gosales.utils import db as dbmod
from gosales.utils.paths import ROOT_DIR


class DummyEngine:
    def __init__(self, url: str):
        self.url = url


@pytest.fixture(autouse=True)
def reset_env(monkeypatch):
    for var in ["AZSQL_SERVER", "AZSQL_DB", "AZSQL_USER", "AZSQL_PWD"]:
        monkeypatch.delenv(var, raising=False)


def test_get_db_connection_falls_back_to_sqlite_when_azure_missing(monkeypatch, caplog):
    custom_path = ROOT_DIR / "data" / "fallback.sqlite"
    cfg = types.SimpleNamespace(
        database=types.SimpleNamespace(engine="azure", sqlite_path=custom_path, strict_db=False)
    )
    monkeypatch.setattr(dbmod, "load_config", lambda: cfg)

    captured = {}

    def fake_create_engine(url: str, *_, **__):
        captured["url"] = url
        return DummyEngine(url)

    monkeypatch.setattr(dbmod, "create_engine", fake_create_engine)

    caplog.set_level("INFO")
    engine = dbmod.get_db_connection()

    expected_url = f"sqlite:///{custom_path}"
    assert isinstance(engine, DummyEngine)
    assert captured["url"] == expected_url
    assert any("falling back to SQLite" in msg for msg in caplog.messages)


def test_get_db_connection_uses_custom_sqlite_path(monkeypatch):
    custom_path = Path("/tmp/gosales/custom.db")
    cfg = types.SimpleNamespace(
        database=types.SimpleNamespace(engine="sqlite", sqlite_path=custom_path, strict_db=False)
    )
    monkeypatch.setattr(dbmod, "load_config", lambda: cfg)

    captured = {}

    def fake_create_engine(url: str, *_, **__):
        captured["url"] = url
        return DummyEngine(url)

    monkeypatch.setattr(dbmod, "create_engine", fake_create_engine)

    engine = dbmod.get_db_connection()

    assert isinstance(engine, DummyEngine)
    assert captured["url"] == f"sqlite:///{custom_path}"


def test_get_db_connection_duckdb(monkeypatch):
    duckdb_path = Path("/tmp/gosales/test.duckdb")
    cfg = types.SimpleNamespace(
        database=types.SimpleNamespace(
            engine="duckdb", sqlite_path=ROOT_DIR / "gosales.db", duckdb_path=duckdb_path, strict_db=False
        )
    )
    monkeypatch.setattr(dbmod, "load_config", lambda: cfg)

    captured = {}

    def fake_create_engine(url: str, *_, **__):
        captured["url"] = url
        return DummyEngine(url)

    monkeypatch.setattr(dbmod, "create_engine", fake_create_engine)

    engine = dbmod.get_db_connection()

    assert isinstance(engine, DummyEngine)
    assert captured["url"] == f"duckdb:///{duckdb_path}"
