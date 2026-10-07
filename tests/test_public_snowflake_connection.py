"""Offline tests of credential boundaries and preserved query/provenance APIs."""
from __future__ import annotations

import importlib
import sys
import types

import pytest

connection_module = importlib.import_module("func.setup.FIGConnect")
FIGConnect = connection_module.FIGConnect


@pytest.fixture(autouse=True)
def clean_snowflake_environment(monkeypatch):
    for name in (*connection_module._ENV_PARAMETERS.values(), "SNOWFLAKE_LOGIN_TIMEOUT"):
        monkeypatch.delenv(name, raising=False)


def test_construction_and_setup_do_not_validate_or_connect(monkeypatch, tmp_path):
    def forbidden(*args, **kwargs):
        raise AssertionError("Connection/credential access occurred during construction")

    monkeypatch.setattr(connection_module, "snowflake_connection_parameters", forbidden)
    monkeypatch.setattr(FIGConnect, "create_conn", forbidden)
    from func.setup.setup import Setup

    setup = Setup(PROJECT_DIR=str(tmp_path), RUN_ID="offline-construction")
    assert set(setup.conns) == {"SNOWFLAKE", "SNOWPARK"}
    for connection in setup.conns.values():
        assert connection.engine is connection.cnxn is connection.session is None
    setup.shutdown()


def test_missing_credentials_fail_before_driver_import():
    connection = FIGConnect()
    with pytest.raises(ValueError, match="SNOWFLAKE_ACCOUNT.*SNOWFLAKE_USER"):
        connection._connection_parameters()


def test_env_and_explicit_parameters_preserve_secret_values(monkeypatch):
    monkeypatch.setenv("SNOWFLAKE_ACCOUNT", "example-account")
    monkeypatch.setenv("SNOWFLAKE_USER", "test-user")
    monkeypatch.setenv("SNOWFLAKE_PASSWORD", " leading-and-trailing spaces ")
    monkeypatch.setenv("SNOWFLAKE_DATABASE", "demo_db")
    connection = FIGConnect(role="ANALYST", timeout=45)
    values = connection._connection_parameters()
    assert values["password"] == " leading-and-trailing spaces "
    assert values["database"] == "DEMO_DB"
    assert values["role"] == "ANALYST"
    assert values["login_timeout"] == 45


def test_named_profile_does_not_require_password(monkeypatch):
    monkeypatch.setenv("SNOWFLAKE_CONNECTION_NAME", "public-demo")
    values = FIGConnect()._connection_parameters()
    assert values["connection_name"] == "public-demo"
    assert "password" not in values


@pytest.mark.parametrize("authenticator,extra", [
    ("externalbrowser", {}),
    ("SNOWFLAKE_JWT", {"private_key_file": "~/example_key.p8"}),
    ("oauth", {"token": "test-token"}),
])
def test_standard_authentication_modes(authenticator, extra):
    values = FIGConnect(account="example-account", username="test-user",
                        authenticator=authenticator,
                        connection_parameters=extra)._connection_parameters()
    assert values["authenticator"] == authenticator
    assert "password" not in values


def test_invalid_schema_rejected_without_echoing_secret(monkeypatch):
    monkeypatch.setenv("SNOWFLAKE_CONNECTION_NAME", "public-demo")
    monkeypatch.setenv("SNOWFLAKE_SCHEMA", "PUBLIC; DROP TABLE X")
    with pytest.raises(ValueError, match="unquoted SQL identifier"):
        FIGConnect()._connection_parameters()


def test_engine_and_native_cursor_share_one_connection(monkeypatch):
    calls = []
    native = object()
    fake_connector = types.ModuleType("snowflake.connector")
    fake_connector.connect = lambda **kwargs: calls.append(kwargs) or native
    fake_snowflake = types.ModuleType("snowflake")
    fake_snowflake.connector = fake_connector
    monkeypatch.setitem(sys.modules, "snowflake", fake_snowflake)
    monkeypatch.setitem(sys.modules, "snowflake.connector", fake_connector)

    class Engine:
        def __init__(self, creator):
            self.creator = creator

        def connect(self):
            return types.SimpleNamespace(connection=types.SimpleNamespace(driver_connection=self.creator()))

    fake_sqlalchemy = types.ModuleType("sqlalchemy")

    def create_engine(url, creator, **kwargs):
        assert url == "snowflake://"  # Credentials never enter URLs.
        return Engine(creator)

    fake_sqlalchemy.create_engine = create_engine
    monkeypatch.setitem(sys.modules, "sqlalchemy", fake_sqlalchemy)
    connection = FIGConnect(connection_name="public-demo")
    assert calls == []
    connection.create_conn(verbose=0)
    connection.create_conn(verbose=0)
    assert len(calls) == 1
    assert connection.snow_conn is native


def test_chunked_query_retains_real_query_id_and_closes_cursor(monkeypatch):
    class Cursor:
        sfqid = "mock-query-id"
        description = [("x",)]
        closed = False

        def __init__(self):
            self.batches = iter([[(1,), (2,)], [(3,)], []])

        def execute(self, query):
            assert query == "SELECT X FROM EXAMPLE"

        def fetchmany(self, size):
            assert size == 2
            return next(self.batches)

        def close(self):
            self.closed = True

    cursor = Cursor()
    connection = FIGConnect()
    connection.snow_conn = types.SimpleNamespace(cursor=lambda: cursor)
    monkeypatch.setattr(connection, "create_conn", lambda **kwargs: None)
    monkeypatch.setattr(connection, "save_info", lambda: None)
    chunks = connection.query_data("SELECT X FROM EXAMPLE", chunksize=2)
    assert connection.query_id == "mock-query-id"
    assert [frame["X"].tolist() for frame in chunks] == [[1, 2], [3]]
    assert cursor.closed


def test_ddl_query_preserves_status_and_query_receipt(monkeypatch):
    from snowflake.connector.errors import NotSupportedError

    class Cursor:
        sfqid = "mock-ddl-id"
        description = [("status",)]
        closed = False

        def execute(self, query):
            assert query == "CREATE TEMP TABLE EXAMPLE (X INT)"

        def fetch_arrow_all(self, **kwargs):
            raise NotSupportedError("Result uses ordinary rows")

        def fetchall(self):
            return [("Table EXAMPLE successfully created.",)]

        def close(self):
            self.closed = True

    cursor = Cursor()
    connection = FIGConnect()
    connection.snow_conn = types.SimpleNamespace(cursor=lambda: cursor)
    monkeypatch.setattr(connection, "create_conn", lambda **kwargs: None)
    monkeypatch.setattr(connection_module, "get_git_commit_hash", lambda: ("test", "test"))
    result = connection.query_data("CREATE TEMP TABLE EXAMPLE (X INT)", query_name="create_example")
    assert result.loc[0, "STATUS"] == "Table EXAMPLE successfully created."
    assert connection.query_id == "mock-ddl-id"
    assert next(iter(connection.queries.values()))["id"] == "mock-ddl-id"
    assert cursor.closed
