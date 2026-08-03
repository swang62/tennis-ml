import duckdb
import pytest

import src.db.client as db_client

CREATE_TABLE_SQL = "CREATE TABLE t (id INTEGER, name VARCHAR)"
INSERT_SQL = "INSERT INTO t VALUES (1, 'Alice'), (2, 'O''Brien')"


@pytest.fixture(scope="module")
def _in_memory_db():
    """Swap client._conn for an in-memory DuckDB; never touch the real DB file."""
    conn = duckdb.connect(":memory:")
    conn.execute(CREATE_TABLE_SQL)
    conn.execute(INSERT_SQL)
    db_client._conn = conn
    yield
    conn.close()
    db_client._conn = None


def test_to_dataframe_returns_expected_columns_and_rows(_in_memory_db):
    df = db_client.to_dataframe("SELECT id, name FROM t ORDER BY id")

    assert list(df.columns) == ["id", "name"]
    assert len(df) == 2
    assert df.iloc[0].to_dict() == {"id": 1, "name": "Alice"}
    assert df.iloc[1].to_dict() == {"id": 2, "name": "O'Brien"}


def test_execute_df_with_placeholder_params(_in_memory_db):
    df = db_client.execute_df("SELECT name FROM t WHERE id = ?", [1])

    assert list(df.columns) == ["name"]
    assert df.to_dict(orient="records") == [{"name": "Alice"}]


def test_execute_df_param_binding_handles_literal_quote(_in_memory_db):
    # If params were interpolated into the SQL, the quote in O'Brien would break it.
    df = db_client.execute_df("SELECT id FROM t WHERE name = ?", ["O'Brien"])

    assert df.to_dict(orient="records") == [{"id": 2}]


def test_execute_df_without_params(_in_memory_db):
    df = db_client.execute_df("SELECT id, name FROM t ORDER BY id")

    assert list(df.columns) == ["id", "name"]
    assert len(df) == 2


def test_first_row_dict_returns_string_keys(_in_memory_db):
    df = db_client.execute_df("SELECT id, name FROM t ORDER BY id")

    row = db_client.first_row_dict(df)

    assert row == {"id": 1, "name": "Alice"}
    assert all(isinstance(key, str) for key in row)
