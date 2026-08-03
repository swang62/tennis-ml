"""End-to-end tests: raw ATP CSV -> bronze -> dbt gold -> inference on a temp DB.

The unit tests patch connections and use synthetic rows; the DB-backed suite
seeds via `infra/duckdb/seed.py`. These tests run the REAL ingest path
(`load_raw_atp_rows` -> `atp_rows_to_bronze` -> `insert_bronze_rows`) followed
by a real `dbt build` and `build_inference_features`, all against a throwaway
DuckDB file. The whole raw file is ingested — Davis Cup ties and round-robin
matches are preserved and encode as round ordinal 0 in gold, and ranks pass
through raw (0 -> NULL in silver), so the gold build succeeds. No dev-DB
mutation, nothing that needs a cluster.
"""

from datetime import date

import duckdb
import pytest

from src.constants import BRONZE_TABLE, GOLD_ROLLING_FEATURES, GOLD_TABLE, ROOT
from src.db import client as db_client
from src.features.columns import FEATURE_COLS
from src.features.inference import build_inference_features
from src.flows import ingest
from src.utils import run_dbt_build

RAW_CSV = ROOT / "data" / "raw" / "2026.csv"
AS_OF = date(2026, 9, 1)  # after every seeded match, like test_inference_features

# match_features carries these 8 metadata columns before the 54 feature columns.
META_COLS = [
    "match_id",
    "match_date",
    "player_id",
    "opponent_id",
    "tournament",
    "round",
    "surface",
    "match_won",
]

PROFILES_YML = """tennis_ml:
  target: local
  outputs:
    local:
      type: duckdb
      path: {db}
      schema: gold
      threads: 1
"""


def _count(conn, table: str) -> int:
    return conn.execute(f"SELECT COUNT(*) FROM {table}").fetchone()[0]


@pytest.fixture(scope="module")
def temp_pipeline(tmp_path_factory):
    """Temp DuckDB + real ingest + real dbt build; module-scoped so the
    few-second dbt build runs once per file. Rebinds get_conn in db.client
    (used by execute_df / inference) and in ingest (used by insert_bronze_rows);
    both are restored in teardown.
    """
    if not RAW_CSV.exists():
        pytest.skip(f"raw ATP CSV not found at {RAW_CSV}")

    db = tmp_path_factory.mktemp("e2e") / "tennis.duckdb"

    conn = duckdb.connect(str(db))
    conn.execute((ROOT / "infra" / "duckdb" / "init.sql").read_text())
    conn.close()

    profiles_dir = tmp_path_factory.mktemp("profiles")
    (profiles_dir / "profiles.yml").write_text(PROFILES_YML.format(db=db))

    def temp_conn():
        return duckdb.connect(str(db))

    original_client_conn = db_client.get_conn
    original_ingest_conn = ingest.get_conn
    db_client.get_conn = temp_conn
    ingest.get_conn = temp_conn
    try:
        rows = ingest.load_raw_atp_rows(RAW_CSV)
        df = ingest.atp_rows_to_bronze(rows)
        inserted = ingest.insert_bronze_rows(df)
        run_dbt_build(profiles_dir=profiles_dir)
        yield {"db": db, "inserted": inserted, "df": df}
    finally:
        db_client.get_conn = original_client_conn
        ingest.get_conn = original_ingest_conn


def test_ingest_and_dbt_round_trip(temp_pipeline):
    """Real ingest path + dbt build produce populated bronze/silver/gold."""
    conn = duckdb.connect(str(temp_pipeline["db"]))
    try:
        assert _count(conn, BRONZE_TABLE) == temp_pipeline["inserted"]
        assert _count(conn, GOLD_ROLLING_FEATURES) > 0
        assert _count(conn, GOLD_TABLE) > 0
    finally:
        conn.close()


def test_reinsert_is_idempotent_upsert(temp_pipeline):
    """Re-ingesting the same rows upserts in place (match_id PK), no doubling."""
    inserted_again = ingest.insert_bronze_rows(temp_pipeline["df"])
    assert inserted_again == temp_pipeline["inserted"]
    conn = duckdb.connect(str(temp_pipeline["db"]))
    try:
        assert _count(conn, BRONZE_TABLE) == temp_pipeline["inserted"]
    finally:
        conn.close()


def test_gold_match_features_schema_matches_python_contract(temp_pipeline):
    """The dbt-built training table's live schema == metadata cols + FEATURE_COLS
    — the parity check that SQL-text tests and inference-builder tests can't see."""
    conn = duckdb.connect(str(temp_pipeline["db"]))
    try:
        cols = [
            row[0]
            for row in conn.execute(
                "SELECT column_name FROM information_schema.columns "
                "WHERE table_name = 'match_features' AND table_schema = 'gold' "
                "ORDER BY ordinal_position"
            ).fetchall()
        ]
    finally:
        conn.close()
    assert cols == [*META_COLS, *FEATURE_COLS]


def _known_pair(conn):
    row = conn.execute(
        "SELECT player1_id, player2_id FROM bronze.match_events "
        "ORDER BY player1_id, player2_id LIMIT 1"
    ).fetchone()
    return row[0], row[1]


def test_e2e_inference_contract(temp_pipeline):
    """build_inference_features against the temp-DB gold: exact 56-col schema,
    one row, canonical ids, finite features."""
    conn = duckdb.connect(str(temp_pipeline["db"]))
    try:
        player_id, opponent_id = _known_pair(conn)
    finally:
        conn.close()
    out = build_inference_features(player_id, opponent_id, "hard", as_of_date=AS_OF)
    assert out.columns.tolist() == [*FEATURE_COLS, "player_id", "opponent_id"]
    assert len(out) == 1
    assert out["player_id"].iloc[0] == min(player_id, opponent_id)
    assert not out[FEATURE_COLS].isnull().to_numpy().any()


def test_e2e_cold_start_imputation(temp_pipeline):  # noqa: ARG001 — fixture applied for its side effects only
    """Unknown players fall back to global aggregates on the temp DB too."""
    out = build_inference_features("ZZZZ", "YYYY", "clay", as_of_date=AS_OF)
    assert out.columns.tolist() == [*FEATURE_COLS, "player_id", "opponent_id"]
    assert len(out) == 1
    assert out["player_id"].iloc[0] == "YYYY"  # 'YYYY' < 'ZZZZ'
