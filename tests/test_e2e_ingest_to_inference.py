"""End-to-end tests against the shared seeded miniset test DB.

The conftest pretest setup (autouse session fixture `seeded_test_db`) builds a
temporary DuckDB once per session: `infra/duckdb/init.sql`, the real ingest
path via `infra/duckdb/seed.py --offline` (the deterministic miniset: the
RECENT most recent matches of the TOP_PLAYERS best-ranked players), and the
dbt gold build, then points TENNIS_DB_PATH at it. These tests verify that
result: populated bronze/gold layers, live gold schema parity with the Python
feature contract, idempotent re-ingest of the seed rows, and inference against
the seeded gold. No full-file ingest, no per-test DB, no dev-DB mutation; the
temp DB is cleaned up after the session.
"""

from datetime import date
from importlib.util import module_from_spec, spec_from_file_location
from typing import cast

import pandas as pd

from src.constants import BRONZE_TABLE, GOLD_ROLLING_FEATURES, GOLD_TABLE, ROOT
from src.db.client import execute_df
from src.features.columns import FEATURE_COLS, MATCH_STATS_COLS
from src.features.inference import build_inference_features
from src.flows import ingest

RAW_CSV = ROOT / "data" / "raw" / "2026.csv"
AS_OF = date(2026, 9, 1)  # after every seeded match, like test_inference_features

# match_features carries these 8 metadata columns before the per-side
# current-match stats and the 80 feature columns.
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

# seed.py lives under infra/duckdb (no package __init__), so load it by file
# path, same pattern as tests/test_seed.py.
SEED_PATH = ROOT / "infra" / "duckdb" / "seed.py"
_spec = spec_from_file_location("seed", SEED_PATH)
assert _spec is not None and _spec.loader is not None
seed = module_from_spec(_spec)
_spec.loader.exec_module(seed)


def _seed_bronze_df() -> pd.DataFrame:
    """The exact bronze rows seed.py's main() writes (deterministic miniset)."""
    matches = sorted(
        ingest.load_raw_atp_rows(RAW_CSV),
        key=lambda m: (int(m["tourney_date"]), m["tourney_id"], m["match_num"]),
    )
    selected = seed.select_matches(matches)
    selected_ids = {
        f"{int(m['tourney_date'])}-{m['tourney_id']}-{int(m['match_num']):03d}" for m in selected
    }
    return ingest.atp_rows_to_bronze(matches, selected_ids=selected_ids)


def test_seeded_db_round_trip():
    """Seed -> dbt build produce populated bronze/gold layers."""
    assert cast(int, execute_df(f"SELECT COUNT(*) FROM {BRONZE_TABLE}").iloc[0, 0]) > 0
    assert cast(int, execute_df(f"SELECT COUNT(*) FROM {GOLD_ROLLING_FEATURES}").iloc[0, 0]) > 0
    assert cast(int, execute_df(f"SELECT COUNT(*) FROM {GOLD_TABLE}").iloc[0, 0]) > 0


def test_reinsert_is_idempotent_upsert():
    """Re-ingesting the seed's own rows upserts in place (match_id PK), no doubling."""
    df = _seed_bronze_df()
    first = ingest.insert_bronze_rows(df)
    again = ingest.insert_bronze_rows(df)
    assert first == again == len(df)
    count = cast(int, execute_df(f"SELECT COUNT(*) FROM {BRONZE_TABLE}").iloc[0, 0])
    assert count == len(df)


def test_gold_match_features_schema_matches_python_contract():
    """The dbt-built training table's live schema == metadata cols +
    per-side current-match stats + FEATURE_COLS — the parity check that
    SQL-text tests and inference-builder tests can't see."""
    cols = execute_df(
        "SELECT column_name FROM information_schema.columns "
        "WHERE table_name = 'match_features' AND table_schema = 'gold' "
        "ORDER BY ordinal_position"
    )["column_name"].tolist()
    assert cols == [
        *META_COLS,
        *[f"player_{c}" for c in MATCH_STATS_COLS],
        *[f"opponent_{c}" for c in MATCH_STATS_COLS],
        *FEATURE_COLS,
    ]


def _known_pair() -> tuple[str, str]:
    row = execute_df(
        "SELECT player1_id, player2_id FROM bronze.match_events "
        "ORDER BY player1_id, player2_id LIMIT 1"
    )
    return str(row["player1_id"].iloc[0]), str(row["player2_id"].iloc[0])


def test_e2e_inference_contract():
    """build_inference_features against the shared gold: exact 82-col schema,
    one row, canonical ids, finite features."""
    player_id, opponent_id = _known_pair()
    out = build_inference_features(player_id, opponent_id, "hard", as_of_date=AS_OF)
    assert out.columns.tolist() == [*FEATURE_COLS, "player_id", "opponent_id"]
    assert len(out) == 1
    assert out["player_id"].iloc[0] == min(player_id, opponent_id)
    assert not out[FEATURE_COLS].isnull().to_numpy().any()


def test_e2e_cold_start_imputation():
    """Unknown players fall back to global aggregates on the seeded DB too."""
    out = build_inference_features("ZZZZ", "YYYY", "clay", as_of_date=AS_OF)
    assert out.columns.tolist() == [*FEATURE_COLS, "player_id", "opponent_id"]
    assert len(out) == 1
    assert out["player_id"].iloc[0] == "YYYY"  # 'YYYY' < 'ZZZZ'
