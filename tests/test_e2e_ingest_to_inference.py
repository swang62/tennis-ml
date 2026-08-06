"""End-to-end tests against the shared seeded miniset test DB.

The conftest pretest setup (autouse session fixture `seeded_test_db`) builds a
temporary DuckDB once per session: `infra/duckdb/init.sql`, the real ingest
path via `infra/duckdb/seed.py` (the deterministic miniset: the
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
from src.features.columns import FEATURE_COLS
from src.features.inference import build_inference_features
from src.flows import ingest

RAW_CSV = ROOT / "data" / "raw" / "2026.csv"
AS_OF = date(2026, 9, 1)  # after every seeded match, like test_inference_features

# match_features carries these 8 metadata columns before the 36 feature columns.
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


def test_reinsert_is_idempotent():
    """Re-ingesting the seed's own rows skips duplicates (match_id PK), no doubling."""
    df = _seed_bronze_df()
    first = ingest.insert_bronze_rows(df)
    again = ingest.insert_bronze_rows(df)
    assert first == again == len(df)
    count = cast(int, execute_df(f"SELECT COUNT(*) FROM {BRONZE_TABLE}").iloc[0, 0])
    assert count == len(df)


def test_reinsert_skips_duplicates_keeps_original_row():
    """DO NOTHING: re-ingesting an existing match_id must not overwrite it."""
    df = _seed_bronze_df()
    ingest.insert_bronze_rows(df)
    match_id = str(df["match_id"].iloc[0])
    original_ranking = int(df["player1_ranking"].iloc[0])

    changed = df.copy()
    changed.loc[0, "player1_ranking"] = 99999  # would refresh the row under DO UPDATE
    ingest.insert_bronze_rows(changed)

    stored = cast(
        int,
        execute_df(
            f"SELECT player1_ranking FROM {BRONZE_TABLE} WHERE match_id = '{match_id}'"
        ).iloc[0, 0],
    )
    assert stored == original_ranking


def test_gold_match_features_schema_matches_python_contract():
    """The dbt-built training table's live schema == metadata cols + FEATURE_COLS
    — the parity check that SQL-text tests and inference-builder tests can't see.
    Current-match serve/break analysis rates are no longer part of the gold
    contract (Task 6)."""
    cols = execute_df(
        "SELECT column_name FROM information_schema.columns "
        "WHERE table_name = 'match_features' AND table_schema = 'gold' "
        "ORDER BY ordinal_position"
    )["column_name"].tolist()
    assert cols == [*META_COLS, *FEATURE_COLS]


def test_gold_has_no_current_match_enrichment_columns():
    """Task 6: the per-side current-match serve/break enrichment columns are
    removed from gold.match_features entirely — they are derived on demand from
    bronze raw counts where the dashboard/analysis needs them."""
    cols = set(
        execute_df(
            "SELECT column_name FROM information_schema.columns "
            "WHERE table_name = 'match_features' AND table_schema = 'gold'"
        )["column_name"]
    )
    for c in (
        "first_serve_win_pct",
        "second_serve_win_pct",
        "serve_win_pct",
        "aces_per_svc_game",
        "df_per_svc_game",
        "break_points_saved_pct",
    ):
        assert f"player_{c}" not in cols, f"{c} still in gold"
        assert f"opponent_{c}" not in cols, f"{c} still in gold"


def test_current_match_rates_derivable_from_bronze():
    """Task 6: current-match serve/break analysis rates are removed from gold
    but remain derivable on demand from bronze raw counts with the existing
    NULLIF zero-denominator behavior."""
    row = execute_df(
        "SELECT match_id, player1_id, player2_id, "
        "player1_first_serve_points_won, player1_first_serves_made, "
        "player1_aces, player1_service_games "
        "FROM bronze.match_events "
        "WHERE player1_total_serve_points > 0 "
        "ORDER BY player1_id, player2_id LIMIT 1"
    )
    assert not row.empty
    m = row.iloc[0]
    res = execute_df(
        "SELECT CAST(player1_first_serve_points_won AS DOUBLE)"
        "  / NULLIF(player1_first_serves_made, 0) AS first_serve_win_pct,"
        " CAST(player1_aces AS DOUBLE) / NULLIF(player1_service_games, 0)"
        "  AS aces_per_svc_game"
        f" FROM {BRONZE_TABLE} WHERE match_id = ?",
        [m["match_id"]],
    )
    # Gold.match_features must NOT carry these current-match rates.
    gold_cols = set(
        execute_df(
            "SELECT column_name FROM information_schema.columns "
            "WHERE table_name = 'match_features' AND table_schema = 'gold'"
        )["column_name"]
    )
    for stem in ("first_serve_win_pct", "aces_per_svc_game"):
        assert f"player_{stem}" not in gold_cols
        assert f"opponent_{stem}" not in gold_cols
    assert not res.isnull().all().iloc[0]  # NULLIF zero-denominator survives


def _known_pair() -> tuple[str, str]:
    row = execute_df(
        "SELECT player1_id, player2_id FROM bronze.match_events "
        "ORDER BY player1_id, player2_id LIMIT 1"
    )
    return str(row["player1_id"].iloc[0]), str(row["player2_id"].iloc[0])


def test_e2e_inference_contract():
    """build_inference_features against the shared gold: exact schema,
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
