"""End-to-end PostgreSQL tests for seed, dbt, and ID inference."""

from datetime import date
from typing import cast

import pandas as pd
import pytest
from starlette.testclient import TestClient

from src.constants import (
    BRONZE_PROFILES_TABLE,
    BRONZE_TABLE,
    GOLD_TABLE,
    ROOT,
    SILVER_ROLLING_FEATURES,
)
from src.db import client, seed
from src.db.client import execute_df
from src.features.columns import FEATURE_COLS, SIMILARITY_COLS
from src.flows import ingest
from src.serving.service import DATA_APP

RAW_CSV = ROOT / "data" / "raw" / "2026.csv"
AS_OF = date(2026, 9, 1)  # after every seeded match, like test_inference_features

# A seeded player with official ranking rows after ingest_rankings() (Sebastian
# Baez, ARG); his match-row rank (45) differs from his official weekly ranks.
_SEEDED_PLAYER = "B0BI"

service_client = TestClient(DATA_APP)

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


def _seed_bronze_df() -> pd.DataFrame:
    """The exact bronze rows seed.main() writes (deterministic miniset)."""
    matches = sorted(
        ingest.load_raw_atp_rows(RAW_CSV),
        key=lambda m: (int(m["tourney_date"]), m["tourney_id"], m["match_num"]),
    )
    selected = seed.select_matches(matches)
    selected_ids = {
        f"{int(m['tourney_date'])}-{m['tourney_id']}-{int(m['match_num']):03d}" for m in selected
    }
    return ingest.atp_rows_to_bronze(matches, selected_ids=selected_ids)


def test_deterministic_miniset_size(postgres_ready):  # noqa: ARG001 — skip-gate fixture, unused in body
    """The seeded fixture is exactly the existing 28-match/35-player miniset."""
    df = _seed_bronze_df()
    assert len(df) == 28
    assert len(set(df["player1_id"]) | set(df["player2_id"])) == 35


def test_seeded_bronze_populated(postgres_ready):  # noqa: ARG001 — skip-gate fixture, unused in body
    assert cast(int, execute_df(f"SELECT COUNT(*) FROM {BRONZE_TABLE}").iloc[0, 0]) == 28


def test_reinsert_is_idempotent(postgres_ready):  # noqa: ARG001 — skip-gate fixture, unused in body
    """Re-ingesting the seed's own rows skips duplicates (match_id PK), no doubling."""
    df = _seed_bronze_df()
    first = ingest.insert_bronze_rows(df)
    again = ingest.insert_bronze_rows(df)
    assert first == again == len(df)
    count = cast(int, execute_df(f"SELECT COUNT(*) FROM {BRONZE_TABLE}").iloc[0, 0])
    assert count == len(df)


def test_reinsert_skips_duplicates_keeps_original_row(postgres_ready):  # noqa: ARG001 — skip-gate fixture, unused in body
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
            f"SELECT player1_ranking FROM {BRONZE_TABLE} WHERE match_id = %s", [match_id]
        ).iloc[0, 0],
    )
    assert stored == original_ranking


def test_profile_upsert_preserves_enrichment(postgres_ready, tmp_path):  # noqa: ARG001 — skip-gate fixture, unused in body
    """Reload ATP metadata without overwriting Wikipedia enrichment."""
    csv = tmp_path / "atp_profiles.csv"
    pd.DataFrame(
        [
            {
                "id": "P1",
                "player": "Player One",
                "atpname": "P. One",
                "birthdate": "19930316",
                "weight": "85",
                "height": "185",
                "turnedpro": "2018",
                "birthplace": "Paris",
                "coaches": "",
                "hand": "R",
                "backhand": "2H",
                "ioc": "FRA",
            },
        ]
    ).to_csv(csv, index=False)

    ingest.load_atp_profiles(csv, player_ids={"P1"})
    with client.transaction() as cur:
        cur.execute(
            f"UPDATE {BRONZE_PROFILES_TABLE} SET summary = %s, enriched_at = CURRENT_TIMESTAMP "
            "WHERE player_id = %s",
            ["Existing enrichment", "P1"],
        )

    ingest.load_atp_profiles(csv, player_ids={"P1"})

    row = execute_df(
        f"SELECT summary, weight FROM {BRONZE_PROFILES_TABLE} WHERE player_id = %s", ["P1"]
    ).iloc[0]
    assert row["summary"] == "Existing enrichment"  # enrichment survives the reload
    assert int(row["weight"]) == 85  # ATP metadata refreshed


def test_gold_layers_populated(gold_ready):  # noqa: ARG001 — skip-gate fixture, unused in body
    assert cast(int, execute_df(f"SELECT COUNT(*) FROM {SILVER_ROLLING_FEATURES}").iloc[0, 0]) > 0
    assert cast(int, execute_df(f"SELECT COUNT(*) FROM {GOLD_TABLE}").iloc[0, 0]) > 0


def test_gold_match_features_schema_matches_python_contract(gold_ready):  # noqa: ARG001 — skip-gate fixture, unused in body
    """Gold schema is metadata, FEATURE_COLS, then similarity-only columns."""
    cols = execute_df(
        "SELECT column_name FROM information_schema.columns "
        "WHERE table_name = 'match_features' AND table_schema = 'gold' "
        "ORDER BY ordinal_position"
    )["column_name"].tolist()
    assert cols == [*META_COLS, *FEATURE_COLS, *SIMILARITY_COLS]


def test_gold_has_no_current_match_enrichment_columns(gold_ready):  # noqa: ARG001 — skip-gate fixture, unused in body
    """Current-match rates stay out of gold and remain derivable from bronze."""
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


def test_current_match_rates_derivable_from_bronze(gold_ready):  # noqa: ARG001 — skip-gate fixture, unused in body
    """Task 6: current/break analysis rates are removed from gold but remain
    derivable on demand from bronze raw counts with NULLIF zero-denominator."""
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
        "SELECT CAST(player1_first_serve_points_won AS DOUBLE PRECISION)"
        "  / NULLIF(player1_first_serves_made, 0) AS first_serve_win_pct,"
        " CAST(player1_aces AS DOUBLE PRECISION) / NULLIF(player1_service_games, 0)"
        "  AS aces_per_svc_game"
        f" FROM {BRONZE_TABLE} WHERE match_id = %s",
        [m["match_id"]],
    )
    assert not res.isnull().all().iloc[0]  # NULLIF zero-denominator survives


def _known_pair() -> tuple[str, str]:
    row = execute_df(
        "SELECT player1_id, player2_id FROM bronze.match_events "
        "ORDER BY player1_id, player2_id LIMIT 1"
    )
    return str(row["player1_id"].iloc[0]), str(row["player2_id"].iloc[0])


def test_e2e_inference_contract(gold_ready):  # noqa: ARG001 — skip-gate fixture, unused in body
    """build_inference_features against the shared gold: exact schema,
    one row, canonical ids, finite features."""
    from src.features.inference import build_inference_features

    player_id, opponent_id = _known_pair()
    out = build_inference_features(player_id, opponent_id, "hard", as_of_date=AS_OF)
    assert out.columns.tolist() == [*FEATURE_COLS, "player_id", "opponent_id"]
    assert len(out) == 1
    assert out["player_id"].iloc[0] == min(player_id, opponent_id)
    assert not out[FEATURE_COLS].isnull().to_numpy().any()


def test_e2e_cold_start_imputation(gold_ready):  # noqa: ARG001 — skip-gate fixture, unused in body
    """Unknown players fall back to global aggregates on the seeded DB too."""
    from src.features.inference import build_inference_features

    out = build_inference_features("ZZZZ", "YYYY", "clay", as_of_date=AS_OF)
    assert out.columns.tolist() == [*FEATURE_COLS, "player_id", "opponent_id"]
    assert len(out) == 1
    assert out["player_id"].iloc[0] == "YYYY"  # 'YYYY' < 'ZZZZ'


def test_official_rankings_table_populated_in_range_and_idempotent(postgres_ready):  # noqa: ARG001
    """bronze.rankings holds only official top-200 ranks via canonical ids, and
    re-running the importer leaves the same logical rows (idempotent)."""
    from src.flows.ingest import ingest_rankings

    first = ingest_rankings()
    assert first["files"] == 7  # all supplied ranking-period files are handled

    n, lo, hi = execute_df("SELECT COUNT(*), MIN(rank), MAX(rank) FROM bronze.rankings").iloc[0]
    assert int(n) == int(first["upserted"])
    assert int(lo) >= 1 and int(hi) <= 200  # every stored rank is official top-200

    # Raw ranking source ids must never reach the table.
    raw = cast(
        int,
        execute_df("SELECT COUNT(*) FROM bronze.rankings WHERE player_id ~ '^[0-9]+$'").iloc[0, 0],
    )
    assert raw == 0

    # Re-running the importer upserts the same logical rows, no duplication.
    ingest_rankings()
    again = cast(int, execute_df("SELECT COUNT(*) FROM bronze.rankings").iloc[0, 0])
    assert again == int(n)

    # The API/chart contract observes the ingested official rows: /rank_history
    # mirrors bronze.rankings week by week, and /players current_rank is the
    # latest official week — never a match-row rank.
    db_history = execute_df(
        "SELECT ranking_date, rank FROM bronze.rankings WHERE player_id = %s ORDER BY ranking_date",
        [_SEEDED_PLAYER],
    )
    assert len(db_history) > 1  # weekly points across multiple ranking Mondays
    api_history = service_client.get(f"/rank_history?player_id={_SEEDED_PLAYER}").json()["data"][
        "rank_history"
    ]
    assert api_history == [
        {
            "rank_date": pd.Timestamp(r["ranking_date"]).date().isoformat(),
            "rank": int(r["rank"]),
        }
        for r in db_history.to_dict("records")
    ]
    latest_official = int(db_history["rank"].iloc[-1])
    players = service_client.get("/players").json()["data"]["players"]
    assert (
        next(p for p in players if p["player_id"] == _SEEDED_PLAYER)["current_rank"]
        == latest_official
    )
