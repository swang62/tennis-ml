"""Prefect flow: Bronze → Gold ETL.

All per-player rolling features computed in DuckDB SQL via window functions
so the gold table is the single source of truth for training and inference.
"""

from prefect import flow, task

from src.db.client import get_conn, to_dataframe
from src.features.validate import run_feature_checks
from src.flows.ingest import enrich_missing as _enrich_missing

BRONZE_TABLE = "bronze.match_events"
GOLD_TABLE = "gold.match_features"
GOLD_SQL = f"""
    INSERT INTO {GOLD_TABLE}

    WITH base AS (
        SELECT
            *,
            wins_last_10 / NULLIF(matches_last_10, 0) AS win_rate_last_10,
            aces / NULLIF(first_serves_made, 0) AS ace_rate,
            double_faults / NULLIF(total_serve_points, 0) AS double_fault_rate,
            first_serves_made / NULLIF(total_serve_points, 0) AS first_serve_pct,
            break_points_won / NULLIF(break_points_total, 0) AS break_points_converted_pct,
            CASE WHEN surface = 'clay'  THEN 1 ELSE 0 END AS is_clay,
            CASE WHEN surface = 'grass' THEN 1 ELSE 0 END AS is_grass,
            CASE WHEN surface = 'hard'  THEN 1 ELSE 0 END AS is_hard
        FROM {BRONZE_TABLE}
        WHERE match_id IS NOT NULL
          AND match_date IS NOT NULL
          AND player_ranking > 0
          AND opponent_ranking > 0
    ),
    with_windows AS (
        SELECT
            *,

            -- Rolling aggregates (exclude current match)
            avg(match_won)  OVER w5  AS win_rate_5,
            avg(match_won)  OVER w10 AS win_rate_10,
            avg(match_won)  OVER w20 AS win_rate_20,

            sum(aces) OVER w5  / NULLIF(sum(first_serves_made) OVER w5,  0) AS ace_rate_5,
            sum(aces) OVER w10 / NULLIF(sum(first_serves_made) OVER w10, 0) AS ace_rate_10,

            sum(first_serves_made) OVER w5
                / NULLIF(sum(total_serve_points) OVER w5,  0) AS first_serve_pct_5,
            sum(first_serves_made) OVER w10
                / NULLIF(sum(total_serve_points) OVER w10, 0) AS first_serve_pct_10,

            sum(break_points_won) OVER w5
                / NULLIF(sum(break_points_total) OVER w5,  0) AS break_pct_5,
            sum(break_points_won) OVER w10
                / NULLIF(sum(break_points_total) OVER w10, 0) AS break_pct_10,

            avg(opponent_ranking) OVER w10 AS avg_opp_rank_10,
            avg(opponent_ranking) OVER w20 AS avg_opp_rank_20,

            avg(player_ranking) OVER w10 - player_ranking AS rank_trend_10,
            avg(player_ranking) OVER w20 - player_ranking AS rank_trend_20,

            -- Days since player's last match
            DATEDIFF('day', LAG(match_date) OVER w_all, match_date) AS days_since_last_match,

            -- Matches in last 30 days before this match
            SUM(CASE WHEN match_date >= match_date - INTERVAL '30 days' THEN 1 ELSE 0 END)
                OVER (PARTITION BY player_id ORDER BY match_date
                      ROWS BETWEEN UNBOUNDED PRECEDING AND 1 PRECEDING) AS matches_30d,

            -- Surface-specific win rate (rolling 10 on same surface)
            avg(match_won) OVER w_surface10 AS surface_win_rate_10

        FROM base

        WINDOW
            w_all  AS (PARTITION BY player_id ORDER BY match_date
                       ROWS BETWEEN UNBOUNDED PRECEDING AND 0 PRECEDING),
            w5     AS (PARTITION BY player_id ORDER BY match_date
                       ROWS BETWEEN 5  PRECEDING AND 1 PRECEDING),
            w10    AS (PARTITION BY player_id ORDER BY match_date
                       ROWS BETWEEN 10 PRECEDING AND 1 PRECEDING),
            w20    AS (PARTITION BY player_id ORDER BY match_date
                       ROWS BETWEEN 20 PRECEDING AND 1 PRECEDING),
            w_surface10 AS (PARTITION BY player_id, surface ORDER BY match_date
                            ROWS BETWEEN 10 PRECEDING AND 1 PRECEDING)
    ),
    with_win_streak AS (
        SELECT
            *,
            row_number() OVER (PARTITION BY player_id ORDER BY match_date) AS rn
        FROM with_windows
    ),
    with_streak AS (
        SELECT
            *,
            rn - (
                MAX(CASE WHEN match_won = 0 THEN rn ELSE 0 END) OVER (
                    PARTITION BY player_id ORDER BY match_date
                    ROWS BETWEEN UNBOUNDED PRECEDING AND 1 PRECEDING
                )
            ) - 1 AS win_streak
        FROM with_win_streak
    )
    SELECT
        match_id, match_date, player_id, opponent_id,
        tournament, round, surface,
        player_ranking, opponent_ranking,
        wins_last_10, matches_last_10,
        aces, double_faults, first_serves_made, total_serve_points,
        break_points_won, break_points_total, match_won,

        win_rate_last_10, ace_rate, double_fault_rate,
        first_serve_pct, break_points_converted_pct,

        win_rate_5, win_rate_10, win_rate_20,
        ace_rate_5, ace_rate_10,
        first_serve_pct_5, first_serve_pct_10,
        break_pct_5, break_pct_10,
        avg_opp_rank_10, avg_opp_rank_20,
        rank_trend_10, rank_trend_20,
        win_streak,
        COALESCE(days_since_last_match, 365) AS days_since_last_match,
        COALESCE(matches_30d, 0) AS matches_30d,
        COALESCE(surface_win_rate_10, 0) AS surface_win_rate_10,

        is_clay, is_grass, is_hard
    FROM with_streak
    ORDER BY match_date, match_id
"""


@task(retries=2, retry_delay_seconds=30)
def bronze_to_gold() -> int:
    conn = get_conn()
    conn.sql(f"DELETE FROM {GOLD_TABLE}")
    conn.sql(GOLD_SQL)
    result = conn.sql(f"SELECT COUNT(*) AS cnt FROM {GOLD_TABLE}")
    row_count = result.fetchone()[0]
    print(f"Gold: {row_count} rows")
    return row_count


@task
def validate_gold():
    df = to_dataframe(f"SELECT * FROM {GOLD_TABLE} LIMIT 10000")
    result = run_feature_checks(df)
    if not result["passed"]:
        raise RuntimeError("Gold validation failed")


@task(retries=1, retry_delay_seconds=10)
def enrich_bios():
    inserted = _enrich_missing()
    print(f"Bios enriched: {inserted} new")
    return inserted


@flow(log_prints=True)
def etl_flow():
    rows = bronze_to_gold()
    if rows > 0:
        validate_gold()
        enrich_bios()
        print(f"ETL complete: {rows} gold rows")
    else:
        print("No rows in bronze, skipping validation")


if __name__ == "__main__":
    etl_flow()
