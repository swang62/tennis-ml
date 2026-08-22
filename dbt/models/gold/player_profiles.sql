-- gold.player_profiles: one row per player of derived aggregates (match
-- counts, service/return/surface aggregates, rolling form, rank points, rank).
-- Metadata identity (display_name, handedness, ...) is NOT duplicated here —
-- bronze.player_profiles owns it and consumers join.
--
-- Plain table rebuilt in full every run: career aggregates change with every
-- new match, so incremental would never save work. Aggregates use weighted
-- sums with NULLIF, never averages of per-match percentages.
--
-- Every bronze profile is preserved, including zero-match players.

WITH player_agg AS (
    SELECT
        pm.player_id,
        -- Each bronze match expands to two player_matches rows with distinct
        -- player_ids, so DISTINCT keeps the count per physical match.
        COUNT(DISTINCT pm.match_id)                               AS match_count,
        MAX(pm.match_date)                                         AS latest_match_date,
        -- Latest positive rank points (ignore newer zero/null obs).
        MAX(pm.player_rank_points) FILTER (WHERE pm.player_rank_points > 0)      AS latest_positive_points,
        -- Deterministic latest positive via (match_date DESC, match_id DESC).
        (ARRAY_AGG(pm.player_rank_points ORDER BY pm.match_date DESC, pm.match_id DESC)
            FILTER (WHERE pm.player_rank_points > 0))[1]                  AS latest_positive_points_det,
        (ARRAY_AGG(pm.player_rank_points ORDER BY pm.match_date ASC, pm.match_id ASC)
            FILTER (WHERE pm.player_rank_points > 0))[1]                  AS earliest_positive_points,
        (ARRAY_AGG(pm.match_date ORDER BY pm.match_date ASC, pm.match_id ASC)
            FILTER (WHERE pm.player_rank_points > 0))[1]                  AS earliest_positive_date,
        (ARRAY_AGG(pm.match_date ORDER BY pm.match_date DESC, pm.match_id DESC)
            FILTER (WHERE pm.player_rank_points > 0))[1]                  AS latest_positive_date,

        -- Last known match-time rank (official ranking fallback).
        (ARRAY_AGG(pm.player_ranking ORDER BY pm.match_date DESC, pm.match_id DESC)
            FILTER (WHERE pm.player_ranking IS NOT NULL))[1]              AS last_match_rank,

        -- Service metrics (weighted sums with NULLIF).
        CAST(SUM(pm.first_serves_made) AS DOUBLE PRECISION)
            / NULLIF(SUM(pm.total_serve_points), 0)
            AS first_serve_in_pct,
        CAST(SUM(pm.aces) AS DOUBLE PRECISION)
            / NULLIF(SUM(pm.first_serves_made), 0)
            AS aces_per_first_serve,
        CAST(SUM(pm.first_serve_points_won) AS DOUBLE PRECISION)
            / NULLIF(SUM(pm.first_serves_made), 0)
            AS first_serve_points_won_pct,
        CAST(SUM(pm.second_serve_points_won) AS DOUBLE PRECISION)
            / NULLIF(SUM(pm.total_serve_points - pm.first_serves_made), 0)
            AS second_serve_points_won_pct,
        CAST(SUM(pm.first_serve_points_won + pm.second_serve_points_won) AS DOUBLE PRECISION)
            / NULLIF(SUM(pm.total_serve_points), 0)
            AS overall_serve_points_won_pct,
        CAST(SUM(pm.double_faults) AS DOUBLE PRECISION)
            / NULLIF(SUM(pm.total_serve_points), 0)
            AS double_faults_per_serve_point,
        CAST(SUM(pm.aces) AS DOUBLE PRECISION)
            / NULLIF(SUM(pm.service_games), 0)
            AS aces_per_service_game,
        CAST(SUM(pm.break_points_saved) AS DOUBLE PRECISION)
            / NULLIF(SUM(pm.break_points_faced), 0)
            AS break_points_saved_pct,

        -- Return metric (from this player's own row).
        CAST(SUM(pm.return_points_won) AS DOUBLE PRECISION)
            / NULLIF(SUM(pm.return_points_available), 0)
            AS return_points_won_pct,
         (
             (SUM(pm.return_points_won) + 1.0)
                 / (SUM(pm.return_points_available) + 2.0)
             / (1.0 - (SUM(pm.first_serve_points_won + pm.second_serve_points_won) + 1.0)
                        / (SUM(pm.total_serve_points) + 2.0))
         )::NUMERIC AS dominance,

        -- Surface counts and win rates.
        COUNT(*) FILTER (WHERE pm.surface = 'hard')                                   AS hard_matches,
        COUNT(*) FILTER (WHERE pm.surface = 'clay')                                   AS clay_matches,
        COUNT(*) FILTER (WHERE pm.surface = 'grass')                                  AS grass_matches,
        CAST(SUM(CASE WHEN pm.surface = 'hard'  THEN pm.match_won ELSE 0 END) AS DOUBLE PRECISION)
            / NULLIF(COUNT(*) FILTER (WHERE pm.surface = 'hard'), 0)                  AS hard_win_rate,
        CAST(SUM(CASE WHEN pm.surface = 'clay'  THEN pm.match_won ELSE 0 END) AS DOUBLE PRECISION)
            / NULLIF(COUNT(*) FILTER (WHERE pm.surface = 'clay'), 0)                  AS clay_win_rate,
        CAST(SUM(CASE WHEN pm.surface = 'grass' THEN pm.match_won ELSE 0 END) AS DOUBLE PRECISION)
            / NULLIF(COUNT(*) FILTER (WHERE pm.surface = 'grass'), 0)                 AS grass_win_rate,
        -- Career win rate (reputation signal for similarity).
        CAST(SUM(pm.match_won) AS DOUBLE PRECISION)
            / NULLIF(COUNT(*), 0)                                                     AS career_win_rate

    FROM {{ ref('player_matches') }} pm
    GROUP BY pm.player_id
),

-- Return metrics from opponent perspective (self-join on match_id).
return_agg AS (
    SELECT
        pm.player_id,
        -- First-serve return: (opp serves - opp serve points won) / opp serves.
        CAST(SUM(opp.first_serves_made - opp.first_serve_points_won) AS DOUBLE PRECISION)
            / NULLIF(SUM(opp.first_serves_made), 0)
            AS first_serve_return_points_won_pct,
        -- Second-serve return analog on the non-first-serve denominator.
        CAST(SUM((opp.total_serve_points - opp.first_serves_made) - opp.second_serve_points_won)
            AS DOUBLE PRECISION)
            / NULLIF(SUM(opp.total_serve_points - opp.first_serves_made), 0)
            AS second_serve_return_points_won_pct,
        -- Return games won: every converted break point ends that game, so
        -- (opp BP faced - opp BP saved) / opp service games.
        CAST(SUM(opp.break_points_faced - opp.break_points_saved) AS DOUBLE PRECISION)
            / NULLIF(SUM(opp.service_games), 0)
            AS return_games_won_pct,
        -- Break point conversion, and per-return-game opportunities.
        CAST(SUM(opp.break_points_faced - opp.break_points_saved) AS DOUBLE PRECISION)
            / NULLIF(SUM(opp.break_points_faced), 0)
            AS break_point_conversion_pct,
        CAST(SUM(opp.break_points_faced) AS DOUBLE PRECISION)
            / NULLIF(SUM(opp.service_games), 0)
            AS break_point_opportunities_per_return_game
    FROM {{ ref('player_matches') }} pm
    JOIN {{ ref('player_matches') }} opp
        ON opp.match_id = pm.match_id
       AND opp.player_id = pm.opponent_id
    GROUP BY pm.player_id
),

-- Latest rolling snapshot per player.
latest_snapshot AS (
    SELECT DISTINCT ON (player_id)
        player_id,
        snapshot_date   AS recent_snapshot_date,
        win_rate_10
    FROM {{ ref('rolling_features') }}
    ORDER BY player_id, snapshot_date DESC
),

-- Latest official ATP weekly ranking per player; served by a backward scan of
-- bronze.idx_rankings_player_date (player_id, ranking_date).
latest_rank AS (
    SELECT DISTINCT ON (r.player_id)
        r.player_id,
        r.rank
    FROM {{ source('bronze', 'rankings') }} r
    ORDER BY r.player_id, r.ranking_date DESC
)

SELECT
    -- player_id comes from bronze (every bronze profile is preserved);
    -- metadata columns themselves stay in bronze, never duplicated here.
    bp.player_id,

    -- Match counts
    COALESCE(pa.match_count, 0)               AS match_count,
    pa.latest_match_date,

    -- Rank
    pa.latest_positive_points_det             AS latest_rank_points,
    pa.earliest_positive_points               AS earliest_rank_points,
    pa.earliest_positive_date                 AS earliest_rank_points_date,
    pa.latest_positive_date                   AS latest_rank_points_date,
    pa.latest_positive_points_det - pa.earliest_positive_points
        AS rank_points_delta,
    -- Official weekly rank, falling back to last match-time rank when absent.
    COALESCE(lr.rank, pa.last_match_rank)       AS current_rank,

    -- Service
    pa.first_serve_in_pct                  AS first_serve_in_pct,
    pa.aces_per_first_serve                AS aces_per_first_serve,
    pa.first_serve_points_won_pct
        AS first_serve_points_won_pct,
    pa.second_serve_points_won_pct
        AS second_serve_points_won_pct,
    pa.overall_serve_points_won_pct
        AS overall_serve_points_won_pct,
    pa.double_faults_per_serve_point
        AS double_faults_per_serve_point,
    pa.aces_per_service_game
        AS aces_per_service_game,
    pa.break_points_saved_pct
        AS break_points_saved_pct,

    -- Return
    pa.return_points_won_pct
        AS return_points_won_pct,
    ra.first_serve_return_points_won_pct
        AS first_serve_return_points_won_pct,
    ra.second_serve_return_points_won_pct
        AS second_serve_return_points_won_pct,
    ra.return_games_won_pct
        AS return_games_won_pct,
    ra.break_point_conversion_pct
        AS break_point_conversion_pct,
    ra.break_point_opportunities_per_return_game
        AS break_point_opportunities_per_return_game,

    -- Surface
    COALESCE(pa.hard_matches, 0)              AS hard_matches,
    COALESCE(pa.clay_matches, 0)              AS clay_matches,
    COALESCE(pa.grass_matches, 0)             AS grass_matches,
    pa.hard_win_rate                         AS hard_win_rate,
    pa.clay_win_rate                         AS clay_win_rate,
    pa.grass_win_rate                        AS grass_win_rate,
    pa.dominance                             AS dominance,
    pa.career_win_rate                       AS career_win_rate,

    -- Recent form
    ls.recent_snapshot_date,
    ls.win_rate_10                           AS win_rate_10
FROM {{ source('bronze', 'player_profiles') }} bp
LEFT JOIN player_agg pa       ON pa.player_id = bp.player_id
LEFT JOIN return_agg ra       ON ra.player_id = bp.player_id
LEFT JOIN latest_snapshot ls  ON ls.player_id = bp.player_id
LEFT JOIN latest_rank lr      ON lr.player_id = bp.player_id
