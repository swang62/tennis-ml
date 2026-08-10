-- gold.player_profiles: one row per player with identity, biography,
-- match counts, career service/return aggregates, surface counts,
-- recent rolling form, rank points, and estimated current rank.
--
-- Every player from bronze.player_profiles is preserved, including
-- zero-match players. Aggregates use weighted sums/denominators with
-- DOUBLE PRECISION and NULLIF — never averages of per-match percentages.
-- Return metrics derive from opponent silver perspectives via self-join,
-- not from widened silver columns.

WITH player_agg AS (
    SELECT
        pm.player_id,
        COUNT(*)                                                   AS match_count,
        MAX(pm.match_date)                                         AS latest_match_date,
        -- latest positive rank points (ignore newer zero/null obs)
        MAX(pm.player_rank_points) FILTER (WHERE pm.player_rank_points > 0)      AS latest_positive_points,
        -- deterministic latest positive via match_date DESC, match_id DESC
        (ARRAY_AGG(pm.player_rank_points ORDER BY pm.match_date DESC, pm.match_id DESC)
            FILTER (WHERE pm.player_rank_points > 0))[1]                  AS latest_positive_points_det,
        -- earliest positive
        (ARRAY_AGG(pm.player_rank_points ORDER BY pm.match_date ASC, pm.match_id ASC)
            FILTER (WHERE pm.player_rank_points > 0))[1]                  AS earliest_positive_points,
        (ARRAY_AGG(pm.match_date ORDER BY pm.match_date ASC, pm.match_id ASC)
            FILTER (WHERE pm.player_rank_points > 0))[1]                  AS earliest_positive_date,
        (ARRAY_AGG(pm.match_date ORDER BY pm.match_date DESC, pm.match_id DESC)
            FILTER (WHERE pm.player_rank_points > 0))[1]                  AS latest_positive_date,

        -- service metrics (weighted sums, DOUBLE PRECISION, NULLIF)
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

        -- return metric: overall (from this player's own row)
        CAST(SUM(pm.return_points_won) AS DOUBLE PRECISION)
            / NULLIF(SUM(pm.return_points_available), 0)
            AS return_points_won_pct,

        -- surface counts and win rates
        COUNT(*) FILTER (WHERE pm.surface = 'hard')                                   AS hard_matches,
        COUNT(*) FILTER (WHERE pm.surface = 'clay')                                   AS clay_matches,
        COUNT(*) FILTER (WHERE pm.surface = 'grass')                                  AS grass_matches,
        CAST(SUM(CASE WHEN pm.surface = 'hard'  THEN pm.match_won ELSE 0 END) AS DOUBLE PRECISION)
            / NULLIF(COUNT(*) FILTER (WHERE pm.surface = 'hard'), 0)                  AS hard_win_rate,
        CAST(SUM(CASE WHEN pm.surface = 'clay'  THEN pm.match_won ELSE 0 END) AS DOUBLE PRECISION)
            / NULLIF(COUNT(*) FILTER (WHERE pm.surface = 'clay'), 0)                  AS clay_win_rate,
        CAST(SUM(CASE WHEN pm.surface = 'grass' THEN pm.match_won ELSE 0 END) AS DOUBLE PRECISION)
            / NULLIF(COUNT(*) FILTER (WHERE pm.surface = 'grass'), 0)                 AS grass_win_rate

    FROM {{ ref('player_matches') }} pm
    GROUP BY pm.player_id
),

-- return metrics from opponent perspective (self-join on same match_id)
return_agg AS (
    SELECT
        pm.player_id,
        -- first-serve return: (opp first serves - opp first serve points won) / opp first serves
        CAST(SUM(opp.first_serves_made - opp.first_serve_points_won) AS DOUBLE PRECISION)
            / NULLIF(SUM(opp.first_serves_made), 0)
            AS first_serve_return_points_won_pct,
        -- second-serve return: (opp second serves - opp second serve points won) / opp second serves
        CAST(SUM((opp.total_serve_points - opp.first_serves_made) - opp.second_serve_points_won)
            AS DOUBLE PRECISION)
            / NULLIF(SUM(opp.total_serve_points - opp.first_serves_made), 0)
            AS second_serve_return_points_won_pct,
        -- break point conversion: (opp BP faced - opp BP saved) / opp BP faced
        CAST(SUM(opp.break_points_faced - opp.break_points_saved) AS DOUBLE PRECISION)
            / NULLIF(SUM(opp.break_points_faced), 0)
            AS break_point_conversion_pct,
        -- BP opportunities per return game: opp BP faced / opp service games
        CAST(SUM(opp.break_points_faced) AS DOUBLE PRECISION)
            / NULLIF(SUM(opp.service_games), 0)
            AS break_point_opportunities_per_return_game
    FROM {{ ref('player_matches') }} pm
    JOIN {{ ref('player_matches') }} opp
        ON opp.match_id = pm.match_id
       AND opp.player_id = pm.opponent_id
    GROUP BY pm.player_id
),

-- latest rolling snapshot per player
latest_snapshot AS (
    SELECT DISTINCT ON (player_id)
        player_id,
        snapshot_date   AS recent_snapshot_date,
        win_rate_10
    FROM {{ ref('rolling_features') }}
    ORDER BY player_id, snapshot_date DESC
)

SELECT
    -- identity/biography (preserve every bronze profile)
    bp.player_id,
    bp.display_name,
    bp.atp_name,
    bp.birthdate,
    bp.weight,
    bp.height,
    bp.turned_pro,
    bp.birthplace,
    bp.coaches,
    bp.handedness,
    bp.backhand,
    bp.ioc,
    bp.summary,
    bp.enriched_at,

    -- match counts
    COALESCE(pa.match_count, 0)               AS match_count,
    pa.latest_match_date,

    -- rank
    pa.latest_positive_points_det             AS latest_rank_points,
    pa.earliest_positive_points               AS earliest_rank_points,
    pa.earliest_positive_date                 AS earliest_rank_points_date,
    pa.latest_positive_date                   AS latest_rank_points_date,
    pa.latest_positive_points_det - pa.earliest_positive_points
        AS rank_points_delta,

    -- service
    pa.first_serve_in_pct,
    pa.aces_per_first_serve,
    pa.first_serve_points_won_pct,
    pa.second_serve_points_won_pct,
    pa.overall_serve_points_won_pct,
    pa.double_faults_per_serve_point,
    pa.aces_per_service_game,
    pa.break_points_saved_pct,

    -- return
    pa.return_points_won_pct,
    ra.first_serve_return_points_won_pct,
    ra.second_serve_return_points_won_pct,
    ra.break_point_conversion_pct,
    ra.break_point_opportunities_per_return_game,

    -- surface
    COALESCE(pa.hard_matches, 0)              AS hard_matches,
    COALESCE(pa.clay_matches, 0)              AS clay_matches,
    COALESCE(pa.grass_matches, 0)             AS grass_matches,
    pa.hard_win_rate,
    pa.clay_win_rate,
    pa.grass_win_rate,

    -- recent form
    ls.recent_snapshot_date,
    ls.win_rate_10
FROM {{ source('bronze', 'player_profiles') }} bp
LEFT JOIN player_agg pa       ON pa.player_id = bp.player_id
LEFT JOIN return_agg ra       ON ra.player_id = bp.player_id
LEFT JOIN latest_snapshot ls  ON ls.player_id = bp.player_id
