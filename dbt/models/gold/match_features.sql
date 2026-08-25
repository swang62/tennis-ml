
{{ config(
    materialized="incremental",
    incremental_strategy="delete+insert",
    unique_key=["player_id", "match_id"],
) }}

WITH
{% if is_incremental() %}
new_matches AS (
    SELECT match_id, player1_id, player2_id, match_date, match_num
    FROM {{ source('bronze', 'match_events') }}
    WHERE ingested_at > COALESCE(
        (SELECT source_watermark FROM bronze.etl_state WHERE pipeline = 'dbt'),
        '-infinity'::TIMESTAMPTZ
    )
),
changed_match_ids AS (
    SELECT match_id
    FROM new_matches
    UNION
    SELECT pm.match_id
    FROM {{ ref('player_matches') }} pm
    JOIN new_matches nm
      ON LEAST(pm.player_id, pm.opponent_id) = LEAST(nm.player1_id, nm.player2_id)
     AND GREATEST(pm.player_id, pm.opponent_id) = GREATEST(nm.player1_id, nm.player2_id)
     AND (pm.match_date, pm.match_num, pm.match_id)
         > (nm.match_date, nm.match_num, nm.match_id)
),
{% endif %}
player_match_enriched AS (
    SELECT
        pm.match_id,
        pm.match_date,
        pm.match_num,
        bron.tournament,
        bron.round,
        pm.surface,
        COALESCE(bron.is_indoor, 0) AS is_indoor,
        COALESCE(bron.best_of, 3) AS best_of,
        pm.player_id,
        pm.opponent_id,
        pm.match_won,

        COALESCE(pr.latest_player_ranking, fd.latest_player_ranking) AS player_ranking,
        COALESCE(pr.latest_player_rank_points, fd.latest_player_rank_points)
            AS player_rank_points,
        COALESCE(pr.latest_player_age, fd.latest_player_age) AS player_age,

        CASE WHEN pr.player_id IS NULL THEN 0
             ELSE CAST(pr.matches_10 AS INTEGER)
        END AS matches_10,

        COALESCE(pr.win_rate_10, fd.win_rate_10) AS win_rate_10,
        COALESCE(pr.ace_rate_10, fd.ace_rate_10) AS ace_rate_10,
        COALESCE(pr.first_serve_pct_10, fd.first_serve_pct_10) AS first_serve_pct_10,
        COALESCE(pr.break_points_saved_pct_10, fd.break_points_saved_pct_10)
            AS break_points_saved_pct_10,
        COALESCE(pr.first_serve_win_pct_10, fd.first_serve_win_pct_10)
            AS first_serve_win_pct_10,
        COALESCE(pr.second_serve_win_pct_10, fd.second_serve_win_pct_10)
            AS second_serve_win_pct_10,
        COALESCE(pr.serve_win_pct_10, fd.serve_win_pct_10) AS serve_win_pct_10,
        COALESCE(pr.return_points_won_pct_10, fd.return_points_won_pct_10)
            AS return_points_won_pct_10,
        COALESCE(pr.dominance, fd.dominance) AS dominance,
        COALESCE(pr.df_rate_10, fd.df_rate_10) AS df_rate_10,
        COALESCE(pr.aces_per_svc_game_10, fd.aces_per_svc_game_10)
            AS aces_per_svc_game_10,

        COALESCE(pr.avg_player_rank_10, fd.avg_player_rank_10)
            - COALESCE(pr.latest_player_ranking, fd.latest_player_ranking)
            AS rank_trend_10,

        COALESCE(pr.avg_rank_faced_10, fd.avg_rank_faced_10) AS avg_rank_faced_10,

        COALESCE(pr.streak, fd.streak) AS streak,

        CASE pm.surface
            WHEN 'clay'  THEN COALESCE(pr.clay_win_rate_10,  fd.clay_win_rate_10)
            WHEN 'grass' THEN COALESCE(pr.grass_win_rate_10, fd.grass_win_rate_10)
            WHEN 'hard'  THEN COALESCE(pr.hard_win_rate_10,  fd.hard_win_rate_10)
            ELSE fd.rate_default
        END AS surface_form,

        LEAST(
            CASE WHEN pr.player_id IS NULL THEN 90.0
                 ELSE pm.match_date - pr.snapshot_date
            END,
            90.0
        ) AS days_since_last_match,

        COALESCE(
            CAST(CASE WHEN prof.handedness = 'L' THEN 1
                      WHEN prof.handedness = 'R' THEN 0 END AS DOUBLE PRECISION),
            fd.left_handed_rate
        ) AS is_left_handed,

        COALESCE(
            CAST(EXTRACT(YEAR FROM pm.match_date) - prof.turned_pro AS DOUBLE PRECISION),
            fd.avg_years_pro
        ) AS years_pro,

        COALESCE(es.pre_elo, 1500.0) AS player_elo,

        COALESCE(pgrad.elo_gradient_10, 0.0) AS player_elo_gradient_10,
        COALESCE(ograd.elo_gradient_10, 0.0) AS opponent_elo_gradient_10

    FROM {{ ref('player_matches') }} pm
{% if is_incremental() %}
    JOIN changed_match_ids cm
        ON cm.match_id = pm.match_id
{% endif %}
    LEFT JOIN LATERAL (
        SELECT rf.* FROM {{ ref('rolling_features') }} rf
        WHERE rf.player_id = pm.player_id
          AND (rf.snapshot_date, rf.match_num, rf.match_id)
              < (pm.match_date, pm.match_num, pm.match_id)
        ORDER BY rf.snapshot_date DESC, rf.match_num DESC, rf.match_id DESC
        LIMIT 1
    ) pr ON true
    CROSS JOIN {{ ref('tour_averages') }} fd
    LEFT JOIN {{ source('bronze', 'match_events') }} bron
        ON bron.match_id = pm.match_id
    LEFT JOIN {{ source('bronze', 'player_profiles') }} prof
        ON prof.player_id = pm.player_id
    LEFT JOIN {{ source('silver', 'elo_snapshots') }} es
        ON es.player_id = pm.player_id
       AND es.match_id = pm.match_id
    LEFT JOIN LATERAL (
        SELECT
            CASE WHEN COUNT(*) < 2 THEN 0.0
            ELSE (COUNT(*) * SUM(idx * post_elo) - SUM(idx) * SUM(post_elo))
                 / (COUNT(*) * SUM(idx * idx) - SUM(idx) * SUM(idx))
            END AS elo_gradient_10
        FROM (
            SELECT
                s.post_elo,
                ROW_NUMBER() OVER (
                    ORDER BY s.match_date, s.match_num, s.match_id
                ) AS idx
            FROM {{ source('silver', 'elo_snapshots') }} s
            WHERE s.player_id = pm.player_id
              AND (s.match_date, s.match_num, s.match_id)
                  < (pm.match_date, pm.match_num, pm.match_id)
            ORDER BY s.match_date DESC, s.match_num DESC, s.match_id DESC
            LIMIT 10
        ) graded
    ) pgrad ON true
    LEFT JOIN LATERAL (
        SELECT
            CASE WHEN COUNT(*) < 2 THEN 0.0
            ELSE (COUNT(*) * SUM(idx * post_elo) - SUM(idx) * SUM(post_elo))
                 / (COUNT(*) * SUM(idx * idx) - SUM(idx) * SUM(idx))
            END AS elo_gradient_10
        FROM (
            SELECT
                s.post_elo,
                ROW_NUMBER() OVER (
                    ORDER BY s.match_date, s.match_num, s.match_id
                ) AS idx
            FROM {{ source('silver', 'elo_snapshots') }} s
            WHERE s.player_id = pm.opponent_id
              AND (s.match_date, s.match_num, s.match_id)
                  < (pm.match_date, pm.match_num, pm.match_id)
            ORDER BY s.match_date DESC, s.match_num DESC, s.match_id DESC
            LIMIT 10
        ) graded
    ) ograd ON true
),
pair_meetings AS (
    SELECT
        current_match.match_id,
        current_match.player_id,
        (meeting.winner_id = current_match.player_id) AS winner_is_current_player,
        (meeting.surface = current_match.surface) AS meeting_surface_matches
    FROM player_match_enriched current_match
    CROSS JOIN LATERAL (
        SELECT winner_id, surface
        FROM {{ source('bronze', 'match_events') }} meeting
        WHERE meeting.player1_id = current_match.player_id
          AND meeting.player2_id = current_match.opponent_id
          AND (meeting.match_date, meeting.match_num, meeting.match_id)
            < (current_match.match_date, current_match.match_num, current_match.match_id)
        ORDER BY meeting.match_date DESC, meeting.match_num DESC, meeting.match_id DESC
    ) meeting
    UNION ALL
    SELECT
        current_match.match_id,
        current_match.player_id,
        (meeting.winner_id = current_match.player_id) AS winner_is_current_player,
        (meeting.surface = current_match.surface) AS meeting_surface_matches
    FROM player_match_enriched current_match
    CROSS JOIN LATERAL (
        SELECT winner_id, surface
        FROM {{ source('bronze', 'match_events') }} meeting
        WHERE meeting.player1_id = current_match.opponent_id
          AND meeting.player2_id = current_match.player_id
          AND (meeting.match_date, meeting.match_num, meeting.match_id)
            < (current_match.match_date, current_match.match_num, current_match.match_id)
        ORDER BY meeting.match_date DESC, meeting.match_num DESC, meeting.match_id DESC
    ) meeting
),
prior_h2h AS (
    SELECT
        match_id,
        player_id,
        COUNT(*) AS recent_meetings,
        SUM(CASE WHEN winner_is_current_player THEN 1 ELSE 0 END) AS wins_for_player,
        COUNT(*) FILTER (WHERE meeting_surface_matches) AS surface_meetings,
        SUM(CASE WHEN winner_is_current_player AND meeting_surface_matches THEN 1 ELSE 0 END)
            AS surface_wins_for_player
    FROM pair_meetings
    GROUP BY match_id, player_id
)
SELECT
    p.match_id,
    p.match_date,
    p.player_id,
    p.opponent_id,
    p.tournament,
    p.round,
    p.surface,
    p.match_won,

    p.player_ranking - o.player_ranking AS rank_diff,
    p.player_rank_points - o.player_rank_points AS rank_points_diff,
    p.player_age - o.player_age AS age_diff,
    p.win_rate_10 - o.win_rate_10 AS form_diff,
    p.ace_rate_10 - o.ace_rate_10 AS ace_rate_diff,
    p.first_serve_pct_10 - o.first_serve_pct_10 AS first_serve_pct_diff,
    p.break_points_saved_pct_10 - o.break_points_saved_pct_10
        AS break_points_saved_pct_diff,
    p.first_serve_win_pct_10 - o.first_serve_win_pct_10 AS first_serve_win_pct_diff,
    p.second_serve_win_pct_10 - o.second_serve_win_pct_10 AS second_serve_win_pct_diff,
    p.serve_win_pct_10 - o.serve_win_pct_10 AS serve_win_pct_diff,
    p.return_points_won_pct_10 - o.return_points_won_pct_10
        AS return_points_won_pct_diff,
    p.dominance - o.dominance AS dominance_diff,
    p.df_rate_10 - o.df_rate_10 AS df_rate_diff,
    p.aces_per_svc_game_10 - o.aces_per_svc_game_10 AS aces_per_svc_game_diff,
    p.rank_trend_10 - o.rank_trend_10 AS rank_trend_diff,
    p.avg_rank_faced_10 - o.avg_rank_faced_10 AS avg_rank_faced_diff,
    p.streak - o.streak AS streak_diff,
    p.surface_form - o.surface_form AS surface_form_diff,
    LN(1.0 + p.days_since_last_match) - LN(1.0 + o.days_since_last_match)
        AS days_since_last_match_diff,

    p.player_elo - o.player_elo AS elo_diff,

    p.player_elo_gradient_10 AS player_elo_gradient_10,
    o.player_elo_gradient_10 AS opponent_elo_gradient_10,

    p.matches_10            AS player_matches_10,
    o.matches_10            AS opponent_matches_10,
    p.is_left_handed        AS player_is_left_handed,
    o.is_left_handed        AS opponent_is_left_handed,
    p.years_pro AS player_years_pro,
    o.years_pro AS opponent_years_pro,

    COALESCE(h.recent_meetings, 0) AS h2h_exposure,
    ((COALESCE(h.wins_for_player, 0) + 1.0)
        / (COALESCE(h.recent_meetings, 0) + 2.0) - 0.5)
        AS h2h_advantage,
    ((COALESCE(h.surface_wins_for_player, 0) + 1.0)
        / (COALESCE(h.surface_meetings, 0) + 2.0) - 0.5)
        AS h2h_surface_advantage,

    CAST(CASE WHEN p.surface = 'clay'  THEN 1 ELSE 0 END AS SMALLINT) AS is_clay,
    CAST(CASE WHEN p.surface = 'grass' THEN 1 ELSE 0 END AS SMALLINT) AS is_grass,
    CAST(CASE WHEN p.surface = 'hard'  THEN 1 ELSE 0 END AS SMALLINT) AS is_hard,
    p.is_indoor,
    CAST(p.best_of AS SMALLINT) AS best_of,
    CAST(CASE p.tournament
        WHEN 'grand_slam' THEN 4 WHEN 'masters' THEN 3
        WHEN 'atp_500' THEN 2 WHEN 'atp_250' THEN 1 ELSE 0
    END AS SMALLINT) AS tournament_level,
    CAST(CASE p.round
        WHEN 'r128' THEN 1 WHEN 'r64' THEN 2 WHEN 'r32' THEN 3 WHEN 'r16' THEN 4
        WHEN 'qf' THEN 5 WHEN 'sf' THEN 6 WHEN 'f' THEN 7 ELSE 0
    END AS SMALLINT) AS round_encoded,

    p.first_serve_pct_10 AS player_first_serve_pct_10,
    o.first_serve_pct_10 AS opponent_first_serve_pct_10,
    p.first_serve_win_pct_10 AS player_first_serve_win_pct_10,
    o.first_serve_win_pct_10 AS opponent_first_serve_win_pct_10,
    p.second_serve_win_pct_10 AS player_second_serve_win_pct_10,
    o.second_serve_win_pct_10 AS opponent_second_serve_win_pct_10,
    p.serve_win_pct_10 AS player_serve_win_pct_10,
    o.serve_win_pct_10 AS opponent_serve_win_pct_10,
    p.return_points_won_pct_10 AS player_return_points_won_pct_10,
    o.return_points_won_pct_10 AS opponent_return_points_won_pct_10
FROM player_match_enriched p
JOIN player_match_enriched o
    ON o.match_id = p.match_id
   AND o.player_id = p.opponent_id
LEFT JOIN prior_h2h h
    ON h.match_id = p.match_id
   AND h.player_id = p.player_id
ORDER BY p.match_date, p.match_id, p.player_id
