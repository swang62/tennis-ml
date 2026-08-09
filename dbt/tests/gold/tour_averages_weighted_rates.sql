-- Assert every weighted tour rate in tour_averages equals a direct
-- SUM/SUM recomputation from silver.player_matches. IS DISTINCT FROM catches
-- NULL vs non-NULL mismatches. Any returned row is a violation.
WITH direct AS (
    SELECT
        CAST(SUM(aces) AS DOUBLE PRECISION)
            / NULLIF(SUM(first_serves_made), 0) AS ace_rate,
        CAST(SUM(first_serves_made) AS DOUBLE PRECISION)
            / NULLIF(SUM(total_serve_points), 0) AS first_serve_pct,
        CAST(SUM(break_points_saved) AS DOUBLE PRECISION)
            / NULLIF(SUM(break_points_faced), 0) AS break_points_saved_pct,
        CAST(SUM(first_serve_points_won) AS DOUBLE PRECISION)
            / NULLIF(SUM(first_serves_made), 0) AS first_serve_win_pct,
        CAST(SUM(second_serve_points_won) AS DOUBLE PRECISION)
            / NULLIF(SUM(total_serve_points - first_serves_made), 0)
            AS second_serve_win_pct,
        CAST(SUM(first_serve_points_won + second_serve_points_won) AS DOUBLE PRECISION)
            / NULLIF(SUM(total_serve_points), 0) AS serve_win_pct,
        CAST(SUM(return_points_won) AS DOUBLE PRECISION)
            / NULLIF(SUM(return_points_available), 0) AS return_points_won_pct,
        CAST(SUM(double_faults) AS DOUBLE PRECISION)
            / NULLIF(SUM(total_serve_points), 0) AS df_rate,
        CAST(SUM(aces) AS DOUBLE PRECISION)
            / NULLIF(SUM(service_games), 0) AS aces_per_svc_game
    FROM {{ ref('player_matches') }}
)
SELECT ta.singleton_id, 'tour_ace_rate' AS column_name,
       ta.tour_ace_rate AS stored, d.ace_rate AS expected
FROM {{ ref('tour_averages') }} ta
CROSS JOIN direct d
WHERE ta.tour_ace_rate IS DISTINCT FROM d.ace_rate
UNION ALL
SELECT ta.singleton_id, 'tour_first_serve_pct', ta.tour_first_serve_pct, d.first_serve_pct
FROM {{ ref('tour_averages') }} ta
CROSS JOIN direct d
WHERE ta.tour_first_serve_pct IS DISTINCT FROM d.first_serve_pct
UNION ALL
SELECT ta.singleton_id, 'tour_break_points_saved_pct',
       ta.tour_break_points_saved_pct, d.break_points_saved_pct
FROM {{ ref('tour_averages') }} ta
CROSS JOIN direct d
WHERE ta.tour_break_points_saved_pct IS DISTINCT FROM d.break_points_saved_pct
UNION ALL
SELECT ta.singleton_id, 'tour_first_serve_win_pct', ta.tour_first_serve_win_pct, d.first_serve_win_pct
FROM {{ ref('tour_averages') }} ta
CROSS JOIN direct d
WHERE ta.tour_first_serve_win_pct IS DISTINCT FROM d.first_serve_win_pct
UNION ALL
SELECT ta.singleton_id, 'tour_second_serve_win_pct',
       ta.tour_second_serve_win_pct, d.second_serve_win_pct
FROM {{ ref('tour_averages') }} ta
CROSS JOIN direct d
WHERE ta.tour_second_serve_win_pct IS DISTINCT FROM d.second_serve_win_pct
UNION ALL
SELECT ta.singleton_id, 'tour_serve_win_pct', ta.tour_serve_win_pct, d.serve_win_pct
FROM {{ ref('tour_averages') }} ta
CROSS JOIN direct d
WHERE ta.tour_serve_win_pct IS DISTINCT FROM d.serve_win_pct
UNION ALL
SELECT ta.singleton_id, 'tour_return_points_won_pct',
       ta.tour_return_points_won_pct, d.return_points_won_pct
FROM {{ ref('tour_averages') }} ta
CROSS JOIN direct d
WHERE ta.tour_return_points_won_pct IS DISTINCT FROM d.return_points_won_pct
UNION ALL
SELECT ta.singleton_id, 'tour_df_rate', ta.tour_df_rate, d.df_rate
FROM {{ ref('tour_averages') }} ta
CROSS JOIN direct d
WHERE ta.tour_df_rate IS DISTINCT FROM d.df_rate
UNION ALL
SELECT ta.singleton_id, 'tour_aces_per_svc_game', ta.tour_aces_per_svc_game, d.aces_per_svc_game
FROM {{ ref('tour_averages') }} ta
CROSS JOIN direct d
WHERE ta.tour_aces_per_svc_game IS DISTINCT FROM d.aces_per_svc_game
