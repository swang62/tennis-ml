-- Assert every applicable rate column in the gold.tour_averages singleton is
-- bounded to [0, 1], while tour_aces_per_svc_game and
-- tour_break_point_opportunities_per_return_game are per-game rates that only
-- need a lower bound of 0. Tour benchmark columns may be NULL (zero denominator);
-- NULLs are skipped. Any returned row violates a bound.
SELECT
    singleton_id,
    'win_rate_10' AS column_name, win_rate_10 AS value
FROM {{ ref('tour_averages') }}
WHERE win_rate_10 IS NOT NULL AND (win_rate_10 < 0 OR win_rate_10 > 1)
UNION ALL
SELECT singleton_id, 'first_serve_pct_10', first_serve_pct_10
FROM {{ ref('tour_averages') }}
WHERE first_serve_pct_10 IS NOT NULL AND (first_serve_pct_10 < 0 OR first_serve_pct_10 > 1)
UNION ALL
SELECT singleton_id, 'break_points_saved_pct_10', break_points_saved_pct_10
FROM {{ ref('tour_averages') }}
WHERE break_points_saved_pct_10 IS NOT NULL
  AND (break_points_saved_pct_10 < 0 OR break_points_saved_pct_10 > 1)
UNION ALL
SELECT singleton_id, 'first_serve_win_pct_10', first_serve_win_pct_10
FROM {{ ref('tour_averages') }}
WHERE first_serve_win_pct_10 IS NOT NULL
  AND (first_serve_win_pct_10 < 0 OR first_serve_win_pct_10 > 1)
UNION ALL
SELECT singleton_id, 'second_serve_win_pct_10', second_serve_win_pct_10
FROM {{ ref('tour_averages') }}
WHERE second_serve_win_pct_10 IS NOT NULL
  AND (second_serve_win_pct_10 < 0 OR second_serve_win_pct_10 > 1)
UNION ALL
SELECT singleton_id, 'serve_win_pct_10', serve_win_pct_10
FROM {{ ref('tour_averages') }}
WHERE serve_win_pct_10 IS NOT NULL AND (serve_win_pct_10 < 0 OR serve_win_pct_10 > 1)
UNION ALL
SELECT singleton_id, 'return_points_won_pct_10', return_points_won_pct_10
FROM {{ ref('tour_averages') }}
WHERE return_points_won_pct_10 IS NOT NULL
  AND (return_points_won_pct_10 < 0 OR return_points_won_pct_10 > 1)
UNION ALL
SELECT singleton_id, 'clay_win_rate_10', clay_win_rate_10
FROM {{ ref('tour_averages') }}
WHERE clay_win_rate_10 IS NOT NULL AND (clay_win_rate_10 < 0 OR clay_win_rate_10 > 1)
UNION ALL
SELECT singleton_id, 'grass_win_rate_10', grass_win_rate_10
FROM {{ ref('tour_averages') }}
WHERE grass_win_rate_10 IS NOT NULL AND (grass_win_rate_10 < 0 OR grass_win_rate_10 > 1)
UNION ALL
SELECT singleton_id, 'hard_win_rate_10', hard_win_rate_10
FROM {{ ref('tour_averages') }}
WHERE hard_win_rate_10 IS NOT NULL AND (hard_win_rate_10 < 0 OR hard_win_rate_10 > 1)
UNION ALL
SELECT singleton_id, 'left_handed_rate', left_handed_rate
FROM {{ ref('tour_averages') }}
WHERE left_handed_rate IS NOT NULL AND (left_handed_rate < 0 OR left_handed_rate > 1)
UNION ALL
SELECT singleton_id, 'rate_default', rate_default
FROM {{ ref('tour_averages') }}
WHERE rate_default IS NOT NULL AND (rate_default < 0 OR rate_default > 1)
UNION ALL
SELECT singleton_id, 'tour_ace_rate', tour_ace_rate
FROM {{ ref('tour_averages') }}
WHERE tour_ace_rate IS NOT NULL AND (tour_ace_rate < 0 OR tour_ace_rate > 1)
UNION ALL
SELECT singleton_id, 'tour_first_serve_pct', tour_first_serve_pct
FROM {{ ref('tour_averages') }}
WHERE tour_first_serve_pct IS NOT NULL
  AND (tour_first_serve_pct < 0 OR tour_first_serve_pct > 1)
UNION ALL
SELECT singleton_id, 'tour_break_points_saved_pct', tour_break_points_saved_pct
FROM {{ ref('tour_averages') }}
WHERE tour_break_points_saved_pct IS NOT NULL
  AND (tour_break_points_saved_pct < 0 OR tour_break_points_saved_pct > 1)
UNION ALL
SELECT singleton_id, 'tour_first_serve_win_pct', tour_first_serve_win_pct
FROM {{ ref('tour_averages') }}
WHERE tour_first_serve_win_pct IS NOT NULL
  AND (tour_first_serve_win_pct < 0 OR tour_first_serve_win_pct > 1)
UNION ALL
SELECT singleton_id, 'tour_second_serve_win_pct', tour_second_serve_win_pct
FROM {{ ref('tour_averages') }}
WHERE tour_second_serve_win_pct IS NOT NULL
  AND (tour_second_serve_win_pct < 0 OR tour_second_serve_win_pct > 1)
UNION ALL
SELECT singleton_id, 'tour_serve_win_pct', tour_serve_win_pct
FROM {{ ref('tour_averages') }}
WHERE tour_serve_win_pct IS NOT NULL AND (tour_serve_win_pct < 0 OR tour_serve_win_pct > 1)
UNION ALL
SELECT singleton_id, 'tour_return_points_won_pct', tour_return_points_won_pct
FROM {{ ref('tour_averages') }}
WHERE tour_return_points_won_pct IS NOT NULL
  AND (tour_return_points_won_pct < 0 OR tour_return_points_won_pct > 1)
UNION ALL
SELECT singleton_id, 'tour_df_rate', tour_df_rate
FROM {{ ref('tour_averages') }}
WHERE tour_df_rate IS NOT NULL AND (tour_df_rate < 0 OR tour_df_rate > 1)
UNION ALL
SELECT singleton_id, 'tour_aces_per_svc_game', tour_aces_per_svc_game
FROM {{ ref('tour_averages') }}
WHERE tour_aces_per_svc_game IS NOT NULL AND tour_aces_per_svc_game < 0
UNION ALL
SELECT singleton_id, 'tour_break_point_opportunities_per_return_game',
       tour_break_point_opportunities_per_return_game
FROM {{ ref('tour_averages') }}
WHERE tour_break_point_opportunities_per_return_game IS NOT NULL
  AND tour_break_point_opportunities_per_return_game < 0
