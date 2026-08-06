-- Assert every snapshot-backed feature of each match_features row comes from
-- the player's PRIOR snapshot only (player_match_number = current match
-- number - 1), for both the canonical player and the opponent side.
--
-- The finalized 36-col contract keeps most rolling values as DIFFS (canonical
-- minus opponent), so the strongest leakage check re-derives each diff from
-- the two prior snapshots and compares it with the stored value. Any mismatch
-- means the row used a current-match snapshot (leakage), its current-match
-- raw stats, or a wrong snapshot. Per-side absolute values (weighted_form_10,
-- surface_win_rate_10, days_since_last_match, matches_30d) are compared
-- directly against the prior snapshot / current silver row. As-of-date values
-- (ranking, rank_points, age) come from the CURRENT silver row (pre-match
-- known, never from current-match raw stats).
--
-- Covered snapshot-backed fields:
--   diff form:      win_rate_diff, streak_diff, surface (via per-side)
--   diff serve/bk:  ace_rate_diff, first_serve_pct_diff,
--                    break_points_saved_pct_diff, first_serve_win_pct_diff,
--                    second_serve_win_pct_diff, serve_win_pct_diff,
--                    df_rate_diff, aces_per_svc_game_diff
--   diff strength:  avg_rank_faced_diff, rank_trend_diff
--   per-side:       player/opponent_weighted_form_10,
--                    player/opponent_surface_win_rate_10,
--                    player/opponent_days_since_last_match,
--                    player/opponent_matches_30d
--   as-of-date:     player/opponent_ranking, player/opponent_age,
--                    rank_points_diff (current event, pre-match known)
{% set diff_cols = [
    "win_rate_10", "ace_rate_10", "first_serve_pct_10",
    "break_points_saved_pct_10", "first_serve_win_pct_10",
    "second_serve_win_pct_10", "serve_win_pct_10",
    "df_rate_10", "aces_per_svc_game_10", "avg_rank_faced_10",
] %}
{% set diff_stored = {
    "win_rate_10": "win_rate_diff", "ace_rate_10": "ace_rate_diff",
    "first_serve_pct_10": "first_serve_pct_diff",
    "break_points_saved_pct_10": "break_points_saved_pct_diff",
    "first_serve_win_pct_10": "first_serve_win_pct_diff",
    "second_serve_win_pct_10": "second_serve_win_pct_diff",
    "serve_win_pct_10": "serve_win_pct_diff",
    "df_rate_10": "df_rate_diff",
    "aces_per_svc_game_10": "aces_per_svc_game_diff",
    "avg_rank_faced_10": "avg_rank_faced_diff",
} %}
WITH prior_snapshot AS (
    SELECT
        mf.match_id,
        mf.player_id,
        mf.opponent_id,
        pm.match_date,
        pm.surface AS match_surface,
        -- Stored diff features
        mf.win_rate_diff, mf.streak_diff,
        mf.ace_rate_diff, mf.first_serve_pct_diff,
        mf.break_points_saved_pct_diff, mf.first_serve_win_pct_diff,
        mf.second_serve_win_pct_diff, mf.serve_win_pct_diff,
        mf.df_rate_diff, mf.aces_per_svc_game_diff,
        mf.avg_rank_faced_diff, mf.rank_trend_diff,
        mf.rank_diff, mf.rank_points_diff, mf.age_diff,
        -- Stored per-side absolute values
        mf.player_weighted_form_10, mf.opponent_weighted_form_10,
        mf.player_surface_win_rate_10, mf.opponent_surface_win_rate_10,
        mf.player_days_since_last_match, mf.opponent_days_since_last_match,
        mf.player_matches_30d, mf.opponent_matches_30d,
        -- Prior snapshot inputs (N-1)
        {% for c in diff_cols %}
        prp.{{ c }} AS player_prior_{{ c }},
        pro.{{ c }} AS opponent_prior_{{ c }},
        {% endfor %}
        prp.streak AS player_prior_streak,
        pro.streak AS opponent_prior_streak,
        prp.weighted_form_10 AS player_prior_weighted_form_10,
        pro.weighted_form_10 AS opponent_prior_weighted_form_10,
        prp.avg_player_rank_10 AS player_prior_avg_rank_10,
        pro.avg_player_rank_10 AS opponent_prior_avg_rank_10,
        prp.snapshot_date AS player_prior_snapshot_date,
        pro.snapshot_date AS opponent_prior_snapshot_date,
        prp.clay_win_rate_10 AS player_prior_clay_win_rate_10,
        prp.grass_win_rate_10 AS player_prior_grass_win_rate_10,
        prp.hard_win_rate_10 AS player_prior_hard_win_rate_10,
        pro.clay_win_rate_10 AS opponent_prior_clay_win_rate_10,
        pro.grass_win_rate_10 AS opponent_prior_grass_win_rate_10,
        pro.hard_win_rate_10 AS opponent_prior_hard_win_rate_10,
        -- Current-event as-of-date values (pre-match known, from silver)
        pm.player_ranking AS player_cur_ranking,
        po.player_ranking AS opponent_cur_ranking,
        pm.player_age AS player_cur_age,
        po.player_age AS opponent_cur_age,
        pm.player_rank_points AS player_cur_rank_points,
        po.player_rank_points AS opponent_cur_rank_points,
        pm.matches_30d_before AS player_cur_matches_30d,
        po.matches_30d_before AS opponent_cur_matches_30d
    FROM {{ ref('match_features') }} mf
    JOIN {{ ref('player_matches') }} pm
      ON pm.match_id = mf.match_id AND pm.player_id = mf.player_id
    JOIN {{ ref('player_matches') }} po
      ON po.match_id = mf.match_id AND po.player_id = mf.opponent_id
    LEFT JOIN {{ ref('rolling_features') }} prp
      ON prp.player_id = mf.player_id
     AND prp.player_match_number = pm.player_match_number - 1
    LEFT JOIN {{ ref('rolling_features') }} pro
      ON pro.player_id = mf.opponent_id
     AND pro.player_match_number = po.player_match_number - 1
),
comparisons AS (
    {% for c in diff_cols %}
    SELECT match_id, '{{ c }}_diff' AS feature, {{ diff_stored[c] }} AS mf_val,
           player_prior_{{ c }} - opponent_prior_{{ c }} AS prior_val
    FROM prior_snapshot
    UNION ALL
    {% endfor %}
    SELECT match_id, 'streak_diff' AS feature, streak_diff AS mf_val,
           player_prior_streak - opponent_prior_streak AS prior_val
    FROM prior_snapshot
    UNION ALL
    SELECT match_id, 'rank_trend_diff' AS feature, rank_trend_diff AS mf_val,
           (player_prior_avg_rank_10 - player_cur_ranking)
           - (opponent_prior_avg_rank_10 - opponent_cur_ranking) AS prior_val
    FROM prior_snapshot
    UNION ALL
    SELECT match_id, 'player_weighted_form_10' AS feature,
           player_weighted_form_10 AS mf_val, player_prior_weighted_form_10 AS prior_val
    FROM prior_snapshot
    UNION ALL
    SELECT match_id, 'opponent_weighted_form_10' AS feature,
           opponent_weighted_form_10 AS mf_val, opponent_prior_weighted_form_10 AS prior_val
    FROM prior_snapshot
    UNION ALL
    SELECT match_id, 'player_days_since_last_match' AS feature,
           player_days_since_last_match AS mf_val,
           CAST(COALESCE(match_date - player_prior_snapshot_date, 365) AS INTEGER) AS prior_val
    FROM prior_snapshot
    UNION ALL
    SELECT match_id, 'opponent_days_since_last_match' AS feature,
           opponent_days_since_last_match AS mf_val,
           CAST(COALESCE(match_date - opponent_prior_snapshot_date, 365) AS INTEGER) AS prior_val
    FROM prior_snapshot
    UNION ALL
    SELECT match_id, 'player_matches_30d' AS feature,
           player_matches_30d AS mf_val,
           CAST(player_cur_matches_30d AS INTEGER) AS prior_val
    FROM prior_snapshot
    UNION ALL
    SELECT match_id, 'opponent_matches_30d' AS feature,
           opponent_matches_30d AS mf_val,
           CAST(opponent_cur_matches_30d AS INTEGER) AS prior_val
    FROM prior_snapshot
    UNION ALL
    SELECT match_id, 'player_surface_win_rate_10' AS feature,
           player_surface_win_rate_10 AS mf_val,
           CASE match_surface
               WHEN 'clay'  THEN player_prior_clay_win_rate_10
               WHEN 'grass' THEN player_prior_grass_win_rate_10
               WHEN 'hard'  THEN player_prior_hard_win_rate_10
           END AS prior_val
    FROM prior_snapshot
    UNION ALL
    SELECT match_id, 'opponent_surface_win_rate_10' AS feature,
           opponent_surface_win_rate_10 AS mf_val,
           CASE match_surface
               WHEN 'clay'  THEN opponent_prior_clay_win_rate_10
               WHEN 'grass' THEN opponent_prior_grass_win_rate_10
               WHEN 'hard'  THEN opponent_prior_hard_win_rate_10
           END AS prior_val
    FROM prior_snapshot
    UNION ALL
    SELECT match_id, 'player_age' AS feature, age_diff AS mf_val,
           player_cur_age - opponent_cur_age AS prior_val
    FROM prior_snapshot
    UNION ALL
    SELECT match_id, 'rank_diff' AS feature, rank_diff AS mf_val,
           player_cur_ranking - opponent_cur_ranking AS prior_val
    FROM prior_snapshot
    UNION ALL
    SELECT match_id, 'rank_points_diff' AS feature, rank_points_diff AS mf_val,
           player_cur_rank_points - opponent_cur_rank_points AS prior_val
    FROM prior_snapshot
)
SELECT
    c.match_id,
    ps.player_id,
    ps.opponent_id,
    c.feature,
    c.mf_val,
    c.prior_val
FROM comparisons c
JOIN prior_snapshot ps USING (match_id)
WHERE c.mf_val IS DISTINCT FROM c.prior_val
ORDER BY c.match_id, c.feature