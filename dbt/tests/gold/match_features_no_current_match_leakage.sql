-- Assert every snapshot-backed feature of each match_features row comes from
-- the player's PRIOR snapshot only (player_match_number = current match
-- number - 1), for both the canonical player and the opponent side. Each
-- comparison row re-derives the stored feature from rolling_features at N-1
-- (or, for as-of-date features, from the CURRENT silver row — pre-match
-- known, never from the current match's outcome or raw stats) and compares
-- it with the stored value. Any mismatch means the row used the current
-- match's own snapshot N (leakage), its current-match raw stats, or a wrong
-- snapshot. A NULL stored value must also be NULL in the prior snapshot
-- (cold start / zero denominator): NULL is never replaced by a current-match
-- value.
--
-- Covered (all rolling/snapshot-backed fields, incl. the Task 2-3 additions):
--   form:            win_rate_5/10/20, weighted_form_10, win_streak,
--                    loss_streak, days_since_last_match, matches_30d,
--                    surface_win_rate_10
--   serve/break:     ace_rate_5/10, first_serve_pct_5/10,
--                    break_points_saved_pct_5/10, first_serve_win_pct_5/10,
--                    second_serve_win_pct_5/10, serve_win_pct_5/10,
--                    df_rate_5/10, aces_per_svc_game_5/10
--   strength:        avg_rank_faced_5/10, rank_trend_10/20
--   as-of-date:      player/opponent_ranking, player/opponent_age,
--                    rank_points_diff (current event, pre-match known)
{% set direct_cols = [
    "win_rate_5", "win_rate_10", "win_rate_20", "weighted_form_10",
    "ace_rate_5", "ace_rate_10",
    "first_serve_pct_5", "first_serve_pct_10",
    "break_points_saved_pct_5", "break_points_saved_pct_10",
    "first_serve_win_pct_5", "first_serve_win_pct_10",
    "second_serve_win_pct_5", "second_serve_win_pct_10",
    "serve_win_pct_5", "serve_win_pct_10",
    "df_rate_5", "df_rate_10",
    "aces_per_svc_game_5", "aces_per_svc_game_10",
    "avg_rank_faced_5", "avg_rank_faced_10",
    "win_streak", "loss_streak",
] %}
{% set cmp = [] %}
{% for side in ["player", "opponent"] %}
  {% for c in direct_cols %}
    {% do cmp.append(
        "SELECT match_id, '" ~ side ~ "_" ~ c ~ "' AS feature, "
        ~ side ~ "_mf_" ~ c ~ " AS mf_val, "
        ~ side ~ "_prior_" ~ c ~ " AS prior_val FROM prior_snapshot"
    ) %}
  {% endfor %}
{% endfor %}
WITH prior_snapshot AS (
    SELECT
        mf.match_id,
        mf.player_id,
        mf.opponent_id,
        pm.match_date AS match_date,
        pm.surface AS match_surface,
        {% for c in direct_cols %}
        mf.player_{{ c }} AS player_mf_{{ c }},
        mf.opponent_{{ c }} AS opponent_mf_{{ c }},
        prp.{{ c }} AS player_prior_{{ c }},
        pro.{{ c }} AS opponent_prior_{{ c }},
        {% endfor %}
        -- Stored values of the features derived from the prior snapshot /
        -- current silver row (compared against the recomputed value below)
        mf.player_rank_trend_10 AS player_mf_rank_trend_10,
        mf.player_rank_trend_20 AS player_mf_rank_trend_20,
        mf.opponent_rank_trend_10 AS opponent_mf_rank_trend_10,
        mf.opponent_rank_trend_20 AS opponent_mf_rank_trend_20,
        mf.player_days_since_last_match AS player_mf_days_since_last_match,
        mf.opponent_days_since_last_match AS opponent_mf_days_since_last_match,
        mf.player_matches_30d AS player_mf_matches_30d,
        mf.opponent_matches_30d AS opponent_mf_matches_30d,
        mf.player_age AS player_mf_age,
        mf.opponent_age AS opponent_mf_age,
        mf.player_ranking AS player_mf_ranking,
        mf.opponent_ranking AS opponent_mf_ranking,
        mf.rank_points_diff AS mf_rank_points_diff,
        mf.player_surface_win_rate_10 AS player_mf_surface_win_rate_10,
        mf.opponent_surface_win_rate_10 AS opponent_mf_surface_win_rate_10,
        -- Current-event as-of-date values (pre-match known, from silver)
        pm.player_ranking AS player_cur_ranking,
        po.player_ranking AS opponent_cur_ranking,
        pm.player_age AS player_cur_age,
        po.player_age AS opponent_cur_age,
        pm.player_rank_points AS player_cur_rank_points,
        po.player_rank_points AS opponent_cur_rank_points,
        pm.matches_30d_before AS player_cur_matches_30d,
        po.matches_30d_before AS opponent_cur_matches_30d,
        -- Prior-snapshot inputs needed to re-derive the derived features
        prp.avg_player_rank_10 AS player_prior_avg_rank_10,
        prp.avg_player_rank_20 AS player_prior_avg_rank_20,
        pro.avg_player_rank_10 AS opponent_prior_avg_rank_10,
        pro.avg_player_rank_20 AS opponent_prior_avg_rank_20,
        prp.snapshot_date AS player_prior_snapshot_date,
        pro.snapshot_date AS opponent_prior_snapshot_date,
        prp.clay_win_rate_10 AS player_prior_clay_win_rate_10,
        prp.grass_win_rate_10 AS player_prior_grass_win_rate_10,
        prp.hard_win_rate_10 AS player_prior_hard_win_rate_10,
        pro.clay_win_rate_10 AS opponent_prior_clay_win_rate_10,
        pro.grass_win_rate_10 AS opponent_prior_grass_win_rate_10,
        pro.hard_win_rate_10 AS opponent_prior_hard_win_rate_10
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
    {{ cmp | join(" UNION ALL ") }}
    -- Derived: rank trend = prior rolling avg rank minus CURRENT event ranking
    UNION ALL
    SELECT match_id, 'player_rank_trend_10' AS feature,
           player_mf_rank_trend_10 AS mf_val,
           player_prior_avg_rank_10 - player_cur_ranking AS prior_val
    FROM prior_snapshot
    UNION ALL
    SELECT match_id, 'player_rank_trend_20' AS feature,
           player_mf_rank_trend_20 AS mf_val,
           player_prior_avg_rank_20 - player_cur_ranking AS prior_val
    FROM prior_snapshot
    UNION ALL
    SELECT match_id, 'opponent_rank_trend_10' AS feature,
           opponent_mf_rank_trend_10 AS mf_val,
           opponent_prior_avg_rank_10 - opponent_cur_ranking AS prior_val
    FROM prior_snapshot
    UNION ALL
    SELECT match_id, 'opponent_rank_trend_20' AS feature,
           opponent_mf_rank_trend_20 AS mf_val,
           opponent_prior_avg_rank_20 - opponent_cur_ranking AS prior_val
    FROM prior_snapshot
    -- Derived: days since last match = match date minus the PRIOR snapshot's
    -- date; 365 when there is no prior snapshot (cold-start fallback)
    UNION ALL
    SELECT match_id, 'player_days_since_last_match' AS feature,
           player_mf_days_since_last_match AS mf_val,
           CAST(COALESCE(DATEDIFF('day', player_prior_snapshot_date, match_date), 365) AS INTEGER) AS prior_val
    FROM prior_snapshot
    UNION ALL
    SELECT match_id, 'opponent_days_since_last_match' AS feature,
           opponent_mf_days_since_last_match AS mf_val,
           CAST(COALESCE(DATEDIFF('day', opponent_prior_snapshot_date, match_date), 365) AS INTEGER) AS prior_val
    FROM prior_snapshot
    -- Derived: matches_30d is the pre-match count from the current silver row
    -- (a COUNT over [match_date - 30d, match_date), never including the
    -- current match or the prior snapshot's count)
    UNION ALL
    SELECT match_id, 'player_matches_30d' AS feature,
           player_mf_matches_30d AS mf_val,
           CAST(player_cur_matches_30d AS INTEGER) AS prior_val
    FROM prior_snapshot
    UNION ALL
    SELECT match_id, 'opponent_matches_30d' AS feature,
           opponent_mf_matches_30d AS mf_val,
           CAST(opponent_cur_matches_30d AS INTEGER) AS prior_val
    FROM prior_snapshot
    -- Derived: as-of-date per-side values come from the CURRENT silver row,
    -- not from the prior snapshot (stale) and not from any current-match raw
    -- stat (leakage)
    UNION ALL
    SELECT match_id, 'player_age' AS feature,
           player_mf_age AS mf_val, player_cur_age AS prior_val
    FROM prior_snapshot
    UNION ALL
    SELECT match_id, 'opponent_age' AS feature,
           opponent_mf_age AS mf_val, opponent_cur_age AS prior_val
    FROM prior_snapshot
    UNION ALL
    SELECT match_id, 'player_ranking' AS feature,
           player_mf_ranking AS mf_val, player_cur_ranking AS prior_val
    FROM prior_snapshot
    UNION ALL
    SELECT match_id, 'opponent_ranking' AS feature,
           opponent_mf_ranking AS mf_val, opponent_cur_ranking AS prior_val
    FROM prior_snapshot
    UNION ALL
    SELECT match_id, 'rank_points_diff' AS feature,
           mf_rank_points_diff AS mf_val,
           player_cur_rank_points - opponent_cur_rank_points AS prior_val
    FROM prior_snapshot
    -- Derived: surface rate = prior snapshot's rate on the CURRENT surface
    UNION ALL
    SELECT match_id, 'player_surface_win_rate_10' AS feature,
           player_mf_surface_win_rate_10 AS mf_val,
           CASE match_surface
               WHEN 'clay'  THEN player_prior_clay_win_rate_10
               WHEN 'grass' THEN player_prior_grass_win_rate_10
               WHEN 'hard'  THEN player_prior_hard_win_rate_10
           END AS prior_val
    FROM prior_snapshot
    UNION ALL
    SELECT match_id, 'opponent_surface_win_rate_10' AS feature,
           opponent_mf_surface_win_rate_10 AS mf_val,
           CASE match_surface
               WHEN 'clay'  THEN opponent_prior_clay_win_rate_10
               WHEN 'grass' THEN opponent_prior_grass_win_rate_10
               WHEN 'hard'  THEN opponent_prior_hard_win_rate_10
           END AS prior_val
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
