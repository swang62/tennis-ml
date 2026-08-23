{% set sample_sizes = {
    "early": 25, "late": 25, "cold": 50, "same_date": 50, "h2h": 50,
} %}
{% set uniform_denom = 256 %}
{% set diff_cols = [
    "win_rate_10", "ace_rate_10", "first_serve_pct_10",
    "break_points_saved_pct_10", "first_serve_win_pct_10",
    "second_serve_win_pct_10", "serve_win_pct_10",
    "return_points_won_pct_10", "df_rate_10", "aces_per_svc_game_10",
    "avg_rank_faced_10",
] %}
{% set diff_stored = {
    "win_rate_10": "win_rate_diff", "ace_rate_10": "ace_rate_diff",
    "first_serve_pct_10": "first_serve_pct_diff",
    "break_points_saved_pct_10": "break_points_saved_pct_diff",
    "first_serve_win_pct_10": "first_serve_win_pct_diff",
    "second_serve_win_pct_10": "second_serve_win_pct_diff",
    "serve_win_pct_10": "serve_win_pct_diff",
    "return_points_won_pct_10": "return_points_won_pct_diff",
    "df_rate_10": "df_rate_diff",
    "aces_per_svc_game_10": "aces_per_svc_game_diff",
    "avg_rank_faced_10": "avg_rank_faced_diff",
} %}
WITH sampled_matches AS (
    SELECT match_id
    FROM (
        (SELECT DISTINCT match_id, match_date
         FROM {{ ref('player_matches') }}
         ORDER BY match_date, match_id
         LIMIT {{ sample_sizes.early }})
        UNION ALL
        (SELECT DISTINCT match_id, match_date
         FROM {{ ref('player_matches') }}
         ORDER BY match_date DESC, match_id DESC
         LIMIT {{ sample_sizes.late }})
        UNION ALL
        (SELECT DISTINCT match_id, match_date
         FROM {{ ref('player_matches') }}
         WHERE player_match_number = 1
         ORDER BY match_date, match_id
         LIMIT {{ sample_sizes.cold }})
        UNION ALL
        (SELECT DISTINCT match_id, match_date
         FROM {{ ref('player_matches') }}
         WHERE (player_id, match_date) IN (
             SELECT player_id, match_date
             FROM {{ ref('player_matches') }}
             GROUP BY player_id, match_date
             HAVING COUNT(*) > 1
         )
         ORDER BY match_date, match_id
         LIMIT {{ sample_sizes.same_date }})
        UNION ALL
        (SELECT DISTINCT match_id, match_date
         FROM {{ ref('player_matches') }}
         WHERE (LEAST(player_id, opponent_id), GREATEST(player_id, opponent_id)) IN (
             SELECT LEAST(player_id, opponent_id), GREATEST(player_id, opponent_id)
             FROM {{ ref('player_matches') }}
             GROUP BY 1, 2
             HAVING COUNT(DISTINCT match_id) >= 2
         )
         ORDER BY match_date, match_id
         LIMIT {{ sample_sizes.h2h }})
        UNION ALL
        (SELECT DISTINCT match_id, match_date
         FROM {{ ref('player_matches') }}
         WHERE get_byte(decode(md5(match_id), 'hex'), 0) % {{ uniform_denom }} = 0)
    ) strata
    GROUP BY match_id
),
prior_snapshot AS (
    SELECT
        mf.match_id,
        mf.player_id,
        mf.opponent_id,
        pm.match_date,
        pm.surface AS match_surface,
        mf.win_rate_diff, mf.streak_diff,
        mf.ace_rate_diff, mf.first_serve_pct_diff,
        mf.break_points_saved_pct_diff, mf.first_serve_win_pct_diff,
        mf.second_serve_win_pct_diff, mf.serve_win_pct_diff,
        mf.return_points_won_pct_diff,
        mf.df_rate_diff, mf.aces_per_svc_game_diff,
        mf.avg_rank_faced_diff, mf.rank_trend_diff,
        mf.rank_diff, mf.rank_points_diff, mf.age_diff,
        mf.surface_form_diff, mf.days_since_last_match_diff,
        mf.player_weighted_form_10, mf.opponent_weighted_form_10,
        {% for c in diff_cols %}
        prp.{{ c }} AS player_raw_{{ c }},
        pro.{{ c }} AS opponent_raw_{{ c }},
        COALESCE(prp.{{ c }}, fd.{{ c }}) AS player_prior_{{ c }},
        COALESCE(pro.{{ c }}, fd.{{ c }}) AS opponent_prior_{{ c }},
        {% endfor %}
        prp.streak AS player_raw_streak,
        pro.streak AS opponent_raw_streak,
        COALESCE(prp.streak, fd.streak) AS player_prior_streak,
        COALESCE(pro.streak, fd.streak) AS opponent_prior_streak,
        prp.weighted_form_10 AS player_raw_weighted_form_10,
        pro.weighted_form_10 AS opponent_raw_weighted_form_10,
        COALESCE(prp.weighted_form_10, fd.weighted_form_10)
            AS player_prior_weighted_form_10,
        COALESCE(pro.weighted_form_10, fd.weighted_form_10)
            AS opponent_prior_weighted_form_10,
        prp.avg_player_rank_10 AS player_raw_avg_rank_10,
        pro.avg_player_rank_10 AS opponent_raw_avg_rank_10,
        COALESCE(prp.avg_player_rank_10, fd.avg_player_rank_10)
            AS player_prior_avg_rank_10,
        COALESCE(pro.avg_player_rank_10, fd.avg_player_rank_10)
            AS opponent_prior_avg_rank_10,
        prp.latest_player_ranking AS player_raw_ranking,
        pro.latest_player_ranking AS opponent_raw_ranking,
        COALESCE(prp.latest_player_ranking, fd.latest_player_ranking)
            AS player_prior_ranking,
        COALESCE(pro.latest_player_ranking, fd.latest_player_ranking)
            AS opponent_prior_ranking,
        prp.latest_player_rank_points AS player_raw_rank_points,
        pro.latest_player_rank_points AS opponent_raw_rank_points,
        COALESCE(prp.latest_player_rank_points, fd.latest_player_rank_points)
            AS player_prior_rank_points,
        COALESCE(pro.latest_player_rank_points, fd.latest_player_rank_points)
            AS opponent_prior_rank_points,
        prp.latest_player_age AS player_raw_age,
        pro.latest_player_age AS opponent_raw_age,
        COALESCE(prp.latest_player_age, fd.latest_player_age)
            AS player_prior_age,
        COALESCE(pro.latest_player_age, fd.latest_player_age)
            AS opponent_prior_age,
        mf.player_matches_10,
        mf.opponent_matches_10,
        prp.matches_10 AS player_raw_matches_10,
        pro.matches_10 AS opponent_raw_matches_10,
        COALESCE(prp.matches_10, 0) AS player_prior_matches_10,
        COALESCE(pro.matches_10, 0) AS opponent_prior_matches_10,
        CASE pm.surface
            WHEN 'clay'  THEN prp.clay_win_rate_10
            WHEN 'grass' THEN prp.grass_win_rate_10
            WHEN 'hard'  THEN prp.hard_win_rate_10
            ELSE NULL
        END AS player_raw_surface_form,
        CASE pm.surface
            WHEN 'clay'  THEN pro.clay_win_rate_10
            WHEN 'grass' THEN pro.grass_win_rate_10
            WHEN 'hard'  THEN pro.hard_win_rate_10
            ELSE NULL
        END AS opponent_raw_surface_form,
        CASE pm.surface
            WHEN 'clay'  THEN COALESCE(prp.clay_win_rate_10,  fd.clay_win_rate_10)
            WHEN 'grass' THEN COALESCE(prp.grass_win_rate_10, fd.grass_win_rate_10)
            WHEN 'hard'  THEN COALESCE(prp.hard_win_rate_10,  fd.hard_win_rate_10)
            ELSE fd.rate_default
        END AS player_prior_surface_form,
        CASE pm.surface
            WHEN 'clay'  THEN COALESCE(pro.clay_win_rate_10,  fd.clay_win_rate_10)
            WHEN 'grass' THEN COALESCE(pro.grass_win_rate_10, fd.grass_win_rate_10)
            WHEN 'hard'  THEN COALESCE(pro.hard_win_rate_10,  fd.hard_win_rate_10)
            ELSE fd.rate_default
        END AS opponent_prior_surface_form,
        pm.match_date - prp.snapshot_date AS player_raw_days_since,
        po.match_date - pro.snapshot_date AS opponent_raw_days_since,
        LEAST(30, CASE WHEN prp.player_id IS NULL THEN fd.days_since_default
             ELSE pm.match_date - prp.snapshot_date END) AS player_prior_days_since,
        LEAST(30, CASE WHEN pro.player_id IS NULL THEN fd.days_since_default
             ELSE po.match_date - pro.snapshot_date END) AS opponent_prior_days_since
    FROM (
        SELECT * FROM {{ ref('match_features') }}
        WHERE match_id IN (SELECT match_id FROM sampled_matches)
    ) mf
    JOIN {{ ref('player_matches') }} pm
      ON pm.match_id = mf.match_id AND pm.player_id = mf.player_id
    JOIN {{ ref('player_matches') }} po
      ON po.match_id = mf.match_id AND po.player_id = mf.opponent_id
    LEFT JOIN LATERAL (
        SELECT * FROM {{ ref('rolling_features') }} rfp
        WHERE rfp.player_id = mf.player_id
          AND (rfp.snapshot_date, rfp.match_num, rfp.match_id)
              < (pm.match_date, pm.match_num, pm.match_id)
        ORDER BY rfp.snapshot_date DESC, rfp.match_num DESC, rfp.match_id DESC
        LIMIT 1
    ) prp ON true
    LEFT JOIN LATERAL (
        SELECT * FROM {{ ref('rolling_features') }} rfo
        WHERE rfo.player_id = mf.opponent_id
          AND (rfo.snapshot_date, rfo.match_num, rfo.match_id)
              < (po.match_date, po.match_num, po.match_id)
        ORDER BY rfo.snapshot_date DESC, rfo.match_num DESC, rfo.match_id DESC
        LIMIT 1
    ) pro ON true
    CROSS JOIN {{ ref('tour_averages') }} fd
),
comparisons AS (
    {% for c in diff_cols %}
    SELECT match_id, player_id, '{{ c }}_diff' AS feature, {{ diff_stored[c] }} AS mf_val,
           player_prior_{{ c }} - opponent_prior_{{ c }} AS prior_val,
           player_raw_{{ c }} IS NOT NULL AND opponent_raw_{{ c }} IS NOT NULL AS guard
    FROM prior_snapshot
    UNION ALL
    {% endfor %}
    SELECT match_id, player_id, 'streak_diff' AS feature, streak_diff AS mf_val,
           player_prior_streak - opponent_prior_streak AS prior_val,
           player_raw_streak IS NOT NULL AND opponent_raw_streak IS NOT NULL AS guard
    FROM prior_snapshot
    UNION ALL
    SELECT match_id, player_id, 'surface_form_diff' AS feature, surface_form_diff AS mf_val,
           player_prior_surface_form - opponent_prior_surface_form AS prior_val,
           player_raw_surface_form IS NOT NULL AND opponent_raw_surface_form IS NOT NULL AS guard
    FROM prior_snapshot
    UNION ALL
    SELECT match_id, player_id, 'days_since_last_match_diff' AS feature, days_since_last_match_diff AS mf_val,
           LN(1.0 + player_prior_days_since) - LN(1.0 + opponent_prior_days_since) AS prior_val,
           player_raw_days_since IS NOT NULL AND opponent_raw_days_since IS NOT NULL AS guard
    FROM prior_snapshot
    UNION ALL
    SELECT match_id, player_id, 'rank_trend_diff' AS feature, rank_trend_diff AS mf_val,
           (player_prior_avg_rank_10 - player_prior_ranking)
               - (opponent_prior_avg_rank_10 - opponent_prior_ranking) AS prior_val,
           player_raw_avg_rank_10 IS NOT NULL AND player_raw_ranking IS NOT NULL
           AND opponent_raw_avg_rank_10 IS NOT NULL AND opponent_raw_ranking IS NOT NULL
           AS guard
    FROM prior_snapshot
    UNION ALL
    SELECT match_id, player_id, 'player_weighted_form_10' AS feature,
           player_weighted_form_10 AS mf_val,
           player_prior_weighted_form_10 AS prior_val,
           player_raw_weighted_form_10 IS NOT NULL AS guard
    FROM prior_snapshot
    UNION ALL
    SELECT match_id, player_id, 'opponent_weighted_form_10' AS feature,
           opponent_weighted_form_10 AS mf_val,
           opponent_prior_weighted_form_10 AS prior_val,
           opponent_raw_weighted_form_10 IS NOT NULL AS guard
    FROM prior_snapshot
    UNION ALL
    SELECT match_id, player_id, 'player_matches_10' AS feature,
           player_matches_10 AS mf_val, player_prior_matches_10 AS prior_val,
           player_raw_matches_10 IS NOT NULL AS guard
    FROM prior_snapshot
    UNION ALL
    SELECT match_id, player_id, 'opponent_matches_10' AS feature,
           opponent_matches_10 AS mf_val, opponent_prior_matches_10 AS prior_val,
           opponent_raw_matches_10 IS NOT NULL AS guard
    FROM prior_snapshot
    UNION ALL
    SELECT match_id, player_id, 'age_diff' AS feature, age_diff AS mf_val,
           player_prior_age - opponent_prior_age AS prior_val,
           player_raw_age IS NOT NULL AND opponent_raw_age IS NOT NULL AS guard
    FROM prior_snapshot
    UNION ALL
    SELECT match_id, player_id, 'rank_diff' AS feature, rank_diff AS mf_val,
           player_prior_ranking - opponent_prior_ranking AS prior_val,
           player_raw_ranking IS NOT NULL AND opponent_raw_ranking IS NOT NULL AS guard
    FROM prior_snapshot
    UNION ALL
    SELECT match_id, player_id, 'rank_points_diff' AS feature, rank_points_diff AS mf_val,
           player_prior_rank_points - opponent_prior_rank_points AS prior_val,
           player_raw_rank_points IS NOT NULL AND opponent_raw_rank_points IS NOT NULL AS guard
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
JOIN prior_snapshot ps
    ON ps.match_id = c.match_id
   AND ps.player_id = c.player_id
WHERE c.guard
  AND (
    (c.mf_val IS NULL) <> (c.prior_val IS NULL)
    OR ABS(c.mf_val - c.prior_val) > 1e-6
  )
ORDER BY c.match_id, c.player_id, c.feature
