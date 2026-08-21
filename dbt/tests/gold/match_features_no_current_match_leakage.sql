-- Assert every snapshot-backed feature of each SAMPLED match_features row comes
-- from the player's PRIOR snapshot only (latest rolling_features snapshot
-- strictly before match_date, the same date-strict semantics inference uses) or,
-- when that prior state is missing (cold start), from the tour_averages singleton.
-- Both sides of each directional row are checked. Any mismatch = used a
-- current-match snapshot (leakage) or the wrong default/raw stats.
--
-- SAMPLED for speed: a deterministic, bounded set of physical match_ids is
-- sampled by (match_date, match_id) from silver.player_matches, then only those
-- directional gold rows are re-derived (both mirrors always). Strata, one per
-- LIMIT-capped branch so size stays bounded as data grows: earliest, latest,
-- cold starts (player_match_number = 1), same-date back-to-backs (the strict <
-- guard excludes the same-day snapshot), repeated unordered-pair H2H, and a
-- deterministic md5(match_id) uniform ~1/256 sample.
--
-- Silver stores the Beta(1,1)-smoothed rates; match_features consumes them
-- directly, so prior values re-read the same stored columns. Fallback cells
-- (prior NULL, imputed from the singleton) are excluded via the guard on each
-- RAW prior value, because incremental rows keep their build-time pool values
-- ("existing data remains unchanged"). Most features are DIFFS re-derived from
-- the two prior snapshots (COALESCE'd to the singleton); as-of-date values come
-- from the prior snapshot, never current-match raw stats.
--
-- match_features emits full arithmetic precision, so plain recomputation and a
-- 1e-6 float tolerance is compared; real leakage shifts a feature by O(0.01).

-- Covered snapshot-backed fields since they are stored in match_features:
--   diff:      win_rate, streak, surface_form, days_since_last_match, ace_rate,
--              first_serve_pct, break_points_saved_pct, first_serve_win_pct,
--              second_serve_win_pct, serve_win_pct, return_points_won_pct,
--              df_rate, aces_per_svc_game, avg_rank_faced, rank_trend, rank,
--              rank_points, age
--   per-side:  player/opponent_weighted_form_10, player/opponent_matches_10
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
    -- Deterministic, bounded physical-match sample. Each stratum is a
    -- per-branch parenthesized LIMIT, so the sample size is capped even as
    -- the dataset grows. (match_date, match_id) is unique per physical match.
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
        -- Cold start: either side's first match has no strictly-prior snapshot.
        (SELECT DISTINCT match_id, match_date
         FROM {{ ref('player_matches') }}
         WHERE player_match_number = 1
         ORDER BY match_date, match_id
         LIMIT {{ sample_sizes.cold }})
        UNION ALL
        -- Same-date back-to-back: the strict < guard must exclude the same-day
        -- snapshot.
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
        -- Repeated unordered-pair H2H (two or more distinct meetings).
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
        -- Deterministic uniform coverage: md5 byte mod denominator (~1/256).
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
        -- Stored diff features
        mf.win_rate_diff, mf.streak_diff,
        mf.ace_rate_diff, mf.first_serve_pct_diff,
        mf.break_points_saved_pct_diff, mf.first_serve_win_pct_diff,
        mf.second_serve_win_pct_diff, mf.serve_win_pct_diff,
        mf.return_points_won_pct_diff,
        mf.df_rate_diff, mf.aces_per_svc_game_diff,
        mf.avg_rank_faced_diff, mf.rank_trend_diff,
        mf.rank_diff, mf.rank_points_diff, mf.age_diff,
        mf.surface_form_diff, mf.days_since_last_match_diff,
        -- Stored per-side absolute values
        mf.player_weighted_form_10, mf.opponent_weighted_form_10,
        -- Prior snapshot inputs (N-1), COALESCE'd to the singleton exactly as
        -- match_features imputes. Each carries its RAW value: comparison is
        -- guarded on it, because a cell that fell back to the singleton is
        -- frozen at build-time pool values the current singleton can't reproduce.
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
        -- Strictly-prior as-of-date rankings/rank points/age (imputed).
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
        -- Stored exposure values (0 for cold start) and the prior snapshot's
        -- matches_10 backing the smoothed 10-match rates.
        mf.player_matches_10,
        mf.opponent_matches_10,
        prp.matches_10 AS player_raw_matches_10,
        pro.matches_10 AS opponent_raw_matches_10,
        COALESCE(prp.matches_10, 0) AS player_prior_matches_10,
        COALESCE(pro.matches_10, 0) AS opponent_prior_matches_10,
        -- Surface form: prior snapshot's carried per-surface win rate chosen by
        -- the match surface (pool mean when unseen; carpet has no rate, so the
        -- fixed 0.5 rate_default applies, never a pool value).
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
        -- Rest: days since the prior snapshot's match; pool median on cold start,
        -- capped at 30 before the log transform, exactly as the model.
        pm.match_date - prp.snapshot_date AS player_raw_days_since,
        po.match_date - pro.snapshot_date AS opponent_raw_days_since,
        LEAST(30, CASE WHEN prp.player_id IS NULL THEN fd.days_since_default
             ELSE pm.match_date - prp.snapshot_date END) AS player_prior_days_since,
        LEAST(30, CASE WHEN pro.player_id IS NULL THEN fd.days_since_default
             ELSE po.match_date - pro.snapshot_date END) AS opponent_prior_days_since
    -- Filter before the lateral expansion so prior lookups run only for sampled rows.
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
          AND rfp.snapshot_date < pm.match_date
        ORDER BY rfp.player_match_number DESC
        LIMIT 1
    ) prp ON true
    LEFT JOIN LATERAL (
        SELECT * FROM {{ ref('rolling_features') }} rfo
        WHERE rfo.player_id = mf.opponent_id
          AND rfo.snapshot_date < po.match_date
        ORDER BY rfo.player_match_number DESC
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
  -- Real leakage shifts a feature by an O(0.01) amount; tolerate float
  -- round-off so recomputation noise never flags a false positive.
  AND (
    (c.mf_val IS NULL) <> (c.prior_val IS NULL)
    OR ABS(c.mf_val - c.prior_val) > 1e-6
  )
ORDER BY c.match_id, c.player_id, c.feature
