-- gold.match_features: symmetric player-perspective training table.
--
-- TWO directional rows per physical match, keyed by (match_id, player_id):
-- one row per player, each in that player's own perspective. The two rows
-- share match_id and the match context, exchange the paired player_*/opponent_*
-- side values, negate the signed differences and the h2h advantage features,
-- and carry
-- complementary match_won labels. Every side-level value (ranking, rank
-- points, age, rolling state, profile, activity) is imputed BEFORE matchup
-- differences are calculated, so every FEATURE_COLS cell is non-null and
-- finite. Rolling values use the player's PRIOR snapshot from
-- silver.rolling_features — the latest snapshot strictly before match_date
-- (snapshot_date < match_date, same-date matches cannot supply it, matching
-- inference), fetched via a per-side lateral join; ranking/rank points/age
-- also come from the prior snapshot, never a same-day one. The prior
-- snapshot's `matches_10` exposure is carried through as player_matches_10 /
-- opponent_matches_10 (0 for cold start). Missing or NULL prior cells fall
-- back to the single-row gold.tour_averages singleton (CROSS JOIN), not a
-- date-keyed defaults row.
--
-- match_won = 1 iff this row's player side won, else 0. It is the LABEL, not
-- a feature.
--
-- H2H reads bronze.match_events (via winner_id, no id canonicalization).
-- h2h_exposure is the count of the FIVE most recent strictly-prior
-- unordered-pair meetings (identical for both mirrors); h2h_advantage and
-- h2h_surface_advantage are Beta(1,1)-smoothed directional advantages built
-- from the same bounded five-meeting window (the surface variant restricted
-- to meetings on the current match's surface) and negate across mirrors.
-- Lookups stay bounded (LIMIT 5 laterals); no lifetime count is kept.
--
-- Player snapshot state stays strictly prior (no current-match leakage);
-- fallback values are intentionally global: the same full-pool singleton is
-- used for old cold-start and currently missing cells, so limited historical
-- leakage into fallback-consuming rows is accepted and reported.
--
-- Similarity-analysis serve/return columns are appended for PlayerSimilarity
-- only; they are never FEATURE_COLS model features.
--
-- Model columns are metadata plus FEATURE_COLS. Height and current-match
-- rates are not model features.
--
-- Incremental boundary: the ETL watermark identifies bronze rows ingested
-- since the last successful dbt build, so each ETL run appends exactly two
-- directional rows per new bronze match without scanning this target. New rows are
-- computed against the FULL silver history (prior snapshots via the
-- date-strict lateral join, H2H over the bounded recent-five meetings, the freshly
-- rebuilt gold.tour_averages singleton), so a new match's rows are exactly
-- what a full rebuild would produce; existing rows are untouched. Re-running
-- with no new bronze matches inserts nothing (idempotent).

{{ config(
    materialized="incremental",
    incremental_strategy="delete+insert",
    unique_key=["player_id", "match_id"],
) }}

WITH
{% if is_incremental() %}
-- New bronze rows since the last successful dbt build. The source watermark is
-- recorded only after the whole build succeeds, so a failed downstream model
-- leaves the batch eligible for the next run.
new_match_ids AS (
    SELECT match_id
    FROM {{ source('bronze', 'match_events') }}
    WHERE ingested_at > COALESCE(
        (SELECT source_watermark FROM bronze.etl_state WHERE pipeline = 'dbt'),
        '-infinity'::TIMESTAMPTZ
    )
),
{% endif %}
player_match_enriched AS (
    SELECT
        pm.match_id,
        pm.match_date,
        bron.tournament,
        bron.round,
        pm.surface,
        -- Indoor context defaults to 0 (outdoor) when unknown.
        COALESCE(bron.is_indoor, 0) AS is_indoor,
        pm.player_id,
        pm.opponent_id,
        pm.match_won,

        -- Strictly-prior ranking/rank points/age (no same-day snapshot),
        -- imputed from the singleton defaults row.
        COALESCE(pr.latest_player_ranking, fd.latest_player_ranking) AS player_ranking,
        COALESCE(pr.latest_player_rank_points, fd.latest_player_rank_points)
            AS player_rank_points,
        COALESCE(pr.latest_player_age, fd.latest_player_age) AS player_age,

        -- Rate-exposure carry: number of prior matches in the 10-match window
        -- backing the smoothed rates; cold start uses literal 0 (no prior match).
        CASE WHEN pr.player_id IS NULL THEN 0
             ELSE CAST(pr.matches_10 AS INTEGER)
        END AS matches_10,

        -- Prior rolling snapshot, imputed from the defaults pool.
        COALESCE(pr.weighted_form_10, fd.weighted_form_10) AS weighted_form_10,
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
        COALESCE(pr.df_rate_10, fd.df_rate_10) AS df_rate_10,
        COALESCE(pr.aces_per_svc_game_10, fd.aces_per_svc_game_10)
            AS aces_per_svc_game_10,

        -- Rank trend: prior rolling avg ranking minus prior latest ranking.
        COALESCE(pr.avg_player_rank_10, fd.avg_player_rank_10)
            - COALESCE(pr.latest_player_ranking, fd.latest_player_ranking)
            AS rank_trend_10,

        -- Strength of schedule: prior snapshot's rolling average opponent rank.
        COALESCE(pr.avg_rank_faced_10, fd.avg_rank_faced_10) AS avg_rank_faced_10,

        COALESCE(pr.streak, fd.streak) AS streak,

        -- Surface form on the CURRENT match's surface: the prior snapshot's
        -- carried per-surface win rate (last 10 on that surface, Beta(1,1)
        -- smoothed), imputed from its pool mean when the player has never
        -- played the surface; carpet has no per-surface pool rate, so the
        -- neutral 0.5 rate_default applies.
        CASE pm.surface
            WHEN 'clay'  THEN COALESCE(pr.clay_win_rate_10,  fd.clay_win_rate_10)
            WHEN 'grass' THEN COALESCE(pr.grass_win_rate_10, fd.grass_win_rate_10)
            WHEN 'hard'  THEN COALESCE(pr.hard_win_rate_10,  fd.hard_win_rate_10)
            ELSE fd.rate_default
        END AS surface_form,

        -- Rest: days since the player's immediately preceding match (the
        -- prior snapshot's date is that match); pool median days-since
        -- fallback on cold start. Same-date matches cannot supply the prior
        -- snapshot (strict <), so a same-week back-to-back is at least 1 day.
        -- Capped at 30 before the downstream ln(1 + days) transform.
        LEAST(30, CASE WHEN pr.player_id IS NULL THEN fd.days_since_default
             ELSE pm.match_date - pr.snapshot_date
        END) AS days_since_last_match,

        -- Missing or non-L/R handedness uses the pool left-handed rate.
        COALESCE(
            CAST(CASE WHEN prof.handedness = 'L' THEN 1
                      WHEN prof.handedness = 'R' THEN 0 END AS DOUBLE PRECISION),
            fd.left_handed_rate
        ) AS is_left_handed,

        -- Time-aware years pro; missing turned_pro uses the pool mean.
        COALESCE(
            CAST(EXTRACT(YEAR FROM pm.match_date) - prof.turned_pro AS DOUBLE PRECISION),
            fd.avg_years_pro
        ) AS years_pro

    FROM {{ ref('player_matches') }} pm
{% if is_incremental() %}
    JOIN new_match_ids nm
        ON nm.match_id = pm.match_id
{% endif %}
    LEFT JOIN LATERAL (
        SELECT * FROM {{ ref('rolling_features') }} rf
        WHERE rf.player_id = pm.player_id
          AND rf.snapshot_date < pm.match_date
        -- (snapshot_date, match_id) DESC == player_match_number DESC (the
        -- ordinal is assigned in exactly that order) and lets the ordered
        -- idx_rolling_pid_date_match serve a bounded backward scan.
        ORDER BY rf.snapshot_date DESC, rf.match_id DESC
        LIMIT 1
    ) pr ON true
    CROSS JOIN {{ ref('tour_averages') }} fd
    LEFT JOIN {{ source('bronze', 'match_events') }} bron
        ON bron.match_id = pm.match_id
    LEFT JOIN {{ source('bronze', 'player_profiles') }} prof
        ON prof.player_id = pm.player_id
),
-- Strictly-prior unordered-pair meetings per directional row, read directly
-- from bronze.match_events (one row per physical match, so no dedup; winner_id
-- orients each meeting to the row's player without any id canonicalization).
-- Each orientation (player as player1 / player as player2) is an indexed
-- bounded lateral lookup capped at the five newest meetings, so at most ten
-- candidate meetings per directional row reach the window below; the global
-- (match_date DESC, match_id DESC) top five is then identical to the old
-- OR-join over all prior meetings while scanning only the newest meetings.
-- meeting_surface_matches marks meetings on the CURRENT match's surface so the
-- surface advantage can be computed from the same bounded top-five window.
pair_meetings AS (
    SELECT match_id, player_id, winner_is_current_player, meeting_surface_matches
    FROM (
        SELECT
            match_id,
            player_id,
            (winner_id = player_id) AS winner_is_current_player,
            (meeting_surface = surface) AS meeting_surface_matches,
            ROW_NUMBER() OVER (
                PARTITION BY match_id, player_id
                ORDER BY match_date DESC, meeting_match_id DESC
            ) AS rn
        FROM (
            -- Player was player1 in the prior meeting.
            SELECT
                current_match.match_id,
                current_match.player_id,
                meeting.winner_id,
                meeting.match_date,
                meeting.match_id AS meeting_match_id,
                meeting.surface AS meeting_surface,
                current_match.surface
            FROM player_match_enriched current_match
            CROSS JOIN LATERAL (
                SELECT winner_id, match_date, match_id, surface
                FROM {{ source('bronze', 'match_events') }} meeting
                WHERE meeting.player1_id = current_match.player_id
                  AND meeting.player2_id = current_match.opponent_id
                  AND meeting.match_date < current_match.match_date
                ORDER BY meeting.match_date DESC, meeting.match_id DESC
                LIMIT 5
            ) meeting
            UNION ALL
            -- Player was player2 in the prior meeting.
            SELECT
                current_match.match_id,
                current_match.player_id,
                meeting.winner_id,
                meeting.match_date,
                meeting.match_id AS meeting_match_id,
                meeting.surface AS meeting_surface,
                current_match.surface
            FROM player_match_enriched current_match
            CROSS JOIN LATERAL (
                SELECT winner_id, match_date, match_id, surface
                FROM {{ source('bronze', 'match_events') }} meeting
                WHERE meeting.player1_id = current_match.opponent_id
                  AND meeting.player2_id = current_match.player_id
                  AND meeting.match_date < current_match.match_date
                ORDER BY meeting.match_date DESC, meeting.match_id DESC
                LIMIT 5
            ) meeting
        ) meetings_union
    ) ranked
    WHERE rn <= 5
),
-- Directional H2H per row: recent-5 wins oriented to the row player plus the
-- surface-restricted recent-5 wins. pair_meetings already restricted each row
-- to its five newest meetings.
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
-- One directional row per player perspective of each physical match.
SELECT
    p.match_id,
    p.match_date,
    p.player_id,
    p.opponent_id,
    p.tournament,
    p.round,
    p.surface,
    p.match_won,

    -- ── Matchup differences (imputed row player side minus imputed opponent) ──
    p.player_ranking - o.player_ranking AS rank_diff,
    p.player_rank_points - o.player_rank_points AS rank_points_diff,
    p.player_age - o.player_age AS age_diff,
    p.win_rate_10 - o.win_rate_10 AS win_rate_diff,
    p.ace_rate_10 - o.ace_rate_10 AS ace_rate_diff,
    p.first_serve_pct_10 - o.first_serve_pct_10 AS first_serve_pct_diff,
    p.break_points_saved_pct_10 - o.break_points_saved_pct_10
        AS break_points_saved_pct_diff,
    p.first_serve_win_pct_10 - o.first_serve_win_pct_10 AS first_serve_win_pct_diff,
    p.second_serve_win_pct_10 - o.second_serve_win_pct_10 AS second_serve_win_pct_diff,
    p.serve_win_pct_10 - o.serve_win_pct_10 AS serve_win_pct_diff,
    p.return_points_won_pct_10 - o.return_points_won_pct_10
        AS return_points_won_pct_diff,
    p.df_rate_10 - o.df_rate_10 AS df_rate_diff,
    p.aces_per_svc_game_10 - o.aces_per_svc_game_10 AS aces_per_svc_game_diff,
    p.rank_trend_10 - o.rank_trend_10 AS rank_trend_diff,
    p.avg_rank_faced_10 - o.avg_rank_faced_10 AS avg_rank_faced_diff,
    p.streak - o.streak AS streak_diff,
    p.surface_form - o.surface_form AS surface_form_diff,
    -- Log-transformed directional rest: ln(1 + player) - ln(1 + opponent),
    -- both inputs capped at 30 before the transform.
    LN(1.0 + p.days_since_last_match) - LN(1.0 + o.days_since_last_match)
        AS days_since_last_match_diff,

    -- ── Absolute state values where both sides matter ──
    p.weighted_form_10 AS player_weighted_form_10,
    o.weighted_form_10 AS opponent_weighted_form_10,
    -- Rate-exposure counts backing the smoothed 10-match rates (0 cold start).
    p.matches_10            AS player_matches_10,
    o.matches_10            AS opponent_matches_10,
    p.is_left_handed        AS player_is_left_handed,
    o.is_left_handed        AS opponent_is_left_handed,
    p.years_pro AS player_years_pro,
    o.years_pro AS opponent_years_pro,

    -- ── Pair-level head-to-head: recent-5 exposure + signed advantages
    --    (overall and current-surface) from the bounded meeting window ──
    COALESCE(h.recent_meetings, 0) AS h2h_exposure,
    ((COALESCE(h.wins_for_player, 0) + 1.0)
        / (COALESCE(h.recent_meetings, 0) + 2.0) - 0.5)
        AS h2h_advantage,
    ((COALESCE(h.surface_wins_for_player, 0) + 1.0)
        / (COALESCE(h.surface_meetings, 0) + 2.0) - 0.5)
        AS h2h_surface_advantage,

    -- ── Numeric match context (one-hot surface for linear and neural models) ──
    CAST(CASE WHEN p.surface = 'clay'  THEN 1 ELSE 0 END AS SMALLINT) AS is_clay,
    CAST(CASE WHEN p.surface = 'grass' THEN 1 ELSE 0 END AS SMALLINT) AS is_grass,
    CAST(CASE WHEN p.surface = 'hard'  THEN 1 ELSE 0 END AS SMALLINT) AS is_hard,
    p.is_indoor,
    CAST(CASE p.tournament
        WHEN 'grand_slam' THEN 4 WHEN 'masters' THEN 3
        WHEN 'atp_500' THEN 2 WHEN 'atp_250' THEN 1 ELSE 0
    END AS SMALLINT) AS tournament_level,
    CAST(CASE p.round
        WHEN 'r128' THEN 1 WHEN 'r64' THEN 2 WHEN 'r32' THEN 3 WHEN 'r16' THEN 4
        WHEN 'qf' THEN 5 WHEN 'sf' THEN 6 WHEN 'f' THEN 7 ELSE 0
    END AS SMALLINT) AS round_encoded,

    -- ── Similarity-analysis serve/return percentages (NOT model features) ──
    -- Appended style signals for PlayerSimilarity, never FEATURE_COLS. They
    -- share the same defaults imputation but are documented as non-model.
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
