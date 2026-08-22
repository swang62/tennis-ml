-- silver.rolling_features: post-match player snapshots. Downstream reads the
-- prior one; windows are post-match and never include future matches.
--
-- [0,1] probability rates are Beta(1,1)-smoothed, (successes + 1) /
-- (opportunities + 2), so a first-match window = 0.5 and a zero-opportunity
-- window is never NULL; `matches_10` exposes the backing match count. Surface
-- rates smooth the same way but stay NULL for unseen surfaces.
-- dominance is a lifetime (unbounded) Dominance Ratio over the player's full
-- history, not a 10-match window.
-- aces_per_svc_game_10, weighted_form_10, streak, and rank averages are NOT
-- probabilities and are left unsmoothed; the training source never zero-fills.
-- Per-surface windows carry forward with conditional MAX (no LAST_VALUE IGNORE
-- NULLS in PostgreSQL). Streak is the signed current win/loss run including
-- this match. match_features derives rank trend; AVG skips NULL rankings.
--
-- Incremental: affected-player rebuild. A snapshot depends on matches up to its
-- own (match_date, match_id), so a historical insert changes every later
-- snapshot for that player; the run returns the FULL history of affected
-- players and delete+insert replaces stale snapshots in place.

{{ config(
    materialized="incremental",
    incremental_strategy="delete+insert",
    unique_key=["player_id", "match_id"],
) }}

WITH
{% if is_incremental() %}
-- Affected players have a missing or changed-ordinal snapshot; checking the
-- ordinal recovers when player_matches committed but rolling_features didn't.
changed_players AS (
    SELECT DISTINCT pm.player_id
    FROM {{ ref('player_matches') }} pm
    LEFT JOIN {{ this }} t
        ON t.player_id = pm.player_id
       AND t.match_id = pm.match_id
    WHERE t.match_id IS NULL
       OR t.player_match_number <> pm.player_match_number
),
{% endif %}
player_surface_matches AS (
    -- Inclusive 10-match rate on each surface, Beta(1,1)-smoothed.
    SELECT
        player_id,
        match_id,
        player_match_number,
        surface,
        (SUM(match_won) OVER (
            PARTITION BY player_id, surface
            ORDER BY match_date, match_id
            ROWS BETWEEN 9 PRECEDING AND CURRENT ROW
        ) + 1.0) / (COUNT(*) OVER (
            PARTITION BY player_id, surface
            ORDER BY match_date, match_id
            ROWS BETWEEN 9 PRECEDING AND CURRENT ROW
        ) + 2.0) AS surface_win_rate_10
    FROM {{ ref('player_matches') }}
),
-- Windows always evaluate over the FULL player_matches history; the incremental
-- filter applies only in the outermost SELECT so rows carry full-rebuild values.
snapshots AS (
    SELECT
        pm.player_id,
        pm.match_id,
        pm.match_date AS snapshot_date,
        pm.player_match_number,
        pm.surface,
        pm.player_ranking,
        pm.opponent_ranking,
        pm.player_rank_points,
        pm.player_age,
        pm.match_won,
        pm.aces,
        pm.double_faults,
        pm.first_serves_made,
        pm.total_serve_points,
        pm.first_serve_points_won,
        pm.second_serve_points_won,
        pm.service_games,
        pm.break_points_saved,
        pm.break_points_faced,
        pm.return_points_won,
        pm.return_points_available,
        -- Latest ordinal in the player's full history; the gap to this row feeds the
-- weighted-form decay (newest = 0, oldest = n-1) without a reversed sort.
        MAX(pm.player_match_number) OVER (
            PARTITION BY pm.player_id
        ) AS player_max_match_number,
        -- Latest ordinal per surface at each snapshot (0 if unseen), carried
        -- forward via conditional MAX (no LAST_VALUE IGNORE NULLS); computed
        -- here so the surface-rate join keys exist as plain columns.
        MAX(CASE WHEN pm.surface = 'clay'  THEN pm.player_match_number ELSE 0 END) OVER (
            PARTITION BY pm.player_id
            ORDER BY pm.match_date, pm.match_id
            ROWS BETWEEN UNBOUNDED PRECEDING AND CURRENT ROW
        ) AS clay_last_match_number,
        MAX(CASE WHEN pm.surface = 'grass' THEN pm.player_match_number ELSE 0 END) OVER (
            PARTITION BY pm.player_id
            ORDER BY pm.match_date, pm.match_id
            ROWS BETWEEN UNBOUNDED PRECEDING AND CURRENT ROW
        ) AS grass_last_match_number,
        MAX(CASE WHEN pm.surface = 'hard'  THEN pm.player_match_number ELSE 0 END) OVER (
            PARTITION BY pm.player_id
            ORDER BY pm.match_date, pm.match_id
            ROWS BETWEEN UNBOUNDED PRECEDING AND CURRENT ROW
        ) AS hard_last_match_number,
        -- Last loss/win ordinals feed the signed streak; same window pass as carries.
        MAX(CASE WHEN pm.match_won = 0 THEN pm.player_match_number ELSE 0 END) OVER (
            PARTITION BY pm.player_id
            ORDER BY pm.match_date, pm.match_id
            ROWS BETWEEN UNBOUNDED PRECEDING AND CURRENT ROW
        ) AS last_loss_match_number,
        MAX(CASE WHEN pm.match_won = 1 THEN pm.player_match_number ELSE 0 END) OVER (
            PARTITION BY pm.player_id
            ORDER BY pm.match_date, pm.match_id
            ROWS BETWEEN UNBOUNDED PRECEDING AND CURRENT ROW
        ) AS last_win_match_number
    FROM {{ ref('player_matches') }} pm
),
computed AS (
SELECT
    s.player_id,
    s.match_id,
    s.snapshot_date,
    s.player_match_number,
    s.surface,

    -- Current ranking, rank points, and age.
    s.player_ranking AS latest_player_ranking,
    s.player_rank_points AS latest_player_rank_points,
    s.player_age AS latest_player_age,

    -- Exponentially-decayed 10-match win rate; newest weight 1, decay exponent
    -- is the ordinal gap to the player's max, so no reversed sort pass.
    SUM(s.match_won * POW(0.9, s.player_max_match_number - s.player_match_number)) OVER w10
        / NULLIF(SUM(POW(0.9, s.player_max_match_number - s.player_match_number)) OVER w10, 0)
        AS weighted_form_10,

    -- Beta(1,1)-smoothed rates over the last 10 matches incl. this one:
    -- (x + 1) / (denominator + 2); first match = 0.5, zero-denominator never NULL.
    (SUM(s.match_won) OVER w10 + 1.0) / (COUNT(*) OVER w10 + 2.0)
        AS win_rate_10,
    (SUM(s.aces) OVER w10 + 1.0) / (SUM(s.first_serves_made) OVER w10 + 2.0)
        AS ace_rate_10,
    (SUM(s.first_serves_made) OVER w10 + 1.0)
        / (SUM(s.total_serve_points) OVER w10 + 2.0)
        AS first_serve_pct_10,
    (SUM(s.break_points_saved) OVER w10 + 1.0)
        / (SUM(s.break_points_faced) OVER w10 + 2.0)
        AS break_points_saved_pct_10,
    (SUM(s.first_serve_points_won) OVER w10 + 1.0)
        / (SUM(s.first_serves_made) OVER w10 + 2.0)
        AS first_serve_win_pct_10,
    (SUM(s.second_serve_points_won) OVER w10 + 1.0)
        / (SUM(s.total_serve_points - s.first_serves_made) OVER w10 + 2.0)
        AS second_serve_win_pct_10,
    (SUM(s.first_serve_points_won + s.second_serve_points_won) OVER w10 + 1.0)
        / (SUM(s.total_serve_points) OVER w10 + 2.0)
        AS serve_win_pct_10,
    (SUM(s.return_points_won) OVER w10 + 1.0)
        / (SUM(s.return_points_available) OVER w10 + 2.0)
        AS return_points_won_pct_10,
    -- Lifetime (unbounded) Dominance Ratio: return strength per unit of serve
    -- weakness, Beta(1,1)-smoothed over the player's full history (w_life).
    -- Both inputs are smoothed rates below 1, so the denominator is never
    -- zero; emitted as the raw calculated numeric (no truncation).
    (
        (SUM(s.return_points_won) OVER w_life + 1.0)
            / (SUM(s.return_points_available) OVER w_life + 2.0)
        / (1.0 - (SUM(s.first_serve_points_won + s.second_serve_points_won) OVER w_life + 1.0)
                   / (SUM(s.total_serve_points) OVER w_life + 2.0))
    )::NUMERIC AS dominance,
    (SUM(s.double_faults) OVER w10 + 1.0)
        / (SUM(s.total_serve_points) OVER w10 + 2.0)
        AS df_rate_10,

    -- Matches backing the smoothed rates (1 for the first, up to 10);
    -- carried to gold as matches_10 exposure.
    COUNT(*) OVER w10 AS matches_10,

    -- Unsmoothed rates (not probabilities): aces per service game and rolling
    -- average ranks; AVG skips NULL opponent rankings.
    CAST(SUM(s.aces) OVER w10 AS DOUBLE PRECISION)
        / NULLIF(SUM(s.service_games) OVER w10, 0)
        AS aces_per_svc_game_10,
    AVG(s.player_ranking) OVER w10 AS avg_player_rank_10,
    AVG(s.opponent_ranking) OVER w10 AS avg_rank_faced_10,

    -- Signed current win/loss run, from the snapshot pass's ordinals.
    s.streak,

    -- Per-surface rates carried forward from the latest surface match.
    psm_clay.surface_win_rate_10  AS clay_win_rate_10,
    psm_grass.surface_win_rate_10 AS grass_win_rate_10,
    psm_hard.surface_win_rate_10  AS hard_win_rate_10

FROM (
    -- Streak is derived here because window results aren't visible to the
    -- JOIN clauses that use the surface-rate keys.
    SELECT s.*,
        CASE WHEN s.match_won = 1
            THEN s.player_match_number - s.last_loss_match_number
            ELSE -1 * (s.player_match_number - s.last_win_match_number)
        END AS streak
    FROM snapshots s
) s
LEFT JOIN player_surface_matches psm_clay
    ON psm_clay.player_id = s.player_id
   AND psm_clay.surface = 'clay'
   AND psm_clay.player_match_number = s.clay_last_match_number
LEFT JOIN player_surface_matches psm_grass
    ON psm_grass.player_id = s.player_id
   AND psm_grass.surface = 'grass'
   AND psm_grass.player_match_number = s.grass_last_match_number
LEFT JOIN player_surface_matches psm_hard
    ON psm_hard.player_id = s.player_id
   AND psm_hard.surface = 'hard'
   AND psm_hard.player_match_number = s.hard_last_match_number
WINDOW
    w10 AS (PARTITION BY s.player_id ORDER BY s.snapshot_date, s.match_id
            ROWS BETWEEN 9 PRECEDING AND CURRENT ROW),
    w_life AS (PARTITION BY s.player_id ORDER BY s.snapshot_date, s.match_id
               ROWS BETWEEN UNBOUNDED PRECEDING AND CURRENT ROW)
)
-- Trim to affected players' full histories; every window above was evaluated
-- over full history, so returned rows carry full-rebuild values.
SELECT * FROM computed
{% if is_incremental() %}
WHERE player_id IN (SELECT player_id FROM changed_players)
{% endif %}
ORDER BY snapshot_date, match_id, player_id
